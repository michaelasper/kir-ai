use crate::{
    DEFAULT_NATIVE_TEXT_MAX_NEW_TOKENS, DEFAULT_NATIVE_TEXT_MAX_PREFILL_TOKENS,
    native_matvec::{
        NativeTextCacheMirrorCleaner, NativeTextCacheMirrorIds, NativeTextCacheMirrorSource,
        NativeTextMatvecBackend, native_text_metal_weight_cache_bytes,
    },
    native_text::{
        NativeTextAdapter, NativeTextDriver, NativeTextNextTokenContext, NativeTextPrefixCache,
        NativeTextPrefixCacheMetrics, NativeTextPrefixCacheNamespace, NativeTextPrefixCacheValue,
        NativeTextPrefixNamespaceContext, NativeTextStopTokens, native_text_prefix_namespace,
    },
};
use async_trait::async_trait;
use futures::stream::BoxStream;
use llm_backend::{
    BackendError, BackendModelMetadata, BackendOutput, BackendRequest, BackendStreamChunk,
    GemmaLayerCache, InferenceScratchpad, ModelBackend, NativeMatvecBackend,
    NativeTextLayerCachesMut, SafeTensorShardStore, SamplingConfig, gemma_cache_count_for_spec,
    gemma_layer_caches_for_spec, gemma_static_f32_tensors_for_spec,
    native_decode_token_with_cache_for_spec_ref, native_prefill_sequence_with_cache_for_spec_ref,
};
use llm_hub::SnapshotManifest;
use llm_models::GemmaModelSpec;
use llm_tokenizer::HuggingFaceTokenizer;
use serde_json::Value;
use std::{
    path::Path,
    sync::{Arc, OnceLock},
};
use tokio_util::sync::CancellationToken;

const DEFAULT_NATIVE_GEMMA_PREFIX_CACHE_BYTES: u64 = 512 * 1024 * 1024;
const NATIVE_GEMMA_PREFIX_CACHE_LAYOUT_VERSION: u32 = 1;

#[derive(Clone)]
pub struct NativeGemmaBackend {
    driver: NativeTextDriver<NativeGemmaAdapter>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativeGemmaLoadOptions {
    pub eager_materialize_shards: bool,
    pub metal_weight_cache_bytes: Option<u64>,
    pub warm_metal_weight_cache: bool,
}

#[derive(Clone)]
pub(crate) struct NativeGemmaAdapter {
    model_id: String,
    metadata: BackendModelMetadata,
    spec: GemmaModelSpec,
    store: SafeTensorShardStore,
    matvec: NativeTextMatvecBackend,
    max_prefill_tokens: usize,
    top_k: usize,
    chunk_rows: usize,
    prefix_cache: Arc<NativeGemmaPrefixCache>,
}

type NativeGemmaPrefixCache = NativeTextPrefixCache<GemmaLayerCache>;
type NativeGemmaPrefixCacheMetrics = NativeTextPrefixCacheMetrics;

fn native_gemma_prefix_cache_metrics() -> &'static NativeGemmaPrefixCacheMetrics {
    static METRICS: OnceLock<NativeGemmaPrefixCacheMetrics> = OnceLock::new();
    METRICS.get_or_init(NativeGemmaPrefixCacheMetrics::default)
}

pub(crate) fn native_gemma_prefix_cache_metrics_snapshot() -> Value {
    native_gemma_prefix_cache_metrics().snapshot()
}

impl NativeTextPrefixCacheValue for GemmaLayerCache {
    fn prefix_cache_entry_bytes(hidden: &[f32], caches: &[Self]) -> u64 {
        let hidden_bytes = std::mem::size_of_val(hidden) as u64;
        caches.iter().fold(hidden_bytes, |total, cache| {
            total.saturating_add(match cache {
                GemmaLayerCache::Attention(cache) => {
                    ((cache.key_storage().len() + cache.value_storage().len())
                        * std::mem::size_of::<f32>()) as u64
                }
            })
        })
    }
}

impl NativeTextCacheMirrorSource for GemmaLayerCache {
    fn append_cache_mirror_ids(&self, ids: &mut NativeTextCacheMirrorIds) {
        match self {
            GemmaLayerCache::Attention(cache) => ids.push_kv(cache.id()),
        }
    }
}

impl NativeGemmaBackend {
    pub async fn open(
        model_id: impl Into<String>,
        snapshot_path: impl AsRef<Path>,
    ) -> anyhow::Result<Self> {
        Self::open_with_options(model_id, snapshot_path, NativeGemmaLoadOptions::default()).await
    }

    pub async fn open_with_options(
        model_id: impl Into<String>,
        snapshot_path: impl AsRef<Path>,
        options: NativeGemmaLoadOptions,
    ) -> anyhow::Result<Self> {
        let model_id = model_id.into();
        let snapshot_path = snapshot_path.as_ref();
        let cache_namespace = snapshot_path.canonicalize()?.to_string_lossy().into_owned();
        let metadata = native_gemma_metadata(&model_id, snapshot_path).await?;
        reject_native_gemma_quantized_snapshot(snapshot_path).await?;
        let config_json = tokio::fs::read_to_string(snapshot_path.join("config.json")).await?;
        let spec = GemmaModelSpec::from_config_json(&config_json)?;
        let store = SafeTensorShardStore::open(snapshot_path)?;
        store.index().validate_gemma4_text_weights(&spec)?;
        if options.eager_materialize_shards {
            store.materialize_all_shards().map_err(|err| {
                anyhow::anyhow!("native Gemma safetensors materialization failed: {err}")
            })?;
        }
        let static_f32_tensors = gemma_static_f32_tensors_for_spec(&spec);
        let static_f32_warmup = store.preload_bf16_f32_tensors(&static_f32_tensors)?;
        tracing::info!(
            candidates = static_f32_warmup.candidates,
            loaded = static_f32_warmup.loaded,
            resident_bytes = static_f32_warmup.resident_bytes,
            already_resident = static_f32_warmup.already_resident,
            "native Gemma static f32 tensor cache warm-up complete"
        );
        let matvec = NativeTextMatvecBackend::system_default(
            native_text_metal_weight_cache_bytes(options.metal_weight_cache_bytes),
            &cache_namespace,
        );
        if options.warm_metal_weight_cache {
            let warmup = matvec.warm_bf16_matrix_cache(&store).await.map_err(|err| {
                anyhow::anyhow!("native Gemma Metal weight cache warm-up failed: {err}")
            })?;
            tracing::info!(
                candidates = warmup.candidates,
                warmed = warmup.warmed,
                already_resident = warmup.already_resident,
                skipped_budget = warmup.skipped_budget,
                skipped_non_metal = warmup.skipped_non_metal,
                "native Gemma Metal BF16 weight cache warm-up complete"
            );
        }
        let tokenizer = HuggingFaceTokenizer::from_file(snapshot_path.join("tokenizer.json"))?;
        let adapter = NativeGemmaAdapter {
            model_id: model_id.clone(),
            metadata: metadata.clone(),
            spec,
            store,
            matvec,
            max_prefill_tokens: DEFAULT_NATIVE_TEXT_MAX_PREFILL_TOKENS,
            top_k: 16,
            chunk_rows: 2048,
            prefix_cache: Arc::new(NativeGemmaPrefixCache::new(
                DEFAULT_NATIVE_GEMMA_PREFIX_CACHE_BYTES,
            )),
        };
        Ok(Self {
            driver: NativeTextDriver::new(
                model_id,
                metadata,
                tokenizer,
                adapter,
                DEFAULT_NATIVE_TEXT_MAX_NEW_TOKENS,
            ),
        })
    }

    pub fn with_max_new_tokens(mut self, max_new_tokens: u32) -> Self {
        self.driver = self.driver.with_max_new_tokens(max_new_tokens);
        self
    }

    pub fn with_max_prefill_tokens(mut self, max_prefill_tokens: usize) -> Self {
        self.driver = self.driver.with_max_prefill_tokens(max_prefill_tokens);
        self
    }

    pub(crate) fn into_driver(self) -> NativeTextDriver<NativeGemmaAdapter> {
        self.driver
    }

    #[cfg(test)]
    fn start_decode_session(
        &self,
        context_tokens: &[usize],
        max_new_tokens: u32,
        request: &BackendRequest,
        cancellation: &CancellationToken,
    ) -> Result<NativeGemmaDecodeSession, BackendError> {
        let driver = &self.driver;
        tokio::task::block_in_place(|| {
            driver.block_on_worker(driver.start_decode_session(
                context_tokens,
                max_new_tokens,
                request,
                cancellation,
                &mut InferenceScratchpad::new(),
            ))?
        })
    }

    #[cfg(test)]
    fn next_token_from_hidden(
        &self,
        hidden: &[f32],
        sampling: SamplingConfig,
    ) -> Result<usize, BackendError> {
        let sampling_draw = if sampling.is_greedy() {
            None
        } else {
            let mut sampling_rng = crate::native_text::NativeTextSamplingRng::from_entropy();
            Some(sampling_rng.draw_f32())
        };
        tokio::task::block_in_place(|| {
            self.driver
                .block_on_worker(self.driver.adapter.next_token_from_hidden(
                    hidden,
                    sampling,
                    sampling_draw,
                    &mut llm_sampler::TopPSamplerScratch::new(),
                ))?
        })
    }
}

#[async_trait]
impl NativeTextAdapter for NativeGemmaAdapter {
    type DecodeSession = NativeGemmaDecodeSession;
    type LayerCache = GemmaLayerCache;

    fn family_display_name(&self) -> &'static str {
        "Gemma"
    }

    fn worker_label(&self) -> &'static str {
        "native Gemma"
    }

    fn set_max_prefill_tokens(&mut self, max_prefill_tokens: usize) {
        self.max_prefill_tokens = max_prefill_tokens.max(1);
    }

    fn encode_prompt(
        &self,
        tokenizer: &HuggingFaceTokenizer,
        request: &BackendRequest,
    ) -> Result<Vec<u32>, BackendError> {
        tokenizer
            .encode(request.prompt(), false)
            .map_err(|err| BackendError::other(err.to_string()))
    }

    fn decode_output(
        &self,
        tokenizer: &HuggingFaceTokenizer,
        output_ids: &[u32],
    ) -> Result<String, BackendError> {
        tokenizer
            .decode(output_ids, false)
            .map_err(|err| BackendError::other(err.to_string()))
    }

    fn stop_tokens(&self) -> NativeTextStopTokens {
        native_gemma_stop_tokens()
    }

    fn max_position_embeddings(&self) -> u32 {
        self.spec.max_position_embeddings
    }

    fn max_prefill_tokens(&self) -> usize {
        self.max_prefill_tokens
    }

    fn prefix_cache(&self) -> &NativeTextPrefixCache<GemmaLayerCache> {
        &self.prefix_cache
    }

    fn prefix_cache_metrics(&self) -> &NativeTextPrefixCacheMetrics {
        native_gemma_prefix_cache_metrics()
    }

    fn prefix_cache_namespace(
        &self,
        request: &BackendRequest,
        cache_tokens: usize,
    ) -> NativeTextPrefixCacheNamespace {
        native_text_prefix_namespace(NativeTextPrefixNamespaceContext {
            model_id: &self.model_id,
            metadata: &self.metadata,
            request,
            cache_layout_version: NATIVE_GEMMA_PREFIX_CACHE_LAYOUT_VERSION,
            cache_tokens,
            max_prefill_tokens: self.max_prefill_tokens,
        })
    }

    fn prefix_cache_hit_is_compatible(
        &self,
        caches: &[GemmaLayerCache],
        cache_tokens: usize,
    ) -> bool {
        caches.iter().all(|cache| match cache {
            GemmaLayerCache::Attention(cache) => cache.max_tokens() >= cache_tokens,
        })
    }

    fn layer_count(&self) -> usize {
        gemma_cache_count_for_spec(&self.spec).unwrap_or(self.spec.num_hidden_layers as usize)
    }

    fn allocate_caches(&self, cache_tokens: usize) -> Result<Vec<GemmaLayerCache>, BackendError> {
        gemma_layer_caches_for_spec(&self.spec, cache_tokens)
            .map_err(|err| BackendError::other(err.to_string()))
    }

    async fn prefill_chunk_with_cache(
        &self,
        token_ids: &[usize],
        caches: &mut [GemmaLayerCache],
        scratch: &mut InferenceScratchpad,
    ) -> Result<Vec<Vec<f32>>, BackendError> {
        native_prefill_sequence_with_cache_for_spec_ref(
            &self.store,
            (&self.spec).into(),
            token_ids,
            NativeTextLayerCachesMut::Gemma(caches),
            &self.matvec,
            scratch,
        )
        .await
        .map_err(|err| BackendError::other(err.to_string()))
    }

    fn make_decode_session(
        &self,
        hidden: Vec<f32>,
        caches: Vec<GemmaLayerCache>,
    ) -> NativeGemmaDecodeSession {
        NativeGemmaDecodeSession {
            hidden,
            caches,
            cache_mirror_cleaner: self.matvec.cache_mirror_cleaner(),
        }
    }

    fn cleanup_cache_mirrors(&self, caches: &[GemmaLayerCache]) {
        if let Some(cleaner) = self.matvec.cache_mirror_cleaner() {
            cleaner.cleanup_cache_mirrors(caches);
        }
    }

    fn hidden<'a>(&self, session: &'a NativeGemmaDecodeSession) -> &'a [f32] {
        session.hidden()
    }

    async fn step(
        &self,
        session: &mut NativeGemmaDecodeSession,
        token_id: usize,
        scratch: &mut InferenceScratchpad,
    ) -> Result<(), BackendError> {
        session
            .step(&self.store, &self.spec, &self.matvec, token_id, scratch)
            .await
    }

    async fn next_token_from_hidden(
        &self,
        hidden: &[f32],
        sampling: SamplingConfig,
        sampling_draw: Option<f32>,
        sampling_scratch: &mut llm_sampler::TopPSamplerScratch,
    ) -> Result<usize, BackendError> {
        NativeTextNextTokenContext {
            store: &self.store,
            spec: (&self.spec).into(),
            top_k: self.top_k,
            chunk_rows: self.chunk_rows,
            matvec: &self.matvec,
            family_display_name: "Gemma",
        }
        .select_next_token(hidden, sampling, sampling_draw, sampling_scratch)
        .await
    }
}

fn native_gemma_stop_tokens() -> NativeTextStopTokens {
    NativeTextStopTokens {
        token_ids: &[1],
        token_strings: &["<eos>", "<turn|>"],
        encoded_token_strings: &[],
    }
}

pub(crate) struct NativeGemmaDecodeSession {
    hidden: Vec<f32>,
    caches: Vec<GemmaLayerCache>,
    cache_mirror_cleaner: Option<Arc<dyn NativeTextCacheMirrorCleaner<GemmaLayerCache>>>,
}

impl NativeGemmaDecodeSession {
    fn hidden(&self) -> &[f32] {
        &self.hidden
    }

    async fn step(
        &mut self,
        store: &SafeTensorShardStore,
        spec: &GemmaModelSpec,
        matvec: &impl NativeMatvecBackend,
        token_id: usize,
        scratch: &mut InferenceScratchpad,
    ) -> Result<(), BackendError> {
        self.hidden = native_decode_token_with_cache_for_spec_ref(
            store,
            spec.into(),
            token_id,
            NativeTextLayerCachesMut::Gemma(&mut self.caches),
            matvec,
            scratch,
        )
        .await
        .map_err(|err| BackendError::other(err.to_string()))?;
        Ok(())
    }
}

impl Drop for NativeGemmaDecodeSession {
    fn drop(&mut self) {
        if let Some(cleaner) = &self.cache_mirror_cleaner {
            cleaner.cleanup_cache_mirrors(&self.caches);
        }
    }
}

async fn native_gemma_metadata(
    model_id: &str,
    snapshot_path: &Path,
) -> anyhow::Result<BackendModelMetadata> {
    let manifest_path = snapshot_path.join("llm-engine-manifest.json");
    let mut metadata =
        BackendModelMetadata::new(model_id.to_owned(), "native-gemma").with_family("gemma");
    let Some(manifest_bytes) = crate::fs_util::read_optional_bytes(&manifest_path).await? else {
        return Ok(metadata);
    };
    let manifest = serde_json::from_slice::<SnapshotManifest>(&manifest_bytes)?;
    if manifest.family != "gemma" {
        anyhow::bail!(
            "native Gemma backend only supports family `gemma`, not `{}`",
            manifest.family
        );
    }
    if manifest.loader != "native-metal" {
        anyhow::bail!(
            "native Gemma backend only supports loader `native-metal`, not `{}`",
            manifest.loader
        );
    }
    metadata.family = Some(manifest.family.clone());
    metadata.quantization = Some(manifest.quantization.clone());
    metadata.repo_id = Some(manifest.repo_id.clone());
    metadata.resolved_commit = Some(manifest.resolved_commit.clone());
    metadata.profile = Some(manifest.profile.clone());
    Ok(metadata)
}

async fn reject_native_gemma_quantized_snapshot(snapshot_path: &Path) -> anyhow::Result<()> {
    let config_path = snapshot_path.join("config.json");
    let Some(config_json) = crate::fs_util::read_optional_string(&config_path).await? else {
        return Ok(());
    };
    let value: Value = serde_json::from_str(&config_json)?;
    let quantization = value
        .get("quantization")
        .or_else(|| value.get("quantization_config"));
    if let Some(quantization) = quantization {
        anyhow::bail!(
            "native Gemma execution currently supports BF16 safetensors, not quantized Gemma weights ({quantization})"
        );
    }
    Ok(())
}

#[async_trait]
impl ModelBackend for NativeGemmaBackend {
    fn model_id(&self) -> &str {
        self.driver.model_id()
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        self.driver.model_metadata()
    }

    async fn generate(&self, request: BackendRequest) -> Result<BackendOutput, BackendError> {
        self.generate_with_cancel(request, CancellationToken::new())
            .await
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        self.driver
            .generate_with_cancel(request, cancellation)
            .await
    }

    fn generate_stream<'a>(
        &'a self,
        request: BackendRequest,
    ) -> BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        self.generate_stream_with_cancel(request, CancellationToken::new())
    }

    fn generate_stream_with_cancel<'a>(
        &'a self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        self.driver
            .generate_stream_with_cancel(request, cancellation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_text::NativeTextStopTokens;
    use crate::sync_ext::FailPoisonedMutex;
    use llm_backend::{BackendCacheContext, BackendToolChoice, LayerKvCache};
    use llm_models::{GemmaFamilyAdapter, ModelFamilyAdapter};
    use serde_json::json;
    use std::{
        path::PathBuf,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        time::{SystemTime, UNIX_EPOCH},
    };

    #[derive(Default)]
    struct TestGemmaCacheMirrorCleaner {
        calls: AtomicUsize,
        cache_count: AtomicUsize,
    }

    impl NativeTextCacheMirrorCleaner<GemmaLayerCache> for TestGemmaCacheMirrorCleaner {
        fn cleanup_cache_mirrors(&self, caches: &[GemmaLayerCache]) {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.cache_count.fetch_add(caches.len(), Ordering::SeqCst);
        }
    }

    fn open_gemma_backend(model_id: &str, snapshot: &Path) -> NativeGemmaBackend {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime");
        rt.block_on(NativeGemmaBackend::open(model_id, snapshot))
            .expect("backend opens snapshot")
    }

    fn open_gemma_backend_with_options(
        model_id: &str,
        snapshot: &Path,
        options: NativeGemmaLoadOptions,
    ) -> NativeGemmaBackend {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime");
        rt.block_on(NativeGemmaBackend::open_with_options(
            model_id, snapshot, options,
        ))
        .expect("backend opens snapshot")
    }

    #[test]
    fn native_gemma_stop_tokens_include_eos_and_turn_literals() {
        let stop_tokens = native_gemma_stop_tokens();
        assert_eq!(
            stop_tokens,
            NativeTextStopTokens {
                token_ids: &[1],
                token_strings: &["<eos>", "<turn|>"],
                encoded_token_strings: &[],
            }
        );

        let tokenizer_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/qwen36")
            .join("tokenizer.json");
        let resolved = stop_tokens.resolve(
            &HuggingFaceTokenizer::from_file(tokenizer_path).expect("fixture tokenizer loads"),
        );
        assert!(resolved.contains(1));
        assert!(!resolved.contains(0));
    }

    #[test]
    fn native_gemma_backend_runs_tiny_prefill_and_selects_tied_lm_head_token() {
        let snapshot = temp_snapshot_dir("native-gemma-prefill");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        write_tiny_gemma4_decoder_snapshot(&snapshot);
        copy_qwen_tokenizer(snapshot.join("tokenizer.json"));

        let backend = open_gemma_backend("local-gemma", &snapshot);
        let decode = backend
            .start_decode_session(
                &[0, 1],
                4,
                &native_gemma_test_request("local-gemma"),
                &CancellationToken::new(),
            )
            .expect("tiny Gemma prefill runs");
        let candidate = backend
            .next_token_from_hidden(decode.hidden(), SamplingConfig::Greedy)
            .expect("tied lm head selects a token");

        assert_eq!(candidate, 1);
        assert_eq!(backend.model_metadata().backend, "native-gemma");
        assert_eq!(backend.model_metadata().family.as_deref(), Some("gemma"));
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn native_gemma_decode_session_cleans_cache_mirrors_on_drop() {
        let cleaner = Arc::new(TestGemmaCacheMirrorCleaner::default());
        let session_cleaner: Arc<dyn NativeTextCacheMirrorCleaner<GemmaLayerCache>> =
            cleaner.clone();

        {
            let cache = GemmaLayerCache::Attention(
                LayerKvCache::new(1, 1, 1).expect("test cache shape is valid"),
            );
            let _session = NativeGemmaDecodeSession {
                hidden: vec![0.0],
                caches: vec![cache],
                cache_mirror_cleaner: Some(session_cleaner),
            };
        }

        assert_eq!(cleaner.calls.load(Ordering::SeqCst), 1);
        assert_eq!(cleaner.cache_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn native_gemma_prefix_cache_reuses_longest_compatible_prefix() {
        let cache = NativeGemmaPrefixCache::new(10_000);
        let metrics = NativeGemmaPrefixCacheMetrics::default();
        let namespace = native_gemma_test_prefix_namespace("base");
        let mut layer_cache = LayerKvCache::new(4, 1, 2).expect("cache shape is valid");
        layer_cache
            .append(&[1.0, 2.0], &[3.0, 4.0])
            .expect("token fits");
        let original_cache_id = layer_cache.id();
        let caches = vec![GemmaLayerCache::Attention(layer_cache)];

        cache.store(namespace.clone(), &[1, 2], &[0.25, 0.75], &caches, &metrics);

        let hit = cache
            .lookup(&namespace, &[1, 2, 3], &metrics)
            .expect("compatible longer prompt reuses stored prefix");
        assert_eq!(hit.token_count, 2);
        assert_eq!(hit.hidden, vec![0.25, 0.75]);
        match &hit.caches[0] {
            GemmaLayerCache::Attention(cache) => {
                assert_ne!(cache.id(), original_cache_id);
                assert_eq!(cache.token_count(), 1);
            }
        }

        let incompatible_namespace = NativeTextPrefixCacheNamespace {
            tool_schema: Some("different-tool-schema".to_owned()),
            ..namespace.clone()
        };
        assert!(
            cache
                .lookup(&incompatible_namespace, &[1, 2], &metrics)
                .is_none(),
            "tool schema changes must not reuse prefix state"
        );
    }

    #[test]
    fn native_gemma_prefix_cache_separates_capacity_manifest_profile_and_required_tool_name() {
        let cache = NativeGemmaPrefixCache::new(10_000);
        let metrics = NativeGemmaPrefixCacheMetrics::default();
        let namespace = native_gemma_test_prefix_namespace("namespace-policy");
        let larger_capacity_namespace = NativeTextPrefixCacheNamespace {
            cache_tokens: namespace.cache_tokens * 2,
            ..namespace.clone()
        };
        let different_manifest_namespace = NativeTextPrefixCacheNamespace {
            resolved_commit: Some("fedcba9876543210fedcba9876543210fedcba98".to_owned()),
            ..namespace.clone()
        };
        let different_profile_namespace = NativeTextPrefixCacheNamespace {
            profile: Some("gemma-other-profile".to_owned()),
            ..namespace.clone()
        };
        let lookup_required_tool = NativeTextPrefixCacheNamespace {
            request_mode: format!(
                "chat,json_object=false,required_tool={:?}",
                BackendToolChoice::RequiredFunction("lookup".to_owned())
            ),
            ..namespace.clone()
        };
        let search_required_tool = NativeTextPrefixCacheNamespace {
            request_mode: format!(
                "chat,json_object=false,required_tool={:?}",
                BackendToolChoice::RequiredFunction("search".to_owned())
            ),
            ..namespace.clone()
        };

        cache.store(namespace.clone(), &[1, 2], &[0.25, 0.75], &[], &metrics);
        cache.store(
            lookup_required_tool.clone(),
            &[1, 2],
            &[0.25, 0.75],
            &[],
            &metrics,
        );

        assert!(
            cache
                .lookup(&larger_capacity_namespace, &[1, 2], &metrics)
                .is_none(),
            "cache capacity changes must not reuse Gemma prefix state"
        );
        assert!(
            cache
                .lookup(&different_manifest_namespace, &[1, 2], &metrics)
                .is_none(),
            "manifest identity changes must not reuse Gemma prefix state"
        );
        assert!(
            cache
                .lookup(&different_profile_namespace, &[1, 2], &metrics)
                .is_none(),
            "profile changes must not reuse Gemma prefix state"
        );
        assert!(
            cache
                .lookup(&search_required_tool, &[1, 2], &metrics)
                .is_none(),
            "required tool-choice names must not reuse Gemma prefix state"
        );
    }

    #[test]
    fn native_gemma_prefix_cache_evicts_lru_entries_to_fit_budget() {
        let cache = NativeGemmaPrefixCache::new(40);
        let metrics = NativeGemmaPrefixCacheMetrics::default();
        let namespace = native_gemma_test_prefix_namespace("eviction");
        let hidden = vec![1.0; 8];

        cache.store(namespace.clone(), &[1], &hidden, &[], &metrics);
        cache.store(namespace.clone(), &[2], &hidden, &[], &metrics);

        assert!(
            cache.lookup(&namespace, &[1], &metrics).is_none(),
            "oldest entry should be evicted"
        );
        assert!(
            cache.lookup(&namespace, &[2], &metrics).is_some(),
            "newest entry should remain resident"
        );
        let inner = cache.inner.lock_or_panic("native Gemma prefix cache");
        assert_eq!(inner.entries.len(), 1);
        assert_eq!(inner.used_bytes, 32);
    }

    #[test]
    fn native_gemma_prefix_cache_metrics_expose_hits_misses_and_evictions() {
        let metrics = NativeGemmaPrefixCacheMetrics::default();

        metrics.record_hit(3);
        metrics.record_miss();
        metrics.record_store(32);
        metrics.record_eviction(16);
        metrics.record_rejected();
        metrics.record_residency(32, 1);
        metrics.record_lookup_scan(5, 4);
        metrics.record_hit_clone_bytes(64);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot["hits"], 1);
        assert_eq!(snapshot["misses"], 1);
        assert_eq!(snapshot["stores"], 1);
        assert_eq!(snapshot["evictions"], 1);
        assert_eq!(snapshot["rejected"], 1);
        assert_eq!(snapshot["reused_tokens"], 3);
        assert_eq!(snapshot["bytes_stored"], 32);
        assert_eq!(snapshot["bytes_evicted"], 16);
        assert_eq!(snapshot["resident_bytes"], 32);
        assert_eq!(snapshot["resident_entries"], 1);
        assert_eq!(snapshot["entries_scanned"], 5);
        assert_eq!(snapshot["namespace_entries_scanned"], 4);
        assert_eq!(snapshot["hit_clone_bytes"], 64);
    }

    #[test]
    fn native_gemma_backend_rejects_quantized_mlx_snapshot_explicitly() {
        let snapshot = temp_snapshot_dir("native-gemma-quantized");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        write_tiny_gemma4_decoder_snapshot(&snapshot);
        copy_qwen_tokenizer(snapshot.join("tokenizer.json"));
        let mut config = serde_json::from_str::<serde_json::Value>(
            &std::fs::read_to_string(snapshot.join("config.json")).expect("config"),
        )
        .expect("config json");
        config["quantization"] = json!({"bits": 4, "group_size": 64, "mode": "affine"});
        std::fs::write(snapshot.join("config.json"), config.to_string()).expect("config");

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime");
        let err = match rt.block_on(NativeGemmaBackend::open("local-gemma", &snapshot)) {
            Err(err) => err,
            Ok(_) => panic!("quantized native Gemma fails explicitly"),
        };

        assert!(err.to_string().contains("not quantized Gemma weights"));
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn native_gemma_backend_accepts_native_metal_cache_options_with_cpu_fallback() {
        let snapshot = temp_snapshot_dir("native-gemma-metal-options");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        write_tiny_gemma4_decoder_snapshot(&snapshot);
        copy_qwen_tokenizer(snapshot.join("tokenizer.json"));

        let backend = open_gemma_backend_with_options(
            "local-gemma",
            &snapshot,
            NativeGemmaLoadOptions {
                metal_weight_cache_bytes: Some(0),
                warm_metal_weight_cache: true,
                ..NativeGemmaLoadOptions::default()
            },
        );

        assert_eq!(backend.model_metadata().backend, "native-gemma");
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[tokio::test]
    #[ignore = "set KIR_AI_GEMMA_BF16_SNAPSHOT to a local BF16 Gemma 4 text snapshot"]
    async fn native_gemma_real_bf16_snapshot_smoke_generates_one_token() {
        let snapshot = std::env::var_os("KIR_AI_GEMMA_BF16_SNAPSHOT")
            .map(PathBuf::from)
            .expect("KIR_AI_GEMMA_BF16_SNAPSHOT must point at a local BF16 Gemma 4 snapshot");
        let backend = NativeGemmaBackend::open("local-gemma", &snapshot)
            .await
            .expect("real BF16 Gemma snapshot opens")
            .with_max_new_tokens(1);
        let output = backend
            .generate(BackendRequest::raw_completion(
                "local-gemma",
                "Hello",
                Some(1),
                SamplingConfig::Greedy,
            ))
            .await
            .expect("real BF16 Gemma snapshot generates");

        assert!(output.prompt_tokens > 0);
        assert!(output.completion_tokens <= 1);
    }

    fn native_gemma_test_request(model: &str) -> BackendRequest {
        BackendRequest::raw_completion(model, "hello", Some(1), SamplingConfig::Greedy)
    }

    fn native_gemma_test_prefix_namespace(label: &str) -> NativeTextPrefixCacheNamespace {
        NativeTextPrefixCacheNamespace {
            model_id: format!("model-{label}"),
            backend: "native-gemma".to_owned(),
            family: Some("gemma".to_owned()),
            quantization: Some("bf16".to_owned()),
            repo_id: Some("local/test".to_owned()),
            resolved_commit: Some("0123456789abcdef0123456789abcdef01234567".to_owned()),
            profile: Some("gemma-test".to_owned()),
            cache_key: BackendCacheContext::chat_template_with_kwargs(
                GemmaFamilyAdapter.cache_template_id(),
                Some("tool-schema-v1".to_owned()),
                GemmaFamilyAdapter
                    .chat_template_kwargs_json()
                    .map(str::to_owned),
            )
            .key
            .as_str()
            .to_owned(),
            tool_schema: Some("tool-schema-v1".to_owned()),
            request_mode: "chat,json_object=false,required_tool=None".to_owned(),
            cache_layout_version: NATIVE_GEMMA_PREFIX_CACHE_LAYOUT_VERSION,
            cache_tokens: 8,
            max_prefill_tokens: 8,
        }
    }

    fn write_tiny_gemma4_decoder_snapshot(root: &Path) {
        std::fs::write(
            root.join("config.json"),
            json!({
                "architectures": ["Gemma4ForConditionalGeneration"],
                "model_type": "gemma4",
                "text_config": {
                    "attention_bias": false,
                    "attention_dropout": 0.0,
                    "attention_k_eq_v": false,
                    "bos_token_id": 2,
                    "dtype": "bfloat16",
                    "enable_moe_block": false,
                    "global_head_dim": null,
                    "head_dim": 2,
                    "hidden_activation": "gelu_pytorch_tanh",
                    "hidden_size": 2,
                    "hidden_size_per_layer_input": 0,
                    "intermediate_size": 1,
                    "layer_types": ["sliding_attention"],
                    "max_position_embeddings": 8,
                    "model_type": "gemma4_text",
                    "num_attention_heads": 1,
                    "num_global_key_value_heads": null,
                    "num_hidden_layers": 1,
                    "num_key_value_heads": 1,
                    "num_kv_shared_layers": 0,
                    "rms_norm_eps": 1e-6,
                    "rope_parameters": {
                        "full_attention": {"partial_rotary_factor": 1.0, "rope_theta": 10000.0},
                        "sliding_attention": {"rope_theta": 10000.0}
                    },
                    "sliding_window": 2,
                    "tie_word_embeddings": true,
                    "use_double_wide_mlp": false,
                    "vocab_size": 3,
                    "vocab_size_per_layer_input": 3
                },
                "tie_word_embeddings": true
            })
            .to_string(),
        )
        .expect("config");
        let tensors = [
            (
                "model.language_model.embed_tokens.weight",
                vec![3, 2],
                vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0],
            ),
            ("model.language_model.norm.weight", vec![2], vec![1.0, 1.0]),
            (
                "model.language_model.layers.0.input_layernorm.weight",
                vec![2],
                vec![1.0, 1.0],
            ),
            (
                "model.language_model.layers.0.self_attn.q_proj.weight",
                vec![2, 2],
                vec![0.0, 0.0, 0.0, 0.0],
            ),
            (
                "model.language_model.layers.0.self_attn.k_proj.weight",
                vec![2, 2],
                vec![0.0, 0.0, 0.0, 0.0],
            ),
            (
                "model.language_model.layers.0.self_attn.v_proj.weight",
                vec![2, 2],
                vec![0.0, 0.0, 0.0, 0.0],
            ),
            (
                "model.language_model.layers.0.self_attn.q_norm.weight",
                vec![2],
                vec![1.0, 1.0],
            ),
            (
                "model.language_model.layers.0.self_attn.k_norm.weight",
                vec![2],
                vec![1.0, 1.0],
            ),
            (
                "model.language_model.layers.0.self_attn.o_proj.weight",
                vec![2, 2],
                vec![0.0, 0.0, 0.0, 0.0],
            ),
            (
                "model.language_model.layers.0.post_attention_layernorm.weight",
                vec![2],
                vec![1.0, 1.0],
            ),
            (
                "model.language_model.layers.0.pre_feedforward_layernorm.weight",
                vec![2],
                vec![1.0, 1.0],
            ),
            (
                "model.language_model.layers.0.mlp.gate_proj.weight",
                vec![1, 2],
                vec![0.0, 0.0],
            ),
            (
                "model.language_model.layers.0.mlp.up_proj.weight",
                vec![1, 2],
                vec![0.0, 0.0],
            ),
            (
                "model.language_model.layers.0.mlp.down_proj.weight",
                vec![2, 1],
                vec![0.0, 0.0],
            ),
            (
                "model.language_model.layers.0.post_feedforward_layernorm.weight",
                vec![2],
                vec![1.0, 1.0],
            ),
            (
                "model.language_model.layers.0.layer_scalar",
                vec![1],
                vec![1.0],
            ),
        ];
        let safetensors = tiny_owned_multi_safetensors_bf16(&tensors);
        std::fs::write(root.join("model.safetensors"), &safetensors).expect("safetensors");
        let weight_map = tensors
            .iter()
            .map(|(tensor, _, _)| {
                (
                    (*tensor).to_owned(),
                    serde_json::Value::String("model.safetensors".to_owned()),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        std::fs::write(
            root.join("model.safetensors.index.json"),
            json!({
                "metadata": {"total_size": safetensors.len()},
                "weight_map": weight_map
            })
            .to_string(),
        )
        .expect("index");
    }

    fn tiny_owned_multi_safetensors_bf16(tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape, values) in tensors {
            let start = data.len();
            for value in values {
                data.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
            }
            let end = data.len();
            header.insert(
                (*name).to_owned(),
                json!({
                    "dtype": "BF16",
                    "shape": shape,
                    "data_offsets": [start, end]
                }),
            );
        }
        let header = serde_json::Value::Object(header).to_string();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&data);
        bytes
    }

    fn copy_qwen_tokenizer(destination: impl AsRef<Path>) {
        let source = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/qwen36")
            .join("tokenizer.json");
        std::fs::copy(&source, destination).expect("copy tokenizer");
    }

    fn temp_snapshot_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!("kir-ai-{name}-{nanos}"))
    }
}
