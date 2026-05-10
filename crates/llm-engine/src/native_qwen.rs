use crate::{
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
    InferenceScratchpad, ModelBackend, NativeMatvecBackend, NativeTextLayerCachesMut,
    QwenLayerCache, SafeTensorShardStore, SamplingConfig,
    native_decode_token_with_cache_for_spec_ref_with_matvec,
    native_prefill_sequence_with_cache_for_spec_ref_with_matvec, qwen_layer_caches_for_spec,
};
use llm_hub::SnapshotManifest;
use llm_models::QwenModelSpec;
use llm_tokenizer::HuggingFaceTokenizer;
use serde_json::Value;
use std::{
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
};
use tokio_util::sync::CancellationToken;

pub const DEFAULT_NATIVE_QWEN_MAX_NEW_TOKENS: u32 = 256;

#[derive(Clone)]
pub struct NativeQwenBackend {
    driver: NativeTextDriver<NativeQwenAdapter>,
}

#[derive(Clone)]
pub(crate) struct NativeQwenAdapter {
    model_id: String,
    metadata: BackendModelMetadata,
    spec: QwenModelSpec,
    store: SafeTensorShardStore,
    matvec: NativeTextMatvecBackend,
    max_prefill_tokens: usize,
    top_k: usize,
    chunk_rows: usize,
    prefix_cache: Arc<NativeQwenPrefixCache>,
}

const DEFAULT_NATIVE_QWEN_PREFIX_CACHE_BYTES: u64 = 512 * 1024 * 1024;
const NATIVE_QWEN_PREFIX_CACHE_LAYOUT_VERSION: u32 = 1;

type NativeQwenPrefixCache = NativeTextPrefixCache<QwenLayerCache>;
#[cfg(test)]
type NativeQwenPrefixCacheNamespace = NativeTextPrefixCacheNamespace;
type NativeQwenPrefixCacheMetrics = NativeTextPrefixCacheMetrics;

fn native_qwen_prefix_cache_metrics() -> &'static NativeQwenPrefixCacheMetrics {
    static METRICS: OnceLock<NativeQwenPrefixCacheMetrics> = OnceLock::new();
    METRICS.get_or_init(NativeQwenPrefixCacheMetrics::default)
}

fn native_qwen_prefix_entry_bytes(hidden: &[f32], caches: &[QwenLayerCache]) -> u64 {
    let hidden_bytes = std::mem::size_of_val(hidden) as u64;
    caches.iter().fold(hidden_bytes, |total, cache| {
        total.saturating_add(match cache {
            QwenLayerCache::Full(cache) => {
                ((cache.key_storage().len() + cache.value_storage().len())
                    * std::mem::size_of::<f32>()) as u64
            }
            QwenLayerCache::Linear(cache) => {
                ((cache.conv_window().len() + cache.recurrent_state().len())
                    * std::mem::size_of::<f32>()) as u64
            }
        })
    })
}

impl NativeTextPrefixCacheValue for QwenLayerCache {
    fn prefix_cache_entry_bytes(hidden: &[f32], caches: &[Self]) -> u64 {
        native_qwen_prefix_entry_bytes(hidden, caches)
    }
}

impl NativeTextCacheMirrorSource for QwenLayerCache {
    fn append_cache_mirror_ids(&self, ids: &mut NativeTextCacheMirrorIds) {
        match self {
            QwenLayerCache::Full(cache) => ids.push_kv(cache.id()),
            QwenLayerCache::Linear(cache) => ids.push_linear(cache.id()),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativeQwenLoadOptions {
    pub eager_materialize_shards: bool,
    pub metal_weight_cache_bytes: Option<u64>,
    pub warm_metal_weight_cache: bool,
}
impl NativeQwenBackend {
    pub async fn open(
        model_id: impl Into<String>,
        snapshot_path: impl AsRef<Path>,
    ) -> anyhow::Result<Self> {
        Self::open_with_options(model_id, snapshot_path, NativeQwenLoadOptions::default()).await
    }

    pub async fn open_with_options(
        model_id: impl Into<String>,
        snapshot_path: impl AsRef<Path>,
        options: NativeQwenLoadOptions,
    ) -> anyhow::Result<Self> {
        let model_id = model_id.into();
        let snapshot_path = snapshot_path.as_ref();
        let cache_namespace = snapshot_path.canonicalize()?.to_string_lossy().into_owned();
        let config_json = std::fs::read_to_string(snapshot_path.join("config.json"))?;
        let metadata = native_qwen_metadata(&model_id, snapshot_path)?;
        let store = SafeTensorShardStore::open(snapshot_path)?;
        let spec = QwenModelSpec::from_config_json(&config_json)?;
        store.index().validate_qwen_text_weights(&spec)?;
        if options.eager_materialize_shards {
            let materialized_bytes = store.materialize_all_shards()?;
            tracing::info!(
                materialized_bytes,
                "materialized native Qwen safetensors shards"
            );
        }
        let matvec = NativeTextMatvecBackend::system_default(
            native_qwen_metal_weight_cache_bytes(options.metal_weight_cache_bytes),
            &cache_namespace,
        );
        if options.warm_metal_weight_cache {
            let warmup = matvec.warm_bf16_matrix_cache(&store).await.map_err(|err| {
                anyhow::anyhow!("native Qwen Metal weight cache warm-up failed: {err}")
            })?;
            tracing::info!(
                candidates = warmup.candidates,
                warmed = warmup.warmed,
                already_resident = warmup.already_resident,
                skipped_budget = warmup.skipped_budget,
                skipped_non_metal = warmup.skipped_non_metal,
                "native Qwen Metal BF16 weight cache warm-up complete"
            );
        }
        let tokenizer = HuggingFaceTokenizer::from_file(snapshot_path.join("tokenizer.json"))?;
        let adapter = NativeQwenAdapter {
            model_id: model_id.clone(),
            metadata: metadata.clone(),
            spec,
            store,
            matvec,
            max_prefill_tokens: 32,
            top_k: 16,
            chunk_rows: 2048,
            prefix_cache: Arc::new(NativeQwenPrefixCache::new(
                DEFAULT_NATIVE_QWEN_PREFIX_CACHE_BYTES,
            )),
        };
        Ok(Self {
            driver: NativeTextDriver::new(
                model_id,
                metadata,
                tokenizer,
                adapter,
                DEFAULT_NATIVE_QWEN_MAX_NEW_TOKENS,
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

    pub(crate) fn into_driver(self) -> NativeTextDriver<NativeQwenAdapter> {
        self.driver
    }

    #[cfg(test)]
    fn generate_blocking_stream(
        &self,
        request: BackendRequest,
        tx: tokio::sync::mpsc::Sender<Result<BackendStreamChunk, BackendError>>,
        cancellation: CancellationToken,
    ) -> Result<(), BackendError> {
        self.driver
            .generate_blocking_stream(request, tx, cancellation)
    }

    #[cfg(test)]
    fn start_decode_session(
        &self,
        context_tokens: &[usize],
        max_new_tokens: u32,
        request: &BackendRequest,
        cancellation: &CancellationToken,
    ) -> Result<NativeQwenDecodeSession, BackendError> {
        let driver = &self.driver;
        tokio::task::block_in_place(|| {
            driver.runtime().block_on(driver.start_decode_session(
                context_tokens,
                max_new_tokens,
                request,
                cancellation,
                &mut InferenceScratchpad::new(),
            ))
        })
    }

    #[cfg(test)]
    fn next_token_from_hidden(
        &self,
        hidden: &[f32],
        sampling: SamplingConfig,
    ) -> Result<NativeQwenCandidate, BackendError> {
        let token_id = tokio::task::block_in_place(|| {
            self.driver
                .runtime()
                .block_on(self.driver.adapter.next_token_from_hidden(
                    hidden,
                    sampling,
                    &mut InferenceScratchpad::new(),
                ))
        })?;
        Ok(NativeQwenCandidate { token_id })
    }
}

#[async_trait]
impl NativeTextAdapter for NativeQwenAdapter {
    type DecodeSession = NativeQwenDecodeSession;
    type LayerCache = QwenLayerCache;

    fn family_display_name(&self) -> &'static str {
        "Qwen"
    }

    fn worker_label(&self) -> &'static str {
        "native Qwen"
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
            .encode(&request.prompt, false)
            .map_err(|err| BackendError::Other(err.to_string()))
    }

    fn decode_output(
        &self,
        tokenizer: &HuggingFaceTokenizer,
        output_ids: &[u32],
    ) -> Result<String, BackendError> {
        tokenizer
            .decode(output_ids, false)
            .map_err(|err| BackendError::Other(err.to_string()))
    }

    fn stop_tokens(&self) -> NativeTextStopTokens {
        NativeTextStopTokens {
            token_ids: &[],
            token_strings: &["<|im_end|>"],
        }
    }

    fn max_position_embeddings(&self) -> u32 {
        self.spec.max_position_embeddings
    }

    fn max_prefill_tokens(&self) -> usize {
        self.max_prefill_tokens
    }

    fn prefix_cache(&self) -> &NativeTextPrefixCache<QwenLayerCache> {
        &self.prefix_cache
    }

    fn prefix_cache_metrics(&self) -> &NativeTextPrefixCacheMetrics {
        native_qwen_prefix_cache_metrics()
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
            cache_layout_version: NATIVE_QWEN_PREFIX_CACHE_LAYOUT_VERSION,
            cache_tokens,
            max_prefill_tokens: self.max_prefill_tokens,
        })
    }

    fn layer_count(&self) -> usize {
        self.spec.num_hidden_layers as usize
    }

    fn allocate_caches(&self, cache_tokens: usize) -> Result<Vec<QwenLayerCache>, BackendError> {
        qwen_layer_caches_for_spec(&self.spec, cache_tokens)
            .map_err(|err| BackendError::Other(err.to_string()))
    }

    async fn prefill_chunk_with_cache(
        &self,
        token_ids: &[usize],
        caches: &mut [QwenLayerCache],
        scratch: &mut InferenceScratchpad,
    ) -> Result<Vec<Vec<f32>>, BackendError> {
        native_prefill_sequence_with_cache_for_spec_ref_with_matvec(
            &self.store,
            (&self.spec).into(),
            token_ids,
            NativeTextLayerCachesMut::Qwen(caches),
            &self.matvec,
            scratch,
        )
        .await
        .map_err(|err| BackendError::Other(err.to_string()))
    }

    fn make_decode_session(
        &self,
        hidden: Vec<f32>,
        caches: Vec<QwenLayerCache>,
    ) -> NativeQwenDecodeSession {
        NativeQwenDecodeSession {
            hidden,
            caches,
            cache_mirror_cleaner: self.matvec.cache_mirror_cleaner(),
        }
    }

    fn cleanup_cache_mirrors(&self, caches: &[QwenLayerCache]) {
        if let Some(cleaner) = self.matvec.cache_mirror_cleaner() {
            cleaner.cleanup_cache_mirrors(caches);
        }
    }

    fn hidden<'a>(&self, session: &'a NativeQwenDecodeSession) -> &'a [f32] {
        session.hidden()
    }

    async fn step(
        &self,
        session: &mut NativeQwenDecodeSession,
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
        _scratch: &mut InferenceScratchpad,
    ) -> Result<usize, BackendError> {
        NativeTextNextTokenContext {
            store: &self.store,
            spec: (&self.spec).into(),
            top_k: self.top_k,
            chunk_rows: self.chunk_rows,
            matvec: &self.matvec,
            family_display_name: "Qwen",
        }
        .select_next_token(hidden, sampling)
        .await
    }
}

pub(crate) struct NativeQwenDecodeSession {
    hidden: Vec<f32>,
    caches: Vec<QwenLayerCache>,
    cache_mirror_cleaner: Option<Arc<dyn NativeTextCacheMirrorCleaner<QwenLayerCache>>>,
}

impl NativeQwenDecodeSession {
    fn hidden(&self) -> &[f32] {
        &self.hidden
    }

    async fn step(
        &mut self,
        store: &SafeTensorShardStore,
        spec: &QwenModelSpec,
        matvec: &impl NativeMatvecBackend,
        token_id: usize,
        scratch: &mut InferenceScratchpad,
    ) -> Result<(), BackendError> {
        self.hidden = native_decode_token_with_cache_for_spec_ref_with_matvec(
            store,
            spec.into(),
            token_id,
            NativeTextLayerCachesMut::Qwen(&mut self.caches),
            matvec,
            scratch,
        )
        .await
        .map_err(|err| BackendError::Other(err.to_string()))?;
        Ok(())
    }
}

impl Drop for NativeQwenDecodeSession {
    fn drop(&mut self) {
        if let Some(cleaner) = &self.cache_mirror_cleaner {
            cleaner.cleanup_cache_mirrors(&self.caches);
        }
    }
}

#[cfg(test)]
fn resolve_native_max_tokens(
    requested: Option<u32>,
    configured_max: u32,
) -> Result<u32, BackendError> {
    crate::native_text::resolve_native_text_max_tokens(requested, configured_max, "Qwen")
}

#[cfg(test)]
#[derive(Debug, Clone)]
struct NativeQwenCandidate {
    token_id: usize,
}

#[async_trait]
impl ModelBackend for NativeQwenBackend {
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

fn native_qwen_metadata(
    model_id: &str,
    snapshot_path: &Path,
) -> anyhow::Result<BackendModelMetadata> {
    let manifest_path = snapshot_path.join("llm-engine-manifest.json");
    let mut metadata =
        BackendModelMetadata::new(model_id.to_owned(), "native-qwen").with_family("qwen");
    metadata.loader = Some("native-metal".to_owned());
    metadata.snapshot_path = Some(PathBuf::from(snapshot_path));
    let manifest_bytes = match std::fs::read(&manifest_path) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(metadata),
        Err(err) => return Err(err.into()),
    };
    let manifest = serde_json::from_slice::<SnapshotManifest>(&manifest_bytes)?;
    if manifest.family != "qwen" {
        anyhow::bail!(
            "native Qwen backend only supports family `qwen`, not `{}`",
            manifest.family
        );
    }
    if manifest.loader != "native-metal" {
        anyhow::bail!(
            "native Qwen backend only supports loader `native-metal`, not `{}`",
            manifest.loader
        );
    }
    metadata.family = Some(manifest.family.clone());
    metadata.loader = Some(manifest.loader.clone());
    metadata.quantization = Some(manifest.quantization.clone());
    metadata.repo_id = Some(manifest.repo_id.clone());
    metadata.resolved_commit = Some(manifest.resolved_commit.clone());
    metadata.profile = Some(manifest.profile.clone());
    metadata.manifest_digest = Some(manifest.digest());
    Ok(metadata)
}

fn native_qwen_metal_weight_cache_bytes(configured: Option<u64>) -> u64 {
    native_text_metal_weight_cache_bytes(configured)
}

#[cfg(test)]
fn native_qwen_warmable_bf16_matrix_tensors(
    store: &SafeTensorShardStore,
) -> Result<
    Vec<crate::native_matvec::NativeTextWarmableBf16MatrixTensor>,
    llm_backend::TensorLoadError,
> {
    crate::native_matvec::native_text_warmable_bf16_matrix_tensors(store)
}

pub(crate) fn native_qwen_prefix_cache_metrics_snapshot() -> Value {
    native_qwen_prefix_cache_metrics().snapshot()
}

#[cfg(test)]
mod tests;
