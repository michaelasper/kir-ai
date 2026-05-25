use crate::{
    DEFAULT_NATIVE_TEXT_MAX_PREFILL_TOKENS,
    native_matvec::{
        NativeTextCacheMirrorCleaner, NativeTextCacheMirrorIds, NativeTextCacheMirrorSource,
        NativeTextMatvecBackend,
    },
    native_text::{
        DEFAULT_NATIVE_TEXT_PREFIX_CACHE_BYTES, NativeTextAdapter, NativeTextDiskCache,
        NativeTextDiskCacheIdentity, NativeTextDriver, NativeTextNextTokenContext,
        NativeTextPrefixCache, NativeTextPrefixCacheMetrics, NativeTextPrefixCacheNamespace,
        NativeTextPrefixCacheValue, NativeTextPrefixNamespaceContext, NativeTextRuntimeOptions,
        NativeTextSnapshotOpen, NativeTextSnapshotOpenFamily, NativeTextStopTokens,
        native_text_disk_cache_snapshot_identity, native_text_prefix_namespace,
        open_native_text_snapshot,
    },
};
use crate::{ResolvedSnapshotBackend, SnapshotBackendLoader};
use async_trait::async_trait;
use futures::stream::BoxStream;
use llm_backend::native::{
    InferenceScratchpad, NativeMatvecBackend, NativeTextLayerCachesMut, QwenLayerCache,
    QwenLayerCachePrefixState, SafeTensorShardStore, native_decode_token_with_cache_for_spec_ref,
    native_prefill_sequence_with_cache_for_spec_ref, qwen_layer_caches_for_spec,
    qwen_static_f32_tensors_for_spec,
};
use llm_backend_contracts::{
    BackendError, BackendModelMetadata, BackendOutput, BackendRequest, BackendStreamChunk,
    ModelBackend, SamplingConfig,
};
use llm_models::{ModelFamily, QwenModelSpec, SafetensorsIndex};
use llm_tokenizer::{HuggingFaceTokenizer, HuggingFaceTokenizerIdentity};
use serde_json::Value;
use std::{
    path::Path,
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
    prefix_disk_cache: Option<Arc<NativeTextDiskCache<QwenLayerCache>>>,
}

const DEFAULT_NATIVE_QWEN_PREFIX_CACHE_BYTES: u64 = DEFAULT_NATIVE_TEXT_PREFIX_CACHE_BYTES;
const NATIVE_QWEN_PREFIX_CACHE_LAYOUT_VERSION: u32 = 1;
const NATIVE_QWEN_PREFIX_ADAPTER_SETTINGS: &str = "native-qwen-prefix-adapter/v1";

type NativeQwenPrefixCache = NativeTextPrefixCache<QwenLayerCache>;
#[cfg(test)]
type NativeQwenPrefixCacheNamespace = NativeTextPrefixCacheNamespace;
type NativeQwenPrefixCacheMetrics = NativeTextPrefixCacheMetrics;

fn native_qwen_prefix_cache_metrics() -> &'static NativeQwenPrefixCacheMetrics {
    static METRICS: OnceLock<NativeQwenPrefixCacheMetrics> = OnceLock::new();
    METRICS.get_or_init(NativeQwenPrefixCacheMetrics::default)
}

impl NativeTextPrefixCacheValue for QwenLayerCache {
    type PrefixCacheState = QwenLayerCachePrefixState;

    fn prefix_cache_state(caches: &[Self]) -> Vec<Self::PrefixCacheState> {
        caches
            .iter()
            .map(QwenLayerCache::prefix_cache_state)
            .collect()
    }

    fn prefix_cache_from_state(states: &[Self::PrefixCacheState]) -> Option<Vec<Self>> {
        states
            .iter()
            .map(QwenLayerCache::from_prefix_cache_state)
            .collect::<Result<Vec<_>, _>>()
            .ok()
    }

    fn prefix_cache_entry_bytes(hidden: &[f32], states: &[Self::PrefixCacheState]) -> u64 {
        let hidden_bytes = std::mem::size_of_val(hidden) as u64;
        states.iter().fold(hidden_bytes, |total, state| {
            total.saturating_add(match state {
                QwenLayerCachePrefixState::Full(state) => state.metadata_bytes(),
                QwenLayerCachePrefixState::Linear(state) => {
                    linear_attention_snapshot_bytes(&state.conv_window, &state.recurrent_state)
                }
                _ => 0,
            })
        })
    }
}

fn linear_attention_snapshot_bytes(conv_window: &[f32], recurrent_state: &[f32]) -> u64 {
    std::mem::size_of_val(conv_window).saturating_add(std::mem::size_of_val(recurrent_state)) as u64
}

impl NativeTextCacheMirrorSource for QwenLayerCache {
    fn append_cache_mirror_ids(&self, ids: &mut NativeTextCacheMirrorIds) {
        match self {
            QwenLayerCache::Full(cache) => ids.push_kv_cache(cache),
            QwenLayerCache::Linear(cache) => ids.push_linear(cache.id()),
            _ => {}
        }
    }
}

pub type NativeQwenLoadOptions = NativeTextRuntimeOptions;

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
        let snapshot_path = snapshot_path.as_ref();
        let identity = ResolvedSnapshotBackend::resolve(
            snapshot_path,
            None,
            None,
            SnapshotBackendLoader::NativeMetal,
            false,
            false,
        )
        .await?;
        Self::open_with_snapshot_identity(model_id, snapshot_path, options, identity).await
    }

    pub(crate) async fn open_with_snapshot_identity(
        model_id: impl Into<String>,
        snapshot_path: impl AsRef<Path>,
        options: NativeQwenLoadOptions,
        identity: ResolvedSnapshotBackend,
    ) -> anyhow::Result<Self> {
        let model_id = model_id.into();
        let snapshot_path = snapshot_path.as_ref();
        let metadata = native_qwen_metadata(&model_id, &identity)?;
        let NativeTextSnapshotOpen {
            model_id,
            metadata,
            spec,
            store,
            matvec,
            tokenizer,
            prefix_cache_bytes,
            prefix_disk_cache,
        } = open_native_text_snapshot(
            model_id,
            snapshot_path,
            options,
            metadata,
            NativeTextSnapshotOpenFamily {
                display_name: "Qwen",
                parse_spec: parse_native_qwen_spec,
                validate_text_weights: validate_native_qwen_text_weights,
                static_f32_tensors_for_spec: qwen_static_f32_tensors_for_spec,
            },
        )
        .await?;
        let prefix_disk_cache = match prefix_disk_cache {
            Some(config) => {
                let snapshot_identity = native_text_disk_cache_snapshot_identity(
                    snapshot_path,
                    identity.manifest_digest(),
                )
                .await;
                Some(Arc::new(
                    NativeTextDiskCache::open(
                        config,
                        NativeTextDiskCacheIdentity::from_model_metadata(
                            &metadata,
                            "qwen",
                            Some(&snapshot_identity),
                        ),
                    )
                    .await?,
                ))
            }
            None => None,
        };
        let adapter = NativeQwenAdapter {
            model_id: model_id.clone(),
            metadata: metadata.clone(),
            spec,
            store,
            matvec,
            max_prefill_tokens: DEFAULT_NATIVE_TEXT_MAX_PREFILL_TOKENS,
            top_k: 16,
            chunk_rows: 2048,
            prefix_cache: Arc::new(NativeQwenPrefixCache::new(
                prefix_cache_bytes.unwrap_or(DEFAULT_NATIVE_QWEN_PREFIX_CACHE_BYTES),
            )),
            prefix_disk_cache,
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
            .encode(request.prompt(), false)
            .map_err(|err| BackendError::tokenizer(err.to_string()))
    }

    fn decode_output(
        &self,
        tokenizer: &HuggingFaceTokenizer,
        output_ids: &[u32],
    ) -> Result<String, BackendError> {
        tokenizer
            .decode(output_ids, false)
            .map_err(|err| BackendError::tokenizer(err.to_string()))
    }

    fn stop_tokens(&self) -> NativeTextStopTokens {
        NativeTextStopTokens {
            token_ids: &[],
            token_strings: &["<|im_end|>"],
            encoded_token_strings: &[],
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

    fn prefix_disk_cache(&self) -> Option<&NativeTextDiskCache<QwenLayerCache>> {
        self.prefix_disk_cache.as_deref()
    }

    fn prefix_cache_namespace(
        &self,
        tokenizer_identity: &HuggingFaceTokenizerIdentity,
        request: &BackendRequest,
        cache_tokens: usize,
    ) -> NativeTextPrefixCacheNamespace {
        native_text_prefix_namespace(NativeTextPrefixNamespaceContext {
            model_id: &self.model_id,
            metadata: &self.metadata,
            tokenizer_identity,
            adapter_settings: self.prefix_cache_adapter_settings(),
            request,
            cache_layout_version: NATIVE_QWEN_PREFIX_CACHE_LAYOUT_VERSION,
            cache_tokens,
            max_prefill_tokens: self.max_prefill_tokens,
        })
    }

    fn prefix_cache_adapter_settings(&self) -> &'static str {
        NATIVE_QWEN_PREFIX_ADAPTER_SETTINGS
    }

    fn prefix_cache_hit_is_compatible(
        &self,
        states: &[QwenLayerCachePrefixState],
        cache_tokens: usize,
    ) -> bool {
        states.iter().all(|state| match state {
            QwenLayerCachePrefixState::Full(state) => state.max_tokens() >= cache_tokens,
            QwenLayerCachePrefixState::Linear(_) => true,
            _ => false,
        })
    }

    fn layer_count(&self) -> usize {
        self.spec.num_hidden_layers as usize
    }

    fn allocate_caches(&self, cache_tokens: usize) -> Result<Vec<QwenLayerCache>, BackendError> {
        qwen_layer_caches_for_spec(&self.spec, cache_tokens).map_err(BackendError::from)
    }

    async fn prefill_chunk_with_cache(
        &self,
        token_ids: &[usize],
        caches: &mut [QwenLayerCache],
        scratch: &mut InferenceScratchpad,
    ) -> Result<Vec<Vec<f32>>, BackendError> {
        native_prefill_sequence_with_cache_for_spec_ref(
            &self.store,
            (&self.spec).into(),
            token_ids,
            NativeTextLayerCachesMut::Qwen(caches),
            &self.matvec,
            scratch,
        )
        .await
        .map_err(BackendError::from)
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
        sampling_draw: Option<f32>,
        sampling_scratch: &mut llm_sampler::TopPSamplerScratch,
    ) -> Result<usize, BackendError> {
        NativeTextNextTokenContext {
            store: &self.store,
            spec: (&self.spec).into(),
            top_k: self.top_k,
            chunk_rows: self.chunk_rows,
            matvec: &self.matvec,
            family_display_name: "Qwen",
        }
        .select_next_token(hidden, sampling, sampling_draw, sampling_scratch)
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
        self.hidden = native_decode_token_with_cache_for_spec_ref(
            store,
            spec.into(),
            token_id,
            NativeTextLayerCachesMut::Qwen(&mut self.caches),
            matvec,
            scratch,
        )
        .await
        .map_err(BackendError::from)?;
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
    identity: &ResolvedSnapshotBackend,
) -> anyhow::Result<BackendModelMetadata> {
    if let Some(family) = identity.family()
        && family != ModelFamily::Qwen
    {
        anyhow::bail!(
            "native Qwen backend only supports family `qwen`, not `{}`",
            family.canonical_slug()
        );
    }
    if identity.loader() != SnapshotBackendLoader::NativeMetal {
        anyhow::bail!(
            "native Qwen backend only supports loader `native-metal`, not `{}`",
            identity.loader().canonical_slug()
        );
    }
    Ok(identity.backend_metadata(model_id.to_owned(), "native-qwen", Some(ModelFamily::Qwen)))
}

fn parse_native_qwen_spec(config_json: &str) -> anyhow::Result<QwenModelSpec> {
    Ok(QwenModelSpec::from_config_json(config_json)?)
}

fn validate_native_qwen_text_weights(
    spec: &QwenModelSpec,
    index: &SafetensorsIndex,
) -> anyhow::Result<()> {
    Ok(spec.validate_text_weights(index)?)
}

#[cfg(test)]
fn native_qwen_metal_weight_cache_bytes(configured: Option<u64>) -> u64 {
    crate::native_matvec::native_text_metal_weight_cache_bytes(configured)
}

#[cfg(test)]
fn native_qwen_warmable_bf16_matrix_tensors(
    store: &SafeTensorShardStore,
) -> Result<
    Vec<crate::warm_order::NativeTextWarmableBf16MatrixTensor>,
    llm_backend::native::TensorLoadError,
> {
    crate::warm_order::native_text_warmable_bf16_matrix_tensors(store)
}

pub fn native_qwen_prefix_cache_metrics_snapshot() -> Value {
    native_qwen_prefix_cache_metrics().snapshot()
}

#[cfg(test)]
mod tests;
