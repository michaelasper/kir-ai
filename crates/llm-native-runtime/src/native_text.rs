#[cfg(feature = "native-gemma")]
use crate::native_gemma::{NativeGemmaAdapter, NativeGemmaBackend};
#[cfg(feature = "native-qwen")]
use crate::native_qwen::{NativeQwenAdapter, NativeQwenBackend};
use crate::{
    ResolvedSnapshotBackend, SnapshotBackendLoader,
    native_matvec::{
        NativeTextMatvecBackend, NativeTextMetalWarmup, native_text_metal_weight_cache_bytes,
    },
};
use async_trait::async_trait;
use futures::stream::BoxStream;
#[allow(unused_imports)]
use llm_backend::native::{F32TensorCacheWarmup, InferenceScratchpad, SafeTensorShardStore};
#[allow(unused_imports)]
use llm_backend_contracts::{
    BackendError, BackendFinishReason, BackendModelMetadata, BackendOutput, BackendRequest,
    BackendStreamChunk, ModelBackend,
};
use llm_models::{ModelFamily, NativeTextModelSpec, SafetensorsIndex};
use llm_tokenizer::HuggingFaceTokenizer;
use std::path::{Path, PathBuf};
use tokio_util::sync::CancellationToken;

mod disk_cache;
mod driver;
mod generation;
mod prefix_cache;
mod streaming;

pub use disk_cache::NativeTextDiskCacheConfig;
#[allow(unused_imports)]
pub(crate) use disk_cache::{
    NativeTextDiskCache, NativeTextDiskCacheIdentity, NativeTextDiskCacheStoreStatus,
    NativeTextDiskCacheValue, native_text_disk_cache_snapshot_identity,
};
#[cfg(test)]
pub(crate) use disk_cache::{
    NativeTextDiskCacheError, NativeTextDiskCacheLayerLayout, NativeTextDiskCacheStateBlock,
    NativeTextDiskCacheTensorArchive, NativeTextDiskCacheTensorSink,
};
#[allow(unused_imports)]
pub(crate) use driver::{
    NativeTextAdapter, NativeTextCandidateDecision, NativeTextDriver, NativeTextResolvedStopTokens,
    NativeTextStopTokens,
};
#[cfg(test)]
pub(crate) use generation::native_text_prefill_context_with_cache;
#[cfg(test)]
pub(crate) use generation::sample_token_id_with_draw;
#[allow(unused_imports)]
pub(crate) use generation::{
    NativeTextNextTokenContext, NativeTextSamplingRng, native_text_cache_namespace_token_bucket,
    native_text_cache_token_capacity, resolve_native_text_max_tokens,
};
#[allow(unused_imports)]
pub(crate) use prefix_cache::{
    NativeTextPrefixCache, NativeTextPrefixCacheCounters, NativeTextPrefixCacheEntry,
    NativeTextPrefixCacheHit, NativeTextPrefixCacheInner, NativeTextPrefixCacheMetrics,
    NativeTextPrefixCacheNamespace, NativeTextPrefixCacheValue, NativeTextPrefixNamespaceContext,
    native_text_prefix_namespace, native_text_prefix_request_mode,
};
#[cfg(test)]
pub(crate) use streaming::NativeStreamTextDeltas;
pub(crate) use streaming::{
    NativeTextStreamDecoder, NativeTokenizerStreamDecoder, native_text_worker_stream,
};

pub const DEFAULT_NATIVE_TEXT_MAX_NEW_TOKENS: u32 = 256;
pub const DEFAULT_NATIVE_TEXT_MAX_PREFILL_TOKENS: usize = 2048;
pub const DEFAULT_NATIVE_TEXT_PREFIX_CACHE_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeTextRuntimeOptions {
    pub eager_materialize_shards: bool,
    pub metal_weight_cache_bytes: Option<u64>,
    pub prefix_cache_bytes: Option<u64>,
    pub prefix_disk_cache: Option<NativeTextDiskCacheConfig>,
    pub warm_metal_weight_cache: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeTextLoadOptions {
    pub family: Option<ModelFamily>,
    pub runtime: NativeTextRuntimeOptions,
}

impl NativeTextLoadOptions {
    pub fn with_runtime_options(runtime: NativeTextRuntimeOptions) -> Self {
        Self {
            family: None,
            runtime,
        }
    }

    #[cfg(feature = "native-qwen")]
    pub fn with_qwen_options(qwen: crate::native_qwen::NativeQwenLoadOptions) -> Self {
        Self::with_runtime_options(qwen)
    }

    #[cfg(feature = "native-gemma")]
    pub fn with_gemma_options(gemma: crate::native_gemma::NativeGemmaLoadOptions) -> Self {
        Self::with_runtime_options(gemma)
    }

    pub fn with_family(mut self, family: ModelFamily) -> Self {
        self.family = Some(family);
        self
    }
}

pub(crate) struct NativeTextSnapshotOpen<S> {
    pub(crate) model_id: String,
    pub(crate) metadata: BackendModelMetadata,
    pub(crate) spec: S,
    pub(crate) store: SafeTensorShardStore,
    pub(crate) matvec: NativeTextMatvecBackend,
    pub(crate) tokenizer: HuggingFaceTokenizer,
    pub(crate) prefix_cache_bytes: Option<u64>,
    pub(crate) prefix_disk_cache: Option<NativeTextDiskCacheConfig>,
}

pub(crate) struct NativeTextSnapshotOpenFamily<S> {
    pub(crate) display_name: &'static str,
    pub(crate) parse_spec: fn(&str) -> anyhow::Result<S>,
    pub(crate) validate_text_weights: fn(&S, &SafetensorsIndex) -> anyhow::Result<()>,
    pub(crate) static_f32_tensors_for_spec: fn(&S) -> Vec<String>,
}

struct NativeTextSnapshotBlockingOpen<S> {
    spec: S,
    store: SafeTensorShardStore,
    matvec: NativeTextMatvecBackend,
    tokenizer: HuggingFaceTokenizer,
    materialized_bytes: Option<usize>,
    static_f32_warmup: F32TensorCacheWarmup,
    metal_warmup: Option<NativeTextMetalWarmup>,
}

// Keep the async open path limited to async filesystem reads and worker
// handoff. Safetensors store construction, mmap/materialization, static tensor
// warmup, optional Metal cache warmup, and tokenizer loading are synchronous
// snapshot-open work and run on Tokio's blocking pool.
pub(crate) async fn open_native_text_snapshot<S>(
    model_id: impl Into<String>,
    snapshot_path: impl AsRef<Path>,
    options: NativeTextRuntimeOptions,
    metadata: BackendModelMetadata,
    family: NativeTextSnapshotOpenFamily<S>,
) -> anyhow::Result<NativeTextSnapshotOpen<S>>
where
    S: Send + 'static,
{
    let model_id = model_id.into();
    let snapshot_path = snapshot_path.as_ref().to_path_buf();
    let prefix_disk_cache = options.prefix_disk_cache.clone();
    let prefix_cache_bytes = options.prefix_cache_bytes;
    let blocking_options = options.clone();
    let cache_namespace = tokio::fs::canonicalize(&snapshot_path)
        .await?
        .to_string_lossy()
        .into_owned();
    let config_json = tokio::fs::read_to_string(snapshot_path.join("config.json")).await?;
    let family_display_name = family.display_name;
    let blocking_open = run_native_text_open_blocking(family_display_name, move || {
        open_native_text_snapshot_blocking(
            snapshot_path,
            config_json,
            blocking_options,
            family,
            cache_namespace,
        )
    })
    .await?;
    if let Some(materialized_bytes) = blocking_open.materialized_bytes {
        tracing::info!(
            family = family_display_name,
            materialized_bytes,
            "materialized native text safetensors shards"
        );
    }
    tracing::info!(
        family = family_display_name,
        candidates = blocking_open.static_f32_warmup.candidates,
        loaded = blocking_open.static_f32_warmup.loaded,
        resident_bytes = blocking_open.static_f32_warmup.resident_bytes,
        already_resident = blocking_open.static_f32_warmup.already_resident,
        "native text static f32 tensor cache warm-up complete"
    );
    if let Some(warmup) = blocking_open.metal_warmup {
        tracing::info!(
            family = family_display_name,
            candidates = warmup.candidates,
            warmed = warmup.warmed,
            already_resident = warmup.already_resident,
            skipped_budget = warmup.skipped_budget,
            skipped_non_metal = warmup.skipped_non_metal,
            "native text Metal BF16 weight cache warm-up complete"
        );
    }
    Ok(NativeTextSnapshotOpen {
        model_id,
        metadata,
        spec: blocking_open.spec,
        store: blocking_open.store,
        matvec: blocking_open.matvec,
        tokenizer: blocking_open.tokenizer,
        prefix_cache_bytes,
        prefix_disk_cache,
    })
}

async fn run_native_text_open_blocking<T>(
    family_display_name: &'static str,
    work: impl FnOnce() -> anyhow::Result<T> + Send + 'static,
) -> anyhow::Result<T>
where
    T: Send + 'static,
{
    tokio::task::spawn_blocking(work).await.map_err(|err| {
        anyhow::anyhow!("native {family_display_name} snapshot open blocking worker failed: {err}")
    })?
}

fn open_native_text_snapshot_blocking<S>(
    snapshot_path: PathBuf,
    config_json: String,
    options: NativeTextRuntimeOptions,
    family: NativeTextSnapshotOpenFamily<S>,
    cache_namespace: String,
) -> anyhow::Result<NativeTextSnapshotBlockingOpen<S>>
where
    S: Send + 'static,
{
    let spec = (family.parse_spec)(&config_json)?;
    let store = SafeTensorShardStore::open(&snapshot_path)?;
    (family.validate_text_weights)(&spec, store.index())?;
    let materialized_bytes = if options.eager_materialize_shards {
        Some(store.materialize_all_shards().map_err(|err| {
            anyhow::anyhow!(
                "native {} safetensors materialization failed: {err}",
                family.display_name
            )
        })?)
    } else {
        None
    };
    let static_f32_tensors = (family.static_f32_tensors_for_spec)(&spec);
    let static_f32_warmup = store.preload_bf16_f32_tensors(&static_f32_tensors)?;
    let matvec = NativeTextMatvecBackend::system_default(
        native_text_metal_weight_cache_bytes(options.metal_weight_cache_bytes),
        &cache_namespace,
    );
    let metal_warmup = if options.warm_metal_weight_cache {
        Some(matvec.warm_bf16_matrix_cache(&store).map_err(|err| {
            anyhow::anyhow!(
                "native {} Metal weight cache warm-up failed: {err}",
                family.display_name
            )
        })?)
    } else {
        None
    };
    let tokenizer = HuggingFaceTokenizer::from_file(snapshot_path.join("tokenizer.json"))?;
    Ok(NativeTextSnapshotBlockingOpen {
        spec,
        store,
        matvec,
        tokenizer,
        materialized_bytes,
        static_f32_warmup,
        metal_warmup,
    })
}

pub fn native_text_metal_metrics_snapshot() -> serde_json::Value {
    crate::native_metrics::native_text_metal_metrics_snapshot()
}

pub fn native_text_prefix_cache_metrics_snapshot(
    qwen_snapshot: serde_json::Value,
) -> serde_json::Value {
    let mut metrics = serde_json::Map::new();
    #[cfg(feature = "native-qwen")]
    metrics.insert("qwen".to_owned(), qwen_snapshot);
    #[cfg(not(feature = "native-qwen"))]
    let _ = qwen_snapshot;
    #[cfg(feature = "native-gemma")]
    metrics.insert(
        "gemma".to_owned(),
        crate::native_gemma::native_gemma_prefix_cache_metrics_snapshot(),
    );
    serde_json::Value::Object(metrics)
}

#[derive(Clone)]
pub struct NativeTextBackend {
    inner: NativeTextBackendInner,
}

#[derive(Clone)]
enum NativeTextBackendInner {
    #[cfg(feature = "native-qwen")]
    Qwen(NativeTextDriver<NativeQwenAdapter>),
    #[cfg(feature = "native-gemma")]
    Gemma(NativeTextDriver<NativeGemmaAdapter>),
}

impl NativeTextBackend {
    pub async fn open(
        model_id: impl Into<String>,
        snapshot_path: impl AsRef<Path>,
    ) -> anyhow::Result<Self> {
        Self::open_with_options(model_id, snapshot_path, NativeTextLoadOptions::default()).await
    }

    pub async fn open_with_options(
        model_id: impl Into<String>,
        snapshot_path: impl AsRef<Path>,
        options: NativeTextLoadOptions,
    ) -> anyhow::Result<Self> {
        let snapshot_path = snapshot_path.as_ref();
        let identity = ResolvedSnapshotBackend::resolve(
            snapshot_path,
            None,
            options.family,
            SnapshotBackendLoader::NativeMetal,
            true,
            false,
        )
        .await?;
        Self::open_with_snapshot_identity(model_id, snapshot_path, options, identity).await
    }

    pub async fn open_with_snapshot_identity(
        model_id: impl Into<String>,
        snapshot_path: impl AsRef<Path>,
        options: NativeTextLoadOptions,
        identity: ResolvedSnapshotBackend,
    ) -> anyhow::Result<Self> {
        let model_id = model_id.into();
        let snapshot_path = snapshot_path.as_ref();
        let family = identity
            .family()
            .or(options.family)
            .ok_or_else(|| anyhow::anyhow!("native text snapshot identity is missing family"))?;
        match family {
            ModelFamily::Gemma => {
                #[cfg(feature = "native-gemma")]
                {
                    let driver = NativeGemmaBackend::open_with_snapshot_identity(
                        model_id,
                        snapshot_path,
                        options.runtime,
                        identity,
                    )
                    .await?
                    .into_driver();
                    Ok(Self {
                        inner: NativeTextBackendInner::Gemma(driver),
                    })
                }
                #[cfg(not(feature = "native-gemma"))]
                {
                    anyhow::bail!(
                        "native text execution for family `gemma` is disabled; rebuild llm-engine with --features native-gemma"
                    );
                }
            }
            ModelFamily::DeepSeek => {
                anyhow::bail!(
                    "native text execution for family `deep_seek` is deferred until native DeepSeek tensor support exists"
                );
            }
            ModelFamily::Llama => {
                anyhow::bail!(
                    "native text execution for family `llama` is deferred until native Llama tensor support exists"
                );
            }
            ModelFamily::Qwen => {
                #[cfg(feature = "native-qwen")]
                {
                    let driver = NativeQwenBackend::open_with_snapshot_identity(
                        model_id,
                        snapshot_path,
                        options.runtime,
                        identity,
                    )
                    .await?
                    .into_driver();
                    Ok(Self {
                        inner: NativeTextBackendInner::Qwen(driver),
                    })
                }
                #[cfg(not(feature = "native-qwen"))]
                {
                    anyhow::bail!(
                        "native text execution for family `qwen` is disabled; rebuild llm-engine with --features native-qwen"
                    );
                }
            }
        }
    }

    pub fn with_max_new_tokens(mut self, max_new_tokens: u32) -> Self {
        self.inner = match self.inner {
            #[cfg(feature = "native-qwen")]
            NativeTextBackendInner::Qwen(driver) => {
                NativeTextBackendInner::Qwen(driver.with_max_new_tokens(max_new_tokens))
            }
            #[cfg(feature = "native-gemma")]
            NativeTextBackendInner::Gemma(driver) => {
                NativeTextBackendInner::Gemma(driver.with_max_new_tokens(max_new_tokens))
            }
        };
        self
    }

    pub fn with_max_prefill_tokens(mut self, max_prefill_tokens: usize) -> Self {
        self.inner = match self.inner {
            #[cfg(feature = "native-qwen")]
            NativeTextBackendInner::Qwen(driver) => {
                NativeTextBackendInner::Qwen(driver.with_max_prefill_tokens(max_prefill_tokens))
            }
            #[cfg(feature = "native-gemma")]
            NativeTextBackendInner::Gemma(driver) => {
                NativeTextBackendInner::Gemma(driver.with_max_prefill_tokens(max_prefill_tokens))
            }
        };
        self
    }
}

pub(crate) async fn infer_native_text_family(snapshot_path: &Path) -> anyhow::Result<ModelFamily> {
    let config_path = snapshot_path.join("config.json");
    let config_json = tokio::fs::read_to_string(&config_path).await.map_err(|err| {
        anyhow::anyhow!(
            "native text snapshot without explicit family metadata requires readable config.json for family detection at `{}`: {err}",
            config_path.display()
        )
    })?;
    Ok(NativeTextModelSpec::infer_from_config_json(&config_json)?.family())
}

#[async_trait]
impl ModelBackend for NativeTextBackend {
    fn model_id(&self) -> &str {
        match &self.inner {
            #[cfg(feature = "native-qwen")]
            NativeTextBackendInner::Qwen(backend) => backend.model_id(),
            #[cfg(feature = "native-gemma")]
            NativeTextBackendInner::Gemma(backend) => backend.model_id(),
        }
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        match &self.inner {
            #[cfg(feature = "native-qwen")]
            NativeTextBackendInner::Qwen(backend) => backend.model_metadata(),
            #[cfg(feature = "native-gemma")]
            NativeTextBackendInner::Gemma(backend) => backend.model_metadata(),
        }
    }

    async fn generate(&self, request: BackendRequest) -> Result<BackendOutput, BackendError> {
        match &self.inner {
            #[cfg(feature = "native-qwen")]
            NativeTextBackendInner::Qwen(backend) => backend.generate(request).await,
            #[cfg(feature = "native-gemma")]
            NativeTextBackendInner::Gemma(backend) => backend.generate(request).await,
        }
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        match &self.inner {
            #[cfg(feature = "native-qwen")]
            NativeTextBackendInner::Qwen(backend) => {
                backend.generate_with_cancel(request, cancellation).await
            }
            #[cfg(feature = "native-gemma")]
            NativeTextBackendInner::Gemma(backend) => {
                backend.generate_with_cancel(request, cancellation).await
            }
        }
    }

    fn generate_stream<'a>(
        &'a self,
        request: BackendRequest,
    ) -> BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        match &self.inner {
            #[cfg(feature = "native-qwen")]
            NativeTextBackendInner::Qwen(backend) => backend.generate_stream(request),
            #[cfg(feature = "native-gemma")]
            NativeTextBackendInner::Gemma(backend) => backend.generate_stream(request),
        }
    }

    fn generate_stream_with_cancel<'a>(
        &'a self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        match &self.inner {
            #[cfg(feature = "native-qwen")]
            NativeTextBackendInner::Qwen(backend) => {
                backend.generate_stream_with_cancel(request, cancellation)
            }
            #[cfg(feature = "native-gemma")]
            NativeTextBackendInner::Gemma(backend) => {
                backend.generate_stream_with_cancel(request, cancellation)
            }
        }
    }
}

#[cfg(all(test, feature = "native-qwen", feature = "native-gemma"))]
mod tests {
    use super::*;
    use crate::native_matvec::{NativeTextCacheMirrorIds, NativeTextCacheMirrorSource};
    use llm_backend_contracts::{
        BackendChatContext, BackendChatMessage, BackendChatRole, BackendFailureClass,
        BackendPrefillChunkAdmission, BackendPrefillChunkAdmissionHook, BackendStreamProgress,
        BackendToolChoice, SamplingConfig,
    };
    use llm_tokenizer::{HuggingFaceTokenizer, HuggingFaceTokenizerIdentity};
    use std::{
        sync::{
            Arc, Mutex, Weak,
            atomic::{AtomicUsize, Ordering},
        },
        time::{Duration, Instant},
    };
    use tokio::sync::Notify;

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TestCache {
        bytes: u64,
        marker: u32,
    }

    impl NativeTextPrefixCacheValue for TestCache {
        type PrefixCacheState = Self;

        fn prefix_cache_state(caches: &[Self]) -> Vec<Self::PrefixCacheState> {
            caches.to_vec()
        }

        fn prefix_cache_from_state(states: &[Self::PrefixCacheState]) -> Option<Vec<Self>> {
            Some(states.to_vec())
        }

        fn prefix_cache_entry_bytes(hidden: &[f32], states: &[Self::PrefixCacheState]) -> u64 {
            std::mem::size_of_val(hidden) as u64
                + states.iter().map(|cache| cache.bytes).sum::<u64>()
        }
    }

    impl NativeTextDiskCacheValue for TestCache {
        fn encode_disk_block_states(
            states: &[Self::PrefixCacheState],
            block_start: usize,
            block_token_count: usize,
            sink: &mut NativeTextDiskCacheTensorSink,
        ) -> Result<Vec<NativeTextDiskCacheLayerLayout>, NativeTextDiskCacheError> {
            let values = states[block_start..block_start + block_token_count]
                .iter()
                .map(|state| state.marker as f32)
                .collect::<Vec<_>>();
            sink.push_f32("test.markers", vec![values.len()], values)?;
            Ok(vec![NativeTextDiskCacheLayerLayout::test_marker_tensor(
                "test.markers",
            )])
        }

        fn decode_disk_states(
            layouts: &[NativeTextDiskCacheLayerLayout],
            archive: &NativeTextDiskCacheTensorArchive<'_>,
        ) -> Result<Vec<Self::PrefixCacheState>, NativeTextDiskCacheError> {
            let Some(layout) = layouts.first() else {
                return Err(NativeTextDiskCacheError::integrity(
                    "missing test marker layout",
                ));
            };
            let tensor = layout
                .test_marker_tensor_name()
                .ok_or_else(|| NativeTextDiskCacheError::integrity("wrong test marker layout"))?;
            archive
                .f32_tensor(tensor)?
                .into_iter()
                .map(|marker| {
                    if marker.fract() != 0.0 || marker < 0.0 {
                        return Err(NativeTextDiskCacheError::integrity(
                            "test marker must be a non-negative integer",
                        ));
                    }
                    Ok(TestCache {
                        bytes: std::mem::size_of::<TestCache>() as u64,
                        marker: marker as u32,
                    })
                })
                .collect()
        }

        fn assemble_disk_block_states(
            blocks: &[NativeTextDiskCacheStateBlock<Self::PrefixCacheState>],
        ) -> Result<Vec<Self::PrefixCacheState>, NativeTextDiskCacheError> {
            Ok(blocks
                .iter()
                .flat_map(|block| block.states.iter().cloned())
                .collect())
        }
    }

    impl NativeTextCacheMirrorSource for TestCache {
        fn append_cache_mirror_ids(&self, _ids: &mut NativeTextCacheMirrorIds) {}
    }

    #[derive(Debug)]
    struct LockObservingCache {
        bytes: u64,
        cache: Weak<NativeTextPrefixCache<LockObservingCache>>,
        cloned_while_locked: Arc<AtomicUsize>,
    }

    impl Clone for LockObservingCache {
        fn clone(&self) -> Self {
            if let Some(cache) = self.cache.upgrade()
                && cache.inner.try_lock().is_err()
            {
                self.cloned_while_locked.fetch_add(1, Ordering::SeqCst);
            }
            Self {
                bytes: self.bytes,
                cache: self.cache.clone(),
                cloned_while_locked: self.cloned_while_locked.clone(),
            }
        }
    }

    impl NativeTextPrefixCacheValue for LockObservingCache {
        type PrefixCacheState = Self;

        fn prefix_cache_state(caches: &[Self]) -> Vec<Self::PrefixCacheState> {
            caches.to_vec()
        }

        fn prefix_cache_from_state(states: &[Self::PrefixCacheState]) -> Option<Vec<Self>> {
            Some(states.to_vec())
        }

        fn prefix_cache_entry_bytes(hidden: &[f32], states: &[Self::PrefixCacheState]) -> u64 {
            std::mem::size_of_val(hidden) as u64
                + states.iter().map(|cache| cache.bytes).sum::<u64>()
        }
    }

    #[derive(Clone)]
    struct TestDecodeSession {
        hidden: Vec<f32>,
    }

    #[derive(Clone)]
    enum TestDecodeOutput {
        TokenTags,
        UnicodeBoundary,
    }

    #[derive(Clone)]
    struct TestAdapter {
        script: std::sync::Arc<[usize]>,
        stop_tokens: NativeTextStopTokens,
        max_prefill_tokens: usize,
        max_position_embeddings: u32,
        decode_output: TestDecodeOutput,
        prefix_cache: std::sync::Arc<NativeTextPrefixCache<TestCache>>,
        prefix_cache_metrics: std::sync::Arc<NativeTextPrefixCacheMetrics>,
        cleanup_calls: Arc<AtomicUsize>,
        cleanup_markers: Arc<Mutex<Vec<Vec<u32>>>>,
        next_token_calls: Arc<AtomicUsize>,
        sampling_draws: Arc<Mutex<Vec<Option<f32>>>>,
        decoded_token_total: Arc<AtomicUsize>,
        stream_decoded_token_total: Arc<AtomicUsize>,
        encoded_prompt: std::sync::Arc<[u32]>,
        next_token_delay: Option<Duration>,
        fail_prefill: bool,
        fail_after_prefill_chunk: Option<usize>,
        cancel_on_prefill: Option<CancellationToken>,
        cancel_after_prefill_chunk: Option<(CancellationToken, usize)>,
        prefill_chunk_calls: Arc<AtomicUsize>,
    }

    impl TestAdapter {
        fn new(script: impl Into<std::sync::Arc<[usize]>>) -> Self {
            Self {
                script: script.into(),
                stop_tokens: NativeTextStopTokens::default(),
                max_prefill_tokens: 4,
                max_position_embeddings: 16,
                decode_output: TestDecodeOutput::TokenTags,
                prefix_cache: std::sync::Arc::new(NativeTextPrefixCache::new(1024)),
                prefix_cache_metrics: std::sync::Arc::new(NativeTextPrefixCacheMetrics::default()),
                cleanup_calls: Arc::new(AtomicUsize::new(0)),
                cleanup_markers: Arc::new(Mutex::new(Vec::new())),
                next_token_calls: Arc::new(AtomicUsize::new(0)),
                sampling_draws: Arc::new(Mutex::new(Vec::new())),
                decoded_token_total: Arc::new(AtomicUsize::new(0)),
                stream_decoded_token_total: Arc::new(AtomicUsize::new(0)),
                encoded_prompt: std::sync::Arc::from([42_u32]),
                next_token_delay: None,
                fail_prefill: false,
                fail_after_prefill_chunk: None,
                cancel_on_prefill: None,
                cancel_after_prefill_chunk: None,
                prefill_chunk_calls: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn with_stop_tokens(mut self, stop_tokens: NativeTextStopTokens) -> Self {
            self.stop_tokens = stop_tokens;
            self
        }

        fn with_prefill_failure(mut self) -> Self {
            self.fail_prefill = true;
            self
        }

        fn with_prefill_failure_after_chunk(mut self, chunk: usize) -> Self {
            self.fail_after_prefill_chunk = Some(chunk);
            self
        }

        fn with_prefill_cancellation(mut self, cancellation: CancellationToken) -> Self {
            self.cancel_on_prefill = Some(cancellation);
            self
        }

        fn with_prefill_cancellation_after_chunk(
            mut self,
            cancellation: CancellationToken,
            chunk: usize,
        ) -> Self {
            self.cancel_after_prefill_chunk = Some((cancellation, chunk));
            self
        }

        fn with_prefix_cache_bytes(mut self, prefix_cache_bytes: u64) -> Self {
            self.prefix_cache = std::sync::Arc::new(NativeTextPrefixCache::new(prefix_cache_bytes));
            self
        }

        fn with_next_token_delay(mut self, delay: Duration) -> Self {
            self.next_token_delay = Some(delay);
            self
        }

        fn with_encoded_prompt(mut self, encoded_prompt: impl Into<std::sync::Arc<[u32]>>) -> Self {
            self.encoded_prompt = encoded_prompt.into();
            self
        }

        fn with_unicode_boundary_decode(mut self) -> Self {
            self.decode_output = TestDecodeOutput::UnicodeBoundary;
            self
        }

        fn with_max_position_embeddings(mut self, max_position_embeddings: u32) -> Self {
            self.max_position_embeddings = max_position_embeddings;
            self
        }

        fn cleanup_calls(&self) -> Arc<AtomicUsize> {
            Arc::clone(&self.cleanup_calls)
        }

        fn cleanup_markers(&self) -> Arc<Mutex<Vec<Vec<u32>>>> {
            Arc::clone(&self.cleanup_markers)
        }

        fn next_token_calls(&self) -> Arc<AtomicUsize> {
            Arc::clone(&self.next_token_calls)
        }

        fn sampling_draws(&self) -> Arc<Mutex<Vec<Option<f32>>>> {
            Arc::clone(&self.sampling_draws)
        }

        fn decoded_token_total(&self) -> Arc<AtomicUsize> {
            Arc::clone(&self.decoded_token_total)
        }

        fn stream_decoded_token_total(&self) -> Arc<AtomicUsize> {
            Arc::clone(&self.stream_decoded_token_total)
        }

        fn prefill_chunk_calls(&self) -> Arc<AtomicUsize> {
            Arc::clone(&self.prefill_chunk_calls)
        }
    }

    #[derive(Debug)]
    struct BlockingPrefillAdmission {
        release: Notify,
        calls: AtomicUsize,
    }

    impl BlockingPrefillAdmission {
        fn new() -> Self {
            Self {
                release: Notify::new(),
                calls: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait]
    impl BackendPrefillChunkAdmission for BlockingPrefillAdmission {
        async fn wait_for_next_chunk(
            &self,
            progress: BackendStreamProgress,
        ) -> Result<(), BackendError> {
            assert_eq!(
                progress,
                BackendStreamProgress::PrefillProgress {
                    chunk: 1,
                    total: 2,
                    tokens: 2,
                    total_tokens: 4,
                }
            );
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.release.notified().await;
            Ok(())
        }
    }

    struct TestStreamDecoder {
        decode_output: TestDecodeOutput,
        decoded_token_total: Arc<AtomicUsize>,
        unicode_boundary_started: bool,
    }

    impl NativeTextStreamDecoder for TestStreamDecoder {
        fn step(&mut self, token_id: u32) -> Result<Option<String>, BackendError> {
            self.decoded_token_total.fetch_add(1, Ordering::SeqCst);
            Ok(match self.decode_output {
                TestDecodeOutput::TokenTags => Some(format!("<{token_id}>")),
                TestDecodeOutput::UnicodeBoundary => {
                    if self.unicode_boundary_started && token_id == 2 {
                        self.unicode_boundary_started = false;
                        Some("é".to_owned())
                    } else if token_id == 1 {
                        self.unicode_boundary_started = true;
                        None
                    } else {
                        Some(format!("<{token_id}>"))
                    }
                }
            })
        }
    }

    #[async_trait]
    impl NativeTextAdapter for TestAdapter {
        type DecodeSession = TestDecodeSession;
        type LayerCache = TestCache;

        fn family_display_name(&self) -> &'static str {
            "Test"
        }

        fn worker_label(&self) -> &'static str {
            "native test"
        }

        fn set_max_prefill_tokens(&mut self, max_prefill_tokens: usize) {
            self.max_prefill_tokens = max_prefill_tokens;
        }

        fn encode_prompt(
            &self,
            _tokenizer: &HuggingFaceTokenizer,
            _request: &BackendRequest,
        ) -> Result<Vec<u32>, BackendError> {
            Ok(self.encoded_prompt.to_vec())
        }

        fn decode_output(
            &self,
            _tokenizer: &HuggingFaceTokenizer,
            output_ids: &[u32],
        ) -> Result<String, BackendError> {
            self.decoded_token_total
                .fetch_add(output_ids.len(), Ordering::SeqCst);
            Ok(match self.decode_output {
                TestDecodeOutput::TokenTags => output_ids
                    .iter()
                    .map(|token_id| format!("<{token_id}>"))
                    .collect::<String>(),
                TestDecodeOutput::UnicodeBoundary => match output_ids {
                    [1] | [2] => "�".to_owned(),
                    [1, 2] => "é".to_owned(),
                    _ => output_ids
                        .iter()
                        .map(|token_id| format!("<{token_id}>"))
                        .collect::<String>(),
                },
            })
        }

        fn stream_decoder<'tokenizer>(
            &self,
            _tokenizer: &'tokenizer HuggingFaceTokenizer,
        ) -> Box<dyn NativeTextStreamDecoder + 'tokenizer> {
            Box::new(TestStreamDecoder {
                decode_output: self.decode_output.clone(),
                decoded_token_total: Arc::clone(&self.stream_decoded_token_total),
                unicode_boundary_started: false,
            })
        }

        fn stop_tokens(&self) -> NativeTextStopTokens {
            self.stop_tokens
        }

        fn max_position_embeddings(&self) -> u32 {
            self.max_position_embeddings
        }

        fn max_prefill_tokens(&self) -> usize {
            self.max_prefill_tokens
        }

        fn prefix_cache(&self) -> &NativeTextPrefixCache<Self::LayerCache> {
            &self.prefix_cache
        }

        fn prefix_cache_metrics(&self) -> &NativeTextPrefixCacheMetrics {
            &self.prefix_cache_metrics
        }

        fn prefix_cache_namespace(
            &self,
            tokenizer_identity: &HuggingFaceTokenizerIdentity,
            _request: &BackendRequest,
            cache_tokens: usize,
        ) -> NativeTextPrefixCacheNamespace {
            NativeTextPrefixCacheNamespace {
                cache_tokens,
                tokenizer_kind: tokenizer_identity.kind.clone(),
                tokenizer_hash: tokenizer_identity.content_hash.clone(),
                tokenizer_normalization: tokenizer_identity.normalization.clone(),
                adapter_settings: self.prefix_cache_adapter_settings().to_owned(),
                ..namespace("driver-test")
            }
        }

        fn prefix_cache_adapter_settings(&self) -> &'static str {
            "native-test-adapter/v1"
        }

        fn layer_count(&self) -> usize {
            1
        }

        fn allocate_caches(
            &self,
            _cache_tokens: usize,
        ) -> Result<Vec<Self::LayerCache>, BackendError> {
            Ok(vec![TestCache {
                bytes: 0,
                marker: 0,
            }])
        }

        async fn prefill_chunk_with_cache(
            &self,
            token_ids: &[usize],
            _caches: &mut [Self::LayerCache],
            _scratch: &mut InferenceScratchpad,
        ) -> Result<Vec<Vec<f32>>, BackendError> {
            let chunk_call = self.prefill_chunk_calls.fetch_add(1, Ordering::SeqCst) + 1;
            if let Some(cancellation) = &self.cancel_on_prefill {
                cancellation.cancel();
            }
            if let Some((cancellation, chunk)) = &self.cancel_after_prefill_chunk
                && chunk_call == *chunk
            {
                cancellation.cancel();
            }
            if self.fail_prefill {
                return Err(BackendError::other("test prefill failed".to_owned()));
            }
            if let Some(chunk) = self.fail_after_prefill_chunk
                && chunk_call == chunk
            {
                return Err(BackendError::other("test prefill failed".to_owned()));
            }
            Ok(token_ids.iter().map(|_| vec![0.0]).collect())
        }

        fn make_decode_session(
            &self,
            hidden: Vec<f32>,
            _caches: Vec<Self::LayerCache>,
        ) -> Self::DecodeSession {
            TestDecodeSession { hidden }
        }

        fn cleanup_cache_mirrors(&self, caches: &[Self::LayerCache]) {
            let markers = caches.iter().map(|cache| cache.marker).collect();
            self.cleanup_markers
                .lock()
                .expect("cleanup markers lock is not poisoned")
                .push(markers);
            self.cleanup_calls.fetch_add(1, Ordering::SeqCst);
        }

        fn hidden<'a>(&self, session: &'a Self::DecodeSession) -> &'a [f32] {
            &session.hidden
        }

        async fn step(
            &self,
            session: &mut Self::DecodeSession,
            _token_id: usize,
            _scratch: &mut InferenceScratchpad,
        ) -> Result<(), BackendError> {
            session.hidden[0] += 1.0;
            Ok(())
        }

        async fn next_token_from_hidden(
            &self,
            hidden: &[f32],
            _sampling: SamplingConfig,
            sampling_draw: Option<f32>,
            _sampling_scratch: &mut llm_sampler::TopPSamplerScratch,
        ) -> Result<usize, BackendError> {
            self.next_token_calls.fetch_add(1, Ordering::SeqCst);
            self.sampling_draws
                .lock()
                .expect("sampling draws lock is not poisoned")
                .push(sampling_draw);
            if let Some(delay) = self.next_token_delay {
                std::thread::sleep(delay);
            }
            let script_index = hidden[0] as usize;
            Ok(*self
                .script
                .get(script_index)
                .expect("test script includes requested token"))
        }
    }

    #[derive(Clone)]
    struct ContextSensitiveTestAdapter {
        base: TestAdapter,
        stop_after_emitted: usize,
    }

    impl ContextSensitiveTestAdapter {
        fn new(script: impl Into<std::sync::Arc<[usize]>>, stop_after_emitted: usize) -> Self {
            Self {
                base: TestAdapter::new(script),
                stop_after_emitted,
            }
        }
    }

    #[async_trait]
    impl NativeTextAdapter for ContextSensitiveTestAdapter {
        type DecodeSession = TestDecodeSession;
        type LayerCache = TestCache;

        fn family_display_name(&self) -> &'static str {
            self.base.family_display_name()
        }

        fn worker_label(&self) -> &'static str {
            self.base.worker_label()
        }

        fn set_max_prefill_tokens(&mut self, max_prefill_tokens: usize) {
            self.base.set_max_prefill_tokens(max_prefill_tokens);
        }

        fn encode_prompt(
            &self,
            tokenizer: &HuggingFaceTokenizer,
            request: &BackendRequest,
        ) -> Result<Vec<u32>, BackendError> {
            self.base.encode_prompt(tokenizer, request)
        }

        fn decode_output(
            &self,
            tokenizer: &HuggingFaceTokenizer,
            output_ids: &[u32],
        ) -> Result<String, BackendError> {
            self.base.decode_output(tokenizer, output_ids)
        }

        fn observe_candidate(
            &self,
            stop_tokens: &NativeTextResolvedStopTokens,
            emitted_tokens: &[u32],
            token_id: usize,
        ) -> Result<NativeTextCandidateDecision, BackendError> {
            if emitted_tokens.len() >= self.stop_after_emitted {
                Ok(NativeTextCandidateDecision::Stop)
            } else {
                self.base
                    .observe_candidate(stop_tokens, emitted_tokens, token_id)
            }
        }

        fn max_position_embeddings(&self) -> u32 {
            self.base.max_position_embeddings()
        }

        fn max_prefill_tokens(&self) -> usize {
            self.base.max_prefill_tokens()
        }

        fn prefix_cache(&self) -> &NativeTextPrefixCache<Self::LayerCache> {
            self.base.prefix_cache()
        }

        fn prefix_cache_metrics(&self) -> &NativeTextPrefixCacheMetrics {
            self.base.prefix_cache_metrics()
        }

        fn prefix_cache_namespace(
            &self,
            tokenizer_identity: &HuggingFaceTokenizerIdentity,
            request: &BackendRequest,
            cache_tokens: usize,
        ) -> NativeTextPrefixCacheNamespace {
            self.base
                .prefix_cache_namespace(tokenizer_identity, request, cache_tokens)
        }

        fn prefix_cache_adapter_settings(&self) -> &'static str {
            self.base.prefix_cache_adapter_settings()
        }

        fn layer_count(&self) -> usize {
            self.base.layer_count()
        }

        fn allocate_caches(
            &self,
            cache_tokens: usize,
        ) -> Result<Vec<Self::LayerCache>, BackendError> {
            self.base.allocate_caches(cache_tokens)
        }

        async fn prefill_chunk_with_cache(
            &self,
            token_ids: &[usize],
            caches: &mut [Self::LayerCache],
            scratch: &mut InferenceScratchpad,
        ) -> Result<Vec<Vec<f32>>, BackendError> {
            self.base
                .prefill_chunk_with_cache(token_ids, caches, scratch)
                .await
        }

        fn make_decode_session(
            &self,
            hidden: Vec<f32>,
            caches: Vec<Self::LayerCache>,
        ) -> Self::DecodeSession {
            self.base.make_decode_session(hidden, caches)
        }

        fn cleanup_cache_mirrors(&self, caches: &[Self::LayerCache]) {
            self.base.cleanup_cache_mirrors(caches);
        }

        fn hidden<'a>(&self, session: &'a Self::DecodeSession) -> &'a [f32] {
            self.base.hidden(session)
        }

        async fn step(
            &self,
            session: &mut Self::DecodeSession,
            token_id: usize,
            scratch: &mut InferenceScratchpad,
        ) -> Result<(), BackendError> {
            self.base.step(session, token_id, scratch).await
        }

        async fn next_token_from_hidden(
            &self,
            hidden: &[f32],
            sampling: SamplingConfig,
            sampling_draw: Option<f32>,
            sampling_scratch: &mut llm_sampler::TopPSamplerScratch,
        ) -> Result<usize, BackendError> {
            self.base
                .next_token_from_hidden(hidden, sampling, sampling_draw, sampling_scratch)
                .await
        }
    }

    fn namespace(label: &str) -> NativeTextPrefixCacheNamespace {
        NativeTextPrefixCacheNamespace {
            model_id: format!("model-{label}"),
            backend: "native-test".to_owned(),
            family: Some("test".to_owned()),
            quantization: Some("bf16".to_owned()),
            repo_id: Some("org/model".to_owned()),
            resolved_commit: Some("abc123".to_owned()),
            profile: Some(label.to_owned()),
            tokenizer_kind: "huggingface-tokenizer-json".to_owned(),
            tokenizer_hash: format!("sha256:tokenizer-{label}"),
            tokenizer_normalization: "llm-tokenizer/hf-json/v1".to_owned(),
            cache_template_id: format!("template-{label}/v1"),
            chat_template_kwargs_hash: None,
            adapter_settings: format!("native-test-adapter-{label}/v1"),
            cache_key: format!("cache-key-{label}"),
            tool_schema: None,
            request_mode: "conversation=false,json_object=false,required_tool=None".to_owned(),
            cache_layout_version: 1,
            cache_tokens: 16,
            max_prefill_tokens: 4,
        }
    }

    fn driver_test_tokenizer() -> HuggingFaceTokenizer {
        let tokenizer_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/qwen36/tokenizer.json");
        HuggingFaceTokenizer::from_file(tokenizer_path).expect("tokenizer loads")
    }

    fn driver_test_tokenizer_identity() -> HuggingFaceTokenizerIdentity {
        driver_test_tokenizer().identity().clone()
    }

    fn driver_test_request(max_tokens: u32) -> BackendRequest {
        BackendRequest::raw_completion(
            "model-test",
            "test",
            Some(max_tokens),
            SamplingConfig::Greedy,
        )
    }

    fn driver_for_test<A>(adapter: A) -> NativeTextDriver<A>
    where
        A: NativeTextAdapter,
    {
        NativeTextDriver::new(
            "model-test".to_owned(),
            BackendModelMetadata::new("model-test", "native-test").with_family("test"),
            driver_test_tokenizer(),
            adapter,
            8,
        )
    }

    fn stream_final_chunk<A>(
        driver: &NativeTextDriver<A>,
        request: BackendRequest,
    ) -> BackendStreamChunk
    where
        A: NativeTextAdapter,
    {
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        driver
            .generate_blocking_stream(request, tx, CancellationToken::new())
            .expect("streaming generation succeeds");
        let mut final_chunk = None;
        while let Some(chunk) = rx.blocking_recv() {
            let chunk = chunk.expect("stream chunk is ok");
            if chunk.finish_reason.is_some() {
                final_chunk = Some(chunk);
            }
        }
        final_chunk.expect("streaming generation emits a final chunk")
    }

    fn store_driver_prefix_hit(
        adapter: &TestAdapter,
        request: &BackendRequest,
        prompt_tokens: usize,
        max_new_tokens: u32,
        prefix_tokens: &[usize],
        cache: TestCache,
    ) -> (NativeTextPrefixCacheNamespace, u64) {
        let cache_tokens = native_text_cache_token_capacity(
            prompt_tokens,
            max_new_tokens,
            adapter.max_prefill_tokens(),
            adapter.max_position_embeddings(),
            adapter.family_display_name(),
        )
        .expect("test cache token capacity is valid");
        let namespace_cache_tokens = native_text_cache_namespace_token_bucket(
            cache_tokens,
            adapter.max_position_embeddings(),
            adapter.family_display_name(),
        )
        .expect("test namespace cache token bucket is valid");
        let tokenizer = driver_test_tokenizer();
        let namespace =
            adapter.prefix_cache_namespace(tokenizer.identity(), request, namespace_cache_tokens);
        let hidden = [0.25_f32];
        let caches = [cache];
        let byte_len = TestCache::prefix_cache_entry_bytes(&hidden, &caches);

        adapter.prefix_cache.store(
            namespace.clone(),
            prefix_tokens,
            &hidden,
            &caches,
            &adapter.prefix_cache_metrics,
        );

        (namespace, byte_len)
    }

    fn driver_prefix_namespace(
        adapter: &TestAdapter,
        request: &BackendRequest,
        prompt_tokens: usize,
        max_new_tokens: u32,
    ) -> NativeTextPrefixCacheNamespace {
        let cache_tokens = native_text_cache_token_capacity(
            prompt_tokens,
            max_new_tokens,
            adapter.max_prefill_tokens(),
            adapter.max_position_embeddings(),
            adapter.family_display_name(),
        )
        .expect("test cache token capacity is valid");
        let namespace_cache_tokens = native_text_cache_namespace_token_bucket(
            cache_tokens,
            adapter.max_position_embeddings(),
            adapter.family_display_name(),
        )
        .expect("test namespace cache token bucket is valid");
        let tokenizer = driver_test_tokenizer();
        adapter.prefix_cache_namespace(tokenizer.identity(), request, namespace_cache_tokens)
    }

    fn assert_prefix_cache_entry(
        cache: &NativeTextPrefixCache<TestCache>,
        namespace: &NativeTextPrefixCacheNamespace,
        tokens: &[usize],
    ) {
        let inner = cache
            .inner
            .lock()
            .expect("prefix cache lock is not poisoned");
        let bucket = inner
            .entries
            .get(namespace)
            .expect("prefix namespace remains resident");
        assert!(
            bucket.contains_key(&tokens.to_vec()),
            "expected checkpoint for tokens {tokens:?}"
        );
    }

    fn assert_no_prefix_cache_entry(
        cache: &NativeTextPrefixCache<TestCache>,
        namespace: &NativeTextPrefixCacheNamespace,
        tokens: &[usize],
    ) {
        let inner = cache
            .inner
            .lock()
            .expect("prefix cache lock is not poisoned");
        assert!(
            inner
                .entries
                .get(namespace)
                .is_none_or(|bucket| !bucket.contains_key(&tokens.to_vec())),
            "did not expect checkpoint for tokens {tokens:?}"
        );
    }

    fn assert_only_prefix_cache_entry(
        cache: &NativeTextPrefixCache<TestCache>,
        namespace: &NativeTextPrefixCacheNamespace,
        tokens: &[usize],
        byte_len: u64,
    ) {
        let inner = cache
            .inner
            .lock()
            .expect("prefix cache lock is not poisoned");
        let bucket = inner
            .entries
            .get(namespace)
            .expect("prefix namespace remains resident");

        assert_eq!(bucket.len(), 1);
        assert!(bucket.contains_key(&tokens.to_vec()));
        assert_eq!(
            inner
                .entries
                .values()
                .map(std::collections::HashMap::len)
                .sum::<usize>(),
            1
        );
        assert_eq!(inner.used_bytes, byte_len);
    }

    #[test]
    fn native_text_driver_clone_shares_inner_state() {
        let driver = driver_for_test(TestAdapter::new([1_usize]));
        let clone = driver.clone();

        assert!(driver.shares_inner_state_with(&clone));
    }

    #[test]
    fn native_text_load_options_store_runtime_options_once() {
        let runtime = NativeTextRuntimeOptions {
            eager_materialize_shards: true,
            metal_weight_cache_bytes: Some(4096),
            prefix_cache_bytes: Some(17),
            prefix_disk_cache: None,
            warm_metal_weight_cache: true,
        };
        let options = NativeTextLoadOptions::with_runtime_options(runtime.clone());

        assert_eq!(options.runtime, runtime);
    }

    #[test]
    fn family_load_options_use_shared_runtime_options() {
        #[cfg(feature = "native-qwen")]
        let _: NativeTextRuntimeOptions = crate::native_qwen::NativeQwenLoadOptions::default();
        #[cfg(feature = "native-gemma")]
        let _: NativeTextRuntimeOptions = crate::native_gemma::NativeGemmaLoadOptions::default();
    }

    #[test]
    fn driver_with_zero_prefix_cache_budget_generates_without_reuse() {
        let adapter = TestAdapter::new([1_usize]).with_prefix_cache_bytes(0);
        let metrics = Arc::clone(&adapter.prefix_cache_metrics);
        let driver = driver_for_test(adapter);

        let first = driver
            .generate_blocking(driver_test_request(1), CancellationToken::new())
            .expect("first generation succeeds");
        let second = driver
            .generate_blocking(driver_test_request(1), CancellationToken::new())
            .expect("second generation succeeds");

        assert_eq!(first.text, "<1>");
        assert_eq!(second.text, "<1>");
        assert_eq!(first.prompt_cached_tokens, Some(0));
        assert_eq!(second.prompt_cached_tokens, Some(0));
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot["stores"], 0);
        assert_eq!(snapshot["rejected"], 2);
        assert_eq!(snapshot["resident_bytes"], 0);
        assert_eq!(snapshot["resident_entries"], 0);
    }

    #[test]
    fn prefix_namespace_copies_metadata_and_request_context() {
        let mut metadata = BackendModelMetadata::new("model-a", "native-test").with_family("test");
        metadata.quantization = Some("bf16".to_owned());
        metadata.repo_id = Some("org/model".to_owned());
        metadata.resolved_commit = Some("abc123".to_owned());
        metadata.profile = Some("profile-a".to_owned());
        let request = BackendRequest::chat_completion(
            "model-a",
            "hello",
            BackendChatContext {
                messages: vec![BackendChatMessage {
                    role: BackendChatRole::User,
                    content: Some("hello".to_owned()),
                    name: None,
                    tool_call_id: None,
                    tool_calls: Vec::new(),
                }],
                tools: Vec::new(),
            },
            Some(1),
            SamplingConfig::Greedy,
            None,
            true,
            llm_backend_contracts::BackendCacheContext::chat_template_with_kwargs(
                "chatml/qwen/v1",
                Some("schema-a".to_owned()),
                Some(r#"{"enable_thinking":false}"#.to_owned()),
            ),
        );
        let expected_cache_key = request.cache_context().key.as_str().to_owned();
        let tokenizer_identity = driver_test_tokenizer_identity();

        let namespace = native_text_prefix_namespace(NativeTextPrefixNamespaceContext {
            model_id: "model-a",
            metadata: &metadata,
            tokenizer_identity: &tokenizer_identity,
            adapter_settings: "native-test-adapter/v1",
            request: &request,
            cache_layout_version: 7,
            cache_tokens: 64,
            max_prefill_tokens: 8,
        });

        assert_eq!(namespace.model_id, "model-a");
        assert_eq!(namespace.backend, "native-test");
        assert_eq!(namespace.family.as_deref(), Some("test"));
        assert_eq!(namespace.quantization.as_deref(), Some("bf16"));
        assert_eq!(namespace.repo_id.as_deref(), Some("org/model"));
        assert_eq!(namespace.resolved_commit.as_deref(), Some("abc123"));
        assert_eq!(namespace.profile.as_deref(), Some("profile-a"));
        assert_eq!(namespace.tokenizer_kind, "huggingface-tokenizer-json");
        assert!(namespace.tokenizer_hash.starts_with("sha256:"));
        assert_eq!(
            namespace.tokenizer_normalization,
            "llm-tokenizer/hf-json/v1"
        );
        assert_eq!(namespace.cache_template_id, "chatml/qwen/v1");
        assert!(
            namespace
                .chat_template_kwargs_hash
                .as_deref()
                .is_some_and(|hash| hash.starts_with("sha256:"))
        );
        assert_eq!(namespace.adapter_settings, "native-test-adapter/v1");
        assert_eq!(namespace.cache_key, expected_cache_key);
        assert_eq!(namespace.tool_schema.as_deref(), Some("schema-a"));
        assert_eq!(
            namespace.request_mode,
            "chat,json_object=true,required_tool=None"
        );
        assert_eq!(namespace.cache_layout_version, 7);
        assert_eq!(namespace.cache_tokens, 64);
        assert_eq!(namespace.max_prefill_tokens, 8);
    }

    #[test]
    fn prefix_cache_reuses_namespace_when_only_sampling_changes() {
        let metadata = BackendModelMetadata::new("model-a", "native-test").with_family("qwen");
        let greedy_request = BackendRequest::raw_completion_with_cache_context(
            "model-a",
            "hello",
            Some(1),
            SamplingConfig::Greedy,
            llm_backend_contracts::BackendCacheContext::chat_template_with_kwargs(
                "chatml/qwen/v1",
                Some("schema-a".to_owned()),
                Some(r#"{"enable_thinking":false}"#.to_owned()),
            ),
        );
        let mut top_p_request = greedy_request.clone();
        top_p_request.sampling = SamplingConfig::TopP {
            temperature: 0.7,
            top_p: 0.8,
        };
        let tokenizer_identity = driver_test_tokenizer_identity();
        let namespace_for = |request: &BackendRequest| {
            native_text_prefix_namespace(NativeTextPrefixNamespaceContext {
                model_id: "model-a",
                metadata: &metadata,
                tokenizer_identity: &tokenizer_identity,
                adapter_settings: "native-test-adapter/v1",
                request,
                cache_layout_version: 1,
                cache_tokens: 16,
                max_prefill_tokens: 8,
            })
        };
        let greedy_namespace = namespace_for(&greedy_request);
        let top_p_namespace = namespace_for(&top_p_request);
        let cache = NativeTextPrefixCache::new(1024);
        let metrics = NativeTextPrefixCacheMetrics::default();
        let caches = vec![TestCache {
            bytes: 8,
            marker: 1,
        }];

        cache.store(
            greedy_namespace.clone(),
            &[1, 2],
            &[0.25, 0.75],
            &caches,
            &metrics,
        );

        assert_eq!(greedy_namespace, top_p_namespace);
        assert!(
            cache
                .lookup(&top_p_namespace, &[1, 2, 3], &metrics)
                .is_some(),
            "sampling controls are intentionally outside the prefix cache namespace"
        );
    }

    #[test]
    fn prefix_cache_namespace_separates_cache_capacity() {
        let cache = NativeTextPrefixCache::new(1024);
        let metrics = NativeTextPrefixCacheMetrics::default();
        let namespace = namespace("capacity-key");
        let larger_capacity_namespace = NativeTextPrefixCacheNamespace {
            cache_tokens: namespace.cache_tokens * 2,
            ..namespace.clone()
        };

        cache.store(
            namespace.clone(),
            &[1, 2],
            &[0.25, 0.75],
            &[TestCache {
                bytes: 8,
                marker: 1,
            }],
            &metrics,
        );

        assert_ne!(namespace, larger_capacity_namespace);
        assert!(
            cache
                .lookup(&larger_capacity_namespace, &[1, 2], &metrics)
                .is_none(),
            "cache capacity buckets are prefix cache compatibility keys"
        );
    }

    #[test]
    fn prefix_cache_namespace_separates_manifest_identity_and_profile() {
        let cache = NativeTextPrefixCache::new(1024);
        let metrics = NativeTextPrefixCacheMetrics::default();
        let namespace = namespace("manifest");
        let different_manifest_namespace = NativeTextPrefixCacheNamespace {
            resolved_commit: Some("def456".to_owned()),
            ..namespace.clone()
        };
        let different_profile_namespace = NativeTextPrefixCacheNamespace {
            profile: Some("profile-b".to_owned()),
            ..namespace.clone()
        };

        cache.store(
            namespace.clone(),
            &[1, 2],
            &[0.25, 0.75],
            &[TestCache {
                bytes: 8,
                marker: 1,
            }],
            &metrics,
        );

        assert_ne!(namespace, different_manifest_namespace);
        assert_ne!(namespace, different_profile_namespace);
        assert!(
            cache
                .lookup(&different_manifest_namespace, &[1, 2], &metrics)
                .is_none(),
            "manifest identity changes must not reuse prefix state"
        );
        assert!(
            cache
                .lookup(&different_profile_namespace, &[1, 2], &metrics)
                .is_none(),
            "profile changes must not reuse prefix state"
        );
    }

    #[test]
    fn prefix_cache_namespace_separates_tokenizer_template_adapter_and_bucket_identity() {
        let cache = NativeTextPrefixCache::new(1024);
        let metrics = NativeTextPrefixCacheMetrics::default();
        let namespace = namespace("shared-route");
        let mismatches = [
            NativeTextPrefixCacheNamespace {
                tokenizer_kind: "different-tokenizer-kind".to_owned(),
                ..namespace.clone()
            },
            NativeTextPrefixCacheNamespace {
                tokenizer_hash: "sha256:different-tokenizer".to_owned(),
                ..namespace.clone()
            },
            NativeTextPrefixCacheNamespace {
                tokenizer_normalization: "llm-tokenizer/hf-json/v2".to_owned(),
                ..namespace.clone()
            },
            NativeTextPrefixCacheNamespace {
                cache_template_id: "template/shared-route/v2".to_owned(),
                ..namespace.clone()
            },
            NativeTextPrefixCacheNamespace {
                chat_template_kwargs_hash: Some("sha256:different-template-kwargs".to_owned()),
                ..namespace.clone()
            },
            NativeTextPrefixCacheNamespace {
                adapter_settings: "native-test-adapter/shared-route/v2".to_owned(),
                ..namespace.clone()
            },
            NativeTextPrefixCacheNamespace {
                cache_tokens: namespace.cache_tokens * 2,
                ..namespace.clone()
            },
        ];

        cache.store(
            namespace.clone(),
            &[1, 2],
            &[0.25, 0.75],
            &[TestCache {
                bytes: 8,
                marker: 1,
            }],
            &metrics,
        );

        assert!(cache.lookup(&namespace, &[1, 2, 3], &metrics).is_some());
        for mismatch in mismatches {
            assert!(
                cache.lookup(&mismatch, &[1, 2, 3], &metrics).is_none(),
                "incompatible shared-prefix identity must miss: {mismatch:?}"
            );
        }
    }

    #[test]
    fn prefix_namespace_identity_changes_with_chat_template_kwargs() {
        let metadata = BackendModelMetadata::new("model-a", "native-test").with_family("qwen");
        let mut request = driver_test_request(1);
        *request.cache_context_mut() =
            llm_backend_contracts::BackendCacheContext::chat_template_with_kwargs(
                "chatml/qwen/v1",
                None,
                Some(r#"{"enable_thinking":false}"#.to_owned()),
            );
        let tokenizer_identity = driver_test_tokenizer_identity();

        let no_thinking = native_text_prefix_namespace(NativeTextPrefixNamespaceContext {
            model_id: "model-a",
            metadata: &metadata,
            tokenizer_identity: &tokenizer_identity,
            adapter_settings: "native-test-adapter/v1",
            request: &request,
            cache_layout_version: 1,
            cache_tokens: 16,
            max_prefill_tokens: 8,
        });
        *request.cache_context_mut() =
            llm_backend_contracts::BackendCacheContext::chat_template_with_kwargs(
                "chatml/qwen/v1",
                None,
                Some(r#"{"enable_thinking":true}"#.to_owned()),
            );
        let thinking = native_text_prefix_namespace(NativeTextPrefixNamespaceContext {
            model_id: "model-a",
            metadata: &metadata,
            tokenizer_identity: &tokenizer_identity,
            adapter_settings: "native-test-adapter/v1",
            request: &request,
            cache_layout_version: 1,
            cache_tokens: 16,
            max_prefill_tokens: 8,
        });

        assert_ne!(no_thinking, thinking);
        assert_ne!(no_thinking.cache_key, thinking.cache_key);
    }

    #[test]
    fn prefix_namespace_identity_changes_with_tool_schema_and_request_mode() {
        fn chat_request(
            tool_schema: &str,
            required_tool_choice: Option<BackendToolChoice>,
            json_object_mode: bool,
        ) -> BackendRequest {
            BackendRequest::chat_completion(
                "model-a",
                "hello",
                BackendChatContext {
                    messages: vec![BackendChatMessage {
                        role: BackendChatRole::User,
                        content: Some("hello".to_owned()),
                        name: None,
                        tool_call_id: None,
                        tool_calls: Vec::new(),
                    }],
                    tools: Vec::new(),
                },
                Some(1),
                SamplingConfig::Greedy,
                required_tool_choice,
                json_object_mode,
                llm_backend_contracts::BackendCacheContext::chat_template(
                    "chatml/qwen/v1",
                    Some(tool_schema.to_owned()),
                ),
            )
        }

        let metadata = BackendModelMetadata::new("model-a", "native-test").with_family("qwen");
        let base_request = chat_request("schema-a", None, false);
        let different_schema_request = chat_request("schema-b", None, false);
        let required_tool_request = chat_request(
            "schema-a",
            Some(BackendToolChoice::RequiredFunction("lookup".to_owned())),
            false,
        );
        let tokenizer_identity = driver_test_tokenizer_identity();

        let namespace_for = |request: &BackendRequest| {
            native_text_prefix_namespace(NativeTextPrefixNamespaceContext {
                model_id: "model-a",
                metadata: &metadata,
                tokenizer_identity: &tokenizer_identity,
                adapter_settings: "native-test-adapter/v1",
                request,
                cache_layout_version: 1,
                cache_tokens: 16,
                max_prefill_tokens: 8,
            })
        };
        let base = namespace_for(&base_request);
        let different_schema = namespace_for(&different_schema_request);
        let required_tool = namespace_for(&required_tool_request);

        assert_ne!(base, different_schema);
        assert_ne!(base.cache_key, different_schema.cache_key);
        assert_ne!(base.tool_schema, different_schema.tool_schema);
        assert_ne!(base, required_tool);
        assert_ne!(base.request_mode, required_tool.request_mode);
    }

    #[test]
    fn prefix_cache_namespace_separates_required_tool_choice_names() {
        fn chat_request(required_tool_name: &str) -> BackendRequest {
            BackendRequest::chat_completion(
                "model-a",
                "hello",
                BackendChatContext {
                    messages: vec![BackendChatMessage {
                        role: BackendChatRole::User,
                        content: Some("hello".to_owned()),
                        name: None,
                        tool_call_id: None,
                        tool_calls: Vec::new(),
                    }],
                    tools: Vec::new(),
                },
                Some(1),
                SamplingConfig::Greedy,
                Some(BackendToolChoice::RequiredFunction(
                    required_tool_name.to_owned(),
                )),
                false,
                llm_backend_contracts::BackendCacheContext::chat_template(
                    "chatml/qwen/v1",
                    Some("schema-a".to_owned()),
                ),
            )
        }

        let metadata = BackendModelMetadata::new("model-a", "native-test").with_family("qwen");
        let lookup_request = chat_request("lookup");
        let search_request = chat_request("search");
        let tokenizer_identity = driver_test_tokenizer_identity();
        let namespace_for = |request: &BackendRequest| {
            native_text_prefix_namespace(NativeTextPrefixNamespaceContext {
                model_id: "model-a",
                metadata: &metadata,
                tokenizer_identity: &tokenizer_identity,
                adapter_settings: "native-test-adapter/v1",
                request,
                cache_layout_version: 1,
                cache_tokens: 16,
                max_prefill_tokens: 8,
            })
        };
        let lookup_namespace = namespace_for(&lookup_request);
        let search_namespace = namespace_for(&search_request);
        let cache = NativeTextPrefixCache::new(1024);
        let metrics = NativeTextPrefixCacheMetrics::default();

        cache.store(
            lookup_namespace.clone(),
            &[1, 2],
            &[0.25, 0.75],
            &[TestCache {
                bytes: 8,
                marker: 1,
            }],
            &metrics,
        );

        assert_ne!(lookup_namespace, search_namespace);
        assert_ne!(lookup_namespace.request_mode, search_namespace.request_mode);
        assert!(
            cache.lookup(&search_namespace, &[1, 2], &metrics).is_none(),
            "required tool-choice names are prefix cache compatibility keys"
        );
    }

    #[test]
    fn stop_tokens_match_literal_ids_and_tokenizer_tokens() {
        let tokenizer_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/qwen36/tokenizer.json");
        let tokenizer = HuggingFaceTokenizer::from_file(tokenizer_path).expect("tokenizer loads");
        let im_end = tokenizer
            .token_to_id("<|im_end|>")
            .expect("qwen tokenizer has im_end token") as usize;
        let stop_tokens = NativeTextStopTokens {
            token_ids: &[1],
            token_strings: &["<|im_end|>"],
            encoded_token_strings: &[],
        };
        let non_stop = (0..16)
            .find(|token_id| *token_id != 1 && *token_id != im_end)
            .expect("small non-stop token id exists");

        let resolved = stop_tokens.resolve(&tokenizer);

        assert_eq!(resolved.token_ids(), vec![1, im_end]);
        assert!(resolved.contains(1));
        assert!(resolved.contains(im_end));
        assert!(!resolved.contains(non_stop));
        assert!(
            !resolved.contains(im_end + (u32::MAX as usize) + 1),
            "candidate ids above u32::MAX must not wrap into a tokenizer stop token"
        );

        const PHRASE_STOP_STRINGS: &[&str] = &["hello rust tokenizer"];
        let missing_literal = PHRASE_STOP_STRINGS[0];
        assert!(
            tokenizer.token_to_id(missing_literal).is_none(),
            "fixture phrase should not be treated as a single literal stop token"
        );
        let literal_only_stop_tokens = NativeTextStopTokens {
            token_ids: &[],
            token_strings: PHRASE_STOP_STRINGS,
            encoded_token_strings: &[],
        }
        .resolve(&tokenizer);
        assert_eq!(literal_only_stop_tokens.token_ids(), Vec::<usize>::new());

        let phrase = PHRASE_STOP_STRINGS[0];
        assert!(
            tokenizer.token_to_id(phrase).is_none(),
            "fixture phrase should exercise encode fallback for non-vocabulary stop strings"
        );
        let phrase_ids = tokenizer
            .encode(phrase, false)
            .expect("fixture phrase encodes");
        let phrase_stop_tokens = NativeTextStopTokens {
            token_ids: &[],
            token_strings: &[],
            encoded_token_strings: PHRASE_STOP_STRINGS,
        }
        .resolve(&tokenizer);
        for token_id in phrase_ids {
            assert!(phrase_stop_tokens.contains(token_id as usize));
        }
    }

    #[test]
    fn driver_stop_token_candidate_is_not_emitted_for_blocking_generation() {
        let driver = driver_for_test(TestAdapter::new([1_usize]).with_stop_tokens(
            NativeTextStopTokens {
                token_ids: &[1],
                token_strings: &[],
                encoded_token_strings: &[],
            },
        ));

        let output = driver
            .generate_blocking(driver_test_request(4), CancellationToken::new())
            .expect("generation stops cleanly");

        assert_eq!(output.text, "");
        assert_eq!(output.completion_tokens, 0);
        assert_eq!(output.finish_reason, BackendFinishReason::Stop);
    }

    #[test]
    fn driver_stop_token_candidate_is_not_emitted_for_streaming_generation() {
        let driver = driver_for_test(TestAdapter::new([1_usize]).with_stop_tokens(
            NativeTextStopTokens {
                token_ids: &[1],
                token_strings: &[],
                encoded_token_strings: &[],
            },
        ));
        let (tx, mut rx) = tokio::sync::mpsc::channel(2);

        driver
            .generate_blocking_stream(driver_test_request(4), tx, CancellationToken::new())
            .expect("streaming generation stops cleanly");
        let final_chunk = loop {
            let chunk = rx
                .blocking_recv()
                .expect("final chunk is sent")
                .expect("final chunk is ok");
            if chunk.finish_reason.is_some() {
                break chunk;
            }
            assert_eq!(chunk.text, "");
            assert_eq!(chunk.completion_tokens, 0);
        };
        assert_eq!(final_chunk.text, "");
        assert_eq!(final_chunk.completion_tokens, 0);
        assert_eq!(final_chunk.finish_reason, Some(BackendFinishReason::Stop));
        assert!(rx.blocking_recv().is_none());
    }

    #[test]
    fn streaming_generation_decodes_each_output_token_once() {
        let adapter = TestAdapter::new([1_usize, 2, 3, 4]);
        let full_decode_token_total = adapter.decoded_token_total();
        let stream_decoded_token_total = adapter.stream_decoded_token_total();
        let driver = driver_for_test(adapter);
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);

        driver
            .generate_blocking_stream(driver_test_request(4), tx, CancellationToken::new())
            .expect("streaming generation succeeds");

        let mut text = String::new();
        while let Some(chunk) = rx.blocking_recv() {
            let chunk = chunk.expect("stream chunk is ok");
            text.push_str(&chunk.text);
        }

        assert_eq!(text, "<1><2><3><4>");
        assert_eq!(stream_decoded_token_total.load(Ordering::SeqCst), 4);
        assert_eq!(full_decode_token_total.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn streaming_generation_preserves_unicode_token_boundaries() {
        let driver = driver_for_test(TestAdapter::new([1_usize, 2]).with_unicode_boundary_decode());
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);

        driver
            .generate_blocking_stream(driver_test_request(2), tx, CancellationToken::new())
            .expect("streaming generation succeeds");

        let mut text = String::new();
        while let Some(chunk) = rx.blocking_recv() {
            let chunk = chunk.expect("stream chunk is ok");
            text.push_str(&chunk.text);
        }

        assert_eq!(text, "é");
    }

    #[test]
    fn streaming_generation_decode_work_scales_with_output_tokens() {
        let script = (1_usize..=64).collect::<Vec<_>>();
        let adapter = TestAdapter::new(std::sync::Arc::<[usize]>::from(script.clone()))
            .with_max_position_embeddings(128);
        let stream_decoded_token_total = adapter.stream_decoded_token_total();
        let full_decode_token_total = adapter.decoded_token_total();
        let driver = driver_for_test(adapter).with_max_new_tokens(64);
        let (tx, mut rx) = tokio::sync::mpsc::channel(128);

        driver
            .generate_blocking_stream(driver_test_request(64), tx, CancellationToken::new())
            .expect("streaming generation succeeds");

        let mut text = String::new();
        while let Some(chunk) = rx.blocking_recv() {
            let chunk = chunk.expect("stream chunk is ok");
            text.push_str(&chunk.text);
        }

        let expected = script
            .iter()
            .map(|token_id| format!("<{token_id}>"))
            .collect::<String>();
        assert_eq!(text, expected);
        assert_eq!(stream_decoded_token_total.load(Ordering::SeqCst), 64);
        assert_eq!(full_decode_token_total.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn driver_supplies_rng_draws_only_for_non_greedy_sampling() {
        let greedy_adapter = TestAdapter::new([1_usize, 2]);
        let greedy_draws = greedy_adapter.sampling_draws();
        let greedy_driver = driver_for_test(greedy_adapter);

        greedy_driver
            .generate_blocking(driver_test_request(2), CancellationToken::new())
            .expect("greedy generation succeeds");

        assert_eq!(
            *greedy_draws
                .lock()
                .expect("greedy sampling draws lock is not poisoned"),
            vec![None, None]
        );

        let top_p_adapter = TestAdapter::new([1_usize, 2]);
        let top_p_draws = top_p_adapter.sampling_draws();
        let top_p_driver = driver_for_test(top_p_adapter);
        let mut request = driver_test_request(2);
        request.sampling = SamplingConfig::TopP {
            temperature: 1.0,
            top_p: 0.9,
        };

        top_p_driver
            .generate_blocking(request, CancellationToken::new())
            .expect("top-p generation succeeds");

        let top_p_draws = top_p_draws
            .lock()
            .expect("top-p sampling draws lock is not poisoned")
            .clone();
        assert_eq!(top_p_draws.len(), 2);
        assert!(
            top_p_draws
                .into_iter()
                .all(|draw| { matches!(draw, Some(value) if (0.0..1.0).contains(&value)) })
        );
    }

    #[test]
    fn driver_reports_prefix_cache_miss_and_hit_for_blocking_generation() {
        let driver = driver_for_test(TestAdapter::new([1_usize]));

        let first = driver
            .generate_blocking(driver_test_request(1), CancellationToken::new())
            .expect("first generation succeeds");
        let second = driver
            .generate_blocking(driver_test_request(1), CancellationToken::new())
            .expect("second generation succeeds");

        assert_eq!(first.prompt_cached_tokens, Some(0));
        assert_eq!(second.prompt_cached_tokens, Some(1));
    }

    #[test]
    fn driver_records_prefill_and_avoided_work_for_warm_prefix() {
        let adapter = TestAdapter::new([1_usize]).with_encoded_prompt([10_u32, 11, 12, 13, 14]);
        let metrics = Arc::clone(&adapter.prefix_cache_metrics);
        let driver = driver_for_test(adapter).with_max_prefill_tokens(2);

        let first = driver
            .generate_blocking(driver_test_request(1), CancellationToken::new())
            .expect("cold generation succeeds");
        let second = driver
            .generate_blocking(driver_test_request(1), CancellationToken::new())
            .expect("warm generation succeeds");

        assert_eq!(first.prompt_cached_tokens, Some(0));
        assert_eq!(second.prompt_cached_tokens, Some(5));
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot["prefill_chunks"], 3);
        assert_eq!(snapshot["prefill_tokens"], 5);
        assert_eq!(snapshot["hit_tokens"], 5);
        assert_eq!(snapshot["miss_tokens"], 5);
        assert_eq!(snapshot["avoided_prefill_tokens"], 5);
    }

    #[test]
    fn driver_records_shared_prefix_reuse_without_exposing_state() {
        let request = driver_test_request(1);
        let adapter = TestAdapter::new([1_usize]).with_encoded_prompt([10_u32, 11, 12, 13]);
        let metrics = Arc::clone(&adapter.prefix_cache_metrics);
        store_driver_prefix_hit(
            &adapter,
            &request,
            4,
            1,
            &[10, 11],
            TestCache {
                bytes: 8,
                marker: 77,
            },
        );
        let driver = driver_for_test(adapter).with_max_prefill_tokens(2);

        let output = driver
            .generate_blocking(request, CancellationToken::new())
            .expect("generation reuses compatible shared prefix");

        assert_eq!(output.prompt_cached_tokens, Some(2));
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot["shared_prefix_hits"], 1);
        assert_eq!(snapshot["shared_prefix_reused_tokens"], 2);
        assert_eq!(snapshot.get("shared_prefix_states"), None);
    }

    #[test]
    fn driver_reports_prefix_cache_miss_and_hit_for_streaming_generation() {
        let driver = driver_for_test(TestAdapter::new([1_usize]));

        let first = stream_final_chunk(&driver, driver_test_request(1));
        let second = stream_final_chunk(&driver, driver_test_request(1));

        assert_eq!(first.prompt_cached_tokens, Some(0));
        assert_eq!(second.prompt_cached_tokens, Some(1));
    }

    #[test]
    fn streaming_generation_emits_prefill_progress_after_each_uncached_chunk() {
        let driver = driver_for_test(
            TestAdapter::new([1_usize]).with_encoded_prompt([10_u32, 11, 12, 13, 14]),
        )
        .with_max_prefill_tokens(2);
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);

        driver
            .generate_blocking_stream(driver_test_request(1), tx, CancellationToken::new())
            .expect("streaming generation succeeds");

        let mut progress = Vec::new();
        while let Some(chunk) = rx.blocking_recv() {
            let chunk = chunk.expect("stream chunk is ok");
            if let Some(event) = chunk.progress {
                progress.push(event);
            }
        }

        assert_eq!(
            progress,
            vec![
                BackendStreamProgress::PrefillProgress {
                    chunk: 1,
                    total: 3,
                    tokens: 2,
                    total_tokens: 5,
                },
                BackendStreamProgress::PrefillProgress {
                    chunk: 2,
                    total: 3,
                    tokens: 4,
                    total_tokens: 5,
                },
                BackendStreamProgress::PrefillProgress {
                    chunk: 3,
                    total: 3,
                    tokens: 5,
                    total_tokens: 5,
                },
            ]
        );
    }

    #[test]
    fn streaming_generation_waits_for_prefill_admission_before_next_uncached_chunk() {
        let adapter = TestAdapter::new([1_usize]).with_encoded_prompt([10_u32, 11, 12, 13]);
        let prefill_chunk_calls = adapter.prefill_chunk_calls();
        let driver = driver_for_test(adapter).with_max_prefill_tokens(2);
        let admission = Arc::new(BlockingPrefillAdmission::new());
        let request = driver_test_request(1).with_prefill_chunk_admission(
            BackendPrefillChunkAdmissionHook::new(Arc::clone(&admission)),
        );
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let worker = std::thread::spawn({
            let driver = driver.clone();
            move || {
                driver
                    .block_on_worker(driver.generate_stream_async(
                        request,
                        tx,
                        CancellationToken::new(),
                    ))
                    .expect("native stream worker runtime succeeds")
            }
        });

        let first = rx
            .blocking_recv()
            .expect("first prefill progress arrives")
            .expect("first prefill progress succeeds");
        assert_eq!(
            first.progress,
            Some(BackendStreamProgress::PrefillProgress {
                chunk: 1,
                total: 2,
                tokens: 2,
                total_tokens: 4,
            })
        );
        let deadline = Instant::now() + Duration::from_millis(500);
        while admission.calls.load(Ordering::SeqCst) == 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(admission.calls.load(Ordering::SeqCst), 1);
        assert_eq!(prefill_chunk_calls.load(Ordering::SeqCst), 1);

        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(
            prefill_chunk_calls.load(Ordering::SeqCst),
            1,
            "native worker must not start the next prefill chunk before admission"
        );

        admission.release.notify_waiters();
        let mut saw_final = false;
        while let Some(chunk) = rx.blocking_recv() {
            let chunk = chunk.expect("stream chunk succeeds after readmission");
            if chunk.finish_reason.is_some() {
                saw_final = true;
            }
        }
        worker
            .join()
            .expect("native stream worker joins")
            .expect("native stream generation succeeds");
        assert!(saw_final);
        assert_eq!(prefill_chunk_calls.load(Ordering::SeqCst), 2);
        assert_eq!(admission.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn driver_reuses_mid_prefill_checkpoint_after_cancellation() {
        let request = driver_test_request(1);
        let cancellation = CancellationToken::new();
        let adapter = TestAdapter::new([1_usize])
            .with_encoded_prompt([10_u32, 11, 12, 13, 14])
            .with_prefill_cancellation_after_chunk(cancellation.clone(), 2);
        let prefix_cache = Arc::clone(&adapter.prefix_cache);
        let metrics = Arc::clone(&adapter.prefix_cache_metrics);
        let namespace = driver_prefix_namespace(&adapter, &request, 5, 1);
        let driver = driver_for_test(adapter).with_max_prefill_tokens(2);

        let err = driver
            .generate_blocking(request.clone(), cancellation)
            .expect_err("prefill cancellation is returned");
        assert!(err.is_cancelled());
        assert_prefix_cache_entry(&prefix_cache, &namespace, &[10, 11]);
        assert_prefix_cache_entry(&prefix_cache, &namespace, &[10, 11, 12, 13]);
        assert_no_prefix_cache_entry(&prefix_cache, &namespace, &[10, 11, 12, 13, 14]);

        let warm = driver
            .generate_blocking(request, CancellationToken::new())
            .expect("warm generation reuses checkpoint and completes suffix");

        assert_eq!(warm.prompt_cached_tokens, Some(4));
        assert_eq!(warm.text, "<1>");
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot["checkpoint_stores"], 2);
        assert_eq!(snapshot["checkpoint_store_tokens"], 6);
        assert_eq!(snapshot["checkpoint_reuse_hits"], 1);
        assert_eq!(snapshot["checkpoint_reused_tokens"], 4);
    }

    #[test]
    fn driver_does_not_checkpoint_failed_prefill_chunk() {
        let request = driver_test_request(1);
        let adapter = TestAdapter::new([1_usize])
            .with_encoded_prompt([30_u32, 31, 32, 33, 34])
            .with_prefill_failure_after_chunk(2);
        let prefix_cache = Arc::clone(&adapter.prefix_cache);
        let metrics = Arc::clone(&adapter.prefix_cache_metrics);
        let namespace = driver_prefix_namespace(&adapter, &request, 5, 1);
        let driver = driver_for_test(adapter).with_max_prefill_tokens(2);

        let err = driver
            .generate_blocking(request.clone(), CancellationToken::new())
            .expect_err("prefill failure is returned");
        assert!(err.to_string().contains("test prefill failed"));
        assert_prefix_cache_entry(&prefix_cache, &namespace, &[30, 31]);
        assert_no_prefix_cache_entry(&prefix_cache, &namespace, &[30, 31, 32, 33]);
        assert_no_prefix_cache_entry(&prefix_cache, &namespace, &[30, 31, 32, 33, 34]);

        let warm = driver
            .generate_blocking(request, CancellationToken::new())
            .expect("warm generation reuses only the successful checkpoint");

        assert_eq!(warm.prompt_cached_tokens, Some(2));
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot["checkpoint_reuse_hits"], 1);
        assert_eq!(snapshot["checkpoint_reused_tokens"], 2);
    }

    #[test]
    fn driver_allows_adapter_context_sensitive_candidate_observation() {
        let driver = driver_for_test(ContextSensitiveTestAdapter::new([7_usize, 8_usize], 1));

        let output = driver
            .generate_blocking(driver_test_request(4), CancellationToken::new())
            .expect("generation stops through adapter hook");

        assert_eq!(output.text, "<7>");
        assert_eq!(output.completion_tokens, 1);
        assert_eq!(output.finish_reason, BackendFinishReason::Stop);
    }

    #[test]
    fn driver_cleans_cache_mirrors_when_prefill_fails_before_session_handoff() {
        let adapter = TestAdapter::new([1_usize]).with_prefill_failure();
        let cleanup_calls = adapter.cleanup_calls();
        let driver = driver_for_test(adapter);

        let err = driver
            .generate_blocking(driver_test_request(1), CancellationToken::new())
            .expect_err("prefill failure is returned");

        assert!(err.to_string().contains("test prefill failed"));
        assert_eq!(cleanup_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn driver_cleans_cloned_prefix_cache_when_suffix_prefill_fails() {
        let request = driver_test_request(1);
        let adapter = TestAdapter::new([1_usize]).with_encoded_prompt([10_u32, 11_u32, 12_u32]);
        let cleanup_calls = adapter.cleanup_calls();
        let cleanup_markers = adapter.cleanup_markers();
        let prefix_cache = Arc::clone(&adapter.prefix_cache);
        let metrics = Arc::clone(&adapter.prefix_cache_metrics);
        let (namespace, byte_len) = store_driver_prefix_hit(
            &adapter,
            &request,
            3,
            1,
            &[10, 11],
            TestCache {
                bytes: 13,
                marker: 77,
            },
        );
        let driver = driver_for_test(adapter.with_prefill_failure());

        let err = driver
            .generate_blocking(request, CancellationToken::new())
            .expect_err("suffix prefill failure is returned");

        assert!(err.to_string().contains("test prefill failed"));
        assert_eq!(cleanup_calls.load(Ordering::SeqCst), 1);
        let markers = cleanup_markers
            .lock()
            .expect("cleanup markers lock is not poisoned")
            .clone();
        assert_eq!(markers, vec![vec![77]]);
        assert_only_prefix_cache_entry(&prefix_cache, &namespace, &[10, 11], byte_len);
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot["hits"], 1);
        assert_eq!(snapshot["hit_tokens"], 2);
        assert_eq!(snapshot["miss_tokens"], 1);
        assert_eq!(snapshot["stores"], 1);
        assert_eq!(snapshot["resident_entries"], 1);
        assert_eq!(snapshot["resident_bytes"], byte_len);
    }

    #[test]
    fn driver_cleans_cloned_prefix_cache_when_suffix_prefill_cancels() {
        let request = driver_test_request(1);
        let cancellation = CancellationToken::new();
        let adapter = TestAdapter::new([1_usize]).with_encoded_prompt([20_u32, 21_u32, 22_u32]);
        let cleanup_calls = adapter.cleanup_calls();
        let cleanup_markers = adapter.cleanup_markers();
        let prefix_cache = Arc::clone(&adapter.prefix_cache);
        let metrics = Arc::clone(&adapter.prefix_cache_metrics);
        let (namespace, byte_len) = store_driver_prefix_hit(
            &adapter,
            &request,
            3,
            1,
            &[20, 21],
            TestCache {
                bytes: 17,
                marker: 88,
            },
        );
        let driver = driver_for_test(adapter.with_prefill_cancellation(cancellation.clone()));

        let err = driver
            .generate_blocking(request, cancellation)
            .expect_err("suffix prefill cancellation is returned");

        assert!(err.is_cancelled());
        assert_eq!(cleanup_calls.load(Ordering::SeqCst), 1);
        let markers = cleanup_markers
            .lock()
            .expect("cleanup markers lock is not poisoned")
            .clone();
        assert_eq!(markers, vec![vec![88]]);
        assert_only_prefix_cache_entry(&prefix_cache, &namespace, &[20, 21], byte_len);
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot["hits"], 1);
        assert_eq!(snapshot["hit_tokens"], 2);
        assert_eq!(snapshot["miss_tokens"], 1);
        assert_eq!(snapshot["stores"], 1);
        assert_eq!(snapshot["resident_entries"], 1);
        assert_eq!(snapshot["resident_bytes"], byte_len);
    }

    #[test]
    fn driver_does_not_clean_cache_mirrors_after_successful_session_handoff() {
        let adapter = TestAdapter::new([1_usize]);
        let cleanup_calls = adapter.cleanup_calls();
        let driver = driver_for_test(adapter);

        driver
            .generate_blocking(driver_test_request(1), CancellationToken::new())
            .expect("generation succeeds");

        assert_eq!(cleanup_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn native_text_open_blocking_work_runs_off_async_runtime() {
        let work_started = Arc::new(AtomicUsize::new(0));
        let work_started_for_closure = Arc::clone(&work_started);

        let open = tokio::spawn(async move {
            run_native_text_open_blocking("Test", move || {
                work_started_for_closure.fetch_add(1, Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(750));
                Ok::<_, anyhow::Error>("opened")
            })
            .await
        });

        let deadline = tokio::time::Instant::now() + Duration::from_millis(200);
        while work_started.load(Ordering::SeqCst) == 0 && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        assert_eq!(work_started.load(Ordering::SeqCst), 1);
        assert!(
            !open.is_finished(),
            "native text snapshot open work should not block the async runtime"
        );
        assert_eq!(
            open.await
                .expect("open task joins")
                .expect("blocking open work succeeds"),
            "opened"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn driver_generate_with_cancel_runs_native_work_off_async_runtime() {
        let adapter = TestAdapter::new([1_usize]).with_next_token_delay(Duration::from_millis(750));
        let next_token_calls = adapter.next_token_calls();
        let driver = driver_for_test(adapter);

        let generation = tokio::spawn(async move {
            driver
                .generate_with_cancel(driver_test_request(1), CancellationToken::new())
                .await
        });

        let deadline = tokio::time::Instant::now() + Duration::from_millis(200);
        while next_token_calls.load(Ordering::SeqCst) == 0 && tokio::time::Instant::now() < deadline
        {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        assert_eq!(next_token_calls.load(Ordering::SeqCst), 1);
        assert!(
            !generation.is_finished(),
            "native generation should not block the async runtime while CPU work is running"
        );
        let output = generation
            .await
            .expect("generation task joins")
            .expect("generation succeeds");
        assert_eq!(output.completion_tokens, 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn driver_generate_with_cancel_cancels_worker_when_future_is_dropped() {
        let adapter = TestAdapter::new([1_usize]).with_next_token_delay(Duration::from_millis(750));
        let next_token_calls = adapter.next_token_calls();
        let driver = driver_for_test(adapter);
        let cancellation = CancellationToken::new();
        let worker_cancellation = cancellation.clone();

        let generation = tokio::spawn(async move {
            driver
                .generate_with_cancel(driver_test_request(1), worker_cancellation)
                .await
        });

        let deadline = tokio::time::Instant::now() + Duration::from_millis(200);
        while next_token_calls.load(Ordering::SeqCst) == 0 && tokio::time::Instant::now() < deadline
        {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(next_token_calls.load(Ordering::SeqCst), 1);

        generation.abort();
        assert!(
            generation
                .await
                .expect_err("generation task is aborted")
                .is_cancelled()
        );
        assert!(
            cancellation.is_cancelled(),
            "dropping the async request future should signal the blocking native worker"
        );
    }

    #[test]
    fn prefill_context_returns_last_hidden_from_last_chunk() {
        let cancellation = CancellationToken::new();
        let mut observed_chunks = Vec::new();

        let mut prefill_caches = [TestCache {
            bytes: 0,
            marker: 0,
        }];
        let mut prefill_scratch = InferenceScratchpad::new();
        let hidden = native_text_prefill_context_with_cache(
            "Test",
            2,
            &[1, 2, 3],
            &mut prefill_caches,
            &cancellation,
            &mut prefill_scratch,
            |chunk, _caches, _scratch| {
                observed_chunks.push(chunk.to_vec());
                Ok(chunk
                    .iter()
                    .map(|token| vec![*token as f32, (*token * 10) as f32])
                    .collect())
            },
        )
        .expect("prefill succeeds");

        assert_eq!(observed_chunks, vec![vec![1, 2], vec![3]]);
        assert_eq!(hidden, vec![3.0, 30.0]);
    }

    #[test]
    fn prefill_context_observes_cancellation_between_chunks() {
        let cancellation = CancellationToken::new();
        let mut calls = 0;

        let mut cancel_caches = [TestCache {
            bytes: 0,
            marker: 0,
        }];
        let mut cancel_scratch = InferenceScratchpad::new();
        let err = native_text_prefill_context_with_cache(
            "Test",
            1,
            &[1, 2],
            &mut cancel_caches,
            &cancellation,
            &mut cancel_scratch,
            |chunk, _caches, _scratch| {
                calls += 1;
                assert_eq!(chunk, &[1]);
                cancellation.cancel();
                Ok(vec![vec![1.0]])
            },
        )
        .expect_err("cancelled after first chunk");

        assert!(err.is_cancelled());
        assert_eq!(calls, 1);
    }

    #[test]
    fn cache_token_capacity_uses_exact_budget_within_position_limit() {
        let capacity = native_text_cache_token_capacity(40, 8, 32, 64, "Test")
            .expect("context and generation budget fits");

        assert_eq!(capacity, 48);
    }

    #[test]
    fn cache_namespace_token_bucket_keeps_prefix_identity_stable() {
        let capacity = native_text_cache_token_capacity(40, 8, 32, 64, "Test")
            .expect("context and generation budget fits");
        let bucket = native_text_cache_namespace_token_bucket(capacity, 64, "Test")
            .expect("namespace bucket fits");

        assert_eq!(capacity, 48);
        assert_eq!(bucket, 64);
    }

    #[test]
    fn cache_token_capacity_rejects_invalid_position_limits() {
        let err = native_text_cache_token_capacity(0, 1, 1, 0, "Test")
            .expect_err("zero position limit fails closed");

        assert_eq!(
            err.backend_failure_class(),
            Some(BackendFailureClass::Config)
        );
        assert_eq!(err.backend_failure_code(), Some("backend_config_failed"));
        assert!(
            err.to_string()
                .contains("native Test model declares zero max_position_embeddings"),
            "error should identify the invalid model position limit: {err}"
        );
    }

    #[test]
    fn prefix_cache_reuses_longest_namespace_compatible_prefix() {
        let cache = NativeTextPrefixCache::new(1024);
        let metrics = NativeTextPrefixCacheMetrics::default();
        let namespace = namespace("base");
        let caches = vec![TestCache {
            bytes: 11,
            marker: 7,
        }];

        cache.store(namespace.clone(), &[1, 2], &[0.5, 1.5], &caches, &metrics);

        let hit = cache
            .lookup(&namespace, &[1, 2, 3], &metrics)
            .expect("longer prompt reuses compatible prefix");
        assert_eq!(hit.token_count, 2);
        assert_eq!(hit.hidden, vec![0.5, 1.5]);
        assert_eq!(hit.caches, caches);

        let incompatible = NativeTextPrefixCacheNamespace {
            cache_key: "different".to_owned(),
            ..namespace
        };
        assert!(cache.lookup(&incompatible, &[1, 2], &metrics).is_none());
    }

    #[test]
    fn prefix_cache_lookup_skips_capacity_incompatible_entries() {
        let cache = NativeTextPrefixCache::new(1024);
        let metrics = NativeTextPrefixCacheMetrics::default();
        let namespace = namespace("capacity");
        let caches = vec![TestCache {
            bytes: 11,
            marker: 7,
        }];

        cache.store(namespace.clone(), &[1, 2], &[0.5, 1.5], &caches, &metrics);

        assert!(
            cache
                .lookup_compatible(&namespace, &[1, 2, 3], &metrics, |caches| {
                    caches.iter().all(|cache| cache.marker != 7)
                })
                .is_none()
        );

        let hit = cache
            .lookup_compatible(&namespace, &[1, 2, 3], &metrics, |caches| {
                caches.iter().all(|cache| cache.marker == 7)
            })
            .expect("compatible entry is reusable");
        assert_eq!(hit.token_count, 2);
    }

    #[test]
    fn prefix_cache_stores_entries_in_namespace_buckets() {
        let cache = NativeTextPrefixCache::new(1024);
        let metrics = NativeTextPrefixCacheMetrics::default();
        let base_namespace = namespace("bucket");
        let other_namespace = namespace("other-bucket");
        let hidden = [1.0];
        let caches = [TestCache {
            bytes: 8,
            marker: 1,
        }];

        cache.store(base_namespace.clone(), &[1], &hidden, &caches, &metrics);
        cache.store(base_namespace.clone(), &[1, 2], &hidden, &caches, &metrics);
        cache.store(other_namespace.clone(), &[9], &hidden, &caches, &metrics);

        let inner = cache.inner.lock().expect("prefix cache lock is available");
        assert_eq!(inner.entries.len(), 2);
        assert_eq!(
            inner
                .entries
                .get(&base_namespace)
                .expect("namespace bucket exists")
                .len(),
            2
        );
        assert_eq!(
            inner
                .entries
                .get(&other_namespace)
                .expect("other namespace bucket exists")
                .len(),
            1
        );
    }

    #[test]
    fn prefix_cache_prefers_longest_prefix_over_recency_and_updates_lru() {
        let cache = NativeTextPrefixCache::new(48);
        let metrics = NativeTextPrefixCacheMetrics::default();
        let base_namespace = namespace("longest-lru");
        let other_namespace = namespace("longest-lru-other");
        let hidden = [1.0, 2.0, 3.0, 4.0];

        cache.store(
            base_namespace.clone(),
            &[1, 2, 3],
            &hidden,
            &[TestCache {
                bytes: 8,
                marker: 3,
            }],
            &metrics,
        );
        cache.store(
            base_namespace.clone(),
            &[1, 2],
            &hidden,
            &[TestCache {
                bytes: 8,
                marker: 2,
            }],
            &metrics,
        );

        let hit = cache
            .lookup(&base_namespace, &[1, 2, 3, 4], &metrics)
            .expect("matching prompt reuses longest prefix");
        assert_eq!(hit.token_count, 3);
        assert_eq!(hit.caches[0].marker, 3);

        cache.store(
            other_namespace.clone(),
            &[9],
            &hidden,
            &[TestCache {
                bytes: 8,
                marker: 9,
            }],
            &metrics,
        );

        assert!(
            cache.lookup(&base_namespace, &[1, 2], &metrics).is_none(),
            "shorter prefix should be least recently used after the longest-prefix hit"
        );
        assert!(
            cache
                .lookup(&base_namespace, &[1, 2, 3], &metrics)
                .is_some(),
            "longest-prefix hit should refresh that entry before eviction"
        );
        assert!(cache.lookup(&other_namespace, &[9], &metrics).is_some());
    }

    #[test]
    fn prefix_cache_clones_payloads_outside_global_lock() {
        let cache = Arc::new(NativeTextPrefixCache::new(1024));
        let metrics = NativeTextPrefixCacheMetrics::default();
        let namespace = namespace("clone-lock");
        let cloned_while_locked = Arc::new(AtomicUsize::new(0));
        let caches = vec![LockObservingCache {
            bytes: 8,
            cache: Arc::downgrade(&cache),
            cloned_while_locked: cloned_while_locked.clone(),
        }];

        cache.store(namespace.clone(), &[1, 2], &[0.5, 1.5], &caches, &metrics);
        let hit = cache
            .lookup(&namespace, &[1, 2, 3], &metrics)
            .expect("compatible longer prompt reuses stored prefix");

        assert_eq!(hit.token_count, 2);
        assert_eq!(
            cloned_while_locked.load(Ordering::SeqCst),
            0,
            "prefix cache must not clone layer-cache payloads while holding its global lock"
        );
    }

    #[test]
    fn prefix_cache_metrics_record_lookup_scans_and_hit_clone_bytes() {
        let cache = NativeTextPrefixCache::new(1024);
        let metrics = NativeTextPrefixCacheMetrics::default();
        let base_namespace = namespace("scan-metrics");
        let other_namespace = namespace("scan-metrics-other");
        let hidden = [1.0, 2.0, 3.0, 4.0];

        cache.store(
            base_namespace.clone(),
            &[1],
            &hidden,
            &[TestCache {
                bytes: 5,
                marker: 1,
            }],
            &metrics,
        );
        cache.store(
            base_namespace.clone(),
            &[1, 2],
            &hidden,
            &[TestCache {
                bytes: 7,
                marker: 2,
            }],
            &metrics,
        );
        cache.store(
            other_namespace,
            &[9],
            &hidden,
            &[TestCache {
                bytes: 11,
                marker: 9,
            }],
            &metrics,
        );

        let hit = cache
            .lookup(&base_namespace, &[1, 2, 3], &metrics)
            .expect("matching prompt reuses longest stored prefix");
        assert_eq!(hit.token_count, 2);
        assert!(cache.lookup(&base_namespace, &[42], &metrics).is_none());

        let snapshot = metrics.snapshot();
        assert_eq!(
            snapshot["entries_scanned"], 4,
            "lookups only scan entries in the matching namespace bucket"
        );
        assert_eq!(snapshot["namespace_entries_scanned"], 4);
        assert_eq!(
            snapshot["hit_clone_bytes"],
            std::mem::size_of_val(&hidden) as u64 + 7
        );
    }

    #[test]
    fn prefix_cache_uses_value_sizing_for_eviction_budget() {
        let cache = NativeTextPrefixCache::new(32);
        let metrics = NativeTextPrefixCacheMetrics::default();
        let namespace = namespace("budget");
        let hidden = vec![1.0; 4];

        cache.store(
            namespace.clone(),
            &[1],
            &hidden,
            &[TestCache {
                bytes: 8,
                marker: 1,
            }],
            &metrics,
        );
        cache.store(
            namespace.clone(),
            &[2],
            &hidden,
            &[TestCache {
                bytes: 8,
                marker: 2,
            }],
            &metrics,
        );

        assert!(cache.lookup(&namespace, &[1], &metrics).is_none());
        assert!(cache.lookup(&namespace, &[2], &metrics).is_some());
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot["evictions"], 1);
        assert_eq!(snapshot["resident_bytes"], 24);
    }
}
