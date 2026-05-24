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
mod tests;
