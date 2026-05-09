use crate::{
    native_matvec::{
        NativeTextMatvecBackend, NativeTextMetalState, native_text_metal_weight_cache_bytes,
    },
    native_text::{
        NativeTextAdapter, NativeTextCandidateDecision, NativeTextDriver,
        NativeTextNextTokenContext, NativeTextPrefixCache, NativeTextPrefixCacheMetrics,
        NativeTextPrefixCacheNamespace, NativeTextPrefixCacheValue,
    },
};
use async_trait::async_trait;
use futures::stream::BoxStream;
use llm_backend::{
    BackendCacheContext, BackendError, BackendModelMetadata, BackendOutput, BackendRequest,
    BackendStreamChunk, ModelBackend, NativeMatvecBackend, QwenLayerCache, SafeTensorShardStore,
    SamplingConfig, qwen_decode_token_with_cache_with_matvec, qwen_layer_caches_for_spec,
    qwen_prefill_sequence_with_cache_with_matvec,
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

pub(crate) type NativeQwenMetalState = NativeTextMetalState;

const DEFAULT_NATIVE_QWEN_PREFIX_CACHE_BYTES: u64 = 512 * 1024 * 1024;
const NATIVE_QWEN_PREFIX_CACHE_LAYOUT_VERSION: u32 = 1;

type NativeQwenPrefixCache = NativeTextPrefixCache<QwenLayerCache>;
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

fn native_qwen_prefix_namespace(
    adapter: &NativeQwenAdapter,
    request: &BackendRequest,
    cache_tokens: usize,
) -> NativeQwenPrefixCacheNamespace {
    NativeQwenPrefixCacheNamespace {
        model_id: adapter.model_id.clone(),
        backend: adapter.metadata.backend.clone(),
        family: adapter.metadata.family.clone(),
        loader: adapter.metadata.loader.clone(),
        quantization: adapter.metadata.quantization.clone(),
        repo_id: adapter.metadata.repo_id.clone(),
        resolved_commit: adapter.metadata.resolved_commit.clone(),
        profile: adapter.metadata.profile.clone(),
        manifest_digest: adapter.metadata.manifest_digest.clone(),
        prompt_template: backend_request_cache_prompt_template(request),
        tool_schema: request.cache_context.tool_schema.clone(),
        request_mode: native_qwen_prefix_request_mode(request),
        cache_layout_version: NATIVE_QWEN_PREFIX_CACHE_LAYOUT_VERSION,
        cache_tokens,
        max_prefill_tokens: adapter.max_prefill_tokens,
    }
}

fn native_qwen_prefix_request_mode(request: &BackendRequest) -> String {
    format!(
        "conversation={},json_object={},required_tool={:?}",
        request.conversation_mode, request.json_object_mode, request.required_tool_choice
    )
}

fn backend_request_cache_prompt_template(request: &BackendRequest) -> String {
    if request.cache_context.prompt_template.is_empty() {
        BackendCacheContext::raw_prompt().prompt_template
    } else {
        request.cache_context.prompt_template.clone()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativeQwenLoadOptions {
    pub eager_materialize_shards: bool,
    pub metal_weight_cache_bytes: Option<u64>,
    pub warm_metal_weight_cache: bool,
}
impl NativeQwenBackend {
    pub fn open(
        model_id: impl Into<String>,
        snapshot_path: impl AsRef<Path>,
    ) -> anyhow::Result<Self> {
        Self::open_with_options(model_id, snapshot_path, NativeQwenLoadOptions::default())
    }

    pub fn open_with_options(
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
            let warmup = matvec.warm_bf16_matrix_cache(&store).map_err(|err| {
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
        self.driver
            .start_decode_session(context_tokens, max_new_tokens, request, cancellation)
    }

    #[cfg(test)]
    fn next_token_from_hidden(
        &self,
        hidden: &[f32],
        sampling: SamplingConfig,
    ) -> Result<NativeQwenCandidate, BackendError> {
        Ok(NativeQwenCandidate {
            token_id: self
                .driver
                .adapter
                .next_token_from_hidden(hidden, sampling)?,
        })
    }
}

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

    fn observe_candidate(
        &self,
        tokenizer: &HuggingFaceTokenizer,
        _emitted_tokens: &[u32],
        token_id: usize,
    ) -> Result<NativeTextCandidateDecision, BackendError> {
        if tokenizer
            .token_to_id("<|im_end|>")
            .is_some_and(|stop_id| token_id == stop_id as usize)
        {
            return Ok(NativeTextCandidateDecision::Stop);
        }
        Ok(NativeTextCandidateDecision::Emit(token_id))
    }

    fn cache_token_capacity(
        &self,
        context_tokens: usize,
        max_new_tokens: u32,
    ) -> Result<usize, BackendError> {
        native_qwen_cache_token_capacity(
            context_tokens,
            max_new_tokens,
            self.max_prefill_tokens,
            self.spec.max_position_embeddings,
        )
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
        native_qwen_prefix_namespace(self, request, cache_tokens)
    }

    fn layer_count(&self) -> usize {
        self.spec.num_hidden_layers as usize
    }

    fn allocate_caches(&self, cache_tokens: usize) -> Result<Vec<QwenLayerCache>, BackendError> {
        qwen_layer_caches_for_spec(&self.spec, cache_tokens)
            .map_err(|err| BackendError::Other(err.to_string()))
    }

    fn prefill_context_with_cache(
        &self,
        context_tokens: &[usize],
        caches: &mut [QwenLayerCache],
        cancellation: &CancellationToken,
    ) -> Result<Vec<f32>, BackendError> {
        native_qwen_prefill_context_with_cache(
            &self.store,
            &self.spec,
            context_tokens,
            caches,
            &self.matvec,
            self.max_prefill_tokens,
            cancellation,
        )
    }

    fn make_decode_session(
        &self,
        hidden: Vec<f32>,
        caches: Vec<QwenLayerCache>,
    ) -> NativeQwenDecodeSession {
        NativeQwenDecodeSession {
            hidden,
            caches,
            metal_state: self.matvec.metal_state(),
        }
    }

    fn hidden<'a>(&self, session: &'a NativeQwenDecodeSession) -> &'a [f32] {
        session.hidden()
    }

    fn step(
        &self,
        session: &mut NativeQwenDecodeSession,
        token_id: usize,
    ) -> Result<(), BackendError> {
        session.step(&self.store, &self.spec, &self.matvec, token_id)
    }

    fn next_token_from_hidden(
        &self,
        hidden: &[f32],
        sampling: SamplingConfig,
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
    }
}

fn native_qwen_prefill_context_with_cache(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    context_tokens: &[usize],
    caches: &mut [QwenLayerCache],
    matvec: &impl NativeMatvecBackend,
    prefill_chunk_tokens: usize,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, BackendError> {
    if cancellation.is_cancelled() {
        return Err(BackendError::Cancelled);
    }
    let mut hidden = None;
    for chunk in context_tokens.chunks(prefill_chunk_tokens.max(1)) {
        if cancellation.is_cancelled() {
            return Err(BackendError::Cancelled);
        }
        let hidden_states =
            qwen_prefill_sequence_with_cache_with_matvec(store, spec, chunk, caches, matvec)
                .map_err(|err| BackendError::Other(err.to_string()))?;
        if cancellation.is_cancelled() {
            return Err(BackendError::Cancelled);
        }
        hidden = hidden_states.last().cloned();
    }
    hidden.ok_or_else(|| BackendError::Other("Qwen prefill returned no hidden states".to_owned()))
}

pub(crate) struct NativeQwenDecodeSession {
    hidden: Vec<f32>,
    caches: Vec<QwenLayerCache>,
    metal_state: Option<Arc<NativeQwenMetalState>>,
}

impl NativeQwenDecodeSession {
    fn hidden(&self) -> &[f32] {
        &self.hidden
    }

    fn step(
        &mut self,
        store: &SafeTensorShardStore,
        spec: &QwenModelSpec,
        matvec: &impl NativeMatvecBackend,
        token_id: usize,
    ) -> Result<(), BackendError> {
        self.hidden = qwen_decode_token_with_cache_with_matvec(
            store,
            spec,
            token_id,
            &mut self.caches,
            matvec,
        )
        .map_err(|err| BackendError::Other(err.to_string()))?;
        Ok(())
    }
}

impl Drop for NativeQwenDecodeSession {
    fn drop(&mut self) {
        if let Some(state) = &self.metal_state {
            state.remove_cache_mirrors(&self.caches);
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

fn native_qwen_cache_token_capacity(
    context_tokens: usize,
    max_new_tokens: u32,
    min_cache_tokens: usize,
    max_position_embeddings: u32,
) -> Result<usize, BackendError> {
    let max_position_embeddings = usize::try_from(max_position_embeddings).map_err(|err| {
        BackendError::Other(format!(
            "native Qwen max_position_embeddings does not fit usize: {err}"
        ))
    })?;
    if max_position_embeddings == 0 {
        return Err(BackendError::UnsupportedRequest(
            "native Qwen model declares zero max_position_embeddings".to_owned(),
        ));
    }
    let max_new_tokens = usize::try_from(max_new_tokens).map_err(|err| {
        BackendError::Other(format!(
            "native Qwen max_new_tokens does not fit usize: {err}"
        ))
    })?;
    let requested_context = context_tokens.checked_add(max_new_tokens).ok_or_else(|| {
        BackendError::UnsupportedRequest(
            "native Qwen context length plus generation budget overflows usize".to_owned(),
        )
    })?;
    if requested_context > max_position_embeddings {
        return Err(BackendError::UnsupportedRequest(format!(
            "native Qwen request needs {context_tokens} prompt tokens plus {max_new_tokens} generation tokens, exceeding model context limit {max_position_embeddings}"
        )));
    }
    let required = requested_context.max(min_cache_tokens.max(1));
    Ok(required
        .checked_next_power_of_two()
        .unwrap_or(max_position_embeddings)
        .min(max_position_embeddings))
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

#[cfg(test)]
fn native_qwen_worker_stream(
    rx: tokio::sync::mpsc::Receiver<Result<BackendStreamChunk, BackendError>>,
    worker: tokio::task::JoinHandle<()>,
) -> BoxStream<'static, Result<BackendStreamChunk, BackendError>> {
    crate::native_text::native_text_worker_stream("native Qwen", rx, worker)
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
