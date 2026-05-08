use async_trait::async_trait;
use axum::{
    Json, Router,
    extract::{Path as AxumPath, State, rejection::JsonRejection},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use futures::{
    Stream, StreamExt,
    stream::{self, BoxStream},
};
use llm_api::{
    ApiError, ChatCompletionRequest, ChatCompletionStreamResponse, CompletionRequest,
    CompletionStreamResponse, FinishReason, ModelCard, ModelList, Usage, ValidateRequest,
};
use llm_backend::{
    BackendError, BackendModelMetadata, BackendOutput, BackendRequest, BackendStreamChunk,
    CpuQwenMatvecBackend, DeterministicBackend, MathError, ModelBackend, QwenLayerCache,
    QwenMatvecBackend, SafeTensorShardStore, SamplingConfig, TensorLoadError, TopKLogit,
    TopKWeight, qwen_decode_token_with_cache_with_matvec, qwen_final_norm_with_matvec,
    qwen_layer_caches_for_spec, qwen_lm_head_logits_with_matvec, qwen_lm_head_top_k_with_matvec,
    qwen_prefill_sequence_with_cache_with_matvec,
};
use llm_hub::{
    DownloadPlan, HubClient, HubError, HubRepoId, ModelProfile, ModelStore, SnapshotManifest,
};
use llm_models::QwenModelSpec;
use llm_runtime::{
    ChatCompletionStreamEvent, CompletionStreamEvent, Runtime, RuntimeError,
    chat_stream_requires_buffering,
};
use llm_sampler::TopPSampler;
use llm_telemetry::{ServerMetrics, TokenCounters};
use llm_tokenizer::HuggingFaceTokenizer;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::{HashMap, HashSet},
    convert::Infallible,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_util::sync::CancellationToken;

type EngineRuntime = Runtime<Box<dyn ModelBackend>>;

#[derive(Clone)]
struct AppState {
    runtime: Arc<EngineRuntime>,
    metrics: Arc<Mutex<ServerMetrics>>,
    generation_phases: Arc<GenerationPhaseMetrics>,
    model_permits: Arc<Semaphore>,
    active_requests: Arc<Mutex<HashMap<String, CancellationToken>>>,
    next_request_id: Arc<AtomicU64>,
    admin_token: Option<Arc<str>>,
    model_home: PathBuf,
    model_store_usage: Arc<Mutex<ModelStoreUsageCache>>,
    hub_client: HubClient,
    hf_token: Option<Arc<str>>,
    stream_stall_timeout: Option<Duration>,
}

#[derive(Debug, Default)]
struct GenerationPhaseMetrics {
    prefill_requests: AtomicU64,
    decode_requests: AtomicU64,
}

impl GenerationPhaseMetrics {
    fn begin(self: &Arc<Self>, phase: GenerationPhase) -> GenerationPhaseGuard {
        self.increment(phase);
        GenerationPhaseGuard {
            metrics: Arc::clone(self),
            phase,
        }
    }

    fn prefill_requests(&self) -> u64 {
        self.prefill_requests.load(Ordering::Relaxed)
    }

    fn decode_requests(&self) -> u64 {
        self.decode_requests.load(Ordering::Relaxed)
    }

    fn increment(&self, phase: GenerationPhase) {
        self.counter(phase).fetch_add(1, Ordering::Relaxed);
    }

    fn decrement(&self, phase: GenerationPhase) {
        self.counter(phase).fetch_sub(1, Ordering::Relaxed);
    }

    fn counter(&self, phase: GenerationPhase) -> &AtomicU64 {
        match phase {
            GenerationPhase::Prefill => &self.prefill_requests,
            GenerationPhase::Decode => &self.decode_requests,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GenerationPhase {
    Prefill,
    Decode,
}

#[derive(Debug)]
struct GenerationPhaseGuard {
    metrics: Arc<GenerationPhaseMetrics>,
    phase: GenerationPhase,
}

impl GenerationPhaseGuard {
    fn transition_to_decode(&mut self) {
        if self.phase == GenerationPhase::Decode {
            return;
        }
        self.metrics.decrement(self.phase);
        self.phase = GenerationPhase::Decode;
        self.metrics.increment(self.phase);
    }
}

impl Drop for GenerationPhaseGuard {
    fn drop(&mut self) {
        self.metrics.decrement(self.phase);
    }
}

#[derive(Debug)]
struct ActiveRequest {
    id: String,
    cancellation: CancellationToken,
    active_requests: Arc<Mutex<HashMap<String, CancellationToken>>>,
}

impl Drop for ActiveRequest {
    fn drop(&mut self) {
        self.active_requests
            .lock()
            .expect("active request lock is not poisoned")
            .remove(&self.id);
    }
}

#[derive(Debug, Clone)]
pub struct EngineOptions {
    pub concurrency_limit: usize,
    pub admin_token: Option<String>,
    pub model_home: Option<PathBuf>,
    pub hub_endpoint: Option<String>,
    pub hf_token: Option<String>,
    pub stream_stall_timeout: Option<Duration>,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            concurrency_limit: 0,
            admin_token: None,
            model_home: None,
            hub_endpoint: None,
            hf_token: None,
            stream_stall_timeout: Some(DEFAULT_STREAM_STALL_TIMEOUT),
        }
    }
}

const DEFAULT_STREAM_STALL_TIMEOUT: Duration = Duration::from_secs(300);

pub fn build_router() -> Router {
    build_router_with_backend(Box::new(default_backend()))
}

pub fn build_router_with_backend(backend: Box<dyn ModelBackend>) -> Router {
    build_router_with_backend_and_concurrency(backend, 1)
}

pub fn build_router_with_backend_and_concurrency(
    backend: Box<dyn ModelBackend>,
    concurrency_limit: usize,
) -> Router {
    build_router_with_backend_and_options(
        backend,
        EngineOptions {
            concurrency_limit,
            ..EngineOptions::default()
        },
    )
    .expect("default engine options are valid")
}

pub fn build_router_with_backend_and_options(
    backend: Box<dyn ModelBackend>,
    options: EngineOptions,
) -> Result<Router, EngineConfigError> {
    let hub_client = options
        .hub_endpoint
        .as_deref()
        .map(parse_hub_client)
        .transpose()?
        .unwrap_or_default();
    let runtime = Runtime::new(backend);
    Ok(Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/admin/models", get(admin_models))
        .route("/admin/models/{alias}", get(admin_model))
        .route("/admin/models/{alias}/verify", post(admin_model_verify))
        .route("/admin/models/{alias}/plan", post(admin_model_plan))
        .route("/admin/models/{alias}/pull", post(admin_model_pull))
        .route(
            "/admin/requests/{request_id}/cancel",
            post(admin_cancel_request),
        )
        .route("/admin/metrics", get(admin_metrics))
        .route(
            "/v1/chat/completions",
            axum::routing::post(chat_completions),
        )
        .route("/v1/completions", axum::routing::post(completions))
        .with_state(AppState {
            runtime: Arc::new(runtime),
            metrics: Arc::new(Mutex::new(ServerMetrics::default())),
            generation_phases: Arc::new(GenerationPhaseMetrics::default()),
            model_permits: Arc::new(Semaphore::new(options.concurrency_limit.max(1))),
            active_requests: Arc::new(Mutex::new(HashMap::new())),
            next_request_id: Arc::new(AtomicU64::new(1)),
            admin_token: options.admin_token.map(Arc::from),
            model_home: options.model_home.unwrap_or_else(default_model_home),
            model_store_usage: Arc::new(Mutex::new(ModelStoreUsageCache::default())),
            hub_client,
            hf_token: options.hf_token.map(Arc::from),
            stream_stall_timeout: options.stream_stall_timeout,
        }))
}

#[derive(Debug)]
pub struct EngineConfigError {
    message: String,
}

impl EngineConfigError {
    fn invalid_hub_endpoint(endpoint: &str, source: url::ParseError) -> Self {
        Self {
            message: format!("invalid hub endpoint `{endpoint}`: {source}"),
        }
    }
}

impl std::fmt::Display for EngineConfigError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for EngineConfigError {}

fn parse_hub_client(endpoint: &str) -> Result<HubClient, EngineConfigError> {
    let endpoint = url::Url::parse(endpoint)
        .map_err(|err| EngineConfigError::invalid_hub_endpoint(endpoint, err))?;
    Ok(HubClient::new(endpoint))
}

fn default_backend() -> DeterministicBackend {
    DeterministicBackend::new("local-qwen36", "hello from rust native backend")
        .with_required_tool_protocol()
        .with_json_object_protocol()
        .with_conversation_protocol()
}

fn default_model_home() -> PathBuf {
    std::env::var_os("LLM_MODEL_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".llm-models"))
}

#[derive(Clone)]
pub struct NativeQwenBackend {
    model_id: String,
    metadata: BackendModelMetadata,
    tokenizer: HuggingFaceTokenizer,
    spec: QwenModelSpec,
    store: SafeTensorShardStore,
    matvec: NativeQwenMatvecBackend,
    max_new_tokens: u32,
    max_prefill_tokens: usize,
    top_k: usize,
    chunk_rows: usize,
}

#[derive(Clone)]
enum NativeQwenMatvecBackend {
    Cpu,
    Metal(Arc<NativeQwenMetalState>),
}

struct NativeQwenMetalState {
    device: llm_metal::MetalDevice,
    bf16_matrices: Mutex<HashMap<Bf16MatrixCacheKey, Arc<llm_metal::Bf16MatrixBuffer>>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Bf16MatrixCacheKey {
    tensor: String,
    element_offset: usize,
    rows: usize,
    columns: usize,
}

#[derive(Debug)]
enum NativeQwenMetalBufferError {
    Shape(String),
    Tensor(TensorLoadError),
    Metal(llm_metal::MetalError),
}

impl std::fmt::Display for NativeQwenMetalBufferError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shape(message) => formatter.write_str(message),
            Self::Tensor(err) => write!(formatter, "{err}"),
            Self::Metal(err) => write!(formatter, "{err}"),
        }
    }
}

impl NativeQwenMetalState {
    fn new(device: llm_metal::MetalDevice) -> Self {
        Self {
            device,
            bf16_matrices: Mutex::new(HashMap::new()),
        }
    }

    fn bf16_matrix_buffer(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        element_offset: usize,
        rows: usize,
        columns: usize,
    ) -> Result<Arc<llm_metal::Bf16MatrixBuffer>, NativeQwenMetalBufferError> {
        let key = Bf16MatrixCacheKey {
            tensor: tensor.to_owned(),
            element_offset,
            rows,
            columns,
        };
        if let Some(buffer) = self
            .bf16_matrices
            .lock()
            .expect("BF16 matrix buffer cache lock is not poisoned")
            .get(&key)
            .cloned()
        {
            native_qwen_metal_metrics().record_bf16_matrix_cache_hit();
            return Ok(buffer);
        }
        native_qwen_metal_metrics().record_bf16_matrix_cache_miss();
        let element_count = rows.checked_mul(columns).ok_or_else(|| {
            NativeQwenMetalBufferError::Shape("BF16 matrix element count overflow".to_owned())
        })?;
        let weights = store
            .bf16_tensor_bits_range(tensor, element_offset, element_count)
            .map_err(NativeQwenMetalBufferError::Tensor)?;
        let buffer = Arc::new(
            self.device
                .new_bf16_matrix_buffer(&weights, rows, columns)
                .map_err(NativeQwenMetalBufferError::Metal)?,
        );
        let mut matrices = self
            .bf16_matrices
            .lock()
            .expect("BF16 matrix buffer cache lock is not poisoned");
        if let Some(existing) = matrices.get(&key) {
            native_qwen_metal_metrics().record_bf16_matrix_cache_hit();
            return Ok(Arc::clone(existing));
        }
        native_qwen_metal_metrics().record_bf16_matrix_cache_upload(buffer.byte_len() as u64);
        matrices.insert(key, Arc::clone(&buffer));
        Ok(buffer)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct MetalKernelCounters {
    attempts: u64,
    successes: u64,
    fallbacks: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct MetalBf16MatrixCacheCounters {
    hits: u64,
    misses: u64,
    uploads: u64,
    bytes_uploaded: u64,
}

#[derive(Debug, Default)]
struct MetalBackendMetrics {
    counters: Mutex<HashMap<&'static str, MetalKernelCounters>>,
    bf16_matrix_cache: Mutex<MetalBf16MatrixCacheCounters>,
    warned_fallbacks: Mutex<HashSet<String>>,
}

impl MetalBackendMetrics {
    fn record_attempt(&self, kernel: &'static str) {
        self.update_counter(kernel, |counters| counters.attempts += 1);
    }

    fn record_success(&self, kernel: &'static str) {
        self.update_counter(kernel, |counters| counters.successes += 1);
    }

    fn record_fallback(
        &self,
        kernel: &'static str,
        bucket: impl Into<String>,
        error: impl std::fmt::Display,
    ) {
        self.update_counter(kernel, |counters| counters.fallbacks += 1);
        let bucket = bucket.into();
        let error = error.to_string();
        let warning_key = format!("{kernel}:{bucket}");
        let should_warn = self
            .warned_fallbacks
            .lock()
            .expect("Metal fallback warning lock is not poisoned")
            .insert(warning_key);
        if should_warn {
            tracing::warn!(
                target: "native_qwen_metal",
                kernel,
                shape_bucket = %bucket,
                error = %error,
                "native Qwen Metal kernel fell back to CPU"
            );
        } else {
            tracing::debug!(
                target: "native_qwen_metal",
                kernel,
                shape_bucket = %bucket,
                error = %error,
                "native Qwen Metal kernel fell back to CPU"
            );
        }
    }

    fn record_bf16_matrix_cache_hit(&self) {
        let mut cache = self
            .bf16_matrix_cache
            .lock()
            .expect("Metal BF16 matrix cache metrics lock is not poisoned");
        cache.hits += 1;
    }

    fn record_bf16_matrix_cache_miss(&self) {
        let mut cache = self
            .bf16_matrix_cache
            .lock()
            .expect("Metal BF16 matrix cache metrics lock is not poisoned");
        cache.misses += 1;
    }

    fn record_bf16_matrix_cache_upload(&self, byte_len: u64) {
        let mut cache = self
            .bf16_matrix_cache
            .lock()
            .expect("Metal BF16 matrix cache metrics lock is not poisoned");
        cache.uploads += 1;
        cache.bytes_uploaded += byte_len;
    }

    fn snapshot(&self) -> Value {
        let counters = self
            .counters
            .lock()
            .expect("Metal metrics lock is not poisoned");
        let bf16_matrix_cache = *self
            .bf16_matrix_cache
            .lock()
            .expect("Metal BF16 matrix cache metrics lock is not poisoned");
        let mut kernels = serde_json::Map::new();
        let mut kernel_names = counters.keys().copied().collect::<Vec<_>>();
        kernel_names.sort_unstable();
        for kernel in kernel_names {
            let counters = counters.get(kernel).copied().unwrap_or_default();
            kernels.insert(
                kernel.to_owned(),
                json!({
                    "attempts": counters.attempts,
                    "successes": counters.successes,
                    "fallbacks": counters.fallbacks,
                }),
            );
        }
        json!({
            "kernels": kernels,
            "bf16_matrix_cache": {
                "hits": bf16_matrix_cache.hits,
                "misses": bf16_matrix_cache.misses,
                "uploads": bf16_matrix_cache.uploads,
                "bytes_uploaded": bf16_matrix_cache.bytes_uploaded,
            }
        })
    }

    fn update_counter(&self, kernel: &'static str, update: impl FnOnce(&mut MetalKernelCounters)) {
        let mut counters = self
            .counters
            .lock()
            .expect("Metal metrics lock is not poisoned");
        update(counters.entry(kernel).or_default());
    }
}

fn native_qwen_metal_metrics() -> &'static MetalBackendMetrics {
    static METRICS: OnceLock<MetalBackendMetrics> = OnceLock::new();
    METRICS.get_or_init(MetalBackendMetrics::default)
}

impl NativeQwenMatvecBackend {
    fn system_default() -> Self {
        match llm_metal::MetalDevice::system_default_result() {
            Ok(Some(device)) => Self::Metal(Arc::new(NativeQwenMetalState::new(device))),
            Ok(None) => Self::Cpu,
            Err(err) => {
                tracing::warn!("Metal Qwen matvec backend unavailable: {err}");
                Self::Cpu
            }
        }
    }

    fn cpu() -> CpuQwenMatvecBackend {
        CpuQwenMatvecBackend
    }

    fn bf16_matrix_shape(
        store: &SafeTensorShardStore,
        tensor: &str,
        input: &[f32],
    ) -> Option<(usize, usize)> {
        let metadata = store.tensor_metadata(tensor).ok()?;
        if metadata.dtype != "BF16" || metadata.shape.len() != 2 {
            return None;
        }
        let rows = metadata.shape[0];
        let columns = metadata.shape[1];
        (input.len() == columns).then_some((rows, columns))
    }

    fn flattened_inputs(inputs: &[Vec<f32>], columns: usize) -> Option<Vec<f32>> {
        let mut flattened = Vec::with_capacity(inputs.len().checked_mul(columns)?);
        for input in inputs {
            if input.len() != columns {
                return None;
            }
            flattened.extend_from_slice(input);
        }
        Some(flattened)
    }

    fn record_metal_fallback(
        kernel: &'static str,
        bucket: impl Into<String>,
        error: impl std::fmt::Display,
    ) {
        native_qwen_metal_metrics().record_fallback(kernel, bucket, error);
    }

    fn run_metal_math<T>(
        kernel: &'static str,
        bucket: impl Into<String>,
        metal: impl FnOnce() -> Result<T, llm_metal::MetalError>,
        cpu: impl FnOnce() -> Result<T, MathError>,
    ) -> Result<T, MathError> {
        let metrics = native_qwen_metal_metrics();
        metrics.record_attempt(kernel);
        match metal() {
            Ok(value) => {
                metrics.record_success(kernel);
                Ok(value)
            }
            Err(err) => {
                metrics.record_fallback(kernel, bucket, err);
                cpu()
            }
        }
    }

    fn run_metal_tensor<T>(
        kernel: &'static str,
        bucket: impl Into<String>,
        metal: impl FnOnce() -> Result<T, llm_metal::MetalError>,
        cpu: impl FnOnce() -> Result<T, TensorLoadError>,
    ) -> Result<T, TensorLoadError> {
        let metrics = native_qwen_metal_metrics();
        metrics.record_attempt(kernel);
        match metal() {
            Ok(value) => {
                metrics.record_success(kernel);
                Ok(value)
            }
            Err(err) => {
                metrics.record_fallback(kernel, bucket, err);
                cpu()
            }
        }
    }
}

impl QwenMatvecBackend for NativeQwenMatvecBackend {
    fn bf16_matvec_row_major_f32(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        input: &[f32],
    ) -> Result<Vec<f32>, TensorLoadError> {
        let Self::Metal(state) = self else {
            return Self::cpu().bf16_matvec_row_major_f32(store, tensor, input);
        };
        let Some((rows, columns)) = Self::bf16_matrix_shape(store, tensor, input) else {
            Self::record_metal_fallback(
                "matvec_bf16_f32",
                format!("tensor={tensor},input_len={}", input.len()),
                "unsupported BF16 matrix shape or input length",
            );
            return Self::cpu().bf16_matvec_row_major_f32(store, tensor, input);
        };
        let matrix = match state.bf16_matrix_buffer(store, tensor, 0, rows, columns) {
            Ok(matrix) => matrix,
            Err(err) => {
                Self::record_metal_fallback(
                    "matvec_bf16_f32",
                    format!("tensor={tensor},rows={rows},cols={columns}"),
                    err,
                );
                return Self::cpu().bf16_matvec_row_major_f32(store, tensor, input);
            }
        };
        Self::run_metal_tensor(
            "matvec_bf16_f32",
            format!("tensor={tensor},rows={rows},cols={columns}"),
            || state.device.matvec_bf16_f32_buffered(&matrix, input),
            || Self::cpu().bf16_matvec_row_major_f32(store, tensor, input),
        )
    }

    fn bf16_matvecs_row_major_f32(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        inputs: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, TensorLoadError> {
        let Self::Metal(state) = self else {
            return Self::cpu().bf16_matvecs_row_major_f32(store, tensor, inputs);
        };
        let Some(first_input) = inputs.first() else {
            return Ok(Vec::new());
        };
        let Some((rows, columns)) = Self::bf16_matrix_shape(store, tensor, first_input) else {
            Self::record_metal_fallback(
                "batched_matvec_bf16_f32",
                format!(
                    "tensor={tensor},inputs={},first_input_len={}",
                    inputs.len(),
                    first_input.len()
                ),
                "unsupported BF16 matrix shape or input length",
            );
            return Self::cpu().bf16_matvecs_row_major_f32(store, tensor, inputs);
        };
        let Some(flattened) = Self::flattened_inputs(inputs, columns) else {
            Self::record_metal_fallback(
                "batched_matvec_bf16_f32",
                format!("tensor={tensor},inputs={},cols={columns}", inputs.len()),
                "batched input width mismatch",
            );
            return Self::cpu().bf16_matvecs_row_major_f32(store, tensor, inputs);
        };
        let matrix = match state.bf16_matrix_buffer(store, tensor, 0, rows, columns) {
            Ok(matrix) => matrix,
            Err(err) => {
                Self::record_metal_fallback(
                    "batched_matvec_bf16_f32",
                    format!("tensor={tensor},rows={rows},cols={columns}"),
                    err,
                );
                return Self::cpu().bf16_matvecs_row_major_f32(store, tensor, inputs);
            }
        };
        Self::run_metal_tensor(
            "batched_matvec_bf16_f32",
            format!(
                "tensor={tensor},rows={rows},cols={columns},inputs={}",
                inputs.len()
            ),
            || {
                state
                    .device
                    .batched_matvec_bf16_f32_buffered(&matrix, &flattened, inputs.len())
                    .map(|values| {
                        values
                            .chunks_exact(rows)
                            .map(|chunk| chunk.to_vec())
                            .collect()
                    })
            },
            || Self::cpu().bf16_matvecs_row_major_f32(store, tensor, inputs),
        )
    }

    fn bf16_matvec_rows_f32(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        input: &[f32],
        chunk_rows: usize,
    ) -> Result<Vec<f32>, TensorLoadError> {
        let Self::Metal(state) = self else {
            return Self::cpu().bf16_matvec_rows_f32(store, tensor, input, chunk_rows);
        };
        if chunk_rows == 0 {
            Self::record_metal_fallback(
                "matvec_bf16_f32",
                format!("tensor={tensor},input_len={},chunk_rows=0", input.len()),
                "zero chunk rows",
            );
            return Self::cpu().bf16_matvec_rows_f32(store, tensor, input, chunk_rows);
        }
        let Some((rows, columns)) = Self::bf16_matrix_shape(store, tensor, input) else {
            Self::record_metal_fallback(
                "matvec_bf16_f32",
                format!(
                    "tensor={tensor},input_len={},chunk_rows={chunk_rows}",
                    input.len()
                ),
                "unsupported BF16 matrix shape or input length",
            );
            return Self::cpu().bf16_matvec_rows_f32(store, tensor, input, chunk_rows);
        };
        let mut output = Vec::with_capacity(rows);
        for row_start in (0..rows).step_by(chunk_rows) {
            let rows_in_chunk = chunk_rows.min(rows - row_start);
            let Some(element_offset) = row_start.checked_mul(columns) else {
                Self::record_metal_fallback(
                    "matvec_bf16_f32",
                    format!("tensor={tensor},row_start={row_start},rows={rows},cols={columns}"),
                    "BF16 row offset overflow",
                );
                return Self::cpu().bf16_matvec_rows_f32(store, tensor, input, chunk_rows);
            };
            let matrix = match state.bf16_matrix_buffer(
                store,
                tensor,
                element_offset,
                rows_in_chunk,
                columns,
            ) {
                Ok(matrix) => matrix,
                Err(err) => {
                    Self::record_metal_fallback(
                        "matvec_bf16_f32",
                        format!(
                            "tensor={tensor},row_start={row_start},rows_in_chunk={rows_in_chunk},cols={columns}"
                        ),
                        err,
                    );
                    return Self::cpu().bf16_matvec_rows_f32(store, tensor, input, chunk_rows);
                }
            };
            let metrics = native_qwen_metal_metrics();
            metrics.record_attempt("matvec_bf16_f32");
            let logits = match state.device.matvec_bf16_f32_buffered(&matrix, input) {
                Ok(logits) => {
                    metrics.record_success("matvec_bf16_f32");
                    logits
                }
                Err(err) => {
                    metrics.record_fallback(
                        "matvec_bf16_f32",
                        format!(
                            "tensor={tensor},row_start={row_start},rows_in_chunk={rows_in_chunk},cols={columns}"
                        ),
                        err,
                    );
                    return Self::cpu().bf16_matvec_rows_f32(store, tensor, input, chunk_rows);
                }
            };
            output.extend(logits);
        }
        Ok(output)
    }

    fn bf16_matvec_top_k_rows_f32(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        input: &[f32],
        top_k: usize,
        chunk_rows: usize,
    ) -> Result<Vec<TopKLogit>, TensorLoadError> {
        let Self::Metal(state) = self else {
            return Self::cpu().bf16_matvec_top_k_rows_f32(store, tensor, input, top_k, chunk_rows);
        };
        if chunk_rows == 0 {
            Self::record_metal_fallback(
                "matvec_bf16_f32",
                format!(
                    "tensor={tensor},input_len={},top_k={top_k},chunk_rows=0",
                    input.len()
                ),
                "zero chunk rows",
            );
            return Self::cpu().bf16_matvec_top_k_rows_f32(store, tensor, input, top_k, chunk_rows);
        }
        let Some((rows, columns)) = Self::bf16_matrix_shape(store, tensor, input) else {
            Self::record_metal_fallback(
                "matvec_bf16_f32",
                format!(
                    "tensor={tensor},input_len={},top_k={top_k},chunk_rows={chunk_rows}",
                    input.len()
                ),
                "unsupported BF16 matrix shape or input length",
            );
            return Self::cpu().bf16_matvec_top_k_rows_f32(store, tensor, input, top_k, chunk_rows);
        };
        if top_k == 0 || top_k > rows {
            Self::record_metal_fallback(
                "top_k_f32",
                format!("tensor={tensor},rows={rows},top_k={top_k}"),
                "unsupported top-k request",
            );
            return Self::cpu().bf16_matvec_top_k_rows_f32(store, tensor, input, top_k, chunk_rows);
        }
        let mut top = Vec::new();
        for row_start in (0..rows).step_by(chunk_rows) {
            let rows_in_chunk = chunk_rows.min(rows - row_start);
            let Some(element_offset) = row_start.checked_mul(columns) else {
                Self::record_metal_fallback(
                    "matvec_bf16_f32",
                    format!("tensor={tensor},row_start={row_start},rows={rows},cols={columns}"),
                    "BF16 row offset overflow",
                );
                return Self::cpu()
                    .bf16_matvec_top_k_rows_f32(store, tensor, input, top_k, chunk_rows);
            };
            let matrix = match state.bf16_matrix_buffer(
                store,
                tensor,
                element_offset,
                rows_in_chunk,
                columns,
            ) {
                Ok(matrix) => matrix,
                Err(err) => {
                    Self::record_metal_fallback(
                        "matvec_bf16_f32",
                        format!(
                            "tensor={tensor},row_start={row_start},rows_in_chunk={rows_in_chunk},cols={columns}"
                        ),
                        err,
                    );
                    return Self::cpu()
                        .bf16_matvec_top_k_rows_f32(store, tensor, input, top_k, chunk_rows);
                }
            };
            let metrics = native_qwen_metal_metrics();
            metrics.record_attempt("matvec_bf16_f32");
            let logits = match state.device.matvec_bf16_f32_buffered(&matrix, input) {
                Ok(logits) => {
                    metrics.record_success("matvec_bf16_f32");
                    logits
                }
                Err(err) => {
                    metrics.record_fallback(
                        "matvec_bf16_f32",
                        format!(
                            "tensor={tensor},row_start={row_start},rows_in_chunk={rows_in_chunk},cols={columns}"
                        ),
                        err,
                    );
                    return Self::cpu()
                        .bf16_matvec_top_k_rows_f32(store, tensor, input, top_k, chunk_rows);
                }
            };
            metrics.record_attempt("top_k_f32");
            let chunk_top = match state.device.top_k_f32(&logits, top_k.min(rows_in_chunk)) {
                Ok(chunk_top) => {
                    metrics.record_success("top_k_f32");
                    chunk_top
                }
                Err(err) => {
                    metrics.record_fallback(
                        "top_k_f32",
                        format!(
                            "tensor={tensor},row_start={row_start},rows_in_chunk={rows_in_chunk},top_k={top_k}"
                        ),
                        err,
                    );
                    return Self::cpu()
                        .bf16_matvec_top_k_rows_f32(store, tensor, input, top_k, chunk_rows);
                }
            };
            top.extend(chunk_top.into_iter().map(|item| TopKLogit {
                index: row_start + item.index,
                logit: item.value,
            }));
        }
        top.sort_by(|left, right| {
            right
                .logit
                .total_cmp(&left.logit)
                .then_with(|| left.index.cmp(&right.index))
        });
        top.truncate(top_k);
        Ok(top)
    }

    fn matvec_row_major_f32(
        &self,
        input: &[f32],
        weights: &[f32],
        rows: usize,
        columns: usize,
    ) -> Result<Vec<f32>, MathError> {
        match self {
            Self::Cpu => Self::cpu().matvec_row_major_f32(input, weights, rows, columns),
            Self::Metal(metal) => Self::run_metal_math(
                "matvec_f32",
                format!("rows={rows},cols={columns},input_len={}", input.len()),
                || metal.device.matvec_f32(weights, rows, columns, input),
                || Self::cpu().matvec_row_major_f32(input, weights, rows, columns),
            ),
        }
    }

    fn qwen_rms_norm_f32(
        &self,
        input: &[f32],
        weight: &[f32],
        eps: f32,
    ) -> Result<Vec<f32>, MathError> {
        match self {
            Self::Cpu => Self::cpu().qwen_rms_norm_f32(input, weight, eps),
            Self::Metal(metal) => Self::run_metal_math(
                "qwen_rms_norm",
                format!("len={},weight_len={}", input.len(), weight.len()),
                || metal.device.qwen_rms_norm_f32(input, weight, eps),
                || Self::cpu().qwen_rms_norm_f32(input, weight, eps),
            ),
        }
    }

    fn softmax_f32(&self, scores: &[f32]) -> Result<Vec<f32>, MathError> {
        match self {
            Self::Cpu => Self::cpu().softmax_f32(scores),
            Self::Metal(metal) => Self::run_metal_math(
                "softmax_f32",
                format!("len={}", scores.len()),
                || metal.device.softmax_f32(scores),
                || Self::cpu().softmax_f32(scores),
            ),
        }
    }

    fn linear_attention_conv1d_silu_f32(
        &self,
        window: &[f32],
        weights: &[f32],
        conv_dim: usize,
        kernel_size: usize,
    ) -> Result<Vec<f32>, MathError> {
        match self {
            Self::Cpu => {
                Self::cpu().linear_attention_conv1d_silu_f32(window, weights, conv_dim, kernel_size)
            }
            Self::Metal(metal) => Self::run_metal_math(
                "linear_attention_conv1d_silu_f32",
                format!(
                    "window_len={},weight_len={},conv_dim={conv_dim},kernel_size={kernel_size}",
                    window.len(),
                    weights.len()
                ),
                || {
                    metal.device.linear_attention_conv1d_silu_f32(
                        window,
                        weights,
                        conv_dim,
                        kernel_size,
                    )
                },
                || {
                    Self::cpu().linear_attention_conv1d_silu_f32(
                        window,
                        weights,
                        conv_dim,
                        kernel_size,
                    )
                },
            ),
        }
    }

    fn softmax_top_k_f32(
        &self,
        logits: &[f32],
        top_k: usize,
    ) -> Result<Vec<TopKWeight>, MathError> {
        match self {
            Self::Cpu => Self::cpu().softmax_top_k_f32(logits, top_k),
            Self::Metal(metal) => {
                if top_k == 0
                    || top_k > logits.len()
                    || logits.iter().any(|value| !value.is_finite())
                {
                    Self::record_metal_fallback(
                        "top_k_f32",
                        format!("logits_len={},top_k={top_k}", logits.len()),
                        "unsupported top-k softmax request",
                    );
                    return Self::cpu().softmax_top_k_f32(logits, top_k);
                }
                let metrics = native_qwen_metal_metrics();
                metrics.record_attempt("top_k_f32");
                let top = match metal.device.top_k_f32(logits, top_k) {
                    Ok(top) => top,
                    Err(err) => {
                        metrics.record_fallback(
                            "top_k_f32",
                            format!("logits_len={},top_k={top_k}", logits.len()),
                            err,
                        );
                        return Self::cpu().softmax_top_k_f32(logits, top_k);
                    }
                };
                match softmax_metal_top_k(top) {
                    Ok(weights) => {
                        metrics.record_success("top_k_f32");
                        Ok(weights)
                    }
                    Err(()) => {
                        metrics.record_fallback(
                            "top_k_f32",
                            format!("logits_len={},top_k={top_k}", logits.len()),
                            "Metal top-k softmax normalization failed",
                        );
                        Self::cpu().softmax_top_k_f32(logits, top_k)
                    }
                }
            }
        }
    }

    fn weighted_sum_f32(
        &self,
        values: &[f32],
        weights: &[f32],
        vector_len: usize,
    ) -> Result<Vec<f32>, MathError> {
        match self {
            Self::Cpu => Self::cpu().weighted_sum_f32(values, weights, vector_len),
            Self::Metal(metal) => Self::run_metal_math(
                "weighted_sum_f32",
                format!(
                    "values_len={},weights_len={},vector_len={vector_len}",
                    values.len(),
                    weights.len()
                ),
                || metal.device.weighted_sum_f32(values, weights, vector_len),
                || Self::cpu().weighted_sum_f32(values, weights, vector_len),
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_attention_recurrent_update_f32(
        &self,
        state: &[f32],
        key: &[f32],
        value: &[f32],
        memory: &[f32],
        beta: f32,
        decay: f32,
        key_head_dim: usize,
        value_head_dim: usize,
    ) -> Result<Vec<f32>, MathError> {
        match self {
            Self::Cpu => Self::cpu().linear_attention_recurrent_update_f32(
                state,
                key,
                value,
                memory,
                beta,
                decay,
                key_head_dim,
                value_head_dim,
            ),
            Self::Metal(metal) => Self::run_metal_math(
                "linear_attention_recurrent_update_f32",
                format!(
                    "state_len={},key_len={},value_len={},memory_len={},key_head_dim={key_head_dim},value_head_dim={value_head_dim}",
                    state.len(),
                    key.len(),
                    value.len(),
                    memory.len()
                ),
                || {
                    metal.device.linear_attention_recurrent_update_f32(
                        state,
                        key,
                        value,
                        memory,
                        beta,
                        decay,
                        key_head_dim,
                        value_head_dim,
                    )
                },
                || {
                    Self::cpu().linear_attention_recurrent_update_f32(
                        state,
                        key,
                        value,
                        memory,
                        beta,
                        decay,
                        key_head_dim,
                        value_head_dim,
                    )
                },
            ),
        }
    }
    fn select_head_rows_f32(
        &self,
        values: &[f32],
        row_count: usize,
        row_len: usize,
        head_start: usize,
        head_len: usize,
    ) -> Result<Vec<f32>, MathError> {
        match self {
            Self::Cpu => {
                Self::cpu().select_head_rows_f32(values, row_count, row_len, head_start, head_len)
            }
            Self::Metal(metal) => Self::run_metal_math(
                "select_head_rows_f32",
                format!(
                    "values_len={},row_count={row_count},row_len={row_len},head_start={head_start},head_len={head_len}",
                    values.len()
                ),
                || {
                    metal
                        .device
                        .select_head_rows_f32(values, row_count, row_len, head_start, head_len)
                },
                || {
                    Self::cpu()
                        .select_head_rows_f32(values, row_count, row_len, head_start, head_len)
                },
            ),
        }
    }
}

fn softmax_metal_top_k(top: Vec<llm_metal::TopKResult>) -> Result<Vec<TopKWeight>, ()> {
    let max = top
        .iter()
        .map(|item| item.value)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut exp_values = top
        .iter()
        .map(|item| (item.value - max).exp())
        .collect::<Vec<_>>();
    let sum = exp_values.iter().sum::<f32>();
    if sum == 0.0 || !sum.is_finite() {
        return Err(());
    }
    Ok(top
        .iter()
        .zip(exp_values.iter_mut())
        .map(|(item, value)| TopKWeight {
            index: item.index,
            weight: *value / sum,
        })
        .collect())
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativeQwenLoadOptions {
    pub eager_materialize_shards: bool,
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
        let config_json = std::fs::read_to_string(snapshot_path.join("config.json"))?;
        let metadata = native_qwen_metadata(&model_id, snapshot_path)?;
        let store = SafeTensorShardStore::open(snapshot_path)?;
        if options.eager_materialize_shards {
            let materialized_bytes = store.materialize_all_shards()?;
            tracing::info!(
                materialized_bytes,
                "materialized native Qwen safetensors shards"
            );
        }
        Ok(Self {
            model_id,
            metadata,
            tokenizer: HuggingFaceTokenizer::from_file(snapshot_path.join("tokenizer.json"))?,
            spec: QwenModelSpec::from_config_json(&config_json)?,
            store,
            matvec: NativeQwenMatvecBackend::system_default(),
            max_new_tokens: 1,
            max_prefill_tokens: 32,
            top_k: 16,
            chunk_rows: 2048,
        })
    }

    pub fn with_max_new_tokens(mut self, max_new_tokens: u32) -> Self {
        self.max_new_tokens = max_new_tokens.max(1);
        self
    }

    pub fn with_max_prefill_tokens(mut self, max_prefill_tokens: usize) -> Self {
        self.max_prefill_tokens = max_prefill_tokens.max(1);
        self
    }

    fn generate_blocking(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        if cancellation.is_cancelled() {
            return Err(BackendError::Cancelled);
        }
        if request.model != self.model_id {
            return Err(BackendError::ModelNotFound {
                requested: request.model,
                available: self.model_id.clone(),
            });
        }
        if cancellation.is_cancelled() {
            return Err(BackendError::Cancelled);
        }
        let prompt_tokens = self
            .tokenizer
            .encode(&request.prompt, false)
            .map_err(|err| BackendError::Other(err.to_string()))?;
        let context_tokens = prompt_tokens
            .iter()
            .map(|token| *token as usize)
            .collect::<Vec<_>>();
        if context_tokens.is_empty() {
            return Err(BackendError::Other(
                "Qwen prompt encoded to zero tokens".to_owned(),
            ));
        }
        let mut output_ids = Vec::new();
        let mut finish_reason = FinishReason::Length;
        let eos_id = self
            .tokenizer
            .token_to_id("<|im_end|>")
            .map(|id| id as usize);
        let requested = resolve_native_max_tokens(request.max_tokens, self.max_new_tokens)?;
        let mut decode = self.start_decode_session(&context_tokens, requested, &cancellation)?;
        if cancellation.is_cancelled() {
            return Err(BackendError::Cancelled);
        }

        for step_idx in 0..requested {
            if cancellation.is_cancelled() {
                return Err(BackendError::Cancelled);
            }
            let candidate = self.next_token_from_hidden(decode.hidden(), request.sampling)?;
            if cancellation.is_cancelled() {
                return Err(BackendError::Cancelled);
            }
            if Some(candidate.token_id) == eos_id {
                finish_reason = FinishReason::Stop;
                break;
            }
            output_ids.push(u32::try_from(candidate.token_id).map_err(|err| {
                BackendError::Other(format!("Qwen token id does not fit u32: {err}"))
            })?);
            if step_idx + 1 < requested {
                decode.step(&self.store, &self.spec, &self.matvec, candidate.token_id)?;
            }
        }

        let text = self
            .tokenizer
            .decode(&output_ids, false)
            .map_err(|err| BackendError::Other(err.to_string()))?;
        Ok(BackendOutput {
            text,
            prompt_tokens: prompt_tokens.len() as u64,
            completion_tokens: output_ids.len() as u64,
            finish_reason,
        })
    }

    fn generate_blocking_stream(
        &self,
        request: BackendRequest,
        tx: tokio::sync::mpsc::Sender<Result<BackendStreamChunk, BackendError>>,
        cancellation: CancellationToken,
    ) -> Result<(), BackendError> {
        if cancellation.is_cancelled() {
            return Ok(());
        }
        if request.model != self.model_id {
            return Err(BackendError::ModelNotFound {
                requested: request.model,
                available: self.model_id.clone(),
            });
        }
        if cancellation.is_cancelled() {
            return Ok(());
        }
        let prompt_tokens = self
            .tokenizer
            .encode(&request.prompt, false)
            .map_err(|err| BackendError::Other(err.to_string()))?;
        let context_tokens = prompt_tokens
            .iter()
            .map(|token| *token as usize)
            .collect::<Vec<_>>();
        if context_tokens.is_empty() {
            return Err(BackendError::Other(
                "Qwen prompt encoded to zero tokens".to_owned(),
            ));
        }
        let mut output_ids = Vec::new();
        let mut decoded = String::new();
        let mut finish_reason = FinishReason::Length;
        let eos_id = self
            .tokenizer
            .token_to_id("<|im_end|>")
            .map(|id| id as usize);
        let requested = resolve_native_max_tokens(request.max_tokens, self.max_new_tokens)?;
        let mut decode = match self.start_decode_session(&context_tokens, requested, &cancellation)
        {
            Ok(decode) => decode,
            Err(BackendError::Cancelled) if cancellation.is_cancelled() => return Ok(()),
            Err(err) => return Err(err),
        };
        if cancellation.is_cancelled() {
            return Ok(());
        }

        for step_idx in 0..requested {
            if cancellation.is_cancelled() {
                return Ok(());
            }
            let candidate = self.next_token_from_hidden(decode.hidden(), request.sampling)?;
            if cancellation.is_cancelled() {
                return Ok(());
            }
            if Some(candidate.token_id) == eos_id {
                finish_reason = FinishReason::Stop;
                break;
            }
            output_ids.push(u32::try_from(candidate.token_id).map_err(|err| {
                BackendError::Other(format!("Qwen token id does not fit u32: {err}"))
            })?);
            let next_decoded = self
                .tokenizer
                .decode(&output_ids, false)
                .map_err(|err| BackendError::Other(err.to_string()))?;
            let delta = next_decoded
                .strip_prefix(&decoded)
                .unwrap_or(&next_decoded)
                .to_owned();
            decoded = next_decoded;
            if cancellation.is_cancelled() {
                return Ok(());
            }
            send_backend_stream_chunk(
                &tx,
                BackendStreamChunk {
                    text: delta,
                    prompt_tokens: prompt_tokens.len() as u64,
                    completion_tokens: 1,
                    finish_reason: None,
                },
            )?;
            if step_idx + 1 < requested {
                if cancellation.is_cancelled() {
                    return Ok(());
                }
                decode.step(&self.store, &self.spec, &self.matvec, candidate.token_id)?;
            }
        }

        if cancellation.is_cancelled() {
            return Ok(());
        }
        send_backend_stream_chunk(
            &tx,
            BackendStreamChunk {
                text: String::new(),
                prompt_tokens: prompt_tokens.len() as u64,
                completion_tokens: 0,
                finish_reason: Some(finish_reason),
            },
        )
    }

    fn start_decode_session(
        &self,
        context_tokens: &[usize],
        max_new_tokens: u32,
        cancellation: &CancellationToken,
    ) -> Result<NativeQwenDecodeSession, BackendError> {
        if cancellation.is_cancelled() {
            return Err(BackendError::Cancelled);
        }
        let cache_tokens =
            native_qwen_cache_token_capacity(self.max_prefill_tokens, max_new_tokens);
        let mut caches = qwen_layer_caches_for_spec(&self.spec, cache_tokens)
            .map_err(|err| BackendError::Other(err.to_string()))?;
        if cancellation.is_cancelled() {
            return Err(BackendError::Cancelled);
        }
        let hidden = native_qwen_prefill_context_with_cache(
            &self.store,
            &self.spec,
            context_tokens,
            &mut caches,
            &self.matvec,
            self.max_prefill_tokens,
            cancellation,
        )?;
        Ok(NativeQwenDecodeSession { hidden, caches })
    }

    fn next_token_from_hidden(
        &self,
        hidden: &[f32],
        sampling: SamplingConfig,
    ) -> Result<NativeQwenCandidate, BackendError> {
        let final_norm = qwen_final_norm_with_matvec(
            &self.store,
            hidden,
            self.spec.hidden_size as usize,
            self.spec.rms_norm_eps,
            &self.matvec,
        )
        .map_err(|err| BackendError::Other(err.to_string()))?;
        if !sampling.is_greedy() {
            let logits = qwen_lm_head_logits_with_matvec(
                &self.store,
                &final_norm,
                self.chunk_rows,
                &self.matvec,
            )
            .map_err(|err| BackendError::Other(err.to_string()))?;
            let sampled_token_id =
                sample_token_id_with_draw(&logits, sampling, native_sampling_draw())?;
            u32::try_from(sampled_token_id).map_err(|err| {
                BackendError::Other(format!("Qwen token id does not fit u32: {err}"))
            })?;
            return Ok(NativeQwenCandidate {
                token_id: sampled_token_id,
            });
        }

        let top_logits = qwen_lm_head_top_k_with_matvec(
            &self.store,
            &final_norm,
            self.top_k,
            self.chunk_rows,
            &self.matvec,
        )
        .map_err(|err| BackendError::Other(err.to_string()))?;

        let item = top_logits
            .into_iter()
            .next()
            .ok_or_else(|| BackendError::Other("Qwen lm head returned no logits".to_owned()))?;
        u32::try_from(item.index)
            .map_err(|err| BackendError::Other(format!("Qwen token id does not fit u32: {err}")))?;
        Ok(NativeQwenCandidate {
            token_id: item.index,
        })
    }
}

fn native_qwen_prefill_context_with_cache(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    context_tokens: &[usize],
    caches: &mut [QwenLayerCache],
    matvec: &impl QwenMatvecBackend,
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

struct NativeQwenDecodeSession {
    hidden: Vec<f32>,
    caches: Vec<QwenLayerCache>,
}

impl NativeQwenDecodeSession {
    fn hidden(&self) -> &[f32] {
        &self.hidden
    }

    fn step(
        &mut self,
        store: &SafeTensorShardStore,
        spec: &QwenModelSpec,
        matvec: &impl QwenMatvecBackend,
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

fn resolve_native_max_tokens(
    requested: Option<u32>,
    configured_max: u32,
) -> Result<u32, BackendError> {
    let configured_max = configured_max.max(1);
    match requested {
        None => Ok(configured_max),
        Some(0) => Err(BackendError::UnsupportedRequest(
            "max_tokens must be greater than 0".to_owned(),
        )),
        Some(value) if value > configured_max => Err(BackendError::UnsupportedRequest(format!(
            "requested max_tokens {value} exceeds configured native Qwen limit {configured_max}"
        ))),
        Some(value) => Ok(value),
    }
}

fn native_qwen_cache_token_capacity(max_prefill_tokens: usize, _max_new_tokens: u32) -> usize {
    max_prefill_tokens.max(1)
}

#[derive(Debug, Clone)]
struct NativeQwenCandidate {
    token_id: usize,
}

fn sample_token_id_with_draw(
    logits: &[f32],
    sampling: SamplingConfig,
    draw: f32,
) -> Result<usize, BackendError> {
    if logits.is_empty() {
        return Err(BackendError::Other(
            "Qwen lm head returned no logits".to_owned(),
        ));
    }
    match sampling {
        SamplingConfig::Greedy => llm_sampler::GreedySampler
            .sample(logits)
            .map_err(|err| BackendError::Other(err.to_string())),
        SamplingConfig::TopP { temperature, top_p } => TopPSampler { temperature, top_p }
            .sample(logits, draw)
            .map_err(|err| BackendError::Other(err.to_string())),
    }
}

static NATIVE_SAMPLING_COUNTER: AtomicU64 = AtomicU64::new(0);

fn native_sampling_draw() -> f32 {
    let time_seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0);
    let counter = NATIVE_SAMPLING_COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut value = time_seed ^ counter.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    let bits = value.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40;
    (bits as f32) / ((1_u32 << 24) as f32)
}

#[async_trait]
impl ModelBackend for NativeQwenBackend {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        self.metadata.clone()
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
        let backend = self.clone();
        tokio::task::spawn_blocking(move || backend.generate_blocking(request, cancellation))
            .await
            .map_err(|err| BackendError::Other(format!("native Qwen worker failed: {err}")))?
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
        let backend = self.clone();
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        tokio::task::spawn_blocking(move || {
            let err_tx = tx.clone();
            if let Err(err) = backend.generate_blocking_stream(request, tx, cancellation) {
                let _ = err_tx.blocking_send(Err(err));
            }
        });
        stream::unfold(rx, |mut rx| async {
            rx.recv().await.map(|item| (item, rx))
        })
        .boxed()
    }
}

fn send_backend_stream_chunk(
    tx: &tokio::sync::mpsc::Sender<Result<BackendStreamChunk, BackendError>>,
    chunk: BackendStreamChunk,
) -> Result<(), BackendError> {
    tx.blocking_send(Ok(chunk))
        .map_err(|_| BackendError::Other("stream receiver dropped".to_owned()))
}

async fn health() -> impl IntoResponse {
    Json(json!({
        "status": "ok",
        "runtime": "rust",
        "python_runtime": false
    }))
}

async fn models(State(state): State<AppState>) -> impl IntoResponse {
    Json(ModelList {
        object: "list".to_owned(),
        data: vec![ModelCard {
            id: state.runtime.model_id().to_owned(),
            object: "model".to_owned(),
            owned_by: "local".to_owned(),
        }],
    })
}

async fn admin_models(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Value>, EngineError> {
    require_admin(&state, &headers)?;
    let metadata = state.runtime.model_metadata();
    Ok(Json(json!({
        "object": "list",
        "data": [admin_model_status(&metadata)],
    })))
}

async fn admin_model(
    AxumPath(alias): AxumPath<String>,
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Value>, EngineError> {
    require_admin(&state, &headers)?;
    let metadata = state.runtime.model_metadata();
    if alias != metadata.id {
        return Err(RuntimeError::Backend(BackendError::ModelNotFound {
            requested: alias,
            available: metadata.id,
        })
        .into());
    }
    Ok(Json(admin_model_status(&metadata)))
}

async fn admin_model_verify(
    AxumPath(alias): AxumPath<String>,
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Value>, EngineError> {
    require_admin(&state, &headers)?;
    let metadata = state.runtime.model_metadata();
    if alias != metadata.id {
        return Err(RuntimeError::Backend(BackendError::ModelNotFound {
            requested: alias,
            available: metadata.id,
        })
        .into());
    }
    let snapshot_path = metadata.snapshot_path.ok_or_else(|| {
        RuntimeError::Api(ApiError::unsupported_capability(
            "model verification requires snapshot metadata",
        ))
    })?;
    let verification = match ModelStore::verify_snapshot(&snapshot_path).await {
        Ok(verification) => verification,
        Err(err) => {
            record_artifact_verification_failure_metrics(&state);
            return Err(EngineError::ModelStore(err));
        }
    };
    Ok(Json(json!({
        "status": "ok",
        "snapshot_path": verification.snapshot.path,
        "repo_id": verification.snapshot.manifest.repo_id,
        "resolved_commit": verification.snapshot.manifest.resolved_commit,
        "manifest_digest": verification.snapshot.manifest_digest,
        "verified_files": verification.verified_files,
        "verified_bytes": verification.verified_bytes,
    })))
}

#[derive(Debug, Deserialize)]
struct AdminModelPlanRequest {
    repo_id: String,
    #[serde(default)]
    revision: Option<String>,
    #[serde(default)]
    profile: Option<String>,
    #[serde(default)]
    metadata_only: bool,
}

async fn admin_model_plan(
    AxumPath(alias): AxumPath<String>,
    State(state): State<AppState>,
    headers: HeaderMap,
    request: Result<Json<AdminModelPlanRequest>, JsonRejection>,
) -> Result<Json<DownloadPlan>, EngineError> {
    require_admin(&state, &headers)?;
    require_model_alias(&state, alias)?;
    let request = parse_json_request(request, &state)?;
    let plan = build_admin_download_plan(&state, request).await?;
    Ok(Json(plan))
}

async fn admin_model_pull(
    AxumPath(alias): AxumPath<String>,
    State(state): State<AppState>,
    headers: HeaderMap,
    request: Result<Json<AdminModelPlanRequest>, JsonRejection>,
) -> Result<Json<Value>, EngineError> {
    require_admin(&state, &headers)?;
    require_model_alias(&state, alias)?;
    let request = parse_json_request(request, &state)?;
    let plan = match build_admin_download_plan(&state, request).await {
        Ok(plan) => plan,
        Err(err) => {
            record_model_pull_failure_metrics(&state);
            return Err(err);
        }
    };
    let snapshot = match ModelStore::new(&state.model_home)
        .pull_plan(&state.hub_client, &plan, state.hf_token.as_deref())
        .await
    {
        Ok(snapshot) => snapshot,
        Err(err) => {
            record_model_pull_failure_metrics(&state);
            return Err(EngineError::ModelStore(err));
        }
    };
    let model_pull_bytes = snapshot.manifest.files.iter().map(|file| file.size).sum();
    record_model_pull_success_metrics(&state, model_pull_bytes);
    invalidate_model_store_usage_cache(&state);
    Ok(Json(json!({
        "snapshot_path": snapshot.path,
        "manifest_digest": snapshot.manifest_digest,
        "repo_id": snapshot.manifest.repo_id,
        "resolved_commit": snapshot.manifest.resolved_commit,
        "profile": snapshot.manifest.profile,
        "files": snapshot.manifest.files.len(),
    })))
}

async fn build_admin_download_plan(
    state: &AppState,
    request: AdminModelPlanRequest,
) -> Result<DownloadPlan, EngineError> {
    let repo_id = HubRepoId::model(request.repo_id).map_err(EngineError::ModelStore)?;
    let revision = request.revision.unwrap_or_else(|| "main".to_owned());
    let profile_name = request
        .profile
        .unwrap_or_else(|| "qwen36-safetensors-bf16".to_owned());
    let profile = model_profile(&profile_name)?;
    let mut plan = state
        .hub_client
        .plan_model(repo_id, &revision, profile, state.hf_token.as_deref())
        .await
        .map_err(EngineError::ModelStore)?;
    if request.metadata_only {
        plan = plan.metadata_only();
    }
    Ok(plan)
}

fn model_profile(name: &str) -> Result<ModelProfile, EngineError> {
    match name {
        "qwen36-mlx-4bit" => Ok(ModelProfile::qwen36_mlx_4bit()),
        "qwen36-safetensors-bf16" => Ok(ModelProfile::qwen36_safetensors_bf16()),
        other => Err(RuntimeError::Api(ApiError::invalid_request(format!(
            "unknown model profile `{other}`"
        )))
        .into()),
    }
}

fn require_model_alias(state: &AppState, alias: String) -> Result<(), EngineError> {
    let model_id = state.runtime.model_id();
    if alias == model_id {
        return Ok(());
    }
    Err(RuntimeError::Backend(BackendError::ModelNotFound {
        requested: alias,
        available: model_id.to_owned(),
    })
    .into())
}

fn admin_model_status(metadata: &BackendModelMetadata) -> Value {
    json!({
        "id": metadata.id,
        "object": "admin.model",
        "status": "ready",
        "runtime": "rust",
        "python_runtime": false,
        "backend": metadata.backend,
        "family": metadata.family,
        "loader": metadata.loader,
        "quantization": metadata.quantization,
        "repo_id": metadata.repo_id,
        "resolved_commit": metadata.resolved_commit,
        "profile": metadata.profile,
        "snapshot_path": metadata.snapshot_path,
        "manifest_digest": metadata.manifest_digest,
    })
}

fn native_qwen_metadata(
    model_id: &str,
    snapshot_path: &Path,
) -> anyhow::Result<BackendModelMetadata> {
    let manifest_path = snapshot_path.join("llm-engine-manifest.json");
    let mut metadata = BackendModelMetadata::new(model_id.to_owned(), "native-qwen");
    metadata.snapshot_path = Some(PathBuf::from(snapshot_path));
    let manifest_bytes = match std::fs::read(&manifest_path) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(metadata),
        Err(err) => return Err(err.into()),
    };
    let manifest = serde_json::from_slice::<SnapshotManifest>(&manifest_bytes)?;
    metadata.family = Some(manifest.family.clone());
    metadata.loader = Some(manifest.loader.clone());
    metadata.quantization = Some(manifest.quantization.clone());
    metadata.repo_id = Some(manifest.repo_id.clone());
    metadata.resolved_commit = Some(manifest.resolved_commit.clone());
    metadata.profile = Some(manifest.profile.clone());
    metadata.manifest_digest = Some(manifest.digest());
    Ok(metadata)
}

async fn admin_metrics(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Value>, EngineError> {
    require_admin(&state, &headers)?;
    let metrics = *state.metrics.lock().expect("metrics lock is not poisoned");
    let tokens = metrics.tokens();
    let request_latency = metrics.request_latency();
    let time_to_first_token = metrics.time_to_first_token();
    let model_store_usage = model_store_usage(&state).await?;
    let active_requests = state
        .active_requests
        .lock()
        .expect("active request lock is not poisoned")
        .len();
    Ok(Json(json!({
        "requests_total": metrics.requests_total(),
        "successful_requests": metrics.successful_requests(),
        "failed_requests": metrics.failed_requests(),
        "streamed_requests": metrics.streamed_requests(),
        "active_requests": active_requests,
        "queued_requests": 0,
        "prefill_requests": state.generation_phases.prefill_requests(),
        "decode_requests": state.generation_phases.decode_requests(),
        "cancelled_requests": metrics.cancelled_requests(),
        "no_progress_failures": metrics.no_progress_failures(),
        "model_pull_operations": metrics.model_pull_operations(),
        "model_pull_successes": metrics.model_pull_successes(),
        "model_pull_failures": metrics.model_pull_failures(),
        "model_pull_bytes": metrics.model_pull_bytes(),
        "model_store_snapshots": model_store_usage.snapshots,
        "model_store_bytes": model_store_usage.bytes,
        "artifact_verification_failures": metrics.artifact_verification_failures(),
        "process_rss_bytes": process_rss_bytes(),
        "tokens_per_second": metrics.tokens_per_second(),
        "native_qwen_metal": native_qwen_metal_metrics().snapshot(),
        "request_latency_ms": {
            "count": request_latency.count(),
            "min": request_latency.min_ms(),
            "max": request_latency.max_ms(),
            "avg": request_latency.avg_ms(),
        },
        "time_to_first_token_ms": {
            "count": time_to_first_token.count(),
            "min": time_to_first_token.min_ms(),
            "max": time_to_first_token.max_ms(),
            "avg": time_to_first_token.avg_ms(),
        },
        "tokens": {
            "prompt_tokens": tokens.prompt_tokens(),
            "completion_tokens": tokens.completion_tokens(),
            "total_tokens": tokens.total_tokens(),
        }
    })))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ModelStoreUsage {
    snapshots: usize,
    bytes: u64,
}

#[derive(Debug, Default)]
struct ModelStoreUsageCache {
    usage: Option<ModelStoreUsage>,
    refreshed_at: Option<Instant>,
}

const MODEL_STORE_USAGE_CACHE_TTL: Duration = Duration::from_secs(30);

impl ModelStoreUsageCache {
    fn current(&self, now: Instant) -> Option<ModelStoreUsage> {
        let usage = self.usage?;
        let refreshed_at = self.refreshed_at?;
        if now.duration_since(refreshed_at) <= MODEL_STORE_USAGE_CACHE_TTL {
            Some(usage)
        } else {
            None
        }
    }

    fn store(&mut self, usage: ModelStoreUsage, refreshed_at: Instant) {
        self.usage = Some(usage);
        self.refreshed_at = Some(refreshed_at);
    }

    fn invalidate(&mut self) {
        self.usage = None;
        self.refreshed_at = None;
    }
}

async fn model_store_usage(state: &AppState) -> Result<ModelStoreUsage, EngineError> {
    let now = Instant::now();
    if let Some(usage) = state
        .model_store_usage
        .lock()
        .expect("model store usage cache lock is not poisoned")
        .current(now)
    {
        return Ok(usage);
    }
    let usage = scan_model_store_usage(&state.model_home).await?;
    state
        .model_store_usage
        .lock()
        .expect("model store usage cache lock is not poisoned")
        .store(usage, Instant::now());
    Ok(usage)
}

async fn scan_model_store_usage(model_home: &Path) -> Result<ModelStoreUsage, EngineError> {
    let snapshots = ModelStore::new(model_home)
        .list_snapshots()
        .await
        .map_err(EngineError::ModelStore)?;
    let bytes = snapshots
        .iter()
        .flat_map(|snapshot| &snapshot.manifest.files)
        .map(|file| file.size)
        .sum();
    Ok(ModelStoreUsage {
        snapshots: snapshots.len(),
        bytes,
    })
}

fn invalidate_model_store_usage_cache(state: &AppState) {
    state
        .model_store_usage
        .lock()
        .expect("model store usage cache lock is not poisoned")
        .invalidate();
}

fn process_rss_bytes() -> u64 {
    platform_process_rss_bytes().unwrap_or(0)
}

#[cfg(target_os = "macos")]
#[allow(deprecated)]
fn platform_process_rss_bytes() -> Option<u64> {
    let mut info = std::mem::MaybeUninit::<libc::mach_task_basic_info>::uninit();
    let mut count = (std::mem::size_of::<libc::mach_task_basic_info>()
        / std::mem::size_of::<libc::natural_t>())
        as libc::mach_msg_type_number_t;
    let task = unsafe { libc::mach_task_self_ };
    let result = unsafe {
        libc::task_info(
            task,
            libc::MACH_TASK_BASIC_INFO,
            info.as_mut_ptr().cast(),
            &mut count,
        )
    };
    if result == libc::KERN_SUCCESS {
        let info = unsafe { info.assume_init() };
        Some(info.resident_size)
    } else {
        None
    }
}

#[cfg(target_os = "linux")]
fn platform_process_rss_bytes() -> Option<u64> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let resident_pages = statm.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_size <= 0 {
        return None;
    }
    resident_pages.checked_mul(page_size as u64)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn platform_process_rss_bytes() -> Option<u64> {
    None
}

async fn admin_cancel_request(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(request_id): AxumPath<String>,
) -> Result<Json<Value>, EngineError> {
    require_admin(&state, &headers)?;
    let cancellation = state
        .active_requests
        .lock()
        .expect("active request lock is not poisoned")
        .get(&request_id)
        .cloned()
        .ok_or_else(|| EngineError::RequestNotFound(request_id.clone()))?;
    cancellation.cancel();
    record_cancellation_metrics(&state);
    Ok(Json(json!({
        "object": "admin.request_cancellation",
        "request_id": request_id,
        "status": "cancelled"
    })))
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    request: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Result<Response, EngineError> {
    let request = parse_json_request(request, &state)?;
    validate_api_request(&request, &state)?;
    let streamed = request.stream;
    if request.stream {
        let permit = acquire_model_permit(&state)?;
        let active_request = register_active_request(&state, &headers)?;
        let phase = state.generation_phases.begin(GenerationPhase::Prefill);
        let request_id = active_request.id.clone();
        let request_started = Instant::now();
        if chat_stream_requires_buffering(&request) {
            let response = match state
                .runtime
                .chat_stream_buffered_with_cancel(request, active_request.cancellation.clone())
                .await
            {
                Ok(response) => response,
                Err(err) => {
                    record_runtime_error_metrics(&state, &err);
                    return Err(err.into());
                }
            };
            let events = async_stream::stream! {
                let _permit = permit;
                let _active_request = active_request;
                let mut phase = phase;
                let mut events = response.into_events();
                let mut ttft_recorded = false;
                loop {
                    match next_stream_event(&mut events, state.stream_stall_timeout).await {
                        Ok(Some(Ok(ChatCompletionStreamEvent::Chunk(chunk)))) => {
                            if !ttft_recorded && chat_chunk_has_real_delta(&chunk) {
                                phase.transition_to_decode();
                                record_time_to_first_token_metrics(&state, request_started.elapsed());
                                ttft_recorded = true;
                            }
                            yield sse_json_event(chunk);
                        }
                        Ok(Some(Ok(ChatCompletionStreamEvent::Complete(usage)))) => {
                            record_success_metrics(&state, &usage, streamed, request_started.elapsed());
                            yield Ok(Event::default().data("[DONE]"));
                        }
                        Ok(Some(Err(err))) => {
                            record_runtime_error_metrics(&state, &err);
                            for event in runtime_error_stream_events(err) {
                                yield event;
                            }
                            return;
                        }
                        Ok(None) => break,
                        Err(StreamStalled) => {
                            record_failure_metrics(&state);
                            for event in stream_stalled_stream_events(state.stream_stall_timeout) {
                                yield event;
                            }
                            return;
                        }
                    }
                }
            };
            let mut response = Sse::new(events)
                .keep_alive(engine_sse_keep_alive())
                .into_response();
            insert_request_id_header(&mut response, &request_id);
            return Ok(response);
        }
        let request_started = Instant::now();
        let events = async_stream::stream! {
                    let _permit = permit;
                    let _active_request = active_request;
                    let mut phase = phase;
                    match state
                        .runtime
                        .chat_stream_with_cancel(request, _active_request.cancellation.clone())
                        .await
                    {
                        Ok(response) => {
                            let mut events = response.into_events();
                            let mut ttft_recorded = false;
                            loop {
                                match next_stream_event(&mut events, state.stream_stall_timeout).await {
                                    Ok(Some(Ok(ChatCompletionStreamEvent::Chunk(chunk)))) => {
                                        if !ttft_recorded && chat_chunk_has_real_delta(&chunk) {
                                            phase.transition_to_decode();
                                            record_time_to_first_token_metrics(&state, request_started.elapsed());
                                            ttft_recorded = true;
                                        }
                                        yield sse_json_event(chunk);
                                    }
                                    Ok(Some(Ok(ChatCompletionStreamEvent::Complete(usage)))) => {
                                        record_success_metrics(&state, &usage, streamed, request_started.elapsed());
                                        yield Ok(Event::default().data("[DONE]"));
                                    }
                                    Ok(Some(Err(err))) => {
                                        record_runtime_error_metrics(&state, &err);
                                        for event in runtime_error_stream_events(err) {
                                            yield event;
                                        }
                                        return;
                                    }
                                    Ok(None) => break,
                                    Err(StreamStalled) => {
                                        record_failure_metrics(&state);
                                        for event in stream_stalled_stream_events(state.stream_stall_timeout) {
                                            yield event;
                                        }
                                        return;
                    }
                }
            }
        }

                        Err(err) => {
                            record_runtime_error_metrics(&state, &err);
                            for event in runtime_error_stream_events(err) {
                                yield event;
                            }
                        }
                    }
                };
        let mut response = Sse::new(events)
            .keep_alive(engine_sse_keep_alive())
            .into_response();
        insert_request_id_header(&mut response, &request_id);
        return Ok(response);
    }
    let _permit = acquire_model_permit(&state)?;
    let active_request = register_active_request(&state, &headers)?;
    let _phase = state.generation_phases.begin(GenerationPhase::Decode);
    let request_id = active_request.id.clone();
    let request_started = Instant::now();
    let response = match state
        .runtime
        .chat_with_cancel(request, active_request.cancellation.clone())
        .await
    {
        Ok(response) => response,
        Err(err) => {
            record_runtime_error_metrics(&state, &err);
            return Err(err.into());
        }
    };
    drop(active_request);
    record_success_metrics(&state, &response.usage, streamed, request_started.elapsed());
    let mut response = Json(response).into_response();
    insert_request_id_header(&mut response, &request_id);
    Ok(response)
}

async fn completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    request: Result<Json<CompletionRequest>, JsonRejection>,
) -> Result<Response, EngineError> {
    let request = parse_json_request(request, &state)?;
    validate_api_request(&request, &state)?;
    let streamed = request.stream;
    if request.stream {
        let permit = acquire_model_permit(&state)?;
        let active_request = register_active_request(&state, &headers)?;
        let phase = state.generation_phases.begin(GenerationPhase::Prefill);
        let request_id = active_request.id.clone();
        let request_started = Instant::now();
        let events = async_stream::stream! {
            let _permit = permit;
            let _active_request = active_request;
            let mut phase = phase;
            match state
                .runtime
                .completion_stream_with_cancel(request, _active_request.cancellation.clone())
                .await
            {
                Ok(response) => {
                    let mut events = response.into_events();
                    let mut ttft_recorded = false;
                    loop {
                        match next_stream_event(&mut events, state.stream_stall_timeout).await {
                            Ok(Some(Ok(CompletionStreamEvent::Chunk(chunk)))) => {
                                if !ttft_recorded && completion_chunk_has_real_delta(&chunk) {
                                    phase.transition_to_decode();
                                    record_time_to_first_token_metrics(&state, request_started.elapsed());
                                    ttft_recorded = true;
                                }
                                yield sse_json_event(chunk);
                            }
                            Ok(Some(Ok(CompletionStreamEvent::Complete(usage)))) => {
                                record_success_metrics(&state, &usage, streamed, request_started.elapsed());
                                yield Ok(Event::default().data("[DONE]"));
                            }
                            Ok(Some(Err(err))) => {
                                record_runtime_error_metrics(&state, &err);
                                for event in runtime_error_stream_events(err) {
                                    yield event;
                                }
                                return;
                            }
                            Ok(None) => break,
                            Err(StreamStalled) => {
                                record_failure_metrics(&state);
                                for event in stream_stalled_stream_events(state.stream_stall_timeout) {
                                    yield event;
                                }
                                return;
                            }
                        }
                    }
                }
                Err(err) => {
                    record_runtime_error_metrics(&state, &err);
                    for event in runtime_error_stream_events(err) {
                        yield event;
                    }
                }
            }
        };
        let mut response = Sse::new(events)
            .keep_alive(engine_sse_keep_alive())
            .into_response();
        insert_request_id_header(&mut response, &request_id);
        return Ok(response);
    }
    let _permit = acquire_model_permit(&state)?;
    let active_request = register_active_request(&state, &headers)?;
    let _phase = state.generation_phases.begin(GenerationPhase::Decode);
    let request_id = active_request.id.clone();
    let request_started = Instant::now();
    let response = match state
        .runtime
        .completion_with_cancel(request, active_request.cancellation.clone())
        .await
    {
        Ok(response) => response,
        Err(err) => {
            record_runtime_error_metrics(&state, &err);
            return Err(err.into());
        }
    };
    drop(active_request);
    record_success_metrics(&state, &response.usage, streamed, request_started.elapsed());
    let mut response = Json(response).into_response();
    insert_request_id_header(&mut response, &request_id);
    Ok(response)
}

fn runtime_error_stream_events(err: RuntimeError) -> Vec<Result<Event, Infallible>> {
    let metadata = runtime_error_metadata(&err);
    vec![
        sse_json_event(json!({
            "error": {
                "message": err.to_string(),
                "code": metadata.code,
                "phase": metadata.phase,
                "retryable": metadata.retryable,
                "type": "llm_engine_error"
            }
        })),
        Ok(Event::default().data("[DONE]")),
    ]
}

#[derive(Debug, Clone, Copy)]
struct RuntimeErrorMetadata {
    status: StatusCode,
    code: &'static str,
    phase: &'static str,
    retryable: bool,
}

fn runtime_error_metadata(err: &RuntimeError) -> RuntimeErrorMetadata {
    let (status, code, phase, retryable) = match err {
        RuntimeError::Api(api) => (
            StatusCode::BAD_REQUEST,
            api.code(),
            "request_validation",
            false,
        ),
        RuntimeError::Backend(BackendError::ModelNotFound { .. }) => (
            StatusCode::NOT_FOUND,
            "model_not_found",
            "model_resolution",
            false,
        ),
        RuntimeError::Backend(BackendError::UnsupportedRequest(_)) => (
            StatusCode::BAD_REQUEST,
            "unsupported_capability",
            "request_validation",
            false,
        ),
        RuntimeError::Backend(BackendError::Cancelled) => {
            (StatusCode::REQUEST_TIMEOUT, "cancelled", "decode", false)
        }
        RuntimeError::Backend(BackendError::Other(_)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "backend_execution_failed",
            "decode",
            true,
        ),
        RuntimeError::Template(_) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            "chat_template_failed",
            "prompt_rendering",
            false,
        ),
        RuntimeError::Parser(err) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            err.code(),
            "response_parsing",
            false,
        ),
        RuntimeError::Json(_) | RuntimeError::JsonMode(_) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            "json_validation_failed",
            "response_validation",
            false,
        ),
        RuntimeError::ToolCallValidation(_) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            "tool_call_validation_failed",
            "response_validation",
            false,
        ),
        RuntimeError::NoProgress(_) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            "no_progress",
            "response_validation",
            false,
        ),
    };
    RuntimeErrorMetadata {
        status,
        code,
        phase,
        retryable,
    }
}

fn stream_stalled_stream_events(timeout: Option<Duration>) -> Vec<Result<Event, Infallible>> {
    let message = match timeout {
        Some(timeout) => format!(
            "stream stalled for {} ms without backend output",
            timeout.as_millis()
        ),
        None => "stream stalled without backend output".to_owned(),
    };
    vec![
        sse_json_event(json!({
            "error": {
                "message": message,
                "code": "stream_stalled",
                "phase": "streaming",
                "retryable": true,
                "type": "llm_engine_error"
            }
        })),
        Ok(Event::default().data("[DONE]")),
    ]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StreamStalled;

async fn next_stream_event<S, T>(
    events: &mut S,
    timeout: Option<Duration>,
) -> Result<Option<Result<T, RuntimeError>>, StreamStalled>
where
    S: Stream<Item = Result<T, RuntimeError>> + Unpin,
{
    match timeout {
        Some(timeout) => tokio::time::timeout(timeout, events.next())
            .await
            .map_err(|_| StreamStalled),
        None => Ok(events.next().await),
    }
}

fn engine_sse_keep_alive() -> KeepAlive {
    KeepAlive::new()
        .interval(Duration::from_millis(100))
        .text("llm-engine-heartbeat")
}

fn sse_json_event(value: impl serde::Serialize) -> Result<Event, Infallible> {
    let data = serde_json::to_string(&value).unwrap_or_else(|err| {
        json!({
            "error": {
                "message": format!("response serialization failed: {err}"),
                "type": "llm_engine_error"
            }
        })
        .to_string()
    });
    Ok(Event::default().data(data))
}

fn chat_chunk_has_real_delta(chunk: &ChatCompletionStreamResponse) -> bool {
    chunk.choices.iter().any(|choice| {
        choice
            .delta
            .content
            .as_ref()
            .is_some_and(|content| !content.is_empty())
            || !choice.delta.tool_calls.is_empty()
    })
}

fn completion_chunk_has_real_delta(chunk: &CompletionStreamResponse) -> bool {
    chunk.choices.iter().any(|choice| !choice.text.is_empty())
}

fn record_success_metrics(state: &AppState, usage: &Usage, streamed: bool, latency: Duration) {
    state
        .metrics
        .lock()
        .expect("metrics lock is not poisoned")
        .record_success(
            TokenCounters::new(usage.prompt_tokens, usage.completion_tokens),
            streamed,
            latency,
        );
}

fn record_failure_metrics(state: &AppState) {
    state
        .metrics
        .lock()
        .expect("metrics lock is not poisoned")
        .record_failure();
}

fn record_runtime_error_metrics(state: &AppState, err: &RuntimeError) {
    let mut metrics = state.metrics.lock().expect("metrics lock is not poisoned");
    if matches!(err, RuntimeError::NoProgress(_)) {
        metrics.record_no_progress_failure();
    }
    metrics.record_failure();
}

fn record_cancellation_metrics(state: &AppState) {
    state
        .metrics
        .lock()
        .expect("metrics lock is not poisoned")
        .record_cancellation();
}

fn record_model_pull_success_metrics(state: &AppState, bytes: u64) {
    state
        .metrics
        .lock()
        .expect("metrics lock is not poisoned")
        .record_model_pull_success(bytes);
}

fn record_model_pull_failure_metrics(state: &AppState) {
    state
        .metrics
        .lock()
        .expect("metrics lock is not poisoned")
        .record_model_pull_failure();
}

fn record_artifact_verification_failure_metrics(state: &AppState) {
    state
        .metrics
        .lock()
        .expect("metrics lock is not poisoned")
        .record_artifact_verification_failure();
}

fn record_time_to_first_token_metrics(state: &AppState, latency: Duration) {
    state
        .metrics
        .lock()
        .expect("metrics lock is not poisoned")
        .record_time_to_first_token(latency);
}

fn acquire_model_permit(state: &AppState) -> Result<OwnedSemaphorePermit, EngineError> {
    state
        .model_permits
        .clone()
        .try_acquire_owned()
        .map_err(|_| {
            record_failure_metrics(state);
            EngineError::Overloaded("model is busy; retry the request later".to_owned())
        })
}

fn register_active_request(
    state: &AppState,
    headers: &HeaderMap,
) -> Result<ActiveRequest, EngineError> {
    let id = match request_id_from_headers(state, headers) {
        Ok(id) => id,
        Err(err) => {
            record_failure_metrics(state);
            return Err(err);
        }
    };
    let cancellation = CancellationToken::new();
    let mut active_requests = state
        .active_requests
        .lock()
        .expect("active request lock is not poisoned");
    if active_requests.contains_key(&id) {
        record_failure_metrics(state);
        return Err(EngineError::RequestConflict(id));
    }
    active_requests.insert(id.clone(), cancellation.clone());
    drop(active_requests);
    Ok(ActiveRequest {
        id,
        cancellation,
        active_requests: state.active_requests.clone(),
    })
}

fn request_id_from_headers(state: &AppState, headers: &HeaderMap) -> Result<String, EngineError> {
    let Some(value) = headers
        .get("x-request-id")
        .or_else(|| headers.get("x-llm-request-id"))
    else {
        let next = state.next_request_id.fetch_add(1, Ordering::Relaxed);
        return Ok(format!("req-{next}"));
    };
    let request_id = value
        .to_str()
        .map_err(|_| EngineError::InvalidRequestId("request id must be visible ASCII".to_owned()))?
        .trim();
    if request_id.is_empty() {
        return Err(EngineError::InvalidRequestId(
            "request id must not be empty".to_owned(),
        ));
    }
    if request_id.len() > 128 {
        return Err(EngineError::InvalidRequestId(
            "request id must be at most 128 bytes".to_owned(),
        ));
    }
    Ok(request_id.to_owned())
}

fn insert_request_id_header(response: &mut Response, request_id: &str) {
    let value = HeaderValue::from_str(request_id)
        .expect("registered request id came from a valid header value or generated ASCII");
    response.headers_mut().insert("x-request-id", value);
}

fn require_admin(state: &AppState, headers: &HeaderMap) -> Result<(), EngineError> {
    let Some(token) = &state.admin_token else {
        return Ok(());
    };
    let Some(header_value) = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
    else {
        return Err(EngineError::UnauthorizedAdmin);
    };
    if header_value == format!("Bearer {token}") {
        return Ok(());
    }
    Err(EngineError::UnauthorizedAdmin)
}

fn parse_json_request<T>(
    request: Result<Json<T>, JsonRejection>,
    state: &AppState,
) -> Result<T, EngineError> {
    match request {
        Ok(Json(request)) => Ok(request),
        Err(err) => {
            record_failure_metrics(state);
            Err(RuntimeError::Api(ApiError::invalid_request(format!(
                "invalid JSON request body: {err}"
            )))
            .into())
        }
    }
}

fn validate_api_request<T: ValidateRequest>(
    request: &T,
    state: &AppState,
) -> Result<(), EngineError> {
    request.validate().map_err(|err| {
        record_failure_metrics(state);
        RuntimeError::Api(err).into()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_backend_metrics_records_attempt_success_and_fallback_by_kernel() {
        let metrics = MetalBackendMetrics::default();

        metrics.record_attempt("matvec_bf16_f32");
        metrics.record_success("matvec_bf16_f32");
        metrics.record_attempt("matvec_bf16_f32");
        metrics.record_fallback("matvec_bf16_f32", "rows=2,cols=3", "execution failed");

        let snapshot = metrics.snapshot();
        let matvec = &snapshot["kernels"]["matvec_bf16_f32"];
        assert_eq!(matvec["attempts"], 2);
        assert_eq!(matvec["successes"], 1);
        assert_eq!(matvec["fallbacks"], 1);
    }

    #[test]
    fn metal_backend_metrics_records_bf16_matrix_cache_activity() {
        let metrics = MetalBackendMetrics::default();

        metrics.record_bf16_matrix_cache_miss();
        metrics.record_bf16_matrix_cache_upload(12);
        metrics.record_bf16_matrix_cache_hit();

        let snapshot = metrics.snapshot();
        let cache = &snapshot["bf16_matrix_cache"];
        assert_eq!(cache["hits"], 1);
        assert_eq!(cache["misses"], 1);
        assert_eq!(cache["uploads"], 1);
        assert_eq!(cache["bytes_uploaded"], 12);
    }

    #[test]
    fn native_max_tokens_defaults_to_configured_cache_limit() {
        assert_eq!(
            resolve_native_max_tokens(None, 4).expect("omitted max tokens uses configured cap"),
            4
        );
    }

    #[test]
    fn native_max_tokens_accepts_multi_token_decode_with_cache() {
        assert_eq!(
            resolve_native_max_tokens(Some(2), 4).expect("multi-token decode uses cache"),
            2
        );
    }

    #[test]
    fn native_max_tokens_rejects_requests_above_configured_limit() {
        let err = resolve_native_max_tokens(Some(5), 4)
            .expect_err("request above configured limit fails closed");

        assert!(matches!(err, BackendError::UnsupportedRequest(_)));
        assert!(err.to_string().contains("configured native Qwen limit"));
    }

    #[test]
    fn native_qwen_cache_capacity_uses_retained_window_not_generation_limit() {
        let capacity = native_qwen_cache_token_capacity(32, 1024);
        let spec = QwenModelSpec {
            family: llm_models::ModelFamily::Qwen,
            architecture: "Qwen3_5MoeForConditionalGeneration".to_owned(),
            model_type: "qwen3_5_moe".to_owned(),
            text_model_type: "qwen3_5_moe_text".to_owned(),
            hidden_size: 2,
            rms_norm_eps: 0.0,
            tie_word_embeddings: false,
            rope_theta: 1_000_000.0,
            partial_rotary_factor: 1.0,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 2,
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_key_head_dim: 1,
            linear_value_head_dim: 1,
            linear_conv_kernel_dim: 1,
            num_experts: 1,
            num_experts_per_tok: 1,
            moe_intermediate_size: 1,
            shared_expert_intermediate_size: 1,
            max_position_embeddings: 32,
            vocab_size: 16,
            layer_kinds: vec![llm_models::AttentionKind::FullAttention],
        };

        let caches = qwen_layer_caches_for_spec(&spec, capacity).expect("cache allocates");
        match &caches[0] {
            QwenLayerCache::Full(cache) => assert_eq!(cache.max_tokens(), 32),
            QwenLayerCache::Linear(_) => panic!("expected full-attention cache"),
        }
    }

    #[test]
    fn native_qwen_start_decode_session_prefills_full_context_with_bounded_cache() {
        let snapshot = temp_snapshot_dir("full-context-prefill");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));
        write_tiny_linear_decoder_snapshot(&snapshot);
        let backend = NativeQwenBackend {
            model_id: "local-qwen36".to_owned(),
            metadata: BackendModelMetadata::new("local-qwen36", "native-qwen"),
            tokenizer: HuggingFaceTokenizer::from_file(snapshot.join("tokenizer.json"))
                .expect("tokenizer loads"),
            spec: tiny_engine_qwen_spec(llm_models::AttentionKind::LinearAttention),
            store: SafeTensorShardStore::open(&snapshot).expect("store opens"),
            matvec: NativeQwenMatvecBackend::Cpu,
            max_new_tokens: 8,
            max_prefill_tokens: 1,
            top_k: 2,
            chunk_rows: 64,
        };

        let decode = backend
            .start_decode_session(&[0, 1, 0], 8, &CancellationToken::new())
            .expect("decode session starts");

        match &decode.caches[0] {
            QwenLayerCache::Linear(cache) => assert_eq!(cache.token_count(), 3),
            QwenLayerCache::Full(_) => panic!("layer 0 should be linear attention"),
        }
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn native_qwen_prefill_context_uses_sequence_cache_path_for_full_context() {
        let snapshot = temp_snapshot_dir("sequence-prefill");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        write_tiny_linear_decoder_snapshot(&snapshot);
        let spec = tiny_engine_qwen_spec(llm_models::AttentionKind::LinearAttention);
        let store = SafeTensorShardStore::open(&snapshot).expect("store opens");
        let mut caches = qwen_layer_caches_for_spec(&spec, 1).expect("caches allocate");

        let hidden = native_qwen_prefill_context_with_cache(
            &store,
            &spec,
            &[0, 1, 0],
            &mut caches,
            &NativeQwenMatvecBackend::Cpu,
            1,
            &CancellationToken::new(),
        )
        .expect("sequence prefill succeeds");

        assert_eq!(hidden.len(), 2);
        match &caches[0] {
            QwenLayerCache::Linear(cache) => assert_eq!(cache.token_count(), 3),
            QwenLayerCache::Full(_) => panic!("layer 0 should be linear attention"),
        }
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn native_qwen_prefill_context_checks_cancellation_between_chunks() {
        let snapshot = temp_snapshot_dir("sequence-prefill-cancel");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        write_tiny_linear_decoder_snapshot(&snapshot);
        let spec = tiny_engine_qwen_spec(llm_models::AttentionKind::LinearAttention);
        let store = SafeTensorShardStore::open(&snapshot).expect("store opens");
        let mut caches = qwen_layer_caches_for_spec(&spec, 1).expect("caches allocate");
        let cancellation = CancellationToken::new();
        let matvec = CancelAfterFirstConv {
            cancellation: cancellation.clone(),
            conv_calls: std::cell::Cell::new(0),
        };

        let err = native_qwen_prefill_context_with_cache(
            &store,
            &spec,
            &[0, 1, 0],
            &mut caches,
            &matvec,
            1,
            &cancellation,
        )
        .expect_err("cancelled after first chunk");

        assert!(matches!(err, BackendError::Cancelled));
        match &caches[0] {
            QwenLayerCache::Linear(cache) => assert_eq!(cache.token_count(), 1),
            QwenLayerCache::Full(_) => panic!("layer 0 should be linear attention"),
        }
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn native_qwen_backend_opens_snapshot_without_engine_manifest() {
        let snapshot = temp_snapshot_dir("no-manifest");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        copy_fixture("config.json", snapshot.join("config.json"));
        copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));
        copy_fixture(
            "model.safetensors.index.json",
            snapshot.join("model.safetensors.index.json"),
        );

        let backend =
            NativeQwenBackend::open("local-qwen36", &snapshot).expect("backend opens snapshot");
        let metadata = backend.model_metadata();

        assert_eq!(metadata.id, "local-qwen36");
        assert_eq!(metadata.backend, "native-qwen");
        assert_eq!(metadata.snapshot_path.as_deref(), Some(snapshot.as_path()));
        assert!(metadata.manifest_digest.is_none());
        assert!(metadata.repo_id.is_none());
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn native_qwen_backend_can_eagerly_materialize_indexed_shards_on_open() {
        let snapshot = temp_snapshot_dir("eager-materialize");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        copy_fixture("config.json", snapshot.join("config.json"));
        copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));
        std::fs::write(
            snapshot.join("model.safetensors.index.json"),
            serde_json::json!({
                "metadata": { "total_size": 2 },
                "weight_map": { "dummy.weight": "dummy.safetensors" }
            })
            .to_string(),
        )
        .expect("index");
        std::fs::write(
            snapshot.join("dummy.safetensors"),
            tiny_safetensors_bf16("dummy.weight", &[1], &[1.0]),
        )
        .expect("dummy shard");

        let backend = NativeQwenBackend::open_with_options(
            "local-qwen36",
            &snapshot,
            NativeQwenLoadOptions {
                eager_materialize_shards: true,
            },
        )
        .expect("backend opens and materializes shards");

        assert_eq!(backend.store.materialized_shard_count(), 1);
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[tokio::test]
    async fn native_qwen_generate_with_cancel_observes_pre_cancelled_token() {
        let snapshot = temp_snapshot_dir("cancelled-generate");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        copy_fixture("config.json", snapshot.join("config.json"));
        copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));
        copy_fixture(
            "model.safetensors.index.json",
            snapshot.join("model.safetensors.index.json"),
        );
        let backend =
            NativeQwenBackend::open("local-qwen36", &snapshot).expect("backend opens snapshot");
        let cancellation = CancellationToken::new();
        cancellation.cancel();

        let err = backend
            .generate_with_cancel(
                BackendRequest {
                    model: "local-qwen36".to_owned(),
                    prompt: "say hi".to_owned(),
                    max_tokens: Some(1),
                    sampling: SamplingConfig::Greedy,
                    required_tool_choice: None,
                    json_object_mode: false,
                    conversation_mode: false,
                },
                cancellation,
            )
            .await
            .expect_err("pre-cancelled generation fails before decode");

        assert!(err.to_string().contains("cancelled"));
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn native_qwen_start_decode_session_observes_pre_cancelled_token() {
        let snapshot = temp_snapshot_dir("cancelled-start-decode");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        copy_fixture("config.json", snapshot.join("config.json"));
        copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));
        copy_fixture(
            "model.safetensors.index.json",
            snapshot.join("model.safetensors.index.json"),
        );
        let backend =
            NativeQwenBackend::open("local-qwen36", &snapshot).expect("backend opens snapshot");
        let cancellation = CancellationToken::new();
        cancellation.cancel();

        match backend.start_decode_session(&[0], 1, &cancellation) {
            Err(BackendError::Cancelled) => {}
            Err(err) => panic!("expected cancellation before prefill, got {err}"),
            Ok(_) => panic!("pre-cancelled decode startup should fail before prefill"),
        }
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn native_qwen_greedy_returns_top_logit_even_when_it_decodes_to_whitespace() {
        let snapshot = temp_snapshot_dir("greedy-whitespace");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));

        let norm_shape = [1_usize];
        let norm = [1.0_f32];
        let lm_head_shape = [221_usize, 1_usize];
        let mut lm_head = vec![0.0_f32; 221];
        lm_head[32] = 1.0;
        lm_head[220] = 2.0;
        let safetensors = tiny_multi_safetensors_bf16(&[
            (
                "model.language_model.norm.weight",
                &norm_shape,
                norm.as_slice(),
            ),
            ("lm_head.weight", &lm_head_shape, lm_head.as_slice()),
        ]);
        std::fs::write(snapshot.join("model.safetensors"), &safetensors)
            .expect("write greedy fixture shard");
        std::fs::write(
            snapshot.join("model.safetensors.index.json"),
            serde_json::json!({
                "metadata": { "total_size": safetensors.len() },
                "weight_map": {
                    "model.language_model.norm.weight": "model.safetensors",
                    "lm_head.weight": "model.safetensors"
                }
            })
            .to_string(),
        )
        .expect("write greedy fixture index");

        let backend = NativeQwenBackend {
            model_id: "local-qwen36".to_owned(),
            metadata: BackendModelMetadata::new("local-qwen36", "native-qwen"),
            tokenizer: HuggingFaceTokenizer::from_file(snapshot.join("tokenizer.json"))
                .expect("tokenizer loads"),
            spec: QwenModelSpec {
                family: llm_models::ModelFamily::Qwen,
                architecture: "Qwen3_5MoeForConditionalGeneration".to_owned(),
                model_type: "qwen3_5_moe".to_owned(),
                text_model_type: "qwen3_5_moe_text".to_owned(),
                hidden_size: 1,
                rms_norm_eps: 0.0,
                tie_word_embeddings: false,
                rope_theta: 1_000_000.0,
                partial_rotary_factor: 1.0,
                num_hidden_layers: 0,
                num_attention_heads: 1,
                num_key_value_heads: 1,
                head_dim: 1,
                linear_num_key_heads: 1,
                linear_num_value_heads: 1,
                linear_key_head_dim: 1,
                linear_value_head_dim: 1,
                linear_conv_kernel_dim: 1,
                num_experts: 1,
                num_experts_per_tok: 1,
                moe_intermediate_size: 1,
                shared_expert_intermediate_size: 1,
                max_position_embeddings: 1,
                vocab_size: 221,
                layer_kinds: Vec::new(),
            },
            store: SafeTensorShardStore::open(&snapshot).expect("store opens"),
            matvec: NativeQwenMatvecBackend::Cpu,
            max_new_tokens: 1,
            max_prefill_tokens: 1,
            top_k: 2,
            chunk_rows: 64,
        };

        let candidate = backend
            .next_token_from_hidden(&[1.0], SamplingConfig::Greedy)
            .expect("greedy candidate");

        assert_eq!(candidate.token_id, 220);
        let decoded = backend
            .tokenizer
            .decode(&[candidate.token_id as u32], false)
            .expect("candidate decodes");
        assert!(decoded.trim().is_empty());
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn native_top_p_sampling_selects_full_vocab_token_from_draw() {
        let token_id = sample_token_id_with_draw(
            &[2.0, 1.0, 0.0],
            SamplingConfig::TopP {
                temperature: 1.0,
                top_p: 0.9,
            },
            0.8,
        )
        .expect("sampling succeeds");

        assert_eq!(token_id, 1);
    }

    fn copy_fixture(name: &str, destination: impl AsRef<Path>) {
        let source = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/qwen36")
            .join(name);
        std::fs::copy(&source, destination).expect("copy fixture");
    }

    fn tiny_multi_safetensors_bf16(tensors: &[(&str, &[usize], &[f32])]) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape, values) in tensors {
            let start = data.len();
            for value in *values {
                data.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
            }
            let end = data.len();
            header.insert(
                (*name).to_owned(),
                serde_json::json!({
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

    fn tiny_safetensors_bf16(name: &str, shape: &[usize], values: &[f32]) -> Vec<u8> {
        let mut data = Vec::with_capacity(values.len() * 2);
        for value in values {
            data.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
        }
        let data_len = data.len();
        let header = serde_json::json!({
            name: {
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [0, data_len]
            }
        })
        .to_string();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&data);
        bytes
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
                serde_json::json!({
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

    fn write_tiny_linear_decoder_snapshot(root: &Path) {
        let tensors = vec![
            (
                "model.language_model.embed_tokens.weight",
                vec![2, 2],
                vec![1.0, 0.0, 0.0, 1.0],
            ),
            (
                "model.language_model.layers.0.input_layernorm.weight",
                vec![2],
                vec![0.0, 0.0],
            ),
            (
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                vec![4, 2],
                vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 4.0],
            ),
            (
                "model.language_model.layers.0.linear_attn.in_proj_z.weight",
                vec![2, 2],
                vec![1.0, 0.0, 0.0, 1.0],
            ),
            (
                "model.language_model.layers.0.linear_attn.in_proj_b.weight",
                vec![1, 2],
                vec![0.0, 0.0],
            ),
            (
                "model.language_model.layers.0.linear_attn.in_proj_a.weight",
                vec![1, 2],
                vec![0.0, 0.0],
            ),
            (
                "model.language_model.layers.0.linear_attn.dt_bias",
                vec![1],
                vec![0.0],
            ),
            (
                "model.language_model.layers.0.linear_attn.A_log",
                vec![1],
                vec![0.0],
            ),
            (
                "model.language_model.layers.0.linear_attn.conv1d.weight",
                vec![4, 1],
                vec![1.0, 1.0, 1.0, 1.0],
            ),
            (
                "model.language_model.layers.0.linear_attn.norm.weight",
                vec![2],
                vec![1.0, 1.0],
            ),
            (
                "model.language_model.layers.0.linear_attn.out_proj.weight",
                vec![2, 2],
                vec![1.0, 0.0, 0.0, 1.0],
            ),
            (
                "model.language_model.layers.0.post_attention_layernorm.weight",
                vec![2],
                vec![0.0, 0.0],
            ),
            (
                "model.language_model.layers.0.mlp.gate.weight",
                vec![1, 2],
                vec![0.0, 0.0],
            ),
            (
                "model.language_model.layers.0.mlp.experts.gate_up_proj",
                vec![2, 2],
                vec![0.0, 0.0, 0.0, 0.0],
            ),
            (
                "model.language_model.layers.0.mlp.experts.down_proj",
                vec![2, 1],
                vec![0.0, 0.0],
            ),
            (
                "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
                vec![1, 2],
                vec![0.0, 0.0],
            ),
            (
                "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
                vec![1, 2],
                vec![0.0, 0.0],
            ),
            (
                "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
                vec![2, 1],
                vec![0.0, 0.0],
            ),
            (
                "model.language_model.layers.0.mlp.shared_expert_gate.weight",
                vec![1, 2],
                vec![0.0, 0.0],
            ),
        ];
        let mut weight_map = serde_json::Map::new();
        for (name, _, _) in &tensors {
            weight_map.insert(
                (*name).to_owned(),
                serde_json::Value::String("model.safetensors".to_owned()),
            );
        }
        let safetensors = tiny_owned_multi_safetensors_bf16(&tensors);
        std::fs::write(snapshot_path(root, "model.safetensors"), &safetensors)
            .expect("write tiny decoder shard");
        std::fs::write(
            snapshot_path(root, "model.safetensors.index.json"),
            serde_json::json!({
                "metadata": { "total_size": safetensors.len() },
                "weight_map": serde_json::Value::Object(weight_map)
            })
            .to_string(),
        )
        .expect("write tiny decoder index");
    }

    fn snapshot_path(root: &Path, name: &str) -> PathBuf {
        root.join(name)
    }

    fn tiny_engine_qwen_spec(kind: llm_models::AttentionKind) -> QwenModelSpec {
        QwenModelSpec {
            family: llm_models::ModelFamily::Qwen,
            architecture: "Qwen3_5MoeForConditionalGeneration".to_owned(),
            model_type: "qwen3_5_moe".to_owned(),
            text_model_type: "qwen3_5_moe_text".to_owned(),
            hidden_size: 2,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: false,
            rope_theta: 1_000_000.0,
            partial_rotary_factor: 1.0,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 2,
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_key_head_dim: 1,
            linear_value_head_dim: 2,
            linear_conv_kernel_dim: 1,
            num_experts: 1,
            num_experts_per_tok: 1,
            moe_intermediate_size: 1,
            shared_expert_intermediate_size: 1,
            max_position_embeddings: 32,
            vocab_size: 2,
            layer_kinds: vec![kind],
        }
    }

    struct CancelAfterFirstConv {
        cancellation: CancellationToken,
        conv_calls: std::cell::Cell<usize>,
    }

    impl QwenMatvecBackend for CancelAfterFirstConv {
        fn linear_attention_conv1d_silu_f32(
            &self,
            window: &[f32],
            weights: &[f32],
            conv_dim: usize,
            kernel_size: usize,
        ) -> Result<Vec<f32>, MathError> {
            self.conv_calls.set(self.conv_calls.get() + 1);
            if self.conv_calls.get() == 1 {
                self.cancellation.cancel();
            }
            CpuQwenMatvecBackend.linear_attention_conv1d_silu_f32(
                window,
                weights,
                conv_dim,
                kernel_size,
            )
        }
    }

    fn temp_snapshot_dir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!("llm-engine-{label}-{}", std::process::id()))
    }
}

#[derive(Debug)]
enum EngineError {
    Runtime(RuntimeError),
    ModelStore(HubError),
    Overloaded(String),
    RequestNotFound(String),
    RequestConflict(String),
    InvalidRequestId(String),
    UnauthorizedAdmin,
}

impl From<RuntimeError> for EngineError {
    fn from(value: RuntimeError) -> Self {
        Self::Runtime(value)
    }
}

impl IntoResponse for EngineError {
    fn into_response(self) -> axum::response::Response {
        let (status, code, phase, retryable, message) = match self {
            Self::Runtime(err) => {
                let metadata = runtime_error_metadata(&err);
                (
                    metadata.status,
                    metadata.code,
                    metadata.phase,
                    metadata.retryable,
                    err.to_string(),
                )
            }
            Self::ModelStore(err) => (
                if err.code() == "model_not_found" {
                    StatusCode::NOT_FOUND
                } else {
                    StatusCode::UNPROCESSABLE_ENTITY
                },
                err.code(),
                "model_artifact_verification",
                false,
                err.to_string(),
            ),
            Self::Overloaded(message) => (
                StatusCode::TOO_MANY_REQUESTS,
                "model_overloaded",
                "scheduler",
                true,
                message,
            ),
            Self::RequestNotFound(request_id) => (
                StatusCode::NOT_FOUND,
                "request_not_found",
                "cancellation",
                false,
                format!("request `{request_id}` is not active"),
            ),
            Self::RequestConflict(request_id) => (
                StatusCode::CONFLICT,
                "request_id_conflict",
                "request_validation",
                false,
                format!("request id `{request_id}` is already active"),
            ),
            Self::InvalidRequestId(message) => (
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "request_validation",
                false,
                message,
            ),
            Self::UnauthorizedAdmin => (
                StatusCode::UNAUTHORIZED,
                "admin_auth_required",
                "admin_auth",
                false,
                "admin bearer token is required".to_owned(),
            ),
        };
        let body = Json(json!({
            "error": {
                "message": message,
                "code": code,
                "phase": phase,
                "retryable": retryable,
                "type": "llm_engine_error"
            }
        }));
        (status, body).into_response()
    }
}
