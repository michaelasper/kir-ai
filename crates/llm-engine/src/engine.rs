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
use futures::{Stream, StreamExt, stream::BoxStream};
use llm_api::{
    ApiError, ChatCompletionRequest, ChatCompletionStreamResponse, CompletionRequest,
    CompletionStreamResponse, FinishReason, ModelCard, ModelList, Usage, ValidateRequest,
};
use llm_backend::{
    BackendCacheContext, BackendError, BackendModelMetadata, BackendOutput, BackendRequest,
    BackendStreamChunk, CpuQwenMatvecBackend, DeterministicBackend, LayerKvCache,
    LinearAttentionCache, MathError, ModelBackend, QwenKvCacheTensor, QwenLayerCache,
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
    collections::{HashMap, HashSet, VecDeque},
    convert::Infallible,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

type EngineRuntime = Runtime<Box<dyn ModelBackend>>;

#[derive(Clone)]
struct AppState {
    runtime: Arc<EngineRuntime>,
    metrics: Arc<Mutex<ServerMetrics>>,
    generation_phases: Arc<GenerationPhaseMetrics>,
    model_scheduler: Arc<ModelScheduler>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SchedulerClass {
    Prefill,
    Decode,
}

impl SchedulerClass {
    fn as_phase(self) -> GenerationPhase {
        match self {
            Self::Prefill => GenerationPhase::Prefill,
            Self::Decode => GenerationPhase::Decode,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ModelSchedulerOptions {
    concurrency_limit: usize,
    queue_limit: usize,
    queue_timeout: Option<Duration>,
    prefill_threshold_chars: usize,
    prefill_burst: usize,
}

#[derive(Debug)]
struct ModelScheduler {
    options: ModelSchedulerOptions,
    state: Mutex<ModelSchedulerState>,
    notify: Notify,
}

#[derive(Debug, Default)]
struct ModelSchedulerState {
    next_ticket: u64,
    queued_prefill: VecDeque<u64>,
    queued_decode: VecDeque<u64>,
    active_prefill: usize,
    active_decode: usize,
    prefill_admissions_since_decode: usize,
    admitted_prefill: u64,
    admitted_decode: u64,
    completed: u64,
    cancelled: u64,
    failed: u64,
    queued_cancelled: u64,
    queue_timeouts: u64,
}

impl ModelSchedulerState {
    fn active_total(&self) -> usize {
        self.active_prefill + self.active_decode
    }

    fn queued_total(&self) -> usize {
        self.queued_prefill.len() + self.queued_decode.len()
    }

    fn queue(&self, class: SchedulerClass) -> &VecDeque<u64> {
        match class {
            SchedulerClass::Prefill => &self.queued_prefill,
            SchedulerClass::Decode => &self.queued_decode,
        }
    }

    fn queue_mut(&mut self, class: SchedulerClass) -> &mut VecDeque<u64> {
        match class {
            SchedulerClass::Prefill => &mut self.queued_prefill,
            SchedulerClass::Decode => &mut self.queued_decode,
        }
    }

    fn start_active(&mut self, admission_class: SchedulerClass, initial_phase: GenerationPhase) {
        match initial_phase {
            GenerationPhase::Prefill => self.active_prefill += 1,
            GenerationPhase::Decode => self.active_decode += 1,
        }
        match admission_class {
            SchedulerClass::Prefill => {
                self.prefill_admissions_since_decode += 1;
                self.admitted_prefill += 1;
            }
            SchedulerClass::Decode => {
                self.prefill_admissions_since_decode = 0;
                self.admitted_decode += 1;
            }
        }
    }

    fn finish_active(&mut self, phase: GenerationPhase, outcome: SchedulerOutcome) {
        match phase {
            GenerationPhase::Prefill => self.active_prefill = self.active_prefill.saturating_sub(1),
            GenerationPhase::Decode => self.active_decode = self.active_decode.saturating_sub(1),
        }
        match outcome {
            SchedulerOutcome::Completed => self.completed += 1,
            SchedulerOutcome::Cancelled => self.cancelled += 1,
            SchedulerOutcome::Failed => self.failed += 1,
        }
    }

    fn transition_to_decode(&mut self) {
        self.active_prefill = self.active_prefill.saturating_sub(1);
        self.active_decode += 1;
    }

    fn next_admissible_class(&self, prefill_burst: usize) -> Option<SchedulerClass> {
        let has_prefill = !self.queued_prefill.is_empty();
        let has_decode = !self.queued_decode.is_empty();
        match (has_prefill, has_decode) {
            (false, false) => None,
            (true, false) => Some(SchedulerClass::Prefill),
            (false, true) => Some(SchedulerClass::Decode),
            (true, true) => {
                if self.prefill_admissions_since_decode >= prefill_burst {
                    return Some(SchedulerClass::Decode);
                }
                let prefill_ticket = self.queued_prefill.front().copied().unwrap_or(u64::MAX);
                let decode_ticket = self.queued_decode.front().copied().unwrap_or(u64::MAX);
                if decode_ticket < prefill_ticket {
                    Some(SchedulerClass::Decode)
                } else {
                    Some(SchedulerClass::Prefill)
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SchedulerOutcome {
    Completed,
    Cancelled,
    Failed,
}

#[derive(Debug)]
struct SchedulerPermit {
    scheduler: Arc<ModelScheduler>,
    phase: GenerationPhase,
    outcome: SchedulerOutcome,
}

impl SchedulerPermit {
    fn transition_to_decode(&mut self) {
        if self.phase == GenerationPhase::Decode {
            return;
        }
        self.scheduler
            .state
            .lock()
            .expect("scheduler lock is not poisoned")
            .transition_to_decode();
        self.phase = GenerationPhase::Decode;
    }

    fn mark_failed(&mut self) {
        self.outcome = SchedulerOutcome::Failed;
    }

    fn mark_cancelled(&mut self) {
        self.outcome = SchedulerOutcome::Cancelled;
    }
}

impl Drop for SchedulerPermit {
    fn drop(&mut self) {
        self.scheduler
            .state
            .lock()
            .expect("scheduler lock is not poisoned")
            .finish_active(self.phase, self.outcome);
        self.scheduler.notify.notify_waiters();
    }
}

#[derive(Debug)]
struct QueuedSchedulerTicket {
    scheduler: Arc<ModelScheduler>,
    id: u64,
    class: SchedulerClass,
    admitted: bool,
    timeout: bool,
}

impl QueuedSchedulerTicket {
    fn admitted(&mut self) {
        self.admitted = true;
    }

    fn timed_out(&mut self) {
        self.timeout = true;
    }
}

impl Drop for QueuedSchedulerTicket {
    fn drop(&mut self) {
        if self.admitted {
            return;
        }
        let mut state = self
            .scheduler
            .state
            .lock()
            .expect("scheduler lock is not poisoned");
        let queue = state.queue_mut(self.class);
        if let Some(index) = queue.iter().position(|ticket| *ticket == self.id) {
            queue.remove(index);
            if self.timeout {
                state.queue_timeouts += 1;
            } else {
                state.queued_cancelled += 1;
            }
        }
        drop(state);
        self.scheduler.notify.notify_waiters();
    }
}

#[derive(Debug, Clone, Copy)]
struct ModelSchedulerSnapshot {
    queued_prefill: usize,
    queued_decode: usize,
    active_prefill: usize,
    active_decode: usize,
    admitted_prefill: u64,
    admitted_decode: u64,
    completed: u64,
    cancelled: u64,
    failed: u64,
    queued_cancelled: u64,
    queue_timeouts: u64,
}

impl ModelSchedulerSnapshot {
    fn queued_total(&self) -> usize {
        self.queued_prefill + self.queued_decode
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SchedulerAcquireError {
    QueueFull,
    QueueTimedOut,
}

impl ModelScheduler {
    fn new(options: ModelSchedulerOptions) -> Self {
        Self {
            options,
            state: Mutex::new(ModelSchedulerState::default()),
            notify: Notify::new(),
        }
    }

    fn classify_chat(&self, request: &ChatCompletionRequest) -> SchedulerClass {
        let chars = request
            .messages
            .iter()
            .map(|message| message.content.as_ref().map_or(0, String::len))
            .sum::<usize>()
            + request
                .tools
                .iter()
                .filter_map(|tool| serde_json::to_string(tool).ok())
                .map(|tool| tool.len())
                .sum::<usize>();
        self.classify_chars(chars)
    }

    fn classify_completion(&self, request: &CompletionRequest) -> SchedulerClass {
        self.classify_chars(request.prompt.len())
    }

    fn classify_chars(&self, chars: usize) -> SchedulerClass {
        if chars >= self.options.prefill_threshold_chars {
            SchedulerClass::Prefill
        } else {
            SchedulerClass::Decode
        }
    }

    async fn acquire(
        self: &Arc<Self>,
        admission_class: SchedulerClass,
        initial_phase: GenerationPhase,
    ) -> Result<SchedulerPermit, SchedulerAcquireError> {
        if let Some(permit) = self.try_acquire_immediate(admission_class, initial_phase) {
            return Ok(permit);
        }
        let mut ticket = self.enqueue(admission_class)?;
        let deadline = self
            .options
            .queue_timeout
            .map(|timeout| tokio::time::Instant::now() + timeout);
        loop {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            if let Some(permit) = self.try_admit_queued(ticket.id, admission_class, initial_phase) {
                ticket.admitted();
                return Ok(permit);
            }
            if let Some(deadline) = deadline {
                tokio::select! {
                    () = &mut notified => {}
                    () = tokio::time::sleep_until(deadline) => {
                        ticket.timed_out();
                        return Err(SchedulerAcquireError::QueueTimedOut);
                    }
                }
            } else {
                notified.await;
            }
        }
    }

    fn try_acquire_immediate(
        self: &Arc<Self>,
        admission_class: SchedulerClass,
        initial_phase: GenerationPhase,
    ) -> Option<SchedulerPermit> {
        let mut state = self.state.lock().expect("scheduler lock is not poisoned");
        if state.active_total() >= self.options.concurrency_limit || state.queued_total() > 0 {
            return None;
        }
        state.start_active(admission_class, initial_phase);
        Some(SchedulerPermit {
            scheduler: Arc::clone(self),
            phase: initial_phase,
            outcome: SchedulerOutcome::Completed,
        })
    }

    fn enqueue(
        self: &Arc<Self>,
        class: SchedulerClass,
    ) -> Result<QueuedSchedulerTicket, SchedulerAcquireError> {
        let mut state = self.state.lock().expect("scheduler lock is not poisoned");
        if state.queued_total() >= self.options.queue_limit {
            return Err(SchedulerAcquireError::QueueFull);
        }
        state.next_ticket += 1;
        let id = state.next_ticket;
        state.queue_mut(class).push_back(id);
        drop(state);
        self.notify.notify_waiters();
        Ok(QueuedSchedulerTicket {
            scheduler: Arc::clone(self),
            id,
            class,
            admitted: false,
            timeout: false,
        })
    }

    fn try_admit_queued(
        self: &Arc<Self>,
        ticket: u64,
        admission_class: SchedulerClass,
        initial_phase: GenerationPhase,
    ) -> Option<SchedulerPermit> {
        let mut state = self.state.lock().expect("scheduler lock is not poisoned");
        if state.active_total() >= self.options.concurrency_limit {
            return None;
        }
        if state.next_admissible_class(self.options.prefill_burst)? != admission_class {
            return None;
        }
        if state.queue(admission_class).front().copied() != Some(ticket) {
            return None;
        }
        state.queue_mut(admission_class).pop_front();
        state.start_active(admission_class, initial_phase);
        Some(SchedulerPermit {
            scheduler: Arc::clone(self),
            phase: initial_phase,
            outcome: SchedulerOutcome::Completed,
        })
    }

    fn snapshot(&self) -> ModelSchedulerSnapshot {
        let state = self.state.lock().expect("scheduler lock is not poisoned");
        ModelSchedulerSnapshot {
            queued_prefill: state.queued_prefill.len(),
            queued_decode: state.queued_decode.len(),
            active_prefill: state.active_prefill,
            active_decode: state.active_decode,
            admitted_prefill: state.admitted_prefill,
            admitted_decode: state.admitted_decode,
            completed: state.completed,
            cancelled: state.cancelled,
            failed: state.failed,
            queued_cancelled: state.queued_cancelled,
            queue_timeouts: state.queue_timeouts,
        }
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
    pub scheduler_queue_limit: usize,
    pub scheduler_queue_timeout: Option<Duration>,
    pub scheduler_prefill_threshold_chars: usize,
    pub scheduler_prefill_burst: usize,
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
            scheduler_queue_limit: 1,
            scheduler_queue_timeout: Some(Duration::from_secs(30)),
            scheduler_prefill_threshold_chars: 4096,
            scheduler_prefill_burst: 1,
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
        .map(|endpoint| parse_hub_client(endpoint, options.hf_token.as_deref()))
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
            model_scheduler: Arc::new(ModelScheduler::new(ModelSchedulerOptions {
                concurrency_limit: options.concurrency_limit.max(1),
                queue_limit: options.scheduler_queue_limit,
                queue_timeout: options.scheduler_queue_timeout,
                prefill_threshold_chars: options.scheduler_prefill_threshold_chars,
                prefill_burst: options.scheduler_prefill_burst.max(1),
            })),
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

    fn insecure_hub_token_endpoint(endpoint: &url::Url) -> Self {
        Self {
            message: format!(
                "refusing to send HF_TOKEN to non-HTTPS hub endpoint `{endpoint}`; use HTTPS or a loopback endpoint for local development"
            ),
        }
    }
}

impl std::fmt::Display for EngineConfigError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for EngineConfigError {}

fn parse_hub_client(
    endpoint: &str,
    hf_token: Option<&str>,
) -> Result<HubClient, EngineConfigError> {
    let endpoint = url::Url::parse(endpoint)
        .map_err(|err| EngineConfigError::invalid_hub_endpoint(endpoint, err))?;
    if hf_token.is_some() && endpoint.scheme() != "https" && !is_loopback_endpoint(&endpoint) {
        return Err(EngineConfigError::insecure_hub_token_endpoint(&endpoint));
    }
    Ok(HubClient::new(endpoint))
}

fn is_loopback_endpoint(endpoint: &url::Url) -> bool {
    match endpoint.host() {
        Some(url::Host::Domain(domain)) => domain.eq_ignore_ascii_case("localhost"),
        Some(url::Host::Ipv4(addr)) => addr.is_loopback(),
        Some(url::Host::Ipv6(addr)) => addr.is_loopback(),
        None => false,
    }
}

fn default_backend() -> DeterministicBackend {
    DeterministicBackend::new("local-qwen36", "hello from rust native backend")
        .with_required_tool_protocol()
        .with_json_object_protocol()
        .with_adaptive_chat_protocol()
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
    prefix_cache: Arc<NativeQwenPrefixCache>,
}

#[derive(Clone)]
enum NativeQwenMatvecBackend {
    Cpu,
    Metal(Arc<NativeQwenMetalState>),
}

const DEFAULT_NATIVE_QWEN_METAL_WEIGHT_CACHE_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const DEFAULT_NATIVE_QWEN_PREFIX_CACHE_BYTES: u64 = 512 * 1024 * 1024;
const NATIVE_QWEN_PREFIX_CACHE_LAYOUT_VERSION: u32 = 1;

#[derive(Debug)]
struct NativeQwenPrefixCache {
    max_bytes: u64,
    inner: Mutex<NativeQwenPrefixCacheInner>,
}

#[derive(Debug, Default)]
struct NativeQwenPrefixCacheInner {
    entries: HashMap<NativeQwenPrefixCacheKey, NativeQwenPrefixCacheEntry>,
    used_bytes: u64,
    next_access: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct NativeQwenPrefixCacheNamespace {
    model_id: String,
    backend: String,
    family: Option<String>,
    loader: Option<String>,
    quantization: Option<String>,
    repo_id: Option<String>,
    resolved_commit: Option<String>,
    profile: Option<String>,
    manifest_digest: Option<String>,
    prompt_template: String,
    tool_schema: Option<String>,
    request_mode: String,
    sampling: String,
    cache_layout_version: u32,
    cache_tokens: usize,
    max_prefill_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct NativeQwenPrefixCacheKey {
    namespace: NativeQwenPrefixCacheNamespace,
    tokens: Vec<usize>,
}

#[derive(Debug, Clone)]
struct NativeQwenPrefixCacheEntry {
    hidden: Vec<f32>,
    caches: Vec<QwenLayerCache>,
    byte_len: u64,
    last_used: u64,
}

#[derive(Debug)]
struct NativeQwenPrefixCacheHit {
    token_count: usize,
    hidden: Vec<f32>,
    caches: Vec<QwenLayerCache>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct NativeQwenPrefixCacheCounters {
    hits: u64,
    misses: u64,
    stores: u64,
    evictions: u64,
    rejected: u64,
    reused_tokens: u64,
    bytes_stored: u64,
    bytes_evicted: u64,
    resident_bytes: u64,
    resident_entries: u64,
}

#[derive(Debug, Default)]
struct NativeQwenPrefixCacheMetrics {
    counters: Mutex<NativeQwenPrefixCacheCounters>,
}

impl NativeQwenPrefixCache {
    fn new(max_bytes: u64) -> Self {
        Self {
            max_bytes,
            inner: Mutex::new(NativeQwenPrefixCacheInner::default()),
        }
    }

    fn lookup(
        &self,
        namespace: &NativeQwenPrefixCacheNamespace,
        tokens: &[usize],
    ) -> Option<NativeQwenPrefixCacheHit> {
        let mut inner = self
            .inner
            .lock()
            .expect("native Qwen prefix cache lock is not poisoned");
        let mut best_key = None;
        let mut best_len = 0;
        for key in inner.entries.keys() {
            if key.namespace == *namespace
                && key.tokens.len() > best_len
                && tokens.starts_with(&key.tokens)
            {
                best_len = key.tokens.len();
                best_key = Some(key.clone());
            }
        }
        let Some(best_key) = best_key else {
            native_qwen_prefix_cache_metrics().record_miss();
            return None;
        };
        let access = inner.next_access();
        let entry = inner
            .entries
            .get_mut(&best_key)
            .expect("best prefix key came from cache entries");
        entry.last_used = access;
        native_qwen_prefix_cache_metrics().record_hit(best_len as u64);
        Some(NativeQwenPrefixCacheHit {
            token_count: best_len,
            hidden: entry.hidden.clone(),
            caches: entry.caches.clone(),
        })
    }

    fn store(
        &self,
        namespace: NativeQwenPrefixCacheNamespace,
        tokens: &[usize],
        hidden: &[f32],
        caches: &[QwenLayerCache],
    ) {
        if tokens.is_empty() {
            return;
        }
        let byte_len = native_qwen_prefix_entry_bytes(hidden, caches);
        if byte_len > self.max_bytes {
            native_qwen_prefix_cache_metrics().record_rejected();
            return;
        }
        let key = NativeQwenPrefixCacheKey {
            namespace,
            tokens: tokens.to_vec(),
        };
        let mut inner = self
            .inner
            .lock()
            .expect("native Qwen prefix cache lock is not poisoned");
        if let Some(existing) = inner.entries.remove(&key) {
            inner.used_bytes = inner.used_bytes.saturating_sub(existing.byte_len);
        }
        while inner.used_bytes.saturating_add(byte_len) > self.max_bytes {
            let Some(lru_key) = inner
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            let Some(evicted) = inner.entries.remove(&lru_key) else {
                break;
            };
            inner.used_bytes = inner.used_bytes.saturating_sub(evicted.byte_len);
            native_qwen_prefix_cache_metrics().record_eviction(evicted.byte_len);
        }
        let access = inner.next_access();
        inner.entries.insert(
            key,
            NativeQwenPrefixCacheEntry {
                hidden: hidden.to_vec(),
                caches: caches.to_vec(),
                byte_len,
                last_used: access,
            },
        );
        inner.used_bytes = inner.used_bytes.saturating_add(byte_len);
        native_qwen_prefix_cache_metrics().record_store(byte_len);
        native_qwen_prefix_cache_metrics()
            .record_residency(inner.used_bytes, inner.entries.len() as u64);
    }
}

impl NativeQwenPrefixCacheInner {
    fn next_access(&mut self) -> u64 {
        let access = self.next_access;
        self.next_access = self.next_access.saturating_add(1);
        access
    }
}

impl NativeQwenPrefixCacheMetrics {
    fn record_hit(&self, tokens: u64) {
        self.update(|counters| {
            counters.hits += 1;
            counters.reused_tokens += tokens;
        });
    }

    fn record_miss(&self) {
        self.update(|counters| counters.misses += 1);
    }

    fn record_store(&self, byte_len: u64) {
        self.update(|counters| {
            counters.stores += 1;
            counters.bytes_stored += byte_len;
        });
    }

    fn record_eviction(&self, byte_len: u64) {
        self.update(|counters| {
            counters.evictions += 1;
            counters.bytes_evicted += byte_len;
        });
    }

    fn record_rejected(&self) {
        self.update(|counters| counters.rejected += 1);
    }

    fn record_residency(&self, bytes: u64, entries: u64) {
        self.update(|counters| {
            counters.resident_bytes = bytes;
            counters.resident_entries = entries;
        });
    }

    fn snapshot(&self) -> Value {
        let counters = *self
            .counters
            .lock()
            .expect("native Qwen prefix cache metrics lock is not poisoned");
        json!({
            "hits": counters.hits,
            "misses": counters.misses,
            "stores": counters.stores,
            "evictions": counters.evictions,
            "rejected": counters.rejected,
            "reused_tokens": counters.reused_tokens,
            "bytes_stored": counters.bytes_stored,
            "bytes_evicted": counters.bytes_evicted,
            "resident_bytes": counters.resident_bytes,
            "resident_entries": counters.resident_entries,
        })
    }

    fn update(&self, update: impl FnOnce(&mut NativeQwenPrefixCacheCounters)) {
        let mut counters = self
            .counters
            .lock()
            .expect("native Qwen prefix cache metrics lock is not poisoned");
        update(&mut counters);
    }
}

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

fn native_qwen_prefix_namespace(
    backend: &NativeQwenBackend,
    request: &BackendRequest,
    cache_tokens: usize,
) -> NativeQwenPrefixCacheNamespace {
    NativeQwenPrefixCacheNamespace {
        model_id: backend.model_id.clone(),
        backend: backend.metadata.backend.clone(),
        family: backend.metadata.family.clone(),
        loader: backend.metadata.loader.clone(),
        quantization: backend.metadata.quantization.clone(),
        repo_id: backend.metadata.repo_id.clone(),
        resolved_commit: backend.metadata.resolved_commit.clone(),
        profile: backend.metadata.profile.clone(),
        manifest_digest: backend.metadata.manifest_digest.clone(),
        prompt_template: backend_request_cache_prompt_template(request),
        tool_schema: request.cache_context.tool_schema.clone(),
        request_mode: native_qwen_prefix_request_mode(request),
        sampling: native_qwen_prefix_sampling_key(request.sampling),
        cache_layout_version: NATIVE_QWEN_PREFIX_CACHE_LAYOUT_VERSION,
        cache_tokens,
        max_prefill_tokens: backend.max_prefill_tokens,
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

fn native_qwen_prefix_sampling_key(sampling: SamplingConfig) -> String {
    match sampling {
        SamplingConfig::Greedy => "greedy".to_owned(),
        SamplingConfig::TopP { temperature, top_p } => {
            format!(
                "top_p:{:08x}:{:08x}",
                temperature.to_bits(),
                top_p.to_bits()
            )
        }
    }
}

struct NativeQwenMetalState {
    device: llm_metal::MetalDevice,
    bf16_matrices: Mutex<Bf16MatrixBufferCache<Arc<llm_metal::Bf16MatrixBuffer>>>,
    kv_caches: Mutex<HashMap<u64, MetalLayerKvCacheMirror>>,
    linear_caches: Mutex<HashMap<u64, MetalLinearAttentionCacheMirror>>,
}

#[derive(Debug)]
struct MetalLayerKvCacheMirror {
    keys: llm_metal::F32Buffer,
    values: llm_metal::F32Buffer,
    revision: u64,
}

#[derive(Debug)]
struct MetalLinearAttentionCacheMirror {
    recurrent_state: llm_metal::F32Buffer,
    revision: u64,
}

type NativeQwenMetalStateRegistry =
    Mutex<HashMap<NativeQwenMetalStateKey, Arc<NativeQwenMetalState>>>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct NativeQwenMetalStateKey {
    cache_namespace: String,
    weight_cache_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Bf16MatrixCacheKey {
    tensor: String,
    element_offset: usize,
    rows: usize,
    columns: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WarmableBf16MatrixTensor {
    name: String,
    rows: usize,
    columns: usize,
    byte_len: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct NativeQwenWeightWarmOrder {
    stage: u8,
    layer: usize,
    item: u8,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct NativeQwenMetalWarmup {
    candidates: u64,
    warmed: u64,
    already_resident: u64,
    skipped_budget: u64,
    skipped_non_metal: u64,
}

#[derive(Debug)]
struct Bf16MatrixBufferCache<T> {
    max_bytes: u64,
    used_bytes: u64,
    next_access: u64,
    entries: HashMap<Bf16MatrixCacheKey, CachedBf16MatrixBuffer<T>>,
}

#[derive(Debug)]
struct CachedBf16MatrixBuffer<T> {
    value: T,
    byte_len: u64,
    last_used: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Bf16MatrixBufferCacheInsert {
    inserted: bool,
    evicted_count: u64,
    evicted_bytes: u64,
}

impl<T: Clone> Bf16MatrixBufferCache<T> {
    fn new(max_bytes: u64) -> Self {
        Self {
            max_bytes,
            used_bytes: 0,
            next_access: 0,
            entries: HashMap::new(),
        }
    }

    fn get(&mut self, key: &Bf16MatrixCacheKey) -> Option<T> {
        let access = self.next_access();
        self.entries.get_mut(key).map(|entry| {
            entry.last_used = access;
            entry.value.clone()
        })
    }

    fn insert(
        &mut self,
        key: Bf16MatrixCacheKey,
        value: T,
        byte_len: u64,
    ) -> Bf16MatrixBufferCacheInsert {
        if byte_len > self.max_bytes {
            return Bf16MatrixBufferCacheInsert::default();
        }
        if let Some(existing) = self.entries.remove(&key) {
            self.used_bytes = self.used_bytes.saturating_sub(existing.byte_len);
        }
        let mut result = Bf16MatrixBufferCacheInsert::default();
        while self.used_bytes.saturating_add(byte_len) > self.max_bytes {
            let Some(lru_key) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            let Some(evicted) = self.entries.remove(&lru_key) else {
                break;
            };
            self.used_bytes = self.used_bytes.saturating_sub(evicted.byte_len);
            result.evicted_count += 1;
            result.evicted_bytes += evicted.byte_len;
        }
        let access = self.next_access();
        self.entries.insert(
            key,
            CachedBf16MatrixBuffer {
                value,
                byte_len,
                last_used: access,
            },
        );
        self.used_bytes = self.used_bytes.saturating_add(byte_len);
        result.inserted = true;
        result
    }

    #[cfg(test)]
    fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    fn resident_bytes(&self) -> u64 {
        self.used_bytes
    }

    fn resident_buffers(&self) -> u64 {
        self.entries.len() as u64
    }

    fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    fn can_insert_without_eviction(&self, byte_len: u64) -> bool {
        byte_len <= self.max_bytes && self.used_bytes.saturating_add(byte_len) <= self.max_bytes
    }

    fn next_access(&mut self) -> u64 {
        let access = self.next_access;
        self.next_access = self.next_access.saturating_add(1);
        access
    }
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
    fn new(device: llm_metal::MetalDevice, weight_cache_bytes: u64) -> Self {
        native_qwen_metal_metrics().record_bf16_matrix_cache_residency(0, 0, weight_cache_bytes);
        Self {
            device,
            bf16_matrices: Mutex::new(Bf16MatrixBufferCache::new(weight_cache_bytes)),
            kv_caches: Mutex::new(HashMap::new()),
            linear_caches: Mutex::new(HashMap::new()),
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
            return Ok(existing);
        }
        let byte_len = buffer.byte_len() as u64;
        let insert = matrices.insert(key, Arc::clone(&buffer), byte_len);
        let metrics = native_qwen_metal_metrics();
        metrics.record_bf16_matrix_cache_upload(byte_len);
        if insert.evicted_count > 0 {
            metrics.record_bf16_matrix_cache_eviction(insert.evicted_count, insert.evicted_bytes);
        }
        metrics.record_bf16_matrix_cache_residency(
            matrices.resident_bytes(),
            matrices.resident_buffers(),
            matrices.max_bytes(),
        );
        Ok(buffer)
    }

    fn warm_bf16_matrix_cache(
        &self,
        store: &SafeTensorShardStore,
    ) -> Result<NativeQwenMetalWarmup, NativeQwenMetalBufferError> {
        let tensors = native_qwen_warmable_bf16_matrix_tensors(store)
            .map_err(NativeQwenMetalBufferError::Tensor)?;
        let mut warmup = NativeQwenMetalWarmup {
            candidates: tensors.len() as u64,
            ..NativeQwenMetalWarmup::default()
        };
        for tensor in tensors {
            let key = Bf16MatrixCacheKey {
                tensor: tensor.name.clone(),
                element_offset: 0,
                rows: tensor.rows,
                columns: tensor.columns,
            };
            {
                let mut matrices = self
                    .bf16_matrices
                    .lock()
                    .expect("BF16 matrix buffer cache lock is not poisoned");
                if matrices.get(&key).is_some() {
                    warmup.already_resident += 1;
                    continue;
                }
                if !matrices.can_insert_without_eviction(tensor.byte_len) {
                    warmup.skipped_budget += 1;
                    continue;
                }
            }
            self.bf16_matrix_buffer(store, &tensor.name, 0, tensor.rows, tensor.columns)?;
            warmup.warmed += 1;
        }
        Ok(warmup)
    }

    fn sync_kv_cache(&self, cache: &LayerKvCache) -> Result<(), llm_metal::MetalError> {
        let byte_len =
            cache_resident_byte_len(cache.key_storage().len() + cache.value_storage().len())?;
        let mut caches = self
            .kv_caches
            .lock()
            .expect("Metal KV cache mirror lock is not poisoned");
        match caches.get_mut(&cache.id()) {
            Some(mirror) if mirror.revision == cache.revision() => Ok(()),
            Some(mirror) => {
                self.device
                    .write_f32_buffer(&mirror.keys, cache.key_storage())?;
                self.device
                    .write_f32_buffer(&mirror.values, cache.value_storage())?;
                mirror.revision = cache.revision();
                native_qwen_metal_metrics().record_kv_cache_sync(byte_len);
                Ok(())
            }
            None => {
                let keys = self.device.new_f32_buffer(cache.key_storage())?;
                let values = self.device.new_f32_buffer(cache.value_storage())?;
                caches.insert(
                    cache.id(),
                    MetalLayerKvCacheMirror {
                        keys,
                        values,
                        revision: cache.revision(),
                    },
                );
                native_qwen_metal_metrics().record_kv_cache_allocation(byte_len);
                self.record_kv_cache_residency_locked(&caches);
                Ok(())
            }
        }
    }

    fn select_kv_cache_head_rows(
        &self,
        cache: &LayerKvCache,
        tensor: QwenKvCacheTensor,
        row_count: usize,
        head_start: usize,
        head_len: usize,
    ) -> Result<Vec<f32>, llm_metal::MetalError> {
        self.sync_kv_cache(cache)?;
        let caches = self
            .kv_caches
            .lock()
            .expect("Metal KV cache mirror lock is not poisoned");
        let mirror = caches.get(&cache.id()).ok_or_else(|| {
            llm_metal::MetalError::InvalidShape(format!(
                "missing Metal KV cache mirror for cache {}",
                cache.id()
            ))
        })?;
        let values = match tensor {
            QwenKvCacheTensor::Key => &mirror.keys,
            QwenKvCacheTensor::Value => &mirror.values,
        };
        self.device.select_head_rows_f32_buffered(
            values,
            row_count,
            cache.vector_len(),
            head_start,
            head_len,
        )
    }

    fn sync_linear_cache(&self, cache: &LinearAttentionCache) -> Result<(), llm_metal::MetalError> {
        let byte_len = cache_resident_byte_len(cache.recurrent_state().len())?;
        let mut caches = self
            .linear_caches
            .lock()
            .expect("Metal linear attention cache mirror lock is not poisoned");
        match caches.get_mut(&cache.id()) {
            Some(mirror) if mirror.revision == cache.revision() => Ok(()),
            Some(mirror) => {
                self.device
                    .write_f32_buffer(&mirror.recurrent_state, cache.recurrent_state())?;
                mirror.revision = cache.revision();
                native_qwen_metal_metrics().record_linear_cache_sync(byte_len);
                Ok(())
            }
            None => {
                let recurrent_state = self.device.new_f32_buffer(cache.recurrent_state())?;
                caches.insert(
                    cache.id(),
                    MetalLinearAttentionCacheMirror {
                        recurrent_state,
                        revision: cache.revision(),
                    },
                );
                native_qwen_metal_metrics().record_linear_cache_allocation(byte_len);
                self.record_linear_cache_residency_locked(&caches);
                Ok(())
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_attention_recurrent_cache_update(
        &self,
        cache: &LinearAttentionCache,
        state_start: usize,
        key: &[f32],
        value: &[f32],
        memory: &[f32],
        beta: f32,
        decay: f32,
        key_head_dim: usize,
        value_head_dim: usize,
    ) -> Result<Vec<f32>, llm_metal::MetalError> {
        self.sync_linear_cache(cache)?;
        let mut caches = self
            .linear_caches
            .lock()
            .expect("Metal linear attention cache mirror lock is not poisoned");
        let mirror = caches.get_mut(&cache.id()).ok_or_else(|| {
            llm_metal::MetalError::InvalidShape(format!(
                "missing Metal linear attention cache mirror for cache {}",
                cache.id()
            ))
        })?;
        let updated = self
            .device
            .linear_attention_recurrent_update_f32_buffered_state(
                &mirror.recurrent_state,
                state_start,
                key,
                value,
                memory,
                beta,
                decay,
                key_head_dim,
                value_head_dim,
            )?;
        mirror.revision = cache.revision().saturating_add(1);
        Ok(updated)
    }

    fn remove_cache_mirrors(&self, caches: &[QwenLayerCache]) {
        let mut kv_removed = Vec::new();
        let mut linear_removed = Vec::new();
        for cache in caches {
            match cache {
                QwenLayerCache::Full(cache) => kv_removed.push(cache.id()),
                QwenLayerCache::Linear(cache) => linear_removed.push(cache.id()),
            }
        }
        if !kv_removed.is_empty() {
            let mut mirrors = self
                .kv_caches
                .lock()
                .expect("Metal KV cache mirror lock is not poisoned");
            let mut bytes = 0_u64;
            let mut count = 0_u64;
            for id in kv_removed {
                if let Some(mirror) = mirrors.remove(&id) {
                    bytes = bytes
                        .saturating_add((mirror.keys.byte_len() + mirror.values.byte_len()) as u64);
                    count += 2;
                }
            }
            if count > 0 {
                native_qwen_metal_metrics().record_kv_cache_eviction(count, bytes);
                self.record_kv_cache_residency_locked(&mirrors);
            }
        }
        if !linear_removed.is_empty() {
            let mut mirrors = self
                .linear_caches
                .lock()
                .expect("Metal linear attention cache mirror lock is not poisoned");
            let mut bytes = 0_u64;
            let mut count = 0_u64;
            for id in linear_removed {
                if let Some(mirror) = mirrors.remove(&id) {
                    bytes = bytes.saturating_add(mirror.recurrent_state.byte_len() as u64);
                    count += 1;
                }
            }
            if count > 0 {
                native_qwen_metal_metrics().record_linear_cache_eviction(count, bytes);
                self.record_linear_cache_residency_locked(&mirrors);
            }
        }
    }

    fn record_kv_cache_residency_locked(&self, caches: &HashMap<u64, MetalLayerKvCacheMirror>) {
        let resident_bytes = caches
            .values()
            .map(|mirror| mirror.keys.byte_len() as u64 + mirror.values.byte_len() as u64)
            .sum();
        native_qwen_metal_metrics()
            .record_kv_cache_residency(resident_bytes, caches.len() as u64 * 2);
    }

    fn record_linear_cache_residency_locked(
        &self,
        caches: &HashMap<u64, MetalLinearAttentionCacheMirror>,
    ) {
        let resident_bytes = caches
            .values()
            .map(|mirror| mirror.recurrent_state.byte_len() as u64)
            .sum();
        native_qwen_metal_metrics()
            .record_linear_cache_residency(resident_bytes, caches.len() as u64);
    }
}

fn cache_resident_byte_len(elements: usize) -> Result<u64, llm_metal::MetalError> {
    elements
        .checked_mul(std::mem::size_of::<f32>())
        .map(|bytes| bytes as u64)
        .ok_or_else(|| {
            llm_metal::MetalError::InvalidShape(
                "Metal resident cache byte length overflows usize".to_owned(),
            )
        })
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
    evictions: u64,
    bytes_evicted: u64,
    resident_bytes: u64,
    resident_buffers: u64,
    budget_bytes: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct MetalCacheCounters {
    allocations: u64,
    syncs: u64,
    evictions: u64,
    bytes_uploaded: u64,
    bytes_evicted: u64,
    resident_bytes: u64,
    resident_buffers: u64,
}

#[derive(Debug, Default)]
struct MetalBackendMetrics {
    counters: Mutex<HashMap<&'static str, MetalKernelCounters>>,
    bf16_matrix_cache: Mutex<MetalBf16MatrixCacheCounters>,
    kv_cache: Mutex<MetalCacheCounters>,
    linear_cache: Mutex<MetalCacheCounters>,
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

    fn record_bf16_matrix_cache_eviction(&self, count: u64, byte_len: u64) {
        let mut cache = self
            .bf16_matrix_cache
            .lock()
            .expect("Metal BF16 matrix cache metrics lock is not poisoned");
        cache.evictions += count;
        cache.bytes_evicted += byte_len;
    }

    fn record_bf16_matrix_cache_residency(
        &self,
        resident_bytes: u64,
        resident_buffers: u64,
        budget_bytes: u64,
    ) {
        let mut cache = self
            .bf16_matrix_cache
            .lock()
            .expect("Metal BF16 matrix cache metrics lock is not poisoned");
        cache.resident_bytes = resident_bytes;
        cache.resident_buffers = resident_buffers;
        cache.budget_bytes = budget_bytes;
    }

    fn record_kv_cache_allocation(&self, byte_len: u64) {
        self.update_cache_counter(CacheMetricKind::Kv, |cache| {
            cache.allocations += 1;
            cache.bytes_uploaded += byte_len;
        });
    }

    fn record_kv_cache_sync(&self, byte_len: u64) {
        self.update_cache_counter(CacheMetricKind::Kv, |cache| {
            cache.syncs += 1;
            cache.bytes_uploaded += byte_len;
        });
    }

    fn record_kv_cache_eviction(&self, count: u64, byte_len: u64) {
        self.update_cache_counter(CacheMetricKind::Kv, |cache| {
            cache.evictions += count;
            cache.bytes_evicted += byte_len;
        });
    }

    fn record_kv_cache_residency(&self, resident_bytes: u64, resident_buffers: u64) {
        self.update_cache_counter(CacheMetricKind::Kv, |cache| {
            cache.resident_bytes = resident_bytes;
            cache.resident_buffers = resident_buffers;
        });
    }

    fn record_linear_cache_allocation(&self, byte_len: u64) {
        self.update_cache_counter(CacheMetricKind::Linear, |cache| {
            cache.allocations += 1;
            cache.bytes_uploaded += byte_len;
        });
    }

    fn record_linear_cache_sync(&self, byte_len: u64) {
        self.update_cache_counter(CacheMetricKind::Linear, |cache| {
            cache.syncs += 1;
            cache.bytes_uploaded += byte_len;
        });
    }

    fn record_linear_cache_eviction(&self, count: u64, byte_len: u64) {
        self.update_cache_counter(CacheMetricKind::Linear, |cache| {
            cache.evictions += count;
            cache.bytes_evicted += byte_len;
        });
    }

    fn record_linear_cache_residency(&self, resident_bytes: u64, resident_buffers: u64) {
        self.update_cache_counter(CacheMetricKind::Linear, |cache| {
            cache.resident_bytes = resident_bytes;
            cache.resident_buffers = resident_buffers;
        });
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
        let kv_cache = *self
            .kv_cache
            .lock()
            .expect("Metal KV cache metrics lock is not poisoned");
        let linear_cache = *self
            .linear_cache
            .lock()
            .expect("Metal linear cache metrics lock is not poisoned");
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
                "evictions": bf16_matrix_cache.evictions,
                "bytes_evicted": bf16_matrix_cache.bytes_evicted,
                "resident_bytes": bf16_matrix_cache.resident_bytes,
                "resident_buffers": bf16_matrix_cache.resident_buffers,
                "budget_bytes": bf16_matrix_cache.budget_bytes,
            },
            "kv_cache": cache_counters_json(kv_cache),
            "linear_attention_cache": cache_counters_json(linear_cache),
        })
    }

    fn update_cache_counter(
        &self,
        kind: CacheMetricKind,
        update: impl FnOnce(&mut MetalCacheCounters),
    ) {
        let cache = match kind {
            CacheMetricKind::Kv => &self.kv_cache,
            CacheMetricKind::Linear => &self.linear_cache,
        };
        let mut cache = cache
            .lock()
            .expect("Metal resident cache metrics lock is not poisoned");
        update(&mut cache);
    }

    fn update_counter(&self, kernel: &'static str, update: impl FnOnce(&mut MetalKernelCounters)) {
        let mut counters = self
            .counters
            .lock()
            .expect("Metal metrics lock is not poisoned");
        update(counters.entry(kernel).or_default());
    }
}

#[derive(Debug, Clone, Copy)]
enum CacheMetricKind {
    Kv,
    Linear,
}

fn cache_counters_json(counters: MetalCacheCounters) -> Value {
    json!({
        "allocations": counters.allocations,
        "syncs": counters.syncs,
        "evictions": counters.evictions,
        "bytes_uploaded": counters.bytes_uploaded,
        "bytes_evicted": counters.bytes_evicted,
        "resident_bytes": counters.resident_bytes,
        "resident_buffers": counters.resident_buffers,
    })
}

fn native_qwen_metal_metrics() -> &'static MetalBackendMetrics {
    static METRICS: OnceLock<MetalBackendMetrics> = OnceLock::new();
    METRICS.get_or_init(MetalBackendMetrics::default)
}

fn native_qwen_metal_state_registry() -> &'static NativeQwenMetalStateRegistry {
    static REGISTRY: OnceLock<NativeQwenMetalStateRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn native_qwen_shared_metal_state(
    weight_cache_bytes: u64,
    cache_namespace: &str,
) -> Result<Option<Arc<NativeQwenMetalState>>, llm_metal::MetalError> {
    let key = NativeQwenMetalStateKey {
        cache_namespace: cache_namespace.to_owned(),
        weight_cache_bytes,
    };
    let registry = native_qwen_metal_state_registry();
    if let Some(state) = registry
        .lock()
        .expect("native Qwen Metal state registry lock is not poisoned")
        .get(&key)
        .cloned()
    {
        return Ok(Some(state));
    }
    let Some(device) = llm_metal::MetalDevice::system_default_result()? else {
        return Ok(None);
    };
    let mut states = registry
        .lock()
        .expect("native Qwen Metal state registry lock is not poisoned");
    if let Some(state) = states.get(&key).cloned() {
        return Ok(Some(state));
    }
    let state = Arc::new(NativeQwenMetalState::new(device, weight_cache_bytes));
    states.insert(key, Arc::clone(&state));
    Ok(Some(state))
}

impl NativeQwenMatvecBackend {
    fn system_default(weight_cache_bytes: u64, cache_namespace: &str) -> Self {
        match native_qwen_shared_metal_state(weight_cache_bytes, cache_namespace) {
            Ok(Some(state)) => Self::Metal(state),
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

    fn metal_state(&self) -> Option<Arc<NativeQwenMetalState>> {
        match self {
            Self::Cpu => None,
            Self::Metal(state) => Some(Arc::clone(state)),
        }
    }

    fn warm_bf16_matrix_cache(
        &self,
        store: &SafeTensorShardStore,
    ) -> Result<NativeQwenMetalWarmup, NativeQwenMetalBufferError> {
        let candidates = native_qwen_warmable_bf16_matrix_tensors(store)
            .map_err(NativeQwenMetalBufferError::Tensor)?
            .len() as u64;
        match self {
            Self::Cpu => Ok(NativeQwenMetalWarmup {
                candidates,
                skipped_non_metal: candidates,
                ..NativeQwenMetalWarmup::default()
            }),
            Self::Metal(metal) => metal.warm_bf16_matrix_cache(store),
        }
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

    fn bf16_matvec_range_row_major_f32(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        element_offset: usize,
        rows: usize,
        columns: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, TensorLoadError> {
        let Self::Metal(state) = self else {
            return Self::cpu().bf16_matvec_range_row_major_f32(
                store,
                tensor,
                element_offset,
                rows,
                columns,
                input,
            );
        };
        if input.len() != columns {
            Self::record_metal_fallback(
                "matvec_bf16_f32",
                format!(
                    "tensor={tensor},offset={element_offset},rows={rows},cols={columns},input_len={}",
                    input.len()
                ),
                "BF16 range input width mismatch",
            );
            return Self::cpu().bf16_matvec_range_row_major_f32(
                store,
                tensor,
                element_offset,
                rows,
                columns,
                input,
            );
        }
        let matrix = match state.bf16_matrix_buffer(store, tensor, element_offset, rows, columns) {
            Ok(matrix) => matrix,
            Err(err) => {
                Self::record_metal_fallback(
                    "matvec_bf16_f32",
                    format!("tensor={tensor},offset={element_offset},rows={rows},cols={columns}"),
                    err,
                );
                return Self::cpu().bf16_matvec_range_row_major_f32(
                    store,
                    tensor,
                    element_offset,
                    rows,
                    columns,
                    input,
                );
            }
        };
        Self::run_metal_tensor(
            "matvec_bf16_f32",
            format!("tensor={tensor},offset={element_offset},rows={rows},cols={columns}"),
            || state.device.matvec_bf16_f32_buffered(&matrix, input),
            || {
                Self::cpu().bf16_matvec_range_row_major_f32(
                    store,
                    tensor,
                    element_offset,
                    rows,
                    columns,
                    input,
                )
            },
        )
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

    fn select_kv_cache_head_rows_f32(
        &self,
        cache: &LayerKvCache,
        tensor: QwenKvCacheTensor,
        row_count: usize,
        head_start: usize,
        head_len: usize,
    ) -> Result<Vec<f32>, MathError> {
        match self {
            Self::Cpu => Self::cpu()
                .select_kv_cache_head_rows_f32(cache, tensor, row_count, head_start, head_len),
            Self::Metal(metal) => Self::run_metal_math(
                "select_head_rows_f32",
                format!(
                    "cache_id={},tensor={tensor:?},row_count={row_count},row_len={},head_start={head_start},head_len={head_len}",
                    cache.id(),
                    cache.vector_len()
                ),
                || metal.select_kv_cache_head_rows(cache, tensor, row_count, head_start, head_len),
                || {
                    Self::cpu().select_kv_cache_head_rows_f32(
                        cache, tensor, row_count, head_start, head_len,
                    )
                },
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_attention_recurrent_cache_update_f32(
        &self,
        cache: &LinearAttentionCache,
        state_start: usize,
        key: &[f32],
        value: &[f32],
        memory: &[f32],
        beta: f32,
        decay: f32,
        key_head_dim: usize,
        value_head_dim: usize,
    ) -> Result<Vec<f32>, MathError> {
        match self {
            Self::Cpu => Self::cpu().linear_attention_recurrent_cache_update_f32(
                cache,
                state_start,
                key,
                value,
                memory,
                beta,
                decay,
                key_head_dim,
                value_head_dim,
            ),
            Self::Metal(metal) => Self::run_metal_math(
                "linear_attention_recurrent_update_state_f32",
                format!(
                    "cache_id={},state_start={state_start},key_head_dim={key_head_dim},value_head_dim={value_head_dim}",
                    cache.id()
                ),
                || {
                    metal.linear_attention_recurrent_cache_update(
                        cache,
                        state_start,
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
                    Self::cpu().linear_attention_recurrent_cache_update_f32(
                        cache,
                        state_start,
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
        if options.eager_materialize_shards {
            let materialized_bytes = store.materialize_all_shards()?;
            tracing::info!(
                materialized_bytes,
                "materialized native Qwen safetensors shards"
            );
        }
        let matvec = NativeQwenMatvecBackend::system_default(
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
        Ok(Self {
            model_id,
            metadata,
            tokenizer: HuggingFaceTokenizer::from_file(snapshot_path.join("tokenizer.json"))?,
            spec: QwenModelSpec::from_config_json(&config_json)?,
            store,
            matvec,
            max_new_tokens: 1,
            max_prefill_tokens: 32,
            top_k: 16,
            chunk_rows: 2048,
            prefix_cache: Arc::new(NativeQwenPrefixCache::new(
                DEFAULT_NATIVE_QWEN_PREFIX_CACHE_BYTES,
            )),
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
        let mut decode =
            self.start_decode_session(&context_tokens, requested, &request, &cancellation)?;
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
        let mut text_deltas = NativeStreamTextDeltas::default();
        let mut unreported_completion_tokens = 0_u64;
        let mut finish_reason = FinishReason::Length;
        let eos_id = self
            .tokenizer
            .token_to_id("<|im_end|>")
            .map(|id| id as usize);
        let requested = resolve_native_max_tokens(request.max_tokens, self.max_new_tokens)?;
        let mut decode =
            match self.start_decode_session(&context_tokens, requested, &request, &cancellation) {
                Ok(decode) => decode,
                Err(BackendError::Cancelled) if cancellation.is_cancelled() => {
                    return Err(BackendError::Cancelled);
                }
                Err(err) => return Err(err),
            };
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
            unreported_completion_tokens += 1;
            let next_decoded = self
                .tokenizer
                .decode(&output_ids, false)
                .map_err(|err| BackendError::Other(err.to_string()))?;
            let delta = text_deltas.observe(next_decoded)?;
            if cancellation.is_cancelled() {
                return Err(BackendError::Cancelled);
            }
            if let Some(delta) = delta {
                send_backend_stream_chunk(
                    &tx,
                    BackendStreamChunk {
                        text: delta,
                        prompt_tokens: prompt_tokens.len() as u64,
                        completion_tokens: std::mem::take(&mut unreported_completion_tokens),
                        finish_reason: None,
                    },
                )?;
            }
            if step_idx + 1 < requested {
                if cancellation.is_cancelled() {
                    return Err(BackendError::Cancelled);
                }
                decode.step(&self.store, &self.spec, &self.matvec, candidate.token_id)?;
            }
        }

        if cancellation.is_cancelled() {
            return Err(BackendError::Cancelled);
        }
        let final_text = if output_ids.is_empty() {
            None
        } else {
            let final_decoded = self
                .tokenizer
                .decode(&output_ids, false)
                .map_err(|err| BackendError::Other(err.to_string()))?;
            text_deltas.finish(final_decoded)?
        };
        send_backend_stream_chunk(
            &tx,
            BackendStreamChunk {
                text: final_text.unwrap_or_default(),
                prompt_tokens: prompt_tokens.len() as u64,
                completion_tokens: std::mem::take(&mut unreported_completion_tokens),
                finish_reason: Some(finish_reason),
            },
        )
    }

    fn start_decode_session(
        &self,
        context_tokens: &[usize],
        max_new_tokens: u32,
        request: &BackendRequest,
        cancellation: &CancellationToken,
    ) -> Result<NativeQwenDecodeSession, BackendError> {
        if cancellation.is_cancelled() {
            return Err(BackendError::Cancelled);
        }
        let cache_tokens =
            native_qwen_cache_token_capacity(self.max_prefill_tokens, max_new_tokens);
        let namespace = native_qwen_prefix_namespace(self, request, cache_tokens);
        let layer_count = self.spec.num_hidden_layers as usize;
        let mut cached_prefix_len = 0_usize;
        let (mut hidden, mut caches) =
            if let Some(hit) = self.prefix_cache.lookup(&namespace, context_tokens) {
                if hit.caches.len() != layer_count {
                    return Err(BackendError::Other(format!(
                        "native Qwen prefix cache entry had {} layers, expected {layer_count}",
                        hit.caches.len()
                    )));
                }
                cached_prefix_len = hit.token_count;
                (Some(hit.hidden), hit.caches)
            } else {
                (
                    None,
                    qwen_layer_caches_for_spec(&self.spec, cache_tokens)
                        .map_err(|err| BackendError::Other(err.to_string()))?,
                )
            };
        if cancellation.is_cancelled() {
            return Err(BackendError::Cancelled);
        }
        if cached_prefix_len < context_tokens.len() {
            hidden = Some(native_qwen_prefill_context_with_cache(
                &self.store,
                &self.spec,
                &context_tokens[cached_prefix_len..],
                &mut caches,
                &self.matvec,
                self.max_prefill_tokens,
                cancellation,
            )?);
        }
        let hidden = hidden.ok_or_else(|| {
            BackendError::Other("Qwen prefill returned no hidden states".to_owned())
        })?;
        if cancellation.is_cancelled() {
            return Err(BackendError::Cancelled);
        }
        self.prefix_cache
            .store(namespace, context_tokens, &hidden, &caches);
        Ok(NativeQwenDecodeSession {
            hidden,
            caches,
            metal_state: self.matvec.metal_state(),
        })
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
    metal_state: Option<Arc<NativeQwenMetalState>>,
}

#[derive(Default)]
struct NativeStreamTextDeltas {
    emitted: String,
    pending: Option<String>,
}

impl NativeStreamTextDeltas {
    fn observe(&mut self, decoded: String) -> Result<Option<String>, BackendError> {
        if !decoded.starts_with(&self.emitted) {
            return Err(non_prefix_stream_error());
        }
        let Some(pending) = self.pending.take() else {
            self.pending = Some(decoded);
            return Ok(None);
        };
        let delta = if pending.starts_with(&self.emitted) && decoded.starts_with(&pending) {
            let delta = pending[self.emitted.len()..].to_owned();
            self.emitted = pending;
            non_empty(delta)
        } else {
            None
        };
        self.pending = Some(decoded);
        Ok(delta)
    }

    fn finish(&mut self, decoded: String) -> Result<Option<String>, BackendError> {
        self.pending = None;
        if !decoded.starts_with(&self.emitted) {
            return Err(non_prefix_stream_error());
        }
        let delta = decoded[self.emitted.len()..].to_owned();
        self.emitted = decoded;
        Ok(non_empty(delta))
    }
}

fn non_empty(value: String) -> Option<String> {
    (!value.is_empty()).then_some(value)
}

fn non_prefix_stream_error() -> BackendError {
    BackendError::Other(
        "native tokenizer streaming decode became non-prefix after emitted delta".to_owned(),
    )
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

impl Drop for NativeQwenDecodeSession {
    fn drop(&mut self) {
        if let Some(state) = &self.metal_state {
            state.remove_cache_mirrors(&self.caches);
        }
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

fn native_qwen_metal_weight_cache_bytes(configured: Option<u64>) -> u64 {
    configured.unwrap_or(DEFAULT_NATIVE_QWEN_METAL_WEIGHT_CACHE_BYTES)
}

fn native_qwen_warmable_bf16_matrix_tensors(
    store: &SafeTensorShardStore,
) -> Result<Vec<WarmableBf16MatrixTensor>, TensorLoadError> {
    let mut tensors = Vec::new();
    for name in store.tensor_names() {
        let metadata = store.tensor_metadata(name)?;
        if metadata.dtype == "BF16" && metadata.shape.len() == 2 {
            tensors.push(WarmableBf16MatrixTensor {
                name: name.to_owned(),
                rows: metadata.shape[0],
                columns: metadata.shape[1],
                byte_len: metadata.byte_len as u64,
            });
        }
    }
    tensors.sort_by(|left, right| {
        native_qwen_bf16_matrix_warm_order(&left.name)
            .cmp(&native_qwen_bf16_matrix_warm_order(&right.name))
            .then_with(|| left.name.cmp(&right.name))
    });
    Ok(tensors)
}

fn native_qwen_bf16_matrix_warm_order(name: &str) -> NativeQwenWeightWarmOrder {
    if name == "model.language_model.embed_tokens.weight" {
        return NativeQwenWeightWarmOrder {
            stage: 0,
            layer: 0,
            item: 0,
        };
    }
    if name == "lm_head.weight" {
        return NativeQwenWeightWarmOrder {
            stage: 3,
            layer: 0,
            item: 0,
        };
    }
    let Some(layer_suffix) = name.strip_prefix("model.language_model.layers.") else {
        return native_qwen_unknown_weight_warm_order();
    };
    let Some((layer, suffix)) = layer_suffix.split_once('.') else {
        return native_qwen_unknown_weight_warm_order();
    };
    let Ok(layer) = layer.parse::<usize>() else {
        return native_qwen_unknown_weight_warm_order();
    };
    let Some((stage, item)) = native_qwen_layer_bf16_matrix_warm_order(suffix) else {
        return native_qwen_unknown_weight_warm_order();
    };
    NativeQwenWeightWarmOrder { stage, layer, item }
}

fn native_qwen_layer_bf16_matrix_warm_order(suffix: &str) -> Option<(u8, u8)> {
    let item = match suffix {
        "self_attn.q_proj.weight" | "linear_attn.in_proj_qkv.weight" => 0,
        "self_attn.k_proj.weight" | "linear_attn.in_proj_z.weight" => 1,
        "self_attn.v_proj.weight" | "linear_attn.in_proj_b.weight" => 2,
        "self_attn.o_proj.weight" | "linear_attn.in_proj_a.weight" => 3,
        "linear_attn.out_proj.weight" => 4,
        "mlp.gate.weight" => 10,
        "mlp.shared_expert.gate_proj.weight" => 11,
        "mlp.shared_expert.up_proj.weight" => 12,
        "mlp.shared_expert.down_proj.weight" => 13,
        "mlp.shared_expert_gate.weight" => 14,
        _ => return None,
    };
    Some((1, item))
}

fn native_qwen_unknown_weight_warm_order() -> NativeQwenWeightWarmOrder {
    NativeQwenWeightWarmOrder {
        stage: 4,
        layer: usize::MAX,
        item: u8::MAX,
    }
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
        let worker = tokio::task::spawn_blocking(move || {
            let err_tx = tx.clone();
            if let Err(err) = backend.generate_blocking_stream(request, tx, cancellation) {
                let _ = err_tx.blocking_send(Err(err));
            }
        });
        native_qwen_worker_stream(rx, worker)
    }
}

fn native_qwen_worker_stream(
    rx: tokio::sync::mpsc::Receiver<Result<BackendStreamChunk, BackendError>>,
    worker: tokio::task::JoinHandle<()>,
) -> BoxStream<'static, Result<BackendStreamChunk, BackendError>> {
    async_stream::stream! {
        let mut rx = rx;
        let mut worker = Some(worker);
        loop {
            let Some(handle) = worker.as_mut() else {
                match rx.recv().await {
                    Some(item) => {
                        yield item;
                        continue;
                    }
                    None => break,
                }
            };
            tokio::select! {
                item = rx.recv() => {
                    match item {
                        Some(item) => yield item,
                        None => {
                            let result = worker
                                .take()
                                .expect("worker handle exists while stream watches it")
                                .await;
                            if let Err(err) = result {
                                yield Err(BackendError::Other(format!(
                                    "native Qwen streaming worker failed: {err}"
                                )));
                            }
                            break;
                        }
                    }
                }
                result = handle => {
                    worker = None;
                    if let Err(err) = result {
                        yield Err(BackendError::Other(format!(
                            "native Qwen streaming worker failed: {err}"
                        )));
                        break;
                    }
                }
            }
        }
    }
    .boxed()
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
    ModelStore::mark_snapshot_used(&snapshot_path)
        .await
        .map_err(EngineError::ModelStore)?;
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
    require_model_alias(&state, &alias)?;
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
    require_model_alias(&state, &alias)?;
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
    ModelStore::mark_snapshot_used(&snapshot.path)
        .await
        .map_err(EngineError::ModelStore)?;
    ModelStore::new(&state.model_home)
        .record_snapshot_alias(&alias, &snapshot.path)
        .await
        .map_err(EngineError::ModelStore)?;
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

fn require_model_alias(state: &AppState, alias: &str) -> Result<(), EngineError> {
    let model_id = state.runtime.model_id();
    if alias == model_id {
        return Ok(());
    }
    Err(RuntimeError::Backend(BackendError::ModelNotFound {
        requested: alias.to_owned(),
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
    let scheduler = state.model_scheduler.snapshot();
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
        "queued_requests": scheduler.queued_total(),
        "queued_prefill_requests": scheduler.queued_prefill,
        "queued_decode_requests": scheduler.queued_decode,
        "prefill_requests": state.generation_phases.prefill_requests(),
        "decode_requests": state.generation_phases.decode_requests(),
        "active_prefill_requests": scheduler.active_prefill,
        "active_decode_requests": scheduler.active_decode,
        "scheduler_admitted_prefill_requests": scheduler.admitted_prefill,
        "scheduler_admitted_decode_requests": scheduler.admitted_decode,
        "scheduler_completed_requests": scheduler.completed,
        "scheduler_cancelled_requests": scheduler.cancelled,
        "scheduler_failed_requests": scheduler.failed,
        "scheduler_queued_cancelled_requests": scheduler.queued_cancelled,
        "scheduler_queue_timeouts": scheduler.queue_timeouts,
        "cancelled_requests": metrics.cancelled_requests(),
        "no_progress_failures": metrics.no_progress_failures(),
        "model_pull_operations": metrics.model_pull_operations(),
        "model_pull_successes": metrics.model_pull_successes(),
        "model_pull_failures": metrics.model_pull_failures(),
        "model_pull_bytes": metrics.model_pull_bytes(),
        "model_store_snapshots": model_store_usage.snapshots,
        "model_store_bytes": model_store_usage.bytes,
        "model_store_quarantined_snapshots": model_store_usage.quarantined_snapshots,
        "model_store_quarantined_bytes": model_store_usage.quarantined_bytes,
        "artifact_verification_failures": metrics.artifact_verification_failures(),
        "process_rss_bytes": process_rss_bytes(),
        "tokens_per_second": metrics.tokens_per_second(),
        "native_qwen_metal": native_qwen_metal_metrics().snapshot(),
        "native_qwen_prefix_cache": native_qwen_prefix_cache_metrics().snapshot(),
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
    quarantined_snapshots: usize,
    quarantined_bytes: u64,
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
    let quarantined = ModelStore::new(model_home)
        .list_quarantined_snapshots()
        .await
        .map_err(EngineError::ModelStore)?;
    let bytes = snapshots
        .iter()
        .flat_map(|snapshot| &snapshot.manifest.files)
        .map(|file| file.size)
        .sum();
    let quarantined_bytes = quarantined.iter().map(|snapshot| snapshot.bytes).sum();
    Ok(ModelStoreUsage {
        snapshots: snapshots.len(),
        bytes,
        quarantined_snapshots: quarantined.len(),
        quarantined_bytes,
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
    let (admission_class, initial_phase) = chat_scheduler_classes(&state, &request);
    if request.stream {
        let mut scheduler_slot =
            acquire_scheduler_slot(&state, admission_class, initial_phase).await?;
        let active_request = match register_active_request(&state, &headers) {
            Ok(active_request) => active_request,
            Err(err) => {
                scheduler_slot.mark_failed();
                return Err(err);
            }
        };
        let phase = state.generation_phases.begin(initial_phase);
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
                    mark_scheduler_runtime_error(&mut scheduler_slot, &err);
                    record_runtime_error_metrics(&state, &err);
                    return Err(err.into());
                }
            };
            let events = async_stream::stream! {
                let mut scheduler_slot = scheduler_slot;
                let _active_request = active_request;
                let mut phase = phase;
                let mut events = response.into_events();
                let mut ttft_recorded = false;
                loop {
                    match next_stream_event(&mut events, state.stream_stall_timeout).await {
                        Ok(Some(Ok(ChatCompletionStreamEvent::Chunk(chunk)))) => {
                            if !ttft_recorded && chat_chunk_has_real_delta(&chunk) {
                                phase.transition_to_decode();
                                scheduler_slot.transition_to_decode();
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
                            mark_scheduler_runtime_error(&mut scheduler_slot, &err);
                            record_runtime_error_metrics(&state, &err);
                            for event in runtime_error_stream_events(err) {
                                yield event;
                            }
                            return;
                        }
                        Ok(None) => break,
                        Err(StreamStalled) => {
                            scheduler_slot.mark_failed();
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
        let events = async_stream::stream! {
            let mut scheduler_slot = scheduler_slot;
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
                                    scheduler_slot.transition_to_decode();
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
                                mark_scheduler_runtime_error(&mut scheduler_slot, &err);
                                record_runtime_error_metrics(&state, &err);
                                for event in runtime_error_stream_events(err) {
                                    yield event;
                                }
                                return;
                            }
                            Ok(None) => break,
                            Err(StreamStalled) => {
                                scheduler_slot.mark_failed();
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
                    mark_scheduler_runtime_error(&mut scheduler_slot, &err);
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
    let mut scheduler_slot = acquire_scheduler_slot(&state, admission_class, initial_phase).await?;
    let active_request = match register_active_request(&state, &headers) {
        Ok(active_request) => active_request,
        Err(err) => {
            scheduler_slot.mark_failed();
            return Err(err);
        }
    };
    let _phase = state.generation_phases.begin(initial_phase);
    let request_id = active_request.id.clone();
    let request_started = Instant::now();
    let response = match state
        .runtime
        .chat_with_cancel(request, active_request.cancellation.clone())
        .await
    {
        Ok(response) => response,
        Err(err) => {
            mark_scheduler_runtime_error(&mut scheduler_slot, &err);
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
    let (admission_class, initial_phase) = completion_scheduler_classes(&state, &request);
    if request.stream {
        let mut scheduler_slot =
            acquire_scheduler_slot(&state, admission_class, initial_phase).await?;
        let active_request = match register_active_request(&state, &headers) {
            Ok(active_request) => active_request,
            Err(err) => {
                scheduler_slot.mark_failed();
                return Err(err);
            }
        };
        let phase = state.generation_phases.begin(initial_phase);
        let request_id = active_request.id.clone();
        let request_started = Instant::now();
        let events = async_stream::stream! {
            let mut scheduler_slot = scheduler_slot;
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
                                    scheduler_slot.transition_to_decode();
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
                                mark_scheduler_runtime_error(&mut scheduler_slot, &err);
                                record_runtime_error_metrics(&state, &err);
                                for event in runtime_error_stream_events(err) {
                                    yield event;
                                }
                                return;
                            }
                            Ok(None) => break,
                            Err(StreamStalled) => {
                                scheduler_slot.mark_failed();
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
                    mark_scheduler_runtime_error(&mut scheduler_slot, &err);
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
    let mut scheduler_slot = acquire_scheduler_slot(&state, admission_class, initial_phase).await?;
    let active_request = match register_active_request(&state, &headers) {
        Ok(active_request) => active_request,
        Err(err) => {
            scheduler_slot.mark_failed();
            return Err(err);
        }
    };
    let _phase = state.generation_phases.begin(initial_phase);
    let request_id = active_request.id.clone();
    let request_started = Instant::now();
    let response = match state
        .runtime
        .completion_with_cancel(request, active_request.cancellation.clone())
        .await
    {
        Ok(response) => response,
        Err(err) => {
            mark_scheduler_runtime_error(&mut scheduler_slot, &err);
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
        RuntimeError::NoProgress(class) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            class.code(),
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

async fn acquire_scheduler_slot(
    state: &AppState,
    admission_class: SchedulerClass,
    initial_phase: GenerationPhase,
) -> Result<SchedulerPermit, EngineError> {
    match state
        .model_scheduler
        .clone()
        .acquire(admission_class, initial_phase)
        .await
    {
        Ok(permit) => Ok(permit),
        Err(SchedulerAcquireError::QueueFull) => {
            record_failure_metrics(state);
            Err(EngineError::Overloaded(
                "model scheduler queue is full; retry the request later".to_owned(),
            ))
        }
        Err(SchedulerAcquireError::QueueTimedOut) => {
            record_failure_metrics(state);
            Err(EngineError::Overloaded(
                "model scheduler queue timed out; retry the request later".to_owned(),
            ))
        }
    }
}

fn chat_scheduler_classes(
    state: &AppState,
    request: &ChatCompletionRequest,
) -> (SchedulerClass, GenerationPhase) {
    let admission = state.model_scheduler.classify_chat(request);
    let initial_phase = if request.stream || admission == SchedulerClass::Prefill {
        GenerationPhase::Prefill
    } else {
        admission.as_phase()
    };
    (admission, initial_phase)
}

fn completion_scheduler_classes(
    state: &AppState,
    request: &CompletionRequest,
) -> (SchedulerClass, GenerationPhase) {
    let admission = state.model_scheduler.classify_completion(request);
    let initial_phase = if request.stream || admission == SchedulerClass::Prefill {
        GenerationPhase::Prefill
    } else {
        admission.as_phase()
    };
    (admission, initial_phase)
}

fn mark_scheduler_runtime_error(permit: &mut SchedulerPermit, err: &RuntimeError) {
    if matches!(err, RuntimeError::Backend(BackendError::Cancelled)) {
        permit.mark_cancelled();
    } else {
        permit.mark_failed();
    }
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
    use llm_models::{ModelFamilyAdapter, QwenFamilyAdapter};

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
        metrics.record_bf16_matrix_cache_eviction(2, 8);
        metrics.record_bf16_matrix_cache_residency(10, 3, 16);
        metrics.record_bf16_matrix_cache_hit();

        let snapshot = metrics.snapshot();
        let cache = &snapshot["bf16_matrix_cache"];
        assert_eq!(cache["hits"], 1);
        assert_eq!(cache["misses"], 1);
        assert_eq!(cache["uploads"], 1);
        assert_eq!(cache["bytes_uploaded"], 12);
        assert_eq!(cache["evictions"], 2);
        assert_eq!(cache["bytes_evicted"], 8);
        assert_eq!(cache["resident_bytes"], 10);
        assert_eq!(cache["resident_buffers"], 3);
        assert_eq!(cache["budget_bytes"], 16);
    }

    #[test]
    fn metal_backend_metrics_records_resident_attention_cache_activity() {
        let metrics = MetalBackendMetrics::default();

        metrics.record_kv_cache_allocation(16);
        metrics.record_kv_cache_sync(8);
        metrics.record_kv_cache_residency(16, 2);
        metrics.record_kv_cache_eviction(2, 16);
        metrics.record_kv_cache_residency(0, 0);
        metrics.record_linear_cache_allocation(12);
        metrics.record_linear_cache_sync(4);
        metrics.record_linear_cache_residency(12, 1);
        metrics.record_linear_cache_eviction(1, 12);
        metrics.record_linear_cache_residency(0, 0);

        let snapshot = metrics.snapshot();
        let kv = &snapshot["kv_cache"];
        assert_eq!(kv["allocations"], 1);
        assert_eq!(kv["syncs"], 1);
        assert_eq!(kv["evictions"], 2);
        assert_eq!(kv["bytes_uploaded"], 24);
        assert_eq!(kv["bytes_evicted"], 16);
        assert_eq!(kv["resident_bytes"], 0);
        assert_eq!(kv["resident_buffers"], 0);
        let linear = &snapshot["linear_attention_cache"];
        assert_eq!(linear["allocations"], 1);
        assert_eq!(linear["syncs"], 1);
        assert_eq!(linear["evictions"], 1);
        assert_eq!(linear["bytes_uploaded"], 16);
        assert_eq!(linear["bytes_evicted"], 12);
        assert_eq!(linear["resident_bytes"], 0);
        assert_eq!(linear["resident_buffers"], 0);
    }

    #[test]
    fn native_qwen_prefix_cache_reuses_longest_compatible_prefix() {
        let cache = NativeQwenPrefixCache::new(10_000);
        let namespace = native_qwen_test_prefix_namespace("base");
        let mut layer_cache = LayerKvCache::new(4, 1, 2).expect("cache shape is valid");
        layer_cache
            .append(&[1.0, 2.0], &[3.0, 4.0])
            .expect("token fits");
        let original_cache_id = layer_cache.id();
        let caches = vec![QwenLayerCache::Full(layer_cache)];

        cache.store(namespace.clone(), &[1, 2], &[0.25, 0.75], &caches);

        let hit = cache
            .lookup(&namespace, &[1, 2, 3])
            .expect("compatible longer prompt reuses stored prefix");
        assert_eq!(hit.token_count, 2);
        assert_eq!(hit.hidden, vec![0.25, 0.75]);
        match &hit.caches[0] {
            QwenLayerCache::Full(cache) => {
                assert_ne!(cache.id(), original_cache_id);
                assert_eq!(cache.token_count(), 1);
            }
            QwenLayerCache::Linear(_) => panic!("expected full-attention cache"),
        }

        let incompatible_namespace = NativeQwenPrefixCacheNamespace {
            tool_schema: Some("different-tool-schema".to_owned()),
            ..namespace.clone()
        };
        assert!(
            cache.lookup(&incompatible_namespace, &[1, 2]).is_none(),
            "tool schema changes must not reuse prefix state"
        );
    }

    #[test]
    fn native_qwen_prefix_cache_evicts_lru_entries_to_fit_budget() {
        let cache = NativeQwenPrefixCache::new(40);
        let namespace = native_qwen_test_prefix_namespace("eviction");
        let hidden = vec![1.0; 8];

        cache.store(namespace.clone(), &[1], &hidden, &[]);
        cache.store(namespace.clone(), &[2], &hidden, &[]);

        assert!(
            cache.lookup(&namespace, &[1]).is_none(),
            "oldest entry should be evicted"
        );
        assert!(
            cache.lookup(&namespace, &[2]).is_some(),
            "newest entry should remain resident"
        );
        let inner = cache
            .inner
            .lock()
            .expect("native Qwen prefix cache lock is not poisoned");
        assert_eq!(inner.entries.len(), 1);
        assert_eq!(inner.used_bytes, 32);
    }

    #[test]
    fn native_qwen_prefix_cache_metrics_expose_hits_misses_and_evictions() {
        let metrics = NativeQwenPrefixCacheMetrics::default();

        metrics.record_hit(3);
        metrics.record_miss();
        metrics.record_store(32);
        metrics.record_eviction(16);
        metrics.record_rejected();
        metrics.record_residency(32, 1);

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
    }

    #[test]
    fn bf16_matrix_buffer_cache_evicts_lru_entries_to_fit_budget() {
        let mut cache = Bf16MatrixBufferCache::new(10);
        let first = Bf16MatrixCacheKey {
            tensor: "first.weight".to_owned(),
            element_offset: 0,
            rows: 2,
            columns: 1,
        };
        let second = Bf16MatrixCacheKey {
            tensor: "second.weight".to_owned(),
            element_offset: 0,
            rows: 2,
            columns: 1,
        };
        let third = Bf16MatrixCacheKey {
            tensor: "third.weight".to_owned(),
            element_offset: 0,
            rows: 3,
            columns: 1,
        };

        assert!(cache.get(&first).is_none());
        assert!(cache.insert(first.clone(), "first", 4).inserted);
        assert!(cache.insert(second.clone(), "second", 4).inserted);
        assert_eq!(cache.get(&first), Some("first"));

        let result = cache.insert(third.clone(), "third", 6);

        assert!(result.inserted);
        assert_eq!(result.evicted_count, 1);
        assert_eq!(result.evicted_bytes, 4);
        assert_eq!(cache.used_bytes(), 10);
        assert_eq!(cache.get(&second), None);
        assert_eq!(cache.get(&first), Some("first"));
        assert_eq!(cache.get(&third), Some("third"));
    }

    #[test]
    fn bf16_matrix_buffer_cache_skips_entries_larger_than_budget() {
        let mut cache = Bf16MatrixBufferCache::new(4);
        let key = Bf16MatrixCacheKey {
            tensor: "large.weight".to_owned(),
            element_offset: 0,
            rows: 3,
            columns: 1,
        };

        let result = cache.insert(key.clone(), "large", 6);

        assert!(!result.inserted);
        assert_eq!(result.evicted_count, 0);
        assert_eq!(cache.used_bytes(), 0);
        assert_eq!(cache.get(&key), None);
    }

    #[test]
    fn native_qwen_metal_weight_cache_bytes_uses_default_or_configured_value() {
        assert_eq!(
            native_qwen_metal_weight_cache_bytes(None),
            DEFAULT_NATIVE_QWEN_METAL_WEIGHT_CACHE_BYTES
        );
        assert_eq!(native_qwen_metal_weight_cache_bytes(Some(0)), 0);
        assert_eq!(native_qwen_metal_weight_cache_bytes(Some(4096)), 4096);
    }

    #[test]
    fn native_qwen_warmable_bf16_matrix_tensors_filters_rank2_bf16() {
        let snapshot = temp_snapshot_dir("warmable-bf16-matrices");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        let tensors = vec![
            ("z.bias", vec![2], vec![1.0, 2.0]),
            ("b.weight", vec![2, 1], vec![3.0, 4.0]),
            ("a.weight", vec![1, 2], vec![5.0, 6.0]),
        ];
        let safetensors = tiny_owned_multi_safetensors_bf16(&tensors);
        std::fs::write(snapshot.join("model.safetensors"), &safetensors).expect("write shard");
        std::fs::write(
            snapshot.join("model.safetensors.index.json"),
            serde_json::json!({
                "metadata": { "total_size": safetensors.len() },
                "weight_map": {
                    "z.bias": "model.safetensors",
                    "b.weight": "model.safetensors",
                    "a.weight": "model.safetensors"
                }
            })
            .to_string(),
        )
        .expect("write index");
        let store = SafeTensorShardStore::open(&snapshot).expect("store opens");

        let warmable = native_qwen_warmable_bf16_matrix_tensors(&store).expect("warmable tensors");

        assert_eq!(
            warmable
                .iter()
                .map(|tensor| (
                    tensor.name.as_str(),
                    tensor.rows,
                    tensor.columns,
                    tensor.byte_len
                ))
                .collect::<Vec<_>>(),
            vec![("a.weight", 1, 2, 4), ("b.weight", 2, 1, 4)]
        );
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn native_qwen_warmable_bf16_matrix_tensors_orders_qwen_execution_weights() {
        let snapshot = temp_snapshot_dir("warmable-qwen-order");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        let tensors = vec![
            ("zz.unclassified.weight", vec![1, 1], vec![1.0]),
            ("lm_head.weight", vec![1, 1], vec![2.0]),
            (
                "model.language_model.layers.10.self_attn.o_proj.weight",
                vec![1, 1],
                vec![3.0],
            ),
            (
                "model.language_model.layers.2.mlp.shared_expert.down_proj.weight",
                vec![1, 1],
                vec![4.0],
            ),
            (
                "model.language_model.layers.2.self_attn.q_proj.weight",
                vec![1, 1],
                vec![5.0],
            ),
            (
                "model.language_model.embed_tokens.weight",
                vec![1, 1],
                vec![6.0],
            ),
            (
                "model.language_model.layers.2.self_attn.k_proj.weight",
                vec![1, 1],
                vec![7.0],
            ),
            (
                "model.language_model.layers.2.mlp.gate.weight",
                vec![1, 1],
                vec![8.0],
            ),
        ];
        let safetensors = tiny_owned_multi_safetensors_bf16(&tensors);
        std::fs::write(snapshot.join("model.safetensors"), &safetensors).expect("write shard");
        std::fs::write(
            snapshot.join("model.safetensors.index.json"),
            serde_json::json!({
                "metadata": { "total_size": safetensors.len() },
                "weight_map": tensors
                    .iter()
                    .map(|(name, _, _)| {
                        (
                            (*name).to_owned(),
                            serde_json::Value::String("model.safetensors".to_owned()),
                        )
                    })
                    .collect::<serde_json::Map<_, _>>()
            })
            .to_string(),
        )
        .expect("write index");
        let store = SafeTensorShardStore::open(&snapshot).expect("store opens");

        let warmable = native_qwen_warmable_bf16_matrix_tensors(&store).expect("warmable tensors");

        assert_eq!(
            warmable
                .iter()
                .map(|tensor| tensor.name.as_str())
                .collect::<Vec<_>>(),
            vec![
                "model.language_model.embed_tokens.weight",
                "model.language_model.layers.2.self_attn.q_proj.weight",
                "model.language_model.layers.2.self_attn.k_proj.weight",
                "model.language_model.layers.2.mlp.gate.weight",
                "model.language_model.layers.2.mlp.shared_expert.down_proj.weight",
                "model.language_model.layers.10.self_attn.o_proj.weight",
                "lm_head.weight",
                "zz.unclassified.weight",
            ]
        );
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn native_qwen_cpu_backend_warmup_reports_non_metal_skip() {
        let snapshot = temp_snapshot_dir("cpu-warmup");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        let safetensors = tiny_owned_multi_safetensors_bf16(&[
            ("a.weight", vec![1, 2], vec![1.0, 2.0]),
            ("b.bias", vec![2], vec![3.0, 4.0]),
        ]);
        std::fs::write(snapshot.join("model.safetensors"), &safetensors).expect("write shard");
        std::fs::write(
            snapshot.join("model.safetensors.index.json"),
            serde_json::json!({
                "metadata": { "total_size": safetensors.len() },
                "weight_map": {
                    "a.weight": "model.safetensors",
                    "b.bias": "model.safetensors"
                }
            })
            .to_string(),
        )
        .expect("write index");
        let store = SafeTensorShardStore::open(&snapshot).expect("store opens");

        let warmup = NativeQwenMatvecBackend::Cpu
            .warm_bf16_matrix_cache(&store)
            .expect("cpu warmup reports stats");

        assert_eq!(
            warmup,
            NativeQwenMetalWarmup {
                candidates: 1,
                skipped_non_metal: 1,
                ..NativeQwenMetalWarmup::default()
            }
        );
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn native_qwen_system_default_reuses_shared_metal_state_for_same_model_budget() {
        let first = NativeQwenMatvecBackend::system_default(1_234_567, "test-shared-model");
        let second = NativeQwenMatvecBackend::system_default(1_234_567, "test-shared-model");
        let other_model = NativeQwenMatvecBackend::system_default(1_234_567, "test-other-model");

        match (&first, &second, &other_model) {
            (
                NativeQwenMatvecBackend::Metal(first),
                NativeQwenMatvecBackend::Metal(second),
                NativeQwenMatvecBackend::Metal(other_model),
            ) => {
                assert!(Arc::ptr_eq(first, second));
                assert!(!Arc::ptr_eq(first, other_model));
            }
            (
                NativeQwenMatvecBackend::Cpu,
                NativeQwenMatvecBackend::Cpu,
                NativeQwenMatvecBackend::Cpu,
            ) => {
                eprintln!("no Metal device available; skipping shared state test");
            }
            _ => panic!("Metal backend availability changed between calls"),
        }
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
            prefix_cache: Arc::new(NativeQwenPrefixCache::new(
                DEFAULT_NATIVE_QWEN_PREFIX_CACHE_BYTES,
            )),
        };

        let decode = backend
            .start_decode_session(
                &[0, 1, 0],
                8,
                &native_qwen_test_request("local-qwen36"),
                &CancellationToken::new(),
            )
            .expect("decode session starts");

        match &decode.caches[0] {
            QwenLayerCache::Linear(cache) => assert_eq!(cache.token_count(), 3),
            QwenLayerCache::Full(_) => panic!("layer 0 should be linear attention"),
        }
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn native_qwen_start_decode_session_reuses_shared_prefix_across_requests() {
        let snapshot = temp_snapshot_dir("shared-prefix-prefill");
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
            prefix_cache: Arc::new(NativeQwenPrefixCache::new(
                DEFAULT_NATIVE_QWEN_PREFIX_CACHE_BYTES,
            )),
        };
        let request = native_qwen_test_request("local-qwen36");
        let before_hits = native_prefix_metric_counter("hits");

        let first = backend
            .start_decode_session(&[0, 1], 8, &request, &CancellationToken::new())
            .expect("first decode session starts");
        drop(first);
        let second = backend
            .start_decode_session(&[0, 1, 0], 8, &request, &CancellationToken::new())
            .expect("second decode session starts");

        assert!(
            native_prefix_metric_counter("hits") > before_hits,
            "second request should hit the shared prefix cache"
        );
        match &second.caches[0] {
            QwenLayerCache::Linear(cache) => assert_eq!(cache.token_count(), 3),
            QwenLayerCache::Full(_) => panic!("layer 0 should be linear attention"),
        }

        let mut expected_caches =
            qwen_layer_caches_for_spec(&backend.spec, native_qwen_cache_token_capacity(1, 8))
                .expect("expected caches allocate");
        let expected_hidden = native_qwen_prefill_context_with_cache(
            &backend.store,
            &backend.spec,
            &[0, 1, 0],
            &mut expected_caches,
            &NativeQwenMatvecBackend::Cpu,
            1,
            &CancellationToken::new(),
        )
        .expect("fresh prefill succeeds");
        assert_close_vec(second.hidden(), &expected_hidden);
        match (&second.caches[0], &expected_caches[0]) {
            (QwenLayerCache::Linear(actual), QwenLayerCache::Linear(expected)) => {
                assert_eq!(actual.token_count(), expected.token_count());
                assert_eq!(actual.conv_window(), expected.conv_window());
                assert_eq!(actual.recurrent_state(), expected.recurrent_state());
            }
            _ => panic!("expected linear attention caches"),
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
                ..NativeQwenLoadOptions::default()
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
                    cache_context: BackendCacheContext::default(),
                },
                cancellation,
            )
            .await
            .expect_err("pre-cancelled generation fails before decode");

        assert!(err.to_string().contains("cancelled"));
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn native_qwen_stream_with_cancel_observes_pre_cancelled_token() {
        let snapshot = temp_snapshot_dir("cancelled-stream");
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
        let (tx, _rx) = tokio::sync::mpsc::channel(1);

        let err = backend
            .generate_blocking_stream(
                BackendRequest {
                    model: "local-qwen36".to_owned(),
                    prompt: "say hi".to_owned(),
                    max_tokens: Some(1),
                    sampling: SamplingConfig::Greedy,
                    required_tool_choice: None,
                    json_object_mode: false,
                    conversation_mode: false,
                    cache_context: BackendCacheContext::default(),
                },
                tx,
                cancellation,
            )
            .expect_err("pre-cancelled stream fails before normal EOF");

        assert!(matches!(err, BackendError::Cancelled));
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[tokio::test]
    async fn native_qwen_worker_stream_reports_join_failure_after_channel_close() {
        let (_tx, rx) = tokio::sync::mpsc::channel(1);
        let worker = tokio::task::spawn_blocking(|| panic!("stream worker panic"));
        let mut stream = native_qwen_worker_stream(rx, worker);

        let err = stream
            .next()
            .await
            .expect("join failure event")
            .expect_err("worker panic is surfaced");

        assert!(
            err.to_string()
                .contains("native Qwen streaming worker failed")
        );
        assert!(stream.next().await.is_none());
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

        match backend.start_decode_session(
            &[0],
            1,
            &native_qwen_test_request("local-qwen36"),
            &cancellation,
        ) {
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
            prefix_cache: Arc::new(NativeQwenPrefixCache::new(
                DEFAULT_NATIVE_QWEN_PREFIX_CACHE_BYTES,
            )),
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
    fn native_stream_text_deltas_withhold_unstable_prefix_until_finish() {
        let mut deltas = NativeStreamTextDeltas::default();

        assert_eq!(deltas.observe("�".to_owned()).expect("observe"), None);
        assert_eq!(deltas.observe("é".to_owned()).expect("observe"), None);

        assert_eq!(
            deltas.finish("é".to_owned()).expect("finish"),
            Some("é".to_owned())
        );
    }

    #[test]
    fn native_stream_text_deltas_emit_stable_prefix_with_one_token_delay() {
        let mut deltas = NativeStreamTextDeltas::default();

        assert_eq!(deltas.observe("a".to_owned()).expect("observe"), None);
        assert_eq!(
            deltas.observe("ab".to_owned()).expect("observe"),
            Some("a".to_owned())
        );
        assert_eq!(
            deltas.observe("abc".to_owned()).expect("observe"),
            Some("b".to_owned())
        );
        assert_eq!(
            deltas.finish("abc".to_owned()).expect("finish"),
            Some("c".to_owned())
        );
    }

    #[test]
    fn native_stream_text_deltas_fail_closed_after_emitted_prefix_changes() {
        let mut deltas = NativeStreamTextDeltas::default();

        assert_eq!(deltas.observe("a".to_owned()).expect("observe"), None);
        assert_eq!(
            deltas.observe("ab".to_owned()).expect("observe"),
            Some("a".to_owned())
        );

        let err = deltas
            .observe("xb".to_owned())
            .expect_err("emitted prefix mismatch fails closed");
        assert!(err.to_string().contains("non-prefix"));
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

    fn native_qwen_test_request(model: &str) -> BackendRequest {
        BackendRequest {
            model: model.to_owned(),
            prompt: "test".to_owned(),
            max_tokens: Some(1),
            sampling: SamplingConfig::Greedy,
            required_tool_choice: None,
            json_object_mode: false,
            conversation_mode: false,
            cache_context: BackendCacheContext::default(),
        }
    }

    fn native_qwen_test_prefix_namespace(label: &str) -> NativeQwenPrefixCacheNamespace {
        NativeQwenPrefixCacheNamespace {
            model_id: format!("model-{label}"),
            backend: "native-qwen".to_owned(),
            family: Some("qwen".to_owned()),
            loader: Some("safetensors".to_owned()),
            quantization: Some("bf16".to_owned()),
            repo_id: Some("local/test".to_owned()),
            resolved_commit: Some("0123456789abcdef0123456789abcdef01234567".to_owned()),
            profile: Some("qwen-test".to_owned()),
            manifest_digest: Some(format!("digest-{label}")),
            prompt_template: QwenFamilyAdapter.cache_template_id().to_owned(),
            tool_schema: Some("tool-schema-v1".to_owned()),
            request_mode: "conversation=true,json_object=false,required_tool=None".to_owned(),
            sampling: "greedy".to_owned(),
            cache_layout_version: NATIVE_QWEN_PREFIX_CACHE_LAYOUT_VERSION,
            cache_tokens: 8,
            max_prefill_tokens: 8,
        }
    }

    fn native_prefix_metric_counter(name: &str) -> u64 {
        native_qwen_prefix_cache_metrics().snapshot()[name]
            .as_u64()
            .unwrap_or_else(|| panic!("prefix metric `{name}` is an unsigned integer"))
    }

    fn assert_close_vec(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "value {index} differed: actual={actual}, expected={expected}"
            );
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
