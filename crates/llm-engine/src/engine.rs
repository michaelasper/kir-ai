use crate::{
    native_qwen::{native_qwen_metal_metrics_snapshot, native_qwen_prefix_cache_metrics_snapshot},
    sync_ext::RecoverPoisonedMutex,
};
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
use futures::{Stream, StreamExt};
use llm_api::{
    ApiError, ChatCompletionRequest, ChatCompletionStreamResponse, CompletionRequest,
    CompletionStreamResponse, ModelCard, ModelList, Usage, ValidateRequest,
};
use llm_backend::{BackendError, BackendModelMetadata, DeterministicBackend, ModelBackend};
use llm_hub::{DownloadPlan, HubClient, HubError, HubRepoId, ModelProfile, ModelStore};
use llm_runtime::{
    ChatCompletionStreamEvent, CompletionStreamEvent, Runtime, RuntimeError,
    chat_stream_requires_buffering,
};
use llm_telemetry::{ServerMetrics, TokenCounters};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::{HashMap, VecDeque},
    convert::Infallible,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
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
            .lock_or_recover("scheduler")
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
            .lock_or_recover("scheduler")
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
        let mut state = self.scheduler.state.lock_or_recover("scheduler");
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
        let mut state = self.state.lock_or_recover("scheduler");
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
        let mut state = self.state.lock_or_recover("scheduler");
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
        let mut state = self.state.lock_or_recover("scheduler");
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
        let state = self.state.lock_or_recover("scheduler");
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
            .lock_or_recover("active request")
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

/// Fails closed because a production router must be constructed with an
/// explicit inference backend.
///
/// Use `build_router_with_backend` or `build_router_with_backend_and_options`
/// for real serving. Use `build_router_with_deterministic_test_backend` only
/// for protocol tests that intentionally do not exercise model inference.
pub fn build_router() -> Result<Router, EngineConfigError> {
    Err(EngineConfigError::missing_backend())
}

pub fn build_router_with_deterministic_test_backend() -> Router {
    build_router_with_backend(Box::new(deterministic_test_backend()))
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
    fn missing_backend() -> Self {
        Self {
            message: "llm-engine router construction requires an explicit backend; use build_router_with_backend(...) for inference or build_router_with_deterministic_test_backend() for protocol tests"
                .to_owned(),
        }
    }

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

fn deterministic_test_backend() -> DeterministicBackend {
    DeterministicBackend::new("local-qwen36", "hello from rust native backend")
        .with_required_tool_protocol()
        .with_json_object_protocol()
}

fn default_model_home() -> PathBuf {
    std::env::var_os("LLM_MODEL_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".llm-models"))
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
    ModelProfile::builtin(name).ok_or_else(|| {
        RuntimeError::Api(ApiError::invalid_request(format!(
            "unknown model profile `{name}`"
        )))
        .into()
    })
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

async fn admin_metrics(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Value>, EngineError> {
    require_admin(&state, &headers)?;
    let metrics = *state.metrics.lock_or_recover("metrics");
    let tokens = metrics.tokens();
    let request_latency = metrics.request_latency();
    let time_to_first_token = metrics.time_to_first_token();
    let model_store_usage = model_store_usage(&state).await?;
    let scheduler = state.model_scheduler.snapshot();
    let active_requests = state
        .active_requests
        .lock_or_recover("active request")
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
        "native_qwen_metal": native_qwen_metal_metrics_snapshot(),
        "native_qwen_prefix_cache": native_qwen_prefix_cache_metrics_snapshot(),
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
        .lock_or_recover("model store usage cache")
        .current(now)
    {
        return Ok(usage);
    }
    let usage = scan_model_store_usage(&state.model_home).await?;
    state
        .model_store_usage
        .lock_or_recover("model store usage cache")
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
        .lock_or_recover("model store usage cache")
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
        .lock_or_recover("active request")
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
    state.metrics.lock_or_recover("metrics").record_success(
        TokenCounters::new(usage.prompt_tokens, usage.completion_tokens),
        streamed,
        latency,
    );
}

fn record_failure_metrics(state: &AppState) {
    state.metrics.lock_or_recover("metrics").record_failure();
}

fn record_runtime_error_metrics(state: &AppState, err: &RuntimeError) {
    let mut metrics = state.metrics.lock_or_recover("metrics");
    if matches!(err, RuntimeError::NoProgress(_)) {
        metrics.record_no_progress_failure();
    }
    metrics.record_failure();
}

fn record_cancellation_metrics(state: &AppState) {
    state
        .metrics
        .lock_or_recover("metrics")
        .record_cancellation();
}

fn record_model_pull_success_metrics(state: &AppState, bytes: u64) {
    state
        .metrics
        .lock_or_recover("metrics")
        .record_model_pull_success(bytes);
}

fn record_model_pull_failure_metrics(state: &AppState) {
    state
        .metrics
        .lock_or_recover("metrics")
        .record_model_pull_failure();
}

fn record_artifact_verification_failure_metrics(state: &AppState) {
    state
        .metrics
        .lock_or_recover("metrics")
        .record_artifact_verification_failure();
}

fn record_time_to_first_token_metrics(state: &AppState, latency: Duration) {
    state
        .metrics
        .lock_or_recover("metrics")
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
    let mut active_requests = state.active_requests.lock_or_recover("active request");
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
    fn poisoned_mutex_lock_recovers_inner_state() {
        let mutex = Mutex::new(7_u32);
        let _ = std::panic::catch_unwind(|| {
            let _guard = mutex.lock().expect("test lock");
            panic!("poison test mutex");
        });

        *mutex.lock_or_recover("test") += 1;

        assert_eq!(*mutex.lock_or_recover("test"), 8);
    }

    #[test]
    fn admin_model_profile_accepts_qwen3_dense_native_profile() {
        let profile = model_profile("qwen3-dense-safetensors-bf16")
            .expect("admin profile matcher accepts dense Qwen3");

        assert_eq!(profile.name, "qwen3-dense-safetensors-bf16");
        assert_eq!(profile.family, "qwen");
        assert_eq!(profile.loader, "native-metal");
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
