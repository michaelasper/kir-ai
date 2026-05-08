use crate::sync_ext::RecoverPoisonedMutex;
use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use futures::{Stream, StreamExt};
use llm_api::{
    ApiError, ChatCompletionRequest, ChatCompletionStreamResponse, CompletionRequest,
    CompletionStreamResponse, Usage, ValidateRequest,
};
use llm_backend::{BackendError, DeterministicBackend, ModelBackend};
use llm_hub::{HubClient, HubError};
use llm_runtime::{
    ChatCompletionStreamEvent, CompletionStreamEvent, Runtime, RuntimeError,
    chat_stream_requires_buffering,
};
use llm_telemetry::{ServerMetrics, TokenCounters};
use serde_json::json;
use std::{
    collections::HashMap,
    convert::Infallible,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};
use tokio_util::sync::CancellationToken;

mod admin;
mod scheduler;
use admin::{
    ModelStoreUsageCache, admin_cancel_request, admin_metrics, admin_model, admin_model_plan,
    admin_model_pull, admin_model_verify, admin_models, health, models,
};
use scheduler::{
    GenerationPhase, GenerationPhaseMetrics, ModelScheduler, ModelSchedulerOptions,
    SchedulerAcquireError, SchedulerClass, SchedulerPermit,
};

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
        let profile = admin::model_profile("qwen3-dense-safetensors-bf16")
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
