#[cfg(feature = "test-utils")]
use super::protocol::protocol_test_backend;
use super::{
    admin::{
        ModelStoreUsageCache, admin_cancel_request, admin_kv_cache, admin_metrics,
        admin_mlx_metrics, admin_model, admin_model_plan, admin_model_pull, admin_model_verify,
        admin_models, admin_tool_stream_metrics, health, models, prometheus_metrics,
    },
    config::{EngineConfigError, EngineOptions, configured_hub_client, default_model_home},
    inference::{chat_completions, completions},
    lifecycle,
    rate_limit::{PublicInferenceClientKey, RateLimitRejection, RateLimitSnapshot},
    requests::ActiveRequestRegistry,
    scheduler::{GenerationPhaseMetrics, ModelScheduler, ModelSchedulerOptions},
    state::AppState,
};
use crate::{NoopServerBackendMetrics, ServerBackendMetrics};
use axum::{
    Router,
    body::Body,
    extract::{ConnectInfo, DefaultBodyLimit, State},
    http::{
        HeaderMap, HeaderValue, Request,
        header::{HeaderName, RETRY_AFTER},
    },
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use llm_backend_contracts::ModelBackend;
use llm_hub::HubClient;
use llm_runtime::{Runtime, RuntimeOptions, ToolSchemaNormalization};
use llm_telemetry::ServerMetrics;
use std::{
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::sync::Semaphore;

/// Fails closed because a production router must be constructed with an
/// explicit inference backend.
///
/// Use `RouterBuilder::new(backend).build()` for real serving. Use
/// `build_router_with_protocol_test_backend` only for protocol tests that
/// intentionally do not exercise model inference.
pub fn build_router() -> Result<Router, EngineConfigError> {
    Err(EngineConfigError::missing_backend())
}

pub struct RouterBuilder {
    backend: Box<dyn ModelBackend>,
    engine_options: EngineOptions,
    backend_metrics: Option<Arc<dyn ServerBackendMetrics>>,
    allow_unauthenticated_admin: bool,
}

impl RouterBuilder {
    pub fn new(backend: Box<dyn ModelBackend>) -> Self {
        Self {
            backend,
            engine_options: EngineOptions::default(),
            backend_metrics: None,
            allow_unauthenticated_admin: false,
        }
    }

    pub fn with_options(mut self, options: EngineOptions) -> Self {
        self.engine_options = options;
        self
    }

    pub fn with_engine_options(self, options: EngineOptions) -> Self {
        self.with_options(options)
    }

    pub fn with_concurrency(mut self, concurrency_limit: usize) -> Self {
        self.engine_options.concurrency_limit = concurrency_limit;
        self
    }

    pub fn with_metrics(mut self, backend_metrics: Arc<dyn ServerBackendMetrics>) -> Self {
        self.backend_metrics = Some(backend_metrics);
        self
    }

    pub fn with_backend_metrics(self, backend_metrics: Arc<dyn ServerBackendMetrics>) -> Self {
        self.with_metrics(backend_metrics)
    }

    pub fn allow_unauthenticated_admin(mut self) -> Self {
        self.allow_unauthenticated_admin = true;
        self
    }

    pub fn with_admin_token(mut self, admin_token: impl Into<String>) -> Self {
        self.engine_options.admin_token = Some(admin_token.into());
        self
    }

    pub fn build(self) -> Result<Router, EngineConfigError> {
        build_router_from_parts(
            self.backend,
            self.engine_options,
            self.allow_unauthenticated_admin,
            self.backend_metrics
                .unwrap_or_else(|| Arc::new(NoopServerBackendMetrics)),
        )
    }
}

#[cfg(feature = "test-utils")]
pub fn build_router_with_protocol_test_backend() -> Router {
    tracing::warn!(
        "protocol test backend initialized — do not use in production; \
         the test-utils feature should never be enabled in release builds"
    );
    RouterBuilder::new(Box::new(protocol_test_backend()))
        .with_options(EngineOptions::default())
        .allow_unauthenticated_admin()
        .build()
        .unwrap_or_else(|err| unreachable!("protocol test backend options are valid: {err}"))
}

#[deprecated(note = "use RouterBuilder::new(backend).build()")]
pub fn build_router_with_backend(
    backend: Box<dyn ModelBackend>,
) -> Result<Router, EngineConfigError> {
    RouterBuilder::new(backend).build()
}

#[deprecated(note = "use RouterBuilder::new(backend).with_concurrency(limit).build()")]
pub fn build_router_with_backend_and_concurrency(
    backend: Box<dyn ModelBackend>,
    concurrency_limit: usize,
) -> Result<Router, EngineConfigError> {
    RouterBuilder::new(backend)
        .with_concurrency(concurrency_limit)
        .build()
}

#[deprecated(note = "use RouterBuilder::new(backend).with_options(options).build()")]
pub fn build_router_with_backend_and_options(
    backend: Box<dyn ModelBackend>,
    options: EngineOptions,
) -> Result<Router, EngineConfigError> {
    RouterBuilder::new(backend).with_options(options).build()
}

#[deprecated(
    note = "use RouterBuilder::new(backend).with_options(options).allow_unauthenticated_admin().build()"
)]
pub fn build_router_with_backend_and_options_allowing_unauthenticated_admin(
    backend: Box<dyn ModelBackend>,
    options: EngineOptions,
) -> Result<Router, EngineConfigError> {
    RouterBuilder::new(backend)
        .with_options(options)
        .allow_unauthenticated_admin()
        .build()
}

#[deprecated(
    note = "use RouterBuilder::new(backend).with_options(options).with_metrics(metrics).build()"
)]
pub fn build_router_with_backend_and_options_and_backend_metrics(
    backend: Box<dyn ModelBackend>,
    options: EngineOptions,
    backend_metrics: Arc<dyn ServerBackendMetrics>,
) -> Result<Router, EngineConfigError> {
    RouterBuilder::new(backend)
        .with_options(options)
        .with_metrics(backend_metrics)
        .build()
}

#[deprecated(
    note = "use RouterBuilder::new(backend).with_options(options).with_metrics(metrics).allow_unauthenticated_admin().build()"
)]
pub fn build_router_with_backend_and_options_allowing_unauthenticated_admin_and_backend_metrics(
    backend: Box<dyn ModelBackend>,
    options: EngineOptions,
    backend_metrics: Arc<dyn ServerBackendMetrics>,
) -> Result<Router, EngineConfigError> {
    RouterBuilder::new(backend)
        .with_options(options)
        .with_metrics(backend_metrics)
        .allow_unauthenticated_admin()
        .build()
}

fn build_router_from_parts(
    backend: Box<dyn ModelBackend>,
    options: EngineOptions,
    allow_unauthenticated_admin: bool,
    backend_metrics: Arc<dyn ServerBackendMetrics>,
) -> Result<Router, EngineConfigError> {
    let hub_client =
        configured_hub_client(options.hub_endpoint.as_deref(), options.hf_token.as_deref())?;
    Ok(router_for_state(engine_state(
        backend,
        options,
        hub_client,
        allow_unauthenticated_admin,
        backend_metrics,
    )))
}

fn router_for_state(state: AppState) -> Router {
    let request_id_state = state.clone();
    let body_timeout_state = state.clone();
    let rate_limit_state = state.clone();
    let json_body_limit = state.request_limits.json_body_bytes;
    let inference_routes = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route_layer(middleware::from_fn_with_state(
            rate_limit_state,
            enforce_public_inference_rate_limit,
        ));

    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(prometheus_metrics))
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
        .route("/admin/kv-cache", get(admin_kv_cache))
        .route("/admin/metrics.mlx", get(admin_mlx_metrics))
        .route("/admin/metrics.tool_stream", get(admin_tool_stream_metrics))
        .merge(inference_routes)
        .with_state(state.clone())
        .layer(DefaultBodyLimit::max(json_body_limit))
        .layer(middleware::from_fn_with_state(
            body_timeout_state,
            enforce_request_body_timeout,
        ))
        .layer(middleware::from_fn_with_state(
            request_id_state,
            log_http_request,
        ))
}

async fn enforce_request_body_timeout(
    State(state): State<AppState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let Some(timeout) = state.request_body_timeout else {
        return next.run(request).await;
    };
    let (parts, body) = request.into_parts();
    let body = super::request_body_timeout::with_request_body_timeout(body, timeout);
    next.run(Request::from_parts(parts, body)).await
}

async fn enforce_public_inference_rate_limit(
    State(state): State<AppState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let peer_addr = request
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|connect_info| connect_info.0);
    let client_key = public_inference_client_key(peer_addr);
    match state.public_inference_rate_limiter.acquire(&client_key) {
        Ok(snapshot) => {
            let mut response = next.run(request).await;
            insert_rate_limit_headers(response.headers_mut(), snapshot);
            response
        }
        Err(rejection) => rate_limited_response(rejection),
    }
}

fn public_inference_client_key(peer_addr: Option<SocketAddr>) -> PublicInferenceClientKey {
    if let Some(peer_addr) = peer_addr {
        return PublicInferenceClientKey::new(format!("peer:{}", peer_addr.ip()));
    }

    PublicInferenceClientKey::anonymous()
}

fn rate_limited_response(rejection: RateLimitRejection) -> Response {
    let mut response = super::EngineError::RateLimited.into_response();
    insert_header(
        response.headers_mut(),
        RETRY_AFTER,
        retry_after_seconds(rejection.retry_after).to_string(),
    );
    insert_rate_limit_headers(response.headers_mut(), rejection.snapshot);
    response
}

fn insert_rate_limit_headers(headers: &mut HeaderMap, snapshot: RateLimitSnapshot) {
    insert_header(
        headers,
        HeaderName::from_static("x-ratelimit-limit-requests"),
        snapshot.limit_requests.to_string(),
    );
    insert_header(
        headers,
        HeaderName::from_static("x-ratelimit-remaining-requests"),
        snapshot.remaining_requests.to_string(),
    );
    insert_header(
        headers,
        HeaderName::from_static("x-ratelimit-reset-requests"),
        ceil_duration_seconds(snapshot.reset_after).to_string(),
    );
}

fn insert_header(headers: &mut HeaderMap, name: HeaderName, value: String) {
    if let Ok(value) = HeaderValue::from_str(&value) {
        headers.insert(name, value);
    }
}

fn retry_after_seconds(duration: Duration) -> u64 {
    ceil_duration_seconds(duration).max(1)
}

fn ceil_duration_seconds(duration: Duration) -> u64 {
    duration
        .as_secs()
        .saturating_add(u64::from(duration.subsec_nanos() > 0))
}

async fn log_http_request(
    State(state): State<AppState>,
    mut request: Request<Body>,
    next: Next,
) -> Response {
    let request_started = Instant::now();
    let request_id = lifecycle::ensure_request_id_header(&state, request.headers_mut());
    let method = request.method().clone();
    let path = request.uri().path().to_owned();
    tracing::debug!(
        target: "llm_server::http_access",
        request_id = %request_id,
        method = %method,
        path = %path,
        "http request started"
    );
    let mut response = next.run(request).await;
    let header_name = HeaderName::from_static("x-request-id");
    if !response.headers().contains_key(&header_name) {
        lifecycle::insert_request_id_header(&mut response, &request_id);
    }
    log_http_request_completed(
        &request_id,
        &method,
        &path,
        response.status(),
        request_started,
    );
    response
}

fn log_http_request_completed(
    request_id: &str,
    method: &axum::http::Method,
    path: &str,
    status: axum::http::StatusCode,
    request_started: Instant,
) {
    let latency_ms = duration_millis_u64(request_started.elapsed());
    if status.is_server_error() {
        tracing::warn!(
            target: "llm_server::http_access",
            request_id = %request_id,
            method = %method,
            path = %path,
            status = status.as_u16(),
            latency_ms,
            "http request completed"
        );
    } else {
        tracing::info!(
            target: "llm_server::http_access",
            request_id = %request_id,
            method = %method,
            path = %path,
            status = status.as_u16(),
            latency_ms,
            "http request completed"
        );
    }
}

fn duration_millis_u64(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn engine_state(
    backend: Box<dyn ModelBackend>,
    options: EngineOptions,
    hub_client: HubClient,
    allow_unauthenticated_admin: bool,
    backend_metrics: Arc<dyn ServerBackendMetrics>,
) -> AppState {
    let runtime_options = RuntimeOptions {
        tool_schema_normalization: if options.canonical_tool_schemas {
            ToolSchemaNormalization::Canonical
        } else {
            ToolSchemaNormalization::Preserve
        },
        request_limits: options.request_limits,
    };
    AppState {
        runtime: Arc::new(Runtime::new_with_options(backend, runtime_options)),
        metrics: Arc::new(Mutex::new(ServerMetrics::default())),
        request_cache: Arc::new(Mutex::new(
            super::metrics::RequestCacheObservations::default(),
        )),
        tool_stream: Arc::new(Mutex::new(super::metrics::ToolStreamObservations::default())),
        generation_phases: Arc::new(GenerationPhaseMetrics::default()),
        model_scheduler: Arc::new(ModelScheduler::new(ModelSchedulerOptions {
            concurrency_limit: options.concurrency_limit.max(1),
            queue_limit: options.scheduler_queue_limit,
            queue_timeout: options.scheduler_queue_timeout,
            prefill_threshold_chars: options.scheduler_prefill_threshold_chars,
            prefill_burst: options.scheduler_prefill_burst.max(1),
        })),
        active_requests: ActiveRequestRegistry::default(),
        public_inference_rate_limiter: Arc::new(
            super::rate_limit::PublicInferenceRateLimiter::new(options.public_inference_rate_limit),
        ),
        backend_metrics,
        admin_token: options.admin_token.map(Arc::from),
        allow_unauthenticated_admin,
        model_home: options.model_home.unwrap_or_else(default_model_home),
        model_store_usage: Arc::new(Mutex::new(ModelStoreUsageCache::default())),
        model_pull_gate: Arc::new(Semaphore::new(1)),
        hub_client,
        hf_token: options.hf_token.map(Arc::from),
        stream_stall_timeout: options.stream_stall_timeout,
        request_body_timeout: options.request_body_timeout,
        request_limits: options.request_limits,
    }
}
