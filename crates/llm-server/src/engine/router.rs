#[cfg(feature = "test-utils")]
use super::protocol::protocol_test_backend;
use super::{
    admin::{
        ModelStoreUsageCache, admin_cancel_request, admin_metrics, admin_model, admin_model_plan,
        admin_model_pull, admin_model_verify, admin_models, health, models,
    },
    config::{EngineConfigError, EngineOptions, default_model_home, parse_hub_client},
    inference::{chat_completions, completions},
    lifecycle,
    requests::ActiveRequestRegistry,
    scheduler::{GenerationPhaseMetrics, ModelScheduler, ModelSchedulerOptions},
    state::AppState,
};
use crate::{NoopServerBackendMetrics, ServerBackendMetrics};
use axum::{
    Router,
    body::Body,
    extract::{DefaultBodyLimit, State},
    http::{Request, header::HeaderName},
    middleware::{self, Next},
    response::Response,
    routing::{get, post},
};
use llm_backend::ModelBackend;
use llm_hub::HubClient;
use llm_runtime::{Runtime, RuntimeOptions, ToolSchemaNormalization};
use llm_telemetry::ServerMetrics;
use std::sync::{Arc, Mutex};

/// Fails closed because a production router must be constructed with an
/// explicit inference backend.
///
/// Use `build_router_with_backend` or `build_router_with_backend_and_options`
/// for real serving. Use `build_router_with_protocol_test_backend` only
/// for protocol tests that intentionally do not exercise model inference.
pub fn build_router() -> Result<Router, EngineConfigError> {
    Err(EngineConfigError::missing_backend())
}

#[cfg(feature = "test-utils")]
pub fn build_router_with_protocol_test_backend() -> Router {
    tracing::warn!(
        "protocol test backend initialized — do not use in production; \
         the test-utils feature should never be enabled in release builds"
    );
    build_router_with_backend_and_options_allowing_unauthenticated_admin(
        Box::new(protocol_test_backend()),
        EngineOptions::default(),
    )
    .unwrap_or_else(|err| unreachable!("protocol test backend options are valid: {err}"))
}

pub fn build_router_with_backend(
    backend: Box<dyn ModelBackend>,
) -> Result<Router, EngineConfigError> {
    build_router_with_backend_and_concurrency(backend, 1)
}

pub fn build_router_with_backend_and_concurrency(
    backend: Box<dyn ModelBackend>,
    concurrency_limit: usize,
) -> Result<Router, EngineConfigError> {
    build_router_with_backend_and_options(
        backend,
        EngineOptions {
            concurrency_limit,
            ..EngineOptions::default()
        },
    )
}

pub fn build_router_with_backend_and_options(
    backend: Box<dyn ModelBackend>,
    options: EngineOptions,
) -> Result<Router, EngineConfigError> {
    build_router_with_backend_and_options_impl(
        backend,
        options,
        false,
        Arc::new(NoopServerBackendMetrics),
    )
}

pub fn build_router_with_backend_and_options_allowing_unauthenticated_admin(
    backend: Box<dyn ModelBackend>,
    options: EngineOptions,
) -> Result<Router, EngineConfigError> {
    build_router_with_backend_and_options_impl(
        backend,
        options,
        true,
        Arc::new(NoopServerBackendMetrics),
    )
}

pub fn build_router_with_backend_and_options_and_backend_metrics(
    backend: Box<dyn ModelBackend>,
    options: EngineOptions,
    backend_metrics: Arc<dyn ServerBackendMetrics>,
) -> Result<Router, EngineConfigError> {
    build_router_with_backend_and_options_impl(backend, options, false, backend_metrics)
}

pub fn build_router_with_backend_and_options_allowing_unauthenticated_admin_and_backend_metrics(
    backend: Box<dyn ModelBackend>,
    options: EngineOptions,
    backend_metrics: Arc<dyn ServerBackendMetrics>,
) -> Result<Router, EngineConfigError> {
    build_router_with_backend_and_options_impl(backend, options, true, backend_metrics)
}

fn build_router_with_backend_and_options_impl(
    backend: Box<dyn ModelBackend>,
    options: EngineOptions,
    allow_unauthenticated_admin: bool,
    backend_metrics: Arc<dyn ServerBackendMetrics>,
) -> Result<Router, EngineConfigError> {
    let hub_client = options
        .hub_endpoint
        .as_deref()
        .map(|endpoint| parse_hub_client(endpoint, options.hf_token.as_deref()))
        .transpose()?
        .unwrap_or_default();
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
    Router::new()
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
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .with_state(state.clone())
        .layer(DefaultBodyLimit::max(llm_api::MAX_JSON_BODY_BYTES))
        .layer(middleware::from_fn_with_state(
            request_id_state,
            attach_request_id_header,
        ))
}

async fn attach_request_id_header(
    State(state): State<AppState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let request_id = lifecycle::response_request_id(&state, request.headers());
    let mut response = next.run(request).await;
    let header_name = HeaderName::from_static("x-request-id");
    if !response.headers().contains_key(&header_name) {
        lifecycle::insert_request_id_header(&mut response, &request_id);
    }
    response
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
    };
    AppState {
        runtime: Arc::new(Runtime::new_with_options(backend, runtime_options)),
        metrics: Arc::new(Mutex::new(ServerMetrics::default())),
        generation_phases: Arc::new(GenerationPhaseMetrics::default()),
        model_scheduler: Arc::new(ModelScheduler::new(ModelSchedulerOptions {
            concurrency_limit: options.concurrency_limit.max(1),
            queue_limit: options.scheduler_queue_limit,
            queue_timeout: options.scheduler_queue_timeout,
            prefill_threshold_chars: options.scheduler_prefill_threshold_chars,
            prefill_burst: options.scheduler_prefill_burst.max(1),
        })),
        active_requests: ActiveRequestRegistry::default(),
        backend_metrics,
        admin_token: options.admin_token.map(Arc::from),
        allow_unauthenticated_admin,
        model_home: options.model_home.unwrap_or_else(default_model_home),
        model_store_usage: Arc::new(Mutex::new(ModelStoreUsageCache::default())),
        hub_client,
        hf_token: options.hf_token.map(Arc::from),
        stream_stall_timeout: options.stream_stall_timeout,
    }
}
