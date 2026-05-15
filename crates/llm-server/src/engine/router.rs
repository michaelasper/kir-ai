#[cfg(feature = "test-utils")]
use super::protocol::protocol_test_backend;
use super::{
    admin::{
        ModelStoreUsageCache, admin_cancel_request, admin_metrics, admin_mlx_metrics, admin_model,
        admin_model_plan, admin_model_pull, admin_model_verify, admin_models, health, models,
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
    RouterBuilder::new(backend).with_concurrency(1).build()
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
    let json_body_limit = state.request_limits.json_body_bytes;
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
        .route("/admin/metrics.mlx", get(admin_mlx_metrics))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .with_state(state.clone())
        .layer(DefaultBodyLimit::max(json_body_limit))
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
        request_limits: options.request_limits,
    };
    AppState {
        runtime: Arc::new(Runtime::new_with_options(backend, runtime_options)),
        metrics: Arc::new(Mutex::new(ServerMetrics::default())),
        request_cache: Arc::new(Mutex::new(
            super::metrics::RequestCacheObservations::default(),
        )),
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
        request_limits: options.request_limits,
    }
}
