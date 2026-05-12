use crate::{
    mlx::mlx_backend_metrics_snapshot,
    native_qwen::native_qwen_prefix_cache_metrics_snapshot,
    native_text::{native_text_metal_metrics_snapshot, native_text_prefix_cache_metrics_snapshot},
};
use llm_backend::ModelBackend;
use llm_server::{
    ServerBackendMetrics, ServerBackendMetricsSnapshot, ServerRouter,
    build_router_with_backend_and_options_allowing_unauthenticated_admin_and_backend_metrics,
    build_router_with_backend_and_options_and_backend_metrics,
};
use std::sync::Arc;

pub use llm_server::{EngineConfigError, EngineOptions};

pub fn build_router() -> Result<ServerRouter, EngineConfigError> {
    llm_server::build_router()
}

#[cfg(feature = "test-utils")]
pub fn build_router_with_protocol_test_backend() -> ServerRouter {
    tracing::warn!(
        "protocol test backend initialized — do not use in production; \
         the test-utils feature should never be enabled in release builds"
    );
    build_router_with_backend_and_options_allowing_unauthenticated_admin(
        Box::new(
            llm_backend::ProtocolTestBackend::new(
                crate::DEFAULT_MODEL_ID,
                "hello from rust native backend",
            )
            .with_required_tool_protocol()
            .with_json_object_protocol(),
        ),
        EngineOptions::default(),
    )
    .unwrap_or_else(|err| unreachable!("protocol test backend options are valid: {err}"))
}

pub fn build_router_with_backend(
    backend: Box<dyn ModelBackend>,
) -> Result<ServerRouter, EngineConfigError> {
    build_router_with_backend_and_concurrency(backend, 1)
}

pub fn build_router_with_backend_and_concurrency(
    backend: Box<dyn ModelBackend>,
    concurrency_limit: usize,
) -> Result<ServerRouter, EngineConfigError> {
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
) -> Result<ServerRouter, EngineConfigError> {
    build_router_with_backend_and_options_and_backend_metrics(
        backend,
        options,
        Arc::new(EngineServerBackendMetrics),
    )
}

pub fn build_router_with_backend_and_options_allowing_unauthenticated_admin(
    backend: Box<dyn ModelBackend>,
    options: EngineOptions,
) -> Result<ServerRouter, EngineConfigError> {
    build_router_with_backend_and_options_allowing_unauthenticated_admin_and_backend_metrics(
        backend,
        options,
        Arc::new(EngineServerBackendMetrics),
    )
}

#[derive(Debug)]
struct EngineServerBackendMetrics;

impl ServerBackendMetrics for EngineServerBackendMetrics {
    fn snapshot(&self) -> ServerBackendMetricsSnapshot {
        let native_text_metal = native_text_metal_metrics_snapshot();
        let native_qwen_prefix_cache = native_qwen_prefix_cache_metrics_snapshot();
        let native_text_prefix_cache =
            native_text_prefix_cache_metrics_snapshot(native_qwen_prefix_cache.clone());
        ServerBackendMetricsSnapshot {
            mlx: mlx_backend_metrics_snapshot(),
            native_text_metal: native_text_metal.clone(),
            native_text_prefix_cache,
            native_qwen_metal: native_text_metal,
            native_qwen_prefix_cache,
        }
    }
}
