#[cfg(feature = "mlx")]
use crate::mlx::mlx_backend_metrics_snapshot;
use llm_backend_contracts::ModelBackend;
#[cfg(feature = "native-qwen")]
use llm_native_runtime::native_qwen_prefix_cache_metrics_snapshot;
#[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
use llm_native_runtime::{
    native_text_metal_metrics_snapshot, native_text_prefix_cache_metrics_snapshot,
};
use llm_server::{RouterBuilder, ServerBackendMetrics, ServerBackendMetricsSnapshot, ServerRouter};
use std::sync::Arc;

pub use llm_server::{
    EngineConfigError, EngineOptions, PublicInferenceRateLimit, configured_hub_client,
};

pub fn build_router() -> Result<ServerRouter, EngineConfigError> {
    llm_server::build_router()
}

pub fn router_builder(backend: Box<dyn ModelBackend>) -> RouterBuilder {
    RouterBuilder::new(backend).with_metrics(Arc::new(EngineServerBackendMetrics))
}

#[cfg(feature = "test-utils")]
pub fn build_router_with_protocol_test_backend() -> ServerRouter {
    tracing::warn!(
        "protocol test backend initialized — do not use in production; \
         the test-utils feature should never be enabled in release builds"
    );
    router_builder(Box::new(
        llm_backend::ProtocolTestBackend::new(
            crate::DEFAULT_MODEL_ID,
            "hello from rust native backend",
        )
        .with_required_tool_protocol()
        .with_json_object_protocol(),
    ))
    .with_options(EngineOptions::default())
    .allow_unauthenticated_admin()
    .build()
    .unwrap_or_else(|err| unreachable!("protocol test backend options are valid: {err}"))
}

#[deprecated(note = "use router_builder(backend).build()")]
pub fn build_router_with_backend(
    backend: Box<dyn ModelBackend>,
) -> Result<ServerRouter, EngineConfigError> {
    router_builder(backend).with_concurrency(1).build()
}

#[deprecated(note = "use router_builder(backend).with_concurrency(limit).build()")]
pub fn build_router_with_backend_and_concurrency(
    backend: Box<dyn ModelBackend>,
    concurrency_limit: usize,
) -> Result<ServerRouter, EngineConfigError> {
    router_builder(backend)
        .with_concurrency(concurrency_limit)
        .build()
}

#[deprecated(note = "use router_builder(backend).with_options(options).build()")]
pub fn build_router_with_backend_and_options(
    backend: Box<dyn ModelBackend>,
    options: EngineOptions,
) -> Result<ServerRouter, EngineConfigError> {
    router_builder(backend).with_options(options).build()
}

#[deprecated(
    note = "use router_builder(backend).with_options(options).allow_unauthenticated_admin().build()"
)]
pub fn build_router_with_backend_and_options_allowing_unauthenticated_admin(
    backend: Box<dyn ModelBackend>,
    options: EngineOptions,
) -> Result<ServerRouter, EngineConfigError> {
    router_builder(backend)
        .with_options(options)
        .allow_unauthenticated_admin()
        .build()
}

#[derive(Debug)]
struct EngineServerBackendMetrics;

impl ServerBackendMetrics for EngineServerBackendMetrics {
    fn snapshot(&self) -> ServerBackendMetricsSnapshot {
        #[cfg(any(feature = "mlx", feature = "native-qwen", feature = "native-gemma"))]
        {
            let mut metrics = std::collections::HashMap::new();
            #[cfg(feature = "mlx")]
            {
                metrics.insert("mlx".to_owned(), mlx_backend_metrics_snapshot());
            }
            #[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
            {
                let native_text_metal = native_text_metal_metrics_snapshot();
                metrics.insert("native_text_metal".to_owned(), native_text_metal.clone());
                #[cfg(feature = "native-qwen")]
                {
                    metrics.insert("native_qwen_metal".to_owned(), native_text_metal);
                }
            }
            #[cfg(feature = "native-qwen")]
            let native_qwen_prefix_cache = native_qwen_prefix_cache_metrics_snapshot();
            #[cfg(feature = "native-qwen")]
            {
                metrics.insert(
                    "native_qwen_prefix_cache".to_owned(),
                    native_qwen_prefix_cache.clone(),
                );
            }
            #[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
            {
                #[cfg(not(feature = "native-qwen"))]
                let native_qwen_prefix_cache = serde_json::json!({});
                metrics.insert(
                    "native_text_prefix_cache".to_owned(),
                    native_text_prefix_cache_metrics_snapshot(native_qwen_prefix_cache),
                );
            }
            ServerBackendMetricsSnapshot { metrics }
        }
        #[cfg(not(any(feature = "mlx", feature = "native-qwen", feature = "native-gemma")))]
        {
            ServerBackendMetricsSnapshot::default()
        }
    }
}
