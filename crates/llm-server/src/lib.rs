mod engine;
mod sync_ext;

pub use axum::Router as ServerRouter;
pub use engine::*;
pub use llm_util::defaults::DEFAULT_MODEL_ID;
use serde_json::{Value, json};

#[derive(Clone, Debug)]
pub struct ServerBackendMetricsSnapshot {
    pub mlx: Value,
    pub native_text_metal: Value,
    pub native_text_prefix_cache: Value,
    pub native_qwen_metal: Value,
    pub native_qwen_prefix_cache: Value,
}

impl Default for ServerBackendMetricsSnapshot {
    fn default() -> Self {
        Self {
            mlx: json!({}),
            native_text_metal: json!({}),
            native_text_prefix_cache: json!({}),
            native_qwen_metal: json!({}),
            native_qwen_prefix_cache: json!({}),
        }
    }
}

pub trait ServerBackendMetrics: Send + Sync {
    fn snapshot(&self) -> ServerBackendMetricsSnapshot;
}

#[derive(Debug, Default)]
pub struct NoopServerBackendMetrics;

impl ServerBackendMetrics for NoopServerBackendMetrics {
    fn snapshot(&self) -> ServerBackendMetricsSnapshot {
        ServerBackendMetricsSnapshot::default()
    }
}

pub async fn serve(listener: tokio::net::TcpListener, router: ServerRouter) -> std::io::Result<()> {
    axum::serve(listener, router).await
}
