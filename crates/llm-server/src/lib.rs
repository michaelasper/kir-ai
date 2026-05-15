mod engine;
mod sync_ext;

pub use axum::Router as ServerRouter;
pub use engine::*;
pub use llm_util::defaults::DEFAULT_MODEL_ID;
use serde_json::Value;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct ServerBackendMetricsSnapshot {
    pub metrics: HashMap<String, Value>,
}

impl Default for ServerBackendMetricsSnapshot {
    fn default() -> Self {
        Self {
            metrics: HashMap::new(),
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
