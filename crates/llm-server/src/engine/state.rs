use super::{
    admin::ModelStoreUsageCache,
    metrics::{RequestCacheObservations, ToolStreamObservations},
    rate_limit::PublicInferenceRateLimiter,
    requests::ActiveRequestRegistry,
    scheduler::{GenerationPhaseMetrics, ModelScheduler},
};
use crate::ServerBackendMetrics;
use llm_api::RequestLimits;
use llm_backend_contracts::ModelBackend;
use llm_hub::HubClient;
use llm_runtime::Runtime;
use llm_telemetry::ServerMetrics;
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::sync::Semaphore;

pub(super) type EngineRuntime = Runtime<Box<dyn ModelBackend>>;

#[derive(Clone)]
pub(super) struct AppState {
    pub(super) runtime: Arc<EngineRuntime>,
    pub(super) metrics: Arc<Mutex<ServerMetrics>>,
    pub(super) request_cache: Arc<Mutex<RequestCacheObservations>>,
    pub(super) tool_stream: Arc<Mutex<ToolStreamObservations>>,
    pub(super) generation_phases: Arc<GenerationPhaseMetrics>,
    pub(super) model_scheduler: Arc<ModelScheduler>,
    pub(super) active_requests: ActiveRequestRegistry,
    pub(super) public_inference_rate_limiter: Arc<PublicInferenceRateLimiter>,
    pub(super) backend_metrics: Arc<dyn ServerBackendMetrics>,
    pub(super) admin_token: Option<Arc<str>>,
    pub(super) allow_unauthenticated_admin: bool,
    pub(super) model_home: PathBuf,
    pub(super) model_store_usage: Arc<Mutex<ModelStoreUsageCache>>,
    pub(super) model_pull_gate: Arc<Semaphore>,
    pub(super) hub_client: HubClient,
    pub(super) hf_token: Option<Arc<str>>,
    pub(super) stream_stall_timeout: Option<Duration>,
    pub(super) request_body_timeout: Option<Duration>,
    pub(super) request_limits: RequestLimits,
}
