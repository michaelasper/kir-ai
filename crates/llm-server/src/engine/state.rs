use super::{
    admin::ModelStoreUsageCache,
    metrics::RequestCacheObservations,
    requests::ActiveRequestRegistry,
    scheduler::{GenerationPhaseMetrics, ModelScheduler},
};
use crate::ServerBackendMetrics;
use llm_backend::ModelBackend;
use llm_hub::HubClient;
use llm_runtime::Runtime;
use llm_telemetry::ServerMetrics;
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};

pub(super) type EngineRuntime = Runtime<Box<dyn ModelBackend>>;

#[derive(Clone)]
pub(super) struct AppState {
    pub(super) runtime: Arc<EngineRuntime>,
    pub(super) metrics: Arc<Mutex<ServerMetrics>>,
    pub(super) request_cache: Arc<Mutex<RequestCacheObservations>>,
    pub(super) generation_phases: Arc<GenerationPhaseMetrics>,
    pub(super) model_scheduler: Arc<ModelScheduler>,
    pub(super) active_requests: ActiveRequestRegistry,
    pub(super) backend_metrics: Arc<dyn ServerBackendMetrics>,
    pub(super) admin_token: Option<Arc<str>>,
    pub(super) allow_unauthenticated_admin: bool,
    pub(super) model_home: PathBuf,
    pub(super) model_store_usage: Arc<Mutex<ModelStoreUsageCache>>,
    pub(super) hub_client: HubClient,
    pub(super) hf_token: Option<Arc<str>>,
    pub(super) stream_stall_timeout: Option<Duration>,
}
