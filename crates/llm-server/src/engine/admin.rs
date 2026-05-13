use super::{
    AppState, EngineError,
    metrics::{
        record_artifact_verification_failure_metrics, record_cancellation_metrics,
        record_model_pull_failure_metrics, record_model_pull_success_metrics,
    },
    requests::CancelRequestResult,
};
use crate::sync_ext::FailPoisonedMutex;
use axum::{
    Json,
    extract::{Path as AxumPath, State, rejection::JsonRejection},
    http::{HeaderMap, header},
    response::IntoResponse,
};
use llm_api::{ApiError, ModelCard, ModelList};
use llm_backend::{BackendError, BackendModelMetadata};
use llm_hub::{DownloadPlan, HubRepoId, ModelProfile, ModelStore};
use llm_runtime::RuntimeError;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    path::Path,
    time::{Duration, Instant},
};
use subtle::ConstantTimeEq;

#[derive(Debug, Serialize, JsonSchema)]
pub(super) struct HealthResponse {
    status: String,
    runtime: String,
    python_runtime: bool,
}

pub(super) async fn health() -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok".to_owned(),
        runtime: "rust".to_owned(),
        python_runtime: false,
    })
}

pub(super) async fn models(State(state): State<AppState>) -> Json<ModelList> {
    Json(ModelList {
        object: "list".to_owned(),
        data: vec![ModelCard {
            id: state.runtime.model_id().to_owned(),
            object: "model".to_owned(),
            owned_by: "local".to_owned(),
        }],
    })
}

#[derive(Debug, Serialize, JsonSchema)]
pub(super) struct AdminModelListResponse {
    object: String,
    data: Vec<AdminModelStatusResponse>,
}

pub(super) async fn admin_models(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<AdminModelListResponse>, EngineError> {
    require_admin(&state, &headers)?;
    let metadata = state.runtime.model_metadata();
    Ok(Json(AdminModelListResponse {
        object: "list".to_owned(),
        data: vec![admin_model_status(&metadata)],
    }))
}

pub(super) async fn admin_model(
    AxumPath(alias): AxumPath<String>,
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<AdminModelStatusResponse>, EngineError> {
    require_admin(&state, &headers)?;
    let metadata = state.runtime.model_metadata();
    if alias != metadata.id {
        return Err(RuntimeError::Backend(BackendError::ModelNotFound {
            requested: alias,
            available: metadata.id,
        })
        .into());
    }
    Ok(Json(admin_model_status(&metadata)))
}

#[derive(Debug, Serialize, JsonSchema)]
pub(super) struct AdminModelVerifyResponse {
    status: String,
    snapshot_path: String,
    repo_id: String,
    resolved_commit: String,
    manifest_digest: String,
    verified_files: u64,
    verified_bytes: u64,
}

pub(super) async fn admin_model_verify(
    AxumPath(alias): AxumPath<String>,
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<AdminModelVerifyResponse>, EngineError> {
    require_admin(&state, &headers)?;
    let metadata = state.runtime.model_metadata();
    if alias != metadata.id {
        return Err(RuntimeError::Backend(BackendError::ModelNotFound {
            requested: alias,
            available: metadata.id,
        })
        .into());
    }
    let snapshot_path = metadata.snapshot_path.ok_or_else(|| {
        RuntimeError::Api(ApiError::unsupported_capability(
            "model verification requires snapshot metadata",
        ))
    })?;
    let verification = match ModelStore::verify_snapshot(&snapshot_path).await {
        Ok(verification) => verification,
        Err(err) => {
            record_artifact_verification_failure_metrics(&state);
            return Err(EngineError::ModelStore(err));
        }
    };
    ModelStore::mark_snapshot_used(&snapshot_path)
        .await
        .map_err(EngineError::ModelStore)?;
    Ok(Json(AdminModelVerifyResponse {
        status: "ok".to_owned(),
        snapshot_path: verification.snapshot.path.to_string_lossy().into_owned(),
        repo_id: verification.snapshot.manifest.repo_id.clone(),
        resolved_commit: verification.snapshot.manifest.resolved_commit.clone(),
        manifest_digest: verification.snapshot.manifest_digest.clone(),
        verified_files: verification.verified_files,
        verified_bytes: verification.verified_bytes,
    }))
}

#[derive(Debug, Deserialize)]
pub(super) struct AdminModelPlanRequest {
    repo_id: String,
    #[serde(default)]
    revision: Option<String>,
    #[serde(default)]
    profile: Option<String>,
    #[serde(default)]
    metadata_only: bool,
}

pub(super) async fn admin_model_plan(
    AxumPath(alias): AxumPath<String>,
    State(state): State<AppState>,
    headers: HeaderMap,
    request: Result<Json<AdminModelPlanRequest>, JsonRejection>,
) -> Result<Json<DownloadPlan>, EngineError> {
    require_admin(&state, &headers)?;
    require_model_alias(&state, &alias)?;
    let request = super::parse_json_request(request, &state)?;
    let plan = build_admin_download_plan(&state, request).await?;
    Ok(Json(plan))
}

#[derive(Debug, Serialize, JsonSchema)]
pub(super) struct AdminModelPullResponse {
    snapshot_path: String,
    manifest_digest: String,
    repo_id: String,
    resolved_commit: String,
    profile: String,
    files: usize,
}

pub(super) async fn admin_model_pull(
    AxumPath(alias): AxumPath<String>,
    State(state): State<AppState>,
    headers: HeaderMap,
    request: Result<Json<AdminModelPlanRequest>, JsonRejection>,
) -> Result<Json<AdminModelPullResponse>, EngineError> {
    require_admin(&state, &headers)?;
    require_model_alias(&state, &alias)?;
    let request = super::parse_json_request(request, &state)?;
    let plan = match build_admin_download_plan(&state, request).await {
        Ok(plan) => plan,
        Err(err) => {
            record_model_pull_failure_metrics(&state);
            return Err(err);
        }
    };
    let snapshot = match ModelStore::new(&state.model_home)
        .pull_plan(&state.hub_client, &plan, state.hf_token.as_deref())
        .await
    {
        Ok(snapshot) => snapshot,
        Err(err) => {
            record_model_pull_failure_metrics(&state);
            return Err(EngineError::ModelStore(err));
        }
    };
    let model_pull_bytes = snapshot.manifest.files.iter().map(|file| file.size).sum();
    ModelStore::mark_snapshot_used(&snapshot.path)
        .await
        .map_err(EngineError::ModelStore)?;
    ModelStore::new(&state.model_home)
        .record_snapshot_alias(&alias, &snapshot.path)
        .await
        .map_err(EngineError::ModelStore)?;
    record_model_pull_success_metrics(&state, model_pull_bytes);
    invalidate_model_store_usage_cache(&state);
    Ok(Json(AdminModelPullResponse {
        snapshot_path: snapshot.path.to_string_lossy().into_owned(),
        manifest_digest: snapshot.manifest_digest,
        repo_id: snapshot.manifest.repo_id,
        resolved_commit: snapshot.manifest.resolved_commit,
        profile: snapshot.manifest.profile,
        files: snapshot.manifest.files.len(),
    }))
}

async fn build_admin_download_plan(
    state: &AppState,
    request: AdminModelPlanRequest,
) -> Result<DownloadPlan, EngineError> {
    let repo_id = HubRepoId::model(request.repo_id).map_err(EngineError::ModelStore)?;
    let revision = request.revision.unwrap_or_else(|| "main".to_owned());
    let profile_name = request
        .profile
        .unwrap_or_else(|| "qwen36-safetensors-bf16".to_owned());
    let profile = model_profile(&profile_name)?;
    let mut plan = state
        .hub_client
        .plan_model(repo_id, &revision, profile, state.hf_token.as_deref())
        .await
        .map_err(EngineError::ModelStore)?;
    if request.metadata_only {
        plan = plan.metadata_only();
    }
    Ok(plan)
}

pub(super) fn model_profile(name: &str) -> Result<ModelProfile, EngineError> {
    ModelProfile::builtin(name).ok_or_else(|| {
        RuntimeError::Api(ApiError::invalid_request(format!(
            "unknown model profile `{name}`"
        )))
        .into()
    })
}

fn require_model_alias(state: &AppState, alias: &str) -> Result<(), EngineError> {
    let model_id = state.runtime.model_id();
    if alias == model_id {
        return Ok(());
    }
    Err(RuntimeError::Backend(BackendError::ModelNotFound {
        requested: alias.to_owned(),
        available: model_id.to_owned(),
    })
    .into())
}

#[derive(Debug, Serialize, JsonSchema)]
pub(super) struct AdminModelStatusResponse {
    id: String,
    object: String,
    status: String,
    runtime: String,
    python_runtime: bool,
    backend: String,
    family: Option<String>,
    loader: Option<String>,
    quantization: Option<String>,
    repo_id: Option<String>,
    resolved_commit: Option<String>,
    profile: Option<String>,
    snapshot_path: Option<String>,
    manifest_digest: Option<String>,
}

fn admin_model_status(metadata: &BackendModelMetadata) -> AdminModelStatusResponse {
    AdminModelStatusResponse {
        id: metadata.id.clone(),
        object: "admin.model".to_owned(),
        status: "ready".to_owned(),
        runtime: "rust".to_owned(),
        python_runtime: false,
        backend: metadata.backend.clone(),
        family: metadata.family.clone(),
        loader: metadata.loader.clone(),
        quantization: metadata.quantization.clone(),
        repo_id: metadata.repo_id.clone(),
        resolved_commit: metadata.resolved_commit.clone(),
        profile: metadata.profile.clone(),
        snapshot_path: metadata
            .snapshot_path
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned()),
        manifest_digest: metadata.manifest_digest.clone(),
    }
}

pub(super) async fn admin_metrics(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<AdminMetricsResponse>, EngineError> {
    require_admin(&state, &headers)?;
    let metrics = *state.metrics.lock_or_panic("metrics");
    let tokens = metrics.tokens();
    let request_latency = metrics.request_latency();
    let non_streamed_request_latency = metrics.non_streamed_request_latency();
    let streamed_request_latency = metrics.streamed_request_latency();
    let time_to_first_token = metrics.time_to_first_token();
    let first_tool_delta = metrics.first_tool_delta();
    let tool_argument_assembly = metrics.tool_argument_assembly();
    let tool_intent_fill = metrics.tool_intent_fill();
    let tool_schema_validation = metrics.tool_schema_validation();
    let tool_finish = metrics.tool_finish();
    let validated_tool_call = metrics.validated_tool_call();
    let model_store_usage = model_store_usage(&state).await?;
    let scheduler = state.model_scheduler.snapshot();
    let active_requests = state.active_requests.active_count();
    let backend_metrics = state.backend_metrics.snapshot();
    let response = AdminMetricsResponse {
        requests_total: metrics.requests_total(),
        successful_requests: metrics.successful_requests(),
        failed_requests: metrics.failed_requests(),
        streamed_requests: metrics.streamed_requests(),
        stream_client_disconnected_requests: metrics.stream_client_disconnected_requests(),
        stream_stalled_requests: metrics.stream_stalled_requests(),
        active_requests,
        queued_requests: scheduler.queued_total(),
        queued_prefill_requests: scheduler.queued_prefill,
        queued_decode_requests: scheduler.queued_decode,
        prefill_requests: state.generation_phases.prefill_requests(),
        decode_requests: state.generation_phases.decode_requests(),
        active_prefill_requests: scheduler.active_prefill,
        active_decode_requests: scheduler.active_decode,
        scheduler_admitted_prefill_requests: scheduler.admitted_prefill,
        scheduler_admitted_decode_requests: scheduler.admitted_decode,
        scheduler_completed_requests: scheduler.completed,
        scheduler_cancelled_requests: scheduler.cancelled,
        scheduler_failed_requests: scheduler.failed,
        scheduler_queued_cancelled_requests: scheduler.queued_cancelled,
        scheduler_queue_timeouts: scheduler.queue_timeouts,
        cancelled_requests: metrics.cancelled_requests(),
        no_progress_failures: metrics.no_progress_failures(),
        model_pull_operations: metrics.model_pull_operations(),
        model_pull_successes: metrics.model_pull_successes(),
        model_pull_failures: metrics.model_pull_failures(),
        model_pull_bytes: metrics.model_pull_bytes(),
        model_store_snapshots: model_store_usage.snapshots,
        model_store_bytes: model_store_usage.bytes,
        model_store_quarantined_snapshots: model_store_usage.quarantined_snapshots,
        model_store_quarantined_bytes: model_store_usage.quarantined_bytes,
        artifact_verification_failures: metrics.artifact_verification_failures(),
        process_rss_bytes: process_rss_bytes(),
        tokens_per_second: metrics.tokens_per_second(),
        mlx: backend_metrics.mlx,
        native_text_metal: backend_metrics.native_text_metal,
        native_text_prefix_cache: backend_metrics.native_text_prefix_cache,
        native_qwen_metal: backend_metrics.native_qwen_metal,
        native_qwen_prefix_cache: backend_metrics.native_qwen_prefix_cache,
        request_latency_ms: LatencySummary::from_metrics(request_latency),
        non_streamed_request_latency_ms: LatencySummary::from_metrics(non_streamed_request_latency),
        streamed_request_latency_ms: LatencySummary::from_metrics(streamed_request_latency),
        time_to_first_token_ms: LatencySummary::from_metrics(time_to_first_token),
        first_tool_delta_ms: LatencySummary::from_metrics(first_tool_delta),
        tool_argument_assembly_ms: LatencySummary::from_metrics(tool_argument_assembly),
        tool_intent_fill_ms: LatencySummary::from_metrics(tool_intent_fill),
        tool_schema_validation_ms: LatencySummary::from_metrics(tool_schema_validation),
        tool_finish_ms: LatencySummary::from_metrics(tool_finish),
        validated_tool_call_ms: LatencySummary::from_metrics(validated_tool_call),
        tokens: TokenSummary {
            prompt_tokens: tokens.prompt_tokens(),
            completion_tokens: tokens.completion_tokens(),
            total_tokens: tokens.total_tokens(),
            prompt_tokens_details: tokens
                .prompt_cached_tokens()
                .map(|cached_tokens| TokenPromptTokensDetailsSummary { cached_tokens }),
        },
    };
    Ok(Json(response))
}

pub(super) async fn admin_mlx_metrics(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Value>, EngineError> {
    require_admin(&state, &headers)?;
    Ok(Json(state.backend_metrics.snapshot().mlx))
}

#[derive(Debug, Serialize, JsonSchema)]
pub(super) struct AdminMetricsResponse {
    requests_total: u64,
    successful_requests: u64,
    failed_requests: u64,
    streamed_requests: u64,
    stream_client_disconnected_requests: u64,
    stream_stalled_requests: u64,
    active_requests: usize,
    queued_requests: usize,
    queued_prefill_requests: usize,
    queued_decode_requests: usize,
    prefill_requests: u64,
    decode_requests: u64,
    active_prefill_requests: usize,
    active_decode_requests: usize,
    scheduler_admitted_prefill_requests: u64,
    scheduler_admitted_decode_requests: u64,
    scheduler_completed_requests: u64,
    scheduler_cancelled_requests: u64,
    scheduler_failed_requests: u64,
    scheduler_queued_cancelled_requests: u64,
    scheduler_queue_timeouts: u64,
    cancelled_requests: u64,
    no_progress_failures: u64,
    model_pull_operations: u64,
    model_pull_successes: u64,
    model_pull_failures: u64,
    model_pull_bytes: u64,
    model_store_snapshots: usize,
    model_store_bytes: u64,
    model_store_quarantined_snapshots: usize,
    model_store_quarantined_bytes: u64,
    artifact_verification_failures: u64,
    process_rss_bytes: u64,
    tokens_per_second: f64,
    mlx: Value,
    native_text_metal: Value,
    native_text_prefix_cache: Value,
    native_qwen_metal: Value,
    native_qwen_prefix_cache: Value,
    request_latency_ms: LatencySummary,
    non_streamed_request_latency_ms: LatencySummary,
    streamed_request_latency_ms: LatencySummary,
    time_to_first_token_ms: LatencySummary,
    first_tool_delta_ms: LatencySummary,
    tool_argument_assembly_ms: LatencySummary,
    tool_intent_fill_ms: LatencySummary,
    tool_schema_validation_ms: LatencySummary,
    tool_finish_ms: LatencySummary,
    validated_tool_call_ms: LatencySummary,
    tokens: TokenSummary,
}

#[derive(Debug, Serialize, JsonSchema)]
pub(super) struct LatencySummary {
    count: u64,
    min: f64,
    max: f64,
    avg: f64,
}

impl LatencySummary {
    fn from_metrics(metrics: llm_telemetry::LatencyMetrics) -> Self {
        Self {
            count: metrics.count(),
            min: metrics.min_ms(),
            max: metrics.max_ms(),
            avg: metrics.avg_ms(),
        }
    }
}

#[derive(Debug, Serialize, JsonSchema)]
pub(super) struct TokenSummary {
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_tokens_details: Option<TokenPromptTokensDetailsSummary>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub(super) struct TokenPromptTokensDetailsSummary {
    cached_tokens: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ModelStoreUsage {
    snapshots: usize,
    bytes: u64,
    quarantined_snapshots: usize,
    quarantined_bytes: u64,
}

#[derive(Debug, Default)]
pub(super) struct ModelStoreUsageCache {
    usage: Option<ModelStoreUsage>,
    refreshed_at: Option<Instant>,
}

const MODEL_STORE_USAGE_CACHE_TTL: Duration = Duration::from_secs(30);

impl ModelStoreUsageCache {
    fn current(&self, now: Instant) -> Option<ModelStoreUsage> {
        let usage = self.usage?;
        let refreshed_at = self.refreshed_at?;
        if now.duration_since(refreshed_at) <= MODEL_STORE_USAGE_CACHE_TTL {
            Some(usage)
        } else {
            None
        }
    }

    fn store(&mut self, usage: ModelStoreUsage, refreshed_at: Instant) {
        self.usage = Some(usage);
        self.refreshed_at = Some(refreshed_at);
    }

    fn invalidate(&mut self) {
        self.usage = None;
        self.refreshed_at = None;
    }
}

async fn model_store_usage(state: &AppState) -> Result<ModelStoreUsage, EngineError> {
    let now = Instant::now();
    if let Some(usage) = state
        .model_store_usage
        .lock_or_panic("model store usage cache")
        .current(now)
    {
        return Ok(usage);
    }
    let usage = scan_model_store_usage(&state.model_home).await?;
    state
        .model_store_usage
        .lock_or_panic("model store usage cache")
        .store(usage, Instant::now());
    Ok(usage)
}

async fn scan_model_store_usage(model_home: &Path) -> Result<ModelStoreUsage, EngineError> {
    let snapshots = ModelStore::new(model_home)
        .list_snapshots()
        .await
        .map_err(EngineError::ModelStore)?;
    let quarantined = ModelStore::new(model_home)
        .list_quarantined_snapshots()
        .await
        .map_err(EngineError::ModelStore)?;
    let bytes = snapshots
        .iter()
        .flat_map(|snapshot| &snapshot.manifest.files)
        .map(|file| file.size)
        .sum();
    let quarantined_bytes = quarantined.iter().map(|snapshot| snapshot.bytes).sum();
    Ok(ModelStoreUsage {
        snapshots: snapshots.len(),
        bytes,
        quarantined_snapshots: quarantined.len(),
        quarantined_bytes,
    })
}

fn invalidate_model_store_usage_cache(state: &AppState) {
    state
        .model_store_usage
        .lock_or_panic("model store usage cache")
        .invalidate();
}

fn process_rss_bytes() -> u64 {
    platform_process_rss_bytes().unwrap_or(0)
}

#[cfg(target_os = "macos")]
#[allow(deprecated)]
fn platform_process_rss_bytes() -> Option<u64> {
    let mut info = std::mem::MaybeUninit::<libc::mach_task_basic_info>::uninit();
    let mut count = (std::mem::size_of::<libc::mach_task_basic_info>()
        / std::mem::size_of::<libc::natural_t>())
        as libc::mach_msg_type_number_t;
    let task = unsafe { libc::mach_task_self_ };
    let result = unsafe {
        libc::task_info(
            task,
            libc::MACH_TASK_BASIC_INFO,
            info.as_mut_ptr().cast(),
            &mut count,
        )
    };
    if result == libc::KERN_SUCCESS {
        let info = unsafe { info.assume_init() };
        Some(info.resident_size)
    } else {
        None
    }
}

#[cfg(target_os = "linux")]
fn platform_process_rss_bytes() -> Option<u64> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let resident_pages = statm.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_size <= 0 {
        return None;
    }
    resident_pages.checked_mul(page_size as u64)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn platform_process_rss_bytes() -> Option<u64> {
    None
}

pub(super) async fn admin_cancel_request(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(request_id): AxumPath<String>,
) -> Result<Json<Value>, EngineError> {
    require_admin(&state, &headers)?;
    let status = match state.active_requests.cancel(&request_id) {
        CancelRequestResult::Cancelled => {
            record_cancellation_metrics(&state);
            "cancelled"
        }
        CancelRequestResult::AlreadyCancelled => "already_cancelled",
        CancelRequestResult::Finished => "already_finished",
        CancelRequestResult::NotFound => return Err(EngineError::RequestNotFound(request_id)),
    };
    Ok(Json(json!({
        "object": "admin.request_cancellation",
        "request_id": request_id,
        "status": status
    })))
}

fn require_admin(state: &AppState, headers: &HeaderMap) -> Result<(), EngineError> {
    let Some(token) = &state.admin_token else {
        if state.allow_unauthenticated_admin {
            return Ok(());
        }
        return Err(EngineError::UnauthorizedAdmin);
    };
    let Some(header_value) = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
    else {
        return Err(EngineError::UnauthorizedAdmin);
    };
    let Some(header_token) = header_value.strip_prefix("Bearer ") else {
        return Err(EngineError::UnauthorizedAdmin);
    };
    if header_token.as_bytes().ct_eq(token.as_bytes()).into() {
        return Ok(());
    }
    Err(EngineError::UnauthorizedAdmin)
}
