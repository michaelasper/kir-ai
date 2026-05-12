use super::AppState;
use crate::sync_ext::FailPoisonedMutex;
use llm_api::Usage;
use llm_runtime::RuntimeError;
use llm_telemetry::TokenCounters;
use std::time::Duration;

pub(super) fn record_success_metrics(
    state: &AppState,
    usage: &Usage,
    streamed: bool,
    latency: Duration,
) {
    state.metrics.lock_or_panic("metrics").record_success(
        TokenCounters::new(usage.prompt_tokens, usage.completion_tokens),
        streamed,
        latency,
    );
}

pub(super) fn record_failure_metrics(state: &AppState) {
    state.metrics.lock_or_panic("metrics").record_failure();
}

pub(super) fn record_runtime_error_metrics(state: &AppState, err: &RuntimeError) {
    let mut metrics = state.metrics.lock_or_panic("metrics");
    if matches!(err, RuntimeError::NoProgress(_)) {
        metrics.record_no_progress_failure();
    }
    metrics.record_failure();
}

pub(super) fn record_cancellation_metrics(state: &AppState) {
    state.metrics.lock_or_panic("metrics").record_cancellation();
}

pub(super) fn record_stream_client_disconnect_metrics(state: &AppState) {
    state
        .metrics
        .lock_or_panic("metrics")
        .record_stream_client_disconnect();
}

pub(super) fn record_stream_stall_metrics(state: &AppState) {
    state.metrics.lock_or_panic("metrics").record_stream_stall();
}

pub(super) fn record_model_pull_success_metrics(state: &AppState, bytes: u64) {
    state
        .metrics
        .lock_or_panic("metrics")
        .record_model_pull_success(bytes);
}

pub(super) fn record_model_pull_failure_metrics(state: &AppState) {
    state
        .metrics
        .lock_or_panic("metrics")
        .record_model_pull_failure();
}

pub(super) fn record_artifact_verification_failure_metrics(state: &AppState) {
    state
        .metrics
        .lock_or_panic("metrics")
        .record_artifact_verification_failure();
}

pub(super) fn record_time_to_first_token_metrics(state: &AppState, latency: Duration) {
    state
        .metrics
        .lock_or_panic("metrics")
        .record_time_to_first_token(latency);
}
