use super::AppState;
use crate::sync_ext::FailPoisonedMutex;
use llm_api::Usage;
use llm_runtime::{RequestCacheIdentity, RuntimeError};
use llm_telemetry::TokenCounters;
use schemars::JsonSchema;
use serde::Serialize;
use std::{collections::VecDeque, time::Duration};

pub(super) const REQUEST_CACHE_OBSERVATION_CAPACITY: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(super) enum RequestCacheStatus {
    Unknown,
    Miss,
    Partial,
    Hit,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, JsonSchema)]
pub(super) struct RequestCacheObservation {
    pub(super) request_id: String,
    pub(super) model: String,
    pub(super) streamed: bool,
    pub(super) prompt_tokens: u64,
    pub(super) cached_tokens: Option<u64>,
    pub(super) uncached_tokens: Option<u64>,
    pub(super) cache_status: RequestCacheStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) prompt_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) cache_template_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) model_family: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) tool_schema_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) system_prompt_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) chat_template_kwargs_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) stable_prefix_key: Option<String>,
    pub(super) latency_ms: u64,
}

impl RequestCacheObservation {
    pub(super) fn from_usage(
        request_id: &str,
        model: &str,
        streamed: bool,
        usage: &Usage,
        latency: Duration,
        identity: Option<&RequestCacheIdentity>,
    ) -> Self {
        let cached_tokens = usage
            .prompt_tokens_details
            .as_ref()
            .map(|details| details.cached_tokens);
        let cache_status = match cached_tokens {
            None => RequestCacheStatus::Unknown,
            Some(0) => RequestCacheStatus::Miss,
            Some(cached) if cached >= usage.prompt_tokens => RequestCacheStatus::Hit,
            Some(_) => RequestCacheStatus::Partial,
        };
        Self {
            request_id: request_id.to_owned(),
            model: model.to_owned(),
            streamed,
            prompt_tokens: usage.prompt_tokens,
            cached_tokens,
            uncached_tokens: cached_tokens.map(|cached| usage.prompt_tokens.saturating_sub(cached)),
            cache_status,
            prompt_hash: identity.map(|identity| identity.prompt_hash.clone()),
            cache_key: identity.map(|identity| identity.cache_key.clone()),
            cache_template_id: identity.map(|identity| identity.cache_template_id.clone()),
            model_family: identity.and_then(|identity| identity.model_family.clone()),
            tool_schema_hash: identity.and_then(|identity| identity.tool_schema_hash.clone()),
            system_prompt_hash: identity.and_then(|identity| identity.system_prompt_hash.clone()),
            chat_template_kwargs_hash: identity
                .and_then(|identity| identity.chat_template_kwargs_hash.clone()),
            stable_prefix_key: identity.and_then(|identity| identity.stable_prefix_key.clone()),
            latency_ms: duration_millis_u64(latency),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, JsonSchema)]
pub(super) struct RequestCacheSnapshot {
    pub(super) capacity: usize,
    pub(super) recent: Vec<RequestCacheObservation>,
}

#[derive(Debug)]
pub(super) struct RequestCacheObservations {
    capacity: usize,
    recent: VecDeque<RequestCacheObservation>,
}

impl Default for RequestCacheObservations {
    fn default() -> Self {
        Self::with_capacity(REQUEST_CACHE_OBSERVATION_CAPACITY)
    }
}

impl RequestCacheObservations {
    pub(super) fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity,
            recent: VecDeque::with_capacity(capacity),
        }
    }

    pub(super) fn record(&mut self, observation: RequestCacheObservation) {
        if self.capacity == 0 {
            return;
        }
        while self.recent.len() >= self.capacity {
            self.recent.pop_front();
        }
        self.recent.push_back(observation);
    }

    pub(super) fn snapshot(&self) -> RequestCacheSnapshot {
        RequestCacheSnapshot {
            capacity: self.capacity,
            recent: self.recent.iter().cloned().collect(),
        }
    }
}

pub(super) fn record_success_metrics(
    state: &AppState,
    request_id: &str,
    model: &str,
    usage: &Usage,
    streamed: bool,
    latency: Duration,
    cache_identity: Option<&RequestCacheIdentity>,
) {
    let prompt_cached_tokens = usage
        .prompt_tokens_details
        .as_ref()
        .map(|details| details.cached_tokens);
    state.metrics.lock_or_panic("metrics").record_success(
        TokenCounters::new(usage.prompt_tokens, usage.completion_tokens)
            .with_prompt_cached_tokens(prompt_cached_tokens),
        streamed,
        latency,
    );
    state
        .request_cache
        .lock_or_panic("request cache observations")
        .record(RequestCacheObservation::from_usage(
            request_id,
            model,
            streamed,
            usage,
            latency,
            cache_identity,
        ));
}

fn duration_millis_u64(latency: Duration) -> u64 {
    u64::try_from(latency.as_millis()).unwrap_or(u64::MAX)
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

#[cfg(feature = "tool-calls")]
pub(super) fn record_first_tool_delta_metrics(state: &AppState, latency: Duration) {
    state
        .metrics
        .lock_or_panic("metrics")
        .record_first_tool_delta(latency);
}

#[cfg(not(feature = "tool-calls"))]
pub(super) fn record_first_tool_delta_metrics(_state: &AppState, _latency: Duration) {}

#[cfg(feature = "tool-calls")]
pub(super) fn record_tool_argument_assembly_metrics(state: &AppState, latency: Duration) {
    state
        .metrics
        .lock_or_panic("metrics")
        .record_tool_argument_assembly(latency);
}

#[cfg(not(feature = "tool-calls"))]
pub(super) fn record_tool_argument_assembly_metrics(_state: &AppState, _latency: Duration) {}

#[cfg(feature = "tool-calls")]
pub(super) fn record_tool_intent_fill_metrics(state: &AppState, latency: Duration) {
    state
        .metrics
        .lock_or_panic("metrics")
        .record_tool_intent_fill(latency);
}

#[cfg(not(feature = "tool-calls"))]
pub(super) fn record_tool_intent_fill_metrics(_state: &AppState, _latency: Duration) {}

#[cfg(feature = "tool-calls")]
pub(super) fn record_tool_schema_validation_metrics(state: &AppState, latency: Duration) {
    state
        .metrics
        .lock_or_panic("metrics")
        .record_tool_schema_validation(latency);
}

#[cfg(not(feature = "tool-calls"))]
pub(super) fn record_tool_schema_validation_metrics(_state: &AppState, _latency: Duration) {}

#[cfg(feature = "tool-calls")]
pub(super) fn record_tool_finish_metrics(state: &AppState, latency: Duration) {
    state
        .metrics
        .lock_or_panic("metrics")
        .record_tool_finish(latency);
}

#[cfg(not(feature = "tool-calls"))]
pub(super) fn record_tool_finish_metrics(_state: &AppState, _latency: Duration) {}

#[cfg(feature = "tool-calls")]
pub(super) fn record_validated_tool_call_metrics(state: &AppState, latency: Duration) {
    state
        .metrics
        .lock_or_panic("metrics")
        .record_validated_tool_call(latency);
}

#[cfg(not(feature = "tool-calls"))]
pub(super) fn record_validated_tool_call_metrics(_state: &AppState, _latency: Duration) {}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_api::Usage;

    #[test]
    fn request_cache_observation_classifies_cached_token_reuse() {
        let miss = RequestCacheObservation::from_usage(
            "miss",
            "model",
            false,
            &Usage::new(10, 1).with_prompt_cached_tokens(Some(0)),
            Duration::from_millis(7),
            None,
        );
        let partial = RequestCacheObservation::from_usage(
            "partial",
            "model",
            false,
            &Usage::new(10, 1).with_prompt_cached_tokens(Some(4)),
            Duration::from_millis(7),
            None,
        );
        let hit = RequestCacheObservation::from_usage(
            "hit",
            "model",
            false,
            &Usage::new(10, 1).with_prompt_cached_tokens(Some(10)),
            Duration::from_millis(7),
            None,
        );
        let overshoot = RequestCacheObservation::from_usage(
            "overshoot",
            "model",
            false,
            &Usage::new(10, 1).with_prompt_cached_tokens(Some(12)),
            Duration::from_millis(7),
            None,
        );
        let unknown = RequestCacheObservation::from_usage(
            "unknown",
            "model",
            false,
            &Usage::new(10, 1),
            Duration::from_millis(7),
            None,
        );

        assert_eq!(miss.cache_status, RequestCacheStatus::Miss);
        assert_eq!(miss.uncached_tokens, Some(10));
        assert_eq!(partial.cache_status, RequestCacheStatus::Partial);
        assert_eq!(partial.uncached_tokens, Some(6));
        assert_eq!(hit.cache_status, RequestCacheStatus::Hit);
        assert_eq!(hit.uncached_tokens, Some(0));
        assert_eq!(overshoot.cache_status, RequestCacheStatus::Hit);
        assert_eq!(overshoot.uncached_tokens, Some(0));
        assert_eq!(unknown.cache_status, RequestCacheStatus::Unknown);
        assert_eq!(unknown.cached_tokens, None);
        assert_eq!(unknown.uncached_tokens, None);
    }

    #[test]
    fn request_cache_observations_evict_oldest_at_capacity() {
        let mut observations = RequestCacheObservations::with_capacity(3);
        for index in 0..4 {
            observations.record(RequestCacheObservation::from_usage(
                &format!("request-{index}"),
                "model",
                false,
                &Usage::new(10, 1).with_prompt_cached_tokens(Some(index)),
                Duration::from_millis(index),
                None,
            ));
        }

        let snapshot = observations.snapshot();
        assert_eq!(snapshot.capacity, 3);
        assert_eq!(
            snapshot
                .recent
                .iter()
                .map(|observation| observation.request_id.as_str())
                .collect::<Vec<_>>(),
            ["request-1", "request-2", "request-3"]
        );
    }
}
