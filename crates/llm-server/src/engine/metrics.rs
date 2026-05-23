use super::AppState;
use crate::sync_ext::FailPoisonedMutex;
use llm_api::Usage;
use llm_backend_contracts::{BackendStreamProgress, BackendStreamTimingMilestone};
use llm_runtime::{RequestCacheIdentity, RuntimeError};
use llm_telemetry::TokenCounters;
use schemars::JsonSchema;
use serde::Serialize;
use std::{collections::VecDeque, time::Duration};

pub(super) const REQUEST_CACHE_OBSERVATION_CAPACITY: usize = 128;
pub(super) const TOOL_STREAM_OBSERVATION_CAPACITY: usize = 128;

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
    fn shares_stable_prefix_with(&self, prior: &Self) -> bool {
        self.stable_prefix_key.is_some()
            && self.stable_prefix_key == prior.stable_prefix_key
            && self.model == prior.model
            && self.cache_key == prior.cache_key
            && self.cache_template_id == prior.cache_template_id
    }

    fn has_same_prompt_as(&self, prior: &Self) -> bool {
        self.prompt_hash.is_some() && self.prompt_hash == prior.prompt_hash
    }

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

    pub(super) fn record(&mut self, mut observation: RequestCacheObservation) {
        if self.capacity == 0 {
            return;
        }
        self.apply_history_fallback(&mut observation);
        while self.recent.len() >= self.capacity {
            self.recent.pop_front();
        }
        self.recent.push_back(observation);
    }

    fn apply_history_fallback(&self, observation: &mut RequestCacheObservation) {
        if observation.cache_status != RequestCacheStatus::Unknown {
            return;
        }
        let Some(prior) = self
            .recent
            .iter()
            .rev()
            .find(|prior| observation.shares_stable_prefix_with(prior))
        else {
            return;
        };
        observation.cache_status = if observation.has_same_prompt_as(prior) {
            observation.cached_tokens = Some(observation.prompt_tokens);
            observation.uncached_tokens = Some(0);
            RequestCacheStatus::Hit
        } else {
            RequestCacheStatus::Partial
        };
        tracing::debug!(
            request_id = %observation.request_id,
            model = %observation.model,
            cache_status = ?observation.cache_status,
            stable_prefix_key = observation.stable_prefix_key.as_deref(),
            "derived request cache status from prior cache identity"
        );
    }

    pub(super) fn snapshot(&self) -> RequestCacheSnapshot {
        RequestCacheSnapshot {
            capacity: self.capacity,
            recent: self.recent.iter().cloned().collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, JsonSchema)]
pub(super) struct ToolStreamObservation {
    pub(super) request_id: String,
    pub(super) model: String,
    pub(super) streamed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) kir_first_tool_delta_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) kir_first_tool_delta_after_ttft_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) tool_argument_assembly_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) tool_intent_fill_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) tool_schema_validation_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) tool_finish_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) validated_tool_call_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) mlx_response_headers_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) mlx_first_upstream_byte_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) mlx_first_parsed_chunk_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) mlx_first_tool_delta_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) mlx_upstream_complete_ms: Option<u64>,
    pub(super) latency_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, JsonSchema)]
pub(super) struct ToolStreamSnapshot {
    pub(super) capacity: usize,
    pub(super) recent: Vec<ToolStreamObservation>,
}

#[derive(Debug)]
pub(super) struct ToolStreamObservations {
    capacity: usize,
    recent: VecDeque<ToolStreamObservation>,
}

impl Default for ToolStreamObservations {
    fn default() -> Self {
        Self::with_capacity(TOOL_STREAM_OBSERVATION_CAPACITY)
    }
}

impl ToolStreamObservations {
    pub(super) fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity,
            recent: VecDeque::with_capacity(capacity),
        }
    }

    pub(super) fn record(&mut self, observation: ToolStreamObservation) {
        if self.capacity == 0 {
            return;
        }
        while self.recent.len() >= self.capacity {
            self.recent.pop_front();
        }
        self.recent.push_back(observation);
    }

    pub(super) fn snapshot(&self) -> ToolStreamSnapshot {
        ToolStreamSnapshot {
            capacity: self.capacity,
            recent: self.recent.iter().cloned().collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct PendingToolStreamObservation {
    request_id: String,
    model: String,
    streamed: bool,
    kir_first_tool_delta_ms: Option<u64>,
    kir_first_tool_delta_after_ttft_ms: Option<u64>,
    tool_argument_assembly_ms: Option<u64>,
    tool_intent_fill_ms: Option<u64>,
    tool_schema_validation_ms: Option<u64>,
    tool_finish_ms: Option<u64>,
    validated_tool_call_ms: Option<u64>,
    mlx_response_headers_ms: Option<u64>,
    mlx_first_upstream_byte_ms: Option<u64>,
    mlx_first_parsed_chunk_ms: Option<u64>,
    mlx_first_tool_delta_ms: Option<u64>,
    mlx_upstream_complete_ms: Option<u64>,
}

impl PendingToolStreamObservation {
    pub(super) fn new(request_id: String, model: String, streamed: bool) -> Self {
        Self {
            request_id,
            model,
            streamed,
            kir_first_tool_delta_ms: None,
            kir_first_tool_delta_after_ttft_ms: None,
            tool_argument_assembly_ms: None,
            tool_intent_fill_ms: None,
            tool_schema_validation_ms: None,
            tool_finish_ms: None,
            validated_tool_call_ms: None,
            mlx_response_headers_ms: None,
            mlx_first_upstream_byte_ms: None,
            mlx_first_parsed_chunk_ms: None,
            mlx_first_tool_delta_ms: None,
            mlx_upstream_complete_ms: None,
        }
    }

    pub(super) fn record_kir_first_tool_delta(&mut self, latency: Duration) {
        set_once_ms(&mut self.kir_first_tool_delta_ms, latency);
    }

    pub(super) fn record_kir_first_tool_delta_after_ttft(&mut self, latency: Duration) {
        set_once_ms(&mut self.kir_first_tool_delta_after_ttft_ms, latency);
    }

    pub(super) fn record_tool_argument_assembly(&mut self, latency: Duration) {
        set_once_ms(&mut self.tool_argument_assembly_ms, latency);
    }

    pub(super) fn record_tool_intent_fill(&mut self, latency: Duration) {
        set_once_ms(&mut self.tool_intent_fill_ms, latency);
    }

    pub(super) fn record_tool_schema_validation(&mut self, latency: Duration) {
        set_once_ms(&mut self.tool_schema_validation_ms, latency);
    }

    pub(super) fn record_tool_finish(&mut self, latency: Duration) {
        set_once_ms(&mut self.tool_finish_ms, latency);
    }

    pub(super) fn record_validated_tool_call(&mut self, latency: Duration) {
        set_once_ms(&mut self.validated_tool_call_ms, latency);
    }

    pub(super) fn record_backend_progress(&mut self, progress: &BackendStreamProgress) -> bool {
        let BackendStreamProgress::MlxStreamTiming {
            milestone,
            latency_ms,
        } = progress
        else {
            return false;
        };
        match milestone {
            BackendStreamTimingMilestone::ResponseHeaders => {
                set_once(&mut self.mlx_response_headers_ms, *latency_ms);
            }
            BackendStreamTimingMilestone::FirstUpstreamByte => {
                set_once(&mut self.mlx_first_upstream_byte_ms, *latency_ms);
            }
            BackendStreamTimingMilestone::FirstParsedChunk => {
                set_once(&mut self.mlx_first_parsed_chunk_ms, *latency_ms);
            }
            BackendStreamTimingMilestone::FirstToolDelta => {
                set_once(&mut self.mlx_first_tool_delta_ms, *latency_ms);
            }
            BackendStreamTimingMilestone::UpstreamComplete => {
                set_once(&mut self.mlx_upstream_complete_ms, *latency_ms);
            }
        }
        true
    }

    pub(super) fn to_observation(&self, latency: Duration) -> Option<ToolStreamObservation> {
        if !self.streamed || !self.observed_tool_stream() {
            return None;
        }
        Some(ToolStreamObservation {
            request_id: self.request_id.clone(),
            model: self.model.clone(),
            streamed: self.streamed,
            kir_first_tool_delta_ms: self.kir_first_tool_delta_ms,
            kir_first_tool_delta_after_ttft_ms: self.kir_first_tool_delta_after_ttft_ms,
            tool_argument_assembly_ms: self.tool_argument_assembly_ms,
            tool_intent_fill_ms: self.tool_intent_fill_ms,
            tool_schema_validation_ms: self.tool_schema_validation_ms,
            tool_finish_ms: self.tool_finish_ms,
            validated_tool_call_ms: self.validated_tool_call_ms,
            mlx_response_headers_ms: self.mlx_response_headers_ms,
            mlx_first_upstream_byte_ms: self.mlx_first_upstream_byte_ms,
            mlx_first_parsed_chunk_ms: self.mlx_first_parsed_chunk_ms,
            mlx_first_tool_delta_ms: self.mlx_first_tool_delta_ms,
            mlx_upstream_complete_ms: self.mlx_upstream_complete_ms,
            latency_ms: duration_millis_u64(latency),
        })
    }

    fn observed_tool_stream(&self) -> bool {
        self.kir_first_tool_delta_ms.is_some()
            || self.kir_first_tool_delta_after_ttft_ms.is_some()
            || self.tool_argument_assembly_ms.is_some()
            || self.tool_intent_fill_ms.is_some()
            || self.tool_schema_validation_ms.is_some()
            || self.tool_finish_ms.is_some()
            || self.validated_tool_call_ms.is_some()
            || self.mlx_first_tool_delta_ms.is_some()
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

pub(super) fn record_tool_stream_observation(state: &AppState, observation: ToolStreamObservation) {
    state
        .tool_stream
        .lock_or_panic("tool stream observations")
        .record(observation);
}

fn duration_millis_u64(latency: Duration) -> u64 {
    u64::try_from(latency.as_millis()).unwrap_or(u64::MAX)
}

fn set_once_ms(target: &mut Option<u64>, latency: Duration) {
    set_once(target, duration_millis_u64(latency));
}

fn set_once(target: &mut Option<u64>, value: u64) {
    if target.is_none() {
        *target = Some(value);
    }
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
pub(super) fn record_first_tool_delta_after_ttft_metrics(state: &AppState, latency: Duration) {
    state
        .metrics
        .lock_or_panic("metrics")
        .record_first_tool_delta_after_ttft(latency);
}

#[cfg(not(feature = "tool-calls"))]
pub(super) fn record_first_tool_delta_after_ttft_metrics(_state: &AppState, _latency: Duration) {}

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

    #[test]
    fn tool_stream_observations_evict_oldest_at_capacity() {
        let mut observations = ToolStreamObservations::with_capacity(2);
        for index in 0..3 {
            observations.record(ToolStreamObservation {
                request_id: format!("request-{index}"),
                model: "model".to_owned(),
                streamed: true,
                kir_first_tool_delta_ms: Some(index),
                kir_first_tool_delta_after_ttft_ms: Some(index.saturating_sub(1)),
                tool_argument_assembly_ms: None,
                tool_intent_fill_ms: None,
                tool_schema_validation_ms: None,
                tool_finish_ms: None,
                validated_tool_call_ms: Some(index + 1),
                mlx_response_headers_ms: None,
                mlx_first_upstream_byte_ms: None,
                mlx_first_parsed_chunk_ms: None,
                mlx_first_tool_delta_ms: None,
                mlx_upstream_complete_ms: None,
                latency_ms: index + 2,
            });
        }

        let snapshot = observations.snapshot();
        assert_eq!(snapshot.capacity, 2);
        assert_eq!(
            snapshot
                .recent
                .iter()
                .map(|observation| observation.request_id.as_str())
                .collect::<Vec<_>>(),
            ["request-1", "request-2"]
        );
    }
}
