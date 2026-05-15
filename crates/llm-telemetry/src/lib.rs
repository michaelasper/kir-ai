use std::time::Duration;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TokenCounters {
    prompt_tokens: u64,
    completion_tokens: u64,
    prompt_cached_tokens: Option<u64>,
}

impl TokenCounters {
    pub fn new(prompt_tokens: u64, completion_tokens: u64) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            prompt_cached_tokens: None,
        }
    }

    pub fn with_prompt_cached_tokens(mut self, cached_tokens: Option<u64>) -> Self {
        self.prompt_cached_tokens = cached_tokens;
        self
    }

    pub fn prompt_tokens(&self) -> u64 {
        self.prompt_tokens
    }

    pub fn completion_tokens(&self) -> u64 {
        self.completion_tokens
    }

    pub fn prompt_cached_tokens(&self) -> Option<u64> {
        self.prompt_cached_tokens
    }

    pub fn total_tokens(&self) -> u64 {
        self.prompt_tokens + self.completion_tokens
    }

    pub fn record_prompt_tokens(&mut self, tokens: u64) {
        self.prompt_tokens += tokens;
    }

    pub fn record_completion_tokens(&mut self, tokens: u64) {
        self.completion_tokens += tokens;
    }

    pub fn record_prompt_cached_tokens(&mut self, tokens: Option<u64>) {
        if let Some(tokens) = tokens {
            self.prompt_cached_tokens = Some(self.prompt_cached_tokens.unwrap_or(0) + tokens);
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LatencyMetrics {
    count: u64,
    total_nanos: u128,
    min_nanos: Option<u128>,
    max_nanos: u128,
}

impl LatencyMetrics {
    pub fn record(&mut self, duration: Duration) {
        let nanos = duration.as_nanos();
        self.count += 1;
        self.total_nanos += nanos;
        self.min_nanos = Some(self.min_nanos.map_or(nanos, |current| current.min(nanos)));
        self.max_nanos = self.max_nanos.max(nanos);
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    pub fn min_ms(&self) -> f64 {
        nanos_to_ms(self.min_nanos.unwrap_or(0))
    }

    pub fn max_ms(&self) -> f64 {
        nanos_to_ms(self.max_nanos)
    }

    pub fn avg_ms(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            nanos_to_ms(self.total_nanos / u128::from(self.count))
        }
    }

    pub fn total_seconds(&self) -> f64 {
        self.total_nanos as f64 / 1_000_000_000.0
    }
}

fn nanos_to_ms(nanos: u128) -> f64 {
    nanos as f64 / 1_000_000.0
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestMetrics {
    model: String,
    tokens: TokenCounters,
}

impl RequestMetrics {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            tokens: TokenCounters::default(),
        }
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn tokens(&self) -> TokenCounters {
        self.tokens
    }

    pub fn record_prompt_tokens(&mut self, tokens: u64) {
        self.tokens.record_prompt_tokens(tokens);
    }

    pub fn record_completion_tokens(&mut self, tokens: u64) {
        self.tokens.record_completion_tokens(tokens);
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ServerMetrics {
    requests_total: u64,
    successful_requests: u64,
    failed_requests: u64,
    streamed_requests: u64,
    stream_client_disconnected_requests: u64,
    stream_stalled_requests: u64,
    cancelled_requests: u64,
    no_progress_failures: u64,
    model_pull_operations: u64,
    model_pull_successes: u64,
    model_pull_failures: u64,
    model_pull_bytes: u64,
    artifact_verification_failures: u64,
    request_latency: LatencyMetrics,
    non_streamed_request_latency: LatencyMetrics,
    streamed_request_latency: LatencyMetrics,
    time_to_first_token: LatencyMetrics,
    #[cfg(feature = "tool-calls")]
    first_tool_delta: LatencyMetrics,
    #[cfg(feature = "tool-calls")]
    tool_argument_assembly: LatencyMetrics,
    #[cfg(feature = "tool-calls")]
    tool_intent_fill: LatencyMetrics,
    #[cfg(feature = "tool-calls")]
    tool_schema_validation: LatencyMetrics,
    #[cfg(feature = "tool-calls")]
    tool_finish: LatencyMetrics,
    #[cfg(feature = "tool-calls")]
    validated_tool_call: LatencyMetrics,
    tokens: TokenCounters,
}

impl ServerMetrics {
    pub fn record_success(&mut self, tokens: TokenCounters, streamed: bool, latency: Duration) {
        self.requests_total += 1;
        self.successful_requests += 1;
        if streamed {
            self.streamed_requests += 1;
            self.streamed_request_latency.record(latency);
        } else {
            self.non_streamed_request_latency.record(latency);
        }
        self.request_latency.record(latency);
        self.tokens.record_prompt_tokens(tokens.prompt_tokens());
        self.tokens
            .record_completion_tokens(tokens.completion_tokens());
        self.tokens
            .record_prompt_cached_tokens(tokens.prompt_cached_tokens());
    }

    pub fn record_failure(&mut self) {
        self.requests_total += 1;
        self.failed_requests += 1;
    }

    pub fn record_stream_client_disconnect(&mut self) {
        self.requests_total += 1;
        self.failed_requests += 1;
        self.stream_client_disconnected_requests += 1;
    }

    pub fn record_stream_stall(&mut self) {
        self.requests_total += 1;
        self.failed_requests += 1;
        self.stream_stalled_requests += 1;
    }

    pub fn record_cancellation(&mut self) {
        self.cancelled_requests += 1;
    }

    pub fn record_no_progress_failure(&mut self) {
        self.no_progress_failures += 1;
    }

    pub fn record_model_pull_success(&mut self, bytes: u64) {
        self.model_pull_operations += 1;
        self.model_pull_successes += 1;
        self.model_pull_bytes += bytes;
    }

    pub fn record_model_pull_failure(&mut self) {
        self.model_pull_operations += 1;
        self.model_pull_failures += 1;
    }

    pub fn record_artifact_verification_failure(&mut self) {
        self.artifact_verification_failures += 1;
    }

    pub fn record_time_to_first_token(&mut self, latency: Duration) {
        self.time_to_first_token.record(latency);
    }

    #[cfg(feature = "tool-calls")]
    pub fn record_first_tool_delta(&mut self, latency: Duration) {
        self.first_tool_delta.record(latency);
    }

    #[cfg(feature = "tool-calls")]
    pub fn record_tool_argument_assembly(&mut self, latency: Duration) {
        self.tool_argument_assembly.record(latency);
    }

    #[cfg(feature = "tool-calls")]
    pub fn record_tool_intent_fill(&mut self, latency: Duration) {
        self.tool_intent_fill.record(latency);
    }

    #[cfg(feature = "tool-calls")]
    pub fn record_tool_schema_validation(&mut self, latency: Duration) {
        self.tool_schema_validation.record(latency);
    }

    #[cfg(feature = "tool-calls")]
    pub fn record_tool_finish(&mut self, latency: Duration) {
        self.tool_finish.record(latency);
    }

    #[cfg(feature = "tool-calls")]
    pub fn record_validated_tool_call(&mut self, latency: Duration) {
        self.validated_tool_call.record(latency);
    }

    pub fn requests_total(&self) -> u64 {
        self.requests_total
    }

    pub fn successful_requests(&self) -> u64 {
        self.successful_requests
    }

    pub fn failed_requests(&self) -> u64 {
        self.failed_requests
    }

    pub fn streamed_requests(&self) -> u64 {
        self.streamed_requests
    }

    pub fn stream_client_disconnected_requests(&self) -> u64 {
        self.stream_client_disconnected_requests
    }

    pub fn stream_stalled_requests(&self) -> u64 {
        self.stream_stalled_requests
    }

    pub fn cancelled_requests(&self) -> u64 {
        self.cancelled_requests
    }

    pub fn no_progress_failures(&self) -> u64 {
        self.no_progress_failures
    }

    pub fn model_pull_operations(&self) -> u64 {
        self.model_pull_operations
    }

    pub fn model_pull_successes(&self) -> u64 {
        self.model_pull_successes
    }

    pub fn model_pull_failures(&self) -> u64 {
        self.model_pull_failures
    }

    pub fn model_pull_bytes(&self) -> u64 {
        self.model_pull_bytes
    }

    pub fn artifact_verification_failures(&self) -> u64 {
        self.artifact_verification_failures
    }

    pub fn request_latency(&self) -> LatencyMetrics {
        self.request_latency
    }

    pub fn non_streamed_request_latency(&self) -> LatencyMetrics {
        self.non_streamed_request_latency
    }

    pub fn streamed_request_latency(&self) -> LatencyMetrics {
        self.streamed_request_latency
    }

    pub fn time_to_first_token(&self) -> LatencyMetrics {
        self.time_to_first_token
    }

    #[cfg(feature = "tool-calls")]
    pub fn first_tool_delta(&self) -> LatencyMetrics {
        self.first_tool_delta
    }

    #[cfg(feature = "tool-calls")]
    pub fn tool_argument_assembly(&self) -> LatencyMetrics {
        self.tool_argument_assembly
    }

    #[cfg(feature = "tool-calls")]
    pub fn tool_intent_fill(&self) -> LatencyMetrics {
        self.tool_intent_fill
    }

    #[cfg(feature = "tool-calls")]
    pub fn tool_schema_validation(&self) -> LatencyMetrics {
        self.tool_schema_validation
    }

    #[cfg(feature = "tool-calls")]
    pub fn tool_finish(&self) -> LatencyMetrics {
        self.tool_finish
    }

    #[cfg(feature = "tool-calls")]
    pub fn validated_tool_call(&self) -> LatencyMetrics {
        self.validated_tool_call
    }

    pub fn tokens_per_second(&self) -> f64 {
        let seconds = self.request_latency.total_seconds();
        if seconds == 0.0 {
            0.0
        } else {
            self.tokens.total_tokens() as f64 / seconds
        }
    }

    pub fn tokens(&self) -> TokenCounters {
        self.tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_counters_report_totals() {
        let mut counters = TokenCounters::new(2, 3);

        counters.record_prompt_tokens(5);
        counters.record_completion_tokens(7);

        assert_eq!(counters.prompt_tokens(), 7);
        assert_eq!(counters.completion_tokens(), 10);
        assert_eq!(counters.total_tokens(), 17);
        assert_eq!(counters.prompt_cached_tokens(), None);
    }

    #[test]
    fn token_counters_sum_cached_prompt_tokens_when_reported() {
        let mut counters = TokenCounters::new(2, 3).with_prompt_cached_tokens(Some(1));

        counters.record_prompt_cached_tokens(Some(4));
        counters.record_prompt_cached_tokens(None);
        counters.record_prompt_tokens(5);
        counters.record_completion_tokens(7);

        assert_eq!(counters.prompt_tokens(), 7);
        assert_eq!(counters.completion_tokens(), 10);
        assert_eq!(counters.total_tokens(), 17);
        assert_eq!(counters.prompt_cached_tokens(), Some(5));
    }

    #[test]
    fn request_metrics_tracks_model_and_tokens() {
        let mut metrics = RequestMetrics::new("local-qwen36");

        metrics.record_prompt_tokens(4);
        metrics.record_completion_tokens(1);

        assert_eq!(metrics.model(), "local-qwen36");
        assert_eq!(metrics.tokens(), TokenCounters::new(4, 1));
    }

    #[test]
    fn server_metrics_tracks_success_failure_streams_and_tokens() {
        let mut metrics = ServerMetrics::default();

        metrics.record_success(
            TokenCounters::new(4, 1).with_prompt_cached_tokens(Some(6)),
            false,
            Duration::from_millis(10),
        );
        metrics.record_success(
            TokenCounters::new(8, 2).with_prompt_cached_tokens(Some(9)),
            true,
            Duration::from_millis(30),
        );
        metrics.record_failure();

        assert_eq!(metrics.requests_total(), 3);
        assert_eq!(metrics.successful_requests(), 2);
        assert_eq!(metrics.failed_requests(), 1);
        assert_eq!(metrics.streamed_requests(), 1);
        assert_eq!(metrics.stream_client_disconnected_requests(), 0);
        assert_eq!(metrics.stream_stalled_requests(), 0);
        assert_eq!(metrics.cancelled_requests(), 0);
        assert_eq!(metrics.no_progress_failures(), 0);
        assert_eq!(metrics.model_pull_operations(), 0);
        assert_eq!(metrics.model_pull_successes(), 0);
        assert_eq!(metrics.model_pull_failures(), 0);
        assert_eq!(metrics.model_pull_bytes(), 0);
        assert_eq!(metrics.artifact_verification_failures(), 0);
        assert_eq!(
            metrics.tokens(),
            TokenCounters::new(12, 3).with_prompt_cached_tokens(Some(15))
        );
        assert_eq!(metrics.request_latency().count(), 2);
        assert_eq!(metrics.request_latency().min_ms(), 10.0);
        assert_eq!(metrics.request_latency().max_ms(), 30.0);
        assert_eq!(metrics.request_latency().avg_ms(), 20.0);
        assert_eq!(metrics.non_streamed_request_latency().count(), 1);
        assert_eq!(metrics.non_streamed_request_latency().avg_ms(), 10.0);
        assert_eq!(metrics.streamed_request_latency().count(), 1);
        assert_eq!(metrics.streamed_request_latency().avg_ms(), 30.0);
        assert_eq!(metrics.tokens_per_second(), 375.0);
        assert_eq!(metrics.time_to_first_token().count(), 0);
        metrics.record_time_to_first_token(Duration::from_millis(7));
        assert_eq!(metrics.time_to_first_token().count(), 1);
        assert_eq!(metrics.time_to_first_token().avg_ms(), 7.0);
        #[cfg(feature = "tool-calls")]
        {
            metrics.record_tool_argument_assembly(Duration::from_millis(8));
            metrics.record_tool_intent_fill(Duration::from_millis(9));
            metrics.record_tool_schema_validation(Duration::from_millis(10));
            metrics.record_tool_finish(Duration::from_millis(11));
            metrics.record_validated_tool_call(Duration::from_millis(11));
            assert_eq!(metrics.tool_argument_assembly().count(), 1);
            assert_eq!(metrics.tool_intent_fill().count(), 1);
            assert_eq!(metrics.tool_schema_validation().count(), 1);
            assert_eq!(metrics.tool_finish().count(), 1);
            assert_eq!(metrics.validated_tool_call().count(), 1);
            assert_eq!(metrics.tool_finish().avg_ms(), 11.0);
            assert_eq!(metrics.validated_tool_call().avg_ms(), 11.0);
        }

        metrics.record_stream_client_disconnect();
        assert_eq!(metrics.requests_total(), 4);
        assert_eq!(metrics.failed_requests(), 2);
        assert_eq!(metrics.stream_client_disconnected_requests(), 1);
        metrics.record_stream_stall();
        assert_eq!(metrics.requests_total(), 5);
        assert_eq!(metrics.failed_requests(), 3);
        assert_eq!(metrics.stream_stalled_requests(), 1);
        metrics.record_cancellation();
        assert_eq!(metrics.cancelled_requests(), 1);
        metrics.record_no_progress_failure();
        assert_eq!(metrics.no_progress_failures(), 1);
        metrics.record_model_pull_success(17);
        metrics.record_model_pull_failure();
        assert_eq!(metrics.model_pull_operations(), 2);
        assert_eq!(metrics.model_pull_successes(), 1);
        assert_eq!(metrics.model_pull_failures(), 1);
        assert_eq!(metrics.model_pull_bytes(), 17);
        metrics.record_artifact_verification_failure();
        assert_eq!(metrics.artifact_verification_failures(), 1);
    }

    #[test]
    fn server_metrics_leave_cached_prompt_tokens_absent_when_not_reported() {
        let mut metrics = ServerMetrics::default();

        metrics.record_success(TokenCounters::new(4, 1), false, Duration::from_millis(10));
        metrics.record_success(TokenCounters::new(8, 2), true, Duration::from_millis(30));

        assert_eq!(metrics.tokens().prompt_cached_tokens(), None);
    }
}
