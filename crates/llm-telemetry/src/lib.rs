use std::{
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    time::Duration,
};

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
        self.prompt_tokens.saturating_add(self.completion_tokens)
    }

    pub fn record_prompt_tokens(&mut self, tokens: u64) {
        self.prompt_tokens = self.prompt_tokens.saturating_add(tokens);
    }

    pub fn record_completion_tokens(&mut self, tokens: u64) {
        self.completion_tokens = self.completion_tokens.saturating_add(tokens);
    }

    pub fn record_prompt_cached_tokens(&mut self, tokens: Option<u64>) {
        if let Some(tokens) = tokens {
            self.prompt_cached_tokens = Some(
                self.prompt_cached_tokens
                    .unwrap_or(0)
                    .saturating_add(tokens),
            );
        }
    }
}

#[derive(Debug, Default)]
struct AtomicTokenCounters {
    prompt_tokens: AtomicU64,
    completion_tokens: AtomicU64,
    prompt_cached_tokens: AtomicU64,
    prompt_cached_tokens_reported: AtomicBool,
}

impl AtomicTokenCounters {
    fn record(&self, tokens: TokenCounters) {
        atomic_saturating_add(&self.prompt_tokens, tokens.prompt_tokens());
        atomic_saturating_add(&self.completion_tokens, tokens.completion_tokens());
        if let Some(cached_tokens) = tokens.prompt_cached_tokens() {
            atomic_saturating_add(&self.prompt_cached_tokens, cached_tokens);
            self.prompt_cached_tokens_reported
                .store(true, Ordering::Relaxed);
        }
    }

    fn snapshot(&self) -> TokenCounters {
        TokenCounters {
            prompt_tokens: self.prompt_tokens.load(Ordering::Relaxed),
            completion_tokens: self.completion_tokens.load(Ordering::Relaxed),
            prompt_cached_tokens: self
                .prompt_cached_tokens_reported
                .load(Ordering::Relaxed)
                .then(|| self.prompt_cached_tokens.load(Ordering::Relaxed)),
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

#[derive(Debug)]
pub struct AtomicLatencyMetrics {
    count: AtomicU64,
    total_nanos: AtomicU64,
    min_nanos: AtomicU64,
    max_nanos: AtomicU64,
}

impl Default for AtomicLatencyMetrics {
    fn default() -> Self {
        Self {
            count: AtomicU64::new(0),
            total_nanos: AtomicU64::new(0),
            min_nanos: AtomicU64::new(u64::MAX),
            max_nanos: AtomicU64::new(0),
        }
    }
}

impl AtomicLatencyMetrics {
    pub fn record(&self, duration: Duration) {
        let nanos = duration_nanos_u64(duration);
        atomic_saturating_add(&self.total_nanos, nanos);
        self.min_nanos.fetch_min(nanos, Ordering::Relaxed);
        self.max_nanos.fetch_max(nanos, Ordering::Relaxed);
        atomic_saturating_add(&self.count, 1);
    }

    pub fn snapshot(&self) -> LatencyMetrics {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            return LatencyMetrics::default();
        }
        let min_nanos = self.min_nanos.load(Ordering::Relaxed);
        LatencyMetrics {
            count,
            total_nanos: u128::from(self.total_nanos.load(Ordering::Relaxed)),
            min_nanos: Some(u128::from(min_nanos)),
            max_nanos: u128::from(self.max_nanos.load(Ordering::Relaxed)),
        }
    }
}

fn duration_nanos_u64(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn atomic_saturating_add(counter: &AtomicU64, value: u64) {
    let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
        Some(current.saturating_add(value))
    });
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

#[derive(Debug, Default)]
pub struct ServerMetrics {
    requests_total: AtomicU64,
    successful_requests: AtomicU64,
    failed_requests: AtomicU64,
    streamed_requests: AtomicU64,
    stream_client_disconnected_requests: AtomicU64,
    stream_stalled_requests: AtomicU64,
    cancelled_requests: AtomicU64,
    no_progress_failures: AtomicU64,
    model_pull_operations: AtomicU64,
    model_pull_successes: AtomicU64,
    model_pull_failures: AtomicU64,
    model_pull_bytes: AtomicU64,
    artifact_verification_failures: AtomicU64,
    request_latency: AtomicLatencyMetrics,
    non_streamed_request_latency: AtomicLatencyMetrics,
    streamed_request_latency: AtomicLatencyMetrics,
    time_to_first_token: AtomicLatencyMetrics,
    #[cfg(feature = "tool-calls")]
    first_tool_delta: AtomicLatencyMetrics,
    #[cfg(feature = "tool-calls")]
    first_tool_delta_after_ttft: AtomicLatencyMetrics,
    #[cfg(feature = "tool-calls")]
    tool_argument_assembly: AtomicLatencyMetrics,
    #[cfg(feature = "tool-calls")]
    tool_intent_fill: AtomicLatencyMetrics,
    #[cfg(feature = "tool-calls")]
    tool_schema_validation: AtomicLatencyMetrics,
    #[cfg(feature = "tool-calls")]
    tool_finish: AtomicLatencyMetrics,
    #[cfg(feature = "tool-calls")]
    validated_tool_call: AtomicLatencyMetrics,
    tokens: AtomicTokenCounters,
}

impl ServerMetrics {
    pub fn record_success(&self, tokens: TokenCounters, streamed: bool, latency: Duration) {
        atomic_saturating_add(&self.requests_total, 1);
        atomic_saturating_add(&self.successful_requests, 1);
        if streamed {
            atomic_saturating_add(&self.streamed_requests, 1);
            self.streamed_request_latency.record(latency);
        } else {
            self.non_streamed_request_latency.record(latency);
        }
        self.request_latency.record(latency);
        self.tokens.record(tokens);
    }

    pub fn record_failure(&self) {
        atomic_saturating_add(&self.requests_total, 1);
        atomic_saturating_add(&self.failed_requests, 1);
    }

    pub fn record_stream_client_disconnect(&self) {
        atomic_saturating_add(&self.requests_total, 1);
        atomic_saturating_add(&self.failed_requests, 1);
        atomic_saturating_add(&self.stream_client_disconnected_requests, 1);
    }

    pub fn record_stream_stall(&self) {
        atomic_saturating_add(&self.requests_total, 1);
        atomic_saturating_add(&self.failed_requests, 1);
        atomic_saturating_add(&self.stream_stalled_requests, 1);
    }

    pub fn record_cancellation(&self) {
        atomic_saturating_add(&self.cancelled_requests, 1);
    }

    pub fn record_no_progress_failure(&self) {
        atomic_saturating_add(&self.no_progress_failures, 1);
    }

    pub fn record_model_pull_success(&self, bytes: u64) {
        atomic_saturating_add(&self.model_pull_operations, 1);
        atomic_saturating_add(&self.model_pull_successes, 1);
        atomic_saturating_add(&self.model_pull_bytes, bytes);
    }

    pub fn record_model_pull_failure(&self) {
        atomic_saturating_add(&self.model_pull_operations, 1);
        atomic_saturating_add(&self.model_pull_failures, 1);
    }

    pub fn record_artifact_verification_failure(&self) {
        atomic_saturating_add(&self.artifact_verification_failures, 1);
    }

    pub fn record_time_to_first_token(&self, latency: Duration) {
        self.time_to_first_token.record(latency);
    }

    #[cfg(feature = "tool-calls")]
    pub fn record_first_tool_delta(&self, latency: Duration) {
        self.first_tool_delta.record(latency);
    }

    #[cfg(feature = "tool-calls")]
    pub fn record_first_tool_delta_after_ttft(&self, latency: Duration) {
        self.first_tool_delta_after_ttft.record(latency);
    }

    #[cfg(feature = "tool-calls")]
    pub fn record_tool_argument_assembly(&self, latency: Duration) {
        self.tool_argument_assembly.record(latency);
    }

    #[cfg(feature = "tool-calls")]
    pub fn record_tool_intent_fill(&self, latency: Duration) {
        self.tool_intent_fill.record(latency);
    }

    #[cfg(feature = "tool-calls")]
    pub fn record_tool_schema_validation(&self, latency: Duration) {
        self.tool_schema_validation.record(latency);
    }

    #[cfg(feature = "tool-calls")]
    pub fn record_tool_finish(&self, latency: Duration) {
        self.tool_finish.record(latency);
    }

    #[cfg(feature = "tool-calls")]
    pub fn record_validated_tool_call(&self, latency: Duration) {
        self.validated_tool_call.record(latency);
    }

    pub fn requests_total(&self) -> u64 {
        self.requests_total.load(Ordering::Relaxed)
    }

    pub fn successful_requests(&self) -> u64 {
        self.successful_requests.load(Ordering::Relaxed)
    }

    pub fn failed_requests(&self) -> u64 {
        self.failed_requests.load(Ordering::Relaxed)
    }

    pub fn streamed_requests(&self) -> u64 {
        self.streamed_requests.load(Ordering::Relaxed)
    }

    pub fn stream_client_disconnected_requests(&self) -> u64 {
        self.stream_client_disconnected_requests
            .load(Ordering::Relaxed)
    }

    pub fn stream_stalled_requests(&self) -> u64 {
        self.stream_stalled_requests.load(Ordering::Relaxed)
    }

    pub fn cancelled_requests(&self) -> u64 {
        self.cancelled_requests.load(Ordering::Relaxed)
    }

    pub fn no_progress_failures(&self) -> u64 {
        self.no_progress_failures.load(Ordering::Relaxed)
    }

    pub fn model_pull_operations(&self) -> u64 {
        self.model_pull_operations.load(Ordering::Relaxed)
    }

    pub fn model_pull_successes(&self) -> u64 {
        self.model_pull_successes.load(Ordering::Relaxed)
    }

    pub fn model_pull_failures(&self) -> u64 {
        self.model_pull_failures.load(Ordering::Relaxed)
    }

    pub fn model_pull_bytes(&self) -> u64 {
        self.model_pull_bytes.load(Ordering::Relaxed)
    }

    pub fn artifact_verification_failures(&self) -> u64 {
        self.artifact_verification_failures.load(Ordering::Relaxed)
    }

    pub fn request_latency(&self) -> LatencyMetrics {
        self.request_latency.snapshot()
    }

    pub fn non_streamed_request_latency(&self) -> LatencyMetrics {
        self.non_streamed_request_latency.snapshot()
    }

    pub fn streamed_request_latency(&self) -> LatencyMetrics {
        self.streamed_request_latency.snapshot()
    }

    pub fn time_to_first_token(&self) -> LatencyMetrics {
        self.time_to_first_token.snapshot()
    }

    #[cfg(feature = "tool-calls")]
    pub fn first_tool_delta(&self) -> LatencyMetrics {
        self.first_tool_delta.snapshot()
    }

    #[cfg(feature = "tool-calls")]
    pub fn first_tool_delta_after_ttft(&self) -> LatencyMetrics {
        self.first_tool_delta_after_ttft.snapshot()
    }

    #[cfg(feature = "tool-calls")]
    pub fn tool_argument_assembly(&self) -> LatencyMetrics {
        self.tool_argument_assembly.snapshot()
    }

    #[cfg(feature = "tool-calls")]
    pub fn tool_intent_fill(&self) -> LatencyMetrics {
        self.tool_intent_fill.snapshot()
    }

    #[cfg(feature = "tool-calls")]
    pub fn tool_schema_validation(&self) -> LatencyMetrics {
        self.tool_schema_validation.snapshot()
    }

    #[cfg(feature = "tool-calls")]
    pub fn tool_finish(&self) -> LatencyMetrics {
        self.tool_finish.snapshot()
    }

    #[cfg(feature = "tool-calls")]
    pub fn validated_tool_call(&self) -> LatencyMetrics {
        self.validated_tool_call.snapshot()
    }

    pub fn tokens_per_second(&self) -> f64 {
        let seconds = self.request_latency().total_seconds();
        if seconds == 0.0 {
            0.0
        } else {
            self.tokens().total_tokens() as f64 / seconds
        }
    }

    pub fn tokens(&self) -> TokenCounters {
        self.tokens.snapshot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{sync::Arc, thread};

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
    fn token_counters_saturate_when_token_totals_overflow() {
        let mut counters = TokenCounters::new(u64::MAX - 1, u64::MAX - 2)
            .with_prompt_cached_tokens(Some(u64::MAX - 3));

        counters.record_prompt_tokens(2);
        counters.record_completion_tokens(3);
        counters.record_prompt_cached_tokens(Some(4));

        assert_eq!(counters.prompt_tokens(), u64::MAX);
        assert_eq!(counters.completion_tokens(), u64::MAX);
        assert_eq!(counters.total_tokens(), u64::MAX);
        assert_eq!(counters.prompt_cached_tokens(), Some(u64::MAX));
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
        let metrics = ServerMetrics::default();

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
            metrics.record_first_tool_delta(Duration::from_millis(27));
            metrics.record_first_tool_delta_after_ttft(Duration::from_millis(4));
            metrics.record_tool_argument_assembly(Duration::from_millis(8));
            metrics.record_tool_intent_fill(Duration::from_millis(9));
            metrics.record_tool_schema_validation(Duration::from_millis(10));
            metrics.record_tool_finish(Duration::from_millis(11));
            metrics.record_validated_tool_call(Duration::from_millis(11));
            assert_eq!(metrics.first_tool_delta().count(), 1);
            assert_eq!(metrics.first_tool_delta().avg_ms(), 27.0);
            assert_eq!(metrics.first_tool_delta_after_ttft().count(), 1);
            assert_eq!(metrics.first_tool_delta_after_ttft().avg_ms(), 4.0);
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
        let metrics = ServerMetrics::default();

        metrics.record_success(TokenCounters::new(4, 1), false, Duration::from_millis(10));
        metrics.record_success(TokenCounters::new(8, 2), true, Duration::from_millis(30));

        assert_eq!(metrics.tokens().prompt_cached_tokens(), None);
    }

    #[test]
    fn server_metrics_record_hot_paths_concurrently_without_external_lock() {
        let metrics = Arc::new(ServerMetrics::default());

        thread::scope(|scope| {
            for _ in 0..8 {
                let metrics = Arc::clone(&metrics);
                scope.spawn(move || {
                    for _ in 0..64 {
                        metrics.record_success(
                            TokenCounters::new(4, 2).with_prompt_cached_tokens(Some(1)),
                            true,
                            Duration::from_millis(5),
                        );
                        metrics.record_time_to_first_token(Duration::from_millis(2));
                        #[cfg(feature = "tool-calls")]
                        metrics.record_first_tool_delta(Duration::from_millis(3));
                    }
                });
            }
        });

        assert_eq!(metrics.requests_total(), 512);
        assert_eq!(metrics.successful_requests(), 512);
        assert_eq!(metrics.streamed_requests(), 512);
        assert_eq!(
            metrics.tokens(),
            TokenCounters::new(2048, 1024).with_prompt_cached_tokens(Some(512))
        );
        assert_eq!(metrics.request_latency().count(), 512);
        assert_eq!(metrics.streamed_request_latency().count(), 512);
        assert_eq!(metrics.time_to_first_token().count(), 512);
        #[cfg(feature = "tool-calls")]
        assert_eq!(metrics.first_tool_delta().count(), 512);
    }
}
