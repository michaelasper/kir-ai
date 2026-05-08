use std::time::Duration;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TokenCounters {
    prompt_tokens: u64,
    completion_tokens: u64,
}

impl TokenCounters {
    pub fn new(prompt_tokens: u64, completion_tokens: u64) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
        }
    }

    pub fn prompt_tokens(&self) -> u64 {
        self.prompt_tokens
    }

    pub fn completion_tokens(&self) -> u64 {
        self.completion_tokens
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
    cancelled_requests: u64,
    no_progress_failures: u64,
    model_pull_operations: u64,
    model_pull_successes: u64,
    model_pull_failures: u64,
    model_pull_bytes: u64,
    request_latency: LatencyMetrics,
    time_to_first_token: LatencyMetrics,
    tokens: TokenCounters,
}

impl ServerMetrics {
    pub fn record_success(&mut self, tokens: TokenCounters, streamed: bool, latency: Duration) {
        self.requests_total += 1;
        self.successful_requests += 1;
        if streamed {
            self.streamed_requests += 1;
        }
        self.request_latency.record(latency);
        self.tokens.record_prompt_tokens(tokens.prompt_tokens());
        self.tokens
            .record_completion_tokens(tokens.completion_tokens());
    }

    pub fn record_failure(&mut self) {
        self.requests_total += 1;
        self.failed_requests += 1;
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

    pub fn record_time_to_first_token(&mut self, latency: Duration) {
        self.time_to_first_token.record(latency);
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

    pub fn request_latency(&self) -> LatencyMetrics {
        self.request_latency
    }

    pub fn time_to_first_token(&self) -> LatencyMetrics {
        self.time_to_first_token
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

        metrics.record_success(TokenCounters::new(4, 1), false, Duration::from_millis(10));
        metrics.record_success(TokenCounters::new(8, 2), true, Duration::from_millis(30));
        metrics.record_failure();

        assert_eq!(metrics.requests_total(), 3);
        assert_eq!(metrics.successful_requests(), 2);
        assert_eq!(metrics.failed_requests(), 1);
        assert_eq!(metrics.streamed_requests(), 1);
        assert_eq!(metrics.cancelled_requests(), 0);
        assert_eq!(metrics.no_progress_failures(), 0);
        assert_eq!(metrics.model_pull_operations(), 0);
        assert_eq!(metrics.model_pull_successes(), 0);
        assert_eq!(metrics.model_pull_failures(), 0);
        assert_eq!(metrics.model_pull_bytes(), 0);
        assert_eq!(metrics.tokens(), TokenCounters::new(12, 3));
        assert_eq!(metrics.request_latency().count(), 2);
        assert_eq!(metrics.request_latency().min_ms(), 10.0);
        assert_eq!(metrics.request_latency().max_ms(), 30.0);
        assert_eq!(metrics.request_latency().avg_ms(), 20.0);
        assert_eq!(metrics.tokens_per_second(), 375.0);
        assert_eq!(metrics.time_to_first_token().count(), 0);
        metrics.record_time_to_first_token(Duration::from_millis(7));
        assert_eq!(metrics.time_to_first_token().count(), 1);
        assert_eq!(metrics.time_to_first_token().avg_ms(), 7.0);

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
    }
}
