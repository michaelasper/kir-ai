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
    tokens: TokenCounters,
}

impl ServerMetrics {
    pub fn record_success(&mut self, tokens: TokenCounters, streamed: bool) {
        self.requests_total += 1;
        self.successful_requests += 1;
        if streamed {
            self.streamed_requests += 1;
        }
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

        metrics.record_success(TokenCounters::new(4, 1), false);
        metrics.record_success(TokenCounters::new(8, 2), true);
        metrics.record_failure();

        assert_eq!(metrics.requests_total(), 3);
        assert_eq!(metrics.successful_requests(), 2);
        assert_eq!(metrics.failed_requests(), 1);
        assert_eq!(metrics.streamed_requests(), 1);
        assert_eq!(metrics.cancelled_requests(), 0);
        assert_eq!(metrics.tokens(), TokenCounters::new(12, 3));

        metrics.record_cancellation();
        assert_eq!(metrics.cancelled_requests(), 1);
    }
}
