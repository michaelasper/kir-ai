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
}
