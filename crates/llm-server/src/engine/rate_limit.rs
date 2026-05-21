use super::config::PublicInferenceRateLimit;
use crate::sync_ext::FailPoisonedMutex;
use std::{
    sync::Mutex,
    time::{Duration, Instant},
};

#[derive(Debug)]
pub(super) struct PublicInferenceRateLimiter {
    max_requests: usize,
    window: Duration,
    state: Mutex<RateLimitWindow>,
}

#[derive(Debug)]
struct RateLimitWindow {
    started_at: Instant,
    accepted: usize,
}

impl PublicInferenceRateLimiter {
    pub(super) fn new(options: PublicInferenceRateLimit) -> Self {
        Self::new_with_started_at(options, Instant::now())
    }

    fn new_with_started_at(options: PublicInferenceRateLimit, started_at: Instant) -> Self {
        let window = if options.window.is_zero() {
            Duration::from_secs(1)
        } else {
            options.window
        };
        Self {
            max_requests: options.max_requests.max(1),
            window,
            state: Mutex::new(RateLimitWindow {
                started_at,
                accepted: 0,
            }),
        }
    }

    pub(super) fn acquire(&self) -> Result<(), Duration> {
        self.acquire_at(Instant::now())
    }

    fn acquire_at(&self, now: Instant) -> Result<(), Duration> {
        let mut state = self.state.lock_or_panic("public inference rate limiter");
        let elapsed = now.saturating_duration_since(state.started_at);
        if elapsed >= self.window {
            state.started_at = now;
            state.accepted = 0;
        }

        if state.accepted < self.max_requests {
            state.accepted += 1;
            return Ok(());
        }

        let retry_after = self
            .window
            .checked_sub(now.saturating_duration_since(state.started_at))
            .unwrap_or_else(|| Duration::from_secs(1));
        Err(retry_after)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limiter_refills_after_window_elapses() {
        let start = Instant::now();
        let limiter = PublicInferenceRateLimiter::new_with_started_at(
            PublicInferenceRateLimit {
                max_requests: 1,
                window: Duration::from_secs(2),
            },
            start,
        );

        assert_eq!(limiter.acquire_at(start), Ok(()));
        assert_eq!(
            limiter.acquire_at(start + Duration::from_secs(1)),
            Err(Duration::from_secs(1))
        );
        assert_eq!(limiter.acquire_at(start + Duration::from_secs(2)), Ok(()));
    }
}
