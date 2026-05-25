use super::config::PublicInferenceRateLimit;
use crate::sync_ext::FailPoisonedMutex;
use std::{
    collections::{HashMap, VecDeque, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
    sync::Mutex,
    time::{Duration, Instant},
};

#[derive(Debug)]
pub(super) struct PublicInferenceRateLimiter {
    max_requests: usize,
    window: Duration,
    ttl: Duration,
    cleanup_interval: Duration,
    state: Mutex<RateLimitState>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct PublicInferenceClientKey(String);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RateLimitSnapshot {
    pub(super) limit_requests: usize,
    pub(super) remaining_requests: usize,
    pub(super) reset_after: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RateLimitRejection {
    pub(super) retry_after: Duration,
    pub(super) snapshot: RateLimitSnapshot,
}

#[derive(Debug)]
struct RateLimitState {
    buckets: HashMap<PublicInferenceClientKey, ClientRequestLog>,
    next_cleanup_at: Instant,
}

#[derive(Debug)]
struct ClientRequestLog {
    accepted_at: VecDeque<Instant>,
    last_seen: Instant,
}

impl PublicInferenceClientKey {
    pub(super) fn new(value: impl Into<String>) -> Self {
        let value = value.into();
        if value.trim().is_empty() {
            Self::anonymous()
        } else {
            Self(value)
        }
    }

    pub(super) fn hashed(prefix: &str, value: &str) -> Self {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        Self(format!("{prefix}:{:016x}", hasher.finish()))
    }

    pub(super) fn anonymous() -> Self {
        Self("anonymous".to_owned())
    }
}

impl ClientRequestLog {
    fn new(now: Instant) -> Self {
        Self {
            accepted_at: VecDeque::new(),
            last_seen: now,
        }
    }

    fn prune(&mut self, now: Instant, window: Duration) {
        while self
            .accepted_at
            .front()
            .is_some_and(|accepted_at| now.saturating_duration_since(*accepted_at) >= window)
        {
            self.accepted_at.pop_front();
        }
    }

    fn reset_after(&self, now: Instant, window: Duration) -> Duration {
        self.accepted_at
            .front()
            .and_then(|accepted_at| window.checked_sub(now.saturating_duration_since(*accepted_at)))
            .unwrap_or(Duration::ZERO)
    }

    fn is_expired(&self, now: Instant, ttl: Duration) -> bool {
        self.accepted_at.is_empty() && now.saturating_duration_since(self.last_seen) >= ttl
    }
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
        let ttl = window.checked_mul(2).unwrap_or(window);
        Self {
            max_requests: options.max_requests.max(1),
            window,
            ttl,
            cleanup_interval: window,
            state: Mutex::new(RateLimitState {
                buckets: HashMap::new(),
                next_cleanup_at: next_cleanup_after(started_at, window),
            }),
        }
    }

    pub(super) fn acquire(
        &self,
        key: &PublicInferenceClientKey,
    ) -> Result<RateLimitSnapshot, RateLimitRejection> {
        self.acquire_at(key, Instant::now())
    }

    fn acquire_at(
        &self,
        key: &PublicInferenceClientKey,
        now: Instant,
    ) -> Result<RateLimitSnapshot, RateLimitRejection> {
        let mut state = self.state.lock_or_panic("public inference rate limiter");
        self.cleanup_expired_buckets(&mut state, now);

        let bucket = state
            .buckets
            .entry(key.clone())
            .or_insert_with(|| ClientRequestLog::new(now));
        bucket.last_seen = now;
        bucket.prune(now, self.window);

        if bucket.accepted_at.len() < self.max_requests {
            bucket.accepted_at.push_back(now);
            return Ok(self.snapshot(bucket, now));
        }

        let retry_after = bucket.reset_after(now, self.window);
        Err(RateLimitRejection {
            retry_after,
            snapshot: self.snapshot(bucket, now),
        })
    }

    fn snapshot(&self, bucket: &ClientRequestLog, now: Instant) -> RateLimitSnapshot {
        RateLimitSnapshot {
            limit_requests: self.max_requests,
            remaining_requests: self.max_requests.saturating_sub(bucket.accepted_at.len()),
            reset_after: bucket.reset_after(now, self.window),
        }
    }

    fn cleanup_expired_buckets(&self, state: &mut RateLimitState, now: Instant) {
        if now < state.next_cleanup_at {
            return;
        }

        for bucket in state.buckets.values_mut() {
            bucket.prune(now, self.window);
        }
        let ttl = self.ttl;
        state
            .buckets
            .retain(|_, bucket| !bucket.is_expired(now, ttl));
        state.next_cleanup_at = next_cleanup_after(now, self.cleanup_interval);
    }

    #[cfg(test)]
    fn tracked_client_count_at(&self, now: Instant) -> usize {
        let mut state = self.state.lock_or_panic("public inference rate limiter");
        self.cleanup_expired_buckets(&mut state, now);
        state.buckets.len()
    }
}

fn next_cleanup_after(now: Instant, interval: Duration) -> Instant {
    now.checked_add(interval).unwrap_or(now)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(value: &str) -> PublicInferenceClientKey {
        PublicInferenceClientKey::new(value)
    }

    fn accepted(result: Result<RateLimitSnapshot, RateLimitRejection>) -> Result<(), Duration> {
        result
            .map(|_| ())
            .map_err(|rejection| rejection.retry_after)
    }

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
        let key = key("client-a");

        assert_eq!(accepted(limiter.acquire_at(&key, start)), Ok(()));
        assert_eq!(
            accepted(limiter.acquire_at(&key, start + Duration::from_secs(1))),
            Err(Duration::from_secs(1))
        );
        assert_eq!(
            accepted(limiter.acquire_at(&key, start + Duration::from_secs(2))),
            Ok(())
        );
    }

    #[test]
    fn limiter_does_not_double_burst_across_window_boundary() {
        let start = Instant::now();
        let limiter = PublicInferenceRateLimiter::new_with_started_at(
            PublicInferenceRateLimit {
                max_requests: 2,
                window: Duration::from_secs(1),
            },
            start,
        );
        let near_boundary = start + Duration::from_millis(999);
        let key = key("client-a");

        assert_eq!(accepted(limiter.acquire_at(&key, near_boundary)), Ok(()));
        assert_eq!(accepted(limiter.acquire_at(&key, near_boundary)), Ok(()));
        assert_eq!(
            accepted(limiter.acquire_at(&key, start + Duration::from_secs(1))),
            Err(Duration::from_millis(999))
        );
    }

    #[test]
    fn limiter_tracks_clients_independently() {
        let start = Instant::now();
        let limiter = PublicInferenceRateLimiter::new_with_started_at(
            PublicInferenceRateLimit {
                max_requests: 1,
                window: Duration::from_secs(60),
            },
            start,
        );
        let first = key("client-a");
        let second = key("client-b");

        assert_eq!(accepted(limiter.acquire_at(&first, start)), Ok(()));
        assert_eq!(accepted(limiter.acquire_at(&second, start)), Ok(()));
        assert_eq!(
            accepted(limiter.acquire_at(&first, start)),
            Err(Duration::from_secs(60))
        );
    }

    #[test]
    fn limiter_cleans_up_idle_client_buckets_after_ttl() {
        let start = Instant::now();
        let limiter = PublicInferenceRateLimiter::new_with_started_at(
            PublicInferenceRateLimit {
                max_requests: 1,
                window: Duration::from_secs(1),
            },
            start,
        );
        let key = key("client-a");

        assert_eq!(accepted(limiter.acquire_at(&key, start)), Ok(()));
        assert_eq!(limiter.tracked_client_count_at(start), 1);
        assert_eq!(
            limiter.tracked_client_count_at(start + Duration::from_secs(2)),
            0
        );
    }
}
