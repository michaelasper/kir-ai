use crate::sync_ext::RecoverPoisonedMutex;
use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone)]
pub(super) struct ActiveRequestRegistry {
    active: Arc<Mutex<HashMap<String, ActiveRequestEntry>>>,
    next_request_id: Arc<AtomicU64>,
}

impl Default for ActiveRequestRegistry {
    fn default() -> Self {
        Self {
            active: Arc::new(Mutex::new(HashMap::new())),
            next_request_id: Arc::new(AtomicU64::new(1)),
        }
    }
}

impl ActiveRequestRegistry {
    pub(super) fn next_request_id(&self) -> String {
        let next = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        format!("req-{next}")
    }

    pub(super) fn register(&self, id: String) -> Result<ActiveRequest, RequestRegistrationError> {
        let cancellation = CancellationToken::new();
        let mut active = self.active.lock_or_recover("active request");
        if active.contains_key(&id) {
            return Err(RequestRegistrationError::Conflict(id));
        }
        active.insert(
            id.clone(),
            ActiveRequestEntry {
                cancellation: cancellation.clone(),
                state: ActiveRequestState::Queued,
            },
        );
        drop(active);
        Ok(ActiveRequest {
            id,
            cancellation,
            started_at: Instant::now(),
            active: self.active.clone(),
        })
    }

    pub(super) fn active_count(&self) -> usize {
        self.active
            .lock_or_recover("active request")
            .values()
            .filter(|entry| entry.state.is_active())
            .count()
    }

    pub(super) fn cancel(&self, request_id: &str) -> CancelRequestResult {
        let mut active = self.active.lock_or_recover("active request");
        let Some(entry) = active.get_mut(request_id) else {
            return CancelRequestResult::NotFound;
        };
        match entry.state {
            ActiveRequestState::Queued | ActiveRequestState::Running => {
                entry.state = ActiveRequestState::Cancelled;
                entry.cancellation.cancel();
                CancelRequestResult::Cancelled
            }
            ActiveRequestState::Cancelled => CancelRequestResult::AlreadyCancelled,
            ActiveRequestState::Finished => CancelRequestResult::Finished,
        }
    }
}

#[derive(Debug)]
pub(super) struct ActiveRequest {
    pub(super) id: String,
    pub(super) cancellation: CancellationToken,
    pub(super) started_at: Instant,
    active: Arc<Mutex<HashMap<String, ActiveRequestEntry>>>,
}

impl ActiveRequest {
    pub(super) fn mark_running(&self) -> RequestStartResult {
        let mut active = self.active.lock_or_recover("active request");
        let Some(entry) = active.get_mut(&self.id) else {
            return RequestStartResult::Missing;
        };
        match entry.state {
            ActiveRequestState::Queued => {
                entry.state = ActiveRequestState::Running;
                RequestStartResult::Running
            }
            ActiveRequestState::Running => RequestStartResult::Running,
            ActiveRequestState::Cancelled => RequestStartResult::Cancelled,
            ActiveRequestState::Finished => RequestStartResult::Finished,
        }
    }

    pub(super) fn mark_finished(&self) -> RequestFinishResult {
        let mut active = self.active.lock_or_recover("active request");
        let Some(entry) = active.get_mut(&self.id) else {
            return RequestFinishResult::Missing;
        };
        match entry.state {
            ActiveRequestState::Queued | ActiveRequestState::Running => {
                entry.state = ActiveRequestState::Finished;
                RequestFinishResult::Finished
            }
            ActiveRequestState::Cancelled => RequestFinishResult::Cancelled,
            ActiveRequestState::Finished => RequestFinishResult::Finished,
        }
    }
}

impl Drop for ActiveRequest {
    fn drop(&mut self) {
        self.active
            .lock_or_recover("active request")
            .remove(&self.id);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum RequestRegistrationError {
    Conflict(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CancelRequestResult {
    Cancelled,
    AlreadyCancelled,
    Finished,
    NotFound,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RequestStartResult {
    Running,
    Cancelled,
    Finished,
    Missing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RequestFinishResult {
    Finished,
    Cancelled,
    Missing,
}

#[derive(Debug)]
struct ActiveRequestEntry {
    cancellation: CancellationToken,
    state: ActiveRequestState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActiveRequestState {
    Queued,
    Running,
    Cancelled,
    Finished,
}

impl ActiveRequestState {
    fn is_active(self) -> bool {
        !matches!(self, Self::Finished)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::{Arc, Barrier},
        thread,
    };

    #[test]
    fn registry_generates_monotonic_request_ids() {
        let registry = ActiveRequestRegistry::default();
        let first_id = registry.next_request_id();
        let second_id = registry.next_request_id();

        let first = registry
            .register(first_id)
            .expect("first request registers");
        let second = registry
            .register(second_id)
            .expect("second request registers");

        assert_eq!(first.id, "req-1");
        assert_eq!(second.id, "req-2");
        assert_eq!(registry.active_count(), 2);
    }

    #[test]
    fn registry_rejects_duplicate_active_request_id_until_guard_drops() {
        let registry = ActiveRequestRegistry::default();
        let active = registry
            .register("same-id".to_owned())
            .expect("first request registers");

        let err = registry
            .register("same-id".to_owned())
            .expect_err("duplicate active request is rejected");
        assert_eq!(
            err,
            RequestRegistrationError::Conflict("same-id".to_owned())
        );

        drop(active);
        registry
            .register("same-id".to_owned())
            .expect("request id can be reused after guard drops");
    }

    #[test]
    fn registry_cancels_active_request_token() {
        let registry = ActiveRequestRegistry::default();
        let active = registry
            .register("cancel-me".to_owned())
            .expect("request registers");
        assert_eq!(active.mark_running(), RequestStartResult::Running);

        assert_eq!(registry.cancel("cancel-me"), CancelRequestResult::Cancelled);
        assert!(active.cancellation.is_cancelled());
        assert_eq!(
            registry.cancel("cancel-me"),
            CancelRequestResult::AlreadyCancelled
        );
        assert_eq!(registry.cancel("missing"), CancelRequestResult::NotFound);
    }

    #[test]
    fn registry_does_not_count_already_cancelled_request_as_new_cancellation() {
        let registry = ActiveRequestRegistry::default();
        let active = registry
            .register("cancel-once".to_owned())
            .expect("request registers");
        assert_eq!(active.mark_running(), RequestStartResult::Running);

        assert_eq!(
            registry.cancel("cancel-once"),
            CancelRequestResult::Cancelled
        );
        assert!(active.cancellation.is_cancelled());
        assert_eq!(
            registry.cancel("cancel-once"),
            CancelRequestResult::AlreadyCancelled
        );
    }

    #[test]
    fn registry_rejects_cancellation_after_request_finishes() {
        let registry = ActiveRequestRegistry::default();
        let active = registry
            .register("finished".to_owned())
            .expect("request registers");
        assert_eq!(active.mark_running(), RequestStartResult::Running);
        assert_eq!(active.mark_finished(), RequestFinishResult::Finished);

        assert_eq!(registry.active_count(), 0);
        assert_eq!(registry.cancel("finished"), CancelRequestResult::Finished);
        assert!(!active.cancellation.is_cancelled());
        assert_eq!(
            registry
                .register("finished".to_owned())
                .expect_err("finished request id remains reserved until guard drops"),
            RequestRegistrationError::Conflict("finished".to_owned())
        );

        drop(active);
        registry
            .register("finished".to_owned())
            .expect("request id can be reused after finished guard drops");
    }

    #[test]
    fn registry_prevents_start_after_queued_request_is_cancelled() {
        let registry = ActiveRequestRegistry::default();
        let active = registry
            .register("cancel-before-running".to_owned())
            .expect("request registers");

        assert_eq!(
            registry.cancel("cancel-before-running"),
            CancelRequestResult::Cancelled
        );
        assert_eq!(active.mark_running(), RequestStartResult::Cancelled);
        assert!(active.cancellation.is_cancelled());
    }

    #[test]
    fn registry_does_not_overwrite_accepted_cancellation_with_finished() {
        let registry = ActiveRequestRegistry::default();
        let active = registry
            .register("cancel-before-finish".to_owned())
            .expect("request registers");
        assert_eq!(active.mark_running(), RequestStartResult::Running);

        assert_eq!(
            registry.cancel("cancel-before-finish"),
            CancelRequestResult::Cancelled
        );
        assert_eq!(active.mark_finished(), RequestFinishResult::Cancelled);
        assert_eq!(
            registry.cancel("cancel-before-finish"),
            CancelRequestResult::AlreadyCancelled
        );
    }

    #[test]
    fn registry_counts_concurrent_cancellation_once() {
        let registry = ActiveRequestRegistry::default();
        let active = registry
            .register("cancel-race".to_owned())
            .expect("request registers");
        assert_eq!(active.mark_running(), RequestStartResult::Running);
        let barrier = Arc::new(Barrier::new(3));

        let first_registry = registry.clone();
        let first_barrier = barrier.clone();
        let first = thread::spawn(move || {
            first_barrier.wait();
            first_registry.cancel("cancel-race")
        });

        let second_registry = registry.clone();
        let second_barrier = barrier.clone();
        let second = thread::spawn(move || {
            second_barrier.wait();
            second_registry.cancel("cancel-race")
        });

        barrier.wait();
        let first = first.join().expect("first cancellation thread joins");
        let second = second.join().expect("second cancellation thread joins");

        assert!(active.cancellation.is_cancelled());
        assert_eq!(
            [first, second]
                .into_iter()
                .filter(|result| *result == CancelRequestResult::Cancelled)
                .count(),
            1
        );
        assert_eq!(
            [first, second]
                .into_iter()
                .filter(|result| *result == CancelRequestResult::AlreadyCancelled)
                .count(),
            1
        );
    }
}
