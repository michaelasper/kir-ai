use crate::sync_ext::FailPoisonedMutex;
use llm_api::{ChatCompletionRequest, CompletionRequest};
use std::{
    collections::VecDeque,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Default)]
pub(super) struct GenerationPhaseMetrics {
    prefill_requests: AtomicU64,
    decode_requests: AtomicU64,
}

impl GenerationPhaseMetrics {
    pub(super) fn begin(self: &Arc<Self>, phase: GenerationPhase) -> GenerationPhaseGuard {
        self.increment(phase);
        GenerationPhaseGuard {
            metrics: Arc::clone(self),
            phase,
        }
    }

    pub(super) fn prefill_requests(&self) -> u64 {
        self.prefill_requests.load(Ordering::Relaxed)
    }

    pub(super) fn decode_requests(&self) -> u64 {
        self.decode_requests.load(Ordering::Relaxed)
    }

    fn increment(&self, phase: GenerationPhase) {
        self.counter(phase).fetch_add(1, Ordering::Relaxed);
    }

    fn decrement(&self, phase: GenerationPhase) {
        self.counter(phase).fetch_sub(1, Ordering::Relaxed);
    }

    fn counter(&self, phase: GenerationPhase) -> &AtomicU64 {
        match phase {
            GenerationPhase::Prefill => &self.prefill_requests,
            GenerationPhase::Decode => &self.decode_requests,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum GenerationPhase {
    Prefill,
    Decode,
}

#[derive(Debug)]
pub(super) struct GenerationPhaseGuard {
    metrics: Arc<GenerationPhaseMetrics>,
    phase: GenerationPhase,
}

impl GenerationPhaseGuard {
    pub(super) fn transition_to_decode(&mut self) {
        if self.phase == GenerationPhase::Decode {
            return;
        }
        self.metrics.decrement(self.phase);
        self.phase = GenerationPhase::Decode;
        self.metrics.increment(self.phase);
    }
}

impl Drop for GenerationPhaseGuard {
    fn drop(&mut self) {
        self.metrics.decrement(self.phase);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SchedulerClass {
    Prefill,
    Decode,
}

impl SchedulerClass {
    pub(super) fn as_phase(self) -> GenerationPhase {
        match self {
            Self::Prefill => GenerationPhase::Prefill,
            Self::Decode => GenerationPhase::Decode,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ModelSchedulerOptions {
    pub(super) concurrency_limit: usize,
    pub(super) queue_limit: usize,
    pub(super) queue_timeout: Option<Duration>,
    pub(super) prefill_threshold_chars: usize,
    pub(super) prefill_burst: usize,
}

#[derive(Debug)]
pub(super) struct ModelScheduler {
    options: ModelSchedulerOptions,
    state: Mutex<ModelSchedulerState>,
    notify: Notify,
}

#[derive(Debug, Default)]
struct ModelSchedulerState {
    next_ticket: u64,
    queued_prefill: VecDeque<u64>,
    queued_decode: VecDeque<u64>,
    active_prefill: usize,
    active_decode: usize,
    prefill_admissions_since_decode: usize,
    admitted_prefill: u64,
    admitted_decode: u64,
    completed: u64,
    cancelled: u64,
    failed: u64,
    queued_cancelled: u64,
    queue_timeouts: u64,
}

impl ModelSchedulerState {
    fn active_total(&self) -> usize {
        self.active_prefill + self.active_decode
    }

    fn queued_total(&self) -> usize {
        self.queued_prefill.len() + self.queued_decode.len()
    }

    fn queue(&self, class: SchedulerClass) -> &VecDeque<u64> {
        match class {
            SchedulerClass::Prefill => &self.queued_prefill,
            SchedulerClass::Decode => &self.queued_decode,
        }
    }

    fn queue_mut(&mut self, class: SchedulerClass) -> &mut VecDeque<u64> {
        match class {
            SchedulerClass::Prefill => &mut self.queued_prefill,
            SchedulerClass::Decode => &mut self.queued_decode,
        }
    }

    fn start_active(&mut self, admission_class: SchedulerClass, initial_phase: GenerationPhase) {
        match initial_phase {
            GenerationPhase::Prefill => self.active_prefill += 1,
            GenerationPhase::Decode => self.active_decode += 1,
        }
        match admission_class {
            SchedulerClass::Prefill => {
                self.prefill_admissions_since_decode += 1;
                self.admitted_prefill += 1;
            }
            SchedulerClass::Decode => {
                self.prefill_admissions_since_decode = 0;
                self.admitted_decode += 1;
            }
        }
    }

    fn finish_active(&mut self, phase: GenerationPhase, outcome: SchedulerOutcome) {
        match phase {
            GenerationPhase::Prefill => self.active_prefill = self.active_prefill.saturating_sub(1),
            GenerationPhase::Decode => self.active_decode = self.active_decode.saturating_sub(1),
        }
        match outcome {
            SchedulerOutcome::Completed => self.completed += 1,
            SchedulerOutcome::Cancelled => self.cancelled += 1,
            SchedulerOutcome::Failed => self.failed += 1,
        }
    }

    fn transition_to_decode(&mut self) {
        self.active_prefill = self.active_prefill.saturating_sub(1);
        self.active_decode += 1;
    }

    fn next_admissible_class(&self, prefill_burst: usize) -> Option<SchedulerClass> {
        let has_prefill = !self.queued_prefill.is_empty();
        let has_decode = !self.queued_decode.is_empty();
        match (has_prefill, has_decode) {
            (false, false) => None,
            (true, false) => Some(SchedulerClass::Prefill),
            (false, true) => Some(SchedulerClass::Decode),
            (true, true) => {
                if self.prefill_admissions_since_decode >= prefill_burst {
                    return Some(SchedulerClass::Decode);
                }
                let prefill_ticket = self.queued_prefill.front().copied().unwrap_or(u64::MAX);
                let decode_ticket = self.queued_decode.front().copied().unwrap_or(u64::MAX);
                if decode_ticket < prefill_ticket {
                    Some(SchedulerClass::Decode)
                } else {
                    Some(SchedulerClass::Prefill)
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SchedulerOutcome {
    Completed,
    Cancelled,
    Failed,
}

#[derive(Debug)]
pub(super) struct SchedulerPermit {
    scheduler: Arc<ModelScheduler>,
    phase: GenerationPhase,
    outcome: SchedulerOutcome,
}

impl SchedulerPermit {
    pub(super) fn transition_to_decode(&mut self) {
        if self.phase == GenerationPhase::Decode {
            return;
        }
        self.scheduler
            .state
            .lock_or_panic("scheduler")
            .transition_to_decode();
        self.phase = GenerationPhase::Decode;
    }

    pub(super) fn mark_failed(&mut self) {
        self.outcome = SchedulerOutcome::Failed;
    }

    pub(super) fn mark_cancelled(&mut self) {
        self.outcome = SchedulerOutcome::Cancelled;
    }
}

impl Drop for SchedulerPermit {
    fn drop(&mut self) {
        self.scheduler
            .state
            .lock_or_panic("scheduler")
            .finish_active(self.phase, self.outcome);
        self.scheduler.notify.notify_waiters();
    }
}

#[derive(Debug)]
struct QueuedSchedulerTicket {
    scheduler: Arc<ModelScheduler>,
    id: u64,
    class: SchedulerClass,
    admitted: bool,
    timeout: bool,
}

impl QueuedSchedulerTicket {
    fn admitted(&mut self) {
        self.admitted = true;
    }

    fn timed_out(&mut self) {
        self.timeout = true;
    }
}

impl Drop for QueuedSchedulerTicket {
    fn drop(&mut self) {
        if self.admitted {
            return;
        }
        let mut state = self.scheduler.state.lock_or_panic("scheduler");
        let queue = state.queue_mut(self.class);
        if let Some(index) = queue.iter().position(|ticket| *ticket == self.id) {
            queue.remove(index);
            if self.timeout {
                state.queue_timeouts += 1;
            } else {
                state.queued_cancelled += 1;
            }
        }
        drop(state);
        self.scheduler.notify.notify_waiters();
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ModelSchedulerSnapshot {
    pub(super) queued_prefill: usize,
    pub(super) queued_decode: usize,
    pub(super) active_prefill: usize,
    pub(super) active_decode: usize,
    pub(super) admitted_prefill: u64,
    pub(super) admitted_decode: u64,
    pub(super) completed: u64,
    pub(super) cancelled: u64,
    pub(super) failed: u64,
    pub(super) queued_cancelled: u64,
    pub(super) queue_timeouts: u64,
}

impl ModelSchedulerSnapshot {
    pub(super) fn queued_total(&self) -> usize {
        self.queued_prefill + self.queued_decode
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SchedulerAcquireError {
    QueueFull,
    QueueTimedOut,
    Cancelled,
}

impl ModelScheduler {
    pub(super) fn new(options: ModelSchedulerOptions) -> Self {
        Self {
            options,
            state: Mutex::new(ModelSchedulerState::default()),
            notify: Notify::new(),
        }
    }

    pub(super) fn classify_chat(&self, request: &ChatCompletionRequest) -> SchedulerClass {
        let chars = request
            .messages
            .iter()
            .map(|message| message.content.as_ref().map_or(0, String::len))
            .sum::<usize>()
            + request
                .tools
                .iter()
                .filter_map(|tool| serde_json::to_string(tool).ok())
                .map(|tool| tool.len())
                .sum::<usize>();
        self.classify_chars(chars)
    }

    pub(super) fn classify_completion(&self, request: &CompletionRequest) -> SchedulerClass {
        self.classify_chars(request.prompt.len())
    }

    fn classify_chars(&self, chars: usize) -> SchedulerClass {
        if chars >= self.options.prefill_threshold_chars {
            SchedulerClass::Prefill
        } else {
            SchedulerClass::Decode
        }
    }

    pub(super) async fn acquire(
        self: &Arc<Self>,
        admission_class: SchedulerClass,
        initial_phase: GenerationPhase,
        cancellation: &CancellationToken,
    ) -> Result<SchedulerPermit, SchedulerAcquireError> {
        if cancellation.is_cancelled() {
            return Err(SchedulerAcquireError::Cancelled);
        }
        if let Some(permit) = self.try_acquire_immediate(admission_class, initial_phase) {
            return Self::admit_unless_cancelled(permit, cancellation);
        }
        let mut ticket = self.enqueue(admission_class)?;
        let deadline = self
            .options
            .queue_timeout
            .map(|timeout| tokio::time::Instant::now() + timeout);
        loop {
            if cancellation.is_cancelled() {
                return Err(SchedulerAcquireError::Cancelled);
            }
            let notified = self.notify.notified();
            tokio::pin!(notified);
            if let Some(permit) = self.try_admit_queued(ticket.id, admission_class, initial_phase) {
                ticket.admitted();
                return Self::admit_unless_cancelled(permit, cancellation);
            }
            if let Some(deadline) = deadline {
                tokio::select! {
                    () = &mut notified => {}
                    () = cancellation.cancelled() => {
                        return Err(SchedulerAcquireError::Cancelled);
                    }
                    () = tokio::time::sleep_until(deadline) => {
                        ticket.timed_out();
                        return Err(SchedulerAcquireError::QueueTimedOut);
                    }
                }
            } else {
                tokio::select! {
                    () = &mut notified => {}
                    () = cancellation.cancelled() => {
                        return Err(SchedulerAcquireError::Cancelled);
                    }
                }
            }
        }
    }

    fn admit_unless_cancelled(
        mut permit: SchedulerPermit,
        cancellation: &CancellationToken,
    ) -> Result<SchedulerPermit, SchedulerAcquireError> {
        if cancellation.is_cancelled() {
            permit.mark_cancelled();
            return Err(SchedulerAcquireError::Cancelled);
        }
        Ok(permit)
    }

    fn try_acquire_immediate(
        self: &Arc<Self>,
        admission_class: SchedulerClass,
        initial_phase: GenerationPhase,
    ) -> Option<SchedulerPermit> {
        let mut state = self.state.lock_or_panic("scheduler");
        if state.active_total() >= self.options.concurrency_limit || state.queued_total() > 0 {
            return None;
        }
        state.start_active(admission_class, initial_phase);
        Some(SchedulerPermit {
            scheduler: Arc::clone(self),
            phase: initial_phase,
            outcome: SchedulerOutcome::Completed,
        })
    }

    fn enqueue(
        self: &Arc<Self>,
        class: SchedulerClass,
    ) -> Result<QueuedSchedulerTicket, SchedulerAcquireError> {
        let mut state = self.state.lock_or_panic("scheduler");
        if state.queued_total() >= self.options.queue_limit {
            return Err(SchedulerAcquireError::QueueFull);
        }
        state.next_ticket += 1;
        let id = state.next_ticket;
        state.queue_mut(class).push_back(id);
        drop(state);
        self.notify.notify_waiters();
        Ok(QueuedSchedulerTicket {
            scheduler: Arc::clone(self),
            id,
            class,
            admitted: false,
            timeout: false,
        })
    }

    fn try_admit_queued(
        self: &Arc<Self>,
        ticket: u64,
        admission_class: SchedulerClass,
        initial_phase: GenerationPhase,
    ) -> Option<SchedulerPermit> {
        let mut state = self.state.lock_or_panic("scheduler");
        if state.active_total() >= self.options.concurrency_limit {
            return None;
        }
        if state.next_admissible_class(self.options.prefill_burst)? != admission_class {
            return None;
        }
        if state.queue(admission_class).front().copied() != Some(ticket) {
            return None;
        }
        state.queue_mut(admission_class).pop_front();
        state.start_active(admission_class, initial_phase);
        Some(SchedulerPermit {
            scheduler: Arc::clone(self),
            phase: initial_phase,
            outcome: SchedulerOutcome::Completed,
        })
    }

    pub(super) fn snapshot(&self) -> ModelSchedulerSnapshot {
        let state = self.state.lock_or_panic("scheduler");
        ModelSchedulerSnapshot {
            queued_prefill: state.queued_prefill.len(),
            queued_decode: state.queued_decode.len(),
            active_prefill: state.active_prefill,
            active_decode: state.active_decode,
            admitted_prefill: state.admitted_prefill,
            admitted_decode: state.admitted_decode,
            completed: state.completed,
            cancelled: state.cancelled,
            failed: state.failed,
            queued_cancelled: state.queued_cancelled,
            queue_timeouts: state.queue_timeouts,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_options() -> ModelSchedulerOptions {
        ModelSchedulerOptions {
            concurrency_limit: 1,
            queue_limit: 1,
            queue_timeout: None,
            prefill_threshold_chars: 16,
            prefill_burst: 1,
        }
    }

    #[test]
    fn admitted_permit_is_cancelled_when_token_cancels_before_return() {
        let scheduler = Arc::new(ModelScheduler::new(test_options()));
        let cancellation = CancellationToken::new();
        let permit = scheduler
            .try_acquire_immediate(SchedulerClass::Decode, GenerationPhase::Decode)
            .expect("permit is admitted");

        cancellation.cancel();
        let result = ModelScheduler::admit_unless_cancelled(permit, &cancellation);

        assert_eq!(result.unwrap_err(), SchedulerAcquireError::Cancelled);
        let snapshot = scheduler.snapshot();
        assert_eq!(snapshot.active_decode, 0);
        assert_eq!(snapshot.cancelled, 1);
    }
}
