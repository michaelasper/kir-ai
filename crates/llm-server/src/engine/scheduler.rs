use crate::sync_ext::FailPoisonedMutex;
use llm_api::{
    ChatCompletionRequest, CompletionRequest, FunctionDefinition, ToolCallType, ToolDefinition,
};
use serde_json::Value;
use std::{
    collections::VecDeque,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
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
    prefill_yields: u64,
    prefill_yields_to_decode: u64,
    prefill_yield_reacquire_waits: u64,
    prefill_yield_reacquire_wait_nanos_total: u64,
    prefill_yield_reacquire_wait_nanos_max: u64,
}

#[derive(Debug, Clone, Copy)]
struct PrefillYieldRelease {
    admitted_decode_before: u64,
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
        self.finish_inactive(outcome);
    }

    fn finish_inactive(&mut self, outcome: SchedulerOutcome) {
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

    fn release_prefill_for_yield(&mut self) -> PrefillYieldRelease {
        self.active_prefill = self.active_prefill.saturating_sub(1);
        PrefillYieldRelease {
            admitted_decode_before: self.admitted_decode,
        }
    }

    fn record_prefill_yield_success(&mut self, release: PrefillYieldRelease, wait: Duration) {
        self.prefill_yields += 1;
        if self.admitted_decode > release.admitted_decode_before {
            self.prefill_yields_to_decode += 1;
        }
        self.record_prefill_yield_reacquire_wait(wait);
    }

    fn record_prefill_yield_reacquire_wait(&mut self, wait: Duration) {
        let wait_nanos = duration_nanos_u64(wait);
        self.prefill_yield_reacquire_waits = self.prefill_yield_reacquire_waits.saturating_add(1);
        self.prefill_yield_reacquire_wait_nanos_total = self
            .prefill_yield_reacquire_wait_nanos_total
            .saturating_add(wait_nanos);
        self.prefill_yield_reacquire_wait_nanos_max =
            self.prefill_yield_reacquire_wait_nanos_max.max(wait_nanos);
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
    active: bool,
    terminal_on_drop: bool,
}

impl SchedulerPermit {
    pub(super) fn transition_to_decode(&mut self) {
        if self.phase == GenerationPhase::Decode || !self.active {
            return;
        }
        self.scheduler
            .state
            .lock_or_panic("scheduler")
            .transition_to_decode();
        self.phase = GenerationPhase::Decode;
    }

    pub(super) async fn yield_prefill_chunk(
        &mut self,
        cancellation: &CancellationToken,
    ) -> Result<(), SchedulerAcquireError> {
        if self.phase == GenerationPhase::Decode || !self.active {
            return Ok(());
        }
        if cancellation.is_cancelled() {
            self.mark_cancelled();
            return Err(SchedulerAcquireError::Cancelled);
        }
        let scheduler = Arc::clone(&self.scheduler);
        let release = {
            let mut state = scheduler.state.lock_or_panic("scheduler");
            state.release_prefill_for_yield()
        };
        self.active = false;
        self.terminal_on_drop = false;
        scheduler.notify.notify_waiters();

        let wait_started = Instant::now();
        let permit = match scheduler
            .clone()
            .acquire(
                SchedulerClass::Prefill,
                GenerationPhase::Prefill,
                cancellation,
            )
            .await
        {
            Ok(permit) => permit,
            Err(err) => {
                self.complete_readmission_error(err);
                return Err(err);
            }
        };
        scheduler
            .state
            .lock_or_panic("scheduler")
            .record_prefill_yield_success(release, wait_started.elapsed());
        *self = permit;
        Ok(())
    }

    fn complete_readmission_error(&mut self, err: SchedulerAcquireError) {
        match err {
            SchedulerAcquireError::Cancelled => {
                self.outcome = SchedulerOutcome::Cancelled;
                self.terminal_on_drop = true;
            }
            SchedulerAcquireError::CancelledAfterAdmission => {
                self.terminal_on_drop = false;
            }
            SchedulerAcquireError::QueueFull | SchedulerAcquireError::QueueTimedOut => {
                self.outcome = SchedulerOutcome::Failed;
                self.terminal_on_drop = true;
            }
        }
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
        if !self.terminal_on_drop {
            return;
        }
        let mut state = self.scheduler.state.lock_or_panic("scheduler");
        if self.active {
            state.finish_active(self.phase, self.outcome);
        } else {
            state.finish_inactive(self.outcome);
        }
        drop(state);
        self.scheduler.notify.notify_waiters();
    }
}

#[derive(Debug, Clone)]
pub(super) struct SharedSchedulerPermit {
    inner: Arc<Mutex<Option<SchedulerPermit>>>,
}

impl SharedSchedulerPermit {
    pub(super) fn new(permit: SchedulerPermit) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Some(permit))),
        }
    }

    pub(super) fn transition_to_decode(&self) {
        if let Some(permit) = self.inner.lock_or_panic("scheduler permit").as_mut() {
            permit.transition_to_decode();
        }
    }

    pub(super) async fn yield_prefill_chunk(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<(), SchedulerAcquireError> {
        let Some(mut permit) = self.inner.lock_or_panic("scheduler permit").take() else {
            return Ok(());
        };
        match permit.yield_prefill_chunk(cancellation).await {
            Ok(()) => {
                *self.inner.lock_or_panic("scheduler permit") = Some(permit);
                Ok(())
            }
            Err(err) => Err(err),
        }
    }

    pub(super) fn mark_failed(&self) {
        if let Some(permit) = self.inner.lock_or_panic("scheduler permit").as_mut() {
            permit.mark_failed();
        }
    }

    pub(super) fn mark_cancelled(&self) {
        if let Some(permit) = self.inner.lock_or_panic("scheduler permit").as_mut() {
            permit.mark_cancelled();
        }
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
    pub(super) prefill_yields: u64,
    pub(super) prefill_yields_to_decode: u64,
    pub(super) prefill_yield_reacquire_waits: u64,
    pub(super) prefill_yield_reacquire_wait_nanos_total: u64,
    pub(super) prefill_yield_reacquire_wait_nanos_max: u64,
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
    CancelledAfterAdmission,
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
        self.classify_estimated_tokens(estimated_chat_tokens(request))
    }

    pub(super) fn classify_completion(&self, request: &CompletionRequest) -> SchedulerClass {
        self.classify_estimated_tokens(estimated_text_tokens(&request.prompt))
    }

    fn classify_estimated_tokens(&self, tokens: usize) -> SchedulerClass {
        // Keep the historical option field name for API compatibility; the
        // scheduler compares it to estimated prompt tokens.
        if tokens >= self.options.prefill_threshold_chars {
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
            return Err(SchedulerAcquireError::CancelledAfterAdmission);
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
            active: true,
            terminal_on_drop: true,
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
            active: true,
            terminal_on_drop: true,
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
            prefill_yields: state.prefill_yields,
            prefill_yields_to_decode: state.prefill_yields_to_decode,
            prefill_yield_reacquire_waits: state.prefill_yield_reacquire_waits,
            prefill_yield_reacquire_wait_nanos_total: state
                .prefill_yield_reacquire_wait_nanos_total,
            prefill_yield_reacquire_wait_nanos_max: state.prefill_yield_reacquire_wait_nanos_max,
        }
    }
}

fn duration_nanos_u64(duration: Duration) -> u64 {
    duration.as_nanos().min(u128::from(u64::MAX)) as u64
}

fn estimated_chat_tokens(request: &ChatCompletionRequest) -> usize {
    request
        .messages
        .iter()
        .map(|message| message.content.as_deref().map_or(0, estimated_text_tokens))
        .chain(request.tools.iter().map(estimated_tool_definition_tokens))
        .fold(0usize, usize::saturating_add)
}

fn estimated_tool_definition_tokens(tool: &ToolDefinition) -> usize {
    let mut tokens = json_object_wrapper_tokens();
    let mut has_field = false;
    add_json_object_field_tokens(
        &mut tokens,
        &mut has_field,
        "type",
        estimated_tool_call_type_tokens(&tool.tool_type),
    );
    add_json_object_field_tokens(
        &mut tokens,
        &mut has_field,
        "function",
        estimated_function_definition_tokens(&tool.function),
    );
    tokens
}

fn estimated_tool_call_type_tokens(tool_type: &ToolCallType) -> usize {
    match tool_type {
        ToolCallType::Function => estimated_json_string_tokens("function"),
    }
}

fn estimated_function_definition_tokens(function: &FunctionDefinition) -> usize {
    let mut tokens = json_object_wrapper_tokens();
    let mut has_field = false;
    add_json_object_field_tokens(
        &mut tokens,
        &mut has_field,
        "name",
        estimated_json_string_tokens(&function.name),
    );
    if let Some(description) = &function.description {
        add_json_object_field_tokens(
            &mut tokens,
            &mut has_field,
            "description",
            estimated_json_string_tokens(description),
        );
    }
    add_json_object_field_tokens(
        &mut tokens,
        &mut has_field,
        "parameters",
        estimated_json_value_tokens(&function.parameters),
    );
    tokens
}

fn estimated_json_value_tokens(value: &Value) -> usize {
    match value {
        Value::Null => 1,
        Value::Bool(_) => 1,
        Value::Number(number) => estimated_json_number_tokens(number),
        Value::String(value) => estimated_json_string_tokens(value),
        Value::Array(values) => {
            let mut tokens = json_array_wrapper_tokens();
            let mut has_item = false;
            for value in values {
                if has_item {
                    tokens = tokens.saturating_add(1);
                } else {
                    has_item = true;
                }
                tokens = tokens.saturating_add(estimated_json_value_tokens(value));
            }
            tokens
        }
        Value::Object(object) => {
            let mut tokens = json_object_wrapper_tokens();
            let mut has_field = false;
            for (key, value) in object {
                add_json_object_field_tokens(
                    &mut tokens,
                    &mut has_field,
                    key,
                    estimated_json_value_tokens(value),
                );
            }
            tokens
        }
    }
}

fn estimated_json_string_tokens(value: &str) -> usize {
    estimated_text_tokens(value).saturating_add(1)
}

fn estimated_json_number_tokens(number: &serde_json::Number) -> usize {
    if let Some(value) = number.as_u64() {
        return decimal_digits(value).div_ceil(4);
    }
    if let Some(value) = number.as_i64() {
        return usize::from(value.is_negative())
            .saturating_add(decimal_digits(value.unsigned_abs()))
            .div_ceil(4);
    }
    if number.as_f64().is_some() {
        return 8;
    }
    usize::MAX / 2
}

fn estimated_text_tokens(value: &str) -> usize {
    let mut ascii_bytes = 0usize;
    let mut non_ascii_tokens = 0usize;
    for character in value.chars() {
        if character.is_ascii() {
            ascii_bytes = ascii_bytes.saturating_add(1);
        } else {
            non_ascii_tokens =
                non_ascii_tokens.saturating_add(character.len_utf8().saturating_sub(1).max(1));
        }
    }
    let byte_estimate = ascii_bytes.div_ceil(4).saturating_add(non_ascii_tokens);
    let word_estimate = value.split_whitespace().count();
    byte_estimate.max(word_estimate)
}

fn decimal_digits(mut value: u64) -> usize {
    let mut digits = 1;
    while value >= 10 {
        value /= 10;
        digits += 1;
    }
    digits
}

fn add_json_object_field_tokens(
    tokens: &mut usize,
    has_field: &mut bool,
    key: &str,
    value_tokens: usize,
) {
    if *has_field {
        *tokens = (*tokens).saturating_add(1);
    } else {
        *has_field = true;
    }
    *tokens = (*tokens)
        .saturating_add(estimated_json_string_tokens(key))
        .saturating_add(1)
        .saturating_add(value_tokens);
}

fn json_object_wrapper_tokens() -> usize {
    1
}

fn json_array_wrapper_tokens() -> usize {
    1
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_api::{ChatMessage, ToolDefinition};
    use serde_json::json;

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

        assert_eq!(
            result.unwrap_err(),
            SchedulerAcquireError::CancelledAfterAdmission
        );
        let snapshot = scheduler.snapshot();
        assert_eq!(snapshot.active_decode, 0);
        assert_eq!(snapshot.cancelled, 1);
    }

    #[tokio::test]
    async fn prefill_yield_reacquires_after_queued_decode() {
        let scheduler = Arc::new(ModelScheduler::new(ModelSchedulerOptions {
            queue_limit: 2,
            ..test_options()
        }));
        let prefill_cancellation = CancellationToken::new();
        let decode_cancellation = CancellationToken::new();
        let mut prefill = scheduler
            .clone()
            .acquire(
                SchedulerClass::Prefill,
                GenerationPhase::Prefill,
                &prefill_cancellation,
            )
            .await
            .expect("initial prefill is admitted");
        let decode_scheduler = Arc::clone(&scheduler);
        let decode = tokio::spawn(async move {
            decode_scheduler
                .clone()
                .acquire(
                    SchedulerClass::Decode,
                    GenerationPhase::Decode,
                    &decode_cancellation,
                )
                .await
                .expect("queued decode is admitted")
        });

        tokio::time::timeout(Duration::from_millis(500), async {
            while scheduler.snapshot().queued_decode != 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("decode queues behind active prefill");

        let mut yield_prefill = Box::pin(prefill.yield_prefill_chunk(&prefill_cancellation));
        let decode_permit = tokio::select! {
            result = decode => result.expect("decode task completes"),
            result = &mut yield_prefill => {
                result.expect("prefill yield should wait for queued decode first");
                panic!("prefill reacquired before queued decode");
            }
        };
        let snapshot = scheduler.snapshot();
        assert_eq!(snapshot.active_prefill, 0);
        assert_eq!(snapshot.active_decode, 1);
        assert_eq!(snapshot.queued_prefill, 1);
        assert_eq!(snapshot.prefill_yields, 0);
        assert_eq!(snapshot.prefill_yields_to_decode, 0);

        drop(decode_permit);
        yield_prefill
            .await
            .expect("prefill reacquires after decode finishes");

        let snapshot = scheduler.snapshot();
        assert_eq!(snapshot.active_prefill, 1);
        assert_eq!(snapshot.active_decode, 0);
        assert_eq!(snapshot.admitted_prefill, 2);
        assert_eq!(snapshot.admitted_decode, 1);
        assert_eq!(snapshot.prefill_yields, 1);
        assert_eq!(snapshot.prefill_yields_to_decode, 1);
        assert_eq!(snapshot.prefill_yield_reacquire_waits, 1);
    }

    #[tokio::test]
    async fn failed_prefill_readmission_does_not_count_successful_yield_metrics() {
        let scheduler = Arc::new(ModelScheduler::new(test_options()));
        let prefill_cancellation = CancellationToken::new();
        let decode_cancellation = CancellationToken::new();
        let mut prefill = scheduler
            .clone()
            .acquire(
                SchedulerClass::Prefill,
                GenerationPhase::Prefill,
                &prefill_cancellation,
            )
            .await
            .expect("initial prefill is admitted");
        let decode_scheduler = Arc::clone(&scheduler);
        let decode = tokio::spawn(async move {
            decode_scheduler
                .clone()
                .acquire(
                    SchedulerClass::Decode,
                    GenerationPhase::Decode,
                    &decode_cancellation,
                )
                .await
        });

        tokio::time::timeout(Duration::from_millis(500), async {
            while scheduler.snapshot().queued_decode != 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("decode queues behind active prefill");

        let err = prefill
            .yield_prefill_chunk(&prefill_cancellation)
            .await
            .expect_err("prefill readmission queue is full");
        assert_eq!(err, SchedulerAcquireError::QueueFull);

        let snapshot = scheduler.snapshot();
        assert_eq!(snapshot.prefill_yields, 0);
        assert_eq!(snapshot.prefill_yields_to_decode, 0);
        assert_eq!(snapshot.prefill_yield_reacquire_waits, 0);

        decode.abort();
    }

    #[test]
    fn yielded_prefill_cancelled_after_readmission_counts_once() {
        let scheduler = Arc::new(ModelScheduler::new(test_options()));
        let cancellation = CancellationToken::new();
        let mut yielded = SchedulerPermit {
            scheduler: Arc::clone(&scheduler),
            phase: GenerationPhase::Prefill,
            outcome: SchedulerOutcome::Completed,
            active: false,
            terminal_on_drop: false,
        };
        let readmitted = scheduler
            .try_acquire_immediate(SchedulerClass::Prefill, GenerationPhase::Prefill)
            .expect("readmission permit is admitted");

        cancellation.cancel();
        let err = ModelScheduler::admit_unless_cancelled(readmitted, &cancellation)
            .expect_err("admitted readmission observes cancellation");
        yielded.complete_readmission_error(err);
        drop(yielded);

        let snapshot = scheduler.snapshot();
        assert_eq!(snapshot.active_prefill, 0);
        assert_eq!(snapshot.cancelled, 1);
    }

    #[test]
    fn chat_classification_counts_tool_schema_with_token_estimate() {
        let scheduler = ModelScheduler::new(ModelSchedulerOptions {
            prefill_threshold_chars: 48,
            ..test_options()
        });
        let tool_description = "x".repeat(96);
        let request = ChatCompletionRequest {
            model: "test-model".to_owned(),
            messages: vec![ChatMessage::user("short")],
            tools: vec![ToolDefinition::function(
                "lookup_customer_profile",
                "looks up customer profile data",
                json!({
                    "type": "object",
                    "properties": {
                        "customer_id": {
                            "type": "string",
                            "description": tool_description,
                        },
                    },
                    "required": ["customer_id"],
                }),
            )],
            ..ChatCompletionRequest::default()
        };

        assert!(estimated_tool_definition_tokens(&request.tools[0]) >= 48);
        assert_eq!(scheduler.classify_chat(&request), SchedulerClass::Prefill);
    }

    #[test]
    fn completion_classification_uses_token_estimate_for_multibyte_text() {
        let scheduler = ModelScheduler::new(ModelSchedulerOptions {
            prefill_threshold_chars: 8,
            ..test_options()
        });
        let prompt = "é".repeat(4);
        let request = CompletionRequest {
            prompt: prompt.clone(),
            ..CompletionRequest::default()
        };

        assert!(prompt.len() >= 8);
        assert_eq!(
            scheduler.classify_completion(&request),
            SchedulerClass::Decode
        );
    }

    #[test]
    fn completion_classification_uses_token_estimate_for_long_ascii_words() {
        let scheduler = ModelScheduler::new(ModelSchedulerOptions {
            prefill_threshold_chars: 16,
            ..test_options()
        });
        let request = CompletionRequest {
            prompt: "antidisestablishmentarianism".to_owned(),
            ..CompletionRequest::default()
        };

        assert!(request.prompt.len() >= 16);
        assert_eq!(
            scheduler.classify_completion(&request),
            SchedulerClass::Decode
        );
    }

    #[test]
    fn completion_classification_counts_ascii_word_boundaries() {
        let scheduler = ModelScheduler::new(ModelSchedulerOptions {
            prefill_threshold_chars: 16,
            ..test_options()
        });
        let request = CompletionRequest {
            prompt: "a ".repeat(20),
            ..CompletionRequest::default()
        };

        assert_eq!(
            scheduler.classify_completion(&request),
            SchedulerClass::Prefill
        );
    }
}
