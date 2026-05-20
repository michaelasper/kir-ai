use super::scheduler::SchedulerPermit;
use super::{
    AppState, EngineErrorBody,
    lifecycle::StreamingGenerationRun,
    metrics::{
        record_failure_metrics, record_first_tool_delta_metrics, record_runtime_error_metrics,
        record_stream_client_disconnect_metrics, record_stream_stall_metrics,
        record_success_metrics, record_time_to_first_token_metrics,
        record_tool_argument_assembly_metrics, record_tool_finish_metrics,
        record_tool_intent_fill_metrics, record_tool_schema_validation_metrics,
        record_validated_tool_call_metrics,
    },
};
use super::{requests::ActiveRequest, scheduler::GenerationPhaseGuard};
use axum::response::sse::{Event, KeepAlive};
use futures::{Stream, StreamExt};
use llm_api::{ChatCompletionStreamResponse, CompletionStreamResponse, Usage};
use llm_backend::BackendStreamProgress;
use llm_runtime::{
    ChatCompletionStreamEvent, ChatCompletionStreamStage, CompletionStreamEvent, RuntimeError,
};
use serde_json::json;
use std::{
    convert::Infallible,
    time::{Duration, Instant},
};
use tokio::time::Instant as TokioInstant;

pub(super) fn stream_runtime_events<'a, E, S>(
    lifecycle: StreamRunLifecycle,
    events: S,
    streamed: bool,
) -> impl Stream<Item = Result<Event, Infallible>> + 'a
where
    E: EngineStreamEvent + 'a,
    E::Chunk: serde::Serialize + StreamChunkProgress + 'a,
    S: Stream<Item = Result<E, RuntimeError>> + Unpin + 'a,
{
    async_stream::stream! {
        let mut lifecycle = lifecycle;
        let mut events = events;
        let mut ttft_recorded = false;
        let mut first_tool_delta_recorded = false;
        let mut validated_tool_call_recorded = false;
        let mut stall_deadline = StreamStallDeadline::new(lifecycle.stream_stall_timeout());
        loop {
            match next_stream_event(
                &mut events,
                stall_deadline.deadline(),
                &lifecycle.active_request.cancellation,
            )
            .await
            {
                Ok(Some(Ok(event))) => match event.into_step() {
                    EngineStreamStep::Chunk(chunk) => {
                        if lifecycle.active_request.cancellation.is_cancelled() {
                            for event in lifecycle.finish_cancellation(
                                "request was cancelled before stream chunk delivery",
                                "decode",
                            ) {
                                yield event;
                            }
                            return;
                        }
                        if !ttft_recorded && chunk.has_real_delta() {
                            lifecycle.transition_to_decode();
                            record_time_to_first_token_metrics(
                                &lifecycle.state,
                                lifecycle.request_started.elapsed(),
                            );
                            ttft_recorded = true;
                        }
                        if !first_tool_delta_recorded && chunk.has_tool_delta() {
                            record_first_tool_delta_metrics(
                                &lifecycle.state,
                                lifecycle.request_started.elapsed(),
                            );
                            first_tool_delta_recorded = true;
                        }
                        if !validated_tool_call_recorded && chunk.has_tool_call_finish() {
                            let latency = lifecycle.request_started.elapsed();
                            record_tool_finish_metrics(&lifecycle.state, latency);
                            record_validated_tool_call_metrics(&lifecycle.state, latency);
                            validated_tool_call_recorded = true;
                        }
                        stall_deadline.record_chunk(&chunk);
                        yield sse_json_event(chunk);
                    }
                    EngineStreamStep::Progress(progress) => {
                        if lifecycle.active_request.cancellation.is_cancelled() {
                            for event in lifecycle.finish_cancellation(
                                "request was cancelled before stream progress delivery",
                                "prefill",
                            ) {
                                yield event;
                            }
                            return;
                        }
                        yield sse_json_event(progress);
                    }
                    EngineStreamStep::InternalProgress { bytes } => {
                        if lifecycle.active_request.cancellation.is_cancelled() {
                            for event in lifecycle.finish_cancellation(
                                "request was cancelled before internal stream progress processing",
                                "decode",
                            ) {
                                yield event;
                            }
                            return;
                        }
                        stall_deadline.record_internal_progress(bytes);
                    }
                    EngineStreamStep::ToolStage(stage) => {
                        if lifecycle.active_request.cancellation.is_cancelled() {
                            for event in lifecycle.finish_cancellation(
                                "request was cancelled before stream stage processing",
                                "decode",
                            ) {
                                yield event;
                            }
                            return;
                        }
                        record_tool_stage_metrics(
                            &lifecycle.state,
                            stage,
                            lifecycle.request_started.elapsed(),
                        );
                    }
                    EngineStreamStep::Complete(usage) => {
                        if let Err(events) = lifecycle.finish_success(&usage, streamed) {
                            for event in events {
                                yield event;
                            }
                            return;
                        }
                        yield Ok(Event::default().data("[DONE]"));
                        return;
                    }
                },
                Ok(Some(Err(err))) => {
                    for event in lifecycle.finish_runtime_error(err) {
                        yield event;
                    }
                    return;
                }
                Ok(None) => {
                    for event in lifecycle.finish_eof() {
                        yield event;
                    }
                    return;
                }
                Err(StreamWaitError::Stalled) => {
                    for event in lifecycle.finish_stall() {
                        yield event;
                    }
                    return;
                }
                Err(StreamWaitError::Cancelled) => {
                    for event in lifecycle.finish_cancellation(
                        "request was cancelled while waiting for stream output",
                        "decode",
                    ) {
                        yield event;
                    }
                    return;
                }
            }
        }
    }
}

fn record_tool_stage_metrics(
    state: &AppState,
    stage: ChatCompletionStreamStage,
    latency: Duration,
) {
    match stage {
        ChatCompletionStreamStage::ToolArgumentAssemblyComplete => {
            record_tool_argument_assembly_metrics(state, latency);
        }
        ChatCompletionStreamStage::ToolIntentFillComplete => {
            record_tool_intent_fill_metrics(state, latency);
        }
        ChatCompletionStreamStage::ToolSchemaValidationComplete => {
            record_tool_schema_validation_metrics(state, latency);
        }
    }
}

pub(super) fn engine_sse_keep_alive() -> KeepAlive {
    KeepAlive::new()
        .interval(Duration::from_millis(100))
        .text("llm-engine-heartbeat")
}

fn runtime_error_stream_events(err: RuntimeError) -> Vec<Result<Event, Infallible>> {
    engine_error_stream_events(EngineErrorBody::from_runtime_error(&err))
}

fn engine_error_stream_events(body: EngineErrorBody) -> Vec<Result<Event, Infallible>> {
    vec![sse_json_event(body), Ok(Event::default().data("[DONE]"))]
}

fn request_cancelled_stream_events(
    message: &'static str,
    phase: &'static str,
) -> Vec<Result<Event, Infallible>> {
    engine_error_stream_events(EngineErrorBody::new(message, "cancelled", phase, false))
}

fn stream_stalled_stream_events(timeout: Option<Duration>) -> Vec<Result<Event, Infallible>> {
    let message = match timeout {
        Some(timeout) => format!(
            "stream stalled for {} ms without meaningful backend output",
            timeout.as_millis()
        ),
        None => "stream stalled without meaningful backend output".to_owned(),
    };
    engine_error_stream_events(EngineErrorBody::new(
        message,
        "stream_stalled",
        "streaming",
        true,
    ))
}

fn stream_ended_without_completion_events() -> Vec<Result<Event, Infallible>> {
    engine_error_stream_events(EngineErrorBody::new(
        "stream ended before completion marker",
        "stream_incomplete",
        "streaming",
        true,
    ))
}

fn sse_json_event(value: impl serde::Serialize) -> Result<Event, Infallible> {
    let data = serde_json::to_string(&value).unwrap_or_else(|err| {
        json!({
            "error": {
                "message": format!("response serialization failed: {err}"),
                "type": "llm_engine_error"
            }
        })
        .to_string()
    });
    Ok(Event::default().data(data))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamTerminalOutcome {
    Success,
    RuntimeError,
    Cancelled,
    Stalled,
    BackendEof,
    ClientDisconnected,
}

pub(super) struct StreamRunLifecycle {
    state: AppState,
    request_id: String,
    model: String,
    active_request: ActiveRequest,
    scheduler_slot: SchedulerPermit,
    phase: GenerationPhaseGuard,
    request_started: Instant,
    terminal: Option<StreamTerminalOutcome>,
}

impl StreamRunLifecycle {
    pub(super) fn new(state: AppState, run: StreamingGenerationRun, model: String) -> Self {
        let StreamingGenerationRun {
            request_id,
            active_request,
            scheduler_slot,
            phase,
            request_started,
        } = run;
        Self {
            state,
            request_id,
            model,
            active_request,
            scheduler_slot,
            phase,
            request_started,
            terminal: None,
        }
    }

    pub(super) fn cancellation(&self) -> tokio_util::sync::CancellationToken {
        self.active_request.cancellation.clone()
    }

    fn stream_stall_timeout(&self) -> Option<Duration> {
        self.state.stream_stall_timeout
    }

    fn transition_to_decode(&mut self) {
        self.phase.transition_to_decode();
        self.scheduler_slot.transition_to_decode();
    }

    fn finish_success(
        &mut self,
        usage: &Usage,
        streamed: bool,
    ) -> Result<(), Vec<Result<Event, Infallible>>> {
        self.terminal = Some(StreamTerminalOutcome::Success);
        match self.active_request.mark_finished() {
            super::requests::RequestFinishResult::Finished => {
                record_success_metrics(
                    &self.state,
                    &self.request_id,
                    &self.model,
                    usage,
                    streamed,
                    self.request_started.elapsed(),
                );
                Ok(())
            }
            super::requests::RequestFinishResult::Cancelled => {
                self.scheduler_slot.mark_cancelled();
                record_failure_metrics(&self.state);
                Err(request_cancelled_stream_events(
                    "request was cancelled before response delivery",
                    "decode",
                ))
            }
            super::requests::RequestFinishResult::Missing => {
                self.scheduler_slot.mark_failed();
                record_failure_metrics(&self.state);
                Err(runtime_error_stream_events(RuntimeError::backend_failed(
                    "request lifecycle was missing before response delivery",
                )))
            }
        }
    }

    pub(super) fn finish_runtime_error(
        &mut self,
        err: RuntimeError,
    ) -> Vec<Result<Event, Infallible>> {
        self.terminal = Some(StreamTerminalOutcome::RuntimeError);
        match self.active_request.mark_finished() {
            super::requests::RequestFinishResult::Finished => {
                super::lifecycle::mark_scheduler_runtime_error(&mut self.scheduler_slot, &err);
                record_runtime_error_metrics(&self.state, &err);
                runtime_error_stream_events(err)
            }
            super::requests::RequestFinishResult::Cancelled => {
                self.scheduler_slot.mark_cancelled();
                record_failure_metrics(&self.state);
                request_cancelled_stream_events(
                    "request was cancelled before error delivery",
                    "decode",
                )
            }
            super::requests::RequestFinishResult::Missing => {
                self.scheduler_slot.mark_failed();
                record_failure_metrics(&self.state);
                runtime_error_stream_events(RuntimeError::backend_failed(
                    "request lifecycle was missing before error delivery",
                ))
            }
        }
    }

    fn finish_cancellation(
        &mut self,
        message: &'static str,
        phase: &'static str,
    ) -> Vec<Result<Event, Infallible>> {
        self.terminal = Some(StreamTerminalOutcome::Cancelled);
        match self.active_request.mark_finished() {
            super::requests::RequestFinishResult::Finished
            | super::requests::RequestFinishResult::Cancelled => {
                self.scheduler_slot.mark_cancelled();
                record_failure_metrics(&self.state);
                request_cancelled_stream_events(message, phase)
            }
            super::requests::RequestFinishResult::Missing => {
                self.scheduler_slot.mark_failed();
                record_failure_metrics(&self.state);
                runtime_error_stream_events(RuntimeError::backend_failed(
                    "request lifecycle was missing before stream cancellation",
                ))
            }
        }
    }

    fn finish_stall(&mut self) -> Vec<Result<Event, Infallible>> {
        self.terminal = Some(StreamTerminalOutcome::Stalled);
        self.active_request.cancellation.cancel();
        match self.active_request.mark_finished() {
            super::requests::RequestFinishResult::Finished => {
                self.scheduler_slot.mark_failed();
                record_stream_stall_metrics(&self.state);
                stream_stalled_stream_events(self.state.stream_stall_timeout)
            }
            super::requests::RequestFinishResult::Cancelled => {
                self.scheduler_slot.mark_cancelled();
                record_failure_metrics(&self.state);
                request_cancelled_stream_events(
                    "request was cancelled before stream stall",
                    "decode",
                )
            }
            super::requests::RequestFinishResult::Missing => {
                self.scheduler_slot.mark_failed();
                record_failure_metrics(&self.state);
                runtime_error_stream_events(RuntimeError::backend_failed(
                    "request lifecycle was missing before stream stall",
                ))
            }
        }
    }

    fn finish_eof(&mut self) -> Vec<Result<Event, Infallible>> {
        self.terminal = Some(StreamTerminalOutcome::BackendEof);
        match self.active_request.mark_finished() {
            super::requests::RequestFinishResult::Finished => {
                self.scheduler_slot.mark_failed();
                record_failure_metrics(&self.state);
                stream_ended_without_completion_events()
            }
            super::requests::RequestFinishResult::Cancelled => {
                self.scheduler_slot.mark_cancelled();
                record_failure_metrics(&self.state);
                request_cancelled_stream_events(
                    "request was cancelled before stream completion",
                    "decode",
                )
            }
            super::requests::RequestFinishResult::Missing => {
                self.scheduler_slot.mark_failed();
                record_failure_metrics(&self.state);
                runtime_error_stream_events(RuntimeError::backend_failed(
                    "request lifecycle was missing before stream completion",
                ))
            }
        }
    }
}

impl Drop for StreamRunLifecycle {
    fn drop(&mut self) {
        if self.terminal.is_some() {
            return;
        }
        self.terminal = Some(StreamTerminalOutcome::ClientDisconnected);
        self.active_request.cancellation.cancel();
        match self.active_request.mark_finished() {
            super::requests::RequestFinishResult::Finished => {
                self.scheduler_slot.mark_cancelled();
                record_stream_client_disconnect_metrics(&self.state);
            }
            super::requests::RequestFinishResult::Cancelled => {
                self.scheduler_slot.mark_cancelled();
                record_failure_metrics(&self.state);
            }
            super::requests::RequestFinishResult::Missing => {
                self.scheduler_slot.mark_failed();
                record_stream_client_disconnect_metrics(&self.state);
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamWaitError {
    Stalled,
    Cancelled,
}

#[derive(Debug)]
struct StreamStallDeadline {
    timeout: Option<Duration>,
    deadline: Option<TokioInstant>,
}

impl StreamStallDeadline {
    fn new(timeout: Option<Duration>) -> Self {
        Self {
            timeout,
            deadline: None,
        }
    }

    fn deadline(&self) -> Option<TokioInstant> {
        self.deadline
    }

    fn record_chunk(&mut self, chunk: &impl StreamChunkProgress) {
        if chunk.has_real_delta() {
            self.record_progress();
        }
    }

    fn record_internal_progress(&mut self, bytes: usize) {
        if bytes > 0 {
            self.record_progress();
        }
    }

    fn record_progress(&mut self) {
        if self.timeout.is_none() {
            return;
        }
        self.reset_deadline();
    }

    fn reset_deadline(&mut self) {
        if let Some(timeout) = self.timeout {
            self.deadline = Some(TokioInstant::now() + timeout);
        }
    }
}

async fn next_stream_event<S, T>(
    events: &mut S,
    deadline: Option<TokioInstant>,
    cancellation: &tokio_util::sync::CancellationToken,
) -> Result<Option<Result<T, RuntimeError>>, StreamWaitError>
where
    S: Stream<Item = Result<T, RuntimeError>> + Unpin,
{
    if cancellation.is_cancelled() {
        return Err(StreamWaitError::Cancelled);
    }
    let next = events.next();
    tokio::pin!(next);
    match deadline {
        Some(deadline) => {
            if TokioInstant::now() >= deadline {
                return Err(StreamWaitError::Stalled);
            }
            let sleep = tokio::time::sleep_until(deadline);
            tokio::pin!(sleep);
            tokio::select! {
                biased;
                () = cancellation.cancelled() => Err(StreamWaitError::Cancelled),
                result = &mut next => Ok(result),
                () = &mut sleep => Err(StreamWaitError::Stalled),
            }
        }
        None => {
            tokio::select! {
                biased;
                () = cancellation.cancelled() => Err(StreamWaitError::Cancelled),
                result = &mut next => Ok(result),
            }
        }
    }
}

pub(super) trait EngineStreamEvent {
    type Chunk;

    fn into_step(self) -> EngineStreamStep<Self::Chunk>;
}

pub(super) enum EngineStreamStep<C> {
    Chunk(C),
    Progress(BackendStreamProgress),
    InternalProgress { bytes: usize },
    ToolStage(ChatCompletionStreamStage),
    Complete(Usage),
}

pub(super) trait StreamChunkProgress {
    fn has_real_delta(&self) -> bool;
    fn has_tool_delta(&self) -> bool;
    fn has_tool_call_finish(&self) -> bool;
    fn real_delta_bytes(&self) -> usize;
}

impl EngineStreamEvent for ChatCompletionStreamEvent {
    type Chunk = ChatCompletionStreamResponse;

    fn into_step(
        self,
    ) -> EngineStreamStep<<ChatCompletionStreamEvent as EngineStreamEvent>::Chunk> {
        match self {
            Self::Chunk(chunk) => EngineStreamStep::Chunk(chunk),
            Self::Progress(progress) => EngineStreamStep::Progress(progress),
            Self::InternalProgress { bytes } => EngineStreamStep::InternalProgress { bytes },
            Self::Stage(stage) => EngineStreamStep::ToolStage(stage),
            Self::Complete(usage) => EngineStreamStep::Complete(usage),
        }
    }
}

impl EngineStreamEvent for CompletionStreamEvent {
    type Chunk = CompletionStreamResponse;

    fn into_step(self) -> EngineStreamStep<<CompletionStreamEvent as EngineStreamEvent>::Chunk> {
        match self {
            Self::Chunk(chunk) => EngineStreamStep::Chunk(chunk),
            Self::Progress(progress) => EngineStreamStep::Progress(progress),
            Self::Complete(usage) => EngineStreamStep::Complete(usage),
        }
    }
}

impl StreamChunkProgress for ChatCompletionStreamResponse {
    fn has_real_delta(&self) -> bool {
        self.choices.iter().any(|choice| {
            choice
                .delta
                .content
                .as_deref()
                .is_some_and(|text| !text.is_empty())
                || !choice.delta.tool_calls.is_empty()
        })
    }

    fn has_tool_delta(&self) -> bool {
        self.choices
            .iter()
            .any(|choice| !choice.delta.tool_calls.is_empty())
    }

    fn has_tool_call_finish(&self) -> bool {
        self.choices
            .iter()
            .any(|choice| choice.finish_reason.as_ref() == Some(&llm_api::FinishReason::ToolCalls))
    }

    fn real_delta_bytes(&self) -> usize {
        self.choices
            .iter()
            .map(|choice| {
                choice.delta.content.as_ref().map_or(0, String::len)
                    + choice
                        .delta
                        .tool_calls
                        .iter()
                        .map(|tool_call| {
                            let bytes = tool_call.id.as_ref().map_or(0, String::len)
                                + tool_call.function.as_ref().map_or(0, |function| {
                                    function.name.as_ref().map_or(0, String::len)
                                        + function.arguments.as_ref().map_or(0, String::len)
                                });
                            bytes.max(1)
                        })
                        .sum::<usize>()
            })
            .sum()
    }
}

impl StreamChunkProgress for CompletionStreamResponse {
    fn has_real_delta(&self) -> bool {
        self.real_delta_bytes() > 0
    }

    fn has_tool_delta(&self) -> bool {
        false
    }

    fn has_tool_call_finish(&self) -> bool {
        false
    }

    fn real_delta_bytes(&self) -> usize {
        self.choices.iter().map(|choice| choice.text.len()).sum()
    }
}
