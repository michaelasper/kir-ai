use super::{
    AppState, record_failure_metrics, record_runtime_error_metrics, record_success_metrics,
    record_time_to_first_token_metrics, runtime_error_metadata,
};
use super::{requests::ActiveRequest, scheduler::GenerationPhaseGuard};
use crate::engine::scheduler::SchedulerPermit;
use axum::response::sse::{Event, KeepAlive};
use futures::{Stream, StreamExt};
use llm_api::{ChatCompletionStreamResponse, CompletionStreamResponse, Usage};
use llm_backend::BackendError;
use llm_runtime::{ChatCompletionStreamEvent, CompletionStreamEvent, RuntimeError};
use serde_json::json;
use std::{
    convert::Infallible,
    time::{Duration, Instant},
};

pub(super) fn stream_runtime_events<'a, E, S>(
    state: AppState,
    active_request: ActiveRequest,
    scheduler_slot: SchedulerPermit,
    phase: GenerationPhaseGuard,
    events: S,
    request_started: Instant,
    streamed: bool,
) -> impl Stream<Item = Result<Event, Infallible>> + 'a
where
    E: EngineStreamEvent + 'a,
    E::Chunk: serde::Serialize + StreamChunkProgress + 'a,
    S: Stream<Item = Result<E, RuntimeError>> + Unpin + 'a,
{
    async_stream::stream! {
        let mut scheduler_slot = scheduler_slot;
        let active_request = active_request;
        let mut phase = phase;
        let mut events = events;
        let mut ttft_recorded = false;
        loop {
            match next_stream_event(
                &mut events,
                state.stream_stall_timeout,
                &active_request.cancellation,
            )
            .await
            {
                Ok(Some(Ok(event))) => match event.into_step() {
                    EngineStreamStep::Chunk(chunk) => {
                        if active_request.cancellation.is_cancelled() {
                            for event in mark_active_request_finished_for_stream_cancellation(
                                &state,
                                &active_request,
                                &mut scheduler_slot,
                                "request was cancelled before stream chunk delivery",
                                "decode",
                            ) {
                                yield event;
                            }
                            return;
                        }
                        if !ttft_recorded && chunk.has_real_delta() {
                            phase.transition_to_decode();
                            scheduler_slot.transition_to_decode();
                            record_time_to_first_token_metrics(&state, request_started.elapsed());
                            ttft_recorded = true;
                        }
                        yield sse_json_event(chunk);
                    }
                    EngineStreamStep::Complete(usage) => {
                        if let Err(events) = mark_active_request_finished_for_stream_success(
                            &state,
                            &active_request,
                            &mut scheduler_slot,
                        ) {
                            for event in events {
                                yield event;
                            }
                            return;
                        }
                        record_success_metrics(&state, &usage, streamed, request_started.elapsed());
                        yield Ok(Event::default().data("[DONE]"));
                        return;
                    }
                },
                Ok(Some(Err(err))) => {
                    for event in mark_active_request_finished_for_stream_error(
                        &state,
                        &active_request,
                        &mut scheduler_slot,
                        err,
                    ) {
                        yield event;
                    }
                    return;
                }
                Ok(None) => {
                    for event in mark_active_request_finished_for_stream_eof(
                        &state,
                        &active_request,
                        &mut scheduler_slot,
                    ) {
                        yield event;
                    }
                    return;
                }
                Err(StreamWaitError::Stalled) => {
                    for event in mark_active_request_finished_for_stream_stall(
                        &state,
                        &active_request,
                        &mut scheduler_slot,
                    ) {
                        yield event;
                    }
                    return;
                }
                Err(StreamWaitError::Cancelled) => {
                    for event in mark_active_request_finished_for_stream_cancellation(
                        &state,
                        &active_request,
                        &mut scheduler_slot,
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

pub(super) fn stream_runtime_error_events(
    state: &AppState,
    active_request: &ActiveRequest,
    scheduler_slot: &mut SchedulerPermit,
    err: RuntimeError,
) -> Vec<Result<Event, Infallible>> {
    mark_active_request_finished_for_stream_error(state, active_request, scheduler_slot, err)
}

pub(super) fn engine_sse_keep_alive() -> KeepAlive {
    KeepAlive::new()
        .interval(Duration::from_millis(100))
        .text("llm-engine-heartbeat")
}

fn runtime_error_stream_events(err: RuntimeError) -> Vec<Result<Event, Infallible>> {
    let metadata = runtime_error_metadata(&err);
    vec![
        sse_json_event(json!({
            "error": {
                "message": err.to_string(),
                "code": metadata.code,
                "phase": metadata.phase,
                "retryable": metadata.retryable,
                "type": "llm_engine_error"
            }
        })),
        Ok(Event::default().data("[DONE]")),
    ]
}

fn request_cancelled_stream_events(
    message: &'static str,
    phase: &'static str,
) -> Vec<Result<Event, Infallible>> {
    vec![
        sse_json_event(json!({
            "error": {
                "message": message,
                "code": "cancelled",
                "phase": phase,
                "retryable": false,
                "type": "llm_engine_error"
            }
        })),
        Ok(Event::default().data("[DONE]")),
    ]
}

fn stream_stalled_stream_events(timeout: Option<Duration>) -> Vec<Result<Event, Infallible>> {
    let message = match timeout {
        Some(timeout) => format!(
            "stream stalled for {} ms without backend output",
            timeout.as_millis()
        ),
        None => "stream stalled without backend output".to_owned(),
    };
    vec![
        sse_json_event(json!({
            "error": {
                "message": message,
                "code": "stream_stalled",
                "phase": "streaming",
                "retryable": true,
                "type": "llm_engine_error"
            }
        })),
        Ok(Event::default().data("[DONE]")),
    ]
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
enum StreamWaitError {
    Stalled,
    Cancelled,
}

async fn next_stream_event<S, T>(
    events: &mut S,
    timeout: Option<Duration>,
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
    match timeout {
        Some(timeout) => {
            tokio::select! {
                biased;
                () = cancellation.cancelled() => Err(StreamWaitError::Cancelled),
                result = &mut next => Ok(result),
                () = tokio::time::sleep(timeout) => Err(StreamWaitError::Stalled),
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
    Complete(Usage),
}

pub(super) trait StreamChunkProgress {
    fn has_real_delta(&self) -> bool;
}

impl EngineStreamEvent for ChatCompletionStreamEvent {
    type Chunk = ChatCompletionStreamResponse;

    fn into_step(
        self,
    ) -> EngineStreamStep<<ChatCompletionStreamEvent as EngineStreamEvent>::Chunk> {
        match self {
            Self::Chunk(chunk) => EngineStreamStep::Chunk(chunk),
            Self::Complete(usage) => EngineStreamStep::Complete(usage),
        }
    }
}

impl EngineStreamEvent for CompletionStreamEvent {
    type Chunk = CompletionStreamResponse;

    fn into_step(self) -> EngineStreamStep<<CompletionStreamEvent as EngineStreamEvent>::Chunk> {
        match self {
            Self::Chunk(chunk) => EngineStreamStep::Chunk(chunk),
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
}

impl StreamChunkProgress for CompletionStreamResponse {
    fn has_real_delta(&self) -> bool {
        self.choices.iter().any(|choice| !choice.text.is_empty())
    }
}

fn mark_active_request_finished_for_stream_success(
    state: &AppState,
    active_request: &ActiveRequest,
    scheduler_slot: &mut SchedulerPermit,
) -> Result<(), Vec<Result<Event, Infallible>>> {
    match active_request.mark_finished() {
        super::requests::RequestFinishResult::Finished => Ok(()),
        super::requests::RequestFinishResult::Cancelled => {
            scheduler_slot.mark_cancelled();
            record_failure_metrics(state);
            Err(request_cancelled_stream_events(
                "request was cancelled before response delivery",
                "decode",
            ))
        }
        super::requests::RequestFinishResult::Missing => {
            scheduler_slot.mark_failed();
            record_failure_metrics(state);
            Err(runtime_error_stream_events(RuntimeError::Backend(
                BackendError::Other(
                    "request lifecycle was missing before response delivery".to_owned(),
                ),
            )))
        }
    }
}

fn mark_active_request_finished_for_stream_error(
    state: &AppState,
    active_request: &ActiveRequest,
    scheduler_slot: &mut SchedulerPermit,
    err: RuntimeError,
) -> Vec<Result<Event, Infallible>> {
    match active_request.mark_finished() {
        super::requests::RequestFinishResult::Finished => {
            super::lifecycle::mark_scheduler_runtime_error(scheduler_slot, &err);
            record_runtime_error_metrics(state, &err);
            runtime_error_stream_events(err)
        }
        super::requests::RequestFinishResult::Cancelled => {
            scheduler_slot.mark_cancelled();
            record_failure_metrics(state);
            request_cancelled_stream_events("request was cancelled before error delivery", "decode")
        }
        super::requests::RequestFinishResult::Missing => {
            scheduler_slot.mark_failed();
            record_failure_metrics(state);
            runtime_error_stream_events(RuntimeError::Backend(BackendError::Other(
                "request lifecycle was missing before error delivery".to_owned(),
            )))
        }
    }
}

fn mark_active_request_finished_for_stream_cancellation(
    state: &AppState,
    active_request: &ActiveRequest,
    scheduler_slot: &mut SchedulerPermit,
    message: &'static str,
    phase: &'static str,
) -> Vec<Result<Event, Infallible>> {
    match active_request.mark_finished() {
        super::requests::RequestFinishResult::Finished
        | super::requests::RequestFinishResult::Cancelled => {
            scheduler_slot.mark_cancelled();
            record_failure_metrics(state);
            request_cancelled_stream_events(message, phase)
        }
        super::requests::RequestFinishResult::Missing => {
            scheduler_slot.mark_failed();
            record_failure_metrics(state);
            runtime_error_stream_events(RuntimeError::Backend(BackendError::Other(
                "request lifecycle was missing before stream cancellation".to_owned(),
            )))
        }
    }
}

fn mark_active_request_finished_for_stream_stall(
    state: &AppState,
    active_request: &ActiveRequest,
    scheduler_slot: &mut SchedulerPermit,
) -> Vec<Result<Event, Infallible>> {
    match active_request.mark_finished() {
        super::requests::RequestFinishResult::Finished => {
            scheduler_slot.mark_failed();
            record_failure_metrics(state);
            stream_stalled_stream_events(state.stream_stall_timeout)
        }
        super::requests::RequestFinishResult::Cancelled => {
            scheduler_slot.mark_cancelled();
            record_failure_metrics(state);
            request_cancelled_stream_events("request was cancelled before stream stall", "decode")
        }
        super::requests::RequestFinishResult::Missing => {
            scheduler_slot.mark_failed();
            record_failure_metrics(state);
            runtime_error_stream_events(RuntimeError::Backend(BackendError::Other(
                "request lifecycle was missing before stream stall".to_owned(),
            )))
        }
    }
}

fn mark_active_request_finished_for_stream_eof(
    state: &AppState,
    active_request: &ActiveRequest,
    scheduler_slot: &mut SchedulerPermit,
) -> Vec<Result<Event, Infallible>> {
    match active_request.mark_finished() {
        super::requests::RequestFinishResult::Finished => Vec::new(),
        super::requests::RequestFinishResult::Cancelled => {
            scheduler_slot.mark_cancelled();
            record_failure_metrics(state);
            request_cancelled_stream_events(
                "request was cancelled before stream completion",
                "decode",
            )
        }
        super::requests::RequestFinishResult::Missing => {
            scheduler_slot.mark_failed();
            record_failure_metrics(state);
            runtime_error_stream_events(RuntimeError::Backend(BackendError::Other(
                "request lifecycle was missing before stream completion".to_owned(),
            )))
        }
    }
}
