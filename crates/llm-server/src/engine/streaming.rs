use super::error::runtime_error_metadata;
use super::scheduler::{SchedulerAcquireError, SharedSchedulerPermit};
use super::{
    AppState, EngineErrorBody,
    lifecycle::StreamingGenerationRun,
    metrics::{
        PendingToolStreamObservation, record_failure_metrics,
        record_first_tool_delta_after_ttft_metrics, record_first_tool_delta_metrics,
        record_runtime_error_metrics, record_stream_client_disconnect_metrics,
        record_stream_stall_metrics, record_success_metrics, record_time_to_first_token_metrics,
        record_tool_argument_assembly_metrics, record_tool_finish_metrics,
        record_tool_intent_fill_metrics, record_tool_schema_validation_metrics,
        record_tool_stream_observation, record_validated_tool_call_metrics,
    },
};
use super::{requests::ActiveRequest, scheduler::GenerationPhaseGuard};
use async_trait::async_trait;
use axum::response::sse::{Event, KeepAlive};
use futures::{Stream, StreamExt};
use llm_api::Usage;
use llm_backend_contracts::{
    BackendError, BackendPrefillChunkAdmission, BackendPrefillChunkAdmissionHook,
    BackendStreamProgress,
};
use llm_runtime::{
    ChatCompletionStreamEvent, ChatCompletionStreamStage, CompletionStreamEvent,
    RequestCacheIdentity, RuntimeError, StreamProgressMetadata,
};
use std::{
    convert::Infallible,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::time::Instant as TokioInstant;

const SSE_SERIALIZATION_FALLBACK_DATA: &str = concat!(
    r#"{"error":{"message":"response serialization failed","#,
    r#""code":"response_serialization_failed","#,
    r#""phase":"response_serialization","#,
    r#""retryable":true,"type":"llm_engine_error"}}"#
);

pub(super) fn stream_runtime_events<'a, E, S>(
    lifecycle: StreamRunLifecycle,
    events: S,
    streamed: bool,
) -> impl Stream<Item = Result<Event, Infallible>> + 'a
where
    E: EngineStreamEvent + 'a,
    E::Chunk: serde::Serialize + 'a,
    S: Stream<Item = Result<E, RuntimeError>> + Unpin + 'a,
{
    async_stream::stream! {
        let mut lifecycle = lifecycle;
        let mut events = events;
        let mut ttft_recorded = false;
        let mut first_tool_delta_recorded = false;
        let mut first_decode_at = None;
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
                    EngineStreamStep::Chunk { chunk, progress } => {
                        if lifecycle.active_request.cancellation.is_cancelled() {
                            for event in lifecycle.finish_cancellation(
                                "request was cancelled before stream chunk delivery",
                                "decode",
                            ) {
                                yield event;
                            }
                            return;
                        }
                        let now = Instant::now();
                        if !ttft_recorded && progress.has_real_delta() {
                            lifecycle.transition_to_decode();
                            first_decode_at = Some(now);
                            record_time_to_first_token_metrics(
                                &lifecycle.state,
                                now.duration_since(lifecycle.request_started),
                            );
                            ttft_recorded = true;
                        }
                        if !first_tool_delta_recorded && progress.has_tool_delta() {
                            let latency = now.duration_since(lifecycle.request_started);
                            record_first_tool_delta_metrics(&lifecycle.state, latency);
                            lifecycle.tool_stream.record_kir_first_tool_delta(latency);
                            if let Some(first_decode_at) = first_decode_at {
                                let after_ttft_latency = now.duration_since(first_decode_at);
                                record_first_tool_delta_after_ttft_metrics(
                                    &lifecycle.state,
                                    after_ttft_latency,
                                );
                                lifecycle
                                    .tool_stream
                                    .record_kir_first_tool_delta_after_ttft(after_ttft_latency);
                            }
                            first_tool_delta_recorded = true;
                        }
                        if !validated_tool_call_recorded && progress.has_tool_call_finish() {
                            let latency = lifecycle.request_started.elapsed();
                            record_tool_finish_metrics(&lifecycle.state, latency);
                            record_validated_tool_call_metrics(&lifecycle.state, latency);
                            lifecycle.tool_stream.record_tool_finish(latency);
                            lifecycle.tool_stream.record_validated_tool_call(latency);
                            validated_tool_call_recorded = true;
                        }
                        stall_deadline.record_progress_metadata(progress);
                        yield sse_json_event(chunk);
                    }
                    EngineStreamStep::Progress(progress) => {
                        if lifecycle.tool_stream.record_backend_progress(&progress) {
                            continue;
                        }
                        if lifecycle.active_request.cancellation.is_cancelled() {
                            for event in lifecycle.finish_cancellation(
                                "request was cancelled before stream progress delivery",
                                "prefill",
                            ) {
                                yield event;
                            }
                            return;
                        }
                        lifecycle.record_prefill_progress(&progress);
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
                            &mut lifecycle.tool_stream,
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
    tool_stream: &mut PendingToolStreamObservation,
    stage: ChatCompletionStreamStage,
    latency: Duration,
) {
    match stage {
        ChatCompletionStreamStage::ToolArgumentAssemblyComplete => {
            record_tool_argument_assembly_metrics(state, latency);
            tool_stream.record_tool_argument_assembly(latency);
        }
        ChatCompletionStreamStage::ToolIntentFillComplete => {
            record_tool_intent_fill_metrics(state, latency);
            tool_stream.record_tool_intent_fill(latency);
        }
        ChatCompletionStreamStage::ToolSchemaValidationComplete => {
            record_tool_schema_validation_metrics(state, latency);
            tool_stream.record_tool_schema_validation(latency);
        }
        _ => {}
    }
}

pub(super) fn engine_sse_keep_alive() -> KeepAlive {
    KeepAlive::new()
        .interval(Duration::from_millis(100))
        .text("llm-engine-heartbeat")
}

fn runtime_error_stream_events(err: RuntimeError) -> Vec<Result<Event, Infallible>> {
    let metadata = runtime_error_metadata(&err);
    let backend_failure_code = match &err {
        RuntimeError::BackendFailed { source } => {
            source.backend_failure_code().unwrap_or("unknown")
        }
        _ => "none",
    };
    tracing::warn!(
        error = %err,
        code = metadata.code,
        backend_failure_code,
        phase = metadata.phase,
        retryable = metadata.retryable,
        "streaming runtime error"
    );
    engine_error_stream_events(runtime_error_stream_body(&err, metadata))
}

fn runtime_error_stream_body(
    err: &RuntimeError,
    metadata: super::error::RuntimeErrorMetadata,
) -> EngineErrorBody {
    match err {
        RuntimeError::InvalidRequest { .. } => EngineErrorBody::from_runtime_error(err),
        _ => EngineErrorBody::new(
            "streaming response failed",
            metadata.code,
            metadata.phase,
            metadata.retryable,
        ),
    }
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
        tracing::error!(error = %err, "failed to serialize SSE event");
        SSE_SERIALIZATION_FALLBACK_DATA.to_owned()
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
    scheduler_slot: SharedSchedulerPermit,
    phase: GenerationPhaseGuard,
    request_started: Instant,
    prefill_chunk_started: Instant,
    cache_identity: Option<RequestCacheIdentity>,
    tool_stream: PendingToolStreamObservation,
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
        let tool_stream =
            PendingToolStreamObservation::new(request_id.clone(), model.clone(), true);
        Self {
            state,
            request_id,
            model,
            active_request,
            scheduler_slot,
            phase,
            request_started,
            prefill_chunk_started: Instant::now(),
            cache_identity: None,
            tool_stream,
            terminal: None,
        }
    }

    pub(super) fn set_cache_identity(&mut self, identity: RequestCacheIdentity) {
        self.cache_identity = Some(identity);
    }

    pub(super) fn cancellation(&self) -> tokio_util::sync::CancellationToken {
        self.active_request.cancellation.clone()
    }

    pub(super) fn prefill_chunk_admission(&self) -> BackendPrefillChunkAdmissionHook {
        BackendPrefillChunkAdmissionHook::new(Arc::new(SchedulerPrefillChunkAdmission {
            scheduler_slot: self.scheduler_slot.clone(),
            cancellation: self.active_request.cancellation.clone(),
        }))
    }

    fn stream_stall_timeout(&self) -> Option<Duration> {
        self.state.stream_stall_timeout
    }

    fn transition_to_decode(&mut self) {
        self.phase.transition_to_decode();
        self.scheduler_slot.transition_to_decode();
    }

    fn record_prefill_progress(&mut self, progress: &BackendStreamProgress) {
        if !matches!(progress, BackendStreamProgress::PrefillProgress { .. }) {
            return;
        }
        let now = Instant::now();
        self.scheduler_slot
            .record_prefill_chunk_latency(now.duration_since(self.prefill_chunk_started));
        self.prefill_chunk_started = now;
    }

    fn finish_success(
        &mut self,
        usage: &Usage,
        streamed: bool,
    ) -> Result<(), Vec<Result<Event, Infallible>>> {
        self.terminal = Some(StreamTerminalOutcome::Success);
        match self.active_request.mark_finished() {
            super::requests::RequestFinishResult::Finished => {
                if let Some(observation) = self
                    .tool_stream
                    .to_observation(self.request_started.elapsed())
                {
                    record_tool_stream_observation(&self.state, observation);
                }
                record_success_metrics(
                    &self.state,
                    &self.request_id,
                    &self.model,
                    usage,
                    streamed,
                    self.request_started.elapsed(),
                    self.cache_identity.as_ref(),
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
                if matches!(err, RuntimeError::Cancelled) {
                    self.scheduler_slot.mark_cancelled();
                } else {
                    self.scheduler_slot.mark_failed();
                }
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

#[derive(Debug)]
struct SchedulerPrefillChunkAdmission {
    scheduler_slot: SharedSchedulerPermit,
    cancellation: tokio_util::sync::CancellationToken,
}

#[async_trait]
impl BackendPrefillChunkAdmission for SchedulerPrefillChunkAdmission {
    async fn wait_for_next_chunk(
        &self,
        _progress: BackendStreamProgress,
    ) -> Result<(), BackendError> {
        self.scheduler_slot
            .yield_prefill_chunk(&self.cancellation)
            .await
            .map_err(prefill_readmission_backend_error)
    }
}

fn prefill_readmission_backend_error(err: SchedulerAcquireError) -> BackendError {
    match err {
        SchedulerAcquireError::Cancelled | SchedulerAcquireError::CancelledAfterAdmission => {
            BackendError::cancelled()
        }
        SchedulerAcquireError::QueueFull => BackendError::scheduler_overloaded(
            "model scheduler queue is full; retry the request later",
        ),
        SchedulerAcquireError::QueueTimedOut => BackendError::scheduler_overloaded(
            "model scheduler queue timed out; retry the request later",
        ),
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

    fn record_progress_metadata(&mut self, progress: StreamProgressMetadata) {
        if progress.has_real_delta() {
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
    Chunk {
        chunk: C,
        progress: StreamProgressMetadata,
    },
    Progress(BackendStreamProgress),
    InternalProgress {
        bytes: usize,
    },
    ToolStage(ChatCompletionStreamStage),
    Complete(Usage),
}

impl EngineStreamEvent for ChatCompletionStreamEvent {
    type Chunk = llm_api::ChatCompletionStreamResponse;

    fn into_step(
        self,
    ) -> EngineStreamStep<<ChatCompletionStreamEvent as EngineStreamEvent>::Chunk> {
        let progress = self.progress_metadata();
        match self {
            Self::Chunk(chunk) => EngineStreamStep::Chunk { chunk, progress },
            Self::Progress(progress) => EngineStreamStep::Progress(progress),
            Self::InternalProgress { bytes } => EngineStreamStep::InternalProgress { bytes },
            Self::Stage(stage) => EngineStreamStep::ToolStage(stage),
            Self::Complete(usage) => EngineStreamStep::Complete(usage),
            _ => EngineStreamStep::InternalProgress { bytes: 0 },
        }
    }
}

impl EngineStreamEvent for CompletionStreamEvent {
    type Chunk = llm_api::CompletionStreamResponse;

    fn into_step(self) -> EngineStreamStep<<CompletionStreamEvent as EngineStreamEvent>::Chunk> {
        let progress = self.progress_metadata();
        match self {
            Self::Chunk(chunk) => EngineStreamStep::Chunk { chunk, progress },
            Self::Progress(progress) => EngineStreamStep::Progress(progress),
            Self::Complete(usage) => EngineStreamStep::Complete(usage),
            _ => EngineStreamStep::InternalProgress { bytes: 0 },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::to_bytes,
        response::{IntoResponse, sse::Sse},
    };
    use serde::{Serialize, Serializer, ser::Error as _};
    use serde_json::Value;

    struct FailingSerialize;

    impl Serialize for FailingSerialize {
        fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            Err(S::Error::custom("forced serialization failure"))
        }
    }

    #[tokio::test]
    async fn sse_json_event_serialization_fallback_preserves_error_metadata() {
        let response =
            Sse::new(futures::stream::iter([sse_json_event(FailingSerialize)])).into_response();
        let bytes = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("SSE response body");
        let body = String::from_utf8(bytes.to_vec()).expect("SSE body is utf8");
        let data = body
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .expect("SSE data line");
        let value: Value = serde_json::from_str(data).expect("fallback event is JSON");

        assert_eq!(value["error"]["message"], "response serialization failed");
        assert_eq!(value["error"]["code"], "response_serialization_failed");
        assert_eq!(value["error"]["phase"], "response_serialization");
        assert_eq!(value["error"]["retryable"], true);
        assert_eq!(value["error"]["type"], "llm_engine_error");
    }
}
