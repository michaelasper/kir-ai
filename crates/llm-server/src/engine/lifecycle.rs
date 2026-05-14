use super::{
    AppState, EngineError,
    metrics::{record_failure_metrics, record_runtime_error_metrics},
    requests::{ActiveRequest, RequestFinishResult, RequestRegistrationError, RequestStartResult},
    scheduler::{
        GenerationPhase, GenerationPhaseGuard, SchedulerAcquireError, SchedulerClass,
        SchedulerPermit,
    },
};
use axum::{
    http::{HeaderMap, HeaderValue},
    response::Response,
};
use llm_api::{ChatCompletionRequest, CompletionRequest};
use llm_runtime::RuntimeError;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

#[derive(Debug)]
pub(super) struct GenerationRun {
    request_id: String,
    phase: GenerationPhaseGuard,
    scheduler_slot: SchedulerPermit,
    active_request: ActiveRequest,
    request_started: Instant,
}

impl GenerationRun {
    pub(super) fn request_id(&self) -> &str {
        &self.request_id
    }

    pub(super) fn cancellation(&self) -> CancellationToken {
        self.active_request.cancellation.clone()
    }

    pub(super) fn finish_success(
        self,
        state: &AppState,
    ) -> Result<FinishedGenerationRun, EngineError> {
        let Self {
            request_id,
            phase,
            mut scheduler_slot,
            active_request,
            request_started,
        } = self;
        mark_active_request_finished_for_success(state, &active_request, &mut scheduler_slot)?;
        drop(active_request);
        Ok(FinishedGenerationRun {
            request_id,
            request_started,
            _scheduler_slot: scheduler_slot,
            _phase: phase,
        })
    }

    pub(super) fn finish_runtime_error(self, state: &AppState, err: RuntimeError) -> EngineError {
        let Self {
            request_id: _request_id,
            phase,
            mut scheduler_slot,
            active_request,
            request_started: _request_started,
        } = self;
        let err = mark_active_request_finished_for_runtime_error(
            state,
            &active_request,
            &mut scheduler_slot,
            err,
        );
        drop(phase);
        drop(scheduler_slot);
        drop(active_request);
        err
    }

    pub(super) fn into_streaming(self) -> StreamingGenerationRun {
        let Self {
            request_id,
            phase,
            scheduler_slot,
            active_request,
            request_started,
        } = self;
        StreamingGenerationRun {
            request_id,
            phase,
            scheduler_slot,
            active_request,
            request_started,
        }
    }
}

#[derive(Debug)]
pub(super) struct FinishedGenerationRun {
    request_id: String,
    request_started: Instant,
    _phase: GenerationPhaseGuard,
    _scheduler_slot: SchedulerPermit,
}

impl FinishedGenerationRun {
    pub(super) fn request_id(&self) -> &str {
        &self.request_id
    }

    pub(super) fn elapsed(&self) -> Duration {
        self.request_started.elapsed()
    }
}

#[derive(Debug)]
pub(super) struct StreamingGenerationRun {
    pub(super) request_id: String,
    pub(super) phase: GenerationPhaseGuard,
    pub(super) scheduler_slot: SchedulerPermit,
    pub(super) active_request: ActiveRequest,
    pub(super) request_started: Instant,
}

pub(super) async fn start_chat_generation(
    state: &AppState,
    headers: &HeaderMap,
    request: &ChatCompletionRequest,
) -> Result<GenerationRun, EngineError> {
    let (admission_class, initial_phase) = chat_scheduler_classes(state, request);
    start_generation(state, headers, admission_class, initial_phase).await
}

pub(super) async fn start_completion_generation(
    state: &AppState,
    headers: &HeaderMap,
    request: &CompletionRequest,
) -> Result<GenerationRun, EngineError> {
    let (admission_class, initial_phase) = completion_scheduler_classes(state, request);
    start_generation(state, headers, admission_class, initial_phase).await
}

async fn start_generation(
    state: &AppState,
    headers: &HeaderMap,
    admission_class: SchedulerClass,
    initial_phase: GenerationPhase,
) -> Result<GenerationRun, EngineError> {
    let active_request = register_active_request(state, headers)?;
    let request_started = active_request.started_at;
    let mut scheduler_slot = acquire_scheduler_slot(
        state,
        admission_class,
        initial_phase,
        &active_request.cancellation,
    )
    .await?;
    mark_active_request_running(state, &active_request, &mut scheduler_slot)?;
    let phase = state.generation_phases.begin(initial_phase);
    let request_id = active_request.id.clone();
    Ok(GenerationRun {
        request_id,
        phase,
        scheduler_slot,
        active_request,
        request_started,
    })
}

async fn acquire_scheduler_slot(
    state: &AppState,
    admission_class: SchedulerClass,
    initial_phase: GenerationPhase,
    cancellation: &CancellationToken,
) -> Result<SchedulerPermit, EngineError> {
    match state
        .model_scheduler
        .clone()
        .acquire(admission_class, initial_phase, cancellation)
        .await
    {
        Ok(permit) => Ok(permit),
        Err(SchedulerAcquireError::QueueFull) => {
            record_failure_metrics(state);
            Err(EngineError::Overloaded(
                "model scheduler queue is full; retry the request later".to_owned(),
            ))
        }
        Err(SchedulerAcquireError::QueueTimedOut) => {
            record_failure_metrics(state);
            Err(EngineError::Overloaded(
                "model scheduler queue timed out; retry the request later".to_owned(),
            ))
        }
        Err(SchedulerAcquireError::Cancelled) => {
            record_failure_metrics(state);
            Err(EngineError::RequestCancelled {
                phase: "scheduler",
                message: "request was cancelled before scheduler admission",
            })
        }
    }
}

fn mark_active_request_running(
    state: &AppState,
    active_request: &ActiveRequest,
    scheduler_slot: &mut SchedulerPermit,
) -> Result<(), EngineError> {
    match active_request.mark_running() {
        RequestStartResult::Running => Ok(()),
        RequestStartResult::Cancelled => {
            scheduler_slot.mark_cancelled();
            record_failure_metrics(state);
            Err(EngineError::RequestCancelled {
                phase: "scheduler",
                message: "request was cancelled before runtime execution",
            })
        }
        RequestStartResult::Finished | RequestStartResult::Missing => {
            scheduler_slot.mark_failed();
            record_failure_metrics(state);
            Err(RuntimeError::BackendExecution(
                "request lifecycle was not runnable after scheduler admission".to_owned(),
            )
            .into())
        }
    }
}

fn mark_active_request_finished_for_success(
    state: &AppState,
    active_request: &ActiveRequest,
    scheduler_slot: &mut SchedulerPermit,
) -> Result<(), EngineError> {
    match active_request.mark_finished() {
        RequestFinishResult::Finished => Ok(()),
        RequestFinishResult::Cancelled => {
            scheduler_slot.mark_cancelled();
            record_failure_metrics(state);
            Err(EngineError::RequestCancelled {
                phase: "decode",
                message: "request was cancelled before response delivery",
            })
        }
        RequestFinishResult::Missing => {
            scheduler_slot.mark_failed();
            record_failure_metrics(state);
            Err(RuntimeError::BackendExecution(
                "request lifecycle was missing before response delivery".to_owned(),
            )
            .into())
        }
    }
}

fn mark_active_request_finished_for_runtime_error(
    state: &AppState,
    active_request: &ActiveRequest,
    scheduler_slot: &mut SchedulerPermit,
    err: RuntimeError,
) -> EngineError {
    match active_request.mark_finished() {
        RequestFinishResult::Finished => {
            mark_scheduler_runtime_error(scheduler_slot, &err);
            record_runtime_error_metrics(state, &err);
            err.into()
        }
        RequestFinishResult::Cancelled => {
            scheduler_slot.mark_cancelled();
            record_failure_metrics(state);
            EngineError::RequestCancelled {
                phase: "decode",
                message: "request was cancelled before error delivery",
            }
        }
        RequestFinishResult::Missing => {
            scheduler_slot.mark_failed();
            record_failure_metrics(state);
            RuntimeError::BackendExecution(
                "request lifecycle was missing before error delivery".to_owned(),
            )
            .into()
        }
    }
}

fn chat_scheduler_classes(
    state: &AppState,
    request: &ChatCompletionRequest,
) -> (SchedulerClass, GenerationPhase) {
    let admission = state.model_scheduler.classify_chat(request);
    let initial_phase = if request.stream || admission == SchedulerClass::Prefill {
        GenerationPhase::Prefill
    } else {
        admission.as_phase()
    };
    (admission, initial_phase)
}

fn completion_scheduler_classes(
    state: &AppState,
    request: &CompletionRequest,
) -> (SchedulerClass, GenerationPhase) {
    let admission = state.model_scheduler.classify_completion(request);
    let initial_phase = if request.stream || admission == SchedulerClass::Prefill {
        GenerationPhase::Prefill
    } else {
        admission.as_phase()
    };
    (admission, initial_phase)
}

pub(super) fn mark_scheduler_runtime_error(permit: &mut SchedulerPermit, err: &RuntimeError) {
    if matches!(err, RuntimeError::Cancelled) {
        permit.mark_cancelled();
    } else {
        permit.mark_failed();
    }
}

fn register_active_request(
    state: &AppState,
    headers: &HeaderMap,
) -> Result<ActiveRequest, EngineError> {
    let id = request_id_from_headers(state, headers).inspect_err(|_| {
        record_failure_metrics(state);
    })?;
    state.active_requests.register(id).map_err(|err| {
        record_failure_metrics(state);
        match err {
            RequestRegistrationError::Conflict(id) => EngineError::RequestConflict(id),
        }
    })
}

pub(super) fn request_id_from_headers(
    state: &AppState,
    headers: &HeaderMap,
) -> Result<String, EngineError> {
    let Some(value) = headers
        .get("x-request-id")
        .or_else(|| headers.get("x-llm-request-id"))
    else {
        return Ok(state.active_requests.next_request_id());
    };
    let request_id = value
        .to_str()
        .map_err(|_| EngineError::InvalidRequestId("request id must be visible ASCII".to_owned()))?
        .trim();
    if request_id.is_empty() {
        return Err(EngineError::InvalidRequestId(
            "request id must not be empty".to_owned(),
        ));
    }
    if request_id.len() > 128 {
        return Err(EngineError::InvalidRequestId(
            "request id must be at most 128 bytes".to_owned(),
        ));
    }
    Ok(request_id.to_owned())
}

pub(super) fn response_request_id(state: &AppState, headers: &HeaderMap) -> String {
    request_id_from_headers(state, headers)
        .unwrap_or_else(|_| state.active_requests.next_request_id())
}

pub(super) fn insert_request_id_header(response: &mut Response, request_id: &str) {
    let value =
        HeaderValue::from_str(request_id).unwrap_or_else(|_| HeaderValue::from_static("unknown"));
    response.headers_mut().insert("x-request-id", value);
}
