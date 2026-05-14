use super::{
    AppState, EngineError, lifecycle,
    metrics::{record_failure_metrics, record_success_metrics},
    parse_json_request,
};
use axum::{
    Json,
    extract::{State, rejection::JsonRejection},
    http::HeaderMap,
    response::{IntoResponse, Response, sse::Sse},
};
use futures::StreamExt;
use llm_api::{ChatCompletionRequest, CompletionRequest, ValidateRequest};
use llm_runtime::RuntimeError;

pub(super) async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    request: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Result<Response, EngineError> {
    let request = parse_json_request(request, &state)?;
    validate_api_request(&request, &state)?;
    let streamed = request.stream;
    if request.stream {
        let model = request.model.clone();
        let run = lifecycle::start_chat_generation(&state, &headers, &request).await?;
        let request_id = run.request_id().to_owned();
        let stream_run = run.into_streaming();
        let stream_state = state.clone();
        let events = async_stream::stream! {
            let mut stream_lifecycle =
                super::streaming::StreamRunLifecycle::new(stream_state.clone(), stream_run, model);
            match stream_state
                .runtime
                .chat_stream_with_cancel(request, stream_lifecycle.cancellation())
                .await
            {
                Ok(response) => {
                    let events = super::streaming::stream_runtime_events(
                        stream_lifecycle,
                        response.into_events(),
                        streamed,
                    );
                    tokio::pin!(events);
                    while let Some(event) = events.next().await {
                        yield event;
                    }
                }
                Err(err) => {
                    for event in stream_lifecycle.finish_runtime_error(err) {
                        yield event;
                    }
                }
            }
        };
        let mut response = Sse::new(events)
            .keep_alive(super::streaming::engine_sse_keep_alive())
            .into_response();
        lifecycle::insert_request_id_header(&mut response, &request_id);
        return Ok(response);
    }
    let run = lifecycle::start_chat_generation(&state, &headers, &request).await?;
    let response = match state
        .runtime
        .chat_with_cancel(request, run.cancellation())
        .await
    {
        Ok(response) => response,
        Err(err) => return Err(run.finish_runtime_error(&state, err)),
    };
    let finished = run.finish_success(&state)?;
    let model = response.model.clone();
    record_success_metrics(
        &state,
        finished.request_id(),
        &model,
        &response.usage,
        streamed,
        finished.elapsed(),
    );
    let mut response = Json(response).into_response();
    lifecycle::insert_request_id_header(&mut response, finished.request_id());
    Ok(response)
}

pub(super) async fn completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    request: Result<Json<CompletionRequest>, JsonRejection>,
) -> Result<Response, EngineError> {
    let request = parse_json_request(request, &state)?;
    validate_api_request(&request, &state)?;
    let streamed = request.stream;
    if request.stream {
        let model = request.model.clone();
        let run = lifecycle::start_completion_generation(&state, &headers, &request).await?;
        let request_id = run.request_id().to_owned();
        let stream_run = run.into_streaming();
        let stream_state = state.clone();
        let events = async_stream::stream! {
            let mut stream_lifecycle =
                super::streaming::StreamRunLifecycle::new(stream_state.clone(), stream_run, model);
            match stream_state
                .runtime
                .completion_stream_with_cancel(request, stream_lifecycle.cancellation())
                .await
            {
                Ok(response) => {
                    let events = super::streaming::stream_runtime_events(
                        stream_lifecycle,
                        response.into_events(),
                        streamed,
                    );
                    tokio::pin!(events);
                    while let Some(event) = events.next().await {
                        yield event;
                    }
                }
                Err(err) => {
                    for event in stream_lifecycle.finish_runtime_error(err) {
                        yield event;
                    }
                }
            }
        };
        let mut response = Sse::new(events)
            .keep_alive(super::streaming::engine_sse_keep_alive())
            .into_response();
        lifecycle::insert_request_id_header(&mut response, &request_id);
        return Ok(response);
    }
    let run = lifecycle::start_completion_generation(&state, &headers, &request).await?;
    let response = match state
        .runtime
        .completion_with_cancel(request, run.cancellation())
        .await
    {
        Ok(response) => response,
        Err(err) => return Err(run.finish_runtime_error(&state, err)),
    };
    let finished = run.finish_success(&state)?;
    let model = response.model.clone();
    record_success_metrics(
        &state,
        finished.request_id(),
        &model,
        &response.usage,
        streamed,
        finished.elapsed(),
    );
    let mut response = Json(response).into_response();
    lifecycle::insert_request_id_header(&mut response, finished.request_id());
    Ok(response)
}

fn validate_api_request<T: ValidateRequest>(
    request: &T,
    state: &AppState,
) -> Result<(), EngineError> {
    request.validate().map_err(|err| {
        record_failure_metrics(state);
        RuntimeError::Api(err).into()
    })
}
