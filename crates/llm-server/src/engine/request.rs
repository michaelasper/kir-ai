use super::{AppState, EngineError, metrics};
use axum::{Json, extract::rejection::JsonRejection};
use llm_api::ApiError;
use llm_runtime::RuntimeError;

pub(super) fn parse_json_request<T>(
    request: Result<Json<T>, JsonRejection>,
    state: &AppState,
) -> Result<T, EngineError> {
    match request {
        Ok(Json(request)) => Ok(request),
        Err(err) => {
            metrics::record_failure_metrics(state);
            if let Some(timeout) = super::request_body_timeout::request_body_timeout(&err) {
                return Err(EngineError::RequestBodyTimeout { timeout });
            }
            Err(RuntimeError::Api(ApiError::invalid_request(format!(
                "invalid JSON request body: {err}"
            )))
            .into())
        }
    }
}
