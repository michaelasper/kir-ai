use axum::{Json, http::StatusCode, response::IntoResponse};
use llm_hub::HubError;
use llm_runtime::RuntimeError;
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct EngineErrorBody {
    error: EngineErrorPayload,
}

impl EngineErrorBody {
    pub(super) fn new(
        message: impl Into<String>,
        code: &'static str,
        phase: &'static str,
        retryable: bool,
    ) -> Self {
        Self {
            error: EngineErrorPayload {
                message: message.into(),
                code,
                phase,
                retryable,
                error_type: "llm_engine_error",
            },
        }
    }

    pub(super) fn from_runtime_error(err: &RuntimeError) -> Self {
        let metadata = runtime_error_metadata(err);
        Self::new(
            err.to_string(),
            metadata.code,
            metadata.phase,
            metadata.retryable,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct EngineErrorPayload {
    message: String,
    code: &'static str,
    phase: &'static str,
    retryable: bool,
    #[serde(rename = "type")]
    error_type: &'static str,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct RuntimeErrorMetadata {
    pub(super) status: StatusCode,
    pub(super) code: &'static str,
    pub(super) phase: &'static str,
    pub(super) retryable: bool,
}

pub(super) fn runtime_error_metadata(err: &RuntimeError) -> RuntimeErrorMetadata {
    let (status, code, phase, retryable) = match err {
        RuntimeError::Api(api) => (
            StatusCode::BAD_REQUEST,
            api.code(),
            "request_validation",
            false,
        ),
        RuntimeError::ModelUnavailable { .. } => (
            StatusCode::NOT_FOUND,
            "model_not_found",
            "model_resolution",
            false,
        ),
        RuntimeError::Cancelled => (StatusCode::REQUEST_TIMEOUT, "cancelled", "decode", false),
        RuntimeError::InvalidRequest { .. } => (
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "request_validation",
            false,
        ),
        RuntimeError::BackendFailed { .. } => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "backend_execution_failed",
            "decode",
            true,
        ),
        RuntimeError::Template(_) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            "chat_template_failed",
            "prompt_rendering",
            false,
        ),
        RuntimeError::Parser(err) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            err.code(),
            "response_parsing",
            false,
        ),
        RuntimeError::Json(_) | RuntimeError::JsonMode(_) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            "json_validation_failed",
            "response_validation",
            false,
        ),
        RuntimeError::ToolCallValidation(_) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            "tool_call_validation_failed",
            "response_validation",
            false,
        ),
        RuntimeError::NoProgress(class) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            class.code(),
            "response_validation",
            false,
        ),
    };
    RuntimeErrorMetadata {
        status,
        code,
        phase,
        retryable,
    }
}

#[derive(Debug)]
pub(super) enum EngineError {
    Runtime(RuntimeError),
    ModelStore(HubError),
    Overloaded(String),
    RequestCancelled {
        phase: &'static str,
        message: &'static str,
    },
    RequestNotFound(String),
    RequestConflict(String),
    InvalidRequestId(String),
    UnauthorizedAdmin,
}

impl From<RuntimeError> for EngineError {
    fn from(value: RuntimeError) -> Self {
        Self::Runtime(value)
    }
}

impl IntoResponse for EngineError {
    fn into_response(self) -> axum::response::Response {
        let (status, body) = match self {
            Self::Runtime(err) => {
                let metadata = runtime_error_metadata(&err);
                (metadata.status, EngineErrorBody::from_runtime_error(&err))
            }
            Self::ModelStore(err) => (
                if err.code() == "model_not_found" {
                    StatusCode::NOT_FOUND
                } else {
                    StatusCode::UNPROCESSABLE_ENTITY
                },
                EngineErrorBody::new(
                    err.to_string(),
                    err.code(),
                    "model_artifact_verification",
                    false,
                ),
            ),
            Self::Overloaded(message) => (
                StatusCode::TOO_MANY_REQUESTS,
                EngineErrorBody::new(message, "model_overloaded", "scheduler", true),
            ),
            Self::RequestCancelled { phase, message } => (
                StatusCode::REQUEST_TIMEOUT,
                EngineErrorBody::new(message, "cancelled", phase, false),
            ),
            Self::RequestNotFound(request_id) => (
                StatusCode::NOT_FOUND,
                EngineErrorBody::new(
                    format!("request `{request_id}` is not active"),
                    "request_not_found",
                    "cancellation",
                    false,
                ),
            ),
            Self::RequestConflict(request_id) => (
                StatusCode::CONFLICT,
                EngineErrorBody::new(
                    format!("request id `{request_id}` is already active"),
                    "request_id_conflict",
                    "request_validation",
                    false,
                ),
            ),
            Self::InvalidRequestId(message) => (
                StatusCode::BAD_REQUEST,
                EngineErrorBody::new(message, "invalid_request", "request_validation", false),
            ),
            Self::UnauthorizedAdmin => (
                StatusCode::UNAUTHORIZED,
                EngineErrorBody::new(
                    "admin bearer token is required",
                    "admin_auth_required",
                    "admin_auth",
                    false,
                ),
            ),
        };
        (status, Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn engine_error_body_serializes_stable_shape() {
        let body = EngineErrorBody::new("stable message", "stable_code", "stable_phase", true);

        let value = serde_json::to_value(&body).expect("engine error body serializes");

        assert_eq!(
            value,
            json!({
                "error": {
                    "message": "stable message",
                    "code": "stable_code",
                    "phase": "stable_phase",
                    "retryable": true,
                    "type": "llm_engine_error"
                }
            })
        );
    }
}
