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
        let message = sanitize_client_error_message(message.into());
        Self {
            error: EngineErrorPayload {
                message,
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

const REDACTED_PATH: &str = "[redacted path]";

fn sanitize_client_error_message(message: String) -> String {
    let mut sanitized = String::with_capacity(message.len());
    let mut cursor = 0;
    let mut redacted = false;

    while cursor < message.len() {
        let Some((relative_start, path_len)) = next_absolute_path(&message[cursor..]) else {
            sanitized.push_str(&message[cursor..]);
            break;
        };
        let start = cursor + relative_start;
        let end = start + path_len;
        sanitized.push_str(&message[cursor..start]);
        sanitized.push_str(REDACTED_PATH);
        cursor = end;
        redacted = true;
    }

    if redacted { sanitized } else { message }
}

fn next_absolute_path(message: &str) -> Option<(usize, usize)> {
    let mut previous = None;
    for (index, ch) in message.char_indices() {
        if let Some(len) = absolute_path_len(&message[index..], previous) {
            return Some((index, len));
        }
        previous = Some(ch);
    }
    None
}

fn absolute_path_len(candidate: &str, previous: Option<char>) -> Option<usize> {
    if starts_with_unix_absolute_path(candidate, previous)
        || starts_with_windows_absolute_path(candidate)
    {
        Some(path_token_len(candidate))
    } else {
        None
    }
}

fn starts_with_unix_absolute_path(candidate: &str, previous: Option<char>) -> bool {
    if !has_path_token_boundary(previous) {
        return false;
    }

    let mut chars = candidate.chars();
    matches!(chars.next(), Some('/'))
        && matches!(chars.next(), Some(ch) if is_unix_path_segment_char(ch))
}

fn has_path_token_boundary(previous: Option<char>) -> bool {
    previous.is_none_or(|ch| {
        ch.is_whitespace() || matches!(ch, '`' | '"' | '\'' | '<' | '(' | '[' | '{' | '=' | ':')
    })
}

fn is_unix_path_segment_char(ch: char) -> bool {
    !is_path_terminator(ch) && ch != '/' && ch != '\\'
}

fn starts_with_windows_absolute_path(candidate: &str) -> bool {
    let bytes = candidate.as_bytes();
    matches!(
        bytes,
        [drive, b':', slash, ..]
            if drive.is_ascii_alphabetic() && (*slash == b'\\' || *slash == b'/')
    ) || candidate.starts_with("\\\\")
}

fn path_token_len(candidate: &str) -> usize {
    candidate
        .char_indices()
        .find_map(|(index, ch)| is_path_terminator(ch).then_some(index))
        .unwrap_or(candidate.len())
}

fn is_path_terminator(ch: char) -> bool {
    ch.is_whitespace()
        || matches!(
            ch,
            '`' | '"' | '\'' | '<' | '>' | '(' | ')' | '[' | ']' | '{' | '}' | ',' | ';'
        )
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
    RateLimited,
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
                log_runtime_error_response(&err, metadata);
                (metadata.status, EngineErrorBody::from_runtime_error(&err))
            }
            Self::ModelStore(err) => {
                let status = if err.code() == "model_not_found" {
                    StatusCode::NOT_FOUND
                } else {
                    StatusCode::UNPROCESSABLE_ENTITY
                };
                tracing::warn!(
                    error = %err,
                    code = err.code(),
                    phase = "model_artifact_verification",
                    retryable = false,
                    status = status.as_u16(),
                    "model-store error response"
                );
                (
                    status,
                    EngineErrorBody::new(
                        err.to_string(),
                        err.code(),
                        "model_artifact_verification",
                        false,
                    ),
                )
            }
            Self::Overloaded(message) => (
                StatusCode::TOO_MANY_REQUESTS,
                EngineErrorBody::new(message, "model_overloaded", "scheduler", true),
            ),
            Self::RateLimited => (
                StatusCode::TOO_MANY_REQUESTS,
                EngineErrorBody::new(
                    "public inference rate limit exceeded; retry later",
                    "rate_limited",
                    "rate_limit",
                    true,
                ),
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

fn log_runtime_error_response(err: &RuntimeError, metadata: RuntimeErrorMetadata) {
    let RuntimeError::BackendFailed { source } = err else {
        return;
    };
    tracing::warn!(
        error = %err,
        code = metadata.code,
        backend_failure_code = source.backend_failure_code().unwrap_or("unknown"),
        phase = metadata.phase,
        retryable = metadata.retryable,
        status = metadata.status.as_u16(),
        "runtime error response"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_backend::{BackendError, TensorLoadError};
    use serde_json::json;

    const DOCUMENTED_ENGINE_ERROR_CODES: &[&str] = &[
        "invalid_request",
        "unsupported_capability",
        "model_not_found",
        "rate_limited",
        "model_overloaded",
        "backend_execution_failed",
        "cancelled",
        "request_not_found",
        "request_id_conflict",
        "admin_auth_required",
        "chat_template_failed",
        "malformed_tool_call",
        "unsupported_multimodal_output",
        "json_validation_failed",
        "tool_call_validation_failed",
        "no_progress_empty_completion",
        "no_progress_empty_high_output_completion",
        "no_progress_hidden_only_output",
        "no_progress_missing_required_tool_call",
        "no_progress_repeated_invalid_tool_call",
        "no_progress_fuzzy_repeated_invalid_tool_call",
        "no_progress_repeated_assistant_content",
        "no_progress_stalled_assistant_turn",
        "stream_stalled",
        "stream_incomplete",
        "response_serialization_failed",
    ];

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

    #[test]
    fn structured_backend_failure_context_survives_server_error_boundary() {
        let err = RuntimeError::from(BackendError::from(TensorLoadError::integrity(
            "bad tensor header",
        )));

        let RuntimeError::BackendFailed { source } = &err else {
            panic!("expected backend failure, got {err:?}");
        };
        assert_eq!(
            source.backend_failure_code(),
            Some("model_integrity_failed")
        );

        let metadata = runtime_error_metadata(&err);
        assert_eq!(metadata.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(metadata.code, "backend_execution_failed");
        assert_eq!(metadata.phase, "decode");
        assert!(metadata.retryable);

        let body = EngineErrorBody::from_runtime_error(&err);
        let value = serde_json::to_value(&body).expect("engine error body serializes");

        assert_eq!(value["error"]["code"], "backend_execution_failed");
        assert_eq!(value["error"]["phase"], "decode");
        assert_eq!(value["error"]["retryable"], true);
        assert!(
            value["error"]["message"]
                .as_str()
                .expect("error message is string")
                .contains("model_integrity_failed: bad tensor header"),
            "server body should retain source failure message while keeping stable metadata: {value}"
        );
    }

    #[test]
    fn engine_error_body_redacts_absolute_unix_paths() {
        let body = EngineErrorBody::new(
            "failed to open `/Users/michaelasper/source/kir-ai/private/model.safetensors`",
            "backend_execution_failed",
            "decode",
            true,
        );

        let value = serde_json::to_value(&body).expect("engine error body serializes");
        let message = value["error"]["message"]
            .as_str()
            .expect("error message is string");

        assert_eq!(message, "failed to open `[redacted path]`");
    }

    #[test]
    fn engine_error_body_redacts_non_allowlisted_unix_paths() {
        let cases = [
            "/data/kir-ai/private/model.safetensors",
            "/models/qwen/private/model.safetensors",
            "/cache/kir-ai/private/model.bin",
            "/nix/store/abc123-kir-ai-model/model.safetensors",
        ];

        for leaked_path in cases {
            let body = EngineErrorBody::new(
                format!("failed to open `{leaked_path}`"),
                "backend_execution_failed",
                "decode",
                true,
            );

            let value = serde_json::to_value(&body).expect("engine error body serializes");
            let message = value["error"]["message"]
                .as_str()
                .expect("error message is string");

            assert_eq!(message, "failed to open `[redacted path]`");
        }
    }

    #[test]
    fn engine_error_body_redacts_windows_absolute_paths() {
        let body = EngineErrorBody::new(
            "failed to read C:\\Users\\michaelasper\\kir-ai\\model.safetensors",
            "backend_execution_failed",
            "decode",
            true,
        );

        let value = serde_json::to_value(&body).expect("engine error body serializes");
        let message = value["error"]["message"]
            .as_str()
            .expect("error message is string");

        assert_eq!(message, "failed to read [redacted path]");
    }

    #[test]
    fn http_api_reference_documents_current_engine_error_codes() {
        let docs = include_str!("../../../../docs/http-api-reference.md");

        for code in DOCUMENTED_ENGINE_ERROR_CODES {
            let row_prefix = format!("| `{code}` |");
            assert!(
                docs.contains(&row_prefix),
                "docs/http-api-reference.md is missing known error code `{code}`"
            );
        }
    }
}
