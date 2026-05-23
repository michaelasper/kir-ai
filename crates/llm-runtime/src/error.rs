use llm_api::ApiError;
use llm_backend_contracts::{BackendError, BackendErrorDomain};
use llm_chat_template::TemplateError;
use llm_tool_parser::ParserError;
use thiserror::Error;

use crate::NoProgressClass;

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error(transparent)]
    Api(#[from] ApiError),
    #[error("model `{requested}` is not loaded; available model is `{available}`")]
    ModelUnavailable {
        requested: String,
        available: String,
    },
    #[error("invalid request: {reason}")]
    InvalidRequest { reason: String },
    #[error("backend generation cancelled")]
    Cancelled,
    #[error("backend error: {source}")]
    BackendFailed { source: BackendError },
    #[error(transparent)]
    Template(#[from] TemplateError),
    #[error(transparent)]
    Parser(#[from] ParserError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    JsonMode(String),
    #[error("tool call validation failed: {0}")]
    ToolCallValidation(String),
    #[error("no progress classified as {0:?}")]
    NoProgress(NoProgressClass),
}

impl RuntimeError {
    pub fn model_unavailable(requested: impl Into<String>, available: impl Into<String>) -> Self {
        Self::ModelUnavailable {
            requested: requested.into(),
            available: available.into(),
        }
    }

    pub fn invalid_request(reason: impl Into<String>) -> Self {
        Self::InvalidRequest {
            reason: reason.into(),
        }
    }

    pub fn backend_failed(message: impl Into<String>) -> Self {
        Self::BackendFailed {
            source: BackendError::other(message),
        }
    }
}

impl From<BackendError> for RuntimeError {
    fn from(value: BackendError) -> Self {
        match value.into_domain() {
            BackendErrorDomain::ModelNotFound {
                requested,
                available,
            } => Self::ModelUnavailable {
                requested,
                available,
            },
            BackendErrorDomain::InvalidRequest { reason } => Self::InvalidRequest { reason },
            BackendErrorDomain::Cancelled => Self::Cancelled,
            BackendErrorDomain::BackendFailure(source) => Self::BackendFailed { source },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_model_not_found_maps_to_model_unavailable() {
        let err = RuntimeError::from(BackendError::model_not_found("requested", "available"));

        match err {
            RuntimeError::ModelUnavailable {
                requested,
                available,
            } => {
                assert_eq!(requested, "requested");
                assert_eq!(available, "available");
            }
            other => panic!("expected model unavailable, got {other:?}"),
        }
    }

    #[test]
    fn backend_request_errors_map_to_invalid_request() {
        let unsupported = RuntimeError::from(BackendError::unsupported_request("unsupported"));
        let invalid_sampling =
            RuntimeError::from(BackendError::invalid_sampling_config("invalid sampling"));

        assert!(matches!(
            unsupported,
            RuntimeError::InvalidRequest { reason } if reason == "unsupported"
        ));
        assert!(matches!(
            invalid_sampling,
            RuntimeError::InvalidRequest { reason } if reason == "invalid sampling"
        ));
    }

    #[test]
    fn backend_cancelled_maps_to_cancelled() {
        assert!(matches!(
            RuntimeError::from(BackendError::cancelled()),
            RuntimeError::Cancelled
        ));
    }

    #[test]
    fn backend_other_maps_to_backend_failed() {
        let err = RuntimeError::from(BackendError::other("backend failed"));

        assert!(matches!(
            err,
            RuntimeError::BackendFailed { source } if source.other_message() == Some("backend failed")
        ));
    }

    #[test]
    fn structured_backend_failure_context_survives_runtime_mapping() {
        let err = RuntimeError::from(BackendError::backend_failure(
            "model_integrity_failed",
            "bad tensor header",
        ));

        assert!(matches!(
            err,
            RuntimeError::BackendFailed { source }
                if source.backend_failure_code() == Some("model_integrity_failed")
        ));
    }
}
