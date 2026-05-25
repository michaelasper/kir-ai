use llm_api::ApiError;
use llm_backend_contracts::{BackendError, BackendErrorDomain};
use llm_chat_template::TemplateError;
use llm_tool_parser::ParserError;
use thiserror::Error;

use crate::NoProgressClass;

/// Error surfaced by runtime request processing.
///
/// Variants preserve the lifecycle phase that failed: API validation,
/// model/backend availability, prompt templating, assistant parsing, response
/// validation, or no-progress classification. The server layer maps these into
/// stable OpenAI-compatible error bodies.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RuntimeError {
    /// Request failed public API validation.
    #[error(transparent)]
    Api(#[from] ApiError),
    /// The requested model differs from the backend loaded in this runtime.
    #[error("model `{requested}` is not loaded; available model is `{available}`")]
    ModelUnavailable {
        /// Model requested by the client.
        requested: String,
        /// Model currently loaded by the backend.
        available: String,
    },
    /// Runtime or backend rejected a semantically invalid request.
    #[error("invalid request: {reason}")]
    InvalidRequest {
        /// Stable human-readable reason.
        reason: String,
    },
    /// Backend observed cancellation before or during generation.
    #[error("backend generation cancelled")]
    Cancelled,
    /// Backend failed after accepting the request.
    #[error("backend error: {source}")]
    BackendFailed {
        /// Backend error with structured failure classification.
        source: BackendError,
    },
    /// Prompt template rendering failed.
    #[error(transparent)]
    Template(#[from] TemplateError),
    /// Assistant tool/parser output was malformed.
    #[error(transparent)]
    Parser(#[from] ParserError),
    /// JSON serialization or validation failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// JSON-object response mode produced invalid assistant content.
    #[error("{0}")]
    JsonMode(String),
    /// Tool call validation failed after assistant parsing.
    #[error("tool call validation failed: {0}")]
    ToolCallValidation(String),
    /// Runtime classified the generation as making no useful progress.
    #[error("no progress classified as {0:?}")]
    NoProgress(NoProgressClass),
}

impl RuntimeError {
    /// Builds a model-unavailable error while preserving requested and available IDs.
    pub fn model_unavailable(requested: impl Into<String>, available: impl Into<String>) -> Self {
        Self::ModelUnavailable {
            requested: requested.into(),
            available: available.into(),
        }
    }

    /// Builds an invalid request error for runtime-level checks.
    pub fn invalid_request(reason: impl Into<String>) -> Self {
        Self::InvalidRequest {
            reason: reason.into(),
        }
    }

    /// Builds a generic backend failure from a message.
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
            _ => Self::BackendFailed {
                source: BackendError::other("backend error domain is not supported by runtime"),
            },
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
