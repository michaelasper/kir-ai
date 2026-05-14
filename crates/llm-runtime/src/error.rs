use llm_api::ApiError;
use llm_backend::BackendError;
use llm_tokenizer::TemplateError;
use llm_tool_parser::ParserError;
use thiserror::Error;

use crate::NoProgressClass;

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error(transparent)]
    Api(#[from] ApiError),
    #[error("model `{requested}` is not loaded; available model is `{available}`")]
    ModelNotFound {
        requested: String,
        available: String,
    },
    #[error("unsupported backend request: {0}")]
    UnsupportedCapability(String),
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("backend generation cancelled")]
    Cancelled,
    #[error("backend error: {0}")]
    BackendExecution(String),
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

impl From<BackendError> for RuntimeError {
    fn from(value: BackendError) -> Self {
        match value {
            BackendError::ModelNotFound {
                requested,
                available,
            } => Self::ModelNotFound {
                requested,
                available,
            },
            BackendError::UnsupportedRequest(message) => Self::UnsupportedCapability(message),
            BackendError::InvalidSamplingConfig(message) => Self::InvalidRequest(message),
            BackendError::Cancelled => Self::Cancelled,
            BackendError::Other(message) => Self::BackendExecution(message),
        }
    }
}
