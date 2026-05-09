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
    #[error(transparent)]
    Backend(#[from] BackendError),
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
