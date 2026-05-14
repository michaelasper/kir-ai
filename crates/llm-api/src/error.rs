use schemars::JsonSchema;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error, JsonSchema)]
#[error("{code}: {message}")]
pub struct ApiError {
    code: &'static str,
    message: String,
}

impl ApiError {
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            code: "invalid_request",
            message: message.into(),
        }
    }

    pub fn unsupported_capability(message: impl Into<String>) -> Self {
        Self {
            code: "unsupported_capability",
            message: message.into(),
        }
    }

    pub fn code(&self) -> &'static str {
        self.code
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}
