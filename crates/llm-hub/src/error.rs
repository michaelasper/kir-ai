use thiserror::Error;

#[derive(Debug, Error)]
#[error("{code}: {message}")]
pub struct HubError {
    code: &'static str,
    message: String,
}

impl HubError {
    pub fn code(&self) -> &'static str {
        self.code
    }

    pub(crate) fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            code: "invalid_request",
            message: message.into(),
        }
    }

    pub(crate) fn invalid_response(message: impl Into<String>) -> Self {
        Self {
            code: "model_integrity_failed",
            message: message.into(),
        }
    }

    pub(crate) fn integrity_failed(message: impl Into<String>) -> Self {
        Self {
            code: "model_integrity_failed",
            message: message.into(),
        }
    }

    #[cfg(feature = "remote")]
    pub(crate) fn auth_failed(message: impl Into<String>) -> Self {
        Self {
            code: "model_auth_failed",
            message: message.into(),
        }
    }

    pub(crate) fn model_not_found(message: impl Into<String>) -> Self {
        Self {
            code: "model_not_found",
            message: message.into(),
        }
    }

    #[cfg(feature = "remote")]
    pub(crate) fn network(message: impl ToString) -> Self {
        Self {
            code: "model_download_interrupted",
            message: message.to_string(),
        }
    }

    pub(crate) fn io(message: impl ToString) -> Self {
        Self {
            code: "model_download_interrupted",
            message: message.to_string(),
        }
    }

    pub(crate) fn model_revision_unresolved(message: impl Into<String>) -> Self {
        Self {
            code: "model_revision_unresolved",
            message: message.into(),
        }
    }
}
