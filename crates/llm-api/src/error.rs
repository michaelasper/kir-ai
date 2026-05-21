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

    /// Returns the canonical HTTP status code for this API error.
    ///
    /// The numeric value keeps `llm-api` independent from any HTTP framework.
    pub fn http_status(&self) -> u16 {
        match self.code {
            "invalid_request" | "unsupported_capability" => 400,
            _ => 500,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_errors_expose_canonical_http_status() {
        let cases = [
            (ApiError::invalid_request("bad request"), 400),
            (ApiError::unsupported_capability("unsupported field"), 400),
        ];

        for (err, expected_status) in cases {
            assert_eq!(
                err.http_status(),
                expected_status,
                "{} should have a stable HTTP status",
                err.code()
            );
        }
    }
}
