use super::safetensors::TensorLoadError;

pub use llm_backend_contracts::*;

impl From<TensorLoadError> for BackendError {
    fn from(value: TensorLoadError) -> Self {
        Self::backend_failure(value.code(), value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::safetensors::TensorLoadError;

    #[test]
    fn tensor_load_error_context_survives_backend_error_conversion() {
        let err = BackendError::from(TensorLoadError::integrity("bad tensor header"));

        assert_eq!(err.backend_failure_code(), Some("model_integrity_failed"));
        assert_eq!(
            err.other_message(),
            Some("model_integrity_failed: bad tensor header")
        );
    }
}
