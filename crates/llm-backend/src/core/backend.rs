use super::safetensors::TensorLoadError;

#[cfg(test)]
pub(crate) use llm_backend_contracts::BackendCacheContext;
pub(super) use llm_backend_contracts::BackendError;
#[cfg(test)]
use llm_backend_contracts::BackendFailureClass;
#[cfg(feature = "test-utils")]
pub(super) use llm_backend_contracts::{
    BackendChatRole, BackendFinishReason, BackendModelMetadata, BackendOutput, BackendRequest,
    BackendToolChoice, BackendToolDefinition, ModelBackend,
};

impl From<TensorLoadError> for BackendError {
    fn from(value: TensorLoadError) -> Self {
        if value.is_cancelled() {
            return Self::cancelled();
        }
        Self::tensor_load(value.code(), value.message())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::safetensors::TensorLoadError;

    #[test]
    fn tensor_load_error_context_survives_backend_error_conversion() {
        let err = BackendError::from(TensorLoadError::integrity("bad tensor header"));

        assert_eq!(
            err.backend_failure_class(),
            Some(BackendFailureClass::TensorLoad)
        );
        assert_eq!(err.backend_failure_code(), Some("model_integrity_failed"));
        assert_eq!(err.other_message(), Some("bad tensor header"));
    }

    #[test]
    fn tensor_load_cancellation_converts_to_backend_cancellation() {
        let err = BackendError::from(TensorLoadError::cancelled());

        assert!(err.is_cancelled());
        assert_eq!(err.backend_failure_class(), None);
    }
}
