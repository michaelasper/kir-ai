use crate::adapters::{SelectedChatAdapter, chat_adapter_for_metadata};
use crate::{RuntimeError, ToolSchemaNormalization};
use llm_api::{RequestLimits, ValidateRequest, Validated};
use llm_backend::{BackendHealth, BackendModelMetadata, ModelBackend};

#[derive(Debug, Clone)]
pub struct Runtime<B> {
    pub(crate) backend: B,
    pub(crate) options: RuntimeOptions,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RuntimeOptions {
    pub tool_schema_normalization: ToolSchemaNormalization,
    pub request_limits: RequestLimits,
}

impl<B> Runtime<B>
where
    B: ModelBackend,
{
    pub fn new(backend: B) -> Self {
        Self::new_with_options(backend, RuntimeOptions::default())
    }

    pub fn new_with_options(backend: B, options: RuntimeOptions) -> Self {
        Self { backend, options }
    }

    pub fn model_id(&self) -> &str {
        self.backend.model_id()
    }

    pub fn model_metadata(&self) -> BackendModelMetadata {
        self.backend.model_metadata()
    }

    pub async fn backend_health(&self) -> BackendHealth {
        self.backend.health().await
    }

    pub(crate) fn ensure_runtime_validated<T>(
        &self,
        request: Validated<T>,
    ) -> Result<Validated<T>, RuntimeError>
    where
        T: ValidateRequest,
    {
        if request.request_limits() == self.options.request_limits {
            return Ok(request);
        }
        request
            .into_inner()
            .into_validated_with_limits(self.options.request_limits)
            .map_err(RuntimeError::from)
    }

    pub(crate) fn chat_adapter(&self) -> Result<SelectedChatAdapter, RuntimeError> {
        chat_adapter_for_metadata(&self.backend.model_metadata())
    }
}
