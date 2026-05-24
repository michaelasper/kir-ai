use crate::adapters::{SelectedChatAdapter, chat_adapter_for_metadata};
use crate::{RuntimeError, ToolSchemaNormalization};
use llm_api::{RequestLimits, ValidateRequest, Validated};
use llm_backend_contracts::{BackendHealth, BackendModelMetadata, ModelBackend};

/// High-level protocol runtime for one loaded backend model.
///
/// `Runtime` is the boundary where public OpenAI-compatible request types become
/// backend requests. It validates request shape and runtime capabilities before
/// prompt rendering, then normalizes backend output into API responses.
#[derive(Debug, Clone)]
pub struct Runtime<B> {
    pub(crate) backend: B,
    pub(crate) options: RuntimeOptions,
}

/// Configuration that affects runtime validation and prompt/backend translation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RuntimeOptions {
    /// Whether tool schemas are preserved as provided or canonicalized before prompt/cache use.
    pub tool_schema_normalization: ToolSchemaNormalization,
    /// Request limits enforced by runtime entry points.
    pub request_limits: RequestLimits,
}

impl<B> Runtime<B>
where
    B: ModelBackend,
{
    /// Creates a runtime with default validation and prompt translation options.
    pub fn new(backend: B) -> Self {
        Self::new_with_options(backend, RuntimeOptions::default())
    }

    /// Creates a runtime with explicit validation and prompt translation options.
    pub fn new_with_options(backend: B, options: RuntimeOptions) -> Self {
        Self { backend, options }
    }

    /// Returns the backend's active model identifier.
    pub fn model_id(&self) -> &str {
        self.backend.model_id()
    }

    /// Returns backend metadata used by model lists, prompt adapters, and diagnostics.
    pub fn model_metadata(&self) -> BackendModelMetadata {
        self.backend.model_metadata()
    }

    /// Queries backend readiness without dispatching generation.
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
