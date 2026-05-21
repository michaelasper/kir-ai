use crate::RuntimeError;
use crate::runtime::Runtime;
use llm_api::{ApiError, ChatCompletionRequest, ChatRole, CompletionRequest, ResponseFormat};
use llm_backend::{BackendCapabilities, ModelBackend, SamplingConfig};

impl<B> Runtime<B>
where
    B: ModelBackend,
{
    pub fn backend_capabilities(&self) -> BackendCapabilities {
        self.backend.capabilities()
    }

    pub fn validate_chat_request_capabilities(
        &self,
        request: &ChatCompletionRequest,
        streaming: bool,
    ) -> Result<(), RuntimeError> {
        let capabilities = self.backend_capabilities();
        if !capabilities.chat_completions {
            return unsupported("backend does not advertise chat completion support");
        }
        validate_streaming_capability(capabilities, streaming)?;
        validate_sampling_capability(
            capabilities,
            SamplingConfig::from_openai_controls(request.temperature, request.top_p)?,
        )?;
        if chat_request_uses_tools(request) && !capabilities.tool_calls {
            return unsupported("backend does not advertise tool-call support");
        }
        if matches!(
            request.response_format.as_ref(),
            Some(ResponseFormat::JsonObject)
        ) && !capabilities.json_object_mode
        {
            return unsupported("backend does not advertise json_object response_format support");
        }
        Ok(())
    }

    pub fn validate_completion_request_capabilities(
        &self,
        request: &CompletionRequest,
        streaming: bool,
    ) -> Result<(), RuntimeError> {
        let capabilities = self.backend_capabilities();
        if !capabilities.raw_completions {
            return unsupported("backend does not advertise raw completion support");
        }
        validate_streaming_capability(capabilities, streaming)?;
        validate_sampling_capability(
            capabilities,
            SamplingConfig::from_openai_controls(request.temperature, request.top_p)?,
        )
    }
}

fn validate_streaming_capability(
    capabilities: BackendCapabilities,
    streaming: bool,
) -> Result<(), RuntimeError> {
    if streaming && !capabilities.streaming {
        return unsupported("backend does not advertise streaming support");
    }
    Ok(())
}

fn validate_sampling_capability(
    capabilities: BackendCapabilities,
    sampling: SamplingConfig,
) -> Result<(), RuntimeError> {
    match sampling {
        SamplingConfig::Greedy if !capabilities.sampling_greedy => {
            unsupported("backend does not advertise greedy sampling support")
        }
        SamplingConfig::TopP { .. } if !capabilities.sampling_top_p => unsupported(
            "backend does not advertise top-p sampling support; use temperature 0 for greedy decoding",
        ),
        SamplingConfig::Greedy | SamplingConfig::TopP { .. } => Ok(()),
    }
}

fn chat_request_uses_tools(request: &ChatCompletionRequest) -> bool {
    !request.tools.is_empty()
        || request.messages.iter().any(|message| {
            matches!(message.role, ChatRole::Tool)
                || message.tool_call_id.is_some()
                || !message.tool_calls.is_empty()
        })
}

fn unsupported(message: &'static str) -> Result<(), RuntimeError> {
    Err(ApiError::unsupported_capability(message).into())
}
