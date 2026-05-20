use crate::adapters::{ChatAdapter, SelectedChatAdapter, chat_adapter_for_metadata};
use crate::{RuntimeError, ToolSchemaNormalization};
use llm_api::{
    RequestLimits, ToolCallType, ToolDefinition, ValidateRequest, Validated,
    canonicalize_json_value, canonicalize_tool_schemas,
};
use llm_backend::{
    BackendCacheContext, BackendChatContext, BackendModelMetadata, BackendToolDefinition,
    BackendToolFunctionDefinition, BackendToolType, ModelBackend,
};
use std::borrow::Cow;

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

    pub(crate) fn prepare_chat_backend(
        &self,
        adapter: SelectedChatAdapter,
        request: &llm_api::ChatCompletionRequest,
    ) -> Result<(BackendCacheContext, String, BackendChatContext), RuntimeError> {
        let effective_tools = self.effective_tools(&request.tools);
        let tool_schema = self.tool_schema_json(&request.tools)?;
        let cache_context = adapter.cache_context(tool_schema);
        let prompt = adapter.render_prompt(&request.messages, effective_tools.as_ref())?;
        let chat_context =
            adapter.backend_chat_context(&request.messages, effective_tools.as_ref());
        Ok((cache_context, prompt, chat_context))
    }

    fn effective_tools<'a>(&self, tools: &'a [ToolDefinition]) -> Cow<'a, [ToolDefinition]> {
        match self.options.tool_schema_normalization {
            ToolSchemaNormalization::Preserve => Cow::Borrowed(tools),
            ToolSchemaNormalization::Canonical => Cow::Owned(canonicalize_tool_schemas(tools)),
        }
    }

    fn tool_schema_json(&self, tools: &[ToolDefinition]) -> Result<Option<String>, RuntimeError> {
        if tools.is_empty() {
            return Ok(None);
        }
        let effective_tools = self.effective_tools(tools);
        let backend_tools = backend_tool_definitions(effective_tools.as_ref());
        let schema = match self.options.tool_schema_normalization {
            ToolSchemaNormalization::Preserve => serde_json::to_string(&backend_tools)?,
            ToolSchemaNormalization::Canonical => {
                let value = serde_json::to_value(&backend_tools)?;
                serde_json::to_string(&canonicalize_json_value(&value))?
            }
        };
        Ok(Some(schema))
    }
}

fn backend_tool_definitions(tools: &[ToolDefinition]) -> Vec<BackendToolDefinition> {
    tools.iter().map(backend_tool_definition).collect()
}

fn backend_tool_definition(tool: &ToolDefinition) -> BackendToolDefinition {
    BackendToolDefinition {
        tool_type: backend_tool_type(&tool.tool_type),
        function: BackendToolFunctionDefinition {
            name: tool.function.name.clone(),
            description: tool.function.description.clone(),
            parameters: tool.function.parameters.clone(),
        },
    }
}

fn backend_tool_type(tool_type: &ToolCallType) -> BackendToolType {
    match tool_type {
        ToolCallType::Function => BackendToolType::Function,
    }
}
