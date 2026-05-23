use crate::RuntimeError;
use crate::adapters::{ChatAdapter, SelectedChatAdapter};
use crate::runtime::Runtime;
use crate::tool_call::{ToolSchemaNormalization, required_backend_tool_choice};
use llm_api::{
    ChatCompletionRequest, CompletionRequest, ResponseFormat, ToolCallType, ToolDefinition,
    canonicalize_json_value, canonicalize_tool_schemas,
};
use llm_backend_contracts::{
    BackendCacheContext, BackendChatContext, BackendRequest, BackendToolDefinition,
    BackendToolFunctionDefinition, BackendToolType, ModelBackend, SamplingConfig,
};
use std::borrow::Cow;

impl<B> Runtime<B>
where
    B: ModelBackend,
{
    pub(crate) fn chat_backend_request(
        &self,
        adapter: SelectedChatAdapter,
        request: &ChatCompletionRequest,
    ) -> Result<BackendRequest, RuntimeError> {
        let (cache_context, prompt, chat_context) = self.prepare_chat_backend(adapter, request)?;
        Ok(BackendRequest::chat_completion(
            request.model.clone(),
            prompt,
            chat_context,
            request.effective_max_tokens(),
            SamplingConfig::from_openai_controls(request.temperature, request.top_p)?,
            required_backend_tool_choice(request),
            matches!(
                request.response_format.as_ref(),
                Some(ResponseFormat::JsonObject)
            ),
            cache_context,
        ))
    }

    pub(crate) fn prepare_chat_backend(
        &self,
        adapter: SelectedChatAdapter,
        request: &ChatCompletionRequest,
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

pub(crate) fn completion_backend_request(
    request: CompletionRequest,
) -> Result<BackendRequest, RuntimeError> {
    Ok(BackendRequest::raw_completion(
        request.model,
        request.prompt,
        request.max_tokens,
        SamplingConfig::from_openai_controls(request.temperature, request.top_p)?,
    ))
}

pub(crate) fn backend_tool_definitions(tools: &[ToolDefinition]) -> Vec<BackendToolDefinition> {
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
