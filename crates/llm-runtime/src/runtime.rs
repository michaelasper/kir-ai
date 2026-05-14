use crate::adapters::{ChatAdapter, SelectedChatAdapter, chat_adapter_for_metadata};
use crate::{RuntimeError, ToolSchemaNormalization};
use llm_api::{ToolDefinition, canonical_tool_schema_json, canonicalize_tool_schemas};
use llm_backend::{BackendCacheContext, BackendChatContext, BackendModelMetadata, ModelBackend};
use std::borrow::Cow;

#[derive(Debug, Clone)]
pub struct Runtime<B> {
    pub(crate) backend: B,
    pub(crate) options: RuntimeOptions,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RuntimeOptions {
    pub tool_schema_normalization: ToolSchemaNormalization,
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

    pub(crate) fn chat_adapter(&self) -> Result<SelectedChatAdapter, RuntimeError> {
        chat_adapter_for_metadata(&self.backend.model_metadata())
    }

    pub(crate) fn prepare_chat_backend(
        &self,
        adapter: SelectedChatAdapter,
        request: &llm_api::ChatCompletionRequest,
    ) -> Result<(BackendCacheContext, String, Option<BackendChatContext>), RuntimeError> {
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
        let schema = match self.options.tool_schema_normalization {
            ToolSchemaNormalization::Preserve => serde_json::to_string(tools)?,
            ToolSchemaNormalization::Canonical => canonical_tool_schema_json(tools)?,
        };
        Ok(Some(schema))
    }
}
