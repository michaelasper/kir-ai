use llm_api::{ApiError, ChatMessage, ChatRole, ToolDefinition};
use llm_backend::{
    BackendCacheContext, BackendChatContext, BackendChatMessage, BackendChatRole,
    BackendModelMetadata,
};
use llm_models::ModelFamily;
use llm_tokenizer::render_family_chat_template;
use llm_tool_parser::{ParsedAssistant, parse_assistant_for_family};

use crate::RuntimeError;

#[derive(Debug, Clone, Copy)]
pub(crate) struct SelectedChatAdapter {
    family: ModelFamily,
}

pub(crate) trait ChatAdapter {
    fn cache_context(self, tools: &[ToolDefinition]) -> Result<BackendCacheContext, RuntimeError>;
    fn backend_chat_context(
        self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> Option<BackendChatContext>;
    fn render_prompt(
        self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> Result<String, RuntimeError>;
    fn parse_complete(self, text: &str) -> Result<ParsedAssistant, RuntimeError>;
}

impl ChatAdapter for SelectedChatAdapter {
    fn cache_context(self, tools: &[ToolDefinition]) -> Result<BackendCacheContext, RuntimeError> {
        let tool_schema = if tools.is_empty() {
            None
        } else {
            Some(serde_json::to_string(tools)?)
        };
        Ok(BackendCacheContext::chat_template(
            self.family.adapter().cache_template_id(),
            tool_schema,
        ))
    }

    fn backend_chat_context(
        self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> Option<BackendChatContext> {
        if !tools.is_empty() {
            return None;
        }
        let messages = messages
            .iter()
            .map(backend_chat_message)
            .collect::<Option<Vec<_>>>()?;
        Some(BackendChatContext { messages })
    }

    fn render_prompt(
        self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> Result<String, RuntimeError> {
        Ok(render_family_chat_template(self.family, messages, tools)?)
    }

    fn parse_complete(self, text: &str) -> Result<ParsedAssistant, RuntimeError> {
        Ok(parse_assistant_for_family(self.family, text)?)
    }
}

pub(crate) fn backend_chat_message(message: &ChatMessage) -> Option<BackendChatMessage> {
    if !message.tool_calls.is_empty() {
        return None;
    }
    let role = match message.role {
        ChatRole::System => BackendChatRole::System,
        ChatRole::User => BackendChatRole::User,
        ChatRole::Assistant => BackendChatRole::Assistant,
        ChatRole::Tool => return None,
    };
    Some(BackendChatMessage {
        role,
        content: message.content.clone().unwrap_or_default(),
    })
}

pub(crate) fn chat_adapter_for_metadata(
    metadata: &BackendModelMetadata,
) -> Result<SelectedChatAdapter, RuntimeError> {
    let Some(family) = metadata.family.as_deref() else {
        return Err(ApiError::unsupported_capability(format!(
            "backend `{}` did not declare a model family for chat rendering",
            metadata.backend
        ))
        .into());
    };
    match parse_metadata_family(family)? {
        family @ (ModelFamily::Qwen | ModelFamily::Gemma) => Ok(SelectedChatAdapter { family }),
        family => Err(unsupported_chat_family(family)),
    }
}

fn parse_metadata_family(family: &str) -> Result<ModelFamily, RuntimeError> {
    ModelFamily::parse_slug(family)
        .map_err(|err| ApiError::unsupported_capability(format!("{err} for chat rendering")).into())
}

fn unsupported_chat_family(family: ModelFamily) -> RuntimeError {
    ApiError::unsupported_capability(format!(
        "{} chat adapter support is deferred until Qwen production parity",
        family.display_name()
    ))
    .into()
}
