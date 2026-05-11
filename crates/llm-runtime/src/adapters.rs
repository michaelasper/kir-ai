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

#[derive(Debug, Clone, Copy)]
pub(crate) struct ToolMarkupPolicy {
    markers: &'static [ToolMarkupMarkers],
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ToolMarkupMarkers {
    pub(crate) start_marker: &'static str,
    pub(crate) end_marker: &'static str,
}

impl ToolMarkupMarkers {
    pub(crate) const fn new(start_marker: &'static str, end_marker: &'static str) -> Self {
        Self {
            start_marker,
            end_marker,
        }
    }
}

pub(crate) const JSON_TOOL_MARKERS: [ToolMarkupMarkers; 1] =
    [ToolMarkupMarkers::new("<tool_call>", "</tool_call>")];
pub(crate) const DEEPSEEK_TOOL_MARKERS: [ToolMarkupMarkers; 2] = [
    ToolMarkupMarkers::new("<｜tool▁calls▁begin｜>", "<｜tool▁calls▁end｜>"),
    ToolMarkupMarkers::new("<dsml_tool_call>", "</dsml_tool_call>"),
];
pub(crate) const GEMMA_TOOL_MARKERS: [ToolMarkupMarkers; 1] =
    [ToolMarkupMarkers::new("<|tool_call>", "<tool_call|>")];

impl ToolMarkupPolicy {
    pub(crate) const fn new(markers: &'static [ToolMarkupMarkers]) -> Self {
        Self { markers }
    }

    pub(crate) fn safe_emit_len(self, content: &str) -> usize {
        if let Some(start) = self
            .markers
            .iter()
            .filter_map(|markers| content.find(markers.start_marker))
            .min()
        {
            return start;
        }
        let withheld_prefix_len = self
            .markers
            .iter()
            .flat_map(|markers| {
                (1..markers.start_marker.len()).filter(move |prefix_len| {
                    markers.start_marker.is_char_boundary(*prefix_len)
                        && content.ends_with(&markers.start_marker[..*prefix_len])
                })
            })
            .max()
            .unwrap_or(0);
        content.len() - withheld_prefix_len
    }

    pub(crate) fn completed_prefix_len(self, content: &str) -> Option<usize> {
        self.markers
            .iter()
            .filter_map(|markers| {
                content
                    .rfind(markers.end_marker)
                    .map(|end| end + markers.end_marker.len())
            })
            .max()
    }

    pub(crate) fn contains_start(self, content: &str) -> bool {
        self.markers
            .iter()
            .any(|markers| content.contains(markers.start_marker))
    }
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
    fn tool_markup_policy(self) -> ToolMarkupPolicy;
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
        _tools: &[ToolDefinition],
    ) -> Option<BackendChatContext> {
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

    fn tool_markup_policy(self) -> ToolMarkupPolicy {
        match self.family {
            ModelFamily::Qwen | ModelFamily::Llama => ToolMarkupPolicy::new(&JSON_TOOL_MARKERS),
            ModelFamily::DeepSeek => ToolMarkupPolicy::new(&DEEPSEEK_TOOL_MARKERS),
            ModelFamily::Gemma => ToolMarkupPolicy::new(&GEMMA_TOOL_MARKERS),
        }
    }
}

impl SelectedChatAdapter {
    pub(crate) fn parses_unmarked_tool_calls(self) -> bool {
        matches!(self.family, ModelFamily::Llama)
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
    Ok(SelectedChatAdapter {
        family: parse_metadata_family(family)?,
    })
}

fn parse_metadata_family(family: &str) -> Result<ModelFamily, RuntimeError> {
    ModelFamily::parse_slug(family)
        .map_err(|err| ApiError::unsupported_capability(format!("{err} for chat rendering")).into())
}
