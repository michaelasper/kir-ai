use llm_api::{ApiError, ChatMessage, ToolDefinition};
use llm_backend_contracts::{
    BackendCacheContext, BackendChatContext, BackendChatMessage, BackendChatRole,
    BackendModelMetadata, BackendToolCall, BackendToolCallFunction, BackendToolCallType,
};
use llm_chat_template::render_family_chat_template_with_tool_instruction;
use llm_models::ModelFamily;
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
pub(crate) struct ToolMarkupStreamState {
    policy: ToolMarkupPolicy,
    first_start: Option<usize>,
    completed_prefix_len: Option<usize>,
    previous_len: usize,
    max_start_marker_len: usize,
    max_end_marker_len: usize,
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
const LLAMA_UNMARKED_JSON_TRUNCATION_TOKENS: [&str; 3] =
    ["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>"];
const QWEN_TOOL_INSTRUCTION: &str =
    "Tools are available. Return tool invocations inside <tool_call> JSON blocks.\n";
const DEEPSEEK_TOOL_INSTRUCTION: &str =
    "You may call tools by emitting DeepSeek tool call blocks with exact tool names.\n";
const LLAMA_TOOL_INSTRUCTION: &str = concat!(
    "Tools are available. To call a function, respond with JSON in the form ",
    r#"{"name":"function_name","arguments":{"argument":"value"}}"#,
    ". Do not use variables.\n"
);

impl ToolMarkupPolicy {
    pub(crate) const fn new(markers: &'static [ToolMarkupMarkers]) -> Self {
        Self { markers }
    }

    pub(crate) fn stream_state(self) -> ToolMarkupStreamState {
        ToolMarkupStreamState {
            policy: self,
            first_start: None,
            completed_prefix_len: None,
            previous_len: 0,
            max_start_marker_len: self
                .markers
                .iter()
                .map(|markers| markers.start_marker.len())
                .max()
                .unwrap_or(0),
            max_end_marker_len: self
                .markers
                .iter()
                .map(|markers| markers.end_marker.len())
                .max()
                .unwrap_or(0),
        }
    }

    pub(crate) fn contains_start(self, content: &str) -> bool {
        self.find_start_from(content, 0).is_some()
    }

    fn find_start_from(self, content: &str, search_start: usize) -> Option<usize> {
        self.markers
            .iter()
            .filter_map(|markers| {
                content[search_start..]
                    .find(markers.start_marker)
                    .map(|start| search_start + start)
            })
            .min()
    }

    fn latest_completed_prefix_from(self, content: &str, search_start: usize) -> Option<usize> {
        self.markers
            .iter()
            .filter_map(|markers| {
                content[search_start..]
                    .rfind(markers.end_marker)
                    .map(|end| search_start + end + markers.end_marker.len())
            })
            .max()
    }

    fn withheld_start_prefix_len(self, content: &str) -> usize {
        self.markers
            .iter()
            .flat_map(|markers| {
                (1..markers.start_marker.len()).filter(move |prefix_len| {
                    markers.start_marker.is_char_boundary(*prefix_len)
                        && content.ends_with(&markers.start_marker[..*prefix_len])
                })
            })
            .max()
            .unwrap_or(0)
    }
}

impl ToolMarkupStreamState {
    pub(crate) fn observe(&mut self, content: &str) {
        if self.first_start.is_none() {
            let search_start = floor_char_boundary(
                content,
                self.previous_len
                    .saturating_sub(self.max_start_marker_len.saturating_sub(1)),
            );
            self.first_start = self.policy.find_start_from(content, search_start);
        }

        let search_start = floor_char_boundary(
            content,
            self.previous_len
                .saturating_sub(self.max_end_marker_len.saturating_sub(1)),
        );
        if let Some(prefix_len) = self
            .policy
            .latest_completed_prefix_from(content, search_start)
        {
            self.completed_prefix_len = Some(
                self.completed_prefix_len
                    .map_or(prefix_len, |current| current.max(prefix_len)),
            );
        }

        self.previous_len = content.len();
    }

    pub(crate) fn safe_emit_len(self, content: &str) -> usize {
        if let Some(start) = self.first_start {
            return start;
        }
        content.len() - self.policy.withheld_start_prefix_len(content)
    }

    pub(crate) const fn completed_prefix_len(self) -> Option<usize> {
        self.completed_prefix_len
    }

    pub(crate) const fn contains_start(self) -> bool {
        self.first_start.is_some()
    }
}

fn floor_char_boundary(content: &str, mut index: usize) -> usize {
    while !content.is_char_boundary(index) {
        index -= 1;
    }
    index
}

pub(crate) trait ChatAdapter {
    fn cache_context(self, tool_schema: Option<String>) -> BackendCacheContext;
    fn backend_chat_context(
        self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> BackendChatContext;
    fn render_prompt(
        self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> Result<String, RuntimeError>;
    fn parse_complete(self, text: &str) -> Result<ParsedAssistant, RuntimeError>;
    fn tool_markup_policy(self) -> ToolMarkupPolicy;
    fn unmarked_tool_json_truncation_tokens(self) -> &'static [&'static str];
}

impl ChatAdapter for SelectedChatAdapter {
    fn cache_context(self, tool_schema: Option<String>) -> BackendCacheContext {
        let adapter = self.family.adapter();
        BackendCacheContext::chat_template_with_kwargs(
            adapter.cache_template_id(),
            tool_schema,
            adapter.chat_template_kwargs_json().map(str::to_owned),
        )
    }

    fn backend_chat_context(
        self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> BackendChatContext {
        BackendChatContext {
            messages: messages.iter().map(backend_chat_message).collect(),
            tools: crate::backend_request::backend_tool_definitions(tools),
        }
    }

    fn render_prompt(
        self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> Result<String, RuntimeError> {
        Ok(render_family_chat_template_with_tool_instruction(
            self.family,
            messages,
            tools,
            self.tool_instruction(tools),
        )?)
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

    fn unmarked_tool_json_truncation_tokens(self) -> &'static [&'static str] {
        match self.family {
            ModelFamily::Llama => &LLAMA_UNMARKED_JSON_TRUNCATION_TOKENS,
            ModelFamily::Qwen | ModelFamily::DeepSeek | ModelFamily::Gemma => &[],
        }
    }
}

impl SelectedChatAdapter {
    fn tool_instruction(self, tools: &[ToolDefinition]) -> Option<&'static str> {
        if tools.is_empty() {
            return None;
        }
        match self.family {
            ModelFamily::Qwen => Some(QWEN_TOOL_INSTRUCTION),
            ModelFamily::DeepSeek => Some(DEEPSEEK_TOOL_INSTRUCTION),
            ModelFamily::Llama => Some(LLAMA_TOOL_INSTRUCTION),
            ModelFamily::Gemma => None,
        }
    }

    pub(crate) fn parses_unmarked_tool_calls(self) -> bool {
        matches!(self.family, ModelFamily::Llama)
    }
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

fn backend_chat_message(message: &ChatMessage) -> BackendChatMessage {
    BackendChatMessage {
        role: backend_chat_role(&message.role),
        content: message.content.clone(),
        name: message.name.clone(),
        tool_call_id: message.tool_call_id.clone(),
        tool_calls: message
            .tool_calls
            .iter()
            .map(|tool_call| BackendToolCall {
                id: tool_call.id.clone(),
                call_type: backend_tool_call_type(&tool_call.call_type),
                function: BackendToolCallFunction {
                    name: tool_call.function.name.clone(),
                    arguments: tool_call.function.arguments.clone(),
                },
            })
            .collect(),
    }
}

fn backend_chat_role(role: &llm_api::ChatRole) -> BackendChatRole {
    match role {
        llm_api::ChatRole::System => BackendChatRole::System,
        llm_api::ChatRole::User => BackendChatRole::User,
        llm_api::ChatRole::Assistant => BackendChatRole::Assistant,
        llm_api::ChatRole::Tool => BackendChatRole::Tool,
    }
}

fn backend_tool_call_type(tool_type: &llm_api::ToolCallType) -> BackendToolCallType {
    match tool_type {
        llm_api::ToolCallType::Function => BackendToolCallType::Function,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_markup_stream_state_withholds_split_start_marker() {
        let policy = ToolMarkupPolicy::new(&JSON_TOOL_MARKERS);
        let mut state = policy.stream_state();
        let mut content = "hello <to".to_owned();

        state.observe(&content);
        assert!(!state.contains_start());
        assert_eq!(state.safe_emit_len(&content), "hello ".len());

        content.push_str("ol_call>{\"name\":\"lookup\"}</tool_call> trailing");
        state.observe(&content);
        assert!(state.contains_start());
        assert_eq!(state.safe_emit_len(&content), "hello ".len());
        assert_eq!(
            state.completed_prefix_len(),
            Some("hello <tool_call>{\"name\":\"lookup\"}</tool_call>".len())
        );
    }

    #[test]
    fn tool_markup_stream_state_finds_latest_completed_prefix() {
        let policy = ToolMarkupPolicy::new(&JSON_TOOL_MARKERS);
        let mut state = policy.stream_state();
        let mut content = "<tool_call>{}</tool_call>".to_owned();

        state.observe(&content);
        assert_eq!(state.completed_prefix_len(), Some(content.len()));

        content.push_str(" text <tool_call>{\"name\":\"second\"}</tool_call>");
        state.observe(&content);
        assert_eq!(state.completed_prefix_len(), Some(content.len()));
    }
}
