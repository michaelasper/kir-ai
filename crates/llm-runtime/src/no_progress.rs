use llm_api::{ChatCompletionRequest, ChatMessage, ChatRole, ToolCall};
use llm_tool_parser::ParsedAssistant;

use crate::adapters::ToolMarkupPolicy;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoProgressClass {
    EmptyCompletion,
    EmptyHighOutputCompletion,
    HiddenOnlyOutput,
    TextFallbackRequiredTool,
    RepeatedInvalidToolCall,
    RepeatedAssistantContent,
    StalledAssistantTurn,
}

impl NoProgressClass {
    pub fn code(self) -> &'static str {
        match self {
            Self::EmptyCompletion => "no_progress_empty_completion",
            Self::EmptyHighOutputCompletion => "no_progress_empty_high_output_completion",
            Self::HiddenOnlyOutput => "no_progress_hidden_only_output",
            Self::TextFallbackRequiredTool => "no_progress_missing_required_tool_call",
            Self::RepeatedInvalidToolCall => "no_progress_repeated_invalid_tool_call",
            Self::RepeatedAssistantContent => "no_progress_repeated_assistant_content",
            Self::StalledAssistantTurn => "no_progress_stalled_assistant_turn",
        }
    }
}

pub fn classify_no_progress(
    content: &str,
    completion_tokens: u64,
) -> Option<NoProgressClass> {
    if content.trim().is_empty() && completion_tokens >= 1024 {
        return Some(NoProgressClass::EmptyHighOutputCompletion);
    }
    if content.trim().is_empty() {
        return Some(NoProgressClass::EmptyCompletion);
    }
    None
}

pub(crate) fn classify_chat_no_progress(
    raw_text: &str,
    parsed: &ParsedAssistant,
    completion_tokens: u64,
    required_tool_pending: bool,
    request: &ChatCompletionRequest,
    tool_markup_policy: ToolMarkupPolicy,
) -> Option<NoProgressClass> {
    if parsed.tool_calls.is_empty() {
        if required_tool_pending && !tool_markup_policy.contains_start(raw_text) {
            return Some(NoProgressClass::TextFallbackRequiredTool);
        }
        if parsed.content.trim().is_empty()
            && parsed
                .reasoning
                .as_deref()
                .is_some_and(|reasoning| !reasoning.trim().is_empty())
        {
            return Some(NoProgressClass::HiddenOnlyOutput);
        }
        if let Some(class) = classify_no_progress(&parsed.content, completion_tokens) {
            return Some(class);
        }
        if repeated_assistant_content(&parsed.content, request) {
            return Some(NoProgressClass::RepeatedAssistantContent);
        }
        if stalled_assistant_turn(&parsed.content) {
            return Some(NoProgressClass::StalledAssistantTurn);
        }
    } else if repeated_invalid_tool_call(parsed, request) {
        return Some(NoProgressClass::RepeatedInvalidToolCall);
    }
    if raw_text.trim().is_empty() {
        return classify_no_progress(raw_text, completion_tokens);
    }
    None
}

fn repeated_assistant_content(content: &str, request: &ChatCompletionRequest) -> bool {
    let normalized = normalized_progress_text(content);
    if normalized.is_empty() {
        return false;
    }
    request
        .messages
        .iter()
        .rev()
        .filter(|message| message.role == ChatRole::Assistant && message.tool_calls.is_empty())
        .filter_map(|message| message.content.as_deref())
        .map(normalized_progress_text)
        .any(|previous| previous == normalized)
}

fn repeated_invalid_tool_call(parsed: &ParsedAssistant, request: &ChatCompletionRequest) -> bool {
    parsed.tool_calls.iter().any(|generated| {
        request
            .messages
            .iter()
            .enumerate()
            .rev()
            .any(|(index, message)| {
                message.role == ChatRole::Assistant
                    && message
                        .tool_calls
                        .iter()
                        .any(|previous| same_tool_call(previous, generated))
                    && following_tool_result_failed(&request.messages[index + 1..])
            })
    })
}

fn same_tool_call(previous: &ToolCall, generated: &ToolCall) -> bool {
    previous.function.name == generated.function.name
        && previous.function.arguments == generated.function.arguments
}

fn following_tool_result_failed(messages: &[ChatMessage]) -> bool {
    for message in messages {
        if message.role == ChatRole::User {
            return false;
        }
        if message.role == ChatRole::Tool
            && message
                .content
                .as_deref()
                .is_some_and(tool_result_indicates_failure)
        {
            return true;
        }
    }
    false
}

fn tool_result_indicates_failure(content: &str) -> bool {
    let lower = content.to_ascii_lowercase();
    [
        "error",
        "failed",
        "failure",
        "invalid",
        "not found",
        "no such file",
        "denied",
        "timeout",
        "exception",
        "panic",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

fn stalled_assistant_turn(content: &str) -> bool {
    let normalized = normalized_progress_text(content);
    if normalized.is_empty() || normalized.split_whitespace().count() > 16 {
        return false;
    }
    [
        "i will get started",
        "i ll get started",
        "i will check",
        "i will look",
        "let me check",
        "let me look",
        "working on it",
        "i can help with that",
        "sure i can help",
    ]
    .iter()
    .any(|phrase| normalized.contains(phrase))
}

fn normalized_progress_text(content: &str) -> String {
    content
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::{
        DEEPSEEK_TOOL_MARKERS, GEMMA_TOOL_MARKERS, JSON_TOOL_MARKERS, ToolMarkupPolicy,
    };
    use llm_api::ChatCompletionRequest;

    fn minimal_request() -> ChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 64
        }))
        .unwrap()
    }

    #[test]
    fn deepseek_marker_prevents_text_fallback_when_parser_fails() {
        let raw_text = r#"<dsml_tool_call>{"name":"lookup","arguments":{"query":"rust"}}</dsml_tool_call>"#;
        let parsed = ParsedAssistant::content(raw_text);
        let policy = ToolMarkupPolicy::new(&DEEPSEEK_TOOL_MARKERS);

        let result = classify_chat_no_progress(
            raw_text,
            &parsed,
            100,
            true,
            &minimal_request(),
            policy,
        );

        assert_ne!(result, Some(NoProgressClass::TextFallbackRequiredTool));
    }

    #[test]
    fn gemma_marker_prevents_text_fallback_when_parser_fails() {
        let raw_text = "<|tool_call>call:lookup{\"query\":\"rust\"}<tool_call|>";
        let parsed = ParsedAssistant::content(raw_text);
        let policy = ToolMarkupPolicy::new(&GEMMA_TOOL_MARKERS);

        let result = classify_chat_no_progress(
            raw_text,
            &parsed,
            100,
            true,
            &minimal_request(),
            policy,
        );

        assert_ne!(result, Some(NoProgressClass::TextFallbackRequiredTool));
    }

    #[test]
    fn deepseek_native_marker_prevents_text_fallback() {
        let raw_text = "<｜tool▁calls▁begin｜>{\"name\":\"lookup\",\"arguments\":{\"query\":\"rust\"}}<｜tool▁calls▁end｜>";
        let parsed = ParsedAssistant::content(raw_text);
        let policy = ToolMarkupPolicy::new(&DEEPSEEK_TOOL_MARKERS);

        let result = classify_chat_no_progress(
            raw_text,
            &parsed,
            100,
            true,
            &minimal_request(),
            policy,
        );

        assert_ne!(result, Some(NoProgressClass::TextFallbackRequiredTool));
    }

    #[test]
    fn text_without_marker_returns_fallback_when_required_tool_pending() {
        let raw_text = "I will help you with that task.";
        let parsed = ParsedAssistant::content(raw_text);
        let policy = ToolMarkupPolicy::new(&DEEPSEEK_TOOL_MARKERS);

        let result = classify_chat_no_progress(
            raw_text,
            &parsed,
            100,
            true,
            &minimal_request(),
            policy,
        );

        assert_eq!(result, Some(NoProgressClass::TextFallbackRequiredTool));
    }

    #[test]
    fn wrong_family_marker_does_not_prevent_fallback() {
        let raw_text = "<dsml_tool_call>{\"name\":\"lookup\"}</dsml_tool_call>";
        let parsed = ParsedAssistant::content(raw_text);
        let policy = ToolMarkupPolicy::new(&GEMMA_TOOL_MARKERS);

        let result = classify_chat_no_progress(
            raw_text,
            &parsed,
            100,
            true,
            &minimal_request(),
            policy,
        );

        assert_eq!(result, Some(NoProgressClass::TextFallbackRequiredTool));
    }

    #[test]
    fn qwen_marker_prevents_text_fallback_when_parser_fails() {
        let raw_text = "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"query\":\"rust\"}}</tool_call>";
        let parsed = ParsedAssistant::content(raw_text);
        let policy = ToolMarkupPolicy::new(&JSON_TOOL_MARKERS);

        let result = classify_chat_no_progress(
            raw_text,
            &parsed,
            100,
            true,
            &minimal_request(),
            policy,
        );

        assert_ne!(result, Some(NoProgressClass::TextFallbackRequiredTool));
    }
}
