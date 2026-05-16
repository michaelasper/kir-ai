use llm_api::{ChatCompletionRequest, ChatMessage, ChatRole, ToolCall};
use llm_tool_parser::ParsedAssistant;
use serde_json::Value;
use std::collections::BTreeSet;

use crate::adapters::ToolMarkupPolicy;
use crate::tool_schema::schema_requires_string_intent_argument;

const EXACT_REPEATED_INVALID_TOOL_CALL_THRESHOLD: usize = 5;
const FUZZY_REPEATED_INVALID_TOOL_CALL_THRESHOLD: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoProgressClass {
    EmptyCompletion,
    EmptyHighOutputCompletion,
    HiddenOnlyOutput,
    TextFallbackRequiredTool,
    RepeatedInvalidToolCall,
    FuzzyRepeatedInvalidToolCall,
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
            Self::FuzzyRepeatedInvalidToolCall => "no_progress_fuzzy_repeated_invalid_tool_call",
            Self::RepeatedAssistantContent => "no_progress_repeated_assistant_content",
            Self::StalledAssistantTurn => "no_progress_stalled_assistant_turn",
        }
    }
}

pub fn classify_no_progress(content: &str, completion_tokens: u64) -> Option<NoProgressClass> {
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
    } else if let Some(class) = classify_repeated_invalid_tool_call_no_progress(parsed, request) {
        return Some(class);
    }
    if parsed.tool_calls.is_empty() && raw_text.trim().is_empty() {
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

pub(crate) fn classify_repeated_invalid_tool_call_no_progress(
    parsed: &ParsedAssistant,
    request: &ChatCompletionRequest,
) -> Option<NoProgressClass> {
    let failed_tool_call_ids = failed_tool_call_ids(&request.messages);
    if failed_tool_call_ids.is_empty() {
        return None;
    }

    for generated in &parsed.tool_calls {
        let generated_arguments = normalized_tool_call_arguments(generated, request);
        let generated_key_set = argument_key_set(&generated_arguments);
        let mut exact_count = 1;
        let mut fuzzy_count = 1;

        for message in &request.messages {
            if message.role != ChatRole::Assistant {
                continue;
            }
            for previous in &message.tool_calls {
                if !failed_tool_call_ids.contains(previous.id.as_str())
                    || previous.function.name != generated.function.name
                {
                    continue;
                }
                let previous_arguments = normalized_tool_call_arguments(previous, request);
                if previous_arguments == generated_arguments {
                    exact_count += 1;
                    continue;
                }
                if generated_key_set.as_ref().is_some_and(|keys| {
                    argument_key_set(&previous_arguments).as_ref() == Some(keys)
                }) {
                    fuzzy_count += 1;
                }
            }
        }

        if exact_count >= EXACT_REPEATED_INVALID_TOOL_CALL_THRESHOLD {
            return Some(NoProgressClass::RepeatedInvalidToolCall);
        }
        if fuzzy_count >= FUZZY_REPEATED_INVALID_TOOL_CALL_THRESHOLD {
            return Some(NoProgressClass::FuzzyRepeatedInvalidToolCall);
        }
    }

    None
}

fn failed_tool_call_ids(messages: &[ChatMessage]) -> BTreeSet<&str> {
    messages
        .iter()
        .filter(|message| message.role == ChatRole::Tool)
        .filter(|message| {
            message
                .content
                .as_deref()
                .is_some_and(tool_result_indicates_failure)
        })
        .filter_map(|message| message.tool_call_id.as_deref())
        .collect()
}

fn normalized_tool_call_arguments(tool_call: &ToolCall, request: &ChatCompletionRequest) -> Value {
    let mut arguments = tool_call.function.arguments.clone();
    if tool_schema_requires_string_intent_argument(&tool_call.function.name, request)
        && let Some(object) = arguments.as_object_mut()
    {
        object.remove("_i");
    }
    arguments
}

fn tool_schema_requires_string_intent_argument(
    tool_name: &str,
    request: &ChatCompletionRequest,
) -> bool {
    request
        .tools
        .iter()
        .find(|tool| tool.function.name == tool_name)
        .is_some_and(|tool| schema_requires_string_intent_argument(&tool.function.parameters))
}

fn argument_key_set(arguments: &Value) -> Option<BTreeSet<&str>> {
    arguments
        .as_object()
        .map(|object| object.keys().map(String::as_str).collect())
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
        let raw_text =
            r#"<dsml_tool_call>{"name":"lookup","arguments":{"query":"rust"}}</dsml_tool_call>"#;
        let parsed = ParsedAssistant::content(raw_text);
        let policy = ToolMarkupPolicy::new(&DEEPSEEK_TOOL_MARKERS);

        let result =
            classify_chat_no_progress(raw_text, &parsed, 100, true, &minimal_request(), policy);

        assert_ne!(result, Some(NoProgressClass::TextFallbackRequiredTool));
    }

    #[test]
    fn gemma_marker_prevents_text_fallback_when_parser_fails() {
        let raw_text = "<|tool_call>call:lookup{\"query\":\"rust\"}<tool_call|>";
        let parsed = ParsedAssistant::content(raw_text);
        let policy = ToolMarkupPolicy::new(&GEMMA_TOOL_MARKERS);

        let result =
            classify_chat_no_progress(raw_text, &parsed, 100, true, &minimal_request(), policy);

        assert_ne!(result, Some(NoProgressClass::TextFallbackRequiredTool));
    }

    #[test]
    fn deepseek_native_marker_prevents_text_fallback() {
        let raw_text = "<｜tool▁calls▁begin｜>{\"name\":\"lookup\",\"arguments\":{\"query\":\"rust\"}}<｜tool▁calls▁end｜>";
        let parsed = ParsedAssistant::content(raw_text);
        let policy = ToolMarkupPolicy::new(&DEEPSEEK_TOOL_MARKERS);

        let result =
            classify_chat_no_progress(raw_text, &parsed, 100, true, &minimal_request(), policy);

        assert_ne!(result, Some(NoProgressClass::TextFallbackRequiredTool));
    }

    #[test]
    fn text_without_marker_returns_fallback_when_required_tool_pending() {
        let raw_text = "I will help you with that task.";
        let parsed = ParsedAssistant::content(raw_text);
        let policy = ToolMarkupPolicy::new(&DEEPSEEK_TOOL_MARKERS);

        let result =
            classify_chat_no_progress(raw_text, &parsed, 100, true, &minimal_request(), policy);

        assert_eq!(result, Some(NoProgressClass::TextFallbackRequiredTool));
    }

    #[test]
    fn wrong_family_marker_does_not_prevent_fallback() {
        let raw_text = "<dsml_tool_call>{\"name\":\"lookup\"}</dsml_tool_call>";
        let parsed = ParsedAssistant::content(raw_text);
        let policy = ToolMarkupPolicy::new(&GEMMA_TOOL_MARKERS);

        let result =
            classify_chat_no_progress(raw_text, &parsed, 100, true, &minimal_request(), policy);

        assert_eq!(result, Some(NoProgressClass::TextFallbackRequiredTool));
    }

    #[test]
    fn qwen_marker_prevents_text_fallback_when_parser_fails() {
        let raw_text =
            "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"query\":\"rust\"}}</tool_call>";
        let parsed = ParsedAssistant::content(raw_text);
        let policy = ToolMarkupPolicy::new(&JSON_TOOL_MARKERS);

        let result =
            classify_chat_no_progress(raw_text, &parsed, 100, true, &minimal_request(), policy);

        assert_ne!(result, Some(NoProgressClass::TextFallbackRequiredTool));
    }
}
