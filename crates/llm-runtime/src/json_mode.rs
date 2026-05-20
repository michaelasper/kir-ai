use crate::RuntimeError;
use crate::adapters::{ChatAdapter, SelectedChatAdapter};
use llm_api::{ChatCompletionRequest, ResponseFormat};
use llm_tool_parser::ParsedAssistant;

pub(crate) fn parse_chat_text(
    adapter: SelectedChatAdapter,
    text: &str,
    request: &ChatCompletionRequest,
) -> Result<ParsedAssistant, RuntimeError> {
    if let Some(content) = unmarked_tool_json_without_declared_tools(request, text, adapter) {
        return Ok(ParsedAssistant::content(content));
    }
    if let Some(content) = json_object_mode_without_tools(request, text, adapter) {
        return Ok(ParsedAssistant::content(content));
    }
    adapter.parse_complete(text)
}

fn unmarked_tool_json_without_declared_tools(
    request: &ChatCompletionRequest,
    text: &str,
    adapter: SelectedChatAdapter,
) -> Option<String> {
    if !adapter.parses_unmarked_tool_calls()
        || !request.tools.is_empty()
        || adapter.tool_markup_policy().contains_start(text)
    {
        return None;
    }
    let content =
        unmarked_tool_json_candidate(text, adapter.unmarked_tool_json_truncation_tokens());
    serde_json::from_str::<serde_json::Value>(content)
        .is_ok_and(|value| value.is_object() || value.is_array())
        .then(|| content.to_owned())
}

fn json_object_mode_without_tools(
    request: &ChatCompletionRequest,
    text: &str,
    adapter: SelectedChatAdapter,
) -> Option<String> {
    if !matches!(request.response_format, Some(ResponseFormat::JsonObject))
        || !request.tools.is_empty()
        || adapter.tool_markup_policy().contains_start(text)
    {
        return None;
    }
    let content =
        unmarked_tool_json_candidate(text, adapter.unmarked_tool_json_truncation_tokens());
    json_object_response_candidate(content).map(str::to_owned)
}

fn json_object_response_candidate(text: &str) -> Option<&str> {
    let trimmed = text.trim();
    if json_value_is_object(trimmed) {
        return Some(trimmed);
    }
    if let Some(fenced) = markdown_fenced_json_object_candidate(trimmed) {
        return Some(fenced);
    }
    if !trimmed.starts_with('{')
        && let Some(candidate) = first_balanced_json_object(trimmed)
        && json_value_is_object(candidate)
    {
        return Some(candidate);
    }
    None
}

fn markdown_fenced_json_object_candidate(text: &str) -> Option<&str> {
    let rest = text.strip_prefix("```")?;
    let body_start = rest.find('\n')? + 1;
    let body_with_close = &rest[body_start..];
    let body_end = body_with_close.find("```")?;
    let candidate = body_with_close[..body_end].trim();
    json_value_is_object(candidate).then_some(candidate)
}

fn first_balanced_json_object(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let mut depth = 0_usize;
    let mut in_string = false;
    let mut escaped = false;
    for (offset, ch) in text[start..].char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    let end = start + offset + ch.len_utf8();
                    return Some(text[start..end].trim());
                }
            }
            _ => {}
        }
    }
    None
}

fn json_value_is_object(text: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(text).is_ok_and(|value| value.is_object())
}

fn unmarked_tool_json_candidate<'a>(
    text: &'a str,
    truncation_tokens: &'static [&'static str],
) -> &'a str {
    truncation_tokens
        .iter()
        .filter_map(|token| text.find(token))
        .min()
        .map_or(text, |index| &text[..index])
        .trim()
}
