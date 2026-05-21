use llm_api::{
    ChatCompletionRequest, ChatMessage, ChatRole,
    NO_PROGRESS_EXACT_REPEATED_INVALID_TOOL_CALL_THRESHOLD,
    NO_PROGRESS_FUZZY_REPEATED_INVALID_TOOL_CALL_THRESHOLD, ToolCall,
};
use llm_tool_parser::ParsedAssistant;
use serde_json::Value;
use std::collections::BTreeSet;

use crate::adapters::ToolMarkupPolicy;
use crate::response_validation::schema_requires_string_intent_argument;

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
        if stalled_assistant_turn(&parsed.content, request) {
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

        if exact_count >= NO_PROGRESS_EXACT_REPEATED_INVALID_TOOL_CALL_THRESHOLD {
            return Some(NoProgressClass::RepeatedInvalidToolCall);
        }
        if fuzzy_count >= NO_PROGRESS_FUZZY_REPEATED_INVALID_TOOL_CALL_THRESHOLD {
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
        "missing required argument",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

fn stalled_assistant_turn(content: &str, request: &ChatCompletionRequest) -> bool {
    let terms = normalized_progress_terms(content);
    if terms.is_empty() || terms.len() > 6 || !plain_short_turn_shape(content) {
        return false;
    }

    let Some(user_content) = latest_user_content(request) else {
        return false;
    };
    let user_terms = normalized_progress_terms(user_content);
    if meaningful_progress_term_count(&user_terms) < 3 {
        return false;
    }

    has_deferred_action_stance(&terms, content)
}

fn normalized_progress_text(content: &str) -> String {
    normalized_progress_terms(content).join(" ")
}

fn normalized_progress_terms(content: &str) -> Vec<String> {
    let mut terms = Vec::new();
    let mut current = String::new();

    for ch in content.chars() {
        if no_space_progress_char(ch) {
            if !current.is_empty() {
                terms.push(std::mem::take(&mut current));
            }
            terms.push(ch.to_string());
        } else if ch.is_alphanumeric() {
            current.extend(ch.to_lowercase());
        } else if !current.is_empty() {
            terms.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        terms.push(current);
    }

    terms
}

fn no_space_progress_char(ch: char) -> bool {
    matches!(
        ch,
        '\u{3400}'..='\u{4DBF}'
            | '\u{4E00}'..='\u{9FFF}'
            | '\u{F900}'..='\u{FAFF}'
            | '\u{3040}'..='\u{309F}'
            | '\u{30A0}'..='\u{30FF}'
            | '\u{AC00}'..='\u{D7AF}'
    )
}

fn plain_short_turn_shape(content: &str) -> bool {
    let trimmed = content.trim();
    if trimmed.is_empty() || trimmed.chars().count() > 96 || trimmed.ends_with('?') {
        return false;
    }
    !trimmed.chars().any(|ch| {
        ch.is_ascii_digit()
            || matches!(
                ch,
                '\n' | '\r' | '`' | '{' | '}' | '[' | ']' | '<' | '>' | '|' | '=' | ':' | ';'
            )
    })
}

fn latest_user_content(request: &ChatCompletionRequest) -> Option<&str> {
    request
        .messages
        .iter()
        .rev()
        .find(|message| message.role == ChatRole::User)
        .and_then(|message| message.content.as_deref())
}

// A stalled turn is a short promise to act later, not merely a terse answer with little
// prompt overlap. Keep this signal stance-based so answers like `This is blue.` pass.
fn has_deferred_action_stance(terms: &[String], content: &str) -> bool {
    has_english_deferred_action_stance(terms)
        || has_spanish_deferred_action_stance(terms)
        || has_cyrillic_deferred_action_stance(terms)
        || has_no_space_deferred_action_stance(content)
}

fn has_english_deferred_action_stance(terms: &[String]) -> bool {
    let has_first_person = terms
        .iter()
        .any(|term| matches!(term.as_str(), "i" | "me" | "my"));
    let has_future_or_modal = terms
        .iter()
        .any(|term| matches!(term.as_str(), "will" | "ll" | "can"));
    let has_action = terms.iter().any(|term| english_deferred_action_term(term));
    let has_working_on_target = terms.windows(3).any(|window| {
        window[0] == "working"
            && window[1] == "on"
            && matches!(window[2].as_str(), "it" | "this" | "that")
    });

    has_working_on_target
        || terms
            .windows(2)
            .any(|window| window[0] == "let" && window[1] == "me")
            && has_action
        || has_first_person && has_future_or_modal && has_action
}

fn english_deferred_action_term(term: &str) -> bool {
    matches!(
        term,
        "check"
            | "checking"
            | "get"
            | "handle"
            | "help"
            | "investigate"
            | "look"
            | "proceed"
            | "start"
            | "started"
            | "starting"
            | "work"
            | "working"
    )
}

fn has_spanish_deferred_action_stance(terms: &[String]) -> bool {
    let has_future_or_modal = terms
        .iter()
        .any(|term| matches!(term.as_str(), "voy" | "puedo"));
    has_future_or_modal && terms.iter().any(|term| spanish_deferred_action_term(term))
}

fn spanish_deferred_action_term(term: &str) -> bool {
    matches!(
        term,
        "ayudar" | "comenzar" | "empezar" | "mirar" | "revisar" | "trabajar" | "verificar"
    )
}

fn has_cyrillic_deferred_action_stance(terms: &[String]) -> bool {
    let has_future_or_modal = terms
        .iter()
        .any(|term| matches!(term.as_str(), "буду" | "будем" | "могу" | "можем"));

    terms
        .iter()
        .any(|term| cyrillic_first_person_future_action_term(term))
        || has_future_or_modal && terms.iter().any(|term| cyrillic_deferred_action_term(term))
}

fn cyrillic_first_person_future_action_term(term: &str) -> bool {
    matches!(
        term,
        "займусь" | "начну" | "помогу" | "посмотрю" | "проверю" | "сделаю"
    )
}

fn cyrillic_deferred_action_term(term: &str) -> bool {
    matches!(
        term,
        "начать"
            | "начинать"
            | "помогать"
            | "помочь"
            | "посмотреть"
            | "проверить"
            | "проверять"
            | "работать"
            | "сделать"
            | "смотреть"
    )
}

fn has_no_space_deferred_action_stance(content: &str) -> bool {
    let compact: String = content
        .chars()
        .filter(|ch| ch.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect();

    has_no_space_first_person_future_stance(&compact) && has_no_space_deferred_action(&compact)
}

fn has_no_space_first_person_future_stance(compact: &str) -> bool {
    ["我会", "我将", "我要", "我来"]
        .iter()
        .any(|marker| compact.contains(marker))
}

fn has_no_space_deferred_action(compact: &str) -> bool {
    ["开始", "处理", "检查", "修复", "着手"]
        .iter()
        .any(|marker| compact.contains(marker))
}

fn meaningful_progress_term_count(terms: &[String]) -> usize {
    terms
        .iter()
        .filter(|term| meaningful_progress_term(term))
        .count()
}

fn meaningful_progress_term(term: &str) -> bool {
    term.chars().count() >= 3 || term.chars().next().is_some_and(no_space_progress_char)
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

    #[test]
    fn stalled_assistant_turn_uses_language_neutral_shape() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Apply the requested code fix now."}],
            "max_tokens": 64
        }))
        .unwrap();

        assert!(stalled_assistant_turn("I will get started.", &request));
        assert!(stalled_assistant_turn("Working on it.", &request));
        assert!(stalled_assistant_turn("Voy a empezar.", &request));
    }

    #[test]
    fn stalled_assistant_turn_ignores_short_content_with_request_terms() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Should I use cargo fmt?"}],
            "max_tokens": 64
        }))
        .unwrap();

        assert!(!stalled_assistant_turn("Use cargo fmt.", &request));
    }

    #[test]
    fn stalled_assistant_turn_ignores_short_answers_to_short_prompts() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "What color?"}],
            "max_tokens": 64
        }))
        .unwrap();

        assert!(!stalled_assistant_turn("It is blue.", &request));
    }

    #[test]
    fn stalled_assistant_turn_allows_short_answer_with_new_answer_term() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "What is the color?"}],
            "max_tokens": 64
        }))
        .unwrap();

        assert!(!stalled_assistant_turn("It is blue.", &request));
    }

    #[test]
    fn stalled_assistant_turn_allows_short_answer_to_command_prompt() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Translate bonjour to English."}],
            "max_tokens": 64
        }))
        .unwrap();

        assert!(!stalled_assistant_turn("It means hello.", &request));
    }

    #[test]
    fn stalled_assistant_turn_allows_bare_working_translation_answer() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Translate \"trabajando\" to English."}],
            "max_tokens": 64
        }))
        .unwrap();

        assert!(!stalled_assistant_turn("Working.", &request));
    }

    #[test]
    fn stalled_assistant_turn_allows_non_first_person_spanish_modal_answer() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Translate \"can help\" to Spanish."}],
            "max_tokens": 64
        }))
        .unwrap();

        assert!(!stalled_assistant_turn("Puede ayudar.", &request));
    }

    #[test]
    fn stalled_assistant_turn_allows_terse_assertive_answers() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "What is the color?"}],
            "max_tokens": 64
        }))
        .unwrap();

        assert!(!stalled_assistant_turn("This is blue.", &request));

        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Translate bonjour to English."}],
            "max_tokens": 64
        }))
        .unwrap();

        assert!(!stalled_assistant_turn("Bonjour means hello.", &request));
    }

    #[test]
    fn stalled_assistant_turn_allows_unicode_terse_substantive_answers() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Translate start to Chinese."}],
            "max_tokens": 64
        }))
        .unwrap();

        assert!(!stalled_assistant_turn("开始。", &request));
        assert!(!stalled_assistant_turn("着手。", &request));

        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "What is the Russian word for beginning?"}],
            "max_tokens": 64
        }))
        .unwrap();

        assert!(!stalled_assistant_turn("Начало.", &request));
        assert!(!stalled_assistant_turn("Начало работы.", &request));
    }

    #[test]
    fn stalled_assistant_turn_detects_unicode_stalls_without_three_whitespace_words() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Проверь падение теста сейчас."}],
            "max_tokens": 64
        }))
        .unwrap();

        assert!(stalled_assistant_turn("Начну проверку.", &request));

        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "请现在修复这个测试失败。"}],
            "max_tokens": 64
        }))
        .unwrap();

        assert!(stalled_assistant_turn("我会开始。", &request));
    }

    #[test]
    fn stalled_assistant_turn_preserves_commitments_that_share_prompt_verbs() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Check the failing test now."}],
            "max_tokens": 64
        }))
        .unwrap();

        assert!(stalled_assistant_turn("I will check.", &request));

        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Look at the parser failure now."}],
            "max_tokens": 64
        }))
        .unwrap();

        assert!(stalled_assistant_turn("Let me look.", &request));

        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Help with the parser failure now."}],
            "max_tokens": 64
        }))
        .unwrap();

        assert!(stalled_assistant_turn("I can help with that.", &request));
    }

    #[test]
    fn stalled_assistant_turn_preserves_get_started_commitment_with_prompt_overlap() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Please get started on the code fix now."}],
            "max_tokens": 64
        }))
        .unwrap();

        assert!(stalled_assistant_turn("I will get started.", &request));
    }

    #[test]
    fn stalled_assistant_turn_ignores_structured_content() {
        let request = minimal_request();

        assert!(!stalled_assistant_turn("status: done", &request));
        assert!(!stalled_assistant_turn("`cargo fmt`", &request));
    }

    #[test]
    fn normalized_progress_text_preserves_unicode_terms() {
        assert_eq!(normalized_progress_text("Voy a empezar."), "voy a empezar");
        assert_eq!(
            normalized_progress_text("Revisare la correccion."),
            "revisare la correccion"
        );
        assert_eq!(
            normalized_progress_text("Начну проверку."),
            "начну проверку"
        );
    }
}
