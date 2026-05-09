use super::{
    BackendError, BackendModelMetadata, BackendOutput, BackendRequest, BackendToolChoice,
    ModelBackend,
};
use async_trait::async_trait;
use llm_api::{FinishReason, ToolDefinition};
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone)]
pub struct ProtocolTestBackend {
    model_id: String,
    text: String,
    required_tool_protocol: bool,
    json_object_protocol: bool,
}

impl ProtocolTestBackend {
    pub fn new(model_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            text: text.into(),
            required_tool_protocol: false,
            json_object_protocol: false,
        }
    }

    pub fn with_required_tool_protocol(mut self) -> Self {
        self.required_tool_protocol = true;
        self
    }

    pub fn with_json_object_protocol(mut self) -> Self {
        self.json_object_protocol = true;
        self
    }
}

#[async_trait]
impl ModelBackend for ProtocolTestBackend {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        BackendModelMetadata::new(self.model_id.clone(), "protocol-test").with_family("qwen")
    }

    async fn generate(&self, request: BackendRequest) -> Result<BackendOutput, BackendError> {
        if request.model != self.model_id {
            return Err(BackendError::ModelNotFound {
                requested: request.model,
                available: self.model_id.clone(),
            });
        }
        if !request.sampling.is_greedy() {
            return Err(BackendError::UnsupportedRequest(
                "protocol test backend does not support non-greedy sampling".to_owned(),
            ));
        }
        let (text, finish_reason) = if self.required_tool_protocol
            && let Some((name, arguments)) =
                protocol_test_tool_call(&request.prompt, request.required_tool_choice.as_ref())
        {
            (
                serde_json::json!({
                    "name": name,
                    "arguments": arguments,
                })
                .to_string(),
                FinishReason::ToolCalls,
            )
        } else if self.json_object_protocol && request.json_object_mode {
            (
                serde_json::json!({
                    "response": "ok",
                })
                .to_string(),
                FinishReason::Stop,
            )
        } else {
            (self.text.clone(), FinishReason::Stop)
        };
        let text = if matches!(finish_reason, FinishReason::ToolCalls) {
            format!("<tool_call>{text}</tool_call>")
        } else {
            text
        };
        Ok(BackendOutput {
            completion_tokens: count_tokens(&text),
            text,
            prompt_tokens: count_tokens(&request.prompt),
            finish_reason,
        })
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        if cancellation.is_cancelled() {
            return Err(BackendError::Cancelled);
        }
        self.generate(request).await
    }
}

fn protocol_test_tool_call(
    prompt: &str,
    required_choice: Option<&BackendToolChoice>,
) -> Option<(String, serde_json::Value)> {
    let tools = rendered_tool_definitions(prompt)?;
    let tool = match required_choice {
        Some(BackendToolChoice::RequiredFunction(name)) => {
            tools.iter().find(|tool| tool.function.name == *name)?
        }
        Some(BackendToolChoice::RequiredAny) => select_required_any_tool(prompt, &tools)?,
        None => select_auto_tool(prompt, &tools)?,
    };
    Some((
        tool.function.name.clone(),
        protocol_test_tool_arguments(prompt, tool),
    ))
}

fn rendered_tool_definitions(prompt: &str) -> Option<Vec<ToolDefinition>> {
    const TOOL_PREAMBLE: &str =
        "Tools are available. Return tool invocations inside <tool_call> JSON blocks.\n";
    let (_, rest) = prompt.split_once(TOOL_PREAMBLE)?;
    let (tools_json, _) = rest.split_once("<|im_end|>")?;
    serde_json::from_str(tools_json).ok()
}

fn select_required_any_tool<'a>(
    prompt: &str,
    tools: &'a [ToolDefinition],
) -> Option<&'a ToolDefinition> {
    select_scored_tool(prompt, tools).or_else(|| (tools.len() == 1).then_some(&tools[0]))
}

fn select_auto_tool<'a>(prompt: &str, tools: &'a [ToolDefinition]) -> Option<&'a ToolDefinition> {
    select_scored_tool(prompt, tools)
}

fn select_scored_tool<'a>(prompt: &str, tools: &'a [ToolDefinition]) -> Option<&'a ToolDefinition> {
    let user_terms = lexical_user_terms(prompt);
    if user_terms.is_empty() {
        return None;
    }
    let mut best = None;
    for tool in tools {
        let score = score_tool_match(&user_terms, tool);
        if score == 0 {
            continue;
        }
        if best
            .map(|(_, best_score)| score > best_score)
            .unwrap_or(true)
        {
            best = Some((tool, score));
        }
    }
    best.map(|(tool, _)| tool)
}

fn lexical_user_terms(prompt: &str) -> Vec<String> {
    let user = last_user_message(prompt);
    let mut terms = lexical_terms(&user);
    if argument_tokens(&user)
        .iter()
        .any(|token| token.contains('.') || token.contains('/'))
    {
        push_lexical_term(&mut terms, "file".to_owned());
        push_lexical_term(&mut terms, "path".to_owned());
    }
    terms
}

fn score_tool_match(user_terms: &[String], tool: &ToolDefinition) -> usize {
    let name_score = lexical_terms(&tool.function.name)
        .iter()
        .filter(|term| contains_term(user_terms, term))
        .count()
        * 3;
    let description_score = tool
        .function
        .description
        .as_deref()
        .map(lexical_terms)
        .unwrap_or_default()
        .iter()
        .filter(|term| contains_term(user_terms, term))
        .count();
    let parameter_score = parameter_terms(&tool.function.parameters)
        .iter()
        .filter(|term| contains_term(user_terms, term))
        .count()
        * 2;
    name_score + description_score + parameter_score
}

fn contains_term(terms: &[String], needle: &str) -> bool {
    terms.iter().any(|term| term == needle)
}

fn parameter_terms(parameters: &serde_json::Value) -> Vec<String> {
    let mut terms = Vec::new();
    for name in required_parameter_names(parameters) {
        terms.extend(lexical_terms(&name));
    }
    if let Some(properties) = parameters
        .get("properties")
        .and_then(serde_json::Value::as_object)
    {
        for name in properties.keys() {
            terms.extend(lexical_terms(name));
        }
    }
    terms
}

fn lexical_terms(text: &str) -> Vec<String> {
    let mut terms = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            current.push(ch.to_ascii_lowercase());
        } else if !current.is_empty() {
            push_lexical_term(&mut terms, std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        push_lexical_term(&mut terms, current);
    }
    terms
}

fn push_lexical_term(terms: &mut Vec<String>, term: String) {
    if term.len() > 1 && !is_lexical_stop_word(&term) && !terms.contains(&term) {
        terms.push(term);
    }
}

fn is_lexical_stop_word(term: &str) -> bool {
    matches!(
        term,
        "a" | "an"
            | "and"
            | "answer"
            | "after"
            | "before"
            | "by"
            | "for"
            | "from"
            | "in"
            | "of"
            | "on"
            | "or"
            | "please"
            | "the"
            | "to"
            | "use"
            | "using"
            | "with"
    )
}

fn protocol_test_tool_arguments(prompt: &str, tool: &ToolDefinition) -> serde_json::Value {
    let user = last_user_message(prompt);
    let mut arguments = serde_json::Map::new();
    for name in required_parameter_names(&tool.function.parameters) {
        if let Some(value) = argument_value_for_parameter(&user, &name) {
            arguments.insert(name, serde_json::Value::String(value));
        }
    }
    serde_json::Value::Object(arguments)
}

fn required_parameter_names(parameters: &serde_json::Value) -> Vec<String> {
    parameters
        .get("required")
        .and_then(serde_json::Value::as_array)
        .map(|required| {
            required
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn argument_value_for_parameter(user: &str, name: &str) -> Option<String> {
    let lower = name.to_ascii_lowercase();
    if lower.contains("path") || lower.contains("file") {
        return extract_file_argument(user);
    }
    if lower == "key" || lower.contains("query") || lower.contains("value") {
        return extract_lookup_argument(user, &lower);
    }
    extract_lookup_argument(user, &lower)
}

fn extract_file_argument(text: &str) -> Option<String> {
    argument_tokens(text)
        .into_iter()
        .find(|token| token.contains('.') || token.contains('/'))
}

fn extract_lookup_argument(text: &str, parameter: &str) -> Option<String> {
    let tokens = argument_tokens(text);
    for (index, token) in tokens.iter().enumerate() {
        let token_lower = token.to_ascii_lowercase();
        if (matches!(token_lower.as_str(), "key" | "query" | "value") || token_lower == parameter)
            && let Some(next) = tokens.get(index + 1)
            && !is_argument_stop_word(next)
        {
            return Some(next.clone());
        }
    }
    tokens
        .into_iter()
        .rev()
        .find(|token| !is_argument_stop_word(token) && !token.contains('_'))
}

fn argument_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.' | '/') {
            current.push(ch);
        } else if !current.is_empty() {
            push_argument_token(&mut tokens, std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        push_argument_token(&mut tokens, current);
    }
    tokens
}

fn push_argument_token(tokens: &mut Vec<String>, token: String) {
    let token = token.trim_end_matches('.').to_owned();
    if !token.is_empty() {
        tokens.push(token);
    }
}

fn is_argument_stop_word(token: &str) -> bool {
    matches!(
        token.to_ascii_lowercase().as_str(),
        "a" | "an" | "and" | "answer" | "after" | "for" | "the" | "to" | "use" | "with"
    )
}

fn last_user_message(prompt: &str) -> String {
    const USER_START: &str = "<|im_start|>user\n";
    let Some(start) = prompt.rfind(USER_START) else {
        return prompt.to_owned();
    };
    let body_start = start + USER_START.len();
    let Some(end_rel) = prompt[body_start..].find("<|im_end|>") else {
        return prompt[body_start..].to_owned();
    };
    prompt[body_start..body_start + end_rel].to_owned()
}

fn count_tokens(text: &str) -> u64 {
    let normalized = text
        .replace("<|im_start|>system", " ")
        .replace("<|im_start|>user", " ")
        .replace("<|im_start|>assistant", " ")
        .replace("<|im_start|>tool", " ")
        .replace("<|im_end|>", " ")
        .replace("<think>", " ")
        .replace("</think>", " ");
    normalized.split_whitespace().count().max(1) as u64
}
