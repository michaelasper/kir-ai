use super::{
    BackendChatRole, BackendError, BackendFinishReason, BackendModelMetadata, BackendOutput,
    BackendRequest, BackendToolChoice, BackendToolDefinition, ModelBackend,
};
use async_trait::async_trait;
use llm_models::ModelFamily;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone)]
pub struct ProtocolTestBackend {
    model_id: String,
    text: String,
    family: ModelFamily,
    required_tool_protocol: bool,
    json_object_protocol: bool,
}

impl ProtocolTestBackend {
    pub fn new(model_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            text: text.into(),
            family: ModelFamily::Qwen,
            required_tool_protocol: false,
            json_object_protocol: false,
        }
    }

    pub fn with_family(mut self, family: ModelFamily) -> Self {
        self.family = family;
        self
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
        BackendModelMetadata::new(self.model_id.clone(), "protocol-test")
            .with_family(self.family.canonical_slug())
    }

    async fn generate(&self, request: BackendRequest) -> Result<BackendOutput, BackendError> {
        if request.model != self.model_id {
            return Err(BackendError::model_not_found(
                request.model,
                self.model_id.clone(),
            ));
        }
        if !request.sampling.is_greedy() && !request.sampling.is_standard() {
            return Err(BackendError::unsupported_request(
                "protocol test backend does not support non-greedy sampling".to_owned(),
            ));
        }
        let (text, finish_reason) = if self.required_tool_protocol
            && let Some((name, arguments)) = protocol_test_tool_call(self.family, &request)
        {
            (
                render_tool_call(self.family, &name, &arguments),
                BackendFinishReason::ToolCalls,
            )
        } else if self.json_object_protocol
            && request.as_chat().is_some_and(|chat| chat.json_object_mode)
        {
            (
                serde_json::json!({
                    "response": "ok",
                })
                .to_string(),
                BackendFinishReason::Stop,
            )
        } else {
            (self.text.clone(), BackendFinishReason::Stop)
        };
        Ok(BackendOutput {
            completion_tokens: count_tokens(self.family, &text),
            text,
            prompt_tokens: count_tokens(self.family, request.prompt()),
            prompt_cached_tokens: None,
            finish_reason,
        })
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        if cancellation.is_cancelled() {
            return Err(BackendError::cancelled());
        }
        self.generate(request).await
    }
}

fn protocol_test_tool_call(
    family: ModelFamily,
    request: &BackendRequest,
) -> Option<(String, serde_json::Value)> {
    let tools = request_tool_definitions(family, request)?;
    let user = request_last_user_message(family, request);
    protocol_test_tool_call_from_tools(
        &user,
        &tools,
        request
            .as_chat()
            .and_then(|chat| chat.required_tool_choice.as_ref()),
    )
}

fn request_tool_definitions(
    family: ModelFamily,
    request: &BackendRequest,
) -> Option<Vec<BackendToolDefinition>> {
    request
        .as_chat()
        .map(|chat| chat.chat_context.tools.clone())
        .filter(|tools| !tools.is_empty())
        .or_else(|| rendered_tool_definitions(family, request.prompt()))
}

fn request_last_user_message(family: ModelFamily, request: &BackendRequest) -> String {
    if let Some(content) = request
        .as_chat()
        .and_then(|context| {
            context
                .chat_context
                .messages
                .iter()
                .rev()
                .find(|message| message.role == BackendChatRole::User)
        })
        .and_then(|message| message.content.as_deref())
    {
        return content.to_owned();
    }
    last_user_message(family, request.prompt())
}

fn protocol_test_tool_call_from_tools(
    user: &str,
    tools: &[BackendToolDefinition],
    required_choice: Option<&BackendToolChoice>,
) -> Option<(String, serde_json::Value)> {
    let tool = match required_choice {
        Some(BackendToolChoice::RequiredFunction(name)) => {
            tools.iter().find(|tool| tool.function.name == *name)?
        }
        Some(BackendToolChoice::RequiredAny) => select_required_any_tool(user, tools)?,
        None => select_auto_tool(user, tools)?,
    };
    Some((
        tool.function.name.clone(),
        protocol_test_tool_arguments(user, tool),
    ))
}

fn rendered_tool_definitions(
    family: ModelFamily,
    prompt: &str,
) -> Option<Vec<BackendToolDefinition>> {
    match family {
        ModelFamily::Qwen => rendered_tool_definitions_between(
            prompt,
            "Tools are available. Return tool invocations inside <tool_call> JSON blocks.\n",
            "<|im_end|>",
        ),
        ModelFamily::DeepSeek => rendered_tool_definitions_between(
            prompt,
            "You may call tools by emitting DeepSeek tool call blocks with exact tool names.\n",
            "\n\n",
        ),
        ModelFamily::Llama => rendered_tool_definitions_between(
            prompt,
            concat!(
                "Tools are available. To call a function, respond with JSON in the form ",
                r#"{"name":"function_name","arguments":{"argument":"value"}}"#,
                ". Do not use variables.\n"
            ),
            "<|eot_id|>",
        ),
        ModelFamily::Gemma => None,
    }
}

fn rendered_tool_definitions_between(
    prompt: &str,
    preamble: &str,
    terminator: &str,
) -> Option<Vec<BackendToolDefinition>> {
    let (_, rest) = prompt.split_once(preamble)?;
    let (tools_json, _) = rest.split_once(terminator)?;
    serde_json::from_str(tools_json).ok()
}

fn select_required_any_tool<'a>(
    prompt: &str,
    tools: &'a [BackendToolDefinition],
) -> Option<&'a BackendToolDefinition> {
    select_scored_tool(prompt, tools).or_else(|| (tools.len() == 1).then_some(&tools[0]))
}

fn select_auto_tool<'a>(
    prompt: &str,
    tools: &'a [BackendToolDefinition],
) -> Option<&'a BackendToolDefinition> {
    select_scored_tool(prompt, tools)
}

fn select_scored_tool<'a>(
    prompt: &str,
    tools: &'a [BackendToolDefinition],
) -> Option<&'a BackendToolDefinition> {
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
    let mut terms = lexical_terms(prompt);
    if argument_tokens(prompt)
        .iter()
        .any(|token| token.contains('.') || token.contains('/'))
    {
        push_lexical_term(&mut terms, "file".to_owned());
        push_lexical_term(&mut terms, "path".to_owned());
    }
    terms
}

fn score_tool_match(user_terms: &[String], tool: &BackendToolDefinition) -> usize {
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

fn protocol_test_tool_arguments(user: &str, tool: &BackendToolDefinition) -> serde_json::Value {
    let mut arguments = serde_json::Map::new();
    for name in required_parameter_names(&tool.function.parameters) {
        if let Some(value) = argument_value_for_parameter(user, &name) {
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

fn render_tool_call(family: ModelFamily, name: &str, arguments: &serde_json::Value) -> String {
    match family {
        ModelFamily::Qwen => format!(
            "<tool_call>{}</tool_call>",
            serde_json::json!({
                "name": name,
                "arguments": arguments,
            })
        ),
        ModelFamily::DeepSeek => format!(
            concat!(
                "<｜tool▁calls▁begin｜>",
                "<｜tool▁call▁begin｜>function<｜tool▁sep｜>{name}\n",
                "```json\n{arguments}\n```",
                "<｜tool▁call▁end｜>",
                "<｜tool▁calls▁end｜>",
                "<｜end▁of▁sentence｜>"
            ),
            name = name,
            arguments = arguments,
        ),
        ModelFamily::Gemma => format!(
            "<|tool_call>call:{name}{}<tool_call|>",
            render_gemma_argument(arguments)
        ),
        ModelFamily::Llama => serde_json::json!({
            "name": name,
            "arguments": arguments,
        })
        .to_string(),
    }
}

fn render_gemma_argument(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "null".to_owned(),
        serde_json::Value::Bool(value) => value.to_string(),
        serde_json::Value::Number(value) => value.to_string(),
        serde_json::Value::String(value) => render_gemma_string(value),
        serde_json::Value::Array(items) => {
            let inner = items
                .iter()
                .map(render_gemma_argument)
                .collect::<Vec<_>>()
                .join(",");
            format!("[{inner}]")
        }
        serde_json::Value::Object(map) => {
            let inner = map
                .iter()
                .map(|(key, value)| format!("{key}:{}", render_gemma_argument(value)))
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{inner}}}")
        }
    }
}

fn render_gemma_string(value: &str) -> String {
    format!("<|\"|>{}<|\"|>", value.replace("<|\"|>", ""))
}

fn last_user_message(family: ModelFamily, prompt: &str) -> String {
    match family {
        ModelFamily::Qwen => last_prompt_body(prompt, "<|im_start|>user\n", &["<|im_end|>"]),
        ModelFamily::Gemma => last_prompt_body(prompt, "<|turn>user\n", &["<turn|>"]),
        ModelFamily::DeepSeek => last_prompt_body(
            prompt,
            "<｜User｜>",
            &["<｜Assistant｜>", "<｜tool▁outputs▁begin｜>"],
        ),
        ModelFamily::Llama => last_prompt_body(
            prompt,
            "<|start_header_id|>user<|end_header_id|>\n\n",
            &["<|eot_id|>"],
        ),
    }
}

fn last_prompt_body(prompt: &str, start_marker: &str, end_markers: &[&str]) -> String {
    let Some(start) = prompt.rfind(start_marker) else {
        return prompt.to_owned();
    };
    let body_start = start + start_marker.len();
    let rest = &prompt[body_start..];
    let body_end = end_markers
        .iter()
        .filter_map(|marker| rest.find(marker))
        .min()
        .unwrap_or(rest.len());
    rest[..body_end].to_owned()
}

fn count_tokens(family: ModelFamily, text: &str) -> u64 {
    let mut normalized = text.to_owned();
    for token in prompt_control_tokens(family) {
        normalized = normalized.replace(token, " ");
    }
    normalized.split_whitespace().count().max(1) as u64
}

fn prompt_control_tokens(family: ModelFamily) -> &'static [&'static str] {
    match family {
        ModelFamily::Qwen => &[
            "<|im_start|>system",
            "<|im_start|>user",
            "<|im_start|>assistant",
            "<|im_start|>tool",
            "<|im_end|>",
            "<tool_call>",
            "</tool_call>",
            "<think>",
            "</think>",
        ],
        ModelFamily::DeepSeek => &[
            "<｜begin▁of▁sentence｜>",
            "<｜end▁of▁sentence｜>",
            "<｜User｜>",
            "<｜Assistant｜>",
            "<｜tool▁calls▁begin｜>",
            "<｜tool▁calls▁end｜>",
            "<｜tool▁call▁begin｜>",
            "<｜tool▁call▁end｜>",
            "<｜tool▁sep｜>",
            "<｜tool▁outputs▁begin｜>",
            "<｜tool▁outputs▁end｜>",
            "<｜tool▁output▁begin｜>",
            "<｜tool▁output▁end｜>",
            "<think>",
            "</think>",
        ],
        ModelFamily::Gemma => &[
            "<bos>",
            "<|turn>system",
            "<|turn>user",
            "<|turn>model",
            "<turn|>",
            "<|channel>thought",
            "<channel|>",
            "<|tool_call>",
            "<tool_call|>",
            "<|tool_response>",
            "<tool_response|>",
            "<|tool>",
            "<tool|>",
            "<|think|>",
        ],
        ModelFamily::Llama => &[
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|start_header_id|>system<|end_header_id|>",
            "<|start_header_id|>user<|end_header_id|>",
            "<|start_header_id|>assistant<|end_header_id|>",
            "<|start_header_id|>ipython<|end_header_id|>",
            "<|eot_id|>",
            "<|eom_id|>",
        ],
    }
}
