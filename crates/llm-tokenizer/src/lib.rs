use llm_api::{ChatMessage, ChatRole, ToolDefinition};
use llm_models::ModelFamily;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct QwenPromptOptions {
    pub enable_thinking: bool,
    pub add_generation_prompt: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GemmaPromptOptions {
    pub enable_thinking: bool,
    pub add_generation_prompt: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeepSeekPromptOptions {
    pub enable_thinking: bool,
    pub add_generation_prompt: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlamaPromptOptions {
    pub add_generation_prompt: bool,
}

#[derive(Debug, Error)]
pub enum TemplateError {
    #[error("tool serialization failed: {0}")]
    ToolSerialization(#[from] serde_json::Error),
    #[error("reserved prompt control token `{0}` is not allowed in request text")]
    ReservedControlToken(&'static str),
    #[error("message role `{0}` cannot be rendered in chat template")]
    UnsupportedRole(String),
    #[error("{0} chat template support is deferred until a production template is implemented")]
    UnsupportedFamily(&'static str),
}

impl TemplateError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::ToolSerialization(_) => "tool_serialization_failed",
            Self::ReservedControlToken(_) => "reserved_prompt_control_token",
            Self::UnsupportedRole(_) => "unsupported_role",
            Self::UnsupportedFamily(_) => "unsupported_template_family",
        }
    }
}

#[derive(Clone)]
pub struct HuggingFaceTokenizer {
    inner: Tokenizer,
}

impl HuggingFaceTokenizer {
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, TokenizerError> {
        let inner = Tokenizer::from_file(path.as_ref())
            .map_err(|err| TokenizerError::Load(err.to_string()))?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, TokenizerError> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|err| TokenizerError::Encode(err.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, TokenizerError> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|err| TokenizerError::Decode(err.to_string()))
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
}

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("failed to load tokenizer: {0}")]
    Load(String),
    #[error("failed to encode text: {0}")]
    Encode(String),
    #[error("failed to decode tokens: {0}")]
    Decode(String),
}

pub fn render_qwen_chatml(
    messages: &[ChatMessage],
    tools: &[ToolDefinition],
    options: &QwenPromptOptions,
) -> Result<String, TemplateError> {
    let mut out = String::new();
    if !tools.is_empty() {
        let tools_json = serde_json::to_string(tools)?;
        reject_reserved_prompt_controls(&tools_json)?;
        out.push_str("<|im_start|>system\n");
        out.push_str(
            "Tools are available. Return tool invocations inside <tool_call> JSON blocks.\n",
        );
        out.push_str(&tools_json);
        out.push_str("<|im_end|>\n");
    }

    for message in messages {
        match message.role {
            ChatRole::System => render_plain(&mut out, "system", message)?,
            ChatRole::User => render_plain(&mut out, "user", message)?,
            ChatRole::Tool => render_plain(&mut out, "tool", message)?,
            ChatRole::Assistant => render_assistant(&mut out, message)?,
        }
    }

    if options.add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
        if !options.enable_thinking {
            out.push_str("<think>\n\n</think>\n\n");
        } else {
            out.push_str("<think>\n");
        }
    }

    Ok(out)
}

pub fn render_gemma4_chat_template(
    messages: &[ChatMessage],
    tools: &[ToolDefinition],
    options: &GemmaPromptOptions,
) -> Result<String, TemplateError> {
    let mut out = String::from("<bos>");
    let mut tool_names_by_id = std::collections::BTreeMap::new();
    for message in messages {
        for call in &message.tool_calls {
            tool_names_by_id.insert(call.id.as_str(), call.function.name.as_str());
        }
    }

    let mut message_index = 0;
    if options.enable_thinking
        || !tools.is_empty()
        || matches!(
            messages.first().map(|message| &message.role),
            Some(ChatRole::System)
        )
    {
        out.push_str("<|turn>system\n");
        if options.enable_thinking {
            out.push_str("<|think|>\n");
        }
        if let Some(message) = messages
            .first()
            .filter(|message| message.role == ChatRole::System)
        {
            if let Some(content) = &message.content {
                reject_gemma4_prompt_controls(content)?;
                out.push_str(content.trim());
            }
            message_index = 1;
        }
        for tool in tools {
            let rendered = render_gemma4_tool_definition(tool)?;
            reject_gemma4_prompt_controls(&rendered)?;
            out.push_str("<|tool>");
            out.push_str(&rendered);
            out.push_str("<tool|>");
        }
        out.push_str("<turn|>\n");
    }

    for message in &messages[message_index..] {
        match message.role {
            ChatRole::System => render_gemma4_turn(&mut out, "system", message)?,
            ChatRole::User => render_gemma4_turn(&mut out, "user", message)?,
            ChatRole::Assistant => render_gemma4_assistant_turn(&mut out, message)?,
            ChatRole::Tool => render_gemma4_tool_response(&mut out, message, &tool_names_by_id)?,
        }
    }

    if options.add_generation_prompt {
        out.push_str("<|turn>model\n");
        if !options.enable_thinking {
            out.push_str("<|channel>thought\n<channel|>");
        }
    }

    Ok(out)
}

pub fn render_deepseek_chat_template(
    messages: &[ChatMessage],
    tools: &[ToolDefinition],
    options: &DeepSeekPromptOptions,
) -> Result<String, TemplateError> {
    let mut out = String::from("<｜begin▁of▁sentence｜>");
    let mut tool_output_open = false;

    for system in messages
        .iter()
        .filter(|message| message.role == ChatRole::System)
        .filter_map(|message| message.content.as_deref())
    {
        reject_deepseek_prompt_controls(system)?;
        out.push_str(system);
        out.push_str("\n\n");
    }

    if !tools.is_empty() {
        let tools_json = serde_json::to_string(tools)?;
        reject_deepseek_prompt_controls(&tools_json)?;
        out.push_str(
            "You may call tools by emitting DeepSeek tool call blocks with exact tool names.\n",
        );
        out.push_str(&tools_json);
        out.push_str("\n\n");
    }

    for message in messages {
        match message.role {
            ChatRole::System => {}
            ChatRole::User => {
                if tool_output_open {
                    out.push_str("<｜tool▁outputs▁end｜>");
                    tool_output_open = false;
                }
                render_deepseek_plain_turn(&mut out, "<｜User｜>", message)?;
            }
            ChatRole::Assistant => {
                if tool_output_open {
                    out.push_str("<｜tool▁outputs▁end｜>");
                    tool_output_open = false;
                }
                render_deepseek_assistant_turn(&mut out, message)?;
            }
            ChatRole::Tool => {
                render_deepseek_tool_response(&mut out, message, &mut tool_output_open)?;
            }
        }
    }

    if tool_output_open {
        out.push_str("<｜tool▁outputs▁end｜>");
    }
    if options.add_generation_prompt {
        out.push_str("<｜Assistant｜>");
        if options.enable_thinking {
            out.push_str("<think>\n");
        }
    }
    Ok(out)
}

pub fn render_llama3_chat_template(
    messages: &[ChatMessage],
    tools: &[ToolDefinition],
    options: &LlamaPromptOptions,
) -> Result<String, TemplateError> {
    let mut out = String::from("<|begin_of_text|>");
    let mut message_index = 0;

    if matches!(
        messages.first().map(|message| &message.role),
        Some(ChatRole::System)
    ) || !tools.is_empty()
    {
        out.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
        if let Some(message) = messages
            .first()
            .filter(|message| message.role == ChatRole::System)
        {
            if let Some(content) = &message.content {
                reject_llama3_prompt_controls(content)?;
                out.push_str(content.trim());
            }
            message_index = 1;
        }
        if !tools.is_empty() {
            if message_index == 1 {
                out.push_str("\n\n");
            }
            let tools_json = serde_json::to_string(tools)?;
            reject_llama3_prompt_controls(&tools_json)?;
            out.push_str("Tools are available. To call a function, respond with JSON in the form ");
            out.push_str(r#"{"name":"function_name","arguments":{"argument":"value"}}"#);
            out.push_str(". Do not use variables.\n");
            out.push_str(&tools_json);
        }
        out.push_str("<|eot_id|>");
    }

    for message in &messages[message_index..] {
        match message.role {
            ChatRole::System => render_llama3_plain_turn(&mut out, "system", message)?,
            ChatRole::User => render_llama3_plain_turn(&mut out, "user", message)?,
            ChatRole::Assistant => render_llama3_assistant_turn(&mut out, message)?,
            ChatRole::Tool => render_llama3_tool_response(&mut out, message)?,
        }
    }

    if options.add_generation_prompt {
        out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    }

    Ok(out)
}

pub fn render_family_chat_template(
    family: ModelFamily,
    messages: &[ChatMessage],
    tools: &[ToolDefinition],
) -> Result<String, TemplateError> {
    match family {
        ModelFamily::Qwen => render_qwen_chatml(
            messages,
            tools,
            &QwenPromptOptions {
                enable_thinking: false,
                add_generation_prompt: true,
            },
        ),
        ModelFamily::DeepSeek => render_deepseek_chat_template(
            messages,
            tools,
            &DeepSeekPromptOptions {
                enable_thinking: false,
                add_generation_prompt: true,
            },
        ),
        ModelFamily::Gemma => render_gemma4_chat_template(
            messages,
            tools,
            &GemmaPromptOptions {
                enable_thinking: false,
                add_generation_prompt: true,
            },
        ),
        ModelFamily::Llama => render_llama3_chat_template(
            messages,
            tools,
            &LlamaPromptOptions {
                add_generation_prompt: true,
            },
        ),
    }
}

fn render_plain(out: &mut String, role: &str, message: &ChatMessage) -> Result<(), TemplateError> {
    out.push_str("<|im_start|>");
    out.push_str(role);
    out.push('\n');
    if let Some(content) = &message.content {
        reject_reserved_prompt_controls(content)?;
        out.push_str(content);
    }
    out.push_str("<|im_end|>\n");
    Ok(())
}

fn render_assistant(out: &mut String, message: &ChatMessage) -> Result<(), TemplateError> {
    out.push_str("<|im_start|>assistant\n");
    if let Some(content) = &message.content {
        reject_reserved_prompt_controls(content)?;
        out.push_str(content);
    }
    for call in &message.tool_calls {
        let payload = serde_json::json!({
            "name": call.function.name,
            "arguments": call.function.arguments,
        });
        let payload_json = serde_json::to_string(&payload)?;
        reject_reserved_prompt_controls(&payload_json)?;
        out.push_str("<tool_call>");
        out.push_str(&payload_json);
        out.push_str("</tool_call>");
    }
    out.push_str("<|im_end|>\n");
    Ok(())
}

fn render_llama3_plain_turn(
    out: &mut String,
    role: &str,
    message: &ChatMessage,
) -> Result<(), TemplateError> {
    out.push_str("<|start_header_id|>");
    out.push_str(role);
    out.push_str("<|end_header_id|>\n\n");
    if let Some(content) = &message.content {
        reject_llama3_prompt_controls(content)?;
        out.push_str(content.trim());
    }
    out.push_str("<|eot_id|>");
    Ok(())
}

fn render_llama3_assistant_turn(
    out: &mut String,
    message: &ChatMessage,
) -> Result<(), TemplateError> {
    out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    if let Some(content) = &message.content {
        reject_llama3_prompt_controls(content)?;
        out.push_str(content.trim());
    }
    for (index, call) in message.tool_calls.iter().enumerate() {
        if message
            .content
            .as_deref()
            .is_some_and(|content| !content.trim().is_empty())
            || index > 0
        {
            out.push('\n');
        }
        let payload = serde_json::json!({
            "name": call.function.name,
            "arguments": call.function.arguments,
        });
        let payload_json = serde_json::to_string(&payload)?;
        reject_llama3_prompt_controls(&payload_json)?;
        out.push_str(&payload_json);
    }
    out.push_str("<|eot_id|>");
    Ok(())
}

fn render_llama3_tool_response(
    out: &mut String,
    message: &ChatMessage,
) -> Result<(), TemplateError> {
    let content = message.content.as_deref().unwrap_or_default();
    reject_llama3_prompt_controls(content)?;
    out.push_str("<|start_header_id|>ipython<|end_header_id|>\n\n");
    out.push_str(content.trim());
    out.push_str("<|eot_id|>");
    Ok(())
}

fn render_deepseek_plain_turn(
    out: &mut String,
    marker: &str,
    message: &ChatMessage,
) -> Result<(), TemplateError> {
    out.push_str(marker);
    if let Some(content) = &message.content {
        reject_deepseek_prompt_controls(content)?;
        out.push_str(content);
    }
    Ok(())
}

fn render_deepseek_assistant_turn(
    out: &mut String,
    message: &ChatMessage,
) -> Result<(), TemplateError> {
    out.push_str("<｜Assistant｜>");
    if let Some(content) = &message.content {
        reject_deepseek_prompt_controls(content)?;
        out.push_str(strip_deepseek_history_reasoning(content));
    }
    if !message.tool_calls.is_empty() {
        out.push_str("<｜tool▁calls▁begin｜>");
        for call in &message.tool_calls {
            reject_deepseek_prompt_controls(&call.function.name)?;
            let arguments_json = serde_json::to_string(&call.function.arguments)?;
            reject_deepseek_prompt_controls(&arguments_json)?;
            out.push_str("<｜tool▁call▁begin｜>function<｜tool▁sep｜>");
            out.push_str(&call.function.name);
            out.push_str("\n```json\n");
            out.push_str(&arguments_json);
            out.push_str("\n```<｜tool▁call▁end｜>");
        }
        out.push_str("<｜tool▁calls▁end｜>");
    }
    out.push_str("<｜end▁of▁sentence｜>");
    Ok(())
}

fn render_deepseek_tool_response(
    out: &mut String,
    message: &ChatMessage,
    tool_output_open: &mut bool,
) -> Result<(), TemplateError> {
    let content = message.content.as_deref().unwrap_or_default();
    reject_deepseek_prompt_controls(content)?;
    if !*tool_output_open {
        out.push_str("<｜tool▁outputs▁begin｜>");
        *tool_output_open = true;
    }
    out.push_str("<｜tool▁output▁begin｜>");
    out.push_str(content);
    out.push_str("<｜tool▁output▁end｜>");
    Ok(())
}

fn strip_deepseek_history_reasoning(content: &str) -> &str {
    content
        .split_once("</think>")
        .map_or(content, |(_, answer)| answer)
}

fn render_gemma4_turn(
    out: &mut String,
    role: &str,
    message: &ChatMessage,
) -> Result<(), TemplateError> {
    out.push_str("<|turn>");
    out.push_str(role);
    out.push('\n');
    if let Some(content) = &message.content {
        reject_gemma4_prompt_controls(content)?;
        out.push_str(content.trim());
    }
    out.push_str("<turn|>\n");
    Ok(())
}

fn render_gemma4_assistant_turn(
    out: &mut String,
    message: &ChatMessage,
) -> Result<(), TemplateError> {
    out.push_str("<|turn>model\n");
    if let Some(content) = &message.content {
        reject_gemma4_prompt_controls(content)?;
        out.push_str(content.trim());
    }
    for call in &message.tool_calls {
        let arguments = render_gemma4_argument(&call.function.arguments);
        reject_gemma4_prompt_controls(&arguments)?;
        out.push_str("<|tool_call>call:");
        out.push_str(&call.function.name);
        out.push_str(&arguments);
        out.push_str("<tool_call|>");
    }
    out.push_str("<turn|>\n");
    Ok(())
}

fn render_gemma4_tool_response(
    out: &mut String,
    message: &ChatMessage,
    tool_names_by_id: &std::collections::BTreeMap<&str, &str>,
) -> Result<(), TemplateError> {
    let content = message.content.as_deref().unwrap_or_default();
    reject_gemma4_prompt_controls(content)?;
    let tool_name = message
        .tool_call_id
        .as_deref()
        .and_then(|id| tool_names_by_id.get(id).copied())
        .or(message.name.as_deref())
        .unwrap_or("unknown");
    out.push_str("<|tool_response>response:");
    out.push_str(tool_name);
    out.push_str("{value:");
    out.push_str(&render_gemma4_string(content));
    out.push_str("}<tool_response|>");
    Ok(())
}

fn render_gemma4_tool_definition(tool: &ToolDefinition) -> Result<String, TemplateError> {
    let mut out = String::new();
    out.push_str("declaration:");
    out.push_str(&tool.function.name);
    out.push('{');
    if let Some(description) = &tool.function.description {
        out.push_str("description:");
        out.push_str(&render_gemma4_string(description));
        out.push(',');
    }
    out.push_str("parameters:");
    out.push_str(&render_gemma4_argument(&tool.function.parameters));
    out.push('}');
    Ok(out)
}

fn render_gemma4_argument(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "null".to_owned(),
        serde_json::Value::Bool(value) => value.to_string(),
        serde_json::Value::Number(value) => value.to_string(),
        serde_json::Value::String(value) => render_gemma4_string(value),
        serde_json::Value::Array(items) => {
            let inner = items
                .iter()
                .map(render_gemma4_argument)
                .collect::<Vec<_>>()
                .join(",");
            format!("[{inner}]")
        }
        serde_json::Value::Object(map) => {
            let inner = map
                .iter()
                .map(|(key, value)| format!("{key}:{}", render_gemma4_argument(value)))
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{inner}}}")
        }
    }
}

fn render_gemma4_string(value: &str) -> String {
    format!("<|\"|>{}<|\"|>", value.replace("<|\"|>", ""))
}

fn reject_reserved_prompt_controls(text: &str) -> Result<(), TemplateError> {
    const RESERVED: [&str; 6] = [
        "<|im_start|>",
        "<|im_end|>",
        "<tool_call>",
        "</tool_call>",
        "<think>",
        "</think>",
    ];
    if let Some((_, token)) = RESERVED
        .iter()
        .filter_map(|token| text.find(token).map(|index| (index, *token)))
        .min_by_key(|(index, _)| *index)
    {
        return Err(TemplateError::ReservedControlToken(token));
    }
    Ok(())
}

fn reject_gemma4_prompt_controls(text: &str) -> Result<(), TemplateError> {
    const RESERVED: [&str; 14] = [
        "<|turn>",
        "<turn|>",
        "<|channel>",
        "<channel|>",
        "<|tool_call>",
        "<tool_call|>",
        "<|tool>",
        "<tool|>",
        "<|tool_response>",
        "<tool_response|>",
        "<|think|>",
        "<|image|>",
        "<|audio|>",
        "<|video|>",
    ];
    if let Some((_, token)) = RESERVED
        .iter()
        .filter_map(|token| text.find(token).map(|index| (index, *token)))
        .min_by_key(|(index, _)| *index)
    {
        return Err(TemplateError::ReservedControlToken(token));
    }
    Ok(())
}

fn reject_deepseek_prompt_controls(text: &str) -> Result<(), TemplateError> {
    const RESERVED: [&str; 15] = [
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
    ];
    if let Some((_, token)) = RESERVED
        .iter()
        .filter_map(|token| text.find(token).map(|index| (index, *token)))
        .min_by_key(|(index, _)| *index)
    {
        return Err(TemplateError::ReservedControlToken(token));
    }
    Ok(())
}

fn reject_llama3_prompt_controls(text: &str) -> Result<(), TemplateError> {
    const RESERVED: [&str; 7] = [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "<|eom_id|>",
        "<|python_tag|>",
    ];
    if let Some((_, token)) = RESERVED
        .iter()
        .filter_map(|token| text.find(token).map(|index| (index, *token)))
        .min_by_key(|(index, _)| *index)
    {
        return Err(TemplateError::ReservedControlToken(token));
    }
    Ok(())
}
