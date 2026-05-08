use llm_api::{ChatMessage, ChatRole, ToolDefinition};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct QwenPromptOptions {
    pub enable_thinking: bool,
    pub add_generation_prompt: bool,
}

#[derive(Debug, Error)]
pub enum TemplateError {
    #[error("tool serialization failed: {0}")]
    ToolSerialization(#[from] serde_json::Error),
    #[error("reserved prompt control token `{0}` is not allowed in request text")]
    ReservedControlToken(&'static str),
    #[error("message role `{0}` cannot be rendered in qwen chatml")]
    UnsupportedRole(String),
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
