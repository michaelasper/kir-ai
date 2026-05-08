use llm_api::{ChatMessage, ChatRole, ToolDefinition};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct QwenPromptOptions {
    pub enable_thinking: bool,
    pub add_generation_prompt: bool,
}

#[derive(Debug, Error)]
pub enum TemplateError {
    #[error("tool serialization failed: {0}")]
    ToolSerialization(#[from] serde_json::Error),
    #[error("message role `{0}` cannot be rendered in qwen chatml")]
    UnsupportedRole(String),
}

pub fn render_qwen_chatml(
    messages: &[ChatMessage],
    tools: &[ToolDefinition],
    options: &QwenPromptOptions,
) -> Result<String, TemplateError> {
    let mut out = String::new();
    if !tools.is_empty() {
        out.push_str("<|im_start|>system\n");
        out.push_str(
            "Tools are available. Return tool invocations inside <tool_call> JSON blocks.\n",
        );
        out.push_str(&serde_json::to_string(tools)?);
        out.push_str("<|im_end|>\n");
    }

    for message in messages {
        match message.role {
            ChatRole::System => render_plain(&mut out, "system", message),
            ChatRole::User => render_plain(&mut out, "user", message),
            ChatRole::Tool => render_plain(&mut out, "tool", message),
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

fn render_plain(out: &mut String, role: &str, message: &ChatMessage) {
    out.push_str("<|im_start|>");
    out.push_str(role);
    out.push('\n');
    if let Some(content) = &message.content {
        out.push_str(content);
    }
    out.push_str("<|im_end|>\n");
}

fn render_assistant(out: &mut String, message: &ChatMessage) -> Result<(), TemplateError> {
    out.push_str("<|im_start|>assistant\n");
    if let Some(content) = &message.content {
        out.push_str(content);
    }
    for call in &message.tool_calls {
        let payload = serde_json::json!({
            "name": call.function.name,
            "arguments": call.function.arguments,
        });
        out.push_str("<tool_call>");
        out.push_str(&serde_json::to_string(&payload)?);
        out.push_str("</tool_call>");
    }
    out.push_str("<|im_end|>\n");
    Ok(())
}
