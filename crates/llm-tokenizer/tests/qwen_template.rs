use llm_api::{ChatMessage, ToolDefinition};
use llm_tokenizer::{QwenPromptOptions, render_qwen_chatml};
use serde_json::json;

#[test]
fn renders_qwen_no_thinking_chatml_with_tools() {
    let tools = vec![ToolDefinition::function(
        "read_file",
        "read a UTF-8 file",
        json!({
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
        }),
    )];
    let messages = vec![
        ChatMessage::system("You are a coding agent."),
        ChatMessage::user("Inspect Cargo.toml"),
    ];

    let rendered = render_qwen_chatml(
        &messages,
        &tools,
        &QwenPromptOptions {
            enable_thinking: false,
            add_generation_prompt: true,
        },
    )
    .expect("template renders");

    assert!(rendered.contains("<|im_start|>system\nYou are a coding agent.<|im_end|>"));
    assert!(rendered.contains("\"name\":\"read_file\""));
    assert!(rendered.contains("<tool_call>"));
    assert!(rendered.ends_with("<|im_start|>assistant\n</think>\n"));
}

#[test]
fn renders_prior_tool_calls_as_structured_chatml() {
    let messages = vec![
        ChatMessage::assistant_tool_call("call_1", "read_file", json!({"path": "Cargo.toml"})),
        ChatMessage::tool("call_1", "{\"workspace\":true}"),
    ];

    let rendered = render_qwen_chatml(&messages, &[], &QwenPromptOptions::default())
        .expect("template renders");

    assert!(rendered.contains("<tool_call>{\"name\":\"read_file\""));
    assert!(rendered.contains("<|im_start|>tool\n{\"workspace\":true}<|im_end|>"));
}
