use llm_tool_parser::{ParsedAssistant, QwenParser};

#[test]
fn parses_reasoning_content_and_hermes_tool_call() {
    let parsed = QwenParser
        .parse_complete(
            "<think>Need a file read.</think><tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"Cargo.toml\"}}</tool_call>",
        )
        .expect("tool call parses");

    assert_eq!(parsed.reasoning.as_deref(), Some("Need a file read."));
    assert_eq!(parsed.content, "");
    assert_eq!(parsed.tool_calls.len(), 1);
    assert_eq!(parsed.tool_calls[0].function.name, "read_file");
    assert_eq!(
        parsed.tool_calls[0].function.arguments["path"],
        "Cargo.toml"
    );
}

#[test]
fn parses_qwen_coder_xml_tool_call() {
    let parsed = QwenParser
        .parse_complete(
            "<tool_call><function=bash><parameter=cmd>cargo test --workspace</parameter></function></tool_call>",
        )
        .expect("xml tool call parses");

    assert_eq!(
        parsed,
        ParsedAssistant::single_tool("bash", serde_json::json!({"cmd": "cargo test --workspace"}))
    );
}

#[test]
fn fails_when_tool_markup_is_malformed() {
    let err = QwenParser
        .parse_complete("<tool_call>{\"name\":\"read_file\",\"arguments\":")
        .expect_err("malformed tool call fails");

    assert_eq!(err.code(), "malformed_tool_call");
}
