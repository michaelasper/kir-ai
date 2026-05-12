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
fn parses_parameters_alias_in_hermes_tool_call() {
    let parsed = QwenParser
        .parse_complete(
            "<tool_call>{\"name\":\"read_file\",\"parameters\":{\"path\":\"Cargo.toml\",\"_i\":0}}</tool_call>",
        )
        .expect("parameters alias parses");

    assert_eq!(parsed.tool_calls.len(), 1);
    assert_eq!(parsed.tool_calls[0].function.name, "read_file");
    assert_eq!(
        parsed.tool_calls[0].function.arguments,
        serde_json::json!({"path": "Cargo.toml", "_i": 0})
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
fn preserves_plain_assistant_whitespace() {
    let parsed = QwenParser
        .parse_complete("  keep leading space\n    indented line\n")
        .expect("plain text parses");

    assert_eq!(parsed.content, "  keep leading space\n    indented line\n");
    assert!(parsed.tool_calls.is_empty());
}

#[test]
fn preserves_content_around_reasoning_and_tool_tags() {
    let parsed = QwenParser
        .parse_complete(
            "  before\n<think>private chain</think>\ninside\n<tool_call>{\"name\":\"lookup\",\"arguments\":{\"query\":\"rust\"}}</tool_call>\n  after\n",
        )
        .expect("tagged output parses");

    assert_eq!(parsed.reasoning.as_deref(), Some("private chain"));
    assert_eq!(parsed.content, "  before\n\ninside\n\n  after\n");
    assert_eq!(parsed.tool_calls.len(), 1);
    assert_eq!(parsed.tool_calls[0].function.name, "lookup");
}

#[test]
fn parses_truncated_reasoning_as_partial_reasoning() {
    let parsed = QwenParser
        .parse_complete("visible prefix\n<think>Need more tokens")
        .expect("truncated reasoning parses as partial output");

    assert_eq!(parsed.reasoning.as_deref(), Some("Need more tokens"));
    assert_eq!(parsed.content, "visible prefix\n");
    assert!(parsed.tool_calls.is_empty());
}

#[test]
fn fails_when_tool_markup_is_malformed() {
    let err = QwenParser
        .parse_complete("<tool_call>{\"name\":\"read_file\",\"arguments\":")
        .expect_err("malformed tool call fails");

    assert_eq!(err.code(), "malformed_tool_call");
}
