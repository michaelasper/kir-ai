use llm_models::ModelFamily;
use llm_tool_parser::{LlamaParser, ToolParserFamily, parse_assistant_for_family};

#[test]
fn parses_llama_plain_content_without_json_tool_confusion() {
    let parsed = LlamaParser
        .parse_complete(r#"{"answer":"plain json response"}"#)
        .expect("plain JSON content stays content");

    assert_eq!(parsed.content, r#"{"answer":"plain json response"}"#);
    assert!(parsed.tool_calls.is_empty());
}

#[test]
fn parses_llama_raw_json_tool_call() {
    let parsed = parse_assistant_for_family(
        ModelFamily::Llama,
        r#"{"name":"lookup","parameters":{"query":"rust"}}"#,
    )
    .expect("llama raw JSON tool call parses");

    assert_eq!(parsed.content, "");
    assert_eq!(parsed.tool_calls.len(), 1);
    assert_eq!(parsed.tool_calls[0].function.name, "lookup");
    assert_eq!(parsed.tool_calls[0].function.arguments["query"], "rust");
}

#[test]
fn parses_llama_openai_structured_tool_call_wrapper() {
    let parsed = LlamaParser
        .parse_complete(
            r#"{"tool_calls":[{"function":{"name":"read_file","arguments":"{\"path\":\"Cargo.toml\"}"}}]}"#,
        )
        .expect("OpenAI wrapper parses");

    assert_eq!(parsed.tool_calls.len(), 1);
    assert_eq!(parsed.tool_calls[0].function.name, "read_file");
    assert_eq!(
        parsed.tool_calls[0].function.arguments["path"],
        "Cargo.toml"
    );
}

#[test]
fn parses_llama_internal_tool_call_markup_from_mlx_structured_deltas() {
    let parsed = LlamaParser
        .parse_complete(r#"<tool_call>{"name":"lookup","arguments":{"query":"mlx"}}</tool_call>"#)
        .expect("internal tool markup parses");

    assert_eq!(parsed.tool_calls.len(), 1);
    assert_eq!(parsed.tool_calls[0].function.name, "lookup");
    assert_eq!(parsed.tool_calls[0].function.arguments["query"], "mlx");
}

#[test]
fn llama_parser_family_alias_routes_to_llama_parser() {
    assert_eq!(
        ToolParserFamily::from_vllm_name("llama3_json_tool_parser"),
        Some(ToolParserFamily::Llama)
    );
}
