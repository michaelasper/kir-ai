use llm_tool_parser::{ToolParserFamily, parse_assistant_for_parser_family};

#[test]
fn vllm_parser_names_route_to_supported_families() {
    assert_eq!(
        ToolParserFamily::from_vllm_name("qwen3xml_tool_parser"),
        Some(ToolParserFamily::Qwen)
    );
    assert_eq!(
        ToolParserFamily::from_vllm_name("deepseek_v3_tool_parser"),
        Some(ToolParserFamily::DeepSeek)
    );
    assert_eq!(
        ToolParserFamily::from_vllm_name("functiongemma_tool_parser"),
        Some(ToolParserFamily::Gemma)
    );
    assert_eq!(
        ToolParserFamily::from_vllm_name("llama3_json_tool_parser"),
        Some(ToolParserFamily::Llama)
    );
    assert_eq!(
        ToolParserFamily::from_vllm_name("granite-20b-fc"),
        Some(ToolParserFamily::Json)
    );
    assert_eq!(
        ToolParserFamily::from_vllm_name("xlam"),
        Some(ToolParserFamily::Xlam)
    );
    assert_eq!(
        ToolParserFamily::from_vllm_name("deepseekv32_tool_parser"),
        None
    );
    assert_eq!(ToolParserFamily::from_vllm_name("glm4_moe"), None);
}

#[test]
fn json_family_parses_openai_structured_tool_call_payloads() {
    let parsed = parse_assistant_for_parser_family(
        ToolParserFamily::Json,
        r#"{"tool_calls":[{"function":{"name":"read_file","arguments":"{\"path\":\"Cargo.toml\"}"}}]}"#,
    )
    .expect("json parser succeeds");

    assert!(parsed.content.is_empty());
    assert_eq!(parsed.tool_calls[0].function.name, "read_file");
    assert_eq!(
        parsed.tool_calls[0].function.arguments["path"],
        "Cargo.toml"
    );
}

#[test]
fn xlam_family_parses_tool_calls_marker_and_code_fences() {
    let marker = parse_assistant_for_parser_family(
        ToolParserFamily::Xlam,
        r#"plan first [TOOL_CALLS][{"name":"lookup","arguments":{"query":"rust"}}]
tail"#,
    )
    .expect("xlam marker parser succeeds");
    assert_eq!(marker.content, "plan first\ntail");
    assert_eq!(marker.tool_calls[0].function.name, "lookup");
    assert_eq!(marker.tool_calls[0].function.arguments["query"], "rust");

    let fenced = parse_assistant_for_parser_family(
        ToolParserFamily::Xlam,
        "```json\n[{\"name\":\"search\",\"arguments\":{\"q\":\"mlx\"}}]\n```",
    )
    .expect("xlam code-fence parser succeeds");
    assert!(fenced.content.is_empty());
    assert_eq!(fenced.tool_calls[0].function.name, "search");
    assert_eq!(fenced.tool_calls[0].function.arguments["q"], "mlx");
}

#[test]
fn auto_parser_does_not_treat_plain_json_code_fences_as_xlam_tools() {
    let text = "Here is config:\n```json\n{\"enabled\":true}\n```";

    let parsed = parse_assistant_for_parser_family(ToolParserFamily::Auto, text)
        .expect("plain fenced JSON is ordinary content in auto mode");

    assert_eq!(parsed.content, text);
    assert!(parsed.tool_calls.is_empty());
}

#[test]
fn auto_parser_still_detects_xlam_tool_call_marker() {
    let parsed = parse_assistant_for_parser_family(
        ToolParserFamily::Auto,
        r#"[TOOL_CALLS][{"name":"lookup","arguments":{"query":"rust"}}]"#,
    )
    .expect("xlam marker parser succeeds");

    assert!(parsed.content.is_empty());
    assert_eq!(parsed.tool_calls[0].function.name, "lookup");
    assert_eq!(parsed.tool_calls[0].function.arguments["query"], "rust");
}
