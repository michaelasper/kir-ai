use llm_models::ModelFamily;
use llm_tool_parser::{GemmaParser, ToolParserFamily, parse_assistant_for_family};

fn assert_generated_tool_call_id_is_opaque(id: &str) {
    assert!(
        id.starts_with("call_"),
        "tool call id must use call_ prefix: {id}"
    );
    assert!(
        id.len() > "call_".len(),
        "tool call id must include an opaque suffix: {id}"
    );
    assert!(
        !id["call_".len()..]
            .chars()
            .all(|character| character.is_ascii_digit()),
        "generated tool call id must not be a predictable numeric sequence: {id}"
    );
}

#[test]
fn generated_tool_call_ids_are_opaque_across_parser_families() {
    let qwen = parse_assistant_for_family(
        ModelFamily::Qwen,
        r#"<tool_call>{"name":"lookup","arguments":{"query":"rust"}}</tool_call>"#,
    )
    .expect("Qwen tool call parses");
    assert_generated_tool_call_id_is_opaque(&qwen.tool_calls[0].id);

    let llama = parse_assistant_for_family(
        ModelFamily::Llama,
        r##"{"tool_calls":[{"function":{"name":"read_file","arguments":"{\"path\":\"Cargo.toml\"}"}}]}"##,
    )
    .expect("Llama JSON tool call parses");
    assert_generated_tool_call_id_is_opaque(&llama.tool_calls[0].id);

    let supplied = parse_assistant_for_family(
        ModelFamily::Llama,
        r##"{"tool_calls":[{"id":"call_read_1","function":{"name":"read_file","arguments":"{\"path\":\"Cargo.toml\"}"}}]}"##,
    )
    .expect("Llama JSON tool call with supplied id parses");
    assert_eq!(supplied.tool_calls[0].id, "call_read_1");

    let deepseek = parse_assistant_for_family(
        ModelFamily::DeepSeek,
        r#"<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>read_file
```json
{"path":"Cargo.toml"}
```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"#,
    )
    .expect("DeepSeek native tool call parses");
    assert_generated_tool_call_id_is_opaque(&deepseek.tool_calls[0].id);

    let gemma = GemmaParser
        .parse_complete(r#"<|tool_call>call:lookup{query:<|"|>rust<|"|>}<tool_call|>"#)
        .expect("Gemma tool call parses");
    assert_generated_tool_call_id_is_opaque(&gemma.tool_calls[0].id);

    let xlam = llm_tool_parser::parse_assistant_for_parser_family(
        ToolParserFamily::Xlam,
        r#"[TOOL_CALLS][{"name":"lookup","arguments":{"query":"rust"}}]"#,
    )
    .expect("XLAM tool call parses");
    assert_generated_tool_call_id_is_opaque(&xlam.tool_calls[0].id);
}
