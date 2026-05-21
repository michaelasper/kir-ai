use llm_models::ModelFamily;
use llm_tool_parser::{
    ParsedAssistant, QwenParser, ToolParserFamily, parse_assistant_for_family,
    parse_assistant_for_parser_family,
};

#[test]
fn parses_deepseek_plain_text_and_reasoning() {
    let plain = parse_assistant_for_family(
        ModelFamily::DeepSeek,
        include_str!("fixtures/deepseek/text.txt"),
    )
    .expect("DeepSeek plain text parses");
    assert_eq!(plain, ParsedAssistant::content("Plain DeepSeek answer.\n"));

    let reasoning = parse_assistant_for_family(
        ModelFamily::DeepSeek,
        include_str!("fixtures/deepseek/reasoning.txt"),
    )
    .expect("DeepSeek reasoning parses");
    assert_eq!(reasoning.reasoning.as_deref(), Some("Need inspect."));
    assert_eq!(reasoning.content, "Use read_file.\n");
    assert!(reasoning.tool_calls.is_empty());
}

#[test]
fn parses_deepseek_dsml_tool_call() {
    let parsed = parse_assistant_for_family(
        ModelFamily::DeepSeek,
        include_str!("fixtures/deepseek/tool_dsml.txt"),
    )
    .expect("DeepSeek DSML tool call parses");

    assert_eq!(parsed.content, "\n");
    assert_eq!(parsed.tool_calls.len(), 1);
    assert_eq!(parsed.tool_calls[0].function.name, "read_file");
    assert_eq!(
        parsed.tool_calls[0].function.arguments["path"],
        "Cargo.toml"
    );
}

#[test]
fn parses_deepseek_native_tool_call_tokens() {
    let parsed = parse_assistant_for_family(
        ModelFamily::DeepSeek,
        "<｜Assistant｜>I need a file.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>read_file\n```json\n{\"path\":\"Cargo.toml\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>",
    )
    .expect("DeepSeek native tool call parses");

    assert_eq!(parsed.content, "I need a file.");
    assert_eq!(parsed.tool_calls.len(), 1);
    assert_eq!(parsed.tool_calls[0].function.name, "read_file");
    assert_eq!(
        parsed.tool_calls[0].function.arguments["path"],
        "Cargo.toml"
    );
}

#[test]
fn parser_family_supports_hermes_deepseek_gemma_qwen_and_auto() {
    let hermes = parse_assistant_for_parser_family(
        ToolParserFamily::Hermes,
        "<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"Cargo.toml\"}}</tool_call>",
    )
    .expect("Hermes parser routes to Qwen-compatible tool syntax");
    assert_eq!(hermes.tool_calls[0].function.name, "read_file");

    let deepseek = parse_assistant_for_parser_family(
        ToolParserFamily::DeepSeek,
        include_str!("fixtures/deepseek/tool_dsml.txt"),
    )
    .expect("DeepSeek parser selected");
    assert_eq!(deepseek.tool_calls[0].function.name, "read_file");

    let gemma = parse_assistant_for_parser_family(
        ToolParserFamily::Gemma,
        "<|tool_call>call:lookup{query:<|\"|>rust<|\"|>}<tool_call|>",
    )
    .expect("Gemma parser selected");
    assert_eq!(gemma.tool_calls[0].function.name, "lookup");

    let qwen = parse_assistant_for_parser_family(
        ToolParserFamily::Qwen,
        "<tool_call>{\"name\":\"write_file\",\"arguments\":{\"path\":\"src/lib.rs\"}}</tool_call>",
    )
    .expect("Qwen parser selected");
    assert_eq!(qwen.tool_calls[0].function.name, "write_file");

    let auto = parse_assistant_for_parser_family(
        ToolParserFamily::Auto,
        include_str!("fixtures/deepseek/tool_dsml.txt"),
    )
    .expect("Auto parser detects DeepSeek DSML");
    assert_eq!(auto.tool_calls[0].function.name, "read_file");
}

#[test]
fn qwen_parser_selection_still_routes_to_qwen_parser() {
    let text = "<think>Need a file.</think><tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"Cargo.toml\"}}</tool_call>";

    let selected = parse_assistant_for_family(ModelFamily::Qwen, text).expect("selected parser");
    let direct = QwenParser.parse_complete(text).expect("direct parser");

    assert_eq!(selected.reasoning, direct.reasoning);
    assert_eq!(selected.content, direct.content);
    assert_eq!(selected.tool_calls.len(), direct.tool_calls.len());
    assert_eq!(
        selected.tool_calls[0].function,
        direct.tool_calls[0].function
    );
    for parsed in [&selected, &direct] {
        let id = &parsed.tool_calls[0].id;
        assert!(id.starts_with("call_"));
        assert!(
            !id["call_".len()..]
                .chars()
                .all(|character| character.is_ascii_digit())
        );
    }
}
