use llm_models::ModelFamily;
use llm_tool_parser::{QwenParser, parse_assistant_for_family};

#[test]
fn deepseek_parser_fixtures_fail_closed_until_qwen_parity() {
    for fixture in [
        include_str!("fixtures/deepseek/text.txt"),
        include_str!("fixtures/deepseek/reasoning.txt"),
        include_str!("fixtures/deepseek/tool_dsml.txt"),
        include_str!("fixtures/deepseek/raw_completion.txt"),
    ] {
        let err = parse_assistant_for_family(ModelFamily::DeepSeek, fixture)
            .expect_err("DeepSeek parser is deferred");

        assert_eq!(err.code(), "unsupported_parser_family");
    }
}

#[test]
fn qwen_parser_selection_still_routes_to_qwen_parser() {
    let text = "<think>Need a file.</think><tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"Cargo.toml\"}}</tool_call>";

    let selected = parse_assistant_for_family(ModelFamily::Qwen, text).expect("selected parser");
    let direct = QwenParser.parse_complete(text).expect("direct parser");

    assert_eq!(selected, direct);
}
