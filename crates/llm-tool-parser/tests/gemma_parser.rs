use llm_models::ModelFamily;
use llm_tool_parser::parse_assistant_for_family;

#[test]
fn gemma_parser_fixtures_fail_closed_until_qwen_parity() {
    for fixture in [
        include_str!("fixtures/gemma/text.txt"),
        include_str!("fixtures/gemma/reasoning_channels.txt"),
        include_str!("fixtures/gemma/tool_channel.txt"),
        include_str!("fixtures/gemma/unsupported_multimodal.txt"),
    ] {
        let err = parse_assistant_for_family(ModelFamily::Gemma, fixture)
            .expect_err("Gemma parser is deferred");

        assert_eq!(err.code(), "unsupported_parser_family");
    }
}
