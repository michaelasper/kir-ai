use llm_test_support::model_output::qwen_tool_history_round_trip;
use llm_tool_parser::QwenParser;

#[test]
fn qwen_model_output_fixture_matches_tool_parser_contract() {
    let fixture = qwen_tool_history_round_trip().expect("qwen fixture pack loads");
    let parsed = QwenParser
        .parse_complete(&fixture.assistant_output)
        .expect("fixture assistant output parses");
    let tool_call = parsed
        .tool_calls
        .first()
        .expect("fixture assistant output has a tool call");

    assert_eq!(
        parsed.reasoning.as_deref(),
        fixture.expected.parsed_reasoning.as_deref()
    );
    assert_eq!(parsed.content, fixture.expected.parsed_content);
    assert_eq!(parsed.tool_calls.len(), 1);
    assert_eq!(tool_call.function.name, fixture.expected.parsed_tool_name);
    assert_eq!(
        tool_call.function.arguments,
        fixture.expected.parsed_tool_arguments
    );
}
