use llm_test_support::model_output::qwen_tool_history_round_trip;

#[test]
fn qwen_model_output_fixture_pack_loads() {
    let fixture = qwen_tool_history_round_trip().expect("qwen model-output fixture loads");

    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.family, "qwen");
    assert_eq!(fixture.case_name, "tool_history_round_trip");
    assert_eq!(fixture.tools.len(), 1);
    assert_eq!(fixture.messages_before_assistant.len(), 2);
    assert!(
        fixture
            .expected
            .rendered_prompt
            .contains("<|im_start|>tool")
    );
}
