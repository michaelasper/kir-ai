use llm_api::ChatMessage;
use llm_test_support::model_output::qwen_tool_history_round_trip;
use llm_tokenizer::{HuggingFaceTokenizer, QwenPromptOptions, render_qwen_chatml};
use llm_tool_parser::QwenParser;

#[test]
fn qwen_tool_history_fixture_matches_rendered_prompt_and_token_ids() {
    let fixture = qwen_tool_history_round_trip().expect("qwen fixture pack loads");
    let parsed = QwenParser
        .parse_complete(&fixture.assistant_output)
        .expect("fixture assistant output parses");
    let tool_call = parsed
        .tool_calls
        .first()
        .expect("fixture assistant output has a tool call");
    let mut messages = fixture.messages_before_assistant.clone();
    messages.push(ChatMessage::assistant_tool_call(
        &tool_call.id,
        &tool_call.function.name,
        tool_call.function.arguments.clone(),
    ));
    messages.push(ChatMessage::tool(
        &tool_call.id,
        fixture.tool_result_content.clone(),
    ));
    messages.push(ChatMessage::user(fixture.follow_up_user.clone()));

    let rendered = render_qwen_chatml(
        &messages,
        &fixture.tools,
        &QwenPromptOptions {
            enable_thinking: fixture.prompt_options.enable_thinking,
            add_generation_prompt: fixture.prompt_options.add_generation_prompt,
        },
    )
    .expect("qwen fixture prompt renders");

    assert_eq!(
        rendered.as_bytes(),
        fixture.expected.rendered_prompt.as_bytes()
    );

    let tokenizer = HuggingFaceTokenizer::from_file(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(&fixture.tokenizer_fixture),
    )
    .expect("qwen tokenizer fixture loads");
    let token_ids = tokenizer
        .encode(&rendered, false)
        .expect("qwen fixture prompt tokenizes");

    assert_eq!(token_ids, fixture.expected.token_ids);
}
