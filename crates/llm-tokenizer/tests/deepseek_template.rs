use llm_api::ChatMessage;
use llm_models::ModelFamily;
use llm_tokenizer::render_family_chat_template;

#[test]
fn deepseek_template_selection_fails_closed_until_qwen_parity() {
    let err =
        render_family_chat_template(ModelFamily::DeepSeek, &[ChatMessage::user("hello")], &[])
            .expect_err("DeepSeek template is deferred");

    assert_eq!(err.code(), "unsupported_template_family");
    assert!(err.to_string().contains("DeepSeek"));
}
