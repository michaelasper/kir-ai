use llm_api::ChatMessage;
use llm_models::ModelFamily;
use llm_tokenizer::render_family_chat_template;

#[test]
fn gemma_template_selection_fails_closed_until_qwen_parity() {
    let err = render_family_chat_template(ModelFamily::Gemma, &[ChatMessage::user("hello")], &[])
        .expect_err("Gemma template is deferred");

    assert_eq!(err.code(), "unsupported_template_family");
    assert!(err.to_string().contains("Gemma"));
}
