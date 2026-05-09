use super::*;

#[tokio::test]
async fn chat_rejects_missing_model_family_before_generation() {
    let runtime = Runtime::new(FamilyMetadataBackend { family: None });
    let err = runtime
        .chat(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("say hi")],
            max_tokens: Some(16),
            ..ChatCompletionRequest::default()
        })
        .await
        .expect_err("missing family should fail before generation");

    assert!(err.to_string().contains("did not declare a model family"));
}

#[tokio::test]
async fn chat_accepts_mlx_backend_when_family_is_qwen() {
    let runtime = Runtime::new(MlxQwenMetadataBackend);
    let response = runtime
        .chat(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("say hi")],
            max_tokens: Some(16),
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("qwen MLX metadata selects Qwen adapter");

    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("hello from mlx")
    );
}

#[tokio::test]
async fn chat_accepts_mlx_backend_when_family_is_gemma() {
    let runtime = Runtime::new(MlxGemmaMetadataBackend);
    let response = runtime
        .chat(ChatCompletionRequest {
            model: "local-gemma4".to_owned(),
            messages: vec![ChatMessage::user("say hi")],
            max_tokens: Some(16),
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("Gemma MLX metadata selects Gemma adapter");

    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("hello from gemma")
    );
}

#[tokio::test]
async fn chat_accepts_mlx_backend_when_family_is_deepseek() {
    let runtime = Runtime::new(MlxDeepSeekMetadataBackend);
    let response = runtime
        .chat(ChatCompletionRequest {
            model: "local-deepseek".to_owned(),
            messages: vec![ChatMessage::user("say hi")],
            max_tokens: Some(16),
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("DeepSeek MLX metadata selects DeepSeek adapter");

    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("hello from deepseek")
    );
}
