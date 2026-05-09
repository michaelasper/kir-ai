use super::*;

#[tokio::test]
async fn json_object_response_format_accepts_object_content() {
    let backend = ProtocolTestBackend::new("local-qwen36", r#"{"answer":"rust"}"#);
    let runtime = Runtime::new(backend);
    let response = runtime
        .chat(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("return json")],
            response_format: Some(ResponseFormat::JsonObject),
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("json object content is valid");

    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some(r#"{"answer":"rust"}"#)
    );
}

#[tokio::test]
async fn protocol_test_backend_returns_json_object_for_json_mode() {
    let backend =
        ProtocolTestBackend::new("local-qwen36", "plain text").with_json_object_protocol();
    let runtime = Runtime::new(backend);
    let response = runtime
        .chat(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("return json")],
            response_format: Some(ResponseFormat::JsonObject),
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("json object protocol mode succeeds");

    let content = response.choices[0]
        .message
        .content
        .as_deref()
        .expect("assistant content");
    assert!(
        serde_json::from_str::<serde_json::Value>(content)
            .expect("valid JSON")
            .is_object()
    );
}

#[tokio::test]
async fn json_object_response_format_rejects_text_content() {
    let backend = ProtocolTestBackend::new("local-qwen36", "not json");
    let runtime = Runtime::new(backend);
    let err = runtime
        .chat(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("return json")],
            response_format: Some(ResponseFormat::JsonObject),
            ..ChatCompletionRequest::default()
        })
        .await
        .expect_err("json object mode validates assistant content");

    assert!(matches!(err, RuntimeError::JsonMode(_)));
}
