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

#[tokio::test]
async fn llama_json_object_mode_keeps_tool_shaped_json_as_content_when_no_tools_are_declared() {
    let runtime = Runtime::new(FamilyStreamBackend {
        model_id: "local-llama",
        family: "llama",
        text: r#"{"name":"report","parameters":{"status":"ok"}}<|eot_id|>"#,
        finish_reason: FinishReason::Stop,
    });
    let response = runtime
        .chat(ChatCompletionRequest {
            model: "local-llama".to_owned(),
            messages: vec![ChatMessage::user("return json")],
            response_format: Some(ResponseFormat::JsonObject),
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("json object content is not treated as an undeclared tool call");

    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some(r#"{"name":"report","parameters":{"status":"ok"}}"#)
    );
    assert!(response.choices[0].message.tool_calls.is_empty());
}
