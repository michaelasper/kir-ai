use super::*;

#[tokio::test]
async fn chat_completions_streams_openai_sse_chunks() {
    let response = build_router_with_protocol_test_backend()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "local-qwen36",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": true,
                        "max_tokens": 8
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("chat stream response");

    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .expect("content type")
            .starts_with("text/event-stream")
    );
    let body = body_text(response.into_body()).await;
    assert!(body.contains("data: {\"id\":\"chatcmpl-"));
    assert!(body.contains("\"object\":\"chat.completion.chunk\""));
    assert!(body.contains("\"delta\":{\"role\":\"assistant\"}"));
    assert!(body.to_ascii_lowercase().contains("rust"));
    assert_eq!(body.matches("data: [DONE]").count(), 1);
}

#[tokio::test]
async fn chat_completions_streams_usage_when_requested() {
    let response = build_router_with_protocol_test_backend()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "local-qwen36",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": true,
                        "stream_options": {"include_usage": true}
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("chat stream response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_text(response.into_body()).await;
    assert!(body.contains("\"choices\":[],\"usage\":{\"prompt_tokens\""));
    assert!(body.contains("\"total_tokens\""));
    assert!(
        body.find("\"choices\":[],\"usage\"").expect("usage chunk")
            < body.find("data: [DONE]").expect("done")
    );
}

#[tokio::test]
async fn chat_completions_streams_tool_call_deltas() {
    let response = build_router_with_backend(Box::new(StaticBackend {
        text: r#"<tool_call>{"name":"lookup","arguments":{"query":"rust"}}</tool_call>"#.to_owned(),
    }))
    .oneshot(
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": "local-qwen36",
                    "messages": [{"role": "user", "content": "lookup rust"}],
                    "tools": [{
                        "type": "function",
                        "function": {"name": "lookup", "parameters": {}}
                    }],
                    "tool_choice": "required",
                    "stream": true
                })
                .to_string(),
            ))
            .expect("request builds"),
    )
    .await
    .expect("chat stream response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_text(response.into_body()).await;
    assert!(body.contains("\"tool_calls\":[{\"index\":0,\"id\":\"call_0\",\"type\":\"function\""));
    assert!(body.contains("\"name\":\"lookup\""));
    assert!(body.contains("\"arguments\":\"{\\\"query\\\":\\\"rust\\\"}\""));
    assert!(body.contains("\"finish_reason\":\"tool_calls\""));
    assert_eq!(body.matches("data: [DONE]").count(), 1);
}

#[tokio::test]
async fn chat_completions_streaming_json_object_validation_errors_are_sse() {
    let response = build_router_with_backend(Box::new(StaticBackend {
        text: "not json".to_owned(),
    }))
    .oneshot(
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": "local-qwen36",
                    "messages": [{"role": "user", "content": "json"}],
                    "stream": true,
                    "response_format": {"type": "json_object"}
                })
                .to_string(),
            ))
            .expect("request builds"),
    )
    .await
    .expect("chat stream response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_text(response.into_body()).await;
    assert!(body.contains("\"code\":\"json_validation_failed\""));
    assert!(body.contains("\"phase\":\"response_validation\""));
    assert_eq!(body.matches("data: [DONE]").count(), 1);
    assert!(!body.contains("\"finish_reason\":\"stop\""));
}
