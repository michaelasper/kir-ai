use super::*;

#[tokio::test]
async fn streaming_chat_validation_errors_return_json_error() {
    let response = build_router_with_protocol_test_backend()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": llm_engine::DEFAULT_MODEL_ID,
                        "messages": [],
                        "stream": true
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("chat response");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(
        !response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .starts_with("text/event-stream")
    );
    let body = body_json(response.into_body()).await;
    assert_eq!(body["error"]["code"], "invalid_request");
    assert_eq!(body["error"]["phase"], "request_validation");
}

#[tokio::test]
async fn streaming_chat_response_validation_errors_emit_sse_error_after_headers() {
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
                    "model": llm_engine::DEFAULT_MODEL_ID,
                    "messages": [{"role": "user", "content": "return json"}],
                    "response_format": {"type": "json_object"},
                    "stream": true
                })
                .to_string(),
            ))
            .expect("request builds"),
    )
    .await
    .expect("chat response");

    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .starts_with("text/event-stream")
    );
    let body = body_text(response.into_body()).await;
    assert!(body.contains("\"code\":\"json_validation_failed\""));
    assert!(body.contains("\"phase\":\"response_validation\""));
    assert_eq!(body.matches("data: [DONE]").count(), 1);
}
