use super::*;

#[tokio::test]
async fn completions_endpoint_returns_openai_text_completion_shape() {
    let response = build_router_with_protocol_test_backend()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": llm_engine::DEFAULT_MODEL_ID,
                        "prompt": "hello",
                        "max_tokens": 8,
                        "stop": " backend"
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("completion response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["object"], "text_completion");
    assert_eq!(body["model"], llm_engine::DEFAULT_MODEL_ID);
    assert_eq!(body["choices"][0]["text"], "hello from rust native");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn completions_endpoint_reports_backend_unsupported_sampling_controls() {
    let response = build_router_with_protocol_test_backend()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": llm_engine::DEFAULT_MODEL_ID,
                        "prompt": "hello",
                        "temperature": 0.7
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("completion response");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["error"]["code"], "unsupported_capability");
    assert_eq!(body["error"]["phase"], "request_validation");
    assert_eq!(body["error"]["retryable"], false);
}

#[tokio::test]
async fn completions_endpoint_rejects_malformed_json_with_stable_error() {
    let response = build_router_with_protocol_test_backend()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from("{not-json"))
                .expect("request builds"),
        )
        .await
        .expect("completion response");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["error"]["code"], "invalid_request");
    assert_eq!(body["error"]["phase"], "request_validation");
    assert_eq!(body["error"]["retryable"], false);
}

#[tokio::test]
async fn streaming_completion_validation_errors_return_json_error() {
    let response = build_router_with_protocol_test_backend()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": llm_engine::DEFAULT_MODEL_ID,
                        "prompt": "",
                        "stream": true
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("completion response");

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
async fn invalid_completion_request_validates_before_busy_model_permit() {
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let app = build_router_with_backend(Box::new(BlockingBackend {
        entered: entered.clone(),
        release: release.clone(),
    }));
    let first = tokio::spawn(app.clone().oneshot(chat_request_body("first")));
    entered.notified().await;

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": llm_engine::DEFAULT_MODEL_ID,
                        "prompt": ""
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("completion response");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["error"]["code"], "invalid_request");
    assert_eq!(body["error"]["phase"], "request_validation");

    release.notify_waiters();
    first
        .await
        .expect("first request task")
        .expect("first response");
}

#[tokio::test]
async fn completions_endpoint_streams_openai_sse_chunks() {
    let response = build_router_with_protocol_test_backend()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": llm_engine::DEFAULT_MODEL_ID,
                        "prompt": "hello",
                        "stream": true,
                        "max_tokens": 8,
                        "stop": " backend"
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("completion stream response");

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
    assert!(body.contains("data: {\"id\":\"cmpl-"));
    assert!(body.contains("\"object\":\"text_completion\""));
    assert!(body.contains("\"text\":\"hello from rust native\""));
    assert!(body.contains("\"finish_reason\":\"stop\""));
    assert_eq!(body.matches("data: [DONE]").count(), 1);
}

#[tokio::test]
async fn completions_endpoint_streams_usage_when_requested() {
    let response = build_router_with_protocol_test_backend()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": llm_engine::DEFAULT_MODEL_ID,
                        "prompt": "hello",
                        "stream": true,
                        "stream_options": {"include_usage": true}
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("completion stream response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_text(response.into_body()).await;
    assert!(body.contains("\"choices\":[],\"usage\":{\"prompt_tokens\""));
    assert!(body.contains("\"total_tokens\""));
    assert!(
        body.find("\"choices\":[],\"usage\"").expect("usage chunk")
            < body.find("data: [DONE]").expect("done")
    );
}
