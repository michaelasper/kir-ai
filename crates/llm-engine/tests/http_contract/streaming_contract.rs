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
                        "model": "local-qwen36",
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
                    "model": "local-qwen36",
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

#[tokio::test]
async fn chat_stream_headers_return_before_backend_completion() {
    let release = Arc::new(Semaphore::new(0));
    let app = build_router_with_backend(Box::new(DelayedStreamBackend {
        release: release.clone(),
    }));
    let response = tokio::time::timeout(
        Duration::from_millis(200),
        app.oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "local-qwen36",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": true
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        ),
    )
    .await
    .expect("streaming response starts before backend release")
    .expect("stream response");

    assert_eq!(response.status(), StatusCode::OK);
    release.add_permits(1);
    let body = body_text(response.into_body()).await;
    assert!(body.contains("\"content\":\"released\""));
    assert_eq!(body.matches("data: [DONE]").count(), 1);
}

#[tokio::test]
async fn chat_stream_emits_heartbeat_before_backend_chunk() {
    let release = Arc::new(Semaphore::new(0));
    let response = build_router_with_backend(Box::new(DelayedStreamBackend {
        release: release.clone(),
    }))
    .oneshot(
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": "local-qwen36",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": true
                })
                .to_string(),
            ))
            .expect("request builds"),
    )
    .await
    .expect("stream response");

    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    let mut seen = String::new();
    tokio::time::timeout(Duration::from_millis(300), async {
        while !seen.contains("llm-engine-heartbeat") {
            let chunk = body
                .next()
                .await
                .expect("body has heartbeat chunk")
                .expect("heartbeat body");
            seen.push_str(std::str::from_utf8(&chunk).expect("utf8 heartbeat"));
            assert!(!seen.contains("\"content\":\"released\""));
        }
    })
    .await
    .expect("heartbeat arrives before backend release");

    release.add_permits(1);
    let mut tail = String::new();
    while let Some(chunk) = body.next().await {
        tail.push_str(std::str::from_utf8(&chunk.expect("body chunk")).expect("utf8 sse"));
    }
    assert!(tail.contains("\"content\":\"released\""));
}

#[tokio::test]
async fn chat_stream_reports_backend_stall_after_configured_timeout() {
    let release = Arc::new(Semaphore::new(0));
    let response = build_router_with_backend_and_options(
        Box::new(DelayedStreamBackend {
            release: release.clone(),
        }),
        EngineOptions {
            stream_stall_timeout: Some(Duration::from_millis(50)),
            ..EngineOptions::default()
        },
    )
    .expect("router builds")
    .oneshot(
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": "local-qwen36",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": true
                })
                .to_string(),
            ))
            .expect("request builds"),
    )
    .await
    .expect("stream response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = tokio::time::timeout(Duration::from_millis(300), body_text(response.into_body()))
        .await
        .expect("stall response completes");
    assert!(body.contains("\"code\":\"stream_stalled\""));
    assert_eq!(body.matches("data: [DONE]").count(), 1);
}

#[tokio::test]
async fn chat_stream_runtime_errors_include_stable_metadata() {
    let response = build_router_with_backend(Box::new(FailingStreamBackend))
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "local-qwen36",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": true
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("stream response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_text(response.into_body()).await;
    assert!(body.contains("\"content\":\"first\""));
    assert!(body.contains("\"code\":\"backend_execution_failed\""));
    assert!(body.contains("\"phase\":\"decode\""));
    assert!(body.contains("\"retryable\":true"));
    assert_eq!(body.matches("data: [DONE]").count(), 1);
}

#[tokio::test]
async fn chat_stream_sends_backend_chunk_before_backend_finishes() {
    let first = Arc::new(Notify::new());
    let finish = Arc::new(Notify::new());
    let app = build_router_with_backend(Box::new(TwoStageStreamBackend {
        first: first.clone(),
        finish: finish.clone(),
    }));
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "local-qwen36",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": true
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("stream response");

    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    first.notify_one();
    let mut seen = String::new();
    tokio::time::timeout(Duration::from_millis(200), async {
        while !seen.contains("\"content\":\"first\"") {
            let chunk = body
                .next()
                .await
                .expect("body has chunk")
                .expect("body chunk");
            seen.push_str(std::str::from_utf8(&chunk).expect("utf8 sse"));
        }
    })
    .await
    .expect("first backend chunk is sent before final backend chunk");
    assert!(
        tokio::time::timeout(Duration::from_millis(50), body.next())
            .await
            .is_err(),
        "body should wait for final backend chunk"
    );

    finish.notify_one();
    let mut tail = seen;
    while let Some(chunk) = body.next().await {
        tail.push_str(std::str::from_utf8(&chunk.expect("body chunk")).expect("utf8 sse"));
    }
    assert!(tail.contains("\"content\":\" second\""));
    assert_eq!(tail.matches("data: [DONE]").count(), 1);
}

#[tokio::test]
async fn chat_stream_with_tools_sends_backend_chunk_before_backend_finishes() {
    let first = Arc::new(Notify::new());
    let finish = Arc::new(Notify::new());
    let app = build_router_with_backend(Box::new(TwoStageStreamBackend {
        first: first.clone(),
        finish: finish.clone(),
    }));
    let response = tokio::time::timeout(
        Duration::from_millis(200),
        app.oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "local-qwen36",
                        "messages": [{"role": "user", "content": "lookup while explaining"}],
                        "tools": [{
                            "type": "function",
                            "function": {"name": "lookup", "parameters": {}}
                        }],
                        "tool_choice": "auto",
                        "stream": true
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        ),
    )
    .await
    .expect("tool-capable stream response should not wait for backend completion")
    .expect("stream response");

    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    first.notify_one();
    let mut seen = String::new();
    tokio::time::timeout(Duration::from_millis(200), async {
        while !seen.contains("\"content\":\"first\"") {
            let chunk = body
                .next()
                .await
                .expect("body has chunk")
                .expect("body chunk");
            seen.push_str(std::str::from_utf8(&chunk).expect("utf8 sse"));
        }
    })
    .await
    .expect("first backend chunk is sent before final backend chunk");

    finish.notify_one();
    while let Some(chunk) = body.next().await {
        seen.push_str(std::str::from_utf8(&chunk.expect("body chunk")).expect("utf8 sse"));
    }
    assert!(seen.contains("\"content\":\" second\""));
    assert_eq!(seen.matches("data: [DONE]").count(), 1);
}

#[tokio::test]
async fn dropping_chat_stream_body_cancels_backend_stream() {
    let cancelled = Arc::new(Notify::new());
    let response = build_router_with_backend(Box::new(CancellableStreamBackend {
        cancelled: cancelled.clone(),
    }))
    .oneshot(
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": "local-qwen36",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": true
                })
                .to_string(),
            ))
            .expect("request builds"),
    )
    .await
    .expect("stream response");

    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    let mut seen = String::new();
    tokio::time::timeout(Duration::from_millis(300), async {
        while !seen.contains("\"content\":\"first\"") {
            let chunk = body
                .next()
                .await
                .expect("body has chunk")
                .expect("body chunk");
            seen.push_str(std::str::from_utf8(&chunk).expect("utf8 sse"));
        }
    })
    .await
    .expect("first backend chunk arrives");

    drop(body);
    tokio::time::timeout(Duration::from_millis(300), cancelled.notified())
        .await
        .expect("backend stream receives cancellation");
}
