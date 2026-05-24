use super::*;

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
                        "model": llm_engine::DEFAULT_MODEL_ID,
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

#[tokio::test(start_paused = true)]
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
                    "model": llm_engine::DEFAULT_MODEL_ID,
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

#[tokio::test(start_paused = true)]
async fn chat_stream_does_not_stall_before_first_backend_output() {
    let release = Arc::new(Semaphore::new(0));
    let app = build_router_with_unauthenticated_admin_and_options(
        Box::new(DelayedStreamBackend {
            release: release.clone(),
        }),
        EngineOptions {
            stream_stall_timeout: Some(Duration::from_millis(50)),
            ..EngineOptions::default()
        },
    )
    .expect("router builds");
    let delayed_release = release.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(120)).await;
        delayed_release.add_permits(1);
    });
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": llm_engine::DEFAULT_MODEL_ID,
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
    let body = tokio::time::timeout(Duration::from_millis(400), body_text(response.into_body()))
        .await
        .expect("stream should survive slow prefill before first output");
    assert!(body.contains("\"content\":\"released\""), "body: {body}");
    assert!(
        !body.contains("\"code\":\"stream_stalled\""),
        "body: {body}"
    );
    assert_eq!(body.matches("data: [DONE]").count(), 1);
    let metrics = wait_for_metrics(&app, |body| body["successful_requests"] == 1).await;
    assert_eq!(metrics["stream_stalled_requests"], 0);
    assert_eq!(metrics["stream_client_disconnected_requests"], 0);
    assert_eq!(metrics["failed_requests"], 0);
    assert_eq!(metrics["scheduler_failed_requests"], 0);
    assert_eq!(metrics["scheduler_completed_requests"], 1);
}

#[tokio::test(start_paused = true)]
async fn chat_stream_stall_cancels_backend_work() {
    let cancelled = Arc::new(Notify::new());
    let response = build_router_with_backend_and_options(
        Box::new(CancellableStreamBackend {
            cancelled: cancelled.clone(),
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
                    "model": llm_engine::DEFAULT_MODEL_ID,
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
    assert!(body.contains("\"content\":\"first\""));
    let frames = sse_json_frames(&body);
    let error_frames: Vec<&Value> = frames
        .iter()
        .filter_map(|frame| frame.get("error"))
        .collect();
    assert_eq!(error_frames.len(), 1, "body: {body}");
    let error = error_frames[0];
    assert_eq!(
        error["message"],
        "stream stalled for 50 ms without meaningful backend output"
    );
    assert_eq!(error["code"], "stream_stalled");
    assert_eq!(error["phase"], "streaming");
    assert_eq!(error["retryable"], true);
    assert_eq!(error["type"], "llm_engine_error");
    assert_eq!(body.matches("data: [DONE]").count(), 1);
    tokio::time::timeout(Duration::from_millis(300), cancelled.notified())
        .await
        .expect("stream stall cancels backend token");
}

#[tokio::test(start_paused = true)]
async fn chat_stream_does_not_stall_on_regular_one_byte_deltas() {
    let app = build_router_with_unauthenticated_admin_and_options(
        Box::new(OneByteDeltaStreamBackend {
            delay: Duration::from_millis(40),
            fragments: vec!["a", "b", "c"],
        }),
        EngineOptions {
            stream_stall_timeout: Some(Duration::from_millis(50)),
            ..EngineOptions::default()
        },
    )
    .expect("router builds");
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": llm_engine::DEFAULT_MODEL_ID,
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
        .expect("one-byte deltas should complete before stream stall timeout");
    assert!(
        !body.contains("\"code\":\"stream_stalled\""),
        "body: {body}"
    );
    assert!(body.contains("\"content\":\"a\""), "body: {body}");
    assert!(body.contains("\"content\":\"b\""), "body: {body}");
    assert!(body.contains("\"content\":\"c\""), "body: {body}");
    assert_eq!(body.matches("data: [DONE]").count(), 1);
    let metrics = wait_for_metrics(&app, |body| body["successful_requests"] == 1).await;
    assert_eq!(metrics["stream_stalled_requests"], 0);
    assert_eq!(metrics["failed_requests"], 0);
}

#[tokio::test(start_paused = true)]
async fn chat_stream_required_tool_buffered_arguments_do_not_stall() {
    let app = build_router_with_unauthenticated_admin_and_options(
        Box::new(SlowStructuredToolArgumentBackend {
            delay: Duration::from_millis(40),
        }),
        EngineOptions {
            stream_stall_timeout: Some(Duration::from_millis(70)),
            ..EngineOptions::default()
        },
    )
    .expect("router builds");
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": llm_engine::DEFAULT_MODEL_ID,
                        "messages": [{"role": "user", "content": "read calculator.py"}],
                        "tools": [{
                            "type": "function",
                            "function": {
                                "name": "read",
                                "description": "read file",
                                "parameters": {
                                    "type": "object",
                                    "required": ["path", "_i"],
                                    "properties": {
                                        "path": {"type": "string"},
                                        "_i": {"type": "string"}
                                    }
                                }
                            }
                        }],
                        "tool_choice": "required",
                        "stream": true
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("stream response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = tokio::time::timeout(Duration::from_millis(500), body_text(response.into_body()))
        .await
        .expect("buffered required-tool arguments should complete before stall timeout");
    assert!(
        !body.contains("\"code\":\"stream_stalled\""),
        "body: {body}"
    );
    assert_eq!(body.matches("data: [DONE]").count(), 1);

    let frames = sse_json_frames(&body);
    let mut reconstructed_arguments = String::new();
    let mut saw_tool_calls_finish = false;

    for frame in &frames {
        let Some(choices) = frame["choices"].as_array() else {
            continue;
        };
        for choice in choices {
            if let Some(tool_calls) = choice["delta"]["tool_calls"].as_array() {
                for tool_call in tool_calls {
                    if let Some(arguments) = tool_call["function"]["arguments"].as_str() {
                        reconstructed_arguments.push_str(arguments);
                    }
                }
            }
            if choice["finish_reason"].as_str() == Some("tool_calls") {
                saw_tool_calls_finish = true;
            }
        }
    }

    assert!(saw_tool_calls_finish);
    let reconstructed_arguments: Value =
        serde_json::from_str(&reconstructed_arguments).expect("client arguments JSON");
    assert_eq!(reconstructed_arguments["path"], "calculator.py");
    assert!(
        reconstructed_arguments["_i"]
            .as_str()
            .is_some_and(|intent| !intent.is_empty())
    );
    let metrics = wait_for_metrics(&app, |body| body["successful_requests"] == 1).await;
    assert_eq!(metrics["stream_stalled_requests"], 0);
    assert_eq!(metrics["failed_requests"], 0);
}
