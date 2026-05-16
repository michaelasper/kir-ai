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

#[tokio::test]
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

#[tokio::test]
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
    assert!(body.contains("\"code\":\"stream_stalled\""));
    tokio::time::timeout(Duration::from_millis(300), cancelled.notified())
        .await
        .expect("stream stall cancels backend token");
}

#[tokio::test]
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

#[tokio::test]
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
async fn chat_stream_emits_prefill_progress_sse_events() {
    let response = build_router_with_backend(Box::new(PrefillProgressStreamBackend))
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
    let body = body_text(response.into_body()).await;
    assert!(
        body.contains(
            "data: {\"type\":\"prefill_progress\",\"chunk\":1,\"total\":3,\"tokens\":2,\"total_tokens\":5}"
        ),
        "body: {body}"
    );
    assert!(body.contains("\"content\":\"done\""), "body: {body}");
    assert_eq!(body.matches("data: [DONE]").count(), 1);
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
                        "model": llm_engine::DEFAULT_MODEL_ID,
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
async fn chat_stream_structured_omp_arguments_include_filled_intent() {
    let response = build_router_with_backend(Box::new(StructuredToolDeltaHttpBackend))
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
    let body = body_text(response.into_body()).await;
    let frames = sse_json_frames(&body);
    let mut reconstructed_arguments = String::new();
    let mut saw_progress_without_arguments = false;
    let mut saw_tool_calls_finish = false;

    for frame in &frames {
        let Some(choices) = frame["choices"].as_array() else {
            continue;
        };
        for choice in choices {
            if let Some(tool_calls) = choice["delta"]["tool_calls"].as_array() {
                for tool_call in tool_calls {
                    let function = tool_call.get("function");
                    if tool_call.get("id").and_then(Value::as_str) == Some("call_read_1")
                        && function
                            .and_then(|function| function.get("name"))
                            .and_then(Value::as_str)
                            == Some("read")
                        && function
                            .and_then(|function| function.get("arguments"))
                            .is_none()
                    {
                        saw_progress_without_arguments = true;
                    }
                    if let Some(arguments) = function
                        .and_then(|function| function.get("arguments"))
                        .and_then(Value::as_str)
                    {
                        assert!(
                            !saw_tool_calls_finish,
                            "arguments must arrive before tool_calls finish"
                        );
                        reconstructed_arguments.push_str(arguments);
                    }
                }
            }
            if choice["finish_reason"].as_str() == Some("tool_calls") {
                assert!(
                    !reconstructed_arguments.is_empty(),
                    "finish must wait for final arguments"
                );
                saw_tool_calls_finish = true;
            }
        }
    }

    assert!(saw_progress_without_arguments);
    assert!(saw_tool_calls_finish);
    let reconstructed_arguments: Value =
        serde_json::from_str(&reconstructed_arguments).expect("client arguments JSON");
    assert_eq!(reconstructed_arguments["path"], "calculator.py");
    assert!(
        reconstructed_arguments["_i"]
            .as_str()
            .is_some_and(|intent| !intent.is_empty())
    );
    assert_eq!(body.matches("data: [DONE]").count(), 1);
}

#[tokio::test]
async fn dropping_chat_stream_body_cancels_backend_stream() {
    let cancelled = Arc::new(Notify::new());
    let app = build_router_with_unauthenticated_admin(Box::new(CancellableStreamBackend {
        cancelled: cancelled.clone(),
    }));
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
    let metrics = wait_for_metrics(&app, |body| {
        body["stream_client_disconnected_requests"] == 1
            && body["scheduler_cancelled_requests"] == 1
    })
    .await;
    assert_eq!(metrics["active_requests"], 0);
    assert_eq!(metrics["successful_requests"], 0);
    assert_eq!(metrics["failed_requests"], 1);
    assert_eq!(metrics["streamed_requests"], 0);
    assert_eq!(metrics["scheduler_completed_requests"], 0);
    assert_eq!(metrics["scheduler_failed_requests"], 0);
    assert_eq!(metrics["time_to_first_token_ms"]["count"], 1);
}

#[tokio::test]
async fn dropping_completion_stream_body_cancels_backend_stream() {
    let cancelled = Arc::new(Notify::new());
    let app = build_router_with_unauthenticated_admin(Box::new(CancellableStreamBackend {
        cancelled: cancelled.clone(),
    }));
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
                        "prompt": "hello",
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
        while !seen.contains("\"text\":\"first\"") {
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
    let metrics = wait_for_metrics(&app, |body| {
        body["stream_client_disconnected_requests"] == 1
            && body["scheduler_cancelled_requests"] == 1
    })
    .await;
    assert_eq!(metrics["active_requests"], 0);
    assert_eq!(metrics["successful_requests"], 0);
    assert_eq!(metrics["failed_requests"], 1);
    assert_eq!(metrics["streamed_requests"], 0);
    assert_eq!(metrics["scheduler_completed_requests"], 0);
    assert_eq!(metrics["scheduler_failed_requests"], 0);
    assert_eq!(metrics["time_to_first_token_ms"]["count"], 1);
}

#[tokio::test]
async fn dropping_chat_stream_before_first_token_records_client_disconnect_without_ttft() {
    let cancelled = Arc::new(Notify::new());
    let app = build_router_with_unauthenticated_admin(Box::new(PendingCancellableStreamBackend {
        cancelled: cancelled.clone(),
    }));
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
    let mut body = response.into_body().into_data_stream();
    let mut seen = String::new();
    tokio::time::timeout(Duration::from_millis(300), async {
        while !seen.contains("llm-engine-heartbeat") {
            let chunk = body
                .next()
                .await
                .expect("body has heartbeat")
                .expect("body chunk");
            seen.push_str(std::str::from_utf8(&chunk).expect("utf8 sse"));
        }
    })
    .await
    .expect("heartbeat arrives before backend output");

    drop(body);
    tokio::time::timeout(Duration::from_millis(300), cancelled.notified())
        .await
        .expect("backend stream receives cancellation");
    let metrics = wait_for_metrics(&app, |body| {
        body["stream_client_disconnected_requests"] == 1
            && body["scheduler_cancelled_requests"] == 1
    })
    .await;
    assert_eq!(metrics["active_requests"], 0);
    assert_eq!(metrics["prefill_requests"], 0);
    assert_eq!(metrics["decode_requests"], 0);
    assert_eq!(metrics["active_prefill_requests"], 0);
    assert_eq!(metrics["active_decode_requests"], 0);
    assert_eq!(metrics["scheduler_completed_requests"], 0);
    assert_eq!(metrics["time_to_first_token_ms"]["count"], 0);
}

#[tokio::test]
async fn dropping_admin_cancelled_stream_does_not_count_as_client_disconnect() {
    let cancelled = Arc::new(Notify::new());
    let app = build_router_with_unauthenticated_admin(Box::new(CancellableStreamBackend {
        cancelled: cancelled.clone(),
    }));
    let request_id = "admin-cancel-then-drop";
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("x-request-id", request_id)
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

    let cancel_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/admin/requests/{request_id}/cancel"))
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("cancel response");
    assert_eq!(cancel_response.status(), StatusCode::OK);
    tokio::time::timeout(Duration::from_millis(300), cancelled.notified())
        .await
        .expect("backend stream receives cancellation");

    drop(body);
    let metrics = wait_for_metrics(&app, |body| {
        body["cancelled_requests"] == 1
            && body["failed_requests"] == 1
            && body["scheduler_cancelled_requests"] == 1
    })
    .await;
    assert_eq!(metrics["stream_client_disconnected_requests"], 0);
    assert_eq!(metrics["scheduler_completed_requests"], 0);
}

struct PrefillProgressStreamBackend;

#[async_trait]
impl ModelBackend for PrefillProgressStreamBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "prefill-progress-stream")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Err(BackendError::other(
            "prefill progress HTTP test must use generate_stream".to_owned(),
        ))
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }

    fn generate_stream_with_cancel<'a>(
        &'a self,
        _request: BackendRequest,
        cancellation: CancellationToken,
    ) -> futures::stream::BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        if cancellation.is_cancelled() {
            return futures::stream::once(async { Err(BackendError::cancelled()) }).boxed();
        }
        async_stream::try_stream! {
            yield BackendStreamChunk {
                text: String::new(),
                tool_call_deltas: Vec::new(),
                prompt_tokens: 5,
                prompt_cached_tokens: Some(0),
                completion_tokens: 0,
                finish_reason: None,
                progress: Some(BackendStreamProgress::PrefillProgress {
                    chunk: 1,
                    total: 3,
                    tokens: 2,
                    total_tokens: 5,
                }),
            };
            yield BackendStreamChunk {
                text: "done".to_owned(),
                tool_call_deltas: Vec::new(),
                prompt_tokens: 5,
                prompt_cached_tokens: Some(0),
                completion_tokens: 1,
                finish_reason: Some(BackendFinishReason::Stop),
                progress: None,
            };
        }
        .boxed()
    }
}

struct OneByteDeltaStreamBackend {
    delay: Duration,
    fragments: Vec<&'static str>,
}

#[async_trait]
impl ModelBackend for OneByteDeltaStreamBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "one-byte-delta-stream")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Err(BackendError::other(
            "generate_stream_with_cancel should be used".to_owned(),
        ))
    }

    async fn generate_with_cancel(
        &self,
        _request: BackendRequest,
        _cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        Err(BackendError::other(
            "generate_stream_with_cancel should be used".to_owned(),
        ))
    }

    fn generate_stream_with_cancel<'a>(
        &'a self,
        _request: BackendRequest,
        _cancellation: CancellationToken,
    ) -> futures::stream::BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        let delay = self.delay;
        let fragments = self.fragments.clone();
        async_stream::try_stream! {
            for (index, fragment) in fragments.iter().enumerate() {
                tokio::time::sleep(delay).await;
                yield BackendStreamChunk {
                    text: (*fragment).to_owned(),
                    tool_call_deltas: Vec::new(),
                    prompt_tokens: 1,
                    prompt_cached_tokens: None,
                    completion_tokens: 1,
                    finish_reason: (index + 1 == fragments.len()).then_some(BackendFinishReason::Stop),
                    progress: None,
                };
            }
        }
        .boxed()
    }
}

struct SlowStructuredToolArgumentBackend {
    delay: Duration,
}

#[async_trait]
impl ModelBackend for SlowStructuredToolArgumentBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "slow-structured-tool-argument-stream")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Err(BackendError::other(
            "slow structured tool argument test must use generate_stream".to_owned(),
        ))
    }

    async fn generate_with_cancel(
        &self,
        _request: BackendRequest,
        _cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        Err(BackendError::other(
            "slow structured tool argument test must use generate_stream".to_owned(),
        ))
    }

    fn generate_stream_with_cancel<'a>(
        &'a self,
        _request: BackendRequest,
        cancellation: CancellationToken,
    ) -> futures::stream::BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        if cancellation.is_cancelled() {
            return futures::stream::once(async { Err(BackendError::cancelled()) }).boxed();
        }
        let delay = self.delay;
        async_stream::try_stream! {
            yield BackendStreamChunk {
                text: String::new(),
                tool_call_deltas: vec![http_structured_tool_delta(
                    0,
                    Some("call_read_1"),
                    Some("read"),
                    Some("{"),
                )],
                prompt_tokens: 1,
                prompt_cached_tokens: None,
                completion_tokens: 1,
                finish_reason: None,
                progress: None,
            };
            for arguments in [
                r#""path""#,
                r#":"#,
                r#""calculator.py""#,
                r#"}"#,
            ] {
                tokio::time::sleep(delay).await;
                yield BackendStreamChunk {
                    text: String::new(),
                    tool_call_deltas: vec![http_structured_tool_delta(
                        0,
                        None,
                        None,
                        Some(arguments),
                    )],
                    prompt_tokens: 1,
                    prompt_cached_tokens: None,
                    completion_tokens: 1,
                    finish_reason: (arguments == "}").then_some(BackendFinishReason::ToolCalls),
                    progress: None,
                };
            }
        }
        .boxed()
    }
}

struct PendingCancellableStreamBackend {
    cancelled: Arc<Notify>,
}

#[async_trait]
impl ModelBackend for PendingCancellableStreamBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "pending-cancellable-stream")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Err(BackendError::other(
            "generate_stream_with_cancel should be used".to_owned(),
        ))
    }

    async fn generate_with_cancel(
        &self,
        _request: BackendRequest,
        _cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        Err(BackendError::other(
            "generate_stream_with_cancel should be used".to_owned(),
        ))
    }

    fn generate_stream_with_cancel<'a>(
        &'a self,
        _request: BackendRequest,
        cancellation: CancellationToken,
    ) -> futures::stream::BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        let cancelled = self.cancelled.clone();
        tokio::spawn(async move {
            cancellation.cancelled().await;
            cancelled.notify_waiters();
        });
        futures::stream::pending().boxed()
    }
}

async fn wait_for_metrics<F>(app: &Router, predicate: F) -> Value
where
    F: Fn(&Value) -> bool,
{
    tokio::time::timeout(Duration::from_millis(500), async {
        loop {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri("/admin/metrics")
                        .body(Body::empty())
                        .expect("request builds"),
                )
                .await
                .expect("metrics response");
            assert_eq!(response.status(), StatusCode::OK);
            let body = body_json(response.into_body()).await;
            if predicate(&body) {
                return body;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("metrics matched predicate")
}

struct StructuredToolDeltaHttpBackend;

#[async_trait]
impl ModelBackend for StructuredToolDeltaHttpBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "structured-tool-delta-http")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Err(BackendError::other(
            "structured tool delta HTTP test must use generate_stream".to_owned(),
        ))
    }

    async fn generate_with_cancel(
        &self,
        _request: BackendRequest,
        _cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        Err(BackendError::other(
            "structured tool delta HTTP test must use generate_stream".to_owned(),
        ))
    }

    fn generate_stream_with_cancel<'a>(
        &'a self,
        _request: BackendRequest,
        cancellation: CancellationToken,
    ) -> futures::stream::BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        if cancellation.is_cancelled() {
            return futures::stream::once(async { Err(BackendError::cancelled()) }).boxed();
        }
        async_stream::try_stream! {
            yield BackendStreamChunk {
                text: String::new(),
                tool_call_deltas: vec![http_structured_tool_delta(
                    0,
                    Some("call_read_1"),
                    Some("read"),
                    Some(r#"{"path":"#),
                )],
                prompt_tokens: 1,
                prompt_cached_tokens: None,
                completion_tokens: 1,
                finish_reason: None,
                progress: None,
            };
            yield BackendStreamChunk {
                text: String::new(),
                tool_call_deltas: vec![http_structured_tool_delta(
                    0,
                    None,
                    None,
                    Some(r#""calculator.py"}"#),
                )],
                prompt_tokens: 1,
                prompt_cached_tokens: None,
                completion_tokens: 1,
                finish_reason: Some(BackendFinishReason::ToolCalls),
                progress: None,
            };
        }
        .boxed()
    }
}

fn http_structured_tool_delta(
    index: u32,
    id: Option<&str>,
    name: Option<&str>,
    arguments: Option<&str>,
) -> BackendToolCallDelta {
    BackendToolCallDelta {
        index,
        id: id.map(str::to_owned),
        call_type: id.map(|_| BackendToolCallType::Function),
        function: (name.is_some() || arguments.is_some()).then(|| BackendToolCallFunctionDelta {
            name: name.map(str::to_owned),
            arguments: arguments.map(str::to_owned),
        }),
    }
}

fn sse_json_frames(body: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).expect("SSE data frame is JSON"))
        .collect()
}
