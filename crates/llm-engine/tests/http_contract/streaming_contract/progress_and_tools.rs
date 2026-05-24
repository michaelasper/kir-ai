use super::*;

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
async fn chat_stream_prefill_progress_yields_scheduler_slot_to_decode_request() {
    let order = Arc::new(Mutex::new(Vec::new()));
    let app = build_router_with_unauthenticated_admin_and_options(
        Box::new(InterleavedPrefillStreamBackend {
            order: Arc::clone(&order),
        }),
        EngineOptions {
            concurrency_limit: 1,
            scheduler_queue_limit: 2,
            scheduler_prefill_threshold_chars: 16,
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
                        "messages": [{"role": "user", "content": "long-prefill xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}],
                        "stream": true
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("long stream response");
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    let mut head = String::new();
    tokio::time::timeout(Duration::from_millis(500), async {
        while !head.contains("\"type\":\"prefill_progress\"") {
            let chunk = body
                .next()
                .await
                .expect("long stream has prefill chunk")
                .expect("long stream chunk");
            head.push_str(std::str::from_utf8(&chunk).expect("utf8 long stream head"));
        }
    })
    .await
    .expect("prefill progress arrives");

    let short = tokio::spawn(app.clone().oneshot(chat_request_body("short-decode")));
    let long_tail = tokio::spawn(async move {
        let mut tail = String::new();
        while let Some(chunk) = body.next().await {
            tail.push_str(std::str::from_utf8(&chunk.expect("tail chunk")).expect("utf8 tail"));
        }
        tail
    });

    let short_response = tokio::time::timeout(Duration::from_millis(500), short)
        .await
        .expect("short decode completes while long prefill is yielded")
        .expect("short task")
        .expect("short response");
    assert_eq!(short_response.status(), StatusCode::OK);
    let short_body = body_text(short_response.into_body()).await;
    assert!(short_body.contains("short-decode"), "body: {short_body}");

    let tail = tokio::time::timeout(Duration::from_millis(500), long_tail)
        .await
        .expect("long stream resumes after short decode")
        .expect("long tail task");
    assert!(
        tail.contains("\"content\":\"long-finished\""),
        "tail: {tail}"
    );

    assert_eq!(
        order.lock().expect("order lock is not poisoned").as_slice(),
        ["long-prefill-start", "short-decode", "long-prefill-resume"]
    );
    let metrics = wait_for_metrics(&app, |body| {
        body["scheduler_prefill_yields"] == 1
            && body["scheduler_prefill_yields_to_decode"] == 1
            && body["scheduler_completed_requests"] == 2
    })
    .await;
    assert_eq!(metrics["scheduler_failed_requests"], 0);
    assert_eq!(metrics["scheduler_cancelled_requests"], 0);
    assert_eq!(metrics["scheduler_prefill_chunk_latency_ms"]["count"], 1);
    assert!(
        metrics["scheduler_prefill_chunk_latency_ms"]["max"]
            .as_f64()
            .expect("prefill chunk max latency is numeric")
            >= metrics["scheduler_prefill_chunk_latency_ms"]["min"]
                .as_f64()
                .expect("prefill chunk min latency is numeric")
    );
    assert_eq!(metrics["scheduler_decode_starvation_events"], 1);
    assert_eq!(metrics["scheduler_decode_starvation_waits"], 1);
    assert!(
        metrics["scheduler_decode_starvation_wait_ms_total"]
            .as_f64()
            .expect("decode starvation total wait is numeric")
            >= metrics["scheduler_decode_starvation_wait_ms_max"]
                .as_f64()
                .expect("decode starvation max wait is numeric")
    );
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
