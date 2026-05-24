use super::*;

#[tokio::test]
async fn admin_metrics_report_process_rss_bytes() {
    let response = build_router_with_protocol_test_backend()
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
    assert!(
        body["process_rss_bytes"]
            .as_u64()
            .expect("process RSS is reported")
            > 0
    );
}

#[tokio::test]
async fn admin_metrics_report_stream_time_to_first_token() {
    let app = build_router_with_protocol_test_backend();
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
    let body = body_text(response.into_body()).await;
    assert!(body.to_ascii_lowercase().contains("rust"));

    let response = app
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
    assert_eq!(body["time_to_first_token_ms"]["count"], 1);
    assert_eq!(body["non_streamed_request_latency_ms"]["count"], 0);
    assert_eq!(body["streamed_request_latency_ms"]["count"], 1);
    assert_eq!(body["first_tool_delta_ms"]["count"], 0);
    assert_eq!(body["first_tool_delta_after_ttft_ms"]["count"], 0);
    assert_eq!(body["tool_argument_assembly_ms"]["count"], 0);
    assert_eq!(body["tool_intent_fill_ms"]["count"], 0);
    assert_eq!(body["tool_schema_validation_ms"]["count"], 0);
    assert_eq!(body["tool_finish_ms"]["count"], 0);
    assert_eq!(body["validated_tool_call_ms"]["count"], 0);
    assert!(
        body["time_to_first_token_ms"]["max"]
            .as_f64()
            .expect("ttft max is numeric")
            >= body["time_to_first_token_ms"]["min"]
                .as_f64()
                .expect("ttft min is numeric")
    );
}

#[tokio::test]
async fn admin_metrics_report_stream_tool_call_timing() {
    let app = build_router_with_protocol_test_backend();
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
                        "messages": [{"role": "user", "content": "lookup rust"}],
                        "tools": [{
                            "type": "function",
                            "function": {"name": "lookup", "parameters": {}}
                        }],
                        "tool_choice": "required",
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
    let body = body_text(response.into_body()).await;
    assert!(body.contains("\"finish_reason\":\"tool_calls\""));

    let response = app
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
    assert_eq!(body["first_tool_delta_ms"]["count"], 1);
    assert_eq!(body["first_tool_delta_after_ttft_ms"]["count"], 1);
    assert_eq!(body["tool_argument_assembly_ms"]["count"], 1);
    assert_eq!(body["tool_intent_fill_ms"]["count"], 1);
    assert_eq!(body["tool_schema_validation_ms"]["count"], 1);
    assert_eq!(body["tool_finish_ms"]["count"], 1);
    assert_eq!(body["validated_tool_call_ms"]["count"], 1);
    assert!(
        body["tool_finish_ms"]["max"]
            .as_f64()
            .expect("tool finish max is numeric")
            >= body["tool_schema_validation_ms"]["min"]
                .as_f64()
                .expect("schema validation min is numeric")
    );
    assert!(
        body["validated_tool_call_ms"]["max"]
            .as_f64()
            .expect("validated tool-call max is numeric")
            >= body["first_tool_delta_ms"]["min"]
                .as_f64()
                .expect("first tool delta min is numeric")
    );
}

#[tokio::test]
async fn admin_metrics_report_first_tool_delta_after_ttft_excludes_prefill_delay() {
    let first = Arc::new(Notify::new());
    let tool = Arc::new(Notify::new());
    let app = build_router_with_unauthenticated_admin(Box::new(TwoStageToolStreamBackend {
        first: first.clone(),
        tool: tool.clone(),
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
                        "messages": [{"role": "user", "content": "lookup rust"}],
                        "tools": [{
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"query": {"type": "string"}},
                                    "required": ["query"]
                                }
                            }
                        }],
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
    let mut stream = response.into_body().into_data_stream();

    tokio::time::sleep(Duration::from_millis(60)).await;
    first.notify_one();
    let mut seen = String::new();
    tokio::time::timeout(Duration::from_millis(500), async {
        while !seen.contains("decode-start") {
            let chunk = stream
                .next()
                .await
                .expect("body has first chunk")
                .expect("body chunk");
            seen.push_str(std::str::from_utf8(&chunk).expect("utf8 sse"));
        }
    })
    .await
    .expect("first streamed content arrives");

    tokio::time::sleep(Duration::from_millis(10)).await;
    tool.notify_one();
    while let Some(chunk) = stream.next().await {
        seen.push_str(std::str::from_utf8(&chunk.expect("body chunk")).expect("utf8 sse"));
    }
    assert!(
        seen.contains("\"tool_calls\""),
        "tool delta streamed: {seen}"
    );

    let response = app
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
    assert_eq!(body["first_tool_delta_ms"]["count"], 1);
    assert_eq!(body["first_tool_delta_after_ttft_ms"]["count"], 1);
    let first_tool_delta_ms = body["first_tool_delta_ms"]["max"]
        .as_f64()
        .expect("first tool delta max is numeric");
    let after_ttft_ms = body["first_tool_delta_after_ttft_ms"]["max"]
        .as_f64()
        .expect("first tool delta after TTFT max is numeric");
    assert!(
        after_ttft_ms < first_tool_delta_ms,
        "after-TTFT metric should exclude the simulated prefill delay: after={after_ttft_ms}, e2e={first_tool_delta_ms}"
    );
}
