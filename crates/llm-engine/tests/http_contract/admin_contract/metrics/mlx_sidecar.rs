use super::*;

#[tokio::test]
async fn admin_metrics_report_mlx_sidecar_activity_after_generation() {
    let server = FakeMlxServer::start(
        "data: {\"choices\":[{\"text\":\"admin mlx\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":3}}\n\ndata: [DONE]\n\n",
    );
    let backend = llm_engine::MlxBackend::open_with_options(
        "local-mlx",
        server.snapshot_path(),
        llm_engine::MlxBackendOptions {
            endpoint: server.endpoint(),
            family: Some(llm_models::ModelFamily::Qwen),
            ..llm_engine::MlxBackendOptions::default()
        },
    )
    .await
    .expect("MLX backend opens");
    let app = build_router_with_unauthenticated_admin(Box::new(backend));

    let before = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/admin/metrics")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("initial metrics response");
    assert_eq!(before.status(), StatusCode::OK);
    let before = body_json(before.into_body()).await;

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "local-mlx",
                        "prompt": "hello mlx",
                        "max_tokens": 8
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("MLX completion response");
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["choices"][0]["text"], "admin mlx");

    let after = app
        .oneshot(
            Request::builder()
                .uri("/admin/metrics")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("metrics response after MLX request");
    assert_eq!(after.status(), StatusCode::OK);
    let after = body_json(after.into_body()).await;

    assert_metric_incremented(
        &before,
        &after,
        &["backend_metrics", "mlx", "requests_total"],
        1,
    );
    assert_metric_incremented(
        &before,
        &after,
        &["backend_metrics", "mlx", "successful_requests"],
        1,
    );
    assert_metric_incremented(
        &before,
        &after,
        &["backend_metrics", "mlx", "completion_requests"],
        1,
    );
    assert_metric_incremented(
        &before,
        &after,
        &["backend_metrics", "mlx", "stream_chunks"],
        1,
    );
    assert_metric_incremented(
        &before,
        &after,
        &["backend_metrics", "mlx", "request_latency_ms", "count"],
        1,
    );
    assert_metric_incremented(
        &before,
        &after,
        &[
            "backend_metrics",
            "mlx",
            "upstream_request_latency_ms",
            "count",
        ],
        1,
    );
    assert_metric_incremented(
        &before,
        &after,
        &[
            "backend_metrics",
            "mlx",
            "blocking_upstream_request_latency_ms",
            "count",
        ],
        1,
    );
}

#[tokio::test]
async fn admin_metrics_report_successful_streamed_mlx_generation() {
    let server = FakeMlxServer::start(
        "data: {\"choices\":[{\"text\":\"streamed mlx\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":3,\"prompt_tokens_details\":{\"cached_tokens\":5}}}\n\ndata: [DONE]\n\n",
    );
    let backend = llm_engine::MlxBackend::open_with_options(
        "local-mlx",
        server.snapshot_path(),
        llm_engine::MlxBackendOptions {
            endpoint: server.endpoint(),
            family: Some(llm_models::ModelFamily::Qwen),
            ..llm_engine::MlxBackendOptions::default()
        },
    )
    .await
    .expect("MLX backend opens");
    let app = build_router_with_unauthenticated_admin(Box::new(backend));

    let before = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/admin/metrics")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("initial metrics response");
    assert_eq!(before.status(), StatusCode::OK);
    let before = body_json(before.into_body()).await;

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "local-mlx",
                        "prompt": "hello mlx",
                        "max_tokens": 8,
                        "stream": true,
                        "stream_options": {"include_usage": true}
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("streaming MLX completion response");
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_text(response.into_body()).await;
    assert!(body.contains("streamed mlx"));
    assert!(
        body.contains("\"prompt_tokens_details\":{\"cached_tokens\":5}"),
        "stream usage chunk should expose cached prompt tokens: {body}"
    );

    let after_response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/admin/metrics")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("metrics response after MLX stream");
    assert_eq!(after_response.status(), StatusCode::OK);
    let after = body_json(after_response.into_body()).await;

    assert_metric_incremented(
        &before,
        &after,
        &["backend_metrics", "mlx", "requests_total"],
        1,
    );
    assert_metric_incremented(
        &before,
        &after,
        &["backend_metrics", "mlx", "successful_requests"],
        1,
    );
    assert_metric_incremented(
        &before,
        &after,
        &["backend_metrics", "mlx", "completion_requests"],
        1,
    );
    assert_metric_incremented(
        &before,
        &after,
        &["backend_metrics", "mlx", "stream_chunks"],
        1,
    );
    assert_metric_incremented(
        &before,
        &after,
        &[
            "backend_metrics",
            "mlx",
            "streaming_upstream_request_latency_ms",
            "count",
        ],
        1,
    );
    assert_metric_unchanged(
        &before,
        &after,
        &["backend_metrics", "mlx", "failed_requests"],
    );
    assert_metric_unchanged(
        &before,
        &after,
        &["backend_metrics", "mlx", "dropped_requests"],
    );
    assert_eq!(after["tokens"]["prompt_tokens"], 2);
    assert_eq!(after["tokens"]["completion_tokens"], 3);
    assert_eq!(after["tokens"]["total_tokens"], 5);
    assert_eq!(after["tokens"]["prompt_tokens_details"]["cached_tokens"], 5);

    let mlx_response = app
        .oneshot(
            Request::builder()
                .uri("/admin/metrics.mlx")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("MLX metrics response");
    assert_eq!(mlx_response.status(), StatusCode::OK);
    let mlx = body_json(mlx_response.into_body()).await;
    assert_metric_incremented(
        &before["backend_metrics"]["mlx"],
        &mlx,
        &["stream_response_headers_ms", "count"],
        1,
    );
    assert_metric_incremented(
        &before["backend_metrics"]["mlx"],
        &mlx,
        &["stream_first_upstream_byte_ms", "count"],
        1,
    );
    assert_metric_incremented(
        &before["backend_metrics"]["mlx"],
        &mlx,
        &["stream_first_parsed_chunk_ms", "count"],
        1,
    );
    assert_metric_incremented(
        &before["backend_metrics"]["mlx"],
        &mlx,
        &["stream_upstream_complete_ms", "count"],
        1,
    );
}

#[tokio::test]
async fn admin_metrics_report_qwen_xml_mlx_streamed_tool_delta_latency() {
    let server = FakeMlxServer::start(
        "data: {\"choices\":[{\"delta\":{\"content\":\"<tool_call><function=read_file>\"},\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":4}}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"<parameter=path>Cargo.toml</parameter></function></tool_call>\"},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"completion_tokens\":5}}\n\ndata: [DONE]\n\n",
    );
    let backend = llm_engine::MlxBackend::open_with_options(
        "local-mlx",
        server.snapshot_path(),
        llm_engine::MlxBackendOptions {
            endpoint: server.endpoint(),
            family: Some(llm_models::ModelFamily::Qwen),
            tool_parser: llm_engine::MlxToolParserMode::QwenXml,
            ..llm_engine::MlxBackendOptions::default()
        },
    )
    .await
    .expect("MLX backend opens");
    let app = build_router_with_unauthenticated_admin(Box::new(backend));
    let request_id = "tool-stream-qwen-xml";

    let before_response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/admin/metrics")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("initial metrics response");
    assert_eq!(before_response.status(), StatusCode::OK);
    let before = body_json(before_response.into_body()).await;

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
                        "model": "local-mlx",
                        "messages": [{"role": "user", "content": "read Cargo.toml"}],
                        "tools": [{
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"path": {"type": "string"}},
                                    "required": ["path"]
                                }
                            }
                        }],
                        "tool_choice": {"type": "function", "function": {"name": "read_file"}},
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
    assert!(
        !body.contains("<tool_call>"),
        "Qwen XML must not leak to client SSE: {body}"
    );
    assert!(body.contains("\"tool_calls\":[{\"index\":0,\"id\":\"call_"));
    assert!(body.contains("\"type\":\"function\""));
    assert!(body.contains("\"name\":\"read_file\""));
    assert!(body.contains("\"finish_reason\":\"tool_calls\""));
    assert!(
        !body.contains("mlx_stream_timing"),
        "internal MLX timing progress must not be exposed to client SSE: {body}"
    );
    assert!(
        !body.contains("first_upstream_byte"),
        "internal MLX timing milestones must not be exposed to client SSE: {body}"
    );

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
    assert_metric_incremented(&before, &body, &["first_tool_delta_ms", "count"], 1);
    assert_metric_incremented(&before, &body, &["validated_tool_call_ms", "count"], 1);
    assert_metric_incremented(
        &before,
        &body,
        &[
            "backend_metrics",
            "mlx",
            "stream_first_tool_delta_ms",
            "count",
        ],
        1,
    );
    let recent = body["tool_stream"]["recent"]
        .as_array()
        .expect("tool stream observations");
    assert_eq!(body["tool_stream"]["capacity"], 128);
    assert_eq!(recent.len(), 1);
    let observation = &recent[0];
    assert_eq!(observation["request_id"], request_id);
    assert_eq!(observation["model"], "local-mlx");
    assert_eq!(observation["streamed"], true);
    assert!(observation["kir_first_tool_delta_ms"].as_u64().is_some());
    assert!(observation["tool_argument_assembly_ms"].as_u64().is_some());
    assert!(observation["tool_intent_fill_ms"].as_u64().is_some());
    assert!(observation["tool_schema_validation_ms"].as_u64().is_some());
    assert!(observation["validated_tool_call_ms"].as_u64().is_some());
    assert!(observation["mlx_response_headers_ms"].as_u64().is_some());
    assert!(observation["mlx_first_upstream_byte_ms"].as_u64().is_some());
    assert!(observation["mlx_first_parsed_chunk_ms"].as_u64().is_some());
    assert!(observation["mlx_first_tool_delta_ms"].as_u64().is_some());
    assert!(observation["mlx_upstream_complete_ms"].as_u64().is_some());
    assert!(observation["latency_ms"].as_u64().is_some());
    for forbidden in [
        "messages",
        "prompt",
        "tools",
        "tool_schema",
        "arguments",
        "request_body",
    ] {
        assert!(
            observation.get(forbidden).is_none(),
            "tool stream observation must not store sensitive field {forbidden}"
        );
    }

    let response = app
        .oneshot(
            Request::builder()
                .uri("/admin/metrics.tool_stream")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("tool stream metrics response");
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response.into_body()).await;
    let recent = body["recent"].as_array().expect("tool stream observations");
    assert_eq!(body["capacity"], 128);
    assert_eq!(recent.len(), 1);
    let observation = &recent[0];
    assert_eq!(observation["request_id"], request_id);
    assert_eq!(observation["model"], "local-mlx");
    assert_eq!(observation["streamed"], true);
    assert!(observation["kir_first_tool_delta_ms"].as_u64().is_some());
    assert!(observation["validated_tool_call_ms"].as_u64().is_some());
    assert!(observation["mlx_first_tool_delta_ms"].as_u64().is_some());
    assert!(observation["latency_ms"].as_u64().is_some());
    for forbidden in [
        "messages",
        "prompt",
        "tools",
        "tool_schema",
        "arguments",
        "request_body",
    ] {
        assert!(
            observation.get(forbidden).is_none(),
            "tool stream endpoint must not store sensitive field {forbidden}"
        );
    }
}
