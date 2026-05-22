use super::*;

#[tokio::test]
async fn admin_metrics_endpoint_reports_supplied_request_id() {
    let request_id = "admin-metrics-request-id";
    let response = build_router_with_protocol_test_backend()
        .oneshot(
            Request::builder()
                .uri("/admin/metrics")
                .header("x-request-id", request_id)
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("admin metrics response");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("x-request-id")
            .expect("request id header")
            .to_str()
            .expect("request id header is string"),
        request_id
    );
    let body = body_json(response.into_body()).await;
    assert!(body.as_object().is_some());
    assert_eq!(body["request_cache"]["capacity"], 128);
    assert_eq!(
        body["request_cache"]["recent"]
            .as_array()
            .expect("recent observations is array")
            .len(),
        0
    );
}

#[tokio::test]
async fn admin_metrics_endpoint_reports_request_id_when_auth_is_required() {
    let request_id = "admin-metrics-auth-request-id";
    let response = build_router_with_backend_and_options(
        Box::new(StaticBackend {
            text: "unused".to_owned(),
        }),
        EngineOptions {
            admin_token: Some("secret-admin-token".to_owned()),
            ..EngineOptions::default()
        },
    )
    .expect("router builds")
    .oneshot(
        Request::builder()
            .uri("/admin/metrics")
            .header("authorization", "Bearer wrong-token")
            .header("x-request-id", request_id)
            .body(Body::empty())
            .expect("request builds"),
    )
    .await
    .expect("admin metrics response");

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        response
            .headers()
            .get("x-request-id")
            .expect("request id header")
            .to_str()
            .expect("request id header is string"),
        request_id
    );
    let body = body_json(response.into_body()).await;
    assert_eq!(body["error"]["code"], "admin_auth_required");
    assert_eq!(body["error"]["phase"], "admin_auth");
}

#[tokio::test]
async fn admin_metrics_report_artifact_verification_failures() {
    let temp = tempfile::tempdir().expect("tempdir");
    let snapshot_path = write_verified_test_snapshot(temp.path()).await;
    tokio::fs::write(snapshot_path.join("config.json"), "bad")
        .await
        .expect("corrupt config");
    ModelStore::new(temp.path())
        .record_snapshot_alias(llm_engine::DEFAULT_MODEL_ID, &snapshot_path)
        .await
        .expect("snapshot alias");
    let app = build_router_with_unauthenticated_admin_and_options(
        Box::new(SnapshotMetadataBackend),
        EngineOptions {
            model_home: Some(temp.path().to_path_buf()),
            ..EngineOptions::default()
        },
    )
    .expect("router builds");

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!(
                    "/admin/models/{}/verify",
                    llm_engine::DEFAULT_MODEL_ID
                ))
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("admin model verify response");
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["error"]["code"], "model_integrity_failed");
    assert_eq!(body["error"]["phase"], "model_artifact_verification");
    let message = body["error"]["message"]
        .as_str()
        .expect("error message is string");
    let model_home = temp.path().to_string_lossy();
    assert!(
        !message.contains(model_home.as_ref()),
        "client error message leaked model home path: {message}"
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
    assert_eq!(body["artifact_verification_failures"], 1);
}

#[tokio::test]
async fn admin_metrics_report_model_pull_operations() {
    let temp = tempfile::tempdir().expect("tempdir");
    let (endpoint, server) = spawn_fake_hub_server(2);
    let app = build_router_with_backend_and_options(
        Box::new(StaticBackend {
            text: "unused".to_owned(),
        }),
        EngineOptions {
            admin_token: Some("secret-admin-token".to_owned()),
            model_home: Some(temp.path().to_path_buf()),
            hub_endpoint: Some(endpoint),
            ..EngineOptions::default()
        },
    )
    .expect("router builds");
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!(
                    "/admin/models/{}/pull",
                    llm_engine::DEFAULT_MODEL_ID
                ))
                .header("authorization", "Bearer secret-admin-token")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "repo_id": "Qwen/Qwen3.6-35B-A3B",
                        "revision": "main",
                        "profile": "qwen36-safetensors-bf16",
                        "metadata_only": true
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("admin pull response");
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/admin/metrics")
                .header("authorization", "Bearer secret-admin-token")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("metrics response");
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["model_pull_operations"], 1);
    assert_eq!(body["model_pull_successes"], 1);
    assert_eq!(body["model_pull_failures"], 0);
    assert_eq!(body["model_pull_bytes"], 2);
    server.join().expect("fake hub exits");
}

#[tokio::test]
async fn admin_metrics_report_model_store_usage() {
    let temp = tempfile::tempdir().expect("tempdir");
    write_verified_test_snapshot(temp.path()).await;
    let app = build_router_with_unauthenticated_admin_and_options(
        Box::new(StaticBackend {
            text: "unused".to_owned(),
        }),
        EngineOptions {
            model_home: Some(temp.path().to_path_buf()),
            ..EngineOptions::default()
        },
    )
    .expect("router builds");
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
    assert_eq!(body["model_store_snapshots"], 1);
    assert_eq!(body["model_store_bytes"], 2);

    std::fs::remove_dir_all(temp.path()).expect("remove model home after first metrics scrape");
    let response = app
        .oneshot(
            Request::builder()
                .uri("/admin/metrics")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("cached metrics response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["model_store_snapshots"], 1);
    assert_eq!(body["model_store_bytes"], 2);
}

#[tokio::test]
async fn admin_metrics_report_quarantined_model_store_usage() {
    let temp = tempfile::tempdir().expect("tempdir");
    let snapshot_path = write_verified_test_snapshot(temp.path()).await;
    ModelStore::new(temp.path())
        .quarantine_snapshot(&snapshot_path, "test corruption")
        .await
        .expect("snapshot quarantined");
    let app = build_router_with_unauthenticated_admin_and_options(
        Box::new(StaticBackend {
            text: "unused".to_owned(),
        }),
        EngineOptions {
            model_home: Some(temp.path().to_path_buf()),
            ..EngineOptions::default()
        },
    )
    .expect("router builds");
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
    assert_eq!(body["model_store_snapshots"], 0);
    assert_eq!(body["model_store_bytes"], 0);
    assert_eq!(body["model_store_quarantined_snapshots"], 1);
    assert_eq!(body["model_store_quarantined_bytes"], 2);
}

#[tokio::test]
async fn admin_metrics_report_inference_counts_and_tokens() {
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
                        "max_tokens": 8
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("chat response");
    assert_eq!(response.status(), StatusCode::OK);

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
    assert_eq!(body["requests_total"], 1);
    assert_eq!(body["successful_requests"], 1);
    assert_eq!(body["failed_requests"], 0);
    assert_eq!(body["streamed_requests"], 0);
    assert_eq!(body["stream_client_disconnected_requests"], 0);
    assert_eq!(body["stream_stalled_requests"], 0);
    assert_eq!(body["tokens"]["prompt_tokens"], 1);
    assert!(
        body["tokens"].get("prompt_tokens_details").is_none(),
        "cached prompt token details should be absent when no successful request reported them"
    );
    let completion_tokens = body["tokens"]["completion_tokens"]
        .as_u64()
        .expect("completion tokens are numeric");
    assert!(completion_tokens > 0);
    assert_eq!(body["tokens"]["total_tokens"], completion_tokens + 1);
    assert_eq!(body["request_latency_ms"]["count"], 1);
    assert_eq!(body["non_streamed_request_latency_ms"]["count"], 1);
    assert_eq!(body["streamed_request_latency_ms"]["count"], 0);
    assert!(
        body["request_latency_ms"]["max"]
            .as_f64()
            .expect("latency max is numeric")
            >= body["request_latency_ms"]["min"]
                .as_f64()
                .expect("latency min is numeric")
    );
    assert!(
        body["tokens_per_second"]
            .as_f64()
            .expect("tokens per second is numeric")
            > 0.0
    );
    assert!(
        body["backend_metrics"]["native_text_metal"]["kernels"].is_object(),
        "native text Metal metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["qwen"]["hits"].is_number(),
        "native text Qwen prefix cache hits are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["qwen"]["entries_scanned"].is_number(),
        "native text Qwen prefix cache lookup scan metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["qwen"]["namespace_entries_scanned"]
            .is_number(),
        "native text Qwen prefix cache namespace scan metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["qwen"]["hit_clone_bytes"].is_number(),
        "native text Qwen prefix cache hit clone bytes are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["qwen"]["prefill_chunks"].is_number(),
        "native text Qwen prefix cache prefill chunk metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["qwen"]["prefill_tokens"].is_number(),
        "native text Qwen prefix cache prefill token metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["qwen"]["hit_tokens"].is_number(),
        "native text Qwen prefix cache hit token metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["qwen"]["miss_tokens"].is_number(),
        "native text Qwen prefix cache miss token metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["qwen"]["avoided_prefill_tokens"]
            .is_number(),
        "native text Qwen prefix cache avoided prefill token metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["gemma"]["hits"].is_number(),
        "native text Gemma prefix cache hits are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["gemma"]["entries_scanned"].is_number(),
        "native text Gemma prefix cache lookup scan metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["gemma"]["namespace_entries_scanned"]
            .is_number(),
        "native text Gemma prefix cache namespace scan metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["gemma"]["hit_clone_bytes"].is_number(),
        "native text Gemma prefix cache hit clone bytes are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["gemma"]["prefill_chunks"].is_number(),
        "native text Gemma prefix cache prefill chunk metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["gemma"]["prefill_tokens"].is_number(),
        "native text Gemma prefix cache prefill token metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["gemma"]["hit_tokens"].is_number(),
        "native text Gemma prefix cache hit token metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["gemma"]["miss_tokens"].is_number(),
        "native text Gemma prefix cache miss token metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_text_prefix_cache"]["gemma"]["avoided_prefill_tokens"]
            .is_number(),
        "native text Gemma prefix cache avoided prefill token metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["mlx"]["requests_total"].is_number(),
        "MLX sidecar request metrics are exposed"
    );
    assert!(body["backend_metrics"]["mlx"]["successful_requests"].is_number());
    assert!(body["backend_metrics"]["mlx"]["failed_requests"].is_number());
    assert!(body["backend_metrics"]["mlx"]["stream_chunks"].is_number());
    assert!(
        body["backend_metrics"]["mlx"]["request_latency_ms"]["count"].is_number(),
        "MLX sidecar latency metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["mlx"]["upstream_request_latency_ms"]["count"].is_number(),
        "MLX upstream sidecar latency metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["mlx"]["blocking_upstream_request_latency_ms"]["count"].is_number(),
        "MLX blocking upstream sidecar latency metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["mlx"]["streaming_upstream_request_latency_ms"]["count"]
            .is_number(),
        "MLX streaming upstream sidecar latency metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_qwen_metal"]["kernels"].is_object(),
        "native Qwen Metal compatibility metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_qwen_metal"]["bf16_matrix_cache"].is_object(),
        "native Qwen Metal BF16 matrix cache metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_qwen_metal"]["bf16_matrix_cache"]["resident_bytes"]
            .is_number(),
        "native Qwen Metal BF16 matrix cache residency is exposed"
    );
    assert!(
        body["backend_metrics"]["native_qwen_metal"]["kv_cache"]["resident_bytes"].is_number(),
        "native Qwen Metal KV cache residency is exposed"
    );
    assert!(
        body["backend_metrics"]["native_qwen_metal"]["linear_attention_cache"]["resident_bytes"]
            .is_number(),
        "native Qwen Metal linear cache residency is exposed"
    );
    assert!(
        body["backend_metrics"]["native_qwen_prefix_cache"].is_object(),
        "native Qwen shared prefix cache metrics are exposed"
    );
    assert!(
        body["backend_metrics"]["native_qwen_prefix_cache"]["hits"].is_number(),
        "native Qwen prefix cache hits are exposed"
    );
    assert!(
        body["backend_metrics"]["native_qwen_prefix_cache"]["misses"].is_number(),
        "native Qwen prefix cache misses are exposed"
    );
    assert!(
        body["backend_metrics"]["native_qwen_prefix_cache"]["evictions"].is_number(),
        "native Qwen prefix cache evictions are exposed"
    );
    assert!(
        body["backend_metrics"]["native_qwen_prefix_cache"]["resident_bytes"].is_number(),
        "native Qwen prefix cache residency is exposed"
    );
    assert!(
        body["backend_metrics"]["native_qwen_prefix_cache"]["entries_scanned"].is_number(),
        "native Qwen prefix cache lookup scan metrics are exposed through the legacy object"
    );
    assert!(
        body["backend_metrics"]["native_qwen_prefix_cache"]["namespace_entries_scanned"]
            .is_number(),
        "native Qwen prefix cache namespace scan metrics are exposed through the legacy object"
    );
    assert!(
        body["backend_metrics"]["native_qwen_prefix_cache"]["hit_clone_bytes"].is_number(),
        "native Qwen prefix cache hit clone bytes are exposed through the legacy object"
    );
    assert!(
        body["backend_metrics"]["native_qwen_prefix_cache"]["prefill_chunks"].is_number(),
        "native Qwen prefix cache prefill chunk metrics are exposed through the legacy object"
    );
    assert!(
        body["backend_metrics"]["native_qwen_prefix_cache"]["prefill_tokens"].is_number(),
        "native Qwen prefix cache prefill token metrics are exposed through the legacy object"
    );
    assert!(
        body["backend_metrics"]["native_qwen_prefix_cache"]["hit_tokens"].is_number(),
        "native Qwen prefix cache hit token metrics are exposed through the legacy object"
    );
    assert!(
        body["backend_metrics"]["native_qwen_prefix_cache"]["miss_tokens"].is_number(),
        "native Qwen prefix cache miss token metrics are exposed through the legacy object"
    );
    assert!(
        body["backend_metrics"]["native_qwen_prefix_cache"]["avoided_prefill_tokens"].is_number(),
        "native Qwen prefix cache avoided prefill token metrics are exposed through the legacy object"
    );
}

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
    assert!(body.contains("\"tool_calls\":[{\"index\":0,\"id\":\"call_0\",\"type\":\"function\""));
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
}

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
async fn admin_metrics_requires_bearer_token_when_configured() {
    let response = build_router_with_backend_and_options(
        Box::new(StaticBackend {
            text: "unused".to_owned(),
        }),
        EngineOptions {
            admin_token: Some("secret-admin-token".to_owned()),
            ..EngineOptions::default()
        },
    )
    .expect("router builds")
    .oneshot(
        Request::builder()
            .uri("/admin/metrics")
            .body(Body::empty())
            .expect("request builds"),
    )
    .await
    .expect("admin metrics response");

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["error"]["code"], "admin_auth_required");
    assert_eq!(body["error"]["phase"], "admin_auth");
}

#[tokio::test]
async fn admin_metrics_accepts_configured_bearer_token() {
    let response = build_router_with_backend_and_options(
        Box::new(StaticBackend {
            text: "unused".to_owned(),
        }),
        EngineOptions {
            admin_token: Some("secret-admin-token".to_owned()),
            ..EngineOptions::default()
        },
    )
    .expect("router builds")
    .oneshot(
        Request::builder()
            .uri("/admin/metrics")
            .header("authorization", "Bearer secret-admin-token")
            .body(Body::empty())
            .expect("request builds"),
    )
    .await
    .expect("admin metrics response");

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn admin_metrics_request_cache_records_buffered_observation() {
    let app = build_router_with_unauthenticated_admin(Box::new(CachedUsageBackend {
        prompt_tokens: 100,
        prompt_cached_tokens: Some(64),
        completion_tokens: 5,
    }));
    let request_id = "cache-buffered";
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
                        "max_tokens": 8
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("chat response");
    assert_eq!(response.status(), StatusCode::OK);

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
    let recent = body["request_cache"]["recent"]
        .as_array()
        .expect("recent observations");
    assert_eq!(recent.len(), 1);
    let observation = &recent[0];
    assert_eq!(observation["request_id"], request_id);
    assert_eq!(observation["model"], llm_engine::DEFAULT_MODEL_ID);
    assert_eq!(observation["streamed"], false);
    assert_eq!(observation["prompt_tokens"], 100);
    assert_eq!(observation["cached_tokens"], 64);
    assert_eq!(observation["uncached_tokens"], 36);
    assert_eq!(observation["cache_status"], "partial");
    assert!(observation["latency_ms"].as_u64().is_some());
}

#[tokio::test]
async fn admin_metrics_request_cache_records_stable_prefix_identity_for_repeat_agent_turns() {
    let app = build_router_with_unauthenticated_admin(Box::new(CacheTransitionBackend {
        prompt_tokens: 100,
        warm_cached_tokens: 100,
        completion_tokens: 5,
        calls: AtomicUsize::new(0),
    }));
    let chat_body = |request_id: &str, user_content: &str| {
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header("x-request-id", request_id)
            .body(Body::from(
                json!({
                    "model": llm_engine::DEFAULT_MODEL_ID,
                    "messages": [
                        {"role": "system", "content": "You are a coding agent."},
                        {"role": "user", "content": user_content}
                    ],
                    "tools": [{
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "Lookup project context.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"}
                                },
                                "required": ["query"]
                            }
                        }
                    }],
                    "max_tokens": 8
                })
                .to_string(),
            ))
            .expect("request builds")
    };

    let first = app
        .clone()
        .oneshot(chat_body("agent-turn-1", "first turn"))
        .await
        .expect("first chat response");
    assert_eq!(first.status(), StatusCode::OK);
    let second = app
        .clone()
        .oneshot(chat_body("agent-turn-2", "second turn"))
        .await
        .expect("second chat response");
    assert_eq!(second.status(), StatusCode::OK);

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
    let recent = body["request_cache"]["recent"]
        .as_array()
        .expect("recent observations");
    assert_eq!(recent.len(), 2);
    let cold = &recent[0];
    let warm = &recent[1];

    assert_eq!(cold["request_id"], "agent-turn-1");
    assert_eq!(cold["cache_status"], "miss");
    assert_eq!(cold["cached_tokens"], 0);
    assert_eq!(cold["uncached_tokens"], 100);
    assert_eq!(warm["request_id"], "agent-turn-2");
    assert_eq!(warm["cache_status"], "hit");
    assert_eq!(warm["cached_tokens"], 100);
    assert_eq!(warm["uncached_tokens"], 0);
    assert_eq!(cold["cache_template_id"], "chatml/qwen/v1");
    assert_eq!(cold["model_family"], "qwen");
    assert_eq!(cold["stable_prefix_key"], warm["stable_prefix_key"]);
    assert_ne!(cold["prompt_hash"], warm["prompt_hash"]);
    assert!(
        cold["cache_key"]
            .as_str()
            .is_some_and(|hash| hash.starts_with("sha256:"))
    );
    assert!(
        cold["prompt_hash"]
            .as_str()
            .is_some_and(|hash| hash.starts_with("sha256:"))
    );
    assert!(
        cold["tool_schema_hash"]
            .as_str()
            .is_some_and(|hash| hash.starts_with("sha256:"))
    );
    assert!(
        cold["system_prompt_hash"]
            .as_str()
            .is_some_and(|hash| hash.starts_with("sha256:"))
    );
}

#[tokio::test]
async fn admin_metrics_request_cache_records_streamed_observation() {
    let app = build_router_with_unauthenticated_admin(Box::new(CachedUsageBackend {
        prompt_tokens: 50,
        prompt_cached_tokens: Some(50),
        completion_tokens: 3,
    }));
    let request_id = "cache-streamed";
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
                        "stream": true,
                        "stream_options": {"include_usage": true},
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
    assert!(body.contains("[DONE]"));

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
    let recent = body["request_cache"]["recent"]
        .as_array()
        .expect("recent observations");
    assert_eq!(recent.len(), 1);
    let observation = &recent[0];
    assert_eq!(observation["request_id"], request_id);
    assert_eq!(observation["streamed"], true);
    assert_eq!(observation["prompt_tokens"], 50);
    assert_eq!(observation["cached_tokens"], 50);
    assert_eq!(observation["uncached_tokens"], 0);
    assert_eq!(observation["cache_status"], "hit");
}

#[tokio::test]
async fn admin_metrics_report_active_and_cancelled_requests() {
    let entered = Arc::new(Notify::new());
    let cancelled = Arc::new(Notify::new());
    let app = build_router_with_unauthenticated_admin(Box::new(AdminCancellableBackend {
        entered: entered.clone(),
        cancelled: cancelled.clone(),
    }));
    let request_id = "metrics-cancel";
    let first = tokio::spawn(
        app.clone()
            .oneshot(chat_request_body_with_id("long running", request_id)),
    );
    entered.notified().await;

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
    assert_eq!(body["active_requests"], 1);
    assert_eq!(body["decode_requests"], 1);
    assert_eq!(body["prefill_requests"], 0);
    assert_eq!(body["cancelled_requests"], 0);

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
        .expect("backend receives cancellation");
    let first = first.await.expect("first task").expect("first response");
    assert_eq!(first.status(), StatusCode::REQUEST_TIMEOUT);

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
    assert_eq!(body["active_requests"], 0);
    assert_eq!(body["decode_requests"], 0);
    assert_eq!(body["prefill_requests"], 0);
    assert_eq!(body["cancelled_requests"], 1);
    assert_eq!(body["stream_client_disconnected_requests"], 0);
    assert_eq!(body["stream_stalled_requests"], 0);
    assert_eq!(body["failed_requests"], 1);
}

#[tokio::test]
async fn admin_metrics_report_no_progress_failures_and_queue_depth() {
    let app = build_router_with_unauthenticated_admin(Box::new(NoProgressBackend));
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
                        "messages": [{"role": "user", "content": "make progress"}]
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("chat response");
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let body = body_json(response.into_body()).await;
    assert_eq!(
        body["error"]["code"],
        "no_progress_empty_high_output_completion"
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
    assert_eq!(body["failed_requests"], 1);
    assert_eq!(body["no_progress_failures"], 1);
    assert_eq!(body["queued_requests"], 0);
}

#[tokio::test]
async fn admin_metrics_report_stream_prefill_phase_before_first_chunk() {
    let release = Arc::new(Semaphore::new(0));
    let app = build_router_with_unauthenticated_admin(Box::new(DelayedStreamBackend {
        release: release.clone(),
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

    let metrics = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/admin/metrics")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("metrics response");
    assert_eq!(metrics.status(), StatusCode::OK);
    let body = body_json(metrics.into_body()).await;
    assert_eq!(body["active_requests"], 1);
    assert_eq!(body["prefill_requests"], 1);
    assert_eq!(body["decode_requests"], 0);

    release.add_permits(1);
    let body = body_text(response.into_body()).await;
    assert!(body.contains("\"content\":\"released\""));

    let metrics = app
        .oneshot(
            Request::builder()
                .uri("/admin/metrics")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("metrics response");
    assert_eq!(metrics.status(), StatusCode::OK);
    let body = body_json(metrics.into_body()).await;
    assert_eq!(body["active_requests"], 0);
    assert_eq!(body["prefill_requests"], 0);
    assert_eq!(body["decode_requests"], 0);
}

fn assert_metric_incremented(before: &Value, after: &Value, path: &[&str], expected_delta: u64) {
    let before = metric_at_path(before, path);
    let after = metric_at_path(after, path);
    assert!(
        after >= before + expected_delta,
        "metric {path:?} should increase by at least {expected_delta}: before={before}, after={after}"
    );
}

fn assert_metric_unchanged(before: &Value, after: &Value, path: &[&str]) {
    let before = metric_at_path(before, path);
    let after = metric_at_path(after, path);
    assert_eq!(
        before, after,
        "metric {path:?} should not change: before={before}, after={after}"
    );
}

fn metric_at_path(metrics: &Value, path: &[&str]) -> u64 {
    let mut value = metrics;
    for segment in path {
        value = &value[*segment];
    }
    value.as_u64().expect("metric is an integer")
}

struct FakeMlxServer {
    endpoint: url::Url,
    snapshot: tempfile::TempDir,
    join: Option<thread::JoinHandle<()>>,
}

impl FakeMlxServer {
    fn start(response_body: &'static str) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind fake mlx server");
        let endpoint = url::Url::parse(&format!(
            "http://{}/v1",
            listener.local_addr().expect("local addr")
        ))
        .expect("endpoint url");
        let join = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept fake mlx request");
            read_http_request(&mut stream);
            write_http_response(&mut stream, "200 OK", response_body);
        });
        Self {
            endpoint,
            snapshot: tempfile::tempdir().expect("snapshot tempdir"),
            join: Some(join),
        }
    }

    fn endpoint(&self) -> url::Url {
        self.endpoint.clone()
    }

    fn snapshot_path(&self) -> &Path {
        self.snapshot.path()
    }
}

impl Drop for FakeMlxServer {
    fn drop(&mut self) {
        if let Some(join) = self.join.take() {
            join.join().expect("fake MLX server exits");
        }
    }
}

fn read_http_request(stream: &mut std::net::TcpStream) {
    let mut bytes = Vec::new();
    let mut buffer = [0_u8; 1024];
    let header_end;
    loop {
        let read = stream.read(&mut buffer).expect("read request");
        assert!(read > 0, "client closed before headers");
        bytes.extend_from_slice(&buffer[..read]);
        if let Some(index) = find_subsequence(&bytes, b"\r\n\r\n") {
            header_end = index + 4;
            break;
        }
    }
    let headers = String::from_utf8_lossy(&bytes[..header_end]);
    let content_length = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().expect("content length"))
        })
        .expect("content-length header");
    while bytes.len() < header_end + content_length {
        let read = stream.read(&mut buffer).expect("read body");
        assert!(read > 0, "client closed before body");
        bytes.extend_from_slice(&buffer[..read]);
    }
}

fn find_subsequence(bytes: &[u8], needle: &[u8]) -> Option<usize> {
    bytes
        .windows(needle.len())
        .position(|window| window == needle)
}

#[tokio::test]
async fn admin_metrics_report_stream_decode_phase_after_first_chunk() {
    let first = Arc::new(Notify::new());
    let finish = Arc::new(Notify::new());
    let app = build_router_with_unauthenticated_admin(Box::new(TwoStageStreamBackend {
        first: first.clone(),
        finish: finish.clone(),
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
    let mut stream = response.into_body().into_data_stream();

    first.notify_one();
    let mut seen = String::new();
    tokio::time::timeout(Duration::from_millis(300), async {
        while !seen.contains("\"content\":\"first\"") {
            let chunk = stream
                .next()
                .await
                .expect("body has chunk")
                .expect("body chunk");
            seen.push_str(std::str::from_utf8(&chunk).expect("utf8 sse"));
        }
    })
    .await
    .expect("first streamed content arrives");

    let metrics = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/admin/metrics")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("metrics response");
    assert_eq!(metrics.status(), StatusCode::OK);
    let body = body_json(metrics.into_body()).await;
    assert_eq!(body["active_requests"], 1);
    assert_eq!(body["prefill_requests"], 0);
    assert_eq!(body["decode_requests"], 1);

    finish.notify_one();
    while stream.next().await.is_some() {}
}
