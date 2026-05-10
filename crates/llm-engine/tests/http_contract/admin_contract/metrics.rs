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
    let app = build_router_with_backend(Box::new(SnapshotMetadataBackend { snapshot_path }));

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
    let app = build_router_with_backend_and_options(
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
    let app = build_router_with_backend_and_options(
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
    let completion_tokens = body["tokens"]["completion_tokens"]
        .as_u64()
        .expect("completion tokens are numeric");
    assert!(completion_tokens > 0);
    assert_eq!(body["tokens"]["total_tokens"], completion_tokens + 1);
    assert_eq!(body["request_latency_ms"]["count"], 1);
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
        body["native_text_metal"]["kernels"].is_object(),
        "native text Metal metrics are exposed"
    );
    assert!(
        body["native_text_prefix_cache"]["qwen"]["hits"].is_number(),
        "native text Qwen prefix cache hits are exposed"
    );
    assert!(
        body["native_text_prefix_cache"]["gemma"]["hits"].is_number(),
        "native text Gemma prefix cache hits are exposed"
    );
    assert!(
        body["mlx"]["requests_total"].is_number(),
        "MLX sidecar request metrics are exposed"
    );
    assert!(body["mlx"]["successful_requests"].is_number());
    assert!(body["mlx"]["failed_requests"].is_number());
    assert!(body["mlx"]["stream_chunks"].is_number());
    assert!(
        body["mlx"]["request_latency_ms"]["count"].is_number(),
        "MLX sidecar latency metrics are exposed"
    );
    assert!(
        body["native_qwen_metal"]["kernels"].is_object(),
        "native Qwen Metal compatibility metrics are exposed"
    );
    assert!(
        body["native_qwen_metal"]["bf16_matrix_cache"].is_object(),
        "native Qwen Metal BF16 matrix cache metrics are exposed"
    );
    assert!(
        body["native_qwen_metal"]["bf16_matrix_cache"]["resident_bytes"].is_number(),
        "native Qwen Metal BF16 matrix cache residency is exposed"
    );
    assert!(
        body["native_qwen_metal"]["kv_cache"]["resident_bytes"].is_number(),
        "native Qwen Metal KV cache residency is exposed"
    );
    assert!(
        body["native_qwen_metal"]["linear_attention_cache"]["resident_bytes"].is_number(),
        "native Qwen Metal linear cache residency is exposed"
    );
    assert!(
        body["native_qwen_prefix_cache"].is_object(),
        "native Qwen shared prefix cache metrics are exposed"
    );
    assert!(
        body["native_qwen_prefix_cache"]["hits"].is_number(),
        "native Qwen prefix cache hits are exposed"
    );
    assert!(
        body["native_qwen_prefix_cache"]["misses"].is_number(),
        "native Qwen prefix cache misses are exposed"
    );
    assert!(
        body["native_qwen_prefix_cache"]["evictions"].is_number(),
        "native Qwen prefix cache evictions are exposed"
    );
    assert!(
        body["native_qwen_prefix_cache"]["resident_bytes"].is_number(),
        "native Qwen prefix cache residency is exposed"
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
        },
    )
    .expect("MLX backend opens");
    let app = build_router_with_backend(Box::new(backend));

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

    assert_metric_incremented(&before, &after, &["mlx", "requests_total"], 1);
    assert_metric_incremented(&before, &after, &["mlx", "successful_requests"], 1);
    assert_metric_incremented(&before, &after, &["mlx", "completion_requests"], 1);
    assert_metric_incremented(&before, &after, &["mlx", "stream_chunks"], 1);
    assert_metric_incremented(&before, &after, &["mlx", "request_latency_ms", "count"], 1);
}

#[tokio::test]
async fn admin_metrics_report_successful_streamed_mlx_generation() {
    let server = FakeMlxServer::start(
        "data: {\"choices\":[{\"text\":\"streamed mlx\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":3}}\n\ndata: [DONE]\n\n",
    );
    let backend = llm_engine::MlxBackend::open_with_options(
        "local-mlx",
        server.snapshot_path(),
        llm_engine::MlxBackendOptions {
            endpoint: server.endpoint(),
            family: Some(llm_models::ModelFamily::Qwen),
        },
    )
    .expect("MLX backend opens");
    let app = build_router_with_backend(Box::new(backend));

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
                        "stream": true
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

    let after = app
        .oneshot(
            Request::builder()
                .uri("/admin/metrics")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("metrics response after MLX stream");
    assert_eq!(after.status(), StatusCode::OK);
    let after = body_json(after.into_body()).await;

    assert_metric_incremented(&before, &after, &["mlx", "requests_total"], 1);
    assert_metric_incremented(&before, &after, &["mlx", "successful_requests"], 1);
    assert_metric_incremented(&before, &after, &["mlx", "completion_requests"], 1);
    assert_metric_incremented(&before, &after, &["mlx", "stream_chunks"], 1);
    assert_metric_unchanged(&before, &after, &["mlx", "failed_requests"]);
    assert_metric_unchanged(&before, &after, &["mlx", "dropped_requests"]);
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
async fn admin_metrics_report_active_and_cancelled_requests() {
    let entered = Arc::new(Notify::new());
    let cancelled = Arc::new(Notify::new());
    let app = build_router_with_backend(Box::new(AdminCancellableBackend {
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
    let app = build_router_with_backend(Box::new(NoProgressBackend));
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
    let app = build_router_with_backend(Box::new(DelayedStreamBackend {
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
    let app = build_router_with_backend(Box::new(TwoStageStreamBackend {
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
