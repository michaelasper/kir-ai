use super::*;

#[tokio::test]
async fn admin_models_endpoint_reports_ready_model() {
    let response = build_router_with_deterministic_test_backend()
        .oneshot(
            Request::builder()
                .uri("/admin/models")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("admin models response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["id"], "local-qwen36");
    assert_eq!(body["data"][0]["status"], "ready");
    assert_eq!(body["data"][0]["python_runtime"], false);
}

#[tokio::test]
async fn admin_model_endpoint_reports_ready_model() {
    let response = build_router_with_deterministic_test_backend()
        .oneshot(
            Request::builder()
                .uri("/admin/models/local-qwen36")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("admin model response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["id"], "local-qwen36");
    assert_eq!(body["status"], "ready");
    assert_eq!(body["python_runtime"], false);
}

#[tokio::test]
async fn admin_model_endpoint_reports_backend_artifact_identity() {
    let response = build_router_with_backend(Box::new(MetadataBackend))
        .oneshot(
            Request::builder()
                .uri("/admin/models/local-qwen36")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("admin model response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["backend"], "native-qwen");
    assert_eq!(body["repo_id"], "Qwen/Qwen3.6-35B-A3B");
    assert_eq!(
        body["resolved_commit"],
        "0123456789abcdef0123456789abcdef01234567"
    );
    assert_eq!(
        body["manifest_digest"],
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    );
}

#[tokio::test]
async fn admin_model_verify_endpoint_verifies_loaded_snapshot() {
    let temp = tempfile::tempdir().expect("tempdir");
    let snapshot_path = write_verified_test_snapshot(temp.path()).await;
    let response = build_router_with_backend(Box::new(SnapshotMetadataBackend {
        snapshot_path: snapshot_path.clone(),
    }))
    .oneshot(
        Request::builder()
            .method("POST")
            .uri("/admin/models/local-qwen36/verify")
            .body(Body::empty())
            .expect("request builds"),
    )
    .await
    .expect("admin model verify response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["status"], "ok");
    assert_eq!(body["repo_id"], "Qwen/Qwen3.6-35B-A3B");
    assert_eq!(body["verified_files"], 1);
    assert_eq!(body["verified_bytes"], 2);
    assert_eq!(body["snapshot_path"], snapshot_path.display().to_string());
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
                .uri("/admin/models/local-qwen36/verify")
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
async fn admin_model_plan_endpoint_returns_download_plan() {
    let (endpoint, server) = spawn_fake_hub_server(1);
    let response = build_router_with_backend_and_options(
        Box::new(StaticBackend {
            text: "unused".to_owned(),
        }),
        EngineOptions {
            admin_token: Some("secret-admin-token".to_owned()),
            hub_endpoint: Some(endpoint),
            ..EngineOptions::default()
        },
    )
    .expect("router builds")
    .oneshot(
        Request::builder()
            .method("POST")
            .uri("/admin/models/local-qwen36/plan")
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
    .expect("admin plan response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["repo_id"]["id"], "Qwen/Qwen3.6-35B-A3B");
    assert_eq!(
        body["resolved_commit"],
        "0123456789abcdef0123456789abcdef01234567"
    );
    assert_eq!(body["metadata_only"], true);
    assert_eq!(
        body["files_to_download"].as_array().expect("files").len(),
        1
    );
    assert_eq!(body["files_to_download"][0]["path"], "config.json");
    server.join().expect("fake hub exits");
}

#[test]
fn engine_options_reject_remote_http_hub_endpoint_with_token() {
    let result = build_router_with_backend_and_options(
        Box::new(llm_backend::DeterministicBackend::new("local-qwen36", "ok")),
        EngineOptions {
            hub_endpoint: Some("http://example.com".to_owned()),
            hf_token: Some("hf_secret".to_owned()),
            ..EngineOptions::default()
        },
    );

    let err = match result {
        Ok(_) => panic!("remote HTTP endpoint with HF_TOKEN should fail"),
        Err(err) => err,
    };
    assert!(
        err.to_string()
            .contains("refusing to send HF_TOKEN to non-HTTPS hub endpoint"),
        "error: {err}"
    );
}

#[test]
fn engine_options_allow_loopback_http_hub_endpoint_with_token() {
    let result = build_router_with_backend_and_options(
        Box::new(llm_backend::DeterministicBackend::new("local-qwen36", "ok")),
        EngineOptions {
            hub_endpoint: Some("http://127.0.0.1:8080".to_owned()),
            hf_token: Some("hf_secret".to_owned()),
            ..EngineOptions::default()
        },
    );

    assert!(result.is_ok());
}

#[tokio::test]
async fn admin_model_pull_endpoint_promotes_snapshot() {
    let temp = tempfile::tempdir().expect("tempdir");
    let (endpoint, server) = spawn_fake_hub_server(2);
    let response = build_router_with_backend_and_options(
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
    .expect("router builds")
    .oneshot(
        Request::builder()
            .method("POST")
            .uri("/admin/models/local-qwen36/pull")
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
    let body = body_json(response.into_body()).await;
    assert_eq!(
        body["resolved_commit"],
        "0123456789abcdef0123456789abcdef01234567"
    );
    assert_eq!(body["files"], 1);
    assert_eq!(
        body["manifest_digest"]
            .as_str()
            .expect("manifest digest")
            .len(),
        64
    );
    let snapshot_path = PathBuf::from(body["snapshot_path"].as_str().expect("snapshot path"));
    assert!(snapshot_path.join("config.json").is_file());
    assert!(snapshot_path.join("llm-engine-manifest.json").is_file());
    server.join().expect("fake hub exits");
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
                .uri("/admin/models/local-qwen36/pull")
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
async fn admin_model_endpoint_uses_stable_missing_model_error() {
    let response = build_router_with_deterministic_test_backend()
        .oneshot(
            Request::builder()
                .uri("/admin/models/not-loaded")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("admin model response");

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["error"]["code"], "model_not_found");
    assert_eq!(body["error"]["phase"], "model_resolution");
    assert_eq!(body["error"]["retryable"], false);
}

#[tokio::test]
async fn admin_metrics_report_inference_counts_and_tokens() {
    let app = build_router_with_deterministic_test_backend();
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "local-qwen36",
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
        body["native_qwen_metal"]["kernels"].is_object(),
        "native Qwen Metal metrics are exposed"
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
async fn admin_metrics_report_process_rss_bytes() {
    let response = build_router_with_deterministic_test_backend()
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
    let app = build_router_with_deterministic_test_backend();
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "local-qwen36",
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
                        "model": "local-qwen36",
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

#[tokio::test]
async fn admin_cancel_request_cancels_active_chat_generation() {
    let entered = Arc::new(Notify::new());
    let cancelled = Arc::new(Notify::new());
    let app = build_router_with_backend(Box::new(AdminCancellableBackend {
        entered: entered.clone(),
        cancelled: cancelled.clone(),
    }));
    let request_id = "cancel-me";
    let first = tokio::spawn(
        app.clone()
            .oneshot(chat_request_body_with_id("long running", request_id)),
    );
    entered.notified().await;

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
    let cancel_body = body_json(cancel_response.into_body()).await;
    assert_eq!(cancel_body["request_id"], request_id);
    assert_eq!(cancel_body["status"], "cancelled");
    tokio::time::timeout(Duration::from_millis(300), cancelled.notified())
        .await
        .expect("backend receives cancellation");

    let first = first.await.expect("first task").expect("first response");
    assert_eq!(first.status(), StatusCode::REQUEST_TIMEOUT);
    let body = body_json(first.into_body()).await;
    assert_eq!(body["error"]["code"], "cancelled");
    assert_eq!(body["error"]["phase"], "decode");
}

#[tokio::test]
async fn admin_cancel_request_cancels_active_text_completion() {
    let entered = Arc::new(Notify::new());
    let cancelled = Arc::new(Notify::new());
    let app = build_router_with_backend(Box::new(AdminCancellableBackend {
        entered: entered.clone(),
        cancelled: cancelled.clone(),
    }));
    let request_id = "cancel-completion";
    let first = tokio::spawn(
        app.clone()
            .oneshot(completion_request_body_with_id("long running", request_id)),
    );
    entered.notified().await;

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
    let body = body_json(first.into_body()).await;
    assert_eq!(body["error"]["code"], "cancelled");
}

#[tokio::test]
async fn admin_cancel_request_reports_unknown_request_id() {
    let response = build_router_with_deterministic_test_backend()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/admin/requests/not-active/cancel")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("cancel response");

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["error"]["code"], "request_not_found");
    assert_eq!(body["error"]["phase"], "cancellation");
}
