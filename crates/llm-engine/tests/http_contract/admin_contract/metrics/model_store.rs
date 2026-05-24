use super::*;

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
