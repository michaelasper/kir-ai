use super::*;

#[tokio::test]
async fn admin_models_endpoint_reports_ready_model() {
    let request_id = "admin-models-request-id";
    let response = build_router_with_protocol_test_backend()
        .oneshot(
            Request::builder()
                .uri("/admin/models")
                .header("x-request-id", request_id)
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("admin models response");

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
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["id"], llm_engine::DEFAULT_MODEL_ID);
    assert_eq!(body["data"][0]["status"], "ready");
    assert_eq!(body["data"][0]["python_runtime"], false);
}

#[tokio::test]
async fn admin_model_endpoint_reports_ready_model() {
    let request_id = "admin-model-request-id";
    let response = build_router_with_protocol_test_backend()
        .oneshot(
            Request::builder()
                .uri(format!("/admin/models/{}", llm_engine::DEFAULT_MODEL_ID))
                .header("x-request-id", request_id)
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("admin model response");

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
    assert_eq!(body["id"], llm_engine::DEFAULT_MODEL_ID);
    assert_eq!(body["status"], "ready");
    assert_eq!(body["python_runtime"], false);
}

#[tokio::test]
async fn admin_model_endpoint_reports_backend_artifact_identity() {
    let temp = tempfile::tempdir().expect("tempdir");
    let snapshot_path = write_verified_test_snapshot(temp.path()).await;
    ModelStore::new(temp.path())
        .record_snapshot_alias(llm_engine::DEFAULT_MODEL_ID, &snapshot_path)
        .await
        .expect("snapshot alias");
    let promoted_snapshot = ModelStore::new(temp.path())
        .resolve_snapshot_alias(llm_engine::DEFAULT_MODEL_ID)
        .await
        .expect("promoted snapshot");
    let response = build_router_with_unauthenticated_admin_and_options(
        Box::new(MetadataBackend),
        EngineOptions {
            model_home: Some(temp.path().to_path_buf()),
            ..EngineOptions::default()
        },
    )
    .expect("router builds")
    .oneshot(
        Request::builder()
            .uri(format!("/admin/models/{}", llm_engine::DEFAULT_MODEL_ID))
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
    assert_eq!(body["loader"], "native-metal");
    assert_eq!(body["snapshot_path"], snapshot_path.display().to_string());
    assert_eq!(body["manifest_digest"], promoted_snapshot.manifest_digest);
}

#[tokio::test]
async fn admin_model_endpoint_reports_mlx_backend_identity() {
    let temp = tempfile::tempdir().expect("tempdir");
    let snapshot_path = write_verified_mlx_test_snapshot(temp.path()).await;
    ModelStore::new(temp.path())
        .record_snapshot_alias("local-qwen36-mlx", &snapshot_path)
        .await
        .expect("snapshot alias");
    let response = build_router_with_unauthenticated_admin_and_options(
        Box::new(MlxMetadataBackend),
        EngineOptions {
            model_home: Some(temp.path().to_path_buf()),
            ..EngineOptions::default()
        },
    )
    .expect("router builds")
    .oneshot(
        Request::builder()
            .uri("/admin/models/local-qwen36-mlx")
            .body(Body::empty())
            .expect("request builds"),
    )
    .await
    .expect("admin model response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["backend"], "mlx");
    assert_eq!(body["family"], "qwen");
    assert_eq!(body["loader"], "mlx");
    assert_eq!(body["profile"], "qwen36-mlx-4bit");
    assert_eq!(body["snapshot_path"], snapshot_path.display().to_string());
}

#[tokio::test]
async fn admin_model_verify_endpoint_verifies_loaded_snapshot() {
    let temp = tempfile::tempdir().expect("tempdir");
    let snapshot_path = write_verified_test_snapshot(temp.path()).await;
    ModelStore::new(temp.path())
        .record_snapshot_alias(llm_engine::DEFAULT_MODEL_ID, &snapshot_path)
        .await
        .expect("snapshot alias");
    let response = build_router_with_unauthenticated_admin_and_options(
        Box::new(SnapshotMetadataBackend),
        EngineOptions {
            model_home: Some(temp.path().to_path_buf()),
            ..EngineOptions::default()
        },
    )
    .expect("router builds")
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

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["status"], "ok");
    assert_eq!(body["repo_id"], "Qwen/Qwen3.6-35B-A3B");
    assert_eq!(body["verified_files"], 1);
    assert_eq!(body["verified_bytes"], 2);
    assert_eq!(body["snapshot_path"], snapshot_path.display().to_string());
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
            .uri(format!(
                "/admin/models/{}/plan",
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

#[tokio::test]
async fn admin_model_plan_endpoint_preserves_request_validation_error_for_unknown_profile() {
    let response = build_router_with_protocol_test_backend()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!(
                    "/admin/models/{}/plan",
                    llm_engine::DEFAULT_MODEL_ID
                ))
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "repo_id": "Qwen/Qwen3.6-35B-A3B",
                        "profile": "unknown-profile"
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("admin plan response");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["error"]["code"], "invalid_request");
    assert_eq!(body["error"]["phase"], "request_validation");
    assert_eq!(body["error"]["retryable"], false);
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
async fn admin_model_pull_endpoint_serializes_concurrent_downloads() {
    let temp = tempfile::tempdir().expect("tempdir");
    let mut hub = spawn_blocking_fake_hub_server();
    let app = build_router_with_backend_and_options(
        Box::new(StaticBackend {
            text: "unused".to_owned(),
        }),
        EngineOptions {
            admin_token: Some("secret-admin-token".to_owned()),
            model_home: Some(temp.path().to_path_buf()),
            hub_endpoint: Some(hub.endpoint.clone()),
            ..EngineOptions::default()
        },
    )
    .expect("router builds");

    let first = tokio::spawn({
        let app = app.clone();
        async move {
            app.oneshot(admin_model_pull_request("TestOrg/FirstModel"))
                .await
                .expect("first admin pull response")
        }
    });
    assert_eq!(next_fake_hub_download(&mut hub).await, "TestOrg/FirstModel");

    let second = tokio::spawn({
        let app = app.clone();
        async move {
            app.oneshot(admin_model_pull_request("TestOrg/SecondModel"))
                .await
                .expect("second admin pull response")
        }
    });
    assert_no_fake_hub_download(&mut hub).await;

    hub.release_one_download();
    let first_response = first.await.expect("first admin pull task");
    assert_eq!(first_response.status(), StatusCode::OK);
    assert_eq!(
        next_fake_hub_download(&mut hub).await,
        "TestOrg/SecondModel"
    );

    hub.release_one_download();
    let second_response = second.await.expect("second admin pull task");
    assert_eq!(second_response.status(), StatusCode::OK);
    assert_eq!(hub.max_active_downloads(), 1);
    hub.server.join().expect("fake hub exits");
}

#[tokio::test]
async fn admin_model_endpoint_uses_stable_missing_model_error() {
    let request_id = "admin-model-not-found-request-id";
    let response = build_router_with_protocol_test_backend()
        .oneshot(
            Request::builder()
                .uri("/admin/models/not-loaded")
                .header("x-request-id", request_id)
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("admin model response");

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
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
    assert_eq!(body["error"]["code"], "model_not_found");
    assert_eq!(body["error"]["phase"], "model_resolution");
    assert_eq!(body["error"]["retryable"], false);
}

fn admin_model_pull_request(repo_id: &str) -> Request<Body> {
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
                "repo_id": repo_id,
                "revision": "main",
                "profile": "qwen36-safetensors-bf16",
                "metadata_only": true
            })
            .to_string(),
        ))
        .expect("request builds")
}

async fn next_fake_hub_download(hub: &mut BlockingFakeHubServer) -> String {
    tokio::time::timeout(Duration::from_secs(2), hub.download_started.recv())
        .await
        .expect("download starts before timeout")
        .expect("fake hub download channel is open")
}

async fn assert_no_fake_hub_download(hub: &mut BlockingFakeHubServer) {
    match tokio::time::timeout(Duration::from_millis(300), hub.download_started.recv()).await {
        Ok(Some(repo_id)) => panic!("concurrent model pull started downloading {repo_id}"),
        Ok(None) => panic!("fake hub download channel closed"),
        Err(_) => {}
    }
}
