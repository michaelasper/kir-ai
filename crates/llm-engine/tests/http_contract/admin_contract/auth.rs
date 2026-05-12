use super::*;

#[tokio::test]
async fn admin_metrics_rejects_requests_when_token_unset_by_default() {
    let response = build_router_with_backend_and_options(
        Box::new(StaticBackend {
            text: "unused".to_owned(),
        }),
        EngineOptions::default(),
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
async fn admin_metrics_allow_requests_when_unauthenticated_admin_is_explicitly_enabled() {
    let response = build_router_with_backend_and_options_allowing_unauthenticated_admin(
        Box::new(StaticBackend {
            text: "unused".to_owned(),
        }),
        EngineOptions::default(),
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

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn admin_metrics_rejects_plain_token_without_bearer_scheme() {
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
            .header("authorization", "secret-admin-token")
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

#[test]
fn engine_options_reject_remote_http_hub_endpoint_with_token() {
    let result = build_router_with_backend_and_options(
        Box::new(llm_backend::ProtocolTestBackend::new(
            llm_engine::DEFAULT_MODEL_ID,
            "ok",
        )),
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
        Box::new(llm_backend::ProtocolTestBackend::new(
            llm_engine::DEFAULT_MODEL_ID,
            "ok",
        )),
        EngineOptions {
            hub_endpoint: Some("http://127.0.0.1:8080".to_owned()),
            hf_token: Some("hf_secret".to_owned()),
            ..EngineOptions::default()
        },
    );

    assert!(result.is_ok());
}
