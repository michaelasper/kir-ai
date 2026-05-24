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
async fn prometheus_metrics_include_paged_kv_cache_metrics() {
    let response = build_router_with_paged_kv_metrics()
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("prometheus metrics response");

    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        response
            .headers()
            .get("content-type")
            .expect("content-type header")
            .to_str()
            .expect("content type is string")
            .starts_with("text/plain"),
        "Prometheus endpoint should return text/plain"
    );
    let body = body_text(response.into_body()).await;

    assert!(
        body.contains("kir_paged_kv_cache_resident_blocks 2"),
        "paged-KV resident block metric should be rendered:\n{body}"
    );
    assert!(
        body.contains("kir_paged_kv_cache_shared_blocks 1"),
        "paged-KV sharing metric should be rendered:\n{body}"
    );
    assert!(
        body.contains("kir_paged_kv_cache_total_cow_clones 1"),
        "paged-KV COW metric should be rendered:\n{body}"
    );
    assert!(
        body.contains("kir_paged_kv_cache_blocks_evicted_lru 2"),
        "paged-KV eviction metric should be rendered:\n{body}"
    );
}

#[tokio::test]
async fn admin_kv_cache_returns_backend_snapshot_with_block_tables_and_refcounts() {
    let response = build_router_with_paged_kv_metrics()
        .oneshot(
            Request::builder()
                .uri("/admin/kv-cache")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("admin KV cache response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response.into_body()).await;

    assert_eq!(body["object"], "kv_cache.block_pool");
    assert_eq!(body["metrics"]["resident_blocks"], 2);
    assert_eq!(body["metrics"]["shared_blocks"], 1);
    assert_eq!(body["metrics"]["total_cow_clones"], 1);
    assert_eq!(body["sessions"][0]["session_id"], 7);
    assert_eq!(body["sessions"][0]["layers"][0]["layer"], 0);
    assert_eq!(
        body["sessions"][0]["layers"][0]["block_table"][0]["block_id"],
        42
    );
    assert_eq!(
        body["sessions"][0]["layers"][0]["block_table"][0]["ref_count"],
        2
    );
    assert_eq!(body["blocks"][0]["block_id"], 42);
    assert_eq!(body["blocks"][0]["ref_count"], 2);
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
async fn admin_tool_stream_metrics_requires_bearer_token_when_configured() {
    let app = build_router_with_backend_and_options(
        Box::new(StaticBackend {
            text: "unused".to_owned(),
        }),
        EngineOptions {
            admin_token: Some("secret-admin-token".to_owned()),
            ..EngineOptions::default()
        },
    )
    .expect("router builds");
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/admin/metrics.tool_stream")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("tool stream metrics response");

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["error"]["code"], "admin_auth_required");
    assert_eq!(body["error"]["phase"], "admin_auth");

    let response = app
        .oneshot(
            Request::builder()
                .uri("/admin/metrics.tool_stream")
                .header("authorization", "Bearer secret-admin-token")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("authenticated tool stream metrics response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["capacity"], 128);
    assert_eq!(
        body["recent"]
            .as_array()
            .expect("recent observations is array")
            .len(),
        0
    );
}
