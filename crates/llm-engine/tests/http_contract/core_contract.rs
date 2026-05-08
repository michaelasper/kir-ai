use super::*;

#[tokio::test]
async fn health_endpoint_reports_no_python_runtime() {
    let response = build_router()
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("health response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["runtime"], "rust");
    assert_eq!(body["python_runtime"], false);
}

#[tokio::test]
async fn models_endpoint_lists_qwen_alias() {
    let response = build_router()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("models response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["id"], "local-qwen36");
}

#[tokio::test]
async fn concurrent_generation_returns_model_overloaded() {
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let app = build_router_with_backend(Box::new(BlockingBackend {
        entered: entered.clone(),
        release: release.clone(),
    }));
    let first = tokio::spawn(app.clone().oneshot(chat_request_body("first")));
    entered.notified().await;

    let second = tokio::time::timeout(
        Duration::from_millis(200),
        app.clone().oneshot(chat_request_body("second")),
    )
    .await
    .expect("overloaded request returns promptly")
    .expect("second response");

    assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
    let body = body_json(second.into_body()).await;
    assert_eq!(body["error"]["code"], "model_overloaded");
    assert_eq!(body["error"]["phase"], "scheduler");
    assert_eq!(body["error"]["retryable"], true);

    release.notify_waiters();
    let first = first.await.expect("first task").expect("first response");
    assert_eq!(first.status(), StatusCode::OK);
}

#[tokio::test]
async fn backend_execution_errors_are_not_reported_as_missing_model() {
    let response = build_router_with_backend(Box::new(FailingBackend))
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "local-qwen36",
                        "messages": [{"role": "user", "content": "hello"}],
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("chat response");

    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["error"]["code"], "backend_execution_failed");
    assert_eq!(body["error"]["phase"], "decode");
    assert_eq!(body["error"]["retryable"], true);
}
