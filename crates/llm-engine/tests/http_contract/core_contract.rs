use super::*;

#[tokio::test]
async fn health_endpoint_reports_no_python_runtime() {
    let response = build_router_with_protocol_test_backend()
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
    let response = build_router_with_protocol_test_backend()
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
async fn concurrent_generation_queues_once_then_returns_model_overloaded_when_full() {
    let order = Arc::new(Mutex::new(Vec::new()));
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Semaphore::new(0));
    let app = build_router_with_backend(Box::new(FairnessBackend {
        order: order.clone(),
        entered: entered.clone(),
        release: release.clone(),
    }));
    let first = tokio::spawn(app.clone().oneshot(chat_request_body("first-long")));
    wait_for_order_len(&order, 1).await;

    let second = tokio::spawn(app.clone().oneshot(chat_request_body("third-short")));
    let metrics = wait_for_metrics(&app, |body| body["queued_requests"] == 1).await;
    assert_eq!(metrics["queued_decode_requests"], 1);
    assert_eq!(metrics["queued_prefill_requests"], 0);

    let third = tokio::time::timeout(
        Duration::from_millis(200),
        app.clone().oneshot(chat_request_body("overflow")),
    )
    .await
    .expect("overloaded request returns promptly")
    .expect("third response");

    assert_eq!(third.status(), StatusCode::TOO_MANY_REQUESTS);
    let body = body_json(third.into_body()).await;
    assert_eq!(body["error"]["code"], "model_overloaded");
    assert_eq!(body["error"]["phase"], "scheduler");
    assert_eq!(body["error"]["retryable"], true);

    release.add_permits(1);
    let first = first.await.expect("first task").expect("first response");
    assert_eq!(first.status(), StatusCode::OK);
    wait_for_order_len(&order, 2).await;
    release.add_permits(1);
    let second = second.await.expect("second task").expect("second response");
    assert_eq!(second.status(), StatusCode::OK);
}

#[tokio::test]
async fn queued_generation_can_be_cancelled_before_scheduler_admission() {
    let order = Arc::new(Mutex::new(Vec::new()));
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Semaphore::new(0));
    let app = build_router_with_backend_and_options(
        Box::new(FairnessBackend {
            order: order.clone(),
            entered,
            release: release.clone(),
        }),
        EngineOptions {
            concurrency_limit: 1,
            scheduler_queue_limit: 1,
            ..EngineOptions::default()
        },
    )
    .expect("router builds");

    let first = tokio::spawn(
        app.clone()
            .oneshot(chat_request_body_with_id("first-long", "active-request")),
    );
    wait_for_order_len(&order, 1).await;

    let queued_request_id = "queued-cancel";
    let queued = tokio::spawn(
        app.clone()
            .oneshot(chat_request_body_with_id("third-short", queued_request_id)),
    );
    let metrics = wait_for_metrics(&app, |body| {
        body["queued_requests"] == 1 && body["active_requests"] == 2
    })
    .await;
    assert_eq!(metrics["queued_decode_requests"], 1);

    let cancel_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/admin/requests/{queued_request_id}/cancel"))
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("cancel response");
    assert_eq!(cancel_response.status(), StatusCode::OK);
    let cancel_body = body_json(cancel_response.into_body()).await;
    assert_eq!(cancel_body["status"], "cancelled");

    let queued = tokio::time::timeout(Duration::from_millis(300), queued)
        .await
        .expect("queued request returns after cancellation")
        .expect("queued task")
        .expect("queued response");
    assert_eq!(queued.status(), StatusCode::REQUEST_TIMEOUT);
    let body = body_json(queued.into_body()).await;
    assert_eq!(body["error"]["code"], "cancelled");
    assert_eq!(body["error"]["phase"], "scheduler");
    assert_eq!(order.lock().expect("order lock is not poisoned").len(), 1);

    let metrics = wait_for_metrics(&app, |body| {
        body["queued_requests"] == 0
            && body["active_requests"] == 1
            && body["scheduler_queued_cancelled_requests"] == 1
    })
    .await;
    assert_eq!(metrics["cancelled_requests"], 1);

    release.add_permits(1);
    let first = first.await.expect("first task").expect("first response");
    assert_eq!(first.status(), StatusCode::OK);
}

#[tokio::test]
async fn queued_generation_latency_includes_scheduler_wait() {
    let order = Arc::new(Mutex::new(Vec::new()));
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Semaphore::new(0));
    let app = build_router_with_backend_and_options(
        Box::new(FairnessBackend {
            order: order.clone(),
            entered,
            release: release.clone(),
        }),
        EngineOptions {
            concurrency_limit: 1,
            scheduler_queue_limit: 1,
            ..EngineOptions::default()
        },
    )
    .expect("router builds");

    let first = tokio::spawn(app.clone().oneshot(chat_request_body("first-long")));
    wait_for_order_len(&order, 1).await;
    let second = tokio::spawn(app.clone().oneshot(chat_request_body("third-short")));
    wait_for_metrics(&app, |body| body["queued_requests"] == 1).await;

    tokio::time::sleep(Duration::from_millis(60)).await;
    release.add_permits(1);
    let first = first.await.expect("first task").expect("first response");
    assert_eq!(first.status(), StatusCode::OK);

    wait_for_order_len(&order, 2).await;
    release.add_permits(1);
    let second = second.await.expect("second task").expect("second response");
    assert_eq!(second.status(), StatusCode::OK);

    let metrics = wait_for_metrics(&app, |body| body["request_latency_ms"]["count"] == 2).await;
    let min_latency = metrics["request_latency_ms"]["min"]
        .as_f64()
        .expect("minimum request latency is numeric");
    assert!(
        min_latency >= 40.0,
        "expected queued lifecycle latency to include scheduler wait, got {min_latency} ms"
    );
}

#[tokio::test]
async fn scheduler_prioritizes_decode_after_prefill_burst() {
    let order = Arc::new(Mutex::new(Vec::new()));
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Semaphore::new(0));
    let app = build_router_with_backend_and_options(
        Box::new(FairnessBackend {
            order: order.clone(),
            entered: entered.clone(),
            release: release.clone(),
        }),
        EngineOptions {
            concurrency_limit: 1,
            scheduler_queue_limit: 2,
            scheduler_prefill_threshold_chars: 16,
            ..EngineOptions::default()
        },
    )
    .expect("router builds");

    let first = tokio::spawn(app.clone().oneshot(chat_request_body(
        "first-long xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    )));
    wait_for_order_len(&order, 1).await;
    let second = tokio::spawn(app.clone().oneshot(chat_request_body(
        "second-long xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    )));
    let third = tokio::spawn(app.clone().oneshot(chat_request_body("third-short")));
    let metrics = wait_for_metrics(&app, |body| {
        body["queued_prefill_requests"] == 1 && body["queued_decode_requests"] == 1
    })
    .await;
    assert_eq!(metrics["active_prefill_requests"], 1);

    release.add_permits(1);
    let first = first.await.expect("first task").expect("first response");
    assert_eq!(first.status(), StatusCode::OK);
    wait_for_order_len(&order, 2).await;
    assert_eq!(
        order.lock().expect("order lock is not poisoned")[1],
        "third-short"
    );

    release.add_permits(1);
    let third = third.await.expect("third task").expect("third response");
    assert_eq!(third.status(), StatusCode::OK);
    wait_for_order_len(&order, 3).await;
    assert_eq!(
        order.lock().expect("order lock is not poisoned")[2],
        "second-long"
    );
    release.add_permits(1);
    let second = second.await.expect("second task").expect("second response");
    assert_eq!(second.status(), StatusCode::OK);
}

async fn wait_for_metrics<F>(app: &Router, predicate: F) -> Value
where
    F: Fn(&Value) -> bool,
{
    tokio::time::timeout(Duration::from_millis(500), async {
        loop {
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
            if predicate(&body) {
                return body;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("metrics matched predicate")
}

async fn wait_for_order_len(order: &Arc<Mutex<Vec<String>>>, expected_len: usize) {
    tokio::time::timeout(Duration::from_millis(500), async {
        loop {
            if order.lock().expect("order lock is not poisoned").len() >= expected_len {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("backend order reached expected length");
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
