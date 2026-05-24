use super::*;

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
