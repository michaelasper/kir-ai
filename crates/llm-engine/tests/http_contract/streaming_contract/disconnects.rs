use super::*;

#[tokio::test]
async fn dropping_chat_stream_body_cancels_backend_stream() {
    let cancelled = Arc::new(Notify::new());
    let app = build_router_with_unauthenticated_admin(Box::new(CancellableStreamBackend {
        cancelled: cancelled.clone(),
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
    let mut body = response.into_body().into_data_stream();
    let mut seen = String::new();
    tokio::time::timeout(Duration::from_millis(300), async {
        while !seen.contains("\"content\":\"first\"") {
            let chunk = body
                .next()
                .await
                .expect("body has chunk")
                .expect("body chunk");
            seen.push_str(std::str::from_utf8(&chunk).expect("utf8 sse"));
        }
    })
    .await
    .expect("first backend chunk arrives");

    drop(body);
    tokio::time::timeout(Duration::from_millis(300), cancelled.notified())
        .await
        .expect("backend stream receives cancellation");
    let metrics = wait_for_metrics(&app, |body| {
        body["stream_client_disconnected_requests"] == 1
            && body["scheduler_cancelled_requests"] == 1
    })
    .await;
    assert_eq!(metrics["active_requests"], 0);
    assert_eq!(metrics["successful_requests"], 0);
    assert_eq!(metrics["failed_requests"], 1);
    assert_eq!(metrics["streamed_requests"], 0);
    assert_eq!(metrics["scheduler_completed_requests"], 0);
    assert_eq!(metrics["scheduler_failed_requests"], 0);
    assert_eq!(metrics["time_to_first_token_ms"]["count"], 1);
}

#[tokio::test]
async fn dropping_completion_stream_body_cancels_backend_stream() {
    let cancelled = Arc::new(Notify::new());
    let app = build_router_with_unauthenticated_admin(Box::new(CancellableStreamBackend {
        cancelled: cancelled.clone(),
    }));
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": llm_engine::DEFAULT_MODEL_ID,
                        "prompt": "hello",
                        "stream": true
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("stream response");

    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    let mut seen = String::new();
    tokio::time::timeout(Duration::from_millis(300), async {
        while !seen.contains("\"text\":\"first\"") {
            let chunk = body
                .next()
                .await
                .expect("body has chunk")
                .expect("body chunk");
            seen.push_str(std::str::from_utf8(&chunk).expect("utf8 sse"));
        }
    })
    .await
    .expect("first backend chunk arrives");

    drop(body);
    tokio::time::timeout(Duration::from_millis(300), cancelled.notified())
        .await
        .expect("backend stream receives cancellation");
    let metrics = wait_for_metrics(&app, |body| {
        body["stream_client_disconnected_requests"] == 1
            && body["scheduler_cancelled_requests"] == 1
    })
    .await;
    assert_eq!(metrics["active_requests"], 0);
    assert_eq!(metrics["successful_requests"], 0);
    assert_eq!(metrics["failed_requests"], 1);
    assert_eq!(metrics["streamed_requests"], 0);
    assert_eq!(metrics["scheduler_completed_requests"], 0);
    assert_eq!(metrics["scheduler_failed_requests"], 0);
    assert_eq!(metrics["time_to_first_token_ms"]["count"], 1);
}

#[tokio::test(start_paused = true)]
async fn dropping_chat_stream_before_first_token_records_client_disconnect_without_ttft() {
    let cancelled = Arc::new(Notify::new());
    let app = build_router_with_unauthenticated_admin(Box::new(PendingCancellableStreamBackend {
        cancelled: cancelled.clone(),
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
    let mut body = response.into_body().into_data_stream();
    let mut seen = String::new();
    tokio::time::timeout(Duration::from_millis(300), async {
        while !seen.contains("llm-engine-heartbeat") {
            let chunk = body
                .next()
                .await
                .expect("body has heartbeat")
                .expect("body chunk");
            seen.push_str(std::str::from_utf8(&chunk).expect("utf8 sse"));
        }
    })
    .await
    .expect("heartbeat arrives before backend output");

    drop(body);
    tokio::time::timeout(Duration::from_millis(300), cancelled.notified())
        .await
        .expect("backend stream receives cancellation");
    let metrics = wait_for_metrics(&app, |body| {
        body["stream_client_disconnected_requests"] == 1
            && body["scheduler_cancelled_requests"] == 1
    })
    .await;
    assert_eq!(metrics["active_requests"], 0);
    assert_eq!(metrics["prefill_requests"], 0);
    assert_eq!(metrics["decode_requests"], 0);
    assert_eq!(metrics["active_prefill_requests"], 0);
    assert_eq!(metrics["active_decode_requests"], 0);
    assert_eq!(metrics["scheduler_completed_requests"], 0);
    assert_eq!(metrics["time_to_first_token_ms"]["count"], 0);
}

#[tokio::test]
async fn dropping_admin_cancelled_stream_does_not_count_as_client_disconnect() {
    let cancelled = Arc::new(Notify::new());
    let app = build_router_with_unauthenticated_admin(Box::new(CancellableStreamBackend {
        cancelled: cancelled.clone(),
    }));
    let request_id = "admin-cancel-then-drop";
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
                        "stream": true
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("stream response");

    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    let mut seen = String::new();
    tokio::time::timeout(Duration::from_millis(300), async {
        while !seen.contains("\"content\":\"first\"") {
            let chunk = body
                .next()
                .await
                .expect("body has chunk")
                .expect("body chunk");
            seen.push_str(std::str::from_utf8(&chunk).expect("utf8 sse"));
        }
    })
    .await
    .expect("first backend chunk arrives");

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
        .expect("backend stream receives cancellation");

    drop(body);
    let metrics = wait_for_metrics(&app, |body| {
        body["cancelled_requests"] == 1
            && body["failed_requests"] == 1
            && body["scheduler_cancelled_requests"] == 1
    })
    .await;
    assert_eq!(metrics["stream_client_disconnected_requests"], 0);
    assert_eq!(metrics["scheduler_completed_requests"], 0);
}
