use super::*;

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
async fn admin_cancel_request_wins_over_late_backend_error() {
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Semaphore::new(0));
    let app = build_router_with_backend(Box::new(AdminLateErrorBackend {
        entered: entered.clone(),
        release: release.clone(),
    }));
    let request_id = "cancel-before-late-error";
    let first = tokio::spawn(
        app.clone()
            .oneshot(chat_request_body_with_id("late backend error", request_id)),
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
    assert_eq!(cancel_body["status"], "cancelled");

    release.add_permits(1);
    let first = first.await.expect("first task").expect("first response");
    assert_eq!(first.status(), StatusCode::REQUEST_TIMEOUT);
    let body = body_json(first.into_body()).await;
    assert_eq!(body["error"]["code"], "cancelled");
    assert_eq!(body["error"]["phase"], "decode");
}

#[tokio::test]
async fn admin_cancel_request_terminates_pending_chat_stream() {
    let cancelled = Arc::new(Notify::new());
    let app = build_router_with_backend(Box::new(CancellableStreamBackend {
        cancelled: cancelled.clone(),
    }));
    let request_id = "cancel-chat-stream";
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
                        "messages": [{"role": "user", "content": "stream"}],
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
                .expect("body has first chunk")
                .expect("body chunk");
            seen.push_str(std::str::from_utf8(&chunk).expect("utf8 sse"));
        }
    })
    .await
    .expect("first stream chunk arrives");

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

    tokio::time::timeout(Duration::from_millis(300), async {
        while !seen.contains("data: [DONE]") {
            let Some(chunk) = body.next().await else {
                break;
            };
            seen.push_str(std::str::from_utf8(&chunk.expect("body chunk")).expect("utf8 sse"));
        }
    })
    .await
    .expect("stream terminates after admin cancellation");
    assert!(seen.contains("\"code\":\"cancelled\""));
    assert_eq!(seen.matches("data: [DONE]").count(), 1);
}

#[tokio::test]
async fn admin_cancel_request_terminates_pending_completion_stream() {
    let cancelled = Arc::new(Notify::new());
    let app = build_router_with_backend(Box::new(CancellableStreamBackend {
        cancelled: cancelled.clone(),
    }));
    let request_id = "cancel-completion-stream";
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .header("x-request-id", request_id)
                .body(Body::from(
                    json!({
                        "model": llm_engine::DEFAULT_MODEL_ID,
                        "prompt": "stream",
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
                .expect("body has first chunk")
                .expect("body chunk");
            seen.push_str(std::str::from_utf8(&chunk).expect("utf8 sse"));
        }
    })
    .await
    .expect("first stream chunk arrives");

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

    tokio::time::timeout(Duration::from_millis(300), async {
        while !seen.contains("data: [DONE]") {
            let Some(chunk) = body.next().await else {
                break;
            };
            seen.push_str(std::str::from_utf8(&chunk.expect("body chunk")).expect("utf8 sse"));
        }
    })
    .await
    .expect("stream terminates after admin cancellation");
    assert!(seen.contains("\"code\":\"cancelled\""));
    assert_eq!(seen.matches("data: [DONE]").count(), 1);
}

#[tokio::test]
async fn admin_cancel_request_reports_unknown_request_id() {
    let response = build_router_with_protocol_test_backend()
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
