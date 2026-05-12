use super::*;

#[tokio::test]
async fn streaming_chat_validation_errors_return_json_error() {
    let response = build_router_with_protocol_test_backend()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": llm_engine::DEFAULT_MODEL_ID,
                        "messages": [],
                        "stream": true
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("chat response");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(
        !response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .starts_with("text/event-stream")
    );
    let body = body_json(response.into_body()).await;
    assert_eq!(body["error"]["code"], "invalid_request");
    assert_eq!(body["error"]["phase"], "request_validation");
}

#[tokio::test]
async fn streaming_chat_response_validation_errors_emit_sse_error_after_headers() {
    let response = build_router_with_backend(Box::new(StaticBackend {
        text: "not json".to_owned(),
    }))
    .oneshot(
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": llm_engine::DEFAULT_MODEL_ID,
                    "messages": [{"role": "user", "content": "return json"}],
                    "response_format": {"type": "json_object"},
                    "stream": true
                })
                .to_string(),
            ))
            .expect("request builds"),
    )
    .await
    .expect("chat response");

    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .starts_with("text/event-stream")
    );
    let body = body_text(response.into_body()).await;
    assert!(body.contains("\"code\":\"json_validation_failed\""));
    assert!(body.contains("\"phase\":\"response_validation\""));
    assert_eq!(body.matches("data: [DONE]").count(), 1);
}

#[tokio::test]
async fn chat_stream_headers_return_before_backend_completion() {
    let release = Arc::new(Semaphore::new(0));
    let app = build_router_with_backend(Box::new(DelayedStreamBackend {
        release: release.clone(),
    }));
    let response = tokio::time::timeout(
        Duration::from_millis(200),
        app.oneshot(
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
        ),
    )
    .await
    .expect("streaming response starts before backend release")
    .expect("stream response");

    assert_eq!(response.status(), StatusCode::OK);
    release.add_permits(1);
    let body = body_text(response.into_body()).await;
    assert!(body.contains("\"content\":\"released\""));
    assert_eq!(body.matches("data: [DONE]").count(), 1);
}

#[tokio::test]
async fn chat_stream_emits_heartbeat_before_backend_chunk() {
    let release = Arc::new(Semaphore::new(0));
    let response = build_router_with_backend(Box::new(DelayedStreamBackend {
        release: release.clone(),
    }))
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
                .expect("body has heartbeat chunk")
                .expect("heartbeat body");
            seen.push_str(std::str::from_utf8(&chunk).expect("utf8 heartbeat"));
            assert!(!seen.contains("\"content\":\"released\""));
        }
    })
    .await
    .expect("heartbeat arrives before backend release");

    release.add_permits(1);
    let mut tail = String::new();
    while let Some(chunk) = body.next().await {
        tail.push_str(std::str::from_utf8(&chunk.expect("body chunk")).expect("utf8 sse"));
    }
    assert!(tail.contains("\"content\":\"released\""));
}

#[tokio::test]
async fn chat_stream_reports_backend_stall_after_configured_timeout() {
    let release = Arc::new(Semaphore::new(0));
    let app = build_router_with_unauthenticated_admin_and_options(
        Box::new(DelayedStreamBackend {
            release: release.clone(),
        }),
        EngineOptions {
            stream_stall_timeout: Some(Duration::from_millis(50)),
            ..EngineOptions::default()
        },
    )
    .expect("router builds");
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
    let body = tokio::time::timeout(Duration::from_millis(300), body_text(response.into_body()))
        .await
        .expect("stall response completes");
    assert!(body.contains("\"code\":\"stream_stalled\""));
    assert_eq!(body.matches("data: [DONE]").count(), 1);
    let metrics = wait_for_metrics(&app, |body| body["stream_stalled_requests"] == 1).await;
    assert_eq!(metrics["stream_client_disconnected_requests"], 0);
    assert_eq!(metrics["failed_requests"], 1);
    assert_eq!(metrics["scheduler_failed_requests"], 1);
}

#[tokio::test]
async fn chat_stream_stall_cancels_backend_work() {
    let cancelled = Arc::new(Notify::new());
    let response = build_router_with_backend_and_options(
        Box::new(CancellableStreamBackend {
            cancelled: cancelled.clone(),
        }),
        EngineOptions {
            stream_stall_timeout: Some(Duration::from_millis(50)),
            ..EngineOptions::default()
        },
    )
    .expect("router builds")
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
    let body = tokio::time::timeout(Duration::from_millis(300), body_text(response.into_body()))
        .await
        .expect("stall response completes");
    assert!(body.contains("\"code\":\"stream_stalled\""));
    tokio::time::timeout(Duration::from_millis(300), cancelled.notified())
        .await
        .expect("stream stall cancels backend token");
}

#[tokio::test]
async fn chat_stream_stall_detects_slow_drip_output() {
    let cancelled = Arc::new(Notify::new());
    let response = build_router_with_backend_and_options(
        Box::new(SlowDripStreamBackend {
            delay: Duration::from_millis(40),
            cancelled: cancelled.clone(),
        }),
        EngineOptions {
            stream_stall_timeout: Some(Duration::from_millis(50)),
            ..EngineOptions::default()
        },
    )
    .expect("router builds")
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
    let body = tokio::time::timeout(Duration::from_millis(250), body_text(response.into_body()))
        .await
        .expect("slow drip should trip stream stall timeout");
    assert!(body.contains("\"code\":\"stream_stalled\""), "body: {body}");
    assert_eq!(body.matches("data: [DONE]").count(), 1);
    tokio::time::timeout(Duration::from_millis(300), cancelled.notified())
        .await
        .expect("stream stall cancels slow drip backend");
}

#[tokio::test]
async fn chat_stream_runtime_errors_include_stable_metadata() {
    let response = build_router_with_backend(Box::new(FailingStreamBackend))
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
    let body = body_text(response.into_body()).await;
    assert!(body.contains("\"content\":\"first\""));
    assert!(body.contains("\"code\":\"backend_execution_failed\""));
    assert!(body.contains("\"phase\":\"decode\""));
    assert!(body.contains("\"retryable\":true"));
    assert_eq!(body.matches("data: [DONE]").count(), 1);
}

#[tokio::test]
async fn chat_stream_sends_backend_chunk_before_backend_finishes() {
    let first = Arc::new(Notify::new());
    let finish = Arc::new(Notify::new());
    let app = build_router_with_backend(Box::new(TwoStageStreamBackend {
        first: first.clone(),
        finish: finish.clone(),
    }));
    let response = app
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
    first.notify_one();
    let mut seen = String::new();
    tokio::time::timeout(Duration::from_millis(200), async {
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
    .expect("first backend chunk is sent before final backend chunk");
    assert!(
        tokio::time::timeout(Duration::from_millis(50), body.next())
            .await
            .is_err(),
        "body should wait for final backend chunk"
    );

    finish.notify_one();
    let mut tail = seen;
    while let Some(chunk) = body.next().await {
        tail.push_str(std::str::from_utf8(&chunk.expect("body chunk")).expect("utf8 sse"));
    }
    assert!(tail.contains("\"content\":\" second\""));
    assert_eq!(tail.matches("data: [DONE]").count(), 1);
}

#[tokio::test]
async fn chat_stream_with_tools_sends_backend_chunk_before_backend_finishes() {
    let first = Arc::new(Notify::new());
    let finish = Arc::new(Notify::new());
    let app = build_router_with_backend(Box::new(TwoStageStreamBackend {
        first: first.clone(),
        finish: finish.clone(),
    }));
    let response = tokio::time::timeout(
        Duration::from_millis(200),
        app.oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": llm_engine::DEFAULT_MODEL_ID,
                        "messages": [{"role": "user", "content": "lookup while explaining"}],
                        "tools": [{
                            "type": "function",
                            "function": {"name": "lookup", "parameters": {}}
                        }],
                        "tool_choice": "auto",
                        "stream": true
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        ),
    )
    .await
    .expect("tool-capable stream response should not wait for backend completion")
    .expect("stream response");

    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    first.notify_one();
    let mut seen = String::new();
    tokio::time::timeout(Duration::from_millis(200), async {
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
    .expect("first backend chunk is sent before final backend chunk");

    finish.notify_one();
    while let Some(chunk) = body.next().await {
        seen.push_str(std::str::from_utf8(&chunk.expect("body chunk")).expect("utf8 sse"));
    }
    assert!(seen.contains("\"content\":\" second\""));
    assert_eq!(seen.matches("data: [DONE]").count(), 1);
}

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

#[tokio::test]
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

struct SlowDripStreamBackend {
    delay: Duration,
    cancelled: Arc<Notify>,
}

#[async_trait]
impl ModelBackend for SlowDripStreamBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "slow-drip-stream")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Err(BackendError::Other(
            "generate_stream_with_cancel should be used".to_owned(),
        ))
    }

    async fn generate_with_cancel(
        &self,
        _request: BackendRequest,
        _cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        Err(BackendError::Other(
            "generate_stream_with_cancel should be used".to_owned(),
        ))
    }

    fn generate_stream_with_cancel<'a>(
        &'a self,
        _request: BackendRequest,
        cancellation: CancellationToken,
    ) -> futures::stream::BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        let delay = self.delay;
        let cancelled = self.cancelled.clone();
        tokio::spawn(async move {
            cancellation.cancelled().await;
            cancelled.notify_waiters();
        });
        async_stream::try_stream! {
            loop {
                tokio::time::sleep(delay).await;
                yield BackendStreamChunk {
                    text: "x".to_owned(),
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    finish_reason: None,
                };
            }
        }
        .boxed()
    }
}

struct PendingCancellableStreamBackend {
    cancelled: Arc<Notify>,
}

#[async_trait]
impl ModelBackend for PendingCancellableStreamBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "pending-cancellable-stream")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Err(BackendError::Other(
            "generate_stream_with_cancel should be used".to_owned(),
        ))
    }

    async fn generate_with_cancel(
        &self,
        _request: BackendRequest,
        _cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        Err(BackendError::Other(
            "generate_stream_with_cancel should be used".to_owned(),
        ))
    }

    fn generate_stream_with_cancel<'a>(
        &'a self,
        _request: BackendRequest,
        cancellation: CancellationToken,
    ) -> futures::stream::BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        let cancelled = self.cancelled.clone();
        tokio::spawn(async move {
            cancellation.cancelled().await;
            cancelled.notify_waiters();
        });
        futures::stream::pending().boxed()
    }
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
