use super::*;

#[tokio::test]
async fn admin_metrics_report_stream_decode_phase_after_first_chunk() {
    let first = Arc::new(Notify::new());
    let finish = Arc::new(Notify::new());
    let app = build_router_with_unauthenticated_admin(Box::new(TwoStageStreamBackend {
        first: first.clone(),
        finish: finish.clone(),
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
    let mut stream = response.into_body().into_data_stream();

    first.notify_one();
    let mut seen = String::new();
    tokio::time::timeout(Duration::from_millis(300), async {
        while !seen.contains("\"content\":\"first\"") {
            let chunk = stream
                .next()
                .await
                .expect("body has chunk")
                .expect("body chunk");
            seen.push_str(std::str::from_utf8(&chunk).expect("utf8 sse"));
        }
    })
    .await
    .expect("first streamed content arrives");

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
    assert_eq!(body["prefill_requests"], 0);
    assert_eq!(body["decode_requests"], 1);

    finish.notify_one();
    while stream.next().await.is_some() {}
}
