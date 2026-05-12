#[cfg(feature = "test-utils")]
#[tokio::test]
async fn protocol_router_serves_chat_without_engine_crate() {
    use axum::{
        body::{Body, to_bytes},
        http::{Request, StatusCode, header},
    };
    use serde_json::{Value, json};
    use tower::ServiceExt;

    let response = llm_server::build_router_with_protocol_test_backend()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": llm_server::DEFAULT_MODEL_ID,
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("chat request reaches router");

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body reads");
    let body: Value = serde_json::from_slice(&bytes).expect("json body");
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "hello from rust native backend"
    );
}
