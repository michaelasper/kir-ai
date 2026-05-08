use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use llm_engine::build_router;
use serde_json::{Value, json};
use tower::ServiceExt;

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
async fn chat_completions_returns_openai_shape() {
    let response = build_router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "local-qwen36",
                        "messages": [{"role": "user", "content": "hello"}],
                        "max_tokens": 8
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("chat response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["model"], "local-qwen36");
    assert!(
        body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .contains("rust")
    );
}

async fn body_json(body: Body) -> Value {
    let bytes = to_bytes(body, usize::MAX).await.expect("body bytes");
    serde_json::from_slice(&bytes).expect("json body")
}
