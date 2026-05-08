use async_trait::async_trait;
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use llm_backend::{
    BackendError, BackendModelMetadata, BackendOutput, BackendRequest, ModelBackend,
};
use llm_engine::{build_router, build_router_with_backend};
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
async fn admin_models_endpoint_reports_ready_model() {
    let response = build_router()
        .oneshot(
            Request::builder()
                .uri("/admin/models")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("admin models response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["id"], "local-qwen36");
    assert_eq!(body["data"][0]["status"], "ready");
    assert_eq!(body["data"][0]["python_runtime"], false);
}

#[tokio::test]
async fn admin_model_endpoint_reports_ready_model() {
    let response = build_router()
        .oneshot(
            Request::builder()
                .uri("/admin/models/local-qwen36")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("admin model response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["id"], "local-qwen36");
    assert_eq!(body["status"], "ready");
    assert_eq!(body["python_runtime"], false);
}

#[tokio::test]
async fn admin_model_endpoint_reports_backend_artifact_identity() {
    let response = build_router_with_backend(Box::new(MetadataBackend))
        .oneshot(
            Request::builder()
                .uri("/admin/models/local-qwen36")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("admin model response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["backend"], "native-qwen");
    assert_eq!(body["repo_id"], "Qwen/Qwen3.6-35B-A3B");
    assert_eq!(
        body["resolved_commit"],
        "0123456789abcdef0123456789abcdef01234567"
    );
    assert_eq!(
        body["manifest_digest"],
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    );
}

#[tokio::test]
async fn admin_model_endpoint_uses_stable_missing_model_error() {
    let response = build_router()
        .oneshot(
            Request::builder()
                .uri("/admin/models/not-loaded")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("admin model response");

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["error"]["code"], "model_not_found");
    assert_eq!(body["error"]["phase"], "model_resolution");
    assert_eq!(body["error"]["retryable"], false);
}

#[tokio::test]
async fn admin_metrics_report_inference_counts_and_tokens() {
    let app = build_router();
    let response = app
        .clone()
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
    assert_eq!(body["requests_total"], 1);
    assert_eq!(body["successful_requests"], 1);
    assert_eq!(body["failed_requests"], 0);
    assert_eq!(body["streamed_requests"], 0);
    assert_eq!(body["tokens"]["prompt_tokens"], 1);
    assert_eq!(body["tokens"]["completion_tokens"], 5);
    assert_eq!(body["tokens"]["total_tokens"], 6);
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

#[tokio::test]
async fn chat_completions_rejects_zero_max_tokens() {
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
                        "max_tokens": 0
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("chat response");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["error"]["code"], "invalid_request");
    assert_eq!(body["error"]["phase"], "request_validation");
    assert_eq!(body["error"]["retryable"], false);
}

#[tokio::test]
async fn chat_completions_rejects_malformed_json_with_stable_error() {
    let response = build_router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from("{not-json"))
                .expect("request builds"),
        )
        .await
        .expect("chat response");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["error"]["code"], "invalid_request");
    assert_eq!(body["error"]["phase"], "request_validation");
    assert_eq!(body["error"]["retryable"], false);
}

#[tokio::test]
async fn chat_completions_rejects_multiple_choices() {
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
                        "n": 2
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("chat response");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["error"]["code"], "unsupported_capability");
    assert_eq!(body["error"]["phase"], "request_validation");
}

#[tokio::test]
async fn chat_completions_rejects_unsupported_penalties() {
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
                        "presence_penalty": 0.5
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("chat response");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["error"]["code"], "unsupported_capability");
    assert_eq!(body["error"]["phase"], "request_validation");
}

#[tokio::test]
async fn chat_completions_rejects_unsupported_logprobs() {
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
                        "logprobs": true
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("chat response");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["error"]["code"], "unsupported_capability");
    assert_eq!(body["error"]["phase"], "request_validation");
}

#[tokio::test]
async fn chat_completions_streams_openai_sse_chunks() {
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
                        "stream": true,
                        "max_tokens": 8
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("chat stream response");

    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .expect("content type")
            .starts_with("text/event-stream")
    );
    let body = body_text(response.into_body()).await;
    assert!(body.contains("data: {\"id\":\"chatcmpl-"));
    assert!(body.contains("\"object\":\"chat.completion.chunk\""));
    assert!(body.contains("\"delta\":{\"role\":\"assistant\"}"));
    assert!(body.contains("\"content\":\"hello from rust native backend\""));
    assert_eq!(body.matches("data: [DONE]").count(), 1);
}

#[tokio::test]
async fn chat_completions_streams_usage_when_requested() {
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
                        "stream": true,
                        "stream_options": {"include_usage": true}
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("chat stream response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_text(response.into_body()).await;
    assert!(body.contains("\"choices\":[],\"usage\":{\"prompt_tokens\""));
    assert!(body.contains("\"total_tokens\""));
    assert!(
        body.find("\"choices\":[],\"usage\"").expect("usage chunk")
            < body.find("data: [DONE]").expect("done")
    );
}

#[tokio::test]
async fn chat_completions_streams_tool_call_deltas() {
    let response = build_router_with_backend(Box::new(StaticBackend {
        text: r#"<tool_call>{"name":"lookup","arguments":{"query":"rust"}}</tool_call>"#.to_owned(),
    }))
    .oneshot(
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": "local-qwen36",
                    "messages": [{"role": "user", "content": "lookup rust"}],
                    "tools": [{
                        "type": "function",
                        "function": {"name": "lookup", "parameters": {}}
                    }],
                    "tool_choice": "required",
                    "stream": true
                })
                .to_string(),
            ))
            .expect("request builds"),
    )
    .await
    .expect("chat stream response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_text(response.into_body()).await;
    assert!(body.contains("\"tool_calls\":[{\"index\":0,\"id\":\"call_0\",\"type\":\"function\""));
    assert!(body.contains("\"name\":\"lookup\""));
    assert!(body.contains("\"arguments\":\"{\\\"query\\\":\\\"rust\\\"}\""));
    assert!(body.contains("\"finish_reason\":\"tool_calls\""));
    assert_eq!(body.matches("data: [DONE]").count(), 1);
}

#[tokio::test]
async fn chat_completions_rejects_invalid_json_object_mode_output() {
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
                    "model": "local-qwen36",
                    "messages": [{"role": "user", "content": "return json"}],
                    "response_format": {"type": "json_object"}
                })
                .to_string(),
            ))
            .expect("request builds"),
    )
    .await
    .expect("chat response");

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["error"]["code"], "json_validation_failed");
    assert_eq!(body["error"]["phase"], "response_validation");
    assert_eq!(body["error"]["retryable"], false);
    assert!(
        body["error"]["message"]
            .as_str()
            .expect("message")
            .contains("json_object")
    );
}

#[tokio::test]
async fn completions_endpoint_returns_openai_text_completion_shape() {
    let response = build_router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "local-qwen36",
                        "prompt": "hello",
                        "max_tokens": 8,
                        "stop": " backend"
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("completion response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["object"], "text_completion");
    assert_eq!(body["model"], "local-qwen36");
    assert_eq!(body["choices"][0]["text"], "hello from rust native");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn completions_endpoint_rejects_unsupported_sampling_controls() {
    let response = build_router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "local-qwen36",
                        "prompt": "hello",
                        "temperature": 0.7
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("completion response");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["error"]["code"], "unsupported_capability");
    assert_eq!(body["error"]["phase"], "request_validation");
    assert_eq!(body["error"]["retryable"], false);
}

#[tokio::test]
async fn completions_endpoint_rejects_malformed_json_with_stable_error() {
    let response = build_router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from("{not-json"))
                .expect("request builds"),
        )
        .await
        .expect("completion response");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["error"]["code"], "invalid_request");
    assert_eq!(body["error"]["phase"], "request_validation");
    assert_eq!(body["error"]["retryable"], false);
}

#[tokio::test]
async fn completions_endpoint_streams_openai_sse_chunks() {
    let response = build_router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "local-qwen36",
                        "prompt": "hello",
                        "stream": true,
                        "max_tokens": 8,
                        "stop": " backend"
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("completion stream response");

    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .expect("content type")
            .starts_with("text/event-stream")
    );
    let body = body_text(response.into_body()).await;
    assert!(body.contains("data: {\"id\":\"cmpl-"));
    assert!(body.contains("\"object\":\"text_completion\""));
    assert!(body.contains("\"text\":\"hello from rust native\""));
    assert!(body.contains("\"finish_reason\":\"stop\""));
    assert_eq!(body.matches("data: [DONE]").count(), 1);
}

#[tokio::test]
async fn completions_endpoint_streams_usage_when_requested() {
    let response = build_router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "local-qwen36",
                        "prompt": "hello",
                        "stream": true,
                        "stream_options": {"include_usage": true}
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("completion stream response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_text(response.into_body()).await;
    assert!(body.contains("\"choices\":[],\"usage\":{\"prompt_tokens\""));
    assert!(body.contains("\"total_tokens\""));
    assert!(
        body.find("\"choices\":[],\"usage\"").expect("usage chunk")
            < body.find("data: [DONE]").expect("done")
    );
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

struct FailingBackend;

#[async_trait]
impl ModelBackend for FailingBackend {
    fn model_id(&self) -> &str {
        "local-qwen36"
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Err(BackendError::Other("execution failed".to_owned()))
    }
}

struct StaticBackend {
    text: String,
}

#[async_trait]
impl ModelBackend for StaticBackend {
    fn model_id(&self) -> &str {
        "local-qwen36"
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Ok(BackendOutput {
            text: self.text.clone(),
            prompt_tokens: 1,
            completion_tokens: 1,
            finish_reason: llm_api::FinishReason::Stop,
        })
    }
}

struct MetadataBackend;

#[async_trait]
impl ModelBackend for MetadataBackend {
    fn model_id(&self) -> &str {
        "local-qwen36"
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        BackendModelMetadata {
            id: "local-qwen36".to_owned(),
            backend: "native-qwen".to_owned(),
            family: Some("qwen".to_owned()),
            loader: Some("native-metal".to_owned()),
            quantization: Some("bf16".to_owned()),
            repo_id: Some("Qwen/Qwen3.6-35B-A3B".to_owned()),
            resolved_commit: Some("0123456789abcdef0123456789abcdef01234567".to_owned()),
            profile: Some("qwen36-safetensors-bf16".to_owned()),
            snapshot_path: Some(std::path::PathBuf::from("/tmp/local-qwen36")),
            manifest_digest: Some(
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
            ),
        }
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Ok(BackendOutput {
            text: "metadata".to_owned(),
            prompt_tokens: 1,
            completion_tokens: 1,
            finish_reason: llm_api::FinishReason::Stop,
        })
    }
}

async fn body_json(body: Body) -> Value {
    let bytes = to_bytes(body, usize::MAX).await.expect("body bytes");
    serde_json::from_slice(&bytes).expect("json body")
}

async fn body_text(body: Body) -> String {
    let bytes = to_bytes(body, usize::MAX).await.expect("body bytes");
    String::from_utf8(bytes.to_vec()).expect("utf8 body")
}
