use super::*;

#[tokio::test]
async fn chat_completions_rejects_malformed_json_with_stable_error() {
    let response = build_router_with_protocol_test_backend()
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
async fn chat_completions_returns_required_tool_arguments_as_json_string() {
    let response = build_router_with_protocol_test_backend()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": llm_engine::DEFAULT_MODEL_ID,
                        "messages": [{
                            "role": "user",
                            "content": "Use lookup_value for key alpha."
                        }],
                        "tools": [{
                            "type": "function",
                            "function": {
                                "name": "lookup_value",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "key": {"type": "string"}
                                    },
                                    "required": ["key"]
                                }
                            }
                        }],
                        "tool_choice": {
                            "type": "function",
                            "function": {"name": "lookup_value"}
                        }
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("chat response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response.into_body()).await;
    let tool_call = &body["choices"][0]["message"]["tool_calls"][0];
    assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(tool_call["type"], "function");
    assert_eq!(tool_call["function"]["name"], "lookup_value");
    let arguments = tool_call["function"]["arguments"]
        .as_str()
        .expect("tool arguments are a JSON string");
    let arguments: Value = serde_json::from_str(arguments).expect("arguments parse as JSON");
    assert_eq!(arguments, json!({"key": "alpha"}));
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
                    "model": llm_engine::DEFAULT_MODEL_ID,
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
async fn chat_completions_returns_json_object_in_protocol_mode() {
    let response = build_router_with_protocol_test_backend()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": llm_engine::DEFAULT_MODEL_ID,
                        "messages": [{"role": "user", "content": "return json"}],
                        "response_format": {"type": "json_object"}
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("chat response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .expect("content");
    assert!(
        serde_json::from_str::<serde_json::Value>(content)
            .expect("valid JSON")
            .is_object()
    );
}
