use super::*;

#[tokio::test]
async fn chat_completions_rejects_required_tool_choice_without_tools() {
    let response = build_router_with_protocol_test_backend()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": llm_engine::DEFAULT_MODEL_ID,
                        "messages": [{"role": "user", "content": "use a tool"}],
                        "tool_choice": "required"
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
}

#[tokio::test]
async fn chat_completions_returns_required_tool_call_in_protocol_mode() {
    let response = build_router_with_protocol_test_backend()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": llm_engine::DEFAULT_MODEL_ID,
                        "messages": [{"role": "user", "content": "lookup rust"}],
                        "tools": [{
                            "type": "function",
                            "function": {"name": "lookup", "parameters": {}}
                        }],
                        "tool_choice": "required"
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("chat response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(
        body["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "lookup"
    );
}

#[tokio::test]
async fn chat_completions_required_any_uses_matching_tool_not_first_tool() {
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
                            "content": "Read secret.txt and answer with the value after LOCAL_BENCH_VALUE."
                        }],
                        "tools": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "lookup_value",
                                    "description": "Lookup a value by key.",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "key": {"type": "string"}
                                        },
                                        "required": ["key"]
                                    }
                                }
                            },
                            {
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "description": "Read a file from disk.",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "path": {"type": "string"}
                                        },
                                        "required": ["path"]
                                    }
                                }
                            }
                        ],
                        "tool_choice": "required"
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
    assert_eq!(tool_call["function"]["name"], "read_file");
    let arguments = tool_call["function"]["arguments"]
        .as_str()
        .expect("tool arguments are a JSON string");
    let arguments: Value = serde_json::from_str(arguments).expect("arguments parse as JSON");
    assert_eq!(arguments, json!({"path": "secret.txt"}));
}

#[tokio::test]
async fn chat_completions_auto_read_intent_returns_read_tool_call() {
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
                            "content": "Read secret.txt and answer with the value after LOCAL_BENCH_VALUE."
                        }],
                        "tools": [{
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "path": {"type": "string"}
                                    },
                                    "required": ["path"]
                                }
                            }
                        }]
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
    assert_eq!(tool_call["function"]["name"], "read_file");
    let arguments = tool_call["function"]["arguments"]
        .as_str()
        .expect("tool arguments are a JSON string");
    let arguments: Value = serde_json::from_str(arguments).expect("arguments parse as JSON");
    assert_eq!(arguments, json!({"path": "secret.txt"}));
}

#[tokio::test]
async fn chat_completions_rejects_parallel_tool_calls() {
    let response = build_router_with_protocol_test_backend()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": llm_engine::DEFAULT_MODEL_ID,
                        "messages": [{"role": "user", "content": "hello"}],
                        "parallel_tool_calls": true
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
async fn chat_completions_rejects_undeclared_generated_tool_call() {
    let response = build_router_with_backend(Box::new(StaticBackend {
        text: r#"<tool_call>{"name":"delete_file","arguments":{"path":"Cargo.toml"}}</tool_call>"#
            .to_owned(),
    }))
    .oneshot(
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": llm_engine::DEFAULT_MODEL_ID,
                    "messages": [{"role": "user", "content": "lookup rust"}],
                    "tools": [{
                        "type": "function",
                        "function": {"name": "lookup", "parameters": {}}
                    }],
                    "tool_choice": "required"
                })
                .to_string(),
            ))
            .expect("request builds"),
    )
    .await
    .expect("chat response");

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["error"]["code"], "tool_call_validation_failed");
    assert_eq!(body["error"]["phase"], "response_validation");
    assert_eq!(body["error"]["retryable"], false);
}

#[tokio::test]
async fn chat_completions_tool_call_validation_failed_includes_schema_hint_for_missing_required_argument()
 {
    let response = build_router_with_backend(Box::new(StaticBackend {
        text: r#"<tool_call>{"name":"read_file","arguments":{}}</tool_call>"#.to_owned(),
    }))
    .oneshot(
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": llm_engine::DEFAULT_MODEL_ID,
                    "messages": [{"role": "user", "content": "read Cargo.toml"}],
                    "tools": [{
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "parameters": {
                                "type": "object",
                                "required": ["path"],
                                "properties": {
                                    "path": {"type": "string"}
                                }
                            }
                        }
                    }],
                    "tool_choice": "required"
                })
                .to_string(),
            ))
            .expect("request builds"),
    )
    .await
    .expect("chat response");

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let body = body_json(response.into_body()).await;
    assert_eq!(body["error"]["code"], "tool_call_validation_failed");
    assert_eq!(body["error"]["phase"], "response_validation");
    assert_eq!(body["error"]["retryable"], false);
    let message = body["error"]["message"]
        .as_str()
        .expect("error message is string");
    assert!(message.contains("missing required argument `path`"));
    assert!(message.contains("required arguments: `path`"));
    assert!(message.contains("expected arguments object"));
    assert!(message.contains(r#""path":"<string>""#));
}

#[tokio::test]
async fn chat_completions_stop_sequence_suppresses_later_tool_calls() {
    let response = build_router_with_backend(Box::new(StaticBackend {
        text:
            r#"content STOP <tool_call>{"name":"lookup","arguments":{"query":"rust"}}</tool_call>"#
                .to_owned(),
    }))
    .oneshot(
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": llm_engine::DEFAULT_MODEL_ID,
                    "messages": [{"role": "user", "content": "lookup rust"}],
                    "tools": [{
                        "type": "function",
                        "function": {"name": "lookup", "parameters": {}}
                    }],
                    "stop": " STOP"
                })
                .to_string(),
            ))
            .expect("request builds"),
    )
    .await
    .expect("chat response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response.into_body()).await;
    let message = &body["choices"][0]["message"];
    assert_eq!(message["content"], "content");
    assert!(message["tool_calls"].is_null());
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
}
