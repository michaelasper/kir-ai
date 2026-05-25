use super::*;

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
    let frames = sse_json_frames(&body);
    let error_frames: Vec<&Value> = frames
        .iter()
        .filter_map(|frame| frame.get("error"))
        .collect();
    assert_eq!(error_frames.len(), 1, "body: {body}");
    let error = error_frames[0];
    assert_eq!(error["message"], "streaming response failed");
    assert_eq!(error["code"], "backend_execution_failed");
    assert_eq!(error["phase"], "decode");
    assert_eq!(error["retryable"], true);
    assert_eq!(error["type"], "llm_engine_error");
    assert!(!body.contains("stream failed"), "body: {body}");
    assert!(!body.contains("/srv/kir-ai/private"), "body: {body}");
    assert!(!body.contains("mlx parser"), "body: {body}");
    assert_eq!(body.matches("data: [DONE]").count(), 1);
}

#[tokio::test]
async fn chat_stream_gemma_mlx_required_tool_error_exposes_attribution() {
    let response = build_router_with_backend(Box::new(GemmaMlxRequiredToolRejectingStreamBackend))
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "gemma4-e2b-mlx-4bit",
                        "messages": [{"role": "user", "content": "record the observation"}],
                        "stream": true,
                        "tools": [{
                            "type": "function",
                            "function": {
                                "name": "record_agentic_observation",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "summary": {"type": "string"}
                                    },
                                    "required": ["summary"]
                                }
                            }
                        }],
                        "tool_choice": {
                            "type": "function",
                            "function": {"name": "record_agentic_observation"}
                        }
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("stream response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_text(response.into_body()).await;
    let frames = sse_json_frames(&body);
    let error_frames: Vec<&Value> = frames
        .iter()
        .filter_map(|frame| frame.get("error"))
        .collect();
    assert_eq!(error_frames.len(), 1, "body: {body}");
    let error = error_frames[0];

    assert_eq!(error["code"], "invalid_request");
    assert_eq!(error["phase"], "request_validation");
    assert_eq!(error["retryable"], false);
    assert_eq!(error["type"], "llm_engine_error");
    let message = error["message"].as_str().expect("error message is string");
    assert!(
        message.contains("model `gemma4-e2b-mlx-4bit`"),
        "body: {body}"
    );
    assert!(message.contains("backend `mlx`"), "body: {body}");
    assert!(message.contains("family `gemma`"), "body: {body}");
    assert!(
        message.contains("function `record_agentic_observation`"),
        "body: {body}"
    );
    let content_deltas: Vec<&str> = frames
        .iter()
        .filter_map(|frame| {
            frame["choices"][0]["delta"]["content"]
                .as_str()
                .filter(|content| !content.is_empty())
        })
        .collect();
    assert!(
        content_deltas.is_empty(),
        "required-tool failure must not stream text fallback: {body}"
    );
    assert_eq!(body.matches("data: [DONE]").count(), 1);
}

#[tokio::test(start_paused = true)]
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
