use super::*;

fn capability_limited_runtime(
    capabilities: BackendCapabilities,
) -> (Runtime<CapabilityLimitedBackend>, Arc<Mutex<bool>>) {
    let generated = Arc::new(Mutex::new(false));
    let runtime = Runtime::new(CapabilityLimitedBackend {
        capabilities,
        generated: generated.clone(),
    });
    (runtime, generated)
}

fn assert_unsupported_capability(err: RuntimeError, expected_message: &str) {
    match err {
        RuntimeError::Api(api_err) => {
            assert_eq!(api_err.code(), "unsupported_capability");
            assert_eq!(api_err.message(), expected_message);
        }
        other => panic!("expected unsupported capability API error, got {other:?}"),
    }
}

#[tokio::test]
async fn runtime_rejects_completion_top_p_when_backend_disallows_sampling() {
    let mut capabilities = BackendCapabilities::all();
    capabilities.sampling_top_p = false;
    let (runtime, generated) = capability_limited_runtime(capabilities);

    let err = runtime
        .completion(CompletionRequest {
            model: "local-qwen36".to_owned(),
            prompt: "sample".to_owned(),
            temperature: Some(0.7),
            top_p: Some(0.9),
            ..CompletionRequest::default()
        })
        .await
        .expect_err("backend capability check rejects top-p sampling");

    assert_unsupported_capability(
        err,
        "backend does not advertise top-p sampling support; use temperature 0 for greedy decoding",
    );
    assert!(
        !*generated.lock().expect("generated flag lock"),
        "backend generation must not run for rejected capabilities"
    );
}

#[tokio::test]
async fn runtime_rejects_chat_tools_when_backend_disallows_tool_calls() {
    let mut capabilities = BackendCapabilities::all();
    capabilities.tool_calls = false;
    let (runtime, generated) = capability_limited_runtime(capabilities);

    let err = runtime
        .chat(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("use a tool")],
            tools: vec![ToolDefinition::function("lookup", "lookup", json!({}))],
            ..ChatCompletionRequest::default()
        })
        .await
        .expect_err("backend capability check rejects tool calls");

    assert_unsupported_capability(err, "backend does not advertise tool-call support");
    assert!(
        !*generated.lock().expect("generated flag lock"),
        "backend generation must not run for rejected capabilities"
    );
}

#[tokio::test]
async fn runtime_rejects_chat_json_object_when_backend_disallows_json_mode() {
    let mut capabilities = BackendCapabilities::all();
    capabilities.json_object_mode = false;
    let (runtime, generated) = capability_limited_runtime(capabilities);

    let err = runtime
        .chat(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("return json")],
            response_format: Some(ResponseFormat::JsonObject),
            ..ChatCompletionRequest::default()
        })
        .await
        .expect_err("backend capability check rejects json_object mode");

    assert_unsupported_capability(
        err,
        "backend does not advertise json_object response_format support",
    );
    assert!(
        !*generated.lock().expect("generated flag lock"),
        "backend generation must not run for rejected capabilities"
    );
}

#[tokio::test]
async fn runtime_rejects_completion_stream_when_backend_disallows_streaming() {
    let mut capabilities = BackendCapabilities::all();
    capabilities.streaming = false;
    let (runtime, generated) = capability_limited_runtime(capabilities);

    let err = runtime
        .completion_stream(CompletionRequest {
            model: "local-qwen36".to_owned(),
            prompt: "stream".to_owned(),
            stream: true,
            ..CompletionRequest::default()
        })
        .await
        .expect_err("backend capability check rejects streaming");

    assert_unsupported_capability(err, "backend does not advertise streaming support");
    assert!(
        !*generated.lock().expect("generated flag lock"),
        "backend stream generation must not run for rejected capabilities"
    );
}
