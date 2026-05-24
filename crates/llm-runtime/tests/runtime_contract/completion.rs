use super::*;

#[tokio::test]
async fn runtime_returns_text_completion() {
    let backend = ProtocolTestBackend::new("local-qwen36", "hello from completion END ignored");
    let runtime = Runtime::new(backend);
    let response = runtime
        .completion(CompletionRequest {
            model: "local-qwen36".to_owned(),
            prompt: "say hi".to_owned(),
            max_tokens: Some(8),
            stop: vec![" END".to_owned()],
            stream: false,
            stream_options: llm_api::StreamOptions::default(),
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            logprobs: None,
            n: None,
        })
        .await
        .expect("completion succeeds");

    assert_eq!(response.object, "text_completion");
    assert_eq!(response.choices[0].text, "hello from completion");
    assert_eq!(response.choices[0].finish_reason, Some(FinishReason::Stop));
    assert_eq!(response.usage.total_tokens, 7);
}

#[tokio::test]
async fn runtime_completion_emits_backend_dispatch_trace() {
    let backend = ProtocolTestBackend::new("local-qwen36", "hello from completion");
    let runtime = Runtime::new(backend);

    let capture = TraceCapture::start();
    let response = runtime
        .completion(CompletionRequest {
            model: "local-qwen36".to_owned(),
            prompt: "say hi".to_owned(),
            max_tokens: Some(8),
            ..CompletionRequest::default()
        })
        .await
        .expect("completion succeeds");
    let events = capture.events();

    assert_eq!(response.object, "text_completion");
    assert!(
        events.iter().any(|event| {
            event.has_field("operation", "runtime_backend_dispatch")
                && event.has_field("request_kind", "completion")
                && event.has_field("stream", "false")
                && event.has_field("model_id", "local-qwen36")
        }),
        "runtime should emit structured backend dispatch trace metadata, got {events:?}"
    );
}

#[tokio::test]
async fn runtime_forwards_completion_sampling_controls_to_backend() {
    let observed = Arc::new(Mutex::new(None));
    let backend = RecordingSamplingBackend {
        observed_sampling: observed.clone(),
    };
    let runtime = Runtime::new(backend);
    runtime
        .completion(CompletionRequest {
            model: "local-qwen36".to_owned(),
            prompt: "sample".to_owned(),
            temperature: Some(0.7),
            top_p: Some(0.9),
            ..CompletionRequest::default()
        })
        .await
        .expect("runtime completion succeeds");

    assert_eq!(
        *observed.lock().expect("observed sampling lock"),
        Some(SamplingConfig::TopP {
            temperature: 0.7,
            top_p: 0.9,
        })
    );
}

#[tokio::test]
async fn runtime_maps_none_temperature_and_top_p_one_to_top_p() {
    let observed = Arc::new(Mutex::new(None));
    let backend = RecordingSamplingBackend {
        observed_sampling: observed.clone(),
    };
    let runtime = Runtime::new(backend);
    runtime
        .completion(CompletionRequest {
            model: "local-qwen36".to_owned(),
            prompt: "sample".to_owned(),
            temperature: None,
            top_p: Some(1.0),
            ..CompletionRequest::default()
        })
        .await
        .expect("runtime completion succeeds");

    assert_eq!(
        *observed.lock().expect("observed sampling lock"),
        Some(SamplingConfig::TopP {
            temperature: 1.0,
            top_p: 1.0,
        })
    );
}

#[tokio::test]
async fn runtime_maps_omitted_completion_sampling_controls_to_top_p() {
    let observed = Arc::new(Mutex::new(None));
    let backend = RecordingSamplingBackend {
        observed_sampling: observed.clone(),
    };
    let runtime = Runtime::new(backend);
    runtime
        .completion(CompletionRequest {
            model: "local-qwen36".to_owned(),
            prompt: "sample".to_owned(),
            temperature: None,
            top_p: None,
            ..CompletionRequest::default()
        })
        .await
        .expect("runtime completion succeeds");

    assert_eq!(
        *observed.lock().expect("observed sampling lock"),
        Some(SamplingConfig::TopP {
            temperature: 1.0,
            top_p: 1.0,
        })
    );
}

#[tokio::test]
async fn runtime_validated_completion_rechecks_requests_validated_with_looser_limits() {
    let backend = ProtocolTestBackend::new("local-qwen36", "should not run");
    let runtime = Runtime::new_with_options(
        backend,
        RuntimeOptions {
            request_limits: RequestLimits {
                completion_prompt_bytes: 8,
                ..RequestLimits::default()
            },
            ..RuntimeOptions::default()
        },
    );
    let request = CompletionRequest {
        model: "local-qwen36".to_owned(),
        prompt: "123456789".to_owned(),
        ..CompletionRequest::default()
    }
    .into_validated_with_limits(RequestLimits::default())
    .expect("default limits accept longer completion prompt");

    let err = runtime
        .completion_validated_with_cancel(request, CancellationToken::new())
        .await
        .expect_err("runtime stricter limits reject default-validated request");

    match err {
        RuntimeError::Api(api_err) => {
            assert_eq!(api_err.code(), "invalid_request");
            assert!(
                api_err.message().contains("prompt must be at most 8 bytes"),
                "runtime limit error should mention strict prompt limit: {api_err:?}"
            );
        }
        other => panic!("expected API validation error, got {other:?}"),
    }
}

#[tokio::test]
async fn chat_preserves_assistant_text_whitespace() {
    let backend =
        ProtocolTestBackend::new("local-qwen36", "  keep leading space\n    indented line\n");
    let runtime = Runtime::new(backend);
    let response = runtime
        .chat(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("preserve whitespace")],
            max_tokens: Some(16),
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("chat succeeds");

    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("  keep leading space\n    indented line\n")
    );
}
