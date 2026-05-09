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
