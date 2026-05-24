use super::*;

#[tokio::test]
async fn runtime_returns_non_streaming_chat_completion() {
    let backend = ProtocolTestBackend::new("local-qwen36", "hello from rust");
    let runtime = Runtime::new(backend);
    let response = runtime
        .chat(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("say hi")],
            max_tokens: Some(16),
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("runtime chat succeeds");

    assert_eq!(response.model, "local-qwen36");
    assert_eq!(response.choices[0].finish_reason, Some(FinishReason::Stop));
    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("hello from rust")
    );
    assert_eq!(response.usage.total_tokens, 5);
}

#[tokio::test]
async fn runtime_returns_streaming_text_completion_chunks() {
    let backend = ProtocolTestBackend::new("local-qwen36", "hello from completion END ignored");
    let runtime = Runtime::new(backend);
    let stream = runtime
        .completion_stream(CompletionRequest {
            model: "local-qwen36".to_owned(),
            prompt: "say hi".to_owned(),
            max_tokens: Some(8),
            stop: vec![" END".to_owned()],
            stream: true,
            stream_options: llm_api::StreamOptions::default(),
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            logprobs: None,
            n: None,
        })
        .await
        .expect("completion stream succeeds");
    let (chunks, _usage) = stream.collect_chunks().await.expect("collect chunks");

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].choices[0].text, "hello from completion");
    assert_eq!(chunks[0].choices[0].finish_reason, None);
    assert_eq!(chunks[1].choices[0].text, "");
    assert_eq!(chunks[1].choices[0].finish_reason, Some(FinishReason::Stop));
}

#[tokio::test]
async fn runtime_completion_stream_chunks_share_stable_identity_fields() {
    let backend = ProtocolTestBackend::new("local-qwen36", "hello from completion END ignored");
    let runtime = Runtime::new(backend);
    let stream = runtime
        .completion_stream(CompletionRequest {
            model: "local-qwen36".to_owned(),
            prompt: "say hi".to_owned(),
            max_tokens: Some(8),
            stop: vec![" END".to_owned()],
            stream: true,
            stream_options: llm_api::StreamOptions::default(),
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            logprobs: None,
            n: None,
        })
        .await
        .expect("completion stream succeeds");
    let (chunks, _usage) = stream.collect_chunks().await.expect("collect chunks");

    let id = chunks[0].id.clone();
    let model = chunks[0].model.clone();
    for chunk in &chunks {
        assert!(Arc::ptr_eq(&id, &chunk.id));
        assert!(Arc::ptr_eq(&model, &chunk.model));
    }
}

#[tokio::test]
async fn runtime_completion_stream_applies_stop_across_backend_chunks() {
    let runtime = Runtime::new(StopStreamingBackend);
    let stream = runtime
        .completion_stream(CompletionRequest {
            model: "local-qwen36".to_owned(),
            prompt: "say hi".to_owned(),
            max_tokens: Some(8),
            stop: vec![" STOP".to_owned()],
            stream: true,
            stream_options: llm_api::StreamOptions::default(),
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            logprobs: None,
            n: None,
        })
        .await
        .expect("completion stream succeeds");
    let (chunks, _usage) = stream.collect_chunks().await.expect("collect chunks");

    let text = chunks
        .iter()
        .map(|chunk| {
            chunk
                .choices
                .first()
                .map(|choice| choice.text.as_str())
                .unwrap_or("")
        })
        .collect::<String>();
    assert_eq!(text, "hello");
    assert_eq!(
        chunks
            .iter()
            .filter_map(|chunk| chunk.choices.first())
            .next_back()
            .and_then(|choice| choice.finish_reason.clone()),
        Some(FinishReason::Stop)
    );
}

#[tokio::test]
async fn runtime_chat_stream_applies_stop_across_backend_chunks() {
    let runtime = Runtime::new(StopStreamingBackend);
    let stream = runtime
        .chat_stream(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("say hi")],
            max_tokens: Some(8),
            stop: vec![" STOP".to_owned()],
            stream: true,
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("chat stream starts without buffering");
    let (chunks, _usage) = stream.collect_chunks().await.expect("collect chunks");

    let content = chunks
        .iter()
        .flat_map(|chunk| &chunk.choices)
        .filter_map(|choice| choice.delta.content.as_deref())
        .collect::<String>();
    assert_eq!(content, "hello");
    assert_eq!(
        chunks
            .iter()
            .filter_map(|chunk| chunk.choices.first())
            .next_back()
            .and_then(|choice| choice.finish_reason.clone()),
        Some(FinishReason::Stop)
    );
}

#[tokio::test]
async fn runtime_completion_stream_returns_before_backend_finishes() {
    let release = Arc::new(Notify::new());
    let runtime = Runtime::new(BlockingTextBackend {
        release: release.clone(),
    });

    let stream = tokio::time::timeout(
        Duration::from_millis(200),
        runtime.completion_stream(CompletionRequest {
            model: "local-qwen36".to_owned(),
            prompt: "say hi".to_owned(),
            stream: true,
            ..CompletionRequest::default()
        }),
    )
    .await
    .expect("stream handle returns before backend finishes")
    .expect("completion stream starts");

    release.notify_one();
    drop(stream);
}

#[tokio::test(start_paused = true)]
async fn runtime_completion_stream_yields_backend_chunks_incrementally() {
    let first = Arc::new(Notify::new());
    let finish = Arc::new(Notify::new());
    let runtime = Runtime::new(TwoChunkStreamBackend {
        first: first.clone(),
        finish: finish.clone(),
    });
    let stream = runtime
        .completion_stream(CompletionRequest {
            model: "local-qwen36".to_owned(),
            prompt: "say hi".to_owned(),
            stream: true,
            ..CompletionRequest::default()
        })
        .await
        .expect("completion stream starts");
    let mut events = stream.into_events();

    first.notify_one();
    let first_event = tokio::time::timeout(Duration::from_millis(200), events.next())
        .await
        .expect("first chunk arrives before completion")
        .expect("first event")
        .expect("first chunk");
    let llm_runtime::CompletionStreamEvent::Chunk(first_chunk) = first_event else {
        panic!("expected first chunk");
    };
    assert_eq!(first_chunk.choices[0].text, "first");
    assert!(
        tokio::time::timeout(Duration::from_millis(50), events.next())
            .await
            .is_err(),
        "stream should wait for the backend's final chunk"
    );

    finish.notify_one();
    let second_event = tokio::time::timeout(Duration::from_millis(200), events.next())
        .await
        .expect("second chunk arrives")
        .expect("second event")
        .expect("second chunk");
    let llm_runtime::CompletionStreamEvent::Chunk(second_chunk) = second_event else {
        panic!("expected second chunk");
    };
    assert_eq!(second_chunk.choices[0].text, " second");
}

#[tokio::test]
async fn runtime_appends_text_completion_stream_usage_when_requested() {
    let backend = ProtocolTestBackend::new("local-qwen36", "hello");
    let runtime = Runtime::new(backend);
    let stream = runtime
        .completion_stream(CompletionRequest {
            model: "local-qwen36".to_owned(),
            prompt: "say hi".to_owned(),
            stream: true,
            stream_options: llm_api::StreamOptions {
                include_usage: true,
            },
            ..CompletionRequest::default()
        })
        .await
        .expect("completion stream succeeds");
    let (chunks, _usage) = stream.collect_chunks().await.expect("collect chunks");

    let usage_chunk = chunks.last().expect("usage chunk");
    assert!(usage_chunk.choices.is_empty());
    assert_eq!(usage_chunk.usage.as_ref().expect("usage").total_tokens, 3);
}

#[tokio::test]
async fn runtime_completion_stream_saturates_overflowed_completion_usage_tokens() {
    let runtime = Runtime::new(OverflowCompletionTokenStreamBackend);
    let stream = runtime
        .completion_stream(CompletionRequest {
            model: "local-qwen36".to_owned(),
            prompt: "say hi".to_owned(),
            stream: true,
            stream_options: llm_api::StreamOptions {
                include_usage: true,
            },
            ..CompletionRequest::default()
        })
        .await
        .expect("completion stream succeeds");
    let (chunks, final_usage) = stream.collect_chunks().await.expect("collect chunks");

    let usage_chunk = chunks.last().expect("usage chunk");
    let usage = usage_chunk.usage.as_ref().expect("usage");
    assert_eq!(usage.completion_tokens, u64::MAX);
    assert_eq!(usage.total_tokens, u64::MAX);
    assert_eq!(final_usage.completion_tokens, u64::MAX);
}
