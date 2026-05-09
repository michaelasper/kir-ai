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

#[tokio::test]
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
async fn streaming_json_object_response_format_rejects_text_content() {
    let backend = ProtocolTestBackend::new("local-qwen36", "not json");
    let runtime = Runtime::new(backend);
    let stream = runtime
        .chat_stream(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("return json")],
            response_format: Some(ResponseFormat::JsonObject),
            stream: true,
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("streaming json object mode starts before final validation");
    let mut emitted_content = String::new();
    let mut events = stream.into_events();
    let err = loop {
        match events.next().await.expect("stream yields validation error") {
            Ok(llm_runtime::ChatCompletionStreamEvent::Chunk(chunk)) => {
                for choice in chunk.choices {
                    if let Some(content) = choice.delta.content {
                        emitted_content.push_str(&content);
                    }
                }
            }
            Ok(llm_runtime::ChatCompletionStreamEvent::Complete(_)) => {
                panic!("invalid JSON stream should not complete successfully")
            }
            Err(err) => break err,
        }
    };

    assert!(matches!(err, RuntimeError::JsonMode(_)));
    assert!(
        emitted_content.is_empty(),
        "invalid JSON content must not be emitted before validation"
    );
}

#[tokio::test]
async fn runtime_returns_text_stream_chunks() {
    let backend = ProtocolTestBackend::new("local-qwen36", "hello");
    let runtime = Runtime::new(backend);
    let stream = runtime
        .chat_stream(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("say hi")],
            stream: true,
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("streaming text succeeds");
    let (chunks, _usage) = stream.collect_chunks().await.expect("collect chunks");

    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].object, "chat.completion.chunk");
    assert_eq!(
        chunks[0].choices[0].delta.role,
        Some(llm_api::ChatRole::Assistant)
    );
    assert_eq!(chunks[1].choices[0].delta.content.as_deref(), Some("hello"));
    assert_eq!(chunks[2].choices[0].finish_reason, Some(FinishReason::Stop));
}

#[tokio::test]
async fn runtime_chat_stream_withholds_undeclared_tool_markup() {
    let backend = ProtocolTestBackend::new(
        "local-qwen36",
        r#"<tool_call>{"name":"delete_file","arguments":{"path":"Cargo.toml"}}</tool_call>"#,
    );
    let runtime = Runtime::new(backend);
    let stream = runtime
        .chat_stream(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("say hi")],
            stream: true,
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("streaming chat starts");

    let mut emitted_content = String::new();
    let mut events = stream.into_events();
    let err = loop {
        match events.next().await.expect("stream yields validation error") {
            Ok(llm_runtime::ChatCompletionStreamEvent::Chunk(chunk)) => {
                for choice in chunk.choices {
                    if let Some(content) = choice.delta.content {
                        emitted_content.push_str(&content);
                    }
                }
            }
            Ok(llm_runtime::ChatCompletionStreamEvent::Complete(_)) => {
                panic!("invalid tool markup should not complete successfully");
            }
            Err(err) => break err,
        }
    };

    assert!(matches!(err, RuntimeError::ToolCallValidation(_)));
    assert!(!emitted_content.contains("<tool_call>"));
    assert!(!emitted_content.contains("</tool_call>"));
}

#[tokio::test]
async fn streaming_required_tool_rejects_text_fallback_without_emitting_content() {
    let backend = ProtocolTestBackend::new("local-qwen36", "plain text fallback");
    let runtime = Runtime::new(backend);
    let stream = runtime
        .chat_stream(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("read Cargo.toml")],
            tools: vec![ToolDefinition::function(
                "read_file",
                "read file",
                json!({}),
            )],
            tool_choice: Some(ToolChoice::Required),
            stream: true,
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("streaming required tool request starts");

    let mut emitted_content = String::new();
    let mut events = stream.into_events();
    let err = loop {
        match events.next().await.expect("stream yields validation error") {
            Ok(llm_runtime::ChatCompletionStreamEvent::Chunk(chunk)) => {
                for choice in chunk.choices {
                    if let Some(content) = choice.delta.content {
                        emitted_content.push_str(&content);
                    }
                }
            }
            Ok(llm_runtime::ChatCompletionStreamEvent::Complete(_)) => {
                panic!("text fallback should not complete successfully")
            }
            Err(err) => break err,
        }
    };

    assert!(matches!(
        err,
        RuntimeError::NoProgress(NoProgressClass::TextFallbackRequiredTool)
    ));
    assert!(
        emitted_content.is_empty(),
        "required-tool streams must not emit text fallback before validation"
    );
}

#[tokio::test]
async fn streaming_tool_call_rejects_missing_required_schema_argument() {
    let backend = ProtocolTestBackend::new(
        "local-qwen36",
        r#"<tool_call>{"name":"read_file","arguments":{}}</tool_call>"#,
    );
    let runtime = Runtime::new(backend);
    let stream = runtime
        .chat_stream(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("read Cargo.toml")],
            tools: vec![ToolDefinition::function(
                "read_file",
                "read file",
                json!({
                    "type": "object",
                    "required": ["path"],
                    "properties": {
                        "path": { "type": "string" }
                    }
                }),
            )],
            tool_choice: Some(ToolChoice::Required),
            stream: true,
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("streaming tool call request starts");

    let mut emitted_tool_calls = 0;
    let mut events = stream.into_events();
    let err = loop {
        match events.next().await.expect("stream yields validation error") {
            Ok(llm_runtime::ChatCompletionStreamEvent::Chunk(chunk)) => {
                emitted_tool_calls += chunk
                    .choices
                    .iter()
                    .map(|choice| choice.delta.tool_calls.len())
                    .sum::<usize>();
            }
            Ok(llm_runtime::ChatCompletionStreamEvent::Complete(_)) => {
                panic!("invalid tool call should not complete successfully")
            }
            Err(err) => break err,
        }
    };

    assert!(matches!(err, RuntimeError::ToolCallValidation(_)));
    assert!(err.to_string().contains("path"));
    assert_eq!(
        emitted_tool_calls, 0,
        "invalid tool call delta must not be emitted before schema validation"
    );
}

#[tokio::test]
async fn runtime_appends_chat_stream_usage_when_requested() {
    let backend = ProtocolTestBackend::new("local-qwen36", "hello");
    let runtime = Runtime::new(backend);
    let stream = runtime
        .chat_stream(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("say hi")],
            stream: true,
            stream_options: llm_api::StreamOptions {
                include_usage: true,
            },
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("streaming text succeeds");
    let (chunks, _usage) = stream.collect_chunks().await.expect("collect chunks");

    let usage_chunk = chunks.last().expect("usage chunk");
    assert!(usage_chunk.choices.is_empty());
    assert_eq!(usage_chunk.usage.as_ref().expect("usage").total_tokens, 3);
}

#[tokio::test]
async fn runtime_streams_generated_tool_call_delta() {
    let backend = ProtocolTestBackend::new(
        "local-qwen36",
        r#"<tool_call>{"name":"lookup","arguments":{"query":"rust"}}</tool_call>"#,
    );
    let runtime = Runtime::new(backend);
    let stream = runtime
        .chat_stream(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("lookup rust")],
            tools: vec![ToolDefinition::function("lookup", "lookup", json!({}))],
            tool_choice: Some(ToolChoice::Required),
            stream: true,
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("streaming tool calls assemble");
    let (chunks, _usage) = stream.collect_chunks().await.expect("collect chunks");

    assert_eq!(chunks.len(), 3);
    let delta = &chunks[1].choices[0].delta.tool_calls[0];
    assert_eq!(delta.index, 0);
    assert_eq!(delta.id.as_deref(), Some("call_0"));
    assert_eq!(
        delta
            .function
            .as_ref()
            .and_then(|function| function.name.as_deref()),
        Some("lookup")
    );
    assert_eq!(
        delta
            .function
            .as_ref()
            .and_then(|function| function.arguments.as_deref()),
        Some(r#"{"query":"rust"}"#)
    );
    assert_eq!(
        chunks[2].choices[0].finish_reason,
        Some(FinishReason::ToolCalls)
    );
}

#[tokio::test]
async fn runtime_streams_tool_call_delta_before_backend_finish() {
    let first = Arc::new(Semaphore::new(0));
    let finish = Arc::new(Semaphore::new(0));
    let backend = ToolBoundaryStreamBackend {
        first: first.clone(),
        finish: finish.clone(),
    };
    let runtime = Runtime::new(backend);
    let stream = runtime
        .chat_stream(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("lookup rust")],
            tools: vec![ToolDefinition::function("lookup", "lookup", json!({}))],
            tool_choice: Some(ToolChoice::Required),
            stream: true,
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("streaming tool calls assemble");
    let mut events = stream.into_events();
    assert!(
        matches!(
            events.next().await,
            Some(Ok(llm_runtime::ChatCompletionStreamEvent::Chunk(_)))
        ),
        "role seed chunk arrives"
    );

    first.add_permits(1);
    let tool_chunk = tokio::time::timeout(Duration::from_millis(300), async {
        loop {
            match events.next().await.expect("stream yields tool chunk") {
                Ok(llm_runtime::ChatCompletionStreamEvent::Chunk(chunk))
                    if chunk
                        .choices
                        .first()
                        .is_some_and(|choice| !choice.delta.tool_calls.is_empty()) =>
                {
                    return chunk;
                }
                Ok(llm_runtime::ChatCompletionStreamEvent::Chunk(_)) => {}
                Ok(llm_runtime::ChatCompletionStreamEvent::Complete(_)) => {
                    panic!("tool call should arrive before completion")
                }
                Err(err) => panic!("stream failed before tool boundary delta: {err}"),
            }
        }
    })
    .await
    .expect("tool delta arrives before backend finish");

    let delta = &tool_chunk.choices[0].delta.tool_calls[0];
    assert_eq!(
        delta
            .function
            .as_ref()
            .and_then(|function| function.name.as_deref()),
        Some("lookup")
    );
    assert_eq!(
        delta
            .function
            .as_ref()
            .and_then(|function| function.arguments.as_deref()),
        Some(r#"{"query":"rust"}"#)
    );

    finish.add_permits(1);
    let mut saw_finish = false;
    while let Some(event) = events.next().await {
        match event.expect("stream event") {
            llm_runtime::ChatCompletionStreamEvent::Chunk(chunk) => {
                saw_finish |= chunk
                    .choices
                    .first()
                    .and_then(|choice| choice.finish_reason.as_ref())
                    == Some(&FinishReason::ToolCalls);
            }
            llm_runtime::ChatCompletionStreamEvent::Complete(_) => break,
        }
    }
    assert!(saw_finish, "stream ends with tool_calls finish reason");
}

#[tokio::test]
async fn dropping_unpolled_chat_stream_cancels_backend_stream() {
    let cancelled = Arc::new(Notify::new());
    let runtime = Runtime::new(CancellableStreamBackend {
        cancelled: cancelled.clone(),
    });
    let stream = runtime
        .chat_stream(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("say hi")],
            stream: true,
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("streaming chat starts");

    drop(stream);
    tokio::time::timeout(Duration::from_millis(100), cancelled.notified())
        .await
        .expect("backend cancellation token is cancelled");
}

#[tokio::test]
async fn dropping_non_streaming_chat_future_cancels_backend_generation() {
    let started = Arc::new(Notify::new());
    let cancelled = Arc::new(Notify::new());
    let runtime = Runtime::new(CancellableGenerateBackend {
        started: started.clone(),
        cancelled: cancelled.clone(),
    });
    let mut future = Box::pin(runtime.chat(ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("say hi")],
        ..ChatCompletionRequest::default()
    }));

    tokio::select! {
        result = &mut future => panic!("backend should stay pending until cancellation: {result:?}"),
        _ = started.notified() => {}
    }

    drop(future);
    tokio::time::timeout(Duration::from_millis(100), cancelled.notified())
        .await
        .expect("backend generation cancellation token is cancelled");
}

#[tokio::test]
async fn dropping_non_streaming_completion_future_cancels_backend_generation() {
    let started = Arc::new(Notify::new());
    let cancelled = Arc::new(Notify::new());
    let runtime = Runtime::new(CancellableGenerateBackend {
        started: started.clone(),
        cancelled: cancelled.clone(),
    });
    let mut future = Box::pin(runtime.completion(CompletionRequest {
        model: "local-qwen36".to_owned(),
        prompt: "say hi".to_owned(),
        ..CompletionRequest::default()
    }));

    tokio::select! {
        result = &mut future => panic!("backend should stay pending until cancellation: {result:?}"),
        _ = started.notified() => {}
    }

    drop(future);
    tokio::time::timeout(Duration::from_millis(100), cancelled.notified())
        .await
        .expect("backend generation cancellation token is cancelled");
}
