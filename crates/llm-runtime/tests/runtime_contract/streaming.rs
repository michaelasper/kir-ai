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
            Ok(llm_runtime::ChatCompletionStreamEvent::Progress(_)) => {}
            Ok(llm_runtime::ChatCompletionStreamEvent::InternalProgress { .. }) => {}
            Ok(llm_runtime::ChatCompletionStreamEvent::Stage(_)) => {}
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

#[test]
fn runtime_classifies_chat_stream_chunk_progress_metadata() {
    let role_only = chat_stream_event(
        llm_api::ChatCompletionDelta {
            role: Some(llm_api::ChatRole::Assistant),
            ..llm_api::ChatCompletionDelta::default()
        },
        None,
    )
    .progress_metadata();
    assert!(!role_only.has_real_delta());
    assert!(!role_only.has_tool_delta());
    assert!(!role_only.has_tool_call_finish());
    assert_eq!(role_only.real_delta_bytes(), 0);

    let empty = chat_stream_event(
        llm_api::ChatCompletionDelta::default(),
        Some(FinishReason::Stop),
    )
    .progress_metadata();
    assert!(!empty.has_real_delta());
    assert!(!empty.has_tool_delta());
    assert!(!empty.has_tool_call_finish());
    assert_eq!(empty.real_delta_bytes(), 0);

    let content = "hello";
    let content_progress = chat_stream_event(
        llm_api::ChatCompletionDelta {
            content: Some(content.to_owned()),
            ..llm_api::ChatCompletionDelta::default()
        },
        None,
    )
    .progress_metadata();
    assert!(content_progress.has_real_delta());
    assert!(!content_progress.has_tool_delta());
    assert!(!content_progress.has_tool_call_finish());
    assert_eq!(content_progress.real_delta_bytes(), content.len());

    let arguments = r#"{"query":"rust"}"#;
    let tool_progress = chat_stream_event(
        llm_api::ChatCompletionDelta {
            tool_calls: vec![llm_api::ToolCallDelta {
                index: 0,
                id: Some("call".to_owned()),
                call_type: Some(llm_api::ToolCallType::Function),
                function: Some(llm_api::ToolCallFunctionDelta {
                    name: Some("lookup".to_owned()),
                    arguments: Some(arguments.to_owned()),
                }),
            }],
            ..llm_api::ChatCompletionDelta::default()
        },
        None,
    )
    .progress_metadata();
    assert!(tool_progress.has_real_delta());
    assert!(tool_progress.has_tool_delta());
    assert!(!tool_progress.has_tool_call_finish());
    assert_eq!(
        tool_progress.real_delta_bytes(),
        "call".len() + "lookup".len() + arguments.len()
    );

    let tool_finish = chat_stream_event(
        llm_api::ChatCompletionDelta::default(),
        Some(FinishReason::ToolCalls),
    )
    .progress_metadata();
    assert!(!tool_finish.has_real_delta());
    assert!(!tool_finish.has_tool_delta());
    assert!(tool_finish.has_tool_call_finish());
    assert_eq!(tool_finish.real_delta_bytes(), 0);
}

#[test]
fn runtime_classifies_completion_stream_chunk_progress_metadata() {
    let empty = completion_stream_event("", Some(FinishReason::Stop)).progress_metadata();
    assert!(!empty.has_real_delta());
    assert!(!empty.has_tool_delta());
    assert!(!empty.has_tool_call_finish());
    assert_eq!(empty.real_delta_bytes(), 0);

    let text = "hello";
    let content = completion_stream_event(text, None).progress_metadata();
    assert!(content.has_real_delta());
    assert!(!content.has_tool_delta());
    assert!(!content.has_tool_call_finish());
    assert_eq!(content.real_delta_bytes(), text.len());
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
            Ok(llm_runtime::ChatCompletionStreamEvent::Progress(_)) => {}
            Ok(llm_runtime::ChatCompletionStreamEvent::InternalProgress { .. }) => {}
            Ok(llm_runtime::ChatCompletionStreamEvent::Stage(_)) => {}
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
            Ok(llm_runtime::ChatCompletionStreamEvent::Progress(_)) => {}
            Ok(llm_runtime::ChatCompletionStreamEvent::InternalProgress { .. }) => {}
            Ok(llm_runtime::ChatCompletionStreamEvent::Stage(_)) => {}
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
            Ok(llm_runtime::ChatCompletionStreamEvent::Progress(_)) => {}
            Ok(llm_runtime::ChatCompletionStreamEvent::InternalProgress { .. }) => {}
            Ok(llm_runtime::ChatCompletionStreamEvent::Stage(_)) => {}
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
async fn streaming_repeated_empty_required_tool_call_fifth_attempt_returns_no_progress_without_emitting_arguments()
 {
    let backend = ProtocolTestBackend::new(
        "local-qwen36",
        r#"<tool_call>{"name":"read","arguments":{}}</tool_call>"#,
    );
    let runtime = Runtime::new(backend);
    let stream = runtime
        .chat_stream(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: streaming_failed_read_attempts(4),
            tools: vec![streaming_read_tool_definition()],
            tool_choice: Some(ToolChoice::Required),
            stream: true,
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("streaming repeated empty read request starts");

    let mut emitted_tool_calls = 0;
    let mut events = stream.into_events();
    let err = loop {
        match events
            .next()
            .await
            .expect("stream yields no-progress error")
        {
            Ok(llm_runtime::ChatCompletionStreamEvent::Chunk(chunk)) => {
                emitted_tool_calls += chunk
                    .choices
                    .iter()
                    .map(|choice| choice.delta.tool_calls.len())
                    .sum::<usize>();
            }
            Ok(llm_runtime::ChatCompletionStreamEvent::Complete(_)) => {
                panic!("repeated invalid tool call should not complete successfully")
            }
            Ok(llm_runtime::ChatCompletionStreamEvent::Progress(_)) => {}
            Ok(llm_runtime::ChatCompletionStreamEvent::InternalProgress { .. }) => {}
            Ok(llm_runtime::ChatCompletionStreamEvent::Stage(_)) => {}
            Err(err) => break err,
        }
    };

    assert!(matches!(
        err,
        RuntimeError::NoProgress(NoProgressClass::RepeatedInvalidToolCall)
    ));
    assert_eq!(
        emitted_tool_calls, 0,
        "invalid repeated tool call arguments must not be emitted"
    );
}

#[tokio::test]
async fn streaming_tool_call_fills_missing_required_omp_intent_argument() {
    let backend = ProtocolTestBackend::new(
        "local-qwen36",
        r#"<tool_call>{"name":"read","arguments":{"path":"calculator.py"}}</tool_call>"#,
    );
    let runtime = Runtime::new(backend);
    let stream = runtime
        .chat_stream(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("read calculator.py")],
            tools: vec![ToolDefinition::function(
                "read",
                "read file",
                json!({
                    "type": "object",
                    "required": ["path", "_i"],
                    "properties": {
                        "path": { "type": "string" },
                        "_i": { "type": "string" }
                    }
                }),
            )],
            tool_choice: Some(ToolChoice::Required),
            stream: true,
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("streaming tool call request starts");
    let (chunks, _usage) = stream.collect_chunks().await.expect("collect chunks");

    let tool_call = chunks
        .iter()
        .flat_map(|chunk| &chunk.choices)
        .flat_map(|choice| &choice.delta.tool_calls)
        .next()
        .expect("tool call delta is emitted");
    let function = tool_call
        .function
        .as_ref()
        .expect("tool call function delta");
    let arguments = function.arguments.as_ref().expect("tool arguments");
    let arguments: Value = serde_json::from_str(arguments).expect("arguments JSON object");
    assert_eq!(arguments["path"], "calculator.py");
    assert!(
        arguments["_i"]
            .as_str()
            .is_some_and(|intent| !intent.is_empty())
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
async fn runtime_chat_stream_usage_includes_backend_cached_prompt_tokens() {
    let runtime = Runtime::new(CachedPromptTokenStreamBackend);
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
    let (chunks, final_usage) = stream.collect_chunks().await.expect("collect chunks");

    let usage_chunk = chunks.last().expect("usage chunk");
    assert_eq!(
        usage_chunk
            .usage
            .as_ref()
            .and_then(|usage| usage.prompt_tokens_details.as_ref())
            .map(|details| details.cached_tokens),
        Some(6)
    );
    assert_eq!(
        final_usage
            .prompt_tokens_details
            .as_ref()
            .map(|details| details.cached_tokens),
        Some(6)
    );
}

#[tokio::test]
async fn runtime_streams_generated_tool_call_delta() {
    let backend = FamilyStreamBackend {
        model_id: "local-qwen36",
        family: "qwen",
        text: r#"<tool_call>{"name":"lookup","arguments":{"query":"rust"}}</tool_call>"#,
        finish_reason: BackendFinishReason::Stop,
    };
    assert_streams_tool_call_delta_without_marker_content(backend, "local-qwen36", &[]).await;
}

#[tokio::test]
async fn runtime_streams_deepseek_tool_call_delta_without_marker_content() {
    let backend = FamilyStreamBackend {
        model_id: "local-deepseek",
        family: "deep_seek",
        text: "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>lookup\n```json\n{\"query\":\"rust\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
        finish_reason: BackendFinishReason::Stop,
    };
    assert_streams_tool_call_delta_without_marker_content(
        backend,
        "local-deepseek",
        &["<｜tool▁calls▁begin｜>", "<｜tool▁calls▁end｜>"],
    )
    .await;
}

struct CachedPromptTokenStreamBackend;

#[async_trait::async_trait]
impl ModelBackend for CachedPromptTokenStreamBackend {
    fn model_id(&self) -> &str {
        "local-qwen36"
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "cached-prompt-token-stream")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Err(BackendError::other(
            "cached prompt token stream test must use generate_stream".to_owned(),
        ))
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }

    fn generate_stream<'a>(
        &'a self,
        _request: BackendRequest,
    ) -> futures::stream::BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        async_stream::try_stream! {
            yield BackendStreamChunk {
                text: "cached".to_owned(),
                tool_call_deltas: Vec::new(),
                prompt_tokens: 10,
                prompt_cached_tokens: Some(6),
                completion_tokens: 1,
                finish_reason: None,
                progress: None,
            };
            yield BackendStreamChunk {
                text: String::new(),
                tool_call_deltas: Vec::new(),
                prompt_tokens: 10,
                prompt_cached_tokens: Some(6),
                completion_tokens: 1,
                finish_reason: Some(BackendFinishReason::Stop),
                progress: None,
            };
        }
        .boxed()
    }

    fn generate_stream_with_cancel<'a>(
        &'a self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> futures::stream::BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        if cancellation.is_cancelled() {
            return futures::stream::once(async { Err(BackendError::cancelled()) }).boxed();
        }
        self.generate_stream(request)
    }
}

#[tokio::test]
async fn runtime_streams_deepseek_dsml_tool_call_delta_without_marker_content() {
    let backend = FamilyStreamBackend {
        model_id: "local-deepseek",
        family: "deep_seek",
        text: r#"<dsml_tool_call>{"name":"lookup","arguments":{"query":"rust"}}</dsml_tool_call>"#,
        finish_reason: BackendFinishReason::Stop,
    };
    assert_streams_tool_call_delta_with_choice_without_marker_content(
        backend,
        "local-deepseek",
        Some(ToolChoice::Auto),
        &["<dsml_tool_call>", "</dsml_tool_call>"],
    )
    .await;
}

#[tokio::test]
async fn runtime_streams_gemma_tool_call_delta_without_marker_content() {
    let backend = FamilyStreamBackend {
        model_id: "local-gemma4",
        family: "gemma",
        text: "<|tool_call>call:lookup{\"query\":\"rust\"}<tool_call|>",
        finish_reason: BackendFinishReason::Stop,
    };
    assert_streams_tool_call_delta_without_marker_content(
        backend,
        "local-gemma4",
        &["<|tool_call>", "<tool_call|>"],
    )
    .await;
}

#[tokio::test]
async fn runtime_streams_llama_raw_json_tool_call_delta_without_content_leak() {
    let backend = FamilyStreamBackend {
        model_id: "local-llama",
        family: "llama",
        text: r#"{"name":"lookup","parameters":{"query":"rust"}}"#,
        finish_reason: BackendFinishReason::Stop,
    };
    assert_streams_tool_call_delta_with_choice_without_marker_content(
        backend,
        "local-llama",
        Some(ToolChoice::Auto),
        &["\"name\":\"lookup\"", "\"parameters\"", "{\"name\""],
    )
    .await;
}

#[tokio::test]
async fn runtime_streams_llama_raw_json_tool_call_with_stop_without_content_leak() {
    let backend = FamilyStreamBackend {
        model_id: "local-llama",
        family: "llama",
        text: r#"{"name":"lookup","parameters":{"query":"rust"}}<|eot_id|>ignored"#,
        finish_reason: BackendFinishReason::Stop,
    };
    let runtime = Runtime::new(backend);
    let stream = runtime
        .chat_stream(ChatCompletionRequest {
            model: "local-llama".to_owned(),
            messages: vec![ChatMessage::user("lookup rust")],
            tools: vec![ToolDefinition::function("lookup", "lookup", json!({}))],
            tool_choice: Some(ToolChoice::Auto),
            stop: vec!["<|eot_id|>".to_owned()],
            stream: true,
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("streaming tool call starts");
    let (chunks, _usage) = stream.collect_chunks().await.expect("collect chunks");

    let emitted_content = chunks
        .iter()
        .flat_map(|chunk| &chunk.choices)
        .filter_map(|choice| choice.delta.content.as_deref())
        .collect::<String>();
    assert!(
        emitted_content.is_empty(),
        "raw Llama JSON tool call leaked as content: {emitted_content}"
    );

    let tool_chunks = chunks
        .iter()
        .flat_map(|chunk| &chunk.choices)
        .filter(|choice| !choice.delta.tool_calls.is_empty())
        .collect::<Vec<_>>();
    assert_eq!(tool_chunks.len(), 1);
    assert_eq!(
        tool_chunks[0].delta.tool_calls[0]
            .function
            .as_ref()
            .and_then(|function| function.name.as_deref()),
        Some("lookup")
    );
    assert_eq!(
        chunks
            .iter()
            .flat_map(|chunk| &chunk.choices)
            .next_back()
            .and_then(|choice| choice.finish_reason.as_ref()),
        Some(&FinishReason::ToolCalls)
    );
}

#[tokio::test]
async fn runtime_streams_llama_text_after_buffering_unmarked_tool_candidate() {
    let backend = FamilyStreamBackend {
        model_id: "local-llama",
        family: "llama",
        text: "plain answer",
        finish_reason: BackendFinishReason::Stop,
    };
    let runtime = Runtime::new(backend);
    let stream = runtime
        .chat_stream(ChatCompletionRequest {
            model: "local-llama".to_owned(),
            messages: vec![ChatMessage::user("lookup rust")],
            tools: vec![ToolDefinition::function("lookup", "lookup", json!({}))],
            tool_choice: Some(ToolChoice::Auto),
            stream: true,
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("streaming starts");
    let (chunks, _usage) = stream.collect_chunks().await.expect("collect chunks");

    let emitted_content = chunks
        .iter()
        .flat_map(|chunk| &chunk.choices)
        .filter_map(|choice| choice.delta.content.as_deref())
        .collect::<String>();
    let emitted_tool_calls = chunks
        .iter()
        .flat_map(|chunk| &chunk.choices)
        .map(|choice| choice.delta.tool_calls.len())
        .sum::<usize>();
    assert_eq!(emitted_content, "plain answer");
    assert_eq!(emitted_tool_calls, 0);
    assert_eq!(
        chunks
            .iter()
            .flat_map(|chunk| &chunk.choices)
            .next_back()
            .and_then(|choice| choice.finish_reason.as_ref()),
        Some(&FinishReason::Stop)
    );
}

#[tokio::test]
async fn runtime_streams_llama_text_with_stop_after_buffering_unmarked_tool_candidate() {
    let backend = FamilyStreamBackend {
        model_id: "local-llama",
        family: "llama",
        text: "plain answer STOP ignored",
        finish_reason: BackendFinishReason::Stop,
    };
    let runtime = Runtime::new(backend);
    let stream = runtime
        .chat_stream(ChatCompletionRequest {
            model: "local-llama".to_owned(),
            messages: vec![ChatMessage::user("lookup rust")],
            stop: vec![" STOP".to_owned()],
            tools: vec![ToolDefinition::function("lookup", "lookup", json!({}))],
            tool_choice: Some(ToolChoice::Auto),
            stream: true,
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("streaming starts");
    let (chunks, _usage) = stream.collect_chunks().await.expect("collect chunks");

    let emitted_content = chunks
        .iter()
        .flat_map(|chunk| &chunk.choices)
        .filter_map(|choice| choice.delta.content.as_deref())
        .collect::<String>();
    let emitted_tool_calls = chunks
        .iter()
        .flat_map(|chunk| &chunk.choices)
        .map(|choice| choice.delta.tool_calls.len())
        .sum::<usize>();
    assert_eq!(emitted_content, "plain answer");
    assert_eq!(emitted_tool_calls, 0);
    assert_eq!(
        chunks
            .iter()
            .flat_map(|chunk| &chunk.choices)
            .next_back()
            .and_then(|choice| choice.finish_reason.as_ref()),
        Some(&FinishReason::Stop)
    );
}

#[tokio::test]
async fn runtime_streams_long_llama_text_before_unmarked_tool_buffer_finishes() {
    const LONG_LLAMA_TEXT: &str = "This is a long plain-text answer with tools declared but no tool call. This is a long plain-text answer with tools declared but no tool call. This is a long plain-text answer with tools declared but no tool call. This is a long plain-text answer with tools declared but no tool call. This is a long plain-text answer with tools declared but no tool call. ";
    let first = Arc::new(Semaphore::new(0));
    let finish = Arc::new(Semaphore::new(0));
    let backend = ToolBoundaryStreamBackend {
        first: first.clone(),
        finish,
        model_id: "local-llama",
        family: "llama",
        text: LONG_LLAMA_TEXT,
    };
    let runtime = Runtime::new(backend);
    let stream = runtime
        .chat_stream(ChatCompletionRequest {
            model: "local-llama".to_owned(),
            messages: vec![ChatMessage::user("explain without tools")],
            tools: vec![ToolDefinition::function("lookup", "lookup", json!({}))],
            tool_choice: Some(ToolChoice::Auto),
            stream: true,
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("streaming starts");
    let mut events = stream.into_events();

    events
        .next()
        .await
        .expect("seed event")
        .expect("seed event succeeds");
    first.add_permits(1);
    let event = tokio::time::timeout(std::time::Duration::from_millis(200), events.next())
        .await
        .expect("long non-tool text should stream before backend finish")
        .expect("content event")
        .expect("content event succeeds");

    let ChatCompletionStreamEvent::Chunk(chunk) = event else {
        panic!("expected content chunk");
    };
    let content = chunk.choices[0]
        .delta
        .content
        .as_deref()
        .expect("content delta");
    assert!(content.starts_with("This is a long plain-text answer"));
    assert!(chunk.choices[0].delta.tool_calls.is_empty());
}

#[tokio::test]
async fn runtime_streams_llama_json_object_with_tools_emits_content_once() {
    let backend = FamilyStreamBackend {
        model_id: "local-llama",
        family: "llama",
        text: r#"{"answer":"ok"}<|eot_id|>"#,
        finish_reason: BackendFinishReason::Stop,
    };
    let runtime = Runtime::new(backend);
    let stream = runtime
        .chat_stream(ChatCompletionRequest {
            model: "local-llama".to_owned(),
            messages: vec![ChatMessage::user("return json")],
            tools: vec![ToolDefinition::function("lookup", "lookup", json!({}))],
            response_format: Some(ResponseFormat::JsonObject),
            stream: true,
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("streaming starts");
    let (chunks, _usage) = stream.collect_chunks().await.expect("collect chunks");

    let emitted_content = chunks
        .iter()
        .flat_map(|chunk| &chunk.choices)
        .filter_map(|choice| choice.delta.content.as_deref())
        .collect::<String>();
    assert_eq!(emitted_content, r#"{"answer":"ok"}"#);
}

async fn assert_streams_tool_call_delta_without_marker_content<B>(
    backend: B,
    model_id: &str,
    forbidden_content: &[&str],
) where
    B: ModelBackend,
{
    assert_streams_tool_call_delta_with_choice_without_marker_content(
        backend,
        model_id,
        Some(ToolChoice::Required),
        forbidden_content,
    )
    .await;
}

async fn assert_streams_tool_call_delta_with_choice_without_marker_content<B>(
    backend: B,
    model_id: &str,
    tool_choice: Option<ToolChoice>,
    forbidden_content: &[&str],
) where
    B: ModelBackend,
{
    let runtime = Runtime::new(backend);
    let stream = runtime
        .chat_stream(ChatCompletionRequest {
            model: model_id.to_owned(),
            messages: vec![ChatMessage::user("lookup rust")],
            tools: vec![ToolDefinition::function("lookup", "lookup", json!({}))],
            tool_choice,
            stream: true,
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("streaming tool calls assemble");
    let (chunks, _usage) = stream.collect_chunks().await.expect("collect chunks");

    let emitted_content = chunks
        .iter()
        .flat_map(|chunk| &chunk.choices)
        .filter_map(|choice| choice.delta.content.as_deref())
        .collect::<String>();
    for marker in forbidden_content {
        assert!(
            !emitted_content.contains(marker),
            "stream content leaked tool marker `{marker}`: {emitted_content}"
        );
    }

    let tool_chunks = chunks
        .iter()
        .flat_map(|chunk| &chunk.choices)
        .filter(|choice| !choice.delta.tool_calls.is_empty())
        .collect::<Vec<_>>();
    assert_eq!(tool_chunks.len(), 1);
    let delta = &tool_chunks[0].delta.tool_calls[0];
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
        chunks
            .iter()
            .flat_map(|chunk| &chunk.choices)
            .next_back()
            .and_then(|choice| choice.finish_reason.as_ref()),
        Some(&FinishReason::ToolCalls)
    );
}

#[tokio::test]
async fn protocol_backend_streams_required_tool_call_delta() {
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
        model_id: "local-qwen36",
        family: "qwen",
        text: r#"<tool_call>{"name":"lookup","arguments":{"query":"rust"}}</tool_call>"#,
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
                Ok(llm_runtime::ChatCompletionStreamEvent::Progress(_)) => {}
                Ok(llm_runtime::ChatCompletionStreamEvent::InternalProgress { .. }) => {}
                Ok(llm_runtime::ChatCompletionStreamEvent::Stage(_)) => {}
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
            llm_runtime::ChatCompletionStreamEvent::Progress(_) => {}
            llm_runtime::ChatCompletionStreamEvent::InternalProgress { .. } => {}
            llm_runtime::ChatCompletionStreamEvent::Stage(_) => {}
            llm_runtime::ChatCompletionStreamEvent::Complete(_) => break,
        }
    }
    assert!(saw_finish, "stream ends with tool_calls finish reason");
}

#[tokio::test]
async fn runtime_streams_structured_tool_delta_before_validated_finish() {
    let first = Arc::new(Semaphore::new(0));
    let finish = Arc::new(Semaphore::new(0));
    let backend = StructuredToolDeltaStreamBackend {
        first: first.clone(),
        finish: finish.clone(),
        model_id: "local-qwen36",
        family: "qwen",
        first_delta: structured_tool_delta(
            0,
            Some("call_lookup_1"),
            Some("look"),
            Some(r#"{"query""#),
        ),
        final_delta: structured_tool_delta(0, None, Some("up"), Some(r#":"rust"}"#)),
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
    let partial = tokio::time::timeout(Duration::from_millis(300), async {
        loop {
            match events.next().await.expect("stream yields partial delta") {
                Ok(llm_runtime::ChatCompletionStreamEvent::Chunk(chunk))
                    if chunk
                        .choices
                        .first()
                        .is_some_and(|choice| !choice.delta.tool_calls.is_empty()) =>
                {
                    return chunk;
                }
                Ok(llm_runtime::ChatCompletionStreamEvent::Chunk(_)) => {}
                Ok(llm_runtime::ChatCompletionStreamEvent::Progress(_)) => {}
                Ok(llm_runtime::ChatCompletionStreamEvent::InternalProgress { .. }) => {}
                Ok(llm_runtime::ChatCompletionStreamEvent::Stage(_)) => {}
                Ok(llm_runtime::ChatCompletionStreamEvent::Complete(_)) => {
                    panic!("partial tool delta should arrive before completion")
                }
                Err(err) => panic!("partial structured tool delta failed early: {err}"),
            }
        }
    })
    .await
    .expect("partial structured tool delta arrives before backend finish");
    assert_eq!(
        partial.choices[0].delta.tool_calls[0]
            .function
            .as_ref()
            .and_then(|function| function.arguments.as_deref()),
        Some(r#"{"query""#)
    );

    finish.add_permits(1);
    let mut saw_final_delta = false;
    let mut saw_finish = false;
    let mut usage = None;
    while let Some(event) = events.next().await {
        match event.expect("stream event") {
            llm_runtime::ChatCompletionStreamEvent::Chunk(chunk) => {
                saw_final_delta |= chunk
                    .choices
                    .first()
                    .is_some_and(|choice| !choice.delta.tool_calls.is_empty());
                saw_finish |= chunk
                    .choices
                    .first()
                    .and_then(|choice| choice.finish_reason.as_ref())
                    == Some(&FinishReason::ToolCalls);
            }
            llm_runtime::ChatCompletionStreamEvent::Progress(_) => {}
            llm_runtime::ChatCompletionStreamEvent::InternalProgress { .. } => {}
            llm_runtime::ChatCompletionStreamEvent::Stage(_) => {}
            llm_runtime::ChatCompletionStreamEvent::Complete(final_usage) => {
                usage = Some(final_usage);
                break;
            }
        }
    }
    assert!(
        saw_final_delta,
        "final structured argument delta is forwarded"
    );
    assert!(saw_finish, "validated tool_calls finish chunk is emitted");
    assert_eq!(
        usage
            .expect("usage emitted")
            .prompt_tokens_details
            .map(|details| details.cached_tokens),
        Some(7)
    );
}

#[tokio::test]
async fn runtime_buffers_structured_omp_arguments_until_validated_finish() {
    let first = Arc::new(Semaphore::new(0));
    let finish = Arc::new(Semaphore::new(0));
    let backend = StructuredToolDeltaStreamBackend {
        first: first.clone(),
        finish: finish.clone(),
        model_id: "local-qwen36",
        family: "qwen",
        first_delta: structured_tool_delta(
            0,
            Some("call_read_1"),
            Some("read"),
            Some(r#"{"path":"#),
        ),
        final_delta: structured_tool_delta(0, None, None, Some(r#""calculator.py"}"#)),
    };
    let runtime = Runtime::new(backend);
    let stream = runtime
        .chat_stream(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("read calculator.py")],
            tools: vec![ToolDefinition::function(
                "read",
                "read file",
                json!({
                    "type": "object",
                    "required": ["path", "_i"],
                    "properties": {
                        "path": {"type": "string"},
                        "_i": {"type": "string"}
                    }
                }),
            )],
            tool_choice: Some(ToolChoice::Required),
            stream: true,
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("streaming starts");
    let mut events = stream.into_events();
    assert!(
        matches!(
            events.next().await,
            Some(Ok(llm_runtime::ChatCompletionStreamEvent::Chunk(_)))
        ),
        "role seed chunk arrives"
    );

    first.add_permits(1);
    let progress = tokio::time::timeout(Duration::from_millis(300), async {
        loop {
            match events.next().await.expect("stream yields progress delta") {
                Ok(llm_runtime::ChatCompletionStreamEvent::Chunk(chunk))
                    if chunk
                        .choices
                        .first()
                        .is_some_and(|choice| !choice.delta.tool_calls.is_empty()) =>
                {
                    return chunk;
                }
                Ok(llm_runtime::ChatCompletionStreamEvent::Chunk(_)) => {}
                Ok(llm_runtime::ChatCompletionStreamEvent::Progress(_)) => {}
                Ok(llm_runtime::ChatCompletionStreamEvent::InternalProgress { .. }) => {}
                Ok(llm_runtime::ChatCompletionStreamEvent::Stage(_)) => {}
                Ok(llm_runtime::ChatCompletionStreamEvent::Complete(_)) => {
                    panic!("tool progress should arrive before completion")
                }
                Err(err) => panic!("stream failed before structured tool progress: {err}"),
            }
        }
    })
    .await
    .expect("tool progress arrives before backend finish");
    let delta = &progress.choices[0].delta.tool_calls[0];
    assert_eq!(delta.id.as_deref(), Some("call_read_1"));
    assert_eq!(
        delta
            .function
            .as_ref()
            .and_then(|function| function.name.as_deref()),
        Some("read")
    );
    assert!(
        delta
            .function
            .as_ref()
            .and_then(|function| function.arguments.as_deref())
            .is_none(),
        "OMP structured argument fragments must be withheld until validation"
    );

    finish.add_permits(1);
    let mut final_arguments = None;
    let mut saw_finish = false;
    let mut stages = Vec::new();
    while let Some(event) = events.next().await {
        match event.expect("stream event") {
            llm_runtime::ChatCompletionStreamEvent::Chunk(chunk) => {
                for choice in &chunk.choices {
                    for delta in &choice.delta.tool_calls {
                        if let Some(arguments) = delta
                            .function
                            .as_ref()
                            .and_then(|function| function.arguments.as_deref())
                        {
                            assert!(
                                !saw_finish,
                                "final arguments delta must arrive before tool_calls finish"
                            );
                            final_arguments = Some(arguments.to_owned());
                        }
                    }
                    if choice.finish_reason.as_ref() == Some(&FinishReason::ToolCalls) {
                        assert!(
                            final_arguments.is_some(),
                            "tool_calls finish must wait for final arguments"
                        );
                        saw_finish = true;
                    }
                }
            }
            llm_runtime::ChatCompletionStreamEvent::Progress(_) => {}
            llm_runtime::ChatCompletionStreamEvent::InternalProgress { .. } => {}
            llm_runtime::ChatCompletionStreamEvent::Stage(stage) => stages.push(stage),
            llm_runtime::ChatCompletionStreamEvent::Complete(_) => break,
        }
    }

    assert_eq!(
        stages,
        vec![
            ChatCompletionStreamStage::ToolArgumentAssemblyComplete,
            ChatCompletionStreamStage::ToolIntentFillComplete,
            ChatCompletionStreamStage::ToolSchemaValidationComplete,
        ]
    );
    assert!(saw_finish, "validated tool_calls finish chunk is emitted");
    let final_arguments = final_arguments.expect("final validated arguments delta");
    let final_arguments: Value =
        serde_json::from_str(&final_arguments).expect("final arguments JSON");
    assert_eq!(final_arguments["path"], "calculator.py");
    assert!(
        final_arguments["_i"]
            .as_str()
            .is_some_and(|intent| !intent.is_empty())
    );
}

#[tokio::test]
async fn runtime_rejects_invalid_structured_omp_args_without_argument_delta_or_finish() {
    let first = Arc::new(Semaphore::new(0));
    let finish = Arc::new(Semaphore::new(0));
    let backend = StructuredToolDeltaStreamBackend {
        first: first.clone(),
        finish: finish.clone(),
        model_id: "local-qwen36",
        family: "qwen",
        first_delta: structured_tool_delta(
            0,
            Some("call_read_1"),
            Some("read"),
            Some(r#"{"path""#),
        ),
        final_delta: structured_tool_delta(0, None, None, Some(":42}")),
    };
    let runtime = Runtime::new(backend);
    let stream = runtime
        .chat_stream(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("read calculator.py")],
            tools: vec![ToolDefinition::function(
                "read",
                "read file",
                json!({
                    "type": "object",
                    "required": ["path", "_i"],
                    "properties": {
                        "path": {"type": "string"},
                        "_i": {"type": "string"}
                    }
                }),
            )],
            tool_choice: Some(ToolChoice::Required),
            stream: true,
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("streaming starts");
    let mut events = stream.into_events();
    events
        .next()
        .await
        .expect("seed event")
        .expect("seed event succeeds");

    first.add_permits(1);
    let progress = events
        .next()
        .await
        .expect("progress event")
        .expect("progress event succeeds");
    let llm_runtime::ChatCompletionStreamEvent::Chunk(progress) = progress else {
        panic!("expected progress chunk");
    };
    assert!(!progress.choices[0].delta.tool_calls.is_empty());
    assert!(
        progress.choices[0].delta.tool_calls.iter().all(|delta| {
            delta
                .function
                .as_ref()
                .and_then(|function| function.arguments.as_ref())
                .is_none()
        }),
        "early OMP progress must not expose arguments"
    );

    finish.add_permits(1);
    let mut saw_arguments = false;
    let mut saw_tool_calls_finish = false;
    let mut stages = Vec::new();
    let err = loop {
        match events.next().await.expect("runtime emits validation error") {
            Ok(llm_runtime::ChatCompletionStreamEvent::Chunk(chunk)) => {
                saw_arguments |= chunk.choices.iter().any(|choice| {
                    choice.delta.tool_calls.iter().any(|delta| {
                        delta
                            .function
                            .as_ref()
                            .and_then(|function| function.arguments.as_ref())
                            .is_some()
                    })
                });
                saw_tool_calls_finish |= chunk
                    .choices
                    .first()
                    .and_then(|choice| choice.finish_reason.as_ref())
                    == Some(&FinishReason::ToolCalls);
            }
            Ok(llm_runtime::ChatCompletionStreamEvent::Stage(stage)) => stages.push(stage),
            Ok(llm_runtime::ChatCompletionStreamEvent::Complete(_)) => {
                panic!("invalid final tool call must not complete successfully")
            }
            Ok(llm_runtime::ChatCompletionStreamEvent::Progress(_)) => {}
            Ok(llm_runtime::ChatCompletionStreamEvent::InternalProgress { .. }) => {}
            Err(err) => break err,
        }
    };
    assert_eq!(
        stages,
        vec![
            ChatCompletionStreamStage::ToolArgumentAssemblyComplete,
            ChatCompletionStreamStage::ToolIntentFillComplete,
        ]
    );
    assert!(!saw_arguments, "invalid OMP stream must not emit arguments");
    assert!(!saw_tool_calls_finish);
    assert!(matches!(err, RuntimeError::ToolCallValidation(_)));
    assert!(err.to_string().contains("path"));
}

fn streaming_read_tool_definition() -> ToolDefinition {
    ToolDefinition::function(
        "read",
        "read file",
        json!({
            "type": "object",
            "required": ["path", "_i"],
            "properties": {
                "path": { "type": "string" },
                "_i": { "type": "string" }
            }
        }),
    )
}

fn streaming_failed_read_attempts(count: usize) -> Vec<ChatMessage> {
    let mut messages = vec![ChatMessage::user("read missing.txt")];
    for index in 0..count {
        let call_id = format!("call_{index}");
        messages.push(ChatMessage::assistant_tool_call(
            call_id.clone(),
            "read",
            json!({}),
        ));
        messages.push(ChatMessage::tool(
            call_id,
            "error: missing path argument or file not found",
        ));
        messages.push(ChatMessage::user("try again"));
    }
    messages
}

fn structured_tool_delta(
    index: u32,
    id: Option<&str>,
    name: Option<&str>,
    arguments: Option<&str>,
) -> BackendToolCallDelta {
    BackendToolCallDelta {
        index,
        id: id.map(str::to_owned),
        call_type: id.map(|_| BackendToolCallType::Function),
        function: (name.is_some() || arguments.is_some()).then(|| BackendToolCallFunctionDelta {
            name: name.map(str::to_owned),
            arguments: arguments.map(str::to_owned),
        }),
    }
}

#[tokio::test(start_paused = true)]
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

#[tokio::test(start_paused = true)]
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

#[tokio::test(start_paused = true)]
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

fn chat_stream_event(
    delta: llm_api::ChatCompletionDelta,
    finish_reason: Option<FinishReason>,
) -> ChatCompletionStreamEvent {
    ChatCompletionStreamEvent::Chunk(llm_api::ChatCompletionStreamResponse {
        id: "chatcmpl-test".to_owned(),
        object: "chat.completion.chunk".to_owned(),
        created: 0,
        model: "local-qwen36".to_owned(),
        choices: vec![llm_api::ChatCompletionStreamChoice {
            index: 0,
            delta,
            finish_reason,
        }],
        usage: None,
    })
}

fn completion_stream_event(
    text: &str,
    finish_reason: Option<FinishReason>,
) -> llm_runtime::CompletionStreamEvent {
    llm_runtime::CompletionStreamEvent::Chunk(llm_api::CompletionStreamResponse {
        id: "cmpl-test".to_owned(),
        object: "text_completion".to_owned(),
        created: 0,
        model: "local-qwen36".to_owned(),
        choices: vec![llm_api::CompletionChoice {
            text: text.to_owned(),
            index: 0,
            finish_reason,
        }],
        usage: None,
    })
}
