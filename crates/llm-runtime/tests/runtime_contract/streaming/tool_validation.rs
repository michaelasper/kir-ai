use super::*;

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
async fn runtime_chat_stream_saturates_overflowed_completion_usage_tokens() {
    let runtime = Runtime::new(OverflowCompletionTokenStreamBackend);
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
    let usage = usage_chunk.usage.as_ref().expect("usage");
    assert_eq!(usage.completion_tokens, u64::MAX);
    assert_eq!(usage.total_tokens, u64::MAX);
    assert_eq!(final_usage.completion_tokens, u64::MAX);
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
