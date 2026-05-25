use super::*;

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
    assert_generated_tool_call_id_is_opaque(delta.id.as_deref().expect("generated tool call id"));
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
async fn streaming_required_tool_choice_preserves_deferred_content_before_tool_call() {
    let backend = ProtocolTestBackend::new(
        "local-qwen36",
        r#"I will look that up.
<tool_call>{"name":"lookup","arguments":{"query":"rust"}}</tool_call>"#,
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
        .expect("streaming tool call with content assembles");
    let (chunks, _usage) = stream.collect_chunks().await.expect("collect chunks");

    let emitted_content = chunks
        .iter()
        .flat_map(|chunk| &chunk.choices)
        .filter_map(|choice| choice.delta.content.as_deref())
        .collect::<String>();
    assert_eq!(emitted_content, "I will look that up.\n");

    let first_content_chunk = chunks
        .iter()
        .position(|chunk| {
            chunk
                .choices
                .iter()
                .any(|choice| choice.delta.content.is_some())
        })
        .expect("content chunk is emitted");
    let first_tool_chunk = chunks
        .iter()
        .position(|chunk| {
            chunk
                .choices
                .iter()
                .any(|choice| !choice.delta.tool_calls.is_empty())
        })
        .expect("tool call chunk is emitted");
    assert!(
        first_content_chunk < first_tool_chunk,
        "deferred content must be emitted before final tool-call deltas"
    );

    let delta = chunks
        .iter()
        .flat_map(|chunk| &chunk.choices)
        .flat_map(|choice| &choice.delta.tool_calls)
        .next()
        .expect("tool call delta is emitted");
    assert_eq!(delta.index, 0);
    assert_generated_tool_call_id_is_opaque(delta.id.as_deref().expect("generated tool call id"));
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
                Ok(other) => panic!("unexpected stream event: {other:?}"),
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
            other => panic!("unexpected stream event: {other:?}"),
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
                Ok(other) => panic!("unexpected stream event: {other:?}"),
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
            other => panic!("unexpected stream event: {other:?}"),
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
                Ok(other) => panic!("unexpected stream event: {other:?}"),
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
            other => panic!("unexpected stream event: {other:?}"),
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
            Ok(other) => panic!("unexpected stream event: {other:?}"),
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
