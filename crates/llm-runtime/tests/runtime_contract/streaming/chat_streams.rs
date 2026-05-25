use super::*;

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
            Ok(other) => panic!("unexpected stream event: {other:?}"),
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
async fn runtime_chat_stream_chunks_share_stable_identity_fields() {
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

    let id = chunks[0].id.clone();
    let model = chunks[0].model.clone();
    for chunk in &chunks {
        assert!(Arc::ptr_eq(&id, &chunk.id));
        assert!(Arc::ptr_eq(&model, &chunk.model));
    }
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
