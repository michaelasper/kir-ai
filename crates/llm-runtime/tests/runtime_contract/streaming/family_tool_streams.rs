use super::*;

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
async fn runtime_streams_malformed_llama_wrapped_tool_json_as_content_with_optional_tools() {
    let text = r#"{"tool_calls":[{"function":{"name":42,"arguments":"{\"query\":\"rust\"}"}}]}"#;
    let backend = FamilyStreamBackend {
        model_id: "local-llama",
        family: "llama",
        text,
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
    let (chunks, _usage) = stream
        .collect_chunks()
        .await
        .expect("malformed wrapped JSON streams as content");

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
    assert_eq!(emitted_content, text);
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
