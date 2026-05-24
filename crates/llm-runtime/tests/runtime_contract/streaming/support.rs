use super::*;

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

struct OverflowCompletionTokenStreamBackend;

#[async_trait::async_trait]
impl ModelBackend for OverflowCompletionTokenStreamBackend {
    fn model_id(&self) -> &str {
        "local-qwen36"
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "overflow-completion-token-stream")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Err(BackendError::other(
            "overflow completion token stream test must use generate_stream".to_owned(),
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
                text: "overflow".to_owned(),
                tool_call_deltas: Vec::new(),
                prompt_tokens: 1,
                prompt_cached_tokens: None,
                completion_tokens: u64::MAX - 1,
                finish_reason: None,
                progress: None,
            };
            yield BackendStreamChunk {
                text: " tokens".to_owned(),
                tool_call_deltas: Vec::new(),
                prompt_tokens: 1,
                prompt_cached_tokens: None,
                completion_tokens: 2,
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

fn chat_stream_event(
    delta: llm_api::ChatCompletionDelta,
    finish_reason: Option<FinishReason>,
) -> ChatCompletionStreamEvent {
    ChatCompletionStreamEvent::Chunk(llm_api::ChatCompletionStreamResponse {
        id: Arc::from("chatcmpl-test"),
        object: "chat.completion.chunk".to_owned(),
        created: 0,
        model: Arc::from("local-qwen36"),
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
        id: Arc::from("cmpl-test"),
        object: "text_completion".to_owned(),
        created: 0,
        model: Arc::from("local-qwen36"),
        choices: vec![llm_api::CompletionChoice {
            text: text.to_owned(),
            index: 0,
            finish_reason,
        }],
        usage: None,
    })
}
