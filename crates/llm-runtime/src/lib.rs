use chrono::Utc;
use futures::{
    StreamExt,
    stream::{self, BoxStream},
};
use llm_api::{
    ApiError, ChatCompletionChoice, ChatCompletionDelta, ChatCompletionRequest,
    ChatCompletionResponse, ChatCompletionStreamChoice, ChatCompletionStreamResponse, ChatMessage,
    ChatRole, CompletionChoice, CompletionRequest, CompletionResponse, CompletionStreamResponse,
    ResponseFormat, ToolCall, ToolCallDelta, ToolCallFunctionDelta, ToolChoice, ToolDefinition,
    Usage, ValidateRequest,
};
use llm_backend::{BackendCacheContext, BackendModelMetadata};
use llm_backend::{
    BackendError, BackendRequest, BackendStreamChunk, BackendToolChoice, ModelBackend,
    SamplingConfig,
};
use llm_models::{ModelFamily, ModelFamilyAdapter, QwenFamilyAdapter};
use llm_tokenizer::{QwenPromptOptions, TemplateError, render_qwen_chatml};
use llm_tool_parser::{ParsedAssistant, ParserError, QwenParser};
use std::collections::BTreeSet;
use std::fmt;
use thiserror::Error;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct Runtime<B> {
    backend: B,
}

impl<B> Runtime<B>
where
    B: ModelBackend,
{
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    pub fn model_id(&self) -> &str {
        self.backend.model_id()
    }

    pub fn model_metadata(&self) -> BackendModelMetadata {
        self.backend.model_metadata()
    }

    pub async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, RuntimeError> {
        self.completion_with_cancel(request, CancellationToken::new())
            .await
    }

    pub async fn completion_with_cancel(
        &self,
        request: CompletionRequest,
        cancellation: CancellationToken,
    ) -> Result<CompletionResponse, RuntimeError> {
        request.validate()?;
        if request.stream {
            return Err(ApiError::unsupported_capability(
                "streaming text completion requests must use Runtime::completion_stream",
            )
            .into());
        }
        let completion = self.complete_text(request, cancellation).await?;
        Ok(CompletionResponse {
            id: completion.id,
            object: "text_completion".to_owned(),
            created: completion.created,
            model: completion.model,
            choices: vec![CompletionChoice {
                text: completion.text,
                index: 0,
                finish_reason: Some(completion.finish_reason),
            }],
            usage: completion.usage,
        })
    }

    pub async fn completion_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionStream<'_>, RuntimeError> {
        self.completion_stream_with_cancel(request, CancellationToken::new())
            .await
    }

    pub async fn completion_stream_with_cancel(
        &self,
        request: CompletionRequest,
        cancellation: CancellationToken,
    ) -> Result<CompletionStream<'_>, RuntimeError> {
        request.validate()?;
        let include_usage = request.stream_options.include_usage;
        let stop = request.stop.clone();
        let completion = RuntimeCompletionSeed {
            id: format!("cmpl-{}", Uuid::now_v7()),
            created: Utc::now().timestamp(),
            model: request.model.clone(),
        };
        let backend_stream = self.backend.generate_stream_with_cancel(
            BackendRequest {
                model: request.model,
                prompt: request.prompt,
                max_tokens: request.max_tokens,
                sampling: SamplingConfig::from_openai_controls(request.temperature, request.top_p),
                required_tool_choice: None,
                json_object_mode: false,
                conversation_mode: false,
                cache_context: BackendCacheContext::raw_prompt(),
            },
            cancellation.clone(),
        );
        Ok(streaming_completion_stream(
            completion,
            backend_stream,
            stop,
            include_usage,
            cancellation,
        ))
    }

    pub async fn chat(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, RuntimeError> {
        self.chat_with_cancel(request, CancellationToken::new())
            .await
    }

    pub async fn chat_with_cancel(
        &self,
        request: ChatCompletionRequest,
        cancellation: CancellationToken,
    ) -> Result<ChatCompletionResponse, RuntimeError> {
        if request.stream {
            return Err(ApiError::unsupported_capability(
                "streaming chat requests must use Runtime::chat_stream",
            )
            .into());
        }
        let completion = self.complete_chat(request, cancellation).await?;
        let message = ChatMessage {
            role: ChatRole::Assistant,
            content: (!completion.parsed.content.is_empty()).then_some(completion.parsed.content),
            name: None,
            tool_call_id: None,
            tool_calls: completion.parsed.tool_calls,
        };
        Ok(ChatCompletionResponse {
            id: completion.id,
            object: "chat.completion".to_owned(),
            created: completion.created,
            model: completion.model,
            choices: vec![ChatCompletionChoice {
                index: 0,
                message,
                finish_reason: Some(completion.finish_reason),
            }],
            usage: completion.usage,
        })
    }

    pub async fn chat_stream(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionStream<'_>, RuntimeError> {
        self.chat_stream_with_cancel(request, CancellationToken::new())
            .await
    }

    pub async fn chat_stream_with_cancel(
        &self,
        request: ChatCompletionRequest,
        cancellation: CancellationToken,
    ) -> Result<ChatCompletionStream<'_>, RuntimeError> {
        if chat_stream_requires_buffering(&request) {
            return self
                .chat_stream_buffered_with_cancel(request, cancellation)
                .await;
        }
        request.validate()?;
        let include_usage = request.stream_options.include_usage;
        let adapter = self.chat_adapter()?;
        let cache_context = adapter.cache_context(&request.tools)?;
        let prompt = adapter.render_prompt(&request.messages, &request.tools)?;
        let completion = RuntimeCompletionSeed {
            id: format!("chatcmpl-{}", Uuid::now_v7()),
            created: Utc::now().timestamp(),
            model: request.model.clone(),
        };
        let backend_stream = self.backend.generate_stream_with_cancel(
            BackendRequest {
                model: request.model.clone(),
                prompt,
                max_tokens: request.effective_max_tokens(),
                sampling: SamplingConfig::from_openai_controls(request.temperature, request.top_p),
                required_tool_choice: required_backend_tool_choice(&request),
                json_object_mode: matches!(
                    request.response_format,
                    Some(ResponseFormat::JsonObject)
                ),
                conversation_mode: true,
                cache_context,
            },
            cancellation.clone(),
        );
        Ok(streaming_chat_stream(
            completion,
            request,
            adapter,
            backend_stream,
            include_usage,
            cancellation,
        ))
    }

    pub async fn chat_stream_buffered(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionStream<'static>, RuntimeError> {
        self.chat_stream_buffered_with_cancel(request, CancellationToken::new())
            .await
    }

    pub async fn chat_stream_buffered_with_cancel(
        &self,
        request: ChatCompletionRequest,
        cancellation: CancellationToken,
    ) -> Result<ChatCompletionStream<'static>, RuntimeError> {
        let include_usage = request.stream_options.include_usage;
        let completion = self.complete_chat(request, cancellation).await?;
        buffered_chat_stream(completion, include_usage)
    }

    async fn complete_chat(
        &self,
        request: ChatCompletionRequest,
        cancellation: CancellationToken,
    ) -> Result<RuntimeChatCompletion, RuntimeError> {
        request.validate()?;
        let adapter = self.chat_adapter()?;
        let cache_context = adapter.cache_context(&request.tools)?;
        let prompt = adapter.render_prompt(&request.messages, &request.tools)?;
        let required_tool_choice = required_backend_tool_choice(&request);
        let _cancel_on_drop = CancelOnDrop::new(cancellation.clone());
        let output = self
            .backend
            .generate_with_cancel(
                BackendRequest {
                    model: request.model.clone(),
                    prompt,
                    max_tokens: request.effective_max_tokens(),
                    sampling: SamplingConfig::from_openai_controls(
                        request.temperature,
                        request.top_p,
                    ),
                    required_tool_choice,
                    json_object_mode: matches!(
                        request.response_format,
                        Some(ResponseFormat::JsonObject)
                    ),
                    conversation_mode: true,
                    cache_context,
                },
                cancellation,
            )
            .await?;
        let mut raw_text = output.text;
        let stopped = apply_stop_sequences(&mut raw_text, &request.stop);
        let parsed = adapter.parse_complete(&raw_text)?;
        validate_tool_call_arguments(&parsed)?;
        validate_tool_calls_against_request(&parsed, &request)?;
        if matches!(request.response_format, Some(ResponseFormat::JsonObject)) {
            validate_json_object_response(&parsed)?;
        }
        let required_tool_pending = matches!(
            request.tool_choice,
            Some(ToolChoice::Required | ToolChoice::Function { .. })
        );
        let no_progress = classify_chat_no_progress(
            &raw_text,
            &parsed,
            output.completion_tokens,
            required_tool_pending && parsed.tool_calls.is_empty(),
            &request,
        );
        if let Some(class) = no_progress {
            return Err(RuntimeError::NoProgress(class));
        }
        let finish_reason = if !parsed.tool_calls.is_empty() {
            llm_api::FinishReason::ToolCalls
        } else if stopped {
            llm_api::FinishReason::Stop
        } else {
            output.finish_reason
        };
        let usage = Usage {
            prompt_tokens: output.prompt_tokens,
            completion_tokens: output.completion_tokens,
            total_tokens: output.prompt_tokens + output.completion_tokens,
        };
        Ok(RuntimeChatCompletion {
            id: format!("chatcmpl-{}", Uuid::now_v7()),
            created: Utc::now().timestamp(),
            model: request.model,
            parsed,
            finish_reason,
            usage,
        })
    }

    fn chat_adapter(&self) -> Result<SelectedChatAdapter, RuntimeError> {
        chat_adapter_for_metadata(&self.backend.model_metadata())
    }

    async fn complete_text(
        &self,
        request: CompletionRequest,
        cancellation: CancellationToken,
    ) -> Result<RuntimeCompletion, RuntimeError> {
        request.validate()?;
        let _cancel_on_drop = CancelOnDrop::new(cancellation.clone());
        let output = self
            .backend
            .generate_with_cancel(
                BackendRequest {
                    model: request.model.clone(),
                    prompt: request.prompt,
                    max_tokens: request.max_tokens,
                    sampling: SamplingConfig::from_openai_controls(
                        request.temperature,
                        request.top_p,
                    ),
                    required_tool_choice: None,
                    json_object_mode: false,
                    conversation_mode: false,
                    cache_context: BackendCacheContext::raw_prompt(),
                },
                cancellation,
            )
            .await?;
        let mut text = output.text;
        let stopped = apply_stop_sequences(&mut text, &request.stop);
        let no_progress = classify_no_progress(&text, output.completion_tokens, false);
        if let Some(class) = no_progress {
            return Err(RuntimeError::NoProgress(class));
        }
        let usage = Usage {
            prompt_tokens: output.prompt_tokens,
            completion_tokens: output.completion_tokens,
            total_tokens: output.prompt_tokens + output.completion_tokens,
        };
        Ok(RuntimeCompletion {
            id: format!("cmpl-{}", Uuid::now_v7()),
            created: Utc::now().timestamp(),
            model: request.model,
            text,
            finish_reason: if stopped {
                llm_api::FinishReason::Stop
            } else {
                output.finish_reason
            },
            usage,
        })
    }
}

pub struct ChatCompletionStream<'a> {
    events: BoxStream<'a, Result<ChatCompletionStreamEvent, RuntimeError>>,
}

impl fmt::Debug for ChatCompletionStream<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ChatCompletionStream { events: <stream> }")
    }
}

impl<'a> ChatCompletionStream<'a> {
    pub fn into_events(self) -> BoxStream<'a, Result<ChatCompletionStreamEvent, RuntimeError>> {
        self.events
    }

    pub async fn collect_chunks(
        self,
    ) -> Result<(Vec<ChatCompletionStreamResponse>, Usage), RuntimeError> {
        let mut chunks = Vec::new();
        let mut usage = None;
        let mut events = self.into_events();
        while let Some(event) = events.next().await {
            match event? {
                ChatCompletionStreamEvent::Chunk(chunk) => chunks.push(chunk),
                ChatCompletionStreamEvent::Complete(final_usage) => usage = Some(final_usage),
            }
        }
        Ok((chunks, usage.unwrap_or_else(empty_usage)))
    }
}

pub struct CompletionStream<'a> {
    events: BoxStream<'a, Result<CompletionStreamEvent, RuntimeError>>,
}

impl fmt::Debug for CompletionStream<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("CompletionStream { events: <stream> }")
    }
}

impl<'a> CompletionStream<'a> {
    pub fn into_events(self) -> BoxStream<'a, Result<CompletionStreamEvent, RuntimeError>> {
        self.events
    }

    pub async fn collect_chunks(
        self,
    ) -> Result<(Vec<CompletionStreamResponse>, Usage), RuntimeError> {
        let mut chunks = Vec::new();
        let mut usage = None;
        let mut events = self.into_events();
        while let Some(event) = events.next().await {
            match event? {
                CompletionStreamEvent::Chunk(chunk) => chunks.push(chunk),
                CompletionStreamEvent::Complete(final_usage) => usage = Some(final_usage),
            }
        }
        Ok((chunks, usage.unwrap_or_else(empty_usage)))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChatCompletionStreamEvent {
    Chunk(ChatCompletionStreamResponse),
    Complete(Usage),
}

#[derive(Debug, Clone, PartialEq)]
pub enum CompletionStreamEvent {
    Chunk(CompletionStreamResponse),
    Complete(Usage),
}

#[derive(Debug)]
struct CancelOnDrop {
    token: CancellationToken,
}

impl CancelOnDrop {
    fn new(token: CancellationToken) -> Self {
        Self { token }
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.token.cancel();
    }
}

#[derive(Debug, Clone, PartialEq)]
struct RuntimeCompletionSeed {
    id: String,
    created: i64,
    model: String,
}

#[derive(Debug, Clone, PartialEq)]
struct RuntimeCompletion {
    id: String,
    created: i64,
    model: String,
    text: String,
    finish_reason: llm_api::FinishReason,
    usage: Usage,
}

#[derive(Debug, Clone, PartialEq)]
struct RuntimeChatCompletion {
    id: String,
    created: i64,
    model: String,
    parsed: ParsedAssistant,
    finish_reason: llm_api::FinishReason,
    usage: Usage,
}

fn streaming_completion_stream<'a>(
    completion: RuntimeCompletionSeed,
    backend_stream: BoxStream<'a, Result<BackendStreamChunk, BackendError>>,
    stop: Vec<String>,
    include_usage: bool,
    cancellation: CancellationToken,
) -> CompletionStream<'a> {
    let cancel_on_drop = CancelOnDrop::new(cancellation);
    let events = async_stream::try_stream! {
        let _cancel_on_drop = cancel_on_drop;
        let mut backend_stream = backend_stream;
        let mut raw_text = String::new();
        let mut emitted_len = 0;
        let mut prompt_tokens = 0;
        let mut completion_tokens = 0;
        let mut finish_reason = llm_api::FinishReason::Length;
        let max_stop_len = max_stop_sequence_len(&stop);
        while let Some(chunk) = backend_stream.next().await {
            let chunk = chunk?;
            prompt_tokens = prompt_tokens.max(chunk.prompt_tokens);
            completion_tokens += chunk.completion_tokens;
            if !chunk.text.is_empty() {
                raw_text.push_str(&chunk.text);
                if let Some(stop_at) = earliest_stop_index(&raw_text, &stop) {
                    if stop_at > emitted_len {
                        yield CompletionStreamEvent::Chunk(completion_stream_seed_chunk(
                            &completion,
                            raw_text[emitted_len..stop_at].to_owned(),
                            None,
                            None,
                        ));
                    }
                    emitted_len = stop_at;
                    finish_reason = llm_api::FinishReason::Stop;
                    break;
                }
                let safe_len = safe_stream_emit_len(&raw_text, max_stop_len);
                if safe_len > emitted_len {
                    yield CompletionStreamEvent::Chunk(completion_stream_seed_chunk(
                        &completion,
                        raw_text[emitted_len..safe_len].to_owned(),
                        None,
                        None,
                    ));
                    emitted_len = safe_len;
                }
            }
            if let Some(reason) = chunk.finish_reason {
                finish_reason = reason;
                break;
            }
        }
        if finish_reason != llm_api::FinishReason::Stop && emitted_len < raw_text.len() {
            yield CompletionStreamEvent::Chunk(completion_stream_seed_chunk(
                &completion,
                raw_text[emitted_len..].to_owned(),
                None,
                None,
            ));
            emitted_len = raw_text.len();
        }
        let visible_text = &raw_text[..emitted_len];
        if let Some(class) = classify_no_progress(visible_text, completion_tokens, false) {
            Err(RuntimeError::NoProgress(class))?;
        }
        let usage = Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        };
        yield CompletionStreamEvent::Chunk(completion_stream_seed_chunk(
            &completion,
            String::new(),
            Some(finish_reason),
            None,
        ));
        if include_usage {
            yield CompletionStreamEvent::Chunk(completion_stream_seed_chunk(
                &completion,
                String::new(),
                None,
                Some(usage.clone()),
            ));
        }
        yield CompletionStreamEvent::Complete(usage);
    };
    CompletionStream {
        events: events.boxed(),
    }
}

fn buffered_chat_stream(
    completion: RuntimeChatCompletion,
    include_usage: bool,
) -> Result<ChatCompletionStream<'static>, RuntimeError> {
    let mut events = Vec::new();
    events.push(Ok(ChatCompletionStreamEvent::Chunk(stream_chunk(
        &completion,
        ChatCompletionDelta {
            role: Some(ChatRole::Assistant),
            ..ChatCompletionDelta::default()
        },
        None,
    ))));
    if !completion.parsed.content.is_empty() {
        events.push(Ok(ChatCompletionStreamEvent::Chunk(stream_chunk(
            &completion,
            ChatCompletionDelta {
                content: Some(completion.parsed.content.clone()),
                ..ChatCompletionDelta::default()
            },
            None,
        ))));
    }
    for (index, tool_call) in completion.parsed.tool_calls.iter().enumerate() {
        events.push(Ok(ChatCompletionStreamEvent::Chunk(stream_chunk(
            &completion,
            ChatCompletionDelta {
                tool_calls: vec![tool_call_delta(index, tool_call)?],
                ..ChatCompletionDelta::default()
            },
            None,
        ))));
    }
    events.push(Ok(ChatCompletionStreamEvent::Chunk(stream_chunk(
        &completion,
        ChatCompletionDelta::default(),
        Some(completion.finish_reason.clone()),
    ))));
    if include_usage {
        events.push(Ok(ChatCompletionStreamEvent::Chunk(stream_usage_chunk(
            &completion,
        ))));
    }
    events.push(Ok(ChatCompletionStreamEvent::Complete(completion.usage)));
    Ok(ChatCompletionStream {
        events: stream::iter(events).boxed(),
    })
}

fn streaming_chat_stream<'a>(
    completion: RuntimeCompletionSeed,
    request: ChatCompletionRequest,
    adapter: SelectedChatAdapter,
    backend_stream: BoxStream<'a, Result<BackendStreamChunk, BackendError>>,
    include_usage: bool,
    cancellation: CancellationToken,
) -> ChatCompletionStream<'a> {
    let cancel_on_drop = CancelOnDrop::new(cancellation);
    let events = async_stream::try_stream! {
        let _cancel_on_drop = cancel_on_drop;
        yield ChatCompletionStreamEvent::Chunk(stream_seed_chunk(
            &completion,
            ChatCompletionDelta {
                role: Some(ChatRole::Assistant),
                ..ChatCompletionDelta::default()
            },
            None,
            None,
        ));

        let mut backend_stream = backend_stream;
        let mut raw_text = String::new();
        let mut emitted_len = 0;
        let mut prompt_tokens = 0;
        let mut completion_tokens = 0;
        let mut finish_reason = llm_api::FinishReason::Length;
        let mut stopped_by_sequence = false;
        let json_object_mode = matches!(request.response_format, Some(ResponseFormat::JsonObject));
        let requires_tool_choice = request_requires_tool_choice(&request);
        let mut emitted_tool_calls = 0;
        let max_stop_len = max_stop_sequence_len(&request.stop);
        while let Some(chunk) = backend_stream.next().await {
            let chunk = chunk?;
            prompt_tokens = prompt_tokens.max(chunk.prompt_tokens);
            completion_tokens += chunk.completion_tokens;
            if !chunk.text.is_empty() {
                raw_text.push_str(&chunk.text);
                if let Some(stop_at) = earliest_stop_index(&raw_text, &request.stop) {
                    if !json_object_mode && !requires_tool_choice && stop_at > emitted_len {
                        yield ChatCompletionStreamEvent::Chunk(stream_seed_chunk(
                            &completion,
                            ChatCompletionDelta {
                                content: Some(raw_text[emitted_len..stop_at].to_owned()),
                                ..ChatCompletionDelta::default()
                            },
                            None,
                            None,
                        ));
                    }
                    emitted_len = stop_at;
                    finish_reason = llm_api::FinishReason::Stop;
                    stopped_by_sequence = true;
                    break;
                }
                if !json_object_mode && !requires_tool_choice {
                    let safe_len = safe_stream_emit_len(&raw_text, max_stop_len)
                        .min(safe_tool_markup_emit_len(&raw_text));
                    if safe_len > emitted_len {
                        yield ChatCompletionStreamEvent::Chunk(stream_seed_chunk(
                            &completion,
                            ChatCompletionDelta {
                                content: Some(raw_text[emitted_len..safe_len].to_owned()),
                                ..ChatCompletionDelta::default()
                            },
                            None,
                            None,
                        ));
                        emitted_len = safe_len;
                    }
                }
                if let Some(tool_prefix_len) = completed_tool_prefix_len(&raw_text)
                    && tool_prefix_len > emitted_len
                {
                    let parsed_prefix = adapter.parse_complete(&raw_text[..tool_prefix_len])?;
                    validate_tool_call_arguments(&parsed_prefix)?;
                    validate_tool_calls_against_request(&parsed_prefix, &request)?;
                    for (index, tool_call) in parsed_prefix
                        .tool_calls
                        .iter()
                        .enumerate()
                        .skip(emitted_tool_calls)
                    {
                        let delta = tool_call_delta(index, tool_call)?;
                        yield ChatCompletionStreamEvent::Chunk(stream_seed_chunk(
                            &completion,
                            ChatCompletionDelta {
                                tool_calls: vec![delta],
                                ..ChatCompletionDelta::default()
                            },
                            None,
                            None,
                        ));
                    }
                    emitted_tool_calls = parsed_prefix.tool_calls.len();
                    emitted_len = emitted_len.max(tool_prefix_len);
                }
            }
            if let Some(reason) = chunk.finish_reason {
                finish_reason = reason;
                break;
            }
        }
        let visible_len = if stopped_by_sequence {
            emitted_len
        } else {
            raw_text.len()
        };
        if !stopped_by_sequence
            && emitted_len < visible_len
            && !json_object_mode
            && !requires_tool_choice
            && !contains_tool_call_start(&raw_text[..visible_len])
        {
            yield ChatCompletionStreamEvent::Chunk(stream_seed_chunk(
                &completion,
                ChatCompletionDelta {
                    content: Some(raw_text[emitted_len..visible_len].to_owned()),
                    ..ChatCompletionDelta::default()
                },
                None,
                None,
            ));
        }

        let visible_text = &raw_text[..visible_len];
        let parsed = adapter.parse_complete(visible_text)?;
        validate_tool_call_arguments(&parsed)?;
        validate_tool_calls_against_request(&parsed, &request)?;
        if json_object_mode {
            validate_json_object_response(&parsed)?;
        }
        if let Some(class) = classify_chat_no_progress(
            visible_text,
            &parsed,
            completion_tokens,
            requires_tool_choice && parsed.tool_calls.is_empty(),
            &request,
        ) {
            Err(RuntimeError::NoProgress(class))?;
        }
        let finish_reason = if !parsed.tool_calls.is_empty() {
            llm_api::FinishReason::ToolCalls
        } else {
            finish_reason
        };
        let usage = Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        };
        if json_object_mode && !parsed.content.is_empty() {
            yield ChatCompletionStreamEvent::Chunk(stream_seed_chunk(
                &completion,
                ChatCompletionDelta {
                    content: Some(parsed.content.clone()),
                    ..ChatCompletionDelta::default()
                },
                None,
                None,
            ));
        }
        for (index, tool_call) in parsed.tool_calls.iter().enumerate().skip(emitted_tool_calls) {
            let delta = tool_call_delta(index, tool_call)?;
            yield ChatCompletionStreamEvent::Chunk(stream_seed_chunk(
                &completion,
                ChatCompletionDelta {
                    tool_calls: vec![delta],
                    ..ChatCompletionDelta::default()
                },
                None,
                None,
            ));
        }
        yield ChatCompletionStreamEvent::Chunk(stream_seed_chunk(
            &completion,
            ChatCompletionDelta::default(),
            Some(finish_reason),
            None,
        ));
        if include_usage {
            yield ChatCompletionStreamEvent::Chunk(stream_seed_chunk(
                &completion,
                ChatCompletionDelta::default(),
                None,
                Some(usage.clone()),
            ));
        }
        yield ChatCompletionStreamEvent::Complete(usage);
    };
    ChatCompletionStream {
        events: events.boxed(),
    }
}

pub fn chat_stream_requires_buffering(_request: &ChatCompletionRequest) -> bool {
    false
}

fn empty_usage() -> Usage {
    Usage {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
    }
}

fn stream_chunk(
    completion: &RuntimeChatCompletion,
    delta: ChatCompletionDelta,
    finish_reason: Option<llm_api::FinishReason>,
) -> ChatCompletionStreamResponse {
    ChatCompletionStreamResponse {
        id: completion.id.clone(),
        object: "chat.completion.chunk".to_owned(),
        created: completion.created,
        model: completion.model.clone(),
        choices: vec![ChatCompletionStreamChoice {
            index: 0,
            delta,
            finish_reason,
        }],
        usage: None,
    }
}

fn stream_usage_chunk(completion: &RuntimeChatCompletion) -> ChatCompletionStreamResponse {
    ChatCompletionStreamResponse {
        id: completion.id.clone(),
        object: "chat.completion.chunk".to_owned(),
        created: completion.created,
        model: completion.model.clone(),
        choices: Vec::new(),
        usage: Some(completion.usage.clone()),
    }
}

fn stream_seed_chunk(
    completion: &RuntimeCompletionSeed,
    delta: ChatCompletionDelta,
    finish_reason: Option<llm_api::FinishReason>,
    usage: Option<Usage>,
) -> ChatCompletionStreamResponse {
    ChatCompletionStreamResponse {
        id: completion.id.clone(),
        object: "chat.completion.chunk".to_owned(),
        created: completion.created,
        model: completion.model.clone(),
        choices: if usage.is_some() {
            Vec::new()
        } else {
            vec![ChatCompletionStreamChoice {
                index: 0,
                delta,
                finish_reason,
            }]
        },
        usage,
    }
}

fn completion_stream_seed_chunk(
    completion: &RuntimeCompletionSeed,
    text: String,
    finish_reason: Option<llm_api::FinishReason>,
    usage: Option<Usage>,
) -> CompletionStreamResponse {
    CompletionStreamResponse {
        id: completion.id.clone(),
        object: "text_completion".to_owned(),
        created: completion.created,
        model: completion.model.clone(),
        choices: if usage.is_some() {
            Vec::new()
        } else {
            vec![CompletionChoice {
                text,
                index: 0,
                finish_reason,
            }]
        },
        usage,
    }
}

fn tool_call_delta(index: usize, tool_call: &ToolCall) -> Result<ToolCallDelta, RuntimeError> {
    Ok(ToolCallDelta {
        index: u32::try_from(index).map_err(|err| {
            ApiError::invalid_request(format!("tool call index does not fit u32: {err}"))
        })?,
        id: Some(tool_call.id.clone()),
        call_type: Some(tool_call.call_type.clone()),
        function: Some(ToolCallFunctionDelta {
            name: Some(tool_call.function.name.clone()),
            arguments: Some(serde_json::to_string(&tool_call.function.arguments)?),
        }),
    })
}

fn apply_stop_sequences(content: &mut String, stop: &[String]) -> bool {
    let Some(stop_at) = stop
        .iter()
        .filter_map(|sequence| content.find(sequence))
        .min()
    else {
        return false;
    };
    content.truncate(stop_at);
    true
}

fn earliest_stop_index(content: &str, stop: &[String]) -> Option<usize> {
    stop.iter()
        .filter_map(|sequence| content.find(sequence))
        .min()
}

fn max_stop_sequence_len(stop: &[String]) -> usize {
    stop.iter().map(String::len).max().unwrap_or(0)
}

fn safe_stream_emit_len(content: &str, max_stop_len: usize) -> usize {
    if max_stop_len <= 1 {
        return content.len();
    }
    floor_char_boundary(content, content.len().saturating_sub(max_stop_len - 1))
}

const TOOL_CALL_START_MARKER: &str = "<tool_call>";
const TOOL_CALL_END_MARKER: &str = "</tool_call>";

fn safe_tool_markup_emit_len(content: &str) -> usize {
    if let Some(start) = content.find(TOOL_CALL_START_MARKER) {
        return start;
    }
    let withheld_prefix_len = (1..TOOL_CALL_START_MARKER.len())
        .rev()
        .find(|prefix_len| content.ends_with(&TOOL_CALL_START_MARKER[..*prefix_len]))
        .unwrap_or(0);
    content.len() - withheld_prefix_len
}

fn completed_tool_prefix_len(content: &str) -> Option<usize> {
    content
        .rfind(TOOL_CALL_END_MARKER)
        .map(|end| end + TOOL_CALL_END_MARKER.len())
}

fn contains_tool_call_start(content: &str) -> bool {
    content.contains(TOOL_CALL_START_MARKER)
}

fn floor_char_boundary(content: &str, mut index: usize) -> usize {
    while !content.is_char_boundary(index) {
        index -= 1;
    }
    index
}

fn validate_tool_call_arguments(parsed: &ParsedAssistant) -> Result<(), RuntimeError> {
    for tool_call in &parsed.tool_calls {
        if !tool_call.function.arguments.is_object() {
            return Err(RuntimeError::JsonMode(format!(
                "tool call `{}` arguments must be a JSON object",
                tool_call.function.name
            )));
        }
    }
    Ok(())
}

fn request_requires_tool_choice(request: &ChatCompletionRequest) -> bool {
    matches!(
        request.tool_choice,
        Some(ToolChoice::Required | ToolChoice::Function { .. })
    )
}

fn validate_tool_calls_against_request(
    parsed: &ParsedAssistant,
    request: &ChatCompletionRequest,
) -> Result<(), RuntimeError> {
    if parsed.tool_calls.is_empty() {
        return Ok(());
    }
    if matches!(request.tool_choice, Some(ToolChoice::None)) {
        return Err(RuntimeError::ToolCallValidation(
            "tool_choice none does not allow generated tool calls".to_owned(),
        ));
    }
    let declared_tools = request
        .tools
        .iter()
        .map(|tool| tool.function.name.as_str())
        .collect::<BTreeSet<_>>();
    for tool_call in &parsed.tool_calls {
        let name = tool_call.function.name.as_str();
        if !declared_tools.contains(name) {
            return Err(RuntimeError::ToolCallValidation(format!(
                "generated tool call `{name}` was not declared in request tools"
            )));
        }
        if let Some(ToolChoice::Function { name: required }) = &request.tool_choice
            && name != required
        {
            return Err(RuntimeError::ToolCallValidation(format!(
                "generated tool call `{name}` did not match required tool `{required}`"
            )));
        }
        let tool = request
            .tools
            .iter()
            .find(|tool| tool.function.name == name)
            .expect("declared tool set already checked");
        validate_tool_call_arguments_against_schema(tool_call, &tool.function.parameters)?;
    }
    Ok(())
}

fn validate_tool_call_arguments_against_schema(
    tool_call: &ToolCall,
    schema: &serde_json::Value,
) -> Result<(), RuntimeError> {
    if !tool_call.function.arguments.is_object() {
        return Err(RuntimeError::ToolCallValidation(format!(
            "generated tool call `{}` arguments must be a JSON object",
            tool_call.function.name
        )));
    }
    let tool_name = tool_call.function.name.as_str();
    validate_json_schema_value(tool_name, "", &tool_call.function.arguments, schema)
}

fn validate_json_schema_value(
    tool_name: &str,
    path: &str,
    value: &serde_json::Value,
    schema: &serde_json::Value,
) -> Result<(), RuntimeError> {
    if schema.is_null() || schema.as_object().is_some_and(serde_json::Map::is_empty) {
        return Ok(());
    }
    let Some(schema_object) = schema.as_object() else {
        return Ok(());
    };
    if let Some(allowed_type) = schema_object.get("type")
        && !schema_type_matches(allowed_type, value)
    {
        return Err(RuntimeError::ToolCallValidation(format!(
            "generated tool call `{tool_name}` argument `{}` does not match schema type {}",
            display_schema_path(path),
            display_schema_type(allowed_type)
        )));
    }
    if let Some(enum_values) = schema_object.get("enum") {
        let enum_values = enum_values.as_array().ok_or_else(|| {
            RuntimeError::ToolCallValidation(format!(
                "tool `{tool_name}` schema enum for `{}` must be an array",
                display_schema_path(path)
            ))
        })?;
        if !enum_values.iter().any(|allowed| allowed == value) {
            return Err(RuntimeError::ToolCallValidation(format!(
                "generated tool call `{tool_name}` argument `{}` is not one of the allowed enum values",
                display_schema_path(path)
            )));
        }
    }
    if let Some(object) = value.as_object() {
        if let Some(required) = schema_object.get("required") {
            let required = required.as_array().ok_or_else(|| {
                RuntimeError::ToolCallValidation(format!(
                    "tool `{tool_name}` schema required for `{}` must be an array",
                    display_schema_path(path)
                ))
            })?;
            for field in required {
                let field = field.as_str().ok_or_else(|| {
                    RuntimeError::ToolCallValidation(format!(
                        "tool `{tool_name}` schema required entries for `{}` must be strings",
                        display_schema_path(path)
                    ))
                })?;
                if !object.contains_key(field) {
                    return Err(RuntimeError::ToolCallValidation(format!(
                        "generated tool call `{tool_name}` missing required argument `{}`",
                        join_schema_path(path, field)
                    )));
                }
            }
        }
        if let Some(properties) = schema_object.get("properties") {
            let properties = properties.as_object().ok_or_else(|| {
                RuntimeError::ToolCallValidation(format!(
                    "tool `{tool_name}` schema properties for `{}` must be an object",
                    display_schema_path(path)
                ))
            })?;
            for (field, field_schema) in properties {
                if let Some(field_value) = object.get(field) {
                    validate_json_schema_value(
                        tool_name,
                        &join_schema_path(path, field),
                        field_value,
                        field_schema,
                    )?;
                }
            }
        }
    }
    if let Some(array) = value.as_array()
        && let Some(items_schema) = schema_object.get("items")
    {
        for (index, item) in array.iter().enumerate() {
            validate_json_schema_value(
                tool_name,
                &format!("{}[{index}]", display_schema_path(path)),
                item,
                items_schema,
            )?;
        }
    }
    Ok(())
}

fn schema_type_matches(schema_type: &serde_json::Value, value: &serde_json::Value) -> bool {
    match schema_type {
        serde_json::Value::String(type_name) => json_type_matches(type_name, value),
        serde_json::Value::Array(types) => types
            .iter()
            .filter_map(serde_json::Value::as_str)
            .any(|type_name| json_type_matches(type_name, value)),
        _ => true,
    }
}

fn json_type_matches(type_name: &str, value: &serde_json::Value) -> bool {
    match type_name {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "number" => value.is_number(),
        _ => true,
    }
}

fn display_schema_type(schema_type: &serde_json::Value) -> String {
    match schema_type {
        serde_json::Value::String(type_name) => format!("`{type_name}`"),
        serde_json::Value::Array(types) => {
            let names = types
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(|type_name| format!("`{type_name}`"))
                .collect::<Vec<_>>();
            format!("[{}]", names.join(", "))
        }
        other => other.to_string(),
    }
}

fn display_schema_path(path: &str) -> String {
    if path.is_empty() {
        "$".to_owned()
    } else {
        path.to_owned()
    }
}

fn join_schema_path(parent: &str, field: &str) -> String {
    if parent.is_empty() {
        field.to_owned()
    } else {
        format!("{parent}.{field}")
    }
}

fn required_backend_tool_choice(request: &ChatCompletionRequest) -> Option<BackendToolChoice> {
    match &request.tool_choice {
        Some(ToolChoice::Required) => Some(BackendToolChoice::RequiredAny),
        Some(ToolChoice::Function { name }) => {
            Some(BackendToolChoice::RequiredFunction(name.clone()))
        }
        Some(ToolChoice::Auto | ToolChoice::None) | None => None,
    }
}

#[derive(Debug, Clone, Copy)]
struct QwenChatAdapter;

#[derive(Debug, Clone, Copy)]
enum SelectedChatAdapter {
    Qwen(QwenChatAdapter),
}

trait ChatAdapter {
    fn cache_context(self, tools: &[ToolDefinition]) -> Result<BackendCacheContext, RuntimeError>;
    fn render_prompt(
        self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> Result<String, RuntimeError>;
    fn parse_complete(self, text: &str) -> Result<ParsedAssistant, RuntimeError>;
}

impl ChatAdapter for SelectedChatAdapter {
    fn cache_context(self, tools: &[ToolDefinition]) -> Result<BackendCacheContext, RuntimeError> {
        match self {
            Self::Qwen(adapter) => adapter.cache_context(tools),
        }
    }

    fn render_prompt(
        self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> Result<String, RuntimeError> {
        match self {
            Self::Qwen(adapter) => adapter.render_prompt(messages, tools),
        }
    }

    fn parse_complete(self, text: &str) -> Result<ParsedAssistant, RuntimeError> {
        match self {
            Self::Qwen(adapter) => adapter.parse_complete(text),
        }
    }
}

impl ChatAdapter for QwenChatAdapter {
    fn cache_context(self, tools: &[ToolDefinition]) -> Result<BackendCacheContext, RuntimeError> {
        let tool_schema = if tools.is_empty() {
            None
        } else {
            Some(serde_json::to_string(tools)?)
        };
        Ok(BackendCacheContext::chat_template(
            QwenFamilyAdapter.cache_template_id(),
            tool_schema,
        ))
    }

    fn render_prompt(
        self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> Result<String, RuntimeError> {
        Ok(render_qwen_chatml(
            messages,
            tools,
            &QwenPromptOptions {
                enable_thinking: false,
                add_generation_prompt: true,
            },
        )?)
    }

    fn parse_complete(self, text: &str) -> Result<ParsedAssistant, RuntimeError> {
        Ok(QwenParser.parse_complete(text)?)
    }
}

fn chat_adapter_for_metadata(
    metadata: &BackendModelMetadata,
) -> Result<SelectedChatAdapter, RuntimeError> {
    let Some(family) = metadata.family.as_deref() else {
        return Err(ApiError::unsupported_capability(format!(
            "backend `{}` did not declare a model family for chat rendering",
            metadata.backend
        ))
        .into());
    };
    match parse_metadata_family(family)? {
        ModelFamily::Qwen => Ok(SelectedChatAdapter::Qwen(QwenChatAdapter)),
        family => Err(unsupported_chat_family(family)),
    }
}

fn parse_metadata_family(family: &str) -> Result<ModelFamily, RuntimeError> {
    ModelFamily::parse_slug(family)
        .map_err(|err| ApiError::unsupported_capability(format!("{err} for chat rendering")).into())
}

fn unsupported_chat_family(family: ModelFamily) -> RuntimeError {
    ApiError::unsupported_capability(format!(
        "{} chat adapter support is deferred until Qwen production parity",
        family.display_name()
    ))
    .into()
}

fn validate_json_object_response(parsed: &ParsedAssistant) -> Result<(), RuntimeError> {
    if !parsed.content.is_empty() {
        let value = serde_json::from_str::<serde_json::Value>(&parsed.content).map_err(|err| {
            RuntimeError::JsonMode(format!(
                "json_object response_format requires valid JSON object content: {err}"
            ))
        })?;
        if !value.is_object() {
            return Err(RuntimeError::JsonMode(
                "json_object response_format requires assistant content to be a JSON object"
                    .to_owned(),
            ));
        }
    } else if parsed.tool_calls.is_empty() {
        return Err(RuntimeError::JsonMode(
            "json_object response_format requires assistant content or tool calls".to_owned(),
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoProgressClass {
    EmptyCompletion,
    EmptyHighOutputCompletion,
    HiddenOnlyOutput,
    TextFallbackRequiredTool,
    RepeatedInvalidToolCall,
    RepeatedAssistantContent,
    StalledAssistantTurn,
}

impl NoProgressClass {
    pub fn code(self) -> &'static str {
        match self {
            Self::EmptyCompletion => "no_progress_empty_completion",
            Self::EmptyHighOutputCompletion => "no_progress_empty_high_output_completion",
            Self::HiddenOnlyOutput => "no_progress_hidden_only_output",
            Self::TextFallbackRequiredTool => "no_progress_missing_required_tool_call",
            Self::RepeatedInvalidToolCall => "no_progress_repeated_invalid_tool_call",
            Self::RepeatedAssistantContent => "no_progress_repeated_assistant_content",
            Self::StalledAssistantTurn => "no_progress_stalled_assistant_turn",
        }
    }
}

pub fn classify_no_progress(
    content: &str,
    completion_tokens: u64,
    required_tool_pending: bool,
) -> Option<NoProgressClass> {
    if content.trim().is_empty() && completion_tokens >= 1024 {
        return Some(NoProgressClass::EmptyHighOutputCompletion);
    }
    if content.trim().is_empty() {
        return Some(NoProgressClass::EmptyCompletion);
    }
    if required_tool_pending && !content.contains("<tool_call>") {
        return Some(NoProgressClass::TextFallbackRequiredTool);
    }
    None
}

fn classify_chat_no_progress(
    raw_text: &str,
    parsed: &ParsedAssistant,
    completion_tokens: u64,
    required_tool_pending: bool,
    request: &ChatCompletionRequest,
) -> Option<NoProgressClass> {
    if parsed.tool_calls.is_empty() {
        if parsed.content.trim().is_empty()
            && parsed
                .reasoning
                .as_deref()
                .is_some_and(|reasoning| !reasoning.trim().is_empty())
        {
            return Some(NoProgressClass::HiddenOnlyOutput);
        }
        if let Some(class) =
            classify_no_progress(&parsed.content, completion_tokens, required_tool_pending)
        {
            return Some(class);
        }
        if repeated_assistant_content(&parsed.content, request) {
            return Some(NoProgressClass::RepeatedAssistantContent);
        }
        if stalled_assistant_turn(&parsed.content) {
            return Some(NoProgressClass::StalledAssistantTurn);
        }
    } else if repeated_invalid_tool_call(parsed, request) {
        return Some(NoProgressClass::RepeatedInvalidToolCall);
    }
    if raw_text.trim().is_empty() {
        return classify_no_progress(raw_text, completion_tokens, required_tool_pending);
    }
    None
}

fn repeated_assistant_content(content: &str, request: &ChatCompletionRequest) -> bool {
    let normalized = normalized_progress_text(content);
    if normalized.is_empty() {
        return false;
    }
    request
        .messages
        .iter()
        .rev()
        .filter(|message| message.role == ChatRole::Assistant && message.tool_calls.is_empty())
        .filter_map(|message| message.content.as_deref())
        .map(normalized_progress_text)
        .any(|previous| previous == normalized)
}

fn repeated_invalid_tool_call(parsed: &ParsedAssistant, request: &ChatCompletionRequest) -> bool {
    parsed.tool_calls.iter().any(|generated| {
        request
            .messages
            .iter()
            .enumerate()
            .rev()
            .any(|(index, message)| {
                message.role == ChatRole::Assistant
                    && message
                        .tool_calls
                        .iter()
                        .any(|previous| same_tool_call(previous, generated))
                    && following_tool_result_failed(&request.messages[index + 1..])
            })
    })
}

fn same_tool_call(previous: &ToolCall, generated: &ToolCall) -> bool {
    previous.function.name == generated.function.name
        && previous.function.arguments == generated.function.arguments
}

fn following_tool_result_failed(messages: &[ChatMessage]) -> bool {
    for message in messages {
        if message.role == ChatRole::User {
            return false;
        }
        if message.role == ChatRole::Tool
            && message
                .content
                .as_deref()
                .is_some_and(tool_result_indicates_failure)
        {
            return true;
        }
    }
    false
}

fn tool_result_indicates_failure(content: &str) -> bool {
    let lower = content.to_ascii_lowercase();
    [
        "error",
        "failed",
        "failure",
        "invalid",
        "not found",
        "no such file",
        "denied",
        "timeout",
        "exception",
        "panic",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

fn stalled_assistant_turn(content: &str) -> bool {
    let normalized = normalized_progress_text(content);
    if normalized.is_empty() || normalized.split_whitespace().count() > 16 {
        return false;
    }
    [
        "i will get started",
        "i ll get started",
        "i will check",
        "i will look",
        "let me check",
        "let me look",
        "working on it",
        "i can help with that",
        "sure i can help",
    ]
    .iter()
    .any(|phrase| normalized.contains(phrase))
}

fn normalized_progress_text(content: &str) -> String {
    content
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error(transparent)]
    Api(#[from] ApiError),
    #[error(transparent)]
    Backend(#[from] BackendError),
    #[error(transparent)]
    Template(#[from] TemplateError),
    #[error(transparent)]
    Parser(#[from] ParserError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    JsonMode(String),
    #[error("tool call validation failed: {0}")]
    ToolCallValidation(String),
    #[error("no progress classified as {0:?}")]
    NoProgress(NoProgressClass),
}
