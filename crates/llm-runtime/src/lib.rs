use chrono::Utc;
use futures::{
    StreamExt,
    stream::{self, BoxStream},
};
use llm_api::{
    ApiError, ChatCompletionChoice, ChatCompletionDelta, ChatCompletionRequest,
    ChatCompletionResponse, ChatCompletionStreamChoice, ChatCompletionStreamResponse, ChatMessage,
    ChatRole, CompletionChoice, CompletionRequest, CompletionResponse, CompletionStreamResponse,
    ResponseFormat, ToolCall, ToolCallDelta, ToolCallFunctionDelta, ToolChoice, Usage,
    ValidateRequest,
};
use llm_backend::{BackendCacheContext, BackendModelMetadata};
use llm_backend::{
    BackendError, BackendRequest, BackendStreamChunk, BackendToolChoice, ModelBackend,
    SamplingConfig,
};
use llm_tool_parser::ParsedAssistant;
use std::collections::BTreeSet;
use std::fmt;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

mod adapters;
mod error;
mod no_progress;

use adapters::{ChatAdapter, SelectedChatAdapter, ToolMarkupPolicy, chat_adapter_for_metadata};
pub use error::RuntimeError;
use no_progress::classify_chat_no_progress;
pub use no_progress::{NoProgressClass, classify_no_progress};

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
                chat_context: None,
                max_tokens: request.max_tokens,
                sampling: SamplingConfig::from_openai_controls(request.temperature, request.top_p)?,
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
        request.validate()?;
        let include_usage = request.stream_options.include_usage;
        let adapter = self.chat_adapter()?;
        let cache_context = adapter.cache_context(&request.tools)?;
        let prompt = adapter.render_prompt(&request.messages, &request.tools)?;
        let chat_context = adapter.backend_chat_context(&request.messages, &request.tools);
        let completion = RuntimeCompletionSeed {
            id: format!("chatcmpl-{}", Uuid::now_v7()),
            created: Utc::now().timestamp(),
            model: request.model.clone(),
        };
        let backend_stream = self.backend.generate_stream_with_cancel(
            BackendRequest {
                model: request.model.clone(),
                prompt,
                chat_context,
                max_tokens: request.effective_max_tokens(),
                sampling: SamplingConfig::from_openai_controls(request.temperature, request.top_p)?,
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
        let chat_context = adapter.backend_chat_context(&request.messages, &request.tools);
        let required_tool_choice = required_backend_tool_choice(&request);
        let _cancel_on_drop = CancelOnDrop::new(cancellation.clone());
        let output = self
            .backend
            .generate_with_cancel(
                BackendRequest {
                    model: request.model.clone(),
                    prompt,
                    chat_context,
                    max_tokens: request.effective_max_tokens(),
                    sampling: SamplingConfig::from_openai_controls(
                        request.temperature,
                        request.top_p,
                    )?,
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
        let mut parsed = parse_chat_text(adapter, &raw_text, &request)?;
        validate_tool_call_arguments(&parsed)?;
        fill_missing_tool_intent_arguments(&mut parsed, &request);
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
            adapter.tool_markup_policy(),
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
                    chat_context: None,
                    max_tokens: request.max_tokens,
                    sampling: SamplingConfig::from_openai_controls(
                        request.temperature,
                        request.top_p,
                    )?,
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
        let no_progress = classify_no_progress(&text, output.completion_tokens);
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
        if let Some(class) = classify_no_progress(visible_text, completion_tokens) {
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

enum DeferredEmission {
    None,
    JsonObjectMode,
    ToolChoiceRequired,
    UnmarkedToolBuffer,
}

const UNMARKED_TOOL_BUFFER_FLUSH_THRESHOLD: usize = 256;

fn deferred_emission_strategy(
    json_object_mode: bool,
    requires_tool_choice: bool,
    buffers_unmarked_tool_candidates: bool,
) -> DeferredEmission {
    // Priority: JsonObjectMode > ToolChoiceRequired > UnmarkedToolBuffer > None.
    // Only one deferred mode can be active at a time; JsonObjectMode takes
    // precedence because it suppresses all inline emission and performs
    // post-parse validation that subsumes the other buffering strategies.
    if json_object_mode {
        DeferredEmission::JsonObjectMode
    } else if requires_tool_choice {
        DeferredEmission::ToolChoiceRequired
    } else if buffers_unmarked_tool_candidates {
        DeferredEmission::UnmarkedToolBuffer
    } else {
        DeferredEmission::None
    }
}

fn unmarked_tool_buffer_can_stream_text(
    raw_text: &str,
    tool_markup_policy: ToolMarkupPolicy,
) -> bool {
    !tool_markup_policy.contains_start(raw_text)
        && !looks_like_unmarked_tool_json_candidate(raw_text)
}

fn looks_like_unmarked_tool_json_candidate(raw_text: &str) -> bool {
    let trimmed = raw_text.trim_start();
    trimmed.is_empty()
        || trimmed.starts_with('{')
        || trimmed.starts_with('[')
        || trimmed.starts_with("```")
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
    let tool_markup_policy = adapter.tool_markup_policy();
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
        let mut stop_at_len = None;
        let deferred = deferred_emission_strategy(
            matches!(request.response_format, Some(ResponseFormat::JsonObject)),
            request_requires_tool_choice(&request),
            adapter.parses_unmarked_tool_calls() && !request.tools.is_empty(),
        );
        let mut emitted_tool_calls = 0;
        let max_stop_len = max_stop_sequence_len(&request.stop);
        while let Some(chunk) = backend_stream.next().await {
            let chunk = chunk?;
            prompt_tokens = prompt_tokens.max(chunk.prompt_tokens);
            completion_tokens += chunk.completion_tokens;
            if !chunk.text.is_empty() {
                raw_text.push_str(&chunk.text);
                if let Some(stop_at) = earliest_stop_index(&raw_text, &request.stop) {
                    if matches!(deferred, DeferredEmission::None) && stop_at > emitted_len
                    {
                        yield ChatCompletionStreamEvent::Chunk(stream_seed_chunk(
                            &completion,
                            ChatCompletionDelta {
                                content: Some(raw_text[emitted_len..stop_at].to_owned()),
                                ..ChatCompletionDelta::default()
                            },
                            None,
                            None,
                        ));
                        emitted_len = stop_at;
                    }
                    stop_at_len = Some(stop_at);
                    finish_reason = llm_api::FinishReason::Stop;
                    stopped_by_sequence = true;
                    break;
                }
                if matches!(deferred, DeferredEmission::None) {
                    let safe_len = safe_stream_emit_len(&raw_text, max_stop_len)
                        .min(tool_markup_policy.safe_emit_len(&raw_text));
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
                if let Some(tool_prefix_len) = tool_markup_policy.completed_prefix_len(&raw_text)
                    && tool_prefix_len > emitted_len
                {
                    let mut parsed_prefix = adapter.parse_complete(&raw_text[..tool_prefix_len])?;
                    validate_tool_call_arguments(&parsed_prefix)?;
                    fill_missing_tool_intent_arguments(&mut parsed_prefix, &request);
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
                if matches!(deferred, DeferredEmission::UnmarkedToolBuffer)
                    && unmarked_tool_buffer_can_stream_text(&raw_text, tool_markup_policy)
                {
                    let safe_len = safe_stream_emit_len(&raw_text, max_stop_len)
                        .min(tool_markup_policy.safe_emit_len(&raw_text));
                    if safe_len.saturating_sub(emitted_len)
                        >= UNMARKED_TOOL_BUFFER_FLUSH_THRESHOLD
                    {
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
            }
            if let Some(reason) = chunk.finish_reason {
                finish_reason = reason;
                break;
            }
        }
        let visible_len = stop_at_len.unwrap_or(raw_text.len());
        if !stopped_by_sequence
            && emitted_len < visible_len
            && matches!(deferred, DeferredEmission::None)
            && !tool_markup_policy.contains_start(&raw_text[..visible_len])
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
        let mut parsed = parse_chat_text(adapter, visible_text, &request)?;
        validate_tool_call_arguments(&parsed)?;
        fill_missing_tool_intent_arguments(&mut parsed, &request);
        validate_tool_calls_against_request(&parsed, &request)?;
        if matches!(deferred, DeferredEmission::JsonObjectMode) {
            validate_json_object_response(&parsed)?;
        }
        if let Some(class) = classify_chat_no_progress(
            visible_text,
            &parsed,
            completion_tokens,
            matches!(deferred, DeferredEmission::ToolChoiceRequired) && parsed.tool_calls.is_empty(),
            &request,
            tool_markup_policy,
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
        match deferred {
            DeferredEmission::JsonObjectMode if !parsed.content.is_empty() => {
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
            DeferredEmission::UnmarkedToolBuffer
                if parsed.tool_calls.is_empty() =>
            {
                if let Some(remaining_content) = parsed.content.get(emitted_len..)
                    && !remaining_content.is_empty()
                {
                    yield ChatCompletionStreamEvent::Chunk(stream_seed_chunk(
                        &completion,
                        ChatCompletionDelta {
                            content: Some(remaining_content.to_owned()),
                            ..ChatCompletionDelta::default()
                        },
                        None,
                        None,
                    ));
                }
            }
            _ => {}
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

fn parse_chat_text(
    adapter: SelectedChatAdapter,
    text: &str,
    request: &ChatCompletionRequest,
) -> Result<ParsedAssistant, RuntimeError> {
    if let Some(content) = unmarked_tool_json_without_declared_tools(request, text, adapter) {
        return Ok(ParsedAssistant::content(content));
    }
    if let Some(content) = json_object_mode_without_tools(request, text, adapter) {
        return Ok(ParsedAssistant::content(content));
    }
    adapter.parse_complete(text)
}

fn unmarked_tool_json_without_declared_tools(
    request: &ChatCompletionRequest,
    text: &str,
    adapter: SelectedChatAdapter,
) -> Option<String> {
    if !adapter.parses_unmarked_tool_calls()
        || !request.tools.is_empty()
        || adapter.tool_markup_policy().contains_start(text)
    {
        return None;
    }
    let content =
        unmarked_tool_json_candidate(text, adapter.unmarked_tool_json_truncation_tokens());
    serde_json::from_str::<serde_json::Value>(content)
        .is_ok_and(|value| value.is_object() || value.is_array())
        .then(|| content.to_owned())
}

fn json_object_mode_without_tools(
    request: &ChatCompletionRequest,
    text: &str,
    adapter: SelectedChatAdapter,
) -> Option<String> {
    if !matches!(request.response_format, Some(ResponseFormat::JsonObject))
        || !request.tools.is_empty()
        || adapter.tool_markup_policy().contains_start(text)
    {
        return None;
    }
    let content =
        unmarked_tool_json_candidate(text, adapter.unmarked_tool_json_truncation_tokens());
    json_object_response_candidate(content).map(str::to_owned)
}

fn json_object_response_candidate(text: &str) -> Option<&str> {
    let trimmed = text.trim();
    if json_value_is_object(trimmed) {
        return Some(trimmed);
    }
    if let Some(fenced) = markdown_fenced_json_object_candidate(trimmed) {
        return Some(fenced);
    }
    if !trimmed.starts_with('{')
        && let Some(candidate) = first_balanced_json_object(trimmed)
        && json_value_is_object(candidate)
    {
        return Some(candidate);
    }
    None
}

fn markdown_fenced_json_object_candidate(text: &str) -> Option<&str> {
    let rest = text.strip_prefix("```")?;
    let body_start = rest.find('\n')? + 1;
    let body_with_close = &rest[body_start..];
    let body_end = body_with_close.find("```")?;
    let candidate = body_with_close[..body_end].trim();
    json_value_is_object(candidate).then_some(candidate)
}

fn first_balanced_json_object(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let mut depth = 0_usize;
    let mut in_string = false;
    let mut escaped = false;
    for (offset, ch) in text[start..].char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    let end = start + offset + ch.len_utf8();
                    return Some(text[start..end].trim());
                }
            }
            _ => {}
        }
    }
    None
}

fn json_value_is_object(text: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(text).is_ok_and(|value| value.is_object())
}

fn unmarked_tool_json_candidate<'a>(
    text: &'a str,
    truncation_tokens: &'static [&'static str],
) -> &'a str {
    truncation_tokens
        .iter()
        .filter_map(|token| text.find(token))
        .min()
        .map_or(text, |index| &text[..index])
        .trim()
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

fn fill_missing_tool_intent_arguments(
    parsed: &mut ParsedAssistant,
    request: &ChatCompletionRequest,
) {
    for tool_call in &mut parsed.tool_calls {
        let Some(arguments) = tool_call.function.arguments.as_object_mut() else {
            continue;
        };
        if arguments.contains_key("_i") {
            continue;
        }
        let Some(tool) = request
            .tools
            .iter()
            .find(|tool| tool.function.name == tool_call.function.name)
        else {
            continue;
        };
        if schema_requires_string_intent_argument(&tool.function.parameters) {
            arguments.insert(
                "_i".to_owned(),
                serde_json::Value::String(default_tool_intent(&tool_call.function.name).to_owned()),
            );
        }
    }
}

fn schema_requires_string_intent_argument(schema: &serde_json::Value) -> bool {
    let Some(schema_object) = schema.as_object() else {
        return false;
    };
    let Some(required) = schema_object
        .get("required")
        .and_then(serde_json::Value::as_array)
    else {
        return false;
    };
    if !required.iter().any(|field| field.as_str() == Some("_i")) {
        return false;
    }
    let Some(intent_schema) = schema_object
        .get("properties")
        .and_then(serde_json::Value::as_object)
        .and_then(|properties| properties.get("_i"))
    else {
        return false;
    };
    intent_schema
        .get("type")
        .is_some_and(schema_type_accepts_string)
}

fn schema_type_accepts_string(schema_type: &serde_json::Value) -> bool {
    match schema_type {
        serde_json::Value::String(type_name) => type_name == "string",
        serde_json::Value::Array(types) => types
            .iter()
            .any(|type_name| type_name.as_str() == Some("string")),
        _ => false,
    }
}

fn default_tool_intent(tool_name: &str) -> &'static str {
    match tool_name {
        "read" => "Reading requested path",
        "bash" => "Running requested command",
        "edit" => "Editing requested file",
        "find" => "Finding requested files",
        name if name.contains("search") || name.contains("grep") => "Searching requested context",
        _ => "Calling requested tool",
    }
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
