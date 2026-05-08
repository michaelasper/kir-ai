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
use llm_backend::BackendModelMetadata;
use llm_backend::{BackendError, BackendRequest, BackendStreamChunk, ModelBackend, SamplingConfig};
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
        request.validate()?;
        if request.stream {
            return Err(ApiError::unsupported_capability(
                "streaming text completion requests must use Runtime::completion_stream",
            )
            .into());
        }
        let completion = self.complete_text(request).await?;
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
        request.validate()?;
        let include_usage = request.stream_options.include_usage;
        let stop = request.stop.clone();
        let completion = RuntimeCompletionSeed {
            id: format!("cmpl-{}", Uuid::now_v7()),
            created: Utc::now().timestamp(),
            model: request.model.clone(),
        };
        let cancellation = CancellationToken::new();
        let backend_stream = self.backend.generate_stream_with_cancel(
            BackendRequest {
                model: request.model,
                prompt: request.prompt,
                max_tokens: request.max_tokens,
                sampling: SamplingConfig::from_openai_controls(request.temperature, request.top_p),
                required_tool_choice: None,
                json_object_mode: false,
                conversation_mode: false,
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
        if request.stream {
            return Err(ApiError::unsupported_capability(
                "streaming chat requests must use Runtime::chat_stream",
            )
            .into());
        }
        let completion = self.complete_chat(request).await?;
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
        if chat_stream_requires_buffering(&request) {
            return self.chat_stream_buffered(request).await;
        }
        request.validate()?;
        let include_usage = request.stream_options.include_usage;
        let prompt = render_qwen_chatml(
            &request.messages,
            &request.tools,
            &QwenPromptOptions {
                enable_thinking: false,
                add_generation_prompt: true,
            },
        )?;
        let completion = RuntimeCompletionSeed {
            id: format!("chatcmpl-{}", Uuid::now_v7()),
            created: Utc::now().timestamp(),
            model: request.model.clone(),
        };
        let cancellation = CancellationToken::new();
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
            },
            cancellation.clone(),
        );
        Ok(streaming_chat_stream(
            completion,
            request,
            backend_stream,
            include_usage,
            cancellation,
        ))
    }

    pub async fn chat_stream_buffered(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionStream<'static>, RuntimeError> {
        let include_usage = request.stream_options.include_usage;
        let completion = self.complete_chat(request).await?;
        buffered_chat_stream(completion, include_usage)
    }

    async fn complete_chat(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<RuntimeChatCompletion, RuntimeError> {
        request.validate()?;
        let prompt = render_qwen_chatml(
            &request.messages,
            &request.tools,
            &QwenPromptOptions {
                enable_thinking: false,
                add_generation_prompt: true,
            },
        )?;
        let required_tool_choice = required_backend_tool_choice(&request);
        let cancellation = CancellationToken::new();
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
                },
                cancellation,
            )
            .await?;
        let mut raw_text = output.text;
        let stopped = apply_stop_sequences(&mut raw_text, &request.stop);
        let parsed = QwenParser.parse_complete(&raw_text)?;
        validate_tool_call_arguments(&parsed)?;
        validate_tool_calls_against_request(&parsed, &request)?;
        if matches!(request.response_format, Some(ResponseFormat::JsonObject)) {
            validate_json_object_response(&parsed)?;
        }
        let required_tool_pending = matches!(
            request.tool_choice,
            Some(ToolChoice::Required | ToolChoice::Function { .. })
        );
        let no_progress = classify_no_progress(
            &raw_text,
            output.completion_tokens,
            required_tool_pending && parsed.tool_calls.is_empty(),
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

    async fn complete_text(
        &self,
        request: CompletionRequest,
    ) -> Result<RuntimeCompletion, RuntimeError> {
        request.validate()?;
        let cancellation = CancellationToken::new();
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
        let max_stop_len = max_stop_sequence_len(&request.stop);
        while let Some(chunk) = backend_stream.next().await {
            let chunk = chunk?;
            prompt_tokens = prompt_tokens.max(chunk.prompt_tokens);
            completion_tokens += chunk.completion_tokens;
            if !chunk.text.is_empty() {
                raw_text.push_str(&chunk.text);
                if let Some(stop_at) = earliest_stop_index(&raw_text, &request.stop) {
                    if stop_at > emitted_len {
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
                    break;
                }
                let safe_len = safe_stream_emit_len(&raw_text, max_stop_len);
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
            if let Some(reason) = chunk.finish_reason {
                finish_reason = reason;
                break;
            }
        }
        if finish_reason != llm_api::FinishReason::Stop && emitted_len < raw_text.len() {
            yield ChatCompletionStreamEvent::Chunk(stream_seed_chunk(
                &completion,
                ChatCompletionDelta {
                    content: Some(raw_text[emitted_len..].to_owned()),
                    ..ChatCompletionDelta::default()
                },
                None,
                None,
            ));
            emitted_len = raw_text.len();
        }

        let visible_text = &raw_text[..emitted_len];
        let parsed = QwenParser.parse_complete(visible_text)?;
        validate_tool_call_arguments(&parsed)?;
        validate_tool_calls_against_request(&parsed, &request)?;
        let required_tool_pending = matches!(
            request.tool_choice,
            Some(ToolChoice::Required | ToolChoice::Function { .. })
        );
        if let Some(class) = classify_no_progress(
            visible_text,
            completion_tokens,
            required_tool_pending && parsed.tool_calls.is_empty(),
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

pub fn chat_stream_requires_buffering(request: &ChatCompletionRequest) -> bool {
    !request.tools.is_empty()
        || !matches!(request.tool_choice, None | Some(ToolChoice::Auto))
        || matches!(request.response_format, Some(ResponseFormat::JsonObject))
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
    }
    Ok(())
}

fn required_backend_tool_choice(request: &ChatCompletionRequest) -> Option<String> {
    match &request.tool_choice {
        Some(ToolChoice::Required) => request.tools.first().map(|tool| tool.function.name.clone()),
        Some(ToolChoice::Function { name }) => Some(name.clone()),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoProgressClass {
    EmptyCompletion,
    EmptyHighOutputCompletion,
    TextFallbackRequiredTool,
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
