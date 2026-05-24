use crate::RuntimeError;
use crate::adapters::{ChatAdapter, SelectedChatAdapter, ToolMarkupStreamState};
use crate::json_mode::parse_chat_text;
use crate::no_progress::{
    classify_chat_no_progress, classify_no_progress,
    classify_repeated_invalid_tool_call_no_progress,
};
use crate::response_validation::{
    validate_json_object_response, validate_tool_call_arguments,
    validate_tool_calls_against_request,
};
use crate::stop::{IncrementalStopDetector, max_stop_sequence_len, safe_stream_emit_len};
use crate::tool_call::{
    StructuredToolDeltaAssembler, ToolCallDeltaSerializer, fill_missing_tool_intent_arguments,
    request_may_fill_tool_intent_arguments, request_requires_tool_choice,
    structured_tool_delta_without_arguments,
};
use futures::{StreamExt, stream::BoxStream};
use llm_api::{
    ChatCompletionDelta, ChatCompletionRequest, ChatCompletionStreamChoice,
    ChatCompletionStreamResponse, ChatRole, CompletionChoice, CompletionStreamResponse,
    ResponseFormat, Usage,
};
use llm_backend_contracts::{
    BackendError, BackendFinishReason, BackendStreamChunk, BackendStreamProgress,
    BackendToolCallDelta, BackendToolCallFunctionDelta, BackendToolCallType,
};
use llm_tool_parser::ParsedAssistant;
use std::{fmt, sync::Arc};
use tokio_util::sync::CancellationToken;

/// Runtime-owned stream for chat completion events.
///
/// Consumers should drain this into SSE chunks and emit exactly one terminal
/// `[DONE]` after the event stream ends. Dropping the stream cancels the
/// underlying backend generation through the cancellation token supplied at
/// creation.
pub struct ChatCompletionStream<'a> {
    events: BoxStream<'a, Result<ChatCompletionStreamEvent, RuntimeError>>,
}

impl fmt::Debug for ChatCompletionStream<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ChatCompletionStream { events: <stream> }")
    }
}

impl<'a> ChatCompletionStream<'a> {
    pub(crate) fn new(
        events: BoxStream<'a, Result<ChatCompletionStreamEvent, RuntimeError>>,
    ) -> Self {
        Self { events }
    }

    /// Consumes the wrapper and returns the underlying event stream.
    pub fn into_events(self) -> BoxStream<'a, Result<ChatCompletionStreamEvent, RuntimeError>> {
        self.events
    }

    /// Collects public chunks and final usage for tests and in-process callers.
    ///
    /// Progress and internal stage events are intentionally ignored because they
    /// are observability signals rather than OpenAI response chunks.
    pub async fn collect_chunks(
        self,
    ) -> Result<(Vec<ChatCompletionStreamResponse>, Usage), RuntimeError> {
        let mut chunks = Vec::new();
        let mut usage = None;
        let mut events = self.into_events();
        while let Some(event) = events.next().await {
            match event? {
                ChatCompletionStreamEvent::Chunk(chunk) => chunks.push(chunk),
                ChatCompletionStreamEvent::Progress(_) => {}
                ChatCompletionStreamEvent::InternalProgress { .. } => {}
                ChatCompletionStreamEvent::Stage(_) => {}
                ChatCompletionStreamEvent::Complete(final_usage) => usage = Some(final_usage),
            }
        }
        Ok((chunks, usage.unwrap_or_else(empty_usage)))
    }
}

/// Runtime-owned stream for legacy completion events.
pub struct CompletionStream<'a> {
    events: BoxStream<'a, Result<CompletionStreamEvent, RuntimeError>>,
}

impl fmt::Debug for CompletionStream<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("CompletionStream { events: <stream> }")
    }
}

impl<'a> CompletionStream<'a> {
    pub(crate) fn new(events: BoxStream<'a, Result<CompletionStreamEvent, RuntimeError>>) -> Self {
        Self { events }
    }

    /// Consumes the wrapper and returns the underlying event stream.
    pub fn into_events(self) -> BoxStream<'a, Result<CompletionStreamEvent, RuntimeError>> {
        self.events
    }

    /// Collects public chunks and final usage for tests and in-process callers.
    pub async fn collect_chunks(
        self,
    ) -> Result<(Vec<CompletionStreamResponse>, Usage), RuntimeError> {
        let mut chunks = Vec::new();
        let mut usage = None;
        let mut events = self.into_events();
        while let Some(event) = events.next().await {
            match event? {
                CompletionStreamEvent::Chunk(chunk) => chunks.push(chunk),
                CompletionStreamEvent::Progress(_) => {}
                CompletionStreamEvent::Complete(final_usage) => usage = Some(final_usage),
            }
        }
        Ok((chunks, usage.unwrap_or_else(empty_usage)))
    }
}

/// Event emitted while producing a streaming chat completion.
#[derive(Debug, Clone, PartialEq)]
pub enum ChatCompletionStreamEvent {
    /// OpenAI-compatible chunk safe to send to the client.
    Chunk(ChatCompletionStreamResponse),
    /// Backend progress signal, such as prefill progress or upstream timing.
    Progress(BackendStreamProgress),
    /// Internal forward-progress signal when bytes were consumed but no public delta is safe yet.
    InternalProgress { bytes: usize },
    /// Runtime validation stage marker for tool-call streaming observability.
    Stage(ChatCompletionStreamStage),
    /// Final usage emitted after all chunks have been produced.
    Complete(Usage),
}

impl ChatCompletionStreamEvent {
    /// Summarizes whether this event advanced visible stream progress.
    pub fn progress_metadata(&self) -> StreamProgressMetadata {
        match self {
            Self::Chunk(chunk) => chat_stream_progress_metadata(chunk),
            Self::Progress(_)
            | Self::InternalProgress { .. }
            | Self::Stage(_)
            | Self::Complete(_) => StreamProgressMetadata::default(),
        }
    }
}

/// Observable validation milestones for streaming chat tool calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatCompletionStreamStage {
    /// Tool arguments have been assembled into complete JSON values.
    ToolArgumentAssemblyComplete,
    /// Missing runtime-managed tool intent arguments have been filled.
    ToolIntentFillComplete,
    /// Tool calls have passed schema and request compatibility validation.
    ToolSchemaValidationComplete,
}

/// Event emitted while producing a streaming legacy text completion.
#[derive(Debug, Clone, PartialEq)]
pub enum CompletionStreamEvent {
    /// OpenAI-compatible chunk safe to send to the client.
    Chunk(CompletionStreamResponse),
    /// Backend progress signal, such as prefill progress or upstream timing.
    Progress(BackendStreamProgress),
    /// Final usage emitted after all chunks have been produced.
    Complete(Usage),
}

impl CompletionStreamEvent {
    /// Summarizes whether this event advanced visible stream progress.
    pub fn progress_metadata(&self) -> StreamProgressMetadata {
        match self {
            Self::Chunk(chunk) => completion_stream_progress_metadata(chunk),
            Self::Progress(_) | Self::Complete(_) => StreamProgressMetadata::default(),
        }
    }
}

/// Compact progress summary used by no-progress and stalled-stream observers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StreamProgressMetadata {
    real_delta_bytes: usize,
    has_tool_delta: bool,
    has_tool_call_finish: bool,
}

impl StreamProgressMetadata {
    /// Returns true when the event exposed non-empty content or tool-call bytes.
    pub const fn has_real_delta(self) -> bool {
        self.real_delta_bytes > 0
    }

    /// Returns true when the event included a tool-call delta.
    pub const fn has_tool_delta(self) -> bool {
        self.has_tool_delta
    }

    /// Returns true when the event completed with a tool-call finish reason.
    pub const fn has_tool_call_finish(self) -> bool {
        self.has_tool_call_finish
    }

    /// Number of bytes counted as externally visible progress.
    pub const fn real_delta_bytes(self) -> usize {
        self.real_delta_bytes
    }
}

fn chat_stream_progress_metadata(chunk: &ChatCompletionStreamResponse) -> StreamProgressMetadata {
    let mut metadata = StreamProgressMetadata::default();
    for choice in &chunk.choices {
        if let Some(content) = &choice.delta.content {
            metadata.real_delta_bytes += content.len();
        }
        for tool_call in &choice.delta.tool_calls {
            metadata.has_tool_delta = true;
            metadata.real_delta_bytes += tool_call_delta_progress_bytes(tool_call).max(1);
        }
        metadata.has_tool_call_finish |=
            choice.finish_reason.as_ref() == Some(&llm_api::FinishReason::ToolCalls);
    }
    metadata
}

fn completion_stream_progress_metadata(chunk: &CompletionStreamResponse) -> StreamProgressMetadata {
    StreamProgressMetadata {
        real_delta_bytes: chunk.choices.iter().map(|choice| choice.text.len()).sum(),
        ..StreamProgressMetadata::default()
    }
}

fn tool_call_delta_progress_bytes(tool_call: &llm_api::ToolCallDelta) -> usize {
    tool_call.id.as_ref().map_or(0, String::len)
        + tool_call.function.as_ref().map_or(0, |function| {
            function.name.as_ref().map_or(0, String::len)
                + function.arguments.as_ref().map_or(0, String::len)
        })
}

#[derive(Debug)]
pub(crate) struct CancelOnDrop {
    token: CancellationToken,
    armed: bool,
}

impl CancelOnDrop {
    pub(crate) fn new(token: CancellationToken) -> Self {
        Self { token, armed: true }
    }

    pub(crate) fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.token.cancel();
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RuntimeCompletionSeed {
    pub(crate) id: Arc<str>,
    pub(crate) created: i64,
    pub(crate) model: Arc<str>,
}

impl RuntimeCompletionSeed {
    pub(crate) fn new(id: String, created: i64, model: &str) -> Self {
        Self {
            id: Arc::from(id),
            created,
            model: Arc::from(model),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RuntimeCompletion {
    pub(crate) id: String,
    pub(crate) created: i64,
    pub(crate) model: String,
    pub(crate) text: String,
    pub(crate) finish_reason: llm_api::FinishReason,
    pub(crate) usage: Usage,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RuntimeChatCompletion {
    pub(crate) id: String,
    pub(crate) created: i64,
    pub(crate) model: String,
    pub(crate) parsed: ParsedAssistant,
    pub(crate) finish_reason: llm_api::FinishReason,
    pub(crate) usage: Usage,
}

fn empty_usage() -> Usage {
    Usage::new(0, 0)
}

pub(crate) fn usage_from_tokens(
    prompt_tokens: u64,
    completion_tokens: u64,
    prompt_cached_tokens: Option<u64>,
) -> Usage {
    Usage::new(prompt_tokens, completion_tokens).with_prompt_cached_tokens(prompt_cached_tokens)
}

pub(crate) fn max_optional_u64(current: Option<u64>, next: Option<u64>) -> Option<u64> {
    match (current, next) {
        (Some(current), Some(next)) => Some(current.max(next)),
        (Some(current), None) => Some(current),
        (None, Some(next)) => Some(next),
        (None, None) => None,
    }
}

pub(crate) fn api_finish_reason(reason: BackendFinishReason) -> llm_api::FinishReason {
    match reason {
        BackendFinishReason::Stop => llm_api::FinishReason::Stop,
        BackendFinishReason::Length => llm_api::FinishReason::Length,
        BackendFinishReason::ToolCalls => llm_api::FinishReason::ToolCalls,
        BackendFinishReason::ContentFilter => llm_api::FinishReason::ContentFilter,
        BackendFinishReason::Error => llm_api::FinishReason::Error,
    }
}

pub(crate) fn stream_seed_chunk(
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

pub(crate) fn completion_stream_seed_chunk(
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

pub(crate) fn streaming_completion_stream<'a>(
    completion: RuntimeCompletionSeed,
    backend_stream: BoxStream<'a, Result<BackendStreamChunk, BackendError>>,
    stop: Vec<String>,
    include_usage: bool,
    cancellation: CancellationToken,
) -> CompletionStream<'a> {
    let cancel_on_drop = CancelOnDrop::new(cancellation);
    let events = async_stream::try_stream! {
        let mut cancel_on_drop = cancel_on_drop;
        let mut backend_stream = backend_stream;
        let mut raw_text = String::new();
        let mut emitted_len = 0;
        let mut prompt_tokens = 0;
        let mut prompt_cached_tokens = None;
        let mut completion_tokens = 0_u64;
        let mut finish_reason = llm_api::FinishReason::Length;
        let max_stop_len = max_stop_sequence_len(&stop);
        let mut stop_detector = IncrementalStopDetector::new(&stop);
        while let Some(chunk) = backend_stream.next().await {
            let chunk = chunk?;
            prompt_tokens = prompt_tokens.max(chunk.prompt_tokens);
            prompt_cached_tokens = max_optional_u64(prompt_cached_tokens, chunk.prompt_cached_tokens);
            completion_tokens = completion_tokens.saturating_add(chunk.completion_tokens);
            if let Some(progress) = chunk.progress {
                yield CompletionStreamEvent::Progress(progress);
            }
            if !chunk.text.is_empty() {
                raw_text.push_str(&chunk.text);
                if let Some(stop_at) = stop_detector.observe(&raw_text, &stop) {
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
                finish_reason = api_finish_reason(reason);
                break;
            }
        }
        cancel_on_drop.disarm();
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
        let usage = usage_from_tokens(prompt_tokens, completion_tokens, prompt_cached_tokens);
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
    CompletionStream::new(events.boxed())
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
    tool_markup_state: ToolMarkupStreamState,
) -> bool {
    !tool_markup_state.contains_start() && !looks_like_unmarked_tool_json_candidate(raw_text)
}

fn looks_like_unmarked_tool_json_candidate(raw_text: &str) -> bool {
    let trimmed = raw_text.trim_start();
    trimmed.is_empty()
        || trimmed.starts_with('{')
        || trimmed.starts_with('[')
        || trimmed.starts_with("```")
}

pub(crate) fn streaming_chat_stream<'a>(
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
        let mut cancel_on_drop = cancel_on_drop;
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
        let mut prompt_cached_tokens = None;
        let mut completion_tokens = 0_u64;
        let mut finish_reason = llm_api::FinishReason::Length;
        let mut stopped_by_sequence = false;
        let mut stop_at_len = None;
        let deferred = deferred_emission_strategy(
            matches!(request.response_format, Some(ResponseFormat::JsonObject)),
            request_requires_tool_choice(&request),
            adapter.parses_unmarked_tool_calls() && !request.tools.is_empty(),
        );
        let mut emitted_tool_calls = 0;
        let mut parsed_tool_prefix_len = 0;
        let mut tool_markup_state = tool_markup_policy.stream_state();
        let mut tool_call_delta_serializer = ToolCallDeltaSerializer::default();
        let mut structured_tool_assembler = StructuredToolDeltaAssembler::default();
        let buffer_structured_tool_arguments = request_may_fill_tool_intent_arguments(&request);
        let max_stop_len = max_stop_sequence_len(&request.stop);
        let mut stop_detector = IncrementalStopDetector::new(&request.stop);
        while let Some(chunk) = backend_stream.next().await {
            let chunk = chunk?;
            let internal_progress_bytes = internal_progress_bytes(&chunk);
            let mut emitted_public_chunk = false;
            prompt_tokens = prompt_tokens.max(chunk.prompt_tokens);
            prompt_cached_tokens = max_optional_u64(prompt_cached_tokens, chunk.prompt_cached_tokens);
            completion_tokens = completion_tokens.saturating_add(chunk.completion_tokens);
            if let Some(progress) = chunk.progress {
                yield ChatCompletionStreamEvent::Progress(progress);
            }
            if !chunk.tool_call_deltas.is_empty() {
                let api_tool_call_deltas = chunk
                    .tool_call_deltas
                    .into_iter()
                    .map(api_tool_call_delta)
                    .collect::<Vec<_>>();
                for delta in &api_tool_call_deltas {
                    structured_tool_assembler.push(delta)?;
                }
                let tool_call_deltas = if buffer_structured_tool_arguments {
                    api_tool_call_deltas
                        .iter()
                        .filter_map(structured_tool_delta_without_arguments)
                        .collect::<Vec<_>>()
                } else {
                    api_tool_call_deltas
                };
                if !tool_call_deltas.is_empty() {
                    emitted_public_chunk = true;
                    yield ChatCompletionStreamEvent::Chunk(stream_seed_chunk(
                        &completion,
                        ChatCompletionDelta {
                            tool_calls: tool_call_deltas,
                            ..ChatCompletionDelta::default()
                        },
                        None,
                        None,
                    ));
                }
            }
            if !chunk.text.is_empty() {
                raw_text.push_str(&chunk.text);
                tool_markup_state.observe(&raw_text);
                if let Some(stop_at) = stop_detector.observe(&raw_text, &request.stop) {
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
                        .min(tool_markup_state.safe_emit_len(&raw_text));
                    if safe_len > emitted_len {
                        emitted_public_chunk = true;
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
                if let Some(tool_prefix_len) = tool_markup_state.completed_prefix_len()
                    && tool_prefix_len > parsed_tool_prefix_len
                {
                    let mut parsed_prefix = adapter.parse_complete(&raw_text[..tool_prefix_len])?;
                    validate_tool_call_arguments(&parsed_prefix)?;
                    fill_missing_tool_intent_arguments(&mut parsed_prefix, &request);
                    if let Some(class) =
                        classify_repeated_invalid_tool_call_no_progress(&parsed_prefix, &request)
                    {
                        Err(RuntimeError::NoProgress(class))?;
                    }
                    validate_tool_calls_against_request(&parsed_prefix, &request)?;
                    let defer_prefix_tool_calls = matches!(
                        deferred,
                        DeferredEmission::ToolChoiceRequired
                    ) && !parsed_prefix.content.is_empty();
                    if !defer_prefix_tool_calls {
                        for (index, tool_call) in parsed_prefix
                            .tool_calls
                            .iter()
                            .enumerate()
                            .skip(emitted_tool_calls)
                        {
                            let delta = tool_call_delta_serializer
                                .tool_call_delta(index, tool_call)?;
                            emitted_public_chunk = true;
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
                    parsed_tool_prefix_len = tool_prefix_len;
                }
                if matches!(deferred, DeferredEmission::UnmarkedToolBuffer)
                    && unmarked_tool_buffer_can_stream_text(&raw_text, tool_markup_state)
                {
                    let safe_len = safe_stream_emit_len(&raw_text, max_stop_len)
                        .min(tool_markup_state.safe_emit_len(&raw_text));
                    if safe_len.saturating_sub(emitted_len)
                        >= UNMARKED_TOOL_BUFFER_FLUSH_THRESHOLD
                    {
                        emitted_public_chunk = true;
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
            if internal_progress_bytes > 0 && !emitted_public_chunk {
                yield ChatCompletionStreamEvent::InternalProgress {
                    bytes: internal_progress_bytes,
                };
            }
            if let Some(reason) = chunk.finish_reason {
                finish_reason = api_finish_reason(reason);
                break;
            }
        }
        cancel_on_drop.disarm();
        let visible_len = stop_at_len.unwrap_or(raw_text.len());
        if !stopped_by_sequence
            && emitted_len < visible_len
            && matches!(deferred, DeferredEmission::None)
            && !tool_markup_state.contains_start()
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
        let structured_tool_deltas_seen = !structured_tool_assembler.is_empty();
        let mut parsed = if structured_tool_deltas_seen {
            structured_tool_assembler.into_parsed(visible_text)?
        } else {
            parse_chat_text(adapter, visible_text, &request)?
        };
        let tool_calls_seen = !parsed.tool_calls.is_empty();
        if tool_calls_seen {
            yield ChatCompletionStreamEvent::Stage(
                ChatCompletionStreamStage::ToolArgumentAssemblyComplete,
            );
        }
        validate_tool_call_arguments(&parsed)?;
        fill_missing_tool_intent_arguments(&mut parsed, &request);
        if let Some(class) = classify_repeated_invalid_tool_call_no_progress(&parsed, &request) {
            Err(RuntimeError::NoProgress(class))?;
        }
        if tool_calls_seen {
            yield ChatCompletionStreamEvent::Stage(
                ChatCompletionStreamStage::ToolIntentFillComplete,
            );
        }
        validate_tool_calls_against_request(&parsed, &request)?;
        if tool_calls_seen {
            yield ChatCompletionStreamEvent::Stage(
                ChatCompletionStreamStage::ToolSchemaValidationComplete,
            );
        }
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
        let usage = usage_from_tokens(prompt_tokens, completion_tokens, prompt_cached_tokens);
        match deferred {
            DeferredEmission::JsonObjectMode | DeferredEmission::ToolChoiceRequired
                if !parsed.content.is_empty() =>
            {
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
        if structured_tool_deltas_seen {
            if buffer_structured_tool_arguments {
                for (index, tool_call) in parsed.tool_calls.iter().enumerate() {
                    let delta = tool_call_delta_serializer
                        .tool_call_arguments_delta(index, tool_call)?;
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
            }
        } else {
            for (index, tool_call) in parsed.tool_calls.iter().enumerate().skip(emitted_tool_calls) {
                let delta = tool_call_delta_serializer
                    .tool_call_delta(index, tool_call)?;
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
    ChatCompletionStream::new(events.boxed())
}

fn api_tool_call_delta(delta: BackendToolCallDelta) -> llm_api::ToolCallDelta {
    llm_api::ToolCallDelta {
        index: delta.index,
        id: delta.id,
        call_type: delta.call_type.map(api_tool_call_type),
        function: delta.function.map(api_tool_call_function_delta),
    }
}

fn internal_progress_bytes(chunk: &BackendStreamChunk) -> usize {
    chunk.text.len()
        + chunk
            .tool_call_deltas
            .iter()
            .map(|delta| {
                delta
                    .function
                    .as_ref()
                    .and_then(|function| function.arguments.as_ref())
                    .map_or(0, String::len)
            })
            .sum::<usize>()
}

fn api_tool_call_type(call_type: BackendToolCallType) -> llm_api::ToolCallType {
    match call_type {
        BackendToolCallType::Function => llm_api::ToolCallType::Function,
    }
}

fn api_tool_call_function_delta(
    function: BackendToolCallFunctionDelta,
) -> llm_api::ToolCallFunctionDelta {
    llm_api::ToolCallFunctionDelta {
        name: function.name,
        arguments: function.arguments,
    }
}
