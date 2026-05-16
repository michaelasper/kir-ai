use crate::RuntimeError;
use crate::adapters::{ChatAdapter, SelectedChatAdapter, ToolMarkupPolicy};
use crate::json_mode::{parse_chat_text, validate_json_object_response};
use crate::no_progress::classify_chat_no_progress;
use crate::stop::{earliest_stop_index, max_stop_sequence_len, safe_stream_emit_len};
use crate::streaming::{
    CancelOnDrop, ChatCompletionStream, ChatCompletionStreamEvent, ChatCompletionStreamStage,
    RuntimeCompletionSeed, api_finish_reason, max_optional_u64, stream_seed_chunk,
    usage_from_tokens,
};
use crate::tool_call::{
    StructuredToolDeltaAssembler, fill_missing_tool_intent_arguments,
    request_may_fill_tool_intent_arguments, request_requires_tool_choice,
    structured_tool_delta_without_arguments, tool_call_arguments_delta, tool_call_delta,
    validate_tool_call_arguments,
};
use crate::tool_schema::validate_tool_calls_against_request;
use futures::{StreamExt, stream::BoxStream};
use llm_api::{ChatCompletionDelta, ChatCompletionRequest, ChatRole, ResponseFormat};
use llm_backend::{
    BackendError, BackendStreamChunk, BackendToolCallDelta, BackendToolCallFunctionDelta,
    BackendToolCallType,
};
use tokio_util::sync::CancellationToken;

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
        let mut prompt_cached_tokens = None;
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
        let mut structured_tool_assembler = StructuredToolDeltaAssembler::default();
        let buffer_structured_tool_arguments = request_may_fill_tool_intent_arguments(&request);
        let max_stop_len = max_stop_sequence_len(&request.stop);
        while let Some(chunk) = backend_stream.next().await {
            let chunk = chunk?;
            let internal_progress_bytes = internal_progress_bytes(&chunk);
            let mut emitted_public_chunk = false;
            prompt_tokens = prompt_tokens.max(chunk.prompt_tokens);
            prompt_cached_tokens = max_optional_u64(prompt_cached_tokens, chunk.prompt_cached_tokens);
            completion_tokens += chunk.completion_tokens;
            if let Some(progress) = chunk.progress.clone() {
                yield ChatCompletionStreamEvent::Progress(progress);
            }
            if !chunk.tool_call_deltas.is_empty() {
                let api_tool_call_deltas = chunk
                    .tool_call_deltas
                    .iter()
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
                if matches!(deferred, DeferredEmission::UnmarkedToolBuffer)
                    && unmarked_tool_buffer_can_stream_text(&raw_text, tool_markup_policy)
                {
                    let safe_len = safe_stream_emit_len(&raw_text, max_stop_len)
                        .min(tool_markup_policy.safe_emit_len(&raw_text));
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
        if structured_tool_deltas_seen {
            if buffer_structured_tool_arguments {
                for (index, tool_call) in parsed.tool_calls.iter().enumerate() {
                    let delta = tool_call_arguments_delta(index, tool_call)?;
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

fn api_tool_call_delta(delta: &BackendToolCallDelta) -> llm_api::ToolCallDelta {
    llm_api::ToolCallDelta {
        index: delta.index,
        id: delta.id.clone(),
        call_type: delta.call_type.as_ref().map(api_tool_call_type),
        function: delta.function.as_ref().map(api_tool_call_function_delta),
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

fn api_tool_call_type(call_type: &BackendToolCallType) -> llm_api::ToolCallType {
    match call_type {
        BackendToolCallType::Function => llm_api::ToolCallType::Function,
    }
}

fn api_tool_call_function_delta(
    function: &BackendToolCallFunctionDelta,
) -> llm_api::ToolCallFunctionDelta {
    llm_api::ToolCallFunctionDelta {
        name: function.name.clone(),
        arguments: function.arguments.clone(),
    }
}
