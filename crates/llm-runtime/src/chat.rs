use crate::RuntimeError;
use crate::adapters::ChatAdapter;
use crate::chat_streaming::streaming_chat_stream;
use crate::json_mode::{parse_chat_text, validate_json_object_response};
use crate::no_progress::{
    classify_chat_no_progress, classify_repeated_invalid_tool_call_no_progress,
};
use crate::runtime::Runtime;
use crate::stop::apply_stop_sequences;
use crate::streaming::{
    CancelOnDrop, ChatCompletionStream, ChatCompletionStreamEvent, ChatCompletionStreamStage,
    RuntimeChatCompletion, RuntimeCompletionSeed, api_finish_reason, stream_chunk,
    stream_usage_chunk, usage_from_tokens,
};
use crate::tool_call::{
    fill_missing_tool_intent_arguments, required_backend_tool_choice, tool_call_delta,
    validate_tool_call_arguments,
};
use crate::tool_schema::validate_tool_calls_against_request;
use chrono::Utc;
use futures::{StreamExt, stream};
use llm_api::{
    ApiError, ChatCompletionChoice, ChatCompletionDelta, ChatCompletionRequest,
    ChatCompletionResponse, ChatMessage, ChatRole, ResponseFormat, ToolChoice, ValidateRequest,
    Validated,
};
use llm_backend::{BackendRequest, ModelBackend, SamplingConfig};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

impl<B> Runtime<B>
where
    B: ModelBackend,
{
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
        let request = request.into_validated_with_limits(self.options.request_limits)?;
        self.chat_validated_with_cancel(request, cancellation).await
    }

    #[doc(hidden)]
    pub async fn chat_validated_with_cancel(
        &self,
        request: Validated<ChatCompletionRequest>,
        cancellation: CancellationToken,
    ) -> Result<ChatCompletionResponse, RuntimeError> {
        if request.as_ref().stream {
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
        let request = request.into_validated_with_limits(self.options.request_limits)?;
        self.chat_stream_validated_with_cancel(request, cancellation)
            .await
    }

    #[doc(hidden)]
    pub async fn chat_stream_validated_with_cancel(
        &self,
        request: Validated<ChatCompletionRequest>,
        cancellation: CancellationToken,
    ) -> Result<ChatCompletionStream<'_>, RuntimeError> {
        let request_ref = request.as_ref();
        let include_usage = request_ref.stream_options.include_usage;
        let adapter = self.chat_adapter()?;
        let (cache_context, prompt, chat_context) =
            self.prepare_chat_backend(adapter, request_ref)?;
        let completion = RuntimeCompletionSeed {
            id: format!("chatcmpl-{}", Uuid::now_v7()),
            created: Utc::now().timestamp(),
            model: request_ref.model.clone(),
        };
        let backend_stream = self.backend.generate_stream_with_cancel(
            BackendRequest {
                model: request_ref.model.clone(),
                prompt,
                chat_context,
                max_tokens: request_ref.effective_max_tokens(),
                sampling: SamplingConfig::from_openai_controls(
                    request_ref.temperature,
                    request_ref.top_p,
                )?,
                required_tool_choice: required_backend_tool_choice(request_ref),
                json_object_mode: matches!(
                    request_ref.response_format.as_ref(),
                    Some(ResponseFormat::JsonObject)
                ),
                conversation_mode: true,
                cache_context,
            },
            cancellation.clone(),
        );
        let request = request.into_inner();
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
        let request = request.into_validated_with_limits(self.options.request_limits)?;
        self.chat_stream_buffered_validated_with_cancel(request, cancellation)
            .await
    }

    #[doc(hidden)]
    pub async fn chat_stream_buffered_validated_with_cancel(
        &self,
        request: Validated<ChatCompletionRequest>,
        cancellation: CancellationToken,
    ) -> Result<ChatCompletionStream<'static>, RuntimeError> {
        let include_usage = request.as_ref().stream_options.include_usage;
        let completion = self.complete_chat(request, cancellation).await?;
        buffered_chat_stream(completion, include_usage)
    }

    async fn complete_chat(
        &self,
        request: Validated<ChatCompletionRequest>,
        cancellation: CancellationToken,
    ) -> Result<RuntimeChatCompletion, RuntimeError> {
        let adapter = self.chat_adapter()?;
        let request_ref = request.as_ref();
        let (cache_context, prompt, chat_context) =
            self.prepare_chat_backend(adapter, request_ref)?;
        let required_tool_choice = required_backend_tool_choice(request_ref);
        let _cancel_on_drop = CancelOnDrop::new(cancellation.clone());
        let output = self
            .backend
            .generate_with_cancel(
                BackendRequest {
                    model: request_ref.model.clone(),
                    prompt,
                    chat_context,
                    max_tokens: request_ref.effective_max_tokens(),
                    sampling: SamplingConfig::from_openai_controls(
                        request_ref.temperature,
                        request_ref.top_p,
                    )?,
                    required_tool_choice,
                    json_object_mode: matches!(
                        request_ref.response_format.as_ref(),
                        Some(ResponseFormat::JsonObject)
                    ),
                    conversation_mode: true,
                    cache_context,
                },
                cancellation,
            )
            .await?;
        let request = request.into_inner();
        let mut raw_text = output.text;
        let stopped = apply_stop_sequences(&mut raw_text, &request.stop);
        let mut parsed = parse_chat_text(adapter, &raw_text, &request)?;
        validate_tool_call_arguments(&parsed)?;
        fill_missing_tool_intent_arguments(&mut parsed, &request);
        if let Some(class) = classify_repeated_invalid_tool_call_no_progress(&parsed, &request) {
            return Err(RuntimeError::NoProgress(class));
        }
        validate_tool_calls_against_request(&parsed, &request)?;
        if matches!(
            request.response_format.as_ref(),
            Some(ResponseFormat::JsonObject)
        ) {
            validate_json_object_response(&parsed)?;
        }
        let required_tool_pending = matches!(
            request.tool_choice.as_ref(),
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
            api_finish_reason(output.finish_reason)
        };
        let usage = usage_from_tokens(
            output.prompt_tokens,
            output.completion_tokens,
            output.prompt_cached_tokens,
        );
        Ok(RuntimeChatCompletion {
            id: format!("chatcmpl-{}", Uuid::now_v7()),
            created: Utc::now().timestamp(),
            model: request.model,
            parsed,
            finish_reason,
            usage,
        })
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
        if index == 0 {
            events.push(Ok(ChatCompletionStreamEvent::Stage(
                ChatCompletionStreamStage::ToolArgumentAssemblyComplete,
            )));
            events.push(Ok(ChatCompletionStreamEvent::Stage(
                ChatCompletionStreamStage::ToolIntentFillComplete,
            )));
            events.push(Ok(ChatCompletionStreamEvent::Stage(
                ChatCompletionStreamStage::ToolSchemaValidationComplete,
            )));
        }
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
    Ok(ChatCompletionStream::new(stream::iter(events).boxed()))
}
