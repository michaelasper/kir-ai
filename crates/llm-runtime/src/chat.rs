use crate::RuntimeError;
use crate::adapters::ChatAdapter;
use crate::json_mode::parse_chat_text;
use crate::no_progress::{
    classify_chat_no_progress, classify_repeated_invalid_tool_call_no_progress,
};
use crate::response_validation::{
    validate_json_object_response_format, validate_tool_call_arguments,
    validate_tool_calls_against_request,
};
use crate::runtime::Runtime;
use crate::stop::apply_stop_sequences;
use crate::streaming::{
    CancelOnDrop, ChatCompletionStream, RuntimeChatCompletion, RuntimeCompletionSeed,
    api_finish_reason, streaming_chat_stream, usage_from_tokens,
};
use crate::tool_call::fill_missing_tool_intent_arguments;
use chrono::Utc;
use llm_api::{
    ApiError, ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage,
    ChatRole, ToolChoice, ValidateRequest, Validated,
};
use llm_backend::ModelBackend;
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
        let request = self.ensure_runtime_validated(request)?;
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
        let request = self.ensure_runtime_validated(request)?;
        let request_ref = request.as_ref();
        let include_usage = request_ref.stream_options.include_usage;
        let adapter = self.chat_adapter()?;
        let backend_request = self.chat_backend_request(adapter, request_ref)?;
        let completion = RuntimeCompletionSeed {
            id: format!("chatcmpl-{}", Uuid::now_v7()),
            created: Utc::now().timestamp(),
            model: request_ref.model.clone(),
        };
        let backend_stream = self
            .backend
            .generate_stream_with_cancel(backend_request, cancellation.clone());
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

    async fn complete_chat(
        &self,
        request: Validated<ChatCompletionRequest>,
        cancellation: CancellationToken,
    ) -> Result<RuntimeChatCompletion, RuntimeError> {
        let adapter = self.chat_adapter()?;
        let request_ref = request.as_ref();
        let backend_request = self.chat_backend_request(adapter, request_ref)?;
        let mut cancel_on_drop = CancelOnDrop::new(cancellation.clone());
        let output = self
            .backend
            .generate_with_cancel(backend_request, cancellation)
            .await;
        cancel_on_drop.disarm();
        let output = output?;
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
        validate_json_object_response_format(&parsed, &request)?;
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
