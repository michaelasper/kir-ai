use crate::RuntimeError;
use crate::no_progress::classify_no_progress;
use crate::runtime::Runtime;
use crate::stop::{
    apply_stop_sequences, earliest_stop_index, max_stop_sequence_len, safe_stream_emit_len,
};
use crate::streaming::{
    CancelOnDrop, CompletionStream, CompletionStreamEvent, RuntimeCompletion,
    RuntimeCompletionSeed, api_finish_reason, completion_stream_seed_chunk, max_optional_u64,
    usage_from_tokens,
};
use chrono::Utc;
use futures::{StreamExt, stream::BoxStream};
use llm_api::{ApiError, CompletionChoice, CompletionRequest, CompletionResponse, ValidateRequest};
use llm_backend::{
    BackendCacheContext, BackendError, BackendRequest, BackendStreamChunk, ModelBackend,
    SamplingConfig,
};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

impl<B> Runtime<B>
where
    B: ModelBackend,
{
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
        request.validate_with_limits(self.options.request_limits)?;
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
        request.validate_with_limits(self.options.request_limits)?;
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

    async fn complete_text(
        &self,
        request: CompletionRequest,
        cancellation: CancellationToken,
    ) -> Result<RuntimeCompletion, RuntimeError> {
        request.validate_with_limits(self.options.request_limits)?;
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
        let usage = usage_from_tokens(
            output.prompt_tokens,
            output.completion_tokens,
            output.prompt_cached_tokens,
        );
        Ok(RuntimeCompletion {
            id: format!("cmpl-{}", Uuid::now_v7()),
            created: Utc::now().timestamp(),
            model: request.model,
            text,
            finish_reason: if stopped {
                llm_api::FinishReason::Stop
            } else {
                api_finish_reason(output.finish_reason)
            },
            usage,
        })
    }
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
        let mut prompt_cached_tokens = None;
        let mut completion_tokens = 0;
        let mut finish_reason = llm_api::FinishReason::Length;
        let max_stop_len = max_stop_sequence_len(&stop);
        while let Some(chunk) = backend_stream.next().await {
            let chunk = chunk?;
            prompt_tokens = prompt_tokens.max(chunk.prompt_tokens);
            prompt_cached_tokens = max_optional_u64(prompt_cached_tokens, chunk.prompt_cached_tokens);
            completion_tokens += chunk.completion_tokens;
            if let Some(progress) = chunk.progress.clone() {
                yield CompletionStreamEvent::Progress(progress);
            }
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
                finish_reason = api_finish_reason(reason);
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
