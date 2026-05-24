use crate::RuntimeError;
use crate::backend_request::completion_backend_request;
use crate::no_progress::classify_no_progress;
use crate::runtime::Runtime;
use crate::stop::apply_stop_sequences;
use crate::streaming::{
    CancelOnDrop, CompletionStream, RuntimeCompletion, RuntimeCompletionSeed, api_finish_reason,
    streaming_completion_stream, usage_from_tokens,
};
use chrono::Utc;
use llm_api::{
    ApiError, CompletionChoice, CompletionRequest, CompletionResponse, ValidateRequest, Validated,
};
use llm_backend_contracts::ModelBackend;
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
        let request = request.into_validated_with_limits(self.options.request_limits)?;
        self.completion_validated_with_cancel(request, cancellation)
            .await
    }

    #[doc(hidden)]
    pub async fn completion_validated_with_cancel(
        &self,
        request: Validated<CompletionRequest>,
        cancellation: CancellationToken,
    ) -> Result<CompletionResponse, RuntimeError> {
        let request = self.ensure_runtime_validated(request)?;
        if request.as_ref().stream {
            return Err(ApiError::unsupported_capability(
                "streaming text completion requests must use Runtime::completion_stream",
            )
            .into());
        }
        self.validate_completion_request_capabilities(request.as_ref(), false)?;
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
        let request = request.into_validated_with_limits(self.options.request_limits)?;
        self.completion_stream_validated_with_cancel(request, cancellation)
            .await
    }

    #[doc(hidden)]
    pub async fn completion_stream_validated_with_cancel(
        &self,
        request: Validated<CompletionRequest>,
        cancellation: CancellationToken,
    ) -> Result<CompletionStream<'_>, RuntimeError> {
        let request = self.ensure_runtime_validated(request)?;
        let request_ref = request.as_ref();
        self.validate_completion_request_capabilities(request_ref, true)?;
        let include_usage = request_ref.stream_options.include_usage;
        let stop = request_ref.stop.clone();
        let completion = RuntimeCompletionSeed {
            id: format!("cmpl-{}", Uuid::now_v7()),
            created: Utc::now().timestamp(),
            model: request_ref.model.clone(),
        };
        let request = request.into_inner();
        let backend_request = completion_backend_request(request)?;
        tracing::debug!(
            operation = "runtime_backend_dispatch",
            request_kind = "completion",
            stream = true,
            model_id = completion.model.as_str(),
            prompt_bytes = backend_request.prompt().len(),
            max_tokens = ?backend_request.max_tokens,
            "dispatching runtime request to backend"
        );
        let backend_stream = self
            .backend
            .generate_stream_with_cancel(backend_request, cancellation.clone());
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
        request: Validated<CompletionRequest>,
        cancellation: CancellationToken,
    ) -> Result<RuntimeCompletion, RuntimeError> {
        let request = request.into_inner();
        let model = request.model.clone();
        let stop = request.stop.clone();
        let backend_request = completion_backend_request(request)?;
        tracing::debug!(
            operation = "runtime_backend_dispatch",
            request_kind = "completion",
            stream = false,
            model_id = model.as_str(),
            prompt_bytes = backend_request.prompt().len(),
            max_tokens = ?backend_request.max_tokens,
            "dispatching runtime request to backend"
        );
        let mut cancel_on_drop = CancelOnDrop::new(cancellation.clone());
        let output = self
            .backend
            .generate_with_cancel(backend_request, cancellation)
            .await;
        cancel_on_drop.disarm();
        let output = output?;
        let mut text = output.text;
        let stopped = apply_stop_sequences(&mut text, &stop);
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
            model,
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
