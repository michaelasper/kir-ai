use futures::StreamExt;
use llm_api::{
    ChatCompletionRequest, ChatMessage, ChatRole, CompletionRequest, FinishReason, ResponseFormat,
    ToolChoice, ToolDefinition,
};
use llm_backend::{
    BackendError, BackendModelMetadata, BackendOutput, BackendRequest, BackendStreamChunk,
    ModelBackend, ProtocolTestBackend, SamplingConfig,
};
use llm_models::ModelFamily;
use llm_runtime::{ChatCompletionStreamEvent, NoProgressClass, Runtime, RuntimeError};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{Notify, Semaphore};
use tokio_util::sync::CancellationToken;

#[path = "runtime_contract/chat.rs"]
mod chat;
#[path = "runtime_contract/completion.rs"]
mod completion;
#[path = "runtime_contract/family_adapters.rs"]
mod family_adapters;
#[path = "runtime_contract/json_mode.rs"]
mod json_mode;
#[path = "runtime_contract/no_progress.rs"]
mod no_progress;
#[path = "runtime_contract/streaming.rs"]
mod streaming;
#[path = "runtime_contract/tool_validation.rs"]
mod tool_validation;

struct RecordingBackend {
    observed_max_tokens: Arc<Mutex<Option<Option<u32>>>>,
}

struct RecordingSamplingBackend {
    observed_sampling: Arc<Mutex<Option<SamplingConfig>>>,
}

struct ReplayBackend {
    output: BackendOutput,
}

struct FamilyMetadataBackend {
    family: Option<String>,
}

struct FamilyStreamBackend {
    model_id: &'static str,
    family: &'static str,
    text: &'static str,
    finish_reason: FinishReason,
}

struct RecordingChatContextBackend {
    observed: Arc<Mutex<Option<BackendRequest>>>,
    family: &'static str,
}

struct MlxQwenMetadataBackend;
struct MlxGemmaMetadataBackend;
struct MlxDeepSeekMetadataBackend;
struct MlxLlamaMetadataBackend;

fn qwen_test_metadata(model_id: &str, backend: &str) -> BackendModelMetadata {
    BackendModelMetadata::new(model_id, backend).with_family("qwen")
}

#[async_trait::async_trait]
impl ModelBackend for RecordingBackend {
    fn model_id(&self) -> &str {
        "local-qwen36"
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "recording")
    }

    async fn generate(&self, request: BackendRequest) -> Result<BackendOutput, BackendError> {
        *self
            .observed_max_tokens
            .lock()
            .expect("observed max_tokens lock") = Some(request.max_tokens);
        Ok(BackendOutput {
            text: "hello".to_owned(),
            prompt_tokens: 1,
            completion_tokens: 1,
            finish_reason: FinishReason::Stop,
        })
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }
}

#[async_trait::async_trait]
impl ModelBackend for RecordingSamplingBackend {
    fn model_id(&self) -> &str {
        "local-qwen36"
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "recording-sampling")
    }

    async fn generate(&self, request: BackendRequest) -> Result<BackendOutput, BackendError> {
        *self
            .observed_sampling
            .lock()
            .expect("observed sampling lock") = Some(request.sampling);
        Ok(BackendOutput {
            text: "hello".to_owned(),
            prompt_tokens: 1,
            completion_tokens: 1,
            finish_reason: FinishReason::Stop,
        })
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }
}

#[async_trait::async_trait]
impl ModelBackend for FamilyMetadataBackend {
    fn model_id(&self) -> &str {
        "local-qwen36"
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        let mut metadata = BackendModelMetadata::new(self.model_id(), "metadata-test");
        metadata.family = self.family.clone();
        metadata
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        panic!("unsupported family should fail before backend generation")
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }
}

#[async_trait::async_trait]
impl ModelBackend for FamilyStreamBackend {
    fn model_id(&self) -> &str {
        self.model_id
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        BackendModelMetadata::new(self.model_id, "family-stream").with_family(self.family)
    }

    async fn generate(&self, request: BackendRequest) -> Result<BackendOutput, BackendError> {
        if request.model != self.model_id {
            return Err(BackendError::ModelNotFound {
                requested: request.model,
                available: self.model_id.to_owned(),
            });
        }
        Ok(BackendOutput {
            text: self.text.to_owned(),
            prompt_tokens: 1,
            completion_tokens: 1,
            finish_reason: self.finish_reason.clone(),
        })
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }
}

#[async_trait::async_trait]
impl ModelBackend for MlxQwenMetadataBackend {
    fn model_id(&self) -> &str {
        "local-qwen36"
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        let mut metadata = BackendModelMetadata::new(self.model_id(), "mlx");
        metadata.family = Some("qwen".to_owned());
        metadata.loader = Some("mlx".to_owned());
        metadata
    }

    async fn generate(&self, request: BackendRequest) -> Result<BackendOutput, BackendError> {
        assert!(
            request.prompt.contains("<|im_start|>user"),
            "Qwen adapter should render ChatML prompt: {}",
            request.prompt
        );
        Ok(BackendOutput {
            text: "hello from mlx".to_owned(),
            prompt_tokens: 1,
            completion_tokens: 3,
            finish_reason: FinishReason::Stop,
        })
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }
}

#[async_trait::async_trait]
impl ModelBackend for MlxGemmaMetadataBackend {
    fn model_id(&self) -> &str {
        "local-gemma4"
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        let mut metadata = BackendModelMetadata::new(self.model_id(), "mlx");
        metadata.family = Some("gemma".to_owned());
        metadata.loader = Some("mlx".to_owned());
        metadata
    }

    async fn generate(&self, request: BackendRequest) -> Result<BackendOutput, BackendError> {
        assert!(
            request.prompt.contains("<|turn>user\nsay hi<turn|>"),
            "Gemma adapter should render Gemma 4 prompt: {}",
            request.prompt
        );
        Ok(BackendOutput {
            text: "hello from gemma<turn|>".to_owned(),
            prompt_tokens: 1,
            completion_tokens: 3,
            finish_reason: FinishReason::Stop,
        })
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }
}

#[async_trait::async_trait]
impl ModelBackend for MlxDeepSeekMetadataBackend {
    fn model_id(&self) -> &str {
        "local-deepseek"
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        let mut metadata = BackendModelMetadata::new(self.model_id(), "mlx");
        metadata.family = Some("deep_seek".to_owned());
        metadata.loader = Some("mlx".to_owned());
        metadata
    }

    async fn generate(&self, request: BackendRequest) -> Result<BackendOutput, BackendError> {
        assert!(
            request.prompt.contains("<｜User｜>say hi<｜Assistant｜>"),
            "DeepSeek adapter should render DeepSeek prompt: {}",
            request.prompt
        );
        Ok(BackendOutput {
            text: "hello from deepseek<｜end▁of▁sentence｜>".to_owned(),
            prompt_tokens: 1,
            completion_tokens: 3,
            finish_reason: FinishReason::Stop,
        })
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }
}

#[async_trait::async_trait]
impl ModelBackend for MlxLlamaMetadataBackend {
    fn model_id(&self) -> &str {
        "local-llama"
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        let mut metadata = BackendModelMetadata::new(self.model_id(), "mlx");
        metadata.family = Some("llama".to_owned());
        metadata.loader = Some("mlx".to_owned());
        metadata
    }

    async fn generate(&self, request: BackendRequest) -> Result<BackendOutput, BackendError> {
        assert!(
            request
                .prompt
                .contains("<|start_header_id|>user<|end_header_id|>\n\nsay hi<|eot_id|>"),
            "Llama adapter should render Llama 3 prompt: {}",
            request.prompt
        );
        Ok(BackendOutput {
            text: "hello from llama<|eot_id|>".to_owned(),
            prompt_tokens: 1,
            completion_tokens: 3,
            finish_reason: FinishReason::Stop,
        })
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }
}

#[async_trait::async_trait]
impl ModelBackend for RecordingChatContextBackend {
    fn model_id(&self) -> &str {
        "local-gemma4"
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        BackendModelMetadata::new(self.model_id(), "recording").with_family(self.family)
    }

    async fn generate(&self, request: BackendRequest) -> Result<BackendOutput, BackendError> {
        *self.observed.lock().expect("observed request lock") = Some(request);
        Ok(BackendOutput {
            text: "hello from gemma".to_owned(),
            prompt_tokens: 4,
            completion_tokens: 3,
            finish_reason: FinishReason::Stop,
        })
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }
}

#[async_trait::async_trait]
impl ModelBackend for ReplayBackend {
    fn model_id(&self) -> &str {
        "local-qwen36"
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "replay")
    }

    async fn generate(&self, request: BackendRequest) -> Result<BackendOutput, BackendError> {
        if request.model != self.model_id() {
            return Err(BackendError::ModelNotFound {
                requested: request.model,
                available: self.model_id().to_owned(),
            });
        }
        Ok(self.output.clone())
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }
}

fn fixture_backend_output(value: &Value) -> BackendOutput {
    BackendOutput {
        text: value["text"]
            .as_str()
            .expect("backend output has text")
            .to_owned(),
        prompt_tokens: value["prompt_tokens"]
            .as_u64()
            .expect("backend output has prompt_tokens"),
        completion_tokens: value["completion_tokens"]
            .as_u64()
            .expect("backend output has completion_tokens"),
        finish_reason: match value["finish_reason"]
            .as_str()
            .expect("backend output has finish_reason")
        {
            "stop" => FinishReason::Stop,
            "length" => FinishReason::Length,
            "tool_calls" => FinishReason::ToolCalls,
            other => panic!("unknown fixture finish_reason `{other}`"),
        },
    }
}

struct BlockingTextBackend {
    release: Arc<Notify>,
}

struct StopStreamingBackend;

struct TwoChunkStreamBackend {
    first: Arc<Notify>,
    finish: Arc<Notify>,
}

struct ToolBoundaryStreamBackend {
    first: Arc<Semaphore>,
    finish: Arc<Semaphore>,
    model_id: &'static str,
    family: &'static str,
    text: &'static str,
}

struct CancellableStreamBackend {
    cancelled: Arc<Notify>,
}

struct CancellableGenerateBackend {
    started: Arc<Notify>,
    cancelled: Arc<Notify>,
}

#[async_trait::async_trait]
impl ModelBackend for CancellableStreamBackend {
    fn model_id(&self) -> &str {
        "local-qwen36"
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "cancellable-stream")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Ok(BackendOutput {
            text: "unused".to_owned(),
            prompt_tokens: 1,
            completion_tokens: 1,
            finish_reason: FinishReason::Stop,
        })
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }

    fn generate_stream_with_cancel<'a>(
        &'a self,
        _request: BackendRequest,
        cancellation: CancellationToken,
    ) -> futures::stream::BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        let cancelled = self.cancelled.clone();
        tokio::spawn(async move {
            cancellation.cancelled().await;
            cancelled.notify_waiters();
        });
        async_stream::try_stream! {
            let chunk = futures::future::pending::<BackendStreamChunk>().await;
            yield chunk;
        }
        .boxed()
    }
}

#[async_trait::async_trait]
impl ModelBackend for CancellableGenerateBackend {
    fn model_id(&self) -> &str {
        "local-qwen36"
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "cancellable-generate")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Ok(BackendOutput {
            text: "unused".to_owned(),
            prompt_tokens: 1,
            completion_tokens: 1,
            finish_reason: FinishReason::Stop,
        })
    }

    async fn generate_with_cancel(
        &self,
        _request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        let started = self.started.clone();
        let cancelled = self.cancelled.clone();
        tokio::spawn(async move {
            started.notify_waiters();
            cancellation.cancelled().await;
            cancelled.notify_waiters();
        });
        futures::future::pending().await
    }
}

#[async_trait::async_trait]
impl ModelBackend for TwoChunkStreamBackend {
    fn model_id(&self) -> &str {
        "local-qwen36"
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "two-chunk-stream")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        self.first.notified().await;
        self.finish.notified().await;
        Ok(BackendOutput {
            text: "first second".to_owned(),
            prompt_tokens: 1,
            completion_tokens: 2,
            finish_reason: FinishReason::Stop,
        })
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
        let first = self.first.clone();
        let finish = self.finish.clone();
        async_stream::try_stream! {
            first.notified().await;
            yield BackendStreamChunk {
                text: "first".to_owned(),
                prompt_tokens: 1,
                completion_tokens: 1,
                finish_reason: None,
            };
            finish.notified().await;
            yield BackendStreamChunk {
                text: " second".to_owned(),
                prompt_tokens: 1,
                completion_tokens: 1,
                finish_reason: Some(FinishReason::Stop),
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
            return futures::stream::once(async { Err(BackendError::Cancelled) }).boxed();
        }
        self.generate_stream(request)
    }
}

#[async_trait::async_trait]
impl ModelBackend for ToolBoundaryStreamBackend {
    fn model_id(&self) -> &str {
        self.model_id
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        BackendModelMetadata::new(self.model_id, "tool-boundary-stream").with_family(self.family)
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Err(BackendError::Other(
            "tool boundary streaming test must use generate_stream".to_owned(),
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
        let first = self.first.clone();
        let finish = self.finish.clone();
        async_stream::try_stream! {
            let _permit = first.acquire().await.expect("first semaphore open");
            yield BackendStreamChunk {
                text: self.text.to_owned(),
                prompt_tokens: 1,
                completion_tokens: 1,
                finish_reason: None,
            };
            let _permit = finish.acquire().await.expect("finish semaphore open");
            yield BackendStreamChunk {
                text: String::new(),
                prompt_tokens: 1,
                completion_tokens: 0,
                finish_reason: Some(FinishReason::ToolCalls),
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
            return futures::stream::once(async { Err(BackendError::Cancelled) }).boxed();
        }
        self.generate_stream(request)
    }
}

#[async_trait::async_trait]
impl ModelBackend for BlockingTextBackend {
    fn model_id(&self) -> &str {
        "local-qwen36"
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "blocking-text")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        self.release.notified().await;
        Ok(BackendOutput {
            text: "released".to_owned(),
            prompt_tokens: 1,
            completion_tokens: 1,
            finish_reason: FinishReason::Stop,
        })
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }
}

#[async_trait::async_trait]
impl ModelBackend for StopStreamingBackend {
    fn model_id(&self) -> &str {
        "local-qwen36"
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "stop-streaming")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Err(BackendError::Other(
            "stop streaming test must use generate_stream".to_owned(),
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
                text: "hello ST".to_owned(),
                prompt_tokens: 1,
                completion_tokens: 1,
                finish_reason: None,
            };
            yield BackendStreamChunk {
                text: "OP ignored".to_owned(),
                prompt_tokens: 1,
                completion_tokens: 1,
                finish_reason: Some(FinishReason::Stop),
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
            return futures::stream::once(async { Err(BackendError::Cancelled) }).boxed();
        }
        self.generate_stream(request)
    }
}

async fn generate_after_pre_cancel<B: ModelBackend + ?Sized>(
    backend: &B,
    request: BackendRequest,
    cancellation: CancellationToken,
) -> Result<BackendOutput, BackendError> {
    if cancellation.is_cancelled() {
        return Err(BackendError::Cancelled);
    }
    backend.generate(request).await
}
