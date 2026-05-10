use async_trait::async_trait;
use futures::{
    StreamExt,
    stream::{self, BoxStream},
};
use llm_api::FinishReason;
use std::path::PathBuf;
use thiserror::Error;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, PartialEq)]
pub struct BackendRequest {
    pub model: String,
    pub prompt: String,
    pub chat_context: Option<BackendChatContext>,
    pub max_tokens: Option<u32>,
    pub sampling: SamplingConfig,
    pub required_tool_choice: Option<BackendToolChoice>,
    pub json_object_mode: bool,
    pub conversation_mode: bool,
    pub cache_context: BackendCacheContext,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendChatContext {
    pub messages: Vec<BackendChatMessage>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendChatMessage {
    pub role: BackendChatRole,
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendChatRole {
    System,
    User,
    Assistant,
    Tool,
}

impl BackendChatRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BackendCacheContext {
    pub prompt_template: String,
    pub tool_schema: Option<String>,
}

impl BackendCacheContext {
    pub fn raw_prompt() -> Self {
        Self {
            prompt_template: "raw-prompt/v1".to_owned(),
            tool_schema: None,
        }
    }

    pub fn chat_template(template_id: impl Into<String>, tool_schema: Option<String>) -> Self {
        Self {
            prompt_template: template_id.into(),
            tool_schema,
        }
    }
}

impl Default for BackendCacheContext {
    fn default() -> Self {
        Self::raw_prompt()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendToolChoice {
    RequiredAny,
    RequiredFunction(String),
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum SamplingConfig {
    #[default]
    Greedy,
    TopP {
        temperature: f32,
        top_p: f32,
    },
}

impl SamplingConfig {
    pub fn from_openai_controls(temperature: Option<f32>, top_p: Option<f32>) -> Self {
        match (temperature, top_p) {
            (Some(0.0), _) => Self::Greedy,
            (None, None) => Self::Greedy,
            (t, p) => Self::TopP {
                temperature: t.unwrap_or(1.0),
                top_p: p.unwrap_or(1.0),
            },
        }
    }

    pub fn is_greedy(self) -> bool {
        matches!(self, Self::Greedy)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendOutput {
    pub text: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendStreamChunk {
    pub text: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendModelMetadata {
    pub id: String,
    pub backend: String,
    pub family: Option<String>,
    pub loader: Option<String>,
    pub quantization: Option<String>,
    pub repo_id: Option<String>,
    pub resolved_commit: Option<String>,
    pub profile: Option<String>,
    pub snapshot_path: Option<PathBuf>,
    pub manifest_digest: Option<String>,
}

impl BackendModelMetadata {
    pub fn new(id: impl Into<String>, backend: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            backend: backend.into(),
            family: None,
            loader: None,
            quantization: None,
            repo_id: None,
            resolved_commit: None,
            profile: None,
            snapshot_path: None,
            manifest_digest: None,
        }
    }

    pub fn with_family(mut self, family: impl Into<String>) -> Self {
        self.family = Some(family.into());
        self
    }
}

#[async_trait]
pub trait ModelBackend: Send + Sync + 'static {
    fn model_id(&self) -> &str;

    fn model_metadata(&self) -> BackendModelMetadata {
        BackendModelMetadata::new(self.model_id(), "unknown")
    }

    /// Non-cancellable generation entry point for direct backend callers.
    async fn generate(&self, request: BackendRequest) -> Result<BackendOutput, BackendError>;

    /// Cancellable generation entry point used by the production runtime.
    ///
    /// Implementations must observe a pre-cancelled token and should bound
    /// cancellation latency during long-running prefill/decode work.
    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError>;

    fn generate_stream<'a>(
        &'a self,
        request: BackendRequest,
    ) -> BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        stream::once(async move {
            self.generate(request)
                .await
                .map(|output| BackendStreamChunk {
                    text: output.text,
                    prompt_tokens: output.prompt_tokens,
                    completion_tokens: output.completion_tokens,
                    finish_reason: Some(output.finish_reason),
                })
        })
        .boxed()
    }

    fn generate_stream_with_cancel<'a>(
        &'a self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        stream::once(async move {
            self.generate_with_cancel(request, cancellation)
                .await
                .map(|output| BackendStreamChunk {
                    text: output.text,
                    prompt_tokens: output.prompt_tokens,
                    completion_tokens: output.completion_tokens,
                    finish_reason: Some(output.finish_reason),
                })
        })
        .boxed()
    }
}

#[async_trait]
impl<T> ModelBackend for Box<T>
where
    T: ModelBackend + ?Sized,
{
    fn model_id(&self) -> &str {
        (**self).model_id()
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        (**self).model_metadata()
    }

    async fn generate(&self, request: BackendRequest) -> Result<BackendOutput, BackendError> {
        (**self).generate(request).await
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        (**self).generate_with_cancel(request, cancellation).await
    }

    fn generate_stream<'a>(
        &'a self,
        request: BackendRequest,
    ) -> BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        (**self).generate_stream(request)
    }

    fn generate_stream_with_cancel<'a>(
        &'a self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        (**self).generate_stream_with_cancel(request, cancellation)
    }
}

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("model `{requested}` is not loaded; available model is `{available}`")]
    ModelNotFound {
        requested: String,
        available: String,
    },
    #[error("unsupported backend request: {0}")]
    UnsupportedRequest(String),
    #[error("backend generation cancelled")]
    Cancelled,
    #[error("backend error: {0}")]
    Other(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use futures::{StreamExt, executor::block_on};
    use llm_api::FinishReason;
    use tokio_util::sync::CancellationToken;

    struct CancelAwareBackend;

    #[async_trait]
    impl ModelBackend for CancelAwareBackend {
        fn model_id(&self) -> &str {
            "local-qwen36"
        }

        async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
            Ok(BackendOutput {
                text: "uncancelled".to_owned(),
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
            if cancellation.is_cancelled() {
                return Err(BackendError::Cancelled);
            }
            self.generate(request).await
        }
    }

    #[test]
    fn default_stream_with_cancel_uses_cancellable_generation() {
        let backend = CancelAwareBackend;
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let mut stream =
            backend.generate_stream_with_cancel(backend_request("hello"), cancellation);

        let result = block_on(stream.next()).expect("stream emits one result");

        assert!(matches!(result, Err(BackendError::Cancelled)));
    }

    fn backend_request(prompt: &str) -> BackendRequest {
        BackendRequest {
            model: "local-qwen36".to_owned(),
            prompt: prompt.to_owned(),
            chat_context: None,
            max_tokens: Some(1),
            sampling: SamplingConfig::Greedy,
            required_tool_choice: None,
            json_object_mode: false,
            conversation_mode: false,
            cache_context: BackendCacheContext::default(),
        }
    }

    #[test]
    fn from_openai_controls_maps_none_temperature_and_top_p_one_to_top_p() {
        assert_eq!(
            SamplingConfig::from_openai_controls(None, Some(1.0)),
            SamplingConfig::TopP {
                temperature: 1.0,
                top_p: 1.0,
            }
        );

        assert_eq!(SamplingConfig::from_openai_controls(None, None), SamplingConfig::Greedy);

        assert_eq!(
            SamplingConfig::from_openai_controls(Some(0.0), Some(1.0)),
            SamplingConfig::Greedy
        );
    }
}
