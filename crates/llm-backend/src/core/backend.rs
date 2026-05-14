use async_trait::async_trait;
use futures::{
    StreamExt,
    stream::{self, BoxStream},
};
use llm_api::{ChatMessage, ChatRole, FinishReason, ToolCallDelta};
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

#[derive(Debug, Clone, PartialEq)]
pub struct BackendChatContext {
    pub messages: Vec<ChatMessage>,
}

pub type BackendChatMessage = ChatMessage;
pub type BackendChatRole = ChatRole;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BackendCacheContext {
    pub prompt_template: String,
    pub tool_schema: Option<String>,
    pub chat_template_kwargs: Option<String>,
}

impl BackendCacheContext {
    pub fn raw_prompt() -> Self {
        Self {
            prompt_template: "raw-prompt/v1".to_owned(),
            tool_schema: None,
            chat_template_kwargs: None,
        }
    }

    pub fn chat_template(template_id: impl Into<String>, tool_schema: Option<String>) -> Self {
        Self::chat_template_with_kwargs(template_id, tool_schema, None)
    }

    pub fn chat_template_with_kwargs(
        template_id: impl Into<String>,
        tool_schema: Option<String>,
        chat_template_kwargs: Option<String>,
    ) -> Self {
        Self {
            prompt_template: template_id.into(),
            tool_schema,
            chat_template_kwargs,
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
    /// Standard multinomial sampling with OpenAI default controls (temperature 1.0, top_p 1.0).
    pub fn standard() -> Self {
        Self::TopP {
            temperature: 1.0,
            top_p: 1.0,
        }
    }

    pub fn from_openai_controls(
        temperature: Option<f32>,
        top_p: Option<f32>,
    ) -> Result<Self, BackendError> {
        if let Some(t) = temperature
            && (!t.is_finite() || !(0.0..=2.0).contains(&t))
        {
            return Err(BackendError::InvalidSamplingConfig(
                "temperature must be finite and in [0, 2]".to_owned(),
            ));
        }
        if let Some(p) = top_p
            && (!p.is_finite() || p <= 0.0 || p > 1.0)
        {
            return Err(BackendError::InvalidSamplingConfig(
                "top_p must be finite and in (0, 1]".to_owned(),
            ));
        }
        Ok(match (temperature, top_p) {
            (Some(0.0), _) => Self::Greedy,
            (None, None) => Self::standard(),
            (t, p) => Self::TopP {
                temperature: t.unwrap_or(1.0),
                top_p: p.unwrap_or(1.0),
            },
        })
    }

    pub fn is_greedy(self) -> bool {
        matches!(self, Self::Greedy)
    }

    pub fn is_standard(self) -> bool {
        self == Self::standard()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendOutput {
    pub text: String,
    pub prompt_tokens: u64,
    pub prompt_cached_tokens: Option<u64>,
    pub completion_tokens: u64,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BackendStreamChunk {
    pub text: String,
    pub tool_call_deltas: Vec<ToolCallDelta>,
    pub prompt_tokens: u64,
    pub prompt_cached_tokens: Option<u64>,
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
                    tool_call_deltas: Vec::new(),
                    prompt_tokens: output.prompt_tokens,
                    prompt_cached_tokens: output.prompt_cached_tokens,
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
                    tool_call_deltas: Vec::new(),
                    prompt_tokens: output.prompt_tokens,
                    prompt_cached_tokens: output.prompt_cached_tokens,
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
    #[error("invalid sampling config: {0}")]
    InvalidSamplingConfig(String),
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
                prompt_cached_tokens: None,
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
            SamplingConfig::from_openai_controls(None, Some(1.0)).expect("valid controls"),
            SamplingConfig::TopP {
                temperature: 1.0,
                top_p: 1.0,
            }
        );

        assert_eq!(
            SamplingConfig::from_openai_controls(None, None).expect("valid controls"),
            SamplingConfig::TopP {
                temperature: 1.0,
                top_p: 1.0,
            }
        );

        assert_eq!(
            SamplingConfig::from_openai_controls(Some(0.0), Some(1.0)).expect("valid controls"),
            SamplingConfig::Greedy
        );
    }

    #[test]
    fn from_openai_controls_rejects_negative_temperature() {
        let err = SamplingConfig::from_openai_controls(Some(-0.5), None)
            .expect_err("negative temperature should be rejected");
        assert!(
            matches!(err, BackendError::InvalidSamplingConfig(_)),
            "expected InvalidSamplingConfig, got {err:?}"
        );
    }

    #[test]
    fn from_openai_controls_rejects_nan_temperature() {
        let err = SamplingConfig::from_openai_controls(Some(f32::NAN), None)
            .expect_err("NaN temperature should be rejected");
        assert!(
            matches!(err, BackendError::InvalidSamplingConfig(_)),
            "expected InvalidSamplingConfig, got {err:?}"
        );
    }

    #[test]
    fn from_openai_controls_rejects_inf_temperature() {
        let err = SamplingConfig::from_openai_controls(Some(f32::INFINITY), None)
            .expect_err("inf temperature should be rejected");
        assert!(
            matches!(err, BackendError::InvalidSamplingConfig(_)),
            "expected InvalidSamplingConfig, got {err:?}"
        );
    }

    #[test]
    fn from_openai_controls_rejects_temperature_above_2() {
        let err = SamplingConfig::from_openai_controls(Some(2.1), None)
            .expect_err("temperature > 2.0 should be rejected");
        assert!(
            matches!(err, BackendError::InvalidSamplingConfig(_)),
            "expected InvalidSamplingConfig, got {err:?}"
        );
    }

    #[test]
    fn from_openai_controls_accepts_temperature_at_upper_bound() {
        let config = SamplingConfig::from_openai_controls(Some(2.0), None)
            .expect("temperature 2.0 is valid");
        assert_eq!(
            config,
            SamplingConfig::TopP {
                temperature: 2.0,
                top_p: 1.0,
            }
        );
    }

    #[test]
    fn from_openai_controls_rejects_zero_top_p() {
        let err = SamplingConfig::from_openai_controls(None, Some(0.0))
            .expect_err("zero top_p should be rejected");
        assert!(
            matches!(err, BackendError::InvalidSamplingConfig(_)),
            "expected InvalidSamplingConfig, got {err:?}"
        );
    }

    #[test]
    fn from_openai_controls_rejects_neg_inf_temperature() {
        let err = SamplingConfig::from_openai_controls(Some(f32::NEG_INFINITY), None)
            .expect_err("neg_inf temperature should be rejected");
        assert!(
            matches!(err, BackendError::InvalidSamplingConfig(_)),
            "expected InvalidSamplingConfig, got {err:?}"
        );
    }

    #[test]
    fn from_openai_controls_rejects_top_p_above_1() {
        let err = SamplingConfig::from_openai_controls(None, Some(1.5))
            .expect_err("top_p > 1.0 should be rejected");
        assert!(
            matches!(err, BackendError::InvalidSamplingConfig(_)),
            "expected InvalidSamplingConfig, got {err:?}"
        );
    }

    #[test]
    fn from_openai_controls_rejects_inf_top_p() {
        let err = SamplingConfig::from_openai_controls(None, Some(f32::INFINITY))
            .expect_err("inf top_p should be rejected");
        assert!(
            matches!(err, BackendError::InvalidSamplingConfig(_)),
            "expected InvalidSamplingConfig, got {err:?}"
        );
    }

    #[test]
    fn from_openai_controls_rejects_nan_top_p() {
        let err = SamplingConfig::from_openai_controls(None, Some(f32::NAN))
            .expect_err("NaN top_p should be rejected");
        assert!(
            matches!(err, BackendError::InvalidSamplingConfig(_)),
            "expected InvalidSamplingConfig, got {err:?}"
        );
    }

    #[test]
    fn from_openai_controls_rejects_negative_top_p() {
        let err = SamplingConfig::from_openai_controls(None, Some(-0.1))
            .expect_err("negative top_p should be rejected");
        assert!(
            matches!(err, BackendError::InvalidSamplingConfig(_)),
            "expected InvalidSamplingConfig, got {err:?}"
        );
    }
}
