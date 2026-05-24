use async_trait::async_trait;
use futures::{
    StreamExt,
    stream::{self, BoxStream},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{fmt, sync::Arc};
use thiserror::Error;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, PartialEq)]
pub struct BackendRequest {
    pub model: String,
    pub max_tokens: Option<u32>,
    pub sampling: SamplingConfig,
    pub kind: BackendRequestKind,
    pub prefill_chunk_admission: Option<BackendPrefillChunkAdmissionHook>,
}

impl BackendRequest {
    pub fn raw_completion(
        model: impl Into<String>,
        prompt: impl Into<String>,
        max_tokens: Option<u32>,
        sampling: SamplingConfig,
    ) -> Self {
        Self::raw_completion_with_cache_context(
            model,
            prompt,
            max_tokens,
            sampling,
            BackendCacheContext::raw_prompt(),
        )
    }

    pub fn raw_completion_with_cache_context(
        model: impl Into<String>,
        prompt: impl Into<String>,
        max_tokens: Option<u32>,
        sampling: SamplingConfig,
        cache_context: BackendCacheContext,
    ) -> Self {
        Self {
            model: model.into(),
            max_tokens,
            sampling,
            kind: BackendRequestKind::RawCompletion(BackendCompletionRequest {
                prompt: prompt.into(),
                cache_context,
            }),
            prefill_chunk_admission: None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn chat_completion(
        model: impl Into<String>,
        prompt: impl Into<String>,
        chat_context: BackendChatContext,
        max_tokens: Option<u32>,
        sampling: SamplingConfig,
        required_tool_choice: Option<BackendToolChoice>,
        json_object_mode: bool,
        cache_context: BackendCacheContext,
    ) -> Self {
        Self {
            model: model.into(),
            max_tokens,
            sampling,
            kind: BackendRequestKind::Chat(BackendChatRequest {
                prompt: prompt.into(),
                chat_context,
                required_tool_choice,
                json_object_mode,
                cache_context,
            }),
            prefill_chunk_admission: None,
        }
    }

    pub fn prompt(&self) -> &str {
        match &self.kind {
            BackendRequestKind::RawCompletion(request) => &request.prompt,
            BackendRequestKind::Chat(request) => &request.prompt,
        }
    }

    pub fn cache_context(&self) -> &BackendCacheContext {
        match &self.kind {
            BackendRequestKind::RawCompletion(request) => &request.cache_context,
            BackendRequestKind::Chat(request) => &request.cache_context,
        }
    }

    pub fn cache_context_mut(&mut self) -> &mut BackendCacheContext {
        match &mut self.kind {
            BackendRequestKind::RawCompletion(request) => &mut request.cache_context,
            BackendRequestKind::Chat(request) => &mut request.cache_context,
        }
    }

    pub fn as_raw_completion(&self) -> Option<&BackendCompletionRequest> {
        match &self.kind {
            BackendRequestKind::RawCompletion(request) => Some(request),
            BackendRequestKind::Chat(_) => None,
        }
    }

    pub fn as_chat(&self) -> Option<&BackendChatRequest> {
        match &self.kind {
            BackendRequestKind::RawCompletion(_) => None,
            BackendRequestKind::Chat(request) => Some(request),
        }
    }

    pub fn with_prefill_chunk_admission(
        mut self,
        admission: BackendPrefillChunkAdmissionHook,
    ) -> Self {
        self.prefill_chunk_admission = Some(admission);
        self
    }

    pub fn prefill_chunk_admission(&self) -> Option<&BackendPrefillChunkAdmissionHook> {
        self.prefill_chunk_admission.as_ref()
    }
}

#[async_trait]
pub trait BackendPrefillChunkAdmission: fmt::Debug + Send + Sync + 'static {
    async fn wait_for_next_chunk(
        &self,
        progress: BackendStreamProgress,
    ) -> Result<(), BackendError>;
}

#[derive(Clone)]
pub struct BackendPrefillChunkAdmissionHook {
    inner: Arc<dyn BackendPrefillChunkAdmission>,
}

impl BackendPrefillChunkAdmissionHook {
    pub fn new<T>(admission: Arc<T>) -> Self
    where
        T: BackendPrefillChunkAdmission,
    {
        let inner: Arc<dyn BackendPrefillChunkAdmission> = admission;
        Self { inner }
    }

    pub async fn wait_for_next_chunk(
        &self,
        progress: BackendStreamProgress,
    ) -> Result<(), BackendError> {
        self.inner.wait_for_next_chunk(progress).await
    }
}

impl fmt::Debug for BackendPrefillChunkAdmissionHook {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BackendPrefillChunkAdmissionHook")
            .finish_non_exhaustive()
    }
}

impl PartialEq for BackendPrefillChunkAdmissionHook {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum BackendRequestKind {
    RawCompletion(BackendCompletionRequest),
    Chat(BackendChatRequest),
}

#[derive(Debug, Clone, PartialEq)]
pub struct BackendCompletionRequest {
    pub prompt: String,
    pub cache_context: BackendCacheContext,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BackendChatRequest {
    pub prompt: String,
    pub chat_context: BackendChatContext,
    pub required_tool_choice: Option<BackendToolChoice>,
    pub json_object_mode: bool,
    pub cache_context: BackendCacheContext,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BackendChatContext {
    pub messages: Vec<BackendChatMessage>,
    pub tools: Vec<BackendToolDefinition>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendChatMessage {
    pub role: BackendChatRole,
    pub content: Option<String>,
    pub name: Option<String>,
    pub tool_call_id: Option<String>,
    pub tool_calls: Vec<BackendToolCall>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendChatRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendToolCall {
    pub id: String,
    pub call_type: BackendToolCallType,
    pub function: BackendToolCallFunction,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendToolCallType {
    Function,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendToolCallFunction {
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendToolType {
    Function,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: BackendToolType,
    pub function: BackendToolFunctionDefinition,
}

impl BackendToolDefinition {
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            tool_type: BackendToolType::Function,
            function: BackendToolFunctionDefinition {
                name: name.into(),
                description: Some(description.into()),
                parameters,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendToolFunctionDefinition {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default = "empty_backend_tool_parameters")]
    pub parameters: serde_json::Value,
}

fn empty_backend_tool_parameters() -> serde_json::Value {
    serde_json::json!({})
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BackendCacheContext {
    pub key: BackendCacheKey,
    pub cache_template_id: String,
    pub tool_schema: Option<String>,
    pub chat_template_kwargs: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BackendCacheKey {
    value: String,
}

impl BackendCacheKey {
    pub fn as_str(&self) -> &str {
        &self.value
    }

    fn from_identity(
        prompt_template: &str,
        tool_schema: Option<&str>,
        chat_template_kwargs: Option<&str>,
    ) -> Self {
        let mut hasher = Sha256::new();
        update_cache_key_component(
            &mut hasher,
            "cache-context-version",
            Some("backend-cache-context/v1"),
        );
        update_cache_key_component(&mut hasher, "prompt-template", Some(prompt_template));
        update_cache_key_component(&mut hasher, "tool-schema", tool_schema);
        update_cache_key_component(&mut hasher, "chat-template-kwargs", chat_template_kwargs);
        Self {
            value: format!("sha256:{:x}", hasher.finalize()),
        }
    }
}

impl BackendCacheContext {
    pub fn raw_prompt() -> Self {
        let prompt_template = "raw-prompt/v1";
        Self {
            key: BackendCacheKey::from_identity(prompt_template, None, None),
            cache_template_id: prompt_template.to_owned(),
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
        let template_id = template_id.into();
        let key = BackendCacheKey::from_identity(
            &template_id,
            tool_schema.as_deref(),
            chat_template_kwargs.as_deref(),
        );
        Self {
            key,
            cache_template_id: template_id,
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

fn update_cache_key_component(hasher: &mut Sha256, name: &str, value: Option<&str>) {
    hasher.update((name.len() as u64).to_le_bytes());
    hasher.update(name.as_bytes());
    match value {
        Some(value) => {
            hasher.update([1]);
            hasher.update((value.len() as u64).to_le_bytes());
            hasher.update(value.as_bytes());
        }
        None => hasher.update([0]),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendToolChoice {
    RequiredAny,
    RequiredFunction(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendFinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    Error,
}

/// Runtime-visible feature support advertised by a backend.
///
/// The default set preserves the legacy backend contract: existing backends
/// are treated as supporting every request shape the runtime could already
/// send. Backends with narrower support should override
/// [`ModelBackend::capabilities`] so callers can reject incompatible requests
/// before generation begins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendCapabilities {
    pub raw_completions: bool,
    pub chat_completions: bool,
    pub streaming: bool,
    pub tool_calls: bool,
    pub json_object_mode: bool,
    pub sampling_greedy: bool,
    pub sampling_top_p: bool,
}

impl BackendCapabilities {
    pub const fn all() -> Self {
        Self {
            raw_completions: true,
            chat_completions: true,
            streaming: true,
            tool_calls: true,
            json_object_mode: true,
            sampling_greedy: true,
            sampling_top_p: true,
        }
    }
}

impl Default for BackendCapabilities {
    fn default() -> Self {
        Self::all()
    }
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
            temperature: llm_util::sampling::DEFAULT_TEMPERATURE,
            top_p: llm_util::sampling::DEFAULT_TOP_P,
        }
    }

    pub fn from_openai_controls(
        temperature: Option<f32>,
        top_p: Option<f32>,
    ) -> Result<Self, BackendError> {
        llm_util::sampling::validate_sampling_controls(temperature, top_p)
            .map_err(|err| BackendError::invalid_sampling_config(err.to_string()))?;
        Ok(match (temperature, top_p) {
            (Some(temperature), _) if temperature == llm_util::sampling::GREEDY_TEMPERATURE => {
                Self::Greedy
            }
            (None, None) => Self::standard(),
            (t, p) => Self::TopP {
                temperature: t.unwrap_or(llm_util::sampling::DEFAULT_TEMPERATURE),
                top_p: p.unwrap_or(llm_util::sampling::DEFAULT_TOP_P),
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
    pub finish_reason: BackendFinishReason,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BackendStreamChunk {
    pub text: String,
    pub tool_call_deltas: Vec<BackendToolCallDelta>,
    pub prompt_tokens: u64,
    pub prompt_cached_tokens: Option<u64>,
    pub completion_tokens: u64,
    pub finish_reason: Option<BackendFinishReason>,
    pub progress: Option<BackendStreamProgress>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BackendStreamProgress {
    PrefillProgress {
        chunk: u64,
        total: u64,
        tokens: u64,
        total_tokens: u64,
    },
    MlxStreamTiming {
        milestone: BackendStreamTimingMilestone,
        latency_ms: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendStreamTimingMilestone {
    ResponseHeaders,
    FirstUpstreamByte,
    FirstParsedChunk,
    FirstToolDelta,
    UpstreamComplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendToolCallDelta {
    pub index: u32,
    pub id: Option<String>,
    pub call_type: Option<BackendToolCallType>,
    pub function: Option<BackendToolCallFunctionDelta>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendToolCallFunctionDelta {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendModelMetadata {
    pub id: String,
    pub backend: String,
    pub family: Option<String>,
    pub quantization: Option<String>,
    pub repo_id: Option<String>,
    pub resolved_commit: Option<String>,
    pub profile: Option<String>,
}

impl BackendModelMetadata {
    pub fn new(id: impl Into<String>, backend: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            backend: backend.into(),
            family: None,
            quantization: None,
            repo_id: None,
            resolved_commit: None,
            profile: None,
        }
    }

    pub fn with_family(mut self, family: impl Into<String>) -> Self {
        self.family = Some(family.into());
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendHealthStatus {
    Ready,
    Unavailable,
}

impl BackendHealthStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Unavailable => "unavailable",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendHealth {
    status: BackendHealthStatus,
    reason: Option<String>,
}

impl BackendHealth {
    pub fn ready() -> Self {
        Self {
            status: BackendHealthStatus::Ready,
            reason: None,
        }
    }

    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self {
            status: BackendHealthStatus::Unavailable,
            reason: Some(reason.into()),
        }
    }

    pub fn is_ready(&self) -> bool {
        self.status == BackendHealthStatus::Ready
    }

    pub fn status(&self) -> BackendHealthStatus {
        self.status
    }

    pub fn reason(&self) -> Option<&str> {
        self.reason.as_deref()
    }
}

#[async_trait]
pub trait ModelBackend: Send + Sync + 'static {
    fn model_id(&self) -> &str;

    fn model_metadata(&self) -> BackendModelMetadata {
        BackendModelMetadata::new(self.model_id(), "unknown")
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::all()
    }

    async fn health(&self) -> BackendHealth {
        BackendHealth::ready()
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
                    progress: None,
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
                    progress: None,
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

    fn capabilities(&self) -> BackendCapabilities {
        (**self).capabilities()
    }

    async fn health(&self) -> BackendHealth {
        (**self).health().await
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

const BACKEND_EXECUTION_FAILED_CODE: &str = "backend_execution_failed";
const TOKENIZER_FAILED_CODE: &str = "tokenizer_failed";
const SAMPLER_FAILED_CODE: &str = "sampler_failed";
const METAL_BACKEND_FAILED_CODE: &str = "metal_backend_failed";
const BACKEND_CONFIG_FAILED_CODE: &str = "backend_config_failed";
const BACKEND_INVARIANT_FAILED_CODE: &str = "backend_invariant_failed";
const SCHEDULER_OVERLOADED_CODE: &str = "model_overloaded";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendFailureClass {
    BackendExecution,
    Scheduler,
    TensorLoad,
    Tokenizer,
    Sampler,
    Metal,
    Config,
    InternalInvariant,
}

impl BackendFailureClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BackendExecution => "backend_execution",
            Self::Scheduler => "scheduler",
            Self::TensorLoad => "tensor_load",
            Self::Tokenizer => "tokenizer",
            Self::Sampler => "sampler",
            Self::Metal => "metal",
            Self::Config => "config",
            Self::InternalInvariant => "internal_invariant",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error(transparent)]
pub struct BackendError {
    kind: BackendErrorKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub(crate) enum BackendErrorKind {
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
    #[error("backend error: {message}")]
    BackendFailure { code: &'static str, message: String },
    #[error("backend scheduler overloaded: {0}")]
    SchedulerOverloaded(String),
    #[error("backend tensor load failed: {message}")]
    TensorLoad { code: &'static str, message: String },
    #[error("backend tokenizer failed: {0}")]
    Tokenizer(String),
    #[error("backend sampler failed: {0}")]
    Sampler(String),
    #[error("backend Metal failed: {0}")]
    Metal(String),
    #[error("backend config failed: {0}")]
    Config(String),
    #[error("backend invariant failed: {0}")]
    InternalInvariant(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendErrorDomain {
    ModelNotFound {
        requested: String,
        available: String,
    },
    InvalidRequest {
        reason: String,
    },
    Cancelled,
    BackendFailure(BackendError),
}

impl BackendError {
    pub fn model_not_found(requested: impl Into<String>, available: impl Into<String>) -> Self {
        Self {
            kind: BackendErrorKind::ModelNotFound {
                requested: requested.into(),
                available: available.into(),
            },
        }
    }

    pub fn unsupported_request(message: impl Into<String>) -> Self {
        Self {
            kind: BackendErrorKind::UnsupportedRequest(message.into()),
        }
    }

    pub fn invalid_sampling_config(message: impl Into<String>) -> Self {
        Self {
            kind: BackendErrorKind::InvalidSamplingConfig(message.into()),
        }
    }

    pub fn cancelled() -> Self {
        Self {
            kind: BackendErrorKind::Cancelled,
        }
    }

    pub fn other(message: impl Into<String>) -> Self {
        Self::backend_failure(BACKEND_EXECUTION_FAILED_CODE, message)
    }

    pub fn backend_failure(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            kind: BackendErrorKind::BackendFailure {
                code,
                message: message.into(),
            },
        }
    }

    pub fn scheduler_overloaded(message: impl Into<String>) -> Self {
        Self {
            kind: BackendErrorKind::SchedulerOverloaded(message.into()),
        }
    }

    pub fn tensor_load(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            kind: BackendErrorKind::TensorLoad {
                code,
                message: message.into(),
            },
        }
    }

    pub fn tokenizer(message: impl Into<String>) -> Self {
        Self {
            kind: BackendErrorKind::Tokenizer(message.into()),
        }
    }

    pub fn sampler(message: impl Into<String>) -> Self {
        Self {
            kind: BackendErrorKind::Sampler(message.into()),
        }
    }

    pub fn metal(message: impl Into<String>) -> Self {
        Self {
            kind: BackendErrorKind::Metal(message.into()),
        }
    }

    pub fn config(message: impl Into<String>) -> Self {
        Self {
            kind: BackendErrorKind::Config(message.into()),
        }
    }

    pub fn internal_invariant(message: impl Into<String>) -> Self {
        Self {
            kind: BackendErrorKind::InternalInvariant(message.into()),
        }
    }

    pub fn is_model_not_found(&self) -> bool {
        matches!(self.kind, BackendErrorKind::ModelNotFound { .. })
    }

    pub fn is_unsupported_request(&self) -> bool {
        matches!(self.kind, BackendErrorKind::UnsupportedRequest(_))
    }

    pub fn is_invalid_sampling_config(&self) -> bool {
        matches!(self.kind, BackendErrorKind::InvalidSamplingConfig(_))
    }

    pub fn is_cancelled(&self) -> bool {
        matches!(self.kind, BackendErrorKind::Cancelled)
    }

    pub fn other_message(&self) -> Option<&str> {
        match &self.kind {
            BackendErrorKind::BackendFailure { message, .. } => Some(message.as_str()),
            BackendErrorKind::SchedulerOverloaded(message) => Some(message.as_str()),
            BackendErrorKind::TensorLoad { message, .. } => Some(message.as_str()),
            BackendErrorKind::Tokenizer(message)
            | BackendErrorKind::Sampler(message)
            | BackendErrorKind::Metal(message)
            | BackendErrorKind::Config(message)
            | BackendErrorKind::InternalInvariant(message) => Some(message.as_str()),
            _ => None,
        }
    }

    pub fn backend_failure_code(&self) -> Option<&'static str> {
        match &self.kind {
            BackendErrorKind::BackendFailure { code, .. } => Some(*code),
            BackendErrorKind::SchedulerOverloaded(_) => Some(SCHEDULER_OVERLOADED_CODE),
            BackendErrorKind::TensorLoad { code, .. } => Some(*code),
            BackendErrorKind::Tokenizer(_) => Some(TOKENIZER_FAILED_CODE),
            BackendErrorKind::Sampler(_) => Some(SAMPLER_FAILED_CODE),
            BackendErrorKind::Metal(_) => Some(METAL_BACKEND_FAILED_CODE),
            BackendErrorKind::Config(_) => Some(BACKEND_CONFIG_FAILED_CODE),
            BackendErrorKind::InternalInvariant(_) => Some(BACKEND_INVARIANT_FAILED_CODE),
            _ => None,
        }
    }

    pub fn backend_failure_class(&self) -> Option<BackendFailureClass> {
        match &self.kind {
            BackendErrorKind::BackendFailure { .. } => Some(BackendFailureClass::BackendExecution),
            BackendErrorKind::SchedulerOverloaded(_) => Some(BackendFailureClass::Scheduler),
            BackendErrorKind::TensorLoad { .. } => Some(BackendFailureClass::TensorLoad),
            BackendErrorKind::Tokenizer(_) => Some(BackendFailureClass::Tokenizer),
            BackendErrorKind::Sampler(_) => Some(BackendFailureClass::Sampler),
            BackendErrorKind::Metal(_) => Some(BackendFailureClass::Metal),
            BackendErrorKind::Config(_) => Some(BackendFailureClass::Config),
            BackendErrorKind::InternalInvariant(_) => Some(BackendFailureClass::InternalInvariant),
            _ => None,
        }
    }

    pub fn into_domain(self) -> BackendErrorDomain {
        match self.kind {
            BackendErrorKind::ModelNotFound {
                requested,
                available,
            } => BackendErrorDomain::ModelNotFound {
                requested,
                available,
            },
            BackendErrorKind::UnsupportedRequest(reason)
            | BackendErrorKind::InvalidSamplingConfig(reason) => {
                BackendErrorDomain::InvalidRequest { reason }
            }
            BackendErrorKind::Cancelled => BackendErrorDomain::Cancelled,
            kind @ (BackendErrorKind::BackendFailure { .. }
            | BackendErrorKind::SchedulerOverloaded(_)
            | BackendErrorKind::TensorLoad { .. }
            | BackendErrorKind::Tokenizer(_)
            | BackendErrorKind::Sampler(_)
            | BackendErrorKind::Metal(_)
            | BackendErrorKind::Config(_)
            | BackendErrorKind::InternalInvariant(_)) => {
                BackendErrorDomain::BackendFailure(BackendError { kind })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use futures::{StreamExt, executor::block_on};
    use tokio_util::sync::CancellationToken;

    struct CancelAwareBackend;
    struct StreamingDisabledBackend;

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
                finish_reason: BackendFinishReason::Stop,
            })
        }

        async fn generate_with_cancel(
            &self,
            request: BackendRequest,
            cancellation: CancellationToken,
        ) -> Result<BackendOutput, BackendError> {
            if cancellation.is_cancelled() {
                return Err(BackendError::cancelled());
            }
            self.generate(request).await
        }
    }

    #[async_trait]
    impl ModelBackend for StreamingDisabledBackend {
        fn model_id(&self) -> &str {
            "local-qwen36"
        }

        fn capabilities(&self) -> BackendCapabilities {
            let mut capabilities = BackendCapabilities::all();
            capabilities.streaming = false;
            capabilities
        }

        async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
            Ok(BackendOutput {
                text: "ok".to_owned(),
                prompt_tokens: 1,
                prompt_cached_tokens: None,
                completion_tokens: 1,
                finish_reason: BackendFinishReason::Stop,
            })
        }

        async fn generate_with_cancel(
            &self,
            request: BackendRequest,
            cancellation: CancellationToken,
        ) -> Result<BackendOutput, BackendError> {
            if cancellation.is_cancelled() {
                return Err(BackendError::cancelled());
            }
            self.generate(request).await
        }
    }

    #[test]
    fn model_backend_default_capabilities_preserve_legacy_request_support() {
        let backend = CancelAwareBackend;

        assert_eq!(backend.capabilities(), BackendCapabilities::all());
        assert_eq!(BackendCapabilities::default(), BackendCapabilities::all());
    }

    #[test]
    fn boxed_model_backend_forwards_capabilities() {
        let backend: Box<dyn ModelBackend> = Box::new(StreamingDisabledBackend);
        let mut expected = BackendCapabilities::all();
        expected.streaming = false;

        assert_eq!(backend.capabilities(), expected);
    }

    #[test]
    fn typed_backend_failures_expose_stable_codes_and_classes() {
        let cases = [
            (
                BackendError::tensor_load("model_integrity_failed", "bad tensor header"),
                BackendFailureClass::TensorLoad,
                "model_integrity_failed",
                "bad tensor header",
            ),
            (
                BackendError::scheduler_overloaded("model scheduler queue is full"),
                BackendFailureClass::Scheduler,
                "model_overloaded",
                "model scheduler queue is full",
            ),
            (
                BackendError::tokenizer("tokenizer decode failed"),
                BackendFailureClass::Tokenizer,
                "tokenizer_failed",
                "tokenizer decode failed",
            ),
            (
                BackendError::sampler("empty logits"),
                BackendFailureClass::Sampler,
                "sampler_failed",
                "empty logits",
            ),
            (
                BackendError::metal("command buffer failed"),
                BackendFailureClass::Metal,
                "metal_backend_failed",
                "command buffer failed",
            ),
            (
                BackendError::config("invalid model config"),
                BackendFailureClass::Config,
                "backend_config_failed",
                "invalid model config",
            ),
            (
                BackendError::internal_invariant("prefill returned no hidden states"),
                BackendFailureClass::InternalInvariant,
                "backend_invariant_failed",
                "prefill returned no hidden states",
            ),
        ];

        for (err, class, code, message) in cases {
            assert_eq!(err.backend_failure_class(), Some(class));
            assert_eq!(err.backend_failure_code(), Some(code));
            assert_eq!(err.other_message(), Some(message));
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

        let err = result.expect_err("pre-cancelled generation should fail");
        assert!(err.is_cancelled(), "expected Cancelled, got {err:?}");
    }

    fn backend_request(prompt: &str) -> BackendRequest {
        BackendRequest::raw_completion("local-qwen36", prompt, Some(1), SamplingConfig::Greedy)
    }

    #[test]
    fn backend_request_kind_separates_raw_completion_and_chat_fields() {
        let raw = BackendRequest::raw_completion(
            "local-qwen36",
            "hello",
            Some(1),
            SamplingConfig::Greedy,
        );
        assert!(matches!(raw.kind, BackendRequestKind::RawCompletion(_)));
        assert!(raw.as_chat().is_none());
        assert_eq!(raw.prompt(), "hello");

        let chat_context = BackendChatContext {
            messages: vec![BackendChatMessage {
                role: BackendChatRole::User,
                content: Some("hello".to_owned()),
                name: None,
                tool_call_id: None,
                tool_calls: Vec::new(),
            }],
            tools: Vec::new(),
        };
        let chat = BackendRequest::chat_completion(
            "local-qwen36",
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n",
            chat_context,
            Some(1),
            SamplingConfig::Greedy,
            Some(BackendToolChoice::RequiredAny),
            true,
            BackendCacheContext::chat_template("chatml/qwen/v1", None),
        );
        let chat_request = chat.as_chat().expect("chat request kind");
        assert_eq!(chat_request.prompt, chat.prompt());
        assert_eq!(
            chat_request.required_tool_choice.as_ref(),
            Some(&BackendToolChoice::RequiredAny)
        );
        assert!(chat_request.json_object_mode);
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
            err.is_invalid_sampling_config(),
            "expected InvalidSamplingConfig, got {err:?}"
        );
    }

    #[test]
    fn from_openai_controls_rejects_nan_temperature() {
        let err = SamplingConfig::from_openai_controls(Some(f32::NAN), None)
            .expect_err("NaN temperature should be rejected");
        assert!(
            err.is_invalid_sampling_config(),
            "expected InvalidSamplingConfig, got {err:?}"
        );
    }

    #[test]
    fn from_openai_controls_rejects_inf_temperature() {
        let err = SamplingConfig::from_openai_controls(Some(f32::INFINITY), None)
            .expect_err("inf temperature should be rejected");
        assert!(
            err.is_invalid_sampling_config(),
            "expected InvalidSamplingConfig, got {err:?}"
        );
    }

    #[test]
    fn from_openai_controls_rejects_temperature_above_2() {
        let err = SamplingConfig::from_openai_controls(Some(2.1), None)
            .expect_err("temperature > 2.0 should be rejected");
        assert!(
            err.is_invalid_sampling_config(),
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
            err.is_invalid_sampling_config(),
            "expected InvalidSamplingConfig, got {err:?}"
        );
    }

    #[test]
    fn from_openai_controls_rejects_neg_inf_temperature() {
        let err = SamplingConfig::from_openai_controls(Some(f32::NEG_INFINITY), None)
            .expect_err("neg_inf temperature should be rejected");
        assert!(
            err.is_invalid_sampling_config(),
            "expected InvalidSamplingConfig, got {err:?}"
        );
    }

    #[test]
    fn from_openai_controls_rejects_top_p_above_1() {
        let err = SamplingConfig::from_openai_controls(None, Some(1.5))
            .expect_err("top_p > 1.0 should be rejected");
        assert!(
            err.is_invalid_sampling_config(),
            "expected InvalidSamplingConfig, got {err:?}"
        );
    }

    #[test]
    fn from_openai_controls_rejects_inf_top_p() {
        let err = SamplingConfig::from_openai_controls(None, Some(f32::INFINITY))
            .expect_err("inf top_p should be rejected");
        assert!(
            err.is_invalid_sampling_config(),
            "expected InvalidSamplingConfig, got {err:?}"
        );
    }

    #[test]
    fn from_openai_controls_rejects_nan_top_p() {
        let err = SamplingConfig::from_openai_controls(None, Some(f32::NAN))
            .expect_err("NaN top_p should be rejected");
        assert!(
            err.is_invalid_sampling_config(),
            "expected InvalidSamplingConfig, got {err:?}"
        );
    }

    #[test]
    fn from_openai_controls_rejects_negative_top_p() {
        let err = SamplingConfig::from_openai_controls(None, Some(-0.1))
            .expect_err("negative top_p should be rejected");
        assert!(
            err.is_invalid_sampling_config(),
            "expected InvalidSamplingConfig, got {err:?}"
        );
    }
}
