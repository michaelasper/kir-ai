//! Backend contract types shared by the runtime and backend implementations.
//!
//! This crate defines the narrow interface a model backend must implement:
//! accept validated/rendered prompts, advertise capabilities, report health,
//! stream progress, and return structured failures that the runtime can map to
//! API-visible errors.

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

/// Rendered generation request sent from the runtime to a backend.
///
/// API request validation and prompt rendering have already happened when this
/// type is created. Backends should treat `kind` as the semantic request shape
/// and should reject unsupported fields with `BackendError::unsupported_request`
/// rather than silently ignoring them.
#[derive(Debug, Clone, PartialEq)]
pub struct BackendRequest {
    /// Model identifier selected by the original API request.
    pub model: String,
    /// Optional maximum number of generated tokens.
    pub max_tokens: Option<u32>,
    /// Sampling configuration normalized from OpenAI controls.
    pub sampling: SamplingConfig,
    /// Rendered request payload.
    pub kind: BackendRequestKind,
    /// Optional hook used to coordinate streaming prefill admission.
    pub prefill_chunk_admission: Option<BackendPrefillChunkAdmissionHook>,
}

impl BackendRequest {
    /// Builds a raw completion request with a default raw-prompt cache context.
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

    /// Builds a raw completion request with an explicit cache context.
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

    /// Builds a chat completion request after chat-template rendering.
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

    /// Returns the rendered prompt text regardless of request kind.
    pub fn prompt(&self) -> &str {
        match &self.kind {
            BackendRequestKind::RawCompletion(request) => &request.prompt,
            BackendRequestKind::Chat(request) => &request.prompt,
        }
    }

    /// Returns immutable cache context for the rendered prompt.
    pub fn cache_context(&self) -> &BackendCacheContext {
        match &self.kind {
            BackendRequestKind::RawCompletion(request) => &request.cache_context,
            BackendRequestKind::Chat(request) => &request.cache_context,
        }
    }

    /// Returns mutable cache context for request staging and tests.
    pub fn cache_context_mut(&mut self) -> &mut BackendCacheContext {
        match &mut self.kind {
            BackendRequestKind::RawCompletion(request) => &mut request.cache_context,
            BackendRequestKind::Chat(request) => &mut request.cache_context,
        }
    }

    /// Returns the raw completion payload when this is a raw request.
    pub fn as_raw_completion(&self) -> Option<&BackendCompletionRequest> {
        match &self.kind {
            BackendRequestKind::RawCompletion(request) => Some(request),
            BackendRequestKind::Chat(_) => None,
        }
    }

    /// Returns the chat payload when this is a chat request.
    pub fn as_chat(&self) -> Option<&BackendChatRequest> {
        match &self.kind {
            BackendRequestKind::RawCompletion(_) => None,
            BackendRequestKind::Chat(request) => Some(request),
        }
    }

    /// Attaches a prefill admission hook to this request.
    pub fn with_prefill_chunk_admission(
        mut self,
        admission: BackendPrefillChunkAdmissionHook,
    ) -> Self {
        self.prefill_chunk_admission = Some(admission);
        self
    }

    /// Returns the optional prefill admission hook.
    pub fn prefill_chunk_admission(&self) -> Option<&BackendPrefillChunkAdmissionHook> {
        self.prefill_chunk_admission.as_ref()
    }
}

/// Async hook invoked by streaming backends between prefill chunks.
///
/// Implementations may block to enforce scheduler or test admission ordering.
#[async_trait]
pub trait BackendPrefillChunkAdmission: fmt::Debug + Send + Sync + 'static {
    /// Waits until the next prefill chunk may continue.
    async fn wait_for_next_chunk(
        &self,
        progress: BackendStreamProgress,
    ) -> Result<(), BackendError>;
}

/// Cloneable dynamic wrapper for a prefill admission hook.
#[derive(Clone)]
pub struct BackendPrefillChunkAdmissionHook {
    inner: Arc<dyn BackendPrefillChunkAdmission>,
}

impl BackendPrefillChunkAdmissionHook {
    /// Wraps an admission hook for storage on backend requests.
    pub fn new<T>(admission: Arc<T>) -> Self
    where
        T: BackendPrefillChunkAdmission,
    {
        let inner: Arc<dyn BackendPrefillChunkAdmission> = admission;
        Self { inner }
    }

    /// Waits until the next prefill chunk may continue.
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

/// Backend request payload variants.
#[derive(Debug, Clone, PartialEq)]
pub enum BackendRequestKind {
    /// Raw text prompt without chat semantics.
    RawCompletion(BackendCompletionRequest),
    /// Chat prompt rendered from messages and tools.
    Chat(BackendChatRequest),
}

/// Backend payload for a legacy raw completion request.
#[derive(Debug, Clone, PartialEq)]
pub struct BackendCompletionRequest {
    /// Rendered raw prompt text.
    pub prompt: String,
    /// Cache identity for the prompt.
    pub cache_context: BackendCacheContext,
}

/// Backend payload for a chat completion request.
#[derive(Debug, Clone, PartialEq)]
pub struct BackendChatRequest {
    /// Rendered chat prompt text.
    pub prompt: String,
    /// Structured chat context retained for backends that need message/tool metadata.
    pub chat_context: BackendChatContext,
    /// Required tool choice, if the API request forced a tool.
    pub required_tool_choice: Option<BackendToolChoice>,
    /// Whether the runtime requires assistant content to parse as a JSON object.
    pub json_object_mode: bool,
    /// Cache identity for the rendered prompt and tool schema.
    pub cache_context: BackendCacheContext,
}

/// Structured chat metadata passed alongside a rendered prompt.
#[derive(Debug, Clone, PartialEq)]
pub struct BackendChatContext {
    /// Original conversation messages after API validation.
    pub messages: Vec<BackendChatMessage>,
    /// Declared function tools after API validation.
    pub tools: Vec<BackendToolDefinition>,
}

/// Chat message representation available to backend implementations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendChatMessage {
    /// Message role.
    pub role: BackendChatRole,
    /// Optional text content.
    pub content: Option<String>,
    /// Optional participant name.
    pub name: Option<String>,
    /// Tool call identifier answered by a tool result message.
    pub tool_call_id: Option<String>,
    /// Assistant tool calls on this message.
    pub tool_calls: Vec<BackendToolCall>,
}

/// Chat message role in backend context.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendChatRole {
    /// System instruction message.
    System,
    /// User message.
    User,
    /// Assistant message.
    Assistant,
    /// Tool result message.
    Tool,
}

/// Tool call included in backend chat context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendToolCall {
    /// Tool call identifier.
    pub id: String,
    /// Tool call type.
    pub call_type: BackendToolCallType,
    /// Function payload.
    pub function: BackendToolCallFunction,
}

/// Tool call type supported by backends.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendToolCallType {
    /// Function call.
    Function,
}

/// Function call payload in backend chat context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendToolCallFunction {
    /// Function name.
    pub name: String,
    /// Parsed JSON arguments.
    pub arguments: serde_json::Value,
}

/// Tool definition kind supported by backends.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendToolType {
    /// Function tool.
    Function,
}

/// Tool declaration passed to a backend.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendToolDefinition {
    /// Tool kind.
    #[serde(rename = "type")]
    pub tool_type: BackendToolType,
    /// Function schema and metadata.
    pub function: BackendToolFunctionDefinition,
}

impl BackendToolDefinition {
    /// Builds a function tool declaration.
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

/// Backend-facing function tool definition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendToolFunctionDefinition {
    /// Function name.
    pub name: String,
    /// Optional model-facing description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON schema object for function arguments.
    #[serde(default = "empty_backend_tool_parameters")]
    pub parameters: serde_json::Value,
}

fn empty_backend_tool_parameters() -> serde_json::Value {
    serde_json::json!({})
}

/// Prompt cache identity attached to backend requests.
///
/// The key is derived from the template ID, serialized tool schema JSON, and
/// family-specific template kwargs rather than from transient request fields.
/// Tool schema JSON is order-insensitive only when the caller applies canonical
/// normalization; the default runtime serialization preserves caller JSON order.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BackendCacheContext {
    /// Stable hash key for this prompt/cache context.
    pub key: BackendCacheKey,
    /// Prompt template identifier.
    pub cache_template_id: String,
    /// Optional serialized tool schema JSON used in the cache key.
    pub tool_schema: Option<String>,
    /// Optional family-specific chat template kwargs JSON.
    pub chat_template_kwargs: Option<String>,
}

/// Stable cache key for a rendered prompt context.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BackendCacheKey {
    value: String,
}

impl BackendCacheKey {
    /// Returns the key string, currently prefixed with `sha256:`.
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
    /// Cache context for raw prompt completions.
    pub fn raw_prompt() -> Self {
        let prompt_template = "raw-prompt/v1";
        Self {
            key: BackendCacheKey::from_identity(prompt_template, None, None),
            cache_template_id: prompt_template.to_owned(),
            tool_schema: None,
            chat_template_kwargs: None,
        }
    }

    /// Cache context for a chat template and optional serialized tool schema.
    pub fn chat_template(template_id: impl Into<String>, tool_schema: Option<String>) -> Self {
        Self::chat_template_with_kwargs(template_id, tool_schema, None)
    }

    /// Cache context for a chat template, optional tool schema, and template kwargs.
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

/// Required tool-call policy after runtime request validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendToolChoice {
    /// At least one declared tool must be called.
    RequiredAny,
    /// A specific declared function must be called.
    RequiredFunction(String),
}

/// Backend stop reason normalized for runtime response mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendFinishReason {
    /// Generation stopped naturally.
    Stop,
    /// Token limit reached.
    Length,
    /// Model generated tool calls.
    ToolCalls,
    /// Content filter stopped generation.
    ContentFilter,
    /// Backend reported an error finish.
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
    /// Backend can handle raw completion requests.
    pub raw_completions: bool,
    /// Backend can handle chat completion requests.
    pub chat_completions: bool,
    /// Backend can produce streaming chunks.
    pub streaming: bool,
    /// Backend can produce or honor tool-call requests.
    pub tool_calls: bool,
    /// Backend can support JSON-object response mode.
    pub json_object_mode: bool,
    /// Backend can run deterministic greedy decoding.
    pub sampling_greedy: bool,
    /// Backend can run top-p sampling.
    pub sampling_top_p: bool,
}

impl BackendCapabilities {
    /// Returns the fully capable default contract.
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

/// Sampling strategy normalized from API controls.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum SamplingConfig {
    /// Deterministic greedy decoding.
    #[default]
    Greedy,
    /// Temperature/top-p multinomial sampling.
    TopP {
        /// Sampling temperature.
        temperature: f32,
        /// Nucleus sampling probability.
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

    /// Converts OpenAI sampling controls into a backend sampling configuration.
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

    /// Returns true for deterministic greedy decoding.
    pub fn is_greedy(self) -> bool {
        matches!(self, Self::Greedy)
    }

    /// Returns true for the default non-greedy OpenAI sampling controls.
    pub fn is_standard(self) -> bool {
        self == Self::standard()
    }
}

/// Non-streaming backend generation result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendOutput {
    /// Generated text payload.
    pub text: String,
    /// Prompt tokens consumed.
    pub prompt_tokens: u64,
    /// Prompt tokens served from cache, if known.
    pub prompt_cached_tokens: Option<u64>,
    /// Completion tokens generated.
    pub completion_tokens: u64,
    /// Backend stop reason.
    pub finish_reason: BackendFinishReason,
}

/// Streaming backend generation chunk.
#[derive(Debug, Clone, PartialEq)]
pub struct BackendStreamChunk {
    /// Text delta from the backend.
    pub text: String,
    /// Structured tool-call deltas from backends that support native tool streaming.
    pub tool_call_deltas: Vec<BackendToolCallDelta>,
    /// Highest prompt token count observed so far.
    pub prompt_tokens: u64,
    /// Highest cached prompt token count observed so far, if known.
    pub prompt_cached_tokens: Option<u64>,
    /// Completion tokens represented by this chunk.
    pub completion_tokens: u64,
    /// Optional terminal finish reason.
    pub finish_reason: Option<BackendFinishReason>,
    /// Optional progress or timing metadata.
    pub progress: Option<BackendStreamProgress>,
}

/// Backend progress event carried alongside streaming chunks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BackendStreamProgress {
    /// Prefill progress for prompt ingestion.
    PrefillProgress {
        /// Current prefill chunk index.
        chunk: u64,
        /// Total prefill chunks.
        total: u64,
        /// Tokens processed in the current progress update.
        tokens: u64,
        /// Total tokens expected for prefill.
        total_tokens: u64,
    },
    /// Timing milestone emitted by upstream MLX streaming.
    MlxStreamTiming {
        /// Milestone reached.
        milestone: BackendStreamTimingMilestone,
        /// Latency from request start in milliseconds.
        latency_ms: u64,
    },
}

/// Timing milestone names emitted by streaming backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendStreamTimingMilestone {
    /// Response headers have been produced.
    ResponseHeaders,
    /// First upstream byte was received.
    FirstUpstreamByte,
    /// First parseable chunk was received.
    FirstParsedChunk,
    /// First tool delta was received.
    FirstToolDelta,
    /// Upstream stream completed.
    UpstreamComplete,
}

/// Structured tool-call delta emitted directly by a backend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendToolCallDelta {
    /// Tool call index.
    pub index: u32,
    /// Optional tool call identifier.
    pub id: Option<String>,
    /// Optional tool call type.
    pub call_type: Option<BackendToolCallType>,
    /// Optional function delta.
    pub function: Option<BackendToolCallFunctionDelta>,
}

/// Structured function delta emitted directly by a backend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendToolCallFunctionDelta {
    /// Function name or name fragment.
    pub name: Option<String>,
    /// JSON argument string or fragment.
    pub arguments: Option<String>,
}

/// Metadata describing the loaded backend model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendModelMetadata {
    /// Model identifier served by the backend.
    pub id: String,
    /// Backend implementation name.
    pub backend: String,
    /// Optional model family slug.
    pub family: Option<String>,
    /// Optional quantization label.
    pub quantization: Option<String>,
    /// Optional source repository ID.
    pub repo_id: Option<String>,
    /// Optional resolved source commit.
    pub resolved_commit: Option<String>,
    /// Optional model profile name.
    pub profile: Option<String>,
}

impl BackendModelMetadata {
    /// Creates metadata with an ID and backend name.
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

    /// Adds a model family slug to metadata.
    pub fn with_family(mut self, family: impl Into<String>) -> Self {
        self.family = Some(family.into());
        self
    }
}

/// Readiness state reported by a backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendHealthStatus {
    /// Backend is ready to accept generation requests.
    Ready,
    /// Backend is not currently able to accept generation requests.
    Unavailable,
}

impl BackendHealthStatus {
    /// Stable lowercase status string.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Unavailable => "unavailable",
        }
    }
}

/// Backend readiness response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendHealth {
    status: BackendHealthStatus,
    reason: Option<String>,
}

impl BackendHealth {
    /// Builds a ready health response.
    pub fn ready() -> Self {
        Self {
            status: BackendHealthStatus::Ready,
            reason: None,
        }
    }

    /// Builds an unavailable health response with a reason.
    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self {
            status: BackendHealthStatus::Unavailable,
            reason: Some(reason.into()),
        }
    }

    /// Returns true when the backend is ready.
    pub fn is_ready(&self) -> bool {
        self.status == BackendHealthStatus::Ready
    }

    /// Returns the backend health status.
    pub fn status(&self) -> BackendHealthStatus {
        self.status
    }

    /// Returns the unavailability reason, if any.
    pub fn reason(&self) -> Option<&str> {
        self.reason.as_deref()
    }
}

/// Trait implemented by local model backends.
///
/// The runtime calls this trait only after API validation and prompt rendering.
/// Implementations must preserve cancellation semantics, return structured
/// `BackendError` values for unsupported or invalid requests, and advertise
/// narrower capabilities through `capabilities` so the runtime can reject
/// incompatible OpenAI requests before generation begins.
#[async_trait]
pub trait ModelBackend: Send + Sync + 'static {
    /// Stable model identifier served by this backend.
    fn model_id(&self) -> &str;

    /// Metadata used by model listing, prompt adapter selection, and diagnostics.
    fn model_metadata(&self) -> BackendModelMetadata {
        BackendModelMetadata::new(self.model_id(), "unknown")
    }

    /// Runtime-visible capabilities supported by this backend.
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::all()
    }

    /// Current backend health.
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

    /// Streaming generation entry point for direct backend callers.
    ///
    /// The default adapter wraps `generate` into a single terminal chunk.
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

    /// Cancellable streaming generation entry point used by the production runtime.
    ///
    /// The default adapter wraps `generate_with_cancel` into a single terminal chunk.
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

/// Coarse failure class for backend errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendFailureClass {
    /// Generic backend execution failure.
    BackendExecution,
    /// Scheduler/admission failure.
    Scheduler,
    /// Tensor loading or tensor metadata failure.
    TensorLoad,
    /// Tokenizer failure.
    Tokenizer,
    /// Sampler failure.
    Sampler,
    /// Metal execution failure.
    Metal,
    /// Backend configuration failure.
    Config,
    /// Internal invariant violation.
    InternalInvariant,
}

impl BackendFailureClass {
    /// Stable class string for logs and error metadata.
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

/// Structured backend error.
///
/// Runtime mapping preserves model-not-found, invalid-request, cancellation, and
/// backend-failure domains so API handlers can return stable phase/error
/// metadata without parsing display strings.
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

/// Domain projection used by the runtime when mapping backend errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendErrorDomain {
    /// Requested model is not the backend's loaded model.
    ModelNotFound {
        /// Requested model ID.
        requested: String,
        /// Available model ID.
        available: String,
    },
    /// Request was invalid for backend capabilities or sampling rules.
    InvalidRequest {
        /// Stable human-readable reason.
        reason: String,
    },
    /// Backend observed cancellation.
    Cancelled,
    /// Backend failed during execution.
    BackendFailure(BackendError),
}

impl BackendError {
    /// Builds a model-not-found error.
    pub fn model_not_found(requested: impl Into<String>, available: impl Into<String>) -> Self {
        Self {
            kind: BackendErrorKind::ModelNotFound {
                requested: requested.into(),
                available: available.into(),
            },
        }
    }

    /// Builds an unsupported backend request error.
    pub fn unsupported_request(message: impl Into<String>) -> Self {
        Self {
            kind: BackendErrorKind::UnsupportedRequest(message.into()),
        }
    }

    /// Builds an invalid sampling configuration error.
    pub fn invalid_sampling_config(message: impl Into<String>) -> Self {
        Self {
            kind: BackendErrorKind::InvalidSamplingConfig(message.into()),
        }
    }

    /// Builds a cancellation error.
    pub fn cancelled() -> Self {
        Self {
            kind: BackendErrorKind::Cancelled,
        }
    }

    /// Builds a generic backend execution failure.
    pub fn other(message: impl Into<String>) -> Self {
        Self::backend_failure(BACKEND_EXECUTION_FAILED_CODE, message)
    }

    /// Builds a backend execution failure with a stable code.
    pub fn backend_failure(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            kind: BackendErrorKind::BackendFailure {
                code,
                message: message.into(),
            },
        }
    }

    /// Builds a scheduler overload error.
    pub fn scheduler_overloaded(message: impl Into<String>) -> Self {
        Self {
            kind: BackendErrorKind::SchedulerOverloaded(message.into()),
        }
    }

    /// Builds a tensor load failure.
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
