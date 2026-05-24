use crate::{
    ChatMessage, ResponseFormat, StreamOptions, ToolChoice, ToolDefinition,
    validation::deserialize_stop_sequences,
};
use serde::{Deserialize, Serialize};

/// Request body for the OpenAI-compatible `POST /v1/chat/completions` endpoint.
///
/// The type preserves the accepted wire shape while `ValidateRequest` enforces
/// local runtime support before a request reaches prompt rendering. Unsupported
/// features such as schema response format, logprobs, and parallel tool calls
/// are represented here so they can fail closed with stable API errors.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model identifier selected by the client.
    pub model: String,
    /// Ordered conversation messages consumed by the chat template renderer.
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    /// Function tools declared for this turn.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    /// Optional tool selection policy from the OpenAI request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    /// OpenAI compatibility flag; `true` is rejected until parallel tool calls are supported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    /// Requested assistant output format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    /// Whether the client requested a streaming response.
    #[serde(default)]
    pub stream: bool,
    /// Streaming options such as terminal usage chunks.
    #[serde(default, skip_serializing_if = "StreamOptions::is_default")]
    pub stream_options: StreamOptions,
    /// Sampling temperature override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus sampling probability override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Presence penalty requested by the client; non-neutral values are rejected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    /// Frequency penalty requested by the client; non-neutral values are rejected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// Chat logprobs flag; `true` is currently unsupported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    /// Number of top logprobs requested per token; currently unsupported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
    /// Legacy OpenAI maximum output token field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Preferred OpenAI chat maximum output token field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    /// Requested number of choices; values other than one are rejected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    /// Stop sequences applied before assistant output is parsed for tools.
    #[serde(
        default,
        deserialize_with = "deserialize_stop_sequences",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub stop: Vec<String>,
}

impl ChatCompletionRequest {
    /// Returns the effective maximum output token limit accepted by the runtime.
    ///
    /// `max_completion_tokens` wins when present, matching the newer chat API
    /// field, while validation requires both fields to agree if a client sends
    /// both.
    pub fn effective_max_tokens(&self) -> Option<u32> {
        self.max_completion_tokens.or(self.max_tokens)
    }
}

/// Request body for the legacy OpenAI-compatible `POST /v1/completions` endpoint.
///
/// The runtime treats this as a raw prompt request without chat-template
/// rendering. Validation still applies shared body limits, sampling controls,
/// stop sequence rules, and fail-closed handling for unsupported logprob
/// fields.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CompletionRequest {
    /// Model identifier selected by the client.
    pub model: String,
    /// Raw prompt text passed to the backend.
    pub prompt: String,
    /// Whether the client requested a streaming response.
    #[serde(default)]
    pub stream: bool,
    /// Streaming options such as terminal usage chunks.
    #[serde(default, skip_serializing_if = "StreamOptions::is_default")]
    pub stream_options: StreamOptions,
    /// Sampling temperature override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus sampling probability override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Presence penalty requested by the client; non-neutral values are rejected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    /// Frequency penalty requested by the client; non-neutral values are rejected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// Legacy completion logprobs request; currently unsupported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<u32>,
    /// Maximum generated token count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Requested number of choices; values other than one are rejected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    /// Stop sequences applied before a completion is returned to the client.
    #[serde(
        default,
        deserialize_with = "deserialize_stop_sequences",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub stop: Vec<String>,
}
