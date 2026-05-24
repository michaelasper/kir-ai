use crate::{ChatMessage, ChatRole, FinishReason, ToolCallType, Usage};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Non-streaming response for the legacy completions endpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionResponse {
    /// Stable response identifier.
    pub id: String,
    /// OpenAI object type, normally `text_completion`.
    pub object: String,
    /// Unix timestamp when the response was created.
    pub created: i64,
    /// Model identifier that handled the request.
    pub model: String,
    /// Completion choices; local inference currently emits one choice.
    pub choices: Vec<CompletionChoice>,
    /// Final token usage for the request.
    pub usage: Usage,
}

/// Streaming chunk for the legacy completions endpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionStreamResponse {
    /// Stable response identifier shared by every chunk in the stream.
    #[serde(with = "arc_str_serde")]
    pub id: Arc<str>,
    /// OpenAI object type, normally `text_completion`.
    pub object: String,
    /// Unix timestamp when the stream was created.
    pub created: i64,
    /// Model identifier that handled the request.
    #[serde(with = "arc_str_serde")]
    pub model: Arc<str>,
    /// Text deltas or an empty list when the chunk carries only usage.
    pub choices: Vec<CompletionChoice>,
    /// Terminal usage payload when `stream_options.include_usage` is enabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// One text completion choice.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionChoice {
    /// Full text for non-streaming responses or text delta for streaming chunks.
    pub text: String,
    /// Choice index in the OpenAI response shape.
    pub index: u32,
    /// Final stop reason; omitted on non-terminal streaming deltas.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
}

/// Non-streaming response for the chat completions endpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    /// Stable response identifier.
    pub id: String,
    /// OpenAI object type, normally `chat.completion`.
    pub object: String,
    /// Unix timestamp when the response was created.
    pub created: i64,
    /// Model identifier that handled the request.
    pub model: String,
    /// Assistant choices; local inference currently emits one choice.
    pub choices: Vec<ChatCompletionChoice>,
    /// Final token usage for the request.
    pub usage: Usage,
}

/// One assistant choice in a non-streaming chat response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionChoice {
    /// Choice index in the OpenAI response shape.
    pub index: u32,
    /// Assistant message after runtime parsing and validation.
    pub message: ChatMessage,
    /// Final stop reason.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
}

/// Streaming chunk for the chat completions endpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionStreamResponse {
    /// Stable response identifier shared by every chunk in the stream.
    #[serde(with = "arc_str_serde")]
    pub id: Arc<str>,
    /// OpenAI object type, normally `chat.completion.chunk`.
    pub object: String,
    /// Unix timestamp when the stream was created.
    pub created: i64,
    /// Model identifier that handled the request.
    #[serde(with = "arc_str_serde")]
    pub model: Arc<str>,
    /// Delta choices or an empty list when the chunk carries only usage.
    pub choices: Vec<ChatCompletionStreamChoice>,
    /// Terminal usage payload when `stream_options.include_usage` is enabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// One choice delta in a streaming chat chunk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionStreamChoice {
    /// Choice index in the OpenAI response shape.
    pub index: u32,
    /// Incremental assistant message delta.
    pub delta: ChatCompletionDelta,
    /// Final stop reason; omitted on non-terminal deltas.
    pub finish_reason: Option<FinishReason>,
}

/// Incremental assistant message payload used in streaming chat responses.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionDelta {
    /// Role delta, emitted at stream start for assistant responses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<ChatRole>,
    /// Text delta when the runtime has validated it is safe to emit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Tool call deltas in the OpenAI streaming wire shape.
    ///
    /// Structured backend deltas may stream incrementally, while parsed text
    /// tool calls can be buffered until runtime validation permits emission.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallDelta>,
}

/// Incremental function tool call delta in the OpenAI streaming shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallDelta {
    /// Tool call index in the assistant message.
    pub index: u32,
    /// Tool call identifier, usually present on the first delta for the call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Tool call type, currently only `function`.
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub call_type: Option<ToolCallType>,
    /// Function name or argument text delta.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function: Option<ToolCallFunctionDelta>,
}

/// Incremental function payload inside a tool call delta.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallFunctionDelta {
    /// Function name or name fragment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// JSON argument string or argument fragment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

mod arc_str_serde {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::sync::Arc;

    pub(super) fn serialize<S>(value: &Arc<str>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(value)
    }

    pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<Arc<str>, D::Error>
    where
        D: Deserializer<'de>,
    {
        String::deserialize(deserializer).map(Arc::from)
    }
}
