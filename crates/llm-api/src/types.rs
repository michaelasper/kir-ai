use crate::ToolCall;
use schemars::JsonSchema;
use serde::{
    Deserialize, Deserializer, Serialize, Serializer, de::Error as _, ser::SerializeStruct,
};
use serde_json::Value;

/// Role assigned to a chat message in the OpenAI wire format.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ChatRole {
    /// System instruction message.
    System,
    /// End-user message.
    User,
    /// Assistant message, including model text and tool calls.
    Assistant,
    /// Tool result message that answers a prior assistant tool call.
    Tool,
}

impl ChatRole {
    /// Returns the lowercase OpenAI role string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

/// A single chat message as accepted and emitted by the OpenAI-compatible API.
///
/// The deserializer accepts plain string content, `null`, or an array of text
/// content parts. Non-text multimodal parts are rejected because this runtime is
/// fail-closed for unsupported request modalities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Message role that determines which other fields are valid.
    pub role: ChatRole,
    /// Text content after compatible text content parts have been flattened.
    #[serde(
        default,
        deserialize_with = "deserialize_message_content",
        skip_serializing_if = "Option::is_none"
    )]
    pub content: Option<String>,
    /// Optional participant name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Tool call identifier answered by a `tool` role message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Tool calls emitted by an assistant message.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
}

impl ChatMessage {
    /// Builds a system text message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::plain(ChatRole::System, content)
    }

    /// Builds a user text message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::plain(ChatRole::User, content)
    }

    /// Builds an assistant text message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::plain(ChatRole::Assistant, content)
    }

    /// Builds a tool result message for a previously emitted tool call.
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Tool,
            content: Some(content.into()),
            name: None,
            tool_call_id: Some(tool_call_id.into()),
            tool_calls: Vec::new(),
        }
    }

    /// Builds an assistant message that contains one function tool call.
    pub fn assistant_tool_call(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: Value,
    ) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: None,
            name: None,
            tool_call_id: None,
            tool_calls: vec![ToolCall {
                id: id.into(),
                call_type: crate::ToolCallType::Function,
                function: crate::ToolCallFunction {
                    name: name.into(),
                    arguments,
                },
            }],
        }
    }

    fn plain(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: Some(content.into()),
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
        }
    }
}

/// Assistant response format requested by a chat client.
///
/// `JsonSchema` is represented for wire compatibility but rejected by
/// validation until schema-constrained generation is implemented.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ResponseFormat {
    /// Default free-form text response.
    Text,
    /// Require the final assistant content to parse as a JSON object.
    JsonObject,
    /// OpenAI JSON schema mode, currently unsupported by the runtime.
    JsonSchema { json_schema: Value },
}

/// OpenAI-compatible reason that generation stopped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum FinishReason {
    /// Generation stopped naturally or due to a client stop sequence.
    Stop,
    /// The configured token limit was reached.
    Length,
    /// The assistant produced one or more tool calls.
    ToolCalls,
    /// Content filtering stopped generation.
    ContentFilter,
    /// Backend reported an error finish condition.
    Error,
}

/// Token accounting returned with terminal non-streaming responses and usage chunks.
///
/// `total_tokens` is serialized as `prompt_tokens + completion_tokens`; custom
/// deserialization rejects mismatched wire values so tests catch accounting
/// regressions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Usage {
    /// Tokens consumed by the prompt.
    pub prompt_tokens: u64,
    /// Tokens generated by the model.
    pub completion_tokens: u64,
    /// Total token count, kept for OpenAI response compatibility.
    pub total_tokens: u64,
    /// Optional details about prompt tokens.
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

/// Additional prompt token accounting.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptTokensDetails {
    /// Prompt tokens served from cache.
    pub cached_tokens: u64,
}

impl Usage {
    /// Creates usage with a computed total token count and no prompt details.
    pub fn new(prompt_tokens: u64, completion_tokens: u64) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
            prompt_tokens_details: None,
        }
    }

    fn computed_total_tokens(&self) -> u64 {
        self.prompt_tokens.saturating_add(self.completion_tokens)
    }

    /// Attaches cached prompt token details when cache metadata is available.
    pub fn with_prompt_cached_tokens(mut self, cached_tokens: Option<u64>) -> Self {
        self.prompt_tokens_details =
            cached_tokens.map(|cached_tokens| PromptTokensDetails { cached_tokens });
        self
    }
}

impl Serialize for Usage {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let field_count = if self.prompt_tokens_details.is_some() {
            4
        } else {
            3
        };
        let mut usage = serializer.serialize_struct("Usage", field_count)?;
        usage.serialize_field("prompt_tokens", &self.prompt_tokens)?;
        usage.serialize_field("completion_tokens", &self.completion_tokens)?;
        usage.serialize_field("total_tokens", &self.computed_total_tokens())?;
        if let Some(details) = &self.prompt_tokens_details {
            usage.serialize_field("prompt_tokens_details", details)?;
        }
        usage.end()
    }
}

impl<'de> Deserialize<'de> for Usage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct WireUsage {
            prompt_tokens: u64,
            completion_tokens: u64,
            total_tokens: u64,
            #[serde(default)]
            prompt_tokens_details: Option<PromptTokensDetails>,
        }

        let usage = WireUsage::deserialize(deserializer)?;
        let expected_total_tokens = usage.prompt_tokens.saturating_add(usage.completion_tokens);
        if usage.total_tokens != expected_total_tokens {
            return Err(D::Error::custom(format!(
                "total_tokens must equal prompt_tokens + completion_tokens ({expected_total_tokens})"
            )));
        }

        Ok(Self {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: expected_total_tokens,
            prompt_tokens_details: usage.prompt_tokens_details,
        })
    }
}

/// Options that only affect streaming response shape.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamOptions {
    /// Whether to include the OpenAI terminal usage chunk before `[DONE]`.
    #[serde(default)]
    pub include_usage: bool,
}

impl StreamOptions {
    /// Returns true when this value serializes identically to omitted stream options.
    pub fn is_default(&self) -> bool {
        !self.include_usage
    }
}

/// OpenAI-compatible model list entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ModelCard {
    /// Model identifier clients use in request bodies.
    pub id: String,
    /// Wire object type, normally `model`.
    pub object: String,
    /// Owner string exposed by the local server.
    pub owned_by: String,
}

/// OpenAI-compatible model list response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ModelList {
    /// Wire object type, normally `list`.
    pub object: String,
    /// Models currently available to the server.
    pub data: Vec<ModelCard>,
}

fn deserialize_message_content<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let Some(value) = Option::<Value>::deserialize(deserializer)? else {
        return Ok(None);
    };
    match value {
        Value::Null => Ok(None),
        Value::String(text) => Ok(Some(text)),
        Value::Array(parts) => {
            let mut text = String::new();
            for part in parts {
                let object = part.as_object().ok_or_else(|| {
                    D::Error::custom("message content parts must be JSON objects")
                })?;
                let part_type = object
                    .get("type")
                    .and_then(Value::as_str)
                    .ok_or_else(|| D::Error::custom("message content part type is required"))?;
                if part_type != "text" {
                    return Err(D::Error::custom(format!(
                        "unsupported message content part type `{part_type}`"
                    )));
                }
                let part_text = object
                    .get("text")
                    .and_then(Value::as_str)
                    .ok_or_else(|| D::Error::custom("text message content part requires text"))?;
                append_message_content_text_part(&mut text, part_text);
            }
            Ok(Some(text))
        }
        _ => Err(D::Error::custom(
            "message content must be a string, null, or an array of text parts",
        )),
    }
}

fn append_message_content_text_part(text: &mut String, part_text: &str) {
    if !text.is_empty()
        && !part_text.is_empty()
        && !text
            .chars()
            .next_back()
            .is_some_and(|last| last.is_whitespace())
        && !part_text
            .chars()
            .next()
            .is_some_and(|first| first.is_whitespace())
    {
        text.push(' ');
    }
    text.push_str(part_text);
}
