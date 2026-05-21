use crate::ToolCall;
use schemars::JsonSchema;
use serde::{
    Deserialize, Deserializer, Serialize, Serializer, de::Error as _, ser::SerializeStruct,
};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

impl ChatRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    #[serde(
        default,
        deserialize_with = "deserialize_message_content",
        skip_serializing_if = "Option::is_none"
    )]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self::plain(ChatRole::System, content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::plain(ChatRole::User, content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::plain(ChatRole::Assistant, content)
    }

    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Tool,
            content: Some(content.into()),
            name: None,
            tool_call_id: Some(tool_call_id.into()),
            tool_calls: Vec::new(),
        }
    }

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: Value },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptTokensDetails {
    pub cached_tokens: u64,
}

impl Usage {
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

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

impl StreamOptions {
    pub fn is_default(&self) -> bool {
        !self.include_usage
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ModelCard {
    pub id: String,
    pub object: String,
    pub owned_by: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ModelList {
    pub object: String,
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
