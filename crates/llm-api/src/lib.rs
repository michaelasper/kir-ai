use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};
use serde_json::Value;
use std::{
    collections::BTreeSet,
    io::{self, Write},
};
use thiserror::Error;

pub const MAX_JSON_BODY_BYTES: usize = 2 * 1024 * 1024;
pub const MAX_CHAT_MESSAGES: usize = 128;
pub const MAX_MESSAGE_CONTENT_BYTES: usize = 1024 * 1024;
pub const MAX_COMPLETION_PROMPT_BYTES: usize = 1024 * 1024;
pub const MAX_NAME_BYTES: usize = 1024;
pub const MAX_TOOLS: usize = 128;
pub const MAX_TOOL_DESCRIPTION_BYTES: usize = 1024 * 1024;
pub const MAX_TOOL_SCHEMA_BYTES: usize = 1024 * 1024;
pub const MAX_TOOL_CALLS_PER_MESSAGE: usize = 128;
pub const MAX_TOOL_ARGUMENT_BYTES: usize = 1024 * 1024;
pub const MAX_STOP_SEQUENCES: usize = 4;
pub const MAX_STOP_SEQUENCE_BYTES: usize = 1024;

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
                call_type: ToolCallType::Function,
                function: ToolCallFunction {
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
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: ToolCallType,
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallType {
    Function,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    #[serde(
        serialize_with = "serialize_tool_call_arguments",
        deserialize_with = "deserialize_tool_call_arguments"
    )]
    pub arguments: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: ToolCallType,
    pub function: FunctionDefinition,
}

impl ToolDefinition {
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
    ) -> Self {
        Self {
            tool_type: ToolCallType::Function,
            function: FunctionDefinition {
                name: name.into(),
                description: Some(description.into()),
                parameters,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default = "empty_object")]
    pub parameters: Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
    Required,
    Function {
        name: String,
    },
}

impl<'de> Deserialize<'de> for ToolChoice {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match value {
            Value::String(s) if s == "auto" => Ok(Self::Auto),
            Value::String(s) if s == "none" => Ok(Self::None),
            Value::String(s) if s == "required" => Ok(Self::Required),
            Value::Object(mut obj) => {
                let kind = obj
                    .remove("type")
                    .and_then(|v| v.as_str().map(str::to_owned))
                    .ok_or_else(|| serde::de::Error::custom("tool_choice.type is required"))?;
                if kind != "function" {
                    return Err(serde::de::Error::custom(
                        "only function tool_choice is supported",
                    ));
                }
                let function = obj
                    .remove("function")
                    .and_then(|v| v.as_object().cloned())
                    .ok_or_else(|| serde::de::Error::custom("tool_choice.function is required"))?;
                let name = function
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        serde::de::Error::custom("tool_choice.function.name is required")
                    })?;
                Ok(Self::Function {
                    name: name.to_owned(),
                })
            }
            _ => Err(serde::de::Error::custom("invalid tool_choice")),
        }
    }
}

impl Serialize for ToolChoice {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Auto => serializer.serialize_str("auto"),
            Self::None => serializer.serialize_str("none"),
            Self::Required => serializer.serialize_str("required"),
            Self::Function { name } => {
                serde_json::json!({"type": "function", "function": {"name": name}})
                    .serialize(serializer)
            }
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

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default, skip_serializing_if = "StreamOptions::is_default")]
    pub stream_options: StreamOptions,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(
        default,
        deserialize_with = "deserialize_stop_sequences",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub stop: Vec<String>,
}

impl ChatCompletionRequest {
    pub fn effective_max_tokens(&self) -> Option<u32> {
        self.max_completion_tokens.or(self.max_tokens)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub stream: bool,
    #[serde(default, skip_serializing_if = "StreamOptions::is_default")]
    pub stream_options: StreamOptions,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(
        default,
        deserialize_with = "deserialize_stop_sequences",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub stop: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionStreamResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatMessage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionStreamResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatCompletionStreamChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionStreamChoice {
    pub index: u32,
    pub delta: ChatCompletionDelta,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionDelta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<ChatRole>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallDelta>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallDelta {
    pub index: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub call_type: Option<ToolCallType>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function: Option<ToolCallFunctionDelta>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallFunctionDelta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
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

pub trait ValidateRequest {
    fn validate(&self) -> Result<(), ApiError>;
}

impl ValidateRequest for ChatCompletionRequest {
    fn validate(&self) -> Result<(), ApiError> {
        if self.model.trim().is_empty() {
            return Err(ApiError::invalid_request("model is required"));
        }
        if self.messages.is_empty() {
            return Err(ApiError::invalid_request("messages must not be empty"));
        }
        validate_len_at_most("messages", self.messages.len(), MAX_CHAT_MESSAGES)?;
        validate_chat_messages(&self.messages)?;
        validate_len_at_most("tools", self.tools.len(), MAX_TOOLS)?;
        validate_tools(&self.tools)?;
        validate_stop_sequences(&self.stop)?;
        if matches!(
            self.response_format,
            Some(ResponseFormat::JsonSchema { .. })
        ) {
            return Err(ApiError::unsupported_capability(
                "json_schema response_format is not supported; use json_object",
            ));
        }
        if matches!(self.tool_choice, Some(ToolChoice::Required)) && self.tools.is_empty() {
            return Err(ApiError::invalid_request(
                "tool_choice required needs at least one declared tool",
            ));
        }
        if let Some(ToolChoice::Function { name }) = &self.tool_choice {
            let names = self
                .tools
                .iter()
                .map(|tool| tool.function.name.as_str())
                .collect::<BTreeSet<_>>();
            if !names.contains(name.as_str()) {
                return Err(ApiError::unsupported_capability(format!(
                    "required tool `{name}` was not declared"
                )));
            }
        }
        if matches!(self.parallel_tool_calls, Some(true)) {
            return Err(ApiError::unsupported_capability(
                "parallel tool calls are not supported yet; use parallel_tool_calls false",
            ));
        }
        validate_sampling_controls(self.temperature, self.top_p)?;
        validate_neutral_penalty("presence_penalty", self.presence_penalty)?;
        validate_neutral_penalty("frequency_penalty", self.frequency_penalty)?;
        if matches!(self.logprobs, Some(true)) {
            return Err(ApiError::unsupported_capability(
                "logprobs are not supported yet; use logprobs false",
            ));
        }
        if self.top_logprobs.is_some() {
            return Err(ApiError::unsupported_capability(
                "top_logprobs are not supported yet",
            ));
        }
        if matches!(self.max_tokens, Some(0)) {
            return Err(ApiError::invalid_request(
                "max_tokens must be greater than 0",
            ));
        }
        if matches!(self.max_completion_tokens, Some(0)) {
            return Err(ApiError::invalid_request(
                "max_completion_tokens must be greater than 0",
            ));
        }
        if let (Some(max_tokens), Some(max_completion_tokens)) =
            (self.max_tokens, self.max_completion_tokens)
            && max_tokens != max_completion_tokens
        {
            return Err(ApiError::invalid_request(
                "max_tokens and max_completion_tokens must match when both are provided",
            ));
        }
        validate_choice_count(self.n)?;
        Ok(())
    }
}

impl ValidateRequest for CompletionRequest {
    fn validate(&self) -> Result<(), ApiError> {
        if self.model.trim().is_empty() {
            return Err(ApiError::invalid_request("model is required"));
        }
        if self.prompt.is_empty() {
            return Err(ApiError::invalid_request("prompt must not be empty"));
        }
        validate_string_bytes("prompt", &self.prompt, MAX_COMPLETION_PROMPT_BYTES)?;
        validate_stop_sequences(&self.stop)?;
        validate_sampling_controls(self.temperature, self.top_p)?;
        validate_neutral_penalty("presence_penalty", self.presence_penalty)?;
        validate_neutral_penalty("frequency_penalty", self.frequency_penalty)?;
        if self.logprobs.is_some() {
            return Err(ApiError::unsupported_capability(
                "completion logprobs are not supported yet",
            ));
        }
        if matches!(self.max_tokens, Some(0)) {
            return Err(ApiError::invalid_request(
                "max_tokens must be greater than 0",
            ));
        }
        validate_choice_count(self.n)?;
        Ok(())
    }
}

fn validate_chat_messages(messages: &[ChatMessage]) -> Result<(), ApiError> {
    for (index, message) in messages.iter().enumerate() {
        let label = format!("messages[{index}].content");
        if let Some(content) = &message.content {
            validate_string_bytes(&label, content, MAX_MESSAGE_CONTENT_BYTES)?;
        }
        if let Some(name) = &message.name {
            validate_string_bytes(&format!("messages[{index}].name"), name, MAX_NAME_BYTES)?;
        }
        if let Some(tool_call_id) = &message.tool_call_id {
            validate_string_bytes(
                &format!("messages[{index}].tool_call_id"),
                tool_call_id,
                MAX_NAME_BYTES,
            )?;
        }
        let tool_calls_label = format!("messages[{index}].tool_calls");
        validate_len_at_most(
            &tool_calls_label,
            message.tool_calls.len(),
            MAX_TOOL_CALLS_PER_MESSAGE,
        )?;
        for (tool_call_index, tool_call) in message.tool_calls.iter().enumerate() {
            validate_string_bytes(
                &format!("messages[{index}].tool_calls[{tool_call_index}].id"),
                &tool_call.id,
                MAX_NAME_BYTES,
            )?;
            validate_string_bytes(
                &format!("messages[{index}].tool_calls[{tool_call_index}].function.name"),
                &tool_call.function.name,
                MAX_NAME_BYTES,
            )?;
            validate_json_bytes_at_most(
                &format!("messages[{index}].tool_calls[{tool_call_index}].function.arguments"),
                &tool_call.function.arguments,
                MAX_TOOL_ARGUMENT_BYTES,
            )?;
        }
    }
    Ok(())
}

fn validate_tools(tools: &[ToolDefinition]) -> Result<(), ApiError> {
    for (index, tool) in tools.iter().enumerate() {
        validate_string_bytes(
            &format!("tools[{index}].function.name"),
            &tool.function.name,
            MAX_NAME_BYTES,
        )?;
        if let Some(description) = &tool.function.description {
            validate_string_bytes(
                &format!("tools[{index}].function.description"),
                description,
                MAX_TOOL_DESCRIPTION_BYTES,
            )?;
        }
        validate_json_bytes_at_most(
            &format!("tools[{index}].function.parameters"),
            &tool.function.parameters,
            MAX_TOOL_SCHEMA_BYTES,
        )?;
    }
    Ok(())
}

fn validate_stop_sequences(stop: &[String]) -> Result<(), ApiError> {
    validate_len_at_most("stop", stop.len(), MAX_STOP_SEQUENCES)?;
    for (index, sequence) in stop.iter().enumerate() {
        if sequence.is_empty() {
            return Err(ApiError::invalid_request(
                "stop sequences must not be empty",
            ));
        }
        validate_string_bytes(&format!("stop[{index}]"), sequence, MAX_STOP_SEQUENCE_BYTES)?;
    }
    Ok(())
}

fn validate_len_at_most(label: &str, actual: usize, max: usize) -> Result<(), ApiError> {
    if actual > max {
        return Err(ApiError::invalid_request(format!(
            "{label} must contain at most {max} entries"
        )));
    }
    Ok(())
}

fn validate_string_bytes(label: &str, value: &str, max: usize) -> Result<(), ApiError> {
    if value.len() > max {
        return Err(ApiError::invalid_request(format!(
            "{label} must be at most {max} bytes"
        )));
    }
    Ok(())
}

fn validate_json_bytes_at_most(label: &str, value: &Value, max: usize) -> Result<(), ApiError> {
    let mut counter = JsonByteCounter::new(max);
    match serde_json::to_writer(&mut counter, value) {
        Ok(()) => Ok(()),
        Err(_) if counter.exceeded() => Err(ApiError::invalid_request(format!(
            "{label} must serialize to at most {max} bytes"
        ))),
        Err(err) => Err(ApiError::invalid_request(format!(
            "{label} must serialize as JSON: {err}"
        ))),
    }
}

struct JsonByteCounter {
    written: usize,
    max: usize,
}

impl JsonByteCounter {
    fn new(max: usize) -> Self {
        Self { written: 0, max }
    }

    fn exceeded(&self) -> bool {
        self.written > self.max
    }
}

impl Write for JsonByteCounter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.written = self.written.saturating_add(buf.len());
        if self.exceeded() {
            return Err(io::Error::other("JSON byte limit exceeded"));
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn validate_choice_count(n: Option<u32>) -> Result<(), ApiError> {
    match n {
        Some(0) => Err(ApiError::invalid_request("n must be greater than 0")),
        Some(1) | None => Ok(()),
        Some(_) => Err(ApiError::unsupported_capability(
            "multiple choices are not supported yet; use n 1",
        )),
    }
}

fn validate_neutral_penalty(name: &str, value: Option<f32>) -> Result<(), ApiError> {
    if let Some(value) = value
        && (!value.is_finite() || value != 0.0)
    {
        return Err(ApiError::unsupported_capability(format!(
            "{name} is not supported yet; use {name} 0"
        )));
    }
    Ok(())
}

fn validate_sampling_controls(
    temperature: Option<f32>,
    top_p: Option<f32>,
) -> Result<(), ApiError> {
    if let Some(temperature) = temperature
        && (!temperature.is_finite() || !(0.0..=2.0).contains(&temperature))
    {
        return Err(ApiError::invalid_request(
            "temperature must be finite and in [0, 2]",
        ));
    }
    if let Some(top_p) = top_p
        && (!top_p.is_finite() || top_p <= 0.0 || top_p > 1.0)
    {
        return Err(ApiError::invalid_request(
            "top_p must be finite and in (0, 1]",
        ));
    }
    Ok(())
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
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

#[derive(Debug, Clone, PartialEq, Eq, Error, JsonSchema)]
#[error("{code}: {message}")]
pub struct ApiError {
    code: &'static str,
    message: String,
}

impl ApiError {
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            code: "invalid_request",
            message: message.into(),
        }
    }

    pub fn unsupported_capability(message: impl Into<String>) -> Self {
        Self {
            code: "unsupported_capability",
            message: message.into(),
        }
    }

    pub fn code(&self) -> &'static str {
        self.code
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

fn empty_object() -> Value {
    serde_json::json!({})
}

fn serialize_tool_call_arguments<S>(arguments: &Value, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let encoded = serde_json::to_string(arguments).map_err(serde::ser::Error::custom)?;
    serializer.serialize_str(&encoded)
}

fn deserialize_tool_call_arguments<'de, D>(deserializer: D) -> Result<Value, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    match value {
        Value::String(arguments) => serde_json::from_str(&arguments).map_err(D::Error::custom),
        arguments => Ok(arguments),
    }
}

fn deserialize_stop_sequences<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<Value>::deserialize(deserializer)?;
    match value {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::String(stop)) => Ok(vec![stop]),
        Some(Value::Array(items)) => items
            .into_iter()
            .map(|item| match item {
                Value::String(stop) => Ok(stop),
                _ => Err(D::Error::custom("stop array must contain only strings")),
            })
            .collect(),
        Some(_) => Err(D::Error::custom(
            "stop must be a string or array of strings",
        )),
    }
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
                text.push_str(part_text);
            }
            Ok(Some(text))
        }
        _ => Err(D::Error::custom(
            "message content must be a string, null, or an array of text parts",
        )),
    }
}
