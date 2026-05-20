mod helpers;

use crate::{
    ApiError, ChatCompletionRequest, CompletionRequest, MAX_CHAT_MESSAGES, MAX_TOOLS,
    RequestLimits, ResponseFormat, ToolChoice,
};
use helpers::{
    validate_chat_messages, validate_choice_count, validate_len_at_most, validate_neutral_penalty,
    validate_sampling_controls, validate_stop_sequence_values, validate_string_bytes,
    validate_tools,
};
use serde::{Deserialize, Deserializer, de::Error as _};
use serde_json::Value;
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Validated<T> {
    inner: T,
    request_limits: RequestLimits,
}

impl<T> Validated<T> {
    pub fn request_limits(&self) -> RequestLimits {
        self.request_limits
    }

    pub fn into_inner(self) -> T {
        self.inner
    }
}

impl<T> AsRef<T> for Validated<T> {
    fn as_ref(&self) -> &T {
        &self.inner
    }
}

pub trait ValidateRequest {
    fn validate(&self) -> Result<(), ApiError> {
        self.validate_with_limits(RequestLimits::default())
    }

    fn into_validated(self) -> Result<Validated<Self>, ApiError>
    where
        Self: Sized,
    {
        self.into_validated_with_limits(RequestLimits::default())
    }

    fn into_validated_with_limits(self, limits: RequestLimits) -> Result<Validated<Self>, ApiError>
    where
        Self: Sized,
    {
        self.validate_with_limits(limits)?;
        Ok(Validated {
            inner: self,
            request_limits: limits,
        })
    }

    fn validate_with_limits(&self, limits: RequestLimits) -> Result<(), ApiError>;
}

impl ValidateRequest for ChatCompletionRequest {
    fn validate_with_limits(&self, limits: RequestLimits) -> Result<(), ApiError> {
        if self.model.trim().is_empty() {
            return Err(ApiError::invalid_request("model is required"));
        }
        if self.messages.is_empty() {
            return Err(ApiError::invalid_request("messages must not be empty"));
        }
        validate_len_at_most("messages", self.messages.len(), MAX_CHAT_MESSAGES)?;
        validate_chat_messages(&self.messages, limits)?;
        validate_len_at_most("tools", self.tools.len(), MAX_TOOLS)?;
        validate_tools(&self.tools)?;
        validate_stop_sequence_values(&self.stop)?;
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
    fn validate_with_limits(&self, limits: RequestLimits) -> Result<(), ApiError> {
        if self.model.trim().is_empty() {
            return Err(ApiError::invalid_request("model is required"));
        }
        if self.prompt.is_empty() {
            return Err(ApiError::invalid_request("prompt must not be empty"));
        }
        validate_string_bytes("prompt", &self.prompt, limits.completion_prompt_bytes)?;
        validate_stop_sequence_values(&self.stop)?;
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

pub fn deserialize_stop_sequences<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
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
