use crate::{
    ApiError, ChatMessage, MAX_MESSAGE_CONTENT_BYTES, MAX_NAME_BYTES, MAX_STOP_SEQUENCE_BYTES,
    MAX_STOP_SEQUENCES, MAX_TOOL_ARGUMENT_BYTES, MAX_TOOL_CALLS_PER_MESSAGE,
    MAX_TOOL_DESCRIPTION_BYTES, MAX_TOOL_SCHEMA_BYTES, ToolDefinition,
};
use serde_json::Value;
use std::io::{self, Write};

pub(super) fn validate_chat_messages(messages: &[ChatMessage]) -> Result<(), ApiError> {
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

pub(super) fn validate_tools(tools: &[ToolDefinition]) -> Result<(), ApiError> {
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

pub(super) fn validate_stop_sequence_values(stop: &[String]) -> Result<(), ApiError> {
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

pub(super) fn validate_len_at_most(label: &str, actual: usize, max: usize) -> Result<(), ApiError> {
    if actual > max {
        return Err(ApiError::invalid_request(format!(
            "{label} must contain at most {max} entries"
        )));
    }
    Ok(())
}

pub(super) fn validate_string_bytes(label: &str, value: &str, max: usize) -> Result<(), ApiError> {
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

pub(super) fn validate_choice_count(n: Option<u32>) -> Result<(), ApiError> {
    match n {
        Some(0) => Err(ApiError::invalid_request("n must be greater than 0")),
        Some(1) | None => Ok(()),
        Some(_) => Err(ApiError::unsupported_capability(
            "multiple choices are not supported yet; use n 1",
        )),
    }
}

pub(super) fn validate_neutral_penalty(name: &str, value: Option<f32>) -> Result<(), ApiError> {
    if let Some(value) = value
        && (!value.is_finite() || value != 0.0)
    {
        return Err(ApiError::unsupported_capability(format!(
            "{name} is not supported yet; use {name} 0"
        )));
    }
    Ok(())
}

pub(super) fn validate_sampling_controls(
    temperature: Option<f32>,
    top_p: Option<f32>,
) -> Result<(), ApiError> {
    llm_util::sampling::validate_sampling_controls(temperature, top_p)
        .map_err(|err| ApiError::invalid_request(err.to_string()))
}
