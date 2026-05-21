use crate::{
    ApiError, ChatMessage, ChatRole, MAX_NAME_BYTES, MAX_STOP_SEQUENCE_BYTES, MAX_STOP_SEQUENCES,
    MAX_TOOL_ARGUMENT_BYTES, MAX_TOOL_CALLS_PER_MESSAGE, MAX_TOOL_DESCRIPTION_BYTES,
    MAX_TOOL_SCHEMA_BYTES, RequestLimits, ToolDefinition,
};
use serde_json::{Map, Value};
use std::{
    collections::BTreeSet,
    io::{self, Write},
};

pub(super) fn validate_chat_messages(
    messages: &[ChatMessage],
    limits: RequestLimits,
) -> Result<(), ApiError> {
    let mut seen_conversation_message = false;
    let mut pending_tool_call_ids = BTreeSet::new();
    let mut pending_tool_call_message = None;

    for (index, message) in messages.iter().enumerate() {
        let label = format!("messages[{index}].content");
        validate_chat_message_role_fields(index, message, &label)?;
        validate_chat_message_order(
            index,
            message,
            &mut seen_conversation_message,
            &mut pending_tool_call_ids,
            &mut pending_tool_call_message,
        )?;
        if let Some(content) = &message.content {
            validate_string_bytes(&label, content, limits.message_content_bytes)?;
        }
        if let Some(name) = &message.name {
            validate_string_bytes(&format!("messages[{index}].name"), name, MAX_NAME_BYTES)?;
        }
        if let Some(tool_call_id) = &message.tool_call_id {
            if message.role == ChatRole::Tool {
                validate_non_empty_string(
                    &format!("messages[{index}].tool_call_id"),
                    tool_call_id,
                )?;
            }
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
            validate_non_empty_string(
                &format!("messages[{index}].tool_calls[{tool_call_index}].function.name"),
                &tool_call.function.name,
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
    if let Some(index) = pending_tool_call_message {
        return Err(ApiError::invalid_request(format!(
            "messages[{index}].tool_calls must be followed by tool messages for every pending tool call"
        )));
    }
    Ok(())
}

fn validate_chat_message_role_fields(
    index: usize,
    message: &ChatMessage,
    content_label: &str,
) -> Result<(), ApiError> {
    match message.role {
        ChatRole::System | ChatRole::User => {
            if message.content.is_none() {
                return Err(ApiError::invalid_request(format!(
                    "{content_label} is required for {} messages",
                    message.role.as_str()
                )));
            }
            reject_tool_call_id_for_non_tool_message(index, message)?;
            reject_tool_calls_for_non_assistant_message(index, message)?;
        }
        ChatRole::Assistant => {
            if message.content.is_none() && message.tool_calls.is_empty() {
                return Err(ApiError::invalid_request(format!(
                    "{content_label} is required for assistant messages without tool_calls"
                )));
            }
            reject_tool_call_id_for_non_tool_message(index, message)?;
        }
        ChatRole::Tool => {
            if message.content.is_none() {
                return Err(ApiError::invalid_request(format!(
                    "{content_label} is required for tool messages"
                )));
            }
            if message.tool_call_id.is_none() {
                return Err(ApiError::invalid_request(format!(
                    "messages[{index}].tool_call_id is required for tool messages"
                )));
            }
            reject_tool_calls_for_non_assistant_message(index, message)?;
        }
    }
    Ok(())
}

fn reject_tool_call_id_for_non_tool_message(
    index: usize,
    message: &ChatMessage,
) -> Result<(), ApiError> {
    if message.tool_call_id.is_some() {
        return Err(ApiError::invalid_request(format!(
            "messages[{index}].tool_call_id is only allowed for tool messages"
        )));
    }
    Ok(())
}

fn reject_tool_calls_for_non_assistant_message(
    index: usize,
    message: &ChatMessage,
) -> Result<(), ApiError> {
    if !message.tool_calls.is_empty() {
        return Err(ApiError::invalid_request(format!(
            "messages[{index}].tool_calls is only allowed for assistant messages"
        )));
    }
    Ok(())
}

fn validate_chat_message_order<'a>(
    index: usize,
    message: &'a ChatMessage,
    seen_conversation_message: &mut bool,
    pending_tool_call_ids: &mut BTreeSet<&'a str>,
    pending_tool_call_message: &mut Option<usize>,
) -> Result<(), ApiError> {
    match message.role {
        ChatRole::System => {
            if *seen_conversation_message {
                return Err(ApiError::invalid_request(format!(
                    "messages[{index}].role system messages must appear before user, assistant, or tool messages"
                )));
            }
        }
        ChatRole::User => {
            reject_non_tool_message_while_tool_calls_pending(
                index,
                pending_tool_call_ids,
                *pending_tool_call_message,
            )?;
            *seen_conversation_message = true;
        }
        ChatRole::Assistant => {
            reject_non_tool_message_while_tool_calls_pending(
                index,
                pending_tool_call_ids,
                *pending_tool_call_message,
            )?;
            *seen_conversation_message = true;
            pending_tool_call_ids.clear();
            *pending_tool_call_message = None;
            for (tool_call_index, tool_call) in message.tool_calls.iter().enumerate() {
                if !pending_tool_call_ids.insert(tool_call.id.as_str()) {
                    return Err(ApiError::invalid_request(format!(
                        "messages[{index}].tool_calls[{tool_call_index}].id duplicates another tool call id in the same assistant message"
                    )));
                }
            }
            if !pending_tool_call_ids.is_empty() {
                *pending_tool_call_message = Some(index);
            }
        }
        ChatRole::Tool => {
            *seen_conversation_message = true;
            let Some(tool_call_id) = message.tool_call_id.as_deref() else {
                return Err(ApiError::invalid_request(format!(
                    "messages[{index}].tool_call_id is required for tool messages"
                )));
            };
            if pending_tool_call_ids.is_empty() {
                return Err(ApiError::invalid_request(format!(
                    "messages[{index}].role tool messages must follow an assistant message with tool_calls"
                )));
            }
            if !pending_tool_call_ids.remove(tool_call_id) {
                return Err(ApiError::invalid_request(format!(
                    "messages[{index}].tool_call_id `{tool_call_id}` does not match a pending assistant tool call"
                )));
            }
            if pending_tool_call_ids.is_empty() {
                *pending_tool_call_message = None;
            }
        }
    }
    Ok(())
}

fn reject_non_tool_message_while_tool_calls_pending(
    index: usize,
    pending_tool_call_ids: &BTreeSet<&str>,
    pending_tool_call_message: Option<usize>,
) -> Result<(), ApiError> {
    if !pending_tool_call_ids.is_empty() {
        let pending = pending_tool_call_message
            .map(|pending| format!(" from messages[{pending}].tool_calls"))
            .unwrap_or_default();
        return Err(ApiError::invalid_request(format!(
            "messages[{index}].role must be tool while assistant tool calls{pending} are pending"
        )));
    }
    Ok(())
}

pub(super) fn validate_tools(tools: &[ToolDefinition]) -> Result<(), ApiError> {
    for (index, tool) in tools.iter().enumerate() {
        validate_non_empty_string(
            &format!("tools[{index}].function.name"),
            &tool.function.name,
        )?;
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
        validate_tool_schema_shape(
            &format!("tools[{index}].function.parameters"),
            &tool.function.parameters,
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

pub(super) fn validate_non_empty_string(label: &str, value: &str) -> Result<(), ApiError> {
    if value.trim().is_empty() {
        return Err(ApiError::invalid_request(format!(
            "{label} must not be empty"
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

fn validate_tool_schema_shape(label: &str, value: &Value) -> Result<(), ApiError> {
    let Some(schema) = value.as_object() else {
        return Err(ApiError::invalid_request(format!(
            "{label} must be a JSON object"
        )));
    };
    validate_schema_object(label, schema)
}

fn validate_schema_object(label: &str, schema: &Map<String, Value>) -> Result<(), ApiError> {
    if let Some(schema_type) = schema.get("type") {
        match schema_type {
            Value::String(_) => {}
            Value::Array(types) if types.iter().all(Value::is_string) => {}
            _ => {
                return Err(ApiError::invalid_request(format!(
                    "{label}.type must be a string or array of strings"
                )));
            }
        }
    }
    if let Some(required) = schema.get("required") {
        let Some(required) = required.as_array() else {
            return Err(ApiError::invalid_request(format!(
                "{label}.required must be an array of strings"
            )));
        };
        if !required.iter().all(Value::is_string) {
            return Err(ApiError::invalid_request(format!(
                "{label}.required must contain only strings"
            )));
        }
    }
    if let Some(enum_values) = schema.get("enum")
        && !enum_values.is_array()
    {
        return Err(ApiError::invalid_request(format!(
            "{label}.enum must be an array"
        )));
    }
    if let Some(properties) = schema.get("properties") {
        let Some(properties) = properties.as_object() else {
            return Err(ApiError::invalid_request(format!(
                "{label}.properties must be a JSON object"
            )));
        };
        for (name, property_schema) in properties {
            if let Some(property_schema) = property_schema.as_object() {
                validate_schema_object(&format!("{label}.properties.{name}"), property_schema)?;
            }
        }
    }
    if let Some(items) = schema.get("items")
        && let Some(items_schema) = items.as_object()
    {
        validate_schema_object(&format!("{label}.items"), items_schema)?;
    }
    Ok(())
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
