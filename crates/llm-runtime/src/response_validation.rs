use crate::RuntimeError;
use llm_api::{ChatCompletionRequest, ResponseFormat, ToolCall, ToolChoice};
use llm_tool_parser::ParsedAssistant;
use std::collections::BTreeSet;

pub(crate) fn schema_requires_string_intent_argument(schema: &serde_json::Value) -> bool {
    let Some(schema_object) = schema.as_object() else {
        return false;
    };
    let Some(required) = schema_object
        .get("required")
        .and_then(serde_json::Value::as_array)
    else {
        return false;
    };
    if !required.iter().any(|field| field.as_str() == Some("_i")) {
        return false;
    }
    let Some(intent_schema) = schema_object
        .get("properties")
        .and_then(serde_json::Value::as_object)
        .and_then(|properties| properties.get("_i"))
    else {
        return false;
    };
    intent_schema
        .get("type")
        .is_some_and(schema_type_accepts_string)
}

pub(crate) fn default_tool_intent(tool_name: &str) -> &'static str {
    match tool_name {
        "read" => "Reading requested path",
        "bash" => "Running requested command",
        "edit" => "Editing requested file",
        "find" => "Finding requested files",
        name if name.contains("search") || name.contains("grep") => "Searching requested context",
        _ => "Calling requested tool",
    }
}

pub(crate) fn validate_tool_call_arguments(parsed: &ParsedAssistant) -> Result<(), RuntimeError> {
    for tool_call in &parsed.tool_calls {
        if !tool_call.function.arguments.is_object() {
            return Err(RuntimeError::JsonMode(format!(
                "tool call `{}` arguments must be a JSON object",
                tool_call.function.name
            )));
        }
    }
    Ok(())
}

pub(crate) fn validate_tool_calls_against_request(
    parsed: &ParsedAssistant,
    request: &ChatCompletionRequest,
) -> Result<(), RuntimeError> {
    if parsed.tool_calls.is_empty() {
        return Ok(());
    }
    if matches!(request.tool_choice, Some(ToolChoice::None)) {
        return Err(RuntimeError::ToolCallValidation(
            "tool_choice none does not allow generated tool calls".to_owned(),
        ));
    }
    let declared_tools = request
        .tools
        .iter()
        .map(|tool| tool.function.name.as_str())
        .collect::<BTreeSet<_>>();
    for tool_call in &parsed.tool_calls {
        let name = tool_call.function.name.as_str();
        if !declared_tools.contains(name) {
            return Err(RuntimeError::ToolCallValidation(format!(
                "generated tool call `{name}` was not declared in request tools"
            )));
        }
        if let Some(ToolChoice::Function { name: required }) = &request.tool_choice
            && name != required
        {
            return Err(RuntimeError::ToolCallValidation(format!(
                "generated tool call `{name}` did not match required tool `{required}`"
            )));
        }
        let Some(tool) = request.tools.iter().find(|tool| tool.function.name == name) else {
            return Err(RuntimeError::ToolCallValidation(format!(
                "generated tool call `{name}` was not declared in request tools"
            )));
        };
        validate_tool_call_arguments_against_schema(tool_call, &tool.function.parameters)?;
    }
    Ok(())
}

pub(crate) fn validate_json_object_response(parsed: &ParsedAssistant) -> Result<(), RuntimeError> {
    if !parsed.content.is_empty() {
        let value = serde_json::from_str::<serde_json::Value>(&parsed.content).map_err(|err| {
            RuntimeError::JsonMode(format!(
                "json_object response_format requires valid JSON object content: {err}"
            ))
        })?;
        if !value.is_object() {
            return Err(RuntimeError::JsonMode(
                "json_object response_format requires assistant content to be a JSON object"
                    .to_owned(),
            ));
        }
    } else if parsed.tool_calls.is_empty() {
        return Err(RuntimeError::JsonMode(
            "json_object response_format requires assistant content or tool calls".to_owned(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_json_object_response_format(
    parsed: &ParsedAssistant,
    request: &ChatCompletionRequest,
) -> Result<(), RuntimeError> {
    if matches!(
        request.response_format.as_ref(),
        Some(ResponseFormat::JsonObject)
    ) {
        validate_json_object_response(parsed)?;
    }
    Ok(())
}

fn validate_tool_call_arguments_against_schema(
    tool_call: &ToolCall,
    schema: &serde_json::Value,
) -> Result<(), RuntimeError> {
    if !tool_call.function.arguments.is_object() {
        return Err(RuntimeError::ToolCallValidation(format!(
            "generated tool call `{}` arguments must be a JSON object",
            tool_call.function.name
        )));
    }
    let tool_name = tool_call.function.name.as_str();
    validate_json_schema_value(tool_name, "", &tool_call.function.arguments, schema)
}

fn validate_json_schema_value(
    tool_name: &str,
    path: &str,
    value: &serde_json::Value,
    schema: &serde_json::Value,
) -> Result<(), RuntimeError> {
    if schema.is_null() || schema.as_object().is_some_and(serde_json::Map::is_empty) {
        return Ok(());
    }
    let Some(schema_object) = schema.as_object() else {
        return Ok(());
    };
    if let Some(allowed_type) = schema_object.get("type")
        && !schema_type_matches(allowed_type, value)
    {
        return Err(RuntimeError::ToolCallValidation(format!(
            "generated tool call `{tool_name}` argument `{}` does not match schema type {}",
            display_schema_path(path),
            display_schema_type(allowed_type)
        )));
    }
    if let Some(enum_values) = schema_object.get("enum") {
        let enum_values = enum_values.as_array().ok_or_else(|| {
            RuntimeError::ToolCallValidation(format!(
                "tool `{tool_name}` schema enum for `{}` must be an array",
                display_schema_path(path)
            ))
        })?;
        if !enum_values.iter().any(|allowed| allowed == value) {
            return Err(RuntimeError::ToolCallValidation(format!(
                "generated tool call `{tool_name}` argument `{}` is not one of the allowed enum values",
                display_schema_path(path)
            )));
        }
    }
    if let Some(object) = value.as_object() {
        if let Some(required) = schema_object.get("required") {
            let required = required.as_array().ok_or_else(|| {
                RuntimeError::ToolCallValidation(format!(
                    "tool `{tool_name}` schema required for `{}` must be an array",
                    display_schema_path(path)
                ))
            })?;
            let required_fields = required
                .iter()
                .map(|field| {
                    field.as_str().ok_or_else(|| {
                        RuntimeError::ToolCallValidation(format!(
                            "tool `{tool_name}` schema required entries for `{}` must be strings",
                            display_schema_path(path)
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            for field in &required_fields {
                if !object.contains_key(*field) {
                    return Err(RuntimeError::ToolCallValidation(
                        missing_required_argument_message(
                            tool_name,
                            path,
                            field,
                            &required_fields,
                            schema_object,
                        ),
                    ));
                }
            }
        }
        if let Some(properties) = schema_object.get("properties") {
            let properties = properties.as_object().ok_or_else(|| {
                RuntimeError::ToolCallValidation(format!(
                    "tool `{tool_name}` schema properties for `{}` must be an object",
                    display_schema_path(path)
                ))
            })?;
            for (field, field_schema) in properties {
                if let Some(field_value) = object.get(field) {
                    validate_json_schema_value(
                        tool_name,
                        &join_schema_path(path, field),
                        field_value,
                        field_schema,
                    )?;
                }
            }
        }
    }
    if let Some(array) = value.as_array()
        && let Some(items_schema) = schema_object.get("items")
    {
        for (index, item) in array.iter().enumerate() {
            validate_json_schema_value(
                tool_name,
                &format!("{}[{index}]", display_schema_path(path)),
                item,
                items_schema,
            )?;
        }
    }
    Ok(())
}

fn missing_required_argument_message(
    tool_name: &str,
    path: &str,
    field: &str,
    required_fields: &[&str],
    schema_object: &serde_json::Map<String, serde_json::Value>,
) -> String {
    let required = required_fields
        .iter()
        .map(|field| format!("`{}`", join_schema_path(path, field)))
        .collect::<Vec<_>>()
        .join(", ");
    let hint = expected_arguments_object_hint(required_fields, schema_object);
    format!(
        "generated tool call `{tool_name}` missing required argument `{}`; required arguments: {required}; expected arguments object: {hint}",
        join_schema_path(path, field)
    )
}

fn expected_arguments_object_hint(
    required_fields: &[&str],
    schema_object: &serde_json::Map<String, serde_json::Value>,
) -> String {
    let properties = schema_object
        .get("properties")
        .and_then(serde_json::Value::as_object);
    let mut hint = serde_json::Map::new();
    for field in required_fields {
        let field_schema = properties.and_then(|properties| properties.get(*field));
        hint.insert(
            (*field).to_owned(),
            expected_schema_value_hint(field_schema),
        );
    }
    serde_json::to_string(&serde_json::Value::Object(hint)).unwrap_or_else(|_| "{}".to_owned())
}

fn expected_schema_value_hint(schema: Option<&serde_json::Value>) -> serde_json::Value {
    let Some(schema_object) = schema.and_then(serde_json::Value::as_object) else {
        return serde_json::Value::String("<value>".to_owned());
    };
    if let Some(enum_values) = schema_object
        .get("enum")
        .and_then(serde_json::Value::as_array)
        && let Some(first) = enum_values.first()
    {
        return first.clone();
    }
    let placeholder = schema_object
        .get("type")
        .and_then(first_schema_type_name)
        .map(|type_name| match type_name {
            "array" => "<array>",
            "boolean" => "<boolean>",
            "integer" => "<integer>",
            "null" => "<null>",
            "number" => "<number>",
            "object" => "<object>",
            "string" => "<string>",
            _ => "<value>",
        })
        .unwrap_or("<value>");
    serde_json::Value::String(placeholder.to_owned())
}

fn first_schema_type_name(schema_type: &serde_json::Value) -> Option<&str> {
    match schema_type {
        serde_json::Value::String(type_name) => Some(type_name),
        serde_json::Value::Array(types) => types.iter().find_map(serde_json::Value::as_str),
        _ => None,
    }
}

fn schema_type_accepts_string(schema_type: &serde_json::Value) -> bool {
    match schema_type {
        serde_json::Value::String(type_name) => type_name == "string",
        serde_json::Value::Array(types) => types
            .iter()
            .any(|type_name| type_name.as_str() == Some("string")),
        _ => false,
    }
}

fn schema_type_matches(schema_type: &serde_json::Value, value: &serde_json::Value) -> bool {
    match schema_type {
        serde_json::Value::String(type_name) => json_type_matches(type_name, value),
        serde_json::Value::Array(types) => types
            .iter()
            .filter_map(serde_json::Value::as_str)
            .any(|type_name| json_type_matches(type_name, value)),
        _ => true,
    }
}

fn json_type_matches(type_name: &str, value: &serde_json::Value) -> bool {
    match type_name {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "number" => value.is_number(),
        _ => true,
    }
}

fn display_schema_type(schema_type: &serde_json::Value) -> String {
    match schema_type {
        serde_json::Value::String(type_name) => format!("`{type_name}`"),
        serde_json::Value::Array(types) => {
            let names = types
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(|type_name| format!("`{type_name}`"))
                .collect::<Vec<_>>();
            format!("[{}]", names.join(", "))
        }
        other => other.to_string(),
    }
}

fn display_schema_path(path: &str) -> String {
    if path.is_empty() {
        "$".to_owned()
    } else {
        path.to_owned()
    }
}

fn join_schema_path(parent: &str, field: &str) -> String {
    if parent.is_empty() {
        field.to_owned()
    } else {
        format!("{parent}.{field}")
    }
}
