use crate::{
    ParsedAssistant, ParserError,
    common::{parse_json_tool_arguments, parse_tool_calls, split_reasoning},
};
use llm_api::{ToolCall, ToolCallFunction, ToolCallType};
use serde_json::Value;

pub(crate) fn parse_json_tool_output(text: &str) -> Result<ParsedAssistant, ParserError> {
    let (reasoning, rest) = split_reasoning(text)?;
    parse_json_tool_output_with_reasoning(reasoning, &rest)
}

pub(crate) fn parse_xlam_tool_output(text: &str) -> Result<ParsedAssistant, ParserError> {
    let (reasoning, rest) = split_reasoning(text)?;
    if rest.contains("<tool_call>") {
        return parse_tool_calls(reasoning, &rest);
    }
    parse_json_tool_output_with_reasoning(reasoning, &rest)
}

fn parse_json_tool_output_with_reasoning(
    reasoning: Option<String>,
    rest: &str,
) -> Result<ParsedAssistant, ParserError> {
    let Some((content, source)) = extract_json_tool_candidate(rest) else {
        return Ok(ParsedAssistant {
            reasoning,
            content: rest.to_owned(),
            tool_calls: Vec::new(),
        });
    };
    let value: Value = serde_json::from_str(source.trim()).map_err(|err| {
        ParserError::malformed_tool(format!("invalid JSON tool-call payload: {err}"))
    })?;
    let tool_calls = parse_json_tool_value(&value)?;
    Ok(ParsedAssistant {
        reasoning,
        content,
        tool_calls,
    })
}

pub(crate) fn parse_json_tool_output_if_tool_like(
    reasoning: Option<String>,
    rest: &str,
) -> Result<ParsedAssistant, ParserError> {
    let Some((content, source)) = extract_json_tool_candidate(rest) else {
        return Ok(ParsedAssistant {
            reasoning,
            content: rest.to_owned(),
            tool_calls: Vec::new(),
        });
    };
    let Ok(value) = serde_json::from_str::<Value>(source.trim()) else {
        return Ok(ParsedAssistant {
            reasoning,
            content: rest.to_owned(),
            tool_calls: Vec::new(),
        });
    };
    if !json_value_has_tool_shape(&value) {
        return Ok(ParsedAssistant {
            reasoning,
            content: rest.to_owned(),
            tool_calls: Vec::new(),
        });
    }
    let tool_calls = parse_json_tool_value(&value)?;
    Ok(ParsedAssistant {
        reasoning,
        content,
        tool_calls,
    })
}

fn extract_json_tool_candidate(rest: &str) -> Option<(String, &str)> {
    if let Some(marker_start) = rest.find("[TOOL_CALLS]") {
        let source_start = marker_start + "[TOOL_CALLS]".len();
        let source = rest[source_start..].trim_start();
        let source_end = source.find('\n').unwrap_or(source.len());
        let mut content = String::new();
        content.push_str(rest[..marker_start].trim_end());
        let suffix = source[source_end..].trim_start_matches('\n');
        if !content.is_empty() && !suffix.is_empty() {
            content.push('\n');
        }
        content.push_str(suffix);
        return Some((content, &source[..source_end]));
    }

    if let Some(fence_start) = rest.find("```") {
        let language_start = fence_start + "```".len();
        let after_language = rest[language_start..]
            .strip_prefix("json")
            .unwrap_or(&rest[language_start..]);
        let after_newline = after_language
            .strip_prefix('\n')
            .unwrap_or(after_language.trim_start());
        if let Some(fence_end_rel) = after_newline.find("```") {
            let body_start = rest.len() - after_newline.len();
            let fence_end = body_start + fence_end_rel + "```".len();
            let mut content = String::new();
            content.push_str(rest[..fence_start].trim_end());
            content.push_str(rest[fence_end..].trim_start());
            return Some((content, &after_newline[..fence_end_rel]));
        }
    }

    let trimmed = rest.trim();
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        return Some((String::new(), trimmed));
    }
    None
}

fn parse_json_tool_value(value: &Value) -> Result<Vec<ToolCall>, ParserError> {
    match value {
        Value::Array(items) => items
            .iter()
            .enumerate()
            .map(|(index, value)| parse_json_tool_value_as_call(value, index))
            .collect(),
        Value::Object(object) => {
            for key in ["tool_calls", "calls", "tools"] {
                if let Some(Value::Array(items)) = object.get(key) {
                    return items
                        .iter()
                        .enumerate()
                        .map(|(index, value)| parse_json_tool_value_as_call(value, index))
                        .collect();
                }
            }
            Ok(vec![parse_json_tool_value_as_call(value, 0)?])
        }
        _ => Err(ParserError::malformed_tool(
            "JSON tool-call payload must be an object or array",
        )),
    }
}

fn json_value_has_tool_shape(value: &Value) -> bool {
    match value {
        Value::Array(items) => {
            !items.is_empty() && items.iter().all(json_value_has_direct_tool_shape)
        }
        Value::Object(object) => {
            ["tool_calls", "calls", "tools"]
                .iter()
                .any(|key| matches!(object.get(*key), Some(Value::Array(_))))
                || json_object_has_direct_tool_shape(object)
        }
        _ => false,
    }
}

fn json_value_has_direct_tool_shape(value: &Value) -> bool {
    let Value::Object(object) = value else {
        return false;
    };
    json_object_has_direct_tool_shape(object)
}

fn json_object_has_direct_tool_shape(object: &serde_json::Map<String, Value>) -> bool {
    let has_name = object
        .get("function")
        .and_then(Value::as_object)
        .and_then(|function| function.get("name"))
        .and_then(Value::as_str)
        .is_some()
        || object.get("function").and_then(Value::as_str).is_some()
        || object.get("name").and_then(Value::as_str).is_some()
        || object.get("tool_name").and_then(Value::as_str).is_some();
    let has_arguments = object
        .get("function")
        .and_then(Value::as_object)
        .and_then(|function| function.get("arguments"))
        .is_some()
        || object.contains_key("arguments")
        || object.contains_key("parameters")
        || object.contains_key("args");
    has_name && has_arguments
}

fn parse_json_tool_value_as_call(value: &Value, index: usize) -> Result<ToolCall, ParserError> {
    let Value::Object(object) = value else {
        return Err(ParserError::malformed_tool(
            "JSON tool-call entry must be an object",
        ));
    };
    let function = object.get("function").and_then(Value::as_object);
    let name = function
        .and_then(|function| function.get("name"))
        .or_else(|| object.get("name"))
        .or_else(|| object.get("tool_name"))
        .or_else(|| object.get("function"))
        .and_then(Value::as_str)
        .ok_or_else(|| ParserError::malformed_tool("JSON tool-call entry missing name"))?;
    let arguments = function
        .and_then(|function| function.get("arguments"))
        .or_else(|| object.get("arguments"))
        .or_else(|| object.get("parameters"))
        .or_else(|| object.get("args"))
        .map(parse_json_tool_arguments)
        .transpose()?
        .unwrap_or_else(|| serde_json::json!({}));
    Ok(ToolCall {
        id: format!("call_{index}"),
        call_type: ToolCallType::Function,
        function: ToolCallFunction {
            name: name.to_owned(),
            arguments,
        },
    })
}
