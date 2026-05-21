use crate::{ParsedAssistant, ParserError};
use llm_api::{ToolCall, ToolCallFunction, ToolCallType, generated_tool_call_id};
use serde_json::Value;

pub(crate) fn split_reasoning(text: &str) -> Result<(Option<String>, String), ParserError> {
    let Some(start) = text.find("<think>") else {
        return Ok((None, text.to_owned()));
    };
    let body_start = start + "<think>".len();
    let Some(end_rel) = text[body_start..].find("</think>") else {
        return Ok((
            Some(text[body_start..].to_owned()),
            text[..start].to_owned(),
        ));
    };
    let end = body_start + end_rel;
    let reasoning = text[body_start..end].to_owned();
    let mut rest = String::new();
    rest.push_str(&text[..start]);
    rest.push_str(&text[end + "</think>".len()..]);
    Ok((Some(reasoning), rest))
}

pub(crate) fn parse_tool_calls(
    reasoning: Option<String>,
    mut rest: &str,
) -> Result<ParsedAssistant, ParserError> {
    let mut calls = Vec::new();
    let mut content = String::new();

    while let Some(start) = rest.find("<tool_call>") {
        content.push_str(&rest[..start]);
        let inner_start = start + "<tool_call>".len();
        let Some(end_rel) = rest[inner_start..].find("</tool_call>") else {
            return Err(ParserError::malformed_tool(
                "unterminated qwen tool_call tag",
            ));
        };
        let inner_end = inner_start + end_rel;
        let inner = rest[inner_start..inner_end].trim();
        let call = if inner.starts_with("<function=") {
            parse_xml_call(inner, calls.len())?
        } else {
            parse_json_call(inner, calls.len())?
        };
        calls.push(call);
        rest = &rest[inner_end + "</tool_call>".len()..];
    }
    content.push_str(rest);

    Ok(ParsedAssistant {
        reasoning,
        content,
        tool_calls: calls,
    })
}

fn parse_json_call(inner: &str, _index: usize) -> Result<ToolCall, ParserError> {
    let value: Value = serde_json::from_str(inner)
        .map_err(|err| ParserError::malformed_tool(format!("invalid qwen tool JSON: {err}")))?;
    let name = value
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| ParserError::malformed_tool("qwen tool JSON missing name"))?;
    let id = value
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(generated_tool_call_id);
    let arguments = value
        .get("arguments")
        .or_else(|| value.get("parameters"))
        .or_else(|| value.get("args"))
        .map(parse_json_tool_arguments)
        .transpose()?
        .unwrap_or_else(|| serde_json::json!({}));
    Ok(ToolCall {
        id,
        call_type: ToolCallType::Function,
        function: ToolCallFunction {
            name: name.to_owned(),
            arguments,
        },
    })
}

fn parse_xml_call(inner: &str, _index: usize) -> Result<ToolCall, ParserError> {
    let Some(name_start) = inner.find("<function=") else {
        return Err(ParserError::malformed_tool("missing qwen function tag"));
    };
    let name_body_start = name_start + "<function=".len();
    let Some(name_end_rel) = inner[name_body_start..].find('>') else {
        return Err(ParserError::malformed_tool(
            "unterminated qwen function tag",
        ));
    };
    let name_end = name_body_start + name_end_rel;
    let name = &inner[name_body_start..name_end];
    let Some(function_end) = inner.find("</function>") else {
        return Err(ParserError::malformed_tool(
            "missing qwen function close tag",
        ));
    };
    let params = &inner[name_end + 1..function_end];
    let mut map = serde_json::Map::new();
    let mut rest = params;
    while let Some(start) = rest.find("<parameter=") {
        let key_start = start + "<parameter=".len();
        let Some(key_end_rel) = rest[key_start..].find('>') else {
            return Err(ParserError::malformed_tool(
                "unterminated qwen parameter tag",
            ));
        };
        let key_end = key_start + key_end_rel;
        let key = &rest[key_start..key_end];
        let value_start = key_end + 1;
        let Some(value_end_rel) = rest[value_start..].find("</parameter>") else {
            return Err(ParserError::malformed_tool(
                "missing qwen parameter close tag",
            ));
        };
        let value_end = value_start + value_end_rel;
        map.insert(
            key.to_owned(),
            Value::String(rest[value_start..value_end].to_owned()),
        );
        rest = &rest[value_end + "</parameter>".len()..];
    }
    Ok(ToolCall {
        id: generated_tool_call_id(),
        call_type: ToolCallType::Function,
        function: ToolCallFunction {
            name: name.to_owned(),
            arguments: Value::Object(map),
        },
    })
}

pub(crate) fn parse_json_tool_arguments(value: &Value) -> Result<Value, ParserError> {
    match value {
        Value::String(arguments) => {
            let trimmed = arguments.trim();
            if trimmed.is_empty() {
                Ok(serde_json::json!({}))
            } else if trimmed.starts_with('{') || trimmed.starts_with('[') {
                serde_json::from_str(trimmed).map_err(|err| {
                    ParserError::malformed_tool(format!("invalid JSON tool-call arguments: {err}"))
                })
            } else {
                Ok(Value::String(arguments.clone()))
            }
        }
        other => Ok(other.clone()),
    }
}
