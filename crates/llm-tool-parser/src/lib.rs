use llm_api::{ToolCall, ToolCallFunction, ToolCallType};
use llm_models::ModelFamily;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParsedAssistant {
    pub reasoning: Option<String>,
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
}

impl ParsedAssistant {
    pub fn content(content: impl Into<String>) -> Self {
        Self {
            reasoning: None,
            content: content.into(),
            tool_calls: Vec::new(),
        }
    }

    pub fn single_tool(name: impl Into<String>, arguments: Value) -> Self {
        Self {
            reasoning: None,
            content: String::new(),
            tool_calls: vec![ToolCall {
                id: "call_0".to_owned(),
                call_type: ToolCallType::Function,
                function: ToolCallFunction {
                    name: name.into(),
                    arguments,
                },
            }],
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct QwenParser;

impl QwenParser {
    pub fn parse_complete(&self, text: &str) -> Result<ParsedAssistant, ParserError> {
        let (reasoning, rest) = split_reasoning(text)?;
        if rest.contains("<tool_call>") {
            return parse_tool_calls(reasoning, &rest);
        }
        Ok(ParsedAssistant {
            reasoning,
            content: rest.trim().to_owned(),
            tool_calls: Vec::new(),
        })
    }
}

#[derive(Debug, Error)]
#[error("{code}: {message}")]
pub struct ParserError {
    code: &'static str,
    message: String,
}

impl ParserError {
    pub fn code(&self) -> &'static str {
        self.code
    }

    fn malformed_tool(message: impl Into<String>) -> Self {
        Self {
            code: "malformed_tool_call",
            message: message.into(),
        }
    }

    fn unsupported_family(family: &'static str) -> Self {
        Self {
            code: "unsupported_parser_family",
            message: format!("{family} parser support is deferred until Qwen production parity"),
        }
    }
}

pub fn parse_assistant_for_family(
    family: ModelFamily,
    text: &str,
) -> Result<ParsedAssistant, ParserError> {
    match family {
        ModelFamily::Qwen => QwenParser.parse_complete(text),
        ModelFamily::DeepSeek => Err(ParserError::unsupported_family("DeepSeek")),
        ModelFamily::Gemma => Err(ParserError::unsupported_family("Gemma")),
    }
}

fn split_reasoning(text: &str) -> Result<(Option<String>, String), ParserError> {
    let Some(start) = text.find("<think>") else {
        return Ok((None, text.to_owned()));
    };
    let body_start = start + "<think>".len();
    let Some(end_rel) = text[body_start..].find("</think>") else {
        return Err(ParserError::malformed_tool(
            "unterminated qwen reasoning tag",
        ));
    };
    let end = body_start + end_rel;
    let reasoning = text[body_start..end].to_owned();
    let mut rest = String::new();
    rest.push_str(text[..start].trim());
    rest.push_str(text[end + "</think>".len()..].trim());
    Ok((Some(reasoning), rest))
}

fn parse_tool_calls(
    reasoning: Option<String>,
    mut rest: &str,
) -> Result<ParsedAssistant, ParserError> {
    let mut calls = Vec::new();
    let mut content = String::new();

    while let Some(start) = rest.find("<tool_call>") {
        content.push_str(rest[..start].trim());
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
    content.push_str(rest.trim());

    Ok(ParsedAssistant {
        reasoning,
        content,
        tool_calls: calls,
    })
}

fn parse_json_call(inner: &str, index: usize) -> Result<ToolCall, ParserError> {
    let value: Value = serde_json::from_str(inner)
        .map_err(|err| ParserError::malformed_tool(format!("invalid qwen tool JSON: {err}")))?;
    let name = value
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| ParserError::malformed_tool("qwen tool JSON missing name"))?;
    let arguments = value
        .get("arguments")
        .cloned()
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

fn parse_xml_call(inner: &str, index: usize) -> Result<ToolCall, ParserError> {
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
        id: format!("call_{index}"),
        call_type: ToolCallType::Function,
        function: ToolCallFunction {
            name: name.to_owned(),
            arguments: Value::Object(map),
        },
    })
}
