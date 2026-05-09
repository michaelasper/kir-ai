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

#[derive(Debug, Default, Clone)]
pub struct GemmaParser;

impl QwenParser {
    pub fn parse_complete(&self, text: &str) -> Result<ParsedAssistant, ParserError> {
        let (reasoning, rest) = split_reasoning(text)?;
        if rest.contains("<tool_call>") {
            return parse_tool_calls(reasoning, &rest);
        }
        Ok(ParsedAssistant {
            reasoning,
            content: rest,
            tool_calls: Vec::new(),
        })
    }
}

impl GemmaParser {
    pub fn parse_complete(&self, text: &str) -> Result<ParsedAssistant, ParserError> {
        reject_gemma_multimodal_markers(text)?;
        let text = trim_gemma_after_stop_control(text);
        let (reasoning, rest) = split_gemma_reasoning(text)?;
        let (content, tool_calls) = parse_gemma_tool_calls(&rest)?;
        Ok(ParsedAssistant {
            reasoning,
            content,
            tool_calls,
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

    fn unsupported_multimodal(message: impl Into<String>) -> Self {
        Self {
            code: "unsupported_multimodal_output",
            message: message.into(),
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
        ModelFamily::Gemma => GemmaParser.parse_complete(text),
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
    rest.push_str(&text[..start]);
    rest.push_str(&text[end + "</think>".len()..]);
    Ok((Some(reasoning), rest))
}

fn parse_tool_calls(
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

fn reject_gemma_multimodal_markers(text: &str) -> Result<(), ParserError> {
    const UNSUPPORTED: [&str; 6] = [
        "<start_of_image>",
        "<|image|>",
        "<image|>",
        "<|audio|>",
        "<audio|>",
        "<|video|>",
    ];
    if let Some(token) = UNSUPPORTED.iter().find(|token| text.contains(**token)) {
        return Err(ParserError::unsupported_multimodal(format!(
            "Gemma multimodal output marker `{token}` is not supported by the text parser"
        )));
    }
    Ok(())
}

fn trim_gemma_after_stop_control(text: &str) -> &str {
    ["<turn|>", "<|tool_response>", "<eos>"]
        .iter()
        .filter_map(|token| text.find(token))
        .min()
        .map_or(text, |index| &text[..index])
}

fn split_gemma_reasoning(text: &str) -> Result<(Option<String>, String), ParserError> {
    let Some(start) = text.find("<|channel>thought\n") else {
        return Ok((None, text.to_owned()));
    };
    let body_start = start + "<|channel>thought\n".len();
    let Some(end_rel) = text[body_start..].find("<channel|>") else {
        return Err(ParserError::malformed_tool(
            "unterminated Gemma thought channel",
        ));
    };
    let end = body_start + end_rel;
    let reasoning = text[body_start..end].trim().to_owned();
    let mut rest = String::new();
    rest.push_str(&text[..start]);
    rest.push_str(&text[end + "<channel|>".len()..]);
    Ok((Some(reasoning), rest))
}

fn parse_gemma_tool_calls(rest: &str) -> Result<(String, Vec<ToolCall>), ParserError> {
    let mut calls = Vec::new();
    let mut content = String::new();
    let mut rest = rest;
    while let Some(start) = rest.find("<|tool_call>") {
        content.push_str(&rest[..start]);
        let inner_start = start + "<|tool_call>".len();
        let Some(end_rel) = rest[inner_start..].find("<tool_call|>") else {
            return Err(ParserError::malformed_tool(
                "unterminated Gemma tool_call tag",
            ));
        };
        let inner_end = inner_start + end_rel;
        let inner = rest[inner_start..inner_end].trim();
        calls.push(parse_gemma_call(inner, calls.len())?);
        rest = &rest[inner_end + "<tool_call|>".len()..];
    }
    content.push_str(rest);
    Ok((content, calls))
}

fn parse_gemma_call(inner: &str, index: usize) -> Result<ToolCall, ParserError> {
    let Some(body) = inner.strip_prefix("call:") else {
        return Err(ParserError::malformed_tool("missing Gemma call prefix"));
    };
    let Some(args_start) = body.find('{') else {
        return Err(ParserError::malformed_tool(
            "Gemma tool call missing arguments",
        ));
    };
    let name = body[..args_start].trim();
    if name.is_empty() {
        return Err(ParserError::malformed_tool("Gemma tool call missing name"));
    }
    let arguments = GemmaArgumentParser::new(&body[args_start..]).parse_complete()?;
    Ok(ToolCall {
        id: format!("call_{index}"),
        call_type: ToolCallType::Function,
        function: ToolCallFunction {
            name: name.to_owned(),
            arguments,
        },
    })
}

struct GemmaArgumentParser<'a> {
    input: &'a str,
    position: usize,
}

impl<'a> GemmaArgumentParser<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, position: 0 }
    }

    fn parse_complete(mut self) -> Result<Value, ParserError> {
        let value = self.parse_value()?;
        self.skip_ws();
        if self.position != self.input.len() {
            return Err(ParserError::malformed_tool(format!(
                "unexpected Gemma tool argument suffix `{}`",
                &self.input[self.position..]
            )));
        }
        Ok(value)
    }

    fn parse_value(&mut self) -> Result<Value, ParserError> {
        self.skip_ws();
        match self.peek_char() {
            Some('{') => self.parse_object(),
            Some('[') => self.parse_array(),
            Some('"') => self.parse_json_string().map(Value::String),
            Some('<') if self.remaining().starts_with("<|\"|>") => {
                self.parse_gemma_string().map(Value::String)
            }
            Some(_) => self.parse_atom(),
            None => Err(ParserError::malformed_tool(
                "Gemma tool argument ended before value",
            )),
        }
    }

    fn parse_object(&mut self) -> Result<Value, ParserError> {
        self.expect_char('{')?;
        let mut map = serde_json::Map::new();
        loop {
            self.skip_ws();
            if self.consume_char('}') {
                break;
            }
            let key = self.parse_key()?;
            self.skip_ws();
            self.expect_char(':')?;
            let value = self.parse_value()?;
            map.insert(key, value);
            self.skip_ws();
            if self.consume_char(',') {
                continue;
            }
            self.expect_char('}')?;
            break;
        }
        Ok(Value::Object(map))
    }

    fn parse_array(&mut self) -> Result<Value, ParserError> {
        self.expect_char('[')?;
        let mut values = Vec::new();
        loop {
            self.skip_ws();
            if self.consume_char(']') {
                break;
            }
            values.push(self.parse_value()?);
            self.skip_ws();
            if self.consume_char(',') {
                continue;
            }
            self.expect_char(']')?;
            break;
        }
        Ok(Value::Array(values))
    }

    fn parse_key(&mut self) -> Result<String, ParserError> {
        self.skip_ws();
        match self.peek_char() {
            Some('"') => self.parse_json_string(),
            Some('<') if self.remaining().starts_with("<|\"|>") => self.parse_gemma_string(),
            Some(_) => {
                let start = self.position;
                while let Some(ch) = self.peek_char() {
                    if ch == ':' || ch.is_whitespace() {
                        break;
                    }
                    self.position += ch.len_utf8();
                }
                if self.position == start {
                    return Err(ParserError::malformed_tool("Gemma object key is empty"));
                }
                Ok(self.input[start..self.position].to_owned())
            }
            None => Err(ParserError::malformed_tool("Gemma object ended before key")),
        }
    }

    fn parse_atom(&mut self) -> Result<Value, ParserError> {
        let start = self.position;
        while let Some(ch) = self.peek_char() {
            if ch == ',' || ch == '}' || ch == ']' || ch.is_whitespace() {
                break;
            }
            self.position += ch.len_utf8();
        }
        let atom = &self.input[start..self.position];
        if atom.is_empty() {
            return Err(ParserError::malformed_tool("Gemma atom is empty"));
        }
        match atom {
            "true" => Ok(Value::Bool(true)),
            "false" => Ok(Value::Bool(false)),
            "null" => Ok(Value::Null),
            _ => {
                serde_json::from_str::<Value>(atom).or_else(|_| Ok(Value::String(atom.to_owned())))
            }
        }
    }

    fn parse_json_string(&mut self) -> Result<String, ParserError> {
        let start = self.position;
        self.expect_char('"')?;
        let mut escaped = false;
        while let Some(ch) = self.peek_char() {
            self.position += ch.len_utf8();
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                return serde_json::from_str::<String>(&self.input[start..self.position]).map_err(
                    |err| ParserError::malformed_tool(format!("invalid Gemma JSON string: {err}")),
                );
            }
        }
        Err(ParserError::malformed_tool(
            "unterminated Gemma JSON string",
        ))
    }

    fn parse_gemma_string(&mut self) -> Result<String, ParserError> {
        self.expect_str("<|\"|>")?;
        let start = self.position;
        let Some(end_rel) = self.remaining().find("<|\"|>") else {
            return Err(ParserError::malformed_tool(
                "unterminated Gemma escaped string",
            ));
        };
        let end = self.position + end_rel;
        self.position = end + "<|\"|>".len();
        Ok(self.input[start..end].to_owned())
    }

    fn skip_ws(&mut self) {
        while let Some(ch) = self.peek_char() {
            if !ch.is_whitespace() {
                break;
            }
            self.position += ch.len_utf8();
        }
    }

    fn consume_char(&mut self, expected: char) -> bool {
        if self.peek_char() == Some(expected) {
            self.position += expected.len_utf8();
            true
        } else {
            false
        }
    }

    fn expect_char(&mut self, expected: char) -> Result<(), ParserError> {
        if self.consume_char(expected) {
            Ok(())
        } else {
            Err(ParserError::malformed_tool(format!(
                "expected Gemma tool argument character `{expected}`"
            )))
        }
    }

    fn expect_str(&mut self, expected: &str) -> Result<(), ParserError> {
        if self.remaining().starts_with(expected) {
            self.position += expected.len();
            Ok(())
        } else {
            Err(ParserError::malformed_tool(format!(
                "expected Gemma tool argument token `{expected}`"
            )))
        }
    }

    fn peek_char(&self) -> Option<char> {
        self.remaining().chars().next()
    }

    fn remaining(&self) -> &'a str {
        &self.input[self.position..]
    }
}
