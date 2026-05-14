use crate::{
    ParsedAssistant, ParserError,
    common::{parse_tool_calls, split_reasoning},
};
use llm_api::{ToolCall, ToolCallFunction, ToolCallType};
use serde_json::Value;

#[derive(Debug, Default, Clone)]
pub struct DeepSeekParser;

impl DeepSeekParser {
    pub fn parse_complete(&self, text: &str) -> Result<ParsedAssistant, ParserError> {
        let (reasoning, rest) = split_reasoning(text)?;
        let rest = trim_deepseek_after_stop_control(strip_deepseek_assistant_prefix(&rest));
        if rest.contains("<｜tool▁calls▁begin｜>") {
            return parse_deepseek_native_tool_calls(reasoning, rest);
        }
        if rest.contains("<dsml_tool_call>") {
            return parse_deepseek_dsml_tool_calls(reasoning, rest);
        }
        if rest.contains("<tool_call>") {
            return parse_tool_calls(reasoning, rest);
        }
        Ok(ParsedAssistant {
            reasoning,
            content: rest.to_owned(),
            tool_calls: Vec::new(),
        })
    }
}

fn parse_deepseek_dsml_tool_calls(
    reasoning: Option<String>,
    mut rest: &str,
) -> Result<ParsedAssistant, ParserError> {
    let mut calls = Vec::new();
    let mut content = String::new();

    while let Some(start) = rest.find("<dsml_tool_call>") {
        content.push_str(&rest[..start]);
        let inner_start = start + "<dsml_tool_call>".len();
        let Some(end_rel) = rest[inner_start..].find("</dsml_tool_call>") else {
            return Err(ParserError::malformed_tool(
                "unterminated DeepSeek dsml_tool_call tag",
            ));
        };
        let inner_end = inner_start + end_rel;
        let inner = rest[inner_start..inner_end].trim();
        calls.push(parse_deepseek_json_call(inner, calls.len())?);
        rest = &rest[inner_end + "</dsml_tool_call>".len()..];
    }
    content.push_str(rest);

    Ok(ParsedAssistant {
        reasoning,
        content,
        tool_calls: calls,
    })
}

fn parse_deepseek_native_tool_calls(
    reasoning: Option<String>,
    rest: &str,
) -> Result<ParsedAssistant, ParserError> {
    let Some(start) = rest.find("<｜tool▁calls▁begin｜>") else {
        return Ok(ParsedAssistant {
            reasoning,
            content: rest.to_owned(),
            tool_calls: Vec::new(),
        });
    };
    let mut content = rest[..start].to_owned();
    let calls_start = start + "<｜tool▁calls▁begin｜>".len();
    let Some(end_rel) = rest[calls_start..].find("<｜tool▁calls▁end｜>") else {
        return Err(ParserError::malformed_tool(
            "unterminated DeepSeek native tool calls block",
        ));
    };
    let calls_end = calls_start + end_rel;
    let mut calls_body = &rest[calls_start..calls_end];
    let mut calls = Vec::new();
    while let Some(call_start) = calls_body.find("<｜tool▁call▁begin｜>") {
        let inner_start = call_start + "<｜tool▁call▁begin｜>".len();
        let Some(call_end_rel) = calls_body[inner_start..].find("<｜tool▁call▁end｜>") else {
            return Err(ParserError::malformed_tool(
                "unterminated DeepSeek native tool call",
            ));
        };
        let inner_end = inner_start + call_end_rel;
        let inner = calls_body[inner_start..inner_end].trim();
        calls.push(parse_deepseek_native_call(inner, calls.len())?);
        calls_body = &calls_body[inner_end + "<｜tool▁call▁end｜>".len()..];
    }
    content.push_str(
        trim_deepseek_after_stop_control(
            rest[calls_end + "<｜tool▁calls▁end｜>".len()..].trim_start(),
        )
        .trim_start_matches("<｜end▁of▁sentence｜>"),
    );
    Ok(ParsedAssistant {
        reasoning,
        content,
        tool_calls: calls,
    })
}

fn parse_deepseek_native_call(inner: &str, index: usize) -> Result<ToolCall, ParserError> {
    let Some((call_type, rest)) = inner.split_once("<｜tool▁sep｜>") else {
        return Err(ParserError::malformed_tool(
            "DeepSeek native tool call missing separator",
        ));
    };
    if call_type.trim() != "function" {
        return Err(ParserError::malformed_tool(format!(
            "unsupported DeepSeek native tool call type `{}`",
            call_type.trim()
        )));
    }
    let Some((name, arguments_block)) = rest.split_once('\n') else {
        return Err(ParserError::malformed_tool(
            "DeepSeek native tool call missing arguments block",
        ));
    };
    let name = name.trim();
    if name.is_empty() {
        return Err(ParserError::malformed_tool(
            "DeepSeek native tool call missing name",
        ));
    }
    let arguments = parse_deepseek_markdown_json(arguments_block.trim())?;
    Ok(ToolCall {
        id: format!("call_{index}"),
        call_type: ToolCallType::Function,
        function: ToolCallFunction {
            name: name.to_owned(),
            arguments,
        },
    })
}

fn parse_deepseek_markdown_json(arguments_block: &str) -> Result<Value, ParserError> {
    let trimmed = arguments_block.trim();
    let json = trimmed
        .strip_prefix("```json")
        .and_then(|body| body.strip_suffix("```"))
        .or_else(|| {
            trimmed
                .strip_prefix("```")
                .and_then(|body| body.strip_suffix("```"))
        })
        .unwrap_or(trimmed)
        .trim();
    serde_json::from_str(json).map_err(|err| {
        ParserError::malformed_tool(format!("invalid DeepSeek native tool JSON: {err}"))
    })
}

fn parse_deepseek_json_call(inner: &str, index: usize) -> Result<ToolCall, ParserError> {
    let value: Value = serde_json::from_str(inner)
        .map_err(|err| ParserError::malformed_tool(format!("invalid DeepSeek tool JSON: {err}")))?;
    let name = value
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| ParserError::malformed_tool("DeepSeek tool JSON missing name"))?;
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

fn strip_deepseek_assistant_prefix(rest: &str) -> &str {
    rest.strip_prefix("<｜Assistant｜>").unwrap_or(rest)
}

fn trim_deepseek_after_stop_control(rest: &str) -> &str {
    ["<｜end▁of▁sentence｜>", "<｜User｜>"]
        .iter()
        .filter_map(|token| rest.find(token))
        .min()
        .map_or(rest, |index| &rest[..index])
}
