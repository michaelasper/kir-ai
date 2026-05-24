use crate::{
    ParsedAssistant, ParserError,
    common::{parse_tool_calls, split_reasoning},
};

/// Parser for Qwen/Hermes-style assistant output with `<think>` and `<tool_call>` tags.
#[derive(Debug, Default, Clone)]
pub struct QwenParser;

impl QwenParser {
    /// Parses one complete assistant output into reasoning, content, and tool calls.
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
