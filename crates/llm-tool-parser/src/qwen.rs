use crate::{
    ParsedAssistant, ParserError,
    common::{parse_tool_calls, split_reasoning},
};

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
            content: rest,
            tool_calls: Vec::new(),
        })
    }
}
