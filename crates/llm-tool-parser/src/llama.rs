use crate::{
    ParsedAssistant, ParserError,
    common::{parse_tool_calls, split_reasoning},
    json::parse_json_tool_output_if_tool_like,
};

#[derive(Debug, Default, Clone)]
pub struct LlamaParser;

impl LlamaParser {
    pub fn parse_complete(&self, text: &str) -> Result<ParsedAssistant, ParserError> {
        let (reasoning, rest) = split_reasoning(text)?;
        let rest = trim_llama_after_stop_control(&rest);
        if rest.contains("<tool_call>") {
            return parse_tool_calls(reasoning, rest);
        }
        parse_json_tool_output_if_tool_like(reasoning, rest)
    }
}

fn trim_llama_after_stop_control(text: &str) -> &str {
    ["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>"]
        .iter()
        .filter_map(|token| text.find(token))
        .min()
        .map_or(text, |index| &text[..index])
}
