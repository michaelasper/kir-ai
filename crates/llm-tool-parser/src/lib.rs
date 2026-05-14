mod common;
mod deepseek;
mod gemma;
mod json;
mod llama;
mod qwen;
mod types;

pub use deepseek::DeepSeekParser;
pub use gemma::GemmaParser;
pub use llama::LlamaParser;
pub use qwen::QwenParser;
pub use types::{ParsedAssistant, ParserError, ToolParserFamily};

use llm_models::ModelFamily;

pub fn parse_assistant_for_family(
    family: ModelFamily,
    text: &str,
) -> Result<ParsedAssistant, ParserError> {
    match family {
        ModelFamily::Qwen => QwenParser.parse_complete(text),
        ModelFamily::DeepSeek => DeepSeekParser.parse_complete(text),
        ModelFamily::Gemma => GemmaParser.parse_complete(text),
        ModelFamily::Llama => LlamaParser.parse_complete(text),
    }
}

pub fn parse_assistant_for_parser_family(
    family: ToolParserFamily,
    text: &str,
) -> Result<ParsedAssistant, ParserError> {
    match family {
        ToolParserFamily::Auto => parse_assistant_auto(text),
        ToolParserFamily::Hermes | ToolParserFamily::Qwen => QwenParser.parse_complete(text),
        ToolParserFamily::DeepSeek => DeepSeekParser.parse_complete(text),
        ToolParserFamily::Gemma => GemmaParser.parse_complete(text),
        ToolParserFamily::Llama => LlamaParser.parse_complete(text),
        ToolParserFamily::Json => json::parse_json_tool_output(text),
        ToolParserFamily::Xlam => json::parse_xlam_tool_output(text),
    }
}

fn parse_assistant_auto(text: &str) -> Result<ParsedAssistant, ParserError> {
    if text.contains("<|tool_call>") || text.contains("<|channel>thought\n") {
        return GemmaParser.parse_complete(text);
    }
    if text.contains("<dsml_tool_call>") || text.contains("<｜tool▁calls▁begin｜>") {
        return DeepSeekParser.parse_complete(text);
    }
    if text.contains("<tool_call>") || text.contains("<think>") {
        return QwenParser.parse_complete(text);
    }
    if text.contains("[TOOL_CALLS]") {
        return json::parse_xlam_tool_output(text);
    }
    Ok(ParsedAssistant::content(text))
}
