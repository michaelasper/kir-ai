//! Family-specific assistant output parsers.
//!
//! The runtime calls this crate after backend generation and before emitting a
//! successful assistant message. Parsers split hidden reasoning, user-visible
//! content, and function tool calls for model families whose prompt templates
//! encode tools differently.

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

/// Parses a complete assistant output using the parser implied by a model family.
pub fn parse_assistant_for_family(
    family: ModelFamily,
    text: &str,
) -> Result<ParsedAssistant, ParserError> {
    let parsed = match family {
        ModelFamily::Qwen => QwenParser.parse_complete(text),
        ModelFamily::DeepSeek => DeepSeekParser.parse_complete(text),
        ModelFamily::Gemma => GemmaParser.parse_complete(text),
        ModelFamily::Llama => LlamaParser.parse_complete(text),
        _ => Err(ParserError::unsupported_model_family(
            family.canonical_slug(),
        )),
    }?;
    tracing::trace!(
        operation = "parse_assistant",
        model_family = family.canonical_slug(),
        input_bytes = text.len(),
        content_bytes = parsed.content.len(),
        tool_call_count = parsed.tool_calls.len(),
        has_reasoning = parsed.reasoning.is_some(),
        "assistant output parsed for model family"
    );
    Ok(parsed)
}

/// Parses a complete assistant output using an explicit parser family.
///
/// `Auto` detects known tool markers and falls back to plain assistant content.
pub fn parse_assistant_for_parser_family(
    family: ToolParserFamily,
    text: &str,
) -> Result<ParsedAssistant, ParserError> {
    let parsed = match family {
        ToolParserFamily::Auto => parse_assistant_auto(text),
        ToolParserFamily::Hermes | ToolParserFamily::Qwen => QwenParser.parse_complete(text),
        ToolParserFamily::DeepSeek => DeepSeekParser.parse_complete(text),
        ToolParserFamily::Gemma => GemmaParser.parse_complete(text),
        ToolParserFamily::Llama => LlamaParser.parse_complete(text),
        ToolParserFamily::Json => json::parse_json_tool_output(text),
        ToolParserFamily::Xlam => json::parse_xlam_tool_output(text),
    }?;
    tracing::trace!(
        operation = "parse_assistant",
        parser_family = ?family,
        input_bytes = text.len(),
        content_bytes = parsed.content.len(),
        tool_call_count = parsed.tool_calls.len(),
        has_reasoning = parsed.reasoning.is_some(),
        "assistant output parsed for parser family"
    );
    Ok(parsed)
}

/// Splits hidden reasoning markers from visible assistant content.
pub fn split_reasoning(text: &str) -> Result<(Option<String>, String), ParserError> {
    common::split_reasoning(text)
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
