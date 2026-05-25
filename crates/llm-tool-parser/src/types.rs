use llm_api::{ToolCall, ToolCallFunction, ToolCallType};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

/// Parsed assistant turn after family-specific tool markup has been interpreted.
///
/// Runtime validation consumes this structure before building API responses so
/// malformed tool calls or invalid JSON-object output never leak as successful
/// assistant deltas.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParsedAssistant {
    /// Hidden reasoning text, when the family exposes a distinct reasoning channel.
    pub reasoning: Option<String>,
    /// User-visible assistant text after tool markup is removed.
    pub content: String,
    /// Function tool calls parsed from the assistant output.
    pub tool_calls: Vec<ToolCall>,
}

impl ParsedAssistant {
    /// Builds a parsed assistant containing only visible text content.
    pub fn content(content: impl Into<String>) -> Self {
        Self {
            reasoning: None,
            content: content.into(),
            tool_calls: Vec::new(),
        }
    }

    /// Builds a parsed assistant containing one function tool call.
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

/// Parser failure with a stable machine-readable code.
#[derive(Debug, Error)]
#[error("{code}: {message}")]
pub struct ParserError {
    code: &'static str,
    message: String,
}

impl ParserError {
    /// Stable parser error code.
    pub fn code(&self) -> &'static str {
        self.code
    }

    pub(crate) fn malformed_tool(message: impl Into<String>) -> Self {
        Self {
            code: "malformed_tool_call",
            message: message.into(),
        }
    }

    pub(crate) fn unsupported_multimodal(message: impl Into<String>) -> Self {
        Self {
            code: "unsupported_multimodal_output",
            message: message.into(),
        }
    }

    pub(crate) fn unsupported_model_family(family: &str) -> Self {
        Self {
            code: "unsupported_model_family",
            message: format!("model family `{family}` does not have a parser"),
        }
    }
}

/// Parser dialect selection compatible with vLLM tool parser names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ToolParserFamily {
    /// Detect parser from known output markers.
    Auto,
    /// Hermes/Qwen XML-style parser route.
    Hermes,
    /// Qwen XML-style parser route.
    Qwen,
    /// DeepSeek native or DSML parser route.
    DeepSeek,
    /// Gemma reasoning/tool-channel parser route.
    Gemma,
    /// Llama parser route.
    Llama,
    /// Generic JSON function-call parser route.
    Json,
    /// xLAM tool-call parser route.
    Xlam,
}

impl ToolParserFamily {
    /// Maps a vLLM parser name or suffixed parser name to this crate's family.
    pub fn from_vllm_name(name: &str) -> Option<Self> {
        let normalized = normalize_parser_name(name);
        VLLM_TOOL_PARSER_ROUTES
            .iter()
            .find_map(|(candidate, family)| (normalized == *candidate).then_some(*family))
    }

    /// Returns the vLLM parser names accepted by `from_vllm_name`.
    pub fn supported_vllm_names() -> &'static [(&'static str, Self)] {
        VLLM_TOOL_PARSER_ROUTES
    }
}

const VLLM_TOOL_PARSER_ROUTES: &[(&str, ToolParserFamily)] = &[
    ("deepseek_v3", ToolParserFamily::DeepSeek),
    ("functiongemma", ToolParserFamily::Gemma),
    ("gemma4", ToolParserFamily::Gemma),
    ("hermes", ToolParserFamily::Hermes),
    ("llama3", ToolParserFamily::Llama),
    ("llama3_json", ToolParserFamily::Llama),
    ("qwen3coder", ToolParserFamily::Qwen),
    ("qwen3xml", ToolParserFamily::Qwen),
    ("xlam", ToolParserFamily::Xlam),
    ("json", ToolParserFamily::Json),
    ("openai", ToolParserFamily::Json),
    ("mistral", ToolParserFamily::Json),
    ("granite", ToolParserFamily::Json),
    ("granite_20b_fc", ToolParserFamily::Json),
    ("hunyuan_a13b", ToolParserFamily::Json),
    ("kimi_k2", ToolParserFamily::Json),
    ("minimax", ToolParserFamily::Json),
    ("minimax_m2", ToolParserFamily::Json),
    ("olmo3", ToolParserFamily::Json),
    ("phi4mini", ToolParserFamily::Json),
    ("seed_oss", ToolParserFamily::Json),
    ("step3", ToolParserFamily::Json),
    ("step3p5", ToolParserFamily::Json),
];

fn normalize_parser_name(name: &str) -> String {
    name.trim()
        .trim_end_matches("_tool_parser")
        .trim_end_matches("_parser")
        .replace('-', "_")
        .to_ascii_lowercase()
}
