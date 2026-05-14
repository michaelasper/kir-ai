use llm_api::{ToolCall, ToolCallFunction, ToolCallType};
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolParserFamily {
    Auto,
    Hermes,
    Qwen,
    DeepSeek,
    Gemma,
    Llama,
    Json,
    Xlam,
}

impl ToolParserFamily {
    pub fn from_vllm_name(name: &str) -> Option<Self> {
        let normalized = normalize_parser_name(name);
        VLLM_TOOL_PARSER_ROUTES
            .iter()
            .find_map(|(candidate, family)| (normalized == *candidate).then_some(*family))
    }

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
