//! Small model-output fixture packs for tokenizer/template/parser drift tests.
//!
//! Fixtures live under `fixtures/model-output/<family>/` and are versioned JSON
//! documents. Each pack captures a model-facing round trip: request messages and
//! tools, raw assistant output markup, the tool result turn, expected rendered
//! prompt bytes, expected token IDs, and expected parser output.

use llm_api::{ChatMessage, ToolDefinition};
use serde::Deserialize;
use serde_json::Value;

#[derive(Clone, Debug, Deserialize)]
pub struct ModelOutputFixturePack {
    pub schema_version: u32,
    pub family: String,
    pub case_name: String,
    pub description: String,
    pub tokenizer_fixture: String,
    pub prompt_options: PromptOptionsFixture,
    pub tools: Vec<ToolDefinition>,
    pub messages_before_assistant: Vec<ChatMessage>,
    pub assistant_output: String,
    pub tool_result_content: String,
    pub follow_up_user: String,
    pub expected: ExpectedModelOutput,
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct PromptOptionsFixture {
    pub enable_thinking: bool,
    pub add_generation_prompt: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ExpectedModelOutput {
    pub rendered_prompt: String,
    pub token_ids: Vec<u32>,
    pub parsed_reasoning: Option<String>,
    pub parsed_content: String,
    pub parsed_tool_name: String,
    pub parsed_tool_arguments: Value,
}

pub fn qwen_tool_history_round_trip() -> Result<ModelOutputFixturePack, serde_json::Error> {
    serde_json::from_str(include_str!(
        "../fixtures/model-output/qwen/tool_history_round_trip.json"
    ))
}
