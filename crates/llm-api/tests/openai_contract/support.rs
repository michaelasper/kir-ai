use llm_api::{
    ChatCompletionDelta, ChatCompletionRequest, ChatCompletionStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, ChatRole, CompletionRequest, CompletionResponse,
    CompletionStreamResponse, FinishReason, MAX_CHAT_MESSAGES, MAX_COMPLETION_PROMPT_BYTES,
    MAX_MESSAGE_CONTENT_BYTES, MAX_STOP_SEQUENCE_BYTES, MAX_STOP_SEQUENCES,
    MAX_TOOL_ARGUMENT_BYTES, MAX_TOOL_DESCRIPTION_BYTES, MAX_TOOL_SCHEMA_BYTES,
    MAX_TOOL_SCHEMA_DEPTH, MAX_TOOL_SCHEMA_ENUM_VALUES, MAX_TOOLS,
    NO_PROGRESS_EXACT_REPEATED_INVALID_TOOL_CALL_THRESHOLD,
    NO_PROGRESS_FUZZY_REPEATED_INVALID_TOOL_CALL_THRESHOLD, RequestLimits, ResponseFormat,
    ToolChoice, ToolDefinition, ValidateRequest, canonical_tool_schema_json, canonicalize_tool_schemas,
};
use serde_json::json;
use std::sync::Arc;
