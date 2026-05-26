/// Maximum accepted JSON request body size in bytes.
pub const MAX_JSON_BODY_BYTES: usize = 16 * 1024 * 1024;
/// Maximum number of chat messages accepted in one request.
pub const MAX_CHAT_MESSAGES: usize = 128;
/// Maximum UTF-8 byte length for one chat message content value.
pub const MAX_MESSAGE_CONTENT_BYTES: usize = 8 * 1024 * 1024;
/// Maximum UTF-8 byte length for a legacy completion prompt.
pub const MAX_COMPLETION_PROMPT_BYTES: usize = 8 * 1024 * 1024;
/// Maximum UTF-8 byte length for OpenAI `name` fields and function names.
pub const MAX_NAME_BYTES: usize = 1024;
/// Maximum number of declared tools accepted in one chat request.
pub const MAX_TOOLS: usize = 128;
/// Maximum UTF-8 byte length for a tool description.
pub const MAX_TOOL_DESCRIPTION_BYTES: usize = 1024 * 1024;
/// Maximum serialized JSON byte length for one tool schema.
pub const MAX_TOOL_SCHEMA_BYTES: usize = 1024 * 1024;
/// Maximum nested JSON Schema object depth below a tool `parameters` root.
pub const MAX_TOOL_SCHEMA_DEPTH: usize = 16;
/// Maximum number of literal values allowed in one JSON Schema `enum` array.
pub const MAX_TOOL_SCHEMA_ENUM_VALUES: usize = 256;
/// Maximum number of tool calls accepted on one assistant message.
pub const MAX_TOOL_CALLS_PER_MESSAGE: usize = 128;
/// Maximum serialized JSON byte length for one tool call argument object.
pub const MAX_TOOL_ARGUMENT_BYTES: usize = 1024 * 1024;
/// Maximum number of stop sequences accepted in one request.
pub const MAX_STOP_SEQUENCES: usize = 4;
/// Maximum UTF-8 byte length for one stop sequence.
pub const MAX_STOP_SEQUENCE_BYTES: usize = 1024;
/// Exact repeated failed tool-call threshold for no-progress detection.
pub const NO_PROGRESS_EXACT_REPEATED_INVALID_TOOL_CALL_THRESHOLD: usize = 5;
/// Fuzzy repeated failed tool-call threshold for no-progress detection.
pub const NO_PROGRESS_FUZZY_REPEATED_INVALID_TOOL_CALL_THRESHOLD: usize = 3;

/// Size limits used when validating API requests.
///
/// Keeping the values in a struct lets tests and embedded runtimes validate
/// with narrower limits without changing the public request types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestLimits {
    /// Maximum JSON request body size in bytes.
    pub json_body_bytes: usize,
    /// Maximum UTF-8 byte length for one chat message content value.
    pub message_content_bytes: usize,
    /// Maximum UTF-8 byte length for a legacy completion prompt.
    pub completion_prompt_bytes: usize,
    /// Maximum nested JSON Schema object depth below a tool `parameters` root.
    pub tool_schema_depth: usize,
    /// Maximum number of literal values allowed in one JSON Schema `enum` array.
    pub tool_schema_enum_values: usize,
}

impl Default for RequestLimits {
    fn default() -> Self {
        Self {
            json_body_bytes: MAX_JSON_BODY_BYTES,
            message_content_bytes: MAX_MESSAGE_CONTENT_BYTES,
            completion_prompt_bytes: MAX_COMPLETION_PROMPT_BYTES,
            tool_schema_depth: MAX_TOOL_SCHEMA_DEPTH,
            tool_schema_enum_values: MAX_TOOL_SCHEMA_ENUM_VALUES,
        }
    }
}
