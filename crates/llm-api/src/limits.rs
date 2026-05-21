pub const MAX_JSON_BODY_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_CHAT_MESSAGES: usize = 128;
pub const MAX_MESSAGE_CONTENT_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_COMPLETION_PROMPT_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_NAME_BYTES: usize = 1024;
pub const MAX_TOOLS: usize = 128;
pub const MAX_TOOL_DESCRIPTION_BYTES: usize = 1024 * 1024;
pub const MAX_TOOL_SCHEMA_BYTES: usize = 1024 * 1024;
pub const MAX_TOOL_CALLS_PER_MESSAGE: usize = 128;
pub const MAX_TOOL_ARGUMENT_BYTES: usize = 1024 * 1024;
pub const MAX_STOP_SEQUENCES: usize = 4;
pub const MAX_STOP_SEQUENCE_BYTES: usize = 1024;
pub const NO_PROGRESS_EXACT_REPEATED_INVALID_TOOL_CALL_THRESHOLD: usize = 5;
pub const NO_PROGRESS_FUZZY_REPEATED_INVALID_TOOL_CALL_THRESHOLD: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestLimits {
    pub json_body_bytes: usize,
    pub message_content_bytes: usize,
    pub completion_prompt_bytes: usize,
}

impl Default for RequestLimits {
    fn default() -> Self {
        Self {
            json_body_bytes: MAX_JSON_BODY_BYTES,
            message_content_bytes: MAX_MESSAGE_CONTENT_BYTES,
            completion_prompt_bytes: MAX_COMPLETION_PROMPT_BYTES,
        }
    }
}
