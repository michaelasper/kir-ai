//! Protocol runtime that connects validated OpenAI requests to local model backends.
//!
//! The runtime owns request lifecycle ordering: validate API shape, render prompts,
//! enforce backend capability gates, dispatch generation, apply stop-sequence
//! boundaries to assistant text, parse that bounded output, validate tool and
//! JSON-mode responses, classify no-progress output, and finally emit
//! OpenAI-compatible responses or stream events.

mod adapters;
mod backend_request;
mod cache_identity;
mod capabilities;
mod chat;
mod completions;
mod error;
mod json_mode;
mod no_progress;
mod response_validation;
mod runtime;
mod stop;
mod streaming;
mod tool_call;

pub use cache_identity::RequestCacheIdentity;
pub use error::RuntimeError;
pub use no_progress::{NoProgressClass, classify_no_progress};
pub use runtime::{Runtime, RuntimeOptions};
pub use streaming::{
    ChatCompletionStream, ChatCompletionStreamEvent, ChatCompletionStreamStage, CompletionStream,
    CompletionStreamEvent, StreamProgressMetadata,
};
pub use tool_call::ToolSchemaNormalization;
