use crate::RuntimeError;
use futures::{StreamExt, stream::BoxStream};
use llm_api::{
    ChatCompletionDelta, ChatCompletionStreamChoice, ChatCompletionStreamResponse,
    CompletionChoice, CompletionStreamResponse, Usage,
};
use llm_tool_parser::ParsedAssistant;
use std::fmt;
use tokio_util::sync::CancellationToken;

pub struct ChatCompletionStream<'a> {
    events: BoxStream<'a, Result<ChatCompletionStreamEvent, RuntimeError>>,
}

impl fmt::Debug for ChatCompletionStream<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ChatCompletionStream { events: <stream> }")
    }
}

impl<'a> ChatCompletionStream<'a> {
    pub(crate) fn new(
        events: BoxStream<'a, Result<ChatCompletionStreamEvent, RuntimeError>>,
    ) -> Self {
        Self { events }
    }

    pub fn into_events(self) -> BoxStream<'a, Result<ChatCompletionStreamEvent, RuntimeError>> {
        self.events
    }

    pub async fn collect_chunks(
        self,
    ) -> Result<(Vec<ChatCompletionStreamResponse>, Usage), RuntimeError> {
        let mut chunks = Vec::new();
        let mut usage = None;
        let mut events = self.into_events();
        while let Some(event) = events.next().await {
            match event? {
                ChatCompletionStreamEvent::Chunk(chunk) => chunks.push(chunk),
                ChatCompletionStreamEvent::Stage(_) => {}
                ChatCompletionStreamEvent::Complete(final_usage) => usage = Some(final_usage),
            }
        }
        Ok((chunks, usage.unwrap_or_else(empty_usage)))
    }
}

pub struct CompletionStream<'a> {
    events: BoxStream<'a, Result<CompletionStreamEvent, RuntimeError>>,
}

impl fmt::Debug for CompletionStream<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("CompletionStream { events: <stream> }")
    }
}

impl<'a> CompletionStream<'a> {
    pub(crate) fn new(events: BoxStream<'a, Result<CompletionStreamEvent, RuntimeError>>) -> Self {
        Self { events }
    }

    pub fn into_events(self) -> BoxStream<'a, Result<CompletionStreamEvent, RuntimeError>> {
        self.events
    }

    pub async fn collect_chunks(
        self,
    ) -> Result<(Vec<CompletionStreamResponse>, Usage), RuntimeError> {
        let mut chunks = Vec::new();
        let mut usage = None;
        let mut events = self.into_events();
        while let Some(event) = events.next().await {
            match event? {
                CompletionStreamEvent::Chunk(chunk) => chunks.push(chunk),
                CompletionStreamEvent::Complete(final_usage) => usage = Some(final_usage),
            }
        }
        Ok((chunks, usage.unwrap_or_else(empty_usage)))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChatCompletionStreamEvent {
    Chunk(ChatCompletionStreamResponse),
    Stage(ChatCompletionStreamStage),
    Complete(Usage),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatCompletionStreamStage {
    ToolArgumentAssemblyComplete,
    ToolIntentFillComplete,
    ToolSchemaValidationComplete,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CompletionStreamEvent {
    Chunk(CompletionStreamResponse),
    Complete(Usage),
}

#[derive(Debug)]
pub(crate) struct CancelOnDrop {
    token: CancellationToken,
}

impl CancelOnDrop {
    pub(crate) fn new(token: CancellationToken) -> Self {
        Self { token }
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.token.cancel();
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RuntimeCompletionSeed {
    pub(crate) id: String,
    pub(crate) created: i64,
    pub(crate) model: String,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RuntimeCompletion {
    pub(crate) id: String,
    pub(crate) created: i64,
    pub(crate) model: String,
    pub(crate) text: String,
    pub(crate) finish_reason: llm_api::FinishReason,
    pub(crate) usage: Usage,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RuntimeChatCompletion {
    pub(crate) id: String,
    pub(crate) created: i64,
    pub(crate) model: String,
    pub(crate) parsed: ParsedAssistant,
    pub(crate) finish_reason: llm_api::FinishReason,
    pub(crate) usage: Usage,
}

fn empty_usage() -> Usage {
    Usage::new(0, 0)
}

pub(crate) fn usage_from_tokens(
    prompt_tokens: u64,
    completion_tokens: u64,
    prompt_cached_tokens: Option<u64>,
) -> Usage {
    Usage::new(prompt_tokens, completion_tokens).with_prompt_cached_tokens(prompt_cached_tokens)
}

pub(crate) fn max_optional_u64(current: Option<u64>, next: Option<u64>) -> Option<u64> {
    match (current, next) {
        (Some(current), Some(next)) => Some(current.max(next)),
        (Some(current), None) => Some(current),
        (None, Some(next)) => Some(next),
        (None, None) => None,
    }
}

pub(crate) fn stream_chunk(
    completion: &RuntimeChatCompletion,
    delta: ChatCompletionDelta,
    finish_reason: Option<llm_api::FinishReason>,
) -> ChatCompletionStreamResponse {
    ChatCompletionStreamResponse {
        id: completion.id.clone(),
        object: "chat.completion.chunk".to_owned(),
        created: completion.created,
        model: completion.model.clone(),
        choices: vec![ChatCompletionStreamChoice {
            index: 0,
            delta,
            finish_reason,
        }],
        usage: None,
    }
}

pub(crate) fn stream_usage_chunk(
    completion: &RuntimeChatCompletion,
) -> ChatCompletionStreamResponse {
    ChatCompletionStreamResponse {
        id: completion.id.clone(),
        object: "chat.completion.chunk".to_owned(),
        created: completion.created,
        model: completion.model.clone(),
        choices: Vec::new(),
        usage: Some(completion.usage.clone()),
    }
}

pub(crate) fn stream_seed_chunk(
    completion: &RuntimeCompletionSeed,
    delta: ChatCompletionDelta,
    finish_reason: Option<llm_api::FinishReason>,
    usage: Option<Usage>,
) -> ChatCompletionStreamResponse {
    ChatCompletionStreamResponse {
        id: completion.id.clone(),
        object: "chat.completion.chunk".to_owned(),
        created: completion.created,
        model: completion.model.clone(),
        choices: if usage.is_some() {
            Vec::new()
        } else {
            vec![ChatCompletionStreamChoice {
                index: 0,
                delta,
                finish_reason,
            }]
        },
        usage,
    }
}

pub(crate) fn completion_stream_seed_chunk(
    completion: &RuntimeCompletionSeed,
    text: String,
    finish_reason: Option<llm_api::FinishReason>,
    usage: Option<Usage>,
) -> CompletionStreamResponse {
    CompletionStreamResponse {
        id: completion.id.clone(),
        object: "text_completion".to_owned(),
        created: completion.created,
        model: completion.model.clone(),
        choices: if usage.is_some() {
            Vec::new()
        } else {
            vec![CompletionChoice {
                text,
                index: 0,
                finish_reason,
            }]
        },
        usage,
    }
}
