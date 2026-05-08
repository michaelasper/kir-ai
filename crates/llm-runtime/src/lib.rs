use chrono::Utc;
use llm_api::{
    ApiError, ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage,
    Usage, ValidateRequest,
};
use llm_backend::{BackendError, BackendRequest, ModelBackend};
use llm_tokenizer::{QwenPromptOptions, TemplateError, render_qwen_chatml};
use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct Runtime<B> {
    backend: B,
}

impl<B> Runtime<B>
where
    B: ModelBackend,
{
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    pub fn model_id(&self) -> &str {
        self.backend.model_id()
    }

    pub async fn chat(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, RuntimeError> {
        request.validate()?;
        let prompt = render_qwen_chatml(
            &request.messages,
            &request.tools,
            &QwenPromptOptions {
                enable_thinking: false,
                add_generation_prompt: true,
            },
        )?;
        let output = self
            .backend
            .generate(BackendRequest {
                model: request.model.clone(),
                prompt,
                max_tokens: request.max_tokens.unwrap_or(4096),
            })
            .await?;
        let no_progress = classify_no_progress(
            &output.text,
            output.completion_tokens,
            !request.tools.is_empty(),
        );
        if let Some(class) = no_progress {
            return Err(RuntimeError::NoProgress(class));
        }
        let usage = Usage {
            prompt_tokens: output.prompt_tokens,
            completion_tokens: output.completion_tokens,
            total_tokens: output.prompt_tokens + output.completion_tokens,
        };
        Ok(ChatCompletionResponse {
            id: format!("chatcmpl-{}", Uuid::now_v7()),
            object: "chat.completion".to_owned(),
            created: Utc::now().timestamp(),
            model: request.model,
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatMessage::assistant(output.text),
                finish_reason: Some(output.finish_reason),
            }],
            usage,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoProgressClass {
    EmptyCompletion,
    EmptyHighOutputCompletion,
    TextFallbackRequiredTool,
}

pub fn classify_no_progress(
    content: &str,
    completion_tokens: u64,
    required_tool_pending: bool,
) -> Option<NoProgressClass> {
    if content.trim().is_empty() && completion_tokens >= 1024 {
        return Some(NoProgressClass::EmptyHighOutputCompletion);
    }
    if content.trim().is_empty() {
        return Some(NoProgressClass::EmptyCompletion);
    }
    if required_tool_pending && !content.contains("<tool_call>") {
        return Some(NoProgressClass::TextFallbackRequiredTool);
    }
    None
}

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error(transparent)]
    Api(#[from] ApiError),
    #[error(transparent)]
    Backend(#[from] BackendError),
    #[error(transparent)]
    Template(#[from] TemplateError),
    #[error("no progress classified as {0:?}")]
    NoProgress(NoProgressClass),
}
