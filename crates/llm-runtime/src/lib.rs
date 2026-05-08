use chrono::Utc;
use llm_api::{
    ApiError, ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage,
    ChatRole, ToolChoice, Usage, ValidateRequest,
};
use llm_backend::{BackendError, BackendRequest, ModelBackend};
use llm_tokenizer::{QwenPromptOptions, TemplateError, render_qwen_chatml};
use llm_tool_parser::{ParserError, QwenParser};
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
        if request.stream {
            return Err(ApiError::unsupported_capability(
                "streaming chat completions are not implemented yet",
            )
            .into());
        }
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
        let parsed = QwenParser.parse_complete(&output.text)?;
        let required_tool_pending = matches!(
            request.tool_choice,
            Some(ToolChoice::Required | ToolChoice::Function { .. })
        );
        let no_progress = classify_no_progress(
            &output.text,
            output.completion_tokens,
            required_tool_pending && parsed.tool_calls.is_empty(),
        );
        if let Some(class) = no_progress {
            return Err(RuntimeError::NoProgress(class));
        }
        let finish_reason = if parsed.tool_calls.is_empty() {
            output.finish_reason
        } else {
            llm_api::FinishReason::ToolCalls
        };
        let message = ChatMessage {
            role: ChatRole::Assistant,
            content: (!parsed.content.is_empty()).then_some(parsed.content),
            name: None,
            tool_call_id: None,
            tool_calls: parsed.tool_calls,
        };
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
                message,
                finish_reason: Some(finish_reason),
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
    #[error(transparent)]
    Parser(#[from] ParserError),
    #[error("no progress classified as {0:?}")]
    NoProgress(NoProgressClass),
}
