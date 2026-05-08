use chrono::Utc;
use llm_api::{
    ApiError, ChatCompletionChoice, ChatCompletionDelta, ChatCompletionRequest,
    ChatCompletionResponse, ChatCompletionStreamChoice, ChatCompletionStreamResponse, ChatMessage,
    ChatRole, CompletionChoice, CompletionRequest, CompletionResponse, CompletionStreamResponse,
    ResponseFormat, ToolCall, ToolCallDelta, ToolCallFunctionDelta, ToolChoice, Usage,
    ValidateRequest,
};
use llm_backend::{BackendError, BackendRequest, ModelBackend};
use llm_tokenizer::{QwenPromptOptions, TemplateError, render_qwen_chatml};
use llm_tool_parser::{ParsedAssistant, ParserError, QwenParser};
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

    pub async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, RuntimeError> {
        request.validate()?;
        if request.stream {
            return Err(ApiError::unsupported_capability(
                "streaming text completion requests must use Runtime::completion_stream",
            )
            .into());
        }
        let completion = self.complete_text(request).await?;
        Ok(CompletionResponse {
            id: completion.id,
            object: "text_completion".to_owned(),
            created: completion.created,
            model: completion.model,
            choices: vec![CompletionChoice {
                text: completion.text,
                index: 0,
                finish_reason: Some(completion.finish_reason),
            }],
            usage: completion.usage,
        })
    }

    pub async fn completion_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionStream, RuntimeError> {
        let include_usage = request.stream_options.include_usage;
        let completion = self.complete_text(request).await?;
        let mut chunks = vec![
            completion_stream_chunk(&completion, completion.text.clone(), None),
            completion_stream_chunk(
                &completion,
                String::new(),
                Some(completion.finish_reason.clone()),
            ),
        ];
        if include_usage {
            chunks.push(completion_stream_usage_chunk(&completion));
        }
        Ok(CompletionStream { chunks })
    }

    pub async fn chat(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, RuntimeError> {
        if request.stream {
            return Err(ApiError::unsupported_capability(
                "streaming chat requests must use Runtime::chat_stream",
            )
            .into());
        }
        let completion = self.complete_chat(request).await?;
        let message = ChatMessage {
            role: ChatRole::Assistant,
            content: (!completion.parsed.content.is_empty()).then_some(completion.parsed.content),
            name: None,
            tool_call_id: None,
            tool_calls: completion.parsed.tool_calls,
        };
        Ok(ChatCompletionResponse {
            id: completion.id,
            object: "chat.completion".to_owned(),
            created: completion.created,
            model: completion.model,
            choices: vec![ChatCompletionChoice {
                index: 0,
                message,
                finish_reason: Some(completion.finish_reason),
            }],
            usage: completion.usage,
        })
    }

    pub async fn chat_stream(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionStream, RuntimeError> {
        let include_usage = request.stream_options.include_usage;
        let completion = self.complete_chat(request).await?;
        let mut chunks = Vec::new();
        chunks.push(stream_chunk(
            &completion,
            ChatCompletionDelta {
                role: Some(ChatRole::Assistant),
                ..ChatCompletionDelta::default()
            },
            None,
        ));
        if !completion.parsed.content.is_empty() {
            chunks.push(stream_chunk(
                &completion,
                ChatCompletionDelta {
                    content: Some(completion.parsed.content.clone()),
                    ..ChatCompletionDelta::default()
                },
                None,
            ));
        }
        for (index, tool_call) in completion.parsed.tool_calls.iter().enumerate() {
            chunks.push(stream_chunk(
                &completion,
                ChatCompletionDelta {
                    tool_calls: vec![tool_call_delta(index, tool_call)?],
                    ..ChatCompletionDelta::default()
                },
                None,
            ));
        }
        chunks.push(stream_chunk(
            &completion,
            ChatCompletionDelta::default(),
            Some(completion.finish_reason.clone()),
        ));
        if include_usage {
            chunks.push(stream_usage_chunk(&completion));
        }
        Ok(ChatCompletionStream { chunks })
    }

    async fn complete_chat(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<RuntimeChatCompletion, RuntimeError> {
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
        let mut parsed = QwenParser.parse_complete(&output.text)?;
        let stopped = apply_stop_sequences(&mut parsed.content, &request.stop);
        validate_tool_call_arguments(&parsed)?;
        if matches!(request.response_format, Some(ResponseFormat::JsonObject)) {
            validate_json_object_response(&parsed)?;
        }
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
        let finish_reason = if !parsed.tool_calls.is_empty() {
            llm_api::FinishReason::ToolCalls
        } else if stopped {
            llm_api::FinishReason::Stop
        } else {
            output.finish_reason
        };
        let usage = Usage {
            prompt_tokens: output.prompt_tokens,
            completion_tokens: output.completion_tokens,
            total_tokens: output.prompt_tokens + output.completion_tokens,
        };
        Ok(RuntimeChatCompletion {
            id: format!("chatcmpl-{}", Uuid::now_v7()),
            created: Utc::now().timestamp(),
            model: request.model,
            parsed,
            finish_reason,
            usage,
        })
    }

    async fn complete_text(
        &self,
        request: CompletionRequest,
    ) -> Result<RuntimeCompletion, RuntimeError> {
        request.validate()?;
        let output = self
            .backend
            .generate(BackendRequest {
                model: request.model.clone(),
                prompt: request.prompt,
                max_tokens: request.max_tokens.unwrap_or(4096),
            })
            .await?;
        let mut text = output.text;
        let stopped = apply_stop_sequences(&mut text, &request.stop);
        let no_progress = classify_no_progress(&text, output.completion_tokens, false);
        if let Some(class) = no_progress {
            return Err(RuntimeError::NoProgress(class));
        }
        let usage = Usage {
            prompt_tokens: output.prompt_tokens,
            completion_tokens: output.completion_tokens,
            total_tokens: output.prompt_tokens + output.completion_tokens,
        };
        Ok(RuntimeCompletion {
            id: format!("cmpl-{}", Uuid::now_v7()),
            created: Utc::now().timestamp(),
            model: request.model,
            text,
            finish_reason: if stopped {
                llm_api::FinishReason::Stop
            } else {
                output.finish_reason
            },
            usage,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChatCompletionStream {
    pub chunks: Vec<ChatCompletionStreamResponse>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompletionStream {
    pub chunks: Vec<CompletionStreamResponse>,
}

#[derive(Debug, Clone, PartialEq)]
struct RuntimeCompletion {
    id: String,
    created: i64,
    model: String,
    text: String,
    finish_reason: llm_api::FinishReason,
    usage: Usage,
}

#[derive(Debug, Clone, PartialEq)]
struct RuntimeChatCompletion {
    id: String,
    created: i64,
    model: String,
    parsed: ParsedAssistant,
    finish_reason: llm_api::FinishReason,
    usage: Usage,
}

fn stream_chunk(
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

fn stream_usage_chunk(completion: &RuntimeChatCompletion) -> ChatCompletionStreamResponse {
    ChatCompletionStreamResponse {
        id: completion.id.clone(),
        object: "chat.completion.chunk".to_owned(),
        created: completion.created,
        model: completion.model.clone(),
        choices: Vec::new(),
        usage: Some(completion.usage.clone()),
    }
}

fn completion_stream_chunk(
    completion: &RuntimeCompletion,
    text: String,
    finish_reason: Option<llm_api::FinishReason>,
) -> CompletionStreamResponse {
    CompletionStreamResponse {
        id: completion.id.clone(),
        object: "text_completion".to_owned(),
        created: completion.created,
        model: completion.model.clone(),
        choices: vec![CompletionChoice {
            text,
            index: 0,
            finish_reason,
        }],
        usage: None,
    }
}

fn completion_stream_usage_chunk(completion: &RuntimeCompletion) -> CompletionStreamResponse {
    CompletionStreamResponse {
        id: completion.id.clone(),
        object: "text_completion".to_owned(),
        created: completion.created,
        model: completion.model.clone(),
        choices: Vec::new(),
        usage: Some(completion.usage.clone()),
    }
}

fn tool_call_delta(index: usize, tool_call: &ToolCall) -> Result<ToolCallDelta, RuntimeError> {
    Ok(ToolCallDelta {
        index: u32::try_from(index).map_err(|err| {
            ApiError::invalid_request(format!("tool call index does not fit u32: {err}"))
        })?,
        id: Some(tool_call.id.clone()),
        call_type: Some(tool_call.call_type.clone()),
        function: Some(ToolCallFunctionDelta {
            name: Some(tool_call.function.name.clone()),
            arguments: Some(serde_json::to_string(&tool_call.function.arguments)?),
        }),
    })
}

fn apply_stop_sequences(content: &mut String, stop: &[String]) -> bool {
    let Some(stop_at) = stop
        .iter()
        .filter_map(|sequence| content.find(sequence))
        .min()
    else {
        return false;
    };
    content.truncate(stop_at);
    true
}

fn validate_tool_call_arguments(parsed: &ParsedAssistant) -> Result<(), RuntimeError> {
    for tool_call in &parsed.tool_calls {
        if !tool_call.function.arguments.is_object() {
            return Err(RuntimeError::JsonMode(format!(
                "tool call `{}` arguments must be a JSON object",
                tool_call.function.name
            )));
        }
    }
    Ok(())
}

fn validate_json_object_response(parsed: &ParsedAssistant) -> Result<(), RuntimeError> {
    if !parsed.content.is_empty() {
        let value = serde_json::from_str::<serde_json::Value>(&parsed.content).map_err(|err| {
            RuntimeError::JsonMode(format!(
                "json_object response_format requires valid JSON object content: {err}"
            ))
        })?;
        if !value.is_object() {
            return Err(RuntimeError::JsonMode(
                "json_object response_format requires assistant content to be a JSON object"
                    .to_owned(),
            ));
        }
    } else if parsed.tool_calls.is_empty() {
        return Err(RuntimeError::JsonMode(
            "json_object response_format requires assistant content or tool calls".to_owned(),
        ));
    }
    Ok(())
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
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    JsonMode(String),
    #[error("no progress classified as {0:?}")]
    NoProgress(NoProgressClass),
}
