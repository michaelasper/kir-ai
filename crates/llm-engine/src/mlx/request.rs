use super::protocol::{
    MlxUpstreamProtocol, mlx_effective_chat_template_kwargs, mlx_upstream_protocol_for_request,
};
use llm_api::ChatMessage;
use llm_backend_contracts::{
    BackendChatMessage, BackendChatRole, BackendError, BackendModelMetadata, BackendRequest,
    BackendToolCall, BackendToolCallFunction, BackendToolCallType, BackendToolChoice,
    BackendToolDefinition, SamplingConfig,
};
use llm_models::ModelFamily;
use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Clone)]
pub(super) struct MlxUpstreamRequest {
    protocol: MlxUpstreamProtocol,
    body: Vec<u8>,
    content_type: &'static str,
}

impl MlxUpstreamRequest {
    fn json<T: Serialize>(protocol: MlxUpstreamProtocol, body: &T) -> Result<Self, BackendError> {
        let body = serde_json::to_vec(body).map_err(|err| {
            BackendError::other(format!("failed to serialize MLX JSON request: {err}"))
        })?;
        Ok(Self {
            protocol,
            body,
            content_type: "application/json",
        })
    }

    pub(super) fn protocol(&self) -> MlxUpstreamProtocol {
        self.protocol
    }

    #[cfg(test)]
    pub(super) fn body(&self) -> &[u8] {
        &self.body
    }

    pub(super) fn content_type(&self) -> &'static str {
        self.content_type
    }

    pub(super) fn into_body(self) -> Vec<u8> {
        self.body
    }
}

pub(super) fn build_upstream_request(
    upstream_model: &str,
    metadata: &BackendModelMetadata,
    request: &BackendRequest,
    stream: bool,
    include_stream_usage: bool,
) -> Result<MlxUpstreamRequest, BackendError> {
    let protocol = mlx_upstream_protocol_for_request(metadata, request)?;
    let (temperature, top_p) = match request.sampling {
        SamplingConfig::Greedy => (0.0, 1.0),
        SamplingConfig::TopP { temperature, top_p } => (temperature, top_p),
        _ => {
            return Err(BackendError::unsupported_request(format!(
                "unsupported MLX sampling config `{:?}`",
                request.sampling
            )));
        }
    };
    let stream_options = (stream && include_stream_usage).then_some(MlxStreamOptions {
        include_usage: true,
    });
    let request = match protocol {
        MlxUpstreamProtocol::Completions => MlxUpstreamRequest::json(
            protocol,
            &MlxCompletionRequest {
                model: upstream_model,
                prompt: request.prompt(),
                max_tokens: request.max_tokens,
                temperature,
                top_p,
                stream,
                stream_options,
            },
        )?,
        MlxUpstreamProtocol::ChatCompletions => {
            validate_mlx_chat_tool_choice_support(metadata, request)?;
            let messages = mlx_chat_messages(request)?;
            let tools = mlx_tools(request);
            let tool_choice = mlx_tool_choice(request)?;
            let response_format = mlx_response_format(request);
            let chat_template_kwargs = mlx_effective_chat_template_kwargs(metadata, request);
            MlxUpstreamRequest::json(
                protocol,
                &MlxChatCompletionRequest {
                    model: upstream_model,
                    messages,
                    tools,
                    tool_choice,
                    response_format,
                    chat_template_kwargs,
                    max_tokens: request.max_tokens,
                    temperature,
                    top_p,
                    stream,
                    stream_options,
                },
            )?
        }
    };
    Ok(request)
}

fn validate_mlx_chat_tool_choice_support(
    metadata: &BackendModelMetadata,
    request: &BackendRequest,
) -> Result<(), BackendError> {
    let Some(chat) = request.as_chat() else {
        return Ok(());
    };
    let Some(choice) = chat.required_tool_choice.as_ref() else {
        return Ok(());
    };
    if metadata_family(metadata)? != Some(ModelFamily::Gemma) {
        return Ok(());
    }
    Err(BackendError::unsupported_request(format!(
        "MLX Gemma required tool_choice is not supported for model `{}` \
         (backend `{}`, family `{}`); required tool choice {} cannot be enforced",
        metadata.id,
        metadata.backend,
        metadata.family.as_deref().unwrap_or("unknown"),
        required_tool_choice_label(choice)
    )))
}

fn metadata_family(metadata: &BackendModelMetadata) -> Result<Option<ModelFamily>, BackendError> {
    metadata
        .family
        .as_deref()
        .map(ModelFamily::parse_slug)
        .transpose()
        .map_err(|err| BackendError::unsupported_request(err.to_string()))
}

fn required_tool_choice_label(choice: &BackendToolChoice) -> String {
    match choice {
        BackendToolChoice::RequiredAny => "any declared tool".to_owned(),
        BackendToolChoice::RequiredFunction(name) => format!("function `{name}`"),
        other => format!("{other:?}"),
    }
}

#[derive(Debug, Serialize)]
struct MlxCompletionRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    max_tokens: Option<u32>,
    temperature: f32,
    top_p: f32,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<MlxStreamOptions>,
}

#[derive(Debug, Serialize)]
struct MlxChatCompletionRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [BackendToolDefinition]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_template_kwargs: Option<Value>,
    max_tokens: Option<u32>,
    temperature: f32,
    top_p: f32,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<MlxStreamOptions>,
}

#[derive(Debug, Serialize)]
struct MlxStreamOptions {
    include_usage: bool,
}

fn mlx_chat_messages(request: &BackendRequest) -> Result<Vec<ChatMessage>, BackendError> {
    if let Some(chat) = request.as_chat() {
        return chat
            .chat_context
            .messages
            .iter()
            .map(mlx_chat_message)
            .collect();
    }
    Ok(vec![ChatMessage::user(request.prompt().to_owned())])
}

fn mlx_chat_message(message: &BackendChatMessage) -> Result<ChatMessage, BackendError> {
    Ok(ChatMessage {
        role: mlx_chat_role(&message.role)?,
        content: message.content.clone(),
        name: message.name.clone(),
        tool_call_id: message.tool_call_id.clone(),
        tool_calls: message
            .tool_calls
            .iter()
            .map(mlx_tool_call)
            .collect::<Result<Vec<_>, _>>()?,
    })
}

fn mlx_chat_role(role: &BackendChatRole) -> Result<llm_api::ChatRole, BackendError> {
    Ok(match role {
        BackendChatRole::System => llm_api::ChatRole::System,
        BackendChatRole::User => llm_api::ChatRole::User,
        BackendChatRole::Assistant => llm_api::ChatRole::Assistant,
        BackendChatRole::Tool => llm_api::ChatRole::Tool,
        _ => {
            return Err(BackendError::unsupported_request(format!(
                "unsupported MLX chat role `{role:?}`"
            )));
        }
    })
}

fn mlx_tool_call(tool_call: &BackendToolCall) -> Result<llm_api::ToolCall, BackendError> {
    Ok(llm_api::ToolCall {
        id: tool_call.id.clone(),
        call_type: mlx_tool_call_type(&tool_call.call_type)?,
        function: mlx_tool_call_function(&tool_call.function),
    })
}

fn mlx_tool_call_type(
    call_type: &BackendToolCallType,
) -> Result<llm_api::ToolCallType, BackendError> {
    Ok(match call_type {
        BackendToolCallType::Function => llm_api::ToolCallType::Function,
        _ => {
            return Err(BackendError::unsupported_request(format!(
                "unsupported MLX tool call type `{call_type:?}`"
            )));
        }
    })
}

fn mlx_tool_call_function(function: &BackendToolCallFunction) -> llm_api::ToolCallFunction {
    llm_api::ToolCallFunction {
        name: function.name.clone(),
        arguments: function.arguments.clone(),
    }
}

fn mlx_tools(request: &BackendRequest) -> Option<&[BackendToolDefinition]> {
    request
        .as_chat()
        .map(|chat| chat.chat_context.tools.as_slice())
        .filter(|tools| !tools.is_empty())
}

fn mlx_tool_choice(request: &BackendRequest) -> Result<Option<Value>, BackendError> {
    request
        .as_chat()
        .and_then(|chat| chat.required_tool_choice.as_ref())
        .map(|choice| {
            Ok(match choice {
                llm_backend_contracts::BackendToolChoice::RequiredAny => {
                    Value::String("required".to_owned())
                }
                llm_backend_contracts::BackendToolChoice::RequiredFunction(name) => {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": name,
                        },
                    })
                }
                _ => {
                    return Err(BackendError::unsupported_request(format!(
                        "unsupported MLX tool choice `{choice:?}`"
                    )));
                }
            })
        })
        .transpose()
}

fn mlx_response_format(request: &BackendRequest) -> Option<Value> {
    request
        .as_chat()
        .is_some_and(|chat| chat.json_object_mode)
        .then(|| serde_json::json!({"type": "json_object"}))
}
