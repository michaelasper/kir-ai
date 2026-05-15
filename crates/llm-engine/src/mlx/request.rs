use super::{
    client::mlx_endpoint_url,
    protocol::{
        MlxUpstreamProtocol, mlx_effective_chat_template_kwargs, mlx_upstream_protocol_for_request,
    },
};
use llm_api::ChatMessage;
use llm_backend::{
    BackendChatMessage, BackendChatRole, BackendError, BackendModelMetadata, BackendRequest,
    BackendToolCall, BackendToolCallFunction, BackendToolCallType, SamplingConfig,
};
use serde::Serialize;
use serde_json::Value;
use url::Url;

pub(super) fn build_upstream_request(
    client: &reqwest::Client,
    endpoint: &Url,
    upstream_model: &str,
    metadata: &BackendModelMetadata,
    request: &BackendRequest,
    stream: bool,
    include_stream_usage: bool,
) -> Result<(MlxUpstreamProtocol, reqwest::RequestBuilder), BackendError> {
    let protocol = mlx_upstream_protocol_for_request(metadata, request);
    let (temperature, top_p) = match request.sampling {
        SamplingConfig::Greedy => (0.0, 1.0),
        SamplingConfig::TopP { temperature, top_p } => (temperature, top_p),
    };
    let stream_options = (stream && include_stream_usage).then_some(MlxStreamOptions {
        include_usage: true,
    });
    let upstream_url = mlx_endpoint_url(endpoint, protocol.endpoint_suffix());
    let request = match protocol {
        MlxUpstreamProtocol::Completions => client.post(upstream_url).json(&MlxCompletionRequest {
            model: upstream_model,
            prompt: &request.prompt,
            max_tokens: request.max_tokens,
            temperature,
            top_p,
            stream,
            stream_options,
        }),
        MlxUpstreamProtocol::ChatCompletions => {
            let messages = mlx_chat_messages(request);
            let tools = mlx_tool_schema(request)?;
            let tool_choice = mlx_tool_choice(request);
            let response_format = mlx_response_format(request);
            let chat_template_kwargs = mlx_effective_chat_template_kwargs(metadata, request);
            client.post(upstream_url).json(&MlxChatCompletionRequest {
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
            })
        }
    };
    Ok((protocol, request))
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
    tools: Option<Value>,
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

fn mlx_chat_messages(request: &BackendRequest) -> Vec<ChatMessage> {
    if let Some(chat_context) = &request.chat_context {
        return chat_context.messages.iter().map(mlx_chat_message).collect();
    }
    vec![ChatMessage::user(request.prompt.clone())]
}

fn mlx_chat_message(message: &BackendChatMessage) -> ChatMessage {
    ChatMessage {
        role: mlx_chat_role(&message.role),
        content: message.content.clone(),
        name: message.name.clone(),
        tool_call_id: message.tool_call_id.clone(),
        tool_calls: message.tool_calls.iter().map(mlx_tool_call).collect(),
    }
}

fn mlx_chat_role(role: &BackendChatRole) -> llm_api::ChatRole {
    match role {
        BackendChatRole::System => llm_api::ChatRole::System,
        BackendChatRole::User => llm_api::ChatRole::User,
        BackendChatRole::Assistant => llm_api::ChatRole::Assistant,
        BackendChatRole::Tool => llm_api::ChatRole::Tool,
    }
}

fn mlx_tool_call(tool_call: &BackendToolCall) -> llm_api::ToolCall {
    llm_api::ToolCall {
        id: tool_call.id.clone(),
        call_type: mlx_tool_call_type(&tool_call.call_type),
        function: mlx_tool_call_function(&tool_call.function),
    }
}

fn mlx_tool_call_type(call_type: &BackendToolCallType) -> llm_api::ToolCallType {
    match call_type {
        BackendToolCallType::Function => llm_api::ToolCallType::Function,
    }
}

fn mlx_tool_call_function(function: &BackendToolCallFunction) -> llm_api::ToolCallFunction {
    llm_api::ToolCallFunction {
        name: function.name.clone(),
        arguments: function.arguments.clone(),
    }
}

fn mlx_tool_schema(request: &BackendRequest) -> Result<Option<Value>, BackendError> {
    request
        .cache_context
        .tool_schema
        .as_deref()
        .map(|schema| {
            serde_json::from_str::<Value>(schema).map_err(|err| {
                BackendError::other(format!("MLX tool schema was not valid JSON: {err}"))
            })
        })
        .transpose()
}

fn mlx_tool_choice(request: &BackendRequest) -> Option<Value> {
    request
        .required_tool_choice
        .as_ref()
        .map(|choice| match choice {
            llm_backend::BackendToolChoice::RequiredAny => Value::String("required".to_owned()),
            llm_backend::BackendToolChoice::RequiredFunction(name) => serde_json::json!({
                "type": "function",
                "function": {
                    "name": name,
                },
            }),
        })
}

fn mlx_response_format(request: &BackendRequest) -> Option<Value> {
    request
        .json_object_mode
        .then(|| serde_json::json!({"type": "json_object"}))
}
