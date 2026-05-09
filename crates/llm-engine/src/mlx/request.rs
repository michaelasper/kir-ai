use super::{
    client::mlx_endpoint_url,
    protocol::{
        MlxUpstreamProtocol, mlx_chat_template_kwargs_for_metadata,
        mlx_upstream_protocol_for_request,
    },
};
use llm_backend::{BackendError, BackendModelMetadata, BackendRequest, SamplingConfig};
use serde::Serialize;
use serde_json::Value;
use url::Url;

pub(super) fn build_upstream_request(
    client: &reqwest::Client,
    endpoint: &Url,
    upstream_model: &str,
    metadata: &BackendModelMetadata,
    request: &BackendRequest,
) -> Result<(MlxUpstreamProtocol, reqwest::RequestBuilder), BackendError> {
    let protocol = mlx_upstream_protocol_for_request(metadata, request);
    let (temperature, top_p) = match request.sampling {
        SamplingConfig::Greedy => (0.0, 1.0),
        SamplingConfig::TopP { temperature, top_p } => (temperature, top_p),
    };
    let upstream_url = mlx_endpoint_url(endpoint, protocol.endpoint_suffix());
    let request = match protocol {
        MlxUpstreamProtocol::Completions => client.post(upstream_url).json(&MlxCompletionRequest {
            model: upstream_model,
            prompt: &request.prompt,
            max_tokens: request.max_tokens,
            temperature,
            top_p,
            stream: true,
        }),
        MlxUpstreamProtocol::ChatCompletions => {
            let messages = mlx_chat_messages(request);
            let tools = mlx_tool_schema(request)?;
            let tool_choice = mlx_tool_choice(request);
            let response_format = mlx_response_format(request);
            let chat_template_kwargs = mlx_chat_template_kwargs_for_metadata(metadata);
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
                stream: true,
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
}

#[derive(Debug, Serialize)]
struct MlxChatCompletionRequest<'a> {
    model: &'a str,
    messages: Vec<MlxChatMessage<'a>>,
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
}

#[derive(Debug, Serialize)]
struct MlxChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

fn mlx_chat_messages(request: &BackendRequest) -> Vec<MlxChatMessage<'_>> {
    if let Some(chat_context) = &request.chat_context {
        return chat_context
            .messages
            .iter()
            .map(|message| MlxChatMessage {
                role: message.role.as_str(),
                content: &message.content,
            })
            .collect();
    }
    vec![MlxChatMessage {
        role: "user",
        content: &request.prompt,
    }]
}

fn mlx_tool_schema(request: &BackendRequest) -> Result<Option<Value>, BackendError> {
    request
        .cache_context
        .tool_schema
        .as_deref()
        .map(|schema| {
            serde_json::from_str::<Value>(schema).map_err(|err| {
                BackendError::Other(format!("MLX tool schema was not valid JSON: {err}"))
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
