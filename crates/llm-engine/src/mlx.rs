use async_trait::async_trait;
use futures::{StreamExt, stream::BoxStream};
use llm_backend::{
    BackendError, BackendModelMetadata, BackendOutput, BackendRequest, BackendStreamChunk,
    ModelBackend, SamplingConfig,
};
use llm_models::ModelFamily;
use serde::Serialize;
use serde_json::Value;
use std::path::Path;
use tokio_util::sync::CancellationToken;
use url::Url;

mod client;
mod metadata;
mod sse;

use client::{is_loopback_endpoint, mlx_endpoint_url};
use metadata::mlx_metadata;
use sse::{
    MlxSseParser, count_whitespace_tokens, mlx_control_stop_tokens_for_metadata,
    mlx_tool_markup_for_metadata, mlx_upstream_protocol_for_request,
};

const MLX_QWEN_CONTROL_STOP_TOKENS: &[&str] = &["<|im_end|>", "<|endoftext|>"];
const MLX_DEEPSEEK_CONTROL_STOP_TOKENS: &[&str] =
    &["<｜end▁of▁sentence｜>", "<｜User｜>", "<|endoftext|>"];
const MLX_GEMMA_CONTROL_STOP_TOKENS: &[&str] =
    &["<turn|>", "<|tool_response>", "<eos>", "<|endoftext|>"];

#[derive(Debug, Clone)]
pub struct MlxBackendOptions {
    pub endpoint: Url,
    pub family: Option<ModelFamily>,
}

#[derive(Debug, Clone)]
pub struct MlxBackend {
    model_id: String,
    metadata: BackendModelMetadata,
    upstream_model: String,
    endpoint: Url,
    control_stop_tokens: &'static [&'static str],
    client: reqwest::Client,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MlxUpstreamProtocol {
    Completions,
    ChatCompletions,
}

impl MlxUpstreamProtocol {
    fn endpoint_suffix(self) -> &'static str {
        match self {
            Self::Completions => "completions",
            Self::ChatCompletions => "chat/completions",
        }
    }
}

impl MlxBackend {
    pub fn open(
        model_id: impl Into<String>,
        snapshot_path: impl AsRef<Path>,
    ) -> anyhow::Result<Self> {
        Self::open_with_options(model_id, snapshot_path, MlxBackendOptions::default())
    }

    pub fn open_with_options(
        model_id: impl Into<String>,
        snapshot_path: impl AsRef<Path>,
        options: MlxBackendOptions,
    ) -> anyhow::Result<Self> {
        if !is_loopback_endpoint(&options.endpoint) {
            anyhow::bail!(
                "MLX endpoint `{}` is not loopback; refusing to proxy generation to a remote sidecar",
                options.endpoint
            );
        }
        let model_id = model_id.into();
        let snapshot_path = snapshot_path.as_ref();
        let upstream_model = snapshot_path.canonicalize()?.to_string_lossy().into_owned();
        let metadata = mlx_metadata(&model_id, snapshot_path, options.family)?;
        let control_stop_tokens = mlx_control_stop_tokens_for_metadata(&metadata);
        Ok(Self {
            model_id: model_id.clone(),
            metadata,
            upstream_model,
            endpoint: options.endpoint,
            control_stop_tokens,
            client: reqwest::Client::new(),
        })
    }

    async fn send_completion_request(
        &self,
        request: &BackendRequest,
    ) -> Result<reqwest::Response, BackendError> {
        if request.model != self.model_id {
            return Err(BackendError::ModelNotFound {
                requested: request.model.clone(),
                available: self.model_id.clone(),
            });
        }
        let (temperature, top_p) = match request.sampling {
            SamplingConfig::Greedy => (0.0, 1.0),
            SamplingConfig::TopP { temperature, top_p } => (temperature, top_p),
        };
        let upstream_protocol = mlx_upstream_protocol_for_request(&self.metadata, request);
        let upstream_url = mlx_endpoint_url(&self.endpoint, upstream_protocol.endpoint_suffix());
        let request = match upstream_protocol {
            MlxUpstreamProtocol::Completions => {
                self.client.post(upstream_url).json(&MlxCompletionRequest {
                    model: &self.upstream_model,
                    prompt: &request.prompt,
                    max_tokens: request.max_tokens,
                    temperature,
                    top_p,
                    stream: true,
                })
            }
            MlxUpstreamProtocol::ChatCompletions => {
                let messages = mlx_chat_messages(request);
                let tools = mlx_tool_schema(request)?;
                let tool_choice = mlx_tool_choice(request);
                let response_format = mlx_response_format(request);
                self.client
                    .post(upstream_url)
                    .json(&MlxChatCompletionRequest {
                        model: &self.upstream_model,
                        messages,
                        tools,
                        tool_choice,
                        response_format,
                        max_tokens: request.max_tokens,
                        temperature,
                        top_p,
                        stream: true,
                    })
            }
        };
        request
            .send()
            .await
            .map_err(|err| BackendError::Other(format!("MLX request failed: {err}")))
    }

    async fn generate_once(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        let mut stream = self.stream_completion(request.clone(), cancellation);
        let mut text = String::new();
        let mut prompt_tokens = 0;
        let mut completion_tokens = 0;
        let mut finish_reason = llm_api::FinishReason::Stop;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            prompt_tokens = prompt_tokens.max(chunk.prompt_tokens);
            completion_tokens += chunk.completion_tokens;
            text.push_str(&chunk.text);
            if let Some(reason) = chunk.finish_reason {
                finish_reason = reason;
            }
        }
        if prompt_tokens == 0 {
            prompt_tokens = count_whitespace_tokens(&request.prompt);
        }
        if completion_tokens == 0 && !text.is_empty() {
            completion_tokens = count_whitespace_tokens(&text);
        }
        Ok(BackendOutput {
            prompt_tokens,
            completion_tokens,
            text,
            finish_reason,
        })
    }

    fn stream_completion<'a>(
        &'a self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        async_stream::try_stream! {
            if cancellation.is_cancelled() {
                Err(BackendError::Cancelled)?;
            }
            let response = tokio::select! {
                response = self.send_completion_request(&request) => response,
                _ = cancellation.cancelled() => Err(BackendError::Cancelled),
            };
            let response = response?;
            let status = response.status();
            if status.is_success() {
                let mut bytes = response.bytes_stream();
                let mut parser = MlxSseParser::new(
                    &request.prompt,
                    self.control_stop_tokens,
                    mlx_tool_markup_for_metadata(&self.metadata),
                );
                loop {
                    let item = tokio::select! {
                        item = bytes.next() => Ok(item),
                        _ = cancellation.cancelled() => Err(BackendError::Cancelled),
                    };
                    let item = item?;
                    let Some(item) = item else {
                        break;
                    };
                    let bytes = item
                        .map_err(|err| BackendError::Other(format!("MLX stream read failed: {err}")))?;
                    let chunk = std::str::from_utf8(&bytes)
                        .map_err(|err| BackendError::Other(format!("MLX stream was not UTF-8: {err}")))?;
                    for parsed in parser.push_str(chunk)? {
                        yield parsed;
                    }
                }
                for parsed in parser.finish()? {
                    yield parsed;
                }
            } else {
                let body = tokio::select! {
                    body = response.text() => body
                        .map_err(|err| BackendError::Other(format!("MLX response read failed: {err}"))),
                    _ = cancellation.cancelled() => Err(BackendError::Cancelled),
                };
                let body = body?;
                Err(BackendError::Other(format!(
                    "MLX server returned HTTP {status}: {body}"
                )))?;
            }
        }
        .boxed()
    }
}

impl Default for MlxBackendOptions {
    fn default() -> Self {
        Self {
            endpoint: Url::parse("http://127.0.0.1:8080/v1").expect("valid default MLX endpoint"),
            family: None,
        }
    }
}

#[async_trait]
impl ModelBackend for MlxBackend {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        self.metadata.clone()
    }

    async fn generate(&self, request: BackendRequest) -> Result<BackendOutput, BackendError> {
        self.generate_once(request, CancellationToken::new()).await
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        if cancellation.is_cancelled() {
            return Err(BackendError::Cancelled);
        }
        self.generate_once(request, cancellation).await
    }

    fn generate_stream<'a>(
        &'a self,
        request: BackendRequest,
    ) -> BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        self.generate_stream_with_cancel(request, CancellationToken::new())
    }

    fn generate_stream_with_cancel<'a>(
        &'a self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        self.stream_completion(request, cancellation)
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

#[cfg(test)]
mod tests {
    use super::sse::MlxToolMarkup;
    use super::*;
    use llm_backend::{
        BackendCacheContext, BackendChatContext, BackendChatMessage, BackendChatRole,
        BackendRequest, ModelBackend, SamplingConfig,
    };
    use serde_json::Value;
    use std::{
        io::{Read, Write},
        net::TcpListener,
        sync::{Arc, Mutex},
        thread,
    };
    use tempfile::TempDir;

    type ParsedMlxChunkForTest = (String, u64, u64, Option<llm_api::FinishReason>);

    fn parse_mlx_sse_for_test(
        chunks: &[&str],
        markup: MlxToolMarkup,
    ) -> Result<Vec<ParsedMlxChunkForTest>, BackendError> {
        let mut parser = MlxSseParser::new("hello mlx", MLX_QWEN_CONTROL_STOP_TOKENS, markup);
        let mut parsed = Vec::new();
        for chunk in chunks {
            parsed.extend(parser.push_str(chunk)?);
        }
        parsed.extend(parser.finish()?);
        Ok(parsed
            .into_iter()
            .map(|chunk| {
                (
                    chunk.text,
                    chunk.prompt_tokens,
                    chunk.completion_tokens,
                    chunk.finish_reason,
                )
            })
            .collect())
    }

    #[tokio::test]
    async fn mlx_backend_posts_prompt_to_completion_endpoint() {
        let server = FakeMlxServer::start(
            "data: {\"choices\":[{\"text\":\"MLX says \",\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":3}}\n\ndata: {\"choices\":[{\"text\":\"hi\",\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":4}}\n\ndata: [DONE]\n\n",
        );
        let backend = MlxBackend::open_with_options(
            "local-mlx",
            server.snapshot_path(),
            MlxBackendOptions {
                endpoint: server.endpoint(),
                family: Some(ModelFamily::Qwen),
            },
        )
        .expect("backend opens");

        let output = backend
            .generate(BackendRequest {
                model: "local-mlx".to_owned(),
                prompt: "hello mlx".to_owned(),
                chat_context: None,
                max_tokens: Some(12),
                sampling: SamplingConfig::TopP {
                    temperature: 0.7,
                    top_p: 0.9,
                },
                required_tool_choice: None,
                json_object_mode: false,
                conversation_mode: false,
                cache_context: BackendCacheContext::raw_prompt(),
            })
            .await
            .expect("mlx generation succeeds");

        assert_eq!(output.text, "MLX says hi");
        assert_eq!(output.prompt_tokens, 3);
        assert_eq!(output.completion_tokens, 4);
        let request = server.received_body();
        assert_eq!(
            request["model"],
            server
                .snapshot_path()
                .canonicalize()
                .expect("canonical snapshot")
                .display()
                .to_string()
        );
        assert_eq!(request["prompt"], "hello mlx");
        assert_eq!(request["max_tokens"], 12);
        assert_eq!(request["temperature"], 0.7);
        assert_eq!(request["top_p"], 0.9);
        assert_eq!(request["stream"], true);
    }

    #[tokio::test]
    async fn mlx_backend_posts_gemma_structured_messages_to_chat_completion_endpoint() {
        let server = FakeMlxServer::start(
            "data: {\"choices\":[{\"delta\":{\"content\":\"gemma says hi\"},\"finish_reason\":\"stop\"}],\"usage\":{\"input_tokens\":6,\"output_tokens\":4}}\n\ndata: [DONE]\n\n",
        );
        let backend = MlxBackend::open_with_options(
            "local-mlx",
            server.snapshot_path(),
            MlxBackendOptions {
                endpoint: server.endpoint(),
                family: Some(ModelFamily::Gemma),
            },
        )
        .expect("backend opens");

        let output = backend
            .generate(BackendRequest {
                model: "local-mlx".to_owned(),
                prompt: "<bos><|turn>user\nhello gemma<turn|>\n<|turn>model\n".to_owned(),
                chat_context: Some(BackendChatContext {
                    messages: vec![
                        BackendChatMessage {
                            role: BackendChatRole::System,
                            content: "You are Kir.".to_owned(),
                        },
                        BackendChatMessage {
                            role: BackendChatRole::User,
                            content: "hello gemma".to_owned(),
                        },
                    ],
                }),
                max_tokens: Some(12),
                sampling: SamplingConfig::Greedy,
                required_tool_choice: None,
                json_object_mode: false,
                conversation_mode: false,
                cache_context: BackendCacheContext::raw_prompt(),
            })
            .await
            .expect("mlx generation succeeds");

        assert_eq!(output.text, "gemma says hi");
        assert_eq!(output.prompt_tokens, 6);
        assert_eq!(output.completion_tokens, 4);
        let request = server.received_body();
        assert_eq!(
            request["model"].as_str(),
            Some(backend.upstream_model.as_str())
        );
        assert_eq!(request["messages"][0]["role"], "system");
        assert_eq!(request["messages"][0]["content"], "You are Kir.");
        assert_eq!(request["messages"][1]["role"], "user");
        assert_eq!(request["messages"][1]["content"], "hello gemma");
        assert_eq!(request["stream"], true);
    }

    #[tokio::test]
    async fn mlx_backend_posts_tool_schema_with_structured_chat_messages() {
        let server = FakeMlxServer::start(
            "data: {\"choices\":[{\"delta\":{\"content\":\"tool fallback\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
        );
        let backend = MlxBackend::open_with_options(
            "local-mlx",
            server.snapshot_path(),
            MlxBackendOptions {
                endpoint: server.endpoint(),
                family: Some(ModelFamily::Gemma),
            },
        )
        .expect("backend opens");

        let output = backend
            .generate(BackendRequest {
                model: "local-mlx".to_owned(),
                prompt: "<bos><|turn>user\nuse lookup<turn|>\n<|turn>model\n".to_owned(),
                chat_context: Some(BackendChatContext {
                    messages: vec![BackendChatMessage {
                        role: BackendChatRole::User,
                        content: "use lookup".to_owned(),
                    }],
                }),
                max_tokens: Some(12),
                sampling: SamplingConfig::Greedy,
                required_tool_choice: None,
                json_object_mode: false,
                conversation_mode: true,
                cache_context: BackendCacheContext::chat_template(
                    "gemma/gemma4/v1",
                    Some(r#"[{"type":"function"}]"#.to_owned()),
                ),
            })
            .await
            .expect("mlx generation succeeds");

        assert_eq!(output.text, "tool fallback");
        let request = server.received_body();
        assert_eq!(request["messages"][0]["role"], "user");
        assert_eq!(request["messages"][0]["content"], "use lookup");
        assert_eq!(request["tools"][0]["type"], "function");
        assert_eq!(
            request["messages"]
                .as_array()
                .expect("messages array")
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn mlx_backend_routes_deepseek_chat_to_chat_completion_endpoint() {
        let server = FakeMlxServer::start(
            "data: {\"choices\":[{\"delta\":{\"content\":\"deepseek says hi\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":6,\"completion_tokens\":4}}\n\ndata: [DONE]\n\n",
        );
        let backend = MlxBackend::open_with_options(
            "local-mlx",
            server.snapshot_path(),
            MlxBackendOptions {
                endpoint: server.endpoint(),
                family: Some(ModelFamily::DeepSeek),
            },
        )
        .expect("backend opens");

        let output = backend
            .generate(BackendRequest {
                model: "local-mlx".to_owned(),
                prompt: "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜>".to_owned(),
                chat_context: Some(BackendChatContext {
                    messages: vec![BackendChatMessage {
                        role: BackendChatRole::User,
                        content: "hello".to_owned(),
                    }],
                }),
                max_tokens: Some(12),
                sampling: SamplingConfig::Greedy,
                required_tool_choice: Some(llm_backend::BackendToolChoice::RequiredFunction(
                    "lookup".to_owned(),
                )),
                json_object_mode: false,
                conversation_mode: true,
                cache_context: BackendCacheContext::chat_template(
                    "deepseek/chat/v1",
                    Some(
                        r#"[{"type":"function","function":{"name":"lookup","parameters":{}}}]"#
                            .to_owned(),
                    ),
                ),
            })
            .await
            .expect("mlx generation succeeds");

        assert_eq!(output.text, "deepseek says hi");
        assert_eq!(server.received_path(), "/v1/chat/completions");
        let request = server.received_body();
        assert_eq!(request["messages"][0]["role"], "user");
        assert_eq!(request["messages"][0]["content"], "hello");
        assert_eq!(request["tools"][0]["function"]["name"], "lookup");
        assert_eq!(
            request["tool_choice"],
            serde_json::json!({"type":"function","function":{"name":"lookup"}})
        );
    }

    #[tokio::test]
    async fn mlx_backend_posts_json_object_response_format_to_chat_completion_endpoint() {
        let server = FakeMlxServer::start(
            "data: {\"choices\":[{\"delta\":{\"content\":\"{\\\"ok\\\":true}\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2}}\n\ndata: [DONE]\n\n",
        );
        let backend = MlxBackend::open_with_options(
            "local-mlx",
            server.snapshot_path(),
            MlxBackendOptions {
                endpoint: server.endpoint(),
                family: Some(ModelFamily::Qwen),
            },
        )
        .expect("backend opens");

        let output = backend
            .generate(BackendRequest {
                model: "local-mlx".to_owned(),
                prompt: "<|im_start|>user\nreturn json<|im_end|>\n<|im_start|>assistant\n"
                    .to_owned(),
                chat_context: Some(BackendChatContext {
                    messages: vec![BackendChatMessage {
                        role: BackendChatRole::User,
                        content: "return json".to_owned(),
                    }],
                }),
                max_tokens: Some(12),
                sampling: SamplingConfig::Greedy,
                required_tool_choice: None,
                json_object_mode: true,
                conversation_mode: true,
                cache_context: BackendCacheContext::chat_template("chatml/qwen/v1", None),
            })
            .await
            .expect("mlx generation succeeds");

        assert_eq!(output.text, "{\"ok\":true}");
        assert_eq!(server.received_path(), "/v1/chat/completions");
        assert_eq!(
            server.received_body()["response_format"],
            serde_json::json!({"type":"json_object"})
        );
    }

    #[tokio::test]
    async fn mlx_backend_strips_control_stop_tokens_from_completion_text() {
        let server = FakeMlxServer::start(
            "data: {\"choices\":[{\"text\":\"otter:19<|im_end|>\\n\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":6}}\n\ndata: [DONE]\n\n",
        );
        let backend = MlxBackend::open_with_options(
            "local-mlx",
            server.snapshot_path(),
            MlxBackendOptions {
                endpoint: server.endpoint(),
                family: Some(ModelFamily::Qwen),
            },
        )
        .expect("backend opens");

        let output = backend
            .generate(BackendRequest {
                model: "local-mlx".to_owned(),
                prompt: "hello mlx".to_owned(),
                chat_context: None,
                max_tokens: Some(12),
                sampling: SamplingConfig::Greedy,
                required_tool_choice: None,
                json_object_mode: false,
                conversation_mode: false,
                cache_context: BackendCacheContext::raw_prompt(),
            })
            .await
            .expect("mlx generation succeeds");

        assert_eq!(output.text, "otter:19");
        assert_eq!(output.finish_reason, llm_api::FinishReason::Stop);
    }

    #[tokio::test]
    async fn mlx_backend_strips_split_control_stop_tokens_from_stream() {
        let server = FakeMlxServer::start(
            "data:{\"choices\":[{\"text\":\"otter:19<|im\",\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":2}}\n\ndata: {\"choices\":[{\"text\":\"_end|>\\n\",\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":6}}\n\ndata: [DONE]\n\n",
        );
        let backend = MlxBackend::open_with_options(
            "local-mlx",
            server.snapshot_path(),
            MlxBackendOptions {
                endpoint: server.endpoint(),
                family: Some(ModelFamily::Qwen),
            },
        )
        .expect("backend opens");

        let chunks = backend
            .generate_stream(BackendRequest {
                model: "local-mlx".to_owned(),
                prompt: "hello mlx".to_owned(),
                chat_context: None,
                max_tokens: Some(12),
                sampling: SamplingConfig::Greedy,
                required_tool_choice: None,
                json_object_mode: false,
                conversation_mode: false,
                cache_context: BackendCacheContext::raw_prompt(),
            })
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .expect("mlx stream succeeds");

        let text = chunks
            .iter()
            .map(|chunk| chunk.text.as_str())
            .collect::<String>();
        assert_eq!(text, "otter:19");
        assert_eq!(
            chunks.last().and_then(|chunk| chunk.finish_reason.clone()),
            Some(llm_api::FinishReason::Stop)
        );
    }

    #[tokio::test]
    async fn mlx_backend_strips_gemma_control_stop_tokens_from_completion_text() {
        let server = FakeMlxServer::start(
            "data: {\"choices\":[{\"text\":\"hello from gemma<turn|>\\n\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":5}}\n\ndata: [DONE]\n\n",
        );
        let backend = MlxBackend::open_with_options(
            "local-mlx",
            server.snapshot_path(),
            MlxBackendOptions {
                endpoint: server.endpoint(),
                family: Some(ModelFamily::Gemma),
            },
        )
        .expect("backend opens");

        let output = backend
            .generate(BackendRequest {
                model: "local-mlx".to_owned(),
                prompt: "hello gemma".to_owned(),
                chat_context: None,
                max_tokens: Some(12),
                sampling: SamplingConfig::Greedy,
                required_tool_choice: None,
                json_object_mode: false,
                conversation_mode: false,
                cache_context: BackendCacheContext::raw_prompt(),
            })
            .await
            .expect("mlx generation succeeds");

        assert_eq!(output.text, "hello from gemma");
        assert_eq!(output.finish_reason, llm_api::FinishReason::Stop);
    }

    #[test]
    fn mlx_sse_parser_flushes_non_stop_prefix_at_done() {
        let mut parser = MlxSseParser::new(
            "hello mlx",
            MLX_QWEN_CONTROL_STOP_TOKENS,
            MlxToolMarkup::Qwen,
        );
        let chunks = parser
            .push_str(
                "data:{\"choices\":[{\"text\":\"keep <|im\",\"finish_reason\":null}]}\n\ndata:[DONE]\n\n",
            )
            .expect("parse chunk");
        let final_chunks = parser.finish().expect("finish parser");
        let chunks = chunks.into_iter().chain(final_chunks).collect::<Vec<_>>();

        let text = chunks
            .iter()
            .map(|chunk| chunk.text.as_str())
            .collect::<String>();
        assert_eq!(text, "keep <|im");
        assert_eq!(
            chunks
                .iter()
                .map(|chunk| chunk.completion_tokens)
                .sum::<u64>(),
            2
        );
    }

    #[test]
    fn mlx_sse_parser_handles_deepseek_non_ascii_stop_prefix_checks() {
        let mut parser = MlxSseParser::new(
            "hello deepseek",
            MLX_DEEPSEEK_CONTROL_STOP_TOKENS,
            MlxToolMarkup::DeepSeek,
        );
        let chunks = parser
            .push_str("data:{\"choices\":[{\"text\":\"plain answer\",\"finish_reason\":null}]}\n\n")
            .expect("DeepSeek parser does not panic while checking non-ASCII stop tokens");

        assert_eq!(chunks[0].text, "plain answer");
    }

    #[test]
    fn mlx_sse_parser_strips_split_deepseek_control_stop_tokens() {
        let mut parser = MlxSseParser::new(
            "hello deepseek",
            MLX_DEEPSEEK_CONTROL_STOP_TOKENS,
            MlxToolMarkup::DeepSeek,
        );
        let chunks = parser
            .push_str(
                "data:{\"choices\":[{\"text\":\"answer <｜end\",\"finish_reason\":null}]}\n\n",
            )
            .expect("first split chunk parses");
        let next_chunks = parser
            .push_str(
                "data:{\"choices\":[{\"text\":\"▁of▁sentence｜> ignored\",\"finish_reason\":\"stop\"}]}\n\ndata:[DONE]\n\n",
            )
            .expect("second split chunk parses");
        let final_chunks = parser.finish().expect("finish parser");
        let text = chunks
            .into_iter()
            .chain(next_chunks)
            .chain(final_chunks)
            .map(|chunk| chunk.text)
            .collect::<String>();

        assert_eq!(text, "answer ");
    }

    #[test]
    fn mlx_sse_parser_is_chunk_boundary_invariant_for_tool_calls() {
        let payload = "data:{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"read_\",\"arguments\":\"{\\\"path\\\"\"}}]},\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":4}}\n\ndata:{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"file\",\"arguments\":\":\\\"Cargo.toml\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"completion_tokens\":5}}\n\ndata:[DONE]\n\n";
        let expected =
            parse_mlx_sse_for_test(&[payload], MlxToolMarkup::Qwen).expect("single chunk parses");

        for split in payload
            .char_indices()
            .map(|(index, _)| index)
            .chain(std::iter::once(payload.len()))
        {
            let actual = parse_mlx_sse_for_test(
                &[&payload[..split], &payload[split..]],
                MlxToolMarkup::Qwen,
            )
            .unwrap_or_else(|err| panic!("split at byte {split} failed: {err}"));
            assert_eq!(actual, expected, "split at byte {split}");
        }
    }

    #[test]
    fn mlx_production_module_does_not_depend_on_protocol_test_backend() {
        let source = include_str!("mlx.rs");
        let production_source = source.split("#[cfg(test)]").next().unwrap_or(source);
        assert!(!production_source.contains("ProtocolTestBackend"));
        assert!(!production_source.contains("protocol_test"));
        assert!(!production_source.contains("fixture"));
    }

    #[tokio::test]
    async fn mlx_backend_streams_completion_chunks() {
        let server = FakeMlxServer::start(
            "data:{\"choices\":[{\"text\":\"one \",\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":2}}\n\ndata: {\"choices\":[{\"text\":\"two\",\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":3}}\n\ndata: [DONE]\n\n",
        );
        let backend = MlxBackend::open_with_options(
            "local-mlx",
            server.snapshot_path(),
            MlxBackendOptions {
                endpoint: server.endpoint(),
                family: Some(ModelFamily::Qwen),
            },
        )
        .expect("backend opens");

        let mut stream = backend.generate_stream(BackendRequest {
            model: "local-mlx".to_owned(),
            prompt: "hello mlx".to_owned(),
            chat_context: None,
            max_tokens: Some(12),
            sampling: SamplingConfig::Greedy,
            required_tool_choice: None,
            json_object_mode: false,
            conversation_mode: false,
            cache_context: BackendCacheContext::raw_prompt(),
        });

        let first = stream
            .next()
            .await
            .expect("first stream item")
            .expect("first chunk");
        let second = stream
            .next()
            .await
            .expect("second stream item")
            .expect("second chunk");
        assert!(stream.next().await.is_none());

        assert_eq!(first.text, "one ");
        assert_eq!(first.prompt_tokens, 2);
        assert_eq!(first.completion_tokens, 0);
        assert_eq!(first.finish_reason, None);
        assert_eq!(second.text, "two");
        assert_eq!(second.completion_tokens, 3);
        assert_eq!(second.finish_reason, Some(llm_api::FinishReason::Stop));
    }

    #[tokio::test]
    async fn mlx_backend_preserves_structured_qwen_tool_call_response() {
        let server = FakeMlxServer::start(
            "data:{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"Cargo.toml\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":5}}\n\ndata:[DONE]\n\n",
        );
        let backend = MlxBackend::open_with_options(
            "local-mlx",
            server.snapshot_path(),
            MlxBackendOptions {
                endpoint: server.endpoint(),
                family: Some(ModelFamily::Qwen),
            },
        )
        .expect("backend opens");

        let output = backend
            .generate(BackendRequest {
                model: "local-mlx".to_owned(),
                prompt: "read a file".to_owned(),
                chat_context: Some(BackendChatContext {
                    messages: vec![BackendChatMessage {
                        role: BackendChatRole::User,
                        content: "read a file".to_owned(),
                    }],
                }),
                max_tokens: Some(12),
                sampling: SamplingConfig::Greedy,
                required_tool_choice: Some(llm_backend::BackendToolChoice::RequiredFunction(
                    "read_file".to_owned(),
                )),
                json_object_mode: false,
                conversation_mode: true,
                cache_context: BackendCacheContext::chat_template(
                    "chatml/qwen/v1",
                    Some(
                        r#"[{"type":"function","function":{"name":"read_file","parameters":{}}}]"#
                            .to_owned(),
                    ),
                ),
            })
            .await
            .expect("mlx generation succeeds");

        assert_eq!(server.received_path(), "/v1/chat/completions");
        let request = server.received_body();
        assert_eq!(request["messages"][0]["role"], "user");
        assert_eq!(request["messages"][0]["content"], "read a file");
        assert_eq!(request["tools"][0]["function"]["name"], "read_file");
        assert_eq!(
            request["tool_choice"],
            serde_json::json!({"type":"function","function":{"name":"read_file"}})
        );
        assert_eq!(output.finish_reason, llm_api::FinishReason::ToolCalls);
        assert!(output.text.starts_with("<tool_call>"));
        assert!(output.text.contains("\"name\":\"read_file\""));
        assert!(output.text.contains("\"path\":\"Cargo.toml\""));
    }

    #[tokio::test]
    async fn mlx_backend_accumulates_streamed_tool_call_fragments() {
        let server = FakeMlxServer::start(
            "data:{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"read_\",\"arguments\":\"{\\\"path\\\"\"}}]},\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":4}}\n\ndata:{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"file\",\"arguments\":\":\\\"Cargo.toml\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"completion_tokens\":5}}\n\ndata:[DONE]\n\n",
        );
        let backend = MlxBackend::open_with_options(
            "local-mlx",
            server.snapshot_path(),
            MlxBackendOptions {
                endpoint: server.endpoint(),
                family: Some(ModelFamily::Qwen),
            },
        )
        .expect("backend opens");

        let chunks = backend
            .generate_stream(BackendRequest {
                model: "local-mlx".to_owned(),
                prompt: "read a file".to_owned(),
                chat_context: None,
                max_tokens: Some(12),
                sampling: SamplingConfig::Greedy,
                required_tool_choice: None,
                json_object_mode: false,
                conversation_mode: false,
                cache_context: BackendCacheContext::raw_prompt(),
            })
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .expect("mlx stream succeeds");

        let text = chunks
            .iter()
            .map(|chunk| chunk.text.as_str())
            .collect::<String>();
        assert!(text.contains("\"name\":\"read_file\""));
        assert!(text.contains("\"path\":\"Cargo.toml\""));
        assert_eq!(
            chunks.last().and_then(|chunk| chunk.finish_reason.clone()),
            Some(llm_api::FinishReason::ToolCalls)
        );
    }

    #[tokio::test]
    async fn mlx_backend_preserves_structured_gemma_tool_call_response() {
        let server = FakeMlxServer::start(
            "data:{\"choices\":[{\"message\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"lookup\",\"arguments\":\"{\\\"query\\\":\\\"rust\\\",\\\"limit\\\":3}\"}}]},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"input_tokens\":4,\"output_tokens\":5}}\n\ndata:[DONE]\n\n",
        );
        let backend = MlxBackend::open_with_options(
            "local-mlx",
            server.snapshot_path(),
            MlxBackendOptions {
                endpoint: server.endpoint(),
                family: Some(ModelFamily::Gemma),
            },
        )
        .expect("backend opens");

        let output = backend
            .generate(BackendRequest {
                model: "local-mlx".to_owned(),
                prompt: "lookup rust".to_owned(),
                chat_context: None,
                max_tokens: Some(12),
                sampling: SamplingConfig::Greedy,
                required_tool_choice: None,
                json_object_mode: false,
                conversation_mode: true,
                cache_context: BackendCacheContext::raw_prompt(),
            })
            .await
            .expect("mlx generation succeeds");

        assert_eq!(output.finish_reason, llm_api::FinishReason::ToolCalls);
        assert!(output.text.starts_with("<|tool_call>call:lookup"));
        assert!(output.text.contains("\"query\":\"rust\""));
        assert!(output.text.contains("\"limit\":3"));
    }

    #[tokio::test]
    async fn mlx_backend_preserves_structured_deepseek_tool_call_response() {
        let server = FakeMlxServer::start(
            "data:{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"lookup\",\"arguments\":\"{\\\"query\\\":\\\"metal\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":5}}\n\ndata:[DONE]\n\n",
        );
        let backend = MlxBackend::open_with_options(
            "local-mlx",
            server.snapshot_path(),
            MlxBackendOptions {
                endpoint: server.endpoint(),
                family: Some(ModelFamily::DeepSeek),
            },
        )
        .expect("backend opens");

        let output = backend
            .generate(BackendRequest {
                model: "local-mlx".to_owned(),
                prompt: "lookup metal".to_owned(),
                chat_context: None,
                max_tokens: Some(12),
                sampling: SamplingConfig::Greedy,
                required_tool_choice: None,
                json_object_mode: false,
                conversation_mode: true,
                cache_context: BackendCacheContext::raw_prompt(),
            })
            .await
            .expect("mlx generation succeeds");

        assert_eq!(output.finish_reason, llm_api::FinishReason::ToolCalls);
        assert!(output.text.starts_with("<｜tool▁calls▁begin｜>"));
        assert!(output.text.contains("<｜tool▁sep｜>lookup"));
        assert!(output.text.contains("\"query\":\"metal\""));
    }

    #[tokio::test]
    async fn mlx_backend_rejects_model_mismatch_before_http_request() {
        let snapshot = tempfile::tempdir().expect("snapshot tempdir");
        let backend = MlxBackend::open_with_options(
            "local-mlx",
            snapshot.path(),
            MlxBackendOptions {
                family: Some(ModelFamily::Qwen),
                ..MlxBackendOptions::default()
            },
        )
        .expect("backend opens");

        let err = backend
            .generate(BackendRequest {
                model: "other-model".to_owned(),
                prompt: "hello".to_owned(),
                chat_context: None,
                max_tokens: Some(1),
                sampling: SamplingConfig::Greedy,
                required_tool_choice: None,
                json_object_mode: false,
                conversation_mode: false,
                cache_context: BackendCacheContext::raw_prompt(),
            })
            .await
            .expect_err("model mismatch fails before HTTP");

        assert!(matches!(err, BackendError::ModelNotFound { .. }));
    }

    #[test]
    fn mlx_backend_rejects_non_loopback_endpoint() {
        let snapshot = tempfile::tempdir().expect("snapshot tempdir");

        let err = MlxBackend::open_with_options(
            "local-mlx",
            snapshot.path(),
            MlxBackendOptions {
                endpoint: Url::parse("https://example.com/v1").expect("url"),
                ..MlxBackendOptions::default()
            },
        )
        .expect_err("remote MLX endpoint is rejected");

        assert!(err.to_string().contains("not loopback"));
    }

    #[test]
    fn mlx_backend_rejects_manifestless_snapshot_without_family() {
        let snapshot = tempfile::tempdir().expect("snapshot tempdir");

        let err = MlxBackend::open_with_options(
            "local-mlx",
            snapshot.path(),
            MlxBackendOptions {
                endpoint: Url::parse("http://127.0.0.1:18080/v1").expect("url"),
                ..MlxBackendOptions::default()
            },
        )
        .expect_err("raw MLX family is required");

        assert!(
            err.to_string()
                .contains("MLX backend requires model family metadata")
        );
    }

    #[test]
    fn mlx_backend_accepts_gemma_requested_family() {
        let snapshot = tempfile::tempdir().expect("snapshot tempdir");

        let backend = MlxBackend::open_with_options(
            "local-mlx",
            snapshot.path(),
            MlxBackendOptions {
                endpoint: Url::parse("http://127.0.0.1:18080/v1").expect("url"),
                family: Some(ModelFamily::Gemma),
            },
        )
        .expect("Gemma MLX backend opens");

        assert_eq!(backend.model_metadata().family.as_deref(), Some("gemma"));
        assert_eq!(backend.model_metadata().loader.as_deref(), Some("mlx"));
    }

    #[test]
    fn mlx_backend_rejects_non_mlx_manifest_loader() {
        let snapshot = tempfile::tempdir().expect("snapshot tempdir");
        write_mlx_manifest(snapshot.path(), "native-metal", "qwen");

        let err = MlxBackend::open_with_options(
            "local-mlx",
            snapshot.path(),
            MlxBackendOptions {
                endpoint: Url::parse("http://127.0.0.1:18080/v1").expect("url"),
                ..MlxBackendOptions::default()
            },
        )
        .expect_err("MLX backend rejects native manifest loader");

        assert!(
            err.to_string()
                .contains("MLX backend requires manifest loader `mlx`")
        );
    }

    #[test]
    fn mlx_backend_rejects_unknown_manifest_family() {
        let snapshot = tempfile::tempdir().expect("snapshot tempdir");
        write_mlx_manifest(snapshot.path(), "mlx", "llama");

        let err = MlxBackend::open_with_options(
            "local-mlx",
            snapshot.path(),
            MlxBackendOptions {
                endpoint: Url::parse("http://127.0.0.1:18080/v1").expect("url"),
                ..MlxBackendOptions::default()
            },
        )
        .expect_err("unknown manifest family is rejected");

        assert!(err.to_string().contains("unsupported model family `llama`"));
    }

    #[tokio::test]
    async fn mlx_backend_rejects_sse_without_done_marker() {
        let server = FakeMlxServer::start(
            "data: {\"choices\":[{\"text\":\"partial\",\"finish_reason\":\"stop\"}]}\n\n",
        );
        let backend = MlxBackend::open_with_options(
            "local-mlx",
            server.snapshot_path(),
            MlxBackendOptions {
                endpoint: server.endpoint(),
                family: Some(ModelFamily::Qwen),
            },
        )
        .expect("backend opens");

        let err = backend
            .generate(BackendRequest {
                model: "local-mlx".to_owned(),
                prompt: "hello".to_owned(),
                chat_context: None,
                max_tokens: Some(1),
                sampling: SamplingConfig::Greedy,
                required_tool_choice: None,
                json_object_mode: false,
                conversation_mode: false,
                cache_context: BackendCacheContext::raw_prompt(),
            })
            .await
            .expect_err("missing DONE fails closed");

        assert!(err.to_string().contains("[DONE]"));
    }

    struct FakeMlxServer {
        endpoint: Url,
        snapshot: TempDir,
        received: Arc<Mutex<Option<Value>>>,
        received_path: Arc<Mutex<Option<String>>>,
        join: Option<thread::JoinHandle<()>>,
    }

    impl FakeMlxServer {
        fn start(response_body: &'static str) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind fake mlx server");
            let endpoint = Url::parse(&format!(
                "http://{}/v1",
                listener.local_addr().expect("addr")
            ))
            .expect("endpoint url");
            let received = Arc::new(Mutex::new(None));
            let received_path = Arc::new(Mutex::new(None));
            let received_for_thread = received.clone();
            let received_path_for_thread = received_path.clone();
            let join = thread::spawn(move || {
                let (mut stream, _) = listener.accept().expect("accept fake mlx request");
                let mut bytes = Vec::new();
                let mut buffer = [0_u8; 1024];
                let header_end;
                loop {
                    let read = stream.read(&mut buffer).expect("read request");
                    assert!(read > 0, "client closed before headers");
                    bytes.extend_from_slice(&buffer[..read]);
                    if let Some(index) = find_subsequence(&bytes, b"\r\n\r\n") {
                        header_end = index + 4;
                        break;
                    }
                }
                let headers = String::from_utf8_lossy(&bytes[..header_end]);
                let request_path = headers
                    .lines()
                    .next()
                    .and_then(|line| line.split_whitespace().nth(1))
                    .expect("request path")
                    .to_owned();
                *received_path_for_thread.lock().expect("received path lock") = Some(request_path);
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().expect("content length"))
                    })
                    .expect("content-length header");
                while bytes.len() < header_end + content_length {
                    let read = stream.read(&mut buffer).expect("read body");
                    assert!(read > 0, "client closed before body");
                    bytes.extend_from_slice(&buffer[..read]);
                }
                let body = &bytes[header_end..header_end + content_length];
                *received_for_thread.lock().expect("received lock") =
                    Some(serde_json::from_slice(body).expect("json request body"));
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response_body.len(),
                    response_body
                )
                .expect("write response");
            });
            Self {
                endpoint,
                snapshot: tempfile::tempdir().expect("snapshot tempdir"),
                received,
                received_path,
                join: Some(join),
            }
        }

        fn endpoint(&self) -> Url {
            self.endpoint.clone()
        }

        fn snapshot_path(&self) -> &Path {
            self.snapshot.path()
        }

        fn received_body(&self) -> Value {
            self.received
                .lock()
                .expect("received lock")
                .clone()
                .expect("received request body")
        }

        fn received_path(&self) -> String {
            self.received_path
                .lock()
                .expect("received path lock")
                .clone()
                .expect("received request path")
        }
    }

    impl Drop for FakeMlxServer {
        fn drop(&mut self) {
            if let Some(join) = self.join.take() {
                join.join().expect("fake server thread");
            }
        }
    }

    fn find_subsequence(bytes: &[u8], needle: &[u8]) -> Option<usize> {
        bytes
            .windows(needle.len())
            .position(|window| window == needle)
    }

    fn write_mlx_manifest(snapshot_path: &Path, loader: &str, family: &str) {
        std::fs::write(
            snapshot_path.join("llm-engine-manifest.json"),
            serde_json::json!({
                "schema_version": 1,
                "source": "huggingface",
                "repo_type": "model",
                "repo_id": "example/model",
                "requested_revision": "main",
                "resolved_commit": "0123456789abcdef0123456789abcdef01234567",
                "profile": "test-mlx",
                "family": family,
                "loader": loader,
                "quantization": "4bit",
                "created_at": "2026-05-08T00:00:00Z",
                "snapshot_path": snapshot_path.display().to_string(),
                "files": [],
                "allow_patterns": [],
                "ignore_patterns": []
            })
            .to_string(),
        )
        .expect("manifest");
    }
}
