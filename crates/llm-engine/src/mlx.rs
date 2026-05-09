use async_trait::async_trait;
use futures::{StreamExt, stream::BoxStream};
use llm_backend::{
    BackendError, BackendModelMetadata, BackendOutput, BackendRequest, BackendStreamChunk,
    ModelBackend, SamplingConfig,
};
use llm_hub::SnapshotManifest;
use llm_models::{BackendKind, ModelFamily};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};
use tokio_util::sync::CancellationToken;
use url::Url;

const MLX_QWEN_CONTROL_STOP_TOKENS: &[&str] = &["<|im_end|>", "<|endoftext|>"];
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
    upstream_url: Url,
    upstream_protocol: MlxUpstreamProtocol,
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
        let upstream_protocol = mlx_upstream_protocol_for_metadata(&metadata);
        let upstream_url = mlx_endpoint_url(&options.endpoint, upstream_protocol.endpoint_suffix());
        let control_stop_tokens = mlx_control_stop_tokens_for_metadata(&metadata);
        Ok(Self {
            model_id: model_id.clone(),
            metadata,
            upstream_model,
            upstream_url,
            upstream_protocol,
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
        let request = match self.upstream_protocol {
            MlxUpstreamProtocol::Completions => {
                self.client
                    .post(self.upstream_url.clone())
                    .json(&MlxCompletionRequest {
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
                self.client
                    .post(self.upstream_url.clone())
                    .json(&MlxChatCompletionRequest {
                        model: &self.upstream_model,
                        messages,
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
    if let (None, Some(chat_context)) = (&request.cache_context.tool_schema, &request.chat_context)
    {
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

#[derive(Debug, Deserialize)]
struct MlxCompletionResponse {
    choices: Vec<MlxCompletionChoice>,
    usage: Option<MlxUsage>,
}

#[derive(Debug, Deserialize)]
struct MlxCompletionChoice {
    text: Option<String>,
    message: Option<MlxMessage>,
    delta: Option<MlxMessage>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MlxMessage {
    content: Option<String>,
    tool_calls: Option<Vec<MlxToolCall>>,
}

#[derive(Debug, Deserialize)]
struct MlxToolCall {
    index: Option<usize>,
    function: Option<MlxToolCallFunction>,
}

#[derive(Debug, Deserialize)]
struct MlxToolCallFunction {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MlxUsage {
    #[serde(alias = "input_tokens")]
    prompt_tokens: Option<u64>,
    #[serde(alias = "output_tokens")]
    completion_tokens: Option<u64>,
}

impl MlxCompletionResponse {
    fn first_choice(self) -> Result<MlxCompletionChoice, BackendError> {
        self.choices
            .into_iter()
            .next()
            .ok_or_else(|| BackendError::Other("MLX completion response had no choices".to_owned()))
    }
}

#[derive(Debug)]
struct MlxSseParser {
    prompt_tokens: u64,
    estimated_completion_tokens: u64,
    emitted_completion_tokens: u64,
    uses_upstream_usage: bool,
    saw_done: bool,
    line_buffer: String,
    stop_filter: MlxControlStopFilter,
    tool_markup: MlxToolMarkup,
    tool_calls: Vec<MlxToolCallAccumulator>,
}

impl MlxSseParser {
    fn new(prompt: &str, stop_tokens: &'static [&'static str], tool_markup: MlxToolMarkup) -> Self {
        Self {
            prompt_tokens: count_whitespace_tokens(prompt),
            estimated_completion_tokens: 0,
            emitted_completion_tokens: 0,
            uses_upstream_usage: false,
            saw_done: false,
            line_buffer: String::new(),
            stop_filter: MlxControlStopFilter::new(stop_tokens),
            tool_markup,
            tool_calls: Vec::new(),
        }
    }

    fn push_str(&mut self, chunk: &str) -> Result<Vec<BackendStreamChunk>, BackendError> {
        self.line_buffer.push_str(chunk);
        let mut chunks = Vec::new();
        while let Some(index) = self.line_buffer.find('\n') {
            let mut line = self.line_buffer.drain(..=index).collect::<String>();
            if line.ends_with('\n') {
                line.pop();
            }
            if line.ends_with('\r') {
                line.pop();
            }
            if let Some(chunk) = self.parse_line(&line)? {
                chunks.push(chunk);
            }
        }
        Ok(chunks)
    }

    fn finish(&mut self) -> Result<Vec<BackendStreamChunk>, BackendError> {
        let mut chunks = Vec::new();
        if !self.line_buffer.is_empty() {
            let line = std::mem::take(&mut self.line_buffer);
            if let Some(chunk) = self.parse_line(line.trim_end_matches('\r'))? {
                chunks.push(chunk);
            }
        }
        if !self.saw_done {
            return Err(BackendError::Other(
                "MLX SSE completion ended before data: [DONE]".to_owned(),
            ));
        }
        let text = self.stop_filter.finish();
        if !text.is_empty() {
            self.estimated_completion_tokens += count_visible_tokens(&text);
            let completion_tokens = self.completion_token_delta(None, true);
            chunks.push(BackendStreamChunk {
                text,
                prompt_tokens: self.prompt_tokens,
                completion_tokens,
                finish_reason: None,
            });
        }
        Ok(chunks)
    }

    fn parse_line(&mut self, line: &str) -> Result<Option<BackendStreamChunk>, BackendError> {
        let Some(data) = mlx_sse_data(line) else {
            return Ok(None);
        };
        if data == "[DONE]" {
            self.saw_done = true;
            return Ok(None);
        }
        let completion = serde_json::from_str::<MlxCompletionResponse>(data).map_err(|err| {
            BackendError::Other(format!("invalid MLX SSE completion JSON: {err}"))
        })?;
        if let Some(prompt_tokens) = completion
            .usage
            .as_ref()
            .and_then(|usage| usage.prompt_tokens)
        {
            self.prompt_tokens = self.prompt_tokens.max(prompt_tokens);
        }
        let usage_completion_tokens = completion
            .usage
            .as_ref()
            .and_then(|usage| usage.completion_tokens);
        let choice = completion.first_choice()?;
        if let Some(tool_calls) = choice
            .delta
            .as_ref()
            .and_then(|message| message.tool_calls.as_ref())
            .or_else(|| {
                choice
                    .message
                    .as_ref()
                    .and_then(|message| message.tool_calls.as_ref())
            })
        {
            self.push_tool_calls(tool_calls);
        }
        let text = choice
            .text
            .or_else(|| choice.delta.and_then(|message| message.content))
            .or_else(|| choice.message.and_then(|message| message.content))
            .unwrap_or_default();
        let mut text = self.stop_filter.push_str(&text);
        self.estimated_completion_tokens += count_visible_tokens(&text);
        let finish_reason = choice
            .finish_reason
            .as_deref()
            .map(|reason| mlx_finish_reason(Some(reason)))
            .transpose()?;
        let finish_reason = finish_reason.or_else(|| {
            self.stop_filter
                .is_stopped()
                .then_some(llm_api::FinishReason::Stop)
        });
        if matches!(finish_reason, Some(llm_api::FinishReason::ToolCalls)) {
            let tool_text = self.render_tool_calls()?;
            self.estimated_completion_tokens += count_visible_tokens(&tool_text);
            text.push_str(&tool_text);
        }
        let completion_tokens =
            self.completion_token_delta(usage_completion_tokens, finish_reason.is_some());
        if text.is_empty() && finish_reason.is_none() && completion_tokens == 0 {
            return Ok(None);
        }
        Ok(Some(BackendStreamChunk {
            text,
            prompt_tokens: self.prompt_tokens,
            completion_tokens,
            finish_reason,
        }))
    }

    fn push_tool_calls(&mut self, tool_calls: &[MlxToolCall]) {
        for call in tool_calls {
            let index = call.index.unwrap_or(self.tool_calls.len());
            if self.tool_calls.len() <= index {
                self.tool_calls
                    .resize_with(index + 1, MlxToolCallAccumulator::default);
            }
            let accumulator = &mut self.tool_calls[index];
            if let Some(function) = &call.function {
                if let Some(name) = &function.name {
                    accumulator.name.push_str(name);
                }
                if let Some(arguments) = &function.arguments {
                    accumulator.arguments.push_str(arguments);
                }
            }
        }
    }

    fn render_tool_calls(&mut self) -> Result<String, BackendError> {
        if self.tool_calls.is_empty() {
            return Ok(String::new());
        }
        let mut rendered = String::new();
        for call in std::mem::take(&mut self.tool_calls) {
            rendered.push_str(&render_mlx_tool_call(&call, self.tool_markup)?);
        }
        Ok(rendered)
    }

    fn completion_token_delta(
        &mut self,
        usage_completion_tokens: Option<u64>,
        is_final_chunk: bool,
    ) -> u64 {
        if let Some(total) = usage_completion_tokens {
            self.uses_upstream_usage = true;
            let delta = total.saturating_sub(self.emitted_completion_tokens);
            self.emitted_completion_tokens = self.emitted_completion_tokens.max(total);
            return delta;
        }
        if self.uses_upstream_usage || !is_final_chunk {
            return 0;
        }
        let delta = self
            .estimated_completion_tokens
            .saturating_sub(self.emitted_completion_tokens);
        self.emitted_completion_tokens = self.estimated_completion_tokens;
        delta
    }
}

#[derive(Debug, Clone, Copy)]
enum MlxToolMarkup {
    Qwen,
    Gemma,
}

#[derive(Debug, Default, Clone)]
struct MlxToolCallAccumulator {
    name: String,
    arguments: String,
}

#[derive(Debug, Clone)]
struct MlxControlStopFilter {
    stop_tokens: &'static [&'static str],
    pending: String,
    stopped: bool,
}

impl MlxControlStopFilter {
    fn new(stop_tokens: &'static [&'static str]) -> Self {
        Self {
            stop_tokens,
            pending: String::new(),
            stopped: false,
        }
    }

    fn is_stopped(&self) -> bool {
        self.stopped
    }

    fn push_str(&mut self, text: &str) -> String {
        if self.stopped || text.is_empty() {
            return String::new();
        }
        self.pending.push_str(text);
        if let Some((index, token_len)) = self.first_stop_token() {
            self.stopped = true;
            let output = self.pending[..index].to_owned();
            self.pending.drain(..index + token_len);
            self.pending.clear();
            return output;
        }
        let withheld = self.pending_stop_prefix_len();
        if withheld == self.pending.len() {
            return String::new();
        }
        let split_at = self.pending.len() - withheld;
        self.pending.drain(..split_at).collect()
    }

    fn finish(&mut self) -> String {
        if self.stopped {
            self.pending.clear();
            return String::new();
        }
        std::mem::take(&mut self.pending)
    }

    fn first_stop_token(&self) -> Option<(usize, usize)> {
        self.stop_tokens
            .iter()
            .filter_map(|token| self.pending.find(token).map(|index| (index, token.len())))
            .min_by_key(|(index, _)| *index)
    }

    fn pending_stop_prefix_len(&self) -> usize {
        self.stop_tokens
            .iter()
            .flat_map(|token| {
                (1..token.len()).filter(move |length| self.pending.ends_with(&token[..*length]))
            })
            .max()
            .unwrap_or(0)
    }
}

fn mlx_sse_data(line: &str) -> Option<&str> {
    line.strip_prefix("data:")
        .map(|data| data.strip_prefix(' ').unwrap_or(data))
}

fn count_visible_tokens(text: &str) -> u64 {
    if text.trim().is_empty() {
        0
    } else {
        count_whitespace_tokens(text)
    }
}

fn mlx_finish_reason(reason: Option<&str>) -> Result<llm_api::FinishReason, BackendError> {
    match reason {
        Some("length") => Ok(llm_api::FinishReason::Length),
        Some("tool_calls") => Ok(llm_api::FinishReason::ToolCalls),
        Some("stop") | None => Ok(llm_api::FinishReason::Stop),
        Some(other) => Err(BackendError::Other(format!(
            "unsupported MLX finish reason `{other}`"
        ))),
    }
}

fn mlx_tool_markup_for_metadata(metadata: &BackendModelMetadata) -> MlxToolMarkup {
    match metadata
        .family
        .as_deref()
        .and_then(|family| ModelFamily::parse_slug(family).ok())
    {
        Some(ModelFamily::Gemma) => MlxToolMarkup::Gemma,
        _ => MlxToolMarkup::Qwen,
    }
}

fn render_mlx_tool_call(
    call: &MlxToolCallAccumulator,
    markup: MlxToolMarkup,
) -> Result<String, BackendError> {
    if call.name.trim().is_empty() {
        return Err(BackendError::Other(
            "MLX structured tool call was missing a function name".to_owned(),
        ));
    }
    let arguments = parse_mlx_tool_arguments(&call.arguments)?;
    match markup {
        MlxToolMarkup::Qwen => Ok(format!(
            "<tool_call>{}</tool_call>",
            serde_json::json!({
                "name": call.name.as_str(),
                "arguments": arguments,
            })
        )),
        MlxToolMarkup::Gemma => {
            let Value::Object(arguments) = arguments else {
                return Err(BackendError::Other(
                    "MLX structured Gemma tool arguments must be a JSON object".to_owned(),
                ));
            };
            Ok(format!(
                "<|tool_call>call:{}{}<tool_call|>",
                call.name,
                render_gemma_tool_object(&arguments)?
            ))
        }
    }
}

fn parse_mlx_tool_arguments(arguments: &str) -> Result<Value, BackendError> {
    let trimmed = arguments.trim();
    if trimmed.is_empty() {
        return Ok(serde_json::json!({}));
    }
    serde_json::from_str::<Value>(trimmed).map_err(|err| {
        BackendError::Other(format!(
            "invalid MLX structured tool call arguments `{trimmed}`: {err}"
        ))
    })
}

fn render_gemma_tool_object(
    object: &serde_json::Map<String, Value>,
) -> Result<String, BackendError> {
    let mut rendered = String::from("{");
    for (index, (key, value)) in object.iter().enumerate() {
        if index > 0 {
            rendered.push(',');
        }
        rendered.push_str(
            &serde_json::to_string(key).map_err(|err| {
                BackendError::Other(format!("Gemma tool key render failed: {err}"))
            })?,
        );
        rendered.push(':');
        rendered.push_str(&render_gemma_tool_value(value)?);
    }
    rendered.push('}');
    Ok(rendered)
}

fn render_gemma_tool_value(value: &Value) -> Result<String, BackendError> {
    match value {
        Value::Object(object) => render_gemma_tool_object(object),
        Value::Array(values) => {
            let mut rendered = String::from("[");
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    rendered.push(',');
                }
                rendered.push_str(&render_gemma_tool_value(value)?);
            }
            rendered.push(']');
            Ok(rendered)
        }
        _ => serde_json::to_string(value)
            .map_err(|err| BackendError::Other(format!("Gemma tool value render failed: {err}"))),
    }
}

fn count_whitespace_tokens(text: &str) -> u64 {
    text.split_whitespace().count().max(1) as u64
}

fn mlx_endpoint_url(base: &Url, suffix: &str) -> Url {
    let mut url = base.clone();
    let path = format!("{}/{}", base.path().trim_end_matches('/'), suffix);
    url.set_path(&path);
    url
}

fn is_loopback_endpoint(endpoint: &Url) -> bool {
    match endpoint.host() {
        Some(url::Host::Domain(domain)) => domain.eq_ignore_ascii_case("localhost"),
        Some(url::Host::Ipv4(addr)) => addr.is_loopback(),
        Some(url::Host::Ipv6(addr)) => addr.is_loopback(),
        None => false,
    }
}

fn mlx_control_stop_tokens_for_metadata(
    metadata: &BackendModelMetadata,
) -> &'static [&'static str] {
    match metadata
        .family
        .as_deref()
        .and_then(|family| ModelFamily::parse_slug(family).ok())
    {
        Some(ModelFamily::Gemma) => MLX_GEMMA_CONTROL_STOP_TOKENS,
        _ => MLX_QWEN_CONTROL_STOP_TOKENS,
    }
}

fn mlx_upstream_protocol_for_metadata(metadata: &BackendModelMetadata) -> MlxUpstreamProtocol {
    match metadata
        .family
        .as_deref()
        .and_then(|family| ModelFamily::parse_slug(family).ok())
    {
        Some(ModelFamily::Gemma) => MlxUpstreamProtocol::ChatCompletions,
        _ => MlxUpstreamProtocol::Completions,
    }
}

fn mlx_metadata(
    model_id: &str,
    snapshot_path: &Path,
    requested_family: Option<ModelFamily>,
) -> anyhow::Result<BackendModelMetadata> {
    let mut metadata = BackendModelMetadata::new(model_id.to_owned(), "mlx");
    metadata.snapshot_path = Some(PathBuf::from(snapshot_path));
    let manifest_path = snapshot_path.join("llm-engine-manifest.json");
    let manifest_bytes = match std::fs::read(&manifest_path) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            let family = requested_family.ok_or_else(|| {
                anyhow::anyhow!(
                    "MLX backend requires model family metadata; add --family qwen for raw MLX snapshots or promote the snapshot with an llm-engine manifest"
                )
            })?;
            validate_mlx_serving_family(family)?;
            metadata.loader = Some("mlx".to_owned());
            metadata.family = Some(family.canonical_slug().to_owned());
            return Ok(metadata);
        }
        Err(err) => return Err(err.into()),
    };
    let manifest = serde_json::from_slice::<SnapshotManifest>(&manifest_bytes)?;
    let manifest_loader = BackendKind::parse_slug(&manifest.loader)?;
    if manifest_loader != BackendKind::Mlx {
        anyhow::bail!(
            "MLX backend requires manifest loader `mlx`, not `{}`",
            manifest_loader.canonical_slug()
        );
    }
    let manifest_family = ModelFamily::parse_slug(&manifest.family)?;
    if let Some(requested_family) = requested_family
        && manifest_family != requested_family
    {
        anyhow::bail!(
            "requested snapshot family `{}` does not match manifest family `{}`",
            requested_family.canonical_slug(),
            manifest_family.canonical_slug()
        );
    }
    validate_mlx_serving_family(manifest_family)?;
    metadata.family = Some(manifest_family.canonical_slug().to_owned());
    metadata.loader = Some(manifest_loader.canonical_slug().to_owned());
    metadata.quantization = Some(manifest.quantization.clone());
    metadata.repo_id = Some(manifest.repo_id.clone());
    metadata.resolved_commit = Some(manifest.resolved_commit.clone());
    metadata.profile = Some(manifest.profile.clone());
    metadata.manifest_digest = Some(manifest.digest());
    Ok(metadata)
}

fn validate_mlx_serving_family(family: ModelFamily) -> anyhow::Result<()> {
    if !family.adapter().capabilities().backend_execution {
        anyhow::bail!(
            "model family `{}` is recognized but not serveable yet; {} serving is deferred until Qwen production parity",
            family.canonical_slug(),
            family.display_name()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
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
    async fn mlx_backend_falls_back_to_rendered_prompt_for_gemma_tool_chat() {
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
        assert_eq!(
            request["messages"][0]["content"],
            "<bos><|turn>user\nuse lookup<turn|>\n<|turn>model\n"
        );
        assert_eq!(
            request["messages"]
                .as_array()
                .expect("messages array")
                .len(),
            1
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
            let received_for_thread = received.clone();
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
