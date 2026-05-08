use async_trait::async_trait;
use futures::{StreamExt, stream::BoxStream};
use llm_backend::{
    BackendError, BackendModelMetadata, BackendOutput, BackendRequest, BackendStreamChunk,
    ModelBackend, SamplingConfig,
};
use llm_hub::SnapshotManifest;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio_util::sync::CancellationToken;
use url::Url;

#[derive(Debug, Clone)]
pub struct MlxBackendOptions {
    pub endpoint: Url,
    pub family: Option<String>,
}

#[derive(Debug, Clone)]
pub struct MlxBackend {
    model_id: String,
    metadata: BackendModelMetadata,
    upstream_model: String,
    completions_url: Url,
    client: reqwest::Client,
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
        let completions_url = mlx_endpoint_url(&options.endpoint, "completions");
        Ok(Self {
            model_id: model_id.clone(),
            metadata: mlx_metadata(&model_id, snapshot_path, options.family.as_deref())?,
            upstream_model,
            completions_url,
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
        let payload = MlxCompletionRequest {
            model: &self.upstream_model,
            prompt: &request.prompt,
            max_tokens: request.max_tokens,
            temperature,
            top_p,
            stream: true,
        };
        self.client
            .post(self.completions_url.clone())
            .json(&payload)
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
                let mut parser = MlxSseParser::new(&request.prompt);
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

#[derive(Debug, Deserialize)]
struct MlxCompletionResponse {
    choices: Vec<MlxCompletionChoice>,
    usage: Option<MlxUsage>,
}

#[derive(Debug, Deserialize)]
struct MlxCompletionChoice {
    text: Option<String>,
    message: Option<MlxMessage>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MlxMessage {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MlxUsage {
    prompt_tokens: Option<u64>,
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
}

impl MlxSseParser {
    fn new(prompt: &str) -> Self {
        Self {
            prompt_tokens: count_whitespace_tokens(prompt),
            estimated_completion_tokens: 0,
            emitted_completion_tokens: 0,
            uses_upstream_usage: false,
            saw_done: false,
            line_buffer: String::new(),
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
        let text = choice
            .text
            .or_else(|| choice.message.and_then(|message| message.content))
            .unwrap_or_default();
        self.estimated_completion_tokens += count_visible_tokens(&text);
        let finish_reason = choice
            .finish_reason
            .as_deref()
            .map(|reason| mlx_finish_reason(Some(reason)))
            .transpose()?;
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

fn mlx_metadata(
    model_id: &str,
    snapshot_path: &Path,
    requested_family: Option<&str>,
) -> anyhow::Result<BackendModelMetadata> {
    let mut metadata = BackendModelMetadata::new(model_id.to_owned(), "mlx");
    metadata.snapshot_path = Some(PathBuf::from(snapshot_path));
    let manifest_path = snapshot_path.join("llm-engine-manifest.json");
    let manifest_bytes = match std::fs::read(&manifest_path) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            metadata.loader = Some("mlx".to_owned());
            metadata.family = requested_family.map(str::to_owned);
            return Ok(metadata);
        }
        Err(err) => return Err(err.into()),
    };
    let manifest = serde_json::from_slice::<SnapshotManifest>(&manifest_bytes)?;
    if let Some(requested_family) = requested_family
        && manifest.family != requested_family
    {
        anyhow::bail!(
            "requested snapshot family `{requested_family}` does not match manifest family `{}`",
            manifest.family
        );
    }
    metadata.family = Some(manifest.family.clone());
    metadata.loader = Some(manifest.loader.clone());
    metadata.quantization = Some(manifest.quantization.clone());
    metadata.repo_id = Some(manifest.repo_id.clone());
    metadata.resolved_commit = Some(manifest.resolved_commit.clone());
    metadata.profile = Some(manifest.profile.clone());
    metadata.manifest_digest = Some(manifest.digest());
    Ok(metadata)
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_backend::{BackendCacheContext, BackendRequest, ModelBackend, SamplingConfig};
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
                ..MlxBackendOptions::default()
            },
        )
        .expect("backend opens");

        let output = backend
            .generate(BackendRequest {
                model: "local-mlx".to_owned(),
                prompt: "hello mlx".to_owned(),
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
    async fn mlx_backend_streams_completion_chunks() {
        let server = FakeMlxServer::start(
            "data:{\"choices\":[{\"text\":\"one \",\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":2}}\n\ndata: {\"choices\":[{\"text\":\"two\",\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":3}}\n\ndata: [DONE]\n\n",
        );
        let backend = MlxBackend::open_with_options(
            "local-mlx",
            server.snapshot_path(),
            MlxBackendOptions {
                endpoint: server.endpoint(),
                ..MlxBackendOptions::default()
            },
        )
        .expect("backend opens");

        let mut stream = backend.generate_stream(BackendRequest {
            model: "local-mlx".to_owned(),
            prompt: "hello mlx".to_owned(),
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
    async fn mlx_backend_rejects_model_mismatch_before_http_request() {
        let snapshot = tempfile::tempdir().expect("snapshot tempdir");
        let backend = MlxBackend::open("local-mlx", snapshot.path()).expect("backend opens");

        let err = backend
            .generate(BackendRequest {
                model: "other-model".to_owned(),
                prompt: "hello".to_owned(),
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
                ..MlxBackendOptions::default()
            },
        )
        .expect("backend opens");

        let err = backend
            .generate(BackendRequest {
                model: "local-mlx".to_owned(),
                prompt: "hello".to_owned(),
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
}
