use super::protocol::{
    MLX_DEEPSEEK_CONTROL_STOP_TOKENS, MLX_QWEN_CONTROL_STOP_TOKENS, MlxToolMarkup,
    MlxUpstreamProtocol,
};
use super::sse::{fold_mlx_chunks, parse_mlx_completion_body};
use super::*;
use llm_api::ChatMessage;
use llm_backend_contracts::{
    BackendCacheContext, BackendChatContext, BackendChatMessage, BackendChatRole,
    BackendFinishReason, BackendModelMetadata, BackendRequest, BackendToolCall,
    BackendToolCallDelta, BackendToolCallFunction, BackendToolCallType, BackendToolDefinition,
    ModelBackend, SamplingConfig,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    io::{Read, Write},
    net::TcpListener,
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};
use tempfile::TempDir;

type ParsedMlxChunkForTest = (
    String,
    Vec<BackendToolCallDelta>,
    u64,
    u64,
    Option<BackendFinishReason>,
);

fn assert_generated_tool_call_id_is_opaque(id: &str) {
    assert!(
        id.starts_with("call_"),
        "tool call id must use call_ prefix: {id}"
    );
    assert!(
        id.len() > "call_".len(),
        "tool call id must include an opaque suffix: {id}"
    );
    assert!(
        !id["call_".len()..]
            .chars()
            .all(|character| character.is_ascii_digit()),
        "generated tool call id must not be a predictable numeric sequence: {id}"
    );
}

fn backend_messages(messages: Vec<ChatMessage>) -> Vec<BackendChatMessage> {
    messages.into_iter().map(backend_message).collect()
}

fn backend_chat_context(messages: Vec<ChatMessage>) -> BackendChatContext {
    backend_chat_context_with_tools(messages, Vec::new())
}

fn backend_chat_context_with_tools(
    messages: Vec<ChatMessage>,
    tools: Vec<BackendToolDefinition>,
) -> BackendChatContext {
    BackendChatContext {
        messages: backend_messages(messages),
        tools,
    }
}

fn backend_message(message: ChatMessage) -> BackendChatMessage {
    BackendChatMessage {
        role: match message.role {
            llm_api::ChatRole::System => BackendChatRole::System,
            llm_api::ChatRole::User => BackendChatRole::User,
            llm_api::ChatRole::Assistant => BackendChatRole::Assistant,
            llm_api::ChatRole::Tool => BackendChatRole::Tool,
            other => panic!("unsupported test chat role: {other:?}"),
        },
        content: message.content,
        name: message.name,
        tool_call_id: message.tool_call_id,
        tool_calls: message
            .tool_calls
            .into_iter()
            .map(|tool_call| BackendToolCall {
                id: tool_call.id,
                call_type: match tool_call.call_type {
                    llm_api::ToolCallType::Function => BackendToolCallType::Function,
                    other => panic!("unsupported test tool call type: {other:?}"),
                },
                function: BackendToolCallFunction {
                    name: tool_call.function.name,
                    arguments: tool_call.function.arguments,
                },
            })
            .collect(),
    }
}

fn parse_mlx_sse_for_test(
    chunks: &[&str],
    markup: MlxToolMarkup,
) -> Result<Vec<ParsedMlxChunkForTest>, BackendError> {
    let mut parser = MlxSseParser::new_streaming("hello mlx", MLX_QWEN_CONTROL_STOP_TOKENS, markup);
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
                chunk.tool_call_deltas,
                chunk.prompt_tokens,
                chunk.completion_tokens,
                chunk.finish_reason,
            )
        })
        .collect())
}

fn parse_mlx_sse_for_test_with_tool_schema(
    chunks: &[&str],
    markup: MlxToolMarkup,
    tool_schema: &str,
) -> Result<Vec<ParsedMlxChunkForTest>, BackendError> {
    let mut parser = MlxSseParser::new_streaming_with_tool_schema(
        "hello mlx",
        MLX_QWEN_CONTROL_STOP_TOKENS,
        markup,
        Some(tool_schema),
    )?;
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
                chunk.tool_call_deltas,
                chunk.prompt_tokens,
                chunk.completion_tokens,
                chunk.finish_reason,
            )
        })
        .collect())
}

fn mlx_text_sse_frame(text: &str, finish_reason: Option<&str>) -> String {
    format!(
        "data:{}\n\n",
        serde_json::json!({
            "choices": [{
                "text": text,
                "finish_reason": finish_reason,
            }]
        })
    )
}

fn qwen_xml_test_tool_schema() -> String {
    serde_json::json!([{
        "type": "function",
        "function": {
            "name": "record",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer"},
                    "active": {"type": "boolean"}
                },
                "required": ["path", "limit", "active"]
            }
        }
    }])
    .to_string()
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
        Self::start_with_status(200, "OK", response_body)
    }

    fn start_with_status(
        status_code: u16,
        reason: &'static str,
        response_body: &'static str,
    ) -> Self {
        Self::start_with_response_delay_and_content_length(
            status_code,
            reason,
            response_body,
            response_body.len(),
            Duration::ZERO,
        )
    }

    fn start_with_response_delay(response_body: &'static str, delay: Duration) -> Self {
        Self::start_with_response_delay_and_content_length(
            200,
            "OK",
            response_body,
            response_body.len(),
            delay,
        )
    }

    fn start_with_initial_body_delay(response_body: &'static str, delay: Duration) -> Self {
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
            let request_content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().expect("content length"))
                })
                .expect("content-length header");
            while bytes.len() < header_end + request_content_length {
                let read = stream.read(&mut buffer).expect("read body");
                assert!(read > 0, "client closed before body");
                bytes.extend_from_slice(&buffer[..read]);
            }
            let body = &bytes[header_end..header_end + request_content_length];
            *received_for_thread.lock().expect("received lock") =
                Some(serde_json::from_slice(body).expect("json request body"));
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
            );
            let _ = stream.flush();
            thread::sleep(delay);
            let _ = write!(
                stream,
                "{:x}\r\n{}\r\n0\r\n\r\n",
                response_body.len(),
                response_body
            );
            let _ = stream.flush();
        });
        Self {
            endpoint,
            snapshot: tempfile::tempdir().expect("snapshot tempdir"),
            received,
            received_path,
            join: Some(join),
        }
    }

    fn start_with_response_content_length(
        status_code: u16,
        reason: &'static str,
        response_body: &'static str,
        response_content_length: usize,
    ) -> Self {
        Self::start_with_response_delay_and_content_length(
            status_code,
            reason,
            response_body,
            response_content_length,
            Duration::ZERO,
        )
    }

    fn start_with_stall(first_chunk: &'static str, stall_duration: Duration) -> Self {
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
        let _join = thread::spawn(move || {
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
            let request_content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().expect("content length"))
                })
                .expect("content-length header");
            while bytes.len() < header_end + request_content_length {
                let read = stream.read(&mut buffer).expect("read body");
                assert!(read > 0, "client closed before body");
                bytes.extend_from_slice(&buffer[..read]);
            }
            let body = &bytes[header_end..header_end + request_content_length];
            *received_for_thread.lock().expect("received lock") =
                Some(serde_json::from_slice(body).expect("json request body"));
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
            );
            let _ = stream.flush();
            let _ = write!(stream, "{:x}\r\n{}\r\n", first_chunk.len(), first_chunk);
            let _ = stream.flush();
            thread::sleep(stall_duration);
        });
        Self {
            endpoint,
            snapshot: tempfile::tempdir().expect("snapshot tempdir"),
            received,
            received_path,
            join: None,
        }
    }

    fn start_with_response_delay_and_content_length(
        status_code: u16,
        reason: &'static str,
        response_body: &'static str,
        response_content_length: usize,
        response_delay: Duration,
    ) -> Self {
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
            let request_content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().expect("content length"))
                })
                .unwrap_or(0);
            while bytes.len() < header_end + request_content_length {
                let read = stream.read(&mut buffer).expect("read body");
                assert!(read > 0, "client closed before body");
                bytes.extend_from_slice(&buffer[..read]);
            }
            let body = &bytes[header_end..header_end + request_content_length];
            if !body.is_empty() {
                *received_for_thread.lock().expect("received lock") =
                    Some(serde_json::from_slice(body).expect("json request body"));
            }
            thread::sleep(response_delay);
            let _ = write!(
                stream,
                "HTTP/1.1 {status_code} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response_content_length, response_body
            );
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
