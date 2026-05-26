use super::client::{
    MlxTransportError, MlxTransportExecutor, MlxTransportRequestKind, MlxTransportResponse,
};
use super::metrics::MlxBackendFailureKind;
use super::*;
use async_trait::async_trait;
use futures::StreamExt;
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};
use tokio_util::sync::CancellationToken;

type ScriptedTransportChunk = Result<Vec<u8>, MlxTransportError>;
type ScriptedTransportResponse = Result<Vec<ScriptedTransportChunk>, MlxTransportError>;

#[derive(Debug, Clone, PartialEq, Eq)]
struct CapturedMlxTransportRequest {
    protocol: MlxUpstreamProtocol,
    kind: MlxTransportRequestKind,
    content_type: &'static str,
    body: Value,
}

#[derive(Debug)]
struct ScriptedMlxTransport {
    state: Mutex<ScriptedMlxTransportState>,
}

#[derive(Debug)]
struct ScriptedMlxTransportState {
    health: BackendHealth,
    health_checks: usize,
    requests: Vec<CapturedMlxTransportRequest>,
    responses: VecDeque<ScriptedTransportResponse>,
}

impl ScriptedMlxTransport {
    fn with_responses(responses: Vec<ScriptedTransportResponse>) -> (MlxTransport, Arc<Self>) {
        Self::with_health_and_responses(BackendHealth::ready(), responses)
    }

    fn with_health_and_responses(
        health: BackendHealth,
        responses: Vec<ScriptedTransportResponse>,
    ) -> (MlxTransport, Arc<Self>) {
        let transport = Arc::new(Self {
            state: Mutex::new(ScriptedMlxTransportState {
                health,
                health_checks: 0,
                requests: Vec::new(),
                responses: responses.into(),
            }),
        });
        (
            MlxTransport::from_executor_for_test(Arc::clone(&transport)),
            transport,
        )
    }

    fn requests(&self) -> Vec<CapturedMlxTransportRequest> {
        self.state
            .lock()
            .expect("scripted transport lock")
            .requests
            .clone()
    }

    fn health_checks(&self) -> usize {
        self.state
            .lock()
            .expect("scripted transport lock")
            .health_checks
    }
}

#[async_trait]
impl MlxTransportExecutor for ScriptedMlxTransport {
    fn label(&self) -> &'static str {
        "scripted"
    }

    async fn execute(
        &self,
        request: super::request::MlxUpstreamRequest,
        kind: MlxTransportRequestKind,
        _cancellation: CancellationToken,
    ) -> Result<MlxTransportResponse, MlxTransportError> {
        let mut state = self.state.lock().expect("scripted transport lock");
        let body = serde_json::from_slice(request.body()).expect("scripted request JSON");
        state.requests.push(CapturedMlxTransportRequest {
            protocol: request.protocol(),
            kind,
            content_type: request.content_type(),
            body,
        });
        let response = state
            .responses
            .pop_front()
            .expect("scripted transport response");
        response.map(|chunks| MlxTransportResponse::new(futures::stream::iter(chunks).boxed()))
    }

    async fn health(&self) -> BackendHealth {
        let mut state = self.state.lock().expect("scripted transport lock");
        state.health_checks += 1;
        state.health.clone()
    }
}

fn backend_with_transport(transport: MlxTransport) -> MlxBackend {
    MlxBackend {
        model_id: "local-mlx".to_owned(),
        metadata: BackendModelMetadata::new("local-mlx", "mlx").with_family("qwen"),
        upstream_model: "/tmp/local-mlx".to_owned(),
        control_stop_tokens: MLX_QWEN_CONTROL_STOP_TOKENS,
        tool_markup: MlxToolMarkup::Json,
        transport,
        include_stream_usage: true,
        metrics: Arc::new(MlxBackendMetrics::default()),
    }
}

fn sse_chunks(chunks: &[&str]) -> Vec<Result<Vec<u8>, MlxTransportError>> {
    chunks
        .iter()
        .map(|chunk| Ok(chunk.as_bytes().to_vec()))
        .collect()
}

#[tokio::test]
async fn mlx_backend_uses_non_http_transport_for_blocking_generation() {
    let (transport, scripted) = ScriptedMlxTransport::with_responses(vec![Ok(sse_chunks(&[
        "data:{\"choices\":[{\"text\":\"MLX says hi\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":4}}\n\ndata:[DONE]\n\n",
    ]))]);
    let backend = backend_with_transport(transport);
    let metrics = Arc::clone(&backend.metrics);

    let output = backend
        .generate(BackendRequest::raw_completion(
            "local-mlx",
            "hello mlx",
            Some(12),
            SamplingConfig::Greedy,
        ))
        .await
        .expect("scripted MLX generation succeeds");

    assert_eq!(output.text, "MLX says hi");
    assert_eq!(output.prompt_tokens, 3);
    assert_eq!(output.completion_tokens, 4);
    let requests = scripted.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].protocol, MlxUpstreamProtocol::Completions);
    assert_eq!(requests[0].kind, MlxTransportRequestKind::Blocking);
    assert_eq!(requests[0].content_type, "application/json");
    assert_eq!(requests[0].body["model"], "/tmp/local-mlx");
    assert_eq!(requests[0].body["prompt"], "hello mlx");
    assert_eq!(requests[0].body["stream"], false);

    let metrics = metrics.snapshot();
    assert_eq!(metrics["successful_requests"], 1);
    assert_eq!(metrics["completion_requests"], 1);
    assert_eq!(metrics["stream_chunks"], 1);
    assert_eq!(metrics["transport_failures"], 0);
}

#[tokio::test]
async fn mlx_backend_uses_non_http_transport_for_streaming_generation() {
    let (transport, scripted) = ScriptedMlxTransport::with_responses(vec![Ok(sse_chunks(&[
        "data:{\"choices\":[{\"text\":\"one \",\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":2}}\n\n",
        "data:{\"choices\":[{\"text\":\"two\",\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":3}}\n\ndata:[DONE]\n\n",
    ]))]);
    let backend = backend_with_transport(transport);
    let metrics = Arc::clone(&backend.metrics);

    let chunks = backend
        .generate_stream(BackendRequest::raw_completion(
            "local-mlx",
            "hello mlx",
            Some(12),
            SamplingConfig::Greedy,
        ))
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("scripted MLX stream succeeds");

    let text = chunks
        .iter()
        .filter(|chunk| chunk.progress.is_none())
        .map(|chunk| chunk.text.as_str())
        .collect::<String>();
    assert_eq!(text, "one two");
    let requests = scripted.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].kind, MlxTransportRequestKind::Streaming);
    assert_eq!(requests[0].body["stream"], true);
    assert_eq!(requests[0].body["stream_options"]["include_usage"], true);

    let metrics = metrics.snapshot();
    assert_eq!(metrics["successful_requests"], 1);
    assert_eq!(metrics["streaming_upstream_request_latency_ms"]["count"], 1);
    assert_eq!(metrics["stream_response_headers_ms"]["count"], 1);
    assert_eq!(metrics["stream_first_upstream_byte_ms"]["count"], 1);
}

#[tokio::test]
async fn mlx_backend_health_uses_non_http_transport_model_list_boundary() {
    let (transport, scripted) = ScriptedMlxTransport::with_health_and_responses(
        BackendHealth::unavailable("binary control plane offline"),
        Vec::new(),
    );
    let backend = backend_with_transport(transport);

    let health = backend.health().await;

    assert!(!health.is_ready());
    assert_eq!(health.reason(), Some("binary control plane offline"));
    assert_eq!(scripted.health_checks(), 1);
    assert!(scripted.requests().is_empty());
}

#[tokio::test]
async fn mlx_backend_records_non_http_transport_errors_without_reqwest() {
    let transport_error = MlxTransportError::new(
        MlxBackendFailureKind::Transport,
        BackendError::other("shared-memory ring unavailable"),
    );
    let (transport, _scripted) = ScriptedMlxTransport::with_responses(vec![Err(transport_error)]);
    let backend = backend_with_transport(transport);
    let metrics = Arc::clone(&backend.metrics);

    let err = backend
        .generate(BackendRequest::raw_completion(
            "local-mlx",
            "hello mlx",
            Some(12),
            SamplingConfig::Greedy,
        ))
        .await
        .expect_err("scripted transport error is surfaced");

    assert_eq!(err.other_message(), Some("shared-memory ring unavailable"));
    let metrics = metrics.snapshot();
    assert_eq!(metrics["failed_requests"], 1);
    assert_eq!(metrics["transport_failures"], 1);
    assert_eq!(metrics["completion_requests"], 1);
}
