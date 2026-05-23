use super::protocol::{MlxUpstreamProtocol, mlx_effective_chat_template_kwargs};
use crate::sync_ext::FailPoisonedMutex;
use llm_backend_contracts::{BackendModelMetadata, BackendRequest, BackendToolChoice};
use llm_telemetry::LatencyMetrics;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, Instant},
};

#[derive(Debug, Default)]
pub(super) struct MlxBackendMetrics {
    counters: Mutex<MlxBackendMetricCounters>,
}

#[derive(Debug, Clone, Default)]
struct MlxBackendMetricCounters {
    requests_total: u64,
    successful_requests: u64,
    failed_requests: u64,
    completion_requests: u64,
    chat_completion_requests: u64,
    stream_chunks: u64,
    response_bytes: u64,
    http_error_responses: u64,
    transport_failures: u64,
    stream_read_failures: u64,
    invalid_utf8_failures: u64,
    sse_parse_failures: u64,
    stall_failures: u64,
    cancelled_requests: u64,
    dropped_requests: u64,
    upstream_request_latency: LatencyMetrics,
    blocking_upstream_request_latency: LatencyMetrics,
    streaming_upstream_request_latency: LatencyMetrics,
    stream_response_headers_latency: LatencyMetrics,
    stream_first_upstream_byte_latency: LatencyMetrics,
    stream_first_parsed_chunk_latency: LatencyMetrics,
    stream_first_tool_delta_latency: LatencyMetrics,
    stream_upstream_complete_latency: LatencyMetrics,
    last_request_fingerprint: Option<Value>,
    zero_output_successes: u64,
    last_zero_output_success: Option<Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MlxBackendFailureKind {
    HttpStatus,
    Transport,
    StreamRead,
    InvalidUtf8,
    SseParse,
    Stall,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MlxBackendRequestKind {
    Blocking,
    Streaming,
}

#[derive(Debug)]
pub(super) struct MlxBackendRequestMetrics {
    metrics: Arc<MlxBackendMetrics>,
    kind: MlxBackendRequestKind,
    started: Instant,
    finished: bool,
    observed_finish_chunk: bool,
    observed_first_upstream_byte: bool,
    observed_first_parsed_chunk: bool,
    observed_first_tool_delta: bool,
    observed_stream_complete: bool,
}

impl MlxBackendMetrics {
    pub(super) fn start_request(
        self: &Arc<Self>,
        protocol: MlxUpstreamProtocol,
        kind: MlxBackendRequestKind,
    ) -> MlxBackendRequestMetrics {
        {
            let mut counters = self.counters.lock_or_panic("MLX backend metrics");
            counters.requests_total += 1;
            match protocol {
                MlxUpstreamProtocol::Completions => counters.completion_requests += 1,
                MlxUpstreamProtocol::ChatCompletions => counters.chat_completion_requests += 1,
            }
        }
        MlxBackendRequestMetrics {
            metrics: Arc::clone(self),
            kind,
            started: Instant::now(),
            finished: false,
            observed_finish_chunk: false,
            observed_first_upstream_byte: false,
            observed_first_parsed_chunk: false,
            observed_first_tool_delta: false,
            observed_stream_complete: false,
        }
    }

    pub(super) fn snapshot(&self) -> Value {
        let counters = self.counters.lock_or_panic("MLX backend metrics").clone();
        json!({
            "requests_total": counters.requests_total,
            "successful_requests": counters.successful_requests,
            "failed_requests": counters.failed_requests,
            "completion_requests": counters.completion_requests,
            "chat_completion_requests": counters.chat_completion_requests,
            "stream_chunks": counters.stream_chunks,
            "response_bytes": counters.response_bytes,
            "http_error_responses": counters.http_error_responses,
            "transport_failures": counters.transport_failures,
            "stream_read_failures": counters.stream_read_failures,
            "invalid_utf8_failures": counters.invalid_utf8_failures,
            "sse_parse_failures": counters.sse_parse_failures,
            "stall_failures": counters.stall_failures,
            "cancelled_requests": counters.cancelled_requests,
            "dropped_requests": counters.dropped_requests,
            "request_latency_ms": latency_summary(counters.upstream_request_latency),
            "upstream_request_latency_ms": latency_summary(counters.upstream_request_latency),
            "blocking_upstream_request_latency_ms": latency_summary(
                counters.blocking_upstream_request_latency,
            ),
            "streaming_upstream_request_latency_ms": latency_summary(
                counters.streaming_upstream_request_latency,
            ),
            "stream_response_headers_ms": latency_summary(
                counters.stream_response_headers_latency,
            ),
            "stream_first_upstream_byte_ms": latency_summary(
                counters.stream_first_upstream_byte_latency,
            ),
            "stream_first_parsed_chunk_ms": latency_summary(
                counters.stream_first_parsed_chunk_latency,
            ),
            "stream_first_tool_delta_ms": latency_summary(
                counters.stream_first_tool_delta_latency,
            ),
            "stream_upstream_complete_ms": latency_summary(
                counters.stream_upstream_complete_latency,
            ),
            "last_request_fingerprint": counters.last_request_fingerprint,
            "zero_output_successes": counters.zero_output_successes,
            "last_zero_output_success": counters.last_zero_output_success,
        })
    }

    fn record_stream_chunks(&self, chunks: u64) {
        self.counters
            .lock_or_panic("MLX backend metrics")
            .stream_chunks += chunks;
    }

    fn record_response_bytes(&self, bytes: u64) {
        self.counters
            .lock_or_panic("MLX backend metrics")
            .response_bytes += bytes;
    }

    fn record_success(&self, kind: MlxBackendRequestKind, latency: Duration) {
        let mut counters = self.counters.lock_or_panic("MLX backend metrics");
        counters.successful_requests += 1;
        record_upstream_latency(&mut counters, kind, latency);
    }

    fn record_request_fingerprint(&self, fingerprint: Value) {
        self.counters
            .lock_or_panic("MLX backend metrics")
            .last_request_fingerprint = Some(fingerprint);
    }

    fn record_zero_output_success(&self, observation: Value) {
        let mut counters = self.counters.lock_or_panic("MLX backend metrics");
        counters.zero_output_successes += 1;
        counters.last_zero_output_success = Some(observation);
    }

    fn record_stream_response_headers(&self, latency: Duration) {
        self.counters
            .lock_or_panic("MLX backend metrics")
            .stream_response_headers_latency
            .record(latency);
    }

    fn record_stream_first_upstream_byte(&self, latency: Duration) {
        self.counters
            .lock_or_panic("MLX backend metrics")
            .stream_first_upstream_byte_latency
            .record(latency);
    }

    fn record_stream_first_parsed_chunk(&self, latency: Duration) {
        self.counters
            .lock_or_panic("MLX backend metrics")
            .stream_first_parsed_chunk_latency
            .record(latency);
    }

    fn record_stream_first_tool_delta(&self, latency: Duration) {
        self.counters
            .lock_or_panic("MLX backend metrics")
            .stream_first_tool_delta_latency
            .record(latency);
    }

    fn record_stream_upstream_complete(&self, latency: Duration) {
        self.counters
            .lock_or_panic("MLX backend metrics")
            .stream_upstream_complete_latency
            .record(latency);
    }

    fn record_failure(
        &self,
        request_kind: MlxBackendRequestKind,
        failure_kind: MlxBackendFailureKind,
        latency: Duration,
    ) {
        let mut counters = self.counters.lock_or_panic("MLX backend metrics");
        counters.failed_requests += 1;
        record_upstream_latency(&mut counters, request_kind, latency);
        match failure_kind {
            MlxBackendFailureKind::HttpStatus => counters.http_error_responses += 1,
            MlxBackendFailureKind::Transport => counters.transport_failures += 1,
            MlxBackendFailureKind::StreamRead => counters.stream_read_failures += 1,
            MlxBackendFailureKind::InvalidUtf8 => counters.invalid_utf8_failures += 1,
            MlxBackendFailureKind::SseParse => counters.sse_parse_failures += 1,
            MlxBackendFailureKind::Stall => counters.stall_failures += 1,
            MlxBackendFailureKind::Cancelled => counters.cancelled_requests += 1,
        }
    }

    fn record_dropped(&self, kind: MlxBackendRequestKind, latency: Duration) {
        let mut counters = self.counters.lock_or_panic("MLX backend metrics");
        counters.failed_requests += 1;
        counters.dropped_requests += 1;
        record_upstream_latency(&mut counters, kind, latency);
    }
}

fn record_upstream_latency(
    counters: &mut MlxBackendMetricCounters,
    kind: MlxBackendRequestKind,
    latency: Duration,
) {
    counters.upstream_request_latency.record(latency);
    match kind {
        MlxBackendRequestKind::Blocking => {
            counters.blocking_upstream_request_latency.record(latency)
        }
        MlxBackendRequestKind::Streaming => {
            counters.streaming_upstream_request_latency.record(latency);
        }
    }
}

impl MlxBackendRequestMetrics {
    pub(super) fn record_finish_chunk(&mut self) {
        self.observed_finish_chunk = true;
    }

    pub(super) fn record_stream_chunks(&self, chunks: usize) {
        self.metrics.record_stream_chunks(chunks as u64);
    }

    pub(super) fn record_response_bytes(&self, bytes: usize) {
        self.metrics.record_response_bytes(bytes as u64);
    }

    pub(super) fn record_request_fingerprint(&self, fingerprint: Value) {
        self.metrics.record_request_fingerprint(fingerprint);
    }

    pub(super) fn record_zero_output_success(&self, observation: Value) {
        self.metrics.record_zero_output_success(observation);
    }

    pub(super) fn record_stream_response_headers(&self) -> Duration {
        let latency = self.started.elapsed();
        self.metrics.record_stream_response_headers(latency);
        latency
    }

    pub(super) fn record_first_upstream_byte(&mut self) -> Option<Duration> {
        if self.observed_first_upstream_byte {
            return None;
        }
        self.observed_first_upstream_byte = true;
        let latency = self.started.elapsed();
        self.metrics.record_stream_first_upstream_byte(latency);
        Some(latency)
    }

    pub(super) fn record_first_parsed_chunk(&mut self) -> Option<Duration> {
        if self.observed_first_parsed_chunk {
            return None;
        }
        self.observed_first_parsed_chunk = true;
        let latency = self.started.elapsed();
        self.metrics.record_stream_first_parsed_chunk(latency);
        Some(latency)
    }

    pub(super) fn record_first_tool_delta(&mut self) -> Option<Duration> {
        if self.observed_first_tool_delta {
            return None;
        }
        self.observed_first_tool_delta = true;
        let latency = self.started.elapsed();
        self.metrics.record_stream_first_tool_delta(latency);
        Some(latency)
    }

    pub(super) fn record_stream_complete(&mut self) -> Option<Duration> {
        if self.observed_stream_complete {
            return None;
        }
        self.observed_stream_complete = true;
        let latency = self.started.elapsed();
        self.metrics.record_stream_upstream_complete(latency);
        Some(latency)
    }

    pub(super) fn finish_success(&mut self) {
        if self.finished {
            return;
        }
        self.metrics
            .record_success(self.kind, self.started.elapsed());
        self.finished = true;
    }

    pub(super) fn finish_failure(&mut self, kind: MlxBackendFailureKind) {
        if self.finished {
            return;
        }
        self.metrics
            .record_failure(self.kind, kind, self.started.elapsed());
        self.finished = true;
    }
}

impl Drop for MlxBackendRequestMetrics {
    fn drop(&mut self) {
        if !self.finished {
            if self.observed_finish_chunk {
                self.metrics
                    .record_success(self.kind, self.started.elapsed());
            } else {
                self.metrics
                    .record_dropped(self.kind, self.started.elapsed());
            }
            self.finished = true;
        }
    }
}

pub(super) fn mlx_backend_metrics() -> Arc<MlxBackendMetrics> {
    static METRICS: OnceLock<Arc<MlxBackendMetrics>> = OnceLock::new();
    Arc::clone(METRICS.get_or_init(|| Arc::new(MlxBackendMetrics::default())))
}

pub(crate) fn mlx_backend_metrics_snapshot() -> Value {
    mlx_backend_metrics().snapshot()
}

pub(super) fn mlx_request_fingerprint(
    protocol: MlxUpstreamProtocol,
    stream: bool,
    metadata: &BackendModelMetadata,
    request: &BackendRequest,
) -> Value {
    json!({
        "protocol": mlx_protocol_label(protocol),
        "stream": stream,
        "cache_key": request.cache_context().key.as_str(),
        "prompt_hash": hash_str(request.prompt()),
        "tool_schema_hash": request.cache_context().tool_schema.as_deref().map(hash_str),
        "messages_hash": request
            .as_chat()
            .and_then(|chat| hash_json(&chat.chat_context.messages)),
        "tool_choice_hash": request
            .as_chat()
            .and_then(|chat| chat.required_tool_choice.as_ref())
            .and_then(hash_tool_choice),
        "chat_template_kwargs_hash": mlx_effective_chat_template_kwargs(metadata, request)
            .as_ref()
            .and_then(hash_json),
        "max_tokens": request.max_tokens,
    })
}

pub(super) fn mlx_protocol_label(protocol: MlxUpstreamProtocol) -> &'static str {
    match protocol {
        MlxUpstreamProtocol::Completions => "completions",
        MlxUpstreamProtocol::ChatCompletions => "chat_completions",
    }
}

fn hash_tool_choice(choice: &BackendToolChoice) -> Option<String> {
    let value = match choice {
        BackendToolChoice::RequiredAny => json!("required"),
        BackendToolChoice::RequiredFunction(name) => json!({
            "type": "function",
            "function": {
                "name": name,
            },
        }),
    };
    hash_json(&value)
}

fn hash_json(value: &impl serde::Serialize) -> Option<String> {
    serde_json::to_vec(value)
        .ok()
        .map(|bytes| hash_bytes(&bytes))
}

fn hash_str(value: &str) -> String {
    hash_bytes(value.as_bytes())
}

fn hash_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    format!("{digest:x}")
}

fn latency_summary(metrics: LatencyMetrics) -> Value {
    json!({
        "count": metrics.count(),
        "min": metrics.min_ms(),
        "max": metrics.max_ms(),
        "avg": metrics.avg_ms(),
    })
}
