use super::protocol::{MlxUpstreamProtocol, mlx_effective_chat_template_kwargs};
use llm_backend_contracts::{BackendModelMetadata, BackendRequest, BackendToolChoice};
use llm_telemetry::AtomicLatencyMetrics;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    sync::{
        Arc, Mutex, MutexGuard, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

#[derive(Debug, Default)]
pub(super) struct MlxBackendMetrics {
    counters: MlxBackendMetricCounters,
    latencies: MlxBackendLatencyMetrics,
    observations: Mutex<MlxBackendObservations>,
}

#[derive(Debug, Default)]
struct MlxBackendMetricCounters {
    requests_total: AtomicU64,
    successful_requests: AtomicU64,
    failed_requests: AtomicU64,
    completion_requests: AtomicU64,
    chat_completion_requests: AtomicU64,
    stream_chunks: AtomicU64,
    response_bytes: AtomicU64,
    http_error_responses: AtomicU64,
    transport_failures: AtomicU64,
    stream_read_failures: AtomicU64,
    invalid_utf8_failures: AtomicU64,
    sse_parse_failures: AtomicU64,
    stall_failures: AtomicU64,
    cancelled_requests: AtomicU64,
    dropped_requests: AtomicU64,
    zero_output_successes: AtomicU64,
}

#[derive(Debug, Default)]
struct MlxBackendLatencyMetrics {
    upstream_request_latency: AtomicLatencyMetrics,
    blocking_upstream_request_latency: AtomicLatencyMetrics,
    streaming_upstream_request_latency: AtomicLatencyMetrics,
    stream_response_headers_latency: AtomicLatencyMetrics,
    stream_first_upstream_byte_latency: AtomicLatencyMetrics,
    stream_first_parsed_chunk_latency: AtomicLatencyMetrics,
    stream_first_tool_delta_latency: AtomicLatencyMetrics,
    stream_upstream_complete_latency: AtomicLatencyMetrics,
}

#[derive(Debug, Clone, Default)]
struct MlxBackendObservations {
    last_request_fingerprint: Option<Value>,
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
        self.counters.requests_total.fetch_add(1, Ordering::Relaxed);
        match protocol {
            MlxUpstreamProtocol::Completions => self
                .counters
                .completion_requests
                .fetch_add(1, Ordering::Relaxed),
            MlxUpstreamProtocol::ChatCompletions => self
                .counters
                .chat_completion_requests
                .fetch_add(1, Ordering::Relaxed),
        };
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
        let observations = self.lock_observations().clone();
        json!({
            "requests_total": self.load_counter(&self.counters.requests_total),
            "successful_requests": self.load_counter(&self.counters.successful_requests),
            "failed_requests": self.load_counter(&self.counters.failed_requests),
            "completion_requests": self.load_counter(&self.counters.completion_requests),
            "chat_completion_requests": self.load_counter(&self.counters.chat_completion_requests),
            "stream_chunks": self.load_counter(&self.counters.stream_chunks),
            "response_bytes": self.load_counter(&self.counters.response_bytes),
            "http_error_responses": self.load_counter(&self.counters.http_error_responses),
            "transport_failures": self.load_counter(&self.counters.transport_failures),
            "stream_read_failures": self.load_counter(&self.counters.stream_read_failures),
            "invalid_utf8_failures": self.load_counter(&self.counters.invalid_utf8_failures),
            "sse_parse_failures": self.load_counter(&self.counters.sse_parse_failures),
            "stall_failures": self.load_counter(&self.counters.stall_failures),
            "cancelled_requests": self.load_counter(&self.counters.cancelled_requests),
            "dropped_requests": self.load_counter(&self.counters.dropped_requests),
            "request_latency_ms": latency_summary(&self.latencies.upstream_request_latency),
            "upstream_request_latency_ms": latency_summary(
                &self.latencies.upstream_request_latency,
            ),
            "blocking_upstream_request_latency_ms": latency_summary(
                &self.latencies.blocking_upstream_request_latency,
            ),
            "streaming_upstream_request_latency_ms": latency_summary(
                &self.latencies.streaming_upstream_request_latency,
            ),
            "stream_response_headers_ms": latency_summary(
                &self.latencies.stream_response_headers_latency,
            ),
            "stream_first_upstream_byte_ms": latency_summary(
                &self.latencies.stream_first_upstream_byte_latency,
            ),
            "stream_first_parsed_chunk_ms": latency_summary(
                &self.latencies.stream_first_parsed_chunk_latency,
            ),
            "stream_first_tool_delta_ms": latency_summary(
                &self.latencies.stream_first_tool_delta_latency,
            ),
            "stream_upstream_complete_ms": latency_summary(
                &self.latencies.stream_upstream_complete_latency,
            ),
            "last_request_fingerprint": observations.last_request_fingerprint,
            "zero_output_successes": self.load_counter(&self.counters.zero_output_successes),
            "last_zero_output_success": observations.last_zero_output_success,
        })
    }

    fn record_stream_chunks(&self, chunks: u64) {
        self.counters
            .stream_chunks
            .fetch_add(chunks, Ordering::Relaxed);
    }

    fn record_response_bytes(&self, bytes: u64) {
        self.counters
            .response_bytes
            .fetch_add(bytes, Ordering::Relaxed);
    }

    fn record_success(&self, kind: MlxBackendRequestKind, latency: Duration) {
        self.counters
            .successful_requests
            .fetch_add(1, Ordering::Relaxed);
        record_upstream_latency(&self.latencies, kind, latency);
    }

    fn record_request_fingerprint(&self, fingerprint: Value) {
        self.lock_observations().last_request_fingerprint = Some(fingerprint);
    }

    fn record_zero_output_success(&self, observation: Value) {
        self.counters
            .zero_output_successes
            .fetch_add(1, Ordering::Relaxed);
        self.lock_observations().last_zero_output_success = Some(observation);
    }

    fn record_stream_response_headers(&self, latency: Duration) {
        self.latencies
            .stream_response_headers_latency
            .record(latency);
    }

    fn record_stream_first_upstream_byte(&self, latency: Duration) {
        self.latencies
            .stream_first_upstream_byte_latency
            .record(latency);
    }

    fn record_stream_first_parsed_chunk(&self, latency: Duration) {
        self.latencies
            .stream_first_parsed_chunk_latency
            .record(latency);
    }

    fn record_stream_first_tool_delta(&self, latency: Duration) {
        self.latencies
            .stream_first_tool_delta_latency
            .record(latency);
    }

    fn record_stream_upstream_complete(&self, latency: Duration) {
        self.latencies
            .stream_upstream_complete_latency
            .record(latency);
    }

    fn record_failure(
        &self,
        request_kind: MlxBackendRequestKind,
        failure_kind: MlxBackendFailureKind,
        latency: Duration,
    ) {
        self.counters
            .failed_requests
            .fetch_add(1, Ordering::Relaxed);
        record_upstream_latency(&self.latencies, request_kind, latency);
        match failure_kind {
            MlxBackendFailureKind::HttpStatus => &self.counters.http_error_responses,
            MlxBackendFailureKind::Transport => &self.counters.transport_failures,
            MlxBackendFailureKind::StreamRead => &self.counters.stream_read_failures,
            MlxBackendFailureKind::InvalidUtf8 => &self.counters.invalid_utf8_failures,
            MlxBackendFailureKind::SseParse => &self.counters.sse_parse_failures,
            MlxBackendFailureKind::Stall => &self.counters.stall_failures,
            MlxBackendFailureKind::Cancelled => &self.counters.cancelled_requests,
        }
        .fetch_add(1, Ordering::Relaxed);
    }

    fn record_dropped(&self, kind: MlxBackendRequestKind, latency: Duration) {
        self.counters
            .failed_requests
            .fetch_add(1, Ordering::Relaxed);
        self.counters
            .dropped_requests
            .fetch_add(1, Ordering::Relaxed);
        record_upstream_latency(&self.latencies, kind, latency);
    }

    fn lock_observations(&self) -> MutexGuard<'_, MlxBackendObservations> {
        recover_metrics_lock(&self.observations, "MLX backend observations")
    }

    fn load_counter(&self, counter: &AtomicU64) -> u64 {
        counter.load(Ordering::Relaxed)
    }
}

fn record_upstream_latency(
    latencies: &MlxBackendLatencyMetrics,
    kind: MlxBackendRequestKind,
    latency: Duration,
) {
    latencies.upstream_request_latency.record(latency);
    match kind {
        MlxBackendRequestKind::Blocking => {
            latencies.blocking_upstream_request_latency.record(latency)
        }
        MlxBackendRequestKind::Streaming => {
            latencies.streaming_upstream_request_latency.record(latency);
        }
    }
}

fn recover_metrics_lock<'a, T>(lock: &'a Mutex<T>, name: &'static str) -> MutexGuard<'a, T> {
    match lock.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            tracing::warn!(mutex = name, "recovering poisoned MLX metrics state");
            lock.clear_poison();
            poisoned.into_inner()
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
        _ => return None,
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

fn latency_summary(metrics: &AtomicLatencyMetrics) -> Value {
    let metrics = metrics.snapshot();
    json!({
        "count": metrics.count(),
        "min": metrics.min_ms(),
        "max": metrics.max_ms(),
        "avg": metrics.avg_ms(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        panic::{AssertUnwindSafe, catch_unwind},
        sync::Mutex,
        thread,
    };

    #[test]
    fn metrics_snapshot_records_atomic_counters_and_latencies() {
        let metrics = Arc::new(MlxBackendMetrics::default());
        let mut request = metrics.start_request(
            MlxUpstreamProtocol::Completions,
            MlxBackendRequestKind::Streaming,
        );

        request.record_stream_chunks(2);
        request.record_response_bytes(128);
        request.record_request_fingerprint(json!({"prompt_hash": "abc123"}));
        request.record_stream_response_headers();
        request.record_first_upstream_byte();
        request.record_first_parsed_chunk();
        request.record_first_tool_delta();
        request.record_stream_complete();
        request.finish_success();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot["requests_total"], json!(1));
        assert_eq!(snapshot["successful_requests"], json!(1));
        assert_eq!(snapshot["completion_requests"], json!(1));
        assert_eq!(snapshot["stream_chunks"], json!(2));
        assert_eq!(snapshot["response_bytes"], json!(128));
        assert_eq!(snapshot["request_latency_ms"]["count"], json!(1));
        assert_eq!(
            snapshot["streaming_upstream_request_latency_ms"]["count"],
            json!(1)
        );
        assert_eq!(snapshot["stream_response_headers_ms"]["count"], json!(1));
        assert_eq!(snapshot["stream_first_upstream_byte_ms"]["count"], json!(1));
        assert_eq!(snapshot["stream_first_parsed_chunk_ms"]["count"], json!(1));
        assert_eq!(snapshot["stream_first_tool_delta_ms"]["count"], json!(1));
        assert_eq!(snapshot["stream_upstream_complete_ms"]["count"], json!(1));
        assert_eq!(
            snapshot["last_request_fingerprint"]["prompt_hash"],
            "abc123"
        );
    }

    #[test]
    fn metrics_record_stream_latency_milestones_concurrently() {
        let metrics = Arc::new(MlxBackendMetrics::default());

        thread::scope(|scope| {
            for _ in 0..8 {
                let metrics = Arc::clone(&metrics);
                scope.spawn(move || {
                    for _ in 0..64 {
                        let mut request = metrics.start_request(
                            MlxUpstreamProtocol::ChatCompletions,
                            MlxBackendRequestKind::Streaming,
                        );
                        request.record_stream_response_headers();
                        request.record_first_upstream_byte();
                        request.record_first_parsed_chunk();
                        request.record_first_tool_delta();
                        request.record_stream_complete();
                        request.finish_success();
                    }
                });
            }
        });

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot["requests_total"], json!(512));
        assert_eq!(snapshot["successful_requests"], json!(512));
        assert_eq!(snapshot["chat_completion_requests"], json!(512));
        assert_eq!(
            snapshot["streaming_upstream_request_latency_ms"]["count"],
            json!(512)
        );
        assert_eq!(snapshot["stream_response_headers_ms"]["count"], json!(512));
        assert_eq!(
            snapshot["stream_first_upstream_byte_ms"]["count"],
            json!(512)
        );
        assert_eq!(
            snapshot["stream_first_parsed_chunk_ms"]["count"],
            json!(512)
        );
        assert_eq!(snapshot["stream_first_tool_delta_ms"]["count"], json!(512));
        assert_eq!(snapshot["stream_upstream_complete_ms"]["count"], json!(512));
    }

    #[test]
    fn metrics_recover_poisoned_observation_lock() {
        let metrics = MlxBackendMetrics::default();

        metrics.record_success(MlxBackendRequestKind::Blocking, Duration::from_millis(5));

        poison_lock(&metrics.observations);
        metrics.record_request_fingerprint(json!({"recovered": true}));
        metrics.record_zero_output_success(json!({"output_tokens": 0}));
        assert!(!metrics.observations.is_poisoned());

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot["successful_requests"], json!(1));
        assert_eq!(
            snapshot["blocking_upstream_request_latency_ms"]["count"],
            json!(1)
        );
        assert_eq!(
            snapshot["last_request_fingerprint"]["recovered"],
            json!(true)
        );
        assert_eq!(snapshot["zero_output_successes"], json!(1));
        assert_eq!(
            snapshot["last_zero_output_success"]["output_tokens"],
            json!(0)
        );
    }

    fn poison_lock<T>(lock: &Mutex<T>) {
        let result = catch_unwind(AssertUnwindSafe(|| {
            let _guard = lock.lock().expect("test lock starts unpoisoned");
            panic!("poison MLX metrics lock");
        }));
        assert!(result.is_err());
        assert!(lock.is_poisoned());
    }
}
