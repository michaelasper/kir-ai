use super::MlxUpstreamProtocol;
use crate::sync_ext::RecoverPoisonedMutex;
use llm_telemetry::LatencyMetrics;
use serde_json::{Value, json};
use std::{
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, Instant},
};

#[derive(Debug, Default)]
pub(super) struct MlxBackendMetrics {
    counters: Mutex<MlxBackendMetricCounters>,
}

#[derive(Debug, Clone, Copy, Default)]
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
    cancelled_requests: u64,
    dropped_requests: u64,
    request_latency: LatencyMetrics,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MlxBackendFailureKind {
    HttpStatus,
    Transport,
    StreamRead,
    InvalidUtf8,
    SseParse,
    Cancelled,
}

#[derive(Debug)]
pub(super) struct MlxBackendRequestMetrics {
    metrics: Arc<MlxBackendMetrics>,
    started: Instant,
    finished: bool,
    observed_finish_chunk: bool,
}

impl MlxBackendMetrics {
    pub(super) fn start_request(
        self: &Arc<Self>,
        protocol: MlxUpstreamProtocol,
    ) -> MlxBackendRequestMetrics {
        {
            let mut counters = self.counters.lock_or_recover("MLX backend metrics");
            counters.requests_total += 1;
            match protocol {
                MlxUpstreamProtocol::Completions => counters.completion_requests += 1,
                MlxUpstreamProtocol::ChatCompletions => counters.chat_completion_requests += 1,
            }
        }
        MlxBackendRequestMetrics {
            metrics: Arc::clone(self),
            started: Instant::now(),
            finished: false,
            observed_finish_chunk: false,
        }
    }

    pub(super) fn snapshot(&self) -> Value {
        let counters = *self.counters.lock_or_recover("MLX backend metrics");
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
            "cancelled_requests": counters.cancelled_requests,
            "dropped_requests": counters.dropped_requests,
            "request_latency_ms": latency_summary(counters.request_latency),
        })
    }

    fn record_stream_chunks(&self, chunks: u64) {
        self.counters
            .lock_or_recover("MLX backend metrics")
            .stream_chunks += chunks;
    }

    fn record_response_bytes(&self, bytes: u64) {
        self.counters
            .lock_or_recover("MLX backend metrics")
            .response_bytes += bytes;
    }

    fn record_success(&self, latency: Duration) {
        let mut counters = self.counters.lock_or_recover("MLX backend metrics");
        counters.successful_requests += 1;
        counters.request_latency.record(latency);
    }

    fn record_failure(&self, kind: MlxBackendFailureKind, latency: Duration) {
        let mut counters = self.counters.lock_or_recover("MLX backend metrics");
        counters.failed_requests += 1;
        counters.request_latency.record(latency);
        match kind {
            MlxBackendFailureKind::HttpStatus => counters.http_error_responses += 1,
            MlxBackendFailureKind::Transport => counters.transport_failures += 1,
            MlxBackendFailureKind::StreamRead => counters.stream_read_failures += 1,
            MlxBackendFailureKind::InvalidUtf8 => counters.invalid_utf8_failures += 1,
            MlxBackendFailureKind::SseParse => counters.sse_parse_failures += 1,
            MlxBackendFailureKind::Cancelled => counters.cancelled_requests += 1,
        }
    }

    fn record_dropped(&self, latency: Duration) {
        let mut counters = self.counters.lock_or_recover("MLX backend metrics");
        counters.failed_requests += 1;
        counters.dropped_requests += 1;
        counters.request_latency.record(latency);
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

    pub(super) fn finish_success(&mut self) {
        if self.finished {
            return;
        }
        self.metrics.record_success(self.started.elapsed());
        self.finished = true;
    }

    pub(super) fn finish_failure(&mut self, kind: MlxBackendFailureKind) {
        if self.finished {
            return;
        }
        self.metrics.record_failure(kind, self.started.elapsed());
        self.finished = true;
    }
}

impl Drop for MlxBackendRequestMetrics {
    fn drop(&mut self) {
        if !self.finished {
            if self.observed_finish_chunk {
                self.metrics.record_success(self.started.elapsed());
            } else {
                self.metrics.record_dropped(self.started.elapsed());
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

fn latency_summary(metrics: LatencyMetrics) -> Value {
    json!({
        "count": metrics.count(),
        "min": metrics.min_ms(),
        "max": metrics.max_ms(),
        "avg": metrics.avg_ms(),
    })
}
