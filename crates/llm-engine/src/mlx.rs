use async_trait::async_trait;
use futures::{StreamExt, stream::BoxStream};
use llm_backend::{
    BackendError, BackendFinishReason, BackendModelMetadata, BackendOutput, BackendRequest,
    BackendStreamChunk, ModelBackend,
};
use llm_models::ModelFamily;
use serde_json::json;
use std::{path::Path, sync::Arc};
use tokio_util::sync::CancellationToken;
use url::Url;

pub use client::MlxTimeouts;

mod client;
mod metadata;
mod metrics;
mod protocol;
mod request;
mod sse;

use client::{MLX_STALL_PREFIX, build_http_client, format_duration, is_loopback_endpoint};
use metadata::mlx_metadata;
pub(crate) use metrics::mlx_backend_metrics_snapshot;
use metrics::{
    MlxBackendFailureKind, MlxBackendMetrics, MlxBackendRequestKind, MlxBackendRequestMetrics,
    mlx_backend_metrics, mlx_protocol_label, mlx_request_fingerprint,
};
use protocol::{
    MlxUpstreamProtocol, mlx_control_stop_tokens_for_metadata, mlx_tool_markup_for_metadata,
};
use request::build_upstream_request;
use sse::{MlxSseParser, parse_mlx_completion_body};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MlxToolParserMode {
    #[default]
    Auto,
    Json,
    QwenXml,
}

impl MlxToolParserMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "auto" => Some(Self::Auto),
            "json" => Some(Self::Json),
            "qwen-xml" => Some(Self::QwenXml),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Json => "json",
            Self::QwenXml => "qwen-xml",
        }
    }
}

#[derive(Debug, Clone)]
pub struct MlxBackendOptions {
    pub endpoint: Url,
    pub family: Option<ModelFamily>,
    pub timeouts: MlxTimeouts,
    pub include_stream_usage: bool,
    pub tool_parser: MlxToolParserMode,
}

#[derive(Debug, Clone)]
pub struct MlxBackend {
    model_id: String,
    metadata: BackendModelMetadata,
    upstream_model: String,
    endpoint: Url,
    control_stop_tokens: &'static [&'static str],
    tool_markup: protocol::MlxToolMarkup,
    client: reqwest::Client,
    timeouts: MlxTimeouts,
    include_stream_usage: bool,
    metrics: Arc<MlxBackendMetrics>,
}

impl MlxBackend {
    pub async fn open(
        model_id: impl Into<String>,
        snapshot_path: impl AsRef<Path>,
    ) -> anyhow::Result<Self> {
        Self::open_with_options(model_id, snapshot_path, MlxBackendOptions::default()).await
    }

    pub async fn open_with_options(
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
        let metadata = mlx_metadata(&model_id, snapshot_path, options.family).await?;
        let control_stop_tokens = mlx_control_stop_tokens_for_metadata(&metadata);
        let tool_markup =
            mlx_tool_markup_for_metadata(&metadata, Some(snapshot_path), options.tool_parser)?;
        let client = build_http_client(options.timeouts);
        let timeouts = options.timeouts;
        let include_stream_usage = options.include_stream_usage;
        Ok(Self {
            model_id: model_id.clone(),
            metadata,
            upstream_model,
            endpoint: options.endpoint,
            control_stop_tokens,
            tool_markup,
            client,
            timeouts,
            include_stream_usage,
            metrics: mlx_backend_metrics(),
        })
    }

    fn validate_model(&self, request: &BackendRequest) -> Result<(), BackendError> {
        if request.model != self.model_id {
            return Err(BackendError::model_not_found(
                request.model.clone(),
                self.model_id.clone(),
            ));
        }
        Ok(())
    }

    async fn generate_once(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        if cancellation.is_cancelled() {
            return Err(BackendError::cancelled());
        }
        self.validate_model(&request)?;
        let (upstream_protocol, upstream_request) = build_upstream_request(
            &self.client,
            &self.endpoint,
            &self.upstream_model,
            &self.metadata,
            &request,
            false,
            self.include_stream_usage,
        )?;
        let mut request_metrics = self
            .metrics
            .start_request(upstream_protocol, MlxBackendRequestKind::Blocking);
        request_metrics.record_request_fingerprint(mlx_request_fingerprint(
            upstream_protocol,
            false,
            &self.metadata,
            &request,
        ));
        let response = tokio::select! {
            response = upstream_request.send() => response
                .map_err(|err| mlx_request_error(err, self.timeouts.request)),
            _ = cancellation.cancelled() => Err(BackendError::cancelled()),
        };
        let response = match response {
            Ok(response) => response,
            Err(err) => {
                request_metrics.finish_failure(mlx_failure_kind_for_backend_error(&err));
                return Err(err);
            }
        };
        let status = response.status();
        if !status.is_success() {
            request_metrics.finish_failure(MlxBackendFailureKind::HttpStatus);
            let body = tokio::select! {
                body = response.text() => body
                    .map_err(|err| BackendError::other(format!("MLX response read failed: {err}"))),
                _ = cancellation.cancelled() => Err(BackendError::cancelled()),
            };
            let body = match body {
                Ok(body) => body,
                Err(err) => {
                    request_metrics.finish_failure(mlx_failure_kind_for_backend_error(&err));
                    return Err(err);
                }
            };
            return Err(BackendError::other(format!(
                "MLX server returned HTTP {status}: {body}"
            )));
        }

        let mut bytes = response.bytes_stream();
        let mut body = Vec::new();
        let mut saw_first_byte = false;
        loop {
            let item = if saw_first_byte {
                tokio::select! {
                    biased;
                    _ = cancellation.cancelled() => Err(BackendError::cancelled()),
                    result = tokio::time::timeout(self.timeouts.read, bytes.next()) => {
                        result.map_err(|_| mlx_stream_stall_error(self.timeouts.read))
                    }
                }
            } else {
                tokio::select! {
                    biased;
                    _ = cancellation.cancelled() => Err(BackendError::cancelled()),
                    item = bytes.next() => Ok(item),
                }
            };
            let item = match item {
                Ok(item) => item,
                Err(err) => {
                    request_metrics.finish_failure(mlx_failure_kind_for_backend_error(&err));
                    return Err(err);
                }
            };
            let Some(item) = item else {
                break;
            };
            let bytes = match item {
                Ok(bytes) => bytes,
                Err(err) => {
                    request_metrics.finish_failure(MlxBackendFailureKind::StreamRead);
                    return Err(BackendError::other(format!(
                        "MLX response read failed: {err}"
                    )));
                }
            };
            saw_first_byte = true;
            request_metrics.record_response_bytes(bytes.len());
            body.extend_from_slice(&bytes);
        }
        let body = match std::str::from_utf8(&body) {
            Ok(body) => body,
            Err(err) => {
                request_metrics.finish_failure(MlxBackendFailureKind::InvalidUtf8);
                return Err(BackendError::other(format!(
                    "MLX response was not UTF-8: {err}"
                )));
            }
        };
        let (output, chunk_count) = match parse_mlx_completion_body(
            body,
            request.prompt(),
            self.control_stop_tokens,
            self.tool_markup,
            request
                .as_chat()
                .and_then(|chat| chat.cache_context.tool_schema.as_deref()),
        ) {
            Ok(parsed) => parsed,
            Err(err) => {
                request_metrics.finish_failure(MlxBackendFailureKind::SseParse);
                return Err(err);
            }
        };
        let output_observation =
            MlxOutputObservation::from_output(&output, chunk_count as u64, body.len() as u64);
        maybe_record_zero_output_success(
            &request_metrics,
            &self.metadata,
            &request.model,
            upstream_protocol,
            false,
            &output_observation,
        );
        request_metrics.record_stream_chunks(chunk_count);
        request_metrics.finish_success();
        Ok(output)
    }

    fn stream_completion<'a>(
        &'a self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        async_stream::try_stream! {
            if cancellation.is_cancelled() {
                Err(BackendError::cancelled())?;
            }
            self.validate_model(&request)?;
            let (upstream_protocol, upstream_request) = build_upstream_request(
                &self.client,
                &self.endpoint,
                &self.upstream_model,
                &self.metadata,
                &request,
                true,
                self.include_stream_usage,
            )?;
            let mut request_metrics = self
                .metrics
                .start_request(upstream_protocol, MlxBackendRequestKind::Streaming);
            request_metrics.record_request_fingerprint(mlx_request_fingerprint(
                upstream_protocol,
                true,
                &self.metadata,
                &request,
            ));
            let response = tokio::select! {
                response = upstream_request.send() => response
                    .map_err(|err| mlx_request_error(err, self.timeouts.request)),
                _ = cancellation.cancelled() => Err(BackendError::cancelled()),
            };
            let response = match response {
                Ok(response) => response,
                Err(err) => {
                    request_metrics.finish_failure(mlx_failure_kind_for_backend_error(&err));
                    Err(err)?
                }
            };
            let status = response.status();
            if status.is_success() {
                request_metrics.record_stream_response_headers();
                let mut bytes = response.bytes_stream();
                let mut parser = MlxSseParser::new_streaming(
                    request.prompt(),
                    self.control_stop_tokens,
                    self.tool_markup,
                );
                if matches!(self.tool_markup, protocol::MlxToolMarkup::QwenXml) {
                    parser = MlxSseParser::new_streaming_with_tool_schema(
                        request.prompt(),
                        self.control_stop_tokens,
                        self.tool_markup,
                        request
                            .as_chat()
                            .and_then(|chat| chat.cache_context.tool_schema.as_deref()),
                    )
                    .inspect_err(|_| {
                        request_metrics.finish_failure(MlxBackendFailureKind::SseParse);
                    })?;
                }
                let mut output_observation = MlxOutputObservation::default();
                let mut saw_first_byte = false;
                loop {
                    let item = if saw_first_byte {
                        tokio::select! {
                            biased;
                            _ = cancellation.cancelled() => Err(BackendError::cancelled()),
                            result = tokio::time::timeout(self.timeouts.read, bytes.next()) => {
                                result.map_err(|_| mlx_stream_stall_error(self.timeouts.read))
                            }
                        }
                    } else {
                        tokio::select! {
                            biased;
                            _ = cancellation.cancelled() => Err(BackendError::cancelled()),
                            item = bytes.next() => Ok(item),
                        }
                    };
                    let item = match item {
                        Ok(item) => item,
                        Err(err) => {
                            request_metrics.finish_failure(mlx_failure_kind_for_backend_error(&err));
                            Err(err)?
                        }
                    };
                    let Some(item) = item else {
                        break;
                    };
                    let bytes = match item {
                        Ok(bytes) => bytes,
                        Err(err) => {
                            request_metrics.finish_failure(MlxBackendFailureKind::StreamRead);
                            Err(BackendError::other(format!("MLX stream read failed: {err}")))?
                        }
                    };
                    saw_first_byte = true;
                    output_observation.response_bytes += bytes.len() as u64;
                    request_metrics.record_first_upstream_byte();
                    request_metrics.record_response_bytes(bytes.len());
                    let chunk = match std::str::from_utf8(&bytes) {
                        Ok(chunk) => chunk,
                        Err(err) => {
                            request_metrics.finish_failure(MlxBackendFailureKind::InvalidUtf8);
                            Err(BackendError::other(format!("MLX stream was not UTF-8: {err}")))?
                        }
                    };
                    let parsed_chunks = match parser.push_str(chunk) {
                        Ok(chunks) => chunks,
                        Err(err) => {
                            request_metrics.finish_failure(MlxBackendFailureKind::SseParse);
                            Err(err)?
                        }
                    };
                    output_observation.stream_chunks += parsed_chunks.len() as u64;
                    request_metrics.record_stream_chunks(parsed_chunks.len());
                    for parsed in parsed_chunks {
                        output_observation.observe_chunk(&parsed);
                        request_metrics.record_first_parsed_chunk();
                        if !parsed.tool_call_deltas.is_empty() {
                            request_metrics.record_first_tool_delta();
                        }
                        if parsed.finish_reason.is_some() {
                            request_metrics.record_finish_chunk();
                            request_metrics.record_stream_complete();
                        }
                        yield parsed;
                    }
                }
                request_metrics.record_stream_complete();
                let final_chunks = match parser.finish() {
                    Ok(chunks) => chunks,
                    Err(err) => {
                        request_metrics.finish_failure(MlxBackendFailureKind::SseParse);
                        Err(err)?
                    }
                };
                output_observation.stream_chunks += final_chunks.len() as u64;
                request_metrics.record_stream_chunks(final_chunks.len());
                for parsed in &final_chunks {
                    output_observation.observe_chunk(parsed);
                }
                maybe_record_zero_output_success(
                    &request_metrics,
                    &self.metadata,
                    &request.model,
                    upstream_protocol,
                    true,
                    &output_observation,
                );
                request_metrics.finish_success();
                for parsed in final_chunks {
                    request_metrics.record_first_parsed_chunk();
                    if !parsed.tool_call_deltas.is_empty() {
                        request_metrics.record_first_tool_delta();
                    }
                    if parsed.finish_reason.is_some() {
                        request_metrics.record_finish_chunk();
                        request_metrics.record_stream_complete();
                    }
                    yield parsed;
                }
            } else {
                request_metrics.finish_failure(MlxBackendFailureKind::HttpStatus);
                let body = tokio::select! {
                    body = response.text() => body
                        .map_err(|err| BackendError::other(format!("MLX response read failed: {err}"))),
                    _ = cancellation.cancelled() => Err(BackendError::cancelled()),
                };
                let body = match body {
                    Ok(body) => body,
                    Err(err) => {
                        request_metrics.finish_failure(mlx_failure_kind_for_backend_error(&err));
                        Err(err)?
                    }
                };
                Err(BackendError::other(format!(
                    "MLX server returned HTTP {status}: {body}"
                )))?;
            }
        }
        .boxed()
    }
}

#[derive(Debug, Default)]
struct MlxOutputObservation {
    saw_text: bool,
    saw_tool_delta: bool,
    prompt_tokens: u64,
    completion_tokens: u64,
    finish_reason: Option<BackendFinishReason>,
    stream_chunks: u64,
    response_bytes: u64,
}

impl MlxOutputObservation {
    fn from_output(output: &BackendOutput, stream_chunks: u64, response_bytes: u64) -> Self {
        Self {
            saw_text: !output.text.is_empty(),
            saw_tool_delta: false,
            prompt_tokens: output.prompt_tokens,
            completion_tokens: output.completion_tokens,
            finish_reason: Some(output.finish_reason),
            stream_chunks,
            response_bytes,
        }
    }

    fn observe_chunk(&mut self, chunk: &BackendStreamChunk) {
        self.saw_text |= !chunk.text.is_empty();
        self.saw_tool_delta |= !chunk.tool_call_deltas.is_empty();
        self.prompt_tokens = self.prompt_tokens.max(chunk.prompt_tokens);
        self.completion_tokens += chunk.completion_tokens;
        if let Some(finish_reason) = chunk.finish_reason {
            self.finish_reason = Some(finish_reason);
        }
    }

    fn is_zero_output_success(&self) -> bool {
        !self.saw_text && !self.saw_tool_delta && self.completion_tokens == 0
    }
}

fn maybe_record_zero_output_success(
    request_metrics: &MlxBackendRequestMetrics,
    metadata: &BackendModelMetadata,
    model: &str,
    protocol: MlxUpstreamProtocol,
    streamed: bool,
    observation: &MlxOutputObservation,
) {
    if !observation.is_zero_output_success() {
        return;
    }
    let protocol = mlx_protocol_label(protocol);
    tracing::warn!(
        model = model,
        family = metadata.family.as_deref().unwrap_or("unknown"),
        protocol,
        streamed,
        prompt_tokens = observation.prompt_tokens,
        completion_tokens = observation.completion_tokens,
        finish_reason = ?observation.finish_reason,
        stream_chunks = observation.stream_chunks,
        response_bytes = observation.response_bytes,
        "MLX successful response produced no completion output"
    );
    request_metrics.record_zero_output_success(json!({
        "model": model,
        "family": metadata.family.as_deref(),
        "protocol": protocol,
        "streamed": streamed,
        "prompt_tokens": observation.prompt_tokens,
        "completion_tokens": observation.completion_tokens,
        "finish_reason": observation.finish_reason,
        "stream_chunks": observation.stream_chunks,
        "response_bytes": observation.response_bytes,
    }));
}

fn mlx_stream_stall_error(read_timeout: std::time::Duration) -> BackendError {
    BackendError::other(format!(
        "{MLX_STALL_PREFIX} stream stalled for {} without data",
        format_duration(read_timeout)
    ))
}

fn mlx_failure_kind_for_backend_error(err: &BackendError) -> MlxBackendFailureKind {
    if err.is_cancelled() {
        MlxBackendFailureKind::Cancelled
    } else if let Some(msg) = err.other_message() {
        if msg.starts_with(MLX_STALL_PREFIX) {
            MlxBackendFailureKind::Stall
        } else {
            MlxBackendFailureKind::Transport
        }
    } else {
        MlxBackendFailureKind::Transport
    }
}

fn mlx_request_error(err: reqwest::Error, request_timeout: std::time::Duration) -> BackendError {
    if err.is_timeout() {
        BackendError::other(format!(
            "{MLX_STALL_PREFIX} request timed out after {}",
            format_duration(request_timeout)
        ))
    } else {
        BackendError::other(format!("MLX request failed: {err}"))
    }
}

const DEFAULT_MLX_ENDPOINT: &str = "http://127.0.0.1:8080/v1";

impl Default for MlxBackendOptions {
    fn default() -> Self {
        Self {
            endpoint: Url::parse(DEFAULT_MLX_ENDPOINT).expect(
                "DEFAULT_MLX_ENDPOINT is a valid URL verified at compile time by this assertion",
            ),
            family: None,
            timeouts: MlxTimeouts::default(),
            include_stream_usage: true,
            tool_parser: MlxToolParserMode::Auto,
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
            return Err(BackendError::cancelled());
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

#[cfg(test)]
mod tests;
