use async_trait::async_trait;
use futures::{StreamExt, stream::BoxStream};
use llm_backend::{
    BackendError, BackendModelMetadata, BackendOutput, BackendRequest, BackendStreamChunk,
    ModelBackend,
};
use llm_models::ModelFamily;
use std::{path::Path, sync::Arc};
use tokio_util::sync::CancellationToken;
use url::Url;

mod client;
mod metadata;
mod metrics;
mod protocol;
mod request;
mod sse;

use client::is_loopback_endpoint;
use metadata::mlx_metadata;
pub(crate) use metrics::mlx_backend_metrics_snapshot;
use metrics::{MlxBackendFailureKind, MlxBackendMetrics, mlx_backend_metrics};
use protocol::{mlx_control_stop_tokens_for_metadata, mlx_tool_markup_for_metadata};
use request::build_upstream_request;
use sse::{MlxSseParser, count_whitespace_tokens};

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
        Ok(Self {
            model_id: model_id.clone(),
            metadata,
            upstream_model,
            endpoint: options.endpoint,
            control_stop_tokens,
            client: reqwest::Client::new(),
            metrics: mlx_backend_metrics(),
        })
    }

    fn validate_model(&self, request: &BackendRequest) -> Result<(), BackendError> {
        if request.model != self.model_id {
            return Err(BackendError::ModelNotFound {
                requested: request.model.clone(),
                available: self.model_id.clone(),
            });
        }
        Ok(())
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
            self.validate_model(&request)?;
            let (upstream_protocol, upstream_request) = build_upstream_request(
                &self.client,
                &self.endpoint,
                &self.upstream_model,
                &self.metadata,
                &request,
            )?;
            let mut request_metrics = self.metrics.start_request(upstream_protocol);
            let response = tokio::select! {
                response = upstream_request.send() => response
                    .map_err(|err| BackendError::Other(format!("MLX request failed: {err}"))),
                _ = cancellation.cancelled() => Err(BackendError::Cancelled),
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
                            Err(BackendError::Other(format!("MLX stream read failed: {err}")))?
                        }
                    };
                    request_metrics.record_response_bytes(bytes.len());
                    let chunk = match std::str::from_utf8(&bytes) {
                        Ok(chunk) => chunk,
                        Err(err) => {
                            request_metrics.finish_failure(MlxBackendFailureKind::InvalidUtf8);
                            Err(BackendError::Other(format!("MLX stream was not UTF-8: {err}")))?
                        }
                    };
                    let parsed_chunks = match parser.push_str(chunk) {
                        Ok(chunks) => chunks,
                        Err(err) => {
                            request_metrics.finish_failure(MlxBackendFailureKind::SseParse);
                            Err(err)?
                        }
                    };
                    request_metrics.record_stream_chunks(parsed_chunks.len());
                    for parsed in parsed_chunks {
                        if parsed.finish_reason.is_some() {
                            request_metrics.record_finish_chunk();
                        }
                        yield parsed;
                    }
                }
                let final_chunks = match parser.finish() {
                    Ok(chunks) => chunks,
                    Err(err) => {
                        request_metrics.finish_failure(MlxBackendFailureKind::SseParse);
                        Err(err)?
                    }
                };
                request_metrics.record_stream_chunks(final_chunks.len());
                request_metrics.finish_success();
                for parsed in final_chunks {
                    if parsed.finish_reason.is_some() {
                        request_metrics.record_finish_chunk();
                    }
                    yield parsed;
                }
            } else {
                request_metrics.finish_failure(MlxBackendFailureKind::HttpStatus);
                let body = tokio::select! {
                    body = response.text() => body
                        .map_err(|err| BackendError::Other(format!("MLX response read failed: {err}"))),
                    _ = cancellation.cancelled() => Err(BackendError::Cancelled),
                };
                let body = match body {
                    Ok(body) => body,
                    Err(err) => {
                        request_metrics.finish_failure(mlx_failure_kind_for_backend_error(&err));
                        Err(err)?
                    }
                };
                Err(BackendError::Other(format!(
                    "MLX server returned HTTP {status}: {body}"
                )))?;
            }
        }
        .boxed()
    }
}

fn mlx_failure_kind_for_backend_error(err: &BackendError) -> MlxBackendFailureKind {
    if matches!(err, BackendError::Cancelled) {
        MlxBackendFailureKind::Cancelled
    } else {
        MlxBackendFailureKind::Transport
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

#[cfg(test)]
mod tests;
