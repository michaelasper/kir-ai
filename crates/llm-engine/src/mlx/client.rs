use std::time::Duration;
use std::{fmt, sync::Arc};
use url::Url;

use async_trait::async_trait;
use futures::{StreamExt, stream::BoxStream};
use llm_backend_contracts::{BackendError, BackendHealth};
use tokio_util::sync::CancellationToken;

use super::metrics::MlxBackendFailureKind;
use super::request::MlxUpstreamRequest;

pub(super) fn mlx_endpoint_url(base: &Url, suffix: &str) -> Url {
    let mut url = base.clone();
    let path = format!("{}/{}", base.path().trim_end_matches('/'), suffix);
    url.set_path(&path);
    url
}

pub(super) fn is_loopback_endpoint(endpoint: &Url) -> bool {
    match endpoint.host() {
        Some(url::Host::Domain(domain)) => domain.eq_ignore_ascii_case("localhost"),
        Some(url::Host::Ipv4(addr)) => addr.is_loopback(),
        Some(url::Host::Ipv6(addr)) => addr.is_loopback(),
        None => false,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlxTimeouts {
    pub connect: Duration,
    pub request: Duration,
    pub read: Duration,
}

impl Default for MlxTimeouts {
    fn default() -> Self {
        Self {
            connect: Duration::from_secs(5),
            request: Duration::from_secs(600),
            read: Duration::from_secs(60),
        }
    }
}

fn build_http_client(timeouts: MlxTimeouts) -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(timeouts.connect)
        .timeout(timeouts.request)
        .build()
        .expect("MLX HTTP client builds")
}

pub(super) type MlxTransportByteStream = BoxStream<'static, Result<Vec<u8>, MlxTransportError>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MlxTransportRequestKind {
    Blocking,
    Streaming,
}

impl MlxTransportRequestKind {
    fn read_error_message(self, err: reqwest::Error) -> String {
        match self {
            Self::Blocking => format!("MLX response read failed: {err}"),
            Self::Streaming => format!("MLX stream read failed: {err}"),
        }
    }
}

#[derive(Debug)]
pub(super) struct MlxTransportError {
    failure_kind: MlxBackendFailureKind,
    error: BackendError,
}

impl MlxTransportError {
    pub(super) fn new(failure_kind: MlxBackendFailureKind, error: BackendError) -> Self {
        Self {
            failure_kind,
            error,
        }
    }

    fn other(failure_kind: MlxBackendFailureKind, message: impl Into<String>) -> Self {
        Self::new(failure_kind, BackendError::other(message))
    }

    fn cancelled() -> Self {
        Self::new(MlxBackendFailureKind::Cancelled, BackendError::cancelled())
    }

    fn request_error(err: reqwest::Error, request_timeout: Duration) -> Self {
        if err.is_timeout() {
            Self::other(
                MlxBackendFailureKind::Stall,
                format!(
                    "{MLX_STALL_PREFIX} request timed out after {}",
                    format_duration(request_timeout)
                ),
            )
        } else {
            Self::other(
                MlxBackendFailureKind::Transport,
                format!("MLX request failed: {err}"),
            )
        }
    }

    fn read_timeout(read_timeout: Duration) -> Self {
        Self::other(
            MlxBackendFailureKind::Stall,
            format!(
                "{MLX_STALL_PREFIX} stream stalled for {} without data",
                format_duration(read_timeout)
            ),
        )
    }

    pub(super) fn failure_kind(&self) -> MlxBackendFailureKind {
        self.failure_kind
    }

    pub(super) fn into_backend_error(self) -> BackendError {
        self.error
    }
}

pub(super) struct MlxTransportResponse {
    body: MlxTransportByteStream,
}

impl MlxTransportResponse {
    pub(super) fn new(body: MlxTransportByteStream) -> Self {
        Self { body }
    }

    pub(super) fn into_byte_stream(self) -> MlxTransportByteStream {
        self.body
    }
}

#[async_trait]
pub(super) trait MlxTransportExecutor: Send + Sync {
    fn label(&self) -> &'static str;

    async fn execute(
        &self,
        request: MlxUpstreamRequest,
        kind: MlxTransportRequestKind,
        cancellation: CancellationToken,
    ) -> Result<MlxTransportResponse, MlxTransportError>;

    async fn health(&self) -> BackendHealth;
}

#[derive(Clone)]
pub(super) struct MlxTransport {
    executor: Arc<dyn MlxTransportExecutor>,
}

impl fmt::Debug for MlxTransport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MlxTransport")
            .field("executor", &self.executor.label())
            .finish()
    }
}

impl MlxTransport {
    pub(super) fn http(endpoint: Url, timeouts: MlxTimeouts) -> Self {
        Self {
            executor: Arc::new(MlxHttpTransport::new(endpoint, timeouts)),
        }
    }

    #[cfg(test)]
    pub(super) fn from_executor_for_test<T>(executor: Arc<T>) -> Self
    where
        T: MlxTransportExecutor + 'static,
    {
        Self { executor }
    }

    pub(super) async fn execute(
        &self,
        request: MlxUpstreamRequest,
        kind: MlxTransportRequestKind,
        cancellation: CancellationToken,
    ) -> Result<MlxTransportResponse, MlxTransportError> {
        self.executor.execute(request, kind, cancellation).await
    }

    pub(super) async fn health(&self) -> BackendHealth {
        self.executor.health().await
    }
}

#[derive(Debug, Clone)]
pub(super) struct MlxHttpTransport {
    endpoint: Url,
    client: reqwest::Client,
    timeouts: MlxTimeouts,
}

impl MlxHttpTransport {
    fn new(endpoint: Url, timeouts: MlxTimeouts) -> Self {
        Self {
            endpoint,
            client: build_http_client(timeouts),
            timeouts,
        }
    }

    fn request_builder(&self, request: MlxUpstreamRequest) -> reqwest::RequestBuilder {
        let upstream_url = mlx_endpoint_url(&self.endpoint, request.protocol().endpoint_suffix());
        self.client
            .post(upstream_url)
            .header(reqwest::header::CONTENT_TYPE, request.content_type())
            .body(request.into_body())
    }

    fn models_request_builder(&self) -> reqwest::RequestBuilder {
        self.client.get(mlx_endpoint_url(&self.endpoint, "models"))
    }

    async fn http_status_error(
        &self,
        response: reqwest::Response,
        cancellation: CancellationToken,
    ) -> MlxTransportError {
        let status = response.status();
        let body = tokio::select! {
            body = response.text() => body
                .map_err(|err| BackendError::other(format!("MLX response read failed: {err}"))),
            _ = cancellation.cancelled() => Err(BackendError::cancelled()),
        };
        match body {
            Ok(body) => MlxTransportError::other(
                MlxBackendFailureKind::HttpStatus,
                format!("MLX server returned HTTP {status}: {body}"),
            ),
            Err(err) => MlxTransportError::new(MlxBackendFailureKind::HttpStatus, err),
        }
    }
}

#[async_trait]
impl MlxTransportExecutor for MlxHttpTransport {
    fn label(&self) -> &'static str {
        "http"
    }

    async fn execute(
        &self,
        request: MlxUpstreamRequest,
        kind: MlxTransportRequestKind,
        cancellation: CancellationToken,
    ) -> Result<MlxTransportResponse, MlxTransportError> {
        let response = tokio::select! {
            response = self.request_builder(request).send() => response
                .map_err(|err| MlxTransportError::request_error(err, self.timeouts.request)),
            _ = cancellation.cancelled() => Err(MlxTransportError::cancelled()),
        }?;
        if !response.status().is_success() {
            return Err(self.http_status_error(response, cancellation).await);
        }
        Ok(MlxTransportResponse::new(http_byte_stream(
            response,
            kind,
            self.timeouts.read,
            cancellation,
        )))
    }

    async fn health(&self) -> BackendHealth {
        let response = self
            .models_request_builder()
            .timeout(self.timeouts.connect)
            .send()
            .await;
        match response {
            Ok(response) if response.status().is_success() => BackendHealth::ready(),
            Ok(response) => BackendHealth::unavailable(format!(
                "MLX model list returned HTTP {}",
                response.status()
            )),
            Err(err) => BackendHealth::unavailable(format!("MLX health request failed: {err}")),
        }
    }
}

fn http_byte_stream(
    response: reqwest::Response,
    kind: MlxTransportRequestKind,
    read_timeout: Duration,
    cancellation: CancellationToken,
) -> MlxTransportByteStream {
    async_stream::try_stream! {
        let mut bytes = response.bytes_stream();
        let mut saw_first_byte = false;
        loop {
            let item = if saw_first_byte {
                tokio::select! {
                    biased;
                    _ = cancellation.cancelled() => Err(MlxTransportError::cancelled()),
                    result = tokio::time::timeout(read_timeout, bytes.next()) => {
                        result.map_err(|_| MlxTransportError::read_timeout(read_timeout))
                    }
                }
            } else {
                tokio::select! {
                    biased;
                    _ = cancellation.cancelled() => Err(MlxTransportError::cancelled()),
                    item = bytes.next() => Ok(item),
                }
            };
            let Some(item) = item? else {
                break;
            };
            let bytes = item.map_err(|err| {
                MlxTransportError::other(
                    MlxBackendFailureKind::StreamRead,
                    kind.read_error_message(err),
                )
            })?;
            saw_first_byte = true;
            yield bytes.to_vec();
        }
    }
    .boxed()
}

pub const MLX_STALL_PREFIX: &str = "MLX_STALL:";

pub(super) fn format_duration(duration: Duration) -> String {
    if duration.as_secs() > 0 {
        format!("{}s", duration.as_secs())
    } else {
        format!("{}ms", duration.as_millis())
    }
}
