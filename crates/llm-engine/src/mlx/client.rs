use std::time::Duration;
use url::Url;

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

#[derive(Debug, Clone)]
pub(super) enum MlxTransport {
    Http(MlxHttpTransport),
}

impl MlxTransport {
    pub(super) fn http(endpoint: Url, timeouts: MlxTimeouts) -> Self {
        Self::Http(MlxHttpTransport::new(endpoint, timeouts))
    }

    pub(super) fn request(&self, request: MlxUpstreamRequest) -> reqwest::RequestBuilder {
        match self {
            Self::Http(transport) => transport.request(request),
        }
    }

    pub(super) fn models_request(&self) -> reqwest::RequestBuilder {
        match self {
            Self::Http(transport) => transport.models_request(),
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct MlxHttpTransport {
    endpoint: Url,
    client: reqwest::Client,
}

impl MlxHttpTransport {
    fn new(endpoint: Url, timeouts: MlxTimeouts) -> Self {
        Self {
            endpoint,
            client: build_http_client(timeouts),
        }
    }

    fn request(&self, request: MlxUpstreamRequest) -> reqwest::RequestBuilder {
        let upstream_url = mlx_endpoint_url(&self.endpoint, request.protocol().endpoint_suffix());
        self.client
            .post(upstream_url)
            .header(reqwest::header::CONTENT_TYPE, request.content_type())
            .body(request.into_body())
    }

    fn models_request(&self) -> reqwest::RequestBuilder {
        self.client.get(mlx_endpoint_url(&self.endpoint, "models"))
    }
}

pub const MLX_STALL_PREFIX: &str = "MLX_STALL:";

pub(super) fn format_duration(duration: Duration) -> String {
    if duration.as_secs() > 0 {
        format!("{}s", duration.as_secs())
    } else {
        format!("{}ms", duration.as_millis())
    }
}
