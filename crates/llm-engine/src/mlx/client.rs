use std::time::Duration;
use url::Url;

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
pub(super) struct MlxTimeouts {
    pub connect: Duration,
    pub request: Duration,
    pub read: Duration,
}

impl Default for MlxTimeouts {
    fn default() -> Self {
        Self {
            connect: Duration::from_secs(5),
            request: Duration::from_secs(300),
            read: Duration::from_secs(60),
        }
    }
}

pub(super) fn build_http_client(timeouts: MlxTimeouts) -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(timeouts.connect)
        .timeout(timeouts.request)
        .build()
        .expect("MLX HTTP client builds")
}

pub(super) fn format_duration(duration: Duration) -> String {
    if duration.as_secs() > 0 {
        format!("{}s", duration.as_secs())
    } else {
        format!("{}ms", duration.as_millis())
    }
}
