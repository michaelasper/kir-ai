use llm_api::RequestLimits;
use llm_hub::HubClient;
use std::{path::PathBuf, time::Duration};

pub(super) const DEFAULT_STREAM_STALL_TIMEOUT: Duration = Duration::from_secs(300);

#[derive(Debug, Clone)]
pub struct EngineOptions {
    pub concurrency_limit: usize,
    pub scheduler_queue_limit: usize,
    pub scheduler_queue_timeout: Option<Duration>,
    pub scheduler_prefill_threshold_chars: usize,
    pub scheduler_prefill_burst: usize,
    pub admin_token: Option<String>,
    pub model_home: Option<PathBuf>,
    pub hub_endpoint: Option<String>,
    pub hf_token: Option<String>,
    pub stream_stall_timeout: Option<Duration>,
    pub canonical_tool_schemas: bool,
    pub request_limits: RequestLimits,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            concurrency_limit: 0,
            scheduler_queue_limit: 1,
            scheduler_queue_timeout: Some(Duration::from_secs(30)),
            scheduler_prefill_threshold_chars: 4096,
            scheduler_prefill_burst: 1,
            admin_token: None,
            model_home: None,
            hub_endpoint: None,
            hf_token: None,
            stream_stall_timeout: Some(DEFAULT_STREAM_STALL_TIMEOUT),
            canonical_tool_schemas: false,
            request_limits: RequestLimits::default(),
        }
    }
}

#[derive(Debug)]
pub struct EngineConfigError {
    message: String,
}

impl EngineConfigError {
    pub(super) fn missing_backend() -> Self {
        Self {
            message: "llm-engine router construction requires an explicit backend; use RouterBuilder::new(backend).build() for inference or build_router_with_protocol_test_backend() for protocol tests"
                .to_owned(),
        }
    }

    fn invalid_hub_endpoint(endpoint: &str, source: url::ParseError) -> Self {
        Self {
            message: format!("invalid hub endpoint `{endpoint}`: {source}"),
        }
    }

    fn insecure_hub_token_endpoint(endpoint: &url::Url) -> Self {
        Self {
            message: format!(
                "refusing to send HF_TOKEN to non-HTTPS hub endpoint `{endpoint}`; use HTTPS or a loopback endpoint for local development"
            ),
        }
    }
}

impl std::fmt::Display for EngineConfigError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for EngineConfigError {}

pub(super) fn parse_hub_client(
    endpoint: &str,
    hf_token: Option<&str>,
) -> Result<HubClient, EngineConfigError> {
    let endpoint = url::Url::parse(endpoint)
        .map_err(|err| EngineConfigError::invalid_hub_endpoint(endpoint, err))?;
    if hf_token.is_some() && endpoint.scheme() != "https" && !is_loopback_endpoint(&endpoint) {
        return Err(EngineConfigError::insecure_hub_token_endpoint(&endpoint));
    }
    Ok(HubClient::new(endpoint))
}

pub(super) fn default_model_home() -> PathBuf {
    std::env::var_os("LLM_MODEL_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".llm-models"))
}

fn is_loopback_endpoint(endpoint: &url::Url) -> bool {
    match endpoint.host() {
        Some(url::Host::Domain(domain)) => domain.eq_ignore_ascii_case("localhost"),
        Some(url::Host::Ipv4(addr)) => addr.is_loopback(),
        Some(url::Host::Ipv6(addr)) => addr.is_loopback(),
        None => false,
    }
}
