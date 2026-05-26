use llm_api::{MAX_TOOL_SCHEMA_DEPTH, MAX_TOOL_SCHEMA_ENUM_VALUES, RequestLimits};
use llm_engine::{
    DEFAULT_INFERENCE_CONCURRENCY_LIMIT, DEFAULT_MODEL_ID, EngineOptions, PublicInferenceRateLimit,
    SnapshotBackendLoader, SnapshotBackendOptions, cli, open_snapshot_backend,
    parse_snapshot_model_family, router_builder,
};
#[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
use llm_engine::{
    DEFAULT_NATIVE_TEXT_MAX_NEW_TOKENS, DEFAULT_NATIVE_TEXT_MAX_PREFILL_TOKENS,
    NativeTextDiskCacheConfig, NativeTextLoadOptions, NativeTextRuntimeOptions,
};
#[cfg(feature = "mlx")]
use llm_engine::{MlxBackendOptions, MlxTimeouts, MlxToolParserMode};
use llm_hub::{ModelStore, SnapshotReadiness};
use std::{future::Future, net::SocketAddr};
use tracing_subscriber::EnvFilter;

#[cfg(feature = "bench")]
mod bench_compat;

const PROTOCOL_TEST_BACKEND_FLAG: &str = "--protocol-test-backend";
const DETERMINISTIC_TEST_BACKEND_FLAG: &str = "--deterministic-test-backend";
const PROTOCOL_TEST_BACKEND_ACK_FLAG: &str = "--i-understand-this-is-not-real-inference";
const DEFAULT_LOG_FILTER: &str = "info";
const MAX_INFERENCE_CONCURRENCY_LIMIT: usize = 256;
#[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
const MAX_NATIVE_TEXT_MAX_NEW_TOKENS: u32 = 65_536;
#[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
const MAX_NATIVE_TEXT_MAX_PREFILL_TOKENS: usize = 262_144;
#[cfg(feature = "test-utils")]
const PROTOCOL_TEST_BACKEND_WARNING: &str =
    "WARNING: SERVING WITH HARDCODED PROTOCOL TEST BACKEND - NOT REAL INFERENCE";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    let command = args.get(1).map(String::as_str).unwrap_or("serve");
    let command_args = args.get(2..).unwrap_or(&[]);
    init_tracing(command, command_args)?;
    ignore_sigpipe();
    match command {
        "serve" => {
            let serve_args = command_args.to_vec();
            if has_flag(&serve_args, "--help") || has_flag(&serve_args, "-h") {
                print_serve_help();
                return Ok(());
            }
            if let Some(protocol_backend_flag) = protocol_test_backend_flag(&serve_args)
                && !has_flag(&serve_args, PROTOCOL_TEST_BACKEND_ACK_FLAG)
            {
                anyhow::bail!(
                    "{protocol_backend_flag} serves hardcoded protocol fixtures and requires {PROTOCOL_TEST_BACKEND_ACK_FLAG}"
                );
            }
            let addr = flag_value(&serve_args, "--addr")
                .unwrap_or("127.0.0.1:3000")
                .parse::<SocketAddr>()?;
            let max_concurrent_requests = max_concurrent_requests_from_args(&serve_args)?;
            let configured_admin_token = flag_value(&serve_args, "--admin-token")
                .map(str::to_owned)
                .or_else(|| std::env::var("LLM_ENGINE_ADMIN_TOKEN").ok());
            let model_home = flag_value(&serve_args, "--model-home")
                .map(std::path::PathBuf::from)
                .or_else(|| std::env::var_os("LLM_MODEL_HOME").map(std::path::PathBuf::from));
            let model_home_for_records = model_home
                .clone()
                .unwrap_or_else(|| std::path::PathBuf::from(".llm-models"));
            let hub_endpoint = flag_value(&serve_args, "--hub-endpoint")
                .map(str::to_owned)
                .or_else(|| std::env::var("LLM_HUB_ENDPOINT").ok());
            let canonical_tool_schemas = canonical_tool_schemas_enabled(&serve_args)?;
            let request_limits = request_limits_from_args(&serve_args)?;
            let public_inference_rate_limit = public_inference_rate_limit_from_args(&serve_args)?;
            let stream_stall_timeout = serve_stream_stall_timeout_from_args(&serve_args)?;
            let tls_config = serve_tls_config_from_args(&serve_args)?;
            let admin_auth = admin_auth_config(configured_admin_token, addr)?;
            if admin_auth.generated {
                emit_generated_admin_token_warning(addr, &admin_auth.token);
            }
            if tls_config.is_none() && !addr.ip().is_loopback() {
                tracing::warn!(
                    %addr,
                    "serving plain HTTP on a non-loopback address; configure --tls-cert/--tls-key or terminate TLS at a reverse proxy"
                );
            }
            let options = EngineOptions {
                concurrency_limit: max_concurrent_requests,
                admin_token: Some(admin_auth.token),
                model_home,
                hub_endpoint,
                hf_token: std::env::var("HF_TOKEN").ok(),
                canonical_tool_schemas,
                public_inference_rate_limit,
                request_limits,
                stream_stall_timeout,
                ..EngineOptions::default()
            };
            let snapshot_alias = flag_value(&serve_args, "--snapshot-alias")
                .or_else(|| flag_value(&serve_args, "--model-alias"));
            if flag_value(&serve_args, "--snapshot").is_some() && snapshot_alias.is_some() {
                anyhow::bail!(
                    "llm-engine serve accepts only one of --snapshot or --snapshot-alias"
                );
            }
            let snapshot_path = if let Some(snapshot_path) = flag_value(&serve_args, "--snapshot") {
                Some(std::path::PathBuf::from(snapshot_path))
            } else if let Some(alias) = snapshot_alias {
                let snapshot = ModelStore::new(&model_home_for_records)
                    .resolve_snapshot_alias(alias)
                    .await?;
                Some(snapshot.path)
            } else {
                None
            };
            let router = if let Some(snapshot_path) = snapshot_path {
                let snapshot_readiness_mode =
                    cli::model::snapshot_readiness_mode_from_args(&serve_args)?;
                let model_id = flag_value(&serve_args, "--model-id")
                    .or(snapshot_alias)
                    .unwrap_or(DEFAULT_MODEL_ID);
                #[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
                let max_new_tokens = native_text_max_new_tokens_from_args(&serve_args)?;
                #[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
                let max_prefill_tokens = native_text_max_prefill_tokens_from_args(&serve_args)?;
                #[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
                let native_metal_weight_cache_bytes =
                    flag_value(&serve_args, "--native-metal-weight-cache-bytes")
                        .map(str::parse::<u64>)
                        .transpose()?;
                #[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
                let native_prefix_cache_bytes = native_prefix_cache_bytes_from_args(&serve_args)?;
                #[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
                let native_prefix_disk_cache =
                    native_prefix_disk_cache_config_from_args(&serve_args)?;
                #[cfg(feature = "mlx")]
                let mlx_endpoint = if let Some(endpoint) = flag_value(&serve_args, "--mlx-endpoint")
                {
                    url::Url::parse(endpoint)?
                } else if let Ok(endpoint) = std::env::var("MLX_LM_ENDPOINT") {
                    url::Url::parse(&endpoint)?
                } else {
                    MlxBackendOptions::default().endpoint
                };
                #[cfg(feature = "mlx")]
                let mlx_stream_usage = mlx_stream_usage_enabled(&serve_args)?;
                #[cfg(feature = "mlx")]
                let mlx_tool_parser = mlx_tool_parser_mode_from_args(&serve_args)?;
                #[cfg(feature = "mlx")]
                let mlx_timeouts = {
                    let defaults = MlxTimeouts::default();
                    let connect = flag_value(&serve_args, "--mlx-connect-timeout")
                        .map(str::parse::<u64>)
                        .transpose()?
                        .map(std::time::Duration::from_secs);
                    let request = flag_value(&serve_args, "--mlx-request-timeout")
                        .map(str::parse::<u64>)
                        .transpose()?
                        .map(std::time::Duration::from_secs);
                    let read = flag_value(&serve_args, "--mlx-read-timeout")
                        .map(str::parse::<u64>)
                        .transpose()?
                        .map(std::time::Duration::from_secs);
                    MlxTimeouts {
                        connect: connect.unwrap_or(defaults.connect),
                        request: request.unwrap_or(defaults.request),
                        read: read.unwrap_or(defaults.read),
                    }
                };
                let loader = flag_value(&serve_args, "--loader")
                    .or_else(|| flag_value(&serve_args, "--backend"))
                    .map(SnapshotBackendLoader::parse)
                    .transpose()?;
                let family = flag_value(&serve_args, "--family")
                    .map(parse_snapshot_model_family)
                    .transpose()?;
                if tokio::fs::try_exists(snapshot_path.join("llm-engine-manifest.json")).await? {
                    let record = ModelStore::inspect_snapshot_readiness_with_mode(
                        &snapshot_path,
                        snapshot_readiness_mode,
                    )
                    .await?;
                    if !matches!(record.readiness, SnapshotReadiness::Ready) {
                        let status = record.readiness.status();
                        let reason = record.readiness.reason().unwrap_or("unknown reason");
                        anyhow::bail!(
                            "snapshot readiness {status} ({}) failed for `{}`: {reason}",
                            snapshot_readiness_mode.as_str(),
                            snapshot_path.display()
                        );
                    }
                    tracing::info!(
                        snapshot = %snapshot_path.display(),
                        readiness_mode = snapshot_readiness_mode.as_str(),
                        "snapshot readiness validated"
                    );
                }
                let backend = open_snapshot_backend(
                    model_id,
                    &snapshot_path,
                    SnapshotBackendOptions {
                        loader,
                        family,
                        #[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
                        native_text: NativeTextLoadOptions::with_runtime_options(
                            NativeTextRuntimeOptions {
                                eager_materialize_shards: has_flag(
                                    &serve_args,
                                    "--eager-materialize-shards",
                                ),
                                metal_weight_cache_bytes: native_metal_weight_cache_bytes,
                                prefix_cache_bytes: native_prefix_cache_bytes,
                                prefix_disk_cache: native_prefix_disk_cache,
                                warm_metal_weight_cache: has_flag(
                                    &serve_args,
                                    "--warm-native-metal-weight-cache",
                                ),
                            },
                        ),
                        #[cfg(feature = "mlx")]
                        mlx: MlxBackendOptions {
                            endpoint: mlx_endpoint,
                            timeouts: mlx_timeouts,
                            include_stream_usage: mlx_stream_usage,
                            tool_parser: mlx_tool_parser,
                            ..MlxBackendOptions::default()
                        },
                        #[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
                        max_new_tokens,
                        #[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
                        max_prefill_tokens,
                    },
                )
                .await?;
                if let Err(err) = ModelStore::mark_snapshot_used(&snapshot_path).await {
                    tracing::warn!(error = %err, snapshot = %snapshot_path.display(), "failed to record snapshot usage");
                }
                if let Err(err) = ModelStore::new(&model_home_for_records)
                    .record_snapshot_alias(model_id, &snapshot_path)
                    .await
                {
                    tracing::warn!(error = %err, alias = model_id, snapshot = %snapshot_path.display(), "failed to record model alias");
                }
                let builder = router_builder(backend).with_options(options);
                builder.build()?
            } else if protocol_test_backend_flag(&serve_args).is_some() {
                #[cfg(feature = "test-utils")]
                {
                    tracing::warn!("{}", PROTOCOL_TEST_BACKEND_WARNING);
                    eprintln!("{PROTOCOL_TEST_BACKEND_WARNING}");
                    let backend = Box::new(
                        llm_backend::ProtocolTestBackend::new(
                            DEFAULT_MODEL_ID,
                            "hello from rust native backend",
                        )
                        .with_required_tool_protocol()
                        .with_json_object_protocol(),
                    );
                    let builder = router_builder(backend).with_options(options);
                    builder.build()?
                }
                #[cfg(not(feature = "test-utils"))]
                {
                    anyhow::bail!(
                        "--protocol-test-backend requires the test-utils feature; \
                         this binary was built without it"
                    );
                }
            } else {
                anyhow::bail!("llm-engine serve requires --snapshot <path> for inference serving");
            };
            let listener = tokio::net::TcpListener::bind(addr).await?;
            if let Some(tls_config) = tls_config {
                tracing::info!(
                    %addr,
                    tls_cert = %tls_config.cert_path().display(),
                    "llm-engine HTTPS listening"
                );
                llm_server::serve_tls_with_graceful_shutdown(
                    listener,
                    router,
                    tls_config,
                    shutdown_signal(),
                )
                .await?;
            } else {
                tracing::info!(%addr, "llm-engine HTTP listening");
                llm_server::serve_with_graceful_shutdown(listener, router, shutdown_signal())
                    .await?;
            }
        }
        #[cfg(feature = "bench")]
        "bench" => bench_compat::run(command_args.to_vec())?,
        #[cfg(not(feature = "bench"))]
        "bench" => anyhow::bail!(
            "the bench command requires the llm-engine `bench` feature; rebuild with --features bench"
        ),
        "model" => cli::model::run(command_args.to_vec()).await?,
        other => anyhow::bail!("unknown command `{other}`"),
    }
    Ok(())
}

fn init_tracing(command: &str, args: &[String]) -> anyhow::Result<()> {
    let filter = tracing_env_filter(command, args, std::env::var("RUST_LOG").ok().as_deref())?;
    tracing_subscriber::fmt().with_env_filter(filter).init();
    Ok(())
}

fn tracing_env_filter(
    command: &str,
    args: &[String],
    rust_log: Option<&str>,
) -> anyhow::Result<EnvFilter> {
    let directive = tracing_filter_directive(command, args, rust_log)?;
    EnvFilter::try_new(&directive)
        .map_err(|err| anyhow::anyhow!("invalid tracing filter `{directive}`: {err}"))
}

fn tracing_filter_directive(
    command: &str,
    args: &[String],
    rust_log: Option<&str>,
) -> anyhow::Result<String> {
    if command == "serve"
        && let Some(level) = flag_value(args, "--log-level")
    {
        validate_log_level(level)?;
        return Ok(level.to_owned());
    }
    let rust_log = rust_log.map(str::trim).filter(|value| !value.is_empty());
    if let Some(directive) = rust_log {
        EnvFilter::try_new(directive)
            .map_err(|err| anyhow::anyhow!("invalid RUST_LOG `{directive}`: {err}"))?;
        return Ok(directive.to_owned());
    }
    Ok(DEFAULT_LOG_FILTER.to_owned())
}

fn validate_log_level(level: &str) -> anyhow::Result<()> {
    match level {
        "trace" | "debug" | "info" | "warn" | "error" | "off" => Ok(()),
        other => {
            anyhow::bail!("--log-level must be trace|debug|info|warn|error|off, got `{other}`")
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ShutdownTrigger {
    Sigint,
    Sigterm,
}

impl ShutdownTrigger {
    fn as_str(self) -> &'static str {
        match self {
            ShutdownTrigger::Sigint => "SIGINT",
            ShutdownTrigger::Sigterm => "SIGTERM",
        }
    }
}

async fn wait_for_shutdown_request<CtrlC, Terminate>(
    ctrl_c: CtrlC,
    terminate: Terminate,
) -> ShutdownTrigger
where
    CtrlC: Future<Output = std::io::Result<()>>,
    Terminate: Future<Output = std::io::Result<()>>,
{
    tokio::select! {
        result = ctrl_c => {
            if let Err(err) = result {
                tracing::warn!(error = %err, "failed while waiting for SIGINT shutdown signal");
            }
            ShutdownTrigger::Sigint
        }
        result = terminate => {
            if let Err(err) = result {
                tracing::warn!(error = %err, "failed while waiting for SIGTERM shutdown signal");
            }
            ShutdownTrigger::Sigterm
        }
    }
}

async fn shutdown_signal() {
    let trigger = wait_for_shutdown_request(tokio::signal::ctrl_c(), sigterm_signal()).await;
    tracing::info!(
        signal = trigger.as_str(),
        "shutdown signal received; draining in-flight requests"
    );
}

#[cfg(unix)]
async fn sigterm_signal() -> std::io::Result<()> {
    let mut signal = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    let _ = signal.recv().await;
    Ok(())
}

#[cfg(not(unix))]
async fn sigterm_signal() -> std::io::Result<()> {
    std::future::pending::<std::io::Result<()>>().await
}

#[cfg(unix)]
fn ignore_sigpipe() {
    // SAFETY: Installing SIG_IGN for SIGPIPE is process-global startup configuration.
    // It converts broken pipe writes into ordinary EPIPE errors instead of terminating
    // the server during streaming responses.
    let previous_handler = unsafe { libc::signal(libc::SIGPIPE, libc::SIG_IGN) };
    if previous_handler == libc::SIG_ERR {
        tracing::warn!("failed to ignore SIGPIPE; broken pipe writes may terminate the process");
    }
}

#[cfg(not(unix))]
fn ignore_sigpipe() {}

fn print_serve_help() {
    println!(
        "\
Usage: llm-engine serve [OPTIONS]

Options:
  --addr <host:port>                         Listen address [default: 127.0.0.1:3000]
  --tls-cert <path>                          PEM certificate chain for HTTPS; requires --tls-key
  --tls-key <path>                           PEM private key for HTTPS; requires --tls-cert
  --snapshot <path>                          Inference snapshot path
  --snapshot-alias <alias>                   Resolve snapshot path from the model store
  --snapshot-readiness <fast|deep>           Startup readiness check [default: fast; deep hashes all manifest files]
  --model-alias <alias>                      Alias for --snapshot-alias
  --model-id <id>                            Served model id [default: {}]
  --loader <native-metal|mlx>                Override snapshot loader when no manifest is present
  --backend <native-metal|mlx>               Alias for --loader
  --family <qwen|deep_seek|gemma|llama>      Model family for raw snapshots without a Kir manifest
                                             Raw native snapshots infer Qwen/Gemma from config.json; raw MLX requires --family
  --max-new-tokens <n>                       Native text maximum generated tokens [default: 256]
  --max-prefill-tokens <n>                   Native text prefill chunk size [default: 2048; lower only for memory-constrained correctness probes]
  --max-concurrent-requests <n>              Maximum concurrent requests [default: 4]
  --log-level <level>                        Startup log level override for serve; one of trace|debug|info|warn|error|off [default: RUST_LOG or info]
  --max-json-body-bytes <bytes>              Maximum JSON request body bytes [default: 16777216]
  --max-message-content-bytes <bytes>        Maximum bytes per chat message content [default: 8388608]
  --max-completion-prompt-bytes <bytes>      Maximum bytes per text completion prompt [default: 8388608]
  --max-tool-schema-depth <depth>            Maximum nested JSON Schema object depth below a tool parameters root [default: {}]
  --max-tool-schema-enum-values <n>          Maximum values in one tool JSON Schema enum array [default: {}]
  --max-public-inference-requests-per-second <n>
                                             Public chat/completion requests per second [default: 60]
  --stream-stall-timeout <secs>              Stream stall timeout after semantic output starts [default: 300]
  --admin-token <token>                      Bearer token for admin endpoints; loopback without one generates a temporary token
  --model-home <path>                        Model store root
  --hub-endpoint <url>                       Hugging Face compatible Hub endpoint
  --mlx-endpoint <url>                       Loopback mlx_lm.server or mlx_vlm.server /v1 endpoint [default: http://127.0.0.1:8080/v1]
  --mlx-connect-timeout <secs>               MLX sidecar connect timeout [default: 5]
  --mlx-request-timeout <secs>               MLX sidecar whole-request timeout [default: 600]
  --mlx-read-timeout <secs>                  MLX sidecar per-chunk read timeout [default: 60]
  --mlx-stream-usage <true|false>            Forward stream_options.include_usage to MLX sidecars [default: true, env: LLM_ENGINE_MLX_STREAM_USAGE]
  --mlx-tool-parser <auto|json|qwen-xml>     MLX streamed tool parser [default: auto]
  --native-prefix-cache-bytes <bytes>        Native prefix cache budget [default: 536870912, env: LLM_ENGINE_PREFIX_CACHE_BYTES]
  --native-prefix-cache-ssd                  Enable opt-in native SSD prefix cache tier [env: LLM_ENGINE_PREFIX_CACHE_SSD=1]
  --native-prefix-cache-ssd-path <path>      Native SSD prefix cache root [default: ~/.cache/kir-ai/kv-cache, env: LLM_ENGINE_PREFIX_CACHE_SSD_PATH]
  --native-prefix-cache-ssd-writer-queue <n> Native SSD prefix cache bounded writer queue [default: 8, env: LLM_ENGINE_PREFIX_CACHE_SSD_WRITER_QUEUE]
  --native-prefix-cache-ssd-block-tokens <n> Native SSD prefix cache token block size [default: 256, env: LLM_ENGINE_PREFIX_CACHE_SSD_BLOCK_TOKENS]
  --native-metal-weight-cache-bytes <bytes>  Native Metal BF16 weight cache budget
  --warm-native-metal-weight-cache           Warm native Metal BF16 weight cache at startup
  --eager-materialize-shards                 Materialize indexed safetensor shards at startup
  --canonical-tool-schemas                   Canonicalize tool schemas before runtime prompt/cache use [env: LLM_ENGINE_CANONICAL_TOOL_SCHEMAS=1]
  -h, --help                                 Print help",
        DEFAULT_MODEL_ID, MAX_TOOL_SCHEMA_DEPTH, MAX_TOOL_SCHEMA_ENUM_VALUES
    );
}

#[derive(Debug, PartialEq, Eq)]
struct AdminAuthConfig {
    token: String,
    generated: bool,
}

fn admin_auth_config(
    configured_admin_token: Option<String>,
    addr: SocketAddr,
) -> anyhow::Result<AdminAuthConfig> {
    if let Some(token) = configured_admin_token {
        return Ok(AdminAuthConfig {
            token,
            generated: false,
        });
    }

    if !addr.ip().is_loopback() {
        anyhow::bail!(
            "serving admin endpoints on a non-loopback address requires --admin-token or LLM_ENGINE_ADMIN_TOKEN"
        );
    }

    Ok(AdminAuthConfig {
        token: generate_admin_token()?,
        generated: true,
    })
}

fn generate_admin_token() -> anyhow::Result<String> {
    let mut token_bytes = [0_u8; 32];
    getrandom::fill(&mut token_bytes)
        .map_err(|err| anyhow::anyhow!("failed to generate admin token: {err}"))?;
    Ok(hex_token(&token_bytes))
}

fn hex_token(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";

    let mut token = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let byte = *byte;
        token.push(HEX[(byte >> 4) as usize] as char);
        token.push(HEX[(byte & 0x0f) as usize] as char);
    }
    token
}

fn emit_generated_admin_token_warning(addr: SocketAddr, token: &str) {
    tracing::warn!(
        %addr,
        "no admin token configured; generated a temporary bearer token for loopback admin endpoints"
    );
    eprintln!(
        "WARNING: no --admin-token or LLM_ENGINE_ADMIN_TOKEN configured; generated a temporary admin token for this process."
    );
    eprintln!("Admin requests must include: Authorization: Bearer {token}");
}

fn request_limits_from_args(args: &[String]) -> anyhow::Result<RequestLimits> {
    let defaults = RequestLimits::default();
    Ok(RequestLimits {
        json_body_bytes: parse_positive_usize_flag(
            args,
            "--max-json-body-bytes",
            defaults.json_body_bytes,
        )?,
        message_content_bytes: parse_positive_usize_flag(
            args,
            "--max-message-content-bytes",
            defaults.message_content_bytes,
        )?,
        completion_prompt_bytes: parse_positive_usize_flag(
            args,
            "--max-completion-prompt-bytes",
            defaults.completion_prompt_bytes,
        )?,
        tool_schema_depth: parse_positive_usize_flag(
            args,
            "--max-tool-schema-depth",
            defaults.tool_schema_depth,
        )?,
        tool_schema_enum_values: parse_positive_usize_flag(
            args,
            "--max-tool-schema-enum-values",
            defaults.tool_schema_enum_values,
        )?,
    })
}

fn max_concurrent_requests_from_args(args: &[String]) -> anyhow::Result<usize> {
    parse_bounded_usize_flag(
        args,
        "--max-concurrent-requests",
        DEFAULT_INFERENCE_CONCURRENCY_LIMIT,
        1,
        MAX_INFERENCE_CONCURRENCY_LIMIT,
    )
}

#[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
fn native_text_max_new_tokens_from_args(args: &[String]) -> anyhow::Result<u32> {
    parse_bounded_u32_flag(
        args,
        "--max-new-tokens",
        DEFAULT_NATIVE_TEXT_MAX_NEW_TOKENS,
        1,
        MAX_NATIVE_TEXT_MAX_NEW_TOKENS,
    )
}

#[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
fn native_text_max_prefill_tokens_from_args(args: &[String]) -> anyhow::Result<usize> {
    parse_bounded_usize_flag(
        args,
        "--max-prefill-tokens",
        DEFAULT_NATIVE_TEXT_MAX_PREFILL_TOKENS,
        1,
        MAX_NATIVE_TEXT_MAX_PREFILL_TOKENS,
    )
}

fn public_inference_rate_limit_from_args(
    args: &[String],
) -> anyhow::Result<PublicInferenceRateLimit> {
    let defaults = PublicInferenceRateLimit::default();
    Ok(PublicInferenceRateLimit {
        max_requests: parse_positive_usize_flag(
            args,
            "--max-public-inference-requests-per-second",
            defaults.max_requests,
        )?,
        window: defaults.window,
    })
}

fn parse_bounded_usize_flag(
    args: &[String],
    flag: &str,
    default: usize,
    min: usize,
    max: usize,
) -> anyhow::Result<usize> {
    let Some(value) = flag_value(args, flag) else {
        return Ok(default);
    };
    let parsed = value.parse::<usize>().map_err(|err| {
        anyhow::anyhow!("{flag} must be an integer between {min} and {max}: {err}")
    })?;
    ensure_inclusive_range(flag, parsed, min, max)?;
    Ok(parsed)
}

#[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
fn parse_bounded_u32_flag(
    args: &[String],
    flag: &str,
    default: u32,
    min: u32,
    max: u32,
) -> anyhow::Result<u32> {
    let Some(value) = flag_value(args, flag) else {
        return Ok(default);
    };
    let parsed = value.parse::<u32>().map_err(|err| {
        anyhow::anyhow!("{flag} must be an integer between {min} and {max}: {err}")
    })?;
    ensure_inclusive_range(flag, parsed, min, max)?;
    Ok(parsed)
}

fn ensure_inclusive_range<T>(name: &str, value: T, min: T, max: T) -> anyhow::Result<()>
where
    T: Copy + Ord + std::fmt::Display,
{
    if value < min || value > max {
        anyhow::bail!("{name} must be between {min} and {max}, got {value}");
    }
    Ok(())
}

fn parse_positive_usize_flag(args: &[String], flag: &str, default: usize) -> anyhow::Result<usize> {
    let Some(value) = flag_value(args, flag) else {
        return Ok(default);
    };
    let parsed = value.parse::<usize>()?;
    if parsed == 0 {
        anyhow::bail!("{flag} must be greater than 0");
    }
    Ok(parsed)
}

fn serve_stream_stall_timeout_from_args(
    args: &[String],
) -> anyhow::Result<Option<std::time::Duration>> {
    let secs = parse_positive_u64_flag(args, "--stream-stall-timeout", 300)?;
    Ok(Some(std::time::Duration::from_secs(secs)))
}

#[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
fn native_prefix_cache_bytes_from_args(args: &[String]) -> anyhow::Result<Option<u64>> {
    let env_value = std::env::var("LLM_ENGINE_PREFIX_CACHE_BYTES").ok();
    native_prefix_cache_bytes_from_env(args, env_value.as_deref())
}

#[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
fn native_prefix_cache_bytes_from_env(
    args: &[String],
    env_value: Option<&str>,
) -> anyhow::Result<Option<u64>> {
    if let Some(value) = flag_value(args, "--native-prefix-cache-bytes") {
        return parse_u64_config("--native-prefix-cache-bytes", value).map(Some);
    }
    env_value
        .map(|value| parse_u64_config("LLM_ENGINE_PREFIX_CACHE_BYTES", value))
        .transpose()
}

#[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
fn native_prefix_disk_cache_config_from_args(
    args: &[String],
) -> anyhow::Result<Option<NativeTextDiskCacheConfig>> {
    let enabled = if has_flag(args, "--native-prefix-cache-ssd") {
        true
    } else if let Ok(value) = std::env::var("LLM_ENGINE_PREFIX_CACHE_SSD") {
        parse_bool_config("LLM_ENGINE_PREFIX_CACHE_SSD", &value)?
    } else {
        false
    };
    if !enabled {
        return Ok(None);
    }
    let root = flag_value(args, "--native-prefix-cache-ssd-path")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("LLM_ENGINE_PREFIX_CACHE_SSD_PATH").map(Into::into))
        .unwrap_or_else(NativeTextDiskCacheConfig::default_root);
    let writer_queue_depth =
        if let Some(value) = flag_value(args, "--native-prefix-cache-ssd-writer-queue") {
            parse_positive_usize_config("--native-prefix-cache-ssd-writer-queue", value)?
        } else if let Ok(value) = std::env::var("LLM_ENGINE_PREFIX_CACHE_SSD_WRITER_QUEUE") {
            parse_positive_usize_config("LLM_ENGINE_PREFIX_CACHE_SSD_WRITER_QUEUE", &value)?
        } else {
            NativeTextDiskCacheConfig::default().writer_queue_depth
        };
    let block_token_count =
        if let Some(value) = flag_value(args, "--native-prefix-cache-ssd-block-tokens") {
            parse_positive_usize_config("--native-prefix-cache-ssd-block-tokens", value)?
        } else if let Ok(value) = std::env::var("LLM_ENGINE_PREFIX_CACHE_SSD_BLOCK_TOKENS") {
            parse_positive_usize_config("LLM_ENGINE_PREFIX_CACHE_SSD_BLOCK_TOKENS", &value)?
        } else {
            NativeTextDiskCacheConfig::default().block_token_count
        };
    Ok(Some(
        NativeTextDiskCacheConfig::for_root(root)
            .with_writer_queue_depth(writer_queue_depth)
            .with_block_token_count(block_token_count),
    ))
}

#[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
fn parse_u64_config(name: &str, value: &str) -> anyhow::Result<u64> {
    value
        .parse::<u64>()
        .map_err(|err| anyhow::anyhow!("{name} must be a non-negative integer: {err}"))
}

#[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
fn parse_positive_usize_config(name: &str, value: &str) -> anyhow::Result<usize> {
    let parsed = value
        .parse::<usize>()
        .map_err(|err| anyhow::anyhow!("{name} must be a positive integer: {err}"))?;
    if parsed == 0 {
        anyhow::bail!("{name} must be greater than 0");
    }
    Ok(parsed)
}

fn parse_positive_u64_flag(args: &[String], flag: &str, default: u64) -> anyhow::Result<u64> {
    let Some(value) = flag_value(args, flag) else {
        return Ok(default);
    };
    let parsed = value
        .parse::<u64>()
        .map_err(|err| anyhow::anyhow!("{flag} must be a positive integer: {err}"))?;
    if parsed == 0 {
        anyhow::bail!("{flag} must be greater than 0");
    }
    Ok(parsed)
}

fn canonical_tool_schemas_enabled(args: &[String]) -> anyhow::Result<bool> {
    if has_flag(args, "--canonical-tool-schemas") {
        return Ok(true);
    }
    let Some(value) = std::env::var("LLM_ENGINE_CANONICAL_TOOL_SCHEMAS").ok() else {
        return Ok(false);
    };
    parse_bool_config("LLM_ENGINE_CANONICAL_TOOL_SCHEMAS", &value)
}

#[cfg(feature = "mlx")]
fn mlx_stream_usage_enabled(args: &[String]) -> anyhow::Result<bool> {
    let env_value = std::env::var("LLM_ENGINE_MLX_STREAM_USAGE").ok();
    mlx_stream_usage_enabled_from_env(args, env_value.as_deref())
}

#[cfg(feature = "mlx")]
fn mlx_tool_parser_mode_from_args(args: &[String]) -> anyhow::Result<MlxToolParserMode> {
    let Some(value) = flag_value(args, "--mlx-tool-parser") else {
        return Ok(MlxToolParserMode::Auto);
    };
    MlxToolParserMode::parse(value).ok_or_else(|| {
        anyhow::anyhow!("--mlx-tool-parser must be auto|json|qwen-xml, got `{value}`")
    })
}

#[cfg(feature = "mlx")]
fn mlx_stream_usage_enabled_from_env(
    args: &[String],
    env_value: Option<&str>,
) -> anyhow::Result<bool> {
    if let Some(value) = flag_value(args, "--mlx-stream-usage") {
        return parse_bool_config("--mlx-stream-usage", value);
    }
    env_value
        .map(|value| parse_bool_config("LLM_ENGINE_MLX_STREAM_USAGE", value))
        .unwrap_or(Ok(true))
}

fn serve_tls_config_from_args(args: &[String]) -> anyhow::Result<Option<llm_server::TlsConfig>> {
    match (
        flag_value(args, "--tls-cert"),
        flag_value(args, "--tls-key"),
    ) {
        (None, None) => Ok(None),
        (Some(cert_path), Some(key_path)) => Ok(Some(llm_server::TlsConfig::new(
            std::path::PathBuf::from(cert_path),
            std::path::PathBuf::from(key_path),
        ))),
        (Some(_), None) => anyhow::bail!("--tls-cert requires --tls-key"),
        (None, Some(_)) => anyhow::bail!("--tls-key requires --tls-cert"),
    }
}

fn parse_bool_config(name: &str, value: &str) -> anyhow::Result<bool> {
    match value {
        "1" | "true" | "TRUE" | "yes" | "YES" => Ok(true),
        "0" | "false" | "FALSE" | "no" | "NO" => Ok(false),
        other => anyhow::bail!("{name} must be 1/0 or true/false, got `{other}`"),
    }
}

#[cfg(test)]
mod serve_arg_tests {
    use super::*;
    use std::time::Duration;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[tokio::test]
    async fn shutdown_request_resolves_on_sigint_future() {
        let trigger = wait_for_shutdown_request(
            async { Ok(()) },
            std::future::pending::<std::io::Result<()>>(),
        )
        .await;

        assert_eq!(trigger, ShutdownTrigger::Sigint);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn shutdown_request_resolves_on_sigterm_future() {
        let trigger =
            wait_for_shutdown_request(std::future::pending::<std::io::Result<()>>(), async {
                Ok(())
            })
            .await;

        assert_eq!(trigger, ShutdownTrigger::Sigterm);
    }

    #[test]
    fn admin_auth_config_preserves_explicit_token() {
        let config = admin_auth_config(
            Some("secret-admin-token".to_owned()),
            "127.0.0.1:3000".parse().expect("socket addr parses"),
        )
        .expect("explicit admin token config is valid");

        assert_eq!(
            config,
            AdminAuthConfig {
                token: "secret-admin-token".to_owned(),
                generated: false,
            }
        );
    }

    #[test]
    fn admin_auth_config_generates_loopback_token_when_unset() {
        let config = admin_auth_config(None, "127.0.0.1:3000".parse().expect("socket addr parses"))
            .expect("loopback without explicit token generates a token");

        assert!(config.generated);
        assert_eq!(config.token.len(), 64);
        assert!(config.token.bytes().all(|byte| byte.is_ascii_hexdigit()));
        assert!(config.token.bytes().all(|byte| !byte.is_ascii_uppercase()));
    }

    #[test]
    fn admin_auth_config_rejects_non_loopback_without_token() {
        let err = admin_auth_config(None, "0.0.0.0:3000".parse().expect("socket addr parses"))
            .expect_err("non-loopback without admin token fails");

        assert!(
            err.to_string()
                .contains("non-loopback address requires --admin-token"),
            "error: {err}"
        );
    }

    #[test]
    fn tracing_filter_defaults_to_info_and_honors_rust_log() {
        assert_eq!(
            tracing_filter_directive("serve", &args(&[]), None).expect("default filter"),
            "info"
        );
        assert_eq!(
            tracing_filter_directive("serve", &args(&[]), Some("llm_engine=debug"))
                .expect("RUST_LOG filter"),
            "llm_engine=debug"
        );
    }

    #[test]
    fn serve_log_level_overrides_rust_log() {
        assert_eq!(
            tracing_filter_directive(
                "serve",
                &args(&["--log-level", "trace"]),
                Some("llm_engine=info"),
            )
            .expect("CLI log level wins"),
            "trace"
        );
        assert_eq!(
            tracing_filter_directive("serve", &args(&["--log-level", "info"]), Some("[invalid"))
                .expect("CLI log level overrides invalid RUST_LOG"),
            "info"
        );

        let err = tracing_filter_directive(
            "serve",
            &args(&["--log-level", "verbose"]),
            Some("llm_engine=info"),
        )
        .expect_err("invalid CLI log level fails before RUST_LOG");
        assert!(err.to_string().contains("--log-level"), "error: {err}");
    }

    #[test]
    fn rust_log_filter_rejects_invalid_directives() {
        let err = tracing_filter_directive("serve", &args(&[]), Some("[invalid"))
            .expect_err("invalid RUST_LOG fails");

        assert!(err.to_string().contains("RUST_LOG"), "error: {err}");
    }

    #[test]
    fn max_concurrent_requests_rejects_values_outside_startup_range() {
        assert_eq!(
            max_concurrent_requests_from_args(&args(&[])).expect("default concurrency"),
            DEFAULT_INFERENCE_CONCURRENCY_LIMIT
        );
        assert_eq!(
            max_concurrent_requests_from_args(&args(&["--max-concurrent-requests", "256"]))
                .expect("upper bound accepted"),
            256
        );

        let zero = max_concurrent_requests_from_args(&args(&["--max-concurrent-requests", "0"]))
            .expect_err("zero concurrency fails");
        assert!(
            zero.to_string().contains("--max-concurrent-requests"),
            "error: {zero}"
        );
        let too_high =
            max_concurrent_requests_from_args(&args(&["--max-concurrent-requests", "257"]))
                .expect_err("high concurrency fails");
        assert!(
            too_high.to_string().contains("--max-concurrent-requests"),
            "error: {too_high}"
        );
    }

    #[test]
    fn request_limits_parse_tool_schema_depth() {
        assert_eq!(
            request_limits_from_args(&args(&[]))
                .expect("default request limits parse")
                .tool_schema_depth,
            MAX_TOOL_SCHEMA_DEPTH
        );
        assert_eq!(
            request_limits_from_args(&args(&["--max-tool-schema-depth", "7"]))
                .expect("custom tool schema depth parses")
                .tool_schema_depth,
            7
        );

        let zero = request_limits_from_args(&args(&["--max-tool-schema-depth", "0"]))
            .expect_err("zero tool schema depth fails");
        assert!(
            zero.to_string().contains("--max-tool-schema-depth"),
            "error: {zero}"
        );
    }

    #[test]
    fn request_limits_parse_tool_schema_enum_values() {
        assert_eq!(
            request_limits_from_args(&args(&[]))
                .expect("default request limits parse")
                .tool_schema_enum_values,
            MAX_TOOL_SCHEMA_ENUM_VALUES
        );
        assert_eq!(
            request_limits_from_args(&args(&["--max-tool-schema-enum-values", "7"]))
                .expect("custom tool schema enum size parses")
                .tool_schema_enum_values,
            7
        );

        let zero = request_limits_from_args(&args(&["--max-tool-schema-enum-values", "0"]))
            .expect_err("zero tool schema enum size fails");
        assert!(
            zero.to_string().contains("--max-tool-schema-enum-values"),
            "error: {zero}"
        );
    }

    #[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
    #[test]
    fn native_text_generation_limits_reject_values_outside_startup_range() {
        assert_eq!(
            native_text_max_new_tokens_from_args(&args(&[])).expect("default max new tokens"),
            DEFAULT_NATIVE_TEXT_MAX_NEW_TOKENS
        );
        assert_eq!(
            native_text_max_new_tokens_from_args(&args(&["--max-new-tokens", "65536"]))
                .expect("max new tokens upper bound"),
            65_536
        );

        let zero = native_text_max_new_tokens_from_args(&args(&["--max-new-tokens", "0"]))
            .expect_err("zero max new tokens fails");
        assert!(
            zero.to_string().contains("--max-new-tokens"),
            "error: {zero}"
        );
        let too_high = native_text_max_new_tokens_from_args(&args(&["--max-new-tokens", "65537"]))
            .expect_err("high max new tokens fails");
        assert!(
            too_high.to_string().contains("--max-new-tokens"),
            "error: {too_high}"
        );
    }

    #[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
    #[test]
    fn native_text_prefill_limits_reject_values_outside_startup_range() {
        assert_eq!(
            native_text_max_prefill_tokens_from_args(&args(&[]))
                .expect("default max prefill tokens"),
            DEFAULT_NATIVE_TEXT_MAX_PREFILL_TOKENS
        );
        assert_eq!(
            native_text_max_prefill_tokens_from_args(&args(&["--max-prefill-tokens", "262144"]))
                .expect("prefill upper bound"),
            262_144
        );

        let zero = native_text_max_prefill_tokens_from_args(&args(&["--max-prefill-tokens", "0"]))
            .expect_err("zero prefill tokens fails");
        assert!(
            zero.to_string().contains("--max-prefill-tokens"),
            "error: {zero}"
        );
        let too_high =
            native_text_max_prefill_tokens_from_args(&args(&["--max-prefill-tokens", "262145"]))
                .expect_err("high prefill tokens fails");
        assert!(
            too_high.to_string().contains("--max-prefill-tokens"),
            "error: {too_high}"
        );
    }

    #[test]
    fn public_inference_rate_limit_defaults_to_60_per_second() {
        assert_eq!(
            public_inference_rate_limit_from_args(&args(&[])).expect("default rate limit parses"),
            PublicInferenceRateLimit {
                max_requests: 60,
                window: Duration::from_secs(1),
            }
        );
    }

    #[test]
    fn public_inference_rate_limit_parses_custom_positive_value() {
        assert_eq!(
            public_inference_rate_limit_from_args(&args(&[
                "--max-public-inference-requests-per-second",
                "7"
            ]))
            .expect("custom rate limit parses"),
            PublicInferenceRateLimit {
                max_requests: 7,
                window: Duration::from_secs(1),
            }
        );
    }

    #[test]
    fn public_inference_rate_limit_rejects_zero() {
        let err = public_inference_rate_limit_from_args(&args(&[
            "--max-public-inference-requests-per-second",
            "0",
        ]))
        .expect_err("zero rate limit fails");
        assert!(err.to_string().contains("greater than 0"), "error: {err}");
    }

    #[test]
    fn stream_stall_timeout_defaults_to_300_seconds() {
        assert_eq!(
            serve_stream_stall_timeout_from_args(&args(&[])).expect("default timeout parses"),
            Some(Duration::from_secs(300))
        );
    }

    #[test]
    fn stream_stall_timeout_parses_custom_seconds() {
        assert_eq!(
            serve_stream_stall_timeout_from_args(&args(&["--stream-stall-timeout", "42"]))
                .expect("custom timeout parses"),
            Some(Duration::from_secs(42))
        );
    }

    #[test]
    fn stream_stall_timeout_rejects_zero_and_non_numeric_values() {
        let zero = serve_stream_stall_timeout_from_args(&args(&["--stream-stall-timeout", "0"]))
            .expect_err("zero timeout fails");
        assert!(zero.to_string().contains("greater than 0"), "error: {zero}");

        let non_numeric =
            serve_stream_stall_timeout_from_args(&args(&["--stream-stall-timeout", "abc"]))
                .expect_err("non-numeric timeout fails");
        assert!(
            non_numeric.to_string().contains("--stream-stall-timeout"),
            "error: {non_numeric}"
        );
    }

    #[test]
    fn tls_config_is_absent_by_default() {
        assert!(
            serve_tls_config_from_args(&args(&[]))
                .expect("default TLS config parses")
                .is_none()
        );
    }

    #[test]
    fn tls_config_requires_certificate_and_key_together() {
        let missing_key = serve_tls_config_from_args(&args(&["--tls-cert", "cert.pem"]))
            .expect_err("certificate without key fails");
        assert!(
            missing_key.to_string().contains("--tls-key"),
            "error: {missing_key}"
        );

        let missing_cert = serve_tls_config_from_args(&args(&["--tls-key", "key.pem"]))
            .expect_err("key without certificate fails");
        assert!(
            missing_cert.to_string().contains("--tls-cert"),
            "error: {missing_cert}"
        );
    }

    #[test]
    fn tls_config_parses_explicit_certificate_and_key_paths() {
        let config =
            serve_tls_config_from_args(&args(&["--tls-cert", "cert.pem", "--tls-key", "key.pem"]))
                .expect("TLS config parses")
                .expect("TLS config present");

        assert_eq!(config.cert_path(), std::path::Path::new("cert.pem"));
        assert_eq!(config.key_path(), std::path::Path::new("key.pem"));
    }

    #[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
    #[test]
    fn native_prefix_cache_bytes_accepts_zero_from_env() {
        assert_eq!(
            native_prefix_cache_bytes_from_env(&args(&[]), Some("0"))
                .expect("zero disables prefix cache stores"),
            Some(0)
        );
    }

    #[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
    #[test]
    fn native_prefix_cache_bytes_flag_overrides_env() {
        assert_eq!(
            native_prefix_cache_bytes_from_env(
                &args(&["--native-prefix-cache-bytes", "64"]),
                Some("0"),
            )
            .expect("flag override parses"),
            Some(64)
        );
    }

    #[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
    #[test]
    fn native_prefix_disk_cache_is_disabled_by_default_and_parses_opt_in_flags() {
        assert!(
            native_prefix_disk_cache_config_from_args(&args(&[]))
                .expect("default SSD config parses")
                .is_none()
        );

        let config = native_prefix_disk_cache_config_from_args(&args(&[
            "--native-prefix-cache-ssd",
            "--native-prefix-cache-ssd-path",
            "/tmp/kir-ai-kv",
            "--native-prefix-cache-ssd-writer-queue",
            "3",
            "--native-prefix-cache-ssd-block-tokens",
            "2",
        ]))
        .expect("opt-in SSD config parses")
        .expect("SSD config is enabled");

        assert_eq!(config.root, std::path::PathBuf::from("/tmp/kir-ai-kv"));
        assert_eq!(config.writer_queue_depth, 3);
        assert_eq!(config.block_token_count, 2);
    }
}

#[cfg(all(test, feature = "mlx"))]
mod tests {
    use super::*;

    #[test]
    fn mlx_stream_usage_defaults_true_and_parses_flag() {
        assert!(mlx_stream_usage_enabled_from_env(&[], None).expect("default parses"));
        assert!(
            !mlx_stream_usage_enabled_from_env(
                &["--mlx-stream-usage".to_owned(), "false".to_owned()],
                None
            )
            .expect("flag parses")
        );
        assert!(
            mlx_stream_usage_enabled_from_env(
                &["--mlx-stream-usage".to_owned(), "true".to_owned()],
                Some("false")
            )
            .expect("flag overrides env")
        );
    }

    #[test]
    fn mlx_stream_usage_parses_env_value() {
        assert!(
            !mlx_stream_usage_enabled_from_env(&[], Some("0")).expect("zero env disables usage")
        );
        assert!(
            mlx_stream_usage_enabled_from_env(&[], Some("yes")).expect("yes env enables usage")
        );
    }

    #[test]
    fn mlx_tool_parser_mode_defaults_auto_and_parses_flag() {
        assert_eq!(
            mlx_tool_parser_mode_from_args(&[]).expect("default parser mode"),
            MlxToolParserMode::Auto
        );
        assert_eq!(
            mlx_tool_parser_mode_from_args(&[
                "--mlx-tool-parser".to_owned(),
                "qwen-xml".to_owned()
            ])
            .expect("qwen XML parser mode"),
            MlxToolParserMode::QwenXml
        );
        let err =
            mlx_tool_parser_mode_from_args(&["--mlx-tool-parser".to_owned(), "xml".to_owned()])
                .expect_err("invalid parser mode fails");
        assert!(err.to_string().contains("auto|json|qwen-xml"));
    }
}

fn flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find_map(|window| (window[0] == flag).then_some(window[1].as_str()))
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|arg| arg == flag)
}

fn protocol_test_backend_flag(args: &[String]) -> Option<&'static str> {
    if has_flag(args, PROTOCOL_TEST_BACKEND_FLAG) {
        Some(PROTOCOL_TEST_BACKEND_FLAG)
    } else if has_flag(args, DETERMINISTIC_TEST_BACKEND_FLAG) {
        Some(DETERMINISTIC_TEST_BACKEND_FLAG)
    } else {
        None
    }
}
