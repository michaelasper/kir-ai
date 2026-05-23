use llm_api::RequestLimits;
#[cfg(feature = "diagnostics")]
use llm_backend::native::{
    CpuNativeMatvecBackend, InferenceScratchpad, QwenMoeDims, SafeTensorFile, SafeTensorShardStore,
    qwen_embedding_and_layer0_norm,
};
#[cfg(feature = "diagnostics")]
use llm_backend::native::{
    qwen_decoder_layer_first_token, qwen_final_norm, qwen_layer_moe_forward_in_place,
    qwen_layer_moe_router, qwen_layer0_linear_attention_first_token,
    qwen_layer0_linear_attention_projections, qwen_layer0_post_attention_norm,
    qwen_linear_decoder_layer_first_token, qwen_lm_head_top_k,
};
use llm_engine::{
    DEFAULT_MODEL_ID, EngineOptions, PublicInferenceRateLimit, SnapshotBackendLoader,
    SnapshotBackendOptions, configured_hub_client, model_cli, open_snapshot_backend,
    parse_snapshot_model_family, router_builder,
};
#[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
use llm_engine::{
    DEFAULT_NATIVE_TEXT_MAX_NEW_TOKENS, DEFAULT_NATIVE_TEXT_MAX_PREFILL_TOKENS,
    NativeTextLoadOptions, NativeTextRuntimeOptions,
};
#[cfg(feature = "mlx")]
use llm_engine::{MlxBackendOptions, MlxTimeouts, MlxToolParserMode};
use llm_hub::{ModelLifecycleService, ModelStore, SnapshotReadiness};
#[cfg(feature = "diagnostics")]
use llm_models::QwenModelSpec;
#[cfg(feature = "diagnostics")]
use llm_tokenizer::HuggingFaceTokenizer;
use std::net::SocketAddr;

const PROTOCOL_TEST_BACKEND_FLAG: &str = "--protocol-test-backend";
const DETERMINISTIC_TEST_BACKEND_FLAG: &str = "--deterministic-test-backend";
const PROTOCOL_TEST_BACKEND_ACK_FLAG: &str = "--i-understand-this-is-not-real-inference";
#[cfg(feature = "test-utils")]
const PROTOCOL_TEST_BACKEND_WARNING: &str =
    "WARNING: SERVING WITH HARDCODED PROTOCOL TEST BACKEND - NOT REAL INFERENCE";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let command = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "serve".to_owned());
    match command.as_str() {
        "serve" => {
            let serve_args = std::env::args().skip(2).collect::<Vec<_>>();
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
            let max_concurrent_requests = flag_value(&serve_args, "--max-concurrent-requests")
                .map(str::parse::<usize>)
                .transpose()?
                .unwrap_or(1);
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
                    model_cli::snapshot_readiness_mode_from_args(&serve_args)?;
                let model_id = flag_value(&serve_args, "--model-id")
                    .or(snapshot_alias)
                    .unwrap_or(DEFAULT_MODEL_ID);
                #[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
                let max_new_tokens = flag_value(&serve_args, "--max-new-tokens")
                    .map(str::parse::<u32>)
                    .transpose()?
                    .unwrap_or(DEFAULT_NATIVE_TEXT_MAX_NEW_TOKENS);
                #[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
                let max_prefill_tokens = flag_value(&serve_args, "--max-prefill-tokens")
                    .map(str::parse::<usize>)
                    .transpose()?
                    .unwrap_or(DEFAULT_NATIVE_TEXT_MAX_PREFILL_TOKENS);
                #[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
                let native_metal_weight_cache_bytes =
                    flag_value(&serve_args, "--native-metal-weight-cache-bytes")
                        .map(str::parse::<u64>)
                        .transpose()?;
                #[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
                let native_prefix_cache_bytes = native_prefix_cache_bytes_from_args(&serve_args)?;
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
                llm_server::serve_tls(listener, router, tls_config).await?;
            } else {
                tracing::info!(%addr, "llm-engine HTTP listening");
                llm_server::serve(listener, router).await?;
            }
        }
        #[cfg(feature = "bench")]
        "bench" => llm_bench::run_bench_command(std::env::args().skip(2).collect()).await?,
        #[cfg(not(feature = "bench"))]
        "bench" => anyhow::bail!(
            "the bench command requires the llm-engine `bench` feature; rebuild with --features bench"
        ),
        "model" => run_model_command(std::env::args().skip(2).collect()).await?,
        other => anyhow::bail!("unknown command `{other}`"),
    }
    Ok(())
}

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
  --max-concurrent-requests <n>              Maximum concurrent requests [default: 1]
  --max-json-body-bytes <bytes>              Maximum JSON request body bytes [default: 16777216]
  --max-message-content-bytes <bytes>        Maximum bytes per chat message content [default: 8388608]
  --max-completion-prompt-bytes <bytes>      Maximum bytes per text completion prompt [default: 8388608]
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
  --native-metal-weight-cache-bytes <bytes>  Native Metal BF16 weight cache budget
  --warm-native-metal-weight-cache           Warm native Metal BF16 weight cache at startup
  --eager-materialize-shards                 Materialize indexed safetensor shards at startup
  --canonical-tool-schemas                   Canonicalize tool schemas before runtime prompt/cache use [env: LLM_ENGINE_CANONICAL_TOOL_SCHEMAS=1]
  -h, --help                                 Print help",
        DEFAULT_MODEL_ID
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
    })
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
fn parse_u64_config(name: &str, value: &str) -> anyhow::Result<u64> {
    value
        .parse::<u64>()
        .map_err(|err| anyhow::anyhow!("{name} must be a non-negative integer: {err}"))
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

async fn run_model_command(args: Vec<String>) -> anyhow::Result<()> {
    let Some(subcommand) = args.first() else {
        anyhow::bail!(
            "usage: llm-engine model plan <repo> [--revision <rev>] [--profile <profile>] [--hub-endpoint <url>]"
        );
    };
    match subcommand.as_str() {
        "list" => {
            let root = model_home_from_args(&args);
            let readiness_mode = model_cli::snapshot_readiness_mode_from_args(&args)?;
            let value = model_cli::model_list_json_with_mode(&root, readiness_mode).await?;
            println!("{}", serde_json::to_string_pretty(&value)?);
        }
        #[cfg(feature = "diagnostics")]
        "inspect" => {
            let snapshot_path = args.get(1).ok_or_else(|| {
                anyhow::anyhow!("usage: llm-engine model inspect <snapshot-path>")
            })?;
            let value = model_cli::model_inspect_json(snapshot_path).await?;
            println!("{}", serde_json::to_string_pretty(&value)?);
        }
        #[cfg(not(feature = "diagnostics"))]
        "inspect" => diagnostics_feature_required("inspect")?,
        "verify" => {
            let snapshot_path = args
                .get(1)
                .ok_or_else(|| anyhow::anyhow!("usage: llm-engine model verify <snapshot-path>"))?;
            let value = model_cli::model_verify_json(snapshot_path).await?;
            println!("{}", serde_json::to_string_pretty(&value)?);
        }
        "prune" => {
            let mode = model_cli::prune_mode_from_args(&args)?;
            let root = model_home_from_args(&args);
            let policy = model_cli::prune_policy_from_args(&args)?;
            let value = model_cli::model_prune_json(&root, policy, mode).await?;
            println!("{}", serde_json::to_string_pretty(&value)?);
        }
        #[cfg(feature = "diagnostics")]
        "inspect-safetensors" => {
            let path = args.get(1).ok_or_else(|| {
                anyhow::anyhow!("usage: llm-engine model inspect-safetensors <path>")
            })?;
            let tensor_file = SafeTensorFile::open(path)?;
            let header = tensor_file.header();
            let sample_tensors: Vec<_> = header.tensor_names().take(8).collect();
            let tensor_name = flag_value(&args, "--tensor");
            let tensor = tensor_name
                .map(|name| {
                    let metadata = header.tensor_metadata(name)?;
                    let range = header.tensor_data_range(name)?;
                    anyhow::Ok(serde_json::json!({
                        "name": metadata.name,
                        "dtype": metadata.dtype,
                        "shape": metadata.shape,
                        "byte_len": metadata.byte_len,
                        "file_byte_range": {
                            "start": range.start,
                            "end": range.end
                        }
                    }))
                })
                .transpose()?;
            let bf16_row = match (tensor_name, flag_value(&args, "--bf16-row")) {
                (Some(name), Some(row)) => {
                    let row = row.parse::<usize>()?;
                    let values = tensor_file.bf16_row_f32(name, row)?;
                    let limit = flag_value(&args, "--limit")
                        .map(str::parse::<usize>)
                        .transpose()?
                        .unwrap_or(8);
                    Some(serde_json::json!({
                        "tensor": name,
                        "row": row,
                        "values_read": values.len(),
                        "values_prefix": values.into_iter().take(limit).collect::<Vec<_>>()
                    }))
                }
                (None, Some(_)) => anyhow::bail!("--bf16-row requires --tensor <name>"),
                _ => None,
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "path": path,
                    "file_len": header.file_len(),
                    "header_len": header.header_len(),
                    "data_start": header.data_start(),
                    "tensor_count": header.tensor_count(),
                    "sample_tensors": sample_tensors,
                    "tensor": tensor,
                    "bf16_row": bf16_row
                }))?
            );
        }
        #[cfg(not(feature = "diagnostics"))]
        "inspect-safetensors" => diagnostics_feature_required("inspect-safetensors")?,
        #[cfg(feature = "diagnostics")]
        "inspect-tensor" => {
            let snapshot_path = args.get(1).ok_or_else(|| {
                anyhow::anyhow!(
                    "usage: llm-engine model inspect-tensor <snapshot-path> --tensor <name>"
                )
            })?;
            let tensor_name = flag_value(&args, "--tensor")
                .ok_or_else(|| anyhow::anyhow!("inspect-tensor requires --tensor <name>"))?;
            let store = SafeTensorShardStore::open(snapshot_path)?;
            let shard_path = store.tensor_shard_path(tensor_name)?;
            let metadata = store.tensor_metadata(tensor_name)?;
            let bf16_row = flag_value(&args, "--bf16-row")
                .map(|row| {
                    let row = row.parse::<usize>()?;
                    let values = store.bf16_row_f32(tensor_name, row)?;
                    let limit = flag_value(&args, "--limit")
                        .map(str::parse::<usize>)
                        .transpose()?
                        .unwrap_or(8);
                    anyhow::Ok(serde_json::json!({
                        "row": row,
                        "values_read": values.len(),
                        "values_prefix": values.into_iter().take(limit).collect::<Vec<_>>()
                    }))
                })
                .transpose()?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "snapshot_path": snapshot_path,
                    "tensor": {
                        "name": metadata.name,
                        "dtype": metadata.dtype,
                        "shape": metadata.shape,
                        "byte_len": metadata.byte_len,
                        "shard_path": shard_path
                    },
                    "bf16_row": bf16_row
                }))?
            );
        }
        #[cfg(not(feature = "diagnostics"))]
        "inspect-tensor" => diagnostics_feature_required("inspect-tensor")?,
        #[cfg(feature = "diagnostics")]
        "inspect-qwen-input" => {
            let snapshot_path = args.get(1).ok_or_else(|| {
                anyhow::anyhow!(
                    "usage: llm-engine model inspect-qwen-input <snapshot-path> --token-id <id>"
                )
            })?;
            let token_id = flag_value(&args, "--token-id")
                .ok_or_else(|| anyhow::anyhow!("inspect-qwen-input requires --token-id <id>"))?
                .parse::<usize>()?;
            let limit = flag_value(&args, "--limit")
                .map(str::parse::<usize>)
                .transpose()?
                .unwrap_or(8);
            let config_json =
                tokio::fs::read_to_string(std::path::Path::new(snapshot_path).join("config.json"))
                    .await?;
            let spec = QwenModelSpec::from_config_json(&config_json)?;
            let store = SafeTensorShardStore::open(snapshot_path)?;
            let lm_head_top_k = flag_value(&args, "--lm-head-top-k")
                .map(str::parse::<usize>)
                .transpose()?;
            let chunk_rows = flag_value(&args, "--chunk-rows")
                .map(str::parse::<usize>)
                .transpose()?
                .unwrap_or(512);
            let tokenizer = lm_head_top_k
                .map(|_| {
                    HuggingFaceTokenizer::from_file(
                        std::path::Path::new(snapshot_path).join("tokenizer.json"),
                    )
                })
                .transpose()?;
            let probe = qwen_embedding_and_layer0_norm(
                &store,
                token_id,
                spec.hidden_size as usize,
                spec.rms_norm_eps,
            )?;
            let linear_layers = if let Some(count) = flag_value(&args, "--linear-layers") {
                let count = count.parse::<usize>()?;
                let mut hidden = probe.embedding.clone();
                let mut layers = Vec::new();
                for layer_idx in 0..count {
                    hidden = qwen_linear_decoder_layer_first_token(
                        &store,
                        &spec,
                        layer_idx,
                        &hidden,
                        &CpuNativeMatvecBackend,
                    )
                    .await?;
                    layers.push(serde_json::json!({
                        "layer": layer_idx,
                        "hidden_prefix": hidden.iter().copied().take(limit).collect::<Vec<_>>()
                    }));
                }
                Some(serde_json::json!({
                    "layers": layers,
                    "final_hidden_prefix": hidden.iter().copied().take(limit).collect::<Vec<_>>()
                }))
            } else {
                None
            };
            let layers = if let Some(count) = flag_value(&args, "--layers") {
                let count = count.parse::<usize>()?;
                let mut hidden = probe.embedding.clone();
                let mut layers = Vec::new();
                for layer_idx in 0..count {
                    hidden = qwen_decoder_layer_first_token(
                        &store,
                        &spec,
                        layer_idx,
                        &hidden,
                        &CpuNativeMatvecBackend,
                    )
                    .await?;
                    layers.push(serde_json::json!({
                        "layer": layer_idx,
                        "kind": format!("{:?}", spec.layer_kinds[layer_idx]),
                        "hidden_prefix": hidden.iter().copied().take(limit).collect::<Vec<_>>()
                    }));
                }
                let lm_head = if let Some(top_k) = lm_head_top_k {
                    Some(
                        qwen_lm_head_json(
                            &store,
                            tokenizer.as_ref(),
                            &hidden,
                            QwenLmHeadJsonOptions {
                                hidden_size: spec.hidden_size as usize,
                                rms_norm_eps: spec.rms_norm_eps,
                                top_k,
                                chunk_rows,
                                limit,
                            },
                        )
                        .await?,
                    )
                } else {
                    None
                };
                Some(serde_json::json!({
                    "layers": layers,
                    "final_hidden_prefix": hidden.iter().copied().take(limit).collect::<Vec<_>>(),
                    "lm_head": lm_head
                }))
            } else {
                None
            };
            let run_layer0_attention = args.iter().any(|arg| arg == "--layer0-attention")
                || args.iter().any(|arg| arg == "--layer0-router")
                || args.iter().any(|arg| arg == "--layer0-moe");
            let run_layer0_projections =
                args.iter().any(|arg| arg == "--layer0-projections") || run_layer0_attention;
            let projections = if run_layer0_projections {
                Some(qwen_layer0_linear_attention_projections(&store, &probe.normalized).await?)
            } else {
                None
            };
            let layer0_attention_output = if run_layer0_attention {
                let projections = projections.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("--layer0-projections must be enabled for --layer0-attention")
                })?;
                Some(qwen_layer0_linear_attention_first_token(&store, &spec, projections).await?)
            } else {
                None
            };
            let layer0_attention = layer0_attention_output.as_ref().map(|output| {
                serde_json::json!({
                    "output_len": output.len(),
                    "output_prefix": output.iter().copied().take(limit).collect::<Vec<_>>()
                })
            });
            let run_layer0_router = args.iter().any(|arg| arg == "--layer0-router")
                || args.iter().any(|arg| arg == "--layer0-moe");
            let mut attention_residual = None;
            let mut post_attention_norm = None;
            let mut router_probe = None;
            let layer0_router = if run_layer0_router {
                let attention_output = layer0_attention_output.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("--layer0-attention must be enabled for --layer0-router")
                })?;
                let residual = probe
                    .embedding
                    .iter()
                    .zip(attention_output)
                    .map(|(embedding, attention)| embedding + attention)
                    .collect::<Vec<_>>();
                let post_attention = qwen_layer0_post_attention_norm(
                    &store,
                    &probe.embedding,
                    attention_output,
                    spec.hidden_size as usize,
                    spec.rms_norm_eps,
                )
                .await?;
                let top_k = flag_value(&args, "--top-k")
                    .map(str::parse::<usize>)
                    .transpose()?
                    .unwrap_or(spec.num_experts_per_tok as usize);
                let router = qwen_layer_moe_router(
                    &store,
                    0,
                    &post_attention,
                    top_k,
                    &CpuNativeMatvecBackend,
                )
                .await?;
                attention_residual = Some(residual);
                post_attention_norm = Some(post_attention.clone());
                router_probe = Some(router.clone());
                Some(serde_json::json!({
                    "post_attention_norm_prefix": post_attention.iter().copied().take(limit).collect::<Vec<_>>(),
                    "logits_len": router.logits.len(),
                    "selected": router.selected.iter().map(|item| {
                        serde_json::json!({
                            "index": item.index,
                            "weight": item.weight
                        })
                    }).collect::<Vec<_>>()
                }))
            } else {
                None
            };
            let layer0_moe = if args.iter().any(|arg| arg == "--layer0-moe") {
                let post_attention = post_attention_norm.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("--layer0-router must be enabled for --layer0-moe")
                })?;
                let router = router_probe.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("--layer0-router must be enabled for --layer0-moe")
                })?;
                let mut moe_output = vec![0.0; spec.hidden_size as usize];
                let mut scratch = InferenceScratchpad::default();
                qwen_layer_moe_forward_in_place(
                    &store,
                    0,
                    &QwenMoeDims::from_spec(&spec),
                    post_attention,
                    router,
                    &CpuNativeMatvecBackend,
                    &mut scratch,
                    &mut moe_output,
                )
                .await?;
                let final_hidden = attention_residual
                    .as_ref()
                    .ok_or_else(|| {
                        anyhow::anyhow!("--layer0-router must be enabled for --layer0-moe")
                    })?
                    .iter()
                    .zip(&moe_output)
                    .map(|(residual, moe)| residual + moe)
                    .collect::<Vec<_>>();
                let lm_head = if let Some(top_k) = lm_head_top_k {
                    Some(
                        qwen_lm_head_json(
                            &store,
                            tokenizer.as_ref(),
                            &final_hidden,
                            QwenLmHeadJsonOptions {
                                hidden_size: spec.hidden_size as usize,
                                rms_norm_eps: spec.rms_norm_eps,
                                top_k,
                                chunk_rows,
                                limit,
                            },
                        )
                        .await?,
                    )
                } else {
                    None
                };
                Some(serde_json::json!({
                    "moe_output_len": moe_output.len(),
                    "moe_output_prefix": moe_output.iter().copied().take(limit).collect::<Vec<_>>(),
                    "final_hidden_prefix": final_hidden.iter().copied().take(limit).collect::<Vec<_>>(),
                    "lm_head": lm_head
                }))
            } else {
                None
            };
            let layer0_projections = projections.as_ref().map(|projections| {
                serde_json::json!({
                    "qkv_len": projections.qkv.len(),
                    "z_len": projections.z.len(),
                    "b_len": projections.b.len(),
                    "a_len": projections.a.len(),
                    "qkv_prefix": projections.qkv.iter().copied().take(limit).collect::<Vec<_>>(),
                    "z_prefix": projections.z.iter().copied().take(limit).collect::<Vec<_>>(),
                    "b_prefix": projections.b.iter().copied().take(limit).collect::<Vec<_>>(),
                    "a_prefix": projections.a.iter().copied().take(limit).collect::<Vec<_>>()
                })
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "snapshot_path": snapshot_path,
                    "token_id": token_id,
                    "hidden_size": spec.hidden_size,
                    "rms_norm_eps": spec.rms_norm_eps,
                    "embedding_prefix": probe.embedding.iter().copied().take(limit).collect::<Vec<_>>(),
                    "normalized_prefix": probe.normalized.iter().copied().take(limit).collect::<Vec<_>>(),
                    "values_read": probe.normalized.len(),
                    "linear_layers": linear_layers,
                    "layers": layers,
                    "layer0_projections": layer0_projections,
                    "layer0_attention": layer0_attention,
                    "layer0_router": layer0_router,
                    "layer0_moe": layer0_moe
                }))?
            );
        }
        #[cfg(not(feature = "diagnostics"))]
        "inspect-qwen-input" => diagnostics_feature_required("inspect-qwen-input")?,
        "plan" | "pull" => {
            let request = model_cli::model_lifecycle_request_from_args(subcommand, &args)?;
            let token = std::env::var("HF_TOKEN").ok();
            let hub_endpoint = flag_value(&args, "--hub-endpoint")
                .map(str::to_owned)
                .or_else(|| std::env::var("LLM_HUB_ENDPOINT").ok());
            let client = configured_hub_client(hub_endpoint.as_deref(), token.as_deref())?;
            let lifecycle = ModelLifecycleService::new(&client, token.as_deref());
            if subcommand == "plan" {
                let plan = lifecycle.plan(&request).await?;
                println!("{}", serde_json::to_string_pretty(&plan)?);
            } else {
                let root = model_home_from_args(&args);
                let store = ModelStore::new(root);
                let snapshot = lifecycle.pull(&store, &request).await?;
                ModelStore::mark_snapshot_used(&snapshot.path).await?;
                if let Some(alias) = flag_value(&args, "--alias") {
                    store.record_snapshot_alias(alias, &snapshot.path).await?;
                }
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "snapshot_path": snapshot.path,
                        "manifest_digest": snapshot.manifest_digest,
                        "resolved_commit": snapshot.manifest.resolved_commit,
                        "files": snapshot.manifest.files.len()
                    }))?
                );
            }
        }
        other => anyhow::bail!("unknown model subcommand `{other}`"),
    }
    Ok(())
}

#[cfg(not(feature = "diagnostics"))]
fn diagnostics_feature_required(subcommand: &str) -> anyhow::Result<()> {
    anyhow::bail!(
        "llm-engine model {subcommand} requires the llm-engine `diagnostics` feature; rebuild with --features diagnostics"
    )
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

fn model_home_from_args(args: &[String]) -> std::path::PathBuf {
    flag_value(args, "--model-home")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("LLM_MODEL_HOME").map(std::path::PathBuf::from))
        .unwrap_or_else(|| std::path::PathBuf::from(".llm-models"))
}

#[cfg(feature = "diagnostics")]
#[derive(Debug, Clone, Copy)]
struct QwenLmHeadJsonOptions {
    hidden_size: usize,
    rms_norm_eps: f32,
    top_k: usize,
    chunk_rows: usize,
    limit: usize,
}

#[cfg(feature = "diagnostics")]
async fn qwen_lm_head_json(
    store: &SafeTensorShardStore,
    tokenizer: Option<&HuggingFaceTokenizer>,
    hidden_states: &[f32],
    options: QwenLmHeadJsonOptions,
) -> anyhow::Result<serde_json::Value> {
    let final_norm = qwen_final_norm(
        store,
        hidden_states,
        options.hidden_size,
        options.rms_norm_eps,
        &CpuNativeMatvecBackend,
    )
    .await?;
    let top_logits = qwen_lm_head_top_k(
        store,
        &final_norm,
        options.top_k,
        options.chunk_rows,
        &CpuNativeMatvecBackend,
    )
    .await?;
    let mut logits = Vec::with_capacity(top_logits.len());
    for item in top_logits {
        let decoded = if let Some(tokenizer) = tokenizer {
            let token_id = u32::try_from(item.index)?;
            Some(tokenizer.decode(&[token_id], false)?)
        } else {
            None
        };
        logits.push(serde_json::json!({
            "index": item.index,
            "logit": item.logit,
            "decoded": decoded
        }));
    }

    Ok(serde_json::json!({
        "final_norm_prefix": final_norm.iter().copied().take(options.limit).collect::<Vec<_>>(),
        "top_logits": logits
    }))
}
