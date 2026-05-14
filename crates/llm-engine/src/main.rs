use llm_backend::{
    CpuNativeMatvecBackend, InferenceScratchpad, QwenMoeDims, SafeTensorFile, SafeTensorShardStore,
    qwen_embedding_and_layer0_norm,
};
use llm_backend::{
    qwen_decoder_layer_first_token, qwen_final_norm, qwen_layer_moe_forward_with_matvec_in_place,
    qwen_layer_moe_router_with_matvec, qwen_layer0_linear_attention_first_token,
    qwen_layer0_linear_attention_projections, qwen_layer0_post_attention_norm,
    qwen_linear_decoder_layer_first_token, qwen_lm_head_top_k,
};
use llm_engine::{
    DEFAULT_MODEL_ID, DEFAULT_NATIVE_TEXT_MAX_NEW_TOKENS, EngineOptions, MlxBackendOptions,
    MlxTimeouts, MlxToolParserMode, NativeTextLoadOptions, NativeTextRuntimeOptions,
    SnapshotBackendLoader, SnapshotBackendOptions, build_router_with_backend_and_options,
    build_router_with_backend_and_options_allowing_unauthenticated_admin, open_snapshot_backend,
    parse_snapshot_model_family,
};
use llm_hub::{
    DeletedSnapshot, HubClient, HubRepoId, ModelProfile, ModelStore, PromotedSnapshot,
    ProtectedSnapshot, PruneCandidate, PrunePlan, PrunePolicy, PruneReport, QuarantinedSnapshot,
    SnapshotRecord,
};
use llm_models::QwenModelSpec;
use llm_tokenizer::HuggingFaceTokenizer;
use serde_json::Value;
use std::{collections::HashMap, net::SocketAddr, path::Path};

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
            let admin_token = flag_value(&serve_args, "--admin-token")
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
            if admin_token.is_none() && !addr.ip().is_loopback() {
                anyhow::bail!(
                    "serving admin endpoints on a non-loopback address requires --admin-token or LLM_ENGINE_ADMIN_TOKEN"
                );
            }
            let allow_unauthenticated_admin = admin_token.is_none() && addr.ip().is_loopback();
            let options = EngineOptions {
                concurrency_limit: max_concurrent_requests,
                admin_token,
                model_home,
                hub_endpoint,
                hf_token: std::env::var("HF_TOKEN").ok(),
                canonical_tool_schemas,
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
                let model_id = flag_value(&serve_args, "--model-id")
                    .or(snapshot_alias)
                    .unwrap_or(DEFAULT_MODEL_ID);
                let max_new_tokens = flag_value(&serve_args, "--max-new-tokens")
                    .map(str::parse::<u32>)
                    .transpose()?
                    .unwrap_or(DEFAULT_NATIVE_TEXT_MAX_NEW_TOKENS);
                let max_prefill_tokens = flag_value(&serve_args, "--max-prefill-tokens")
                    .map(str::parse::<usize>)
                    .transpose()?
                    .unwrap_or(32);
                let native_metal_weight_cache_bytes =
                    flag_value(&serve_args, "--native-metal-weight-cache-bytes")
                        .map(str::parse::<u64>)
                        .transpose()?;
                let mlx_endpoint = if let Some(endpoint) = flag_value(&serve_args, "--mlx-endpoint")
                {
                    url::Url::parse(endpoint)?
                } else if let Ok(endpoint) = std::env::var("MLX_LM_ENDPOINT") {
                    url::Url::parse(&endpoint)?
                } else {
                    MlxBackendOptions::default().endpoint
                };
                let mlx_stream_usage = mlx_stream_usage_enabled(&serve_args)?;
                let mlx_tool_parser = mlx_tool_parser_mode_from_args(&serve_args)?;
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
                    ModelStore::verify_runnable_snapshot(&snapshot_path).await?;
                }
                let backend = open_snapshot_backend(
                    model_id,
                    &snapshot_path,
                    SnapshotBackendOptions {
                        loader,
                        family,
                        native_text: NativeTextLoadOptions::with_runtime_options(
                            NativeTextRuntimeOptions {
                                eager_materialize_shards: has_flag(
                                    &serve_args,
                                    "--eager-materialize-shards",
                                ),
                                metal_weight_cache_bytes: native_metal_weight_cache_bytes,
                                warm_metal_weight_cache: has_flag(
                                    &serve_args,
                                    "--warm-native-metal-weight-cache",
                                ),
                            },
                        ),
                        mlx: MlxBackendOptions {
                            endpoint: mlx_endpoint,
                            timeouts: mlx_timeouts,
                            include_stream_usage: mlx_stream_usage,
                            tool_parser: mlx_tool_parser,
                            ..MlxBackendOptions::default()
                        },
                        max_new_tokens,
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
                if allow_unauthenticated_admin {
                    build_router_with_backend_and_options_allowing_unauthenticated_admin(
                        backend, options,
                    )?
                } else {
                    build_router_with_backend_and_options(backend, options)?
                }
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
                    if allow_unauthenticated_admin {
                        build_router_with_backend_and_options_allowing_unauthenticated_admin(
                            backend, options,
                        )?
                    } else {
                        build_router_with_backend_and_options(backend, options)?
                    }
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
            tracing::info!(%addr, "llm-engine listening");
            llm_server::serve(listener, router).await?;
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
  --snapshot <path>                          Inference snapshot path
  --snapshot-alias <alias>                   Resolve snapshot path from the model store
  --model-alias <alias>                      Alias for --snapshot-alias
  --model-id <id>                            Served model id [default: {}]
  --loader <native-metal|mlx>                Override snapshot loader when no manifest is present
  --backend <native-metal|mlx>               Alias for --loader
  --family <qwen|deep_seek|gemma|llama>      Model family for raw snapshots without a Kir manifest
                                             Raw native snapshots infer Qwen/Gemma from config.json; raw MLX requires --family
  --max-new-tokens <n>                       Native text maximum generated tokens [default: 256]
  --max-prefill-tokens <n>                   Native text maximum prefill tokens
  --max-concurrent-requests <n>              Maximum concurrent requests [default: 1]
  --admin-token <token>                      Bearer token for admin endpoints
  --model-home <path>                        Model store root
  --hub-endpoint <url>                       Hugging Face compatible Hub endpoint
  --mlx-endpoint <url>                       Loopback mlx_lm.server or mlx_vlm.server /v1 endpoint [default: http://127.0.0.1:8080/v1]
  --mlx-connect-timeout <secs>               MLX sidecar connect timeout [default: 5]
  --mlx-request-timeout <secs>               MLX sidecar whole-request timeout [default: 600]
  --mlx-read-timeout <secs>                  MLX sidecar per-chunk read timeout [default: 60]
  --mlx-stream-usage <true|false>            Forward stream_options.include_usage to MLX sidecars [default: true, env: LLM_ENGINE_MLX_STREAM_USAGE]
  --mlx-tool-parser <auto|json|qwen-xml>     MLX streamed tool parser [default: auto]
  --native-metal-weight-cache-bytes <bytes>  Native Metal BF16 weight cache budget
  --warm-native-metal-weight-cache           Warm native Metal BF16 weight cache at startup
  --eager-materialize-shards                 Materialize indexed safetensor shards at startup
  --canonical-tool-schemas                   Canonicalize tool schemas before runtime prompt/cache use [env: LLM_ENGINE_CANONICAL_TOOL_SCHEMAS=1]
  -h, --help                                 Print help",
        DEFAULT_MODEL_ID
    );
}

fn canonical_tool_schemas_enabled(args: &[String]) -> anyhow::Result<bool> {
    if has_flag(args, "--canonical-tool-schemas") {
        return Ok(true);
    }
    let Some(value) = std::env::var("LLM_ENGINE_CANONICAL_TOOL_SCHEMAS").ok() else {
        return Ok(false);
    };
    match value.as_str() {
        "1" | "true" | "TRUE" | "yes" | "YES" => Ok(true),
        "0" | "false" | "FALSE" | "no" | "NO" => Ok(false),
        other => anyhow::bail!(
            "LLM_ENGINE_CANONICAL_TOOL_SCHEMAS must be 1/0 or true/false, got `{other}`"
        ),
    }
}

fn mlx_stream_usage_enabled(args: &[String]) -> anyhow::Result<bool> {
    let env_value = std::env::var("LLM_ENGINE_MLX_STREAM_USAGE").ok();
    mlx_stream_usage_enabled_from_env(args, env_value.as_deref())
}

fn mlx_tool_parser_mode_from_args(args: &[String]) -> anyhow::Result<MlxToolParserMode> {
    let Some(value) = flag_value(args, "--mlx-tool-parser") else {
        return Ok(MlxToolParserMode::Auto);
    };
    MlxToolParserMode::parse(value).ok_or_else(|| {
        anyhow::anyhow!("--mlx-tool-parser must be auto|json|qwen-xml, got `{value}`")
    })
}

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

fn parse_bool_config(name: &str, value: &str) -> anyhow::Result<bool> {
    match value {
        "1" | "true" | "TRUE" | "yes" | "YES" => Ok(true),
        "0" | "false" | "FALSE" | "no" | "NO" => Ok(false),
        other => anyhow::bail!("{name} must be 1/0 or true/false, got `{other}`"),
    }
}

#[cfg(test)]
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
            "usage: llm-engine model plan <repo> [--revision <rev>] [--profile <profile>]"
        );
    };
    match subcommand.as_str() {
        "list" => {
            let root = model_home_from_args(&args);
            let store = ModelStore::new(root);
            let aliases = store.list_aliases().await?;
            let mut aliases_by_path: HashMap<std::path::PathBuf, Vec<String>> = HashMap::new();
            for alias in aliases {
                aliases_by_path
                    .entry(alias.snapshot_path)
                    .or_default()
                    .push(alias.alias);
            }
            let inventory = store.snapshot_inventory().await?;
            let snapshots = inventory
                .ready_snapshots
                .into_iter()
                .map(|snapshot| {
                    let aliases = aliases_by_path.remove(&snapshot.path).unwrap_or_default();
                    promoted_snapshot_json(snapshot, "ready", None, aliases)
                })
                .collect::<Vec<_>>();
            let metadata_only = inventory
                .metadata_only_snapshots
                .into_iter()
                .map(|record| {
                    let aliases = aliases_by_path
                        .remove(&record.snapshot.path)
                        .unwrap_or_default();
                    snapshot_record_json(record, aliases)
                })
                .collect::<Vec<_>>();
            let quarantined = inventory
                .quarantined_snapshots
                .into_iter()
                .map(quarantined_snapshot_json)
                .collect::<Vec<_>>();
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "snapshots": snapshots,
                    "metadata_only_snapshots": metadata_only,
                    "quarantined_snapshots": quarantined,
                }))?
            );
        }
        "inspect" => {
            let snapshot_path = args.get(1).ok_or_else(|| {
                anyhow::anyhow!("usage: llm-engine model inspect <snapshot-path>")
            })?;
            if let Ok(quarantine) = ModelStore::inspect_quarantined_snapshot(snapshot_path).await {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&quarantined_snapshot_json(quarantine))?
                );
                return Ok(());
            }
            match ModelStore::inspect_snapshot_readiness(snapshot_path).await {
                Ok(record) => {
                    let snapshot = record.snapshot;
                    let total_bytes = snapshot
                        .manifest
                        .files
                        .iter()
                        .map(|file| file.size)
                        .sum::<u64>();
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&serde_json::json!({
                            "status": record.readiness.status(),
                            "readiness_reason": record.readiness.reason(),
                            "snapshot_path": snapshot.path,
                            "repo_id": snapshot.manifest.repo_id,
                            "requested_revision": snapshot.manifest.requested_revision,
                            "resolved_commit": snapshot.manifest.resolved_commit,
                            "profile": snapshot.manifest.profile,
                            "family": snapshot.manifest.family,
                            "loader": snapshot.manifest.loader,
                            "quantization": snapshot.manifest.quantization,
                            "manifest_digest": snapshot.manifest_digest,
                            "files": snapshot.manifest.files.len(),
                            "total_bytes": total_bytes,
                        }))?
                    );
                }
                Err(snapshot_err) => return Err(snapshot_err.into()),
            }
        }
        "verify" => {
            let snapshot_path = args
                .get(1)
                .ok_or_else(|| anyhow::anyhow!("usage: llm-engine model verify <snapshot-path>"))?;
            let verification = ModelStore::verify_runnable_snapshot(snapshot_path).await?;
            ModelStore::mark_snapshot_used(snapshot_path).await?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "status": "ok",
                    "snapshot_path": verification.snapshot.path,
                    "repo_id": verification.snapshot.manifest.repo_id,
                    "resolved_commit": verification.snapshot.manifest.resolved_commit,
                    "manifest_digest": verification.snapshot.manifest_digest,
                    "verified_files": verification.verified_files,
                    "verified_bytes": verification.verified_bytes,
                }))?
            );
        }
        "prune" => {
            let dry_run = has_flag(&args, "--dry-run");
            let confirmed = has_flag(&args, "--confirm-delete");
            match (dry_run, confirmed) {
                (true, false) | (false, true) => {}
                (false, false) => {
                    anyhow::bail!("llm-engine model prune requires --dry-run or --confirm-delete")
                }
                (true, true) => anyhow::bail!(
                    "llm-engine model prune accepts only one of --dry-run or --confirm-delete"
                ),
            }
            let root = model_home_from_args(&args);
            let store = ModelStore::new(root);
            let policy = prune_policy_from_args(&args)?;
            let plan = store.prune_plan(policy).await?;
            let report = if confirmed {
                Some(store.apply_prune_plan(&plan).await?)
            } else {
                None
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&prune_output_json(dry_run, &plan, report.as_ref()))?
            );
        }
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
                    hidden =
                        qwen_linear_decoder_layer_first_token(&store, &spec, layer_idx, &hidden)
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
                    hidden =
                        qwen_decoder_layer_first_token(&store, &spec, layer_idx, &hidden).await?;
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
                let router = qwen_layer_moe_router_with_matvec(
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
                qwen_layer_moe_forward_with_matvec_in_place(
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
        "plan" | "pull" => {
            let repo = args
                .get(1)
                .ok_or_else(|| anyhow::anyhow!("usage: llm-engine model {subcommand} <repo>"))?;
            let revision = flag_value(&args, "--revision").unwrap_or("main");
            let profile_name = flag_value(&args, "--profile").unwrap_or("qwen36-safetensors-bf16");
            let metadata_only = args.iter().any(|arg| arg == "--metadata-only");
            let profile = ModelProfile::builtin(profile_name)
                .ok_or_else(|| anyhow::anyhow!("unknown model profile `{profile_name}`"))?;
            let repo_id = HubRepoId::model(repo)?;
            let token = std::env::var("HF_TOKEN").ok();
            let client = HubClient::default();
            let mut plan = client
                .plan_model(repo_id, revision, profile, token.as_deref())
                .await?;
            if metadata_only {
                plan = plan.metadata_only();
            }
            if subcommand == "plan" {
                println!("{}", serde_json::to_string_pretty(&plan)?);
            } else {
                let root = model_home_from_args(&args);
                let store = ModelStore::new(root);
                let snapshot = store.pull_plan(&client, &plan, token.as_deref()).await?;
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

fn prune_policy_from_args(args: &[String]) -> anyhow::Result<PrunePolicy> {
    let mut policy = PrunePolicy::default();
    if let Some(days) = flag_value(args, "--older-than-days") {
        let days = days.parse::<u64>()?;
        let seconds = days
            .checked_mul(24 * 60 * 60)
            .ok_or_else(|| anyhow::anyhow!("--older-than-days is too large"))?;
        policy.keep_recent = Some(std::time::Duration::from_secs(seconds));
    }
    if let Some(count) = flag_value(args, "--keep-min-per-profile") {
        policy.keep_min_per_profile = count.parse::<usize>()?;
    }
    if let Some(profile) = flag_value(args, "--profile") {
        policy.profile = Some(profile.to_owned());
    }
    if let Some(now) = flag_value(args, "--now") {
        policy.now = chrono::DateTime::parse_from_rfc3339(now)?.with_timezone(&chrono::Utc);
    }
    Ok(policy)
}

fn prune_output_json(dry_run: bool, plan: &PrunePlan, report: Option<&PruneReport>) -> Value {
    let candidates = plan
        .candidates
        .iter()
        .map(prune_candidate_json)
        .collect::<Vec<_>>();
    let protected = plan
        .protected
        .iter()
        .map(protected_snapshot_json)
        .collect::<Vec<_>>();
    let deleted = report
        .map(|report| {
            report
                .deleted
                .iter()
                .map(deleted_snapshot_json)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let quarantined = report
        .map(|report| {
            report
                .quarantined
                .iter()
                .cloned()
                .map(quarantined_snapshot_json)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let deleted_bytes = report.map_or(0, |report| report.deleted_bytes);
    serde_json::json!({
        "dry_run": dry_run,
        "confirmed": report.is_some(),
        "snapshots": plan.scanned_snapshots,
        "total_bytes": plan.total_bytes,
        "reclaimable_bytes": plan.reclaimable_bytes,
        "deleted_bytes": deleted_bytes,
        "candidates": candidates,
        "protected": protected,
        "deleted": deleted,
        "quarantined": quarantined,
    })
}

fn prune_candidate_json(candidate: &PruneCandidate) -> Value {
    serde_json::json!({
        "path": path_string(&candidate.path),
        "repo_id": &candidate.repo_id,
        "resolved_commit": &candidate.resolved_commit,
        "profile": &candidate.profile,
        "manifest_digest": &candidate.manifest_digest,
        "bytes": candidate.bytes,
        "last_used_at": candidate.last_used_at,
        "aliases": &candidate.aliases,
        "would_delete": true,
    })
}

fn protected_snapshot_json(snapshot: &ProtectedSnapshot) -> Value {
    serde_json::json!({
        "path": path_string(&snapshot.path),
        "repo_id": &snapshot.repo_id,
        "resolved_commit": &snapshot.resolved_commit,
        "profile": &snapshot.profile,
        "manifest_digest": &snapshot.manifest_digest,
        "bytes": snapshot.bytes,
        "last_used_at": snapshot.last_used_at,
        "aliases": &snapshot.aliases,
        "reasons": &snapshot.reasons,
        "would_delete": false,
    })
}

fn deleted_snapshot_json(snapshot: &DeletedSnapshot) -> Value {
    serde_json::json!({
        "path": path_string(&snapshot.path),
        "bytes": snapshot.bytes,
    })
}

fn snapshot_record_json(record: SnapshotRecord, aliases: Vec<String>) -> Value {
    let reason = record.readiness.reason().map(str::to_owned);
    let status = record.readiness.status();
    promoted_snapshot_json(record.snapshot, status, reason, aliases)
}

fn promoted_snapshot_json(
    snapshot: PromotedSnapshot,
    status: &str,
    reason: Option<String>,
    aliases: Vec<String>,
) -> Value {
    serde_json::json!({
        "status": status,
        "path": path_string(&snapshot.path),
        "repo_id": snapshot.manifest.repo_id,
        "requested_revision": snapshot.manifest.requested_revision,
        "resolved_commit": snapshot.manifest.resolved_commit,
        "profile": snapshot.manifest.profile,
        "family": snapshot.manifest.family,
        "loader": snapshot.manifest.loader,
        "quantization": snapshot.manifest.quantization,
        "manifest_digest": snapshot.manifest_digest,
        "files": snapshot.manifest.files.len(),
        "readiness_reason": reason,
        "aliases": aliases,
    })
}

fn quarantined_snapshot_json(snapshot: QuarantinedSnapshot) -> Value {
    serde_json::json!({
        "status": "quarantined",
        "path": path_string(&snapshot.path),
        "original_path": path_string(&snapshot.metadata.original_path),
        "reason": snapshot.metadata.reason,
        "quarantined_at": snapshot.metadata.quarantined_at,
        "manifest_digest": snapshot.metadata.manifest_digest,
        "bytes": snapshot.bytes,
    })
}

fn path_string(path: &Path) -> String {
    path.display().to_string()
}

#[derive(Debug, Clone, Copy)]
struct QwenLmHeadJsonOptions {
    hidden_size: usize,
    rms_norm_eps: f32,
    top_k: usize,
    chunk_rows: usize,
    limit: usize,
}

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
    )
    .await?;
    let top_logits =
        qwen_lm_head_top_k(store, &final_norm, options.top_k, options.chunk_rows).await?;
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
