use llm_backend::{
    QwenMoeDims, SafeTensorFile, SafeTensorShardStore, qwen_embedding_and_layer0_norm,
};
use llm_backend::{
    qwen_decoder_layer_first_token, qwen_final_norm, qwen_layer0_linear_attention_first_token,
    qwen_layer0_linear_attention_projections, qwen_layer0_moe_forward, qwen_layer0_moe_router,
    qwen_layer0_post_attention_norm, qwen_linear_decoder_layer_first_token, qwen_lm_head_top_k,
};
use llm_engine::{
    DEFAULT_NATIVE_QWEN_MAX_NEW_TOKENS, EngineOptions, MlxBackendOptions, NativeQwenLoadOptions,
    SnapshotBackendOptions, build_router_with_backend_and_options, open_snapshot_backend,
};
use llm_hub::{
    DeletedSnapshot, HubClient, HubRepoId, ModelProfile, ModelStore, ProtectedSnapshot,
    PruneCandidate, PrunePlan, PrunePolicy, PruneReport, QuarantinedSnapshot,
};
use llm_models::QwenModelSpec;
use llm_tokenizer::HuggingFaceTokenizer;
use serde_json::Value;
use std::{collections::HashMap, net::SocketAddr, path::Path};

mod bench;

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
            if admin_token.is_none() && !addr.ip().is_loopback() {
                anyhow::bail!(
                    "serving admin endpoints on a non-loopback address requires --admin-token or LLM_ENGINE_ADMIN_TOKEN"
                );
            }
            let options = EngineOptions {
                concurrency_limit: max_concurrent_requests,
                admin_token,
                model_home,
                hub_endpoint,
                hf_token: std::env::var("HF_TOKEN").ok(),
                ..EngineOptions::default()
            };
            let router = if let Some(snapshot_path) = flag_value(&serve_args, "--snapshot") {
                let model_id = flag_value(&serve_args, "--model-id").unwrap_or("local-qwen36");
                let snapshot_path = std::path::PathBuf::from(snapshot_path);
                let max_new_tokens = flag_value(&serve_args, "--max-new-tokens")
                    .map(str::parse::<u32>)
                    .transpose()?
                    .unwrap_or(DEFAULT_NATIVE_QWEN_MAX_NEW_TOKENS);
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
                let backend = open_snapshot_backend(
                    model_id,
                    &snapshot_path,
                    SnapshotBackendOptions {
                        native_qwen: NativeQwenLoadOptions {
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
                        mlx: MlxBackendOptions {
                            endpoint: mlx_endpoint,
                        },
                        max_new_tokens,
                        max_prefill_tokens,
                    },
                )?;
                if let Err(err) = ModelStore::mark_snapshot_used(&snapshot_path).await {
                    tracing::warn!(error = %err, snapshot = %snapshot_path.display(), "failed to record snapshot usage");
                }
                if let Err(err) = ModelStore::new(&model_home_for_records)
                    .record_snapshot_alias(model_id, &snapshot_path)
                    .await
                {
                    tracing::warn!(error = %err, alias = model_id, snapshot = %snapshot_path.display(), "failed to record model alias");
                }
                build_router_with_backend_and_options(backend, options)?
            } else if has_flag(&serve_args, "--deterministic-test-backend") {
                build_router_with_backend_and_options(
                    Box::new(
                        llm_backend::DeterministicBackend::new(
                            "local-qwen36",
                            "hello from rust native backend",
                        )
                        .with_required_tool_protocol()
                        .with_json_object_protocol(),
                    ),
                    options,
                )?
            } else {
                anyhow::bail!(
                    "llm-engine serve requires --snapshot <path> for native Qwen serving; use --deterministic-test-backend only for protocol tests"
                );
            };
            let listener = tokio::net::TcpListener::bind(addr).await?;
            tracing::info!(%addr, "llm-engine listening");
            axum::serve(listener, router).await?;
        }
        "bench" => bench::run_bench_command(std::env::args().skip(2).collect()).await?,
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
  --snapshot <path>                          Native Qwen snapshot path
  --model-id <id>                            Served model id [default: local-qwen36]
  --deterministic-test-backend               Use deterministic protocol backend
  --max-new-tokens <n>                       Native Qwen maximum generated tokens [default: 256]
  --max-prefill-tokens <n>                   Native Qwen maximum prefill tokens
  --max-concurrent-requests <n>              Maximum concurrent requests [default: 1]
  --admin-token <token>                      Bearer token for admin endpoints
  --model-home <path>                        Model store root
  --hub-endpoint <url>                       Hugging Face compatible Hub endpoint
  --mlx-endpoint <url>                       Loopback mlx_lm.server /v1 endpoint [default: http://127.0.0.1:8080/v1]
  --native-metal-weight-cache-bytes <bytes>  Native Metal BF16 weight cache budget
  --warm-native-metal-weight-cache           Warm native Metal BF16 weight cache at startup
  --eager-materialize-shards                 Materialize indexed safetensor shards at startup
  -h, --help                                 Print help"
    );
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
            let snapshots = store.list_snapshots().await?;
            let snapshots = snapshots
                .into_iter()
                .map(|snapshot| {
                    let aliases = aliases_by_path.remove(&snapshot.path).unwrap_or_default();
                    serde_json::json!({
                        "status": "ready",
                        "path": snapshot.path,
                        "repo_id": snapshot.manifest.repo_id,
                        "requested_revision": snapshot.manifest.requested_revision,
                        "resolved_commit": snapshot.manifest.resolved_commit,
                        "profile": snapshot.manifest.profile,
                        "family": snapshot.manifest.family,
                        "loader": snapshot.manifest.loader,
                        "quantization": snapshot.manifest.quantization,
                        "manifest_digest": snapshot.manifest_digest,
                        "files": snapshot.manifest.files.len(),
                        "aliases": aliases,
                    })
                })
                .collect::<Vec<_>>();
            let quarantined = store
                .list_quarantined_snapshots()
                .await?
                .into_iter()
                .map(quarantined_snapshot_json)
                .collect::<Vec<_>>();
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "snapshots": snapshots,
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
            match ModelStore::inspect_snapshot(snapshot_path).await {
                Ok(snapshot) => {
                    let total_bytes = snapshot
                        .manifest
                        .files
                        .iter()
                        .map(|file| file.size)
                        .sum::<u64>();
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&serde_json::json!({
                            "status": "ready",
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
            let verification = ModelStore::verify_snapshot(snapshot_path).await?;
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
                std::fs::read_to_string(std::path::Path::new(snapshot_path).join("config.json"))?;
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
            let linear_layers = flag_value(&args, "--linear-layers")
                .map(|count| {
                    let count = count.parse::<usize>()?;
                    let mut hidden = probe.embedding.clone();
                    let mut layers = Vec::new();
                    for layer_idx in 0..count {
                        hidden =
                            qwen_linear_decoder_layer_first_token(&store, &spec, layer_idx, &hidden)?;
                        layers.push(serde_json::json!({
                            "layer": layer_idx,
                            "hidden_prefix": hidden.iter().copied().take(limit).collect::<Vec<_>>()
                        }));
                    }
                    anyhow::Ok(serde_json::json!({
                        "layers": layers,
                        "final_hidden_prefix": hidden.iter().copied().take(limit).collect::<Vec<_>>()
                    }))
                })
                .transpose()?;
            let layers = flag_value(&args, "--layers")
                .map(|count| {
                    let count = count.parse::<usize>()?;
                    let mut hidden = probe.embedding.clone();
                    let mut layers = Vec::new();
                    for layer_idx in 0..count {
                        hidden = qwen_decoder_layer_first_token(&store, &spec, layer_idx, &hidden)?;
                        layers.push(serde_json::json!({
                            "layer": layer_idx,
                            "kind": format!("{:?}", spec.layer_kinds[layer_idx]),
                            "hidden_prefix": hidden.iter().copied().take(limit).collect::<Vec<_>>()
                        }));
                    }
                    let lm_head = lm_head_top_k
                        .map(|top_k| {
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
                        })
                        .transpose()?;
                    anyhow::Ok(serde_json::json!({
                        "layers": layers,
                        "final_hidden_prefix": hidden.iter().copied().take(limit).collect::<Vec<_>>(),
                        "lm_head": lm_head
                    }))
                })
                .transpose()?;
            let run_layer0_attention = args.iter().any(|arg| arg == "--layer0-attention")
                || args.iter().any(|arg| arg == "--layer0-router")
                || args.iter().any(|arg| arg == "--layer0-moe");
            let run_layer0_projections =
                args.iter().any(|arg| arg == "--layer0-projections") || run_layer0_attention;
            let projections = if run_layer0_projections {
                Some(qwen_layer0_linear_attention_projections(
                    &store,
                    &probe.normalized,
                )?)
            } else {
                None
            };
            let layer0_attention_output = if run_layer0_attention {
                let projections = projections.as_ref().expect("projections are computed");
                Some(qwen_layer0_linear_attention_first_token(
                    &store,
                    &spec,
                    projections,
                )?)
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
                let attention_output = layer0_attention_output
                    .as_ref()
                    .expect("attention output is computed");
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
                )?;
                let top_k = flag_value(&args, "--top-k")
                    .map(str::parse::<usize>)
                    .transpose()?
                    .unwrap_or(spec.num_experts_per_tok as usize);
                let router = qwen_layer0_moe_router(&store, &post_attention, top_k)?;
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
                let post_attention = post_attention_norm
                    .as_ref()
                    .expect("post-attention norm is computed");
                let router = router_probe.as_ref().expect("router is computed");
                let moe_output = qwen_layer0_moe_forward(
                    &store,
                    &QwenMoeDims::from_spec(&spec),
                    post_attention,
                    router,
                )?;
                let final_hidden = attention_residual
                    .as_ref()
                    .expect("attention residual is computed")
                    .iter()
                    .zip(&moe_output)
                    .map(|(residual, moe)| residual + moe)
                    .collect::<Vec<_>>();
                let lm_head = lm_head_top_k
                    .map(|top_k| {
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
                    })
                    .transpose()?;
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

fn qwen_lm_head_json(
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
    )?;
    let top_logits = qwen_lm_head_top_k(store, &final_norm, options.top_k, options.chunk_rows)?;
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
