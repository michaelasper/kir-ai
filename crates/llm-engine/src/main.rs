use llm_backend::{SafeTensorFile, SafeTensorShardStore, qwen_embedding_and_layer0_norm};
use llm_engine::build_router;
use llm_hub::{HubClient, HubRepoId, ModelProfile, ModelStore};
use llm_models::QwenModelSpec;
use std::net::SocketAddr;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let command = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "serve".to_owned());
    match command.as_str() {
        "serve" => {
            let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
            let listener = tokio::net::TcpListener::bind(addr).await?;
            tracing::info!(%addr, "llm-engine listening");
            axum::serve(listener, build_router()).await?;
        }
        "model" => run_model_command(std::env::args().skip(2).collect()).await?,
        other => anyhow::bail!("unknown command `{other}`"),
    }
    Ok(())
}

async fn run_model_command(args: Vec<String>) -> anyhow::Result<()> {
    let Some(subcommand) = args.first() else {
        anyhow::bail!(
            "usage: llm-engine model plan <repo> [--revision <rev>] [--profile <profile>]"
        );
    };
    match subcommand.as_str() {
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
            let probe = qwen_embedding_and_layer0_norm(
                &store,
                token_id,
                spec.hidden_size as usize,
                spec.rms_norm_eps,
            )?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "snapshot_path": snapshot_path,
                    "token_id": token_id,
                    "hidden_size": spec.hidden_size,
                    "rms_norm_eps": spec.rms_norm_eps,
                    "embedding_prefix": probe.embedding.iter().copied().take(limit).collect::<Vec<_>>(),
                    "normalized_prefix": probe.normalized.iter().copied().take(limit).collect::<Vec<_>>(),
                    "values_read": probe.normalized.len()
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
            let profile = match profile_name {
                "qwen36-mlx-4bit" => ModelProfile::qwen36_mlx_4bit(),
                "qwen36-safetensors-bf16" => ModelProfile::qwen36_safetensors_bf16(),
                other => anyhow::bail!("unknown model profile `{other}`"),
            };
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
                let root = flag_value(&args, "--model-home")
                    .map(std::path::PathBuf::from)
                    .or_else(|| std::env::var_os("LLM_MODEL_HOME").map(std::path::PathBuf::from))
                    .unwrap_or_else(|| std::path::PathBuf::from(".llm-models"));
                let snapshot = ModelStore::new(root)
                    .pull_plan(&client, &plan, token.as_deref())
                    .await?;
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
