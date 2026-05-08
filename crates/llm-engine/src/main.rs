use llm_backend::SafeTensorHeader;
use llm_engine::build_router;
use llm_hub::{HubClient, HubRepoId, ModelProfile, ModelStore};
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
            let header = SafeTensorHeader::from_file(path)?;
            let sample_tensors: Vec<_> = header.tensor_names().take(8).collect();
            let tensor = flag_value(&args, "--tensor")
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
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "path": path,
                    "file_len": header.file_len(),
                    "header_len": header.header_len(),
                    "data_start": header.data_start(),
                    "tensor_count": header.tensor_count(),
                    "sample_tensors": sample_tensors,
                    "tensor": tensor
                }))?
            );
        }
        "plan" | "pull" => {
            let repo = args
                .get(1)
                .ok_or_else(|| anyhow::anyhow!("usage: llm-engine model {subcommand} <repo>"))?;
            let revision = flag_value(&args, "--revision").unwrap_or("main");
            let profile_name = flag_value(&args, "--profile").unwrap_or("qwen36-mlx-4bit");
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
