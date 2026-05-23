use llm_backend::native::{
    CpuNativeMatvecBackend, InferenceScratchpad, QwenMoeDims, SafeTensorFile, SafeTensorShardStore,
    qwen_decoder_layer_first_token, qwen_embedding_and_layer0_norm, qwen_final_norm,
    qwen_layer_moe_forward_in_place, qwen_layer_moe_router,
    qwen_layer0_linear_attention_first_token, qwen_layer0_linear_attention_projections,
    qwen_layer0_post_attention_norm, qwen_linear_decoder_layer_first_token, qwen_lm_head_top_k,
};
use llm_models::QwenModelSpec;
use llm_tokenizer::HuggingFaceTokenizer;
use std::path::Path;

pub async fn run(subcommand: &str, args: &[String]) -> anyhow::Result<()> {
    match subcommand {
        "inspect-safetensors" => {
            let path = args.get(1).ok_or_else(|| {
                anyhow::anyhow!("usage: llm-engine model inspect-safetensors <path>")
            })?;
            let tensor_file = SafeTensorFile::open(path)?;
            let header = tensor_file.header();
            let sample_tensors: Vec<_> = header.tensor_names().take(8).collect();
            let tensor_name = flag_value(args, "--tensor");
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
            let bf16_row = match (tensor_name, flag_value(args, "--bf16-row")) {
                (Some(name), Some(row)) => {
                    let row = row.parse::<usize>()?;
                    let values = tensor_file.bf16_row_f32(name, row)?;
                    let limit = flag_value(args, "--limit")
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
            let tensor_name = flag_value(args, "--tensor")
                .ok_or_else(|| anyhow::anyhow!("inspect-tensor requires --tensor <name>"))?;
            let store = SafeTensorShardStore::open(snapshot_path)?;
            let shard_path = store.tensor_shard_path(tensor_name)?;
            let metadata = store.tensor_metadata(tensor_name)?;
            let bf16_row = flag_value(args, "--bf16-row")
                .map(|row| {
                    let row = row.parse::<usize>()?;
                    let values = store.bf16_row_f32(tensor_name, row)?;
                    let limit = flag_value(args, "--limit")
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
            let token_id = flag_value(args, "--token-id")
                .ok_or_else(|| anyhow::anyhow!("inspect-qwen-input requires --token-id <id>"))?
                .parse::<usize>()?;
            let limit = flag_value(args, "--limit")
                .map(str::parse::<usize>)
                .transpose()?
                .unwrap_or(8);
            let config_json =
                tokio::fs::read_to_string(Path::new(snapshot_path).join("config.json")).await?;
            let spec = QwenModelSpec::from_config_json(&config_json)?;
            let store = SafeTensorShardStore::open(snapshot_path)?;
            let lm_head_top_k = flag_value(args, "--lm-head-top-k")
                .map(str::parse::<usize>)
                .transpose()?;
            let chunk_rows = flag_value(args, "--chunk-rows")
                .map(str::parse::<usize>)
                .transpose()?
                .unwrap_or(512);
            let tokenizer = lm_head_top_k
                .map(|_| {
                    HuggingFaceTokenizer::from_file(Path::new(snapshot_path).join("tokenizer.json"))
                })
                .transpose()?;
            let probe = qwen_embedding_and_layer0_norm(
                &store,
                token_id,
                spec.hidden_size as usize,
                spec.rms_norm_eps,
            )?;
            let linear_layers = if let Some(count) = flag_value(args, "--linear-layers") {
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
            let layers = if let Some(count) = flag_value(args, "--layers") {
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
                let top_k = flag_value(args, "--top-k")
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
        other => anyhow::bail!("unknown native probe subcommand `{other}`"),
    }
    Ok(())
}

fn flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find_map(|window| (window[0] == flag).then_some(window[1].as_str()))
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
