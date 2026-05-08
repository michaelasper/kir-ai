use llm_backend::{
    CpuQwenMatvecBackend, MathError, QWEN_FINAL_NORM_WEIGHT, QwenLayerCache, QwenMatvecBackend,
    SafeTensorArchive, SafeTensorFile, SafeTensorHeader, SafeTensorShardStore, TensorLoadError,
    TopKLogit, qwen_decode_token_with_cache, qwen_decode_token_with_cache_with_matvec,
    qwen_embedding_and_layer0_norm, qwen_final_norm, qwen_final_norm_with_matvec,
    qwen_layer_caches_for_spec, qwen_layer_full_attention_first_token,
    qwen_layer_full_attention_sequence, qwen_layer_full_attention_sequence_with_cache,
    qwen_layer_full_attention_sequence_with_cache_with_matvec,
    qwen_layer_full_attention_step_with_cache,
    qwen_layer_full_attention_step_with_cache_with_matvec, qwen_layer_linear_attention_first_token,
    qwen_layer_linear_attention_projections, qwen_layer_linear_attention_sequence,
    qwen_layer_linear_attention_sequence_with_cache,
    qwen_layer_linear_attention_sequence_with_cache_with_matvec,
    qwen_layer_linear_attention_step_with_cache, qwen_layer_moe_forward_with_matvec,
    qwen_layer_moe_router_with_matvec, qwen_layer0_linear_attention_projections,
    qwen_layer0_moe_forward, qwen_layer0_moe_router, qwen_layer0_post_attention_norm,
    qwen_lm_head_logits, qwen_lm_head_logits_with_matvec, qwen_lm_head_top_k,
    qwen_lm_head_top_k_with_matvec, qwen_prefill_sequence, qwen_prefill_sequence_with_cache,
    qwen_prefill_sequence_with_cache_with_matvec, qwen_rms_norm_f32,
};
use llm_backend::{QwenMoeDims, QwenMoeRouterProbe, TopKWeight};
use llm_kv_cache::{LayerKvCache, LinearAttentionCache};
use llm_models::{AttentionKind, ModelFamily, QwenModelSpec};
use std::cell::Cell;

#[test]
fn reads_safetensors_metadata_and_f32_tensor() {
    let bytes = tiny_safetensors_f32("linear.weight", &[2, 2], &[1.0, 2.0, 3.0, 4.0]);

    let archive = SafeTensorArchive::from_bytes(&bytes).expect("archive loads");
    let metadata = archive.tensor_metadata("linear.weight").expect("metadata");
    assert_eq!(metadata.dtype, "F32");
    assert_eq!(metadata.shape, vec![2, 2]);
    assert_eq!(metadata.byte_len, 16);

    let values = archive
        .f32_tensor("linear.weight")
        .expect("f32 tensor decodes");
    assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn rejects_wrong_dtype_for_f32_reader() {
    let bytes = tiny_safetensors("linear.weight", "BF16", &[1], &[0_u8, 0_u8]);

    let archive = SafeTensorArchive::from_bytes(&bytes).expect("archive loads");
    let err = archive
        .f32_tensor("linear.weight")
        .expect_err("wrong dtype fails");
    assert_eq!(err.code(), "unsupported_capability");
}

#[test]
fn reads_bf16_header_metadata_without_decoding_payload() {
    let bytes = tiny_safetensors(
        "model.layers.0.mlp.gate.weight",
        "BF16",
        &[2, 4],
        &[0_u8; 16],
    );

    let header = SafeTensorHeader::from_bytes(&bytes).expect("header loads");
    let metadata = header
        .tensor_metadata("model.layers.0.mlp.gate.weight")
        .expect("metadata");

    assert_eq!(header.tensor_count(), 1);
    assert_eq!(metadata.dtype, "BF16");
    assert_eq!(metadata.shape, vec![2, 4]);
    assert_eq!(metadata.byte_len, 16);
    assert_eq!(
        header
            .tensor_data_range("model.layers.0.mlp.gate.weight")
            .expect("range"),
        header.data_start()..header.data_start() + 16
    );
}

#[test]
fn reads_header_from_file_with_large_payload() {
    let mut payload = vec![0_u8; 1024 * 1024];
    payload[0] = 7;
    let last = payload.len() - 1;
    payload[last] = 9;
    let bytes = tiny_safetensors("large.weight", "BF16", &[512, 1024], &payload);
    let path = std::env::temp_dir().join(format!(
        "llm-backend-safetensors-header-{}.safetensors",
        std::process::id()
    ));
    std::fs::write(&path, bytes).expect("write fixture");

    let header = SafeTensorHeader::from_file(&path).expect("header loads from file");
    let metadata = header.tensor_metadata("large.weight").expect("metadata");

    assert_eq!(metadata.dtype, "BF16");
    assert_eq!(metadata.shape, vec![512, 1024]);
    assert_eq!(metadata.byte_len, 1024 * 1024);
    std::fs::remove_file(path).ok();
}

#[test]
fn rejects_header_offsets_outside_payload() {
    let mut bytes = tiny_safetensors("broken.weight", "BF16", &[8], &[0_u8; 16]);
    let header_len = u64::from_le_bytes(bytes[0..8].try_into().expect("header prefix")) as usize;
    let header = serde_json::json!({
        "broken.weight": {
            "dtype": "BF16",
            "shape": [8],
            "data_offsets": [0, 32]
        }
    })
    .to_string();
    bytes.splice(0..8 + header_len, {
        let mut replacement = Vec::new();
        replacement.extend_from_slice(&(header.len() as u64).to_le_bytes());
        replacement.extend_from_slice(header.as_bytes());
        replacement
    });

    let err = SafeTensorHeader::from_bytes(&bytes).expect_err("offsets fail");
    assert_eq!(err.code(), "model_integrity_failed");
}

#[test]
fn reads_bf16_ranges_from_file() {
    let bytes = tiny_safetensors_bf16("embed.weight", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let path = temp_safetensors_path("bf16-ranges");
    std::fs::write(&path, bytes).expect("write fixture");

    let file = SafeTensorFile::open(&path).expect("open tensor file");

    assert_eq!(
        file.bf16_tensor_f32_range("embed.weight", 2, 3)
            .expect("range"),
        vec![3.0, 4.0, 5.0]
    );
    assert_eq!(
        file.bf16_tensor_bits_range("embed.weight", 2, 3)
            .expect("raw bf16 range"),
        vec![bf16_bits(3.0), bf16_bits(4.0), bf16_bits(5.0)]
    );
    assert_eq!(
        file.bf16_row_f32("embed.weight", 1).expect("row"),
        vec![4.0, 5.0, 6.0]
    );
    std::fs::remove_file(path).ok();
}

#[test]
fn rejects_bf16_range_outside_tensor() {
    let bytes = tiny_safetensors_bf16("embed.weight", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let path = temp_safetensors_path("bf16-oob");
    std::fs::write(&path, bytes).expect("write fixture");
    let file = SafeTensorFile::open(&path).expect("open tensor file");

    let err = file
        .bf16_tensor_f32_range("embed.weight", 5, 2)
        .expect_err("range fails");

    assert_eq!(err.code(), "model_integrity_failed");
    std::fs::remove_file(path).ok();
}

#[test]
fn shard_store_reads_bf16_row_by_tensor_name() {
    let root = temp_snapshot_dir("indexed-store");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 12 },
            "weight_map": { "embed.weight": "model-00001-of-00001.safetensors" }
        })
        .to_string(),
    )
    .expect("index");
    std::fs::write(
        root.join("model-00001-of-00001.safetensors"),
        tiny_safetensors_bf16("embed.weight", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )
    .expect("shard");

    let store = SafeTensorShardStore::open(&root).expect("store opens");
    assert_eq!(store.cached_shard_count(), 0);

    assert_eq!(
        store.bf16_row_f32("embed.weight", 1).expect("row"),
        vec![4.0, 5.0, 6.0]
    );
    assert_eq!(
        store
            .bf16_tensor_bits_range("embed.weight", 3, 2)
            .expect("raw bf16 range"),
        vec![bf16_bits(4.0), bf16_bits(5.0)]
    );
    assert_eq!(store.cached_shard_count(), 1);
    assert_eq!(
        store
            .bf16_matvec_row_major_f32("embed.weight", &[1.0, 2.0, 3.0])
            .expect("matvec"),
        vec![14.0, 32.0]
    );
    assert_eq!(
        store
            .bf16_matvecs_row_major_f32("embed.weight", &[vec![1.0, 2.0, 3.0], vec![3.0, 2.0, 1.0]])
            .expect("batched matvec"),
        vec![vec![14.0, 32.0], vec![10.0, 28.0]]
    );
    let top = store
        .bf16_matvec_top_k_rows_f32("embed.weight", &[1.0, 2.0, 3.0], 1, 1)
        .expect("top logits");
    assert_eq!(top[0].index, 1);
    assert_eq!(top[0].logit, 32.0);
    assert_eq!(
        store
            .tensor_shard_path("embed.weight")
            .expect("shard path")
            .file_name()
            .and_then(|name| name.to_str()),
        Some("model-00001-of-00001.safetensors")
    );
    assert_eq!(store.cached_shard_count(), 1);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn shard_store_materializes_shard_once_and_reuses_it_for_reads() {
    let root = temp_snapshot_dir("materialized-store");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 12 },
            "weight_map": { "embed.weight": "model-00001-of-00001.safetensors" }
        })
        .to_string(),
    )
    .expect("index");
    let shard_bytes =
        tiny_safetensors_bf16("embed.weight", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let shard_len = shard_bytes.len();
    std::fs::write(root.join("model-00001-of-00001.safetensors"), shard_bytes).expect("shard");

    let store = SafeTensorShardStore::open(&root).expect("store opens");
    assert_eq!(store.materialized_shard_count(), 0);

    assert_eq!(
        store
            .materialize_shard_for_tensor("embed.weight")
            .expect("materialized shard"),
        shard_len
    );
    assert_eq!(store.materialized_shard_count(), 1);
    assert_eq!(
        store
            .materialize_shard_for_tensor("embed.weight")
            .expect("reused materialized shard"),
        shard_len
    );
    assert_eq!(store.materialized_shard_count(), 1);
    assert_eq!(
        store
            .bf16_tensor_f32_range("embed.weight", 1, 4)
            .expect("range reads from materialized shard"),
        vec![2.0, 3.0, 4.0, 5.0]
    );
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn shard_store_materializes_all_indexed_shards_once() {
    let root = temp_snapshot_dir("materialized-all-store");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 16 },
            "weight_map": {
                "embed.weight": "embed.safetensors",
                "norm.weight": "norm.safetensors"
            }
        })
        .to_string(),
    )
    .expect("index");
    let embed = tiny_safetensors_bf16("embed.weight", &[2], &[1.0, 2.0]);
    let norm = tiny_safetensors_bf16("norm.weight", &[2], &[3.0, 4.0]);
    let expected_bytes = embed.len() + norm.len();
    std::fs::write(root.join("embed.safetensors"), embed).expect("embed shard");
    std::fs::write(root.join("norm.safetensors"), norm).expect("norm shard");

    let store = SafeTensorShardStore::open(&root).expect("store opens");
    assert_eq!(store.cached_shard_count(), 0);
    assert_eq!(store.materialized_shard_count(), 0);

    assert_eq!(
        store
            .materialize_all_shards()
            .expect("all shards materialize"),
        expected_bytes
    );
    assert_eq!(store.cached_shard_count(), 2);
    assert_eq!(store.materialized_shard_count(), 2);
    assert_eq!(
        store
            .materialize_all_shards()
            .expect("materialized shards are reused"),
        expected_bytes
    );
    assert_eq!(store.materialized_shard_count(), 2);
    assert_eq!(
        store.bf16_tensor_f32("norm.weight").expect("read norm"),
        vec![3.0, 4.0]
    );
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn shard_store_rejects_unsafe_index_shard_paths_on_open() {
    for shard_path in [
        "../outside.safetensors",
        "/tmp/outside.safetensors",
        "nested\\outside.safetensors",
    ] {
        let root = temp_snapshot_dir(&format!("unsafe-index-{}", shard_path.len()));
        std::fs::remove_dir_all(&root).ok();
        std::fs::create_dir_all(&root).expect("snapshot dir");
        std::fs::write(
            root.join("model.safetensors.index.json"),
            serde_json::json!({
                "metadata": { "total_size": 2 },
                "weight_map": { "embed.weight": shard_path }
            })
            .to_string(),
        )
        .expect("index");

        let err = SafeTensorShardStore::open(&root).expect_err("unsafe index fails closed");

        assert_eq!(err.code(), "model_integrity_failed");
        std::fs::remove_dir_all(root).ok();
    }
}

#[cfg(unix)]
#[test]
fn shard_store_rejects_symlink_that_escapes_snapshot_root() {
    let root = temp_snapshot_dir("symlink-escape");
    let outside = temp_safetensors_path("symlink-outside");
    std::fs::remove_dir_all(&root).ok();
    std::fs::remove_file(&outside).ok();
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 2 },
            "weight_map": { "embed.weight": "linked.safetensors" }
        })
        .to_string(),
    )
    .expect("index");
    std::fs::write(
        &outside,
        tiny_safetensors_bf16("embed.weight", &[1], &[1.0]),
    )
    .expect("outside shard");
    std::os::unix::fs::symlink(&outside, root.join("linked.safetensors")).expect("escape symlink");

    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let err = store
        .bf16_tensor_f32("embed.weight")
        .expect_err("escaped symlink fails closed");

    assert_eq!(err.code(), "model_integrity_failed");
    std::fs::remove_dir_all(root).ok();
    std::fs::remove_file(outside).ok();
}

#[test]
fn qwen_embedding_probe_reads_and_normalizes_token() {
    let root = temp_snapshot_dir("qwen-embed");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 20 },
            "weight_map": {
                "model.language_model.embed_tokens.weight": "embed.safetensors",
                "model.language_model.layers.0.input_layernorm.weight": "norm.safetensors"
            }
        })
        .to_string(),
    )
    .expect("index");
    std::fs::write(
        root.join("embed.safetensors"),
        tiny_safetensors_bf16(
            "model.language_model.embed_tokens.weight",
            &[2, 2],
            &[3.0, 4.0, 6.0, 8.0],
        ),
    )
    .expect("embedding shard");
    std::fs::write(
        root.join("norm.safetensors"),
        tiny_safetensors_bf16(
            "model.language_model.layers.0.input_layernorm.weight",
            &[2],
            &[0.0, 1.0],
        ),
    )
    .expect("norm shard");
    let store = SafeTensorShardStore::open(&root).expect("store opens");

    let probe = qwen_embedding_and_layer0_norm(&store, 1, 2, 0.0).expect("probe");

    assert_eq!(probe.embedding, vec![6.0, 8.0]);
    assert_close(&probe.normalized, &[0.84852815, 2.2627418], 1e-6);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_layer0_projection_probe_reads_bf16_matrices() {
    let root = temp_snapshot_dir("qwen-projections");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 80 },
            "weight_map": {
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": "qkv.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_z.weight": "z.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_b.weight": "b.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_a.weight": "a.safetensors"
            }
        })
        .to_string(),
    )
    .expect("index");
    std::fs::write(
        root.join("qkv.safetensors"),
        tiny_safetensors_bf16(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            &[2, 2],
            &[1.0, 0.0, 0.0, 1.0],
        ),
    )
    .expect("qkv");
    std::fs::write(
        root.join("z.safetensors"),
        tiny_safetensors_bf16(
            "model.language_model.layers.0.linear_attn.in_proj_z.weight",
            &[1, 2],
            &[1.0, 1.0],
        ),
    )
    .expect("z");
    std::fs::write(
        root.join("b.safetensors"),
        tiny_safetensors_bf16(
            "model.language_model.layers.0.linear_attn.in_proj_b.weight",
            &[1, 2],
            &[2.0, 0.0],
        ),
    )
    .expect("b");
    std::fs::write(
        root.join("a.safetensors"),
        tiny_safetensors_bf16(
            "model.language_model.layers.0.linear_attn.in_proj_a.weight",
            &[1, 2],
            &[0.0, 3.0],
        ),
    )
    .expect("a");
    let store = SafeTensorShardStore::open(&root).expect("store opens");

    let projections =
        qwen_layer0_linear_attention_projections(&store, &[4.0, 5.0]).expect("projections");

    assert_eq!(projections.qkv, vec![4.0, 5.0]);
    assert_eq!(projections.z, vec![9.0]);
    assert_eq!(projections.b, vec![8.0]);
    assert_eq!(projections.a, vec![15.0]);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_moe_router_probe_selects_top_experts() {
    let root = temp_snapshot_dir("qwen-router");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 24 },
            "weight_map": {
                "model.language_model.layers.0.mlp.gate.weight": "router.safetensors"
            }
        })
        .to_string(),
    )
    .expect("index");
    std::fs::write(
        root.join("router.safetensors"),
        tiny_safetensors_bf16(
            "model.language_model.layers.0.mlp.gate.weight",
            &[3, 2],
            &[
                1.0, 0.0, //
                0.0, 2.0, //
                1.0, 1.0,
            ],
        ),
    )
    .expect("router");
    let store = SafeTensorShardStore::open(&root).expect("store opens");

    let router = qwen_layer0_moe_router(&store, &[2.0, 3.0], 2).expect("router");

    assert_eq!(router.selected[0].index, 1);
    assert_eq!(router.selected[1].index, 2);
    assert_close(
        &[router.selected[0].weight, router.selected[1].weight],
        &[0.7310586, 0.26894143],
        1e-6,
    );
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_moe_router_uses_configured_top_k_backend() {
    let root = temp_snapshot_dir("qwen-router-custom-top-k");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 24 },
            "weight_map": {
                "model.language_model.layers.0.mlp.gate.weight": "router.safetensors"
            }
        })
        .to_string(),
    )
    .expect("index");
    std::fs::write(
        root.join("router.safetensors"),
        tiny_safetensors_bf16(
            "model.language_model.layers.0.mlp.gate.weight",
            &[3, 2],
            &[
                1.0, 0.0, //
                0.0, 2.0, //
                1.0, 1.0,
            ],
        ),
    )
    .expect("router");
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let matvec = RecordingMatvecBackend::default();

    let router =
        qwen_layer_moe_router_with_matvec(&store, 0, &[2.0, 3.0], 2, &matvec).expect("router");

    assert_eq!(router.selected[0].index, 1);
    assert_eq!(router.selected[1].index, 2);
    assert_eq!(matvec.softmax_top_k_calls.get(), 1);
    assert_close(
        &[router.selected[0].weight, router.selected[1].weight],
        &[0.7310586, 0.26894143],
        1e-6,
    );
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_post_attention_norm_adds_residual_and_normalizes() {
    let root = temp_snapshot_dir("qwen-post-attn-norm");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 4 },
            "weight_map": {
                "model.language_model.layers.0.post_attention_layernorm.weight": "post_norm.safetensors"
            }
        })
        .to_string(),
    )
    .expect("index");
    std::fs::write(
        root.join("post_norm.safetensors"),
        tiny_safetensors_bf16(
            "model.language_model.layers.0.post_attention_layernorm.weight",
            &[2],
            &[0.0, 1.0],
        ),
    )
    .expect("post norm");
    let store = SafeTensorShardStore::open(&root).expect("store opens");

    let normalized = qwen_layer0_post_attention_norm(&store, &[3.0, 4.0], &[3.0, 4.0], 2, 0.0)
        .expect("post attention norm");

    assert_close(&normalized, &[0.84852815, 2.2627418], 1e-6);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_moe_forward_reads_selected_expert_slices() {
    let root = temp_snapshot_dir("qwen-moe-forward");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 80 },
            "weight_map": {
                "model.language_model.layers.0.mlp.experts.gate_up_proj": "gate_up.safetensors",
                "model.language_model.layers.0.mlp.experts.down_proj": "down.safetensors",
                "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight": "shared_gate.safetensors",
                "model.language_model.layers.0.mlp.shared_expert.up_proj.weight": "shared_up.safetensors",
                "model.language_model.layers.0.mlp.shared_expert.down_proj.weight": "shared_down.safetensors",
                "model.language_model.layers.0.mlp.shared_expert_gate.weight": "shared_expert_gate.safetensors"
            }
        })
        .to_string(),
    )
    .expect("index");
    std::fs::write(
        root.join("gate_up.safetensors"),
        tiny_safetensors_bf16(
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            &[2, 2, 2],
            &[
                1.0, 0.0, 0.0, 1.0, //
                0.0, 1.0, 1.0, 0.0,
            ],
        ),
    )
    .expect("gate up");
    std::fs::write(
        root.join("down.safetensors"),
        tiny_safetensors_bf16(
            "model.language_model.layers.0.mlp.experts.down_proj",
            &[2, 2, 1],
            &[
                1.0, 2.0, //
                3.0, 4.0,
            ],
        ),
    )
    .expect("down");
    std::fs::write(
        root.join("shared_gate.safetensors"),
        tiny_safetensors_bf16(
            "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
            &[1, 2],
            &[0.0, 0.0],
        ),
    )
    .expect("shared gate");
    std::fs::write(
        root.join("shared_up.safetensors"),
        tiny_safetensors_bf16(
            "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
            &[1, 2],
            &[0.0, 0.0],
        ),
    )
    .expect("shared up");
    std::fs::write(
        root.join("shared_down.safetensors"),
        tiny_safetensors_bf16(
            "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
            &[2, 1],
            &[0.0, 0.0],
        ),
    )
    .expect("shared down");
    std::fs::write(
        root.join("shared_expert_gate.safetensors"),
        tiny_safetensors_bf16(
            "model.language_model.layers.0.mlp.shared_expert_gate.weight",
            &[1, 2],
            &[0.0, 0.0],
        ),
    )
    .expect("shared expert gate");
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let dims = QwenMoeDims {
        hidden_size: 2,
        num_experts: 2,
        moe_intermediate_size: 1,
        shared_expert_intermediate_size: 1,
    };
    let router = QwenMoeRouterProbe {
        logits: vec![0.0, 0.0],
        selected: vec![TopKWeight {
            index: 0,
            weight: 1.0,
        }],
    };

    let output = qwen_layer0_moe_forward(&store, &dims, &[1.0, 2.0], &router).expect("moe");

    assert_close(&output, &[1.4621172, 2.9242344], 1e-6);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_moe_forward_accumulation_uses_configured_backend() {
    let root = temp_snapshot_dir("qwen-moe-forward-accum-backend");
    write_tiny_moe_forward_snapshot(&root);
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let dims = QwenMoeDims {
        hidden_size: 2,
        num_experts: 2,
        moe_intermediate_size: 1,
        shared_expert_intermediate_size: 1,
    };
    let router = QwenMoeRouterProbe {
        logits: vec![0.0, 0.0],
        selected: vec![TopKWeight {
            index: 0,
            weight: 1.0,
        }],
    };
    let expected = qwen_layer0_moe_forward(&store, &dims, &[1.0, 2.0], &router).expect("cpu moe");
    let matvec = RecordingMatvecBackend::default();

    let output =
        qwen_layer_moe_forward_with_matvec(&store, 0, &dims, &[1.0, 2.0], &router, &matvec)
            .expect("recording moe");

    assert_close(&output, &expected, 1e-6);
    assert_eq!(matvec.weighted_sum_calls.get(), 2);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_full_attention_first_token_requires_key_and_norm_weights() {
    let root = temp_snapshot_dir("qwen-full-attn-required");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 48 },
            "weight_map": {
                "model.language_model.layers.0.self_attn.q_proj.weight": "q.safetensors",
                "model.language_model.layers.0.self_attn.v_proj.weight": "v.safetensors",
                "model.language_model.layers.0.self_attn.o_proj.weight": "o.safetensors"
            }
        })
        .to_string(),
    )
    .expect("index");
    std::fs::write(
        root.join("q.safetensors"),
        tiny_safetensors_bf16(
            "model.language_model.layers.0.self_attn.q_proj.weight",
            &[4, 2],
            &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ),
    )
    .expect("q");
    std::fs::write(
        root.join("v.safetensors"),
        tiny_safetensors_bf16(
            "model.language_model.layers.0.self_attn.v_proj.weight",
            &[2, 2],
            &[1.0, 0.0, 0.0, 1.0],
        ),
    )
    .expect("v");
    std::fs::write(
        root.join("o.safetensors"),
        tiny_safetensors_bf16(
            "model.language_model.layers.0.self_attn.o_proj.weight",
            &[2, 2],
            &[1.0, 0.0, 0.0, 1.0],
        ),
    )
    .expect("o");
    let store = SafeTensorShardStore::open(&root).expect("store opens");

    let err = qwen_layer_full_attention_first_token(
        &store,
        &tiny_qwen_spec(AttentionKind::FullAttention),
        0,
        &[1.0, 1.0],
    )
    .expect_err("full attention requires k_proj/q_norm/k_norm");

    assert_eq!(err.code(), "model_artifact_missing");
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_full_attention_sequence_with_cache_uses_indexed_weights() {
    let root = temp_snapshot_dir("qwen-full-attn-cache");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 64 },
            "weight_map": {
                "model.language_model.layers.0.self_attn.q_proj.weight": "q.safetensors",
                "model.language_model.layers.0.self_attn.k_proj.weight": "k.safetensors",
                "model.language_model.layers.0.self_attn.v_proj.weight": "v.safetensors",
                "model.language_model.layers.0.self_attn.q_norm.weight": "q_norm.safetensors",
                "model.language_model.layers.0.self_attn.k_norm.weight": "k_norm.safetensors",
                "model.language_model.layers.0.self_attn.o_proj.weight": "o.safetensors"
            }
        })
        .to_string(),
    )
    .expect("index");
    for (filename, tensor, shape, values) in [
        (
            "q.safetensors",
            "model.language_model.layers.0.self_attn.q_proj.weight",
            vec![4, 2],
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ),
        (
            "k.safetensors",
            "model.language_model.layers.0.self_attn.k_proj.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "v.safetensors",
            "model.language_model.layers.0.self_attn.v_proj.weight",
            vec![2, 2],
            vec![2.0, 0.0, 0.0, 4.0],
        ),
        (
            "q_norm.safetensors",
            "model.language_model.layers.0.self_attn.q_norm.weight",
            vec![2],
            vec![0.0, 0.0],
        ),
        (
            "k_norm.safetensors",
            "model.language_model.layers.0.self_attn.k_norm.weight",
            vec![2],
            vec![0.0, 0.0],
        ),
        (
            "o.safetensors",
            "model.language_model.layers.0.self_attn.o_proj.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
    ] {
        std::fs::write(
            root.join(filename),
            tiny_safetensors_bf16(tensor, &shape, &values),
        )
        .expect("tensor");
    }
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::FullAttention);
    let hidden_states = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let mut cache = LayerKvCache::new(2, 1, 2).expect("cache shape");

    let output =
        qwen_layer_full_attention_sequence_with_cache(&store, &spec, 0, &hidden_states, &mut cache)
            .expect("full attention sequence with cache");
    let expected = qwen_layer_full_attention_sequence(&store, &spec, 0, &hidden_states)
        .expect("full attention sequence");

    assert_eq!(cache.token_count(), 2);
    assert_close(&output[0], &expected[0], 1e-6);
    assert_close(&output[1], &expected[1], 1e-6);
    assert_close(cache.value(1).expect("value 1"), &[0.0, 4.0], 1e-6);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_full_attention_step_with_cache_uses_indexed_weights() {
    let root = temp_snapshot_dir("qwen-full-attn-step-cache");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 64 },
            "weight_map": {
                "model.language_model.layers.0.self_attn.q_proj.weight": "q.safetensors",
                "model.language_model.layers.0.self_attn.k_proj.weight": "k.safetensors",
                "model.language_model.layers.0.self_attn.v_proj.weight": "v.safetensors",
                "model.language_model.layers.0.self_attn.q_norm.weight": "q_norm.safetensors",
                "model.language_model.layers.0.self_attn.k_norm.weight": "k_norm.safetensors",
                "model.language_model.layers.0.self_attn.o_proj.weight": "o.safetensors"
            }
        })
        .to_string(),
    )
    .expect("index");
    for (filename, tensor, shape, values) in [
        (
            "q.safetensors",
            "model.language_model.layers.0.self_attn.q_proj.weight",
            vec![4, 2],
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ),
        (
            "k.safetensors",
            "model.language_model.layers.0.self_attn.k_proj.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "v.safetensors",
            "model.language_model.layers.0.self_attn.v_proj.weight",
            vec![2, 2],
            vec![2.0, 0.0, 0.0, 4.0],
        ),
        (
            "q_norm.safetensors",
            "model.language_model.layers.0.self_attn.q_norm.weight",
            vec![2],
            vec![0.0, 0.0],
        ),
        (
            "k_norm.safetensors",
            "model.language_model.layers.0.self_attn.k_norm.weight",
            vec![2],
            vec![0.0, 0.0],
        ),
        (
            "o.safetensors",
            "model.language_model.layers.0.self_attn.o_proj.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
    ] {
        std::fs::write(
            root.join(filename),
            tiny_safetensors_bf16(tensor, &shape, &values),
        )
        .expect("tensor");
    }
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::FullAttention);
    let hidden_states = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let mut expected_cache = LayerKvCache::new(3, 1, 2).expect("cache shape");
    let expected_output = qwen_layer_full_attention_sequence_with_cache(
        &store,
        &spec,
        0,
        &hidden_states,
        &mut expected_cache,
    )
    .expect("full cached sequence");
    let prefill = hidden_states[..2].to_vec();
    let mut cache = LayerKvCache::new(3, 1, 2).expect("cache shape");
    qwen_layer_full_attention_sequence_with_cache(&store, &spec, 0, &prefill, &mut cache)
        .expect("initial cached sequence");

    let output =
        qwen_layer_full_attention_step_with_cache(&store, &spec, 0, &hidden_states[2], &mut cache)
            .expect("full attention step");

    assert_close(&output, &expected_output[2], 1e-6);
    assert_eq!(cache.token_count(), 3);
    assert_close(cache.keys(), expected_cache.keys(), 1e-6);
    assert_close(cache.values(), expected_cache.values(), 1e-6);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_full_attention_normalization_uses_configured_matvec_backend() {
    let root = temp_snapshot_dir("qwen-full-attn-custom-norm");
    write_tiny_full_attention_snapshot(&root);
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::FullAttention);
    let hidden_states = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let mut expected_cache = LayerKvCache::new(3, 1, 2).expect("expected cache shape");
    let expected_output = qwen_layer_full_attention_sequence_with_cache(
        &store,
        &spec,
        0,
        &hidden_states,
        &mut expected_cache,
    )
    .expect("full cached sequence");
    let prefill = hidden_states[..2].to_vec();
    let mut cache = LayerKvCache::new(3, 1, 2).expect("cache shape");
    let matvec = RecordingMatvecBackend::default();

    let output = qwen_layer_full_attention_sequence_with_cache_with_matvec(
        &store, &spec, 0, &prefill, &mut cache, &matvec,
    )
    .expect("recording full cached sequence");
    let after_prefill_norm_calls = matvec.rms_norm_calls.get();
    let decoded = qwen_layer_full_attention_step_with_cache_with_matvec(
        &store,
        &spec,
        0,
        &hidden_states[2],
        &mut cache,
        &matvec,
    )
    .expect("recording full attention step");

    assert_close(&output[0], &expected_output[0], 1e-6);
    assert_close(&output[1], &expected_output[1], 1e-6);
    assert_close(&decoded, &expected_output[2], 1e-6);
    assert_eq!(after_prefill_norm_calls, 4);
    assert_eq!(matvec.rms_norm_calls.get(), 6);
    assert_close(cache.keys(), expected_cache.keys(), 1e-6);
    assert_close(cache.values(), expected_cache.values(), 1e-6);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_full_attention_softmax_uses_configured_matvec_backend() {
    let root = temp_snapshot_dir("qwen-full-attn-custom-softmax");
    write_tiny_full_attention_snapshot(&root);
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::FullAttention);
    let hidden_states = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let mut expected_cache = LayerKvCache::new(3, 1, 2).expect("expected cache shape");
    let expected_output = qwen_layer_full_attention_sequence_with_cache(
        &store,
        &spec,
        0,
        &hidden_states,
        &mut expected_cache,
    )
    .expect("full cached sequence");
    let prefill = hidden_states[..2].to_vec();
    let mut cache = LayerKvCache::new(3, 1, 2).expect("cache shape");
    let matvec = RecordingMatvecBackend::default();

    let output = qwen_layer_full_attention_sequence_with_cache_with_matvec(
        &store, &spec, 0, &prefill, &mut cache, &matvec,
    )
    .expect("recording full cached sequence");
    let after_prefill_softmax_calls = matvec.softmax_calls.get();
    let decoded = qwen_layer_full_attention_step_with_cache_with_matvec(
        &store,
        &spec,
        0,
        &hidden_states[2],
        &mut cache,
        &matvec,
    )
    .expect("recording full attention step");

    assert_close(&output[0], &expected_output[0], 1e-6);
    assert_close(&output[1], &expected_output[1], 1e-6);
    assert_close(&decoded, &expected_output[2], 1e-6);
    assert_eq!(after_prefill_softmax_calls, 2);
    assert_eq!(matvec.softmax_calls.get(), 3);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_full_attention_scores_use_configured_matvec_backend() {
    let root = temp_snapshot_dir("qwen-full-attn-custom-scores");
    write_tiny_full_attention_snapshot(&root);
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::FullAttention);
    let hidden_states = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let mut expected_cache = LayerKvCache::new(3, 1, 2).expect("expected cache shape");
    let expected_output = qwen_layer_full_attention_sequence_with_cache(
        &store,
        &spec,
        0,
        &hidden_states,
        &mut expected_cache,
    )
    .expect("full cached sequence");
    let prefill = hidden_states[..2].to_vec();
    let mut cache = LayerKvCache::new(3, 1, 2).expect("cache shape");
    let matvec = RecordingMatvecBackend::default();

    let output = qwen_layer_full_attention_sequence_with_cache_with_matvec(
        &store, &spec, 0, &prefill, &mut cache, &matvec,
    )
    .expect("recording full cached sequence");
    let after_prefill_dense_calls = matvec.dense_f32_calls.get();
    let decoded = qwen_layer_full_attention_step_with_cache_with_matvec(
        &store,
        &spec,
        0,
        &hidden_states[2],
        &mut cache,
        &matvec,
    )
    .expect("recording full attention step");

    assert_close(&output[0], &expected_output[0], 1e-6);
    assert_close(&output[1], &expected_output[1], 1e-6);
    assert_close(&decoded, &expected_output[2], 1e-6);
    assert_eq!(after_prefill_dense_calls, 4);
    assert_eq!(matvec.dense_f32_calls.get(), 6);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_full_attention_value_mix_uses_configured_matvec_backend() {
    let root = temp_snapshot_dir("qwen-full-attn-custom-value-mix");
    write_tiny_full_attention_snapshot(&root);
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::FullAttention);
    let hidden_states = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let mut expected_cache = LayerKvCache::new(3, 1, 2).expect("expected cache shape");
    let expected_output = qwen_layer_full_attention_sequence_with_cache(
        &store,
        &spec,
        0,
        &hidden_states,
        &mut expected_cache,
    )
    .expect("full cached sequence");
    let prefill = hidden_states[..2].to_vec();
    let mut cache = LayerKvCache::new(3, 1, 2).expect("cache shape");
    let matvec = RecordingMatvecBackend::default();

    let output = qwen_layer_full_attention_sequence_with_cache_with_matvec(
        &store, &spec, 0, &prefill, &mut cache, &matvec,
    )
    .expect("recording full cached sequence");
    let after_prefill_weighted_sum_calls = matvec.weighted_sum_calls.get();
    let decoded = qwen_layer_full_attention_step_with_cache_with_matvec(
        &store,
        &spec,
        0,
        &hidden_states[2],
        &mut cache,
        &matvec,
    )
    .expect("recording full attention step");

    assert_close(&output[0], &expected_output[0], 1e-6);
    assert_close(&output[1], &expected_output[1], 1e-6);
    assert_close(&decoded, &expected_output[2], 1e-6);
    assert_eq!(after_prefill_weighted_sum_calls, 2);
    assert_eq!(matvec.weighted_sum_calls.get(), 3);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_linear_attention_first_token_requires_delta_parameters() {
    let root = temp_snapshot_dir("qwen-linear-delta-required");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 96 },
            "weight_map": {
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": "qkv.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_z.weight": "z.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_b.weight": "b.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_a.weight": "a.safetensors",
                "model.language_model.layers.0.linear_attn.conv1d.weight": "conv.safetensors",
                "model.language_model.layers.0.linear_attn.norm.weight": "norm.safetensors",
                "model.language_model.layers.0.linear_attn.out_proj.weight": "out.safetensors"
            }
        })
        .to_string(),
    )
    .expect("index");
    std::fs::write(
        root.join("qkv.safetensors"),
        tiny_safetensors_bf16(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            &[4, 2],
            &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0],
        ),
    )
    .expect("qkv");
    for (filename, tensor, shape, values) in [
        (
            "z.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_z.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "b.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_b.weight",
            vec![1, 2],
            vec![1.0, 0.0],
        ),
        (
            "a.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_a.weight",
            vec![1, 2],
            vec![1.0, 0.0],
        ),
        (
            "conv.safetensors",
            "model.language_model.layers.0.linear_attn.conv1d.weight",
            vec![4, 1],
            vec![1.0, 1.0, 1.0, 1.0],
        ),
        (
            "norm.safetensors",
            "model.language_model.layers.0.linear_attn.norm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "out.safetensors",
            "model.language_model.layers.0.linear_attn.out_proj.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
    ] {
        std::fs::write(
            root.join(filename),
            tiny_safetensors_bf16(tensor, &shape, &values),
        )
        .expect("tensor");
    }
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let projections =
        qwen_layer_linear_attention_projections(&store, 0, &[1.0, 1.0]).expect("projections");

    let err = qwen_layer_linear_attention_first_token(
        &store,
        &tiny_qwen_spec(AttentionKind::LinearAttention),
        0,
        &projections,
    )
    .expect_err("linear attention requires A_log and dt_bias");

    assert_eq!(err.code(), "model_artifact_missing");
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_linear_attention_sequence_with_cache_uses_indexed_weights() {
    let root = temp_snapshot_dir("qwen-linear-attn-cache");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 96 },
            "weight_map": {
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": "qkv.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_z.weight": "z.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_b.weight": "b.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_a.weight": "a.safetensors",
                "model.language_model.layers.0.linear_attn.dt_bias": "dt.safetensors",
                "model.language_model.layers.0.linear_attn.A_log": "a_log.safetensors",
                "model.language_model.layers.0.linear_attn.conv1d.weight": "conv.safetensors",
                "model.language_model.layers.0.linear_attn.norm.weight": "norm.safetensors",
                "model.language_model.layers.0.linear_attn.out_proj.weight": "out.safetensors"
            }
        })
        .to_string(),
    )
    .expect("index");
    for (filename, tensor, shape, values) in [
        (
            "qkv.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            vec![4, 2],
            vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 4.0],
        ),
        (
            "z.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_z.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "b.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_b.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "a.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_a.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "dt.safetensors",
            "model.language_model.layers.0.linear_attn.dt_bias",
            vec![1],
            vec![0.0],
        ),
        (
            "a_log.safetensors",
            "model.language_model.layers.0.linear_attn.A_log",
            vec![1],
            vec![0.0],
        ),
        (
            "conv.safetensors",
            "model.language_model.layers.0.linear_attn.conv1d.weight",
            vec![4, 1],
            vec![1.0, 1.0, 1.0, 1.0],
        ),
        (
            "norm.safetensors",
            "model.language_model.layers.0.linear_attn.norm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "out.safetensors",
            "model.language_model.layers.0.linear_attn.out_proj.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
    ] {
        std::fs::write(
            root.join(filename),
            tiny_safetensors_bf16(tensor, &shape, &values),
        )
        .expect("tensor");
    }
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::LinearAttention);
    let hidden_states = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let mut cache = LinearAttentionCache::new(1, 4, 1, 1, 2).expect("cache shape");

    let output = qwen_layer_linear_attention_sequence_with_cache(
        &store,
        &spec,
        0,
        &hidden_states,
        &mut cache,
    )
    .expect("linear attention sequence with cache");
    let expected = qwen_layer_linear_attention_sequence(&store, &spec, 0, &hidden_states)
        .expect("linear attention sequence");

    assert_eq!(cache.token_count(), 2);
    assert_close(&output[0], &expected[0], 1e-6);
    assert_close(&output[1], &expected[1], 1e-6);
    assert_close(cache.conv_window(), &[0.0, 1.0, 0.0, 4.0], 1e-6);
    assert!(cache.recurrent_state().iter().any(|value| *value != 0.0));
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_linear_attention_step_with_cache_uses_indexed_weights() {
    let root = temp_snapshot_dir("qwen-linear-attn-step-cache");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 96 },
            "weight_map": {
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": "qkv.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_z.weight": "z.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_b.weight": "b.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_a.weight": "a.safetensors",
                "model.language_model.layers.0.linear_attn.dt_bias": "dt.safetensors",
                "model.language_model.layers.0.linear_attn.A_log": "a_log.safetensors",
                "model.language_model.layers.0.linear_attn.conv1d.weight": "conv.safetensors",
                "model.language_model.layers.0.linear_attn.norm.weight": "norm.safetensors",
                "model.language_model.layers.0.linear_attn.out_proj.weight": "out.safetensors"
            }
        })
        .to_string(),
    )
    .expect("index");
    for (filename, tensor, shape, values) in [
        (
            "qkv.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            vec![4, 2],
            vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 4.0],
        ),
        (
            "z.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_z.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "b.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_b.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "a.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_a.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "dt.safetensors",
            "model.language_model.layers.0.linear_attn.dt_bias",
            vec![1],
            vec![0.0],
        ),
        (
            "a_log.safetensors",
            "model.language_model.layers.0.linear_attn.A_log",
            vec![1],
            vec![0.0],
        ),
        (
            "conv.safetensors",
            "model.language_model.layers.0.linear_attn.conv1d.weight",
            vec![4, 1],
            vec![1.0, 1.0, 1.0, 1.0],
        ),
        (
            "norm.safetensors",
            "model.language_model.layers.0.linear_attn.norm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "out.safetensors",
            "model.language_model.layers.0.linear_attn.out_proj.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
    ] {
        std::fs::write(
            root.join(filename),
            tiny_safetensors_bf16(tensor, &shape, &values),
        )
        .expect("tensor");
    }
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::LinearAttention);
    let hidden_states = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let mut expected_cache = LinearAttentionCache::new(1, 4, 1, 1, 2).expect("cache shape");
    let expected_output = qwen_layer_linear_attention_sequence_with_cache(
        &store,
        &spec,
        0,
        &hidden_states,
        &mut expected_cache,
    )
    .expect("full cached sequence");
    let prefill = hidden_states[..2].to_vec();
    let mut cache = LinearAttentionCache::new(1, 4, 1, 1, 2).expect("cache shape");
    qwen_layer_linear_attention_sequence_with_cache(&store, &spec, 0, &prefill, &mut cache)
        .expect("initial cached sequence");

    let output = qwen_layer_linear_attention_step_with_cache(
        &store,
        &spec,
        0,
        &hidden_states[2],
        &mut cache,
    )
    .expect("linear attention step");

    assert_close(&output, &expected_output[2], 1e-6);
    assert_eq!(cache.token_count(), 3);
    assert_close(cache.conv_window(), expected_cache.conv_window(), 1e-6);
    assert_close(
        cache.recurrent_state(),
        expected_cache.recurrent_state(),
        1e-6,
    );
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_linear_attention_normalization_uses_configured_matvec_backend() {
    let root = temp_snapshot_dir("qwen-linear-attn-custom-norm");
    write_tiny_linear_decoder_snapshot(&root);
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::LinearAttention);
    let hidden_states = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let mut expected_cache = LinearAttentionCache::new(1, 4, 1, 1, 2).expect("expected cache");
    let expected = qwen_layer_linear_attention_sequence_with_cache(
        &store,
        &spec,
        0,
        &hidden_states,
        &mut expected_cache,
    )
    .expect("cpu cached sequence");
    let mut cache = LinearAttentionCache::new(1, 4, 1, 1, 2).expect("recording cache");
    let matvec = RecordingMatvecBackend::default();

    let output = qwen_layer_linear_attention_sequence_with_cache_with_matvec(
        &store,
        &spec,
        0,
        &hidden_states,
        &mut cache,
        &matvec,
    )
    .expect("recording cached sequence");

    assert_close(&output[0], &expected[0], 1e-6);
    assert_close(&output[1], &expected[1], 1e-6);
    assert_eq!(matvec.rms_norm_calls.get(), 6);
    assert_close(cache.conv_window(), expected_cache.conv_window(), 1e-6);
    assert_close(
        cache.recurrent_state(),
        expected_cache.recurrent_state(),
        1e-6,
    );
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_linear_attention_recurrent_matvecs_use_configured_backend() {
    let root = temp_snapshot_dir("qwen-linear-attn-recurrent-matvecs");
    write_tiny_linear_decoder_snapshot(&root);
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::LinearAttention);
    let hidden_states = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let mut expected_cache = LinearAttentionCache::new(1, 4, 1, 1, 2).expect("expected cache");
    let expected = qwen_layer_linear_attention_sequence_with_cache(
        &store,
        &spec,
        0,
        &hidden_states,
        &mut expected_cache,
    )
    .expect("cpu cached sequence");
    let mut cache = LinearAttentionCache::new(1, 4, 1, 1, 2).expect("recording cache");
    let matvec = RecordingMatvecBackend::default();

    let output = qwen_layer_linear_attention_sequence_with_cache_with_matvec(
        &store,
        &spec,
        0,
        &hidden_states,
        &mut cache,
        &matvec,
    )
    .expect("recording cached sequence");

    assert_close(&output[0], &expected[0], 1e-6);
    assert_close(&output[1], &expected[1], 1e-6);
    assert_eq!(matvec.dense_f32_calls.get(), 6);
    assert_close(
        cache.recurrent_state(),
        expected_cache.recurrent_state(),
        1e-6,
    );
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_linear_attention_recurrent_decay_and_update_use_configured_backend() {
    let root = temp_snapshot_dir("qwen-linear-attn-recurrent-update");
    write_tiny_linear_decoder_snapshot(&root);
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::LinearAttention);
    let hidden_states = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let mut expected_cache = LinearAttentionCache::new(1, 4, 1, 1, 2).expect("expected cache");
    let expected = qwen_layer_linear_attention_sequence_with_cache(
        &store,
        &spec,
        0,
        &hidden_states,
        &mut expected_cache,
    )
    .expect("cpu cached sequence");
    let mut cache = LinearAttentionCache::new(1, 4, 1, 1, 2).expect("recording cache");
    let matvec = RecordingMatvecBackend::default();

    let output = qwen_layer_linear_attention_sequence_with_cache_with_matvec(
        &store,
        &spec,
        0,
        &hidden_states,
        &mut cache,
        &matvec,
    )
    .expect("recording cached sequence");

    assert_close(&output[0], &expected[0], 1e-6);
    assert_close(&output[1], &expected[1], 1e-6);
    assert_eq!(matvec.recurrent_update_calls.get(), 4);
    assert_close(
        cache.recurrent_state(),
        expected_cache.recurrent_state(),
        1e-6,
    );
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_linear_attention_convolution_uses_configured_backend() {
    let root = temp_snapshot_dir("qwen-linear-attn-conv-backend");
    write_tiny_linear_decoder_snapshot(&root);
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::LinearAttention);
    let hidden_states = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let mut expected_cache = LinearAttentionCache::new(1, 4, 1, 1, 2).expect("expected cache");
    let expected = qwen_layer_linear_attention_sequence_with_cache(
        &store,
        &spec,
        0,
        &hidden_states,
        &mut expected_cache,
    )
    .expect("cpu cached sequence");
    let mut cache = LinearAttentionCache::new(1, 4, 1, 1, 2).expect("recording cache");
    let matvec = RecordingMatvecBackend::default();

    let output = qwen_layer_linear_attention_sequence_with_cache_with_matvec(
        &store,
        &spec,
        0,
        &hidden_states,
        &mut cache,
        &matvec,
    )
    .expect("recording cached sequence");

    assert_close(&output[0], &expected[0], 1e-6);
    assert_close(&output[1], &expected[1], 1e-6);
    assert_eq!(matvec.conv1d_calls.get(), 2);
    assert_close(cache.conv_window(), expected_cache.conv_window(), 1e-6);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_final_norm_and_lm_head_top_k_use_indexed_weights() {
    let root = temp_snapshot_dir("qwen-lm-head");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 16 },
            "weight_map": {
                "model.language_model.norm.weight": "norm.safetensors",
                "lm_head.weight": "lm_head.safetensors"
            }
        })
        .to_string(),
    )
    .expect("index");
    std::fs::write(
        root.join("norm.safetensors"),
        tiny_safetensors_bf16("model.language_model.norm.weight", &[2], &[0.0, 1.0]),
    )
    .expect("norm");
    std::fs::write(
        root.join("lm_head.safetensors"),
        tiny_safetensors_bf16(
            "lm_head.weight",
            &[2, 2],
            &[
                1.0, 0.0, //
                0.0, 1.0,
            ],
        ),
    )
    .expect("lm head");
    let store = SafeTensorShardStore::open(&root).expect("store opens");

    let normalized = qwen_final_norm(&store, &[3.0, 4.0], 2, 0.0).expect("final norm");
    let top = qwen_lm_head_top_k(&store, &normalized, 1, 1).expect("lm head");
    let logits = qwen_lm_head_logits(&store, &normalized, 1).expect("lm head logits");

    assert_close(&normalized, &[0.84852815, 2.2627418], 1e-6);
    assert_eq!(top[0].index, 1);
    assert_close(&[top[0].logit], &[2.2627418], 1e-6);
    assert_close(&logits, &[0.84852815, 2.2627418], 1e-6);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_layer_caches_match_hybrid_attention_shapes() {
    let mut spec = tiny_qwen_spec(AttentionKind::LinearAttention);
    spec.num_hidden_layers = 2;
    spec.layer_kinds = vec![AttentionKind::LinearAttention, AttentionKind::FullAttention];

    let caches = qwen_layer_caches_for_spec(&spec, 4).expect("layer caches");

    assert_eq!(caches.len(), 2);
    match &caches[0] {
        QwenLayerCache::Linear(cache) => {
            assert_eq!(cache.conv_kernel_size(), 1);
            assert_eq!(cache.conv_dim(), 4);
            assert_eq!(cache.num_value_heads(), 1);
            assert_eq!(cache.key_head_dim(), 1);
            assert_eq!(cache.value_head_dim(), 2);
        }
        QwenLayerCache::Full(_) => panic!("layer 0 should be linear attention"),
    }
    match &caches[1] {
        QwenLayerCache::Full(cache) => {
            assert_eq!(cache.max_tokens(), 4);
            assert_eq!(cache.key_value_heads(), 1);
            assert_eq!(cache.head_dim(), 2);
        }
        QwenLayerCache::Linear(_) => panic!("layer 1 should be full attention"),
    }
}

#[test]
fn qwen_prefill_sequence_with_cache_updates_layer_cache() {
    let root = temp_snapshot_dir("qwen-prefill-cache");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 256 },
            "weight_map": {
                "model.language_model.embed_tokens.weight": "embed.safetensors",
                "model.language_model.layers.0.input_layernorm.weight": "input_norm.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": "qkv.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_z.weight": "z.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_b.weight": "b.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_a.weight": "a.safetensors",
                "model.language_model.layers.0.linear_attn.dt_bias": "dt.safetensors",
                "model.language_model.layers.0.linear_attn.A_log": "a_log.safetensors",
                "model.language_model.layers.0.linear_attn.conv1d.weight": "conv.safetensors",
                "model.language_model.layers.0.linear_attn.norm.weight": "attn_norm.safetensors",
                "model.language_model.layers.0.linear_attn.out_proj.weight": "out.safetensors",
                "model.language_model.layers.0.post_attention_layernorm.weight": "post_norm.safetensors",
                "model.language_model.layers.0.mlp.gate.weight": "router.safetensors",
                "model.language_model.layers.0.mlp.experts.gate_up_proj": "experts_gate_up.safetensors",
                "model.language_model.layers.0.mlp.experts.down_proj": "experts_down.safetensors",
                "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight": "shared_gate.safetensors",
                "model.language_model.layers.0.mlp.shared_expert.up_proj.weight": "shared_up.safetensors",
                "model.language_model.layers.0.mlp.shared_expert.down_proj.weight": "shared_down.safetensors",
                "model.language_model.layers.0.mlp.shared_expert_gate.weight": "shared_expert_gate.safetensors"
            }
        })
        .to_string(),
    )
    .expect("index");
    for (filename, tensor, shape, values) in [
        (
            "embed.safetensors",
            "model.language_model.embed_tokens.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "input_norm.safetensors",
            "model.language_model.layers.0.input_layernorm.weight",
            vec![2],
            vec![0.0, 0.0],
        ),
        (
            "qkv.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            vec![4, 2],
            vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 4.0],
        ),
        (
            "z.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_z.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "b.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_b.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "a.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_a.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "dt.safetensors",
            "model.language_model.layers.0.linear_attn.dt_bias",
            vec![1],
            vec![0.0],
        ),
        (
            "a_log.safetensors",
            "model.language_model.layers.0.linear_attn.A_log",
            vec![1],
            vec![0.0],
        ),
        (
            "conv.safetensors",
            "model.language_model.layers.0.linear_attn.conv1d.weight",
            vec![4, 1],
            vec![1.0, 1.0, 1.0, 1.0],
        ),
        (
            "attn_norm.safetensors",
            "model.language_model.layers.0.linear_attn.norm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "out.safetensors",
            "model.language_model.layers.0.linear_attn.out_proj.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "post_norm.safetensors",
            "model.language_model.layers.0.post_attention_layernorm.weight",
            vec![2],
            vec![0.0, 0.0],
        ),
        (
            "router.safetensors",
            "model.language_model.layers.0.mlp.gate.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "experts_gate_up.safetensors",
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            vec![2, 2],
            vec![0.0, 0.0, 0.0, 0.0],
        ),
        (
            "experts_down.safetensors",
            "model.language_model.layers.0.mlp.experts.down_proj",
            vec![2, 1],
            vec![0.0, 0.0],
        ),
        (
            "shared_gate.safetensors",
            "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "shared_up.safetensors",
            "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "shared_down.safetensors",
            "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
            vec![2, 1],
            vec![0.0, 0.0],
        ),
        (
            "shared_expert_gate.safetensors",
            "model.language_model.layers.0.mlp.shared_expert_gate.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
    ] {
        std::fs::write(
            root.join(filename),
            tiny_safetensors_bf16(tensor, &shape, &values),
        )
        .expect("tensor");
    }
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::LinearAttention);
    let mut caches = qwen_layer_caches_for_spec(&spec, 2).expect("layer caches");

    let output = qwen_prefill_sequence_with_cache(&store, &spec, &[0, 1], &mut caches)
        .expect("cached prefill");
    let expected = qwen_prefill_sequence(&store, &spec, &[0, 1]).expect("uncached prefill");

    assert_eq!(output.len(), expected.len());
    assert_close(&output[0], &expected[0], 1e-5);
    assert_close(&output[1], &expected[1], 1e-5);
    match &caches[0] {
        QwenLayerCache::Linear(cache) => {
            assert_eq!(cache.token_count(), 2);
            assert!(cache.recurrent_state().iter().any(|value| *value != 0.0));
        }
        QwenLayerCache::Full(_) => panic!("layer 0 should be linear attention"),
    }
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_decode_token_with_cache_matches_cached_prefill_suffix() {
    let root = temp_snapshot_dir("qwen-decode-token-cache");
    write_tiny_linear_decoder_snapshot(&root);
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::LinearAttention);
    let mut expected_caches = qwen_layer_caches_for_spec(&spec, 3).expect("expected caches");
    let expected =
        qwen_prefill_sequence_with_cache(&store, &spec, &[0, 1, 0], &mut expected_caches)
            .expect("full cached prefill");
    let mut caches = qwen_layer_caches_for_spec(&spec, 3).expect("layer caches");
    qwen_prefill_sequence_with_cache(&store, &spec, &[0, 1], &mut caches)
        .expect("initial cached prefill");

    let output =
        qwen_decode_token_with_cache(&store, &spec, 0, &mut caches).expect("cached token decode");

    assert_close(&output, &expected[2], 1e-5);
    match &caches[0] {
        QwenLayerCache::Linear(cache) => assert_eq!(cache.token_count(), 3),
        QwenLayerCache::Full(_) => panic!("layer 0 should be linear attention"),
    }
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_prefill_and_decode_use_configured_matvec_backend() {
    let root = temp_snapshot_dir("qwen-custom-matvec-cache");
    write_tiny_linear_decoder_snapshot(&root);
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::LinearAttention);
    let matvec = RecordingMatvecBackend::default();
    let expected =
        qwen_prefill_sequence_with_cache(&store, &spec, &[0, 1], &mut caches_for_spec(&spec, 3))
            .expect("cpu cached prefill");
    let mut recording_caches = qwen_layer_caches_for_spec(&spec, 3).expect("recording caches");

    let output = qwen_prefill_sequence_with_cache_with_matvec(
        &store,
        &spec,
        &[0, 1],
        &mut recording_caches,
        &matvec,
    )
    .expect("recording cached prefill");
    let decoded =
        qwen_decode_token_with_cache_with_matvec(&store, &spec, 0, &mut recording_caches, &matvec)
            .expect("recording cached decode");

    assert_eq!(output.len(), expected.len());
    assert_close(&output[0], &expected[0], 1e-5);
    assert_close(&output[1], &expected[1], 1e-5);
    assert_eq!(decoded.len(), spec.hidden_size as usize);
    assert!(matvec.batched_bf16_calls.get() > 0);
    assert!(matvec.single_bf16_calls.get() > 0);
    assert!(matvec.dense_f32_calls.get() > 0);
    assert!(matvec.rms_norm_calls.get() > 0);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_lm_head_uses_configured_matvec_backend() {
    let root = temp_snapshot_dir("qwen-lm-head-custom-matvec");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 12 },
            "weight_map": { "lm_head.weight": "lm_head.safetensors" }
        })
        .to_string(),
    )
    .expect("index");
    std::fs::write(
        root.join("lm_head.safetensors"),
        tiny_safetensors_bf16("lm_head.weight", &[3, 2], &[1.0, 0.0, 0.0, 2.0, -1.0, 1.0]),
    )
    .expect("lm head");
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let matvec = RecordingMatvecBackend::default();

    let top = qwen_lm_head_top_k_with_matvec(&store, &[1.0, 2.0], 2, 2, &matvec)
        .expect("top-k uses recording matvec");
    let logits = qwen_lm_head_logits_with_matvec(&store, &[1.0, 2.0], 2, &matvec)
        .expect("full logits use recording matvec");

    assert_eq!(top[0].index, 1);
    assert_eq!(top[0].logit, 4.0);
    assert_eq!(top[1].index, 0);
    assert_eq!(top[1].logit, 1.0);
    assert_eq!(logits, vec![1.0, 4.0, 1.0]);
    assert_eq!(matvec.top_k_bf16_calls.get(), 1);
    assert_eq!(matvec.rows_bf16_calls.get(), 1);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_final_norm_uses_configured_rms_norm_backend() {
    let root = temp_snapshot_dir("qwen-final-norm-custom-matvec");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 4 },
            "weight_map": { QWEN_FINAL_NORM_WEIGHT: "norm.safetensors" }
        })
        .to_string(),
    )
    .expect("index");
    std::fs::write(
        root.join("norm.safetensors"),
        tiny_safetensors_bf16(QWEN_FINAL_NORM_WEIGHT, &[2], &[0.0, 1.0]),
    )
    .expect("norm");
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let matvec = RecordingMatvecBackend::default();
    let expected = qwen_final_norm(&store, &[3.0, 4.0], 2, 0.0).expect("cpu final norm");

    let output = qwen_final_norm_with_matvec(&store, &[3.0, 4.0], 2, 0.0, &matvec)
        .expect("final norm uses recording backend");

    assert_close(&output, &expected, 1e-6);
    assert_eq!(matvec.rms_norm_calls.get(), 1);
    std::fs::remove_dir_all(root).ok();
}

#[derive(Default)]
struct RecordingMatvecBackend {
    single_bf16_calls: Cell<usize>,
    batched_bf16_calls: Cell<usize>,
    rows_bf16_calls: Cell<usize>,
    top_k_bf16_calls: Cell<usize>,
    dense_f32_calls: Cell<usize>,
    rms_norm_calls: Cell<usize>,
    softmax_calls: Cell<usize>,
    conv1d_calls: Cell<usize>,
    softmax_top_k_calls: Cell<usize>,
    weighted_sum_calls: Cell<usize>,
    recurrent_update_calls: Cell<usize>,
}

impl QwenMatvecBackend for RecordingMatvecBackend {
    fn bf16_matvec_row_major_f32(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        input: &[f32],
    ) -> Result<Vec<f32>, TensorLoadError> {
        self.single_bf16_calls.set(self.single_bf16_calls.get() + 1);
        CpuQwenMatvecBackend.bf16_matvec_row_major_f32(store, tensor, input)
    }

    fn bf16_matvecs_row_major_f32(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        inputs: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, TensorLoadError> {
        self.batched_bf16_calls
            .set(self.batched_bf16_calls.get() + 1);
        CpuQwenMatvecBackend.bf16_matvecs_row_major_f32(store, tensor, inputs)
    }

    fn bf16_matvec_rows_f32(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        input: &[f32],
        chunk_rows: usize,
    ) -> Result<Vec<f32>, TensorLoadError> {
        self.rows_bf16_calls.set(self.rows_bf16_calls.get() + 1);
        CpuQwenMatvecBackend.bf16_matvec_rows_f32(store, tensor, input, chunk_rows)
    }

    fn bf16_matvec_top_k_rows_f32(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        input: &[f32],
        top_k: usize,
        chunk_rows: usize,
    ) -> Result<Vec<TopKLogit>, TensorLoadError> {
        self.top_k_bf16_calls.set(self.top_k_bf16_calls.get() + 1);
        CpuQwenMatvecBackend.bf16_matvec_top_k_rows_f32(store, tensor, input, top_k, chunk_rows)
    }

    fn matvec_row_major_f32(
        &self,
        input: &[f32],
        weights: &[f32],
        rows: usize,
        columns: usize,
    ) -> Result<Vec<f32>, MathError> {
        self.dense_f32_calls.set(self.dense_f32_calls.get() + 1);
        CpuQwenMatvecBackend.matvec_row_major_f32(input, weights, rows, columns)
    }

    fn qwen_rms_norm_f32(
        &self,
        input: &[f32],
        weight: &[f32],
        eps: f32,
    ) -> Result<Vec<f32>, MathError> {
        self.rms_norm_calls.set(self.rms_norm_calls.get() + 1);
        qwen_rms_norm_f32(input, weight, eps)
    }

    fn softmax_f32(&self, scores: &[f32]) -> Result<Vec<f32>, MathError> {
        self.softmax_calls.set(self.softmax_calls.get() + 1);
        CpuQwenMatvecBackend.softmax_f32(scores)
    }

    fn linear_attention_conv1d_silu_f32(
        &self,
        window: &[f32],
        weights: &[f32],
        conv_dim: usize,
        kernel_size: usize,
    ) -> Result<Vec<f32>, MathError> {
        self.conv1d_calls.set(self.conv1d_calls.get() + 1);
        CpuQwenMatvecBackend.linear_attention_conv1d_silu_f32(
            window,
            weights,
            conv_dim,
            kernel_size,
        )
    }

    fn softmax_top_k_f32(
        &self,
        logits: &[f32],
        top_k: usize,
    ) -> Result<Vec<TopKWeight>, MathError> {
        self.softmax_top_k_calls
            .set(self.softmax_top_k_calls.get() + 1);
        CpuQwenMatvecBackend.softmax_top_k_f32(logits, top_k)
    }

    fn weighted_sum_f32(
        &self,
        values: &[f32],
        weights: &[f32],
        vector_len: usize,
    ) -> Result<Vec<f32>, MathError> {
        self.weighted_sum_calls
            .set(self.weighted_sum_calls.get() + 1);
        CpuQwenMatvecBackend.weighted_sum_f32(values, weights, vector_len)
    }

    fn linear_attention_recurrent_update_f32(
        &self,
        state: &[f32],
        key: &[f32],
        value: &[f32],
        memory: &[f32],
        beta: f32,
        decay: f32,
        key_head_dim: usize,
        value_head_dim: usize,
    ) -> Result<Vec<f32>, MathError> {
        self.recurrent_update_calls
            .set(self.recurrent_update_calls.get() + 1);
        CpuQwenMatvecBackend.linear_attention_recurrent_update_f32(
            state,
            key,
            value,
            memory,
            beta,
            decay,
            key_head_dim,
            value_head_dim,
        )
    }
}

fn caches_for_spec(spec: &QwenModelSpec, capacity: usize) -> Vec<QwenLayerCache> {
    qwen_layer_caches_for_spec(spec, capacity).expect("temporary caches")
}

fn write_tiny_moe_forward_snapshot(root: &std::path::Path) {
    std::fs::create_dir_all(root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 80 },
            "weight_map": {
                "model.language_model.layers.0.mlp.experts.gate_up_proj": "gate_up.safetensors",
                "model.language_model.layers.0.mlp.experts.down_proj": "down.safetensors",
                "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight": "shared_gate.safetensors",
                "model.language_model.layers.0.mlp.shared_expert.up_proj.weight": "shared_up.safetensors",
                "model.language_model.layers.0.mlp.shared_expert.down_proj.weight": "shared_down.safetensors",
                "model.language_model.layers.0.mlp.shared_expert_gate.weight": "shared_expert_gate.safetensors"
            }
        })
        .to_string(),
    )
    .expect("index");
    for (filename, tensor, shape, values) in [
        (
            "gate_up.safetensors",
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            vec![2, 2, 2],
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        ),
        (
            "down.safetensors",
            "model.language_model.layers.0.mlp.experts.down_proj",
            vec![2, 2, 1],
            vec![1.0, 2.0, 3.0, 4.0],
        ),
        (
            "shared_gate.safetensors",
            "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "shared_up.safetensors",
            "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "shared_down.safetensors",
            "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
            vec![2, 1],
            vec![0.0, 0.0],
        ),
        (
            "shared_expert_gate.safetensors",
            "model.language_model.layers.0.mlp.shared_expert_gate.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
    ] {
        std::fs::write(
            root.join(filename),
            tiny_safetensors_bf16(tensor, &shape, &values),
        )
        .expect("tensor");
    }
}

fn write_tiny_full_attention_snapshot(root: &std::path::Path) {
    std::fs::create_dir_all(root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 64 },
            "weight_map": {
                "model.language_model.layers.0.self_attn.q_proj.weight": "q.safetensors",
                "model.language_model.layers.0.self_attn.k_proj.weight": "k.safetensors",
                "model.language_model.layers.0.self_attn.v_proj.weight": "v.safetensors",
                "model.language_model.layers.0.self_attn.q_norm.weight": "q_norm.safetensors",
                "model.language_model.layers.0.self_attn.k_norm.weight": "k_norm.safetensors",
                "model.language_model.layers.0.self_attn.o_proj.weight": "o.safetensors"
            }
        })
        .to_string(),
    )
    .expect("index");
    for (filename, tensor, shape, values) in [
        (
            "q.safetensors",
            "model.language_model.layers.0.self_attn.q_proj.weight",
            vec![4, 2],
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ),
        (
            "k.safetensors",
            "model.language_model.layers.0.self_attn.k_proj.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "v.safetensors",
            "model.language_model.layers.0.self_attn.v_proj.weight",
            vec![2, 2],
            vec![2.0, 0.0, 0.0, 4.0],
        ),
        (
            "q_norm.safetensors",
            "model.language_model.layers.0.self_attn.q_norm.weight",
            vec![2],
            vec![0.0, 0.0],
        ),
        (
            "k_norm.safetensors",
            "model.language_model.layers.0.self_attn.k_norm.weight",
            vec![2],
            vec![0.0, 0.0],
        ),
        (
            "o.safetensors",
            "model.language_model.layers.0.self_attn.o_proj.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
    ] {
        std::fs::write(
            root.join(filename),
            tiny_safetensors_bf16(tensor, &shape, &values),
        )
        .expect("tensor");
    }
}

fn write_tiny_linear_decoder_snapshot(root: &std::path::Path) {
    std::fs::create_dir_all(root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 256 },
            "weight_map": {
                "model.language_model.embed_tokens.weight": "embed.safetensors",
                "model.language_model.layers.0.input_layernorm.weight": "input_norm.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": "qkv.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_z.weight": "z.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_b.weight": "b.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_a.weight": "a.safetensors",
                "model.language_model.layers.0.linear_attn.dt_bias": "dt.safetensors",
                "model.language_model.layers.0.linear_attn.A_log": "a_log.safetensors",
                "model.language_model.layers.0.linear_attn.conv1d.weight": "conv.safetensors",
                "model.language_model.layers.0.linear_attn.norm.weight": "attn_norm.safetensors",
                "model.language_model.layers.0.linear_attn.out_proj.weight": "out.safetensors",
                "model.language_model.layers.0.post_attention_layernorm.weight": "post_norm.safetensors",
                "model.language_model.layers.0.mlp.gate.weight": "router.safetensors",
                "model.language_model.layers.0.mlp.experts.gate_up_proj": "experts_gate_up.safetensors",
                "model.language_model.layers.0.mlp.experts.down_proj": "experts_down.safetensors",
                "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight": "shared_gate.safetensors",
                "model.language_model.layers.0.mlp.shared_expert.up_proj.weight": "shared_up.safetensors",
                "model.language_model.layers.0.mlp.shared_expert.down_proj.weight": "shared_down.safetensors",
                "model.language_model.layers.0.mlp.shared_expert_gate.weight": "shared_expert_gate.safetensors"
            }
        })
        .to_string(),
    )
    .expect("index");
    for (filename, tensor, shape, values) in [
        (
            "embed.safetensors",
            "model.language_model.embed_tokens.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "input_norm.safetensors",
            "model.language_model.layers.0.input_layernorm.weight",
            vec![2],
            vec![0.0, 0.0],
        ),
        (
            "qkv.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            vec![4, 2],
            vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 4.0],
        ),
        (
            "z.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_z.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "b.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_b.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "a.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_a.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "dt.safetensors",
            "model.language_model.layers.0.linear_attn.dt_bias",
            vec![1],
            vec![0.0],
        ),
        (
            "a_log.safetensors",
            "model.language_model.layers.0.linear_attn.A_log",
            vec![1],
            vec![0.0],
        ),
        (
            "conv.safetensors",
            "model.language_model.layers.0.linear_attn.conv1d.weight",
            vec![4, 1],
            vec![1.0, 1.0, 1.0, 1.0],
        ),
        (
            "attn_norm.safetensors",
            "model.language_model.layers.0.linear_attn.norm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "out.safetensors",
            "model.language_model.layers.0.linear_attn.out_proj.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "post_norm.safetensors",
            "model.language_model.layers.0.post_attention_layernorm.weight",
            vec![2],
            vec![0.0, 0.0],
        ),
        (
            "router.safetensors",
            "model.language_model.layers.0.mlp.gate.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "experts_gate_up.safetensors",
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            vec![2, 2],
            vec![0.0, 0.0, 0.0, 0.0],
        ),
        (
            "experts_down.safetensors",
            "model.language_model.layers.0.mlp.experts.down_proj",
            vec![2, 1],
            vec![0.0, 0.0],
        ),
        (
            "shared_gate.safetensors",
            "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "shared_up.safetensors",
            "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "shared_down.safetensors",
            "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
            vec![2, 1],
            vec![0.0, 0.0],
        ),
        (
            "shared_expert_gate.safetensors",
            "model.language_model.layers.0.mlp.shared_expert_gate.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
    ] {
        std::fs::write(
            root.join(filename),
            tiny_safetensors_bf16(tensor, &shape, &values),
        )
        .expect("tensor");
    }
}

fn tiny_safetensors_f32(name: &str, shape: &[usize], values: &[f32]) -> Vec<u8> {
    let mut data = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        data.extend_from_slice(&value.to_le_bytes());
    }
    tiny_safetensors(name, "F32", shape, &data)
}

fn tiny_safetensors_bf16(name: &str, shape: &[usize], values: &[f32]) -> Vec<u8> {
    let mut data = Vec::with_capacity(values.len() * 2);
    for value in values {
        data.extend_from_slice(&bf16_bits(*value).to_le_bytes());
    }
    tiny_safetensors(name, "BF16", shape, &data)
}

fn bf16_bits(value: f32) -> u16 {
    (value.to_bits() >> 16) as u16
}

fn tiny_safetensors(name: &str, dtype: &str, shape: &[usize], data: &[u8]) -> Vec<u8> {
    let data_len = data.len();
    let header = serde_json::json!({
        name: {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [0, data_len]
        }
    })
    .to_string();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(data);
    bytes
}

fn temp_safetensors_path(label: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "llm-backend-{label}-{}.safetensors",
        std::process::id()
    ))
}

fn temp_snapshot_dir(label: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("llm-backend-{label}-{}", std::process::id()))
}

fn tiny_qwen_spec(kind: AttentionKind) -> QwenModelSpec {
    QwenModelSpec {
        family: ModelFamily::Qwen,
        architecture: "Qwen3_5MoeForConditionalGeneration".to_owned(),
        model_type: "qwen3_5_moe".to_owned(),
        text_model_type: "qwen3_5_moe_text".to_owned(),
        hidden_size: 2,
        rms_norm_eps: 1e-6,
        tie_word_embeddings: false,
        rope_theta: 10_000.0,
        partial_rotary_factor: 1.0,
        num_hidden_layers: 1,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 2,
        linear_num_key_heads: 1,
        linear_num_value_heads: 1,
        linear_key_head_dim: 1,
        linear_value_head_dim: 2,
        linear_conv_kernel_dim: 1,
        num_experts: 1,
        num_experts_per_tok: 1,
        moe_intermediate_size: 1,
        shared_expert_intermediate_size: 1,
        max_position_embeddings: 128,
        vocab_size: 8,
        layer_kinds: vec![kind],
    }
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual} expected {expected}"
        );
    }
}
