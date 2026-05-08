use llm_backend::{QwenMoeDims, QwenMoeRouterProbe, TopKWeight};
use llm_backend::{
    SafeTensorArchive, SafeTensorFile, SafeTensorHeader, SafeTensorShardStore,
    qwen_embedding_and_layer0_norm, qwen_layer0_linear_attention_projections,
    qwen_layer0_moe_forward, qwen_layer0_moe_router, qwen_layer0_post_attention_norm,
};

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

    assert_eq!(
        store.bf16_row_f32("embed.weight", 1).expect("row"),
        vec![4.0, 5.0, 6.0]
    );
    assert_eq!(
        store
            .bf16_matvec_row_major_f32("embed.weight", &[1.0, 2.0, 3.0])
            .expect("matvec"),
        vec![14.0, 32.0]
    );
    assert_eq!(
        store
            .tensor_shard_path("embed.weight")
            .expect("shard path")
            .file_name()
            .and_then(|name| name.to_str()),
        Some("model-00001-of-00001.safetensors")
    );
    std::fs::remove_dir_all(root).ok();
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
        data.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
    }
    tiny_safetensors(name, "BF16", shape, &data)
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

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual} expected {expected}"
        );
    }
}
