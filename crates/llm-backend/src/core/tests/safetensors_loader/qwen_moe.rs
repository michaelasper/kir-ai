use super::*;

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
        qwen_layer_moe_router(&store, 0, &[2.0, 3.0], 2, &matvec).expect("router");

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
        qwen_layer_moe_forward(&store, 0, &dims, &[1.0, 2.0], &router, &matvec)
            .expect("recording moe");

    assert_close(&output, &expected, 1e-6);
    assert_eq!(matvec.range_bf16_calls.get(), 3);
    assert_eq!(matvec.single_bf16_calls.get(), 4);
    assert_eq!(matvec.weighted_sum_calls.get(), 2);
    std::fs::remove_dir_all(root).ok();
}
