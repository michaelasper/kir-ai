use llm_models::{AttentionKind, ModelFamily, QwenModelSpec, SafetensorsIndex};

#[test]
fn parses_official_qwen36_config_as_hybrid_deltanet_moe() {
    let spec =
        QwenModelSpec::from_config_json(include_str!("../../../fixtures/qwen36/config.json"))
            .expect("official qwen3.6 config parses");

    assert_eq!(spec.family, ModelFamily::Qwen);
    assert_eq!(spec.architecture, "Qwen3_5MoeForConditionalGeneration");
    assert_eq!(spec.model_type, "qwen3_5_moe");
    assert_eq!(spec.hidden_size, 2048);
    assert_eq!(spec.rms_norm_eps, 1e-6);
    assert!(!spec.tie_word_embeddings);
    assert_eq!(spec.rope_theta, 10_000_000.0);
    assert_eq!(spec.partial_rotary_factor, 0.25);
    assert_eq!(spec.linear_conv_kernel_dim, 4);
    assert_eq!(spec.num_hidden_layers, 40);
    assert_eq!(spec.num_experts, 256);
    assert_eq!(spec.num_experts_per_tok, 8);
    assert_eq!(spec.max_position_embeddings, 262_144);
    assert_eq!(spec.layer_kinds.len(), 40);
    assert_eq!(spec.layer_kinds[0], AttentionKind::LinearAttention);
    assert_eq!(spec.layer_kinds[3], AttentionKind::FullAttention);
    assert_eq!(
        spec.layer_kinds
            .iter()
            .filter(|kind| **kind == AttentionKind::LinearAttention)
            .count(),
        30
    );
    assert_eq!(
        spec.layer_kinds
            .iter()
            .filter(|kind| **kind == AttentionKind::FullAttention)
            .count(),
        10
    );
}

#[test]
fn rejects_non_qwen_architecture_for_qwen_spec() {
    let err = QwenModelSpec::from_config_json(
        r#"{"architectures":["LlamaForCausalLM"],"model_type":"llama"}"#,
    )
    .expect_err("wrong architecture fails closed");

    assert_eq!(err.code(), "unsupported_capability");
}

#[test]
fn validates_official_qwen36_safetensors_index_against_spec() {
    let spec =
        QwenModelSpec::from_config_json(include_str!("../../../fixtures/qwen36/config.json"))
            .expect("official qwen3.6 config parses");
    let index = SafetensorsIndex::from_json(include_str!(
        "../../../fixtures/qwen36/model.safetensors.index.json"
    ))
    .expect("official index parses");

    assert_eq!(index.total_size_bytes, 71_903_645_408);
    assert_eq!(index.tensor_count(), 1045);
    assert_eq!(index.shard_count(), 26);
    assert_eq!(
        index.shard_for("model.language_model.embed_tokens.weight"),
        Some("model-00001-of-00026.safetensors")
    );
    assert!(index.contains("model.language_model.layers.0.linear_attn.in_proj_qkv.weight"));
    assert!(!index.contains("model.language_model.layers.0.self_attn.q_proj.weight"));
    assert!(index.contains("model.language_model.layers.3.self_attn.q_proj.weight"));
    assert!(!index.contains("model.language_model.layers.3.linear_attn.in_proj_qkv.weight"));

    spec.validate_text_weights(&index)
        .expect("official qwen index satisfies text loader requirements");
}

#[test]
fn rejects_unsafe_safetensors_index_shard_paths() {
    for shard_path in [
        "../outside.safetensors",
        "/tmp/outside.safetensors",
        "nested\\outside.safetensors",
        "nested//outside.safetensors",
        "bad\u{0}path.safetensors",
        "",
    ] {
        let err = SafetensorsIndex::from_json(
            &serde_json::json!({
                "metadata": { "total_size": 1 },
                "weight_map": { "tensor.weight": shard_path }
            })
            .to_string(),
        )
        .expect_err("unsafe shard path fails closed");

        assert_eq!(err.code(), "invalid_request");
    }
}
