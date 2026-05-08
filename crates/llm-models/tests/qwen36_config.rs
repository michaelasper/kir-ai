use llm_models::{AttentionKind, ModelFamily, QwenModelSpec};

#[test]
fn parses_official_qwen36_config_as_hybrid_deltanet_moe() {
    let spec =
        QwenModelSpec::from_config_json(include_str!("../../../fixtures/qwen36/config.json"))
            .expect("official qwen3.6 config parses");

    assert_eq!(spec.family, ModelFamily::Qwen);
    assert_eq!(spec.architecture, "Qwen3_5MoeForConditionalGeneration");
    assert_eq!(spec.model_type, "qwen3_5_moe");
    assert_eq!(spec.hidden_size, 2048);
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
