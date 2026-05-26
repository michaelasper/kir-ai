use llm_models::{AttentionKind, NativeTextModelSpec, QwenModelSpec, SafetensorsIndex};

#[test]
fn parses_official_qwen3_dense_config_as_full_attention_swiglu() {
    let spec = QwenModelSpec::from_config_json(
        r#"{
          "architectures": ["Qwen3ForCausalLM"],
          "model_type": "qwen3",
          "hidden_size": 1024,
          "intermediate_size": 3072,
          "max_position_embeddings": 40960,
          "num_attention_heads": 16,
          "num_hidden_layers": 2,
          "num_key_value_heads": 8,
          "head_dim": 128,
          "rms_norm_eps": 1e-6,
          "rope_theta": 1000000,
          "tie_word_embeddings": true,
          "vocab_size": 151936
        }"#,
    )
    .expect("Qwen3 dense config parses");

    assert_eq!(spec.architecture, "Qwen3ForCausalLM");
    assert_eq!(spec.model_type, "qwen3");
    assert_eq!(spec.text_model_type, "qwen3");
    assert_eq!(spec.hidden_size, 1024);
    assert_eq!(spec.moe_intermediate_size, 3072);
    assert_eq!(spec.num_experts, 0);
    assert_eq!(spec.num_experts_per_tok, 0);
    assert_eq!(spec.layer_kinds, vec![AttentionKind::FullAttention; 2]);
    assert!(spec.tie_word_embeddings);
}

#[test]
fn parses_qwen3_dense_sliding_window_config() {
    let spec = QwenModelSpec::from_config_json(include_str!(
        "../../../fixtures/qwen3-dense-sliding-window/config.json"
    ))
    .expect("Qwen3 dense sliding-window config parses");

    assert_eq!(spec.architecture, "Qwen3ForCausalLM");
    assert_eq!(spec.model_type, "qwen3");
    assert_eq!(spec.layer_kinds, vec![AttentionKind::FullAttention; 2]);
    assert_eq!(spec.max_position_embeddings, 40960);
    assert_eq!(spec.sliding_window, Some(2048));
}

#[test]
fn qwen3_dense_ignores_sliding_window_value_when_disabled() {
    let spec = QwenModelSpec::from_config_json(
        r#"{
          "architectures": ["Qwen3ForCausalLM"],
          "model_type": "qwen3",
          "hidden_size": 1024,
          "intermediate_size": 3072,
          "max_position_embeddings": 40960,
          "num_attention_heads": 16,
          "num_hidden_layers": 2,
          "num_key_value_heads": 8,
          "head_dim": 128,
          "rms_norm_eps": 1e-6,
          "rope_theta": 1000000,
          "sliding_window": 2048,
          "tie_word_embeddings": true,
          "use_sliding_window": false,
          "vocab_size": 151936
        }"#,
    )
    .expect("disabled Qwen3 dense sliding-window metadata parses");

    assert_eq!(spec.sliding_window, None);
    assert_eq!(spec.layer_kinds, vec![AttentionKind::FullAttention; 2]);
}

#[test]
fn qwen3_dense_requires_window_when_sliding_window_enabled() {
    let err = QwenModelSpec::from_config_json(
        r#"{
          "architectures": ["Qwen3ForCausalLM"],
          "model_type": "qwen3",
          "hidden_size": 1024,
          "intermediate_size": 3072,
          "max_position_embeddings": 40960,
          "num_attention_heads": 16,
          "num_hidden_layers": 2,
          "num_key_value_heads": 8,
          "head_dim": 128,
          "rms_norm_eps": 1e-6,
          "rope_theta": 1000000,
          "tie_word_embeddings": true,
          "use_sliding_window": true,
          "vocab_size": 151936
        }"#,
    )
    .expect_err("enabled Qwen3 dense sliding-window config requires a window");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.to_string().contains("sliding_window"));
}

#[test]
fn qwen3_dense_index_accepts_model_namespace_and_tied_embeddings() {
    let spec = QwenModelSpec::from_config_json(
        r#"{
          "architectures": ["Qwen3ForCausalLM"],
          "model_type": "qwen3",
          "hidden_size": 2,
          "intermediate_size": 1,
          "max_position_embeddings": 16,
          "num_attention_heads": 1,
          "num_hidden_layers": 1,
          "num_key_value_heads": 1,
          "head_dim": 2,
          "rms_norm_eps": 1e-6,
          "rope_theta": 1000000,
          "tie_word_embeddings": true,
          "vocab_size": 8
        }"#,
    )
    .expect("Qwen3 dense config parses");
    let index = SafetensorsIndex::from_json(
        r#"{
          "metadata": {"total_size": 1},
          "weight_map": {
            "model.embed_tokens.weight": "model.safetensors",
            "model.norm.weight": "model.safetensors",
            "model.layers.0.input_layernorm.weight": "model.safetensors",
            "model.layers.0.post_attention_layernorm.weight": "model.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model.safetensors",
            "model.layers.0.self_attn.k_proj.weight": "model.safetensors",
            "model.layers.0.self_attn.v_proj.weight": "model.safetensors",
            "model.layers.0.self_attn.o_proj.weight": "model.safetensors",
            "model.layers.0.self_attn.q_norm.weight": "model.safetensors",
            "model.layers.0.self_attn.k_norm.weight": "model.safetensors",
            "model.layers.0.mlp.gate_proj.weight": "model.safetensors",
            "model.layers.0.mlp.up_proj.weight": "model.safetensors",
            "model.layers.0.mlp.down_proj.weight": "model.safetensors"
          }
        }"#,
    )
    .expect("index parses");

    spec.validate_text_weights(&index)
        .expect("Qwen3 dense tied embedding weights validate without lm_head");
    NativeTextModelSpec::Qwen(spec)
        .validate_text_weights(&index)
        .expect("generic native text validation routes to Qwen weights");
}

#[test]
fn qwen3_dense_config_rejects_unsupported_runtime_options() {
    for (field, value) in [
        ("hidden_act", r#""gelu""#),
        ("attention_bias", "true"),
        ("rope_scaling", r#"{"type":"linear","factor":2.0}"#),
    ] {
        let config = format!(
            r#"{{
              "architectures": ["Qwen3ForCausalLM"],
              "model_type": "qwen3",
              "hidden_size": 1024,
              "intermediate_size": 3072,
              "max_position_embeddings": 40960,
              "num_attention_heads": 16,
              "num_hidden_layers": 2,
              "num_key_value_heads": 8,
              "head_dim": 128,
              "rms_norm_eps": 1e-6,
              "rope_theta": 1000000,
              "tie_word_embeddings": true,
              "vocab_size": 151936,
              "{field}": {value}
            }}"#
        );

        let err = QwenModelSpec::from_config_json(&config)
            .expect_err("unsupported dense Qwen3 option must fail closed");

        assert_eq!(err.code(), "unsupported_capability");
        assert!(
            err.to_string().contains(field),
            "error should name unsupported field {field}: {err}"
        );
    }
}
