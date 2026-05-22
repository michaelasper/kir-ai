use llm_models::{GemmaModelSpec, QwenModelSpec, SafetensorsIndex};

#[test]
fn family_files_do_not_extend_safetensors_index_for_weight_validation() {
    for (path, source) in [
        ("src/qwen.rs", include_str!("../src/qwen.rs")),
        ("src/gemma.rs", include_str!("../src/gemma.rs")),
    ] {
        assert!(
            !source.contains("impl SafetensorsIndex"),
            "{path} should own weight validation through the family spec, not an impl SafetensorsIndex block"
        );
    }
}

#[test]
fn qwen_spec_owns_text_weight_validation() {
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
        .expect("Qwen text validation remains family-owned");
}

#[test]
fn gemma_spec_owns_text_weight_validation() {
    let spec = GemmaModelSpec::from_config_json(
        r#"{
          "architectures": ["Gemma4ForConditionalGeneration"],
          "model_type": "gemma4",
          "text_config": {
            "attention_bias": false,
            "attention_dropout": 0.0,
            "attention_k_eq_v": false,
            "enable_moe_block": false,
            "global_head_dim": 512,
            "head_dim": 256,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 1536,
            "hidden_size_per_layer_input": 0,
            "intermediate_size": 6144,
            "layer_types": ["full_attention"],
            "max_position_embeddings": 131072,
            "model_type": "gemma4_text",
            "num_attention_heads": 8,
            "num_global_key_value_heads": null,
            "num_hidden_layers": 1,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 0,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
              "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
                "rope_type": "proportional"
              },
              "sliding_attention": {
                "rope_theta": 10000.0,
                "rope_type": "default"
              }
            },
            "sliding_window": 512,
            "tie_word_embeddings": true,
            "use_double_wide_mlp": false,
            "vocab_size": 262144,
            "vocab_size_per_layer_input": 262144
          },
          "tie_word_embeddings": true
        }"#,
    )
    .expect("Gemma 4 config parses");
    let index = SafetensorsIndex::from_json(
        r#"{
          "metadata": {"total_size": 1},
          "weight_map": {
            "model.language_model.embed_tokens.weight": "model.safetensors",
            "model.language_model.norm.weight": "model.safetensors",
            "model.language_model.layers.0.input_layernorm.weight": "model.safetensors",
            "model.language_model.layers.0.layer_scalar": "model.safetensors",
            "model.language_model.layers.0.post_attention_layernorm.weight": "model.safetensors",
            "model.language_model.layers.0.pre_feedforward_layernorm.weight": "model.safetensors",
            "model.language_model.layers.0.post_feedforward_layernorm.weight": "model.safetensors",
            "model.language_model.layers.0.self_attn.q_proj.weight": "model.safetensors",
            "model.language_model.layers.0.self_attn.k_proj.weight": "model.safetensors",
            "model.language_model.layers.0.self_attn.v_proj.weight": "model.safetensors",
            "model.language_model.layers.0.self_attn.o_proj.weight": "model.safetensors",
            "model.language_model.layers.0.self_attn.q_norm.weight": "model.safetensors",
            "model.language_model.layers.0.self_attn.k_norm.weight": "model.safetensors",
            "model.language_model.layers.0.mlp.gate_proj.weight": "model.safetensors",
            "model.language_model.layers.0.mlp.up_proj.weight": "model.safetensors",
            "model.language_model.layers.0.mlp.down_proj.weight": "model.safetensors"
          }
        }"#,
    )
    .expect("index parses");

    spec.validate_text_weights(&index)
        .expect("Gemma text validation remains family-owned");
}
