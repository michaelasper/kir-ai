use llm_models::{
    GemmaAttentionKind, GemmaModelSpec, ModelFamily, NativeTextModelSpec, SafetensorsIndex,
};
use serde_json::{Value, json};

#[test]
fn parses_representative_gemma4_dense_text_config() {
    let spec = GemmaModelSpec::from_config_json(&dense_gemma4_config())
        .expect("representative Gemma 4 dense config parses");

    assert_eq!(spec.family, ModelFamily::Gemma);
    assert_eq!(spec.architecture, "Gemma4ForConditionalGeneration");
    assert_eq!(spec.model_type, "gemma4");
    assert_eq!(spec.text_model_type, "gemma4_text");
    assert_eq!(spec.hidden_size, 1536);
    assert_eq!(spec.hidden_size_per_layer_input, 256);
    assert_eq!(spec.num_hidden_layers, 2);
    assert_eq!(spec.num_kv_shared_layers, 1);
    assert_eq!(spec.max_position_embeddings, 131_072);
    assert_eq!(
        spec.layer_kinds,
        vec![
            GemmaAttentionKind::SlidingAttention,
            GemmaAttentionKind::FullAttention
        ]
    );
    assert!(spec.uses_per_layer_input());
    assert!(!spec.is_kv_shared_layer(0));
    assert!(spec.is_kv_shared_layer(1));
    assert!(spec.requires_value_projection(0));
    assert!(!spec.requires_key_value_projection(1));
}

#[test]
fn parses_gemma4_text_only_config_with_model_root_tensors() {
    let spec = GemmaModelSpec::from_config_json(&gemma4_text_only_config())
        .expect("Gemma 4 text-only config parses");

    assert_eq!(spec.family, ModelFamily::Gemma);
    assert_eq!(spec.architecture, "Gemma4TextForCausalLM");
    assert_eq!(spec.model_type, "gemma4_text");
    assert_eq!(spec.text_model_type, "gemma4_text");
    assert_eq!(spec.tensor_root(), "model");
    assert_eq!(spec.embed_tokens_weight(), "model.embed_tokens.weight");
    assert_eq!(
        spec.layer_tensor(0, "layer_scalar"),
        "model.layers.0.layer_scalar"
    );
    assert_eq!(
        spec.self_attn_tensor(0, "q_proj.weight"),
        "model.layers.0.self_attn.q_proj.weight"
    );
    assert!(spec.uses_per_layer_input());
}

#[test]
fn validates_gemma4_text_only_index_with_model_root_tensors() {
    let spec = GemmaModelSpec::from_config_json(&gemma4_text_only_config())
        .expect("Gemma 4 text-only config parses");
    let index = SafetensorsIndex::from_json(
        &serde_json::json!({
            "metadata": {"total_size": 1},
            "weight_map": text_only_gemma4_text_weight_map()
        })
        .to_string(),
    )
    .expect("index parses");

    index
        .validate_gemma4_text_weights(&spec)
        .expect("Gemma 4 text-only tensors validate");
}

#[test]
fn validates_gemma4_dense_text_index_without_requiring_multimodal_tensors() {
    let spec = GemmaModelSpec::from_config_json(&dense_gemma4_config_without_ple())
        .expect("representative Gemma 4 dense config parses");
    let index = SafetensorsIndex::from_json(
        &serde_json::json!({
            "metadata": {"total_size": 1},
            "weight_map": dense_gemma4_text_weight_map()
        })
        .to_string(),
    )
    .expect("index parses");

    assert!(index.contains("model.embed_vision.embedding_projection.weight"));
    assert!(!index.contains("model.vision_tower.patch_embedder.input_proj.weight"));
    index
        .validate_gemma4_text_weights(&spec)
        .expect("Gemma 4 text tensors validate without requiring vision tensors");
    NativeTextModelSpec::Gemma(spec)
        .validate_text_weights(&index)
        .expect("generic native text validation routes to Gemma weights");
}

#[test]
fn validates_gemma4_moe_text_index() {
    let spec = GemmaModelSpec::from_config_json(&moe_gemma4_config())
        .expect("representative Gemma 4 MoE config parses");
    let index = SafetensorsIndex::from_json(
        &serde_json::json!({
            "metadata": {"total_size": 1},
            "weight_map": moe_gemma4_text_weight_map()
        })
        .to_string(),
    )
    .expect("index parses");

    assert!(spec.uses_moe());
    index
        .validate_gemma4_text_weights(&spec)
        .expect("Gemma 4 MoE text tensors validate");
}

#[test]
fn rejects_gemma4_index_missing_required_text_tensor() {
    let spec = GemmaModelSpec::from_config_json(&dense_gemma4_config_without_ple())
        .expect("representative Gemma 4 dense config parses");
    let mut weight_map = dense_gemma4_text_weight_map();
    weight_map
        .as_object_mut()
        .expect("weight map object")
        .remove("model.language_model.layers.0.self_attn.k_proj.weight");
    let index = SafetensorsIndex::from_json(
        &serde_json::json!({
            "metadata": {"total_size": 1},
            "weight_map": weight_map
        })
        .to_string(),
    )
    .expect("index parses");

    let err = index
        .validate_gemma4_text_weights(&spec)
        .expect_err("missing Gemma text tensor fails validation");

    assert_eq!(err.code(), "invalid_request");
    assert!(
        err.to_string()
            .contains("model.language_model.layers.0.self_attn.k_proj.weight")
    );
}

#[test]
fn rejects_multimodal_gemma4_config_without_text_config() {
    let err = GemmaModelSpec::from_config_json(
        r#"{
          "architectures": ["Gemma4ForConditionalGeneration"],
          "model_type": "gemma4",
          "vision_config": {"model_type": "gemma4_vision"}
        }"#,
    )
    .expect_err("Gemma 4 multimodal config without text_config fails closed");

    assert_eq!(err.code(), "unsupported_capability");
    assert!(err.to_string().contains("text_config"));
}

#[test]
fn rejects_gemma4_kv_sharing_beyond_layer_count() {
    let err = GemmaModelSpec::from_config_json(&gemma4_config(json!({
        "attention_k_eq_v": false,
        "enable_moe_block": false,
        "hidden_size": 1536,
        "hidden_size_per_layer_input": 0,
        "intermediate_size": 6144,
        "layer_types": ["sliding_attention", "full_attention"],
        "max_position_embeddings": 131072,
        "num_attention_heads": 8,
        "num_global_key_value_heads": null,
        "num_hidden_layers": 2,
        "num_key_value_heads": 1,
        "num_kv_shared_layers": 3,
        "sliding_window": 512,
        "use_double_wide_mlp": false
    })))
    .expect_err("invalid Gemma KV sharing range fails closed");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.to_string().contains("num_kv_shared_layers"));
}

fn dense_gemma4_config() -> String {
    gemma4_config(json!({
        "attention_k_eq_v": false,
        "enable_moe_block": false,
        "hidden_size": 1536,
        "hidden_size_per_layer_input": 256,
        "intermediate_size": 6144,
        "layer_types": ["sliding_attention", "full_attention"],
        "max_position_embeddings": 131072,
        "num_attention_heads": 8,
        "num_global_key_value_heads": null,
        "num_hidden_layers": 2,
        "num_key_value_heads": 1,
        "num_kv_shared_layers": 1,
        "sliding_window": 512,
        "use_double_wide_mlp": true
    }))
}

fn dense_gemma4_config_without_ple() -> String {
    gemma4_config(json!({
        "attention_k_eq_v": true,
        "enable_moe_block": false,
        "hidden_size": 5376,
        "hidden_size_per_layer_input": 0,
        "intermediate_size": 21504,
        "layer_types": ["sliding_attention", "full_attention"],
        "max_position_embeddings": 262144,
        "num_attention_heads": 32,
        "num_global_key_value_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 16,
        "num_kv_shared_layers": 0,
        "sliding_window": 1024,
        "use_double_wide_mlp": false
    }))
}

fn moe_gemma4_config() -> String {
    gemma4_config(json!({
        "attention_k_eq_v": true,
        "enable_moe_block": true,
        "hidden_size": 2816,
        "hidden_size_per_layer_input": 0,
        "intermediate_size": 2112,
        "layer_types": ["sliding_attention"],
        "max_position_embeddings": 262144,
        "moe_intermediate_size": 704,
        "num_attention_heads": 16,
        "num_experts": 128,
        "num_global_key_value_heads": 2,
        "num_hidden_layers": 1,
        "num_key_value_heads": 8,
        "num_kv_shared_layers": 0,
        "sliding_window": 1024,
        "top_k_experts": 8,
        "use_double_wide_mlp": false
    }))
}

fn gemma4_config(text_overrides: Value) -> String {
    let mut text_config = serde_json::Map::from_iter([
        ("attention_bias".to_owned(), json!(false)),
        ("attention_dropout".to_owned(), json!(0.0)),
        ("bos_token_id".to_owned(), json!(2)),
        ("dtype".to_owned(), json!("bfloat16")),
        ("final_logit_softcapping".to_owned(), json!(30.0)),
        ("global_head_dim".to_owned(), json!(512)),
        ("head_dim".to_owned(), json!(256)),
        ("hidden_activation".to_owned(), json!("gelu_pytorch_tanh")),
        ("model_type".to_owned(), json!("gemma4_text")),
        ("pad_token_id".to_owned(), json!(0)),
        ("rms_norm_eps".to_owned(), json!(1e-6)),
        (
            "rope_parameters".to_owned(),
            json!({
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional"
                },
                "sliding_attention": {
                    "rope_theta": 10000.0,
                    "rope_type": "default"
                }
            }),
        ),
        ("tie_word_embeddings".to_owned(), json!(true)),
        ("use_cache".to_owned(), json!(true)),
        ("vocab_size".to_owned(), json!(262144)),
        ("vocab_size_per_layer_input".to_owned(), json!(262144)),
    ]);
    text_config.extend(
        text_overrides
            .as_object()
            .expect("overrides object")
            .clone(),
    );
    serde_json::json!({
        "architectures": ["Gemma4ForConditionalGeneration"],
        "audio_config": null,
        "model_type": "gemma4",
        "text_config": Value::Object(text_config),
        "tie_word_embeddings": true,
        "vision_config": {"model_type": "gemma4_vision"}
    })
    .to_string()
}

fn gemma4_text_only_config() -> String {
    let mut value = serde_json::from_str::<Value>(&gemma4_config(json!({
        "attention_k_eq_v": false,
        "enable_moe_block": false,
        "hidden_size": 1536,
        "hidden_size_per_layer_input": 256,
        "intermediate_size": 6144,
        "layer_types": ["sliding_attention", "full_attention"],
        "max_position_embeddings": 131072,
        "num_attention_heads": 8,
        "num_global_key_value_heads": null,
        "num_hidden_layers": 2,
        "num_key_value_heads": 1,
        "num_kv_shared_layers": 1,
        "sliding_window": 512,
        "use_double_wide_mlp": true
    })))
    .expect("config json");
    value
        .as_object_mut()
        .expect("config object")
        .remove("text_config")
        .expect("text_config exists");
    value.as_object_mut().expect("config object").extend(
        serde_json::from_str::<Value>(&gemma4_config(json!({
            "attention_k_eq_v": false,
            "enable_moe_block": false,
            "hidden_size": 1536,
            "hidden_size_per_layer_input": 256,
            "intermediate_size": 6144,
            "layer_types": ["sliding_attention", "full_attention"],
            "max_position_embeddings": 131072,
            "num_attention_heads": 8,
            "num_global_key_value_heads": null,
            "num_hidden_layers": 2,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 1,
            "sliding_window": 512,
            "use_double_wide_mlp": true
        })))
        .expect("config json")["text_config"]
            .as_object()
            .expect("text config object")
            .clone(),
    );
    value["model_type"] = json!("gemma4_text");
    value
        .as_object_mut()
        .expect("config object")
        .remove("architectures");
    value.to_string()
}

fn dense_gemma4_text_weight_map() -> Value {
    json!({
        "model.embed_vision.embedding_projection.weight": "model.safetensors",
        "model.language_model.embed_tokens.weight": "model.safetensors",
        "model.language_model.norm.weight": "model.safetensors",
        "model.language_model.layers.0.input_layernorm.weight": "model.safetensors",
        "model.language_model.layers.0.layer_scalar": "model.safetensors",
        "model.language_model.layers.0.mlp.down_proj.weight": "model.safetensors",
        "model.language_model.layers.0.mlp.gate_proj.weight": "model.safetensors",
        "model.language_model.layers.0.mlp.up_proj.weight": "model.safetensors",
        "model.language_model.layers.0.post_attention_layernorm.weight": "model.safetensors",
        "model.language_model.layers.0.post_feedforward_layernorm.weight": "model.safetensors",
        "model.language_model.layers.0.pre_feedforward_layernorm.weight": "model.safetensors",
        "model.language_model.layers.0.self_attn.k_norm.weight": "model.safetensors",
        "model.language_model.layers.0.self_attn.k_proj.weight": "model.safetensors",
        "model.language_model.layers.0.self_attn.o_proj.weight": "model.safetensors",
        "model.language_model.layers.0.self_attn.q_norm.weight": "model.safetensors",
        "model.language_model.layers.0.self_attn.q_proj.weight": "model.safetensors",
        "model.language_model.layers.0.self_attn.v_proj.weight": "model.safetensors",
        "model.language_model.layers.1.input_layernorm.weight": "model.safetensors",
        "model.language_model.layers.1.layer_scalar": "model.safetensors",
        "model.language_model.layers.1.mlp.down_proj.weight": "model.safetensors",
        "model.language_model.layers.1.mlp.gate_proj.weight": "model.safetensors",
        "model.language_model.layers.1.mlp.up_proj.weight": "model.safetensors",
        "model.language_model.layers.1.post_attention_layernorm.weight": "model.safetensors",
        "model.language_model.layers.1.post_feedforward_layernorm.weight": "model.safetensors",
        "model.language_model.layers.1.pre_feedforward_layernorm.weight": "model.safetensors",
        "model.language_model.layers.1.self_attn.k_norm.weight": "model.safetensors",
        "model.language_model.layers.1.self_attn.k_proj.weight": "model.safetensors",
        "model.language_model.layers.1.self_attn.o_proj.weight": "model.safetensors",
        "model.language_model.layers.1.self_attn.q_norm.weight": "model.safetensors",
        "model.language_model.layers.1.self_attn.q_proj.weight": "model.safetensors"
    })
}

fn text_only_gemma4_text_weight_map() -> Value {
    let mut object = dense_gemma4_text_weight_map()
        .as_object()
        .expect("weight map object")
        .iter()
        .filter_map(|(key, value)| {
            key.strip_prefix("model.language_model.")
                .map(|suffix| (format!("model.{suffix}"), value.clone()))
        })
        .collect::<serde_json::Map<_, _>>();
    for tensor in [
        "model.embed_tokens_per_layer.weight",
        "model.per_layer_model_projection.weight",
        "model.per_layer_projection_norm.weight",
        "model.layers.0.per_layer_input_gate.weight",
        "model.layers.0.per_layer_projection.weight",
        "model.layers.0.post_per_layer_input_norm.weight",
        "model.layers.1.per_layer_input_gate.weight",
        "model.layers.1.per_layer_projection.weight",
        "model.layers.1.post_per_layer_input_norm.weight",
    ] {
        object.insert(tensor.to_owned(), json!("model.safetensors"));
    }
    Value::Object(object)
}

fn moe_gemma4_text_weight_map() -> Value {
    let mut weight_map = dense_gemma4_text_weight_map()
        .as_object()
        .expect("weight map object")
        .iter()
        .filter(|(key, _)| !key.contains("layers.1."))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect::<serde_json::Map<_, _>>();
    weight_map.extend(serde_json::Map::from_iter([
        (
            "model.language_model.layers.0.experts.down_proj".to_owned(),
            json!("model.safetensors"),
        ),
        (
            "model.language_model.layers.0.experts.gate_up_proj".to_owned(),
            json!("model.safetensors"),
        ),
        (
            "model.language_model.layers.0.post_feedforward_layernorm_1.weight".to_owned(),
            json!("model.safetensors"),
        ),
        (
            "model.language_model.layers.0.post_feedforward_layernorm_2.weight".to_owned(),
            json!("model.safetensors"),
        ),
        (
            "model.language_model.layers.0.pre_feedforward_layernorm_2.weight".to_owned(),
            json!("model.safetensors"),
        ),
        (
            "model.language_model.layers.0.router.per_expert_scale".to_owned(),
            json!("model.safetensors"),
        ),
        (
            "model.language_model.layers.0.router.proj.weight".to_owned(),
            json!("model.safetensors"),
        ),
        (
            "model.language_model.layers.0.router.scale".to_owned(),
            json!("model.safetensors"),
        ),
    ]));
    Value::Object(weight_map)
}
