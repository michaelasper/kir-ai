use llm_models::{
    BackendKind, DeepSeekFamilyAdapter, GemmaFamilyAdapter, LlamaFamilyAdapter, ModelFamily,
    ModelFamilyAdapter, NativeTextModelSpec, PromotionStage, QwenFamilyAdapter,
};

#[test]
fn model_family_parses_aliases_to_canonical_slugs_and_adapters() {
    let qwen = ModelFamily::parse_slug("qwen").expect("qwen parses");
    let deepseek = ModelFamily::parse_slug("deepseek").expect("deepseek alias parses");
    let deep_seek = ModelFamily::parse_slug("deep_seek").expect("deep_seek parses");
    let gemma = ModelFamily::parse_slug("gemma").expect("gemma parses");
    let llama = ModelFamily::parse_slug("llama").expect("llama parses");

    assert_eq!(qwen, ModelFamily::Qwen);
    assert_eq!(deepseek, ModelFamily::DeepSeek);
    assert_eq!(deep_seek, ModelFamily::DeepSeek);
    assert_eq!(gemma, ModelFamily::Gemma);
    assert_eq!(llama, ModelFamily::Llama);
    assert_eq!(deepseek.canonical_slug(), "deep_seek");
    assert_eq!(deepseek.adapter().tensor_namespace(), "deepseek");
    assert_eq!(llama.adapter().tensor_namespace(), "llama");
}

#[test]
fn backend_kind_parses_manifest_and_cli_loader_slugs() {
    let native = BackendKind::parse_slug("native-metal").expect("hyphenated native loader parses");
    let native_alias =
        BackendKind::parse_slug("native_metal").expect("snake-case native loader parses");
    let mlx = BackendKind::parse_slug("mlx").expect("mlx loader parses");

    assert_eq!(native, BackendKind::NativeMetal);
    assert_eq!(native_alias, BackendKind::NativeMetal);
    assert_eq!(native.canonical_slug(), "native-metal");
    assert_eq!(mlx, BackendKind::Mlx);
    assert_eq!(mlx.canonical_slug(), "mlx");
}

#[test]
fn backend_kind_serializes_to_manifest_loader_slugs() {
    assert_eq!(
        serde_json::to_value(BackendKind::NativeMetal).expect("serialize native backend"),
        serde_json::json!("native-metal")
    );
    assert_eq!(
        serde_json::from_value::<BackendKind>(serde_json::json!("native-metal"))
            .expect("deserialize native manifest slug"),
        BackendKind::NativeMetal
    );
    assert_eq!(
        serde_json::from_value::<BackendKind>(serde_json::json!("native_metal"))
            .expect("deserialize legacy snake-case alias"),
        BackendKind::NativeMetal
    );
}

#[test]
fn qwen_family_declares_mlx_as_required_production_backend() {
    let adapter = QwenFamilyAdapter;
    let backends = adapter.production_backends();

    assert_eq!(adapter.family(), ModelFamily::Qwen);
    assert!(backends.contains(&BackendKind::Mlx));
    assert!(backends.contains(&BackendKind::NativeMetal));
    assert_eq!(adapter.cache_template_id(), "chatml/qwen/v1");
    assert_eq!(adapter.tensor_namespace(), "qwen");
    assert_eq!(adapter.promotion_stage(), PromotionStage::Production);
    assert!(adapter.capabilities().backend_execution);
}

#[test]
fn native_text_model_spec_routes_qwen_config_through_family_contract() {
    let config = r#"{
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
        }"#;
    let spec = NativeTextModelSpec::from_config_json(ModelFamily::Qwen, config)
        .expect("native Qwen text spec parses");
    let inferred = NativeTextModelSpec::infer_from_config_json(config)
        .expect("native Qwen text spec infers from config");

    assert_eq!(spec.family(), ModelFamily::Qwen);
    assert_eq!(inferred.family(), ModelFamily::Qwen);
    assert_eq!(spec.max_position_embeddings(), 40960);
    assert_eq!(spec.num_hidden_layers(), 2);
    assert_eq!(spec.hidden_size(), 1024);
    assert!(spec.is_qwen3_dense());
    assert!(spec.as_qwen().is_some());
}

#[test]
fn native_text_model_spec_rejects_families_without_native_tensor_support() {
    let err = NativeTextModelSpec::from_config_json(
        ModelFamily::DeepSeek,
        r#"{"architectures":["DeepseekV3ForCausalLM"],"model_type":"deepseek_v3"}"#,
    )
    .expect_err("deferred native text family fails closed");

    assert_eq!(err.code(), "unsupported_capability");
    assert!(err.to_string().contains("deep_seek"));
}

#[test]
fn deepseek_family_declares_mlx_production_backend() {
    let adapter = DeepSeekFamilyAdapter;
    let capabilities = adapter.capabilities();

    assert_eq!(adapter.family(), ModelFamily::DeepSeek);
    assert_eq!(adapter.production_backends(), &[BackendKind::Mlx]);
    assert_eq!(adapter.cache_template_id(), "deepseek/chat/v1");
    assert_eq!(adapter.promotion_stage(), PromotionStage::Production);
    assert!(capabilities.text);
    assert!(capabilities.reasoning);
    assert!(capabilities.tool_calls);
    assert!(capabilities.dsml_tools);
    assert!(capabilities.raw_completion);
    assert!(!capabilities.reasoning_channels);
    assert!(!capabilities.multimodal_artifacts);
    assert!(!capabilities.backend_execution);
}

#[test]
fn gemma_family_declares_native_and_mlx_text_backends() {
    let adapter = GemmaFamilyAdapter;
    let capabilities = adapter.capabilities();

    assert_eq!(adapter.family(), ModelFamily::Gemma);
    assert_eq!(
        adapter.production_backends(),
        &[BackendKind::NativeMetal, BackendKind::Mlx]
    );
    assert_eq!(adapter.cache_template_id(), "gemma/text-it/v1");
    assert_eq!(adapter.tensor_namespace(), "gemma4_text");
    assert_eq!(adapter.promotion_stage(), PromotionStage::Production);
    assert!(capabilities.text);
    assert!(capabilities.reasoning);
    assert!(capabilities.tool_calls);
    assert!(!capabilities.dsml_tools);
    assert!(capabilities.raw_completion);
    assert!(capabilities.reasoning_channels);
    assert!(!capabilities.multimodal_artifacts);
    assert!(capabilities.backend_execution);
}

#[test]
fn llama_family_declares_mlx_text_chat_contract() {
    let adapter = LlamaFamilyAdapter;
    let capabilities = adapter.capabilities();

    assert_eq!(adapter.family(), ModelFamily::Llama);
    assert_eq!(adapter.production_backends(), &[BackendKind::Mlx]);
    assert_eq!(adapter.cache_template_id(), "llama3/instruct/v1");
    assert_eq!(adapter.tensor_namespace(), "llama");
    assert_eq!(adapter.promotion_stage(), PromotionStage::Production);
    assert!(capabilities.text);
    assert!(!capabilities.reasoning);
    assert!(capabilities.tool_calls);
    assert!(!capabilities.dsml_tools);
    assert!(capabilities.raw_completion);
    assert!(!capabilities.reasoning_channels);
    assert!(!capabilities.multimodal_artifacts);
    assert!(!capabilities.backend_execution);
}

#[test]
fn backend_execution_capability_matches_native_text_spec_support() {
    let cases = [
        (ModelFamily::Qwen, native_qwen_config()),
        (
            ModelFamily::DeepSeek,
            r#"{"model_type":"deepseek_v3"}"#.to_owned(),
        ),
        (ModelFamily::Gemma, native_gemma_config()),
        (ModelFamily::Llama, r#"{"model_type":"llama"}"#.to_owned()),
    ];

    for (family, config) in cases {
        let supports_native_text = NativeTextModelSpec::from_config_json(family, &config).is_ok();

        assert_eq!(
            family.adapter().capabilities().backend_execution,
            supports_native_text,
            "{family} backend_execution capability must match native text spec support"
        );
    }
}

#[test]
fn promotion_stage_rejects_qwen_specific_deferral_slug() {
    let err =
        serde_json::from_value::<PromotionStage>(serde_json::json!("deferred_until_qwen_parity"))
            .expect_err("promotion stages must not expose qwen-specific deferral states");

    assert!(err.to_string().contains("unknown variant"));
}

fn native_qwen_config() -> String {
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
    }"#
    .to_owned()
}

fn native_gemma_config() -> String {
    serde_json::json!({
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "text_config": {
            "attention_k_eq_v": false,
            "enable_moe_block": false,
            "global_head_dim": 512,
            "head_dim": 256,
            "hidden_size": 1536,
            "hidden_size_per_layer_input": 256,
            "intermediate_size": 6144,
            "layer_types": ["sliding_attention", "full_attention"],
            "max_position_embeddings": 131072,
            "model_type": "gemma4_text",
            "num_attention_heads": 8,
            "num_global_key_value_heads": null,
            "num_hidden_layers": 2,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 1,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0
                },
                "sliding_attention": {
                    "rope_theta": 10000.0
                }
            },
            "sliding_window": 512,
            "use_double_wide_mlp": true,
            "vocab_size": 262144,
            "vocab_size_per_layer_input": 262144
        },
        "tie_word_embeddings": true
    })
    .to_string()
}
