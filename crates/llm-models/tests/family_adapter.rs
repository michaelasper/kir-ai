use llm_models::{
    BackendKind, DeepSeekFamilyAdapter, GemmaFamilyAdapter, ModelFamily, ModelFamilyAdapter,
    PromotionStage, QwenFamilyAdapter,
};

#[test]
fn model_family_parses_aliases_to_canonical_slugs_and_adapters() {
    let qwen = ModelFamily::parse_slug("qwen").expect("qwen parses");
    let deepseek = ModelFamily::parse_slug("deepseek").expect("deepseek alias parses");
    let deep_seek = ModelFamily::parse_slug("deep_seek").expect("deep_seek parses");
    let gemma = ModelFamily::parse_slug("gemma").expect("gemma parses");

    assert_eq!(qwen, ModelFamily::Qwen);
    assert_eq!(deepseek, ModelFamily::DeepSeek);
    assert_eq!(deep_seek, ModelFamily::DeepSeek);
    assert_eq!(gemma, ModelFamily::Gemma);
    assert_eq!(deepseek.canonical_slug(), "deep_seek");
    assert_eq!(deepseek.adapter().tensor_namespace(), "deepseek_v4");
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
fn deepseek_family_is_deferred_until_qwen_parity() {
    let adapter = DeepSeekFamilyAdapter;
    let capabilities = adapter.capabilities();

    assert_eq!(adapter.family(), ModelFamily::DeepSeek);
    assert!(adapter.production_backends().is_empty());
    assert_eq!(
        adapter.promotion_stage(),
        PromotionStage::DeferredUntilQwenParity
    );
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
fn gemma_family_is_deferred_as_text_only_until_qwen_parity() {
    let adapter = GemmaFamilyAdapter;
    let capabilities = adapter.capabilities();

    assert_eq!(adapter.family(), ModelFamily::Gemma);
    assert!(adapter.production_backends().is_empty());
    assert_eq!(adapter.cache_template_id(), "gemma/text-it/v1");
    assert_eq!(adapter.tensor_namespace(), "gemma4_text");
    assert_eq!(
        adapter.promotion_stage(),
        PromotionStage::DeferredUntilQwenParity
    );
    assert!(capabilities.text);
    assert!(capabilities.reasoning);
    assert!(capabilities.tool_calls);
    assert!(!capabilities.dsml_tools);
    assert!(capabilities.raw_completion);
    assert!(capabilities.reasoning_channels);
    assert!(!capabilities.multimodal_artifacts);
    assert!(!capabilities.backend_execution);
}
