use llm_models::{
    BackendKind, DeepSeekFamilyAdapter, ModelFamily, ModelFamilyAdapter, PromotionStage,
    QwenFamilyAdapter,
};

#[test]
fn qwen_family_declares_mlx_as_required_production_backend() {
    let adapter = QwenFamilyAdapter;
    let backends = adapter.production_backends();

    assert_eq!(adapter.family(), ModelFamily::Qwen);
    assert!(backends.contains(&BackendKind::Mlx));
    assert!(backends.contains(&BackendKind::NativeMetal));
    assert_eq!(adapter.cache_template_id(), "chatml/qwen/v1");
    assert_eq!(adapter.tensor_namespace(), "qwen3_5_moe");
    assert_eq!(adapter.promotion_stage(), PromotionStage::Production);
    assert!(adapter.capabilities().backend_execution);
}

#[test]
fn deepseek_family_is_deferred_until_qwen_parity() {
    let adapter = DeepSeekFamilyAdapter;
    let capabilities = adapter.capabilities();

    assert_eq!(adapter.family(), ModelFamily::DeepSeek);
    assert_eq!(adapter.production_backends(), &[BackendKind::Mlx]);
    assert_eq!(
        adapter.promotion_stage(),
        PromotionStage::DeferredUntilQwenParity
    );
    assert!(capabilities.text);
    assert!(capabilities.reasoning);
    assert!(capabilities.tool_calls);
    assert!(capabilities.dsml_tools);
    assert!(capabilities.raw_completion);
    assert!(!capabilities.backend_execution);
}
