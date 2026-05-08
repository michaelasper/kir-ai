use llm_models::{BackendKind, ModelFamily, ModelFamilyAdapter, QwenFamilyAdapter};

#[test]
fn qwen_family_declares_mlx_as_required_production_backend() {
    let adapter = QwenFamilyAdapter;
    let backends = adapter.production_backends();

    assert_eq!(adapter.family(), ModelFamily::Qwen);
    assert!(backends.contains(&BackendKind::Mlx));
    assert!(backends.contains(&BackendKind::NativeMetal));
    assert_eq!(adapter.cache_template_id(), "chatml/qwen/v1");
    assert_eq!(adapter.tensor_namespace(), "qwen3_5_moe");
}
