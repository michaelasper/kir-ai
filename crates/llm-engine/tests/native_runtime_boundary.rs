#![cfg(all(feature = "native-qwen", feature = "native-gemma"))]

use llm_backend_contracts::ModelBackend;
use llm_models::ModelFamily;
use llm_native_runtime::{
    NativeTextBackend, NativeTextLoadOptions, NativeTextRuntimeOptions,
    native_text_metal_metrics_snapshot,
};

fn assert_model_backend<T: ModelBackend>() {}

#[test]
fn native_runtime_crate_exposes_backend_constructor_api() {
    assert_model_backend::<NativeTextBackend>();

    let options = NativeTextLoadOptions::with_runtime_options(NativeTextRuntimeOptions {
        eager_materialize_shards: true,
        metal_weight_cache_bytes: Some(4096),
        prefix_cache_bytes: Some(2048),
        prefix_disk_cache: None,
        warm_metal_weight_cache: true,
    })
    .with_family(ModelFamily::Qwen);

    assert_eq!(options.family, Some(ModelFamily::Qwen));

    let metrics = native_text_metal_metrics_snapshot();
    assert!(metrics.get("bf16_matrix_cache").is_some());
}
