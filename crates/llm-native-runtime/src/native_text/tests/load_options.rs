use super::*;

#[test]
fn native_text_driver_clone_shares_inner_state() {
    let driver = driver_for_test(TestAdapter::new([1_usize]));
    let clone = driver.clone();

    assert!(driver.shares_inner_state_with(&clone));
}

#[test]
fn native_text_load_options_store_runtime_options_once() {
    let runtime = NativeTextRuntimeOptions {
        eager_materialize_shards: true,
        metal_weight_cache_bytes: Some(4096),
        prefix_cache_bytes: Some(17),
        prefix_disk_cache: None,
        warm_metal_weight_cache: true,
    };
    let options = NativeTextLoadOptions::with_runtime_options(runtime.clone());

    assert_eq!(options.runtime, runtime);
}

#[test]
fn family_load_options_use_shared_runtime_options() {
    #[cfg(feature = "native-qwen")]
    let _: NativeTextRuntimeOptions = crate::native_qwen::NativeQwenLoadOptions::default();
    #[cfg(feature = "native-gemma")]
    let _: NativeTextRuntimeOptions = crate::native_gemma::NativeGemmaLoadOptions::default();
}

#[test]
fn driver_with_zero_prefix_cache_budget_generates_without_reuse() {
    let adapter = TestAdapter::new([1_usize]).with_prefix_cache_bytes(0);
    let metrics = Arc::clone(&adapter.prefix_cache_metrics);
    let driver = driver_for_test(adapter);

    let first = driver
        .generate_blocking(driver_test_request(1), CancellationToken::new())
        .expect("first generation succeeds");
    let second = driver
        .generate_blocking(driver_test_request(1), CancellationToken::new())
        .expect("second generation succeeds");

    assert_eq!(first.text, "<1>");
    assert_eq!(second.text, "<1>");
    assert_eq!(first.prompt_cached_tokens, Some(0));
    assert_eq!(second.prompt_cached_tokens, Some(0));
    let snapshot = metrics.snapshot();
    assert_eq!(snapshot["stores"], 0);
    assert_eq!(snapshot["rejected"], 2);
    assert_eq!(snapshot["resident_bytes"], 0);
    assert_eq!(snapshot["resident_entries"], 0);
}
