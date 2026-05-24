use llm_kv_cache::{
    AsymmetricVqCacheConfig, KvCacheConfig, KvCacheError, KvCacheFormat,
    KvCacheValueQuantizationBits, LayerKvCache,
};

fn assert_close_slice(expected: &[f32], actual: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        assert!(
            (expected - actual).abs() <= tolerance,
            "index {index}: expected {expected} to be within {tolerance} of {actual}"
        );
    }
}

#[test]
fn default_layer_kv_cache_uses_f32_format_without_phase3_sidecar() {
    let cache = LayerKvCache::new(2, 1, 4).expect("cache shape is valid");

    assert_eq!(cache.config(), KvCacheConfig::f32());
    assert_eq!(cache.format(), KvCacheFormat::F32);
    assert!(
        cache
            .phase3_dequantized_values()
            .expect("phase3 inspection succeeds")
            .is_none()
    );

    let metrics = cache
        .format_metrics()
        .expect("format metrics are available");
    assert_eq!(metrics.active_format(), KvCacheFormat::F32);
    assert_eq!(metrics.phase3_value_payload_bytes(), 0);
    assert_eq!(metrics.phase3_uploaded_bytes(), 0);
    assert_eq!(metrics.phase3_reconstruction_error(), None);
    assert_eq!(metrics.f32_uploaded_bytes(), 0);
    assert_eq!(metrics.f16_uploaded_bytes(), 0);
    assert_eq!(metrics.int8_uploaded_bytes(), 0);
    assert_eq!(metrics.f32_resident_bytes(), cache.resident_bytes());
}

#[test]
fn non_empty_default_layer_kv_cache_reports_no_phase3_metrics() {
    let mut cache = LayerKvCache::new(2, 1, 4).expect("cache shape is valid");

    cache
        .append(&[-0.25, 0.25, 0.75, 1.25], &[-1.25, -0.55, 0.31, 1.10])
        .expect("first token fits");
    cache
        .append(&[1.50, 1.75, 2.00, 2.25], &[-1.20, -0.52, 0.33, 1.13])
        .expect("second token fits");

    assert_eq!(cache.config(), KvCacheConfig::f32());
    assert_eq!(cache.format(), KvCacheFormat::F32);
    assert!(
        cache
            .phase3_dequantized_values()
            .expect("phase3 inspection succeeds")
            .is_none()
    );

    let metrics = cache
        .format_metrics()
        .expect("format metrics are available");
    assert_eq!(metrics.active_format(), KvCacheFormat::F32);
    assert_eq!(metrics.phase3_value_bits(), None);
    assert_eq!(metrics.phase3_value_payload_bytes(), 0);
    assert_eq!(metrics.phase3_value_metadata_bytes(), 0);
    assert_eq!(metrics.phase3_resident_bytes(), 0);
    assert_eq!(metrics.phase3_uploaded_bytes(), 0);
    assert_eq!(metrics.phase3_reconstruction_error(), None);
    assert_eq!(metrics.f32_uploaded_bytes(), 64);
    assert_eq!(metrics.f16_uploaded_bytes(), 32);
    assert_eq!(metrics.int8_uploaded_bytes(), 16);
    assert_eq!(metrics.f32_resident_bytes(), cache.resident_bytes());
}

#[test]
fn asymmetric_vq_int4_is_opt_in_and_round_trips_values_with_error_metrics() {
    let mut cache = LayerKvCache::new_with_config(
        2,
        1,
        4,
        KvCacheConfig::asymmetric_vq(AsymmetricVqCacheConfig::new(
            KvCacheValueQuantizationBits::Four,
        )),
    )
    .expect("phase3 cache shape is valid");

    cache
        .append(&[-0.25, 0.25, 0.75, 1.25], &[-1.25, -0.55, 0.31, 1.10])
        .expect("first token fits");
    cache
        .append(&[1.50, 1.75, 2.00, 2.25], &[-1.20, -0.52, 0.33, 1.13])
        .expect("second token fits");

    assert_eq!(cache.format(), KvCacheFormat::AsymmetricVq);
    assert_eq!(
        cache.keys(),
        &[-0.25, 0.25, 0.75, 1.25, 1.50, 1.75, 2.00, 2.25]
    );
    assert_eq!(
        cache.values(),
        &[-1.25, -0.55, 0.31, 1.10, -1.20, -0.52, 0.33, 1.13]
    );

    let decoded = cache
        .phase3_dequantized_values()
        .expect("phase3 dequantizes")
        .expect("phase3 sidecar exists");
    assert_close_slice(cache.values(), &decoded, 0.08);

    let metrics = cache
        .format_metrics()
        .expect("format metrics are available");
    let error = metrics
        .phase3_reconstruction_error()
        .expect("phase3 error metrics are present");
    assert_eq!(
        metrics.phase3_value_bits(),
        Some(KvCacheValueQuantizationBits::Four)
    );
    assert_eq!(metrics.f32_uploaded_bytes(), 64);
    assert_eq!(metrics.f16_uploaded_bytes(), 32);
    assert_eq!(metrics.int8_uploaded_bytes(), 16);
    assert_eq!(metrics.phase3_value_payload_bytes(), 4);
    assert!(metrics.phase3_value_metadata_bytes() > 0);
    assert!(metrics.phase3_resident_bytes() >= metrics.phase3_value_payload_bytes());
    assert!(metrics.phase3_uploaded_bytes() > metrics.phase3_value_payload_bytes());
    assert!(error.max_abs() <= 0.08);
    assert!(error.mse() < 0.0025);
}

#[test]
fn asymmetric_vq_survives_sliding_clone_snapshot_and_prefix_restore() {
    let config = KvCacheConfig::asymmetric_vq(AsymmetricVqCacheConfig::new(
        KvCacheValueQuantizationBits::Four,
    ));
    let mut cache =
        LayerKvCache::new_with_config(2, 1, 2, config).expect("phase3 cache shape is valid");

    cache
        .append_sliding(&[1.0, 2.0], &[10.0, 11.0])
        .expect("first token fits");
    cache
        .append_sliding(&[3.0, 4.0], &[30.0, 31.0])
        .expect("second token fits");
    cache
        .append_sliding(&[5.0, 6.0], &[50.0, 51.0])
        .expect("sliding append evicts oldest token");

    assert_eq!(cache.keys(), &[3.0, 4.0, 5.0, 6.0]);
    assert_eq!(cache.values(), &[30.0, 31.0, 50.0, 51.0]);
    assert_phase3_values_match(&cache);

    let clone = cache.clone();
    assert_eq!(clone.config(), config);
    assert_eq!(clone.values(), cache.values());
    assert_phase3_values_match(&clone);

    let restored = LayerKvCache::from_snapshot(cache.snapshot()).expect("phase3 snapshot restores");
    assert_eq!(restored.config(), config);
    assert_eq!(restored.keys(), cache.keys());
    assert_eq!(restored.values(), cache.values());
    assert_phase3_values_match(&restored);

    let prefix_restored = LayerKvCache::from_prefix_cache_state(&cache.prefix_cache_state())
        .expect("phase3 prefix state restores");
    assert_eq!(prefix_restored.config(), config);
    assert_eq!(prefix_restored.keys(), cache.keys());
    assert_eq!(prefix_restored.values(), cache.values());
    assert_phase3_values_match(&prefix_restored);
}

#[test]
fn f16_and_int8_formats_fail_closed_until_storage_paths_exist() {
    let f16 = LayerKvCache::new_with_config(2, 1, 2, KvCacheConfig::f16())
        .expect_err("f16 CPU cache storage is not implemented");
    assert_eq!(
        f16,
        KvCacheError::UnsupportedFormat {
            format: KvCacheFormat::F16
        }
    );

    let int8 = LayerKvCache::new_with_config(2, 1, 2, KvCacheConfig::int8())
        .expect_err("int8 CPU cache storage is not implemented");
    assert_eq!(
        int8,
        KvCacheError::UnsupportedFormat {
            format: KvCacheFormat::Int8
        }
    );
}

#[test]
fn asymmetric_vq_rejects_non_finite_values_before_mutating_cache() {
    let mut cache = LayerKvCache::new_with_config(
        2,
        1,
        2,
        KvCacheConfig::asymmetric_vq(AsymmetricVqCacheConfig::new(
            KvCacheValueQuantizationBits::Four,
        )),
    )
    .expect("phase3 cache shape is valid");

    let err = cache
        .append(&[1.0, 2.0], &[f32::NAN, 3.0])
        .expect_err("phase3 values must be finite");

    assert_eq!(err, KvCacheError::NonFiniteValue);
    assert_eq!(cache.token_count(), 0);
    assert_eq!(cache.keys(), &[]);
    assert_eq!(cache.values(), &[]);
}

fn assert_phase3_values_match(cache: &LayerKvCache) {
    let decoded = cache
        .phase3_dequantized_values()
        .expect("phase3 dequantizes")
        .expect("phase3 sidecar exists");
    assert_close_slice(cache.values(), &decoded, 0.08);
}
