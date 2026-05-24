use llm_kv_cache::{KvCacheConfig, KvCacheFormat, LayerKvCache};

fn assert_close_slice(expected: &[f32], actual: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        assert!(
            (expected - actual).abs() <= tolerance,
            "index {index}: expected {actual} to be within {tolerance} of {expected}"
        );
    }
}

#[test]
fn int8_layer_kv_cache_appends_and_reports_compressed_metrics() {
    let mut cache = LayerKvCache::new_with_config(2, 1, 16, KvCacheConfig::int8())
        .expect("int8 cache shape is valid");

    let key0 = (0..16).map(|value| value as f32 / 16.0).collect::<Vec<_>>();
    let value0 = (0..16)
        .map(|value| (value as f32 - 8.0) / 8.0)
        .collect::<Vec<_>>();
    let key1 = (16..32)
        .map(|value| value as f32 / 16.0)
        .collect::<Vec<_>>();
    let value1 = (16..32)
        .map(|value| (value as f32 - 24.0) / 8.0)
        .collect::<Vec<_>>();

    assert_eq!(cache.append(&key0, &value0).expect("first token fits"), 0);
    assert_eq!(cache.append(&key1, &value1).expect("second token fits"), 1);

    assert_eq!(cache.format(), KvCacheFormat::Int8);
    assert_eq!(cache.keys(), [key0.as_slice(), key1.as_slice()].concat());
    assert_eq!(
        cache.values(),
        [value0.as_slice(), value1.as_slice()].concat()
    );

    let decoded_keys = cache
        .int8_dequantized_keys()
        .expect("int8 keys dequantize")
        .expect("int8 key sidecar exists");
    let decoded_values = cache
        .int8_dequantized_values()
        .expect("int8 values dequantize")
        .expect("int8 value sidecar exists");
    assert_close_slice(cache.keys(), &decoded_keys, 0.01);
    assert_close_slice(cache.values(), &decoded_values, 0.01);

    let metrics = cache
        .format_metrics()
        .expect("format metrics are available");
    assert_eq!(metrics.active_format(), KvCacheFormat::Int8);
    assert_eq!(metrics.f32_uploaded_bytes(), 256);
    assert_eq!(metrics.f16_uploaded_bytes(), 128);
    assert!(metrics.int8_uploaded_bytes() < metrics.f16_uploaded_bytes());
    assert_eq!(metrics.phase3_value_bits(), None);
    assert_eq!(metrics.phase3_resident_bytes(), 0);
    assert!(metrics.f32_resident_bytes() > metrics.int8_resident_bytes());
    assert!(metrics.f16_resident_bytes() > metrics.int8_resident_bytes());
    assert!(metrics.int8_resident_bytes() > 0);
}

#[test]
fn int8_layer_kv_cache_survives_sliding_clone_snapshot_prefix_restore_and_identity_fork() {
    let config = KvCacheConfig::int8();
    let mut cache =
        LayerKvCache::new_with_config(2, 1, 2, config).expect("int8 cache shape is valid");

    cache
        .append_sliding(&[1.0, 2.0], &[0.5, 1.0])
        .expect("first token fits");
    let prefix_block_id = cache.block_ids()[0];
    cache
        .append_sliding(&[3.0, 4.0], &[1.5, 2.0])
        .expect("second token fits");
    cache
        .append_sliding(&[5.0, 6.0], &[2.5, 3.0])
        .expect("sliding append evicts oldest token");

    assert_eq!(cache.keys(), &[3.0, 4.0, 5.0, 6.0]);
    assert_eq!(cache.values(), &[1.5, 2.0, 2.5, 3.0]);
    let decoded_values = cache
        .int8_dequantized_values()
        .expect("int8 values dequantize")
        .expect("int8 sidecar exists");
    assert_close_slice(cache.values(), &decoded_values, 0.03);

    let mut clone = cache.clone();
    assert_ne!(clone.id(), cache.id());
    assert_eq!(clone.config(), config);
    assert_eq!(clone.keys(), cache.keys());
    assert_eq!(clone.values(), cache.values());

    clone
        .append_sliding(&[7.0, 8.0], &[3.5, 4.0])
        .expect("clone write succeeds");
    assert_ne!(
        clone.block_ids()[0],
        cache.block_ids()[0],
        "writing through a cloned int8 cache must fork block identity"
    );
    assert_ne!(
        clone.block_ids()[0],
        prefix_block_id,
        "the original prefix block identity must not be reused after a write"
    );

    let restored = LayerKvCache::from_snapshot(cache.snapshot()).expect("snapshot restores");
    assert_eq!(restored.config(), config);
    assert_eq!(restored.keys(), cache.keys());
    assert_eq!(restored.values(), cache.values());
    assert_close_slice(
        restored.values(),
        &restored
            .int8_dequantized_values()
            .expect("restored int8 values dequantize")
            .expect("restored int8 sidecar exists"),
        0.03,
    );

    let prefix_restored = LayerKvCache::from_prefix_cache_state(&cache.prefix_cache_state())
        .expect("prefix state restores");
    assert_eq!(prefix_restored.config(), config);
    assert_eq!(prefix_restored.keys(), cache.keys());
    assert_eq!(prefix_restored.values(), cache.values());
    assert_close_slice(
        prefix_restored.values(),
        &prefix_restored
            .int8_dequantized_values()
            .expect("prefix int8 values dequantize")
            .expect("prefix int8 sidecar exists"),
        0.03,
    );
}
