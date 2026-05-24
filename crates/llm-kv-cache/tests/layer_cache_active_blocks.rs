use llm_kv_cache::LayerKvCache;

#[test]
fn active_blocks_report_physical_token_start_after_sliding_wrap() {
    let mut cache = LayerKvCache::new(3, 1, 1).expect("cache shape is valid");

    cache.append_sliding(&[1.0], &[10.0]).expect("token fits");
    cache.append_sliding(&[2.0], &[20.0]).expect("token fits");
    cache.append_sliding(&[3.0], &[30.0]).expect("token fits");
    cache
        .append_sliding(&[4.0], &[40.0])
        .expect("sliding append wraps the physical window");

    let active_blocks = cache.active_blocks().expect("active blocks are valid");

    assert_eq!(cache.keys(), &[2.0, 3.0, 4.0]);
    assert_eq!(active_blocks.len(), 2);
    assert_eq!(active_blocks[0].logical_token_start(), 0);
    assert_eq!(active_blocks[0].physical_token_start(), 1);
    assert_eq!(active_blocks[0].token_count(), 2);
    assert_eq!(active_blocks[1].logical_token_start(), 2);
    assert_eq!(active_blocks[1].physical_token_start(), 0);
    assert_eq!(active_blocks[1].token_count(), 1);
}
