use super::*;

#[tokio::test]
async fn qwen_attention_linear_decode_uses_in_place_matvec_cache_path() {
    let root = temp_snapshot_dir("qwen-attention-linear-decode-matvec-path");
    std::fs::remove_dir_all(&root).ok();
    write_tiny_linear_decoder_snapshot(&root);
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::LinearAttention);
    let mut expected_caches = qwen_layer_caches_for_spec(&spec, 4).expect("expected caches");
    let mut expected_scratch = InferenceScratchpad::default();
    let expected = qwen_prefill_sequence_with_cache(
        &store,
        &spec,
        &[0, 1, 0],
        &mut expected_caches,
        &CpuNativeMatvecBackend,
        &mut expected_scratch,
    )
    .await
    .expect("full cached sequence");

    let mut caches = qwen_layer_caches_for_spec(&spec, 4).expect("decode caches");
    let mut scratch = InferenceScratchpad::default();
    let matvec = RecordingMatvecBackend::default();
    qwen_prefill_sequence_with_cache(&store, &spec, &[0, 1], &mut caches, &matvec, &mut scratch)
        .await
        .expect("initial cached sequence");

    let decoded =
        qwen_decode_token_with_cache(&store, &spec, 0, &mut caches, &matvec, &mut scratch)
            .await
            .expect("decode token");

    assert_close(&decoded, &expected[2], 1e-5);
    let QwenLayerCache::Linear(cache) = &caches[0] else {
        panic!("linear cache");
    };
    let QwenLayerCache::Linear(expected_cache) = &expected_caches[0] else {
        panic!("expected linear cache");
    };
    assert_eq!(cache.token_count(), 3);
    assert_close(cache.conv_window(), expected_cache.conv_window(), 1e-5);
    assert_close(
        cache.recurrent_state(),
        expected_cache.recurrent_state(),
        1e-5,
    );
    assert_eq!(matvec.conv1d_calls(), 3);
    assert_eq!(matvec.recurrent_cache_update_calls(), 6);
    assert!(matvec.dense_f32_calls() >= 6);
    assert!(matvec.bf16_row_major_calls() >= 15);
    std::fs::remove_dir_all(root).ok();
}
