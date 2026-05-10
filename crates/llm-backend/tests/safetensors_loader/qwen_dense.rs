use super::*;

#[tokio::test]
async fn qwen3_dense_prefill_uses_model_namespace_and_dense_mlp() {
    let root = temp_snapshot_dir("qwen3-dense-prefill");
    std::fs::remove_dir_all(&root).ok();
    write_tiny_qwen3_dense_decoder_snapshot(&root);
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen3_dense_spec();
    let mut caches = caches_for_spec(&spec, 4);

    let hidden = qwen_prefill_sequence_with_cache(&store, &spec, &[0, 1], &mut caches)
        .await
        .expect("prefill");

    assert_eq!(hidden.len(), 2);
    assert_eq!(hidden[0].len(), 2);
    match &caches[0] {
        QwenLayerCache::Full(cache) => assert_eq!(cache.token_count(), 2),
        QwenLayerCache::Linear(_) => panic!("Qwen3 dense should use full attention cache"),
    }
    std::fs::remove_dir_all(root).ok();
}

#[tokio::test]
async fn qwen3_dense_prefill_rejects_wrong_down_proj_output_width() {
    let root = temp_snapshot_dir("qwen3-dense-bad-down-proj");
    std::fs::remove_dir_all(&root).ok();
    write_tiny_qwen3_dense_decoder_snapshot(&root);
    std::fs::write(
        root.join("down.safetensors"),
        tiny_safetensors_bf16("model.layers.0.mlp.down_proj.weight", &[1, 1], &[0.0]),
    )
    .expect("bad down tensor");
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen3_dense_spec();
    let mut caches = caches_for_spec(&spec, 4);

    let err = qwen_prefill_sequence_with_cache(&store, &spec, &[0, 1], &mut caches)
        .await
        .expect_err("bad down projection width must fail closed");

    assert!(
        err.to_string().contains("down output length"),
        "error should name down output length: {err}"
    );
    std::fs::remove_dir_all(root).ok();
}

#[tokio::test]
async fn qwen3_dense_final_norm_and_tied_lm_head_use_model_namespace() {
    let root = temp_snapshot_dir("qwen3-dense-lm-head");
    std::fs::remove_dir_all(&root).ok();
    write_tiny_qwen3_dense_lm_head_snapshot(&root);
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen3_dense_spec();

    let normalized = qwen_final_norm_for_spec(&store, &spec, &[1.0, 0.0])
        .await
        .expect("final norm uses model.norm");
    let top = qwen_lm_head_top_k_for_spec(&store, &spec, &normalized, 2, 2)
        .await
        .expect("tied top-k works");
    let logits = qwen_lm_head_logits_for_spec(&store, &spec, &normalized, 2)
        .await
        .expect("tied logits work");

    assert_close(&normalized, &[std::f32::consts::SQRT_2, 0.0], 1e-5);
    assert_eq!(top[0].index, 0);
    assert_eq!(logits.len(), 2);
    std::fs::remove_dir_all(root).ok();
}
