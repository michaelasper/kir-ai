use super::*;

#[tokio::test]
async fn qwen_final_norm_and_lm_head_top_k_use_indexed_weights() {
    let root = temp_snapshot_dir("qwen-lm-head");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 16 },
            "weight_map": {
                "model.language_model.norm.weight": "norm.safetensors",
                "lm_head.weight": "lm_head.safetensors"
            }
        })
        .to_string(),
    )
    .expect("index");
    std::fs::write(
        root.join("norm.safetensors"),
        tiny_safetensors_bf16("model.language_model.norm.weight", &[2], &[0.0, 1.0]),
    )
    .expect("norm");
    std::fs::write(
        root.join("lm_head.safetensors"),
        tiny_safetensors_bf16(
            "lm_head.weight",
            &[2, 2],
            &[
                1.0, 0.0, //
                0.0, 1.0,
            ],
        ),
    )
    .expect("lm head");
    let store = SafeTensorShardStore::open(&root).expect("store opens");

    let normalized = qwen_final_norm(&store, &[3.0, 4.0], 2, 0.0).await.expect("final norm");
    let top = qwen_lm_head_top_k(&store, &normalized, 1, 1).await.expect("lm head");
    let logits = qwen_lm_head_logits(&store, &normalized, 1).await.expect("lm head logits");

    assert_close(&normalized, &[0.84852815, 2.2627418], 1e-6);
    assert_eq!(top[0].index, 1);
    assert_close(&[top[0].logit], &[2.2627418], 1e-6);
    assert_close(&logits, &[0.84852815, 2.2627418], 1e-6);
    std::fs::remove_dir_all(root).ok();
}

#[tokio::test]
async fn qwen_lm_head_uses_configured_matvec_backend() {
    let root = temp_snapshot_dir("qwen-lm-head-custom-matvec");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 12 },
            "weight_map": { "lm_head.weight": "lm_head.safetensors" }
        })
        .to_string(),
    )
    .expect("index");
    std::fs::write(
        root.join("lm_head.safetensors"),
        tiny_safetensors_bf16("lm_head.weight", &[3, 2], &[1.0, 0.0, 0.0, 2.0, -1.0, 1.0]),
    )
    .expect("lm head");
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let matvec = RecordingMatvecBackend::default();

    let top = qwen_lm_head_top_k_with_matvec(&store, &[1.0, 2.0], 2, 2, &matvec)
        .await
        .expect("top-k uses recording matvec");
    let logits = qwen_lm_head_logits_with_matvec(&store, &[1.0, 2.0], 2, &matvec)
        .await
        .expect("full logits use recording matvec");

    assert_eq!(top[0].index, 1);
    assert_eq!(top[0].logit, 4.0);
    assert_eq!(top[1].index, 0);
    assert_eq!(top[1].logit, 1.0);
    assert_eq!(logits, vec![1.0, 4.0, 1.0]);
    assert_eq!(matvec.top_k_bf16_calls.load(Ordering::Relaxed), 0);
    assert_eq!(matvec.rows_bf16_calls.load(Ordering::Relaxed), 1);
    std::fs::remove_dir_all(root).ok();
}

#[tokio::test]
async fn qwen_final_norm_uses_configured_rms_norm_backend() {
    let root = temp_snapshot_dir("qwen-final-norm-custom-matvec");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 4 },
            "weight_map": { QWEN_FINAL_NORM_WEIGHT: "norm.safetensors" }
        })
        .to_string(),
    )
    .expect("index");
    std::fs::write(
        root.join("norm.safetensors"),
        tiny_safetensors_bf16(QWEN_FINAL_NORM_WEIGHT, &[2], &[0.0, 1.0]),
    )
    .expect("norm");
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let matvec = RecordingMatvecBackend::default();
    let expected = qwen_final_norm(&store, &[3.0, 4.0], 2, 0.0).await.expect("cpu final norm");

    let output = qwen_final_norm_with_matvec(&store, &[3.0, 4.0], 2, 0.0, &matvec)
        .await
        .expect("final norm uses recording backend");

    assert_close(&output, &expected, 1e-6);
    assert_eq!(matvec.rms_norm_calls.load(Ordering::Relaxed), 1);
    std::fs::remove_dir_all(root).ok();
}
