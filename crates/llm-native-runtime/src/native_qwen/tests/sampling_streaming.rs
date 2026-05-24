use super::*;

#[tokio::test]
async fn native_qwen_greedy_returns_top_logit_even_when_it_decodes_to_whitespace() {
    let snapshot = temp_snapshot_dir("greedy-whitespace");
    std::fs::remove_dir_all(&snapshot).ok();
    std::fs::create_dir_all(&snapshot).expect("snapshot dir");
    copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));

    let norm_shape = [1_usize];
    let norm = [1.0_f32];
    let lm_head_shape = [221_usize, 1_usize];
    let mut lm_head = vec![0.0_f32; 221];
    lm_head[32] = 1.0;
    lm_head[220] = 2.0;
    let safetensors = tiny_multi_safetensors_bf16(&[
        (
            "model.language_model.norm.weight",
            &norm_shape,
            norm.as_slice(),
        ),
        ("lm_head.weight", &lm_head_shape, lm_head.as_slice()),
    ]);
    std::fs::write(snapshot.join("model.safetensors"), &safetensors)
        .expect("write greedy fixture shard");
    std::fs::write(
        snapshot.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": safetensors.len() },
            "weight_map": {
                "model.language_model.norm.weight": "model.safetensors",
                "lm_head.weight": "model.safetensors"
            }
        })
        .to_string(),
    )
    .expect("write greedy fixture index");

    let backend = native_qwen_test_backend(
        &snapshot,
        crate::DEFAULT_MODEL_ID,
        zero_layer_qwen_spec(1, 221),
        1,
        1,
        2,
        64,
    );

    let token_id = select_qwen_token(&backend, &[1.0], SamplingConfig::Greedy)
        .await
        .expect("greedy candidate");

    assert_eq!(token_id, 220);
    let decoded = backend
        .driver
        .tokenizer
        .decode(&[token_id as u32], false)
        .expect("candidate decodes");
    assert!(decoded.trim().is_empty());
    std::fs::remove_dir_all(snapshot).ok();
}

#[tokio::test]
async fn native_qwen_greedy_clamps_top_k_to_vocab_size() {
    let snapshot = temp_snapshot_dir("greedy-top-k-clamp");
    std::fs::remove_dir_all(&snapshot).ok();
    std::fs::create_dir_all(&snapshot).expect("snapshot dir");
    copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));

    let norm_shape = [1_usize];
    let norm = [1.0_f32];
    let lm_head_shape = [2_usize, 1_usize];
    let lm_head = [1.0_f32, 2.0];
    let safetensors = tiny_multi_safetensors_bf16(&[
        (
            "model.language_model.norm.weight",
            &norm_shape,
            norm.as_slice(),
        ),
        ("lm_head.weight", &lm_head_shape, lm_head.as_slice()),
    ]);
    std::fs::write(snapshot.join("model.safetensors"), &safetensors)
        .expect("write greedy top-k clamp fixture shard");
    std::fs::write(
        snapshot.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": safetensors.len() },
            "weight_map": {
                "model.language_model.norm.weight": "model.safetensors",
                "lm_head.weight": "model.safetensors"
            }
        })
        .to_string(),
    )
    .expect("write greedy top-k clamp fixture index");

    let backend = native_qwen_test_backend(
        &snapshot,
        crate::DEFAULT_MODEL_ID,
        zero_layer_qwen_spec(1, 2),
        1,
        1,
        99,
        64,
    );

    let token_id = select_qwen_token(&backend, &[1.0], SamplingConfig::Greedy)
        .await
        .expect("greedy candidate with clamped top_k");

    assert_eq!(token_id, 1);
    std::fs::remove_dir_all(snapshot).ok();
}

#[test]
fn native_stream_text_deltas_withhold_unstable_prefix_until_finish() {
    let mut deltas = NativeStreamTextDeltas::default();

    assert_eq!(deltas.observe("�".to_owned()).expect("observe"), None);
    assert_eq!(deltas.observe("é".to_owned()).expect("observe"), None);

    assert_eq!(
        deltas.finish("é".to_owned()).expect("finish"),
        Some("é".to_owned())
    );
}

#[test]
fn native_stream_text_deltas_emit_stable_prefix_with_one_token_delay() {
    let mut deltas = NativeStreamTextDeltas::default();

    assert_eq!(deltas.observe("a".to_owned()).expect("observe"), None);
    assert_eq!(
        deltas.observe("ab".to_owned()).expect("observe"),
        Some("a".to_owned())
    );
    assert_eq!(
        deltas.observe("abc".to_owned()).expect("observe"),
        Some("b".to_owned())
    );
    assert_eq!(
        deltas.finish("abc".to_owned()).expect("finish"),
        Some("c".to_owned())
    );
}

#[test]
fn native_stream_text_deltas_emit_common_prefix_when_pending_decode_shifts() {
    let mut deltas = NativeStreamTextDeltas::default();

    assert_eq!(deltas.observe("a".to_owned()).expect("observe"), None);
    assert_eq!(
        deltas.observe("abc".to_owned()).expect("observe"),
        Some("a".to_owned())
    );
    assert_eq!(
        deltas.observe("abd".to_owned()).expect("observe"),
        Some("b".to_owned())
    );
    assert_eq!(
        deltas.finish("abd".to_owned()).expect("finish"),
        Some("d".to_owned())
    );
}

#[test]
fn native_stream_text_deltas_emit_incremental_pieces_with_one_token_delay() {
    let mut deltas = NativeStreamTextDeltas::default();

    assert_eq!(deltas.observe_incremental("a".to_owned()), None);
    assert_eq!(
        deltas.observe_incremental("b".to_owned()),
        Some("a".to_owned())
    );
    assert_eq!(
        deltas.observe_incremental("c".to_owned()),
        Some("b".to_owned())
    );
    assert_eq!(deltas.finish_incremental(), Some("c".to_owned()));
}

#[test]
fn native_stream_text_deltas_fail_closed_after_emitted_prefix_changes() {
    let mut deltas = NativeStreamTextDeltas::default();

    assert_eq!(deltas.observe("a".to_owned()).expect("observe"), None);
    assert_eq!(
        deltas.observe("ab".to_owned()).expect("observe"),
        Some("a".to_owned())
    );

    let err = deltas
        .observe("xb".to_owned())
        .expect_err("emitted prefix mismatch fails closed");
    assert!(err.to_string().contains("non-prefix"));
}

#[test]
fn native_top_p_sampling_selects_full_vocab_token_from_draw() {
    let token_id = sample_token_id_with_draw(
        &[2.0, 1.0, 0.0],
        SamplingConfig::TopP {
            temperature: 1.0,
            top_p: 0.9,
        },
        0.8,
        "Qwen",
    )
    .expect("sampling succeeds");

    assert_eq!(token_id, 1);
}
