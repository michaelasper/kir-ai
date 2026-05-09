use super::*;

#[test]
fn qwen_post_attention_norm_adds_residual_and_normalizes() {
    let root = temp_snapshot_dir("qwen-post-attn-norm");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 4 },
            "weight_map": {
                "model.language_model.layers.0.post_attention_layernorm.weight": "post_norm.safetensors"
            }
        })
        .to_string(),
    )
    .expect("index");
    std::fs::write(
        root.join("post_norm.safetensors"),
        tiny_safetensors_bf16(
            "model.language_model.layers.0.post_attention_layernorm.weight",
            &[2],
            &[0.0, 1.0],
        ),
    )
    .expect("post norm");
    let store = SafeTensorShardStore::open(&root).expect("store opens");

    let normalized = qwen_layer0_post_attention_norm(&store, &[3.0, 4.0], &[3.0, 4.0], 2, 0.0)
        .expect("post attention norm");

    assert_close(&normalized, &[0.84852815, 2.2627418], 1e-6);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_full_attention_first_token_requires_key_and_norm_weights() {
    let root = temp_snapshot_dir("qwen-full-attn-required");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 48 },
            "weight_map": {
                "model.language_model.layers.0.self_attn.q_proj.weight": "q.safetensors",
                "model.language_model.layers.0.self_attn.v_proj.weight": "v.safetensors",
                "model.language_model.layers.0.self_attn.o_proj.weight": "o.safetensors"
            }
        })
        .to_string(),
    )
    .expect("index");
    std::fs::write(
        root.join("q.safetensors"),
        tiny_safetensors_bf16(
            "model.language_model.layers.0.self_attn.q_proj.weight",
            &[4, 2],
            &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ),
    )
    .expect("q");
    std::fs::write(
        root.join("v.safetensors"),
        tiny_safetensors_bf16(
            "model.language_model.layers.0.self_attn.v_proj.weight",
            &[2, 2],
            &[1.0, 0.0, 0.0, 1.0],
        ),
    )
    .expect("v");
    std::fs::write(
        root.join("o.safetensors"),
        tiny_safetensors_bf16(
            "model.language_model.layers.0.self_attn.o_proj.weight",
            &[2, 2],
            &[1.0, 0.0, 0.0, 1.0],
        ),
    )
    .expect("o");
    let store = SafeTensorShardStore::open(&root).expect("store opens");

    let err = qwen_layer_full_attention_first_token(
        &store,
        &tiny_qwen_spec(AttentionKind::FullAttention),
        0,
        &[1.0, 1.0],
    )
    .expect_err("full attention requires k_proj/q_norm/k_norm");

    assert_eq!(err.code(), "model_artifact_missing");
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_full_attention_sequence_with_cache_uses_indexed_weights() {
    let root = temp_snapshot_dir("qwen-full-attn-cache");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 64 },
            "weight_map": {
                "model.language_model.layers.0.self_attn.q_proj.weight": "q.safetensors",
                "model.language_model.layers.0.self_attn.k_proj.weight": "k.safetensors",
                "model.language_model.layers.0.self_attn.v_proj.weight": "v.safetensors",
                "model.language_model.layers.0.self_attn.q_norm.weight": "q_norm.safetensors",
                "model.language_model.layers.0.self_attn.k_norm.weight": "k_norm.safetensors",
                "model.language_model.layers.0.self_attn.o_proj.weight": "o.safetensors"
            }
        })
        .to_string(),
    )
    .expect("index");
    for (filename, tensor, shape, values) in [
        (
            "q.safetensors",
            "model.language_model.layers.0.self_attn.q_proj.weight",
            vec![4, 2],
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ),
        (
            "k.safetensors",
            "model.language_model.layers.0.self_attn.k_proj.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "v.safetensors",
            "model.language_model.layers.0.self_attn.v_proj.weight",
            vec![2, 2],
            vec![2.0, 0.0, 0.0, 4.0],
        ),
        (
            "q_norm.safetensors",
            "model.language_model.layers.0.self_attn.q_norm.weight",
            vec![2],
            vec![0.0, 0.0],
        ),
        (
            "k_norm.safetensors",
            "model.language_model.layers.0.self_attn.k_norm.weight",
            vec![2],
            vec![0.0, 0.0],
        ),
        (
            "o.safetensors",
            "model.language_model.layers.0.self_attn.o_proj.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
    ] {
        std::fs::write(
            root.join(filename),
            tiny_safetensors_bf16(tensor, &shape, &values),
        )
        .expect("tensor");
    }
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::FullAttention);
    let hidden_states = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let mut cache = LayerKvCache::new(2, 1, 2).expect("cache shape");

    let output =
        qwen_layer_full_attention_sequence_with_cache(&store, &spec, 0, &hidden_states, &mut cache)
            .expect("full attention sequence with cache");
    let expected = qwen_layer_full_attention_sequence(&store, &spec, 0, &hidden_states)
        .expect("full attention sequence");

    assert_eq!(cache.token_count(), 2);
    assert_close(&output[0], &expected[0], 1e-6);
    assert_close(&output[1], &expected[1], 1e-6);
    assert_close(cache.value(1).expect("value 1"), &[0.0, 4.0], 1e-6);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_full_attention_step_with_cache_uses_indexed_weights() {
    let root = temp_snapshot_dir("qwen-full-attn-step-cache");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 64 },
            "weight_map": {
                "model.language_model.layers.0.self_attn.q_proj.weight": "q.safetensors",
                "model.language_model.layers.0.self_attn.k_proj.weight": "k.safetensors",
                "model.language_model.layers.0.self_attn.v_proj.weight": "v.safetensors",
                "model.language_model.layers.0.self_attn.q_norm.weight": "q_norm.safetensors",
                "model.language_model.layers.0.self_attn.k_norm.weight": "k_norm.safetensors",
                "model.language_model.layers.0.self_attn.o_proj.weight": "o.safetensors"
            }
        })
        .to_string(),
    )
    .expect("index");
    for (filename, tensor, shape, values) in [
        (
            "q.safetensors",
            "model.language_model.layers.0.self_attn.q_proj.weight",
            vec![4, 2],
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ),
        (
            "k.safetensors",
            "model.language_model.layers.0.self_attn.k_proj.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "v.safetensors",
            "model.language_model.layers.0.self_attn.v_proj.weight",
            vec![2, 2],
            vec![2.0, 0.0, 0.0, 4.0],
        ),
        (
            "q_norm.safetensors",
            "model.language_model.layers.0.self_attn.q_norm.weight",
            vec![2],
            vec![0.0, 0.0],
        ),
        (
            "k_norm.safetensors",
            "model.language_model.layers.0.self_attn.k_norm.weight",
            vec![2],
            vec![0.0, 0.0],
        ),
        (
            "o.safetensors",
            "model.language_model.layers.0.self_attn.o_proj.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
    ] {
        std::fs::write(
            root.join(filename),
            tiny_safetensors_bf16(tensor, &shape, &values),
        )
        .expect("tensor");
    }
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::FullAttention);
    let hidden_states = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let mut expected_cache = LayerKvCache::new(3, 1, 2).expect("cache shape");
    let expected_output = qwen_layer_full_attention_sequence_with_cache(
        &store,
        &spec,
        0,
        &hidden_states,
        &mut expected_cache,
    )
    .expect("full cached sequence");
    let prefill = hidden_states[..2].to_vec();
    let mut cache = LayerKvCache::new(3, 1, 2).expect("cache shape");
    qwen_layer_full_attention_sequence_with_cache(&store, &spec, 0, &prefill, &mut cache)
        .expect("initial cached sequence");

    let output =
        qwen_layer_full_attention_step_with_cache(&store, &spec, 0, &hidden_states[2], &mut cache)
            .expect("full attention step");

    assert_close(&output, &expected_output[2], 1e-6);
    assert_eq!(cache.token_count(), 3);
    assert_close(cache.keys(), expected_cache.keys(), 1e-6);
    assert_close(cache.values(), expected_cache.values(), 1e-6);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_full_attention_sequence_with_cache_appends_to_existing_cache_chunk() {
    let root = temp_snapshot_dir("qwen-full-attn-cache-chunk");
    write_tiny_full_attention_snapshot(&root);
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::FullAttention);
    let hidden_states = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let mut expected_cache = LayerKvCache::new(3, 1, 2).expect("cache shape");
    let expected_output = qwen_layer_full_attention_sequence_with_cache(
        &store,
        &spec,
        0,
        &hidden_states,
        &mut expected_cache,
    )
    .expect("full cached sequence");
    let mut cache = LayerKvCache::new(3, 1, 2).expect("cache shape");
    qwen_layer_full_attention_sequence_with_cache(
        &store,
        &spec,
        0,
        &hidden_states[..2],
        &mut cache,
    )
    .expect("initial cached chunk");

    let output = qwen_layer_full_attention_sequence_with_cache(
        &store,
        &spec,
        0,
        &hidden_states[2..],
        &mut cache,
    )
    .expect("second cached chunk");

    assert_eq!(output.len(), 1);
    assert_close(&output[0], &expected_output[2], 1e-6);
    assert_eq!(cache.token_count(), 3);
    assert_close(cache.keys(), expected_cache.keys(), 1e-6);
    assert_close(cache.values(), expected_cache.values(), 1e-6);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_full_attention_normalization_uses_configured_matvec_backend() {
    let root = temp_snapshot_dir("qwen-full-attn-custom-norm");
    write_tiny_full_attention_snapshot(&root);
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::FullAttention);
    let hidden_states = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let mut expected_cache = LayerKvCache::new(3, 1, 2).expect("expected cache shape");
    let expected_output = qwen_layer_full_attention_sequence_with_cache(
        &store,
        &spec,
        0,
        &hidden_states,
        &mut expected_cache,
    )
    .expect("full cached sequence");
    let prefill = hidden_states[..2].to_vec();
    let mut cache = LayerKvCache::new(3, 1, 2).expect("cache shape");
    let matvec = RecordingMatvecBackend::default();

    let output = qwen_layer_full_attention_sequence_with_cache_with_matvec(
        &store, &spec, 0, &prefill, &mut cache, &matvec,
    )
    .expect("recording full cached sequence");
    let after_prefill_norm_calls = matvec.rms_norm_calls.get();
    let decoded = qwen_layer_full_attention_step_with_cache_with_matvec(
        &store,
        &spec,
        0,
        &hidden_states[2],
        &mut cache,
        &matvec,
    )
    .expect("recording full attention step");

    assert_close(&output[0], &expected_output[0], 1e-6);
    assert_close(&output[1], &expected_output[1], 1e-6);
    assert_close(&decoded, &expected_output[2], 1e-6);
    assert_eq!(after_prefill_norm_calls, 4);
    assert_eq!(matvec.rms_norm_calls.get(), 6);
    assert_close(cache.keys(), expected_cache.keys(), 1e-6);
    assert_close(cache.values(), expected_cache.values(), 1e-6);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_full_attention_softmax_uses_configured_matvec_backend() {
    let root = temp_snapshot_dir("qwen-full-attn-custom-softmax");
    write_tiny_full_attention_snapshot(&root);
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::FullAttention);
    let hidden_states = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let mut expected_cache = LayerKvCache::new(3, 1, 2).expect("expected cache shape");
    let expected_output = qwen_layer_full_attention_sequence_with_cache(
        &store,
        &spec,
        0,
        &hidden_states,
        &mut expected_cache,
    )
    .expect("full cached sequence");
    let prefill = hidden_states[..2].to_vec();
    let mut cache = LayerKvCache::new(3, 1, 2).expect("cache shape");
    let matvec = RecordingMatvecBackend::default();

    let output = qwen_layer_full_attention_sequence_with_cache_with_matvec(
        &store, &spec, 0, &prefill, &mut cache, &matvec,
    )
    .expect("recording full cached sequence");
    let after_prefill_softmax_calls = matvec.softmax_calls.get();
    let decoded = qwen_layer_full_attention_step_with_cache_with_matvec(
        &store,
        &spec,
        0,
        &hidden_states[2],
        &mut cache,
        &matvec,
    )
    .expect("recording full attention step");

    assert_close(&output[0], &expected_output[0], 1e-6);
    assert_close(&output[1], &expected_output[1], 1e-6);
    assert_close(&decoded, &expected_output[2], 1e-6);
    assert_eq!(after_prefill_softmax_calls, 2);
    assert_eq!(matvec.softmax_calls.get(), 3);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_full_attention_scores_use_configured_matvec_backend() {
    let root = temp_snapshot_dir("qwen-full-attn-custom-scores");
    write_tiny_full_attention_snapshot(&root);
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::FullAttention);
    let hidden_states = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let mut expected_cache = LayerKvCache::new(3, 1, 2).expect("expected cache shape");
    let expected_output = qwen_layer_full_attention_sequence_with_cache(
        &store,
        &spec,
        0,
        &hidden_states,
        &mut expected_cache,
    )
    .expect("full cached sequence");
    let prefill = hidden_states[..2].to_vec();
    let mut cache = LayerKvCache::new(3, 1, 2).expect("cache shape");
    let matvec = RecordingMatvecBackend::default();

    let output = qwen_layer_full_attention_sequence_with_cache_with_matvec(
        &store, &spec, 0, &prefill, &mut cache, &matvec,
    )
    .expect("recording full cached sequence");
    let after_prefill_dense_calls = matvec.dense_f32_calls.get();
    let decoded = qwen_layer_full_attention_step_with_cache_with_matvec(
        &store,
        &spec,
        0,
        &hidden_states[2],
        &mut cache,
        &matvec,
    )
    .expect("recording full attention step");

    assert_close(&output[0], &expected_output[0], 1e-6);
    assert_close(&output[1], &expected_output[1], 1e-6);
    assert_close(&decoded, &expected_output[2], 1e-6);
    assert_eq!(after_prefill_dense_calls, 4);
    assert_eq!(matvec.dense_f32_calls.get(), 6);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_full_attention_value_mix_uses_configured_matvec_backend() {
    let root = temp_snapshot_dir("qwen-full-attn-custom-value-mix");
    write_tiny_full_attention_snapshot(&root);
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::FullAttention);
    let hidden_states = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let mut expected_cache = LayerKvCache::new(3, 1, 2).expect("expected cache shape");
    let expected_output = qwen_layer_full_attention_sequence_with_cache(
        &store,
        &spec,
        0,
        &hidden_states,
        &mut expected_cache,
    )
    .expect("full cached sequence");
    let prefill = hidden_states[..2].to_vec();
    let mut cache = LayerKvCache::new(3, 1, 2).expect("cache shape");
    let matvec = RecordingMatvecBackend::default();

    let output = qwen_layer_full_attention_sequence_with_cache_with_matvec(
        &store, &spec, 0, &prefill, &mut cache, &matvec,
    )
    .expect("recording full cached sequence");
    let after_prefill_weighted_sum_calls = matvec.weighted_sum_calls.get();
    let decoded = qwen_layer_full_attention_step_with_cache_with_matvec(
        &store,
        &spec,
        0,
        &hidden_states[2],
        &mut cache,
        &matvec,
    )
    .expect("recording full attention step");

    assert_close(&output[0], &expected_output[0], 1e-6);
    assert_close(&output[1], &expected_output[1], 1e-6);
    assert_close(&decoded, &expected_output[2], 1e-6);
    assert_eq!(after_prefill_weighted_sum_calls, 2);
    assert_eq!(matvec.weighted_sum_calls.get(), 3);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_full_attention_cache_rows_use_configured_matvec_backend() {
    let root = temp_snapshot_dir("qwen-full-attn-cache-row-backend");
    write_tiny_full_attention_snapshot(&root);
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::FullAttention);
    let hidden_states = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let mut expected_cache = LayerKvCache::new(3, 1, 2).expect("expected cache shape");
    let expected_output = qwen_layer_full_attention_sequence_with_cache(
        &store,
        &spec,
        0,
        &hidden_states,
        &mut expected_cache,
    )
    .expect("full cached sequence");
    let prefill = hidden_states[..2].to_vec();
    let mut cache = LayerKvCache::new(3, 1, 2).expect("cache shape");
    let matvec = RecordingMatvecBackend::default();

    let output = qwen_layer_full_attention_sequence_with_cache_with_matvec(
        &store, &spec, 0, &prefill, &mut cache, &matvec,
    )
    .expect("recording full cached sequence");
    let after_prefill_head_row_calls = matvec.kv_cache_head_row_calls.get();
    let decoded = qwen_layer_full_attention_step_with_cache_with_matvec(
        &store,
        &spec,
        0,
        &hidden_states[2],
        &mut cache,
        &matvec,
    )
    .expect("recording full attention step");

    assert_close(&output[0], &expected_output[0], 1e-6);
    assert_close(&output[1], &expected_output[1], 1e-6);
    assert_close(&decoded, &expected_output[2], 1e-6);
    assert_eq!(after_prefill_head_row_calls, 4);
    assert_eq!(matvec.kv_cache_head_row_calls.get(), 6);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_linear_attention_first_token_requires_delta_parameters() {
    let root = temp_snapshot_dir("qwen-linear-delta-required");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 96 },
            "weight_map": {
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": "qkv.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_z.weight": "z.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_b.weight": "b.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_a.weight": "a.safetensors",
                "model.language_model.layers.0.linear_attn.conv1d.weight": "conv.safetensors",
                "model.language_model.layers.0.linear_attn.norm.weight": "norm.safetensors",
                "model.language_model.layers.0.linear_attn.out_proj.weight": "out.safetensors"
            }
        })
        .to_string(),
    )
    .expect("index");
    std::fs::write(
        root.join("qkv.safetensors"),
        tiny_safetensors_bf16(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            &[4, 2],
            &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0],
        ),
    )
    .expect("qkv");
    for (filename, tensor, shape, values) in [
        (
            "z.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_z.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "b.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_b.weight",
            vec![1, 2],
            vec![1.0, 0.0],
        ),
        (
            "a.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_a.weight",
            vec![1, 2],
            vec![1.0, 0.0],
        ),
        (
            "conv.safetensors",
            "model.language_model.layers.0.linear_attn.conv1d.weight",
            vec![4, 1],
            vec![1.0, 1.0, 1.0, 1.0],
        ),
        (
            "norm.safetensors",
            "model.language_model.layers.0.linear_attn.norm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "out.safetensors",
            "model.language_model.layers.0.linear_attn.out_proj.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
    ] {
        std::fs::write(
            root.join(filename),
            tiny_safetensors_bf16(tensor, &shape, &values),
        )
        .expect("tensor");
    }
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let projections =
        qwen_layer_linear_attention_projections(&store, 0, &[1.0, 1.0]).expect("projections");

    let err = qwen_layer_linear_attention_first_token(
        &store,
        &tiny_qwen_spec(AttentionKind::LinearAttention),
        0,
        &projections,
    )
    .expect_err("linear attention requires A_log and dt_bias");

    assert_eq!(err.code(), "model_artifact_missing");
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_linear_attention_sequence_with_cache_uses_indexed_weights() {
    let root = temp_snapshot_dir("qwen-linear-attn-cache");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 96 },
            "weight_map": {
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": "qkv.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_z.weight": "z.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_b.weight": "b.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_a.weight": "a.safetensors",
                "model.language_model.layers.0.linear_attn.dt_bias": "dt.safetensors",
                "model.language_model.layers.0.linear_attn.A_log": "a_log.safetensors",
                "model.language_model.layers.0.linear_attn.conv1d.weight": "conv.safetensors",
                "model.language_model.layers.0.linear_attn.norm.weight": "norm.safetensors",
                "model.language_model.layers.0.linear_attn.out_proj.weight": "out.safetensors"
            }
        })
        .to_string(),
    )
    .expect("index");
    for (filename, tensor, shape, values) in [
        (
            "qkv.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            vec![4, 2],
            vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 4.0],
        ),
        (
            "z.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_z.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "b.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_b.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "a.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_a.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "dt.safetensors",
            "model.language_model.layers.0.linear_attn.dt_bias",
            vec![1],
            vec![0.0],
        ),
        (
            "a_log.safetensors",
            "model.language_model.layers.0.linear_attn.A_log",
            vec![1],
            vec![0.0],
        ),
        (
            "conv.safetensors",
            "model.language_model.layers.0.linear_attn.conv1d.weight",
            vec![4, 1],
            vec![1.0, 1.0, 1.0, 1.0],
        ),
        (
            "norm.safetensors",
            "model.language_model.layers.0.linear_attn.norm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "out.safetensors",
            "model.language_model.layers.0.linear_attn.out_proj.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
    ] {
        std::fs::write(
            root.join(filename),
            tiny_safetensors_bf16(tensor, &shape, &values),
        )
        .expect("tensor");
    }
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::LinearAttention);
    let hidden_states = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let mut cache = LinearAttentionCache::new(1, 4, 1, 1, 2).expect("cache shape");

    let output = qwen_layer_linear_attention_sequence_with_cache(
        &store,
        &spec,
        0,
        &hidden_states,
        &mut cache,
    )
    .expect("linear attention sequence with cache");
    let expected = qwen_layer_linear_attention_sequence(&store, &spec, 0, &hidden_states)
        .expect("linear attention sequence");

    assert_eq!(cache.token_count(), 2);
    assert_close(&output[0], &expected[0], 1e-6);
    assert_close(&output[1], &expected[1], 1e-6);
    assert_close(cache.conv_window(), &[0.0, 1.0, 0.0, 4.0], 1e-6);
    assert!(cache.recurrent_state().iter().any(|value| *value != 0.0));
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_linear_attention_step_with_cache_uses_indexed_weights() {
    let root = temp_snapshot_dir("qwen-linear-attn-step-cache");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 96 },
            "weight_map": {
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": "qkv.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_z.weight": "z.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_b.weight": "b.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_a.weight": "a.safetensors",
                "model.language_model.layers.0.linear_attn.dt_bias": "dt.safetensors",
                "model.language_model.layers.0.linear_attn.A_log": "a_log.safetensors",
                "model.language_model.layers.0.linear_attn.conv1d.weight": "conv.safetensors",
                "model.language_model.layers.0.linear_attn.norm.weight": "norm.safetensors",
                "model.language_model.layers.0.linear_attn.out_proj.weight": "out.safetensors"
            }
        })
        .to_string(),
    )
    .expect("index");
    for (filename, tensor, shape, values) in [
        (
            "qkv.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            vec![4, 2],
            vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 4.0],
        ),
        (
            "z.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_z.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "b.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_b.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "a.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_a.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "dt.safetensors",
            "model.language_model.layers.0.linear_attn.dt_bias",
            vec![1],
            vec![0.0],
        ),
        (
            "a_log.safetensors",
            "model.language_model.layers.0.linear_attn.A_log",
            vec![1],
            vec![0.0],
        ),
        (
            "conv.safetensors",
            "model.language_model.layers.0.linear_attn.conv1d.weight",
            vec![4, 1],
            vec![1.0, 1.0, 1.0, 1.0],
        ),
        (
            "norm.safetensors",
            "model.language_model.layers.0.linear_attn.norm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "out.safetensors",
            "model.language_model.layers.0.linear_attn.out_proj.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
    ] {
        std::fs::write(
            root.join(filename),
            tiny_safetensors_bf16(tensor, &shape, &values),
        )
        .expect("tensor");
    }
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::LinearAttention);
    let hidden_states = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let mut expected_cache = LinearAttentionCache::new(1, 4, 1, 1, 2).expect("cache shape");
    let expected_output = qwen_layer_linear_attention_sequence_with_cache(
        &store,
        &spec,
        0,
        &hidden_states,
        &mut expected_cache,
    )
    .expect("full cached sequence");
    let prefill = hidden_states[..2].to_vec();
    let mut cache = LinearAttentionCache::new(1, 4, 1, 1, 2).expect("cache shape");
    qwen_layer_linear_attention_sequence_with_cache(&store, &spec, 0, &prefill, &mut cache)
        .expect("initial cached sequence");

    let output = qwen_layer_linear_attention_step_with_cache(
        &store,
        &spec,
        0,
        &hidden_states[2],
        &mut cache,
    )
    .expect("linear attention step");

    assert_close(&output, &expected_output[2], 1e-6);
    assert_eq!(cache.token_count(), 3);
    assert_close(cache.conv_window(), expected_cache.conv_window(), 1e-6);
    assert_close(
        cache.recurrent_state(),
        expected_cache.recurrent_state(),
        1e-6,
    );
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_linear_attention_normalization_uses_configured_matvec_backend() {
    let root = temp_snapshot_dir("qwen-linear-attn-custom-norm");
    write_tiny_linear_decoder_snapshot(&root);
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::LinearAttention);
    let hidden_states = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let mut expected_cache = LinearAttentionCache::new(1, 4, 1, 1, 2).expect("expected cache");
    let expected = qwen_layer_linear_attention_sequence_with_cache(
        &store,
        &spec,
        0,
        &hidden_states,
        &mut expected_cache,
    )
    .expect("cpu cached sequence");
    let mut cache = LinearAttentionCache::new(1, 4, 1, 1, 2).expect("recording cache");
    let matvec = RecordingMatvecBackend::default();

    let output = qwen_layer_linear_attention_sequence_with_cache_with_matvec(
        &store,
        &spec,
        0,
        &hidden_states,
        &mut cache,
        &matvec,
    )
    .expect("recording cached sequence");

    assert_close(&output[0], &expected[0], 1e-6);
    assert_close(&output[1], &expected[1], 1e-6);
    assert_eq!(matvec.rms_norm_calls.get(), 6);
    assert_close(cache.conv_window(), expected_cache.conv_window(), 1e-6);
    assert_close(
        cache.recurrent_state(),
        expected_cache.recurrent_state(),
        1e-6,
    );
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_linear_attention_recurrent_matvecs_use_configured_backend() {
    let root = temp_snapshot_dir("qwen-linear-attn-recurrent-matvecs");
    write_tiny_linear_decoder_snapshot(&root);
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::LinearAttention);
    let hidden_states = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let mut expected_cache = LinearAttentionCache::new(1, 4, 1, 1, 2).expect("expected cache");
    let expected = qwen_layer_linear_attention_sequence_with_cache(
        &store,
        &spec,
        0,
        &hidden_states,
        &mut expected_cache,
    )
    .expect("cpu cached sequence");
    let mut cache = LinearAttentionCache::new(1, 4, 1, 1, 2).expect("recording cache");
    let matvec = RecordingMatvecBackend::default();

    let output = qwen_layer_linear_attention_sequence_with_cache_with_matvec(
        &store,
        &spec,
        0,
        &hidden_states,
        &mut cache,
        &matvec,
    )
    .expect("recording cached sequence");

    assert_close(&output[0], &expected[0], 1e-6);
    assert_close(&output[1], &expected[1], 1e-6);
    assert_eq!(matvec.dense_f32_calls.get(), 6);
    assert_close(
        cache.recurrent_state(),
        expected_cache.recurrent_state(),
        1e-6,
    );
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_linear_attention_recurrent_decay_and_update_use_configured_backend() {
    let root = temp_snapshot_dir("qwen-linear-attn-recurrent-update");
    write_tiny_linear_decoder_snapshot(&root);
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::LinearAttention);
    let hidden_states = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let mut expected_cache = LinearAttentionCache::new(1, 4, 1, 1, 2).expect("expected cache");
    let expected = qwen_layer_linear_attention_sequence_with_cache(
        &store,
        &spec,
        0,
        &hidden_states,
        &mut expected_cache,
    )
    .expect("cpu cached sequence");
    let mut cache = LinearAttentionCache::new(1, 4, 1, 1, 2).expect("recording cache");
    let matvec = RecordingMatvecBackend::default();

    let output = qwen_layer_linear_attention_sequence_with_cache_with_matvec(
        &store,
        &spec,
        0,
        &hidden_states,
        &mut cache,
        &matvec,
    )
    .expect("recording cached sequence");

    assert_close(&output[0], &expected[0], 1e-6);
    assert_close(&output[1], &expected[1], 1e-6);
    assert_eq!(matvec.recurrent_cache_update_calls.get(), 4);
    assert_close(
        cache.recurrent_state(),
        expected_cache.recurrent_state(),
        1e-6,
    );
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_linear_attention_convolution_uses_configured_backend() {
    let root = temp_snapshot_dir("qwen-linear-attn-conv-backend");
    write_tiny_linear_decoder_snapshot(&root);
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::LinearAttention);
    let hidden_states = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let mut expected_cache = LinearAttentionCache::new(1, 4, 1, 1, 2).expect("expected cache");
    let expected = qwen_layer_linear_attention_sequence_with_cache(
        &store,
        &spec,
        0,
        &hidden_states,
        &mut expected_cache,
    )
    .expect("cpu cached sequence");
    let mut cache = LinearAttentionCache::new(1, 4, 1, 1, 2).expect("recording cache");
    let matvec = RecordingMatvecBackend::default();

    let output = qwen_layer_linear_attention_sequence_with_cache_with_matvec(
        &store,
        &spec,
        0,
        &hidden_states,
        &mut cache,
        &matvec,
    )
    .expect("recording cached sequence");

    assert_close(&output[0], &expected[0], 1e-6);
    assert_close(&output[1], &expected[1], 1e-6);
    assert_eq!(matvec.conv1d_calls.get(), 2);
    assert_close(cache.conv_window(), expected_cache.conv_window(), 1e-6);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_layer_caches_match_hybrid_attention_shapes() {
    let mut spec = tiny_qwen_spec(AttentionKind::LinearAttention);
    spec.num_hidden_layers = 2;
    spec.layer_kinds = vec![AttentionKind::LinearAttention, AttentionKind::FullAttention];

    let caches = qwen_layer_caches_for_spec(&spec, 4).expect("layer caches");

    assert_eq!(caches.len(), 2);
    match &caches[0] {
        QwenLayerCache::Linear(cache) => {
            assert_eq!(cache.conv_kernel_size(), 1);
            assert_eq!(cache.conv_dim(), 4);
            assert_eq!(cache.num_value_heads(), 1);
            assert_eq!(cache.key_head_dim(), 1);
            assert_eq!(cache.value_head_dim(), 2);
        }
        QwenLayerCache::Full(_) => panic!("layer 0 should be linear attention"),
    }
    match &caches[1] {
        QwenLayerCache::Full(cache) => {
            assert_eq!(cache.max_tokens(), 4);
            assert_eq!(cache.key_value_heads(), 1);
            assert_eq!(cache.head_dim(), 2);
        }
        QwenLayerCache::Linear(_) => panic!("layer 1 should be full attention"),
    }
}
