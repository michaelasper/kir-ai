use super::*;

#[test]
fn qwen_embedding_probe_reads_and_normalizes_token() {
    let root = temp_snapshot_dir("qwen-embed");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 20 },
            "weight_map": {
                "model.language_model.embed_tokens.weight": "embed.safetensors",
                "model.language_model.layers.0.input_layernorm.weight": "norm.safetensors"
            }
        })
        .to_string(),
    )
    .expect("index");
    std::fs::write(
        root.join("embed.safetensors"),
        tiny_safetensors_bf16(
            "model.language_model.embed_tokens.weight",
            &[2, 2],
            &[3.0, 4.0, 6.0, 8.0],
        ),
    )
    .expect("embedding shard");
    std::fs::write(
        root.join("norm.safetensors"),
        tiny_safetensors_bf16(
            "model.language_model.layers.0.input_layernorm.weight",
            &[2],
            &[0.0, 1.0],
        ),
    )
    .expect("norm shard");
    let store = SafeTensorShardStore::open(&root).expect("store opens");

    let probe = qwen_embedding_and_layer0_norm(&store, 1, 2, 0.0).expect("probe");

    assert_eq!(probe.embedding, vec![6.0, 8.0]);
    assert_close(&probe.normalized, &[0.84852815, 2.2627418], 1e-6);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_layer0_projection_probe_reads_bf16_matrices() {
    let root = temp_snapshot_dir("qwen-projections");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 80 },
            "weight_map": {
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": "qkv.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_z.weight": "z.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_b.weight": "b.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_a.weight": "a.safetensors"
            }
        })
        .to_string(),
    )
    .expect("index");
    std::fs::write(
        root.join("qkv.safetensors"),
        tiny_safetensors_bf16(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            &[2, 2],
            &[1.0, 0.0, 0.0, 1.0],
        ),
    )
    .expect("qkv");
    std::fs::write(
        root.join("z.safetensors"),
        tiny_safetensors_bf16(
            "model.language_model.layers.0.linear_attn.in_proj_z.weight",
            &[1, 2],
            &[1.0, 1.0],
        ),
    )
    .expect("z");
    std::fs::write(
        root.join("b.safetensors"),
        tiny_safetensors_bf16(
            "model.language_model.layers.0.linear_attn.in_proj_b.weight",
            &[1, 2],
            &[2.0, 0.0],
        ),
    )
    .expect("b");
    std::fs::write(
        root.join("a.safetensors"),
        tiny_safetensors_bf16(
            "model.language_model.layers.0.linear_attn.in_proj_a.weight",
            &[1, 2],
            &[0.0, 3.0],
        ),
    )
    .expect("a");
    let store = SafeTensorShardStore::open(&root).expect("store opens");

    let projections =
        qwen_layer0_linear_attention_projections(&store, &[4.0, 5.0]).expect("projections");

    assert_eq!(projections.qkv, vec![4.0, 5.0]);
    assert_eq!(projections.z, vec![9.0]);
    assert_eq!(projections.b, vec![8.0]);
    assert_eq!(projections.a, vec![15.0]);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_prefill_sequence_with_cache_updates_layer_cache() {
    let root = temp_snapshot_dir("qwen-prefill-cache");
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 256 },
            "weight_map": {
                "model.language_model.embed_tokens.weight": "embed.safetensors",
                "model.language_model.layers.0.input_layernorm.weight": "input_norm.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": "qkv.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_z.weight": "z.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_b.weight": "b.safetensors",
                "model.language_model.layers.0.linear_attn.in_proj_a.weight": "a.safetensors",
                "model.language_model.layers.0.linear_attn.dt_bias": "dt.safetensors",
                "model.language_model.layers.0.linear_attn.A_log": "a_log.safetensors",
                "model.language_model.layers.0.linear_attn.conv1d.weight": "conv.safetensors",
                "model.language_model.layers.0.linear_attn.norm.weight": "attn_norm.safetensors",
                "model.language_model.layers.0.linear_attn.out_proj.weight": "out.safetensors",
                "model.language_model.layers.0.post_attention_layernorm.weight": "post_norm.safetensors",
                "model.language_model.layers.0.mlp.gate.weight": "router.safetensors",
                "model.language_model.layers.0.mlp.experts.gate_up_proj": "experts_gate_up.safetensors",
                "model.language_model.layers.0.mlp.experts.down_proj": "experts_down.safetensors",
                "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight": "shared_gate.safetensors",
                "model.language_model.layers.0.mlp.shared_expert.up_proj.weight": "shared_up.safetensors",
                "model.language_model.layers.0.mlp.shared_expert.down_proj.weight": "shared_down.safetensors",
                "model.language_model.layers.0.mlp.shared_expert_gate.weight": "shared_expert_gate.safetensors"
            }
        })
        .to_string(),
    )
    .expect("index");
    for (filename, tensor, shape, values) in [
        (
            "embed.safetensors",
            "model.language_model.embed_tokens.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "input_norm.safetensors",
            "model.language_model.layers.0.input_layernorm.weight",
            vec![2],
            vec![0.0, 0.0],
        ),
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
            "attn_norm.safetensors",
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
        (
            "post_norm.safetensors",
            "model.language_model.layers.0.post_attention_layernorm.weight",
            vec![2],
            vec![0.0, 0.0],
        ),
        (
            "router.safetensors",
            "model.language_model.layers.0.mlp.gate.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "experts_gate_up.safetensors",
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            vec![2, 2],
            vec![0.0, 0.0, 0.0, 0.0],
        ),
        (
            "experts_down.safetensors",
            "model.language_model.layers.0.mlp.experts.down_proj",
            vec![2, 1],
            vec![0.0, 0.0],
        ),
        (
            "shared_gate.safetensors",
            "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "shared_up.safetensors",
            "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "shared_down.safetensors",
            "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
            vec![2, 1],
            vec![0.0, 0.0],
        ),
        (
            "shared_expert_gate.safetensors",
            "model.language_model.layers.0.mlp.shared_expert_gate.weight",
            vec![1, 2],
            vec![0.0, 0.0],
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
    let mut caches = qwen_layer_caches_for_spec(&spec, 2).expect("layer caches");

    let output = qwen_prefill_sequence_with_cache(&store, &spec, &[0, 1], &mut caches)
        .expect("cached prefill");
    let expected = qwen_prefill_sequence(&store, &spec, &[0, 1]).expect("uncached prefill");

    assert_eq!(output.len(), expected.len());
    assert_close(&output[0], &expected[0], 1e-5);
    assert_close(&output[1], &expected[1], 1e-5);
    match &caches[0] {
        QwenLayerCache::Linear(cache) => {
            assert_eq!(cache.token_count(), 2);
            assert!(cache.recurrent_state().iter().any(|value| *value != 0.0));
        }
        QwenLayerCache::Full(_) => panic!("layer 0 should be linear attention"),
    }
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_decode_token_with_cache_matches_cached_prefill_suffix() {
    let root = temp_snapshot_dir("qwen-decode-token-cache");
    write_tiny_linear_decoder_snapshot(&root);
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::LinearAttention);
    let mut expected_caches = qwen_layer_caches_for_spec(&spec, 3).expect("expected caches");
    let expected =
        qwen_prefill_sequence_with_cache(&store, &spec, &[0, 1, 0], &mut expected_caches)
            .expect("full cached prefill");
    let mut caches = qwen_layer_caches_for_spec(&spec, 3).expect("layer caches");
    qwen_prefill_sequence_with_cache(&store, &spec, &[0, 1], &mut caches)
        .expect("initial cached prefill");

    let output =
        qwen_decode_token_with_cache(&store, &spec, 0, &mut caches).expect("cached token decode");

    assert_close(&output, &expected[2], 1e-5);
    match &caches[0] {
        QwenLayerCache::Linear(cache) => assert_eq!(cache.token_count(), 3),
        QwenLayerCache::Full(_) => panic!("layer 0 should be linear attention"),
    }
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn native_text_dispatch_matches_direct_qwen_prefill_and_decode() {
    let root = temp_snapshot_dir("native-text-dispatch-qwen");
    write_tiny_linear_decoder_snapshot(&root);
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let qwen_spec = tiny_qwen_spec(AttentionKind::LinearAttention);
    let native_spec = NativeTextModelSpec::Qwen(qwen_spec.clone());

    let mut direct_caches = qwen_layer_caches_for_spec(&qwen_spec, 3).expect("direct caches");
    let direct_prefill =
        qwen_prefill_sequence_with_cache(&store, &qwen_spec, &[0, 1], &mut direct_caches)
            .expect("direct prefill");
    let direct_decode = qwen_decode_token_with_cache(&store, &qwen_spec, 0, &mut direct_caches)
        .expect("direct decode");

    let mut native_caches =
        native_layer_caches_for_spec(&native_spec, 3).expect("native text caches");
    assert!(matches!(&native_caches, NativeTextLayerCaches::Qwen(_)));
    let native_prefill =
        native_prefill_sequence_with_cache(&store, &native_spec, &[0, 1], &mut native_caches)
            .expect("native text prefill");
    let native_decode = native_decode_token_with_cache(&store, &native_spec, 0, &mut native_caches)
        .expect("native text decode");

    let mut ref_caches = qwen_layer_caches_for_spec(&qwen_spec, 3).expect("ref caches");
    let ref_prefill = native_prefill_sequence_with_cache_for_spec_ref_with_matvec(
        &store,
        (&qwen_spec).into(),
        &[0, 1],
        NativeTextLayerCachesMut::Qwen(&mut ref_caches),
        &CpuNativeMatvecBackend,
    )
    .expect("native text spec-ref prefill");
    let ref_decode = native_decode_token_with_cache_for_spec_ref_with_matvec(
        &store,
        (&qwen_spec).into(),
        0,
        NativeTextLayerCachesMut::Qwen(&mut ref_caches),
        &CpuNativeMatvecBackend,
    )
    .expect("native text spec-ref decode");

    assert_eq!(native_prefill.len(), direct_prefill.len());
    assert_close(&native_prefill[0], &direct_prefill[0], 1e-5);
    assert_close(&native_prefill[1], &direct_prefill[1], 1e-5);
    assert_close(&native_decode, &direct_decode, 1e-5);
    assert_eq!(ref_prefill.len(), direct_prefill.len());
    assert_close(&ref_prefill[0], &direct_prefill[0], 1e-5);
    assert_close(&ref_prefill[1], &direct_prefill[1], 1e-5);
    assert_close(&ref_decode, &direct_decode, 1e-5);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_prefill_sequence_with_cache_appends_to_existing_linear_cache_chunk() {
    let root = temp_snapshot_dir("qwen-linear-prefill-cache-chunk");
    write_tiny_linear_decoder_snapshot(&root);
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::LinearAttention);
    let mut expected_caches = qwen_layer_caches_for_spec(&spec, 3).expect("expected caches");
    let expected =
        qwen_prefill_sequence_with_cache(&store, &spec, &[0, 1, 0], &mut expected_caches)
            .expect("full cached prefill");
    let mut caches = qwen_layer_caches_for_spec(&spec, 3).expect("layer caches");
    qwen_prefill_sequence_with_cache(&store, &spec, &[0, 1], &mut caches)
        .expect("initial cached chunk");

    let output =
        qwen_prefill_sequence_with_cache(&store, &spec, &[0], &mut caches).expect("second chunk");

    assert_eq!(output.len(), 1);
    assert_close(&output[0], &expected[2], 1e-5);
    match &caches[0] {
        QwenLayerCache::Linear(cache) => assert_eq!(cache.token_count(), 3),
        QwenLayerCache::Full(_) => panic!("layer 0 should be linear attention"),
    }
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_prefill_and_decode_use_configured_matvec_backend() {
    let root = temp_snapshot_dir("qwen-custom-matvec-cache");
    write_tiny_linear_decoder_snapshot(&root);
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::LinearAttention);
    let matvec = RecordingMatvecBackend::default();
    let expected =
        qwen_prefill_sequence_with_cache(&store, &spec, &[0, 1], &mut caches_for_spec(&spec, 3))
            .expect("cpu cached prefill");
    let mut recording_caches = qwen_layer_caches_for_spec(&spec, 3).expect("recording caches");

    let output = qwen_prefill_sequence_with_cache_with_matvec(
        &store,
        &spec,
        &[0, 1],
        &mut recording_caches,
        &matvec,
    )
    .expect("recording cached prefill");
    let decoded =
        qwen_decode_token_with_cache_with_matvec(&store, &spec, 0, &mut recording_caches, &matvec)
            .expect("recording cached decode");

    assert_eq!(output.len(), expected.len());
    assert_close(&output[0], &expected[0], 1e-5);
    assert_close(&output[1], &expected[1], 1e-5);
    assert_eq!(decoded.len(), spec.hidden_size as usize);
    assert!(matvec.batched_bf16_calls.get() > 0);
    assert!(matvec.single_bf16_calls.get() > 0);
    assert!(matvec.dense_f32_calls.get() > 0);
    assert!(matvec.rms_norm_calls.get() > 0);
    std::fs::remove_dir_all(root).ok();
}
