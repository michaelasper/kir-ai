use super::*;

#[tokio::test]
async fn native_qwen_start_decode_session_prefills_full_context_with_bounded_cache() {
    let snapshot = temp_snapshot_dir("full-context-prefill");
    std::fs::remove_dir_all(&snapshot).ok();
    std::fs::create_dir_all(&snapshot).expect("snapshot dir");
    copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));
    write_tiny_linear_decoder_snapshot(&snapshot);
    let backend = native_qwen_test_backend(
        &snapshot,
        crate::DEFAULT_MODEL_ID,
        tiny_engine_qwen_spec(llm_models::AttentionKind::LinearAttention),
        8,
        16,
        2,
        64,
    );

    let decode = start_qwen_decode_session(
        &backend,
        &[0, 1, 0],
        8,
        &native_qwen_test_request(crate::DEFAULT_MODEL_ID),
        &CancellationToken::new(),
    )
    .await
    .expect("decode session starts");

    match &decode.caches[0] {
        QwenLayerCache::Linear(cache) => assert_eq!(cache.token_count(), 3),
        QwenLayerCache::Full(_) => panic!("layer 0 should be linear attention"),
        other => panic!("unexpected Qwen cache variant: {other:?}"),
    }
    std::fs::remove_dir_all(snapshot).ok();
}

#[test]
fn native_qwen_start_decode_session_reuses_shared_prefix_across_requests() {
    let snapshot = temp_snapshot_dir("shared-prefix-prefill");
    std::fs::remove_dir_all(&snapshot).ok();
    std::fs::create_dir_all(&snapshot).expect("snapshot dir");
    copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));
    write_tiny_linear_decoder_snapshot(&snapshot);
    let backend = native_qwen_test_backend(
        &snapshot,
        crate::DEFAULT_MODEL_ID,
        tiny_engine_qwen_spec(llm_models::AttentionKind::LinearAttention),
        8,
        1,
        2,
        64,
    );
    let request = native_qwen_test_request(crate::DEFAULT_MODEL_ID);
    let mut top_p_request = request.clone();
    top_p_request.sampling = SamplingConfig::TopP {
        temperature: 0.2,
        top_p: 0.9,
    };
    let before_hits = native_prefix_metric_counter("hits");
    let rt = test_runtime();

    let first = rt
        .block_on(start_qwen_decode_session(
            &backend,
            &[0, 1],
            8,
            &request,
            &CancellationToken::new(),
        ))
        .expect("first decode session starts");
    drop(first);
    let second = rt
        .block_on(start_qwen_decode_session(
            &backend,
            &[0, 1, 0],
            8,
            &top_p_request,
            &CancellationToken::new(),
        ))
        .expect("second decode session starts");

    assert!(
        native_prefix_metric_counter("hits") > before_hits,
        "second request should hit the shared prefix cache"
    );
    match &second.caches[0] {
        QwenLayerCache::Linear(cache) => assert_eq!(cache.token_count(), 3),
        QwenLayerCache::Full(_) => panic!("layer 0 should be linear attention"),
        other => panic!("unexpected Qwen cache variant: {other:?}"),
    }

    let mut expected_caches = qwen_layer_caches_for_spec(
        &backend.driver.adapter.spec,
        native_text_cache_token_capacity(
            3,
            8,
            backend.driver.adapter.max_prefill_tokens,
            backend.driver.adapter.spec.max_position_embeddings,
            "Qwen",
        )
        .expect("expected cache capacity"),
    )
    .expect("expected caches allocate");
    let expected_cancellation = CancellationToken::new();
    let mut expected_scratch = InferenceScratchpad::new();
    let expected_hidden = native_text_prefill_context_with_cache(
        "Qwen",
        1,
        &[0, 1, 0],
        &mut expected_caches,
        &expected_cancellation,
        &mut expected_scratch,
        |chunk, caches, scratch| {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("test runtime");
            rt.block_on(qwen_prefill_sequence_with_cache(
                &backend.driver.adapter.store,
                &backend.driver.adapter.spec,
                chunk,
                caches,
                &NativeTextMatvecBackend::Cpu,
                scratch,
            ))
            .map_err(BackendError::from)
        },
    );
    let expected_hidden = expected_hidden.expect("fresh prefill succeeds");
    assert_close_vec(second.hidden(), &expected_hidden);
    match (&second.caches[0], &expected_caches[0]) {
        (QwenLayerCache::Linear(actual), QwenLayerCache::Linear(expected)) => {
            assert_eq!(actual.token_count(), expected.token_count());
            assert_eq!(actual.conv_window(), expected.conv_window());
            assert_eq!(actual.recurrent_state(), expected.recurrent_state());
        }
        _ => panic!("expected linear attention caches"),
    }
    std::fs::remove_dir_all(snapshot).ok();
}

#[test]
fn native_qwen_prefill_context_uses_sequence_cache_path_for_full_context() {
    let snapshot = temp_snapshot_dir("sequence-prefill");
    std::fs::remove_dir_all(&snapshot).ok();
    std::fs::create_dir_all(&snapshot).expect("snapshot dir");
    write_tiny_linear_decoder_snapshot(&snapshot);
    let spec = tiny_engine_qwen_spec(llm_models::AttentionKind::LinearAttention);
    let store = SafeTensorShardStore::open(&snapshot).expect("store opens");
    let mut caches = qwen_layer_caches_for_spec(&spec, 1).expect("caches allocate");

    let prefill_cancellation = CancellationToken::new();
    let mut prefill_scratch = InferenceScratchpad::new();
    let hidden = native_text_prefill_context_with_cache(
        "Qwen",
        1,
        &[0, 1, 0],
        &mut caches,
        &prefill_cancellation,
        &mut prefill_scratch,
        |chunk, caches, scratch| {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("test runtime");
            rt.block_on(qwen_prefill_sequence_with_cache(
                &store,
                &spec,
                chunk,
                caches,
                &NativeTextMatvecBackend::Cpu,
                scratch,
            ))
            .map_err(BackendError::from)
        },
    );
    let hidden = hidden.expect("sequence prefill succeeds");

    assert_eq!(hidden.len(), 2);
    match &caches[0] {
        QwenLayerCache::Linear(cache) => assert_eq!(cache.token_count(), 3),
        QwenLayerCache::Full(_) => panic!("layer 0 should be linear attention"),
        other => panic!("unexpected Qwen cache variant: {other:?}"),
    }
    std::fs::remove_dir_all(snapshot).ok();
}

#[test]
fn native_qwen_prefill_context_checks_cancellation_between_chunks() {
    let snapshot = temp_snapshot_dir("sequence-prefill-cancel");
    std::fs::remove_dir_all(&snapshot).ok();
    std::fs::create_dir_all(&snapshot).expect("snapshot dir");
    write_tiny_linear_decoder_snapshot(&snapshot);
    let spec = tiny_engine_qwen_spec(llm_models::AttentionKind::LinearAttention);
    let store = SafeTensorShardStore::open(&snapshot).expect("store opens");
    let mut caches = qwen_layer_caches_for_spec(&spec, 1).expect("caches allocate");
    let cancellation = CancellationToken::new();
    let matvec = CancelAfterFirstConv {
        cancellation: cancellation.clone(),
        conv_calls: std::cell::Cell::new(0),
    };

    let mut cancel_scratch = InferenceScratchpad::new();
    let err = native_text_prefill_context_with_cache(
        "Qwen",
        1,
        &[0, 1, 0],
        &mut caches,
        &cancellation,
        &mut cancel_scratch,
        |chunk, caches, scratch| {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("test runtime");
            rt.block_on(qwen_prefill_sequence_with_cache(
                &store, &spec, chunk, caches, &matvec, scratch,
            ))
            .map_err(BackendError::from)
        },
    )
    .expect_err("cancelled after first chunk");

    assert!(err.is_cancelled());
    match &caches[0] {
        QwenLayerCache::Linear(cache) => assert_eq!(cache.token_count(), 1),
        QwenLayerCache::Full(_) => panic!("layer 0 should be linear attention"),
        other => panic!("unexpected Qwen cache variant: {other:?}"),
    }
    std::fs::remove_dir_all(snapshot).ok();
}

#[test]
fn native_qwen_backend_opens_snapshot_without_engine_manifest() {
    let snapshot = temp_snapshot_dir("no-manifest");
    std::fs::remove_dir_all(&snapshot).ok();
    std::fs::create_dir_all(&snapshot).expect("snapshot dir");
    copy_fixture("config.json", snapshot.join("config.json"));
    copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));
    copy_fixture(
        "model.safetensors.index.json",
        snapshot.join("model.safetensors.index.json"),
    );

    let backend = open_qwen_backend_blocking(crate::DEFAULT_MODEL_ID, &snapshot);
    let metadata = backend.model_metadata();

    assert_eq!(
        backend.driver.max_new_tokens,
        DEFAULT_NATIVE_QWEN_MAX_NEW_TOKENS
    );
    assert_eq!(metadata.id, crate::DEFAULT_MODEL_ID);
    assert_eq!(metadata.backend, "native-qwen");
    assert!(metadata.repo_id.is_none());
    std::fs::remove_dir_all(snapshot).ok();
}

#[tokio::test]
async fn native_qwen_backend_runs_qwen3_dense_single_file_prefill() {
    let snapshot = temp_snapshot_dir("qwen3-dense-single-file");
    std::fs::remove_dir_all(&snapshot).ok();
    std::fs::create_dir_all(&snapshot).expect("snapshot dir");
    write_tiny_qwen3_dense_single_file_decoder_snapshot(&snapshot);
    copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));

    let mut backend = NativeQwenBackend::open("local-qwen3", &snapshot)
        .await
        .expect("backend opens snapshot");
    backend.driver.adapter.top_k = 2;
    let decode = start_qwen_decode_session(
        &backend,
        &[0, 1],
        4,
        &native_qwen_test_request("local-qwen3"),
        &CancellationToken::new(),
    )
    .await
    .expect("dense single-file prefill runs");
    let token_id = select_qwen_token(&backend, decode.hidden(), SamplingConfig::Greedy)
        .await
        .expect("dense tied lm head can select a token");

    assert!(backend.driver.adapter.spec.is_qwen3_dense());
    assert!(token_id < 2);
    match &decode.caches[0] {
        QwenLayerCache::Full(cache) => assert_eq!(cache.token_count(), 2),
        QwenLayerCache::Linear(_) => panic!("dense Qwen3 should use full attention cache"),
        other => panic!("unexpected Qwen cache variant: {other:?}"),
    }
    std::fs::remove_dir_all(snapshot).ok();
}

#[tokio::test]
async fn native_qwen_backend_runs_qwen3_dense_sliding_window_prefill() {
    let snapshot = temp_snapshot_dir("qwen3-dense-sliding-window");
    std::fs::remove_dir_all(&snapshot).ok();
    std::fs::create_dir_all(&snapshot).expect("snapshot dir");
    write_tiny_qwen3_dense_single_file_decoder_snapshot(&snapshot);
    copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));
    let config_path = snapshot.join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&config_path).expect("config reads"))
            .expect("config json parses");
    let config_object = config.as_object_mut().expect("config is an object");
    config_object.insert("use_sliding_window".to_owned(), serde_json::json!(true));
    config_object.insert("sliding_window".to_owned(), serde_json::json!(2));
    std::fs::write(&config_path, config.to_string()).expect("config writes");

    let mut backend = NativeQwenBackend::open("local-qwen3", &snapshot)
        .await
        .expect("backend opens sliding-window snapshot");
    backend.driver.adapter.top_k = 2;
    let decode = start_qwen_decode_session(
        &backend,
        &[0, 1, 0],
        4,
        &native_qwen_test_request("local-qwen3"),
        &CancellationToken::new(),
    )
    .await
    .expect("dense sliding-window prefill runs");

    assert_eq!(backend.driver.adapter.spec.sliding_window, Some(2));
    match &decode.caches[0] {
        QwenLayerCache::Full(cache) => {
            assert_eq!(cache.max_tokens(), 2);
            assert_eq!(cache.token_count(), 2);
            assert!(cache.key(1).is_some(), "latest prompt token must remain");
        }
        QwenLayerCache::Linear(_) => panic!("dense Qwen3 should use full attention cache"),
        other => panic!("unexpected Qwen cache variant: {other:?}"),
    }
    std::fs::remove_dir_all(snapshot).ok();
}

#[tokio::test]
async fn native_qwen_full_attention_prefill_keeps_context_beyond_chunk_size() {
    let snapshot = temp_snapshot_dir("qwen3-dense-long-prefill");
    std::fs::remove_dir_all(&snapshot).ok();
    std::fs::create_dir_all(&snapshot).expect("snapshot dir");
    write_tiny_qwen3_dense_single_file_decoder_snapshot(&snapshot);
    copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));

    let mut backend = NativeQwenBackend::open("local-qwen3", &snapshot)
        .await
        .expect("backend opens snapshot");
    backend.driver.adapter.max_prefill_tokens = 1;
    let context = [0, 1].repeat(6);
    let decode = start_qwen_decode_session(
        &backend,
        &context,
        4,
        &native_qwen_test_request("local-qwen3"),
        &CancellationToken::new(),
    )
    .await
    .expect("dense full-attention prefill keeps the accepted context");

    match &decode.caches[0] {
        QwenLayerCache::Full(cache) => {
            assert_eq!(cache.max_tokens(), 16);
            assert_eq!(cache.token_count(), context.len());
            assert!(cache.key(0).is_some(), "oldest prompt token must remain");
            assert!(
                cache.key(context.len() - 1).is_some(),
                "latest prompt token must remain"
            );
        }
        QwenLayerCache::Linear(_) => panic!("dense Qwen3 should use full attention cache"),
        other => panic!("unexpected Qwen cache variant: {other:?}"),
    }
    std::fs::remove_dir_all(snapshot).ok();
}

#[test]
fn native_qwen_backend_can_eagerly_materialize_indexed_shards_on_open() {
    let snapshot = temp_snapshot_dir("eager-materialize");
    std::fs::remove_dir_all(&snapshot).ok();
    std::fs::create_dir_all(&snapshot).expect("snapshot dir");
    copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));
    write_tiny_qwen3_dense_single_file_decoder_snapshot(&snapshot);
    write_tiny_qwen3_dense_model_index(&snapshot);

    let backend = open_qwen_backend_with_options_blocking(
        crate::DEFAULT_MODEL_ID,
        &snapshot,
        NativeQwenLoadOptions {
            eager_materialize_shards: true,
            ..NativeQwenLoadOptions::default()
        },
    );

    assert_eq!(backend.driver.adapter.store.materialized_shard_count(), 1);
    std::fs::remove_dir_all(snapshot).ok();
}

#[test]
fn native_qwen_backend_uses_configured_prefix_cache_budget() {
    let snapshot = temp_snapshot_dir("prefix-cache-budget");
    std::fs::remove_dir_all(&snapshot).ok();
    std::fs::create_dir_all(&snapshot).expect("snapshot dir");
    copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));
    write_tiny_qwen3_dense_single_file_decoder_snapshot(&snapshot);
    write_tiny_qwen3_dense_model_index(&snapshot);

    let backend = open_qwen_backend_with_options_blocking(
        crate::DEFAULT_MODEL_ID,
        &snapshot,
        NativeQwenLoadOptions {
            prefix_cache_bytes: Some(7),
            ..NativeQwenLoadOptions::default()
        },
    );

    assert_eq!(backend.driver.adapter.prefix_cache.max_bytes, 7);
    std::fs::remove_dir_all(snapshot).ok();
}

#[test]
fn native_qwen_backend_opens_with_raw_path_prefix_disk_cache() {
    let snapshot = temp_snapshot_dir("prefix-disk-cache");
    std::fs::remove_dir_all(&snapshot).ok();
    std::fs::create_dir_all(&snapshot).expect("snapshot dir");
    copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));
    write_tiny_qwen3_dense_single_file_decoder_snapshot(&snapshot);
    write_tiny_qwen3_dense_model_index(&snapshot);

    let backend = open_qwen_backend_with_options_blocking(
        crate::DEFAULT_MODEL_ID,
        &snapshot,
        NativeQwenLoadOptions {
            prefix_disk_cache: Some(crate::native_text::NativeTextDiskCacheConfig::for_root(
                snapshot.join("disk-cache"),
            )),
            ..NativeQwenLoadOptions::default()
        },
    );

    assert!(backend.driver.adapter.prefix_disk_cache.is_some());
    std::fs::remove_dir_all(snapshot).ok();
}
