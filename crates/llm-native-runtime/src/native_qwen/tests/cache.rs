use super::*;

#[test]
fn native_qwen_decode_session_cleans_cache_mirrors_on_drop() {
    let cleaner = Arc::new(TestQwenCacheMirrorCleaner::default());
    let session_cleaner: Arc<dyn NativeTextCacheMirrorCleaner<QwenLayerCache>> = cleaner.clone();

    {
        let cache =
            QwenLayerCache::Full(LayerKvCache::new(1, 1, 1).expect("test cache shape is valid"));
        let _session = NativeQwenDecodeSession {
            hidden: vec![0.0],
            caches: vec![cache],
            cache_mirror_cleaner: Some(session_cleaner),
        };
    }

    assert_eq!(cleaner.calls.load(Ordering::SeqCst), 1);
    assert_eq!(cleaner.cache_count.load(Ordering::SeqCst), 1);
}

#[test]
fn native_qwen_prefix_cache_reuses_longest_compatible_prefix() {
    let cache = NativeQwenPrefixCache::new(10_000);
    let metrics = NativeQwenPrefixCacheMetrics::default();
    let namespace = native_qwen_test_prefix_namespace("base");
    let mut layer_cache = LayerKvCache::new(4, 1, 2).expect("cache shape is valid");
    layer_cache
        .append(&[1.0, 2.0], &[3.0, 4.0])
        .expect("token fits");
    let original_cache_id = layer_cache.id();
    let original_block_id = layer_cache.block_ids()[0];
    let original_block_ptr = layer_cache.active_blocks().expect("active blocks")[0]
        .key_storage()
        .as_ptr();
    let caches = vec![QwenLayerCache::Full(layer_cache)];

    cache.store(namespace.clone(), &[1, 2], &[0.25, 0.75], &caches, &metrics);

    let hit = cache
        .lookup(&namespace, &[1, 2, 3], &metrics)
        .expect("compatible longer prompt reuses stored prefix");
    assert_eq!(hit.token_count, 2);
    assert_eq!(hit.hidden, vec![0.25, 0.75]);
    match &hit.caches[0] {
        QwenLayerCache::Full(cache) => {
            assert_ne!(cache.id(), original_cache_id);
            assert_eq!(cache.block_ids()[0], original_block_id);
            assert_eq!(
                cache.active_blocks().expect("active blocks")[0]
                    .key_storage()
                    .as_ptr(),
                original_block_ptr,
                "Qwen prefix hits should share retained KV block storage"
            );
            assert_eq!(cache.token_count(), 1);
        }
        QwenLayerCache::Linear(_) => panic!("expected full-attention cache"),
    }

    let incompatible_namespace = NativeQwenPrefixCacheNamespace {
        tool_schema: Some("different-tool-schema".to_owned()),
        ..namespace.clone()
    };
    assert!(
        cache
            .lookup(&incompatible_namespace, &[1, 2], &metrics)
            .is_none(),
        "tool schema changes must not reuse prefix state"
    );
}

#[test]
fn native_qwen_prefix_cache_separates_capacity_manifest_profile_and_required_tool_name() {
    let cache = NativeQwenPrefixCache::new(10_000);
    let metrics = NativeQwenPrefixCacheMetrics::default();
    let namespace = native_qwen_test_prefix_namespace("namespace-policy");
    let larger_capacity_namespace = NativeQwenPrefixCacheNamespace {
        cache_tokens: namespace.cache_tokens * 2,
        ..namespace.clone()
    };
    let different_manifest_namespace = NativeQwenPrefixCacheNamespace {
        resolved_commit: Some("fedcba9876543210fedcba9876543210fedcba98".to_owned()),
        ..namespace.clone()
    };
    let different_profile_namespace = NativeQwenPrefixCacheNamespace {
        profile: Some("qwen-other-profile".to_owned()),
        ..namespace.clone()
    };
    let lookup_required_tool = NativeQwenPrefixCacheNamespace {
        request_mode: format!(
            "chat,json_object=false,required_tool={:?}",
            BackendToolChoice::RequiredFunction("lookup".to_owned())
        ),
        ..namespace.clone()
    };
    let search_required_tool = NativeQwenPrefixCacheNamespace {
        request_mode: format!(
            "chat,json_object=false,required_tool={:?}",
            BackendToolChoice::RequiredFunction("search".to_owned())
        ),
        ..namespace.clone()
    };

    cache.store(namespace.clone(), &[1, 2], &[0.25, 0.75], &[], &metrics);
    cache.store(
        lookup_required_tool.clone(),
        &[1, 2],
        &[0.25, 0.75],
        &[],
        &metrics,
    );

    assert!(
        cache
            .lookup(&larger_capacity_namespace, &[1, 2], &metrics)
            .is_none(),
        "cache capacity changes must not reuse Qwen prefix state"
    );
    assert!(
        cache
            .lookup(&different_manifest_namespace, &[1, 2], &metrics)
            .is_none(),
        "manifest identity changes must not reuse Qwen prefix state"
    );
    assert!(
        cache
            .lookup(&different_profile_namespace, &[1, 2], &metrics)
            .is_none(),
        "profile changes must not reuse Qwen prefix state"
    );
    assert!(
        cache
            .lookup(&search_required_tool, &[1, 2], &metrics)
            .is_none(),
        "required tool-choice names must not reuse Qwen prefix state"
    );
}

#[test]
fn native_qwen_prefix_cache_evicts_lru_entries_to_fit_budget() {
    let cache = NativeQwenPrefixCache::new(40);
    let metrics = NativeQwenPrefixCacheMetrics::default();
    let namespace = native_qwen_test_prefix_namespace("eviction");
    let hidden = vec![1.0; 8];

    cache.store(namespace.clone(), &[1], &hidden, &[], &metrics);
    cache.store(namespace.clone(), &[2], &hidden, &[], &metrics);

    assert!(
        cache.lookup(&namespace, &[1], &metrics).is_none(),
        "oldest entry should be evicted"
    );
    assert!(
        cache.lookup(&namespace, &[2], &metrics).is_some(),
        "newest entry should remain resident"
    );
    let inner = cache.inner.lock_or_panic("native Qwen prefix cache");
    assert_eq!(inner.entries.len(), 1);
    assert_eq!(inner.used_bytes, 32);
}

#[test]
fn native_qwen_prefix_cache_rejects_entries_over_small_budget() {
    let cache = NativeQwenPrefixCache::new(4);
    let metrics = NativeQwenPrefixCacheMetrics::default();
    let namespace = native_qwen_test_prefix_namespace("small-budget");
    let hidden = vec![1.0; 2];

    cache.store(namespace.clone(), &[1], &hidden, &[], &metrics);

    assert!(cache.lookup(&namespace, &[1], &metrics).is_none());
    let snapshot = metrics.snapshot();
    assert_eq!(snapshot["stores"], 0);
    assert_eq!(snapshot["rejected"], 1);
    assert_eq!(snapshot["resident_bytes"], 0);
    assert_eq!(snapshot["resident_entries"], 0);
}

#[test]
fn native_qwen_prefix_cache_metrics_expose_hits_misses_and_evictions() {
    let metrics = NativeQwenPrefixCacheMetrics::default();

    metrics.record_hit(3);
    metrics.record_miss();
    metrics.record_miss_tokens(5);
    metrics.record_prefill_chunk(5);
    metrics.record_store(32);
    metrics.record_eviction(16);
    metrics.record_rejected();
    metrics.record_residency(32, 1);
    metrics.record_lookup_scan(5, 4);
    metrics.record_hit_clone_bytes(64);

    let snapshot = metrics.snapshot();
    assert_eq!(snapshot["hits"], 1);
    assert_eq!(snapshot["misses"], 1);
    assert_eq!(snapshot["stores"], 1);
    assert_eq!(snapshot["evictions"], 1);
    assert_eq!(snapshot["rejected"], 1);
    assert_eq!(snapshot["reused_tokens"], 3);
    assert_eq!(snapshot["prefill_chunks"], 1);
    assert_eq!(snapshot["prefill_tokens"], 5);
    assert_eq!(snapshot["hit_tokens"], 3);
    assert_eq!(snapshot["miss_tokens"], 5);
    assert_eq!(snapshot["avoided_prefill_tokens"], 3);
    assert_eq!(snapshot["bytes_stored"], 32);
    assert_eq!(snapshot["bytes_evicted"], 16);
    assert_eq!(snapshot["resident_bytes"], 32);
    assert_eq!(snapshot["resident_entries"], 1);
    assert_eq!(snapshot["entries_scanned"], 5);
    assert_eq!(snapshot["namespace_entries_scanned"], 4);
    assert_eq!(snapshot["hit_clone_bytes"], 64);
}

#[test]
fn bf16_matrix_buffer_cache_evicts_lru_entries_to_fit_budget() {
    let mut cache = Bf16MatrixBufferCache::new(10);
    let first = Bf16MatrixCacheKey {
        tensor: "first.weight".to_owned(),
        element_offset: 0,
        rows: 2,
        columns: 1,
    };
    let second = Bf16MatrixCacheKey {
        tensor: "second.weight".to_owned(),
        element_offset: 0,
        rows: 2,
        columns: 1,
    };
    let third = Bf16MatrixCacheKey {
        tensor: "third.weight".to_owned(),
        element_offset: 0,
        rows: 3,
        columns: 1,
    };

    assert!(cache.get(&first).is_none());
    assert!(cache.insert(first.clone(), "first", 4).inserted);
    assert!(cache.insert(second.clone(), "second", 4).inserted);
    assert_eq!(cache.get(&first), Some("first"));

    let result = cache.insert(third.clone(), "third", 6);

    assert!(result.inserted);
    assert_eq!(result.evicted_count, 1);
    assert_eq!(result.evicted_bytes, 4);
    assert_eq!(cache.used_bytes(), 10);
    assert_eq!(cache.get(&second), None);
    assert_eq!(cache.get(&first), Some("first"));
    assert_eq!(cache.get(&third), Some("third"));
}

#[test]
fn bf16_matrix_buffer_cache_skips_entries_larger_than_budget() {
    let mut cache = Bf16MatrixBufferCache::new(4);
    let key = Bf16MatrixCacheKey {
        tensor: "large.weight".to_owned(),
        element_offset: 0,
        rows: 3,
        columns: 1,
    };

    let result = cache.insert(key.clone(), "large", 6);

    assert!(!result.inserted);
    assert_eq!(result.evicted_count, 0);
    assert_eq!(cache.used_bytes(), 0);
    assert_eq!(cache.get(&key), None);
}

#[test]
fn native_qwen_metal_weight_cache_bytes_uses_default_or_configured_value() {
    assert_eq!(
        native_qwen_metal_weight_cache_bytes(None),
        DEFAULT_NATIVE_QWEN_METAL_WEIGHT_CACHE_BYTES
    );
    assert_eq!(native_qwen_metal_weight_cache_bytes(Some(0)), 0);
    assert_eq!(native_qwen_metal_weight_cache_bytes(Some(4096)), 4096);
}

#[test]
fn native_qwen_warmable_bf16_matrix_tensors_filters_rank2_bf16() {
    let snapshot = temp_snapshot_dir("warmable-bf16-matrices");
    std::fs::remove_dir_all(&snapshot).ok();
    std::fs::create_dir_all(&snapshot).expect("snapshot dir");
    let tensors = vec![
        ("z.bias", vec![2], vec![1.0, 2.0]),
        ("b.weight", vec![2, 1], vec![3.0, 4.0]),
        ("a.weight", vec![1, 2], vec![5.0, 6.0]),
    ];
    let safetensors = tiny_owned_multi_safetensors_bf16(&tensors);
    std::fs::write(snapshot.join("model.safetensors"), &safetensors).expect("write shard");
    std::fs::write(
        snapshot.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": safetensors.len() },
            "weight_map": {
                "z.bias": "model.safetensors",
                "b.weight": "model.safetensors",
                "a.weight": "model.safetensors"
            }
        })
        .to_string(),
    )
    .expect("write index");
    let store = SafeTensorShardStore::open(&snapshot).expect("store opens");

    let warmable = native_qwen_warmable_bf16_matrix_tensors(&store).expect("warmable tensors");

    assert_eq!(
        warmable
            .iter()
            .map(|tensor| (
                tensor.name.as_str(),
                tensor.rows,
                tensor.columns,
                tensor.byte_len
            ))
            .collect::<Vec<_>>(),
        vec![("a.weight", 1, 2, 4), ("b.weight", 2, 1, 4)]
    );
    std::fs::remove_dir_all(snapshot).ok();
}

#[test]
fn native_qwen_warmable_bf16_matrix_tensors_orders_qwen_execution_weights() {
    let snapshot = temp_snapshot_dir("warmable-qwen-order");
    std::fs::remove_dir_all(&snapshot).ok();
    std::fs::create_dir_all(&snapshot).expect("snapshot dir");
    let tensors = vec![
        ("zz.unclassified.weight", vec![1, 1], vec![1.0]),
        ("lm_head.weight", vec![1, 1], vec![2.0]),
        (
            "model.language_model.layers.10.self_attn.o_proj.weight",
            vec![1, 1],
            vec![3.0],
        ),
        (
            "model.language_model.layers.2.mlp.shared_expert.down_proj.weight",
            vec![1, 1],
            vec![4.0],
        ),
        (
            "model.language_model.layers.2.self_attn.q_proj.weight",
            vec![1, 1],
            vec![5.0],
        ),
        (
            "model.language_model.embed_tokens.weight",
            vec![1, 1],
            vec![6.0],
        ),
        (
            "model.language_model.layers.2.self_attn.k_proj.weight",
            vec![1, 1],
            vec![7.0],
        ),
        (
            "model.language_model.layers.2.mlp.gate.weight",
            vec![1, 1],
            vec![8.0],
        ),
    ];
    let safetensors = tiny_owned_multi_safetensors_bf16(&tensors);
    std::fs::write(snapshot.join("model.safetensors"), &safetensors).expect("write shard");
    std::fs::write(
        snapshot.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": safetensors.len() },
            "weight_map": tensors
                .iter()
                .map(|(name, _, _)| {
                    (
                        (*name).to_owned(),
                        serde_json::Value::String("model.safetensors".to_owned()),
                    )
                })
                .collect::<serde_json::Map<_, _>>()
        })
        .to_string(),
    )
    .expect("write index");
    let store = SafeTensorShardStore::open(&snapshot).expect("store opens");

    let warmable = native_qwen_warmable_bf16_matrix_tensors(&store).expect("warmable tensors");

    assert_eq!(
        warmable
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect::<Vec<_>>(),
        vec![
            "model.language_model.embed_tokens.weight",
            "model.language_model.layers.2.self_attn.q_proj.weight",
            "model.language_model.layers.2.self_attn.k_proj.weight",
            "model.language_model.layers.2.mlp.gate.weight",
            "model.language_model.layers.2.mlp.shared_expert.down_proj.weight",
            "model.language_model.layers.10.self_attn.o_proj.weight",
            "lm_head.weight",
            "zz.unclassified.weight",
        ]
    );
    std::fs::remove_dir_all(snapshot).ok();
}

#[test]
fn native_qwen_cpu_backend_warmup_reports_non_metal_skip() {
    let snapshot = temp_snapshot_dir("cpu-warmup");
    std::fs::remove_dir_all(&snapshot).ok();
    std::fs::create_dir_all(&snapshot).expect("snapshot dir");
    let safetensors = tiny_owned_multi_safetensors_bf16(&[
        ("a.weight", vec![1, 2], vec![1.0, 2.0]),
        ("b.bias", vec![2], vec![3.0, 4.0]),
    ]);
    std::fs::write(snapshot.join("model.safetensors"), &safetensors).expect("write shard");
    std::fs::write(
        snapshot.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": safetensors.len() },
            "weight_map": {
                "a.weight": "model.safetensors",
                "b.bias": "model.safetensors"
            }
        })
        .to_string(),
    )
    .expect("write index");
    let store = SafeTensorShardStore::open(&snapshot).expect("store opens");

    let warmup = NativeTextMatvecBackend::Cpu
        .warm_bf16_matrix_cache(&store)
        .expect("cpu warmup reports stats");

    assert_eq!(
        warmup,
        NativeQwenMetalWarmup {
            candidates: 1,
            skipped_non_metal: 1,
            ..NativeQwenMetalWarmup::default()
        }
    );
    std::fs::remove_dir_all(snapshot).ok();
}

#[test]
fn native_qwen_system_default_reuses_shared_metal_state_for_same_model_budget() {
    let first = NativeTextMatvecBackend::system_default(1_234_567, "test-shared-model");
    let second = NativeTextMatvecBackend::system_default(1_234_567, "test-shared-model");
    let other_model = NativeTextMatvecBackend::system_default(1_234_567, "test-other-model");

    match (&first, &second, &other_model) {
        (
            NativeTextMatvecBackend::Metal(first),
            NativeTextMatvecBackend::Metal(second),
            NativeTextMatvecBackend::Metal(other_model),
        ) => {
            assert!(Arc::ptr_eq(first, second));
            assert!(!Arc::ptr_eq(first, other_model));
        }
        (
            NativeTextMatvecBackend::Cpu,
            NativeTextMatvecBackend::Cpu,
            NativeTextMatvecBackend::Cpu,
        ) => {
            eprintln!("no Metal device available; skipping shared state test");
        }
        _ => panic!("Metal backend availability changed between calls"),
    }
}
