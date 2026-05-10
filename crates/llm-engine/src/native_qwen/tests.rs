use super::*;
use crate::native_matvec::{
    Bf16MatrixBufferCache, Bf16MatrixCacheKey,
    DEFAULT_NATIVE_TEXT_METAL_WEIGHT_CACHE_BYTES as DEFAULT_NATIVE_QWEN_METAL_WEIGHT_CACHE_BYTES,
    MetalBackendMetrics, NativeTextMetalWarmup as NativeQwenMetalWarmup,
};
use crate::native_text::{
    NativeStreamTextDeltas, NativeTextCandidateDecision, NativeTextStopTokens,
    native_text_cache_token_capacity, native_text_prefill_context_with_cache,
    native_text_worker_stream, sample_token_id_with_draw,
};
use crate::sync_ext::RecoverPoisonedMutex;
use futures::StreamExt;
use llm_backend::{
    BackendCacheContext, CpuNativeMatvecBackend, InferenceScratchpad, LayerKvCache, MathError,
    NativeMatvecBackend, SafeTensorShardStore, TensorLoadError, qwen_layer_caches_for_spec,
    qwen_prefill_sequence_with_cache_with_matvec,
};
use llm_models::QwenModelSpec;
use llm_models::{ModelFamilyAdapter, QwenFamilyAdapter};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

fn test_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime")
}

fn open_qwen_backend_blocking(model_id: &str, snapshot: &Path) -> NativeQwenBackend {
    test_runtime()
        .block_on(NativeQwenBackend::open(model_id, snapshot))
        .expect("backend opens snapshot")
}

fn open_qwen_backend_with_options_blocking(
    model_id: &str,
    snapshot: &Path,
    options: NativeQwenLoadOptions,
) -> NativeQwenBackend {
    test_runtime()
        .block_on(NativeQwenBackend::open_with_options(
            model_id, snapshot, options,
        ))
        .expect("backend opens snapshot")
}

#[derive(Default)]
struct TestQwenCacheMirrorCleaner {
    calls: AtomicUsize,
    cache_count: AtomicUsize,
}

impl NativeTextCacheMirrorCleaner<QwenLayerCache> for TestQwenCacheMirrorCleaner {
    fn cleanup_cache_mirrors(&self, caches: &[QwenLayerCache]) {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.cache_count.fetch_add(caches.len(), Ordering::SeqCst);
    }
}

#[test]
fn metal_backend_metrics_records_attempt_success_and_fallback_by_kernel() {
    let metrics = MetalBackendMetrics::default();

    metrics.record_attempt("matvec_bf16_f32");
    metrics.record_success("matvec_bf16_f32");
    metrics.record_attempt("matvec_bf16_f32");
    metrics.record_fallback("matvec_bf16_f32", "rows=2,cols=3", "execution failed");

    let snapshot = metrics.snapshot();
    let matvec = &snapshot["kernels"]["matvec_bf16_f32"];
    assert_eq!(matvec["attempts"], 2);
    assert_eq!(matvec["successes"], 1);
    assert_eq!(matvec["fallbacks"], 1);
}

#[test]
fn metal_backend_metrics_records_bf16_matrix_cache_activity() {
    let metrics = MetalBackendMetrics::default();

    metrics.record_bf16_matrix_cache_miss();
    metrics.record_bf16_matrix_cache_upload(12);
    metrics.record_bf16_matrix_cache_eviction(2, 8);
    metrics.record_bf16_matrix_cache_residency(10, 3, 16);
    metrics.record_bf16_matrix_cache_hit();

    let snapshot = metrics.snapshot();
    let cache = &snapshot["bf16_matrix_cache"];
    assert_eq!(cache["hits"], 1);
    assert_eq!(cache["misses"], 1);
    assert_eq!(cache["uploads"], 1);
    assert_eq!(cache["bytes_uploaded"], 12);
    assert_eq!(cache["evictions"], 2);
    assert_eq!(cache["bytes_evicted"], 8);
    assert_eq!(cache["resident_bytes"], 10);
    assert_eq!(cache["resident_buffers"], 3);
    assert_eq!(cache["budget_bytes"], 16);
}

#[test]
fn metal_backend_metrics_records_resident_attention_cache_activity() {
    let metrics = MetalBackendMetrics::default();

    metrics.record_kv_cache_allocation(16);
    metrics.record_kv_cache_sync(8);
    metrics.record_kv_cache_residency(16, 2);
    metrics.record_kv_cache_eviction(2, 16);
    metrics.record_kv_cache_residency(0, 0);
    metrics.record_linear_cache_allocation(12);
    metrics.record_linear_cache_sync(4);
    metrics.record_linear_cache_residency(12, 1);
    metrics.record_linear_cache_eviction(1, 12);
    metrics.record_linear_cache_residency(0, 0);

    let snapshot = metrics.snapshot();
    let kv = &snapshot["kv_cache"];
    assert_eq!(kv["allocations"], 1);
    assert_eq!(kv["syncs"], 1);
    assert_eq!(kv["evictions"], 2);
    assert_eq!(kv["bytes_uploaded"], 24);
    assert_eq!(kv["bytes_evicted"], 16);
    assert_eq!(kv["resident_bytes"], 0);
    assert_eq!(kv["resident_buffers"], 0);
    let linear = &snapshot["linear_attention_cache"];
    assert_eq!(linear["allocations"], 1);
    assert_eq!(linear["syncs"], 1);
    assert_eq!(linear["evictions"], 1);
    assert_eq!(linear["bytes_uploaded"], 16);
    assert_eq!(linear["bytes_evicted"], 12);
    assert_eq!(linear["resident_bytes"], 0);
    assert_eq!(linear["resident_buffers"], 0);
}

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
    let inner = cache.inner.lock_or_recover("native Qwen prefix cache");
    assert_eq!(inner.entries.len(), 1);
    assert_eq!(inner.used_bytes, 32);
}

#[test]
fn native_qwen_prefix_cache_metrics_expose_hits_misses_and_evictions() {
    let metrics = NativeQwenPrefixCacheMetrics::default();

    metrics.record_hit(3);
    metrics.record_miss();
    metrics.record_store(32);
    metrics.record_eviction(16);
    metrics.record_rejected();
    metrics.record_residency(32, 1);

    let snapshot = metrics.snapshot();
    assert_eq!(snapshot["hits"], 1);
    assert_eq!(snapshot["misses"], 1);
    assert_eq!(snapshot["stores"], 1);
    assert_eq!(snapshot["evictions"], 1);
    assert_eq!(snapshot["rejected"], 1);
    assert_eq!(snapshot["reused_tokens"], 3);
    assert_eq!(snapshot["bytes_stored"], 32);
    assert_eq!(snapshot["bytes_evicted"], 16);
    assert_eq!(snapshot["resident_bytes"], 32);
    assert_eq!(snapshot["resident_entries"], 1);
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

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime");
    let warmup = rt
        .block_on(NativeTextMatvecBackend::Cpu.warm_bf16_matrix_cache(&store))
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

#[test]
fn native_max_tokens_defaults_to_configured_cache_limit() {
    assert_eq!(
        resolve_native_max_tokens(None, 4).expect("omitted max tokens uses configured cap"),
        4
    );
}

#[test]
fn native_qwen_default_max_new_tokens_is_interactive_budget() {
    assert_eq!(DEFAULT_NATIVE_QWEN_MAX_NEW_TOKENS, 256);
    assert_eq!(
        resolve_native_max_tokens(None, DEFAULT_NATIVE_QWEN_MAX_NEW_TOKENS)
            .expect("omitted max tokens uses native default"),
        256
    );
    assert_eq!(
        resolve_native_max_tokens(Some(128), DEFAULT_NATIVE_QWEN_MAX_NEW_TOKENS)
            .expect("requests below native default are accepted"),
        128
    );
}

#[test]
fn native_max_tokens_accepts_multi_token_decode_with_cache() {
    assert_eq!(
        resolve_native_max_tokens(Some(2), 4).expect("multi-token decode uses cache"),
        2
    );
}

#[test]
fn native_max_tokens_rejects_requests_above_configured_limit() {
    let err = resolve_native_max_tokens(Some(5), 4)
        .expect_err("request above configured limit fails closed");

    assert!(matches!(err, BackendError::UnsupportedRequest(_)));
    assert!(err.to_string().contains("configured native Qwen limit"));
}

#[test]
fn native_qwen_cache_capacity_preserves_prompt_and_generation_budget() {
    let capacity = native_text_cache_token_capacity(40, 8, 32, 64, "Qwen")
        .expect("prompt plus generation budget fits context");
    let spec = QwenModelSpec {
        family: llm_models::ModelFamily::Qwen,
        architecture: "Qwen3_5MoeForConditionalGeneration".to_owned(),
        model_type: "qwen3_5_moe".to_owned(),
        text_model_type: "qwen3_5_moe_text".to_owned(),
        hidden_size: 2,
        rms_norm_eps: 0.0,
        tie_word_embeddings: false,
        rope_theta: 1_000_000.0,
        partial_rotary_factor: 1.0,
        num_hidden_layers: 1,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 2,
        linear_num_key_heads: 1,
        linear_num_value_heads: 1,
        linear_key_head_dim: 1,
        linear_value_head_dim: 1,
        linear_conv_kernel_dim: 1,
        num_experts: 1,
        num_experts_per_tok: 1,
        moe_intermediate_size: 1,
        shared_expert_intermediate_size: 1,
        max_position_embeddings: 32,
        vocab_size: 16,
        layer_kinds: vec![llm_models::AttentionKind::FullAttention],
    };

    let caches = qwen_layer_caches_for_spec(&spec, capacity).expect("cache allocates");
    match &caches[0] {
        QwenLayerCache::Full(cache) => assert_eq!(cache.max_tokens(), 64),
        QwenLayerCache::Linear(_) => panic!("expected full-attention cache"),
    }
}

#[test]
fn native_qwen_cache_capacity_rejects_context_beyond_position_limit() {
    let err = native_text_cache_token_capacity(60, 8, 32, 64, "Qwen")
        .expect_err("context beyond model position limit fails closed");

    assert!(matches!(err, BackendError::UnsupportedRequest(_)));
    assert!(
        err.to_string().contains("model context limit"),
        "error should name context limit: {err}"
    );
}

#[test]
fn native_qwen_adapter_stop_tokens_use_chatml_im_end() {
    let snapshot = temp_snapshot_dir("qwen-stop-tokens");
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
    let im_end = backend
        .driver
        .tokenizer
        .token_to_id("<|im_end|>")
        .expect("qwen tokenizer has im_end token") as usize;
    let non_stop = (0..16)
        .find(|token_id| *token_id != im_end)
        .expect("small non-stop token id exists");

    assert_eq!(
        backend.driver.adapter.stop_tokens(),
        NativeTextStopTokens {
            token_ids: &[],
            token_strings: &["<|im_end|>"],
        }
    );
    assert!(matches!(
        backend
            .driver
            .adapter
            .observe_candidate(&backend.driver.tokenizer, &[], im_end)
            .expect("im_end candidate is observed"),
        NativeTextCandidateDecision::Stop
    ));
    assert!(matches!(
        backend
            .driver
            .adapter
            .observe_candidate(&backend.driver.tokenizer, &[], non_stop)
            .expect("non-stop candidate is observed"),
        NativeTextCandidateDecision::Emit(token_id) if token_id == non_stop
    ));
    std::fs::remove_dir_all(snapshot).ok();
}

#[test]
fn native_qwen_start_decode_session_prefills_full_context_with_bounded_cache() {
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

    let decode = backend
        .start_decode_session(
            &[0, 1, 0],
            8,
            &native_qwen_test_request(crate::DEFAULT_MODEL_ID),
            &CancellationToken::new(),
        )
        .expect("decode session starts");

    match &decode.caches[0] {
        QwenLayerCache::Linear(cache) => assert_eq!(cache.token_count(), 3),
        QwenLayerCache::Full(_) => panic!("layer 0 should be linear attention"),
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

    let first = backend
        .start_decode_session(&[0, 1], 8, &request, &CancellationToken::new())
        .expect("first decode session starts");
    drop(first);
    let second = backend
        .start_decode_session(&[0, 1, 0], 8, &top_p_request, &CancellationToken::new())
        .expect("second decode session starts");

    assert!(
        native_prefix_metric_counter("hits") > before_hits,
        "second request should hit the shared prefix cache"
    );
    match &second.caches[0] {
        QwenLayerCache::Linear(cache) => assert_eq!(cache.token_count(), 3),
        QwenLayerCache::Full(_) => panic!("layer 0 should be linear attention"),
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
            rt.block_on(qwen_prefill_sequence_with_cache_with_matvec(
                &backend.driver.adapter.store,
                &backend.driver.adapter.spec,
                chunk,
                caches,
                &NativeTextMatvecBackend::Cpu,
                scratch,
            ))
            .map_err(|err| BackendError::Other(err.to_string()))
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
            rt.block_on(qwen_prefill_sequence_with_cache_with_matvec(
                &store,
                &spec,
                chunk,
                caches,
                &NativeTextMatvecBackend::Cpu,
                scratch,
            ))
            .map_err(|err| BackendError::Other(err.to_string()))
        },
    );
    let hidden = hidden.expect("sequence prefill succeeds");

    assert_eq!(hidden.len(), 2);
    match &caches[0] {
        QwenLayerCache::Linear(cache) => assert_eq!(cache.token_count(), 3),
        QwenLayerCache::Full(_) => panic!("layer 0 should be linear attention"),
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
            rt.block_on(qwen_prefill_sequence_with_cache_with_matvec(
                &store, &spec, chunk, caches, &matvec, scratch,
            ))
            .map_err(|err| BackendError::Other(err.to_string()))
        },
    )
    .expect_err("cancelled after first chunk");

    assert!(matches!(err, BackendError::Cancelled));
    match &caches[0] {
        QwenLayerCache::Linear(cache) => assert_eq!(cache.token_count(), 1),
        QwenLayerCache::Full(_) => panic!("layer 0 should be linear attention"),
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
    assert_eq!(metadata.snapshot_path.as_deref(), Some(snapshot.as_path()));
    assert!(metadata.manifest_digest.is_none());
    assert!(metadata.repo_id.is_none());
    std::fs::remove_dir_all(snapshot).ok();
}

#[test]
fn native_qwen_backend_runs_qwen3_dense_single_file_prefill() {
    let snapshot = temp_snapshot_dir("qwen3-dense-single-file");
    std::fs::remove_dir_all(&snapshot).ok();
    std::fs::create_dir_all(&snapshot).expect("snapshot dir");
    write_tiny_qwen3_dense_single_file_decoder_snapshot(&snapshot);
    copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));

    let mut backend = open_qwen_backend_blocking("local-qwen3", &snapshot);
    backend.driver.adapter.top_k = 2;
    let decode = backend
        .start_decode_session(
            &[0, 1],
            4,
            &native_qwen_test_request("local-qwen3"),
            &CancellationToken::new(),
        )
        .expect("dense single-file prefill runs");
    let candidate = backend
        .next_token_from_hidden(decode.hidden(), SamplingConfig::Greedy)
        .expect("dense tied lm head can select a token");

    assert!(backend.driver.adapter.spec.is_qwen3_dense());
    assert!(candidate.token_id < 2);
    match &decode.caches[0] {
        QwenLayerCache::Full(cache) => assert_eq!(cache.token_count(), 2),
        QwenLayerCache::Linear(_) => panic!("dense Qwen3 should use full attention cache"),
    }
    std::fs::remove_dir_all(snapshot).ok();
}

#[test]
fn native_qwen_full_attention_prefill_keeps_context_beyond_chunk_size() {
    let snapshot = temp_snapshot_dir("qwen3-dense-long-prefill");
    std::fs::remove_dir_all(&snapshot).ok();
    std::fs::create_dir_all(&snapshot).expect("snapshot dir");
    write_tiny_qwen3_dense_single_file_decoder_snapshot(&snapshot);
    copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));

    let mut backend = open_qwen_backend_blocking("local-qwen3", &snapshot);
    backend.driver.adapter.max_prefill_tokens = 1;
    let context = [0, 1].repeat(6);
    let decode = backend
        .start_decode_session(
            &context,
            4,
            &native_qwen_test_request("local-qwen3"),
            &CancellationToken::new(),
        )
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

#[tokio::test]
async fn native_qwen_generate_with_cancel_observes_pre_cancelled_token() {
    let snapshot = temp_snapshot_dir("cancelled-generate");
    std::fs::remove_dir_all(&snapshot).ok();
    std::fs::create_dir_all(&snapshot).expect("snapshot dir");
    copy_fixture("config.json", snapshot.join("config.json"));
    copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));
    copy_fixture(
        "model.safetensors.index.json",
        snapshot.join("model.safetensors.index.json"),
    );
    let backend = NativeQwenBackend::open(crate::DEFAULT_MODEL_ID, &snapshot)
        .await
        .expect("backend opens snapshot");
    let cancellation = CancellationToken::new();
    cancellation.cancel();

    let err = backend
        .generate_with_cancel(
            BackendRequest {
                model: crate::DEFAULT_MODEL_ID.to_owned(),
                prompt: "say hi".to_owned(),
                chat_context: None,
                max_tokens: Some(1),
                sampling: SamplingConfig::Greedy,
                required_tool_choice: None,
                json_object_mode: false,
                conversation_mode: false,
                cache_context: BackendCacheContext::default(),
            },
            cancellation,
        )
        .await
        .expect_err("pre-cancelled generation fails before decode");

    assert!(err.to_string().contains("cancelled"));
    std::fs::remove_dir_all(snapshot).ok();
}

#[test]
fn native_qwen_stream_with_cancel_observes_pre_cancelled_token() {
    let snapshot = temp_snapshot_dir("cancelled-stream");
    std::fs::remove_dir_all(&snapshot).ok();
    std::fs::create_dir_all(&snapshot).expect("snapshot dir");
    copy_fixture("config.json", snapshot.join("config.json"));
    copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));
    copy_fixture(
        "model.safetensors.index.json",
        snapshot.join("model.safetensors.index.json"),
    );
    let backend = open_qwen_backend_blocking(crate::DEFAULT_MODEL_ID, &snapshot);
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let (tx, _rx) = tokio::sync::mpsc::channel(1);

    let err = backend
        .generate_blocking_stream(
            BackendRequest {
                model: crate::DEFAULT_MODEL_ID.to_owned(),
                prompt: "say hi".to_owned(),
                chat_context: None,
                max_tokens: Some(1),
                sampling: SamplingConfig::Greedy,
                required_tool_choice: None,
                json_object_mode: false,
                conversation_mode: false,
                cache_context: BackendCacheContext::default(),
            },
            tx,
            cancellation,
        )
        .expect_err("pre-cancelled stream fails before normal EOF");

    assert!(matches!(err, BackendError::Cancelled));
    std::fs::remove_dir_all(snapshot).ok();
}

#[tokio::test]
async fn native_qwen_worker_stream_reports_join_failure_after_channel_close() {
    let (_tx, rx) = tokio::sync::mpsc::channel(1);
    let worker = tokio::task::spawn_blocking(|| panic!("stream worker panic"));
    let mut stream = native_text_worker_stream("native Qwen", rx, worker);

    let err = stream
        .next()
        .await
        .expect("join failure event")
        .expect_err("worker panic is surfaced");

    assert!(
        err.to_string()
            .contains("native Qwen streaming worker failed")
    );
    assert!(stream.next().await.is_none());
}

#[test]
fn native_qwen_start_decode_session_observes_pre_cancelled_token() {
    let snapshot = temp_snapshot_dir("cancelled-start-decode");
    std::fs::remove_dir_all(&snapshot).ok();
    std::fs::create_dir_all(&snapshot).expect("snapshot dir");
    copy_fixture("config.json", snapshot.join("config.json"));
    copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));
    copy_fixture(
        "model.safetensors.index.json",
        snapshot.join("model.safetensors.index.json"),
    );
    let backend = open_qwen_backend_blocking(crate::DEFAULT_MODEL_ID, &snapshot);
    let cancellation = CancellationToken::new();
    cancellation.cancel();

    match backend.start_decode_session(
        &[0],
        1,
        &native_qwen_test_request(crate::DEFAULT_MODEL_ID),
        &cancellation,
    ) {
        Err(BackendError::Cancelled) => {}
        Err(err) => panic!("expected cancellation before prefill, got {err}"),
        Ok(_) => panic!("pre-cancelled decode startup should fail before prefill"),
    }
    std::fs::remove_dir_all(snapshot).ok();
}

#[test]
fn native_qwen_greedy_returns_top_logit_even_when_it_decodes_to_whitespace() {
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

    let candidate = backend
        .next_token_from_hidden(&[1.0], SamplingConfig::Greedy)
        .expect("greedy candidate");

    assert_eq!(candidate.token_id, 220);
    let decoded = backend
        .driver
        .tokenizer
        .decode(&[candidate.token_id as u32], false)
        .expect("candidate decodes");
    assert!(decoded.trim().is_empty());
    std::fs::remove_dir_all(snapshot).ok();
}

#[test]
fn native_qwen_greedy_clamps_top_k_to_vocab_size() {
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

    let candidate = backend
        .next_token_from_hidden(&[1.0], SamplingConfig::Greedy)
        .expect("greedy candidate with clamped top_k");

    assert_eq!(candidate.token_id, 1);
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
fn copy_fixture(name: &str, destination: impl AsRef<Path>) {
    let source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/qwen36")
        .join(name);
    std::fs::copy(&source, destination).expect("copy fixture");
}

fn write_tiny_qwen3_dense_single_file_decoder_snapshot(root: &Path) {
    std::fs::write(
        root.join("config.json"),
        serde_json::json!({
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "attention_bias": false,
            "hidden_act": "silu",
            "hidden_size": 2,
            "intermediate_size": 1,
            "max_position_embeddings": 16,
            "num_attention_heads": 1,
            "num_hidden_layers": 1,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "rms_norm_eps": 1e-6,
            "rope_scaling": null,
            "rope_theta": 1_000_000,
            "sliding_window": null,
            "tie_word_embeddings": true,
            "use_sliding_window": false,
            "vocab_size": 2
        })
        .to_string(),
    )
    .expect("config");
    std::fs::write(
        root.join("model.safetensors"),
        tiny_multi_safetensors_bf16(&[
            ("model.embed_tokens.weight", &[2, 2], &[1.0, 0.0, 0.0, 1.0]),
            ("model.norm.weight", &[2], &[1.0, 1.0]),
            ("model.layers.0.input_layernorm.weight", &[2], &[1.0, 1.0]),
            (
                "model.layers.0.self_attn.q_proj.weight",
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            (
                "model.layers.0.self_attn.k_proj.weight",
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            (
                "model.layers.0.self_attn.v_proj.weight",
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            ("model.layers.0.self_attn.q_norm.weight", &[2], &[1.0, 1.0]),
            ("model.layers.0.self_attn.k_norm.weight", &[2], &[1.0, 1.0]),
            (
                "model.layers.0.self_attn.o_proj.weight",
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            (
                "model.layers.0.post_attention_layernorm.weight",
                &[2],
                &[1.0, 1.0],
            ),
            ("model.layers.0.mlp.gate_proj.weight", &[1, 2], &[0.0, 0.0]),
            ("model.layers.0.mlp.up_proj.weight", &[1, 2], &[0.0, 0.0]),
            ("model.layers.0.mlp.down_proj.weight", &[2, 1], &[0.0, 0.0]),
        ]),
    )
    .expect("single safetensors");
}

fn write_tiny_qwen3_dense_model_index(root: &Path) {
    let weight_map = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ]
    .into_iter()
    .map(|tensor| {
        (
            tensor.to_owned(),
            serde_json::Value::String("model.safetensors".to_owned()),
        )
    })
    .collect::<serde_json::Map<_, _>>();
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 1 },
            "weight_map": weight_map
        })
        .to_string(),
    )
    .expect("tiny Qwen index");
}

fn tiny_multi_safetensors_bf16(tensors: &[(&str, &[usize], &[f32])]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape, values) in tensors {
        let start = data.len();
        for value in *values {
            data.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
        }
        let end = data.len();
        header.insert(
            (*name).to_owned(),
            serde_json::json!({
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [start, end]
            }),
        );
    }
    let header = serde_json::Value::Object(header).to_string();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&data);
    bytes
}

fn tiny_owned_multi_safetensors_bf16(tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape, values) in tensors {
        let start = data.len();
        for value in values {
            data.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
        }
        let end = data.len();
        header.insert(
            (*name).to_owned(),
            serde_json::json!({
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [start, end]
            }),
        );
    }
    let header = serde_json::Value::Object(header).to_string();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&data);
    bytes
}

fn write_tiny_linear_decoder_snapshot(root: &Path) {
    let tensors = vec![
        (
            "model.language_model.embed_tokens.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "model.language_model.layers.0.input_layernorm.weight",
            vec![2],
            vec![0.0, 0.0],
        ),
        (
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            vec![4, 2],
            vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 4.0],
        ),
        (
            "model.language_model.layers.0.linear_attn.in_proj_z.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "model.language_model.layers.0.linear_attn.in_proj_b.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "model.language_model.layers.0.linear_attn.in_proj_a.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "model.language_model.layers.0.linear_attn.dt_bias",
            vec![1],
            vec![0.0],
        ),
        (
            "model.language_model.layers.0.linear_attn.A_log",
            vec![1],
            vec![0.0],
        ),
        (
            "model.language_model.layers.0.linear_attn.conv1d.weight",
            vec![4, 1],
            vec![1.0, 1.0, 1.0, 1.0],
        ),
        (
            "model.language_model.layers.0.linear_attn.norm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "model.language_model.layers.0.linear_attn.out_proj.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "model.language_model.layers.0.post_attention_layernorm.weight",
            vec![2],
            vec![0.0, 0.0],
        ),
        (
            "model.language_model.layers.0.mlp.gate.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            vec![2, 2],
            vec![0.0, 0.0, 0.0, 0.0],
        ),
        (
            "model.language_model.layers.0.mlp.experts.down_proj",
            vec![2, 1],
            vec![0.0, 0.0],
        ),
        (
            "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
            vec![2, 1],
            vec![0.0, 0.0],
        ),
        (
            "model.language_model.layers.0.mlp.shared_expert_gate.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
    ];
    let mut weight_map = serde_json::Map::new();
    for (name, _, _) in &tensors {
        weight_map.insert(
            (*name).to_owned(),
            serde_json::Value::String("model.safetensors".to_owned()),
        );
    }
    let safetensors = tiny_owned_multi_safetensors_bf16(&tensors);
    std::fs::write(snapshot_path(root, "model.safetensors"), &safetensors)
        .expect("write tiny decoder shard");
    std::fs::write(
        snapshot_path(root, "model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": safetensors.len() },
            "weight_map": serde_json::Value::Object(weight_map)
        })
        .to_string(),
    )
    .expect("write tiny decoder index");
}

fn snapshot_path(root: &Path, name: &str) -> PathBuf {
    root.join(name)
}

fn zero_layer_qwen_spec(hidden_size: u32, vocab_size: u32) -> QwenModelSpec {
    QwenModelSpec {
        family: llm_models::ModelFamily::Qwen,
        architecture: "Qwen3_5MoeForConditionalGeneration".to_owned(),
        model_type: "qwen3_5_moe".to_owned(),
        text_model_type: "qwen3_5_moe_text".to_owned(),
        hidden_size,
        rms_norm_eps: 0.0,
        tie_word_embeddings: false,
        rope_theta: 1_000_000.0,
        partial_rotary_factor: 1.0,
        num_hidden_layers: 0,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: hidden_size,
        linear_num_key_heads: 1,
        linear_num_value_heads: 1,
        linear_key_head_dim: 1,
        linear_value_head_dim: hidden_size,
        linear_conv_kernel_dim: 1,
        num_experts: 1,
        num_experts_per_tok: 1,
        moe_intermediate_size: 1,
        shared_expert_intermediate_size: 1,
        max_position_embeddings: 1,
        vocab_size,
        layer_kinds: Vec::new(),
    }
}

fn tiny_engine_qwen_spec(kind: llm_models::AttentionKind) -> QwenModelSpec {
    QwenModelSpec {
        family: llm_models::ModelFamily::Qwen,
        architecture: "Qwen3_5MoeForConditionalGeneration".to_owned(),
        model_type: "qwen3_5_moe".to_owned(),
        text_model_type: "qwen3_5_moe_text".to_owned(),
        hidden_size: 2,
        rms_norm_eps: 1e-6,
        tie_word_embeddings: false,
        rope_theta: 1_000_000.0,
        partial_rotary_factor: 1.0,
        num_hidden_layers: 1,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 2,
        linear_num_key_heads: 1,
        linear_num_value_heads: 1,
        linear_key_head_dim: 1,
        linear_value_head_dim: 2,
        linear_conv_kernel_dim: 1,
        num_experts: 1,
        num_experts_per_tok: 1,
        moe_intermediate_size: 1,
        shared_expert_intermediate_size: 1,
        max_position_embeddings: 32,
        vocab_size: 2,
        layer_kinds: vec![kind],
    }
}

fn native_qwen_test_backend(
    snapshot: &Path,
    model_id: &str,
    spec: QwenModelSpec,
    max_new_tokens: u32,
    max_prefill_tokens: usize,
    top_k: usize,
    chunk_rows: usize,
) -> NativeQwenBackend {
    let metadata = BackendModelMetadata::new(model_id.to_owned(), "native-qwen");
    let tokenizer =
        HuggingFaceTokenizer::from_file(snapshot.join("tokenizer.json")).expect("tokenizer loads");
    let adapter = NativeQwenAdapter {
        model_id: model_id.to_owned(),
        metadata: metadata.clone(),
        spec,
        store: SafeTensorShardStore::open(snapshot).expect("store opens"),
        matvec: NativeTextMatvecBackend::Cpu,
        max_prefill_tokens,
        top_k,
        chunk_rows,
        prefix_cache: Arc::new(NativeQwenPrefixCache::new(
            DEFAULT_NATIVE_QWEN_PREFIX_CACHE_BYTES,
        )),
    };
    NativeQwenBackend {
        driver: NativeTextDriver::new(
            model_id.to_owned(),
            metadata,
            tokenizer,
            adapter,
            max_new_tokens,
        ),
    }
}

fn native_qwen_test_request(model: &str) -> BackendRequest {
    BackendRequest {
        model: model.to_owned(),
        prompt: "test".to_owned(),
        chat_context: None,
        max_tokens: Some(1),
        sampling: SamplingConfig::Greedy,
        required_tool_choice: None,
        json_object_mode: false,
        conversation_mode: false,
        cache_context: BackendCacheContext::default(),
    }
}
fn native_qwen_test_prefix_namespace(label: &str) -> NativeQwenPrefixCacheNamespace {
    NativeQwenPrefixCacheNamespace {
        model_id: format!("model-{label}"),
        backend: "native-qwen".to_owned(),
        family: Some("qwen".to_owned()),
        loader: Some("safetensors".to_owned()),
        quantization: Some("bf16".to_owned()),
        repo_id: Some("local/test".to_owned()),
        resolved_commit: Some("0123456789abcdef0123456789abcdef01234567".to_owned()),
        profile: Some("qwen-test".to_owned()),
        manifest_digest: Some(format!("digest-{label}")),
        prompt_template: QwenFamilyAdapter.cache_template_id().to_owned(),
        tool_schema: Some("tool-schema-v1".to_owned()),
        request_mode: "conversation=true,json_object=false,required_tool=None".to_owned(),
        cache_layout_version: NATIVE_QWEN_PREFIX_CACHE_LAYOUT_VERSION,
        cache_tokens: 8,
        max_prefill_tokens: 8,
    }
}

fn native_prefix_metric_counter(name: &str) -> u64 {
    native_qwen_prefix_cache_metrics().snapshot()[name]
        .as_u64()
        .unwrap_or_else(|| panic!("prefix metric `{name}` is an unsigned integer"))
}

fn assert_close_vec(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "value {index} differed: actual={actual}, expected={expected}"
        );
    }
}

struct CancelAfterFirstConv {
    cancellation: CancellationToken,
    conv_calls: std::cell::Cell<usize>,
}

impl NativeMatvecBackend for CancelAfterFirstConv {
    async fn bf16_matvec_row_major_f32_in_place(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), TensorLoadError> {
        CpuNativeMatvecBackend
            .bf16_matvec_row_major_f32_in_place(store, tensor, input, output)
            .await
    }

    async fn bf16_matvec_rows_f32_in_place(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        input: &[f32],
        chunk_rows: usize,
        output: &mut [f32],
    ) -> Result<(), TensorLoadError> {
        CpuNativeMatvecBackend
            .bf16_matvec_rows_f32_in_place(store, tensor, input, chunk_rows, output)
            .await
    }

    async fn matvec_row_major_f32_in_place(
        &self,
        input: &[f32],
        weights: &[f32],
        rows: usize,
        columns: usize,
        output: &mut [f32],
    ) -> Result<(), MathError> {
        CpuNativeMatvecBackend
            .matvec_row_major_f32_in_place(input, weights, rows, columns, output)
            .await
    }

    async fn rms_norm_one_centered_f32_in_place(
        &self,
        input: &[f32],
        weight: &[f32],
        eps: f32,
        output: &mut [f32],
    ) -> Result<(), MathError> {
        CpuNativeMatvecBackend
            .rms_norm_one_centered_f32_in_place(input, weight, eps, output)
            .await
    }

    async fn softmax_f32_in_place(
        &self,
        scores: &[f32],
        output: &mut [f32],
    ) -> Result<(), MathError> {
        CpuNativeMatvecBackend
            .softmax_f32_in_place(scores, output)
            .await
    }

    async fn linear_attention_conv1d_silu_f32_in_place(
        &self,
        window: &[f32],
        weights: &[f32],
        conv_dim: usize,
        kernel_size: usize,
        output: &mut [f32],
    ) -> Result<(), MathError> {
        self.conv_calls.set(self.conv_calls.get() + 1);
        if self.conv_calls.get() == 1 {
            self.cancellation.cancel();
        }
        CpuNativeMatvecBackend
            .linear_attention_conv1d_silu_f32_in_place(
                window,
                weights,
                conv_dim,
                kernel_size,
                output,
            )
            .await
    }

    async fn weighted_sum_f32_in_place(
        &self,
        values: &[f32],
        weights: &[f32],
        vector_len: usize,
        output: &mut [f32],
    ) -> Result<(), MathError> {
        CpuNativeMatvecBackend
            .weighted_sum_f32_in_place(values, weights, vector_len, output)
            .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn linear_attention_recurrent_update_f32_in_place(
        &self,
        state: &[f32],
        key: &[f32],
        value: &[f32],
        memory: &[f32],
        beta: f32,
        decay: f32,
        key_head_dim: usize,
        value_head_dim: usize,
        output: &mut [f32],
    ) -> Result<(), MathError> {
        CpuNativeMatvecBackend
            .linear_attention_recurrent_update_f32_in_place(
                state,
                key,
                value,
                memory,
                beta,
                decay,
                key_head_dim,
                value_head_dim,
                output,
            )
            .await
    }

    async fn select_head_rows_f32_in_place(
        &self,
        values: &[f32],
        row_count: usize,
        row_len: usize,
        head_start: usize,
        head_len: usize,
        output: &mut [f32],
    ) -> Result<(), MathError> {
        CpuNativeMatvecBackend
            .select_head_rows_f32_in_place(values, row_count, row_len, head_start, head_len, output)
            .await
    }
}
fn temp_snapshot_dir(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!("llm-engine-{label}-{}", std::process::id()))
}
