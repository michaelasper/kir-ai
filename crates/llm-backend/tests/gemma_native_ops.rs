use llm_backend::{
    CpuNativeMatvecBackend, GemmaLayerCache, InferenceScratchpad, MathError, NativeMatvecBackend,
    NativeTextLayerCaches, NativeTextLayerCachesMut, NativeTextModelSpec, SafeTensorShardStore,
    TensorLoadError, gemma_decode_token_with_cache, gemma_final_norm_for_spec,
    gemma_layer_caches_for_spec, gemma_lm_head_top_k_for_spec, gemma_prefill_sequence_with_cache,
    gemma_prefill_sequence_with_cache_with_matvec,
    native_decode_token_with_cache as native_text_decode_token_with_cache,
    native_decode_token_with_cache_for_spec_ref_with_matvec,
    native_final_norm_for_spec as native_text_final_norm_for_spec,
    native_layer_caches_for_spec as native_text_layer_caches_for_spec,
    native_lm_head_top_k_for_spec as native_text_lm_head_top_k_for_spec,
    native_prefill_sequence_with_cache as native_text_prefill_sequence_with_cache,
    native_prefill_sequence_with_cache_for_spec_ref_with_matvec, qwen_layer_caches_for_spec,
};
use llm_models::{AttentionKind, GemmaModelSpec, ModelFamily, QwenModelSpec};
use serde_json::json;
use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Default)]
struct RecordingGemmaMatvecBackend {
    dense_f32_calls: AtomicUsize,
    softmax_calls: AtomicUsize,
    weighted_sum_calls: AtomicUsize,
    kv_cache_head_row_calls: AtomicUsize,
}

impl NativeMatvecBackend for RecordingGemmaMatvecBackend {
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
        self.dense_f32_calls.fetch_add(1, Ordering::Relaxed);
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
        self.softmax_calls.fetch_add(1, Ordering::Relaxed);
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
        self.weighted_sum_calls.fetch_add(1, Ordering::Relaxed);
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
        self.kv_cache_head_row_calls.fetch_add(1, Ordering::Relaxed);
        CpuNativeMatvecBackend
            .select_head_rows_f32_in_place(values, row_count, row_len, head_start, head_len, output)
            .await
    }
}

#[tokio::test]
async fn gemma_layer_caches_cap_sliding_layers_to_sliding_window() {
    let spec = GemmaModelSpec::from_config_json(&tiny_gemma4_config(1, 8, &["sliding_attention"]))
        .expect("tiny Gemma config parses");

    let caches = gemma_layer_caches_for_spec(&spec, 8).expect("Gemma caches allocate");

    assert_eq!(caches.len(), 1);
    match &caches[0] {
        GemmaLayerCache::Attention(cache) => {
            assert_eq!(cache.max_tokens(), 2);
            assert_eq!(cache.key_value_heads(), 1);
            assert_eq!(cache.head_dim(), 2);
        }
    }
}

#[tokio::test]
async fn gemma_prefill_and_decode_produce_deterministic_tiny_outputs() {
    let root = temp_snapshot_dir("gemma-prefill-decode");
    std::fs::remove_dir_all(&root).ok();
    std::fs::create_dir_all(&root).expect("snapshot dir");
    write_tiny_gemma4_decoder_snapshot(&root);
    let spec = GemmaModelSpec::from_config_json(
        &std::fs::read_to_string(root.join("config.json")).expect("config"),
    )
    .expect("tiny Gemma config parses");
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let mut caches = gemma_layer_caches_for_spec(&spec, 8).expect("Gemma caches allocate");

    let mut scratch = InferenceScratchpad::default();
    let prefill =
        gemma_prefill_sequence_with_cache(&store, &spec, &[0, 1], &mut caches, &mut scratch)
            .await
            .expect("prefill");
    assert_close(&prefill[0], &[2.0_f32.sqrt(), 0.0], 1e-5);
    assert_close(&prefill[1], &[0.0, 2.0_f32.sqrt()], 1e-5);
    match &caches[0] {
        GemmaLayerCache::Attention(cache) => assert_eq!(cache.token_count(), 2),
    }

    let decoded = gemma_decode_token_with_cache(&store, &spec, 2, &mut caches, &mut scratch)
        .await
        .expect("decode token");
    assert_close(&decoded, &[2.0 * 2.0_f32.sqrt(), 0.0], 1e-5);
    match &caches[0] {
        GemmaLayerCache::Attention(cache) => {
            assert_eq!(cache.token_count(), 2);
            assert_eq!(cache.next_position(), 3);
        }
    }

    std::fs::remove_dir_all(root).ok();
}

#[tokio::test]
async fn gemma_prefill_supports_per_layer_inputs() {
    let root = temp_snapshot_dir("gemma-ple-prefill");
    std::fs::remove_dir_all(&root).ok();
    std::fs::create_dir_all(&root).expect("snapshot dir");
    write_tiny_gemma4_decoder_snapshot_with_options(&root, true, 1, &["sliding_attention"], 0);
    let spec = GemmaModelSpec::from_config_json(
        &std::fs::read_to_string(root.join("config.json")).expect("config"),
    )
    .expect("tiny Gemma PLE config parses");
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let mut caches = gemma_layer_caches_for_spec(&spec, 8).expect("Gemma caches allocate");

    let mut scratch = InferenceScratchpad::default();
    let prefill =
        gemma_prefill_sequence_with_cache(&store, &spec, &[0, 1], &mut caches, &mut scratch)
            .await
            .expect("prefill");

    assert!(spec.uses_per_layer_input());
    assert_close(&prefill[0], &[2.0 * 2.0_f32.sqrt(), 0.0], 1e-4);
    assert_close(&prefill[1], &[0.0, 2.0_f32.sqrt()], 1e-5);
    std::fs::remove_dir_all(root).ok();
}

#[tokio::test]
async fn gemma_prefill_reuses_shared_kv_cache_layers() {
    let root = temp_snapshot_dir("gemma-shared-kv");
    std::fs::remove_dir_all(&root).ok();
    std::fs::create_dir_all(&root).expect("snapshot dir");
    write_tiny_gemma4_decoder_snapshot_with_options(
        &root,
        false,
        2,
        &["sliding_attention", "sliding_attention"],
        1,
    );
    let spec = GemmaModelSpec::from_config_json(
        &std::fs::read_to_string(root.join("config.json")).expect("config"),
    )
    .expect("tiny Gemma shared KV config parses");
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let mut caches = gemma_layer_caches_for_spec(&spec, 8).expect("Gemma caches allocate");

    let mut scratch = InferenceScratchpad::default();
    let prefill =
        gemma_prefill_sequence_with_cache(&store, &spec, &[0, 1], &mut caches, &mut scratch)
            .await
            .expect("prefill");

    assert_eq!(caches.len(), 1);
    assert!(spec.is_kv_shared_layer(1));
    assert_close(&prefill[0], &[2.0_f32.sqrt(), 0.0], 1e-5);
    assert_close(&prefill[1], &[0.0, 2.0_f32.sqrt()], 1e-5);
    std::fs::remove_dir_all(root).ok();
}

#[tokio::test]
async fn gemma_attention_uses_configured_matvec_backend_for_shared_and_concrete_layers() {
    let root = temp_snapshot_dir("gemma-attn-matvec-hooks");
    std::fs::remove_dir_all(&root).ok();
    std::fs::create_dir_all(&root).expect("snapshot dir");
    write_tiny_gemma4_decoder_snapshot_with_options(
        &root,
        false,
        2,
        &["sliding_attention", "sliding_attention"],
        1,
    );
    let spec = GemmaModelSpec::from_config_json(
        &std::fs::read_to_string(root.join("config.json")).expect("config"),
    )
    .expect("tiny Gemma shared KV config parses");
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let mut caches = gemma_layer_caches_for_spec(&spec, 8).expect("Gemma caches allocate");
    let matvec = RecordingGemmaMatvecBackend::default();
    let mut scratch = InferenceScratchpad::default();

    let prefill = gemma_prefill_sequence_with_cache_with_matvec(
        &store,
        &spec,
        &[0, 1],
        &mut caches,
        &matvec,
        &mut scratch,
    )
    .await
    .expect("prefill with recording matvec");

    assert_eq!(prefill.len(), 2);
    assert_eq!(matvec.softmax_calls.load(Ordering::Relaxed), 4);
    assert_eq!(matvec.weighted_sum_calls.load(Ordering::Relaxed), 4);
    assert_eq!(matvec.kv_cache_head_row_calls.load(Ordering::Relaxed), 8);
    assert_eq!(matvec.dense_f32_calls.load(Ordering::Relaxed), 8);
    std::fs::remove_dir_all(root).ok();
}

#[tokio::test]
async fn gemma_final_norm_and_tied_lm_head_select_top_token() {
    let root = temp_snapshot_dir("gemma-lm-head");
    std::fs::remove_dir_all(&root).ok();
    std::fs::create_dir_all(&root).expect("snapshot dir");
    write_tiny_gemma4_decoder_snapshot(&root);
    let spec = GemmaModelSpec::from_config_json(
        &std::fs::read_to_string(root.join("config.json")).expect("config"),
    )
    .expect("tiny Gemma config parses");
    let store = SafeTensorShardStore::open(&root).expect("store opens");

    let mut final_norm = vec![0.0; 2];
    gemma_final_norm_for_spec(&store, &spec, &[2.0 * 2.0_f32.sqrt(), 0.0], &mut final_norm)
        .await
        .expect("norm");
    assert_close(&final_norm, &[2.0_f32.sqrt(), 0.0], 1e-5);
    let top = gemma_lm_head_top_k_for_spec(&store, &spec, &final_norm, 2, 64)
        .await
        .expect("top logits");

    assert_eq!(top[0].index, 2);
    assert!((top[0].logit - 2.0 * 2.0_f32.sqrt()).abs() < 1e-5);
    std::fs::remove_dir_all(root).ok();
}

#[tokio::test]
async fn native_text_dispatch_matches_direct_gemma_prefill_decode_and_lm_head() {
    let root = temp_snapshot_dir("native-text-dispatch-gemma");
    std::fs::remove_dir_all(&root).ok();
    std::fs::create_dir_all(&root).expect("snapshot dir");
    write_tiny_gemma4_decoder_snapshot(&root);
    let spec = GemmaModelSpec::from_config_json(
        &std::fs::read_to_string(root.join("config.json")).expect("config"),
    )
    .expect("tiny Gemma config parses");
    let native_spec = NativeTextModelSpec::Gemma(spec.clone());
    let store = SafeTensorShardStore::open(&root).expect("store opens");

    let mut direct_caches = gemma_layer_caches_for_spec(&spec, 8).expect("direct caches");
    let mut direct_scratch = InferenceScratchpad::default();
    let direct_prefill = gemma_prefill_sequence_with_cache(
        &store,
        &spec,
        &[0, 1],
        &mut direct_caches,
        &mut direct_scratch,
    )
    .await
    .expect("direct prefill");
    let direct_decode =
        gemma_decode_token_with_cache(&store, &spec, 2, &mut direct_caches, &mut direct_scratch)
            .await
            .expect("direct decode");

    let mut native_caches =
        native_text_layer_caches_for_spec(&native_spec, 8).expect("native text caches");
    assert!(matches!(native_caches, NativeTextLayerCaches::Gemma(_)));
    let mut native_scratch = InferenceScratchpad::default();
    let native_prefill = native_text_prefill_sequence_with_cache(
        &store,
        &native_spec,
        &[0, 1],
        &mut native_caches,
        &mut native_scratch,
    )
    .await
    .expect("native text prefill");
    let native_decode = native_text_decode_token_with_cache(
        &store,
        &native_spec,
        2,
        &mut native_caches,
        &mut native_scratch,
    )
    .await
    .expect("native text decode");
    let mut ref_caches = gemma_layer_caches_for_spec(&spec, 8).expect("spec-ref caches");
    let mut ref_scratch = InferenceScratchpad::default();
    let ref_prefill = native_prefill_sequence_with_cache_for_spec_ref_with_matvec(
        &store,
        (&spec).into(),
        &[0, 1],
        NativeTextLayerCachesMut::Gemma(&mut ref_caches),
        &CpuNativeMatvecBackend,
        &mut ref_scratch,
    )
    .await
    .expect("native text spec-ref prefill");
    let ref_decode = native_decode_token_with_cache_for_spec_ref_with_matvec(
        &store,
        (&spec).into(),
        2,
        NativeTextLayerCachesMut::Gemma(&mut ref_caches),
        &CpuNativeMatvecBackend,
        &mut ref_scratch,
    )
    .await
    .expect("native text spec-ref decode");
    let native_norm = native_text_final_norm_for_spec(&store, &native_spec, &native_decode)
        .await
        .expect("native norm");
    let native_top = native_text_lm_head_top_k_for_spec(&store, &native_spec, &native_norm, 2, 64)
        .await
        .expect("native top logits");

    assert_eq!(native_prefill.len(), direct_prefill.len());
    assert_close(&native_prefill[0], &direct_prefill[0], 1e-5);
    assert_close(&native_prefill[1], &direct_prefill[1], 1e-5);
    assert_close(&native_decode, &direct_decode, 1e-5);
    assert_eq!(ref_prefill.len(), direct_prefill.len());
    assert_close(&ref_prefill[0], &direct_prefill[0], 1e-5);
    assert_close(&ref_prefill[1], &direct_prefill[1], 1e-5);
    assert_close(&ref_decode, &direct_decode, 1e-5);
    assert_eq!(native_top[0].index, 2);
    std::fs::remove_dir_all(root).ok();
}

#[tokio::test]
async fn native_text_spec_ref_rejects_mismatched_cache_families() {
    let root = temp_snapshot_dir("native-text-cache-family-mismatch");
    std::fs::remove_dir_all(&root).ok();
    std::fs::create_dir_all(&root).expect("snapshot dir");
    write_tiny_gemma4_decoder_snapshot(&root);
    let gemma_spec = GemmaModelSpec::from_config_json(
        &std::fs::read_to_string(root.join("config.json")).expect("config"),
    )
    .expect("tiny Gemma config parses");
    let qwen_spec = tiny_qwen_spec(AttentionKind::LinearAttention);
    let store = SafeTensorShardStore::open(&root).expect("store opens");

    let mut gemma_prefill_caches =
        gemma_layer_caches_for_spec(&gemma_spec, 8).expect("Gemma prefill caches");
    let mut scratch = InferenceScratchpad::default();
    let err = native_prefill_sequence_with_cache_for_spec_ref_with_matvec(
        &store,
        (&qwen_spec).into(),
        &[0, 1],
        NativeTextLayerCachesMut::Gemma(&mut gemma_prefill_caches),
        &CpuNativeMatvecBackend,
        &mut scratch,
    )
    .await
    .expect_err("Qwen prefill rejects Gemma caches");
    assert_cache_family_mismatch(err, "prefill", "gemma", "qwen");

    let mut gemma_decode_caches =
        gemma_layer_caches_for_spec(&gemma_spec, 8).expect("Gemma decode caches");
    let mut scratch = InferenceScratchpad::default();
    let err = native_decode_token_with_cache_for_spec_ref_with_matvec(
        &store,
        (&qwen_spec).into(),
        0,
        NativeTextLayerCachesMut::Gemma(&mut gemma_decode_caches),
        &CpuNativeMatvecBackend,
        &mut scratch,
    )
    .await
    .expect_err("Qwen decode rejects Gemma caches");
    assert_cache_family_mismatch(err, "decode", "gemma", "qwen");

    let mut qwen_prefill_caches =
        qwen_layer_caches_for_spec(&qwen_spec, 8).expect("Qwen prefill caches");
    let mut scratch = InferenceScratchpad::default();
    let err = native_prefill_sequence_with_cache_for_spec_ref_with_matvec(
        &store,
        (&gemma_spec).into(),
        &[0, 1],
        NativeTextLayerCachesMut::Qwen(&mut qwen_prefill_caches),
        &CpuNativeMatvecBackend,
        &mut scratch,
    )
    .await
    .expect_err("Gemma prefill rejects Qwen caches");
    assert_cache_family_mismatch(err, "prefill", "qwen", "gemma");

    let mut qwen_decode_caches =
        qwen_layer_caches_for_spec(&qwen_spec, 8).expect("Qwen decode caches");
    let mut scratch = InferenceScratchpad::default();
    let err = native_decode_token_with_cache_for_spec_ref_with_matvec(
        &store,
        (&gemma_spec).into(),
        0,
        NativeTextLayerCachesMut::Qwen(&mut qwen_decode_caches),
        &CpuNativeMatvecBackend,
        &mut scratch,
    )
    .await
    .expect_err("Gemma decode rejects Qwen caches");
    assert_cache_family_mismatch(err, "decode", "qwen", "gemma");

    std::fs::remove_dir_all(root).ok();
}

fn write_tiny_gemma4_decoder_snapshot(root: &Path) {
    write_tiny_gemma4_decoder_snapshot_with_options(root, false, 1, &["sliding_attention"], 0);
}

fn write_tiny_gemma4_decoder_snapshot_with_options(
    root: &Path,
    per_layer_inputs: bool,
    num_hidden_layers: u32,
    layer_types: &[&str],
    num_kv_shared_layers: u32,
) {
    std::fs::write(
        root.join("config.json"),
        tiny_gemma4_config_with_options(
            num_hidden_layers,
            8,
            layer_types,
            per_layer_inputs,
            num_kv_shared_layers,
        ),
    )
    .expect("config");
    let layer_count = num_hidden_layers as usize;
    let mut per_layer_embedding = vec![0.0; 3 * layer_count];
    if per_layer_inputs {
        per_layer_embedding[0] = 1.0;
    }
    let mut tensors = vec![
        (
            "model.language_model.embed_tokens.weight",
            vec![3, 2],
            vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0],
        ),
        ("model.language_model.norm.weight", vec![2], vec![1.0, 1.0]),
        (
            "model.language_model.layers.0.input_layernorm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "model.language_model.layers.0.self_attn.q_proj.weight",
            vec![2, 2],
            vec![0.0, 0.0, 0.0, 0.0],
        ),
        (
            "model.language_model.layers.0.self_attn.k_proj.weight",
            vec![2, 2],
            vec![0.0, 0.0, 0.0, 0.0],
        ),
        (
            "model.language_model.layers.0.self_attn.v_proj.weight",
            vec![2, 2],
            vec![0.0, 0.0, 0.0, 0.0],
        ),
        (
            "model.language_model.layers.0.self_attn.q_norm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "model.language_model.layers.0.self_attn.k_norm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "model.language_model.layers.0.self_attn.o_proj.weight",
            vec![2, 2],
            vec![0.0, 0.0, 0.0, 0.0],
        ),
        (
            "model.language_model.layers.0.post_attention_layernorm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "model.language_model.layers.0.pre_feedforward_layernorm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "model.language_model.layers.0.mlp.gate_proj.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "model.language_model.layers.0.mlp.up_proj.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "model.language_model.layers.0.mlp.down_proj.weight",
            vec![2, 1],
            vec![0.0, 0.0],
        ),
        (
            "model.language_model.layers.0.post_feedforward_layernorm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "model.language_model.layers.0.layer_scalar",
            vec![1],
            vec![1.0],
        ),
    ];
    if per_layer_inputs {
        tensors.push((
            "model.language_model.embed_tokens_per_layer.weight",
            vec![3, layer_count],
            per_layer_embedding,
        ));
        tensors.push((
            "model.language_model.per_layer_model_projection.weight",
            vec![layer_count, 2],
            vec![0.0; 2 * layer_count],
        ));
        tensors.push((
            "model.language_model.per_layer_projection_norm.weight",
            vec![1],
            vec![1.0],
        ));
        tensors.push((
            "model.language_model.layers.0.per_layer_input_gate.weight",
            vec![1, 2],
            vec![1.0, 0.0],
        ));
        tensors.push((
            "model.language_model.layers.0.per_layer_projection.weight",
            vec![2, 1],
            vec![1.0, 0.0],
        ));
        tensors.push((
            "model.language_model.layers.0.post_per_layer_input_norm.weight",
            vec![2],
            vec![1.0, 1.0],
        ));
    }
    if num_hidden_layers > 1 {
        tensors.extend([
            (
                "model.language_model.layers.1.input_layernorm.weight",
                vec![2],
                vec![1.0, 1.0],
            ),
            (
                "model.language_model.layers.1.self_attn.q_proj.weight",
                vec![2, 2],
                vec![0.0, 0.0, 0.0, 0.0],
            ),
            (
                "model.language_model.layers.1.self_attn.q_norm.weight",
                vec![2],
                vec![1.0, 1.0],
            ),
            (
                "model.language_model.layers.1.self_attn.o_proj.weight",
                vec![2, 2],
                vec![0.0, 0.0, 0.0, 0.0],
            ),
            (
                "model.language_model.layers.1.post_attention_layernorm.weight",
                vec![2],
                vec![1.0, 1.0],
            ),
            (
                "model.language_model.layers.1.pre_feedforward_layernorm.weight",
                vec![2],
                vec![1.0, 1.0],
            ),
            (
                "model.language_model.layers.1.mlp.gate_proj.weight",
                vec![1, 2],
                vec![0.0, 0.0],
            ),
            (
                "model.language_model.layers.1.mlp.up_proj.weight",
                vec![1, 2],
                vec![0.0, 0.0],
            ),
            (
                "model.language_model.layers.1.mlp.down_proj.weight",
                vec![2, 1],
                vec![0.0, 0.0],
            ),
            (
                "model.language_model.layers.1.post_feedforward_layernorm.weight",
                vec![2],
                vec![1.0, 1.0],
            ),
            (
                "model.language_model.layers.1.layer_scalar",
                vec![1],
                vec![1.0],
            ),
        ]);
        if per_layer_inputs {
            tensors.extend([
                (
                    "model.language_model.layers.1.per_layer_input_gate.weight",
                    vec![1, 2],
                    vec![0.0, 0.0],
                ),
                (
                    "model.language_model.layers.1.per_layer_projection.weight",
                    vec![2, 1],
                    vec![0.0, 0.0],
                ),
                (
                    "model.language_model.layers.1.post_per_layer_input_norm.weight",
                    vec![2],
                    vec![1.0, 1.0],
                ),
            ]);
        }
    }
    let safetensors = tiny_owned_multi_safetensors_bf16(&tensors);
    std::fs::write(root.join("model.safetensors"), &safetensors).expect("safetensors");
    let weight_map = tensors
        .iter()
        .map(|(tensor, _, _)| {
            (
                (*tensor).to_owned(),
                serde_json::Value::String("model.safetensors".to_owned()),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    std::fs::write(
        root.join("model.safetensors.index.json"),
        json!({
            "metadata": {"total_size": safetensors.len()},
            "weight_map": weight_map
        })
        .to_string(),
    )
    .expect("index");
}

fn tiny_gemma4_config(
    num_hidden_layers: u32,
    max_position_embeddings: u32,
    layer_types: &[&str],
) -> String {
    tiny_gemma4_config_with_options(
        num_hidden_layers,
        max_position_embeddings,
        layer_types,
        false,
        0,
    )
}

fn tiny_gemma4_config_with_options(
    num_hidden_layers: u32,
    max_position_embeddings: u32,
    layer_types: &[&str],
    per_layer_inputs: bool,
    num_kv_shared_layers: u32,
) -> String {
    json!({
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "text_config": {
            "attention_bias": false,
            "attention_dropout": 0.0,
            "attention_k_eq_v": false,
            "bos_token_id": 2,
            "dtype": "bfloat16",
            "enable_moe_block": false,
            "global_head_dim": null,
            "head_dim": 2,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 2,
            "hidden_size_per_layer_input": if per_layer_inputs { 1 } else { 0 },
            "intermediate_size": 1,
            "layer_types": layer_types,
            "max_position_embeddings": max_position_embeddings,
            "model_type": "gemma4_text",
            "num_attention_heads": 1,
            "num_global_key_value_heads": null,
            "num_hidden_layers": num_hidden_layers,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": num_kv_shared_layers,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "full_attention": {"partial_rotary_factor": 1.0, "rope_theta": 10000.0},
                "sliding_attention": {"rope_theta": 10000.0}
            },
            "sliding_window": 2,
            "tie_word_embeddings": true,
            "use_double_wide_mlp": false,
            "vocab_size": 3,
            "vocab_size_per_layer_input": 3
        },
        "tie_word_embeddings": true
    })
    .to_string()
}

fn tiny_qwen_spec(kind: AttentionKind) -> QwenModelSpec {
    QwenModelSpec {
        family: ModelFamily::Qwen,
        architecture: "Qwen3_5MoeForConditionalGeneration".to_owned(),
        model_type: "qwen3_5_moe".to_owned(),
        text_model_type: "qwen3_5_moe_text".to_owned(),
        hidden_size: 2,
        rms_norm_eps: 1e-6,
        tie_word_embeddings: false,
        rope_theta: 10_000.0,
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
        max_position_embeddings: 128,
        vocab_size: 8,
        layer_kinds: vec![kind],
    }
}

fn assert_cache_family_mismatch(
    err: TensorLoadError,
    operation: &str,
    cache_family: &str,
    spec_family: &str,
) {
    assert_eq!(err.code(), "unsupported_capability");
    assert!(
        err.to_string().contains(&format!(
            "native text {operation} received `{cache_family}` caches for `{spec_family}` spec"
        )),
        "{err}"
    );
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
            json!({
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

fn temp_snapshot_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("kir-ai-{name}-{nanos}"))
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "index {idx}: expected {expected}, got {actual}"
        );
    }
}
