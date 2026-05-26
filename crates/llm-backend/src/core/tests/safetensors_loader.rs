use super::super::math::{InferenceScratchpad, MathError};
use super::super::native_matvec::{CpuNativeMatvecBackend, NativeMatvecBackend};
use super::super::qwen::ops::{
    QWEN_EMBED_TOKENS_WEIGHT, QWEN_FINAL_NORM_WEIGHT, QWEN_LAYER0_INPUT_NORM_WEIGHT,
    QwenLayerCache, qwen_decode_token_with_cache, qwen_embedding_and_layer0_norm,
    qwen_embedding_sequence_for_spec, qwen_final_norm, qwen_final_norm_for_spec,
    qwen_layer_caches_for_spec, qwen_layer_full_attention_sequence_with_cache,
    qwen_layer_linear_attention_sequence_with_cache, qwen_lm_head_logits,
    qwen_lm_head_logits_for_spec, qwen_lm_head_top_k, qwen_lm_head_top_k_for_spec,
    qwen_prefill_sequence_with_cache, qwen_static_f32_tensors_for_spec,
};
use super::super::safetensors::{SafeTensorShardStore, TensorLoadError};
use llm_models::{AttentionKind, ModelFamily, QwenModelSpec};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

// #[path = "safetensors_loader/metadata.rs"]
// mod metadata;
// #[path = "safetensors_loader/qwen_attention.rs"]
// mod qwen_attention;
// #[path = "safetensors_loader/qwen_core.rs"]
// mod qwen_core;
#[path = "safetensors_loader/qwen_dense.rs"]
mod qwen_dense;
#[path = "safetensors_loader/qwen_lm_head.rs"]
mod qwen_lm_head;
// #[path = "safetensors_loader/qwen_moe.rs"]
// mod qwen_moe;
#[path = "safetensors_loader/shard_store.rs"]
mod shard_store;

#[derive(Default)]
struct RecordingMatvecBackend {
    single_bf16_calls: AtomicUsize,
    batched_bf16_calls: AtomicUsize,
    rows_bf16_calls: AtomicUsize,
    range_bf16_calls: AtomicUsize,
    top_k_bf16_calls: AtomicUsize,
    bf16_output_projection_calls: AtomicUsize,
    dense_output_projection_calls: AtomicUsize,
    dense_f32_calls: AtomicUsize,
    rms_norm_calls: AtomicUsize,
    softmax_calls: AtomicUsize,
    conv1d_calls: AtomicUsize,
    softmax_top_k_calls: AtomicUsize,
    weighted_sum_calls: AtomicUsize,
    recurrent_update_calls: AtomicUsize,
    recurrent_cache_update_calls: AtomicUsize,
    head_row_calls: AtomicUsize,
    kv_cache_head_row_calls: AtomicUsize,
}

impl NativeMatvecBackend for RecordingMatvecBackend {
    async fn bf16_matvec_row_major_f32_in_place(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), TensorLoadError> {
        self.single_bf16_calls.fetch_add(1, Ordering::Relaxed);
        if tensor.ends_with("self_attn.o_proj.weight")
            || tensor.ends_with("linear_attn.out_proj.weight")
        {
            self.bf16_output_projection_calls
                .fetch_add(1, Ordering::Relaxed);
        }
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
        self.rows_bf16_calls.fetch_add(1, Ordering::Relaxed);
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
        if rows == 2 && columns == 2 {
            self.dense_output_projection_calls
                .fetch_add(1, Ordering::Relaxed);
        }
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
        self.rms_norm_calls.fetch_add(1, Ordering::Relaxed);
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
        self.conv1d_calls.fetch_add(1, Ordering::Relaxed);
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
        self.recurrent_update_calls.fetch_add(1, Ordering::Relaxed);
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
        self.head_row_calls.fetch_add(1, Ordering::Relaxed);
        CpuNativeMatvecBackend
            .select_head_rows_f32_in_place(values, row_count, row_len, head_start, head_len, output)
            .await
    }
}

fn caches_for_spec(spec: &QwenModelSpec, capacity: usize) -> Vec<QwenLayerCache> {
    qwen_layer_caches_for_spec(spec, capacity).expect("temporary caches")
}

#[tokio::test]
async fn qwen_embedding_probe_rejects_token_id_outside_embedding_vocab() {
    let root = temp_snapshot_dir("qwen-embed-token-bounds");
    std::fs::remove_dir_all(&root).ok();
    std::fs::create_dir_all(&root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 20 },
            "weight_map": {
                QWEN_EMBED_TOKENS_WEIGHT: "embed.safetensors",
                QWEN_LAYER0_INPUT_NORM_WEIGHT: "norm.safetensors"
            }
        })
        .to_string(),
    )
    .expect("index");
    std::fs::write(
        root.join("embed.safetensors"),
        tiny_safetensors_bf16(QWEN_EMBED_TOKENS_WEIGHT, &[2, 2], &[3.0, 4.0, 6.0, 8.0]),
    )
    .expect("embedding shard");
    std::fs::write(
        root.join("norm.safetensors"),
        tiny_safetensors_bf16(QWEN_LAYER0_INPUT_NORM_WEIGHT, &[2], &[0.0, 1.0]),
    )
    .expect("norm shard");
    let store = SafeTensorShardStore::open(&root).expect("store opens");

    let err = qwen_embedding_and_layer0_norm(&store, 2, 2, 0.0)
        .expect_err("token id equal to vocab size is rejected");

    assert_eq!(err.code(), "model_integrity_failed");
    assert!(
        err.to_string()
            .contains("Qwen token id 2 is outside vocab size 2"),
        "{err}"
    );
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_spec_embedding_rejects_token_id_outside_configured_vocab() {
    let root = temp_snapshot_dir("qwen-spec-embed-token-bounds");
    std::fs::remove_dir_all(&root).ok();
    write_tiny_linear_decoder_snapshot(&root);
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::LinearAttention);

    let err = qwen_embedding_sequence_for_spec(&store, &spec, &[spec.vocab_size as usize])
        .expect_err("token id equal to configured vocab size is rejected");

    assert_eq!(err.code(), "model_integrity_failed");
    assert!(
        err.to_string()
            .contains("Qwen token id 8 at position 0 is outside vocab size 8"),
        "{err}"
    );
    std::fs::remove_dir_all(root).ok();
}

#[tokio::test]
async fn qwen_decode_rejects_token_id_outside_configured_vocab() {
    let root = temp_snapshot_dir("qwen-decode-token-bounds");
    std::fs::remove_dir_all(&root).ok();
    write_tiny_linear_decoder_snapshot(&root);
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::LinearAttention);
    let mut caches = caches_for_spec(&spec, 4);
    let mut scratch = InferenceScratchpad::default();

    let err = qwen_decode_token_with_cache(
        &store,
        &spec,
        spec.vocab_size as usize,
        &mut caches,
        &CpuNativeMatvecBackend,
        &mut scratch,
    )
    .await
    .expect_err("decode rejects token id equal to configured vocab size");

    assert_eq!(err.code(), "model_integrity_failed");
    assert!(
        err.to_string()
            .contains("Qwen token id 8 is outside vocab size 8"),
        "{err}"
    );
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn qwen_static_f32_tensor_list_excludes_output_projections() {
    let spec = QwenModelSpec {
        num_hidden_layers: 2,
        layer_kinds: vec![AttentionKind::LinearAttention, AttentionKind::FullAttention],
        ..tiny_qwen_spec(AttentionKind::LinearAttention)
    };

    let tensors = qwen_static_f32_tensors_for_spec(&spec);

    assert!(tensors.contains(&"model.language_model.norm.weight".to_owned()));
    assert!(tensors.contains(&"model.language_model.layers.0.input_layernorm.weight".to_owned()));
    assert!(tensors.contains(&"model.language_model.layers.0.linear_attn.dt_bias".to_owned()));
    assert!(tensors.contains(&"model.language_model.layers.0.linear_attn.A_log".to_owned()));
    assert!(
        tensors.contains(&"model.language_model.layers.0.linear_attn.conv1d.weight".to_owned())
    );
    assert!(tensors.contains(&"model.language_model.layers.0.linear_attn.norm.weight".to_owned()));
    assert!(tensors.contains(&"model.language_model.layers.1.self_attn.q_norm.weight".to_owned()));
    assert!(tensors.contains(&"model.language_model.layers.1.self_attn.k_norm.weight".to_owned()));
    assert!(
        !tensors
            .iter()
            .any(|tensor| tensor.ends_with("self_attn.o_proj.weight"))
    );
    assert!(
        !tensors
            .iter()
            .any(|tensor| tensor.ends_with("linear_attn.out_proj.weight"))
    );
}

#[tokio::test]
async fn qwen_full_attention_output_projection_uses_bf16_matvec() {
    let root = temp_snapshot_dir("qwen-full-output-projection-bf16");
    std::fs::remove_dir_all(&root).ok();
    write_tiny_full_attention_snapshot(&root);
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::FullAttention);
    let mut caches = caches_for_spec(&spec, 4);
    let QwenLayerCache::Full(cache) = &mut caches[0] else {
        panic!("full attention cache");
    };
    let matvec = RecordingMatvecBackend::default();

    qwen_layer_full_attention_sequence_with_cache(
        &store,
        &spec,
        0,
        &[vec![1.0, 0.0]],
        cache,
        &matvec,
    )
    .await
    .expect("full attention succeeds");

    assert_eq!(
        matvec.bf16_output_projection_calls.load(Ordering::Relaxed),
        1
    );
    assert_eq!(
        matvec.dense_output_projection_calls.load(Ordering::Relaxed),
        0
    );
    std::fs::remove_dir_all(root).ok();
}

#[tokio::test]
async fn qwen_linear_attention_output_projection_uses_bf16_matvec() {
    let root = temp_snapshot_dir("qwen-linear-output-projection-bf16");
    std::fs::remove_dir_all(&root).ok();
    write_tiny_linear_decoder_snapshot(&root);
    let store = SafeTensorShardStore::open(&root).expect("store opens");
    let spec = tiny_qwen_spec(AttentionKind::LinearAttention);
    let mut caches = caches_for_spec(&spec, 4);
    let QwenLayerCache::Linear(cache) = &mut caches[0] else {
        panic!("linear attention cache");
    };
    let matvec = RecordingMatvecBackend::default();

    qwen_layer_linear_attention_sequence_with_cache(
        &store,
        &spec,
        0,
        &[vec![1.0, 0.0]],
        cache,
        &matvec,
    )
    .await
    .expect("linear attention succeeds");

    assert_eq!(
        matvec.bf16_output_projection_calls.load(Ordering::Relaxed),
        1
    );
    assert_eq!(
        matvec.dense_output_projection_calls.load(Ordering::Relaxed),
        0
    );
    std::fs::remove_dir_all(root).ok();
}

fn write_tiny_moe_forward_snapshot(root: &std::path::Path) {
    std::fs::create_dir_all(root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 80 },
            "weight_map": {
                "model.language_model.layers.0.mlp.experts.gate_up_proj": "gate_up.safetensors",
                "model.language_model.layers.0.mlp.experts.down_proj": "down.safetensors",
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
            "gate_up.safetensors",
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            vec![2, 2, 2],
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        ),
        (
            "down.safetensors",
            "model.language_model.layers.0.mlp.experts.down_proj",
            vec![2, 2, 1],
            vec![1.0, 2.0, 3.0, 4.0],
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
}

fn write_tiny_full_attention_snapshot(root: &std::path::Path) {
    std::fs::create_dir_all(root).expect("snapshot dir");
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
}

fn write_tiny_linear_decoder_snapshot(root: &std::path::Path) {
    std::fs::create_dir_all(root).expect("snapshot dir");
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
}

fn write_tiny_qwen3_dense_decoder_snapshot(root: &std::path::Path) {
    std::fs::create_dir_all(root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 128 },
            "weight_map": {
                "model.embed_tokens.weight": "embed.safetensors",
                "model.layers.0.input_layernorm.weight": "input_norm.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "q.safetensors",
                "model.layers.0.self_attn.k_proj.weight": "k.safetensors",
                "model.layers.0.self_attn.v_proj.weight": "v.safetensors",
                "model.layers.0.self_attn.q_norm.weight": "q_norm.safetensors",
                "model.layers.0.self_attn.k_norm.weight": "k_norm.safetensors",
                "model.layers.0.self_attn.o_proj.weight": "o.safetensors",
                "model.layers.0.post_attention_layernorm.weight": "post_norm.safetensors",
                "model.layers.0.mlp.gate_proj.weight": "gate.safetensors",
                "model.layers.0.mlp.up_proj.weight": "up.safetensors",
                "model.layers.0.mlp.down_proj.weight": "down.safetensors"
            }
        })
        .to_string(),
    )
    .expect("index");
    for (filename, tensor, shape, values) in [
        (
            "embed.safetensors",
            "model.embed_tokens.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "input_norm.safetensors",
            "model.layers.0.input_layernorm.weight",
            vec![2],
            vec![0.0, 0.0],
        ),
        (
            "q.safetensors",
            "model.layers.0.self_attn.q_proj.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "k.safetensors",
            "model.layers.0.self_attn.k_proj.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "v.safetensors",
            "model.layers.0.self_attn.v_proj.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "q_norm.safetensors",
            "model.layers.0.self_attn.q_norm.weight",
            vec![2],
            vec![0.0, 0.0],
        ),
        (
            "k_norm.safetensors",
            "model.layers.0.self_attn.k_norm.weight",
            vec![2],
            vec![0.0, 0.0],
        ),
        (
            "o.safetensors",
            "model.layers.0.self_attn.o_proj.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "post_norm.safetensors",
            "model.layers.0.post_attention_layernorm.weight",
            vec![2],
            vec![0.0, 0.0],
        ),
        (
            "gate.safetensors",
            "model.layers.0.mlp.gate_proj.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "up.safetensors",
            "model.layers.0.mlp.up_proj.weight",
            vec![1, 2],
            vec![0.0, 0.0],
        ),
        (
            "down.safetensors",
            "model.layers.0.mlp.down_proj.weight",
            vec![2, 1],
            vec![0.0, 0.0],
        ),
    ] {
        std::fs::write(
            root.join(filename),
            tiny_safetensors_bf16(tensor, &shape, &values),
        )
        .expect("tensor");
    }
}

fn write_tiny_qwen3_dense_lm_head_snapshot(root: &std::path::Path) {
    std::fs::create_dir_all(root).expect("snapshot dir");
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({
            "metadata": { "total_size": 16 },
            "weight_map": {
                "model.embed_tokens.weight": "embed.safetensors",
                "model.norm.weight": "norm.safetensors"
            }
        })
        .to_string(),
    )
    .expect("index");
    for (filename, tensor, shape, values) in [
        (
            "embed.safetensors",
            "model.embed_tokens.weight",
            vec![2, 2],
            vec![2.0, 0.0, 0.0, 1.0],
        ),
        (
            "norm.safetensors",
            "model.norm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
    ] {
        std::fs::write(
            root.join(filename),
            tiny_safetensors_bf16(tensor, &shape, &values),
        )
        .expect("tensor");
    }
}

fn tiny_safetensors_f32(name: &str, shape: &[usize], values: &[f32]) -> Vec<u8> {
    let mut data = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        data.extend_from_slice(&value.to_le_bytes());
    }
    tiny_safetensors(name, "F32", shape, &data)
}

fn tiny_safetensors_bf16(name: &str, shape: &[usize], values: &[f32]) -> Vec<u8> {
    let mut data = Vec::with_capacity(values.len() * 2);
    for value in values {
        data.extend_from_slice(&bf16_bits(*value).to_le_bytes());
    }
    tiny_safetensors(name, "BF16", shape, &data)
}

fn bf16_bits(value: f32) -> u16 {
    (value.to_bits() >> 16) as u16
}

fn tiny_safetensors(name: &str, dtype: &str, shape: &[usize], data: &[u8]) -> Vec<u8> {
    let data_len = data.len();
    let header = serde_json::json!({
        name: {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [0, data_len]
        }
    })
    .to_string();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(data);
    bytes
}

fn temp_safetensors_path(label: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "llm-backend-{label}-{}.safetensors",
        std::process::id()
    ))
}

fn temp_snapshot_dir(label: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("llm-backend-{label}-{}", std::process::id()))
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
        sliding_window: None,
        vocab_size: 8,
        layer_kinds: vec![kind],
    }
}

fn tiny_qwen3_dense_spec() -> QwenModelSpec {
    QwenModelSpec {
        family: ModelFamily::Qwen,
        architecture: "Qwen3ForCausalLM".to_owned(),
        model_type: "qwen3".to_owned(),
        text_model_type: "qwen3".to_owned(),
        hidden_size: 2,
        rms_norm_eps: 1e-6,
        tie_word_embeddings: true,
        rope_theta: 10_000.0,
        partial_rotary_factor: 1.0,
        num_hidden_layers: 1,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 2,
        linear_num_key_heads: 0,
        linear_num_value_heads: 0,
        linear_key_head_dim: 0,
        linear_value_head_dim: 0,
        linear_conv_kernel_dim: 0,
        num_experts: 0,
        num_experts_per_tok: 0,
        moe_intermediate_size: 1,
        shared_expert_intermediate_size: 0,
        max_position_embeddings: 128,
        sliding_window: None,
        vocab_size: 8,
        layer_kinds: vec![AttentionKind::FullAttention],
    }
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual} expected {expected}"
        );
    }
}
