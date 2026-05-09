use llm_backend::{
    CpuQwenMatvecBackend, MathError, NativeMatvecBackend, QWEN_FINAL_NORM_WEIGHT,
    QwenKvCacheTensor, QwenLayerCache, SafeTensorArchive, SafeTensorFile, SafeTensorHeader,
    SafeTensorShardStore, TensorLoadError, TopKLogit, native_decode_token_with_cache,
    native_layer_caches_for_spec, native_prefill_sequence_with_cache, qwen_decode_token_with_cache,
    qwen_decode_token_with_cache_with_matvec, qwen_embedding_and_layer0_norm, qwen_final_norm,
    qwen_final_norm_for_spec, qwen_final_norm_with_matvec, qwen_layer_caches_for_spec,
    qwen_layer_full_attention_first_token, qwen_layer_full_attention_sequence,
    qwen_layer_full_attention_sequence_with_cache,
    qwen_layer_full_attention_sequence_with_cache_with_matvec,
    qwen_layer_full_attention_step_with_cache,
    qwen_layer_full_attention_step_with_cache_with_matvec, qwen_layer_linear_attention_first_token,
    qwen_layer_linear_attention_projections, qwen_layer_linear_attention_sequence,
    qwen_layer_linear_attention_sequence_with_cache,
    qwen_layer_linear_attention_sequence_with_cache_with_matvec,
    qwen_layer_linear_attention_step_with_cache, qwen_layer_moe_forward_with_matvec,
    qwen_layer_moe_router_with_matvec, qwen_layer0_linear_attention_projections,
    qwen_layer0_moe_forward, qwen_layer0_moe_router, qwen_layer0_post_attention_norm,
    qwen_lm_head_logits, qwen_lm_head_logits_for_spec, qwen_lm_head_logits_with_matvec,
    qwen_lm_head_top_k, qwen_lm_head_top_k_for_spec, qwen_lm_head_top_k_with_matvec,
    qwen_prefill_sequence, qwen_prefill_sequence_with_cache,
    qwen_prefill_sequence_with_cache_with_matvec, qwen_rms_norm_f32,
};
use llm_backend::{QwenMoeDims, QwenMoeRouterProbe, TopKWeight};
use llm_kv_cache::{LayerKvCache, LinearAttentionCache};
use llm_models::{AttentionKind, ModelFamily, NativeTextModelSpec, QwenModelSpec};
use std::cell::Cell;

#[path = "safetensors_loader/metadata.rs"]
mod metadata;
#[path = "safetensors_loader/qwen_attention.rs"]
mod qwen_attention;
#[path = "safetensors_loader/qwen_core.rs"]
mod qwen_core;
#[path = "safetensors_loader/qwen_dense.rs"]
mod qwen_dense;
#[path = "safetensors_loader/qwen_lm_head.rs"]
mod qwen_lm_head;
#[path = "safetensors_loader/qwen_moe.rs"]
mod qwen_moe;
#[path = "safetensors_loader/shard_store.rs"]
mod shard_store;

#[derive(Default)]
struct RecordingMatvecBackend {
    single_bf16_calls: Cell<usize>,
    batched_bf16_calls: Cell<usize>,
    rows_bf16_calls: Cell<usize>,
    range_bf16_calls: Cell<usize>,
    top_k_bf16_calls: Cell<usize>,
    dense_f32_calls: Cell<usize>,
    rms_norm_calls: Cell<usize>,
    softmax_calls: Cell<usize>,
    conv1d_calls: Cell<usize>,
    softmax_top_k_calls: Cell<usize>,
    weighted_sum_calls: Cell<usize>,
    recurrent_update_calls: Cell<usize>,
    recurrent_cache_update_calls: Cell<usize>,
    head_row_calls: Cell<usize>,
    kv_cache_head_row_calls: Cell<usize>,
}

impl NativeMatvecBackend for RecordingMatvecBackend {
    fn bf16_matvec_row_major_f32(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        input: &[f32],
    ) -> Result<Vec<f32>, TensorLoadError> {
        self.single_bf16_calls.set(self.single_bf16_calls.get() + 1);
        CpuQwenMatvecBackend.bf16_matvec_row_major_f32(store, tensor, input)
    }

    fn bf16_matvecs_row_major_f32(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        inputs: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, TensorLoadError> {
        self.batched_bf16_calls
            .set(self.batched_bf16_calls.get() + 1);
        CpuQwenMatvecBackend.bf16_matvecs_row_major_f32(store, tensor, inputs)
    }

    fn bf16_matvec_rows_f32(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        input: &[f32],
        chunk_rows: usize,
    ) -> Result<Vec<f32>, TensorLoadError> {
        self.rows_bf16_calls.set(self.rows_bf16_calls.get() + 1);
        CpuQwenMatvecBackend.bf16_matvec_rows_f32(store, tensor, input, chunk_rows)
    }

    fn bf16_matvec_range_row_major_f32(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        element_offset: usize,
        rows: usize,
        columns: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, TensorLoadError> {
        self.range_bf16_calls.set(self.range_bf16_calls.get() + 1);
        CpuQwenMatvecBackend.bf16_matvec_range_row_major_f32(
            store,
            tensor,
            element_offset,
            rows,
            columns,
            input,
        )
    }

    fn bf16_matvec_top_k_rows_f32(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        input: &[f32],
        top_k: usize,
        chunk_rows: usize,
    ) -> Result<Vec<TopKLogit>, TensorLoadError> {
        self.top_k_bf16_calls.set(self.top_k_bf16_calls.get() + 1);
        CpuQwenMatvecBackend.bf16_matvec_top_k_rows_f32(store, tensor, input, top_k, chunk_rows)
    }

    fn matvec_row_major_f32(
        &self,
        input: &[f32],
        weights: &[f32],
        rows: usize,
        columns: usize,
    ) -> Result<Vec<f32>, MathError> {
        self.dense_f32_calls.set(self.dense_f32_calls.get() + 1);
        CpuQwenMatvecBackend.matvec_row_major_f32(input, weights, rows, columns)
    }

    fn qwen_rms_norm_f32(
        &self,
        input: &[f32],
        weight: &[f32],
        eps: f32,
    ) -> Result<Vec<f32>, MathError> {
        self.rms_norm_calls.set(self.rms_norm_calls.get() + 1);
        qwen_rms_norm_f32(input, weight, eps)
    }

    fn softmax_f32(&self, scores: &[f32]) -> Result<Vec<f32>, MathError> {
        self.softmax_calls.set(self.softmax_calls.get() + 1);
        CpuQwenMatvecBackend.softmax_f32(scores)
    }

    fn linear_attention_conv1d_silu_f32(
        &self,
        window: &[f32],
        weights: &[f32],
        conv_dim: usize,
        kernel_size: usize,
    ) -> Result<Vec<f32>, MathError> {
        self.conv1d_calls.set(self.conv1d_calls.get() + 1);
        CpuQwenMatvecBackend.linear_attention_conv1d_silu_f32(
            window,
            weights,
            conv_dim,
            kernel_size,
        )
    }

    fn softmax_top_k_f32(
        &self,
        logits: &[f32],
        top_k: usize,
    ) -> Result<Vec<TopKWeight>, MathError> {
        self.softmax_top_k_calls
            .set(self.softmax_top_k_calls.get() + 1);
        CpuQwenMatvecBackend.softmax_top_k_f32(logits, top_k)
    }

    fn weighted_sum_f32(
        &self,
        values: &[f32],
        weights: &[f32],
        vector_len: usize,
    ) -> Result<Vec<f32>, MathError> {
        self.weighted_sum_calls
            .set(self.weighted_sum_calls.get() + 1);
        CpuQwenMatvecBackend.weighted_sum_f32(values, weights, vector_len)
    }

    fn linear_attention_recurrent_update_f32(
        &self,
        state: &[f32],
        key: &[f32],
        value: &[f32],
        memory: &[f32],
        beta: f32,
        decay: f32,
        key_head_dim: usize,
        value_head_dim: usize,
    ) -> Result<Vec<f32>, MathError> {
        self.recurrent_update_calls
            .set(self.recurrent_update_calls.get() + 1);
        CpuQwenMatvecBackend.linear_attention_recurrent_update_f32(
            state,
            key,
            value,
            memory,
            beta,
            decay,
            key_head_dim,
            value_head_dim,
        )
    }

    fn linear_attention_recurrent_cache_update_f32(
        &self,
        cache: &LinearAttentionCache,
        state_start: usize,
        key: &[f32],
        value: &[f32],
        memory: &[f32],
        beta: f32,
        decay: f32,
        key_head_dim: usize,
        value_head_dim: usize,
    ) -> Result<Vec<f32>, MathError> {
        self.recurrent_cache_update_calls
            .set(self.recurrent_cache_update_calls.get() + 1);
        CpuQwenMatvecBackend.linear_attention_recurrent_cache_update_f32(
            cache,
            state_start,
            key,
            value,
            memory,
            beta,
            decay,
            key_head_dim,
            value_head_dim,
        )
    }

    fn select_head_rows_f32(
        &self,
        values: &[f32],
        row_count: usize,
        row_len: usize,
        head_start: usize,
        head_len: usize,
    ) -> Result<Vec<f32>, MathError> {
        self.head_row_calls.set(self.head_row_calls.get() + 1);
        CpuQwenMatvecBackend.select_head_rows_f32(values, row_count, row_len, head_start, head_len)
    }

    fn select_kv_cache_head_rows_f32(
        &self,
        cache: &LayerKvCache,
        tensor: QwenKvCacheTensor,
        row_count: usize,
        head_start: usize,
        head_len: usize,
    ) -> Result<Vec<f32>, MathError> {
        self.kv_cache_head_row_calls
            .set(self.kv_cache_head_row_calls.get() + 1);
        CpuQwenMatvecBackend
            .select_kv_cache_head_rows_f32(cache, tensor, row_count, head_start, head_len)
    }
}

fn caches_for_spec(spec: &QwenModelSpec, capacity: usize) -> Vec<QwenLayerCache> {
    qwen_layer_caches_for_spec(spec, capacity).expect("temporary caches")
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
