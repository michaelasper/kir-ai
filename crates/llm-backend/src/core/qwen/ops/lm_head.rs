use super::*;

pub(super) fn qwen_layer_tensor(layer_idx: usize, suffix: &str) -> String {
    format!("model.language_model.layers.{layer_idx}.{suffix}")
}

pub(super) fn qwen_linear_attn_tensor(layer_idx: usize, suffix: &str) -> String {
    qwen_layer_tensor(layer_idx, &format!("linear_attn.{suffix}"))
}

pub(super) fn qwen_mlp_tensor(layer_idx: usize, suffix: &str) -> String {
    qwen_layer_tensor(layer_idx, &format!("mlp.{suffix}"))
}

pub fn qwen_final_norm(
    store: &SafeTensorShardStore,
    hidden_states: &[f32],
    hidden_size: usize,
    rms_norm_eps: f32,
) -> Result<Vec<f32>, TensorLoadError> {
    qwen_final_norm_with_matvec(
        store,
        hidden_states,
        hidden_size,
        rms_norm_eps,
        &CpuQwenMatvecBackend,
    )
}

pub fn qwen_final_norm_with_matvec(
    store: &SafeTensorShardStore,
    hidden_states: &[f32],
    hidden_size: usize,
    rms_norm_eps: f32,
    matvec: &impl QwenMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    if hidden_states.len() != hidden_size {
        return Err(TensorLoadError::integrity(format!(
            "Qwen final norm hidden length {} must match hidden size {hidden_size}",
            hidden_states.len()
        )));
    }
    let norm_weight = store.bf16_tensor_f32_range(QWEN_FINAL_NORM_WEIGHT, 0, hidden_size)?;
    matvec
        .qwen_rms_norm_f32(hidden_states, &norm_weight, rms_norm_eps)
        .map_err(|err| TensorLoadError::integrity(format!("Qwen final RMSNorm failed: {err}")))
}

pub fn qwen_final_norm_for_spec(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    hidden_states: &[f32],
) -> Result<Vec<f32>, TensorLoadError> {
    qwen_final_norm_for_spec_with_matvec(store, spec, hidden_states, &CpuQwenMatvecBackend)
}

pub fn qwen_final_norm_for_spec_with_matvec(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    hidden_states: &[f32],
    matvec: &impl QwenMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    let hidden_size = spec.hidden_size as usize;
    if hidden_states.len() != hidden_size {
        return Err(TensorLoadError::integrity(format!(
            "Qwen final norm hidden length {} must match hidden size {hidden_size}",
            hidden_states.len()
        )));
    }
    let norm_weight = store.bf16_tensor_f32_range(&spec.final_norm_weight(), 0, hidden_size)?;
    qwen_rms_norm_for_spec_with_matvec(spec, hidden_states, &norm_weight, matvec)
        .map_err(|err| TensorLoadError::integrity(format!("Qwen final RMSNorm failed: {err}")))
}

pub fn qwen_lm_head_top_k(
    store: &SafeTensorShardStore,
    hidden_states: &[f32],
    top_k: usize,
    chunk_rows: usize,
) -> Result<Vec<TopKLogit>, TensorLoadError> {
    qwen_lm_head_top_k_with_matvec(
        store,
        hidden_states,
        top_k,
        chunk_rows,
        &CpuQwenMatvecBackend,
    )
}

pub fn qwen_lm_head_top_k_with_matvec(
    store: &SafeTensorShardStore,
    hidden_states: &[f32],
    top_k: usize,
    chunk_rows: usize,
    matvec: &impl QwenMatvecBackend,
) -> Result<Vec<TopKLogit>, TensorLoadError> {
    matvec.bf16_matvec_top_k_rows_f32(store, QWEN_LM_HEAD_WEIGHT, hidden_states, top_k, chunk_rows)
}

pub fn qwen_lm_head_top_k_for_spec(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    hidden_states: &[f32],
    top_k: usize,
    chunk_rows: usize,
) -> Result<Vec<TopKLogit>, TensorLoadError> {
    qwen_lm_head_top_k_for_spec_with_matvec(
        store,
        spec,
        hidden_states,
        top_k,
        chunk_rows,
        &CpuQwenMatvecBackend,
    )
}

pub fn qwen_lm_head_top_k_for_spec_with_matvec(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    hidden_states: &[f32],
    top_k: usize,
    chunk_rows: usize,
    matvec: &impl QwenMatvecBackend,
) -> Result<Vec<TopKLogit>, TensorLoadError> {
    matvec.bf16_matvec_top_k_rows_f32(
        store,
        &spec.lm_head_weight(),
        hidden_states,
        top_k,
        chunk_rows,
    )
}

pub fn qwen_lm_head_logits(
    store: &SafeTensorShardStore,
    hidden_states: &[f32],
    chunk_rows: usize,
) -> Result<Vec<f32>, TensorLoadError> {
    qwen_lm_head_logits_with_matvec(store, hidden_states, chunk_rows, &CpuQwenMatvecBackend)
}

pub fn qwen_lm_head_logits_with_matvec(
    store: &SafeTensorShardStore,
    hidden_states: &[f32],
    chunk_rows: usize,
    matvec: &impl QwenMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    matvec.bf16_matvec_rows_f32(store, QWEN_LM_HEAD_WEIGHT, hidden_states, chunk_rows)
}

pub fn qwen_lm_head_logits_for_spec(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    hidden_states: &[f32],
    chunk_rows: usize,
) -> Result<Vec<f32>, TensorLoadError> {
    qwen_lm_head_logits_for_spec_with_matvec(
        store,
        spec,
        hidden_states,
        chunk_rows,
        &CpuQwenMatvecBackend,
    )
}

pub fn qwen_lm_head_logits_for_spec_with_matvec(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    hidden_states: &[f32],
    chunk_rows: usize,
    matvec: &impl QwenMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    matvec.bf16_matvec_rows_f32(store, &spec.lm_head_weight(), hidden_states, chunk_rows)
}
