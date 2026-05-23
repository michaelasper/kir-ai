#![allow(dead_code)]
// LM-head logits helpers are kept for backend parity tests and optional
// diagnostics while the stable public API exposes only selected entry points.

use super::super::super::math::TopKLogit;
use super::super::super::{NativeMatvecBackend, SafeTensorShardStore, TensorLoadError};
use super::norm::qwen_rms_norm_for_spec_in_place;
use super::tensor_names::{QWEN_FINAL_NORM_WEIGHT, QWEN_LM_HEAD_WEIGHT};
use llm_models::QwenModelSpec;

pub async fn qwen_final_norm(
    store: &SafeTensorShardStore,
    hidden_states: &[f32],
    hidden_size: usize,
    rms_norm_eps: f32,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    let mut output = vec![0.0; hidden_size];
    qwen_final_norm_in_place(
        store,
        hidden_states,
        hidden_size,
        rms_norm_eps,
        matvec,
        &mut output,
    )
    .await?;
    Ok(output)
}

pub(crate) async fn qwen_final_norm_in_place(
    store: &SafeTensorShardStore,
    hidden_states: &[f32],
    hidden_size: usize,
    rms_norm_eps: f32,
    matvec: &impl NativeMatvecBackend,
    output: &mut [f32],
) -> Result<(), TensorLoadError> {
    if hidden_states.len() != hidden_size {
        return Err(TensorLoadError::integrity(format!(
            "Qwen final norm hidden length {} must match hidden size {hidden_size}",
            hidden_states.len()
        )));
    }
    let norm_weight = store.bf16_tensor_f32_cached_arc(QWEN_FINAL_NORM_WEIGHT)?;
    matvec
        .rms_norm_one_centered_f32_in_place(
            hidden_states,
            norm_weight.as_ref(),
            rms_norm_eps,
            output,
        )
        .await
        .map_err(|err| TensorLoadError::integrity(format!("Qwen final RMSNorm failed: {err}")))
}

pub async fn qwen_final_norm_for_spec(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    hidden_states: &[f32],
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    let mut output = vec![0.0; spec.hidden_size as usize];
    qwen_final_norm_for_spec_in_place(store, spec, hidden_states, matvec, &mut output).await?;
    Ok(output)
}

pub(crate) async fn qwen_final_norm_for_spec_in_place(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    hidden_states: &[f32],
    matvec: &impl NativeMatvecBackend,
    output: &mut [f32],
) -> Result<(), TensorLoadError> {
    let hidden_size = spec.hidden_size as usize;
    if hidden_states.len() != hidden_size {
        return Err(TensorLoadError::integrity(format!(
            "Qwen final norm hidden length {} must match hidden size {hidden_size}",
            hidden_states.len()
        )));
    }
    let norm_weight = store.bf16_tensor_f32_cached_arc(&spec.final_norm_weight())?;
    qwen_rms_norm_for_spec_in_place(spec, hidden_states, norm_weight.as_ref(), matvec, output)
        .await
        .map_err(|err| TensorLoadError::integrity(format!("Qwen final RMSNorm failed: {err}")))
}

pub async fn qwen_lm_head_top_k(
    store: &SafeTensorShardStore,
    hidden_states: &[f32],
    top_k: usize,
    chunk_rows: usize,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<TopKLogit>, TensorLoadError> {
    matvec
        .bf16_matvec_top_k_rows_f32(store, QWEN_LM_HEAD_WEIGHT, hidden_states, top_k, chunk_rows)
        .await
}

pub async fn qwen_lm_head_top_k_for_spec(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    hidden_states: &[f32],
    top_k: usize,
    chunk_rows: usize,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<TopKLogit>, TensorLoadError> {
    matvec
        .bf16_matvec_top_k_rows_f32(
            store,
            &spec.lm_head_weight(),
            hidden_states,
            top_k,
            chunk_rows,
        )
        .await
}

pub(crate) async fn qwen_lm_head_logits(
    store: &SafeTensorShardStore,
    hidden_states: &[f32],
    chunk_rows: usize,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    matvec
        .bf16_matvec_rows_f32(store, QWEN_LM_HEAD_WEIGHT, hidden_states, chunk_rows)
        .await
}

pub(crate) async fn qwen_lm_head_logits_in_place(
    store: &SafeTensorShardStore,
    hidden_states: &[f32],
    chunk_rows: usize,
    matvec: &impl NativeMatvecBackend,
    output: &mut [f32],
) -> Result<(), TensorLoadError> {
    matvec
        .bf16_matvec_rows_f32_in_place(
            store,
            QWEN_LM_HEAD_WEIGHT,
            hidden_states,
            chunk_rows,
            output,
        )
        .await
}

pub async fn qwen_lm_head_logits_for_spec(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    hidden_states: &[f32],
    chunk_rows: usize,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    matvec
        .bf16_matvec_rows_f32(store, &spec.lm_head_weight(), hidden_states, chunk_rows)
        .await
}

pub(crate) async fn qwen_lm_head_logits_for_spec_in_place(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    hidden_states: &[f32],
    chunk_rows: usize,
    matvec: &impl NativeMatvecBackend,
    output: &mut [f32],
) -> Result<(), TensorLoadError> {
    matvec
        .bf16_matvec_rows_f32_in_place(
            store,
            &spec.lm_head_weight(),
            hidden_states,
            chunk_rows,
            output,
        )
        .await
}
