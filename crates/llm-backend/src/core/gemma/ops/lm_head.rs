use super::super::super::math::{TopKLogit, rms_norm_f32_in_place};
use super::super::super::{NativeMatvecBackend, SafeTensorShardStore, TensorLoadError};
use llm_models::GemmaModelSpec;

pub async fn gemma_final_norm_for_spec(
    store: &SafeTensorShardStore,
    spec: &GemmaModelSpec,
    hidden_states: &[f32],
    output: &mut [f32],
) -> Result<(), TensorLoadError> {
    let hidden_size = spec.hidden_size as usize;
    if hidden_states.len() != hidden_size {
        return Err(TensorLoadError::integrity(format!(
            "Gemma final norm hidden length {} must match hidden size {hidden_size}",
            hidden_states.len()
        )));
    }
    let norm_weight = store.bf16_tensor_f32_cached_arc(&spec.final_norm_weight())?;
    rms_norm_f32_in_place(
        hidden_states,
        norm_weight.as_ref(),
        spec.rms_norm_eps,
        output,
    )
    .map_err(|err| TensorLoadError::integrity(format!("Gemma final RMSNorm failed: {err}")))
}

pub async fn gemma_lm_head_top_k_for_spec(
    store: &SafeTensorShardStore,
    spec: &GemmaModelSpec,
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

pub async fn gemma_lm_head_logits_for_spec(
    store: &SafeTensorShardStore,
    spec: &GemmaModelSpec,
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
