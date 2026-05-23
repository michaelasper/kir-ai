use super::super::super::math::{InferenceScratchpad, MathError};
use super::super::super::{
    CpuNativeMatvecBackend, NativeMatvecBackend, SafeTensorShardStore, TensorLoadError,
};
use super::super::matvec::rms_norm_f32_in_place;
use super::tensor_names::qwen_layer_tensor;
use llm_models::QwenModelSpec;

pub(crate) async fn qwen_layer_input_norm(
    store: &SafeTensorShardStore,
    layer_idx: usize,
    hidden_states: &[f32],
    hidden_size: usize,
    rms_norm_eps: f32,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    if hidden_states.len() != hidden_size {
        return Err(TensorLoadError::integrity(format!(
            "Qwen layer input hidden length {} must match hidden size {hidden_size}",
            hidden_states.len()
        )));
    }
    let norm_weight = store
        .bf16_tensor_f32_cached_arc(&qwen_layer_tensor(layer_idx, "input_layernorm.weight"))?;
    matvec
        .rms_norm_one_centered_f32(hidden_states, norm_weight.as_ref(), rms_norm_eps)
        .await
        .map_err(|err| {
            TensorLoadError::integrity(format!("Qwen layer input RMSNorm failed: {err}"))
        })
}

pub(crate) async fn qwen_layer_input_norm_for_spec(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[f32],
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    let mut output = vec![0.0; hidden_states.len()];
    qwen_layer_input_norm_for_spec_in_place(
        store,
        spec,
        layer_idx,
        hidden_states,
        matvec,
        &mut output,
    )
    .await?;
    Ok(output)
}

pub(crate) async fn qwen_layer_input_norm_for_spec_in_place(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[f32],
    matvec: &impl NativeMatvecBackend,
    output: &mut [f32],
) -> Result<(), TensorLoadError> {
    let hidden_size = spec.hidden_size as usize;
    if hidden_states.len() != hidden_size {
        return Err(TensorLoadError::integrity(format!(
            "Qwen layer input hidden length {} must match hidden size {hidden_size}",
            hidden_states.len()
        )));
    }
    let norm_weight = store
        .bf16_tensor_f32_cached_arc(&spec.layer_tensor(layer_idx, "input_layernorm.weight"))?;
    qwen_rms_norm_for_spec_in_place(spec, hidden_states, norm_weight.as_ref(), matvec, output)
        .await
        .map_err(|err| {
            TensorLoadError::integrity(format!("Qwen layer input RMSNorm failed: {err}"))
        })
}

pub(crate) async fn qwen_layer_input_norm_sequence_for_spec(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[Vec<f32>],
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    let hidden_size = spec.hidden_size as usize;
    let norm_weight = store
        .bf16_tensor_f32_cached_arc(&spec.layer_tensor(layer_idx, "input_layernorm.weight"))?;
    let mut results = Vec::with_capacity(hidden_states.len());
    for hidden in hidden_states {
        if hidden.len() != hidden_size {
            return Err(TensorLoadError::integrity(format!(
                "Qwen layer input hidden length {} must match hidden size {hidden_size}",
                hidden.len()
            )));
        }
        results.push(
            qwen_rms_norm_for_spec(spec, hidden, norm_weight.as_ref(), matvec)
                .await
                .map_err(|err| {
                    TensorLoadError::integrity(format!(
                        "Qwen layer input RMSNorm sequence failed: {err}"
                    ))
                })?,
        );
    }
    Ok(results)
}

async fn qwen_rms_norm_for_spec(
    spec: &QwenModelSpec,
    input: &[f32],
    weight: &[f32],
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, MathError> {
    let mut output = vec![0.0; input.len()];
    qwen_rms_norm_for_spec_in_place(spec, input, weight, matvec, &mut output).await?;
    Ok(output)
}

pub(crate) async fn qwen_rms_norm_for_spec_in_place(
    spec: &QwenModelSpec,
    input: &[f32],
    weight: &[f32],
    matvec: &impl NativeMatvecBackend,
    output: &mut [f32],
) -> Result<(), MathError> {
    if spec.is_qwen3_dense() {
        rms_norm_f32_in_place(input, weight, spec.rms_norm_eps, matvec, output).await
    } else {
        matvec
            .rms_norm_one_centered_f32_in_place(input, weight, spec.rms_norm_eps, output)
            .await
    }
}

pub async fn qwen_layer0_post_attention_norm(
    store: &SafeTensorShardStore,
    residual: &[f32],
    attention_output: &[f32],
    hidden_size: usize,
    rms_norm_eps: f32,
) -> Result<Vec<f32>, TensorLoadError> {
    qwen_layer_post_attention_norm(
        store,
        0,
        residual,
        attention_output,
        hidden_size,
        rms_norm_eps,
        &CpuNativeMatvecBackend,
    )
    .await
}

pub(crate) async fn qwen_layer_post_attention_norm(
    store: &SafeTensorShardStore,
    layer_idx: usize,
    residual: &[f32],
    attention_output: &[f32],
    hidden_size: usize,
    rms_norm_eps: f32,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    if residual.len() != hidden_size || attention_output.len() != hidden_size {
        return Err(TensorLoadError::integrity(format!(
            "Qwen post-attention residual lengths {}, {} must match hidden size {hidden_size}",
            residual.len(),
            attention_output.len()
        )));
    }
    let hidden_states = residual
        .iter()
        .zip(attention_output)
        .map(|(residual, attention)| residual + attention)
        .collect::<Vec<_>>();
    let norm_weight = store.bf16_tensor_f32_cached_arc(&qwen_layer_tensor(
        layer_idx,
        "post_attention_layernorm.weight",
    ))?;
    matvec
        .rms_norm_one_centered_f32(&hidden_states, norm_weight.as_ref(), rms_norm_eps)
        .await
        .map_err(|err| {
            TensorLoadError::integrity(format!("Qwen layer post-attention RMSNorm failed: {err}"))
        })
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn qwen_layer_post_attention_norm_for_spec_in_place(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    residual: &[f32],
    attention_output: &[f32],
    matvec: &impl NativeMatvecBackend,
    scratch: &mut InferenceScratchpad,
    output: &mut [f32],
) -> Result<(), TensorLoadError> {
    let hidden_size = spec.hidden_size as usize;
    if residual.len() != hidden_size || attention_output.len() != hidden_size {
        return Err(TensorLoadError::integrity(format!(
            "Qwen post-attention residual lengths {}, {} must match hidden size {hidden_size}",
            residual.len(),
            attention_output.len()
        )));
    }
    let hidden_states = InferenceScratchpad::get_mut(&mut scratch.buf4, hidden_size);
    for (h, (r, a)) in hidden_states
        .iter_mut()
        .zip(residual.iter().zip(attention_output))
    {
        *h = *r + *a;
    }
    let norm_weight = store.bf16_tensor_f32_cached_arc(
        &spec.layer_tensor(layer_idx, "post_attention_layernorm.weight"),
    )?;
    qwen_rms_norm_for_spec_in_place(spec, hidden_states, norm_weight.as_ref(), matvec, output)
        .await
        .map_err(|err| {
            TensorLoadError::integrity(format!("Qwen layer post-attention RMSNorm failed: {err}"))
        })
}

pub(crate) async fn qwen_layer_post_attention_norm_for_spec(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    residual: &[f32],
    attention_output: &[f32],
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    let hidden_size = spec.hidden_size as usize;
    let mut output = vec![0.0; hidden_size];
    let mut scratch = InferenceScratchpad::new();
    qwen_layer_post_attention_norm_for_spec_in_place(
        store,
        spec,
        layer_idx,
        residual,
        attention_output,
        matvec,
        &mut scratch,
        &mut output,
    )
    .await?;
    Ok(output)
}

pub(crate) async fn qwen_layer_post_attention_norm_sequence_for_spec(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    residual: &[Vec<f32>],
    attention_output: &[Vec<f32>],
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    let hidden_size = spec.hidden_size as usize;
    if residual.len() != attention_output.len() {
        return Err(TensorLoadError::integrity(
            "Qwen post-attention sequence lengths must match",
        ));
    }
    let norm_weight = store.bf16_tensor_f32_cached_arc(
        &spec.layer_tensor(layer_idx, "post_attention_layernorm.weight"),
    )?;
    let mut results = Vec::with_capacity(residual.len());
    for (residual, attention) in residual.iter().zip(attention_output) {
        if residual.len() != hidden_size || attention.len() != hidden_size {
            return Err(TensorLoadError::integrity(format!(
                "Qwen post-attention residual lengths {}, {} must match hidden size {hidden_size}",
                residual.len(),
                attention.len()
            )));
        }
        let hidden_states = residual
            .iter()
            .zip(attention)
            .map(|(residual, attention)| residual + attention)
            .collect::<Vec<_>>();
        results.push(
            qwen_rms_norm_for_spec(spec, &hidden_states, norm_weight.as_ref(), matvec)
                .await
                .map_err(|err| {
                    TensorLoadError::integrity(format!(
                        "Qwen post-attention RMSNorm sequence failed: {err}"
                    ))
                })?,
        );
    }
    Ok(results)
}
