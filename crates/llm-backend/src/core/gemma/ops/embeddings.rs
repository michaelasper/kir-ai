use super::super::super::math::{InferenceScratchpad, rms_norm_f32_in_place};
use super::super::super::{
    NativeBatchedMatvecInputBuffer, NativeMatvecBackend, SafeTensorShardStore, TensorLoadError,
};
use super::activation::gelu_pytorch_tanh_f32;
use llm_models::GemmaModelSpec;

fn validate_gemma_token_ids(
    label: &str,
    token_ids: &[usize],
    vocab_size: usize,
) -> Result<(), TensorLoadError> {
    for (position, token_id) in token_ids.iter().enumerate() {
        if *token_id >= vocab_size {
            return Err(TensorLoadError::integrity(format!(
                "{label} token id {token_id} at position {position} is outside vocab size {vocab_size}"
            )));
        }
    }
    Ok(())
}

pub(crate) fn gemma_embedding_sequence_for_spec(
    store: &SafeTensorShardStore,
    spec: &GemmaModelSpec,
    token_ids: &[usize],
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    let hidden_size = spec.hidden_size as usize;
    let scale = (hidden_size as f32).sqrt();
    validate_gemma_token_ids("Gemma", token_ids, spec.vocab_size as usize)?;
    token_ids
        .iter()
        .map(|token_id| {
            let mut embedding = store.bf16_row_f32(&spec.embed_tokens_weight(), *token_id)?;
            if embedding.len() != hidden_size {
                return Err(TensorLoadError::integrity(format!(
                    "Gemma embedding row has length {}, expected hidden size {hidden_size}",
                    embedding.len()
                )));
            }
            for value in &mut embedding {
                *value *= scale;
            }
            Ok(embedding)
        })
        .collect()
}

pub(crate) async fn gemma_per_layer_inputs_sequence(
    store: &SafeTensorShardStore,
    spec: &GemmaModelSpec,
    token_ids: &[usize],
    input_embeddings: &[Vec<f32>],
    matvec: &impl NativeMatvecBackend,
    _scratch: &mut InferenceScratchpad,
) -> Result<Vec<Vec<Vec<f32>>>, TensorLoadError> {
    let layer_count = spec.num_hidden_layers as usize;
    let per_layer_size = spec.hidden_size_per_layer_input as usize;
    if per_layer_size == 0 {
        return Ok(vec![Vec::new(); layer_count]);
    }
    if input_embeddings.len() != token_ids.len() {
        return Err(TensorLoadError::integrity(format!(
            "Gemma PLE input embedding count {} must match token count {}",
            input_embeddings.len(),
            token_ids.len()
        )));
    }
    validate_gemma_token_ids(
        "Gemma per-layer input",
        token_ids,
        spec.vocab_size_per_layer_input as usize,
    )?;
    let total_per_token = layer_count
        .checked_mul(per_layer_size)
        .ok_or_else(|| TensorLoadError::integrity("Gemma PLE shape overflow"))?;
    let projection_norm_weight =
        store.bf16_tensor_f32_cached_arc(&spec.per_layer_projection_norm_weight())?;
    let input_columns = input_embeddings.first().map_or(0, Vec::len);
    let flat_input_embeddings =
        NativeBatchedMatvecInputBuffer::from_rows(input_embeddings, input_columns)?;
    let projected = matvec
        .bf16_matvecs_row_major_f32_flat_inputs(
            store,
            &spec.per_layer_model_projection_weight(),
            flat_input_embeddings.values(),
            flat_input_embeddings.input_count(),
        )
        .await?;
    if projected.row_count() != token_ids.len() {
        return Err(TensorLoadError::integrity(format!(
            "Gemma PLE projection count {} must match token count {}",
            projected.row_count(),
            token_ids.len()
        )));
    }

    let token_embedding_scale = (per_layer_size as f32).sqrt();
    let model_projection_scale = (spec.hidden_size as f32).powf(-0.5);
    let combine_scale = 2.0_f32.sqrt().recip();
    let mut layer_inputs = vec![Vec::with_capacity(token_ids.len()); layer_count];
    for (token_idx, token_id) in token_ids.iter().enumerate() {
        let mut token_per_layer =
            store.bf16_row_f32(&spec.embed_tokens_per_layer_weight(), *token_id)?;
        if token_per_layer.len() != total_per_token {
            return Err(TensorLoadError::integrity(format!(
                "Gemma token PLE row has length {}, expected {total_per_token}",
                token_per_layer.len()
            )));
        }
        for value in &mut token_per_layer {
            *value *= token_embedding_scale;
        }
        let projected_token = projected.row(token_idx).ok_or_else(|| {
            TensorLoadError::integrity(format!("Gemma PLE projection row {token_idx} is missing"))
        })?;
        if projected_token.len() != total_per_token {
            return Err(TensorLoadError::integrity(format!(
                "Gemma PLE projection row has length {}, expected {total_per_token}",
                projected_token.len()
            )));
        }
        for (layer_idx, layer_inputs_for_layer) in layer_inputs.iter_mut().enumerate() {
            let start = layer_idx
                .checked_mul(per_layer_size)
                .ok_or_else(|| TensorLoadError::integrity("Gemma PLE layer offset overflow"))?;
            let end = start + per_layer_size;
            let projected_slice = projected_token[start..end]
                .iter()
                .map(|value| value * model_projection_scale)
                .collect::<Vec<_>>();
            let mut normalized_projection = vec![0.0; per_layer_size];
            rms_norm_f32_in_place(
                &projected_slice,
                projection_norm_weight.as_ref(),
                spec.rms_norm_eps,
                &mut normalized_projection,
            )
            .map_err(|err| {
                TensorLoadError::integrity(format!("Gemma PLE projection RMSNorm failed: {err}"))
            })?;
            let combined = normalized_projection
                .iter()
                .zip(&token_per_layer[start..end])
                .map(|(projection, token_embedding)| (projection + token_embedding) * combine_scale)
                .collect::<Vec<_>>();
            layer_inputs_for_layer.push(combined);
        }
    }
    Ok(layer_inputs)
}

pub(crate) async fn gemma_apply_per_layer_input_sequence(
    store: &SafeTensorShardStore,
    spec: &GemmaModelSpec,
    layer_idx: usize,
    hidden_states: &[Vec<f32>],
    per_layer_inputs: &[Vec<f32>],
    matvec: &impl NativeMatvecBackend,
    scratch: &mut InferenceScratchpad,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    let hidden_size = spec.hidden_size as usize;
    let per_layer_size = spec.hidden_size_per_layer_input as usize;
    if hidden_states.len() != per_layer_inputs.len() {
        return Err(TensorLoadError::integrity(format!(
            "Gemma layer{layer_idx} PLE sequence length {} must match hidden sequence length {}",
            per_layer_inputs.len(),
            hidden_states.len()
        )));
    }
    let norm_weight = store.bf16_tensor_f32_cached_arc(
        &spec.layer_tensor(layer_idx, "post_per_layer_input_norm.weight"),
    )?;
    let mut results = Vec::with_capacity(hidden_states.len());
    for (hidden, per_layer_input) in hidden_states.iter().zip(per_layer_inputs) {
        if hidden.len() != hidden_size {
            return Err(TensorLoadError::integrity(format!(
                "Gemma layer{layer_idx} PLE hidden length {} must match hidden size {hidden_size}",
                hidden.len()
            )));
        }
        if per_layer_input.len() != per_layer_size {
            return Err(TensorLoadError::integrity(format!(
                "Gemma layer{layer_idx} PLE input length {} must match per-layer size {per_layer_size}",
                per_layer_input.len()
            )));
        }
        let gate = InferenceScratchpad::get_mut(&mut scratch.buf0, per_layer_size);
        matvec
            .bf16_matvec_row_major_f32_in_place(
                store,
                &spec.layer_tensor(layer_idx, "per_layer_input_gate.weight"),
                hidden,
                gate,
            )
            .await?;

        let activated = InferenceScratchpad::get_mut(&mut scratch.buf1, per_layer_size);
        for (a, (g, i)) in activated.iter_mut().zip(gate.iter().zip(per_layer_input)) {
            *a = gelu_pytorch_tanh_f32(*g) * *i;
        }

        let projected = InferenceScratchpad::get_mut(&mut scratch.buf2, hidden_size);
        matvec
            .bf16_matvec_row_major_f32_in_place(
                store,
                &spec.layer_tensor(layer_idx, "per_layer_projection.weight"),
                activated,
                projected,
            )
            .await?;

        let normalized = InferenceScratchpad::get_mut(&mut scratch.buf3, hidden_size);
        rms_norm_f32_in_place(
            projected,
            norm_weight.as_ref(),
            spec.rms_norm_eps,
            normalized,
        )
        .map_err(|err| {
            TensorLoadError::integrity(format!("Gemma layer{layer_idx} PLE RMSNorm failed: {err}"))
        })?;

        results.push(
            hidden
                .iter()
                .zip(normalized)
                .map(|(h, u)| *h + *u)
                .collect(),
        );
    }
    Ok(results)
}
