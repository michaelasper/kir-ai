use super::super::super::{
    InferenceScratchpad, NativeMatvecBackend, SafeTensorShardStore, TensorLoadError,
};
use super::attention::gemma_layer_attention_sequence_with_cache;
use super::cache::{GemmaLayerCache, gemma_concrete_cache_count};
use super::embeddings::{
    gemma_apply_per_layer_input_sequence, gemma_embedding_sequence_for_spec,
    gemma_per_layer_inputs_sequence,
};
use super::mlp::gemma_layer_dense_mlp;
use super::norm::{
    add_sequence, apply_gemma_layer_scalar, gemma_layer_norm_sequence,
    gemma_norm_sequence_after_projection,
};
use llm_models::GemmaModelSpec;

pub async fn gemma_prefill_sequence_with_cache(
    store: &SafeTensorShardStore,
    spec: &GemmaModelSpec,
    token_ids: &[usize],
    caches: &mut [GemmaLayerCache],
    matvec: &impl NativeMatvecBackend,
    scratch: &mut InferenceScratchpad,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    ensure_supported_gemma_execution(spec)?;
    let expected_caches = gemma_concrete_cache_count(spec)?;
    if caches.len() != expected_caches {
        return Err(TensorLoadError::integrity(format!(
            "Gemma prefill expected {expected_caches} layer caches, got {}",
            caches.len()
        )));
    }
    let input_embeddings = gemma_embedding_sequence_for_spec(store, spec, token_ids)?;
    let per_layer_inputs = if spec.uses_per_layer_input() {
        Some(
            gemma_per_layer_inputs_sequence(
                store,
                spec,
                token_ids,
                &input_embeddings,
                matvec,
                scratch,
            )
            .await?,
        )
    } else {
        None
    };
    let mut hidden_states = input_embeddings;
    for layer_idx in 0..spec.num_hidden_layers as usize {
        hidden_states = gemma_decoder_layer_sequence_with_cache(
            store,
            spec,
            layer_idx,
            &hidden_states,
            per_layer_inputs
                .as_ref()
                .map(|inputs| inputs[layer_idx].as_slice()),
            caches,
            matvec,
            scratch,
        )
        .await?;
    }
    Ok(hidden_states)
}

pub async fn gemma_decode_token_with_cache(
    store: &SafeTensorShardStore,
    spec: &GemmaModelSpec,
    token_id: usize,
    caches: &mut [GemmaLayerCache],
    matvec: &impl NativeMatvecBackend,
    scratch: &mut InferenceScratchpad,
) -> Result<Vec<f32>, TensorLoadError> {
    let hidden_states =
        gemma_prefill_sequence_with_cache(store, spec, &[token_id], caches, matvec, scratch)
            .await?;
    hidden_states
        .into_iter()
        .next()
        .ok_or_else(|| TensorLoadError::integrity("Gemma decode returned no hidden state"))
}

#[allow(clippy::too_many_arguments)]
async fn gemma_decoder_layer_sequence_with_cache(
    store: &SafeTensorShardStore,
    spec: &GemmaModelSpec,
    layer_idx: usize,
    hidden_states: &[Vec<f32>],
    per_layer_input: Option<&[Vec<f32>]>,
    caches: &mut [GemmaLayerCache],
    matvec: &impl NativeMatvecBackend,
    scratch: &mut InferenceScratchpad,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    let input_norm = gemma_layer_norm_sequence(
        store,
        spec,
        layer_idx,
        "input_layernorm.weight",
        hidden_states,
    )?;
    let attention_output = gemma_layer_attention_sequence_with_cache(
        store,
        spec,
        layer_idx,
        &input_norm,
        caches,
        matvec,
        scratch,
    )
    .await?;
    let post_attention = gemma_norm_sequence_after_projection(
        store,
        spec,
        layer_idx,
        "post_attention_layernorm.weight",
        &attention_output,
    )?;
    let after_attention = add_sequence(hidden_states, &post_attention, spec.hidden_size as usize)?;
    let pre_feed_forward = gemma_layer_norm_sequence(
        store,
        spec,
        layer_idx,
        "pre_feedforward_layernorm.weight",
        &after_attention,
    )?;
    let mut mlp_output = Vec::with_capacity(pre_feed_forward.len());
    for hidden in &pre_feed_forward {
        let mut output = vec![0.0; spec.hidden_size as usize];
        gemma_layer_dense_mlp(store, spec, layer_idx, hidden, matvec, scratch, &mut output).await?;
        mlp_output.push(output);
    }
    let post_feed_forward = gemma_norm_sequence_after_projection(
        store,
        spec,
        layer_idx,
        "post_feedforward_layernorm.weight",
        &mlp_output,
    )?;
    let mut output = add_sequence(
        &after_attention,
        &post_feed_forward,
        spec.hidden_size as usize,
    )?;
    if let Some(per_layer_input) = per_layer_input {
        output = gemma_apply_per_layer_input_sequence(
            store,
            spec,
            layer_idx,
            &output,
            per_layer_input,
            matvec,
            scratch,
        )
        .await?;
    }
    apply_gemma_layer_scalar(store, spec, layer_idx, &mut output)?;
    Ok(output)
}

fn ensure_supported_gemma_execution(spec: &GemmaModelSpec) -> Result<(), TensorLoadError> {
    if spec.uses_moe() {
        return Err(TensorLoadError::unsupported(
            "Gemma MoE native execution is not implemented yet",
        ));
    }
    Ok(())
}
