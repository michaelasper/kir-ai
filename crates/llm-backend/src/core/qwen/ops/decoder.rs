use super::super::super::{
    CpuNativeMatvecBackend, InferenceScratchpad, NativeMatvecBackend, SafeTensorShardStore,
    TensorLoadError,
};
use super::attention_full::{
    qwen_layer_full_attention_first_token, qwen_layer_full_attention_sequence_impl,
    qwen_layer_full_attention_sequence_with_cache, qwen_layer_full_attention_step_with_cache,
};
use super::attention_linear::{
    qwen_layer_linear_attention_first_token, qwen_layer_linear_attention_projections,
    qwen_layer_linear_attention_sequence_impl, qwen_layer_linear_attention_sequence_with_cache,
    qwen_layer_linear_attention_step_with_cache,
};
use super::cache::QwenLayerCache;
use super::embedding::{qwen_embedding_sequence_for_spec, validate_qwen_token_id};
use super::moe::qwen_layer_feed_forward;
use super::norm::{
    qwen_layer_input_norm_for_spec, qwen_layer_input_norm_for_spec_in_place,
    qwen_layer_input_norm_sequence_for_spec, qwen_layer_post_attention_norm_for_spec,
    qwen_layer_post_attention_norm_for_spec_in_place,
    qwen_layer_post_attention_norm_sequence_for_spec,
};
use llm_models::{AttentionKind, QwenModelSpec};

pub async fn qwen_linear_decoder_layer_first_token(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[f32],
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    match spec.layer_kinds.get(layer_idx) {
        Some(AttentionKind::LinearAttention) => {}
        Some(AttentionKind::FullAttention) => {
            return Err(TensorLoadError::unsupported(format!(
                "Qwen layer {layer_idx} is full attention, not linear attention"
            )));
        }
        None => {
            return Err(TensorLoadError::missing(format!(
                "Qwen layer {layer_idx} is outside configured layer count"
            )));
        }
    }
    let input_norm =
        qwen_layer_input_norm_for_spec(store, spec, layer_idx, hidden_states, matvec).await?;
    let projections =
        qwen_layer_linear_attention_projections(store, layer_idx, &input_norm, matvec).await?;
    let attention_output =
        qwen_layer_linear_attention_first_token(store, spec, layer_idx, &projections, matvec)
            .await?;
    let post_attention = qwen_layer_post_attention_norm_for_spec(
        store,
        spec,
        layer_idx,
        hidden_states,
        &attention_output,
        matvec,
    )
    .await?;
    let mut mlp_output = vec![0.0; spec.hidden_size as usize];
    let mut scratch = InferenceScratchpad::default();
    qwen_layer_feed_forward(
        store,
        spec,
        layer_idx,
        &post_attention,
        matvec,
        &mut scratch,
        &mut mlp_output,
    )
    .await?;
    hidden_states
        .iter()
        .zip(attention_output)
        .zip(mlp_output)
        .map(|((hidden, attention), mlp)| Ok(hidden + attention + mlp))
        .collect()
}

pub(crate) async fn qwen_full_decoder_layer_first_token(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[f32],
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    match spec.layer_kinds.get(layer_idx) {
        Some(AttentionKind::FullAttention) => {}
        Some(AttentionKind::LinearAttention) => {
            return Err(TensorLoadError::unsupported(format!(
                "Qwen layer {layer_idx} is linear attention, not full attention"
            )));
        }
        None => {
            return Err(TensorLoadError::missing(format!(
                "Qwen layer {layer_idx} is outside configured layer count"
            )));
        }
    }
    let input_norm =
        qwen_layer_input_norm_for_spec(store, spec, layer_idx, hidden_states, matvec).await?;
    let attention_output =
        qwen_layer_full_attention_first_token(store, spec, layer_idx, &input_norm, matvec).await?;
    let post_attention = qwen_layer_post_attention_norm_for_spec(
        store,
        spec,
        layer_idx,
        hidden_states,
        &attention_output,
        matvec,
    )
    .await?;
    let mut mlp_output = vec![0.0; spec.hidden_size as usize];
    let mut scratch = InferenceScratchpad::default();
    qwen_layer_feed_forward(
        store,
        spec,
        layer_idx,
        &post_attention,
        matvec,
        &mut scratch,
        &mut mlp_output,
    )
    .await?;
    hidden_states
        .iter()
        .zip(attention_output)
        .zip(mlp_output)
        .map(|((hidden, attention), mlp)| Ok(hidden + attention + mlp))
        .collect()
}

pub async fn qwen_decoder_layer_first_token(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[f32],
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    match spec.layer_kinds.get(layer_idx) {
        Some(AttentionKind::LinearAttention) => {
            qwen_linear_decoder_layer_first_token(store, spec, layer_idx, hidden_states, matvec)
                .await
        }
        Some(AttentionKind::FullAttention) => {
            qwen_full_decoder_layer_first_token(store, spec, layer_idx, hidden_states, matvec).await
        }
        None => Err(TensorLoadError::missing(format!(
            "Qwen layer {layer_idx} is outside configured layer count"
        ))),
    }
}

pub(crate) async fn qwen_decoder_layer_sequence(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    let mut scratch = InferenceScratchpad::default();
    qwen_decoder_layer_sequence_impl(
        store,
        spec,
        layer_idx,
        hidden_states,
        None,
        &CpuNativeMatvecBackend,
        &mut scratch,
    )
    .await
}

pub(crate) async fn qwen_decoder_layer_sequence_with_cache(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[Vec<f32>],
    cache: &mut QwenLayerCache,
    matvec: &impl NativeMatvecBackend,
    scratch: &mut InferenceScratchpad,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    qwen_decoder_layer_sequence_impl(
        store,
        spec,
        layer_idx,
        hidden_states,
        Some(cache),
        matvec,
        scratch,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn qwen_decoder_layer_step_with_cache(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[f32],
    cache: &mut QwenLayerCache,
    matvec: &impl NativeMatvecBackend,
    scratch: &mut InferenceScratchpad,
    output: &mut [f32],
) -> Result<(), TensorLoadError> {
    let hidden_size = spec.hidden_size as usize;
    let mut input_norm = vec![0.0; hidden_size];
    qwen_layer_input_norm_for_spec_in_place(
        store,
        spec,
        layer_idx,
        hidden_states,
        matvec,
        &mut input_norm,
    )
    .await?;
    let mut attention_output = vec![0.0; hidden_size];
    match spec.layer_kinds.get(layer_idx) {
        Some(AttentionKind::LinearAttention) => match cache {
            QwenLayerCache::Linear(cache) => {
                qwen_layer_linear_attention_step_with_cache(
                    store,
                    spec,
                    layer_idx,
                    &input_norm,
                    cache,
                    matvec,
                    scratch,
                    &mut attention_output,
                )
                .await?
            }
            _ => {
                return Err(TensorLoadError::integrity(format!(
                    "Qwen layer{layer_idx} expected linear attention cache"
                )));
            }
        },
        Some(AttentionKind::FullAttention) => match cache {
            QwenLayerCache::Full(cache) => {
                qwen_layer_full_attention_step_with_cache(
                    store,
                    spec,
                    layer_idx,
                    &input_norm,
                    cache,
                    matvec,
                    scratch,
                    &mut attention_output,
                )
                .await?
            }
            _ => {
                return Err(TensorLoadError::integrity(format!(
                    "Qwen layer{layer_idx} expected full attention cache"
                )));
            }
        },
        None => {
            return Err(TensorLoadError::missing(format!(
                "Qwen layer {layer_idx} is outside configured layer count"
            )));
        }
    };
    let mut post_attention_norm = vec![0.0; hidden_size];
    qwen_layer_post_attention_norm_for_spec_in_place(
        store,
        spec,
        layer_idx,
        hidden_states,
        &attention_output,
        matvec,
        scratch,
        &mut post_attention_norm,
    )
    .await?;
    qwen_layer_feed_forward(
        store,
        spec,
        layer_idx,
        &post_attention_norm,
        matvec,
        scratch,
        output,
    )
    .await?;

    if output.len() < hidden_size {
        return Err(TensorLoadError::integrity("output buffer too small"));
    }
    for i in 0..hidden_size {
        output[i] += hidden_states[i] + attention_output[i];
    }
    Ok(())
}

async fn qwen_decoder_layer_sequence_impl(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[Vec<f32>],
    cache: Option<&mut QwenLayerCache>,
    matvec: &impl NativeMatvecBackend,
    scratch: &mut InferenceScratchpad,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    let input_norm =
        qwen_layer_input_norm_sequence_for_spec(store, spec, layer_idx, hidden_states, matvec)
            .await?;
    let attention_output = match spec.layer_kinds.get(layer_idx) {
        Some(AttentionKind::LinearAttention) => match cache {
            Some(QwenLayerCache::Linear(cache)) => {
                qwen_layer_linear_attention_sequence_with_cache(
                    store,
                    spec,
                    layer_idx,
                    &input_norm,
                    cache,
                    matvec,
                )
                .await?
            }
            Some(QwenLayerCache::Full(_)) => {
                return Err(TensorLoadError::integrity(format!(
                    "Qwen layer{layer_idx} expected linear attention cache"
                )));
            }
            None => {
                qwen_layer_linear_attention_sequence_impl(
                    store,
                    spec,
                    layer_idx,
                    &input_norm,
                    None,
                    matvec,
                )
                .await?
            }
        },
        Some(AttentionKind::FullAttention) => match cache {
            Some(QwenLayerCache::Full(cache)) => {
                qwen_layer_full_attention_sequence_with_cache(
                    store,
                    spec,
                    layer_idx,
                    &input_norm,
                    cache,
                    matvec,
                )
                .await?
            }
            Some(QwenLayerCache::Linear(_)) => {
                return Err(TensorLoadError::integrity(format!(
                    "Qwen layer{layer_idx} expected full attention cache"
                )));
            }
            None => {
                qwen_layer_full_attention_sequence_impl(
                    store,
                    spec,
                    layer_idx,
                    &input_norm,
                    None,
                    matvec,
                )
                .await?
            }
        },
        None => {
            return Err(TensorLoadError::missing(format!(
                "Qwen layer {layer_idx} is outside configured layer count"
            )));
        }
    };
    let post_attention = qwen_layer_post_attention_norm_sequence_for_spec(
        store,
        spec,
        layer_idx,
        hidden_states,
        &attention_output,
        matvec,
    )
    .await?;
    let mut results = Vec::with_capacity(hidden_states.len());
    for ((hidden, attention), post_attention) in hidden_states
        .iter()
        .zip(attention_output)
        .zip(post_attention)
    {
        let mut mlp_output = vec![0.0; spec.hidden_size as usize];
        qwen_layer_feed_forward(
            store,
            spec,
            layer_idx,
            &post_attention,
            matvec,
            scratch,
            &mut mlp_output,
        )
        .await?;
        results.push(
            hidden
                .iter()
                .zip(attention)
                .zip(mlp_output)
                .map(|((hidden, attention), mlp)| hidden + attention + mlp)
                .collect::<Vec<_>>(),
        );
    }
    Ok(results)
}

pub(crate) async fn qwen_prefill_sequence(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    token_ids: &[usize],
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    let mut hidden_states = qwen_embedding_sequence_for_spec(store, spec, token_ids)?;
    for layer_idx in 0..spec.num_hidden_layers as usize {
        hidden_states = qwen_decoder_layer_sequence(store, spec, layer_idx, &hidden_states).await?;
    }
    Ok(hidden_states)
}

pub async fn qwen_prefill_sequence_with_cache(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    token_ids: &[usize],
    caches: &mut [QwenLayerCache],
    matvec: &impl NativeMatvecBackend,
    scratch: &mut InferenceScratchpad,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    let layer_count = spec.num_hidden_layers as usize;
    if caches.len() != layer_count {
        return Err(TensorLoadError::integrity(format!(
            "Qwen prefill expected {layer_count} layer caches, got {}",
            caches.len()
        )));
    }
    let mut hidden_states = qwen_embedding_sequence_for_spec(store, spec, token_ids)?;
    for (layer_idx, cache) in caches.iter_mut().enumerate().take(layer_count) {
        hidden_states = qwen_decoder_layer_sequence_with_cache(
            store,
            spec,
            layer_idx,
            &hidden_states,
            cache,
            matvec,
            scratch,
        )
        .await?;
    }
    Ok(hidden_states)
}

pub async fn qwen_decode_token_with_cache(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    token_id: usize,
    caches: &mut [QwenLayerCache],
    matvec: &impl NativeMatvecBackend,
    scratch: &mut InferenceScratchpad,
) -> Result<Vec<f32>, TensorLoadError> {
    let layer_count = spec.num_hidden_layers as usize;
    if caches.len() != layer_count {
        return Err(TensorLoadError::integrity(format!(
            "Qwen decode expected {layer_count} layer caches, got {}",
            caches.len()
        )));
    }
    validate_qwen_token_id(token_id, spec.vocab_size as usize)?;
    let mut current_hidden = store.bf16_row_f32(&spec.embed_tokens_weight(), token_id)?;
    if current_hidden.len() != spec.hidden_size as usize {
        return Err(TensorLoadError::integrity(format!(
            "Qwen embedding row has length {}, expected hidden size {}",
            current_hidden.len(),
            spec.hidden_size
        )));
    }
    let mut next_hidden = vec![0.0; spec.hidden_size as usize];
    for (layer_idx, cache) in caches.iter_mut().enumerate().take(layer_count) {
        qwen_decoder_layer_step_with_cache(
            store,
            spec,
            layer_idx,
            &current_hidden,
            cache,
            matvec,
            scratch,
            &mut next_hidden,
        )
        .await?;
        current_hidden.copy_from_slice(&next_hidden);
    }
    Ok(current_hidden)
}
