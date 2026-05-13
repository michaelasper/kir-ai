use super::super::math::{
    InferenceScratchpad, MathError, TopKLogit, apply_rope_to_head, require_len,
    rms_norm_f32_in_place,
};
use super::super::{
    CpuNativeMatvecBackend, LayerKvCache, NativeF32Rows, NativeFullAttentionCacheSequenceParts,
    NativeFullAttentionDims, NativeFullAttentionSequenceParts, NativeMatvecBackend,
    NativeOutputProjection, SafeTensorShardStore, TensorLoadError,
    native_full_attention_sequence_from_cache_parts_with_matvec,
    native_full_attention_sequence_with_cache_from_parts_with_matvec,
};
use llm_models::{GemmaAttentionKind, GemmaModelSpec};

#[derive(Debug, Clone, PartialEq)]
pub enum GemmaLayerCache {
    Attention(LayerKvCache),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GemmaAttentionDims {
    hidden_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
}

impl GemmaAttentionDims {
    fn from_spec(spec: &GemmaModelSpec, layer_idx: usize) -> Result<Self, TensorLoadError> {
        let kind = spec.layer_kinds.get(layer_idx).ok_or_else(|| {
            TensorLoadError::missing(format!(
                "Gemma layer {layer_idx} is outside configured layer count"
            ))
        })?;
        let is_full = matches!(kind, GemmaAttentionKind::FullAttention);
        let head_dim = if is_full {
            spec.global_head_dim.unwrap_or(spec.head_dim)
        } else {
            spec.head_dim
        } as usize;
        let num_key_value_heads = if spec.attention_k_eq_v && is_full {
            spec.num_global_key_value_heads
                .unwrap_or(spec.num_key_value_heads)
        } else {
            spec.num_key_value_heads
        } as usize;
        Ok(Self {
            hidden_size: spec.hidden_size as usize,
            num_attention_heads: spec.num_attention_heads as usize,
            num_key_value_heads,
            head_dim,
        })
    }

    fn attention_dim(&self) -> Result<usize, MathError> {
        self.num_attention_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| MathError::InvalidShape("Gemma attention dimension overflow".to_owned()))
    }

    fn key_value_dim(&self) -> Result<usize, MathError> {
        self.num_key_value_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| MathError::InvalidShape("Gemma KV dimension overflow".to_owned()))
    }

    fn native(self) -> NativeFullAttentionDims {
        NativeFullAttentionDims {
            hidden_size: self.hidden_size,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
        }
    }
}

pub fn gemma_layer_caches_for_spec(
    spec: &GemmaModelSpec,
    max_tokens: usize,
) -> Result<Vec<GemmaLayerCache>, TensorLoadError> {
    let concrete_count = gemma_concrete_cache_count(spec)?;
    (0..concrete_count)
        .map(|layer_idx| gemma_layer_cache_for_spec(spec, layer_idx, max_tokens))
        .collect()
}

pub fn gemma_cache_count_for_spec(spec: &GemmaModelSpec) -> Result<usize, TensorLoadError> {
    gemma_concrete_cache_count(spec)
}

pub fn gemma_static_f32_tensors_for_spec(spec: &GemmaModelSpec) -> Vec<String> {
    let mut tensors = Vec::new();
    tensors.push(spec.final_norm_weight());
    if spec.uses_per_layer_input() {
        tensors.push(spec.per_layer_projection_norm_weight());
    }
    for layer_idx in 0..spec.num_hidden_layers as usize {
        tensors.push(spec.layer_tensor(layer_idx, "input_layernorm.weight"));
        tensors.push(spec.layer_tensor(layer_idx, "layer_scalar"));
        tensors.push(spec.layer_tensor(layer_idx, "post_attention_layernorm.weight"));
        tensors.push(spec.layer_tensor(layer_idx, "pre_feedforward_layernorm.weight"));
        tensors.push(spec.layer_tensor(layer_idx, "post_feedforward_layernorm.weight"));
        tensors.push(spec.self_attn_tensor(layer_idx, "q_norm.weight"));
        if spec.requires_key_value_projection(layer_idx) {
            tensors.push(spec.self_attn_tensor(layer_idx, "k_norm.weight"));
        }
        if spec.uses_per_layer_input() {
            tensors.push(spec.layer_tensor(layer_idx, "post_per_layer_input_norm.weight"));
        }
        if spec.uses_moe() {
            tensors.push(spec.layer_tensor(layer_idx, "router.per_expert_scale"));
            tensors.push(spec.layer_tensor(layer_idx, "router.scale"));
            tensors.push(spec.layer_tensor(layer_idx, "pre_feedforward_layernorm_2.weight"));
            tensors.push(spec.layer_tensor(layer_idx, "post_feedforward_layernorm_1.weight"));
            tensors.push(spec.layer_tensor(layer_idx, "post_feedforward_layernorm_2.weight"));
        }
    }
    tensors
}

fn gemma_layer_cache_for_spec(
    spec: &GemmaModelSpec,
    layer_idx: usize,
    max_tokens: usize,
) -> Result<GemmaLayerCache, TensorLoadError> {
    let dims = GemmaAttentionDims::from_spec(spec, layer_idx)?;
    let kind = spec.layer_kinds[layer_idx];
    let max_tokens = match kind {
        GemmaAttentionKind::SlidingAttention => {
            let sliding_window = spec.sliding_window as usize;
            max_tokens.min(sliding_window).max(1)
        }
        GemmaAttentionKind::FullAttention => max_tokens.max(1),
    };
    LayerKvCache::new(max_tokens, dims.num_key_value_heads, dims.head_dim)
        .map(GemmaLayerCache::Attention)
        .map_err(|err| {
            TensorLoadError::integrity(format!(
                "Gemma layer{layer_idx} attention cache allocation failed: {err}"
            ))
        })
}

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

pub async fn gemma_prefill_sequence_with_cache(
    store: &SafeTensorShardStore,
    spec: &GemmaModelSpec,
    token_ids: &[usize],
    caches: &mut [GemmaLayerCache],
    scratch: &mut InferenceScratchpad,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    gemma_prefill_sequence_with_cache_with_matvec(
        store,
        spec,
        token_ids,
        caches,
        &CpuNativeMatvecBackend,
        scratch,
    )
    .await
}

pub async fn gemma_prefill_sequence_with_cache_with_matvec(
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
            gemma_per_layer_inputs_sequence_with_matvec(
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
        hidden_states = gemma_decoder_layer_sequence_with_cache_with_matvec(
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
    scratch: &mut InferenceScratchpad,
) -> Result<Vec<f32>, TensorLoadError> {
    gemma_decode_token_with_cache_with_matvec(
        store,
        spec,
        token_id,
        caches,
        &CpuNativeMatvecBackend,
        scratch,
    )
    .await
}

pub async fn gemma_decode_token_with_cache_with_matvec(
    store: &SafeTensorShardStore,
    spec: &GemmaModelSpec,
    token_id: usize,
    caches: &mut [GemmaLayerCache],
    matvec: &impl NativeMatvecBackend,
    scratch: &mut InferenceScratchpad,
) -> Result<Vec<f32>, TensorLoadError> {
    let hidden_states = gemma_prefill_sequence_with_cache_with_matvec(
        store,
        spec,
        &[token_id],
        caches,
        matvec,
        scratch,
    )
    .await?;
    hidden_states
        .into_iter()
        .next()
        .ok_or_else(|| TensorLoadError::integrity("Gemma decode returned no hidden state"))
}

fn gemma_embedding_sequence_for_spec(
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

async fn gemma_per_layer_inputs_sequence_with_matvec(
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
    let projected = matvec
        .bf16_matvecs_row_major_f32(
            store,
            &spec.per_layer_model_projection_weight(),
            input_embeddings,
        )
        .await?;
    if projected.len() != token_ids.len() {
        return Err(TensorLoadError::integrity(format!(
            "Gemma PLE projection count {} must match token count {}",
            projected.len(),
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
        let projected_token = &projected[token_idx];
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

#[allow(clippy::too_many_arguments)]
async fn gemma_decoder_layer_sequence_with_cache_with_matvec(
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
    let attention_output = gemma_layer_attention_sequence_with_cache_with_matvec(
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
        gemma_layer_dense_mlp_with_matvec(
            store,
            spec,
            layer_idx,
            hidden,
            matvec,
            scratch,
            &mut output,
        )
        .await?;
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
        output = gemma_apply_per_layer_input_sequence_with_matvec(
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

async fn gemma_apply_per_layer_input_sequence_with_matvec(
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

async fn gemma_layer_attention_sequence_with_cache_with_matvec(
    store: &SafeTensorShardStore,
    spec: &GemmaModelSpec,
    layer_idx: usize,
    hidden_states: &[Vec<f32>],
    caches: &mut [GemmaLayerCache],
    matvec: &impl NativeMatvecBackend,
    _scratch: &mut InferenceScratchpad,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    if hidden_states.is_empty() {
        return Ok(Vec::new());
    }
    let dims = GemmaAttentionDims::from_spec(spec, layer_idx)?;
    let attention_dim = dims.attention_dim().map_err(|err| {
        TensorLoadError::integrity(format!("Gemma attention shape failed: {err}"))
    })?;
    let key_value_dim = dims
        .key_value_dim()
        .map_err(|err| TensorLoadError::integrity(format!("Gemma KV shape failed: {err}")))?;
    let kind = spec.layer_kinds[layer_idx];
    let cache_idx = gemma_cache_index_for_layer(spec, layer_idx)?;
    let is_shared_layer = spec.is_kv_shared_layer(layer_idx);
    let cache = caches.get_mut(cache_idx).ok_or_else(|| {
        TensorLoadError::integrity(format!(
            "Gemma layer{layer_idx} cache index {cache_idx} is missing"
        ))
    })?;
    let GemmaLayerCache::Attention(cache) = cache;
    require_gemma_attention_cache_shape(&dims, cache, layer_idx)?;

    let q_proj = matvec
        .bf16_matvecs_row_major_f32_flat(
            store,
            &spec.self_attn_tensor(layer_idx, "q_proj.weight"),
            hidden_states,
        )
        .await?;
    let k_proj = if is_shared_layer {
        None
    } else {
        Some(
            matvec
                .bf16_matvecs_row_major_f32_flat(
                    store,
                    &spec.self_attn_tensor(layer_idx, "k_proj.weight"),
                    hidden_states,
                )
                .await?,
        )
    };
    let use_k_eq_v = spec.attention_k_eq_v && matches!(kind, GemmaAttentionKind::FullAttention);
    let v_proj = if is_shared_layer || use_k_eq_v {
        None
    } else {
        Some(
            matvec
                .bf16_matvecs_row_major_f32_flat(
                    store,
                    &spec.self_attn_tensor(layer_idx, "v_proj.weight"),
                    hidden_states,
                )
                .await?,
        )
    };
    let q_norm_weight =
        store.bf16_tensor_f32_cached_arc(&spec.self_attn_tensor(layer_idx, "q_norm.weight"))?;
    let k_norm_weight = if is_shared_layer {
        None
    } else {
        Some(store.bf16_tensor_f32_cached_arc(&spec.self_attn_tensor(layer_idx, "k_norm.weight"))?)
    };
    let o_proj_tensor = spec.self_attn_tensor(layer_idx, "o_proj.weight");
    let position_offset = if is_shared_layer {
        cache.next_position().saturating_sub(hidden_states.len())
    } else {
        cache.next_position()
    };
    let rotary = gemma_rotary_config(spec, kind);
    let q_proj_rows = NativeF32Rows::from_batched_matvec(&q_proj).map_err(|err| {
        TensorLoadError::integrity(format!("Gemma q projection rows failed: {err}"))
    })?;
    let k_proj_rows = k_proj
        .as_ref()
        .map(NativeF32Rows::from_batched_matvec)
        .transpose()
        .map_err(|err| {
            TensorLoadError::integrity(format!("Gemma k projection rows failed: {err}"))
        })?;
    let v_proj_rows = v_proj
        .as_ref()
        .map(NativeF32Rows::from_batched_matvec)
        .transpose()
        .map_err(|err| {
            TensorLoadError::integrity(format!("Gemma v projection rows failed: {err}"))
        })?;
    let attention_values_len = hidden_states
        .len()
        .checked_mul(attention_dim)
        .ok_or_else(|| TensorLoadError::integrity("Gemma attention sequence shape overflow"))?;
    let key_values_len = hidden_states
        .len()
        .checked_mul(key_value_dim)
        .ok_or_else(|| TensorLoadError::integrity("Gemma KV sequence shape overflow"))?;
    let mut queries = vec![0.0; attention_values_len];
    let mut keys = if is_shared_layer {
        Vec::new()
    } else {
        vec![0.0; key_values_len]
    };
    let mut values = if is_shared_layer {
        Vec::new()
    } else {
        vec![0.0; key_values_len]
    };
    let mut source_counts = if is_shared_layer {
        Vec::with_capacity(hidden_states.len())
    } else {
        Vec::new()
    };
    for token_idx in 0..hidden_states.len() {
        let q_projection = q_proj_rows.row(token_idx);
        require_len("Gemma q projection", q_projection.len(), attention_dim)
            .map_err(|err| TensorLoadError::integrity(err.to_string()))?;
        let position = position_offset
            .checked_add(token_idx)
            .ok_or_else(|| TensorLoadError::integrity("Gemma RoPE position overflow"))?;
        let query = gemma_projected_heads_normed_and_rotary(
            q_projection,
            dims.num_attention_heads,
            dims.head_dim,
            q_norm_weight.as_ref(),
            spec.rms_norm_eps,
            position,
            rotary,
        )?;
        let query_row_start = token_idx * attention_dim;
        queries[query_row_start..query_row_start + attention_dim].copy_from_slice(&query);

        if !is_shared_layer {
            let k_projection = k_proj_rows
                .expect("non-shared Gemma attention has k projection")
                .row(token_idx);
            require_len("Gemma k projection", k_projection.len(), key_value_dim)
                .map_err(|err| TensorLoadError::integrity(err.to_string()))?;
            let key = gemma_projected_heads_normed_and_rotary(
                k_projection,
                dims.num_key_value_heads,
                dims.head_dim,
                k_norm_weight
                    .as_ref()
                    .expect("non-shared Gemma attention has k norm")
                    .as_ref(),
                spec.rms_norm_eps,
                position,
                rotary,
            )?;
            let value_source = if use_k_eq_v {
                k_projection
            } else {
                let v_projection = v_proj_rows
                    .expect("Gemma attention without K=V has v projection")
                    .row(token_idx);
                require_len("Gemma v projection", v_projection.len(), key_value_dim)
                    .map_err(|err| TensorLoadError::integrity(err.to_string()))?;
                v_projection
            };
            let value = gemma_projected_heads_rms_norm_no_scale(
                value_source,
                dims.num_key_value_heads,
                dims.head_dim,
                spec.rms_norm_eps,
            )?;
            let key_row_start = token_idx * key_value_dim;
            keys[key_row_start..key_row_start + key_value_dim].copy_from_slice(&key);
            values[key_row_start..key_row_start + key_value_dim].copy_from_slice(&value);
        } else {
            source_counts.push(
                position
                    .checked_add(1)
                    .unwrap_or(cache.token_count())
                    .min(cache.token_count()),
            );
        }
    }

    let attention_output = if is_shared_layer {
        native_full_attention_sequence_from_cache_parts_with_matvec(
            dims.native(),
            &NativeFullAttentionCacheSequenceParts {
                queries: NativeF32Rows::flat(&queries, attention_dim)
                    .map_err(|err| TensorLoadError::integrity(err.to_string()))?,
                gates: None,
                source_counts: &source_counts,
                output_projection: NativeOutputProjection::Bf16Tensor {
                    store,
                    tensor: &o_proj_tensor,
                },
                score_scale: 1.0,
            },
            cache,
            matvec,
        )
        .await
    } else {
        native_full_attention_sequence_with_cache_from_parts_with_matvec(
            dims.native(),
            &NativeFullAttentionSequenceParts {
                queries: NativeF32Rows::flat(&queries, attention_dim)
                    .map_err(|err| TensorLoadError::integrity(err.to_string()))?,
                keys: NativeF32Rows::flat(&keys, key_value_dim)
                    .map_err(|err| TensorLoadError::integrity(err.to_string()))?,
                values: NativeF32Rows::flat(&values, key_value_dim)
                    .map_err(|err| TensorLoadError::integrity(err.to_string()))?,
                gates: None,
                output_projection: NativeOutputProjection::Bf16Tensor {
                    store,
                    tensor: &o_proj_tensor,
                },
                score_scale: 1.0,
            },
            cache,
            matvec,
        )
        .await
    }
    .map_err(|err| {
        TensorLoadError::integrity(format!("Gemma layer{layer_idx} attention failed: {err}"))
    })?;
    Ok(attention_output)
}

async fn gemma_layer_dense_mlp_with_matvec(
    store: &SafeTensorShardStore,
    spec: &GemmaModelSpec,
    layer_idx: usize,
    hidden_states: &[f32],
    matvec: &impl NativeMatvecBackend,
    scratch: &mut InferenceScratchpad,
    output: &mut [f32],
) -> Result<(), TensorLoadError> {
    let hidden_size = spec.hidden_size as usize;
    let intermediate_size = gemma_intermediate_size_for_layer(spec, layer_idx);
    if hidden_states.len() != hidden_size {
        return Err(TensorLoadError::integrity(format!(
            "Gemma dense MLP hidden length {} must match hidden size {hidden_size}",
            hidden_states.len()
        )));
    }
    let gate = InferenceScratchpad::get_mut(&mut scratch.buf0, intermediate_size);
    matvec
        .bf16_matvec_row_major_f32_in_place(
            store,
            &spec.mlp_tensor(layer_idx, "gate_proj.weight"),
            hidden_states,
            gate,
        )
        .await?;

    let up = InferenceScratchpad::get_mut(&mut scratch.buf1, intermediate_size);
    matvec
        .bf16_matvec_row_major_f32_in_place(
            store,
            &spec.mlp_tensor(layer_idx, "up_proj.weight"),
            hidden_states,
            up,
        )
        .await?;

    if gate.len() != intermediate_size || up.len() != intermediate_size {
        return Err(TensorLoadError::integrity(format!(
            "Gemma dense MLP gate/up lengths {}, {} must match intermediate size {intermediate_size}",
            gate.len(),
            up.len()
        )));
    }

    let activated = InferenceScratchpad::get_mut(&mut scratch.buf2, intermediate_size);
    for (a, (g, u)) in activated.iter_mut().zip(gate.iter().zip(up.iter())) {
        *a = gelu_pytorch_tanh_f32(*g) * *u;
    }

    matvec
        .bf16_matvec_row_major_f32_in_place(
            store,
            &spec.mlp_tensor(layer_idx, "down_proj.weight"),
            activated,
            output,
        )
        .await?;

    if output.len() != hidden_size {
        return Err(TensorLoadError::integrity(format!(
            "Gemma dense MLP down output length {} must match hidden size {hidden_size}",
            output.len()
        )));
    }
    Ok(())
}

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
) -> Result<Vec<TopKLogit>, TensorLoadError> {
    gemma_lm_head_top_k_for_spec_with_matvec(
        store,
        spec,
        hidden_states,
        top_k,
        chunk_rows,
        &CpuNativeMatvecBackend,
    )
    .await
}

pub async fn gemma_lm_head_top_k_for_spec_with_matvec(
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

pub async fn gemma_lm_head_logits_for_spec_with_matvec(
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

fn gemma_layer_norm_sequence(
    store: &SafeTensorShardStore,
    spec: &GemmaModelSpec,
    layer_idx: usize,
    suffix: &str,
    hidden_states: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    gemma_norm_sequence_after_projection(store, spec, layer_idx, suffix, hidden_states)
}

fn gemma_norm_sequence_after_projection(
    store: &SafeTensorShardStore,
    spec: &GemmaModelSpec,
    layer_idx: usize,
    suffix: &str,
    hidden_states: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    let hidden_size = spec.hidden_size as usize;
    let norm_weight = store.bf16_tensor_f32_cached_arc(&spec.layer_tensor(layer_idx, suffix))?;
    hidden_states
        .iter()
        .map(|hidden| {
            if hidden.len() != hidden_size {
                return Err(TensorLoadError::integrity(format!(
                    "Gemma hidden length {} must match hidden size {hidden_size}",
                    hidden.len()
                )));
            }
            let mut output = vec![0.0; hidden_size];
            rms_norm_f32_in_place(hidden, norm_weight.as_ref(), spec.rms_norm_eps, &mut output)
                .map_err(|err| {
                    TensorLoadError::integrity(format!("Gemma layer RMSNorm failed: {err}"))
                })?;
            Ok(output)
        })
        .collect()
}

fn add_sequence(
    left: &[Vec<f32>],
    right: &[Vec<f32>],
    hidden_size: usize,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    if left.len() != right.len() {
        return Err(TensorLoadError::integrity(
            "Gemma residual sequence lengths must match",
        ));
    }
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            if left.len() != hidden_size || right.len() != hidden_size {
                return Err(TensorLoadError::integrity(format!(
                    "Gemma residual hidden lengths {}, {} must match hidden size {hidden_size}",
                    left.len(),
                    right.len()
                )));
            }
            Ok(left
                .iter()
                .zip(right)
                .map(|(left, right)| left + right)
                .collect())
        })
        .collect()
}

fn apply_gemma_layer_scalar(
    store: &SafeTensorShardStore,
    spec: &GemmaModelSpec,
    layer_idx: usize,
    hidden_states: &mut [Vec<f32>],
) -> Result<(), TensorLoadError> {
    let scalar = store.bf16_tensor_f32_cached_arc(&spec.layer_tensor(layer_idx, "layer_scalar"))?;
    match scalar.as_ref() {
        [value] => {
            for hidden in hidden_states {
                for item in hidden {
                    *item *= *value;
                }
            }
            Ok(())
        }
        values if values.len() == spec.hidden_size as usize => {
            for hidden in hidden_states {
                for (item, scale) in hidden.iter_mut().zip(values) {
                    *item *= scale;
                }
            }
            Ok(())
        }
        values => Err(TensorLoadError::integrity(format!(
            "Gemma layer scalar length {} must be 1 or hidden size {}",
            values.len(),
            spec.hidden_size
        ))),
    }
}

fn gemma_projected_heads_normed_and_rotary(
    projected: &[f32],
    head_count: usize,
    head_dim: usize,
    norm_weight: &[f32],
    eps: f32,
    position: usize,
    rotary: GemmaRotaryConfig,
) -> Result<Vec<f32>, TensorLoadError> {
    require_len("Gemma attention norm weight", norm_weight.len(), head_dim)
        .map_err(|err| TensorLoadError::integrity(err.to_string()))?;
    let expected = head_count
        .checked_mul(head_dim)
        .ok_or_else(|| TensorLoadError::integrity("Gemma projected head shape overflow"))?;
    require_len("Gemma projected heads", projected.len(), expected)
        .map_err(|err| TensorLoadError::integrity(err.to_string()))?;
    let mut output = vec![0.0; expected];
    for head in 0..head_count {
        let start = head * head_dim;
        let mut normalized = vec![0.0; head_dim];
        rms_norm_f32_in_place(
            &projected[start..start + head_dim],
            norm_weight,
            eps,
            &mut normalized,
        )
        .map_err(|err| {
            TensorLoadError::integrity(format!("Gemma attention RMSNorm failed: {err}"))
        })?;
        apply_rope_to_head(
            &mut normalized,
            position,
            rotary.rotary_dim(head_dim)?,
            rotary.theta,
        );
        output[start..start + head_dim].copy_from_slice(&normalized);
    }
    Ok(output)
}

fn gemma_projected_heads_rms_norm_no_scale(
    projected: &[f32],
    head_count: usize,
    head_dim: usize,
    eps: f32,
) -> Result<Vec<f32>, TensorLoadError> {
    let expected = head_count
        .checked_mul(head_dim)
        .ok_or_else(|| TensorLoadError::integrity("Gemma value head shape overflow"))?;
    require_len("Gemma value heads", projected.len(), expected)
        .map_err(|err| TensorLoadError::integrity(err.to_string()))?;
    let mut output = vec![0.0; expected];
    for head in 0..head_count {
        let start = head * head_dim;
        let normalized =
            rms_norm_no_scale_f32(&projected[start..start + head_dim], eps).map_err(|err| {
                TensorLoadError::integrity(format!("Gemma value RMSNorm failed: {err}"))
            })?;
        output[start..start + head_dim].copy_from_slice(&normalized);
    }
    Ok(output)
}

fn rms_norm_no_scale_f32(input: &[f32], eps: f32) -> Result<Vec<f32>, MathError> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    if eps < 0.0 {
        return Err(MathError::InvalidShape(
            "rms norm epsilon must be non-negative".to_owned(),
        ));
    }
    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let scale = (mean_square + eps).sqrt().recip();
    Ok(input.iter().map(|value| value * scale).collect())
}

fn require_gemma_attention_cache_shape(
    dims: &GemmaAttentionDims,
    cache: &LayerKvCache,
    layer_idx: usize,
) -> Result<(), TensorLoadError> {
    if cache.key_value_heads() != dims.num_key_value_heads || cache.head_dim() != dims.head_dim {
        return Err(TensorLoadError::integrity(format!(
            "Gemma layer{layer_idx} attention cache shape does not match dims: cache key_value_heads={}, head_dim={}; dims key_value_heads={}, head_dim={}",
            cache.key_value_heads(),
            cache.head_dim(),
            dims.num_key_value_heads,
            dims.head_dim
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct GemmaRotaryConfig {
    theta: f32,
    partial_rotary_factor: f32,
}

impl GemmaRotaryConfig {
    fn rotary_dim(&self, head_dim: usize) -> Result<usize, TensorLoadError> {
        if self.theta <= 0.0 || self.partial_rotary_factor < 0.0 {
            return Err(TensorLoadError::integrity(
                "Gemma RoPE parameters must be positive",
            ));
        }
        let rotary_dim = ((head_dim as f32) * self.partial_rotary_factor).round() as usize;
        if rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
            return Err(TensorLoadError::integrity(format!(
                "Gemma rotary dimension {rotary_dim} must be even and <= head dim {head_dim}"
            )));
        }
        Ok(rotary_dim)
    }
}

fn gemma_rotary_config(spec: &GemmaModelSpec, kind: GemmaAttentionKind) -> GemmaRotaryConfig {
    match kind {
        GemmaAttentionKind::SlidingAttention => GemmaRotaryConfig {
            theta: spec.sliding_rope_theta,
            partial_rotary_factor: 1.0,
        },
        GemmaAttentionKind::FullAttention => GemmaRotaryConfig {
            theta: spec.full_rope_theta,
            partial_rotary_factor: spec.full_partial_rotary_factor,
        },
    }
}

fn gemma_intermediate_size_for_layer(spec: &GemmaModelSpec, layer_idx: usize) -> usize {
    let multiplier = if spec.use_double_wide_mlp && spec.is_kv_shared_layer(layer_idx) {
        2
    } else {
        1
    };
    spec.intermediate_size as usize * multiplier
}

fn gemma_concrete_cache_count(spec: &GemmaModelSpec) -> Result<usize, TensorLoadError> {
    let layer_count = spec.num_hidden_layers as usize;
    if spec.layer_kinds.len() != layer_count {
        return Err(TensorLoadError::integrity(format!(
            "Gemma spec declares {layer_count} layers but has {} attention kind entries",
            spec.layer_kinds.len()
        )));
    }
    if spec.num_kv_shared_layers == 0 {
        return Ok(layer_count);
    }
    let first_shared = first_gemma_kv_shared_layer_idx(spec);
    if first_shared == 0 {
        return Err(TensorLoadError::unsupported(
            "Gemma KV sharing requires at least one concrete layer before shared layers",
        ));
    }
    for layer_idx in first_shared..layer_count {
        gemma_cache_index_for_layer(spec, layer_idx)?;
    }
    Ok(first_shared)
}

fn gemma_cache_index_for_layer(
    spec: &GemmaModelSpec,
    layer_idx: usize,
) -> Result<usize, TensorLoadError> {
    let first_shared = first_gemma_kv_shared_layer_idx(spec);
    if layer_idx < first_shared {
        return Ok(layer_idx);
    }
    let kind = *spec.layer_kinds.get(layer_idx).ok_or_else(|| {
        TensorLoadError::missing(format!(
            "Gemma layer {layer_idx} is outside configured layer count"
        ))
    })?;
    spec.layer_kinds[..first_shared]
        .iter()
        .enumerate()
        .rev()
        .find_map(|(idx, candidate)| (*candidate == kind).then_some(idx))
        .ok_or_else(|| {
            TensorLoadError::unsupported(format!(
                "Gemma shared layer {layer_idx} has no earlier concrete {kind:?} cache"
            ))
        })
}

fn first_gemma_kv_shared_layer_idx(spec: &GemmaModelSpec) -> usize {
    (spec.num_hidden_layers - spec.num_kv_shared_layers) as usize
}

fn ensure_supported_gemma_execution(spec: &GemmaModelSpec) -> Result<(), TensorLoadError> {
    if spec.uses_moe() {
        return Err(TensorLoadError::unsupported(
            "Gemma MoE native execution is not implemented yet",
        ));
    }
    Ok(())
}

fn gelu_pytorch_tanh_f32(value: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    0.5 * value * (1.0 + (SQRT_2_OVER_PI * (value + 0.044_715 * value.powi(3))).tanh())
}
