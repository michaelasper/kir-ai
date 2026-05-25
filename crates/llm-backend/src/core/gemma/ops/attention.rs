use super::super::super::math::{
    MathError, apply_rope_to_head, require_len, rms_norm_f32_in_place, rms_norm_scale_f32,
};
use super::super::super::native_attention::{
    NativeF32Rows, NativeFullAttentionCacheSequenceParts, NativeFullAttentionDims,
    NativeFullAttentionSequenceParts, NativeOutputProjection,
    native_full_attention_sequence_from_cache_parts,
    native_full_attention_sequence_with_cache_from_parts,
};
use super::super::super::{
    InferenceScratchpad, LayerKvCache, NativeBatchedMatvecInputBuffer, NativeMatvecBackend,
    SafeTensorShardStore, TensorLoadError,
};
use super::cache::{GemmaLayerCache, gemma_cache_index_for_layer};
use llm_models::{GemmaAttentionKind, GemmaModelSpec};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GemmaAttentionDims {
    pub(crate) hidden_size: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,
}

impl GemmaAttentionDims {
    pub(crate) fn from_spec(
        spec: &GemmaModelSpec,
        layer_idx: usize,
    ) -> Result<Self, TensorLoadError> {
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

pub(crate) async fn gemma_layer_attention_sequence_with_cache(
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

    let input_columns = hidden_states.first().map_or(0, Vec::len);
    let flat_hidden_states =
        NativeBatchedMatvecInputBuffer::from_rows(hidden_states, input_columns)?;
    let q_proj_tensor = spec.self_attn_tensor(layer_idx, "q_proj.weight");
    let k_proj_tensor = spec.self_attn_tensor(layer_idx, "k_proj.weight");
    let v_proj_tensor = spec.self_attn_tensor(layer_idx, "v_proj.weight");
    let q_proj = matvec
        .bf16_matvecs_row_major_f32_flat_inputs(
            store,
            &q_proj_tensor,
            flat_hidden_states.values(),
            flat_hidden_states.input_count(),
        )
        .await?;
    let k_proj = if is_shared_layer {
        None
    } else {
        Some(
            matvec
                .bf16_matvecs_row_major_f32_flat_inputs(
                    store,
                    &k_proj_tensor,
                    flat_hidden_states.values(),
                    flat_hidden_states.input_count(),
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
                .bf16_matvecs_row_major_f32_flat_inputs(
                    store,
                    &v_proj_tensor,
                    flat_hidden_states.values(),
                    flat_hidden_states.input_count(),
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
    let rotary = gemma_rotary_config(spec, kind)?;
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
        native_full_attention_sequence_from_cache_parts(
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
        native_full_attention_sequence_with_cache_from_parts(
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
    let scale = rms_norm_scale_f32(mean_square, eps);
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

fn gemma_rotary_config(
    spec: &GemmaModelSpec,
    kind: GemmaAttentionKind,
) -> Result<GemmaRotaryConfig, TensorLoadError> {
    Ok(match kind {
        GemmaAttentionKind::SlidingAttention => GemmaRotaryConfig {
            theta: spec.sliding_rope_theta,
            partial_rotary_factor: 1.0,
        },
        GemmaAttentionKind::FullAttention => GemmaRotaryConfig {
            theta: spec.full_rope_theta,
            partial_rotary_factor: spec.full_partial_rotary_factor,
        },
        _ => {
            return Err(TensorLoadError::unsupported(format!(
                "Gemma uses unsupported attention kind `{kind:?}`"
            )));
        }
    })
}
