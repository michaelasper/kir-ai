#![allow(dead_code)]

use super::super::super::math::{
    InferenceScratchpad, MathError, apply_rope_to_head, require_len,
    rms_norm_f32_in_place as cpu_rms_norm_f32_in_place,
    rms_norm_one_centered_f32_in_place as cpu_rms_norm_one_centered_f32_in_place, sigmoid_f32,
};
use super::super::super::native_attention::{
    NativeF32Rows, NativeFullAttentionDims, NativeFullAttentionSequenceParts,
    NativeFullAttentionStepParts, NativeOutputProjection,
    native_full_attention_sequence_from_parts,
    native_full_attention_sequence_with_cache_from_parts,
    native_full_attention_step_with_cache_from_parts, native_output_projection,
    require_full_attention_cache_shape,
};
use super::super::super::{
    CpuNativeMatvecBackend, LayerKvCache, NativeBatchedMatvecInputBuffer, NativeMatvecBackend,
    SafeTensorShardStore, TensorLoadError,
};
use super::super::matvec::rms_norm_f32;
use llm_models::QwenModelSpec;

pub(crate) struct QwenFullAttentionSequenceParts<'a> {
    pub q_proj: NativeF32Rows<'a>,
    pub k_proj: NativeF32Rows<'a>,
    pub v_proj: NativeF32Rows<'a>,
    pub q_norm_weight: &'a [f32],
    pub k_norm_weight: &'a [f32],
    pub o_proj_weight: NativeOutputProjection<'a>,
}

pub(crate) struct QwenFullAttentionStepParts<'a> {
    pub q_proj: &'a [f32],
    pub k_proj: &'a [f32],
    pub v_proj: &'a [f32],
    pub q_norm_weight: &'a [f32],
    pub k_norm_weight: &'a [f32],
    pub o_proj_weight: NativeOutputProjection<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct QwenFullAttentionSequenceConfig {
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    pub q_projection_gate: bool,
    pub one_centered_rms_norm: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct QwenFullAttentionDims {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
}

impl QwenFullAttentionDims {
    pub fn from_spec(spec: &QwenModelSpec) -> Self {
        Self {
            hidden_size: spec.hidden_size as usize,
            num_attention_heads: spec.num_attention_heads as usize,
            num_key_value_heads: spec.num_key_value_heads as usize,
            head_dim: spec.head_dim as usize,
        }
    }

    fn native(self) -> NativeFullAttentionDims {
        NativeFullAttentionDims {
            hidden_size: self.hidden_size,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
        }
    }

    fn attention_dim(&self) -> Result<usize, MathError> {
        self.num_attention_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| MathError::InvalidShape("Qwen attention dimension overflow".to_owned()))
    }

    fn key_value_dim(&self) -> Result<usize, MathError> {
        self.num_key_value_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| MathError::InvalidShape("Qwen KV dimension overflow".to_owned()))
    }
}

async fn qwen_attention_rms_norm(
    input: &[f32],
    weight: &[f32],
    config: QwenFullAttentionSequenceConfig,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, MathError> {
    if config.one_centered_rms_norm {
        matvec
            .rms_norm_one_centered_f32(input, weight, config.rms_norm_eps)
            .await
    } else {
        rms_norm_f32(input, weight, config.rms_norm_eps, matvec).await
    }
}

fn qwen_attention_rms_norm_cpu_in_place(
    input: &[f32],
    weight: &[f32],
    config: QwenFullAttentionSequenceConfig,
    output: &mut [f32],
) -> Result<(), MathError> {
    if config.one_centered_rms_norm {
        cpu_rms_norm_one_centered_f32_in_place(input, weight, config.rms_norm_eps, output)
    } else {
        cpu_rms_norm_f32_in_place(input, weight, config.rms_norm_eps, output)
    }
}

pub(crate) async fn qwen_full_attention_first_token_from_parts(
    dims: &QwenFullAttentionDims,
    q_proj: &[f32],
    v_proj: &[f32],
    o_proj_weight: &[f32],
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, MathError> {
    if dims.num_attention_heads == 0
        || dims.num_key_value_heads == 0
        || dims.head_dim == 0
        || dims.hidden_size == 0
    {
        return Err(MathError::InvalidShape(
            "Qwen full attention dimensions must be non-zero".to_owned(),
        ));
    }
    if !dims
        .num_attention_heads
        .is_multiple_of(dims.num_key_value_heads)
    {
        return Err(MathError::InvalidShape(
            "Qwen attention heads must be divisible by key/value heads".to_owned(),
        ));
    }
    let attention_dim = dims.attention_dim()?;
    let key_value_dim = dims.key_value_dim()?;
    require_len("q projection", q_proj.len(), attention_dim * 2)?;
    require_len("v projection", v_proj.len(), key_value_dim)?;
    require_len(
        "o projection weight",
        o_proj_weight.len(),
        dims.hidden_size
            .checked_mul(attention_dim)
            .ok_or_else(|| MathError::InvalidShape("Qwen o projection overflow".to_owned()))?,
    )?;

    let groups = dims.num_attention_heads / dims.num_key_value_heads;
    let mut gated = vec![0.0; attention_dim];
    for head in 0..dims.num_attention_heads {
        let q_proj_head_start = head * dims.head_dim * 2;
        let gate_start = q_proj_head_start + dims.head_dim;
        let kv_head = head / groups;
        let value_start = kv_head * dims.head_dim;
        let output_start = head * dims.head_dim;
        for offset in 0..dims.head_dim {
            gated[output_start + offset] =
                v_proj[value_start + offset] * sigmoid_f32(q_proj[gate_start + offset]);
        }
    }

    native_output_projection(
        matvec,
        NativeOutputProjection::F32(o_proj_weight),
        &gated,
        dims.hidden_size,
        attention_dim,
    )
    .await
}

pub(crate) async fn qwen_full_attention_sequence_from_parts(
    dims: &QwenFullAttentionDims,
    parts: &QwenFullAttentionSequenceParts<'_>,
    config: QwenFullAttentionSequenceConfig,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<Vec<f32>>, MathError> {
    qwen_full_attention_sequence_from_parts_impl(dims, parts, config, None, matvec).await
}

pub(crate) async fn qwen_full_attention_sequence_with_cache_from_parts(
    dims: &QwenFullAttentionDims,
    parts: &QwenFullAttentionSequenceParts<'_>,
    config: QwenFullAttentionSequenceConfig,
    cache: &mut LayerKvCache,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<Vec<f32>>, MathError> {
    qwen_full_attention_sequence_from_parts_impl(dims, parts, config, Some(cache), matvec).await
}

pub(crate) async fn qwen_full_attention_step_with_cache_from_parts(
    dims: &QwenFullAttentionDims,
    parts: &QwenFullAttentionStepParts<'_>,
    config: QwenFullAttentionSequenceConfig,
    cache: &mut LayerKvCache,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, MathError> {
    if dims.num_attention_heads == 0
        || dims.num_key_value_heads == 0
        || dims.head_dim == 0
        || dims.hidden_size == 0
    {
        return Err(MathError::InvalidShape(
            "Qwen full attention dimensions must be non-zero".to_owned(),
        ));
    }
    if !dims
        .num_attention_heads
        .is_multiple_of(dims.num_key_value_heads)
    {
        return Err(MathError::InvalidShape(
            "Qwen attention heads must be divisible by key/value heads".to_owned(),
        ));
    }
    if config.rope_theta <= 0.0 || config.partial_rotary_factor < 0.0 {
        return Err(MathError::InvalidShape(
            "Qwen RoPE parameters must be positive".to_owned(),
        ));
    }
    require_full_attention_cache_shape("Qwen full attention", dims.native(), cache)?;
    let attention_dim = dims.attention_dim()?;
    let key_value_dim = dims.key_value_dim()?;
    let q_projection_width = if config.q_projection_gate {
        attention_dim
            .checked_mul(2)
            .ok_or_else(|| MathError::InvalidShape("Qwen q projection overflow".to_owned()))?
    } else {
        attention_dim
    };
    require_len("q projection", parts.q_proj.len(), q_projection_width)?;
    require_len("k projection", parts.k_proj.len(), key_value_dim)?;
    require_len("v projection", parts.v_proj.len(), key_value_dim)?;
    require_len("q norm weight", parts.q_norm_weight.len(), dims.head_dim)?;
    require_len("k norm weight", parts.k_norm_weight.len(), dims.head_dim)?;
    let rotary_dim = ((dims.head_dim as f32) * config.partial_rotary_factor).round() as usize;
    if rotary_dim > dims.head_dim || !rotary_dim.is_multiple_of(2) {
        return Err(MathError::InvalidShape(format!(
            "Qwen rotary dimension {rotary_dim} must be even and <= head dim {}",
            dims.head_dim
        )));
    }

    let position = cache.next_position();
    let mut query = vec![0.0; attention_dim];
    let mut gate = vec![0.0; attention_dim];
    for head in 0..dims.num_attention_heads {
        let projected_head_start = if config.q_projection_gate {
            head * dims.head_dim * 2
        } else {
            head * dims.head_dim
        };
        let q_start = head * dims.head_dim;
        qwen_attention_rms_norm_cpu_in_place(
            &parts.q_proj[projected_head_start..projected_head_start + dims.head_dim],
            parts.q_norm_weight,
            config,
            &mut query[q_start..q_start + dims.head_dim],
        )?;
        if config.q_projection_gate {
            gate[q_start..q_start + dims.head_dim].copy_from_slice(
                &parts.q_proj[projected_head_start + dims.head_dim
                    ..projected_head_start + dims.head_dim * 2],
            );
        }
        apply_rope_to_head(
            &mut query[q_start..q_start + dims.head_dim],
            position,
            rotary_dim,
            config.rope_theta,
        );
    }

    let mut key = vec![0.0; key_value_dim];
    for head in 0..dims.num_key_value_heads {
        let head_start = head * dims.head_dim;
        qwen_attention_rms_norm_cpu_in_place(
            &parts.k_proj[head_start..head_start + dims.head_dim],
            parts.k_norm_weight,
            config,
            &mut key[head_start..head_start + dims.head_dim],
        )?;
        apply_rope_to_head(
            &mut key[head_start..head_start + dims.head_dim],
            position,
            rotary_dim,
            config.rope_theta,
        );
    }
    native_full_attention_step_with_cache_from_parts(
        dims.native(),
        &NativeFullAttentionStepParts {
            query: &query,
            key: &key,
            value: parts.v_proj,
            gate: config.q_projection_gate.then_some(gate.as_slice()),
            output_projection: parts.o_proj_weight,
            score_scale: (dims.head_dim as f32).sqrt().recip(),
        },
        cache,
        matvec,
    )
    .await
}

async fn qwen_full_attention_sequence_from_parts_impl(
    dims: &QwenFullAttentionDims,
    parts: &QwenFullAttentionSequenceParts<'_>,
    config: QwenFullAttentionSequenceConfig,
    cache: Option<&mut LayerKvCache>,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<Vec<f32>>, MathError> {
    let q_proj = parts.q_proj;
    let k_proj = parts.k_proj;
    let v_proj = parts.v_proj;
    if q_proj.is_empty() {
        return Ok(Vec::new());
    }
    if dims.num_attention_heads == 0
        || dims.num_key_value_heads == 0
        || dims.head_dim == 0
        || dims.hidden_size == 0
    {
        return Err(MathError::InvalidShape(
            "Qwen full attention dimensions must be non-zero".to_owned(),
        ));
    }
    if !dims
        .num_attention_heads
        .is_multiple_of(dims.num_key_value_heads)
    {
        return Err(MathError::InvalidShape(
            "Qwen attention heads must be divisible by key/value heads".to_owned(),
        ));
    }
    if config.rope_theta <= 0.0 || config.partial_rotary_factor < 0.0 {
        return Err(MathError::InvalidShape(
            "Qwen RoPE parameters must be positive".to_owned(),
        ));
    }
    let seq_len = q_proj.len();
    if k_proj.len() != seq_len || v_proj.len() != seq_len {
        return Err(MathError::InvalidShape(
            "Qwen full attention sequence inputs must have the same length".to_owned(),
        ));
    }
    let attention_dim = dims.attention_dim()?;
    let key_value_dim = dims.key_value_dim()?;
    let q_projection_width = if config.q_projection_gate {
        attention_dim
            .checked_mul(2)
            .ok_or_else(|| MathError::InvalidShape("Qwen q projection overflow".to_owned()))?
    } else {
        attention_dim
    };
    require_len("q norm weight", parts.q_norm_weight.len(), dims.head_dim)?;
    require_len("k norm weight", parts.k_norm_weight.len(), dims.head_dim)?;
    let rotary_dim = ((dims.head_dim as f32) * config.partial_rotary_factor).round() as usize;
    if rotary_dim > dims.head_dim || !rotary_dim.is_multiple_of(2) {
        return Err(MathError::InvalidShape(format!(
            "Qwen rotary dimension {rotary_dim} must be even and <= head dim {}",
            dims.head_dim
        )));
    }

    let position_offset = cache.as_deref().map_or(0, LayerKvCache::next_position);
    let attention_values_len = seq_len.checked_mul(attention_dim).ok_or_else(|| {
        MathError::InvalidShape("Qwen attention sequence shape overflow".to_owned())
    })?;
    let key_values_len = seq_len
        .checked_mul(key_value_dim)
        .ok_or_else(|| MathError::InvalidShape("Qwen KV sequence shape overflow".to_owned()))?;
    let mut queries = vec![0.0; attention_values_len];
    let mut gates = vec![0.0; attention_values_len];
    let mut keys = vec![0.0; key_values_len];
    for token_idx in 0..seq_len {
        let position = position_offset
            .checked_add(token_idx)
            .ok_or_else(|| MathError::InvalidShape("Qwen RoPE position overflow".to_owned()))?;
        let q_projection = q_proj.row(token_idx);
        let k_projection = k_proj.row(token_idx);
        let v_projection = v_proj.row(token_idx);
        require_len("q projection", q_projection.len(), q_projection_width)?;
        require_len("k projection", k_projection.len(), key_value_dim)?;
        require_len("v projection", v_projection.len(), key_value_dim)?;
        let query_row_start = token_idx * attention_dim;
        let key_row_start = token_idx * key_value_dim;
        let query_row = &mut queries[query_row_start..query_row_start + attention_dim];
        let gate_row = &mut gates[query_row_start..query_row_start + attention_dim];
        let key_row = &mut keys[key_row_start..key_row_start + key_value_dim];

        for head in 0..dims.num_attention_heads {
            let projected_head_start = if config.q_projection_gate {
                head * dims.head_dim * 2
            } else {
                head * dims.head_dim
            };
            let q_start = head * dims.head_dim;
            let query = qwen_attention_rms_norm(
                &q_projection[projected_head_start..projected_head_start + dims.head_dim],
                parts.q_norm_weight,
                config,
                matvec,
            )
            .await?;
            query_row[q_start..q_start + dims.head_dim].copy_from_slice(&query);
            if config.q_projection_gate {
                gate_row[q_start..q_start + dims.head_dim].copy_from_slice(
                    &q_projection[projected_head_start + dims.head_dim
                        ..projected_head_start + dims.head_dim * 2],
                );
            }
            apply_rope_to_head(
                &mut query_row[q_start..q_start + dims.head_dim],
                position,
                rotary_dim,
                config.rope_theta,
            );
        }
        for head in 0..dims.num_key_value_heads {
            let head_start = head * dims.head_dim;
            let key = qwen_attention_rms_norm(
                &k_projection[head_start..head_start + dims.head_dim],
                parts.k_norm_weight,
                config,
                matvec,
            )
            .await?;
            key_row[head_start..head_start + dims.head_dim].copy_from_slice(&key);
            apply_rope_to_head(
                &mut key_row[head_start..head_start + dims.head_dim],
                position,
                rotary_dim,
                config.rope_theta,
            );
        }
    }
    let generic_parts = NativeFullAttentionSequenceParts {
        queries: NativeF32Rows::flat(&queries, attention_dim)?,
        keys: NativeF32Rows::flat(&keys, key_value_dim)?,
        values: v_proj,
        gates: config
            .q_projection_gate
            .then(|| NativeF32Rows::flat(&gates, attention_dim))
            .transpose()?,
        output_projection: parts.o_proj_weight,
        score_scale: (dims.head_dim as f32).sqrt().recip(),
    };
    if let Some(cache) = cache {
        native_full_attention_sequence_with_cache_from_parts(
            dims.native(),
            &generic_parts,
            cache,
            matvec,
        )
        .await
    } else {
        native_full_attention_sequence_from_parts(dims.native(), &generic_parts, matvec).await
    }
}

pub(crate) async fn qwen_layer_full_attention_first_token(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[f32],
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    let dims = QwenFullAttentionDims::from_spec(spec);
    let q_proj = matvec
        .bf16_matvec_row_major_f32(
            store,
            &spec.self_attn_tensor(layer_idx, "q_proj.weight"),
            hidden_states,
        )
        .await?;
    let k_proj = matvec
        .bf16_matvec_row_major_f32(
            store,
            &spec.self_attn_tensor(layer_idx, "k_proj.weight"),
            hidden_states,
        )
        .await?;
    let v_proj = matvec
        .bf16_matvec_row_major_f32(
            store,
            &spec.self_attn_tensor(layer_idx, "v_proj.weight"),
            hidden_states,
        )
        .await?;
    let q_norm_weight =
        store.bf16_tensor_f32_cached_arc(&spec.self_attn_tensor(layer_idx, "q_norm.weight"))?;
    let k_norm_weight =
        store.bf16_tensor_f32_cached_arc(&spec.self_attn_tensor(layer_idx, "k_norm.weight"))?;
    let o_proj_tensor = spec.self_attn_tensor(layer_idx, "o_proj.weight");
    let q_proj = vec![q_proj];
    let k_proj = vec![k_proj];
    let v_proj = vec![v_proj];
    qwen_full_attention_sequence_from_parts(
        &dims,
        &QwenFullAttentionSequenceParts {
            q_proj: NativeF32Rows::from_rows(&q_proj),
            k_proj: NativeF32Rows::from_rows(&k_proj),
            v_proj: NativeF32Rows::from_rows(&v_proj),
            q_norm_weight: q_norm_weight.as_ref(),
            k_norm_weight: k_norm_weight.as_ref(),
            o_proj_weight: NativeOutputProjection::Bf16Tensor {
                store,
                tensor: &o_proj_tensor,
            },
        },
        QwenFullAttentionSequenceConfig {
            rms_norm_eps: spec.rms_norm_eps,
            rope_theta: spec.rope_theta,
            partial_rotary_factor: spec.partial_rotary_factor,
            q_projection_gate: !spec.is_qwen3_dense(),
            one_centered_rms_norm: !spec.is_qwen3_dense(),
        },
        matvec,
    )
    .await
    .map(|mut outputs| outputs.remove(0))
    .map_err(|err| {
        TensorLoadError::integrity(format!(
            "Qwen layer{layer_idx} full attention failed: {err}"
        ))
    })
}

pub(crate) async fn qwen_layer_full_attention_sequence(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    qwen_layer_full_attention_sequence_impl(
        store,
        spec,
        layer_idx,
        hidden_states,
        None,
        &CpuNativeMatvecBackend,
    )
    .await
}

pub(crate) async fn qwen_layer_full_attention_sequence_with_cache(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[Vec<f32>],
    cache: &mut LayerKvCache,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    qwen_layer_full_attention_sequence_impl(
        store,
        spec,
        layer_idx,
        hidden_states,
        Some(cache),
        matvec,
    )
    .await
}

pub(crate) async fn qwen_layer_full_attention_sequence_impl(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[Vec<f32>],
    cache: Option<&mut LayerKvCache>,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    let dims = QwenFullAttentionDims::from_spec(spec);
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
    let k_proj = matvec
        .bf16_matvecs_row_major_f32_flat_inputs(
            store,
            &k_proj_tensor,
            flat_hidden_states.values(),
            flat_hidden_states.input_count(),
        )
        .await?;
    let v_proj = matvec
        .bf16_matvecs_row_major_f32_flat_inputs(
            store,
            &v_proj_tensor,
            flat_hidden_states.values(),
            flat_hidden_states.input_count(),
        )
        .await?;
    let q_norm_weight =
        store.bf16_tensor_f32_cached_arc(&spec.self_attn_tensor(layer_idx, "q_norm.weight"))?;
    let k_norm_weight =
        store.bf16_tensor_f32_cached_arc(&spec.self_attn_tensor(layer_idx, "k_norm.weight"))?;
    let o_proj_tensor = spec.self_attn_tensor(layer_idx, "o_proj.weight");
    let parts = QwenFullAttentionSequenceParts {
        q_proj: NativeF32Rows::from_batched_matvec(&q_proj).map_err(|err| {
            TensorLoadError::integrity(format!("Qwen q projection rows failed: {err}"))
        })?,
        k_proj: NativeF32Rows::from_batched_matvec(&k_proj).map_err(|err| {
            TensorLoadError::integrity(format!("Qwen k projection rows failed: {err}"))
        })?,
        v_proj: NativeF32Rows::from_batched_matvec(&v_proj).map_err(|err| {
            TensorLoadError::integrity(format!("Qwen v projection rows failed: {err}"))
        })?,
        q_norm_weight: q_norm_weight.as_ref(),
        k_norm_weight: k_norm_weight.as_ref(),
        o_proj_weight: NativeOutputProjection::Bf16Tensor {
            store,
            tensor: &o_proj_tensor,
        },
    };
    let config = QwenFullAttentionSequenceConfig {
        rms_norm_eps: spec.rms_norm_eps,
        rope_theta: spec.rope_theta,
        partial_rotary_factor: spec.partial_rotary_factor,
        q_projection_gate: !spec.is_qwen3_dense(),
        one_centered_rms_norm: !spec.is_qwen3_dense(),
    };
    let result = if let Some(cache) = cache {
        qwen_full_attention_sequence_with_cache_from_parts(&dims, &parts, config, cache, matvec)
            .await
    } else {
        qwen_full_attention_sequence_from_parts(&dims, &parts, config, matvec).await
    };
    result.map_err(|err| {
        TensorLoadError::integrity(format!(
            "Qwen layer{layer_idx} full attention sequence failed: {err}"
        ))
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn qwen_layer_full_attention_step_with_cache(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[f32],
    cache: &mut LayerKvCache,
    matvec: &impl NativeMatvecBackend,
    scratch: &mut InferenceScratchpad,
    output: &mut [f32],
) -> Result<(), TensorLoadError> {
    let dims = QwenFullAttentionDims::from_spec(spec);
    let attention_dim = dims
        .attention_dim()
        .map_err(|err| TensorLoadError::integrity(format!("Qwen attention shape failed: {err}")))?;
    let key_value_dim = dims
        .key_value_dim()
        .map_err(|err| TensorLoadError::integrity(format!("Qwen KV shape failed: {err}")))?;

    let q_proj = InferenceScratchpad::get_mut(&mut scratch.buf0, attention_dim);
    matvec
        .bf16_matvec_row_major_f32_in_place(
            store,
            &spec.self_attn_tensor(layer_idx, "q_proj.weight"),
            hidden_states,
            q_proj,
        )
        .await?;
    let k_proj = InferenceScratchpad::get_mut(&mut scratch.buf1, key_value_dim);
    matvec
        .bf16_matvec_row_major_f32_in_place(
            store,
            &spec.self_attn_tensor(layer_idx, "k_proj.weight"),
            hidden_states,
            k_proj,
        )
        .await?;
    let v_proj = InferenceScratchpad::get_mut(&mut scratch.buf2, key_value_dim);
    matvec
        .bf16_matvec_row_major_f32_in_place(
            store,
            &spec.self_attn_tensor(layer_idx, "v_proj.weight"),
            hidden_states,
            v_proj,
        )
        .await?;
    let q_norm_weight =
        store.bf16_tensor_f32_cached_arc(&spec.self_attn_tensor(layer_idx, "q_norm.weight"))?;
    let k_norm_weight =
        store.bf16_tensor_f32_cached_arc(&spec.self_attn_tensor(layer_idx, "k_norm.weight"))?;
    let o_proj_tensor = spec.self_attn_tensor(layer_idx, "o_proj.weight");
    let config = QwenFullAttentionSequenceConfig {
        rms_norm_eps: spec.rms_norm_eps,
        rope_theta: spec.rope_theta,
        partial_rotary_factor: spec.partial_rotary_factor,
        q_projection_gate: !spec.is_qwen3_dense(),
        one_centered_rms_norm: !spec.is_qwen3_dense(),
    };
    let step_output = qwen_full_attention_step_with_cache_from_parts(
        &dims,
        &QwenFullAttentionStepParts {
            q_proj,
            k_proj,
            v_proj,
            q_norm_weight: q_norm_weight.as_ref(),
            k_norm_weight: k_norm_weight.as_ref(),
            o_proj_weight: NativeOutputProjection::Bf16Tensor {
                store,
                tensor: &o_proj_tensor,
            },
        },
        config,
        cache,
        matvec,
    )
    .await
    .map_err(|err| {
        TensorLoadError::integrity(format!(
            "Qwen layer{layer_idx} full attention failed: {err}"
        ))
    })?;
    let hidden_size = spec.hidden_size as usize;
    output[..hidden_size].copy_from_slice(&step_output[..hidden_size]);
    Ok(())
}
