#![allow(dead_code)]

use super::super::super::math::{
    InferenceScratchpad, MathError, require_len, sigmoid_f32, silu_f32, softplus_f32,
};
use super::super::super::native_attention::{
    NativeOutputProjection, native_output_projection, native_output_projection_in_place_unchecked,
    native_output_projection_unchecked, require_native_output_projection_shape,
};
use super::super::super::{
    CpuNativeMatvecBackend, LinearAttentionCache, NativeMatvecBackend, SafeTensorShardStore,
    TensorLoadError,
};
use super::super::matvec::{l2_normalize_f32, rms_norm_f32};
use super::tensor_names::qwen_linear_attn_tensor;
use llm_models::QwenModelSpec;

#[derive(Debug, Clone, PartialEq)]
pub struct QwenLinearAttentionProjectionProbe {
    pub qkv: Vec<f32>,
    pub z: Vec<f32>,
    pub b: Vec<f32>,
    pub a: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct QwenLinearAttentionProjectionSequence {
    pub qkv: Vec<Vec<f32>>,
    pub z: Vec<Vec<f32>>,
    pub b: Vec<Vec<f32>>,
    pub a: Vec<Vec<f32>>,
}

pub(crate) struct QwenLinearAttentionFirstTokenParts<'a> {
    pub qkv: &'a [f32],
    pub z: &'a [f32],
    pub b: &'a [f32],
    pub conv1d_weight: &'a [f32],
    pub norm_weight: &'a [f32],
    pub out_proj_weight: NativeOutputProjection<'a>,
}

pub(crate) struct QwenLinearAttentionSequenceParts<'a> {
    pub qkv: &'a [Vec<f32>],
    pub z: &'a [Vec<f32>],
    pub b: &'a [Vec<f32>],
    pub a: &'a [Vec<f32>],
    pub dt_bias: &'a [f32],
    pub a_log: &'a [f32],
    pub conv1d_weight: &'a [f32],
    pub norm_weight: &'a [f32],
    pub out_proj_weight: NativeOutputProjection<'a>,
}

pub(crate) struct QwenLinearAttentionStepParts<'a> {
    pub qkv: &'a [f32],
    pub z: &'a [f32],
    pub b: &'a [f32],
    pub a: &'a [f32],
    pub dt_bias: &'a [f32],
    pub a_log: &'a [f32],
    pub conv1d_weight: &'a [f32],
    pub norm_weight: &'a [f32],
    pub out_proj_weight: NativeOutputProjection<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct QwenLinearAttentionDims {
    pub hidden_size: usize,
    pub num_key_heads: usize,
    pub num_value_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub conv_kernel_size: usize,
    pub rms_norm_eps: f32,
}

impl QwenLinearAttentionDims {
    pub fn from_spec(spec: &QwenModelSpec) -> Self {
        Self {
            hidden_size: spec.hidden_size as usize,
            num_key_heads: spec.linear_num_key_heads as usize,
            num_value_heads: spec.linear_num_value_heads as usize,
            key_head_dim: spec.linear_key_head_dim as usize,
            value_head_dim: spec.linear_value_head_dim as usize,
            conv_kernel_size: spec.linear_conv_kernel_dim as usize,
            rms_norm_eps: spec.rms_norm_eps,
        }
    }

    fn key_dim(&self) -> Result<usize, MathError> {
        self.num_key_heads
            .checked_mul(self.key_head_dim)
            .ok_or_else(|| MathError::InvalidShape("Qwen key dimension overflow".to_owned()))
    }

    fn value_dim(&self) -> Result<usize, MathError> {
        self.num_value_heads
            .checked_mul(self.value_head_dim)
            .ok_or_else(|| MathError::InvalidShape("Qwen value dimension overflow".to_owned()))
    }

    pub(crate) fn conv_dim(&self) -> Result<usize, MathError> {
        let key_dim = self.key_dim()?;
        let value_dim = self.value_dim()?;
        key_dim
            .checked_mul(2)
            .and_then(|key_parts| key_parts.checked_add(value_dim))
            .ok_or_else(|| MathError::InvalidShape("Qwen conv dimension overflow".to_owned()))
    }
}

pub(crate) async fn qwen_linear_attention_first_token_from_parts(
    dims: &QwenLinearAttentionDims,
    parts: &QwenLinearAttentionFirstTokenParts<'_>,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, MathError> {
    let qkv = parts.qkv;
    let z = parts.z;
    let b = parts.b;
    let conv1d_weight = parts.conv1d_weight;
    let norm_weight = parts.norm_weight;
    let out_proj_weight = parts.out_proj_weight;
    if dims.num_key_heads == 0
        || dims.num_value_heads == 0
        || dims.key_head_dim == 0
        || dims.value_head_dim == 0
        || dims.conv_kernel_size == 0
    {
        return Err(MathError::InvalidShape(
            "Qwen linear attention dimensions must be non-zero".to_owned(),
        ));
    }
    if !dims.num_value_heads.is_multiple_of(dims.num_key_heads) {
        return Err(MathError::InvalidShape(
            "Qwen value heads must be divisible by key heads".to_owned(),
        ));
    }
    let key_dim = dims.key_dim()?;
    let value_dim = dims.value_dim()?;
    let conv_dim = dims.conv_dim()?;
    require_len("qkv projection", qkv.len(), conv_dim)?;
    require_len("z projection", z.len(), value_dim)?;
    require_len("b projection", b.len(), dims.num_value_heads)?;
    require_len("norm weight", norm_weight.len(), dims.value_head_dim)?;
    require_len(
        "conv1d weight",
        conv1d_weight.len(),
        conv_dim
            .checked_mul(dims.conv_kernel_size)
            .ok_or_else(|| MathError::InvalidShape("conv1d weight shape overflow".to_owned()))?,
    )?;
    let mut conv_window = vec![0.0; conv_dim * dims.conv_kernel_size];
    let current_start = (dims.conv_kernel_size - 1) * conv_dim;
    conv_window[current_start..current_start + conv_dim].copy_from_slice(qkv);
    let mixed_qkv = matvec
        .linear_attention_conv1d_silu_f32(
            &conv_window,
            conv1d_weight,
            conv_dim,
            dims.conv_kernel_size,
        )
        .await?;

    let query = &mixed_qkv[..key_dim];
    let key = &mixed_qkv[key_dim..key_dim * 2];
    let value = &mixed_qkv[key_dim * 2..];
    let repeat = dims.num_value_heads / dims.num_key_heads;
    let scale = (dims.key_head_dim as f32).sqrt().recip();
    let mut gated = vec![0.0; value_dim];

    for (value_head, beta_logit) in b.iter().enumerate().take(dims.num_value_heads) {
        let key_head = value_head / repeat;
        let key_start = key_head * dims.key_head_dim;
        let value_start = value_head * dims.value_head_dim;
        let query_head = l2_normalize_f32(
            &query[key_start..key_start + dims.key_head_dim],
            1e-6,
            matvec,
        )
        .await?;
        let key_head_values =
            l2_normalize_f32(&key[key_start..key_start + dims.key_head_dim], 1e-6, matvec).await?;
        let attention_score = query_head
            .iter()
            .zip(&key_head_values)
            .map(|(query, key)| query * key)
            .sum::<f32>()
            * scale;
        let beta = sigmoid_f32(*beta_logit);
        let mut core_head = Vec::with_capacity(dims.value_head_dim);
        for offset in 0..dims.value_head_dim {
            core_head.push(attention_score * value[value_start + offset] * beta);
        }
        let normalized = rms_norm_f32(&core_head, norm_weight, dims.rms_norm_eps, matvec).await?;
        for offset in 0..dims.value_head_dim {
            gated[value_start + offset] = normalized[offset] * silu_f32(z[value_start + offset]);
        }
    }

    native_output_projection(matvec, out_proj_weight, &gated, dims.hidden_size, value_dim).await
}

pub(crate) async fn qwen_linear_attention_sequence_from_parts(
    dims: &QwenLinearAttentionDims,
    parts: &QwenLinearAttentionSequenceParts<'_>,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<Vec<f32>>, MathError> {
    qwen_linear_attention_sequence_from_parts_impl(dims, parts, None, matvec).await
}

pub(crate) async fn qwen_linear_attention_sequence_with_cache_from_parts(
    dims: &QwenLinearAttentionDims,
    parts: &QwenLinearAttentionSequenceParts<'_>,
    cache: &mut LinearAttentionCache,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<Vec<f32>>, MathError> {
    qwen_linear_attention_sequence_from_parts_impl(dims, parts, Some(cache), matvec).await
}

pub(crate) async fn qwen_linear_attention_step_with_cache_from_parts(
    dims: &QwenLinearAttentionDims,
    parts: &QwenLinearAttentionStepParts<'_>,
    cache: &mut LinearAttentionCache,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, MathError> {
    let mut scratch = InferenceScratchpad::default();
    let mut output = vec![0.0; dims.hidden_size];
    qwen_linear_attention_step_with_cache_from_parts_in_place(
        dims,
        parts,
        cache,
        matvec,
        &mut scratch,
        &mut output,
    )
    .await?;
    Ok(output)
}

pub(crate) async fn qwen_linear_attention_step_with_cache_from_parts_in_place(
    dims: &QwenLinearAttentionDims,
    parts: &QwenLinearAttentionStepParts<'_>,
    cache: &mut LinearAttentionCache,
    matvec: &impl NativeMatvecBackend,
    scratch: &mut InferenceScratchpad,
    output: &mut [f32],
) -> Result<(), MathError> {
    let key_dim = dims.key_dim()?;
    let value_dim = dims.value_dim()?;
    let conv_dim = dims.conv_dim()?;
    require_native_output_projection_shape(parts.out_proj_weight, dims.hidden_size, value_dim)?;
    let repeat = dims.num_value_heads / dims.num_key_heads;
    let scale = 1.0 / (dims.key_head_dim as f32).sqrt();

    let conv_output = InferenceScratchpad::get_mut(&mut scratch.buf0, conv_dim);
    cache
        .push_conv_input(parts.qkv)
        .map_err(|err| MathError::InvalidShape(format!("KV cache append failed: {err}")))?;
    matvec
        .linear_attention_conv1d_silu_f32_in_place(
            cache.conv_window(),
            parts.conv1d_weight,
            conv_dim,
            dims.conv_kernel_size,
            conv_output,
        )
        .await?;

    let query = &conv_output[..key_dim];
    let key = &conv_output[key_dim..key_dim * 2];
    let value = &conv_output[key_dim * 2..];

    let zero_memory = vec![0.0; dims.value_head_dim];
    let mut gated = vec![0.0; value_dim];
    let mut value_major_state = Vec::new();
    let mut query_scaled = vec![0.0; dims.key_head_dim];

    for value_head in 0..dims.num_value_heads {
        let key_head = value_head / repeat;
        let key_start = key_head * dims.key_head_dim;
        let value_start = value_head * dims.value_head_dim;
        let query_head = l2_normalize_f32(
            &query[key_start..key_start + dims.key_head_dim],
            1e-6,
            matvec,
        )
        .await?;
        let key_head_values =
            l2_normalize_f32(&key[key_start..key_start + dims.key_head_dim], 1e-6, matvec).await?;
        for (o, v) in query_scaled.iter_mut().zip(&query_head) {
            *o = v * scale;
        }
        let beta = sigmoid_f32(parts.b[value_head]);
        let decay = (-parts.a_log[value_head].exp()
            * softplus_f32(parts.a[value_head] + parts.dt_bias[value_head]))
        .exp();

        let state_start = value_head * dims.key_head_dim * dims.value_head_dim;
        let _state_len = dims.key_head_dim * dims.value_head_dim;

        let decayed_state = matvec
            .linear_attention_recurrent_cache_update_f32(
                cache,
                state_start,
                &key_head_values,
                &value[value_start..value_start + dims.value_head_dim],
                &zero_memory,
                0.0,
                decay,
                dims.key_head_dim,
                dims.value_head_dim,
            )
            .await?;
        cache
            .replace_recurrent_state_range(state_start, &decayed_state)
            .map_err(|err| {
                MathError::InvalidShape(format!("linear attention cache update failed: {err}"))
            })?;

        copy_linear_attention_value_major_state_rows(
            cache.recurrent_state(),
            state_start,
            dims.key_head_dim,
            dims.value_head_dim,
            &mut value_major_state,
        )?;
        let memory = matvec
            .matvec_row_major_f32(
                &key_head_values,
                &value_major_state,
                dims.value_head_dim,
                dims.key_head_dim,
            )
            .await?;

        let updated_state = matvec
            .linear_attention_recurrent_cache_update_f32(
                cache,
                state_start,
                &key_head_values,
                &value[value_start..value_start + dims.value_head_dim],
                &memory,
                beta,
                1.0,
                dims.key_head_dim,
                dims.value_head_dim,
            )
            .await?;
        cache
            .replace_recurrent_state_range(state_start, &updated_state)
            .map_err(|err| {
                MathError::InvalidShape(format!("linear attention cache update failed: {err}"))
            })?;

        copy_linear_attention_value_major_state_rows(
            cache.recurrent_state(),
            state_start,
            dims.key_head_dim,
            dims.value_head_dim,
            &mut value_major_state,
        )?;
        let core_head = matvec
            .matvec_row_major_f32(
                &query_scaled,
                &value_major_state,
                dims.value_head_dim,
                dims.key_head_dim,
            )
            .await?;
        let normalized =
            rms_norm_f32(&core_head, parts.norm_weight, dims.rms_norm_eps, matvec).await?;
        for value_offset in 0..dims.value_head_dim {
            gated[value_start + value_offset] =
                normalized[value_offset] * silu_f32(parts.z[value_start + value_offset]);
        }
    }
    native_output_projection_in_place_unchecked(
        matvec,
        parts.out_proj_weight,
        &gated,
        dims.hidden_size,
        value_dim,
        output,
    )
    .await?;
    Ok(())
}

async fn qwen_linear_attention_sequence_from_parts_impl(
    dims: &QwenLinearAttentionDims,
    parts: &QwenLinearAttentionSequenceParts<'_>,
    mut cache: Option<&mut LinearAttentionCache>,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<Vec<f32>>, MathError> {
    let qkv = parts.qkv;
    let z = parts.z;
    let b = parts.b;
    let a = parts.a;
    if qkv.is_empty() {
        return Ok(Vec::new());
    }
    if dims.num_key_heads == 0
        || dims.num_value_heads == 0
        || dims.key_head_dim == 0
        || dims.value_head_dim == 0
        || dims.conv_kernel_size == 0
    {
        return Err(MathError::InvalidShape(
            "Qwen linear attention dimensions must be non-zero".to_owned(),
        ));
    }
    if !dims.num_value_heads.is_multiple_of(dims.num_key_heads) {
        return Err(MathError::InvalidShape(
            "Qwen value heads must be divisible by key heads".to_owned(),
        ));
    }
    let seq_len = qkv.len();
    if z.len() != seq_len || b.len() != seq_len || a.len() != seq_len {
        return Err(MathError::InvalidShape(
            "Qwen linear attention sequence inputs must have the same length".to_owned(),
        ));
    }
    let key_dim = dims.key_dim()?;
    let value_dim = dims.value_dim()?;
    let conv_dim = dims.conv_dim()?;
    if let Some(cache) = cache.as_ref() {
        require_linear_attention_cache_shape(dims, conv_dim, cache)?;
    }
    require_len("dt bias", parts.dt_bias.len(), dims.num_value_heads)?;
    require_len("A log", parts.a_log.len(), dims.num_value_heads)?;
    require_len("norm weight", parts.norm_weight.len(), dims.value_head_dim)?;
    require_native_output_projection_shape(parts.out_proj_weight, dims.hidden_size, value_dim)?;
    require_len(
        "conv1d weight",
        parts.conv1d_weight.len(),
        conv_dim
            .checked_mul(dims.conv_kernel_size)
            .ok_or_else(|| MathError::InvalidShape("conv1d weight shape overflow".to_owned()))?,
    )?;
    for token_idx in 0..seq_len {
        require_len("qkv projection", qkv[token_idx].len(), conv_dim)?;
        require_len("z projection", z[token_idx].len(), value_dim)?;
        require_len("b projection", b[token_idx].len(), dims.num_value_heads)?;
        require_len("a projection", a[token_idx].len(), dims.num_value_heads)?;
    }

    let mut mixed_tokens = vec![vec![0.0; conv_dim]; seq_len];
    for token_idx in 0..seq_len {
        if let Some(cache) = cache.as_mut() {
            cache.push_conv_input(&qkv[token_idx]).map_err(|err| {
                MathError::InvalidShape(format!("linear attention cache update failed: {err}"))
            })?;
            mixed_tokens[token_idx] = matvec
                .linear_attention_conv1d_silu_f32(
                    cache.conv_window(),
                    parts.conv1d_weight,
                    conv_dim,
                    dims.conv_kernel_size,
                )
                .await?;
        } else {
            let mut conv_window = vec![0.0; conv_dim * dims.conv_kernel_size];
            for kernel_idx in 0..dims.conv_kernel_size {
                let lookback = dims.conv_kernel_size - 1 - kernel_idx;
                if token_idx >= lookback {
                    let window_start = kernel_idx * conv_dim;
                    conv_window[window_start..window_start + conv_dim]
                        .copy_from_slice(&qkv[token_idx - lookback]);
                }
            }
            mixed_tokens[token_idx] = matvec
                .linear_attention_conv1d_silu_f32(
                    &conv_window,
                    parts.conv1d_weight,
                    conv_dim,
                    dims.conv_kernel_size,
                )
                .await?;
        }
    }

    let repeat = dims.num_value_heads / dims.num_key_heads;
    let scale = (dims.key_head_dim as f32).sqrt().recip();
    let mut recurrent_state = cache
        .as_ref()
        .map(|cache| cache.recurrent_state().to_vec())
        .unwrap_or_else(|| {
            vec![0.0; dims.num_value_heads * dims.key_head_dim * dims.value_head_dim]
        });
    let value_major_len = dims
        .value_head_dim
        .checked_mul(dims.key_head_dim)
        .ok_or_else(|| MathError::InvalidShape("value-major state shape overflow".to_owned()))?;
    let mut _l2_weight_scratch: Vec<f32> = Vec::with_capacity(dims.key_head_dim);
    let zero_memory = vec![0.0; dims.value_head_dim];
    let mut value_major_state = Vec::with_capacity(value_major_len);
    let mut query_scaled = vec![0.0; dims.key_head_dim];
    let mut outputs = Vec::with_capacity(seq_len);

    for token_idx in 0..seq_len {
        let mixed_qkv = &mixed_tokens[token_idx];
        let query = &mixed_qkv[..key_dim];
        let key = &mixed_qkv[key_dim..key_dim * 2];
        let value = &mixed_qkv[key_dim * 2..];
        let mut gated = vec![0.0; value_dim];

        for value_head in 0..dims.num_value_heads {
            let key_head = value_head / repeat;
            let key_start = key_head * dims.key_head_dim;
            let value_start = value_head * dims.value_head_dim;
            let query_head = l2_normalize_f32(
                &query[key_start..key_start + dims.key_head_dim],
                1e-6,
                matvec,
            )
            .await?;
            let key_head_values =
                l2_normalize_f32(&key[key_start..key_start + dims.key_head_dim], 1e-6, matvec)
                    .await?;
            for (output, value) in query_scaled.iter_mut().zip(&query_head) {
                *output = value * scale;
            }
            let beta = sigmoid_f32(b[token_idx][value_head]);
            let decay = (-parts.a_log[value_head].exp()
                * softplus_f32(a[token_idx][value_head] + parts.dt_bias[value_head]))
            .exp();

            let state_start = value_head * dims.key_head_dim * dims.value_head_dim;
            let state_end = state_start + dims.key_head_dim * dims.value_head_dim;
            let decayed_state = if let Some(cache) = cache.as_ref() {
                matvec
                    .linear_attention_recurrent_cache_update_f32(
                        cache,
                        state_start,
                        &key_head_values,
                        &value[value_start..value_start + dims.value_head_dim],
                        &zero_memory,
                        0.0,
                        decay,
                        dims.key_head_dim,
                        dims.value_head_dim,
                    )
                    .await?
            } else {
                matvec
                    .linear_attention_recurrent_update_f32(
                        &recurrent_state[state_start..state_end],
                        &key_head_values,
                        &value[value_start..value_start + dims.value_head_dim],
                        &zero_memory,
                        0.0,
                        decay,
                        dims.key_head_dim,
                        dims.value_head_dim,
                    )
                    .await?
            };
            recurrent_state[state_start..state_end].copy_from_slice(&decayed_state);
            if let Some(cache) = cache.as_mut() {
                cache
                    .replace_recurrent_state_range(state_start, &decayed_state)
                    .map_err(|err| {
                        MathError::InvalidShape(format!(
                            "linear attention cache update failed: {err}"
                        ))
                    })?;
            }

            copy_linear_attention_value_major_state_rows(
                &recurrent_state,
                state_start,
                dims.key_head_dim,
                dims.value_head_dim,
                &mut value_major_state,
            )?;
            let memory = matvec
                .matvec_row_major_f32(
                    &key_head_values,
                    &value_major_state,
                    dims.value_head_dim,
                    dims.key_head_dim,
                )
                .await?;

            let updated_state = if let Some(cache) = cache.as_ref() {
                matvec
                    .linear_attention_recurrent_cache_update_f32(
                        cache,
                        state_start,
                        &key_head_values,
                        &value[value_start..value_start + dims.value_head_dim],
                        &memory,
                        beta,
                        1.0,
                        dims.key_head_dim,
                        dims.value_head_dim,
                    )
                    .await?
            } else {
                matvec
                    .linear_attention_recurrent_update_f32(
                        &recurrent_state[state_start..state_end],
                        &key_head_values,
                        &value[value_start..value_start + dims.value_head_dim],
                        &memory,
                        beta,
                        1.0,
                        dims.key_head_dim,
                        dims.value_head_dim,
                    )
                    .await?
            };
            recurrent_state[state_start..state_end].copy_from_slice(&updated_state);
            if let Some(cache) = cache.as_mut() {
                cache
                    .replace_recurrent_state_range(state_start, &updated_state)
                    .map_err(|err| {
                        MathError::InvalidShape(format!(
                            "linear attention cache update failed: {err}"
                        ))
                    })?;
            }

            copy_linear_attention_value_major_state_rows(
                &recurrent_state,
                state_start,
                dims.key_head_dim,
                dims.value_head_dim,
                &mut value_major_state,
            )?;
            let core_head = matvec
                .matvec_row_major_f32(
                    &query_scaled,
                    &value_major_state,
                    dims.value_head_dim,
                    dims.key_head_dim,
                )
                .await?;
            let normalized =
                rms_norm_f32(&core_head, parts.norm_weight, dims.rms_norm_eps, matvec).await?;
            for value_offset in 0..dims.value_head_dim {
                gated[value_start + value_offset] =
                    normalized[value_offset] * silu_f32(z[token_idx][value_start + value_offset]);
            }
        }
        outputs.push(
            native_output_projection_unchecked(
                matvec,
                parts.out_proj_weight,
                &gated,
                dims.hidden_size,
                value_dim,
            )
            .await?,
        );
    }

    Ok(outputs)
}

fn copy_linear_attention_value_major_state_rows(
    recurrent_state: &[f32],
    state_start: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    rows: &mut Vec<f32>,
) -> Result<(), MathError> {
    let row_len = value_head_dim
        .checked_mul(key_head_dim)
        .ok_or_else(|| MathError::InvalidShape("value-major state shape overflow".to_owned()))?;
    let state_end = state_start
        .checked_add(row_len)
        .ok_or_else(|| MathError::InvalidShape("recurrent state offset overflow".to_owned()))?;
    if recurrent_state.len() < state_end {
        return Err(MathError::InvalidShape(format!(
            "recurrent state slice too short: expected at least {state_end}, got {}",
            recurrent_state.len()
        )));
    }
    rows.clear();
    rows.resize(row_len, 0.0);
    for value_offset in 0..value_head_dim {
        for key_offset in 0..key_head_dim {
            rows[value_offset * key_head_dim + key_offset] =
                recurrent_state[state_start + key_offset * value_head_dim + value_offset];
        }
    }
    Ok(())
}

fn require_linear_attention_cache_shape(
    dims: &QwenLinearAttentionDims,
    conv_dim: usize,
    cache: &LinearAttentionCache,
) -> Result<(), MathError> {
    if cache.conv_kernel_size() != dims.conv_kernel_size
        || cache.conv_dim() != conv_dim
        || cache.num_value_heads() != dims.num_value_heads
        || cache.key_head_dim() != dims.key_head_dim
        || cache.value_head_dim() != dims.value_head_dim
    {
        return Err(MathError::InvalidShape(format!(
            "Qwen linear attention cache shape does not match dims: cache conv_kernel_size={}, conv_dim={}, value_heads={}, key_head_dim={}, value_head_dim={}; dims conv_kernel_size={}, conv_dim={}, value_heads={}, key_head_dim={}, value_head_dim={}",
            cache.conv_kernel_size(),
            cache.conv_dim(),
            cache.num_value_heads(),
            cache.key_head_dim(),
            cache.value_head_dim(),
            dims.conv_kernel_size,
            conv_dim,
            dims.num_value_heads,
            dims.key_head_dim,
            dims.value_head_dim
        )));
    }
    Ok(())
}

pub async fn qwen_layer0_linear_attention_first_token(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    projections: &QwenLinearAttentionProjectionProbe,
) -> Result<Vec<f32>, TensorLoadError> {
    qwen_layer_linear_attention_first_token(store, spec, 0, projections, &CpuNativeMatvecBackend)
        .await
}

pub(crate) async fn qwen_layer_linear_attention_first_token(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    projections: &QwenLinearAttentionProjectionProbe,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    let dims = QwenLinearAttentionDims::from_spec(spec);
    let dt_bias =
        store.bf16_tensor_f32_cached_arc(&qwen_linear_attn_tensor(layer_idx, "dt_bias"))?;
    let a_log = store.bf16_tensor_f32_cached_arc(&qwen_linear_attn_tensor(layer_idx, "A_log"))?;
    let conv1d_weight =
        store.bf16_tensor_f32_cached_arc(&qwen_linear_attn_tensor(layer_idx, "conv1d.weight"))?;
    let norm_weight =
        store.bf16_tensor_f32_cached_arc(&qwen_linear_attn_tensor(layer_idx, "norm.weight"))?;
    let out_proj_tensor = qwen_linear_attn_tensor(layer_idx, "out_proj.weight");
    let qkv = vec![projections.qkv.clone()];
    let z = vec![projections.z.clone()];
    let b = vec![projections.b.clone()];
    let a = vec![projections.a.clone()];
    qwen_linear_attention_sequence_from_parts(
        &dims,
        &QwenLinearAttentionSequenceParts {
            qkv: &qkv,
            z: &z,
            b: &b,
            a: &a,
            dt_bias: dt_bias.as_ref(),
            a_log: a_log.as_ref(),
            conv1d_weight: conv1d_weight.as_ref(),
            norm_weight: norm_weight.as_ref(),
            out_proj_weight: NativeOutputProjection::Bf16Tensor {
                store,
                tensor: &out_proj_tensor,
            },
        },
        matvec,
    )
    .await
    .map(|mut outputs| outputs.remove(0))
    .map_err(|err| {
        TensorLoadError::integrity(format!("Qwen layer0 linear attention failed: {err}"))
    })
}

pub(crate) async fn qwen_layer_linear_attention_sequence(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    qwen_layer_linear_attention_sequence_impl(
        store,
        spec,
        layer_idx,
        hidden_states,
        None,
        &CpuNativeMatvecBackend,
    )
    .await
}

pub(crate) async fn qwen_layer_linear_attention_sequence_with_cache(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[Vec<f32>],
    cache: &mut LinearAttentionCache,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    qwen_layer_linear_attention_sequence_impl(
        store,
        spec,
        layer_idx,
        hidden_states,
        Some(cache),
        matvec,
    )
    .await
}

pub(crate) async fn qwen_layer_linear_attention_sequence_impl(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[Vec<f32>],
    cache: Option<&mut LinearAttentionCache>,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    let projections = QwenLinearAttentionProjectionSequence {
        qkv: matvec
            .bf16_matvecs_row_major_f32(
                store,
                &qwen_linear_attn_tensor(layer_idx, "in_proj_qkv.weight"),
                hidden_states,
            )
            .await?,
        z: matvec
            .bf16_matvecs_row_major_f32(
                store,
                &qwen_linear_attn_tensor(layer_idx, "in_proj_z.weight"),
                hidden_states,
            )
            .await?,
        b: matvec
            .bf16_matvecs_row_major_f32(
                store,
                &qwen_linear_attn_tensor(layer_idx, "in_proj_b.weight"),
                hidden_states,
            )
            .await?,
        a: matvec
            .bf16_matvecs_row_major_f32(
                store,
                &qwen_linear_attn_tensor(layer_idx, "in_proj_a.weight"),
                hidden_states,
            )
            .await?,
    };
    let dims = QwenLinearAttentionDims::from_spec(spec);
    let dt_bias =
        store.bf16_tensor_f32_cached_arc(&qwen_linear_attn_tensor(layer_idx, "dt_bias"))?;
    let a_log = store.bf16_tensor_f32_cached_arc(&qwen_linear_attn_tensor(layer_idx, "A_log"))?;
    let conv1d_weight =
        store.bf16_tensor_f32_cached_arc(&qwen_linear_attn_tensor(layer_idx, "conv1d.weight"))?;
    let norm_weight =
        store.bf16_tensor_f32_cached_arc(&qwen_linear_attn_tensor(layer_idx, "norm.weight"))?;
    let out_proj_tensor = qwen_linear_attn_tensor(layer_idx, "out_proj.weight");
    let parts = QwenLinearAttentionSequenceParts {
        qkv: &projections.qkv,
        z: &projections.z,
        b: &projections.b,
        a: &projections.a,
        dt_bias: dt_bias.as_ref(),
        a_log: a_log.as_ref(),
        conv1d_weight: conv1d_weight.as_ref(),
        norm_weight: norm_weight.as_ref(),
        out_proj_weight: NativeOutputProjection::Bf16Tensor {
            store,
            tensor: &out_proj_tensor,
        },
    };
    let result = if let Some(cache) = cache {
        qwen_linear_attention_sequence_with_cache_from_parts(&dims, &parts, cache, matvec).await
    } else {
        qwen_linear_attention_sequence_from_parts(&dims, &parts, matvec).await
    };
    result.map_err(|err| {
        TensorLoadError::integrity(format!(
            "Qwen layer{layer_idx} linear attention sequence failed: {err}"
        ))
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn qwen_layer_linear_attention_step_with_cache(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[f32],
    cache: &mut LinearAttentionCache,
    matvec: &impl NativeMatvecBackend,
    scratch: &mut InferenceScratchpad,
    output: &mut [f32],
) -> Result<(), TensorLoadError> {
    let projections =
        qwen_layer_linear_attention_projections(store, layer_idx, hidden_states, matvec).await?;
    let dims = QwenLinearAttentionDims::from_spec(spec);
    let dt_bias =
        store.bf16_tensor_f32_cached_arc(&qwen_linear_attn_tensor(layer_idx, "dt_bias"))?;
    let a_log = store.bf16_tensor_f32_cached_arc(&qwen_linear_attn_tensor(layer_idx, "A_log"))?;
    let conv1d_weight =
        store.bf16_tensor_f32_cached_arc(&qwen_linear_attn_tensor(layer_idx, "conv1d.weight"))?;
    let norm_weight =
        store.bf16_tensor_f32_cached_arc(&qwen_linear_attn_tensor(layer_idx, "norm.weight"))?;
    let out_proj_tensor = qwen_linear_attn_tensor(layer_idx, "out_proj.weight");
    qwen_linear_attention_step_with_cache_from_parts_in_place(
        &dims,
        &QwenLinearAttentionStepParts {
            qkv: &projections.qkv,
            z: &projections.z,
            b: &projections.b,
            a: &projections.a,
            dt_bias: dt_bias.as_ref(),
            a_log: a_log.as_ref(),
            conv1d_weight: conv1d_weight.as_ref(),
            norm_weight: norm_weight.as_ref(),
            out_proj_weight: NativeOutputProjection::Bf16Tensor {
                store,
                tensor: &out_proj_tensor,
            },
        },
        cache,
        matvec,
        scratch,
        output,
    )
    .await
    .map_err(|err| {
        TensorLoadError::integrity(format!(
            "Qwen layer{layer_idx} linear attention step failed: {err}"
        ))
    })
}

pub async fn qwen_layer0_linear_attention_projections(
    store: &SafeTensorShardStore,
    hidden_states: &[f32],
) -> Result<QwenLinearAttentionProjectionProbe, TensorLoadError> {
    qwen_layer_linear_attention_projections(store, 0, hidden_states, &CpuNativeMatvecBackend).await
}

pub(crate) async fn qwen_layer_linear_attention_projections(
    store: &SafeTensorShardStore,
    layer_idx: usize,
    hidden_states: &[f32],
    matvec: &impl NativeMatvecBackend,
) -> Result<QwenLinearAttentionProjectionProbe, TensorLoadError> {
    Ok(QwenLinearAttentionProjectionProbe {
        qkv: matvec
            .bf16_matvec_row_major_f32(
                store,
                &qwen_linear_attn_tensor(layer_idx, "in_proj_qkv.weight"),
                hidden_states,
            )
            .await?,
        z: matvec
            .bf16_matvec_row_major_f32(
                store,
                &qwen_linear_attn_tensor(layer_idx, "in_proj_z.weight"),
                hidden_states,
            )
            .await?,
        b: matvec
            .bf16_matvec_row_major_f32(
                store,
                &qwen_linear_attn_tensor(layer_idx, "in_proj_b.weight"),
                hidden_states,
            )
            .await?,
        a: matvec
            .bf16_matvec_row_major_f32(
                store,
                &qwen_linear_attn_tensor(layer_idx, "in_proj_a.weight"),
                hidden_states,
            )
            .await?,
    })
}

mod tests {
    #[test]
    fn linear_attention_value_major_state_rows_reuses_scratch_buffer() {
        let recurrent_state = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut scratch = Vec::with_capacity(16);

        super::copy_linear_attention_value_major_state_rows(
            &recurrent_state,
            0,
            2,
            3,
            &mut scratch,
        )
        .expect("state transpose succeeds");

        assert_eq!(scratch, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(scratch.capacity(), 16);
    }
}
