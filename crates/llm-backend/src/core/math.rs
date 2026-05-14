use thiserror::Error;

#[derive(Debug, Error)]
pub enum MathError {
    #[error("invalid math shape: {0}")]
    InvalidShape(String),
}

#[derive(Debug, Clone, Default)]
pub struct InferenceScratchpad {
    pub buf0: Vec<f32>,
    pub buf1: Vec<f32>,
    pub buf2: Vec<f32>,
    pub buf3: Vec<f32>,
    pub buf4: Vec<f32>,
}

impl InferenceScratchpad {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_mut(buf: &mut Vec<f32>, len: usize) -> &mut [f32] {
        if buf.len() < len {
            buf.resize(len, 0.0);
        }
        &mut buf[..len]
    }
}

pub(crate) fn rms_norm_f32(input: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>, MathError> {
    let mut out = vec![0.0; input.len()];
    rms_norm_f32_in_place(input, weight, eps, &mut out)?;
    Ok(out)
}

pub(crate) fn rms_norm_f32_in_place(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    output: &mut [f32],
) -> Result<(), MathError> {
    rms_norm_with_weight_offset_f32_in_place(input, weight, eps, 0.0, output)
}

pub(crate) fn rms_norm_one_centered_f32(
    input: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<Vec<f32>, MathError> {
    let mut out = vec![0.0; input.len()];
    rms_norm_one_centered_f32_in_place(input, weight, eps, &mut out)?;
    Ok(out)
}

pub(crate) fn rms_norm_one_centered_f32_in_place(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    output: &mut [f32],
) -> Result<(), MathError> {
    rms_norm_with_weight_offset_f32_in_place(input, weight, eps, 1.0, output)
}

fn rms_norm_with_weight_offset_f32_in_place(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    weight_offset: f32,
    output: &mut [f32],
) -> Result<(), MathError> {
    if input.len() != weight.len() {
        return Err(MathError::InvalidShape(
            "input and weight must have the same length".to_owned(),
        ));
    }
    if output.len() < input.len() {
        return Err(MathError::InvalidShape(
            "output buffer is too small".to_owned(),
        ));
    }
    if input.is_empty() {
        return Ok(());
    }
    if eps < 0.0 {
        return Err(MathError::InvalidShape(
            "rms norm epsilon must be non-negative".to_owned(),
        ));
    }
    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let scale = rms_norm_scale_f32(mean_square, eps);
    for ((out, val), w) in output.iter_mut().zip(input).zip(weight) {
        *out = val * scale * (weight_offset + w);
    }
    Ok(())
}

pub(crate) fn rms_norm_scale_f32(mean_square: f32, eps: f32) -> f32 {
    let variance = mean_square + eps;
    if variance == 0.0 {
        0.0
    } else {
        variance.sqrt().recip()
    }
}

pub(crate) fn matvec_row_major_f32(
    input: &[f32],
    weights: &[f32],
    rows: usize,
    columns: usize,
) -> Result<Vec<f32>, MathError> {
    let mut out = vec![0.0; rows];
    matvec_row_major_f32_in_place(input, weights, rows, columns, &mut out)?;
    Ok(out)
}

pub(crate) fn matvec_row_major_f32_in_place(
    input: &[f32],
    weights: &[f32],
    rows: usize,
    columns: usize,
    output: &mut [f32],
) -> Result<(), MathError> {
    if input.len() != columns {
        return Err(MathError::InvalidShape(format!(
            "input length {} does not match matvec columns {columns}",
            input.len()
        )));
    }
    let expected_weights = rows
        .checked_mul(columns)
        .ok_or_else(|| MathError::InvalidShape("matvec shape overflows usize".to_owned()))?;
    if weights.len() != expected_weights {
        return Err(MathError::InvalidShape(format!(
            "weight length {} does not match rows {rows} * columns {columns}",
            weights.len()
        )));
    }
    if output.len() < rows {
        return Err(MathError::InvalidShape(
            "output buffer too small".to_owned(),
        ));
    }
    for (out, row) in output.iter_mut().zip(weights.chunks_exact(columns)) {
        *out = row
            .iter()
            .zip(input)
            .map(|(weight, value)| weight * value)
            .sum();
    }
    Ok(())
}

pub(crate) fn matvecs_row_major_f32(
    inputs: &[Vec<f32>],
    weights: &[f32],
    rows: usize,
    columns: usize,
) -> Result<Vec<Vec<f32>>, MathError> {
    inputs
        .iter()
        .map(|input| matvec_row_major_f32(input, weights, rows, columns))
        .collect()
}

pub(crate) fn swiglu_mlp_f32(
    input: &[f32],
    gate_weight: &[f32],
    up_weight: &[f32],
    down_weight: &[f32],
    intermediate_size: usize,
) -> Result<Vec<f32>, MathError> {
    let mut scratch = InferenceScratchpad::new();
    let mut out = vec![0.0; down_weight.len() / intermediate_size];
    swiglu_mlp_f32_in_place(
        input,
        gate_weight,
        up_weight,
        down_weight,
        intermediate_size,
        &mut scratch,
        &mut out,
    )?;
    Ok(out)
}

pub(crate) fn swiglu_mlp_f32_in_place(
    input: &[f32],
    gate_weight: &[f32],
    up_weight: &[f32],
    down_weight: &[f32],
    intermediate_size: usize,
    scratch: &mut InferenceScratchpad,
    output: &mut [f32],
) -> Result<(), MathError> {
    let gate = InferenceScratchpad::get_mut(&mut scratch.buf0, intermediate_size);
    matvec_row_major_f32_in_place(input, gate_weight, intermediate_size, input.len(), gate)?;
    let up = InferenceScratchpad::get_mut(&mut scratch.buf1, intermediate_size);
    matvec_row_major_f32_in_place(input, up_weight, intermediate_size, input.len(), up)?;
    let activated = InferenceScratchpad::get_mut(&mut scratch.buf2, intermediate_size);
    for (a, (g, u)) in activated.iter_mut().zip(gate.iter().zip(up.iter())) {
        *a = silu_f32(*g) * *u;
    }
    if !down_weight.len().is_multiple_of(intermediate_size) {
        return Err(MathError::InvalidShape(format!(
            "down projection length {} is not divisible by intermediate size {intermediate_size}",
            down_weight.len()
        )));
    }
    let rows = down_weight.len() / intermediate_size;
    matvec_row_major_f32_in_place(activated, down_weight, rows, intermediate_size, output)
}

pub(crate) fn silu_f32(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

pub(crate) fn linear_attention_conv1d_silu_f32_in_place(
    window: &[f32],
    weights: &[f32],
    conv_dim: usize,
    kernel_size: usize,
    output: &mut [f32],
) -> Result<(), MathError> {
    if kernel_size == 0 {
        return Err(MathError::InvalidShape(
            "linear attention conv kernel size must be non-zero".to_owned(),
        ));
    }
    let expected_len = conv_dim.checked_mul(kernel_size).ok_or_else(|| {
        MathError::InvalidShape("linear attention conv shape overflows usize".to_owned())
    })?;
    require_len("conv window", window.len(), expected_len)?;
    require_len("conv weight", weights.len(), expected_len)?;
    if output.len() < conv_dim {
        return Err(MathError::InvalidShape(
            "conv output buffer too small".to_owned(),
        ));
    }
    for channel in 0..conv_dim {
        let mut mixed = 0.0;
        for kernel_idx in 0..kernel_size {
            mixed += window[kernel_idx * conv_dim + channel]
                * weights[channel * kernel_size + kernel_idx];
        }
        output[channel] = silu_f32(mixed);
    }
    Ok(())
}

pub(crate) fn weighted_sum_f32_in_place(
    values: &[f32],
    weights: &[f32],
    vector_len: usize,
    output: &mut [f32],
) -> Result<(), MathError> {
    let row_count = weights.len();
    let expected_len = row_count
        .checked_mul(vector_len)
        .ok_or_else(|| MathError::InvalidShape("weighted sum shape overflows usize".to_owned()))?;
    require_len("weighted sum values", values.len(), expected_len)?;
    if vector_len == 0 {
        return Ok(());
    }
    if output.len() < vector_len {
        return Err(MathError::InvalidShape(
            "weighted sum output buffer too small".to_owned(),
        ));
    }
    output[..vector_len].fill(0.0);
    for (row_idx, weight) in weights.iter().enumerate() {
        let row_start = row_idx * vector_len;
        for offset in 0..vector_len {
            output[offset] += values[row_start + offset] * weight;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn linear_attention_recurrent_update_f32_in_place(
    state: &[f32],
    key: &[f32],
    value: &[f32],
    memory: &[f32],
    beta: f32,
    decay: f32,
    key_head_dim: usize,
    value_head_dim: usize,
    output: &mut [f32],
) -> Result<(), MathError> {
    if key_head_dim == 0 || value_head_dim == 0 {
        return Err(MathError::InvalidShape(
            "linear attention recurrent update dimensions must be non-zero".to_owned(),
        ));
    }
    let element_count = key_head_dim.checked_mul(value_head_dim).ok_or_else(|| {
        MathError::InvalidShape(
            "linear attention recurrent update shape overflows usize".to_owned(),
        )
    })?;
    require_len(
        "linear attention recurrent state",
        state.len(),
        element_count,
    )?;
    require_len("linear attention recurrent key", key.len(), key_head_dim)?;
    require_len(
        "linear attention recurrent value",
        value.len(),
        value_head_dim,
    )?;
    require_len(
        "linear attention recurrent memory",
        memory.len(),
        value_head_dim,
    )?;
    if output.len() < element_count {
        return Err(MathError::InvalidShape(
            "linear attention recurrent output buffer too small".to_owned(),
        ));
    }
    for (key_offset, key_value) in key.iter().enumerate().take(key_head_dim) {
        let row_start = key_offset * value_head_dim;
        for value_offset in 0..value_head_dim {
            let delta = (value[value_offset] - memory[value_offset]) * beta;
            output[row_start + value_offset] =
                state[row_start + value_offset] * decay + key_value * delta;
        }
    }
    Ok(())
}

pub(crate) fn select_head_rows_f32_in_place(
    values: &[f32],
    row_count: usize,
    row_len: usize,
    head_start: usize,
    head_len: usize,
    output: &mut [f32],
) -> Result<(), MathError> {
    let used_len = row_count.checked_mul(row_len).ok_or_else(|| {
        MathError::InvalidShape("head row selection shape overflows usize".to_owned())
    })?;
    if values.len() < used_len {
        return Err(MathError::InvalidShape(format!(
            "head row selection value length {} is shorter than row_count {row_count} * row_len {row_len}",
            values.len()
        )));
    }
    let head_end = head_start.checked_add(head_len).ok_or_else(|| {
        MathError::InvalidShape("head row selection range overflows usize".to_owned())
    })?;
    if head_end > row_len {
        return Err(MathError::InvalidShape(format!(
            "head row selection range {head_start}..{head_end} exceeds row length {row_len}"
        )));
    }
    let output_len = row_count.checked_mul(head_len).ok_or_else(|| {
        MathError::InvalidShape("head row selection output shape overflows usize".to_owned())
    })?;
    if output.len() < output_len {
        return Err(MathError::InvalidShape(
            "head row selection output buffer too small".to_owned(),
        ));
    }
    for row_idx in 0..row_count {
        let row_start = row_idx * row_len + head_start;
        let output_start = row_idx * head_len;
        output[output_start..output_start + head_len]
            .copy_from_slice(&values[row_start..row_start + head_len]);
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopKWeight {
    pub index: usize,
    pub weight: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopKLogit {
    pub index: usize,
    pub logit: f32,
}

pub(crate) fn softmax_top_k_f32(
    logits: &[f32],
    top_k: usize,
) -> Result<Vec<TopKWeight>, MathError> {
    if top_k == 0 || top_k > logits.len() {
        return Err(MathError::InvalidShape(format!(
            "top_k {top_k} must be in 1..={}",
            logits.len()
        )));
    }
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(MathError::InvalidShape(
            "router logits must be finite".to_owned(),
        ));
    }
    let mut selected = logits.iter().copied().enumerate().collect::<Vec<_>>();
    selected.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    selected.truncate(top_k);
    let max = selected
        .iter()
        .map(|(_, value)| *value)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut exp_values = selected
        .iter()
        .map(|(_, value)| (*value - max).exp())
        .collect::<Vec<_>>();
    let sum = exp_values.iter().sum::<f32>();
    if sum == 0.0 || !sum.is_finite() {
        return Err(MathError::InvalidShape(
            "router softmax denominator is invalid".to_owned(),
        ));
    }
    Ok(selected
        .iter()
        .zip(exp_values.iter_mut())
        .map(|((index, _), value)| TopKWeight {
            index: *index,
            weight: *value / sum,
        })
        .collect())
}

pub(crate) fn require_len(name: &str, actual: usize, expected: usize) -> Result<(), MathError> {
    if actual == expected {
        Ok(())
    } else {
        Err(MathError::InvalidShape(format!(
            "{name} length {actual} does not match expected {expected}"
        )))
    }
}

pub(crate) fn sigmoid_f32(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

pub(crate) fn softmax_f32_in_place(scores: &[f32], output: &mut [f32]) -> Result<(), MathError> {
    if output.len() < scores.len() {
        return Err(MathError::InvalidShape(
            "softmax output buffer too small".to_owned(),
        ));
    }
    if scores.is_empty() {
        return Ok(());
    }
    if scores.iter().any(|value| !value.is_finite()) {
        return Err(MathError::InvalidShape(
            "softmax scores must be finite".to_owned(),
        ));
    }
    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for (out, score) in output.iter_mut().zip(scores) {
        *out = (*score - max_score).exp();
        sum += *out;
    }
    if sum == 0.0 || !sum.is_finite() {
        return Err(MathError::InvalidShape(
            "softmax denominator is invalid".to_owned(),
        ));
    }
    for value in &mut output[..scores.len()] {
        *value /= sum;
    }
    Ok(())
}

pub(crate) fn softplus_f32(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        (1.0 + value.exp()).ln()
    }
}

pub(crate) fn apply_rope_to_head(head: &mut [f32], position: usize, rotary_dim: usize, theta: f32) {
    if rotary_dim == 0 {
        return;
    }
    let half = rotary_dim / 2;
    for offset in 0..half {
        let inv_freq = theta.powf(-((2 * offset) as f32) / rotary_dim as f32);
        let angle = position as f32 * inv_freq;
        let (sin, cos) = angle.sin_cos();
        let first = head[offset];
        let second = head[offset + half];
        head[offset] = first * cos - second * sin;
        head[offset + half] = second * cos + first * sin;
    }
}

pub(crate) fn push_top_logit(top: &mut Vec<TopKLogit>, candidate: TopKLogit, top_k: usize) {
    top.push(candidate);
    top.sort_by(|left, right| {
        right
            .logit
            .total_cmp(&left.logit)
            .then_with(|| left.index.cmp(&right.index))
    });
    top.truncate(top_k);
}

pub(crate) fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}
