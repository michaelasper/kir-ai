use super::super::math::{
    MathError, TopKLogit, TopKWeight, linear_attention_conv1d_silu_f32,
    linear_attention_recurrent_update_f32, matvec_row_major_f32, qwen_rms_norm_f32,
    select_head_rows_f32, silu_f32, softmax_f32, softmax_top_k_f32, weighted_sum_f32,
};
use super::super::{LayerKvCache, LinearAttentionCache, SafeTensorShardStore, TensorLoadError};

pub(super) fn rms_norm_f32_with_matvec(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    matvec: &impl QwenMatvecBackend,
) -> Result<Vec<f32>, MathError> {
    if input.len() != weight.len() {
        return Err(MathError::InvalidShape(
            "input and weight must have the same length".to_owned(),
        ));
    }
    let qwen_weight = weight.iter().map(|value| value - 1.0).collect::<Vec<_>>();
    matvec.qwen_rms_norm_f32(input, &qwen_weight, eps)
}

pub(super) fn l2_normalize_f32_with_matvec(
    input: &[f32],
    eps: f32,
    matvec: &impl QwenMatvecBackend,
) -> Result<Vec<f32>, MathError> {
    let mut qwen_weight = Vec::new();
    l2_normalize_f32_with_matvec_and_weight_scratch(input, eps, matvec, &mut qwen_weight)
}

pub(super) fn l2_normalize_f32_with_matvec_and_weight_scratch(
    input: &[f32],
    eps: f32,
    matvec: &impl QwenMatvecBackend,
    qwen_weight: &mut Vec<f32>,
) -> Result<Vec<f32>, MathError> {
    if input.is_empty() {
        qwen_weight.clear();
        return Ok(Vec::new());
    }
    if eps < 0.0 {
        return Err(MathError::InvalidShape(
            "l2 norm epsilon must be non-negative".to_owned(),
        ));
    }
    let weight_scale = (input.len() as f32).sqrt().recip();
    qwen_weight.clear();
    qwen_weight.resize(input.len(), weight_scale - 1.0);
    matvec.qwen_rms_norm_f32(input, qwen_weight, eps / input.len() as f32)
}

pub fn swiglu_mlp_f32_with_matvec(
    input: &[f32],
    gate_weight: &[f32],
    up_weight: &[f32],
    down_weight: &[f32],
    intermediate_size: usize,
    matvec: &impl QwenMatvecBackend,
) -> Result<Vec<f32>, MathError> {
    let gate = matvec.matvec_row_major_f32(input, gate_weight, intermediate_size, input.len())?;
    let up = matvec.matvec_row_major_f32(input, up_weight, intermediate_size, input.len())?;
    let activated = gate
        .iter()
        .zip(up)
        .map(|(gate, up)| silu_f32(*gate) * up)
        .collect::<Vec<_>>();
    if !down_weight.len().is_multiple_of(intermediate_size) {
        return Err(MathError::InvalidShape(format!(
            "down projection length {} is not divisible by intermediate size {intermediate_size}",
            down_weight.len()
        )));
    }
    let rows = down_weight.len() / intermediate_size;
    matvec.matvec_row_major_f32(&activated, down_weight, rows, intermediate_size)
}

pub trait NativeMatvecBackend {
    fn bf16_matvec_row_major_f32(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        input: &[f32],
    ) -> Result<Vec<f32>, TensorLoadError> {
        store.bf16_matvec_row_major_f32(tensor, input)
    }

    fn bf16_matvecs_row_major_f32(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        inputs: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, TensorLoadError> {
        store.bf16_matvecs_row_major_f32(tensor, inputs)
    }

    fn bf16_matvec_rows_f32(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        input: &[f32],
        chunk_rows: usize,
    ) -> Result<Vec<f32>, TensorLoadError> {
        store.bf16_matvec_rows_f32(tensor, input, chunk_rows)
    }

    fn bf16_matvec_range_row_major_f32(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        element_offset: usize,
        rows: usize,
        columns: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, TensorLoadError> {
        if input.len() != columns {
            return Err(TensorLoadError::integrity(format!(
                "BF16 range matvec input length {} must match columns {columns}",
                input.len()
            )));
        }
        let element_count = rows
            .checked_mul(columns)
            .ok_or_else(|| TensorLoadError::integrity("BF16 range matvec shape overflow"))?;
        let weights = store.bf16_tensor_f32_range(tensor, element_offset, element_count)?;
        self.matvec_row_major_f32(input, &weights, rows, columns)
            .map_err(|err| TensorLoadError::integrity(format!("BF16 range matvec failed: {err}")))
    }

    fn bf16_matvec_top_k_rows_f32(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        input: &[f32],
        top_k: usize,
        chunk_rows: usize,
    ) -> Result<Vec<TopKLogit>, TensorLoadError> {
        store.bf16_matvec_top_k_rows_f32(tensor, input, top_k, chunk_rows)
    }

    fn matvec_row_major_f32(
        &self,
        input: &[f32],
        weights: &[f32],
        rows: usize,
        columns: usize,
    ) -> Result<Vec<f32>, MathError> {
        matvec_row_major_f32(input, weights, rows, columns)
    }

    fn qwen_rms_norm_f32(
        &self,
        input: &[f32],
        weight: &[f32],
        eps: f32,
    ) -> Result<Vec<f32>, MathError> {
        self.rms_norm_one_centered_f32(input, weight, eps)
    }

    fn rms_norm_one_centered_f32(
        &self,
        input: &[f32],
        weight: &[f32],
        eps: f32,
    ) -> Result<Vec<f32>, MathError> {
        qwen_rms_norm_f32(input, weight, eps)
    }

    fn softmax_f32(&self, scores: &[f32]) -> Result<Vec<f32>, MathError> {
        softmax_f32(scores)
    }

    fn linear_attention_conv1d_silu_f32(
        &self,
        window: &[f32],
        weights: &[f32],
        conv_dim: usize,
        kernel_size: usize,
    ) -> Result<Vec<f32>, MathError> {
        linear_attention_conv1d_silu_f32(window, weights, conv_dim, kernel_size)
    }

    fn weighted_sum_f32(
        &self,
        values: &[f32],
        weights: &[f32],
        vector_len: usize,
    ) -> Result<Vec<f32>, MathError> {
        weighted_sum_f32(values, weights, vector_len)
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_attention_recurrent_update_f32(
        &self,
        state: &[f32],
        key: &[f32],
        value: &[f32],
        memory: &[f32],
        beta: f32,
        decay: f32,
        key_head_dim: usize,
        value_head_dim: usize,
    ) -> Result<Vec<f32>, MathError> {
        linear_attention_recurrent_update_f32(
            state,
            key,
            value,
            memory,
            beta,
            decay,
            key_head_dim,
            value_head_dim,
        )
    }

    fn select_head_rows_f32(
        &self,
        values: &[f32],
        row_count: usize,
        row_len: usize,
        head_start: usize,
        head_len: usize,
    ) -> Result<Vec<f32>, MathError> {
        select_head_rows_f32(values, row_count, row_len, head_start, head_len)
    }

    fn select_kv_cache_head_rows_f32(
        &self,
        cache: &LayerKvCache,
        tensor: QwenKvCacheTensor,
        row_count: usize,
        head_start: usize,
        head_len: usize,
    ) -> Result<Vec<f32>, MathError> {
        let values = match tensor {
            QwenKvCacheTensor::Key => cache.keys(),
            QwenKvCacheTensor::Value => cache.values(),
        };
        self.select_head_rows_f32(values, row_count, cache.vector_len(), head_start, head_len)
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_attention_recurrent_cache_update_f32(
        &self,
        cache: &LinearAttentionCache,
        state_start: usize,
        key: &[f32],
        value: &[f32],
        memory: &[f32],
        beta: f32,
        decay: f32,
        key_head_dim: usize,
        value_head_dim: usize,
    ) -> Result<Vec<f32>, MathError> {
        let state_len = key_head_dim.checked_mul(value_head_dim).ok_or_else(|| {
            MathError::InvalidShape(
                "linear attention recurrent cache state shape overflows usize".to_owned(),
            )
        })?;
        let state_end = state_start.checked_add(state_len).ok_or_else(|| {
            MathError::InvalidShape(
                "linear attention recurrent cache state offset overflows usize".to_owned(),
            )
        })?;
        let recurrent_state = cache.recurrent_state();
        if state_end > recurrent_state.len() {
            return Err(MathError::InvalidShape(format!(
                "linear attention recurrent cache state range {state_start}..{state_end} exceeds state length {}",
                recurrent_state.len()
            )));
        }
        self.linear_attention_recurrent_update_f32(
            &recurrent_state[state_start..state_end],
            key,
            value,
            memory,
            beta,
            decay,
            key_head_dim,
            value_head_dim,
        )
    }

    fn softmax_top_k_f32(
        &self,
        logits: &[f32],
        top_k: usize,
    ) -> Result<Vec<TopKWeight>, MathError> {
        softmax_top_k_f32(logits, top_k)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenKvCacheTensor {
    Key,
    Value,
}

pub trait QwenMatvecBackend: NativeMatvecBackend {}

impl<T> QwenMatvecBackend for T where T: NativeMatvecBackend + ?Sized {}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CpuNativeMatvecBackend;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CpuQwenMatvecBackend;

impl NativeMatvecBackend for CpuNativeMatvecBackend {}
impl NativeMatvecBackend for CpuQwenMatvecBackend {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_normalize_f32_with_matvec_reuses_weight_scratch() {
        let mut qwen_weight = Vec::with_capacity(8);

        let normalized = l2_normalize_f32_with_matvec_and_weight_scratch(
            &[3.0, 4.0],
            1e-6,
            &CpuQwenMatvecBackend,
            &mut qwen_weight,
        )
        .expect("l2 normalize succeeds");

        assert!((normalized[0] - 0.6).abs() < 1e-5);
        assert!((normalized[1] - 0.8).abs() < 1e-5);
        assert_eq!(qwen_weight, vec![2.0_f32.sqrt().recip() - 1.0; 2]);
        assert_eq!(qwen_weight.capacity(), 8);
    }
}
