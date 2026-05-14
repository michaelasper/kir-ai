use super::math::{
    InferenceScratchpad, MathError, TopKLogit, TopKWeight,
    linear_attention_conv1d_silu_f32_in_place, linear_attention_recurrent_update_f32_in_place,
    matvec_row_major_f32_in_place, rms_norm_f32_in_place as math_rms_norm_f32_in_place,
    rms_norm_one_centered_f32_in_place, select_head_rows_f32_in_place, silu_f32,
    softmax_f32_in_place, softmax_top_k_f32, weighted_sum_f32_in_place,
};
use super::{LayerKvCache, LinearAttentionCache, SafeTensorShardStore, TensorLoadError};

#[derive(Debug, Clone, PartialEq)]
pub struct NativeBatchedMatvecOutput {
    values: Vec<f32>,
    row_len: usize,
}

impl NativeBatchedMatvecOutput {
    pub fn new(values: Vec<f32>, row_len: usize) -> Result<Self, TensorLoadError> {
        if row_len == 0 {
            if values.is_empty() {
                return Ok(Self { values, row_len });
            }
            return Err(TensorLoadError::integrity(
                "batched matvec row length must be non-zero for non-empty values",
            ));
        }
        if !values.len().is_multiple_of(row_len) {
            return Err(TensorLoadError::integrity(format!(
                "batched matvec values length {} must be divisible by row length {row_len}",
                values.len()
            )));
        }
        Ok(Self { values, row_len })
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn row_len(&self) -> usize {
        self.row_len
    }

    pub fn row_count(&self) -> usize {
        self.values.len().checked_div(self.row_len).unwrap_or(0)
    }

    pub fn into_rows(self) -> Vec<Vec<f32>> {
        if self.row_len == 0 {
            return Vec::new();
        }
        self.values
            .chunks_exact(self.row_len)
            .map(<[f32]>::to_vec)
            .collect()
    }

    pub fn into_values(self) -> Vec<f32> {
        self.values
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn swiglu_mlp_f32_with_matvec(
    input: &[f32],
    gate_weight: &[f32],
    up_weight: &[f32],
    down_weight: &[f32],
    intermediate_size: usize,
    matvec: &impl NativeMatvecBackend,
    scratch: &mut InferenceScratchpad,
    output: &mut [f32],
) -> Result<(), MathError> {
    let gate = InferenceScratchpad::get_mut(&mut scratch.buf0, intermediate_size);
    matvec
        .matvec_row_major_f32_in_place(input, gate_weight, intermediate_size, input.len(), gate)
        .await?;
    let up = InferenceScratchpad::get_mut(&mut scratch.buf1, intermediate_size);
    matvec
        .matvec_row_major_f32_in_place(input, up_weight, intermediate_size, input.len(), up)
        .await?;
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
    matvec
        .matvec_row_major_f32_in_place(activated, down_weight, rows, intermediate_size, output)
        .await
}

#[allow(async_fn_in_trait)]
pub trait NativeMatvecBackend {
    async fn bf16_matvec_row_major_f32(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        input: &[f32],
    ) -> Result<Vec<f32>, TensorLoadError> {
        let mut output = vec![0.0; store.tensor_metadata(tensor)?.shape[0]];
        self.bf16_matvec_row_major_f32_in_place(store, tensor, input, &mut output)
            .await?;
        Ok(output)
    }

    async fn bf16_matvec_row_major_f32_in_place(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), TensorLoadError>;

    async fn bf16_matvecs_row_major_f32(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        inputs: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, TensorLoadError> {
        Ok(self
            .bf16_matvecs_row_major_f32_flat(store, tensor, inputs)
            .await?
            .into_rows())
    }

    async fn bf16_matvecs_row_major_f32_flat(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        inputs: &[Vec<f32>],
    ) -> Result<NativeBatchedMatvecOutput, TensorLoadError> {
        store.bf16_matvecs_row_major_f32_flat(tensor, inputs)
    }

    async fn bf16_matvec_rows_f32(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        input: &[f32],
        chunk_rows: usize,
    ) -> Result<Vec<f32>, TensorLoadError> {
        let mut output = vec![0.0; store.tensor_metadata(tensor)?.shape[0]];
        self.bf16_matvec_rows_f32_in_place(store, tensor, input, chunk_rows, &mut output)
            .await?;
        Ok(output)
    }

    async fn bf16_matvec_rows_f32_in_place(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        input: &[f32],
        chunk_rows: usize,
        output: &mut [f32],
    ) -> Result<(), TensorLoadError>;

    async fn bf16_matvec_range_row_major_f32(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        element_offset: usize,
        rows: usize,
        columns: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, TensorLoadError> {
        let mut output = vec![0.0; rows];
        self.bf16_matvec_range_row_major_f32_in_place(
            store,
            tensor,
            element_offset,
            rows,
            columns,
            input,
            &mut output,
        )
        .await?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    async fn bf16_matvec_range_row_major_f32_in_place(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        element_offset: usize,
        rows: usize,
        columns: usize,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), TensorLoadError> {
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
        self.matvec_row_major_f32_in_place(input, &weights, rows, columns, output)
            .await
            .map_err(|err| TensorLoadError::integrity(format!("BF16 range matvec failed: {err}")))
    }

    async fn bf16_matvec_top_k_rows_f32(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        input: &[f32],
        top_k: usize,
        chunk_rows: usize,
    ) -> Result<Vec<TopKLogit>, TensorLoadError> {
        store.bf16_matvec_top_k_rows_f32(tensor, input, top_k, chunk_rows)
    }

    async fn matvec_row_major_f32(
        &self,
        input: &[f32],
        weights: &[f32],
        rows: usize,
        columns: usize,
    ) -> Result<Vec<f32>, MathError> {
        let mut output = vec![0.0; rows];
        self.matvec_row_major_f32_in_place(input, weights, rows, columns, &mut output)
            .await?;
        Ok(output)
    }

    async fn matvec_row_major_f32_in_place(
        &self,
        input: &[f32],
        weights: &[f32],
        rows: usize,
        columns: usize,
        output: &mut [f32],
    ) -> Result<(), MathError>;

    async fn rms_norm_f32(
        &self,
        input: &[f32],
        weight: &[f32],
        eps: f32,
    ) -> Result<Vec<f32>, MathError> {
        let mut output = vec![0.0; input.len()];
        self.rms_norm_f32_in_place(input, weight, eps, &mut output)
            .await?;
        Ok(output)
    }

    async fn rms_norm_f32_in_place(
        &self,
        input: &[f32],
        weight: &[f32],
        eps: f32,
        output: &mut [f32],
    ) -> Result<(), MathError> {
        math_rms_norm_f32_in_place(input, weight, eps, output)
    }

    async fn rms_norm_one_centered_f32(
        &self,
        input: &[f32],
        weight: &[f32],
        eps: f32,
    ) -> Result<Vec<f32>, MathError> {
        let mut output = vec![0.0; input.len()];
        self.rms_norm_one_centered_f32_in_place(input, weight, eps, &mut output)
            .await?;
        Ok(output)
    }

    async fn rms_norm_one_centered_f32_in_place(
        &self,
        input: &[f32],
        weight: &[f32],
        eps: f32,
        output: &mut [f32],
    ) -> Result<(), MathError>;

    async fn softmax_f32(&self, scores: &[f32]) -> Result<Vec<f32>, MathError> {
        let mut output = vec![0.0; scores.len()];
        self.softmax_f32_in_place(scores, &mut output).await?;
        Ok(output)
    }

    async fn softmax_f32_in_place(
        &self,
        scores: &[f32],
        output: &mut [f32],
    ) -> Result<(), MathError>;

    async fn linear_attention_conv1d_silu_f32(
        &self,
        window: &[f32],
        weights: &[f32],
        conv_dim: usize,
        kernel_size: usize,
    ) -> Result<Vec<f32>, MathError> {
        let mut output = vec![0.0; conv_dim];
        self.linear_attention_conv1d_silu_f32_in_place(
            window,
            weights,
            conv_dim,
            kernel_size,
            &mut output,
        )
        .await?;
        Ok(output)
    }

    async fn linear_attention_conv1d_silu_f32_in_place(
        &self,
        window: &[f32],
        weights: &[f32],
        conv_dim: usize,
        kernel_size: usize,
        output: &mut [f32],
    ) -> Result<(), MathError>;

    async fn weighted_sum_f32(
        &self,
        values: &[f32],
        weights: &[f32],
        vector_len: usize,
    ) -> Result<Vec<f32>, MathError> {
        let mut output = vec![0.0; vector_len];
        self.weighted_sum_f32_in_place(values, weights, vector_len, &mut output)
            .await?;
        Ok(output)
    }

    async fn weighted_sum_f32_in_place(
        &self,
        values: &[f32],
        weights: &[f32],
        vector_len: usize,
        output: &mut [f32],
    ) -> Result<(), MathError>;

    #[allow(clippy::too_many_arguments)]
    async fn linear_attention_recurrent_update_f32(
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
        let mut output = vec![0.0; state.len()];
        self.linear_attention_recurrent_update_f32_in_place(
            state,
            key,
            value,
            memory,
            beta,
            decay,
            key_head_dim,
            value_head_dim,
            &mut output,
        )
        .await?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    async fn linear_attention_recurrent_update_f32_in_place(
        &self,
        state: &[f32],
        key: &[f32],
        value: &[f32],
        memory: &[f32],
        beta: f32,
        decay: f32,
        key_head_dim: usize,
        value_head_dim: usize,
        output: &mut [f32],
    ) -> Result<(), MathError>;

    async fn select_head_rows_f32(
        &self,
        values: &[f32],
        row_count: usize,
        row_len: usize,
        head_start: usize,
        head_len: usize,
    ) -> Result<Vec<f32>, MathError> {
        let mut output = vec![0.0; row_count * head_len];
        self.select_head_rows_f32_in_place(
            values,
            row_count,
            row_len,
            head_start,
            head_len,
            &mut output,
        )
        .await?;
        Ok(output)
    }

    async fn select_head_rows_f32_in_place(
        &self,
        values: &[f32],
        row_count: usize,
        row_len: usize,
        head_start: usize,
        head_len: usize,
        output: &mut [f32],
    ) -> Result<(), MathError>;

    async fn select_kv_cache_head_rows_f32(
        &self,
        cache: &LayerKvCache,
        tensor: NativeKvCacheTensor,
        row_count: usize,
        head_start: usize,
        head_len: usize,
    ) -> Result<Vec<f32>, MathError> {
        let mut output = vec![0.0; row_count * head_len];
        self.select_kv_cache_head_rows_f32_in_place(
            cache,
            tensor,
            row_count,
            head_start,
            head_len,
            &mut output,
        )
        .await?;
        Ok(output)
    }

    async fn select_kv_cache_head_rows_f32_in_place(
        &self,
        cache: &LayerKvCache,
        tensor: NativeKvCacheTensor,
        row_count: usize,
        head_start: usize,
        head_len: usize,
        output: &mut [f32],
    ) -> Result<(), MathError> {
        let values = match tensor {
            NativeKvCacheTensor::Key => cache.keys(),
            NativeKvCacheTensor::Value => cache.values(),
        };
        self.select_head_rows_f32_in_place(
            values,
            row_count,
            cache.vector_len(),
            head_start,
            head_len,
            output,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn full_attention_cache_mix_f32_in_place(
        &self,
        _cache: &LayerKvCache,
        _query: &[f32],
        _row_count: usize,
        _num_attention_heads: usize,
        _num_key_value_heads: usize,
        _head_dim: usize,
        _score_scale: f32,
        _output: &mut [f32],
    ) -> Result<bool, MathError> {
        Ok(false)
    }

    #[allow(clippy::too_many_arguments)]
    async fn linear_attention_recurrent_cache_update_f32(
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
        let mut output = vec![0.0; key_head_dim * value_head_dim];
        self.linear_attention_recurrent_cache_update_f32_in_place(
            cache,
            state_start,
            key,
            value,
            memory,
            beta,
            decay,
            key_head_dim,
            value_head_dim,
            &mut output,
        )
        .await?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    async fn linear_attention_recurrent_cache_update_f32_in_place(
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
        output: &mut [f32],
    ) -> Result<(), MathError> {
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
        self.linear_attention_recurrent_update_f32_in_place(
            &recurrent_state[state_start..state_end],
            key,
            value,
            memory,
            beta,
            decay,
            key_head_dim,
            value_head_dim,
            output,
        )
        .await
    }

    async fn softmax_top_k_f32(
        &self,
        logits: &[f32],
        top_k: usize,
    ) -> Result<Vec<TopKWeight>, MathError> {
        softmax_top_k_f32(logits, top_k)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeKvCacheTensor {
    Key,
    Value,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CpuNativeMatvecBackend;

impl NativeMatvecBackend for CpuNativeMatvecBackend {
    async fn bf16_matvec_row_major_f32_in_place(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), TensorLoadError> {
        store.bf16_matvec_row_major_f32_in_place(tensor, input, output)
    }

    async fn bf16_matvec_rows_f32_in_place(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        input: &[f32],
        chunk_rows: usize,
        output: &mut [f32],
    ) -> Result<(), TensorLoadError> {
        store.bf16_matvec_rows_f32_in_place(tensor, input, chunk_rows, output)
    }

    async fn matvec_row_major_f32_in_place(
        &self,
        input: &[f32],
        weights: &[f32],
        rows: usize,
        columns: usize,
        output: &mut [f32],
    ) -> Result<(), MathError> {
        matvec_row_major_f32_in_place(input, weights, rows, columns, output)
    }

    async fn rms_norm_one_centered_f32_in_place(
        &self,
        input: &[f32],
        weight: &[f32],
        eps: f32,
        output: &mut [f32],
    ) -> Result<(), MathError> {
        rms_norm_one_centered_f32_in_place(input, weight, eps, output)
    }

    async fn softmax_f32_in_place(
        &self,
        scores: &[f32],
        output: &mut [f32],
    ) -> Result<(), MathError> {
        softmax_f32_in_place(scores, output)
    }

    async fn linear_attention_conv1d_silu_f32_in_place(
        &self,
        window: &[f32],
        weights: &[f32],
        conv_dim: usize,
        kernel_size: usize,
        output: &mut [f32],
    ) -> Result<(), MathError> {
        linear_attention_conv1d_silu_f32_in_place(window, weights, conv_dim, kernel_size, output)
    }

    async fn weighted_sum_f32_in_place(
        &self,
        values: &[f32],
        weights: &[f32],
        vector_len: usize,
        output: &mut [f32],
    ) -> Result<(), MathError> {
        weighted_sum_f32_in_place(values, weights, vector_len, output)
    }

    async fn linear_attention_recurrent_update_f32_in_place(
        &self,
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
        linear_attention_recurrent_update_f32_in_place(
            state,
            key,
            value,
            memory,
            beta,
            decay,
            key_head_dim,
            value_head_dim,
            output,
        )
    }

    async fn select_head_rows_f32_in_place(
        &self,
        values: &[f32],
        row_count: usize,
        row_len: usize,
        head_start: usize,
        head_len: usize,
        output: &mut [f32],
    ) -> Result<(), MathError> {
        select_head_rows_f32_in_place(values, row_count, row_len, head_start, head_len, output)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn cpu_in_place_methods_do_not_delegate_through_allocating_vec_wrappers() {
        let source = include_str!("native_matvec.rs");
        for disallowed in [
            concat!("let vec = ", "store.bf16_matvec_row_major_f32("),
            concat!("let vec = ", "store.bf16_matvec_rows_f32("),
            concat!("let vec = ", "softmax_f32("),
            concat!("let vec = ", "linear_attention_conv1d_silu_f32("),
            concat!("let vec = ", "weighted_sum_f32("),
            concat!("let vec = ", "linear_attention_recurrent_update_f32("),
            concat!("let vec = ", "select_head_rows_f32("),
        ] {
            assert!(
                !source.contains(disallowed),
                "CpuNativeMatvecBackend in-place method still delegates through `{disallowed}`"
            );
        }
    }
}
