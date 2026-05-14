use super::super::NativeMatvecBackend;
use super::super::math::MathError;

#[cfg(test)]
use super::super::CpuNativeMatvecBackend;

pub(super) async fn rms_norm_f32(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, MathError> {
    let mut output = vec![0.0; input.len()];
    rms_norm_f32_in_place(input, weight, eps, matvec, &mut output).await?;
    Ok(output)
}

pub(super) async fn rms_norm_f32_in_place(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    matvec: &impl NativeMatvecBackend,
    output: &mut [f32],
) -> Result<(), MathError> {
    if input.len() != weight.len() {
        return Err(MathError::InvalidShape(
            "input and weight must have the same length".to_owned(),
        ));
    }
    matvec
        .rms_norm_f32_in_place(input, weight, eps, output)
        .await
}

pub(super) async fn l2_normalize_f32(
    input: &[f32],
    eps: f32,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, MathError> {
    let mut qwen_weight = Vec::new();
    let mut output = vec![0.0; input.len()];
    l2_normalize_f32_and_weight_scratch(input, eps, matvec, &mut qwen_weight, &mut output).await?;
    Ok(output)
}

pub(super) async fn l2_normalize_f32_and_weight_scratch(
    input: &[f32],
    eps: f32,
    matvec: &impl NativeMatvecBackend,
    qwen_weight: &mut Vec<f32>,
    output: &mut [f32],
) -> Result<(), MathError> {
    if input.is_empty() {
        qwen_weight.clear();
        return Ok(());
    }
    if eps < 0.0 {
        return Err(MathError::InvalidShape(
            "l2 norm epsilon must be non-negative".to_owned(),
        ));
    }
    let weight_scale = (input.len() as f32).sqrt().recip();
    qwen_weight.clear();
    qwen_weight.resize(input.len(), weight_scale - 1.0);
    matvec
        .rms_norm_one_centered_f32_in_place(input, qwen_weight, eps / input.len() as f32, output)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{
        LayerKvCache, LinearAttentionCache, NativeKvCacheTensor, SafeTensorShardStore,
        TensorLoadError, TopKLogit, TopKWeight,
    };
    use std::sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    };

    #[derive(Default)]
    struct RecordingRmsNormBackend {
        raw_calls: AtomicUsize,
        one_centered_calls: AtomicUsize,
        observed_weight: Mutex<Vec<f32>>,
    }

    impl NativeMatvecBackend for RecordingRmsNormBackend {
        async fn bf16_matvec_row_major_f32_in_place(
            &self,
            store: &SafeTensorShardStore,
            tensor: &str,
            input: &[f32],
            output: &mut [f32],
        ) -> Result<(), TensorLoadError> {
            CpuNativeMatvecBackend
                .bf16_matvec_row_major_f32_in_place(store, tensor, input, output)
                .await
        }

        async fn bf16_matvec_rows_f32_in_place(
            &self,
            store: &SafeTensorShardStore,
            tensor: &str,
            input: &[f32],
            chunk_rows: usize,
            output: &mut [f32],
        ) -> Result<(), TensorLoadError> {
            CpuNativeMatvecBackend
                .bf16_matvec_rows_f32_in_place(store, tensor, input, chunk_rows, output)
                .await
        }

        async fn bf16_matvec_top_k_rows_f32(
            &self,
            store: &SafeTensorShardStore,
            tensor: &str,
            input: &[f32],
            top_k: usize,
            chunk_rows: usize,
        ) -> Result<Vec<TopKLogit>, TensorLoadError> {
            CpuNativeMatvecBackend
                .bf16_matvec_top_k_rows_f32(store, tensor, input, top_k, chunk_rows)
                .await
        }

        async fn matvec_row_major_f32_in_place(
            &self,
            input: &[f32],
            weights: &[f32],
            rows: usize,
            columns: usize,
            output: &mut [f32],
        ) -> Result<(), MathError> {
            CpuNativeMatvecBackend
                .matvec_row_major_f32_in_place(input, weights, rows, columns, output)
                .await
        }

        async fn rms_norm_f32_in_place(
            &self,
            input: &[f32],
            weight: &[f32],
            eps: f32,
            output: &mut [f32],
        ) -> Result<(), MathError> {
            self.raw_calls.fetch_add(1, Ordering::Relaxed);
            *self.observed_weight.lock().expect("observed weight lock") = weight.to_vec();
            CpuNativeMatvecBackend
                .rms_norm_f32_in_place(input, weight, eps, output)
                .await
        }

        async fn rms_norm_one_centered_f32_in_place(
            &self,
            input: &[f32],
            weight: &[f32],
            eps: f32,
            output: &mut [f32],
        ) -> Result<(), MathError> {
            self.one_centered_calls.fetch_add(1, Ordering::Relaxed);
            CpuNativeMatvecBackend
                .rms_norm_one_centered_f32_in_place(input, weight, eps, output)
                .await
        }

        async fn softmax_f32_in_place(
            &self,
            scores: &[f32],
            output: &mut [f32],
        ) -> Result<(), MathError> {
            CpuNativeMatvecBackend
                .softmax_f32_in_place(scores, output)
                .await
        }

        async fn linear_attention_conv1d_silu_f32_in_place(
            &self,
            window: &[f32],
            weights: &[f32],
            conv_dim: usize,
            kernel_size: usize,
            output: &mut [f32],
        ) -> Result<(), MathError> {
            CpuNativeMatvecBackend
                .linear_attention_conv1d_silu_f32_in_place(
                    window,
                    weights,
                    conv_dim,
                    kernel_size,
                    output,
                )
                .await
        }

        async fn weighted_sum_f32_in_place(
            &self,
            values: &[f32],
            weights: &[f32],
            vector_len: usize,
            output: &mut [f32],
        ) -> Result<(), MathError> {
            CpuNativeMatvecBackend
                .weighted_sum_f32_in_place(values, weights, vector_len, output)
                .await
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
        ) -> Result<(), MathError> {
            CpuNativeMatvecBackend
                .linear_attention_recurrent_update_f32_in_place(
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
                .await
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
            CpuNativeMatvecBackend
                .select_head_rows_f32_in_place(
                    values, row_count, row_len, head_start, head_len, output,
                )
                .await
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
            CpuNativeMatvecBackend
                .select_kv_cache_head_rows_f32_in_place(
                    cache, tensor, row_count, head_start, head_len, output,
                )
                .await
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
            CpuNativeMatvecBackend
                .linear_attention_recurrent_cache_update_f32_in_place(
                    cache,
                    state_start,
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
            CpuNativeMatvecBackend
                .softmax_top_k_f32(logits, top_k)
                .await
        }
    }

    #[tokio::test]
    async fn rms_norm_f32_forwards_raw_weight_without_one_center_scratch() {
        let matvec = RecordingRmsNormBackend::default();
        let mut output = vec![0.0; 2];

        rms_norm_f32_in_place(&[3.0, 4.0], &[1.5, 2.5], 0.0, &matvec, &mut output)
            .await
            .expect("rms norm succeeds");

        assert_eq!(matvec.raw_calls.load(Ordering::Relaxed), 1);
        assert_eq!(matvec.one_centered_calls.load(Ordering::Relaxed), 0);
        assert_eq!(
            *matvec.observed_weight.lock().expect("observed weight lock"),
            vec![1.5, 2.5]
        );
        assert!((output[0] - 1.2727922).abs() < 1e-5);
        assert!((output[1] - 2.828427).abs() < 1e-5);
    }

    #[tokio::test]
    async fn l2_normalize_f32_reuses_weight_scratch() {
        let mut qwen_weight = Vec::with_capacity(8);
        let mut output = vec![0.0; 2];

        l2_normalize_f32_and_weight_scratch(
            &[3.0, 4.0],
            1e-6,
            &CpuNativeMatvecBackend,
            &mut qwen_weight,
            &mut output,
        )
        .await
        .expect("l2 normalize succeeds");

        assert!((output[0] - 0.6).abs() < 1e-5);
        assert!((output[1] - 0.8).abs() < 1e-5);
        assert_eq!(qwen_weight, vec![2.0_f32.sqrt().recip() - 1.0; 2]);
        assert_eq!(qwen_weight.capacity(), 8);
    }
}
