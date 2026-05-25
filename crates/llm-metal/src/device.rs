use metal::Device;
use std::sync::{Arc, Mutex};

mod buffers;
mod command;
mod error;
mod matvec;
mod pipeline;
mod primitives;
mod reductions;
mod shaders;
mod transformer;

pub use buffers::{Bf16MatrixBuffer, F16Buffer, F32Buffer, I8Buffer};
pub use error::MetalError;
pub use reductions::{ArgmaxResult, TopKResult};

use self::{buffers::MetalBufferPool, pipeline::MetalKernel};

pub(crate) fn power_of_two_at_most(value: u64) -> u64 {
    debug_assert!(value > 0);
    1_u64 << (u64::BITS - 1 - value.leading_zeros())
}

pub(crate) fn metal_buffer_byte_len<T>(
    element_count: usize,
    context: &str,
) -> Result<u64, MetalError> {
    let byte_len = element_count
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| {
            MetalError::InvalidShape(format!("{context} byte length overflows usize"))
        })?;
    u64::try_from(byte_len).map_err(|err| {
        MetalError::InvalidShape(format!("{context} byte length does not fit u64: {err}"))
    })
}

#[derive(Debug, Clone)]
pub struct MetalDevice {
    pub(crate) device: Device,
    pub(crate) scratch_buffers: Arc<Mutex<MetalBufferPool>>,
    pub(crate) vector_add: Arc<MetalKernel>,
    pub(crate) rms_norm_f32_kernel: Arc<MetalKernel>,
    pub(crate) softmax_f32: Arc<MetalKernel>,
    pub(crate) attention_scores_f32: Arc<MetalKernel>,
    pub(crate) attention_scores_f16: Arc<MetalKernel>,
    pub(crate) attention_scores_int8: Arc<MetalKernel>,
    pub(crate) softmax_rows_f32: Arc<MetalKernel>,
    pub(crate) attention_weighted_sum_f32: Arc<MetalKernel>,
    pub(crate) attention_weighted_sum_f16: Arc<MetalKernel>,
    pub(crate) attention_weighted_sum_int8: Arc<MetalKernel>,
    pub(crate) linear_attention_conv1d_silu_f32: Arc<MetalKernel>,
    pub(crate) weighted_sum_f32: Arc<MetalKernel>,
    pub(crate) linear_attention_recurrent_update_f32: Arc<MetalKernel>,
    pub(crate) linear_attention_recurrent_update_state_f32: Arc<MetalKernel>,
    pub(crate) select_head_rows_f32: Arc<MetalKernel>,
    pub(crate) select_head_rows_f16: Arc<MetalKernel>,
    pub(crate) select_head_rows_int8: Arc<MetalKernel>,
    pub(crate) matvec_f32: Arc<MetalKernel>,
    pub(crate) matvec_bf16_f32: Arc<MetalKernel>,
    pub(crate) batched_matvec_bf16_f32: Arc<MetalKernel>,
    pub(crate) argmax_f32: Arc<MetalKernel>,
    pub(crate) top_k_f32: Arc<MetalKernel>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_buffer_byte_len_accepts_representable_lengths() {
        assert_eq!(
            metal_buffer_byte_len::<f32>(3, "test buffer").expect("byte length fits"),
            12
        );
    }

    #[test]
    fn metal_buffer_byte_len_rejects_overflowing_lengths() {
        let element_count = usize::MAX / std::mem::size_of::<f32>() + 1;
        let err = metal_buffer_byte_len::<f32>(element_count, "test buffer")
            .expect_err("byte length should overflow");

        assert!(matches!(err, MetalError::InvalidShape(_)));
        assert!(err.to_string().contains("test buffer"));
        assert!(err.to_string().contains("overflows"));
    }
}
