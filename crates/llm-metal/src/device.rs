use metal::Device;
use std::sync::Arc;

mod buffers;
mod command;
mod error;
mod matvec;
mod pipeline;
mod primitives;
mod qwen;
mod reductions;
mod shaders;

pub use buffers::{Bf16MatrixBuffer, F32Buffer};
pub use error::MetalError;
pub use reductions::{ArgmaxResult, TopKResult};

use self::pipeline::MetalKernel;

#[derive(Debug, Clone)]
pub struct MetalDevice {
    pub(crate) device: Device,
    pub(crate) vector_add: Arc<MetalKernel>,
    pub(crate) qwen_rms_norm: Arc<MetalKernel>,
    pub(crate) softmax_f32: Arc<MetalKernel>,
    pub(crate) linear_attention_conv1d_silu_f32: Arc<MetalKernel>,
    pub(crate) weighted_sum_f32: Arc<MetalKernel>,
    pub(crate) linear_attention_recurrent_update_f32: Arc<MetalKernel>,
    pub(crate) linear_attention_recurrent_update_state_f32: Arc<MetalKernel>,
    pub(crate) select_head_rows_f32: Arc<MetalKernel>,
    pub(crate) matvec_f32: Arc<MetalKernel>,
    pub(crate) matvec_bf16_f32: Arc<MetalKernel>,
    pub(crate) batched_matvec_bf16_f32: Arc<MetalKernel>,
    pub(crate) argmax_f32: Arc<MetalKernel>,
    pub(crate) top_k_f32: Arc<MetalKernel>,
}
