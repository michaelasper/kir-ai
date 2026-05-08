use super::shaders::METAL_SOURCE;
use super::{MetalDevice, MetalError};
use metal::{CommandQueue, CompileOptions, ComputePipelineState, Device};
use std::sync::Arc;

#[derive(Debug)]
pub(crate) struct MetalKernel {
    pub(crate) pipeline: ComputePipelineState,
    pub(crate) queue: CommandQueue,
}

impl MetalDevice {
    pub fn system_default() -> Option<Self> {
        Self::system_default_result().ok().flatten()
    }

    pub fn system_default_result() -> Result<Option<Self>, MetalError> {
        Device::system_default().map(Self::new).transpose()
    }

    pub fn vector_add_thread_execution_width(&self) -> u64 {
        self.vector_add.pipeline.thread_execution_width()
    }

    fn new(device: Device) -> Result<Self, MetalError> {
        let library = device
            .new_library_with_source(METAL_SOURCE, &CompileOptions::new())
            .map_err(MetalError::Compile)?;
        let vector_add = Self::kernel(&device, &library, "vector_add")?;
        let qwen_rms_norm = Self::kernel(&device, &library, "qwen_rms_norm")?;
        let softmax_f32 = Self::kernel(&device, &library, "softmax_f32")?;
        let linear_attention_conv1d_silu_f32 =
            Self::kernel(&device, &library, "linear_attention_conv1d_silu_f32")?;
        let weighted_sum_f32 = Self::kernel(&device, &library, "weighted_sum_f32")?;
        let linear_attention_recurrent_update_f32 =
            Self::kernel(&device, &library, "linear_attention_recurrent_update_f32")?;
        let linear_attention_recurrent_update_state_f32 = Self::kernel(
            &device,
            &library,
            "linear_attention_recurrent_update_state_f32",
        )?;
        let select_head_rows_f32 = Self::kernel(&device, &library, "select_head_rows_f32")?;
        let matvec_f32 = Self::kernel(&device, &library, "matvec_f32")?;
        let matvec_bf16_f32 = Self::kernel(&device, &library, "matvec_bf16_f32")?;
        let batched_matvec_bf16_f32 = Self::kernel(&device, &library, "batched_matvec_bf16_f32")?;
        let argmax_f32 = Self::kernel(&device, &library, "argmax_f32")?;
        let top_k_f32 = Self::kernel(&device, &library, "top_k_f32")?;
        Ok(Self {
            device,
            vector_add,
            qwen_rms_norm,
            softmax_f32,
            linear_attention_conv1d_silu_f32,
            weighted_sum_f32,
            linear_attention_recurrent_update_f32,
            linear_attention_recurrent_update_state_f32,
            select_head_rows_f32,
            matvec_f32,
            matvec_bf16_f32,
            batched_matvec_bf16_f32,
            argmax_f32,
            top_k_f32,
        })
    }

    fn kernel(
        device: &Device,
        library: &metal::Library,
        name: &str,
    ) -> Result<Arc<MetalKernel>, MetalError> {
        let function = library
            .get_function(name, None)
            .map_err(MetalError::Compile)?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| MetalError::Pipeline(format!("{err:?}")))?;
        let queue = device.new_command_queue();
        Ok(Arc::new(MetalKernel { pipeline, queue }))
    }
}
