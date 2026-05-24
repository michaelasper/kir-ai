use super::shaders::METAL_SOURCE;
use super::{MetalDevice, MetalError};
use metal::{CommandQueue, CompileOptions, ComputePipelineState, Device};
use std::sync::Arc;

#[derive(Debug)]
pub(crate) struct MetalKernel {
    pub(crate) pipeline: ComputePipelineState,
    pub(crate) queue: Arc<CommandQueue>,
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
        let command_queue = Arc::new(device.new_command_queue());
        let vector_add = Self::kernel(&device, &library, &command_queue, "vector_add")?;
        let rms_norm_f32_kernel = Self::kernel(&device, &library, &command_queue, "rms_norm_f32")?;
        let softmax_f32 = Self::kernel(&device, &library, &command_queue, "softmax_f32")?;
        let attention_scores_f32 =
            Self::kernel(&device, &library, &command_queue, "attention_scores_f32")?;
        let attention_scores_f16 =
            Self::kernel(&device, &library, &command_queue, "attention_scores_f16")?;
        let attention_scores_int8 =
            Self::kernel(&device, &library, &command_queue, "attention_scores_int8")?;
        let softmax_rows_f32 = Self::kernel(&device, &library, &command_queue, "softmax_rows_f32")?;
        let attention_weighted_sum_f32 = Self::kernel(
            &device,
            &library,
            &command_queue,
            "attention_weighted_sum_f32",
        )?;
        let attention_weighted_sum_f16 = Self::kernel(
            &device,
            &library,
            &command_queue,
            "attention_weighted_sum_f16",
        )?;
        let attention_weighted_sum_int8 = Self::kernel(
            &device,
            &library,
            &command_queue,
            "attention_weighted_sum_int8",
        )?;
        let linear_attention_conv1d_silu_f32 = Self::kernel(
            &device,
            &library,
            &command_queue,
            "linear_attention_conv1d_silu_f32",
        )?;
        let weighted_sum_f32 = Self::kernel(&device, &library, &command_queue, "weighted_sum_f32")?;
        let linear_attention_recurrent_update_f32 = Self::kernel(
            &device,
            &library,
            &command_queue,
            "linear_attention_recurrent_update_f32",
        )?;
        let linear_attention_recurrent_update_state_f32 = Self::kernel(
            &device,
            &library,
            &command_queue,
            "linear_attention_recurrent_update_state_f32",
        )?;
        let select_head_rows_f32 =
            Self::kernel(&device, &library, &command_queue, "select_head_rows_f32")?;
        let select_head_rows_f16 =
            Self::kernel(&device, &library, &command_queue, "select_head_rows_f16")?;
        let select_head_rows_int8 =
            Self::kernel(&device, &library, &command_queue, "select_head_rows_int8")?;
        let matvec_f32 = Self::kernel(&device, &library, &command_queue, "matvec_f32")?;
        let matvec_bf16_f32 = Self::kernel(&device, &library, &command_queue, "matvec_bf16_f32")?;
        let batched_matvec_bf16_f32 =
            Self::kernel(&device, &library, &command_queue, "batched_matvec_bf16_f32")?;
        let argmax_f32 = Self::kernel(&device, &library, &command_queue, "argmax_f32")?;
        let top_k_f32 = Self::kernel(&device, &library, &command_queue, "top_k_f32")?;
        Ok(Self {
            device,
            synchronization: Arc::new(super::command::MetalSynchronization::new()),
            scratch_buffers: Arc::new(std::sync::Mutex::new(
                super::buffers::MetalBufferPool::default(),
            )),
            vector_add,
            rms_norm_f32_kernel,
            softmax_f32,
            attention_scores_f32,
            attention_scores_f16,
            attention_scores_int8,
            softmax_rows_f32,
            attention_weighted_sum_f32,
            attention_weighted_sum_f16,
            attention_weighted_sum_int8,
            linear_attention_conv1d_silu_f32,
            weighted_sum_f32,
            linear_attention_recurrent_update_f32,
            linear_attention_recurrent_update_state_f32,
            select_head_rows_f32,
            select_head_rows_f16,
            select_head_rows_int8,
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
        queue: &Arc<CommandQueue>,
        name: &str,
    ) -> Result<Arc<MetalKernel>, MetalError> {
        let function = library
            .get_function(name, None)
            .map_err(|err| kernel_compile_error(name, err))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| kernel_pipeline_error(name, err))?;
        Ok(Arc::new(MetalKernel {
            pipeline,
            queue: Arc::clone(queue),
        }))
    }
}

fn kernel_compile_error(name: &str, err: String) -> MetalError {
    MetalError::Compile(format!("kernel '{name}': {err}"))
}

fn kernel_pipeline_error(name: &str, err: impl std::fmt::Debug) -> MetalError {
    MetalError::Pipeline(format!("kernel '{name}': {err:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_compile_error_includes_kernel_name() {
        let err = kernel_compile_error("attention_scores_f16", "function missing".to_owned());

        assert!(matches!(&err, MetalError::Compile(_)));
        assert_eq!(
            err.to_string(),
            "Metal compile error: kernel 'attention_scores_f16': function missing"
        );
    }

    #[test]
    fn kernel_pipeline_error_includes_kernel_name() {
        let err = kernel_pipeline_error("matvec_f32", "pipeline failed");

        assert!(matches!(&err, MetalError::Pipeline(_)));
        assert_eq!(
            err.to_string(),
            "Metal pipeline error: kernel 'matvec_f32': \"pipeline failed\""
        );
    }

    #[test]
    fn metal_device_uses_one_command_queue_for_all_kernels() {
        let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
        else {
            eprintln!("no Metal device available; skipping queue sharing test");
            return;
        };

        let queue = &device.vector_add.queue;
        assert!(Arc::ptr_eq(queue, &device.rms_norm_f32_kernel.queue));
        assert!(Arc::ptr_eq(queue, &device.softmax_f32.queue));
        assert!(Arc::ptr_eq(queue, &device.attention_scores_f32.queue));
        assert!(Arc::ptr_eq(queue, &device.attention_scores_f16.queue));
        assert!(Arc::ptr_eq(queue, &device.attention_scores_int8.queue));
        assert!(Arc::ptr_eq(queue, &device.softmax_rows_f32.queue));
        assert!(Arc::ptr_eq(queue, &device.attention_weighted_sum_f32.queue));
        assert!(Arc::ptr_eq(queue, &device.attention_weighted_sum_f16.queue));
        assert!(Arc::ptr_eq(
            queue,
            &device.attention_weighted_sum_int8.queue
        ));
        assert!(Arc::ptr_eq(
            queue,
            &device.linear_attention_conv1d_silu_f32.queue
        ));
        assert!(Arc::ptr_eq(queue, &device.weighted_sum_f32.queue));
        assert!(Arc::ptr_eq(
            queue,
            &device.linear_attention_recurrent_update_f32.queue
        ));
        assert!(Arc::ptr_eq(
            queue,
            &device.linear_attention_recurrent_update_state_f32.queue
        ));
        assert!(Arc::ptr_eq(queue, &device.select_head_rows_f32.queue));
        assert!(Arc::ptr_eq(queue, &device.select_head_rows_f16.queue));
        assert!(Arc::ptr_eq(queue, &device.select_head_rows_int8.queue));
        assert!(Arc::ptr_eq(queue, &device.matvec_f32.queue));
        assert!(Arc::ptr_eq(queue, &device.matvec_bf16_f32.queue));
        assert!(Arc::ptr_eq(queue, &device.batched_matvec_bf16_f32.queue));
        assert!(Arc::ptr_eq(queue, &device.argmax_f32.queue));
        assert!(Arc::ptr_eq(queue, &device.top_k_f32.queue));

        let cloned = device.clone();
        assert!(Arc::ptr_eq(queue, &cloned.vector_add.queue));
        assert!(Arc::ptr_eq(queue, &cloned.top_k_f32.queue));
    }
}
