use super::shaders::{
    METAL_SOURCE, SHADER_SOURCE_SHA256, ShaderLibraryLoadCandidate, embedded_metallib,
    shader_library_load_plan,
};
use super::{MetalDevice, MetalError};
use metal::{CommandQueue, CompileOptions, ComputePipelineState, Device};
use std::sync::Arc;

#[derive(Debug)]
pub(crate) struct MetalKernel {
    pub(crate) pipeline: ComputePipelineState,
    pub(crate) queue: Arc<CommandQueue>,
}

#[derive(Debug, Clone)]
pub(crate) struct MetalCommandQueues {
    pub(crate) compute: Arc<CommandQueue>,
    pub(crate) attention: Arc<CommandQueue>,
    pub(crate) transfer: Arc<CommandQueue>,
}

impl MetalCommandQueues {
    fn new(device: &Device) -> Self {
        let compute = Arc::new(device.new_command_queue());
        compute.set_label("llm-metal.compute");
        let attention = Arc::new(device.new_command_queue());
        attention.set_label("llm-metal.attention");
        let transfer = Arc::new(device.new_command_queue());
        transfer.set_label("llm-metal.transfer");
        Self {
            compute,
            attention,
            transfer,
        }
    }

    fn compute(&self) -> &Arc<CommandQueue> {
        &self.compute
    }

    fn attention(&self) -> &Arc<CommandQueue> {
        &self.attention
    }
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
        let library = Self::shader_library(&device)?;
        let queues = MetalCommandQueues::new(&device);
        let vector_add = Self::kernel(&device, &library, queues.compute(), "vector_add")?;
        let rms_norm_f32_kernel =
            Self::kernel(&device, &library, queues.compute(), "rms_norm_f32")?;
        let softmax_f32 = Self::kernel(&device, &library, queues.compute(), "softmax_f32")?;
        let attention_scores_f32 = Self::kernel(
            &device,
            &library,
            queues.attention(),
            "attention_scores_f32",
        )?;
        let attention_scores_f16 = Self::kernel(
            &device,
            &library,
            queues.attention(),
            "attention_scores_f16",
        )?;
        let attention_scores_int8 = Self::kernel(
            &device,
            &library,
            queues.attention(),
            "attention_scores_int8",
        )?;
        let softmax_rows_f32 =
            Self::kernel(&device, &library, queues.attention(), "softmax_rows_f32")?;
        let attention_weighted_sum_f32 = Self::kernel(
            &device,
            &library,
            queues.attention(),
            "attention_weighted_sum_f32",
        )?;
        let attention_weighted_sum_f16 = Self::kernel(
            &device,
            &library,
            queues.attention(),
            "attention_weighted_sum_f16",
        )?;
        let attention_weighted_sum_int8 = Self::kernel(
            &device,
            &library,
            queues.attention(),
            "attention_weighted_sum_int8",
        )?;
        let linear_attention_conv1d_silu_f32 = Self::kernel(
            &device,
            &library,
            queues.attention(),
            "linear_attention_conv1d_silu_f32",
        )?;
        let weighted_sum_f32 =
            Self::kernel(&device, &library, queues.compute(), "weighted_sum_f32")?;
        let linear_attention_recurrent_update_f32 = Self::kernel(
            &device,
            &library,
            queues.attention(),
            "linear_attention_recurrent_update_f32",
        )?;
        let linear_attention_recurrent_update_state_f32 = Self::kernel(
            &device,
            &library,
            queues.attention(),
            "linear_attention_recurrent_update_state_f32",
        )?;
        let select_head_rows_f32 = Self::kernel(
            &device,
            &library,
            queues.attention(),
            "select_head_rows_f32",
        )?;
        let select_head_rows_f16 = Self::kernel(
            &device,
            &library,
            queues.attention(),
            "select_head_rows_f16",
        )?;
        let select_head_rows_int8 = Self::kernel(
            &device,
            &library,
            queues.attention(),
            "select_head_rows_int8",
        )?;
        let matvec_f32 = Self::kernel(&device, &library, queues.compute(), "matvec_f32")?;
        let matvec_bf16_f32 = Self::kernel(&device, &library, queues.compute(), "matvec_bf16_f32")?;
        let batched_matvec_bf16_f32 = Self::kernel(
            &device,
            &library,
            queues.compute(),
            "batched_matvec_bf16_f32",
        )?;
        let argmax_f32 = Self::kernel(&device, &library, queues.compute(), "argmax_f32")?;
        let top_k_f32 = Self::kernel(&device, &library, queues.compute(), "top_k_f32")?;
        Ok(Self {
            device,
            queues,
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

    fn shader_library(device: &Device) -> Result<metal::Library, MetalError> {
        let mut embedded_error = None;
        for candidate in shader_library_load_plan() {
            match candidate {
                ShaderLibraryLoadCandidate::EmbeddedMetallib => {
                    let Some(bytes) = embedded_metallib() else {
                        continue;
                    };
                    match device.new_library_with_data(bytes) {
                        Ok(library) => return Ok(library),
                        Err(err) => {
                            tracing::warn!(
                                error = %err,
                                shader_source_sha256 = SHADER_SOURCE_SHA256,
                                "failed to load embedded Metal shader library; falling back to source compilation"
                            );
                            embedded_error = Some(err);
                        }
                    }
                }
                ShaderLibraryLoadCandidate::Source => {
                    return device
                        .new_library_with_source(METAL_SOURCE, &CompileOptions::new())
                        .map_err(|err| source_compile_error(err, embedded_error.as_deref()));
                }
            }
        }

        device
            .new_library_with_source(METAL_SOURCE, &CompileOptions::new())
            .map_err(MetalError::Compile)
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

    pub(crate) fn transfer_queue(&self) -> &Arc<CommandQueue> {
        &self.queues.transfer
    }
}

fn source_compile_error(err: String, embedded_error: Option<&str>) -> MetalError {
    match embedded_error {
        Some(embedded_error) => MetalError::Compile(format!(
            "embedded metallib load failed ({embedded_error}); source fallback failed: {err}"
        )),
        None => MetalError::Compile(err),
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
    fn metal_device_uses_distinct_command_queues_for_compute_and_transfer() {
        let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
        else {
            eprintln!("no Metal device available; skipping queue sharing test");
            return;
        };

        let compute = &device.queues.compute;
        let attention = &device.queues.attention;
        let transfer = &device.queues.transfer;

        assert_eq!(compute.label(), "llm-metal.compute");
        assert_eq!(attention.label(), "llm-metal.attention");
        assert_eq!(transfer.label(), "llm-metal.transfer");
        assert!(Arc::ptr_eq(compute, &device.vector_add.queue));
        assert!(Arc::ptr_eq(compute, &device.rms_norm_f32_kernel.queue));
        assert!(Arc::ptr_eq(compute, &device.softmax_f32.queue));
        assert!(Arc::ptr_eq(attention, &device.attention_scores_f32.queue));
        assert!(Arc::ptr_eq(attention, &device.attention_scores_f16.queue));
        assert!(Arc::ptr_eq(attention, &device.attention_scores_int8.queue));
        assert!(Arc::ptr_eq(attention, &device.softmax_rows_f32.queue));
        assert!(Arc::ptr_eq(
            attention,
            &device.attention_weighted_sum_f32.queue
        ));
        assert!(Arc::ptr_eq(
            attention,
            &device.attention_weighted_sum_f16.queue
        ));
        assert!(Arc::ptr_eq(
            attention,
            &device.attention_weighted_sum_int8.queue
        ));
        assert!(Arc::ptr_eq(
            attention,
            &device.linear_attention_conv1d_silu_f32.queue
        ));
        assert!(Arc::ptr_eq(compute, &device.weighted_sum_f32.queue));
        assert!(Arc::ptr_eq(
            attention,
            &device.linear_attention_recurrent_update_f32.queue
        ));
        assert!(Arc::ptr_eq(
            attention,
            &device.linear_attention_recurrent_update_state_f32.queue
        ));
        assert!(Arc::ptr_eq(attention, &device.select_head_rows_f32.queue));
        assert!(Arc::ptr_eq(attention, &device.select_head_rows_f16.queue));
        assert!(Arc::ptr_eq(attention, &device.select_head_rows_int8.queue));
        assert!(Arc::ptr_eq(compute, &device.matvec_f32.queue));
        assert!(Arc::ptr_eq(compute, &device.matvec_bf16_f32.queue));
        assert!(Arc::ptr_eq(compute, &device.batched_matvec_bf16_f32.queue));
        assert!(Arc::ptr_eq(compute, &device.argmax_f32.queue));
        assert!(Arc::ptr_eq(compute, &device.top_k_f32.queue));
        assert!(!Arc::ptr_eq(compute, attention));
        assert!(!Arc::ptr_eq(compute, transfer));
        assert!(!Arc::ptr_eq(attention, transfer));

        let cloned = device.clone();
        assert!(Arc::ptr_eq(compute, &cloned.queues.compute));
        assert!(Arc::ptr_eq(attention, &cloned.queues.attention));
        assert!(Arc::ptr_eq(transfer, &cloned.queues.transfer));
        assert!(Arc::ptr_eq(compute, &cloned.vector_add.queue));
        assert!(Arc::ptr_eq(compute, &cloned.top_k_f32.queue));
        assert!(Arc::ptr_eq(attention, &cloned.attention_scores_f32.queue));
    }
}
