use metal::{
    CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};
use std::ffi::c_void;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct MetalDevice {
    device: Device,
    vector_add: Arc<MetalKernel>,
    qwen_rms_norm: Arc<MetalKernel>,
}

#[derive(Debug)]
struct MetalKernel {
    pipeline: ComputePipelineState,
    queue: CommandQueue,
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
        Ok(Self {
            device,
            vector_add,
            qwen_rms_norm,
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

    pub fn add_f32(&self, left: &[f32], right: &[f32]) -> Result<Vec<f32>, MetalError> {
        if left.len() != right.len() {
            return Err(MetalError::InvalidShape(
                "left and right inputs must have the same length".to_owned(),
            ));
        }
        if left.is_empty() {
            return Ok(Vec::new());
        }

        let byte_len = std::mem::size_of_val(left) as u64;
        let left_buffer = self.device.new_buffer_with_data(
            left.as_ptr().cast::<c_void>(),
            byte_len,
            MTLResourceOptions::StorageModeShared,
        );
        let right_buffer = self.device.new_buffer_with_data(
            right.as_ptr().cast::<c_void>(),
            byte_len,
            MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self
            .device
            .new_buffer(byte_len, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.vector_add.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.vector_add.pipeline);
        encoder.set_buffer(0, Some(&left_buffer), 0);
        encoder.set_buffer(1, Some(&right_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        let threads = MTLSize {
            width: left.len() as u64,
            height: 1,
            depth: 1,
        };
        let group_width = self
            .vector_add
            .pipeline
            .thread_execution_width()
            .min(left.len() as u64);
        let threads_per_group = MTLSize {
            width: group_width,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads, threads_per_group);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // SAFETY: output_buffer is a StorageModeShared Metal buffer allocated with
        // byte_len bytes above. The command buffer has completed, and the buffer
        // remains alive for the duration of this read. The pointer is interpreted
        // as f32 values matching the byte length used to allocate it.
        let values = unsafe {
            let ptr = output_buffer.contents().cast::<f32>();
            std::slice::from_raw_parts(ptr, left.len()).to_vec()
        };
        Ok(values)
    }

    pub fn qwen_rms_norm_f32(
        &self,
        input: &[f32],
        weight: &[f32],
        eps: f32,
    ) -> Result<Vec<f32>, MetalError> {
        if input.len() != weight.len() {
            return Err(MetalError::InvalidShape(
                "input and weight must have the same length".to_owned(),
            ));
        }
        if !eps.is_finite() || eps < 0.0 {
            return Err(MetalError::InvalidShape(
                "epsilon must be finite and non-negative".to_owned(),
            ));
        }
        if input.is_empty() {
            return Ok(Vec::new());
        }
        let len = u32::try_from(input.len()).map_err(|err| {
            MetalError::InvalidShape(format!("input length does not fit u32: {err}"))
        })?;
        let byte_len = std::mem::size_of_val(input) as u64;
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr().cast::<c_void>(),
            byte_len,
            MTLResourceOptions::StorageModeShared,
        );
        let weight_buffer = self.device.new_buffer_with_data(
            weight.as_ptr().cast::<c_void>(),
            byte_len,
            MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self
            .device
            .new_buffer(byte_len, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.qwen_rms_norm.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.qwen_rms_norm.pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&weight_buffer), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of_val(&len) as u64,
            (&len as *const u32).cast::<c_void>(),
        );
        encoder.set_bytes(
            3,
            std::mem::size_of_val(&eps) as u64,
            (&eps as *const f32).cast::<c_void>(),
        );
        encoder.set_buffer(4, Some(&output_buffer), 0);
        let threads = MTLSize {
            width: input.len() as u64,
            height: 1,
            depth: 1,
        };
        let group_width = self
            .qwen_rms_norm
            .pipeline
            .thread_execution_width()
            .min(input.len() as u64);
        let threads_per_group = MTLSize {
            width: group_width,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads, threads_per_group);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // SAFETY: output_buffer is a completed StorageModeShared Metal buffer
        // with the same byte length as the input slice.
        let values = unsafe {
            let ptr = output_buffer.contents().cast::<f32>();
            std::slice::from_raw_parts(ptr, input.len()).to_vec()
        };
        Ok(values)
    }
}

const METAL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void vector_add(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = left[id] + right[id];
}

kernel void qwen_rms_norm(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    constant uint& len [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    device float* output [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= len) {
        return;
    }
    float sum = 0.0;
    for (uint index = 0; index < len; index++) {
        float value = input[index];
        sum += value * value;
    }
    float inv_rms = rsqrt((sum / float(len)) + eps);
    output[id] = input[id] * inv_rms * (weight[id] + 1.0);
}
"#;

#[derive(Debug, Error)]
pub enum MetalError {
    #[error("invalid Metal input shape: {0}")]
    InvalidShape(String),
    #[error("Metal compile error: {0}")]
    Compile(String),
    #[error("Metal pipeline error: {0}")]
    Pipeline(String),
    #[error("Metal execution error: {0}")]
    Execution(String),
}
