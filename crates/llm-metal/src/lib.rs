use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};
use std::ffi::c_void;
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct MetalDevice {
    device: Device,
}

impl MetalDevice {
    pub fn system_default() -> Option<Self> {
        Device::system_default().map(|device| Self { device })
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

        let library = self
            .device
            .new_library_with_source(VECTOR_ADD_SOURCE, &CompileOptions::new())
            .map_err(MetalError::Compile)?;
        let function = library
            .get_function("vector_add", None)
            .map_err(MetalError::Compile)?;
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| MetalError::Pipeline(format!("{err:?}")))?;
        let queue = self.device.new_command_queue();
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

        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&left_buffer), 0);
        encoder.set_buffer(1, Some(&right_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        let threads = MTLSize {
            width: left.len() as u64,
            height: 1,
            depth: 1,
        };
        let group_width = pipeline.thread_execution_width().min(left.len() as u64);
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
}

const VECTOR_ADD_SOURCE: &str = r#"
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
