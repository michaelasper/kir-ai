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
    matvec_f32: Arc<MetalKernel>,
    matvec_bf16_f32: Arc<MetalKernel>,
    batched_matvec_bf16_f32: Arc<MetalKernel>,
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
        let matvec_f32 = Self::kernel(&device, &library, "matvec_f32")?;
        let matvec_bf16_f32 = Self::kernel(&device, &library, "matvec_bf16_f32")?;
        let batched_matvec_bf16_f32 = Self::kernel(&device, &library, "batched_matvec_bf16_f32")?;
        Ok(Self {
            device,
            vector_add,
            qwen_rms_norm,
            matvec_f32,
            matvec_bf16_f32,
            batched_matvec_bf16_f32,
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

    pub fn matvec_f32(
        &self,
        matrix: &[f32],
        rows: usize,
        cols: usize,
        vector: &[f32],
    ) -> Result<Vec<f32>, MetalError> {
        let expected_matrix_len = rows
            .checked_mul(cols)
            .ok_or_else(|| MetalError::InvalidShape("matrix shape overflows usize".to_owned()))?;
        if matrix.len() != expected_matrix_len {
            return Err(MetalError::InvalidShape(format!(
                "matrix length {} does not match rows {rows} * cols {cols}",
                matrix.len()
            )));
        }
        if vector.len() != cols {
            return Err(MetalError::InvalidShape(format!(
                "vector length {} does not match cols {cols}",
                vector.len()
            )));
        }
        if rows == 0 {
            return Ok(Vec::new());
        }
        if cols == 0 {
            return Ok(vec![0.0; rows]);
        }
        let rows_u32 = u32::try_from(rows).map_err(|err| {
            MetalError::InvalidShape(format!("row count does not fit u32: {err}"))
        })?;
        let cols_u32 = u32::try_from(cols).map_err(|err| {
            MetalError::InvalidShape(format!("column count does not fit u32: {err}"))
        })?;
        let matrix_byte_len = std::mem::size_of_val(matrix) as u64;
        let vector_byte_len = std::mem::size_of_val(vector) as u64;
        let output_byte_len = rows
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| MetalError::InvalidShape("output byte length overflow".to_owned()))?
            as u64;
        let matrix_buffer = self.device.new_buffer_with_data(
            matrix.as_ptr().cast::<c_void>(),
            matrix_byte_len,
            MTLResourceOptions::StorageModeShared,
        );
        let vector_buffer = self.device.new_buffer_with_data(
            vector.as_ptr().cast::<c_void>(),
            vector_byte_len,
            MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self
            .device
            .new_buffer(output_byte_len, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.matvec_f32.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.matvec_f32.pipeline);
        encoder.set_buffer(0, Some(&matrix_buffer), 0);
        encoder.set_buffer(1, Some(&vector_buffer), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of_val(&rows_u32) as u64,
            (&rows_u32 as *const u32).cast::<c_void>(),
        );
        encoder.set_bytes(
            3,
            std::mem::size_of_val(&cols_u32) as u64,
            (&cols_u32 as *const u32).cast::<c_void>(),
        );
        encoder.set_buffer(4, Some(&output_buffer), 0);
        let threads = MTLSize {
            width: rows as u64,
            height: 1,
            depth: 1,
        };
        let group_width = self
            .matvec_f32
            .pipeline
            .thread_execution_width()
            .min(rows as u64);
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
        // containing one f32 per requested matrix row.
        let values = unsafe {
            let ptr = output_buffer.contents().cast::<f32>();
            std::slice::from_raw_parts(ptr, rows).to_vec()
        };
        Ok(values)
    }

    pub fn matvec_bf16_f32(
        &self,
        matrix: &[u16],
        rows: usize,
        cols: usize,
        vector: &[f32],
    ) -> Result<Vec<f32>, MetalError> {
        let expected_matrix_len = rows
            .checked_mul(cols)
            .ok_or_else(|| MetalError::InvalidShape("matrix shape overflows usize".to_owned()))?;
        if matrix.len() != expected_matrix_len {
            return Err(MetalError::InvalidShape(format!(
                "matrix length {} does not match rows {rows} * cols {cols}",
                matrix.len()
            )));
        }
        if vector.len() != cols {
            return Err(MetalError::InvalidShape(format!(
                "vector length {} does not match cols {cols}",
                vector.len()
            )));
        }
        if rows == 0 {
            return Ok(Vec::new());
        }
        if cols == 0 {
            return Ok(vec![0.0; rows]);
        }
        let rows_u32 = u32::try_from(rows).map_err(|err| {
            MetalError::InvalidShape(format!("row count does not fit u32: {err}"))
        })?;
        let cols_u32 = u32::try_from(cols).map_err(|err| {
            MetalError::InvalidShape(format!("column count does not fit u32: {err}"))
        })?;
        let matrix_byte_len = std::mem::size_of_val(matrix) as u64;
        let vector_byte_len = std::mem::size_of_val(vector) as u64;
        let output_byte_len = rows
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| MetalError::InvalidShape("output byte length overflow".to_owned()))?
            as u64;
        let matrix_buffer = self.device.new_buffer_with_data(
            matrix.as_ptr().cast::<c_void>(),
            matrix_byte_len,
            MTLResourceOptions::StorageModeShared,
        );
        let vector_buffer = self.device.new_buffer_with_data(
            vector.as_ptr().cast::<c_void>(),
            vector_byte_len,
            MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self
            .device
            .new_buffer(output_byte_len, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.matvec_bf16_f32.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.matvec_bf16_f32.pipeline);
        encoder.set_buffer(0, Some(&matrix_buffer), 0);
        encoder.set_buffer(1, Some(&vector_buffer), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of_val(&rows_u32) as u64,
            (&rows_u32 as *const u32).cast::<c_void>(),
        );
        encoder.set_bytes(
            3,
            std::mem::size_of_val(&cols_u32) as u64,
            (&cols_u32 as *const u32).cast::<c_void>(),
        );
        encoder.set_buffer(4, Some(&output_buffer), 0);
        let threads = MTLSize {
            width: rows as u64,
            height: 1,
            depth: 1,
        };
        let group_width = self
            .matvec_bf16_f32
            .pipeline
            .thread_execution_width()
            .min(rows as u64);
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
        // containing one f32 per requested matrix row.
        let values = unsafe {
            let ptr = output_buffer.contents().cast::<f32>();
            std::slice::from_raw_parts(ptr, rows).to_vec()
        };
        Ok(values)
    }

    pub fn batched_matvec_bf16_f32(
        &self,
        matrix: &[u16],
        rows: usize,
        cols: usize,
        vectors: &[f32],
        vector_count: usize,
    ) -> Result<Vec<f32>, MetalError> {
        let expected_matrix_len = rows
            .checked_mul(cols)
            .ok_or_else(|| MetalError::InvalidShape("matrix shape overflows usize".to_owned()))?;
        if matrix.len() != expected_matrix_len {
            return Err(MetalError::InvalidShape(format!(
                "matrix length {} does not match rows {rows} * cols {cols}",
                matrix.len()
            )));
        }
        let expected_vectors_len = vector_count.checked_mul(cols).ok_or_else(|| {
            MetalError::InvalidShape("batched vector shape overflows usize".to_owned())
        })?;
        if vectors.len() != expected_vectors_len {
            return Err(MetalError::InvalidShape(format!(
                "batched vector length {} does not match vector_count {vector_count} * cols {cols}",
                vectors.len()
            )));
        }
        let output_len = vector_count.checked_mul(rows).ok_or_else(|| {
            MetalError::InvalidShape("batched output shape overflows usize".to_owned())
        })?;
        if rows == 0 || vector_count == 0 {
            return Ok(Vec::new());
        }
        if cols == 0 {
            return Ok(vec![0.0; output_len]);
        }
        let rows_u32 = u32::try_from(rows).map_err(|err| {
            MetalError::InvalidShape(format!("row count does not fit u32: {err}"))
        })?;
        let cols_u32 = u32::try_from(cols).map_err(|err| {
            MetalError::InvalidShape(format!("column count does not fit u32: {err}"))
        })?;
        let vector_count_u32 = u32::try_from(vector_count).map_err(|err| {
            MetalError::InvalidShape(format!("vector count does not fit u32: {err}"))
        })?;
        let matrix_byte_len = std::mem::size_of_val(matrix) as u64;
        let vector_byte_len = std::mem::size_of_val(vectors) as u64;
        let output_byte_len = output_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| MetalError::InvalidShape("output byte length overflow".to_owned()))?
            as u64;
        let matrix_buffer = self.device.new_buffer_with_data(
            matrix.as_ptr().cast::<c_void>(),
            matrix_byte_len,
            MTLResourceOptions::StorageModeShared,
        );
        let vector_buffer = self.device.new_buffer_with_data(
            vectors.as_ptr().cast::<c_void>(),
            vector_byte_len,
            MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self
            .device
            .new_buffer(output_byte_len, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.batched_matvec_bf16_f32.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.batched_matvec_bf16_f32.pipeline);
        encoder.set_buffer(0, Some(&matrix_buffer), 0);
        encoder.set_buffer(1, Some(&vector_buffer), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of_val(&rows_u32) as u64,
            (&rows_u32 as *const u32).cast::<c_void>(),
        );
        encoder.set_bytes(
            3,
            std::mem::size_of_val(&cols_u32) as u64,
            (&cols_u32 as *const u32).cast::<c_void>(),
        );
        encoder.set_bytes(
            4,
            std::mem::size_of_val(&vector_count_u32) as u64,
            (&vector_count_u32 as *const u32).cast::<c_void>(),
        );
        encoder.set_buffer(5, Some(&output_buffer), 0);
        let threads = MTLSize {
            width: rows as u64,
            height: vector_count as u64,
            depth: 1,
        };
        let group_width = self
            .batched_matvec_bf16_f32
            .pipeline
            .thread_execution_width()
            .min(rows as u64);
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
        // containing vector_count * rows f32 values in input-major order.
        let values = unsafe {
            let ptr = output_buffer.contents().cast::<f32>();
            std::slice::from_raw_parts(ptr, output_len).to_vec()
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

kernel void matvec_f32(
    device const float* matrix [[buffer(0)]],
    device const float* vector [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    device float* output [[buffer(4)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) {
        return;
    }
    float sum = 0.0;
    uint row_offset = row * cols;
    for (uint col = 0; col < cols; col++) {
        sum += matrix[row_offset + col] * vector[col];
    }
    output[row] = sum;
}

kernel void matvec_bf16_f32(
    device const ushort* matrix [[buffer(0)]],
    device const float* vector [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    device float* output [[buffer(4)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) {
        return;
    }
    float sum = 0.0;
    uint row_offset = row * cols;
    for (uint col = 0; col < cols; col++) {
        uint bits = uint(matrix[row_offset + col]) << 16;
        float weight = as_type<float>(bits);
        sum += weight * vector[col];
    }
    output[row] = sum;
}

kernel void batched_matvec_bf16_f32(
    device const ushort* matrix [[buffer(0)]],
    device const float* vectors [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& vector_count [[buffer(4)]],
    device float* output [[buffer(5)]],
    uint2 id [[thread_position_in_grid]]
) {
    uint row = id.x;
    uint vector_index = id.y;
    if (row >= rows || vector_index >= vector_count) {
        return;
    }
    float sum = 0.0;
    uint row_offset = row * cols;
    uint vector_offset = vector_index * cols;
    for (uint col = 0; col < cols; col++) {
        uint bits = uint(matrix[row_offset + col]) << 16;
        float weight = as_type<float>(bits);
        sum += weight * vectors[vector_offset + col];
    }
    output[(vector_index * rows) + row] = sum;
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
