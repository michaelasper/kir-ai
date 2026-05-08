use metal::{
    Buffer, CommandBufferRef, CommandQueue, CompileOptions, ComputePipelineState, Device,
    MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};
use std::ffi::c_void;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct MetalDevice {
    device: Device,
    vector_add: Arc<MetalKernel>,
    qwen_rms_norm: Arc<MetalKernel>,
    softmax_f32: Arc<MetalKernel>,
    linear_attention_conv1d_silu_f32: Arc<MetalKernel>,
    weighted_sum_f32: Arc<MetalKernel>,
    linear_attention_recurrent_update_f32: Arc<MetalKernel>,
    select_head_rows_f32: Arc<MetalKernel>,
    matvec_f32: Arc<MetalKernel>,
    matvec_bf16_f32: Arc<MetalKernel>,
    batched_matvec_bf16_f32: Arc<MetalKernel>,
    argmax_f32: Arc<MetalKernel>,
    top_k_f32: Arc<MetalKernel>,
}

#[derive(Debug)]
struct MetalKernel {
    pipeline: ComputePipelineState,
    queue: CommandQueue,
}

fn finish_command_buffer(
    command_buffer: &CommandBufferRef,
    kernel_name: &str,
) -> Result<(), MetalError> {
    command_buffer.commit();
    command_buffer.wait_until_completed();
    command_buffer_status_result(command_buffer.status(), kernel_name)
}

fn command_buffer_status_result(
    status: MTLCommandBufferStatus,
    kernel_name: &str,
) -> Result<(), MetalError> {
    match status {
        MTLCommandBufferStatus::Completed => Ok(()),
        MTLCommandBufferStatus::Error => Err(MetalError::Execution(format!(
            "{kernel_name} command buffer failed with status {status:?}"
        ))),
        other => Err(MetalError::Execution(format!(
            "{kernel_name} command buffer finished with unexpected status {other:?}"
        ))),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArgmaxResult {
    pub index: usize,
    pub value: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopKResult {
    pub index: usize,
    pub value: f32,
}

#[derive(Debug, Clone)]
pub struct Bf16MatrixBuffer {
    buffer: Option<Buffer>,
    rows: usize,
    columns: usize,
    byte_len: usize,
}

impl Bf16MatrixBuffer {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn byte_len(&self) -> usize {
        self.byte_len
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
        finish_command_buffer(command_buffer, "vector_add")?;

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
        finish_command_buffer(command_buffer, "qwen_rms_norm")?;

        // SAFETY: output_buffer is a completed StorageModeShared Metal buffer
        // with the same byte length as the input slice.
        let values = unsafe {
            let ptr = output_buffer.contents().cast::<f32>();
            std::slice::from_raw_parts(ptr, input.len()).to_vec()
        };
        Ok(values)
    }

    pub fn softmax_f32(&self, scores: &[f32]) -> Result<Vec<f32>, MetalError> {
        if scores.is_empty() {
            return Ok(Vec::new());
        }
        if let Some((index, _)) = scores
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(MetalError::InvalidShape(format!(
                "softmax input contains non-finite value at index {index}"
            )));
        }
        let len_u32 = u32::try_from(scores.len()).map_err(|err| {
            MetalError::InvalidShape(format!("softmax input length does not fit u32: {err}"))
        })?;
        let byte_len = std::mem::size_of_val(scores) as u64;
        let scores_buffer = self.device.new_buffer_with_data(
            scores.as_ptr().cast::<c_void>(),
            byte_len,
            MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self
            .device
            .new_buffer(byte_len, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.softmax_f32.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.softmax_f32.pipeline);
        encoder.set_buffer(0, Some(&scores_buffer), 0);
        encoder.set_bytes(
            1,
            std::mem::size_of_val(&len_u32) as u64,
            (&len_u32 as *const u32).cast::<c_void>(),
        );
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        finish_command_buffer(command_buffer, "softmax_f32")?;

        // SAFETY: output_buffer is a completed StorageModeShared Metal buffer
        // with the same byte length as the input scores.
        let values = unsafe {
            let ptr = output_buffer.contents().cast::<f32>();
            std::slice::from_raw_parts(ptr, scores.len()).to_vec()
        };
        Ok(values)
    }

    pub fn linear_attention_conv1d_silu_f32(
        &self,
        window: &[f32],
        weights: &[f32],
        conv_dim: usize,
        kernel_size: usize,
    ) -> Result<Vec<f32>, MetalError> {
        if kernel_size == 0 {
            return Err(MetalError::InvalidShape(
                "linear attention conv kernel size must be non-zero".to_owned(),
            ));
        }
        let expected_len = conv_dim.checked_mul(kernel_size).ok_or_else(|| {
            MetalError::InvalidShape("linear attention conv shape overflows usize".to_owned())
        })?;
        if window.len() != expected_len {
            return Err(MetalError::InvalidShape(format!(
                "conv window length {} does not match conv_dim {conv_dim} * kernel_size {kernel_size}",
                window.len()
            )));
        }
        if weights.len() != expected_len {
            return Err(MetalError::InvalidShape(format!(
                "conv weight length {} does not match conv_dim {conv_dim} * kernel_size {kernel_size}",
                weights.len()
            )));
        }
        if conv_dim == 0 {
            return Ok(Vec::new());
        }
        let conv_dim_u32 = u32::try_from(conv_dim)
            .map_err(|err| MetalError::InvalidShape(format!("conv dim does not fit u32: {err}")))?;
        let kernel_size_u32 = u32::try_from(kernel_size).map_err(|err| {
            MetalError::InvalidShape(format!("kernel size does not fit u32: {err}"))
        })?;
        let input_byte_len = std::mem::size_of_val(window) as u64;
        let output_byte_len = (conv_dim * std::mem::size_of::<f32>()) as u64;
        let window_buffer = self.device.new_buffer_with_data(
            window.as_ptr().cast::<c_void>(),
            input_byte_len,
            MTLResourceOptions::StorageModeShared,
        );
        let weight_buffer = self.device.new_buffer_with_data(
            weights.as_ptr().cast::<c_void>(),
            input_byte_len,
            MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self
            .device
            .new_buffer(output_byte_len, MTLResourceOptions::StorageModeShared);

        let command_buffer = self
            .linear_attention_conv1d_silu_f32
            .queue
            .new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.linear_attention_conv1d_silu_f32.pipeline);
        encoder.set_buffer(0, Some(&window_buffer), 0);
        encoder.set_buffer(1, Some(&weight_buffer), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of_val(&conv_dim_u32) as u64,
            (&conv_dim_u32 as *const u32).cast::<c_void>(),
        );
        encoder.set_bytes(
            3,
            std::mem::size_of_val(&kernel_size_u32) as u64,
            (&kernel_size_u32 as *const u32).cast::<c_void>(),
        );
        encoder.set_buffer(4, Some(&output_buffer), 0);
        let threads = MTLSize {
            width: conv_dim as u64,
            height: 1,
            depth: 1,
        };
        let group_width = self
            .linear_attention_conv1d_silu_f32
            .pipeline
            .thread_execution_width()
            .min(conv_dim as u64);
        let threads_per_group = MTLSize {
            width: group_width,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads, threads_per_group);
        encoder.end_encoding();
        finish_command_buffer(command_buffer, "linear_attention_conv1d_silu_f32")?;

        // SAFETY: output_buffer is a completed StorageModeShared Metal buffer
        // containing one f32 per convolution channel.
        let values = unsafe {
            let ptr = output_buffer.contents().cast::<f32>();
            std::slice::from_raw_parts(ptr, conv_dim).to_vec()
        };
        Ok(values)
    }

    pub fn weighted_sum_f32(
        &self,
        values: &[f32],
        weights: &[f32],
        vector_len: usize,
    ) -> Result<Vec<f32>, MetalError> {
        let expected_values = weights.len().checked_mul(vector_len).ok_or_else(|| {
            MetalError::InvalidShape("weighted sum shape overflows usize".to_owned())
        })?;
        if values.len() != expected_values {
            return Err(MetalError::InvalidShape(format!(
                "weighted sum value length {} does not match weights {} * vector_len {vector_len}",
                values.len(),
                weights.len()
            )));
        }
        if vector_len == 0 {
            return Ok(Vec::new());
        }
        if weights.is_empty() {
            return Ok(vec![0.0; vector_len]);
        }
        let row_count_u32 = u32::try_from(weights.len()).map_err(|err| {
            MetalError::InvalidShape(format!("weighted sum row count does not fit u32: {err}"))
        })?;
        let vector_len_u32 = u32::try_from(vector_len).map_err(|err| {
            MetalError::InvalidShape(format!(
                "weighted sum vector length does not fit u32: {err}"
            ))
        })?;
        let values_byte_len = std::mem::size_of_val(values) as u64;
        let weights_byte_len = std::mem::size_of_val(weights) as u64;
        let output_byte_len = (vector_len * std::mem::size_of::<f32>()) as u64;
        let values_buffer = self.device.new_buffer_with_data(
            values.as_ptr().cast::<c_void>(),
            values_byte_len,
            MTLResourceOptions::StorageModeShared,
        );
        let weights_buffer = self.device.new_buffer_with_data(
            weights.as_ptr().cast::<c_void>(),
            weights_byte_len,
            MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self
            .device
            .new_buffer(output_byte_len, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.weighted_sum_f32.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.weighted_sum_f32.pipeline);
        encoder.set_buffer(0, Some(&values_buffer), 0);
        encoder.set_buffer(1, Some(&weights_buffer), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of_val(&row_count_u32) as u64,
            (&row_count_u32 as *const u32).cast::<c_void>(),
        );
        encoder.set_bytes(
            3,
            std::mem::size_of_val(&vector_len_u32) as u64,
            (&vector_len_u32 as *const u32).cast::<c_void>(),
        );
        encoder.set_buffer(4, Some(&output_buffer), 0);
        let threads = MTLSize {
            width: vector_len as u64,
            height: 1,
            depth: 1,
        };
        let group_width = self
            .weighted_sum_f32
            .pipeline
            .thread_execution_width()
            .min(vector_len as u64);
        let threads_per_group = MTLSize {
            width: group_width,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads, threads_per_group);
        encoder.end_encoding();
        finish_command_buffer(command_buffer, "weighted_sum_f32")?;

        // SAFETY: output_buffer is a completed StorageModeShared Metal buffer
        // containing one f32 per output column.
        let output = unsafe {
            let ptr = output_buffer.contents().cast::<f32>();
            std::slice::from_raw_parts(ptr, vector_len).to_vec()
        };
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn linear_attention_recurrent_update_f32(
        &self,
        state: &[f32],
        key: &[f32],
        value: &[f32],
        memory: &[f32],
        beta: f32,
        decay: f32,
        key_head_dim: usize,
        value_head_dim: usize,
    ) -> Result<Vec<f32>, MetalError> {
        if key_head_dim == 0 || value_head_dim == 0 {
            return Err(MetalError::InvalidShape(
                "linear attention recurrent update dimensions must be non-zero".to_owned(),
            ));
        }
        let element_count = key_head_dim.checked_mul(value_head_dim).ok_or_else(|| {
            MetalError::InvalidShape(
                "linear attention recurrent update shape overflows usize".to_owned(),
            )
        })?;
        if state.len() != element_count {
            return Err(MetalError::InvalidShape(format!(
                "linear attention recurrent state length {} does not match key_head_dim {key_head_dim} * value_head_dim {value_head_dim}",
                state.len()
            )));
        }
        if key.len() != key_head_dim {
            return Err(MetalError::InvalidShape(format!(
                "linear attention recurrent key length {} does not match key_head_dim {key_head_dim}",
                key.len()
            )));
        }
        if value.len() != value_head_dim {
            return Err(MetalError::InvalidShape(format!(
                "linear attention recurrent value length {} does not match value_head_dim {value_head_dim}",
                value.len()
            )));
        }
        if memory.len() != value_head_dim {
            return Err(MetalError::InvalidShape(format!(
                "linear attention recurrent memory length {} does not match value_head_dim {value_head_dim}",
                memory.len()
            )));
        }
        let value_head_dim_u32 = u32::try_from(value_head_dim).map_err(|err| {
            MetalError::InvalidShape(format!(
                "linear attention recurrent value head dimension does not fit u32: {err}"
            ))
        })?;
        let element_count_u32 = u32::try_from(element_count).map_err(|err| {
            MetalError::InvalidShape(format!(
                "linear attention recurrent element count does not fit u32: {err}"
            ))
        })?;
        let state_byte_len = std::mem::size_of_val(state) as u64;
        let key_byte_len = std::mem::size_of_val(key) as u64;
        let value_byte_len = std::mem::size_of_val(value) as u64;
        let state_buffer = self.device.new_buffer_with_data(
            state.as_ptr().cast::<c_void>(),
            state_byte_len,
            MTLResourceOptions::StorageModeShared,
        );
        let key_buffer = self.device.new_buffer_with_data(
            key.as_ptr().cast::<c_void>(),
            key_byte_len,
            MTLResourceOptions::StorageModeShared,
        );
        let value_buffer = self.device.new_buffer_with_data(
            value.as_ptr().cast::<c_void>(),
            value_byte_len,
            MTLResourceOptions::StorageModeShared,
        );
        let memory_buffer = self.device.new_buffer_with_data(
            memory.as_ptr().cast::<c_void>(),
            value_byte_len,
            MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self
            .device
            .new_buffer(state_byte_len, MTLResourceOptions::StorageModeShared);

        let command_buffer = self
            .linear_attention_recurrent_update_f32
            .queue
            .new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.linear_attention_recurrent_update_f32.pipeline);
        encoder.set_buffer(0, Some(&state_buffer), 0);
        encoder.set_buffer(1, Some(&key_buffer), 0);
        encoder.set_buffer(2, Some(&value_buffer), 0);
        encoder.set_buffer(3, Some(&memory_buffer), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of_val(&beta) as u64,
            (&beta as *const f32).cast::<c_void>(),
        );
        encoder.set_bytes(
            5,
            std::mem::size_of_val(&decay) as u64,
            (&decay as *const f32).cast::<c_void>(),
        );
        encoder.set_bytes(
            6,
            std::mem::size_of_val(&value_head_dim_u32) as u64,
            (&value_head_dim_u32 as *const u32).cast::<c_void>(),
        );
        encoder.set_bytes(
            7,
            std::mem::size_of_val(&element_count_u32) as u64,
            (&element_count_u32 as *const u32).cast::<c_void>(),
        );
        encoder.set_buffer(8, Some(&output_buffer), 0);
        let threads = MTLSize {
            width: element_count as u64,
            height: 1,
            depth: 1,
        };
        let group_width = self
            .linear_attention_recurrent_update_f32
            .pipeline
            .thread_execution_width()
            .min(element_count as u64);
        let threads_per_group = MTLSize {
            width: group_width,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads, threads_per_group);
        encoder.end_encoding();
        finish_command_buffer(command_buffer, "linear_attention_recurrent_update_f32")?;

        // SAFETY: output_buffer is a completed StorageModeShared Metal buffer
        // containing one f32 per recurrent-state element.
        let output = unsafe {
            let ptr = output_buffer.contents().cast::<f32>();
            std::slice::from_raw_parts(ptr, element_count).to_vec()
        };
        Ok(output)
    }

    pub fn select_head_rows_f32(
        &self,
        values: &[f32],
        row_count: usize,
        row_len: usize,
        head_start: usize,
        head_len: usize,
    ) -> Result<Vec<f32>, MetalError> {
        let used_len = row_count.checked_mul(row_len).ok_or_else(|| {
            MetalError::InvalidShape("head row selection shape overflows usize".to_owned())
        })?;
        if values.len() < used_len {
            return Err(MetalError::InvalidShape(format!(
                "head row selection value length {} is shorter than row_count {row_count} * row_len {row_len}",
                values.len()
            )));
        }
        let head_end = head_start.checked_add(head_len).ok_or_else(|| {
            MetalError::InvalidShape("head row selection range overflows usize".to_owned())
        })?;
        if head_end > row_len {
            return Err(MetalError::InvalidShape(format!(
                "head row selection range {head_start}..{head_end} exceeds row length {row_len}"
            )));
        }
        let output_len = row_count.checked_mul(head_len).ok_or_else(|| {
            MetalError::InvalidShape("head row selection output shape overflows usize".to_owned())
        })?;
        if output_len == 0 {
            return Ok(Vec::new());
        }
        let row_len_u32 = u32::try_from(row_len).map_err(|err| {
            MetalError::InvalidShape(format!("head row length does not fit u32: {err}"))
        })?;
        let head_start_u32 = u32::try_from(head_start).map_err(|err| {
            MetalError::InvalidShape(format!("head row start does not fit u32: {err}"))
        })?;
        let head_len_u32 = u32::try_from(head_len).map_err(|err| {
            MetalError::InvalidShape(format!("head row length does not fit u32: {err}"))
        })?;
        let output_len_u32 = u32::try_from(output_len).map_err(|err| {
            MetalError::InvalidShape(format!("head row output length does not fit u32: {err}"))
        })?;
        let values_byte_len = std::mem::size_of_val(values) as u64;
        let output_byte_len = (output_len * std::mem::size_of::<f32>()) as u64;
        let values_buffer = self.device.new_buffer_with_data(
            values.as_ptr().cast::<c_void>(),
            values_byte_len,
            MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self
            .device
            .new_buffer(output_byte_len, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.select_head_rows_f32.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.select_head_rows_f32.pipeline);
        encoder.set_buffer(0, Some(&values_buffer), 0);
        encoder.set_bytes(
            1,
            std::mem::size_of_val(&row_len_u32) as u64,
            (&row_len_u32 as *const u32).cast::<c_void>(),
        );
        encoder.set_bytes(
            2,
            std::mem::size_of_val(&head_start_u32) as u64,
            (&head_start_u32 as *const u32).cast::<c_void>(),
        );
        encoder.set_bytes(
            3,
            std::mem::size_of_val(&head_len_u32) as u64,
            (&head_len_u32 as *const u32).cast::<c_void>(),
        );
        encoder.set_bytes(
            4,
            std::mem::size_of_val(&output_len_u32) as u64,
            (&output_len_u32 as *const u32).cast::<c_void>(),
        );
        encoder.set_buffer(5, Some(&output_buffer), 0);
        let threads = MTLSize {
            width: output_len as u64,
            height: 1,
            depth: 1,
        };
        let group_width = self
            .select_head_rows_f32
            .pipeline
            .thread_execution_width()
            .min(output_len as u64);
        let threads_per_group = MTLSize {
            width: group_width,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads, threads_per_group);
        encoder.end_encoding();
        finish_command_buffer(command_buffer, "select_head_rows_f32")?;

        // SAFETY: output_buffer is a completed StorageModeShared Metal buffer
        // containing one f32 per selected row element.
        let output = unsafe {
            let ptr = output_buffer.contents().cast::<f32>();
            std::slice::from_raw_parts(ptr, output_len).to_vec()
        };
        Ok(output)
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
        finish_command_buffer(command_buffer, "matvec_f32")?;

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
        let matrix_buffer = self.new_bf16_matrix_buffer(matrix, rows, cols)?;
        self.matvec_bf16_f32_buffered(&matrix_buffer, vector)
    }

    pub fn new_bf16_matrix_buffer(
        &self,
        matrix: &[u16],
        rows: usize,
        cols: usize,
    ) -> Result<Bf16MatrixBuffer, MetalError> {
        let expected_matrix_len = rows
            .checked_mul(cols)
            .ok_or_else(|| MetalError::InvalidShape("matrix shape overflows usize".to_owned()))?;
        if matrix.len() != expected_matrix_len {
            return Err(MetalError::InvalidShape(format!(
                "matrix length {} does not match rows {rows} * cols {cols}",
                matrix.len()
            )));
        }
        let byte_len = std::mem::size_of_val(matrix);
        let buffer = if byte_len == 0 {
            None
        } else {
            Some(self.device.new_buffer_with_data(
                matrix.as_ptr().cast::<c_void>(),
                byte_len as u64,
                MTLResourceOptions::StorageModeShared,
            ))
        };
        Ok(Bf16MatrixBuffer {
            buffer,
            rows,
            columns: cols,
            byte_len,
        })
    }

    pub fn matvec_bf16_f32_buffered(
        &self,
        matrix: &Bf16MatrixBuffer,
        vector: &[f32],
    ) -> Result<Vec<f32>, MetalError> {
        let rows = matrix.rows;
        let cols = matrix.columns;
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
        let vector_byte_len = std::mem::size_of_val(vector) as u64;
        let output_byte_len = rows
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| MetalError::InvalidShape("output byte length overflow".to_owned()))?
            as u64;
        let Some(matrix_buffer) = matrix.buffer.as_ref() else {
            return Err(MetalError::InvalidShape(
                "non-empty BF16 matvec requires a matrix buffer".to_owned(),
            ));
        };
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
        encoder.set_buffer(0, Some(matrix_buffer), 0);
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
        finish_command_buffer(command_buffer, "matvec_bf16_f32")?;

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
        let matrix_buffer = self.new_bf16_matrix_buffer(matrix, rows, cols)?;
        self.batched_matvec_bf16_f32_buffered(&matrix_buffer, vectors, vector_count)
    }

    pub fn batched_matvec_bf16_f32_buffered(
        &self,
        matrix: &Bf16MatrixBuffer,
        vectors: &[f32],
        vector_count: usize,
    ) -> Result<Vec<f32>, MetalError> {
        let rows = matrix.rows;
        let cols = matrix.columns;
        let expected_matrix_len = rows
            .checked_mul(cols)
            .ok_or_else(|| MetalError::InvalidShape("matrix shape overflows usize".to_owned()))?;
        debug_assert_eq!(
            matrix.byte_len / std::mem::size_of::<u16>(),
            expected_matrix_len
        );
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
        let vector_byte_len = std::mem::size_of_val(vectors) as u64;
        let output_byte_len = output_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| MetalError::InvalidShape("output byte length overflow".to_owned()))?
            as u64;
        let Some(matrix_buffer) = matrix.buffer.as_ref() else {
            return Err(MetalError::InvalidShape(
                "non-empty batched BF16 matvec requires a matrix buffer".to_owned(),
            ));
        };
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
        encoder.set_buffer(0, Some(matrix_buffer), 0);
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
        finish_command_buffer(command_buffer, "batched_matvec_bf16_f32")?;

        // SAFETY: output_buffer is a completed StorageModeShared Metal buffer
        // containing vector_count * rows f32 values in input-major order.
        let values = unsafe {
            let ptr = output_buffer.contents().cast::<f32>();
            std::slice::from_raw_parts(ptr, output_len).to_vec()
        };
        Ok(values)
    }

    pub fn argmax_f32(&self, logits: &[f32]) -> Result<ArgmaxResult, MetalError> {
        if logits.is_empty() {
            return Err(MetalError::InvalidShape(
                "argmax input must not be empty".to_owned(),
            ));
        }
        if let Some((index, _)) = logits.iter().enumerate().find(|(_, value)| value.is_nan()) {
            return Err(MetalError::InvalidShape(format!(
                "argmax input contains NaN at index {index}"
            )));
        }
        let len_u32 = u32::try_from(logits.len()).map_err(|err| {
            MetalError::InvalidShape(format!("argmax input length does not fit u32: {err}"))
        })?;
        let chunk_size = 256_u32;
        let chunk_count = logits.len().div_ceil(chunk_size as usize);
        let chunk_count_u32 = u32::try_from(chunk_count).map_err(|err| {
            MetalError::InvalidShape(format!("argmax chunk count does not fit u32: {err}"))
        })?;
        let logits_byte_len = std::mem::size_of_val(logits) as u64;
        let chunk_indices_byte_len = (chunk_count * std::mem::size_of::<u32>()) as u64;
        let chunk_values_byte_len = (chunk_count * std::mem::size_of::<f32>()) as u64;
        let logits_buffer = self.device.new_buffer_with_data(
            logits.as_ptr().cast::<c_void>(),
            logits_byte_len,
            MTLResourceOptions::StorageModeShared,
        );
        let chunk_indices_buffer = self.device.new_buffer(
            chunk_indices_byte_len,
            MTLResourceOptions::StorageModeShared,
        );
        let chunk_values_buffer = self
            .device
            .new_buffer(chunk_values_byte_len, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.argmax_f32.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.argmax_f32.pipeline);
        encoder.set_buffer(0, Some(&logits_buffer), 0);
        encoder.set_bytes(
            1,
            std::mem::size_of_val(&len_u32) as u64,
            (&len_u32 as *const u32).cast::<c_void>(),
        );
        encoder.set_bytes(
            2,
            std::mem::size_of_val(&chunk_size) as u64,
            (&chunk_size as *const u32).cast::<c_void>(),
        );
        encoder.set_buffer(3, Some(&chunk_indices_buffer), 0);
        encoder.set_buffer(4, Some(&chunk_values_buffer), 0);
        let threads = MTLSize {
            width: chunk_count_u32 as u64,
            height: 1,
            depth: 1,
        };
        let group_width = self
            .argmax_f32
            .pipeline
            .thread_execution_width()
            .min(chunk_count_u32 as u64);
        let threads_per_group = MTLSize {
            width: group_width,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads, threads_per_group);
        encoder.end_encoding();
        finish_command_buffer(command_buffer, "argmax_f32")?;

        // SAFETY: both output buffers are StorageModeShared buffers sized for
        // exactly chunk_count values. The command buffer has completed.
        let (chunk_indices, chunk_values) = unsafe {
            let indices_ptr = chunk_indices_buffer.contents().cast::<u32>();
            let values_ptr = chunk_values_buffer.contents().cast::<f32>();
            (
                std::slice::from_raw_parts(indices_ptr, chunk_count),
                std::slice::from_raw_parts(values_ptr, chunk_count),
            )
        };
        let mut best = ArgmaxResult {
            index: chunk_indices[0] as usize,
            value: chunk_values[0],
        };
        for (&index, &value) in chunk_indices.iter().zip(chunk_values).skip(1) {
            let index = index as usize;
            if value > best.value || (value == best.value && index < best.index) {
                best = ArgmaxResult { index, value };
            }
        }
        Ok(best)
    }

    pub fn top_k_f32(&self, logits: &[f32], k: usize) -> Result<Vec<TopKResult>, MetalError> {
        if k == 0 {
            return Ok(Vec::new());
        }
        if logits.is_empty() {
            return Err(MetalError::InvalidShape(
                "top-k input must not be empty".to_owned(),
            ));
        }
        if k > MAX_METAL_TOP_K {
            return Err(MetalError::InvalidShape(format!(
                "top-k count {k} exceeds maximum {MAX_METAL_TOP_K}"
            )));
        }
        if let Some((index, _)) = logits.iter().enumerate().find(|(_, value)| value.is_nan()) {
            return Err(MetalError::InvalidShape(format!(
                "top-k input contains NaN at index {index}"
            )));
        }
        let final_k = k.min(logits.len());
        let len_u32 = u32::try_from(logits.len()).map_err(|err| {
            MetalError::InvalidShape(format!("top-k input length does not fit u32: {err}"))
        })?;
        let k_u32 = u32::try_from(k).map_err(|err| {
            MetalError::InvalidShape(format!("top-k count does not fit u32: {err}"))
        })?;
        let chunk_size = 256_u32;
        let chunk_count = logits.len().div_ceil(chunk_size as usize);
        let chunk_count_u32 = u32::try_from(chunk_count).map_err(|err| {
            MetalError::InvalidShape(format!("top-k chunk count does not fit u32: {err}"))
        })?;
        let candidate_count = chunk_count
            .checked_mul(k)
            .ok_or_else(|| MetalError::InvalidShape("top-k output shape overflows".to_owned()))?;
        let logits_byte_len = std::mem::size_of_val(logits) as u64;
        let indices_byte_len = (candidate_count * std::mem::size_of::<u32>()) as u64;
        let values_byte_len = (candidate_count * std::mem::size_of::<f32>()) as u64;
        let logits_buffer = self.device.new_buffer_with_data(
            logits.as_ptr().cast::<c_void>(),
            logits_byte_len,
            MTLResourceOptions::StorageModeShared,
        );
        let indices_buffer = self
            .device
            .new_buffer(indices_byte_len, MTLResourceOptions::StorageModeShared);
        let values_buffer = self
            .device
            .new_buffer(values_byte_len, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.top_k_f32.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.top_k_f32.pipeline);
        encoder.set_buffer(0, Some(&logits_buffer), 0);
        encoder.set_bytes(
            1,
            std::mem::size_of_val(&len_u32) as u64,
            (&len_u32 as *const u32).cast::<c_void>(),
        );
        encoder.set_bytes(
            2,
            std::mem::size_of_val(&chunk_size) as u64,
            (&chunk_size as *const u32).cast::<c_void>(),
        );
        encoder.set_bytes(
            3,
            std::mem::size_of_val(&k_u32) as u64,
            (&k_u32 as *const u32).cast::<c_void>(),
        );
        encoder.set_buffer(4, Some(&indices_buffer), 0);
        encoder.set_buffer(5, Some(&values_buffer), 0);
        let threads = MTLSize {
            width: chunk_count_u32 as u64,
            height: 1,
            depth: 1,
        };
        let group_width = self
            .top_k_f32
            .pipeline
            .thread_execution_width()
            .min(chunk_count_u32 as u64);
        let threads_per_group = MTLSize {
            width: group_width,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads, threads_per_group);
        encoder.end_encoding();
        finish_command_buffer(command_buffer, "top_k_f32")?;

        // SAFETY: both output buffers are StorageModeShared buffers sized for
        // exactly candidate_count values. The command buffer has completed.
        let mut candidates = unsafe {
            let indices_ptr = indices_buffer.contents().cast::<u32>();
            let values_ptr = values_buffer.contents().cast::<f32>();
            std::slice::from_raw_parts(indices_ptr, candidate_count)
                .iter()
                .copied()
                .zip(
                    std::slice::from_raw_parts(values_ptr, candidate_count)
                        .iter()
                        .copied(),
                )
                .filter_map(|(index, value)| {
                    (index != u32::MAX).then_some(TopKResult {
                        index: index as usize,
                        value,
                    })
                })
                .collect::<Vec<_>>()
        };
        candidates.sort_by(|left, right| {
            right
                .value
                .total_cmp(&left.value)
                .then_with(|| left.index.cmp(&right.index))
        });
        candidates.truncate(final_k);
        Ok(candidates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal::MTLCommandBufferStatus;

    #[test]
    fn command_buffer_status_result_accepts_completed_status() {
        assert!(
            command_buffer_status_result(MTLCommandBufferStatus::Completed, "matvec_f32").is_ok()
        );
    }

    #[test]
    fn command_buffer_status_result_rejects_error_status() {
        let err = command_buffer_status_result(MTLCommandBufferStatus::Error, "softmax_f32")
            .expect_err("error status should fail");

        assert!(matches!(err, MetalError::Execution(_)));
        assert!(err.to_string().contains("softmax_f32"));
    }

    #[test]
    fn command_buffer_status_result_rejects_unfinished_status() {
        let err = command_buffer_status_result(MTLCommandBufferStatus::Scheduled, "top_k_f32")
            .expect_err("unfinished status should fail");

        assert!(matches!(err, MetalError::Execution(_)));
        assert!(err.to_string().contains("unexpected status"));
    }
}

const MAX_METAL_TOP_K: usize = 64;

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

kernel void softmax_f32(
    device const float* scores [[buffer(0)]],
    constant uint& len [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id != 0 || len == 0) {
        return;
    }
    float max_score = scores[0];
    for (uint index = 1; index < len; index++) {
        max_score = max(max_score, scores[index]);
    }
    float denominator = 0.0;
    for (uint index = 0; index < len; index++) {
        denominator += exp(scores[index] - max_score);
    }
    for (uint index = 0; index < len; index++) {
        output[index] = exp(scores[index] - max_score) / denominator;
    }
}

kernel void linear_attention_conv1d_silu_f32(
    device const float* window [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    constant uint& conv_dim [[buffer(2)]],
    constant uint& kernel_size [[buffer(3)]],
    device float* output [[buffer(4)]],
    uint channel [[thread_position_in_grid]]
) {
    if (channel >= conv_dim) {
        return;
    }
    float mixed = 0.0;
    for (uint kernel_index = 0; kernel_index < kernel_size; kernel_index++) {
        mixed += window[(kernel_index * conv_dim) + channel]
            * weights[(channel * kernel_size) + kernel_index];
    }
    output[channel] = mixed / (1.0 + exp(-mixed));
}

kernel void weighted_sum_f32(
    device const float* values [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    constant uint& row_count [[buffer(2)]],
    constant uint& vector_len [[buffer(3)]],
    device float* output [[buffer(4)]],
    uint column [[thread_position_in_grid]]
) {
    if (column >= vector_len) {
        return;
    }
    float sum = 0.0;
    for (uint row = 0; row < row_count; row++) {
        sum += values[(row * vector_len) + column] * weights[row];
    }
    output[column] = sum;
}

kernel void linear_attention_recurrent_update_f32(
    device const float* state [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device const float* memory [[buffer(3)]],
    constant float& beta [[buffer(4)]],
    constant float& decay [[buffer(5)]],
    constant uint& value_head_dim [[buffer(6)]],
    constant uint& element_count [[buffer(7)]],
    device float* output [[buffer(8)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= element_count) {
        return;
    }
    uint key_index = index / value_head_dim;
    uint value_index = index % value_head_dim;
    float delta = (value[value_index] - memory[value_index]) * beta;
    output[index] = (state[index] * decay) + (key[key_index] * delta);
}

kernel void select_head_rows_f32(
    device const float* values [[buffer(0)]],
    constant uint& row_len [[buffer(1)]],
    constant uint& head_start [[buffer(2)]],
    constant uint& head_len [[buffer(3)]],
    constant uint& output_len [[buffer(4)]],
    device float* output [[buffer(5)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= output_len) {
        return;
    }
    uint row = index / head_len;
    uint offset = index % head_len;
    output[index] = values[(row * row_len) + head_start + offset];
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

kernel void argmax_f32(
    device const float* logits [[buffer(0)]],
    constant uint& len [[buffer(1)]],
    constant uint& chunk_size [[buffer(2)]],
    device uint* chunk_indices [[buffer(3)]],
    device float* chunk_values [[buffer(4)]],
    uint chunk [[thread_position_in_grid]]
) {
    uint start = chunk * chunk_size;
    if (start >= len) {
        return;
    }
    uint end = min(start + chunk_size, len);
    uint best_index = start;
    float best_value = logits[start];
    for (uint index = start + 1; index < end; index++) {
        float value = logits[index];
        if (value > best_value || (value == best_value && index < best_index)) {
            best_value = value;
            best_index = index;
        }
    }
    chunk_indices[chunk] = best_index;
    chunk_values[chunk] = best_value;
}

constant uint MAX_TOP_K = 64;
constant uint INVALID_TOP_K_INDEX = 0xffffffff;
constant float NEGATIVE_MAX_FLOAT = -3.4028234663852886e38f;

kernel void top_k_f32(
    device const float* logits [[buffer(0)]],
    constant uint& len [[buffer(1)]],
    constant uint& chunk_size [[buffer(2)]],
    constant uint& top_k [[buffer(3)]],
    device uint* output_indices [[buffer(4)]],
    device float* output_values [[buffer(5)]],
    uint chunk [[thread_position_in_grid]]
) {
    uint output_offset = chunk * top_k;
    for (uint rank = 0; rank < top_k; rank++) {
        output_indices[output_offset + rank] = INVALID_TOP_K_INDEX;
        output_values[output_offset + rank] = NEGATIVE_MAX_FLOAT;
    }

    uint start = chunk * chunk_size;
    if (start >= len || top_k == 0 || top_k > MAX_TOP_K) {
        return;
    }
    uint end = min(start + chunk_size, len);
    uint best_indices[MAX_TOP_K];
    float best_values[MAX_TOP_K];
    for (uint rank = 0; rank < top_k; rank++) {
        best_indices[rank] = INVALID_TOP_K_INDEX;
        best_values[rank] = NEGATIVE_MAX_FLOAT;
    }

    for (uint index = start; index < end; index++) {
        float value = logits[index];
        for (uint rank = 0; rank < top_k; rank++) {
            uint current_index = best_indices[rank];
            float current_value = best_values[rank];
            if (current_index == INVALID_TOP_K_INDEX ||
                value > current_value ||
                (value == current_value && index < current_index)) {
                for (uint shift = top_k - 1; shift > rank; shift--) {
                    best_indices[shift] = best_indices[shift - 1];
                    best_values[shift] = best_values[shift - 1];
                }
                best_indices[rank] = index;
                best_values[rank] = value;
                break;
            }
        }
    }

    for (uint rank = 0; rank < top_k; rank++) {
        output_indices[output_offset + rank] = best_indices[rank];
        output_values[output_offset + rank] = best_values[rank];
    }
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
