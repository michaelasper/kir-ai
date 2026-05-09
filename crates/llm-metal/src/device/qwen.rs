use super::command::finish_command_buffer_async;
use super::{F32Buffer, MetalDevice, MetalError};
use metal::{MTLResourceOptions, MTLSize};
use std::ffi::c_void;

impl MetalDevice {
    pub async fn qwen_rms_norm_f32(
        &self,
        input: &[f32],
        weight: &[f32],
        eps: f32,
        output: &mut [f32],
    ) -> Result<(), MetalError> {
        if input.len() != weight.len() {
            return Err(MetalError::InvalidShape(
                "input and weight must have the same length".to_owned(),
            ));
        }
        if output.len() < input.len() {
            return Err(MetalError::InvalidShape(format!(
                "output length {} is smaller than input length {}",
                output.len(),
                input.len()
            )));
        }
        if !eps.is_finite() || eps < 0.0 {
            return Err(MetalError::InvalidShape(
                "epsilon must be finite and non-negative".to_owned(),
            ));
        }
        if input.is_empty() {
            return Ok(());
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
        finish_command_buffer_async(command_buffer, "qwen_rms_norm").await?;

        // SAFETY: output_buffer is a completed StorageModeShared Metal buffer
        // with the same byte length as the input slice.
        unsafe {
            let ptr = output_buffer.contents().cast::<f32>();
            let values = std::slice::from_raw_parts(ptr, input.len());
            output[..input.len()].copy_from_slice(values);
        };
        Ok(())
    }

    pub async fn linear_attention_conv1d_silu_f32(
        &self,
        window: &[f32],
        weights: &[f32],
        conv_dim: usize,
        kernel_size: usize,
        output: &mut [f32],
    ) -> Result<(), MetalError> {
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
        if output.len() < conv_dim {
            return Err(MetalError::InvalidShape(format!(
                "output length {} is smaller than conv_dim {conv_dim}",
                output.len()
            )));
        }
        if conv_dim == 0 {
            return Ok(());
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
        finish_command_buffer_async(command_buffer, "linear_attention_conv1d_silu_f32").await?;

        // SAFETY: output_buffer is a completed StorageModeShared Metal buffer
        // containing one f32 per convolution channel.
        unsafe {
            let ptr = output_buffer.contents().cast::<f32>();
            let values = std::slice::from_raw_parts(ptr, conv_dim);
            output[..conv_dim].copy_from_slice(values);
        };
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn linear_attention_recurrent_update_f32(
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
    ) -> Result<(), MetalError> {
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
        if output.len() < element_count {
            return Err(MetalError::InvalidShape(format!(
                "output length {} is smaller than state element count {element_count}",
                output.len()
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
        finish_command_buffer_async(command_buffer, "linear_attention_recurrent_update_f32").await?;

        // SAFETY: output_buffer is a completed StorageModeShared Metal buffer
        // containing one f32 per recurrent-state element.
        unsafe {
            let ptr = output_buffer.contents().cast::<f32>();
            let values = std::slice::from_raw_parts(ptr, element_count);
            output[..element_count].copy_from_slice(values);
        };
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn linear_attention_recurrent_update_f32_buffered_state(
        &self,
        state: &F32Buffer,
        state_start: usize,
        key: &[f32],
        value: &[f32],
        memory: &[f32],
        beta: f32,
        decay: f32,
        key_head_dim: usize,
        value_head_dim: usize,
    ) -> Result<(), MetalError> {
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
        let state_end = state_start.checked_add(element_count).ok_or_else(|| {
            MetalError::InvalidShape(
                "linear attention recurrent state range overflows usize".to_owned(),
            )
        })?;
        if state_end > state.len {
            return Err(MetalError::InvalidShape(format!(
                "linear attention recurrent state range {state_start}..{state_end} exceeds state length {}",
                state.len
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
        if element_count == 0 {
            return Ok(());
        }
        let Some(state_buffer) = state.buffer.as_ref() else {
            return Err(MetalError::InvalidShape(
                "non-empty recurrent update requires a state buffer".to_owned(),
            ));
        };
        let state_start_u32 = u32::try_from(state_start).map_err(|err| {
            MetalError::InvalidShape(format!(
                "linear attention recurrent state start does not fit u32: {err}"
            ))
        })?;
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
        let key_byte_len = std::mem::size_of_val(key) as u64;
        let value_byte_len = std::mem::size_of_val(value) as u64;
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

        let command_buffer = self
            .linear_attention_recurrent_update_state_f32
            .queue
            .new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder
            .set_compute_pipeline_state(&self.linear_attention_recurrent_update_state_f32.pipeline);
        encoder.set_buffer(0, Some(state_buffer), 0);
        encoder.set_bytes(
            1,
            std::mem::size_of_val(&state_start_u32) as u64,
            (&state_start_u32 as *const u32).cast::<c_void>(),
        );
        encoder.set_buffer(2, Some(&key_buffer), 0);
        encoder.set_buffer(3, Some(&value_buffer), 0);
        encoder.set_buffer(4, Some(&memory_buffer), 0);
        encoder.set_bytes(
            5,
            std::mem::size_of_val(&beta) as u64,
            (&beta as *const f32).cast::<c_void>(),
        );
        encoder.set_bytes(
            6,
            std::mem::size_of_val(&decay) as u64,
            (&decay as *const f32).cast::<c_void>(),
        );
        encoder.set_bytes(
            7,
            std::mem::size_of_val(&value_head_dim_u32) as u64,
            (&value_head_dim_u32 as *const u32).cast::<c_void>(),
        );
        encoder.set_bytes(
            8,
            std::mem::size_of_val(&element_count_u32) as u64,
            (&element_count_u32 as *const u32).cast::<c_void>(),
        );
        let threads = MTLSize {
            width: element_count as u64,
            height: 1,
            depth: 1,
        };
        let group_width = self
            .linear_attention_recurrent_update_state_f32
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
        finish_command_buffer_async(
            command_buffer,
            "linear_attention_recurrent_update_state_f32",
        ).await?;
        Ok(())
    }
}
