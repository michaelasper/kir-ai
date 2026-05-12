use super::command::finish_command_buffer_async;
use super::{F32Buffer, MetalDevice, MetalError};
use metal::{MTLResourceOptions, MTLSize};
use std::ffi::c_void;

impl MetalDevice {
    pub async fn add_f32(
        &self,
        left: &[f32],
        right: &[f32],
        output: &mut [f32],
    ) -> Result<(), MetalError> {
        if left.len() != right.len() {
            return Err(MetalError::InvalidShape(
                "left and right inputs must have the same length".to_owned(),
            ));
        }
        if output.len() < left.len() {
            return Err(MetalError::InvalidShape(format!(
                "output length {} is smaller than input length {}",
                output.len(),
                left.len()
            )));
        }
        if left.is_empty() {
            return Ok(());
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
        finish_command_buffer_async(&self.synchronization, command_buffer, "vector_add").await?;

        // SAFETY: output_buffer is a StorageModeShared Metal buffer allocated with
        // byte_len bytes above. The command buffer has completed, and the buffer
        // remains alive for the duration of this read. The pointer is interpreted
        // as f32 values matching the byte length used to allocate it.
        unsafe {
            let ptr = output_buffer.contents().cast::<f32>();
            let values = std::slice::from_raw_parts(ptr, left.len());
            output[..left.len()].copy_from_slice(values);
        };
        Ok(())
    }

    pub async fn softmax_f32(&self, scores: &[f32], output: &mut [f32]) -> Result<(), MetalError> {
        if scores.is_empty() {
            return Ok(());
        }
        if output.len() < scores.len() {
            return Err(MetalError::InvalidShape(format!(
                "output length {} is smaller than scores length {}",
                output.len(),
                scores.len()
            )));
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
        finish_command_buffer_async(&self.synchronization, command_buffer, "softmax_f32").await?;

        // SAFETY: output_buffer is a completed StorageModeShared Metal buffer
        // with the same byte length as the input scores.
        unsafe {
            let ptr = output_buffer.contents().cast::<f32>();
            let values = std::slice::from_raw_parts(ptr, scores.len());
            output[..scores.len()].copy_from_slice(values);
        };
        Ok(())
    }

    pub async fn weighted_sum_f32(
        &self,
        values: &[f32],
        weights: &[f32],
        vector_len: usize,
        output: &mut [f32],
    ) -> Result<(), MetalError> {
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
        if output.len() < vector_len {
            return Err(MetalError::InvalidShape(format!(
                "output length {} is smaller than vector length {vector_len}",
                output.len()
            )));
        }
        if vector_len == 0 {
            return Ok(());
        }
        if weights.is_empty() {
            output[..vector_len].fill(0.0);
            return Ok(());
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
        finish_command_buffer_async(&self.synchronization, command_buffer, "weighted_sum_f32")
            .await?;

        // SAFETY: output_buffer is a completed StorageModeShared Metal buffer
        // containing one f32 per output column.
        unsafe {
            let ptr = output_buffer.contents().cast::<f32>();
            let values = std::slice::from_raw_parts(ptr, vector_len);
            output[..vector_len].copy_from_slice(values);
        };
        Ok(())
    }

    pub async fn select_head_rows_f32(
        &self,
        values: &[f32],
        row_count: usize,
        row_len: usize,
        head_start: usize,
        head_len: usize,
        output: &mut [f32],
    ) -> Result<(), MetalError> {
        let values_buffer = self.new_f32_buffer(values)?;
        self.select_head_rows_f32_buffered(
            &values_buffer,
            row_count,
            row_len,
            head_start,
            head_len,
            output,
        )
        .await
    }

    pub async fn select_head_rows_f32_buffered(
        &self,
        values: &F32Buffer,
        row_count: usize,
        row_len: usize,
        head_start: usize,
        head_len: usize,
        output: &mut [f32],
    ) -> Result<(), MetalError> {
        let used_len = row_count.checked_mul(row_len).ok_or_else(|| {
            MetalError::InvalidShape("head row selection shape overflows usize".to_owned())
        })?;
        if values.len < used_len {
            return Err(MetalError::InvalidShape(format!(
                "head row selection value length {} is shorter than row_count {row_count} * row_len {row_len}",
                values.len
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
        if output.len() < output_len {
            return Err(MetalError::InvalidShape(format!(
                "output length {} is smaller than expected {output_len}",
                output.len()
            )));
        }
        if output_len == 0 {
            return Ok(());
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
        let output_byte_len = (output_len * std::mem::size_of::<f32>()) as u64;
        let Some(values_buffer) = values.buffer.as_ref() else {
            return Err(MetalError::InvalidShape(
                "non-empty head row selection requires a values buffer".to_owned(),
            ));
        };
        let output_buffer = self
            .device
            .new_buffer(output_byte_len, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.select_head_rows_f32.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.select_head_rows_f32.pipeline);
        encoder.set_buffer(0, Some(values_buffer), 0);
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
        finish_command_buffer_async(
            &self.synchronization,
            command_buffer,
            "select_head_rows_f32",
        )
        .await?;

        // SAFETY: output_buffer is a completed StorageModeShared Metal buffer
        // containing one f32 per selected row element.
        unsafe {
            let ptr = output_buffer.contents().cast::<f32>();
            let values = std::slice::from_raw_parts(ptr, output_len);
            output[..output_len].copy_from_slice(values);
        };
        Ok(())
    }
}
