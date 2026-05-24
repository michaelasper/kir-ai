use super::command::finish_command_buffer_async;
use super::{
    F16Buffer, F32Buffer, I8Buffer, MetalDevice, MetalError, metal_buffer_byte_len,
    power_of_two_at_most,
};
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
        let max_threads = self
            .softmax_f32
            .pipeline
            .max_total_threads_per_threadgroup()
            .max(1);
        let group_width = power_of_two_at_most((scores.len() as u64).min(max_threads));
        let thread_count = u32::try_from(group_width).map_err(|err| {
            MetalError::InvalidShape(format!("softmax threadgroup width does not fit u32: {err}"))
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
        encoder.set_bytes(
            3,
            std::mem::size_of_val(&thread_count) as u64,
            (&thread_count as *const u32).cast::<c_void>(),
        );
        encoder.set_threadgroup_memory_length(0, group_width * std::mem::size_of::<f32>() as u64);
        let threadgroups = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let threads_per_group = MTLSize {
            width: group_width,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups, threads_per_group);
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
        let output_byte_len = metal_buffer_byte_len::<f32>(vector_len, "weighted sum output")?;
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

    #[allow(clippy::too_many_arguments)]
    pub async fn full_attention_cache_mix_f32_buffered(
        &self,
        keys: &F32Buffer,
        values: &F32Buffer,
        query: &[f32],
        row_count: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        score_scale: f32,
        output: &mut [f32],
    ) -> Result<(), MetalError> {
        if num_attention_heads == 0 || num_key_value_heads == 0 || head_dim == 0 {
            return Err(MetalError::InvalidShape(
                "full attention dimensions must be non-zero".to_owned(),
            ));
        }
        if !num_attention_heads.is_multiple_of(num_key_value_heads) {
            return Err(MetalError::InvalidShape(
                "attention heads must be divisible by key/value heads".to_owned(),
            ));
        }
        let attention_dim = num_attention_heads.checked_mul(head_dim).ok_or_else(|| {
            MetalError::InvalidShape("attention output dimension overflows usize".to_owned())
        })?;
        let kv_vector_len = num_key_value_heads.checked_mul(head_dim).ok_or_else(|| {
            MetalError::InvalidShape("attention KV vector dimension overflows usize".to_owned())
        })?;
        let used_kv_len = row_count.checked_mul(kv_vector_len).ok_or_else(|| {
            MetalError::InvalidShape("attention KV cache shape overflows usize".to_owned())
        })?;
        let score_len = num_attention_heads.checked_mul(row_count).ok_or_else(|| {
            MetalError::InvalidShape("attention score shape overflows usize".to_owned())
        })?;
        if query.len() != attention_dim {
            return Err(MetalError::InvalidShape(format!(
                "query length {} does not match attention dimension {attention_dim}",
                query.len()
            )));
        }
        if output.len() < attention_dim {
            return Err(MetalError::InvalidShape(format!(
                "output length {} is smaller than attention dimension {attention_dim}",
                output.len()
            )));
        }
        if row_count == 0 {
            output[..attention_dim].fill(0.0);
            return Ok(());
        }
        if keys.len < used_kv_len || values.len < used_kv_len {
            return Err(MetalError::InvalidShape(format!(
                "KV cache buffers are shorter than row_count {row_count} * vector_len {kv_vector_len}"
            )));
        }
        let Some(keys_buffer) = keys.buffer.as_ref() else {
            return Err(MetalError::InvalidShape(
                "non-empty attention requires a key cache buffer".to_owned(),
            ));
        };
        let Some(values_buffer) = values.buffer.as_ref() else {
            return Err(MetalError::InvalidShape(
                "non-empty attention requires a value cache buffer".to_owned(),
            ));
        };
        let row_count_u32 = u32::try_from(row_count).map_err(|err| {
            MetalError::InvalidShape(format!("attention row count does not fit u32: {err}"))
        })?;
        let num_attention_heads_u32 = u32::try_from(num_attention_heads).map_err(|err| {
            MetalError::InvalidShape(format!("attention head count does not fit u32: {err}"))
        })?;
        let num_key_value_heads_u32 = u32::try_from(num_key_value_heads).map_err(|err| {
            MetalError::InvalidShape(format!("KV head count does not fit u32: {err}"))
        })?;
        let head_dim_u32 = u32::try_from(head_dim).map_err(|err| {
            MetalError::InvalidShape(format!("attention head dimension does not fit u32: {err}"))
        })?;
        let groups = num_attention_heads / num_key_value_heads;
        let groups_u32 = u32::try_from(groups).map_err(|err| {
            MetalError::InvalidShape(format!("attention group count does not fit u32: {err}"))
        })?;
        let query_buffer = self.new_f32_buffer(query)?;
        let Some(query_buffer) = query_buffer.buffer.as_ref() else {
            return Err(MetalError::InvalidShape(
                "non-empty attention requires a query buffer".to_owned(),
            ));
        };
        let score_byte_len = metal_buffer_byte_len::<f32>(score_len, "attention score")?;
        let output_byte_len = metal_buffer_byte_len::<f32>(attention_dim, "attention output")?;
        let scores_buffer = self
            .device
            .new_buffer(score_byte_len, MTLResourceOptions::StorageModeShared);
        let weights_buffer = self
            .device
            .new_buffer(score_byte_len, MTLResourceOptions::StorageModeShared);
        let output_buffer = self
            .device
            .new_buffer(output_byte_len, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.attention_scores_f32.queue.new_command_buffer();
        let score_encoder = command_buffer.new_compute_command_encoder();
        score_encoder.set_compute_pipeline_state(&self.attention_scores_f32.pipeline);
        score_encoder.set_buffer(0, Some(query_buffer), 0);
        score_encoder.set_buffer(1, Some(keys_buffer), 0);
        score_encoder.set_bytes(
            2,
            std::mem::size_of_val(&row_count_u32) as u64,
            (&row_count_u32 as *const u32).cast::<c_void>(),
        );
        score_encoder.set_bytes(
            3,
            std::mem::size_of_val(&num_attention_heads_u32) as u64,
            (&num_attention_heads_u32 as *const u32).cast::<c_void>(),
        );
        score_encoder.set_bytes(
            4,
            std::mem::size_of_val(&num_key_value_heads_u32) as u64,
            (&num_key_value_heads_u32 as *const u32).cast::<c_void>(),
        );
        score_encoder.set_bytes(
            5,
            std::mem::size_of_val(&head_dim_u32) as u64,
            (&head_dim_u32 as *const u32).cast::<c_void>(),
        );
        score_encoder.set_bytes(
            6,
            std::mem::size_of_val(&groups_u32) as u64,
            (&groups_u32 as *const u32).cast::<c_void>(),
        );
        score_encoder.set_bytes(
            7,
            std::mem::size_of_val(&score_scale) as u64,
            (&score_scale as *const f32).cast::<c_void>(),
        );
        score_encoder.set_buffer(8, Some(&scores_buffer), 0);
        let score_threads = MTLSize {
            width: row_count as u64,
            height: num_attention_heads as u64,
            depth: 1,
        };
        let score_group_width = self
            .attention_scores_f32
            .pipeline
            .thread_execution_width()
            .min(row_count as u64);
        let score_threads_per_group = MTLSize {
            width: score_group_width,
            height: 1,
            depth: 1,
        };
        score_encoder.dispatch_threads(score_threads, score_threads_per_group);
        score_encoder.end_encoding();

        let max_threads = self
            .softmax_rows_f32
            .pipeline
            .max_total_threads_per_threadgroup()
            .max(1);
        let softmax_group_width = power_of_two_at_most((row_count as u64).min(max_threads));
        let softmax_thread_count = u32::try_from(softmax_group_width).map_err(|err| {
            MetalError::InvalidShape(format!(
                "attention softmax threadgroup width does not fit u32: {err}"
            ))
        })?;
        let softmax_encoder = command_buffer.new_compute_command_encoder();
        softmax_encoder.set_compute_pipeline_state(&self.softmax_rows_f32.pipeline);
        softmax_encoder.set_buffer(0, Some(&scores_buffer), 0);
        softmax_encoder.set_bytes(
            1,
            std::mem::size_of_val(&row_count_u32) as u64,
            (&row_count_u32 as *const u32).cast::<c_void>(),
        );
        softmax_encoder.set_buffer(2, Some(&weights_buffer), 0);
        softmax_encoder.set_bytes(
            3,
            std::mem::size_of_val(&softmax_thread_count) as u64,
            (&softmax_thread_count as *const u32).cast::<c_void>(),
        );
        softmax_encoder.set_threadgroup_memory_length(
            0,
            softmax_group_width * std::mem::size_of::<f32>() as u64,
        );
        softmax_encoder.dispatch_thread_groups(
            MTLSize {
                width: num_attention_heads as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: softmax_group_width,
                height: 1,
                depth: 1,
            },
        );
        softmax_encoder.end_encoding();

        let sum_encoder = command_buffer.new_compute_command_encoder();
        sum_encoder.set_compute_pipeline_state(&self.attention_weighted_sum_f32.pipeline);
        sum_encoder.set_buffer(0, Some(values_buffer), 0);
        sum_encoder.set_buffer(1, Some(&weights_buffer), 0);
        sum_encoder.set_bytes(
            2,
            std::mem::size_of_val(&row_count_u32) as u64,
            (&row_count_u32 as *const u32).cast::<c_void>(),
        );
        sum_encoder.set_bytes(
            3,
            std::mem::size_of_val(&num_attention_heads_u32) as u64,
            (&num_attention_heads_u32 as *const u32).cast::<c_void>(),
        );
        sum_encoder.set_bytes(
            4,
            std::mem::size_of_val(&num_key_value_heads_u32) as u64,
            (&num_key_value_heads_u32 as *const u32).cast::<c_void>(),
        );
        sum_encoder.set_bytes(
            5,
            std::mem::size_of_val(&head_dim_u32) as u64,
            (&head_dim_u32 as *const u32).cast::<c_void>(),
        );
        sum_encoder.set_bytes(
            6,
            std::mem::size_of_val(&groups_u32) as u64,
            (&groups_u32 as *const u32).cast::<c_void>(),
        );
        sum_encoder.set_buffer(7, Some(&output_buffer), 0);
        let sum_threads = MTLSize {
            width: head_dim as u64,
            height: num_attention_heads as u64,
            depth: 1,
        };
        let sum_group_width = self
            .attention_weighted_sum_f32
            .pipeline
            .thread_execution_width()
            .min(head_dim as u64);
        let sum_threads_per_group = MTLSize {
            width: sum_group_width,
            height: 1,
            depth: 1,
        };
        sum_encoder.dispatch_threads(sum_threads, sum_threads_per_group);
        sum_encoder.end_encoding();

        finish_command_buffer_async(
            &self.synchronization,
            command_buffer,
            "full_attention_cache_mix_f32",
        )
        .await?;

        // SAFETY: output_buffer is a completed StorageModeShared Metal buffer
        // containing one f32 for every attention output element.
        unsafe {
            let ptr = output_buffer.contents().cast::<f32>();
            let values = std::slice::from_raw_parts(ptr, attention_dim);
            output[..attention_dim].copy_from_slice(values);
        };
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn full_attention_cache_mix_f16_buffered(
        &self,
        keys: &F16Buffer,
        values: &F16Buffer,
        query: &[f32],
        row_count: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        score_scale: f32,
        output: &mut [f32],
    ) -> Result<(), MetalError> {
        self.full_attention_cache_mix_f16_buffered_at(
            keys,
            0,
            values,
            0,
            query,
            row_count,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            score_scale,
            output,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn full_attention_cache_mix_f16_buffered_at(
        &self,
        keys: &F16Buffer,
        key_element_offset: usize,
        values: &F16Buffer,
        value_element_offset: usize,
        query: &[f32],
        row_count: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        score_scale: f32,
        output: &mut [f32],
    ) -> Result<(), MetalError> {
        if num_attention_heads == 0 || num_key_value_heads == 0 || head_dim == 0 {
            return Err(MetalError::InvalidShape(
                "full attention dimensions must be non-zero".to_owned(),
            ));
        }
        if !num_attention_heads.is_multiple_of(num_key_value_heads) {
            return Err(MetalError::InvalidShape(
                "attention heads must be divisible by key/value heads".to_owned(),
            ));
        }
        let attention_dim = num_attention_heads.checked_mul(head_dim).ok_or_else(|| {
            MetalError::InvalidShape("attention output dimension overflows usize".to_owned())
        })?;
        let kv_vector_len = num_key_value_heads.checked_mul(head_dim).ok_or_else(|| {
            MetalError::InvalidShape("attention KV vector dimension overflows usize".to_owned())
        })?;
        let used_kv_len = row_count.checked_mul(kv_vector_len).ok_or_else(|| {
            MetalError::InvalidShape("attention KV cache shape overflows usize".to_owned())
        })?;
        let score_len = num_attention_heads.checked_mul(row_count).ok_or_else(|| {
            MetalError::InvalidShape("attention score shape overflows usize".to_owned())
        })?;
        if query.len() != attention_dim {
            return Err(MetalError::InvalidShape(format!(
                "query length {} does not match attention dimension {attention_dim}",
                query.len()
            )));
        }
        if output.len() < attention_dim {
            return Err(MetalError::InvalidShape(format!(
                "output length {} is smaller than attention dimension {attention_dim}",
                output.len()
            )));
        }
        if row_count == 0 {
            output[..attention_dim].fill(0.0);
            return Ok(());
        }
        let key_end = key_element_offset.checked_add(used_kv_len).ok_or_else(|| {
            MetalError::InvalidShape("attention key buffer offset overflows usize".to_owned())
        })?;
        let value_end = value_element_offset
            .checked_add(used_kv_len)
            .ok_or_else(|| {
                MetalError::InvalidShape("attention value buffer offset overflows usize".to_owned())
            })?;
        if keys.len < key_end || values.len < value_end {
            return Err(MetalError::InvalidShape(format!(
                "KV cache buffers are shorter than requested offsets plus row_count {row_count} * vector_len {kv_vector_len}"
            )));
        }
        let key_byte_offset =
            metal_buffer_byte_len::<u16>(key_element_offset, "attention key buffer offset")?;
        let value_byte_offset =
            metal_buffer_byte_len::<u16>(value_element_offset, "attention value buffer offset")?;
        let Some(keys_buffer) = keys.buffer.as_ref() else {
            return Err(MetalError::InvalidShape(
                "non-empty attention requires a key cache buffer".to_owned(),
            ));
        };
        let Some(values_buffer) = values.buffer.as_ref() else {
            return Err(MetalError::InvalidShape(
                "non-empty attention requires a value cache buffer".to_owned(),
            ));
        };
        let row_count_u32 = u32::try_from(row_count).map_err(|err| {
            MetalError::InvalidShape(format!("attention row count does not fit u32: {err}"))
        })?;
        let num_attention_heads_u32 = u32::try_from(num_attention_heads).map_err(|err| {
            MetalError::InvalidShape(format!("attention head count does not fit u32: {err}"))
        })?;
        let num_key_value_heads_u32 = u32::try_from(num_key_value_heads).map_err(|err| {
            MetalError::InvalidShape(format!("KV head count does not fit u32: {err}"))
        })?;
        let head_dim_u32 = u32::try_from(head_dim).map_err(|err| {
            MetalError::InvalidShape(format!("attention head dimension does not fit u32: {err}"))
        })?;
        let groups = num_attention_heads / num_key_value_heads;
        let groups_u32 = u32::try_from(groups).map_err(|err| {
            MetalError::InvalidShape(format!("attention group count does not fit u32: {err}"))
        })?;
        let query_buffer = self.new_f32_buffer(query)?;
        let Some(query_buffer) = query_buffer.buffer.as_ref() else {
            return Err(MetalError::InvalidShape(
                "non-empty attention requires a query buffer".to_owned(),
            ));
        };
        let score_byte_len = metal_buffer_byte_len::<f32>(score_len, "attention score")?;
        let output_byte_len = metal_buffer_byte_len::<f32>(attention_dim, "attention output")?;
        let scores_buffer = self
            .device
            .new_buffer(score_byte_len, MTLResourceOptions::StorageModeShared);
        let weights_buffer = self
            .device
            .new_buffer(score_byte_len, MTLResourceOptions::StorageModeShared);
        let output_buffer = self
            .device
            .new_buffer(output_byte_len, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.attention_scores_f16.queue.new_command_buffer();
        let score_encoder = command_buffer.new_compute_command_encoder();
        score_encoder.set_compute_pipeline_state(&self.attention_scores_f16.pipeline);
        score_encoder.set_buffer(0, Some(query_buffer), 0);
        score_encoder.set_buffer(1, Some(keys_buffer), key_byte_offset);
        score_encoder.set_bytes(
            2,
            std::mem::size_of_val(&row_count_u32) as u64,
            (&row_count_u32 as *const u32).cast::<c_void>(),
        );
        score_encoder.set_bytes(
            3,
            std::mem::size_of_val(&num_attention_heads_u32) as u64,
            (&num_attention_heads_u32 as *const u32).cast::<c_void>(),
        );
        score_encoder.set_bytes(
            4,
            std::mem::size_of_val(&num_key_value_heads_u32) as u64,
            (&num_key_value_heads_u32 as *const u32).cast::<c_void>(),
        );
        score_encoder.set_bytes(
            5,
            std::mem::size_of_val(&head_dim_u32) as u64,
            (&head_dim_u32 as *const u32).cast::<c_void>(),
        );
        score_encoder.set_bytes(
            6,
            std::mem::size_of_val(&groups_u32) as u64,
            (&groups_u32 as *const u32).cast::<c_void>(),
        );
        score_encoder.set_bytes(
            7,
            std::mem::size_of_val(&score_scale) as u64,
            (&score_scale as *const f32).cast::<c_void>(),
        );
        score_encoder.set_buffer(8, Some(&scores_buffer), 0);
        let score_threads = MTLSize {
            width: row_count as u64,
            height: num_attention_heads as u64,
            depth: 1,
        };
        let score_group_width = self
            .attention_scores_f16
            .pipeline
            .thread_execution_width()
            .min(row_count as u64);
        let score_threads_per_group = MTLSize {
            width: score_group_width,
            height: 1,
            depth: 1,
        };
        score_encoder.dispatch_threads(score_threads, score_threads_per_group);
        score_encoder.end_encoding();

        let max_threads = self
            .softmax_rows_f32
            .pipeline
            .max_total_threads_per_threadgroup()
            .max(1);
        let softmax_group_width = power_of_two_at_most((row_count as u64).min(max_threads));
        let softmax_thread_count = u32::try_from(softmax_group_width).map_err(|err| {
            MetalError::InvalidShape(format!(
                "attention softmax threadgroup width does not fit u32: {err}"
            ))
        })?;
        let softmax_encoder = command_buffer.new_compute_command_encoder();
        softmax_encoder.set_compute_pipeline_state(&self.softmax_rows_f32.pipeline);
        softmax_encoder.set_buffer(0, Some(&scores_buffer), 0);
        softmax_encoder.set_bytes(
            1,
            std::mem::size_of_val(&row_count_u32) as u64,
            (&row_count_u32 as *const u32).cast::<c_void>(),
        );
        softmax_encoder.set_buffer(2, Some(&weights_buffer), 0);
        softmax_encoder.set_bytes(
            3,
            std::mem::size_of_val(&softmax_thread_count) as u64,
            (&softmax_thread_count as *const u32).cast::<c_void>(),
        );
        softmax_encoder.set_threadgroup_memory_length(
            0,
            softmax_group_width * std::mem::size_of::<f32>() as u64,
        );
        softmax_encoder.dispatch_thread_groups(
            MTLSize {
                width: num_attention_heads as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: softmax_group_width,
                height: 1,
                depth: 1,
            },
        );
        softmax_encoder.end_encoding();

        let sum_encoder = command_buffer.new_compute_command_encoder();
        sum_encoder.set_compute_pipeline_state(&self.attention_weighted_sum_f16.pipeline);
        sum_encoder.set_buffer(0, Some(values_buffer), value_byte_offset);
        sum_encoder.set_buffer(1, Some(&weights_buffer), 0);
        sum_encoder.set_bytes(
            2,
            std::mem::size_of_val(&row_count_u32) as u64,
            (&row_count_u32 as *const u32).cast::<c_void>(),
        );
        sum_encoder.set_bytes(
            3,
            std::mem::size_of_val(&num_attention_heads_u32) as u64,
            (&num_attention_heads_u32 as *const u32).cast::<c_void>(),
        );
        sum_encoder.set_bytes(
            4,
            std::mem::size_of_val(&num_key_value_heads_u32) as u64,
            (&num_key_value_heads_u32 as *const u32).cast::<c_void>(),
        );
        sum_encoder.set_bytes(
            5,
            std::mem::size_of_val(&head_dim_u32) as u64,
            (&head_dim_u32 as *const u32).cast::<c_void>(),
        );
        sum_encoder.set_bytes(
            6,
            std::mem::size_of_val(&groups_u32) as u64,
            (&groups_u32 as *const u32).cast::<c_void>(),
        );
        sum_encoder.set_buffer(7, Some(&output_buffer), 0);
        let sum_threads = MTLSize {
            width: head_dim as u64,
            height: num_attention_heads as u64,
            depth: 1,
        };
        let sum_group_width = self
            .attention_weighted_sum_f16
            .pipeline
            .thread_execution_width()
            .min(head_dim as u64);
        let sum_threads_per_group = MTLSize {
            width: sum_group_width,
            height: 1,
            depth: 1,
        };
        sum_encoder.dispatch_threads(sum_threads, sum_threads_per_group);
        sum_encoder.end_encoding();

        finish_command_buffer_async(
            &self.synchronization,
            command_buffer,
            "full_attention_cache_mix_f16",
        )
        .await?;

        // SAFETY: output_buffer is a completed StorageModeShared Metal buffer
        // containing one f32 for every attention output element.
        unsafe {
            let ptr = output_buffer.contents().cast::<f32>();
            let values = std::slice::from_raw_parts(ptr, attention_dim);
            output[..attention_dim].copy_from_slice(values);
        };
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn full_attention_cache_mix_int8_buffered(
        &self,
        keys: &I8Buffer,
        key_scales: &F32Buffer,
        values: &I8Buffer,
        value_scales: &F32Buffer,
        query: &[f32],
        row_count: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        score_scale: f32,
        output: &mut [f32],
    ) -> Result<(), MetalError> {
        self.full_attention_cache_mix_int8_buffered_at(
            keys,
            0,
            key_scales,
            0,
            values,
            0,
            value_scales,
            0,
            query,
            row_count,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            score_scale,
            output,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn full_attention_cache_mix_int8_buffered_at(
        &self,
        keys: &I8Buffer,
        key_element_offset: usize,
        key_scales: &F32Buffer,
        key_scale_offset: usize,
        values: &I8Buffer,
        value_element_offset: usize,
        value_scales: &F32Buffer,
        value_scale_offset: usize,
        query: &[f32],
        row_count: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        score_scale: f32,
        output: &mut [f32],
    ) -> Result<(), MetalError> {
        if num_attention_heads == 0 || num_key_value_heads == 0 || head_dim == 0 {
            return Err(MetalError::InvalidShape(
                "full attention dimensions must be non-zero".to_owned(),
            ));
        }
        if !num_attention_heads.is_multiple_of(num_key_value_heads) {
            return Err(MetalError::InvalidShape(
                "attention heads must be divisible by key/value heads".to_owned(),
            ));
        }
        let attention_dim = num_attention_heads.checked_mul(head_dim).ok_or_else(|| {
            MetalError::InvalidShape("attention output dimension overflows usize".to_owned())
        })?;
        let kv_vector_len = num_key_value_heads.checked_mul(head_dim).ok_or_else(|| {
            MetalError::InvalidShape("attention KV vector dimension overflows usize".to_owned())
        })?;
        let used_kv_len = row_count.checked_mul(kv_vector_len).ok_or_else(|| {
            MetalError::InvalidShape("attention KV cache shape overflows usize".to_owned())
        })?;
        let score_len = num_attention_heads.checked_mul(row_count).ok_or_else(|| {
            MetalError::InvalidShape("attention score shape overflows usize".to_owned())
        })?;
        if query.len() != attention_dim {
            return Err(MetalError::InvalidShape(format!(
                "query length {} does not match attention dimension {attention_dim}",
                query.len()
            )));
        }
        if output.len() < attention_dim {
            return Err(MetalError::InvalidShape(format!(
                "output length {} is smaller than attention dimension {attention_dim}",
                output.len()
            )));
        }
        if row_count == 0 {
            output[..attention_dim].fill(0.0);
            return Ok(());
        }
        let key_end = key_element_offset.checked_add(used_kv_len).ok_or_else(|| {
            MetalError::InvalidShape("INT8 attention key buffer offset overflows usize".to_owned())
        })?;
        let value_end = value_element_offset
            .checked_add(used_kv_len)
            .ok_or_else(|| {
                MetalError::InvalidShape(
                    "INT8 attention value buffer offset overflows usize".to_owned(),
                )
            })?;
        let key_scale_end = key_scale_offset.checked_add(row_count).ok_or_else(|| {
            MetalError::InvalidShape("INT8 attention key scale offset overflows usize".to_owned())
        })?;
        let value_scale_end = value_scale_offset.checked_add(row_count).ok_or_else(|| {
            MetalError::InvalidShape("INT8 attention value scale offset overflows usize".to_owned())
        })?;
        if keys.len < key_end || values.len < value_end {
            return Err(MetalError::InvalidShape(format!(
                "INT8 KV cache buffers are shorter than requested offsets plus row_count {row_count} * vector_len {kv_vector_len}"
            )));
        }
        if key_scales.len < key_scale_end || value_scales.len < value_scale_end {
            return Err(MetalError::InvalidShape(format!(
                "INT8 KV scale buffers are shorter than requested offsets plus row_count {row_count}"
            )));
        }
        let key_byte_offset =
            metal_buffer_byte_len::<i8>(key_element_offset, "INT8 attention key buffer offset")?;
        let value_byte_offset = metal_buffer_byte_len::<i8>(
            value_element_offset,
            "INT8 attention value buffer offset",
        )?;
        let key_scale_byte_offset = metal_buffer_byte_len::<f32>(
            key_scale_offset,
            "INT8 attention key scale buffer offset",
        )?;
        let value_scale_byte_offset = metal_buffer_byte_len::<f32>(
            value_scale_offset,
            "INT8 attention value scale buffer offset",
        )?;
        let Some(keys_buffer) = keys.buffer.as_ref() else {
            return Err(MetalError::InvalidShape(
                "non-empty attention requires an INT8 key cache buffer".to_owned(),
            ));
        };
        let Some(key_scales_buffer) = key_scales.buffer.as_ref() else {
            return Err(MetalError::InvalidShape(
                "non-empty attention requires an INT8 key scale buffer".to_owned(),
            ));
        };
        let Some(values_buffer) = values.buffer.as_ref() else {
            return Err(MetalError::InvalidShape(
                "non-empty attention requires an INT8 value cache buffer".to_owned(),
            ));
        };
        let Some(value_scales_buffer) = value_scales.buffer.as_ref() else {
            return Err(MetalError::InvalidShape(
                "non-empty attention requires an INT8 value scale buffer".to_owned(),
            ));
        };
        let row_count_u32 = u32::try_from(row_count).map_err(|err| {
            MetalError::InvalidShape(format!("attention row count does not fit u32: {err}"))
        })?;
        let num_attention_heads_u32 = u32::try_from(num_attention_heads).map_err(|err| {
            MetalError::InvalidShape(format!("attention head count does not fit u32: {err}"))
        })?;
        let num_key_value_heads_u32 = u32::try_from(num_key_value_heads).map_err(|err| {
            MetalError::InvalidShape(format!("KV head count does not fit u32: {err}"))
        })?;
        let head_dim_u32 = u32::try_from(head_dim).map_err(|err| {
            MetalError::InvalidShape(format!("attention head dimension does not fit u32: {err}"))
        })?;
        let groups = num_attention_heads / num_key_value_heads;
        let groups_u32 = u32::try_from(groups).map_err(|err| {
            MetalError::InvalidShape(format!("attention group count does not fit u32: {err}"))
        })?;
        let query_buffer = self.new_f32_buffer(query)?;
        let Some(query_buffer) = query_buffer.buffer.as_ref() else {
            return Err(MetalError::InvalidShape(
                "non-empty attention requires a query buffer".to_owned(),
            ));
        };
        let score_byte_len = metal_buffer_byte_len::<f32>(score_len, "attention score")?;
        let output_byte_len = metal_buffer_byte_len::<f32>(attention_dim, "attention output")?;
        let scores_buffer = self
            .device
            .new_buffer(score_byte_len, MTLResourceOptions::StorageModeShared);
        let weights_buffer = self
            .device
            .new_buffer(score_byte_len, MTLResourceOptions::StorageModeShared);
        let output_buffer = self
            .device
            .new_buffer(output_byte_len, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.attention_scores_int8.queue.new_command_buffer();
        let score_encoder = command_buffer.new_compute_command_encoder();
        score_encoder.set_compute_pipeline_state(&self.attention_scores_int8.pipeline);
        score_encoder.set_buffer(0, Some(query_buffer), 0);
        score_encoder.set_buffer(1, Some(keys_buffer), key_byte_offset);
        score_encoder.set_buffer(2, Some(key_scales_buffer), key_scale_byte_offset);
        score_encoder.set_bytes(
            3,
            std::mem::size_of_val(&row_count_u32) as u64,
            (&row_count_u32 as *const u32).cast::<c_void>(),
        );
        score_encoder.set_bytes(
            4,
            std::mem::size_of_val(&num_attention_heads_u32) as u64,
            (&num_attention_heads_u32 as *const u32).cast::<c_void>(),
        );
        score_encoder.set_bytes(
            5,
            std::mem::size_of_val(&num_key_value_heads_u32) as u64,
            (&num_key_value_heads_u32 as *const u32).cast::<c_void>(),
        );
        score_encoder.set_bytes(
            6,
            std::mem::size_of_val(&head_dim_u32) as u64,
            (&head_dim_u32 as *const u32).cast::<c_void>(),
        );
        score_encoder.set_bytes(
            7,
            std::mem::size_of_val(&groups_u32) as u64,
            (&groups_u32 as *const u32).cast::<c_void>(),
        );
        score_encoder.set_bytes(
            8,
            std::mem::size_of_val(&score_scale) as u64,
            (&score_scale as *const f32).cast::<c_void>(),
        );
        score_encoder.set_buffer(9, Some(&scores_buffer), 0);
        let score_threads = MTLSize {
            width: row_count as u64,
            height: num_attention_heads as u64,
            depth: 1,
        };
        let score_group_width = self
            .attention_scores_int8
            .pipeline
            .thread_execution_width()
            .min(row_count as u64);
        let score_threads_per_group = MTLSize {
            width: score_group_width,
            height: 1,
            depth: 1,
        };
        score_encoder.dispatch_threads(score_threads, score_threads_per_group);
        score_encoder.end_encoding();

        let max_threads = self
            .softmax_rows_f32
            .pipeline
            .max_total_threads_per_threadgroup()
            .max(1);
        let softmax_group_width = power_of_two_at_most((row_count as u64).min(max_threads));
        let softmax_thread_count = u32::try_from(softmax_group_width).map_err(|err| {
            MetalError::InvalidShape(format!(
                "attention softmax threadgroup width does not fit u32: {err}"
            ))
        })?;
        let softmax_encoder = command_buffer.new_compute_command_encoder();
        softmax_encoder.set_compute_pipeline_state(&self.softmax_rows_f32.pipeline);
        softmax_encoder.set_buffer(0, Some(&scores_buffer), 0);
        softmax_encoder.set_bytes(
            1,
            std::mem::size_of_val(&row_count_u32) as u64,
            (&row_count_u32 as *const u32).cast::<c_void>(),
        );
        softmax_encoder.set_buffer(2, Some(&weights_buffer), 0);
        softmax_encoder.set_bytes(
            3,
            std::mem::size_of_val(&softmax_thread_count) as u64,
            (&softmax_thread_count as *const u32).cast::<c_void>(),
        );
        softmax_encoder.set_threadgroup_memory_length(
            0,
            softmax_group_width * std::mem::size_of::<f32>() as u64,
        );
        softmax_encoder.dispatch_thread_groups(
            MTLSize {
                width: num_attention_heads as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: softmax_group_width,
                height: 1,
                depth: 1,
            },
        );
        softmax_encoder.end_encoding();

        let sum_encoder = command_buffer.new_compute_command_encoder();
        sum_encoder.set_compute_pipeline_state(&self.attention_weighted_sum_int8.pipeline);
        sum_encoder.set_buffer(0, Some(values_buffer), value_byte_offset);
        sum_encoder.set_buffer(1, Some(value_scales_buffer), value_scale_byte_offset);
        sum_encoder.set_buffer(2, Some(&weights_buffer), 0);
        sum_encoder.set_bytes(
            3,
            std::mem::size_of_val(&row_count_u32) as u64,
            (&row_count_u32 as *const u32).cast::<c_void>(),
        );
        sum_encoder.set_bytes(
            4,
            std::mem::size_of_val(&num_attention_heads_u32) as u64,
            (&num_attention_heads_u32 as *const u32).cast::<c_void>(),
        );
        sum_encoder.set_bytes(
            5,
            std::mem::size_of_val(&num_key_value_heads_u32) as u64,
            (&num_key_value_heads_u32 as *const u32).cast::<c_void>(),
        );
        sum_encoder.set_bytes(
            6,
            std::mem::size_of_val(&head_dim_u32) as u64,
            (&head_dim_u32 as *const u32).cast::<c_void>(),
        );
        sum_encoder.set_bytes(
            7,
            std::mem::size_of_val(&groups_u32) as u64,
            (&groups_u32 as *const u32).cast::<c_void>(),
        );
        sum_encoder.set_buffer(8, Some(&output_buffer), 0);
        let sum_threads = MTLSize {
            width: head_dim as u64,
            height: num_attention_heads as u64,
            depth: 1,
        };
        let sum_group_width = self
            .attention_weighted_sum_int8
            .pipeline
            .thread_execution_width()
            .min(head_dim as u64);
        let sum_threads_per_group = MTLSize {
            width: sum_group_width,
            height: 1,
            depth: 1,
        };
        sum_encoder.dispatch_threads(sum_threads, sum_threads_per_group);
        sum_encoder.end_encoding();

        finish_command_buffer_async(
            &self.synchronization,
            command_buffer,
            "full_attention_cache_mix_int8",
        )
        .await?;

        // SAFETY: output_buffer is a completed StorageModeShared Metal buffer
        // containing one f32 for every attention output element.
        unsafe {
            let ptr = output_buffer.contents().cast::<f32>();
            let values = std::slice::from_raw_parts(ptr, attention_dim);
            output[..attention_dim].copy_from_slice(values);
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
        let output_byte_len =
            metal_buffer_byte_len::<f32>(output_len, "head row selection output")?;
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

    pub async fn select_head_rows_f16_buffered(
        &self,
        values: &F16Buffer,
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
        let output_byte_len =
            metal_buffer_byte_len::<f32>(output_len, "head row selection output")?;
        let Some(values_buffer) = values.buffer.as_ref() else {
            return Err(MetalError::InvalidShape(
                "non-empty head row selection requires a values buffer".to_owned(),
            ));
        };
        let output_buffer = self
            .device
            .new_buffer(output_byte_len, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.select_head_rows_f16.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.select_head_rows_f16.pipeline);
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
            .select_head_rows_f16
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
            "select_head_rows_f16",
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

    #[allow(clippy::too_many_arguments)]
    pub async fn select_head_rows_int8_buffered(
        &self,
        values: &I8Buffer,
        scales: &F32Buffer,
        row_count: usize,
        row_len: usize,
        head_start: usize,
        head_len: usize,
        output: &mut [f32],
    ) -> Result<(), MetalError> {
        let used_len = row_count.checked_mul(row_len).ok_or_else(|| {
            MetalError::InvalidShape("INT8 head row selection shape overflows usize".to_owned())
        })?;
        if values.len < used_len {
            return Err(MetalError::InvalidShape(format!(
                "INT8 head row selection value length {} is shorter than row_count {row_count} * row_len {row_len}",
                values.len
            )));
        }
        if scales.len < row_count {
            return Err(MetalError::InvalidShape(format!(
                "INT8 head row selection scale length {} is shorter than row_count {row_count}",
                scales.len
            )));
        }
        let head_end = head_start.checked_add(head_len).ok_or_else(|| {
            MetalError::InvalidShape("INT8 head row selection range overflows usize".to_owned())
        })?;
        if head_end > row_len {
            return Err(MetalError::InvalidShape(format!(
                "INT8 head row selection range {head_start}..{head_end} exceeds row length {row_len}"
            )));
        }
        let output_len = row_count.checked_mul(head_len).ok_or_else(|| {
            MetalError::InvalidShape(
                "INT8 head row selection output shape overflows usize".to_owned(),
            )
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
            MetalError::InvalidShape(format!("INT8 head row length does not fit u32: {err}"))
        })?;
        let head_start_u32 = u32::try_from(head_start).map_err(|err| {
            MetalError::InvalidShape(format!("INT8 head row start does not fit u32: {err}"))
        })?;
        let head_len_u32 = u32::try_from(head_len).map_err(|err| {
            MetalError::InvalidShape(format!("INT8 head row length does not fit u32: {err}"))
        })?;
        let output_len_u32 = u32::try_from(output_len).map_err(|err| {
            MetalError::InvalidShape(format!(
                "INT8 head row output length does not fit u32: {err}"
            ))
        })?;
        let output_byte_len =
            metal_buffer_byte_len::<f32>(output_len, "INT8 head row selection output")?;
        let Some(values_buffer) = values.buffer.as_ref() else {
            return Err(MetalError::InvalidShape(
                "non-empty INT8 head row selection requires a values buffer".to_owned(),
            ));
        };
        let Some(scales_buffer) = scales.buffer.as_ref() else {
            return Err(MetalError::InvalidShape(
                "non-empty INT8 head row selection requires a scale buffer".to_owned(),
            ));
        };
        let output_buffer = self
            .device
            .new_buffer(output_byte_len, MTLResourceOptions::StorageModeShared);

        let command_buffer = self.select_head_rows_int8.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.select_head_rows_int8.pipeline);
        encoder.set_buffer(0, Some(values_buffer), 0);
        encoder.set_buffer(1, Some(scales_buffer), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of_val(&row_len_u32) as u64,
            (&row_len_u32 as *const u32).cast::<c_void>(),
        );
        encoder.set_bytes(
            3,
            std::mem::size_of_val(&head_start_u32) as u64,
            (&head_start_u32 as *const u32).cast::<c_void>(),
        );
        encoder.set_bytes(
            4,
            std::mem::size_of_val(&head_len_u32) as u64,
            (&head_len_u32 as *const u32).cast::<c_void>(),
        );
        encoder.set_bytes(
            5,
            std::mem::size_of_val(&output_len_u32) as u64,
            (&output_len_u32 as *const u32).cast::<c_void>(),
        );
        encoder.set_buffer(6, Some(&output_buffer), 0);
        let threads = MTLSize {
            width: output_len as u64,
            height: 1,
            depth: 1,
        };
        let group_width = self
            .select_head_rows_int8
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
            "select_head_rows_int8",
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

#[cfg(test)]
#[path = "primitives/tests.rs"]
mod tests;
