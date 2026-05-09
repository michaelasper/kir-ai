use super::command::finish_command_buffer_async;
use super::{MetalDevice, MetalError};
use metal::{MTLResourceOptions, MTLSize};
use std::ffi::c_void;

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

const MAX_METAL_TOP_K: usize = 64;

impl MetalDevice {
    pub async fn argmax_f32(&self, logits: &[f32]) -> Result<ArgmaxResult, MetalError> {
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
        finish_command_buffer_async(command_buffer, "argmax_f32").await?;

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

    pub async fn top_k_f32(&self, logits: &[f32], k: usize) -> Result<Vec<TopKResult>, MetalError> {
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
        finish_command_buffer_async(command_buffer, "top_k_f32").await?;

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
