use super::command::finish_command_buffer_async;
use super::{MetalDevice, MetalError, metal_buffer_byte_len};
use metal::MTLSize;
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
            return Err(MetalError::InvalidInput(
                "argmax input must not be empty".to_owned(),
            ));
        }
        if let Some((index, _)) = logits.iter().enumerate().find(|(_, value)| value.is_nan()) {
            return Err(MetalError::InvalidInput(format!(
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
        let chunk_indices_byte_len =
            metal_buffer_byte_len::<u32>(chunk_count, "argmax chunk indices")?;
        let chunk_values_byte_len =
            metal_buffer_byte_len::<f32>(chunk_count, "argmax chunk values")?;
        let logits_buffer = self.take_scratch_f32_buffer(logits);
        let chunk_indices_buffer = self.take_scratch_buffer(chunk_indices_byte_len);
        let chunk_values_buffer = self.take_scratch_buffer(chunk_values_byte_len);

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
        finish_command_buffer_async(&[], command_buffer, "argmax_f32").await?;

        // SAFETY: both output buffers are StorageModeShared buffers sized for
        // exactly chunk_count values. The command buffer has completed.
        let best = {
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
            best
        };
        self.return_scratch_buffer(logits_byte_len, logits_buffer);
        self.return_scratch_buffer(chunk_indices_byte_len, chunk_indices_buffer);
        self.return_scratch_buffer(chunk_values_byte_len, chunk_values_buffer);
        Ok(best)
    }

    pub async fn top_k_f32(
        &self,
        logits: &[f32],
        k: usize,
        output: &mut [TopKResult],
    ) -> Result<(), MetalError> {
        if k == 0 {
            return Ok(());
        }
        if logits.is_empty() {
            return Err(MetalError::InvalidInput(
                "top-k input must not be empty".to_owned(),
            ));
        }
        if k > MAX_METAL_TOP_K {
            return Err(MetalError::InvalidInput(format!(
                "top-k count {k} exceeds maximum {MAX_METAL_TOP_K}"
            )));
        }
        if output.len() < k {
            return Err(MetalError::InvalidShape(format!(
                "output buffer size {} is smaller than k {k}",
                output.len()
            )));
        }
        if let Some((index, _)) = logits.iter().enumerate().find(|(_, value)| value.is_nan()) {
            return Err(MetalError::InvalidInput(format!(
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
        let indices_byte_len =
            metal_buffer_byte_len::<u32>(candidate_count, "top-k candidate indices")?;
        let values_byte_len =
            metal_buffer_byte_len::<f32>(candidate_count, "top-k candidate values")?;
        let logits_buffer = self.take_scratch_f32_buffer(logits);
        let indices_buffer = self.take_scratch_buffer(indices_byte_len);
        let values_buffer = self.take_scratch_buffer(values_byte_len);

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
        finish_command_buffer_async(&[], command_buffer, "top_k_f32").await?;

        // SAFETY: both output buffers are StorageModeShared buffers sized for
        // exactly candidate_count values. The command buffer has completed.
        let mut candidates = {
            let (indices, values) = unsafe {
                (
                    std::slice::from_raw_parts(
                        indices_buffer.contents().cast::<u32>(),
                        candidate_count,
                    ),
                    std::slice::from_raw_parts(
                        values_buffer.contents().cast::<f32>(),
                        candidate_count,
                    ),
                )
            };
            indices
                .iter()
                .copied()
                .zip(values.iter().copied())
                .filter_map(|(index, value)| {
                    (index != u32::MAX).then_some(TopKResult {
                        index: index as usize,
                        value,
                    })
                })
                .collect::<Vec<_>>()
        };
        self.return_scratch_buffer(logits_byte_len, logits_buffer);
        self.return_scratch_buffer(indices_byte_len, indices_buffer);
        self.return_scratch_buffer(values_byte_len, values_buffer);
        candidates.sort_by(|left, right| {
            right
                .value
                .total_cmp(&left.value)
                .then_with(|| left.index.cmp(&right.index))
        });
        candidates.truncate(final_k);
        output[..final_k].copy_from_slice(&candidates);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::MetalDevice;

    #[tokio::test]
    async fn argmax_f32_reuses_transient_scratch_buffers() {
        let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
        else {
            eprintln!("no Metal device available; skipping smoke test");
            return;
        };
        let mut logits = vec![-1.0; 600];
        logits[42] = 4.5;
        logits[311] = 4.5;
        logits[599] = 3.25;

        let first = device
            .argmax_f32(&logits)
            .await
            .expect("first argmax succeeds");
        let second = device
            .argmax_f32(&logits)
            .await
            .expect("second argmax succeeds");

        assert_eq!(first.index, 42);
        assert_eq!(second.index, 42);
        assert_eq!(second.value, 4.5);
        assert_eq!(
            device.scratch_buffer_count_for_test(600 * std::mem::size_of::<f32>() as u64),
            1
        );
        assert_eq!(
            device.scratch_buffer_count_for_test(3 * std::mem::size_of::<u32>() as u64),
            2
        );
    }
}
