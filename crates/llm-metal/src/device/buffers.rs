use super::{MetalDevice, MetalError};
use metal::{Buffer, MTLResourceOptions};
use std::ffi::c_void;

#[derive(Debug, Default)]
pub(crate) struct MetalBufferPool {
    buffers: Vec<PooledMetalBuffer>,
}

#[derive(Debug)]
struct PooledMetalBuffer {
    byte_len: u64,
    buffer: Buffer,
}

impl MetalBufferPool {
    fn take(&mut self, byte_len: u64) -> Option<Buffer> {
        let position = self
            .buffers
            .iter()
            .position(|buffer| buffer.byte_len == byte_len)?;
        Some(self.buffers.swap_remove(position).buffer)
    }

    fn put(&mut self, byte_len: u64, buffer: Buffer) {
        self.buffers.push(PooledMetalBuffer { byte_len, buffer });
    }

    #[cfg(test)]
    fn count_for_len(&self, byte_len: u64) -> usize {
        self.buffers
            .iter()
            .filter(|buffer| buffer.byte_len == byte_len)
            .count()
    }
}

#[derive(Debug, Clone)]
pub struct Bf16MatrixBuffer {
    pub(crate) buffer: Option<Buffer>,
    pub(crate) rows: usize,
    pub(crate) columns: usize,
    pub(crate) byte_len: usize,
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

#[derive(Debug, Clone)]
pub struct F32Buffer {
    pub(crate) buffer: Option<Buffer>,
    pub(crate) len: usize,
    pub(crate) byte_len: usize,
}

impl F32Buffer {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn byte_len(&self) -> usize {
        self.byte_len
    }
}

impl MetalDevice {
    pub(crate) fn take_scratch_f32_buffer(&self, values: &[f32]) -> Buffer {
        let byte_len = std::mem::size_of_val(values) as u64;
        let buffer = self
            .scratch_buffers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take(byte_len)
            .unwrap_or_else(|| {
                self.device
                    .new_buffer(byte_len, MTLResourceOptions::StorageModeShared)
            });
        if !values.is_empty() {
            // SAFETY: buffer has exactly byte_len bytes, and byte_len was
            // computed from values.len() * size_of::<f32>().
            unsafe {
                std::ptr::copy_nonoverlapping(
                    values.as_ptr(),
                    buffer.contents().cast::<f32>(),
                    values.len(),
                );
            }
        }
        buffer
    }

    pub(crate) fn take_scratch_buffer(&self, byte_len: u64) -> Buffer {
        self.scratch_buffers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take(byte_len)
            .unwrap_or_else(|| {
                self.device
                    .new_buffer(byte_len, MTLResourceOptions::StorageModeShared)
            })
    }

    pub(crate) fn return_scratch_buffer(&self, byte_len: u64, buffer: Buffer) {
        self.scratch_buffers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .put(byte_len, buffer);
    }

    #[cfg(test)]
    pub fn scratch_buffer_count_for_test(&self, byte_len: u64) -> usize {
        self.scratch_buffers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .count_for_len(byte_len)
    }

    pub fn new_f32_buffer(&self, values: &[f32]) -> Result<F32Buffer, MetalError> {
        let byte_len = std::mem::size_of_val(values);
        let buffer = if byte_len == 0 {
            None
        } else {
            Some(self.device.new_buffer_with_data(
                values.as_ptr().cast::<c_void>(),
                byte_len as u64,
                MTLResourceOptions::StorageModeShared,
            ))
        };
        Ok(F32Buffer {
            buffer,
            len: values.len(),
            byte_len,
        })
    }

    pub fn write_f32_buffer(&self, buffer: &F32Buffer, values: &[f32]) -> Result<(), MetalError> {
        if values.len() != buffer.len {
            return Err(MetalError::InvalidShape(format!(
                "f32 buffer write length {} does not match buffer length {}",
                values.len(),
                buffer.len
            )));
        }
        if values.is_empty() {
            return Ok(());
        }
        let Some(metal_buffer) = buffer.buffer.as_ref() else {
            return Err(MetalError::InvalidShape(
                "non-empty f32 buffer write requires a Metal buffer".to_owned(),
            ));
        };
        let _cpu_access = self.synchronization.begin_cpu_access();
        // SAFETY: metal_buffer was allocated with byte_len bytes for len f32
        // values. The device synchronization guard above waits for in-flight
        // GPU commands and prevents new command submissions during this copy.
        // The caller provides exactly len f32 values above, and both pointers
        // remain valid for the duration of this copy.
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                metal_buffer.contents().cast::<f32>(),
                values.len(),
            );
        }
        Ok(())
    }

    pub fn read_f32_buffer(&self, buffer: &F32Buffer) -> Result<Vec<f32>, MetalError> {
        self.read_f32_buffer_range(buffer, 0, buffer.len)
    }

    pub fn read_f32_buffer_range(
        &self,
        buffer: &F32Buffer,
        start: usize,
        len: usize,
    ) -> Result<Vec<f32>, MetalError> {
        let mut output = vec![0.0; len];
        self.read_f32_buffer_range_in_place(buffer, start, len, &mut output)?;
        Ok(output)
    }

    pub fn read_f32_buffer_range_in_place(
        &self,
        buffer: &F32Buffer,
        start: usize,
        len: usize,
        output: &mut [f32],
    ) -> Result<(), MetalError> {
        let end = start.checked_add(len).ok_or_else(|| {
            MetalError::InvalidShape("f32 buffer read range overflows usize".to_owned())
        })?;
        if end > buffer.len {
            return Err(MetalError::InvalidShape(format!(
                "f32 buffer read range {start}..{end} exceeds buffer length {}",
                buffer.len
            )));
        }
        if output.len() < len {
            return Err(MetalError::InvalidShape(format!(
                "output length {} is smaller than requested read length {len}",
                output.len()
            )));
        }
        if len == 0 {
            return Ok(());
        }
        let Some(metal_buffer) = buffer.buffer.as_ref() else {
            return Err(MetalError::InvalidShape(
                "non-empty f32 buffer read requires a Metal buffer".to_owned(),
            ));
        };
        let _cpu_access = self.synchronization.begin_cpu_access();
        // SAFETY: the requested range is bounds-checked above against the f32
        // length used to allocate the StorageModeShared buffer. The device
        // synchronization guard above waits for in-flight GPU commands and
        // prevents new command submissions during this copy.
        unsafe {
            let ptr = metal_buffer.contents().cast::<f32>().add(start);
            let values = std::slice::from_raw_parts(ptr, len);
            output[..len].copy_from_slice(values);
        };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::MetalDevice;

    #[tokio::test]
    async fn matvec_f32_reuses_transient_scratch_buffers() {
        let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
        else {
            eprintln!("no Metal device available; skipping smoke test");
            return;
        };

        let matrix = [1.0, 2.0, 3.0, -1.0, 0.5, 4.0];
        let vector = [0.5, -1.0, 0.25];
        let mut output = vec![0.0; 2];

        device
            .matvec_f32(&matrix, 2, 3, &vector, &mut output)
            .await
            .expect("first metal matvec succeeds");
        device
            .matvec_f32(&matrix, 2, 3, &vector, &mut output)
            .await
            .expect("second metal matvec succeeds");

        assert_eq!(
            device.scratch_buffer_count_for_test(6 * std::mem::size_of::<f32>() as u64),
            1
        );
        assert_eq!(
            device.scratch_buffer_count_for_test(3 * std::mem::size_of::<f32>() as u64),
            1
        );
        assert_eq!(
            device.scratch_buffer_count_for_test(2 * std::mem::size_of::<f32>() as u64),
            1
        );
        assert!((output[0] + 0.75).abs() < 1e-6);
        assert!(output[1].abs() < 1e-6);
    }
}
