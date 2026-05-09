use super::{MetalDevice, MetalError};
use metal::{Buffer, MTLResourceOptions};
use std::ffi::c_void;

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
        // SAFETY: metal_buffer was allocated with byte_len bytes for len f32
        // values. The caller provides exactly len f32 values above, and both
        // pointers remain valid for the duration of this copy.
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
        // SAFETY: the requested range is bounds-checked above against the f32
        // length used to allocate the StorageModeShared buffer.
        unsafe {
            let ptr = metal_buffer.contents().cast::<f32>().add(start);
            let values = std::slice::from_raw_parts(ptr, len);
            output[..len].copy_from_slice(values);
        };
        Ok(())
    }
}
