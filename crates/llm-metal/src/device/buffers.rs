use super::{
    MetalDevice, MetalError,
    command::{MetalBufferHazard, finish_command_buffer_async},
    metal_buffer_byte_len,
};
use metal::{Buffer, MTLResourceOptions};
use std::{collections::HashMap, ffi::c_void, sync::Arc};

#[derive(Debug, Default)]
pub(crate) struct MetalBufferPool {
    buffers_by_len: HashMap<u64, Vec<Buffer>>,
}

impl MetalBufferPool {
    fn take(&mut self, byte_len: u64) -> Option<Buffer> {
        self.buffers_by_len.get_mut(&byte_len)?.pop()
    }

    fn put(&mut self, byte_len: u64, buffer: Buffer) {
        self.buffers_by_len
            .entry(byte_len)
            .or_default()
            .push(buffer);
    }

    #[cfg(test)]
    fn count_for_len(&self, byte_len: u64) -> usize {
        self.buffers_by_len.get(&byte_len).map_or(0, Vec::len)
    }

    #[cfg(test)]
    fn bucket_count(&self) -> usize {
        self.buffers_by_len.len()
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
    pub(crate) hazard: Arc<MetalBufferHazard>,
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

#[derive(Debug)]
pub struct F32TransientBuffer {
    pub(crate) buffer: Option<Buffer>,
    pub(crate) hazard: Arc<MetalBufferHazard>,
    pub(crate) len: usize,
    pub(crate) byte_len: u64,
}

impl F32TransientBuffer {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn byte_len(&self) -> u64 {
        self.byte_len
    }
}

#[derive(Debug, Clone)]
pub struct F16Buffer {
    pub(crate) buffer: Option<Buffer>,
    pub(crate) hazard: Arc<MetalBufferHazard>,
    pub(crate) len: usize,
    pub(crate) byte_len: usize,
}

impl F16Buffer {
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

#[derive(Debug, Clone)]
pub struct I8Buffer {
    pub(crate) buffer: Option<Buffer>,
    pub(crate) hazard: Arc<MetalBufferHazard>,
    pub(crate) len: usize,
    pub(crate) byte_len: usize,
}

impl I8Buffer {
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

pub(crate) fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;

    if exponent == 0xff {
        if mantissa == 0 {
            return sign | 0x7c00;
        }
        return sign | 0x7e00;
    }

    let mut half_exponent = exponent - 127 + 15;
    if half_exponent >= 0x1f {
        return sign | 0x7c00;
    }

    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let mantissa = mantissa | 0x0080_0000;
        let shift = (14 - half_exponent) as u32;
        let mut half_mantissa = (mantissa >> shift) as u16;
        let round_bit = 1_u32 << (shift - 1);
        let remainder = mantissa & (round_bit - 1);
        if (mantissa & round_bit) != 0 && (remainder != 0 || (half_mantissa & 1) != 0) {
            half_mantissa = half_mantissa.saturating_add(1);
        }
        return sign | half_mantissa;
    }

    let mut half_mantissa = (mantissa >> 13) as u16;
    let round = mantissa & 0x0000_1fff;
    if round > 0x1000 || (round == 0x1000 && (half_mantissa & 1) != 0) {
        half_mantissa += 1;
        if half_mantissa == 0x0400 {
            half_mantissa = 0;
            half_exponent += 1;
            if half_exponent >= 0x1f {
                return sign | 0x7c00;
            }
        }
    }

    sign | ((half_exponent as u16) << 10) | half_mantissa
}

#[cfg(test)]
mod f16_tests {
    use super::*;

    #[test]
    fn f32_to_f16_bits_handles_common_values_and_rounding() {
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(-0.0), 0x8000);
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(-2.0), 0xc000);
        assert_eq!(f32_to_f16_bits(65_504.0), 0x7bff);
        assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16_bits(f32::NEG_INFINITY), 0xfc00);
        assert_ne!(f32_to_f16_bits(f32::NAN) & 0x03ff, 0);

        let half_ulp_at_one = 2.0_f32.powi(-10);
        assert_eq!(f32_to_f16_bits(1.0 + half_ulp_at_one / 2.0), 0x3c00);
        assert_eq!(
            f32_to_f16_bits(1.0 + half_ulp_at_one + half_ulp_at_one / 2.0),
            0x3c02
        );
    }

    #[test]
    fn f16_buffer_len_uses_two_bytes_per_element() {
        let Some(device) = MetalDevice::system_default_result().expect("Metal initializes") else {
            eprintln!("no Metal device available; skipping f16 buffer test");
            return;
        };

        let buffer = device.new_f16_buffer_len(9).expect("buffer length fits");

        assert_eq!(buffer.len(), 9);
        assert_eq!(buffer.byte_len(), 18);
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

    pub(crate) fn take_scratch_bf16_buffer(&self, values: &[u16]) -> Buffer {
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
            // computed from values.len() * size_of::<u16>().
            unsafe {
                std::ptr::copy_nonoverlapping(
                    values.as_ptr(),
                    buffer.contents().cast::<u16>(),
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

    pub(crate) fn take_scratch_f32_transient_buffer_len(
        &self,
        len: usize,
        context: &str,
    ) -> Result<F32TransientBuffer, MetalError> {
        let byte_len = metal_buffer_byte_len::<f32>(len, context)?;
        let buffer = if byte_len == 0 {
            None
        } else {
            Some(self.take_scratch_buffer(byte_len))
        };
        Ok(F32TransientBuffer {
            buffer,
            hazard: Arc::new(MetalBufferHazard::new()),
            len,
            byte_len,
        })
    }

    pub fn recycle_f32_transient_buffer(&self, mut buffer: F32TransientBuffer) {
        if let Some(metal_buffer) = buffer.buffer.take() {
            self.return_scratch_buffer(buffer.byte_len, metal_buffer);
        }
    }

    #[cfg(test)]
    pub fn scratch_buffer_count_for_test(&self, byte_len: u64) -> usize {
        self.scratch_buffers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .count_for_len(byte_len)
    }

    #[cfg(test)]
    pub fn scratch_buffer_bucket_count_for_test(&self) -> usize {
        self.scratch_buffers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .bucket_count()
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
            hazard: Arc::new(MetalBufferHazard::new()),
            len: values.len(),
            byte_len,
        })
    }

    pub fn new_f32_buffer_len(&self, len: usize) -> Result<F32Buffer, MetalError> {
        let byte_len = len.checked_mul(std::mem::size_of::<f32>()).ok_or_else(|| {
            MetalError::InvalidShape("f32 buffer byte length overflows usize".to_owned())
        })?;
        let buffer = if byte_len == 0 {
            None
        } else {
            Some(
                self.device
                    .new_buffer(byte_len as u64, MTLResourceOptions::StorageModeShared),
            )
        };
        Ok(F32Buffer {
            buffer,
            hazard: Arc::new(MetalBufferHazard::new()),
            len,
            byte_len,
        })
    }

    pub fn new_f16_buffer_from_f32(&self, values: &[f32]) -> Result<F16Buffer, MetalError> {
        let bits = values
            .iter()
            .copied()
            .map(f32_to_f16_bits)
            .collect::<Vec<_>>();
        let byte_len = bits
            .len()
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| {
                MetalError::InvalidShape("f16 buffer byte length overflows usize".to_owned())
            })?;
        let buffer = if byte_len == 0 {
            None
        } else {
            Some(self.device.new_buffer_with_data(
                bits.as_ptr().cast::<c_void>(),
                byte_len as u64,
                MTLResourceOptions::StorageModeShared,
            ))
        };
        Ok(F16Buffer {
            buffer,
            hazard: Arc::new(MetalBufferHazard::new()),
            len: values.len(),
            byte_len,
        })
    }

    pub fn new_f16_buffer_len(&self, len: usize) -> Result<F16Buffer, MetalError> {
        let byte_len = len.checked_mul(std::mem::size_of::<u16>()).ok_or_else(|| {
            MetalError::InvalidShape("f16 buffer byte length overflows usize".to_owned())
        })?;
        let buffer = if byte_len == 0 {
            None
        } else {
            Some(
                self.device
                    .new_buffer(byte_len as u64, MTLResourceOptions::StorageModeShared),
            )
        };
        Ok(F16Buffer {
            buffer,
            hazard: Arc::new(MetalBufferHazard::new()),
            len,
            byte_len,
        })
    }

    pub fn new_i8_buffer(&self, values: &[i8]) -> Result<I8Buffer, MetalError> {
        let byte_len = values
            .len()
            .checked_mul(std::mem::size_of::<i8>())
            .ok_or_else(|| {
                MetalError::InvalidShape("i8 buffer byte length overflows usize".to_owned())
            })?;
        let buffer = if byte_len == 0 {
            None
        } else {
            Some(self.device.new_buffer_with_data(
                values.as_ptr().cast::<c_void>(),
                byte_len as u64,
                MTLResourceOptions::StorageModeShared,
            ))
        };
        Ok(I8Buffer {
            buffer,
            hazard: Arc::new(MetalBufferHazard::new()),
            len: values.len(),
            byte_len,
        })
    }

    pub fn new_i8_buffer_len(&self, len: usize) -> Result<I8Buffer, MetalError> {
        let byte_len = len.checked_mul(std::mem::size_of::<i8>()).ok_or_else(|| {
            MetalError::InvalidShape("i8 buffer byte length overflows usize".to_owned())
        })?;
        let buffer = if byte_len == 0 {
            None
        } else {
            Some(
                self.device
                    .new_buffer(byte_len as u64, MTLResourceOptions::StorageModeShared),
            )
        };
        Ok(I8Buffer {
            buffer,
            hazard: Arc::new(MetalBufferHazard::new()),
            len,
            byte_len,
        })
    }

    pub fn write_i8_buffer_range(
        &self,
        buffer: &I8Buffer,
        start: usize,
        values: &[i8],
    ) -> Result<(), MetalError> {
        let end = start.checked_add(values.len()).ok_or_else(|| {
            MetalError::InvalidShape("i8 buffer write range overflows usize".to_owned())
        })?;
        if end > buffer.len {
            return Err(MetalError::InvalidShape(format!(
                "i8 buffer write range {start}..{end} exceeds buffer length {}",
                buffer.len
            )));
        }
        if values.is_empty() {
            return Ok(());
        }
        let Some(metal_buffer) = buffer.buffer.as_ref() else {
            return Err(MetalError::InvalidShape(
                "non-empty i8 buffer range write requires a Metal buffer".to_owned(),
            ));
        };
        let _cpu_access = buffer.hazard.begin_cpu_access();
        // SAFETY: the destination range is bounds-checked above against the i8
        // element length used to allocate the StorageModeShared buffer. The
        // buffer hazard guard waits for in-flight GPU commands touching this
        // buffer and prevents same-buffer submissions during this copy.
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                metal_buffer.contents().cast::<i8>().add(start),
                values.len(),
            );
        }
        Ok(())
    }

    pub async fn copy_i8_buffer_range(
        &self,
        source: &I8Buffer,
        source_start: usize,
        destination: &I8Buffer,
        destination_start: usize,
        len: usize,
    ) -> Result<(), MetalError> {
        let source_end = source_start.checked_add(len).ok_or_else(|| {
            MetalError::InvalidShape("source i8 buffer copy range overflows usize".to_owned())
        })?;
        if source_end > source.len {
            return Err(MetalError::InvalidShape(format!(
                "source i8 buffer copy range {source_start}..{source_end} exceeds buffer length {}",
                source.len
            )));
        }
        let destination_end = destination_start.checked_add(len).ok_or_else(|| {
            MetalError::InvalidShape("destination i8 buffer copy range overflows usize".to_owned())
        })?;
        if destination_end > destination.len {
            return Err(MetalError::InvalidShape(format!(
                "destination i8 buffer copy range {destination_start}..{destination_end} exceeds buffer length {}",
                destination.len
            )));
        }
        if len == 0 {
            return Ok(());
        }
        let Some(source_buffer) = source.buffer.as_ref() else {
            return Err(MetalError::InvalidShape(
                "non-empty i8 buffer copy requires a source Metal buffer".to_owned(),
            ));
        };
        let Some(destination_buffer) = destination.buffer.as_ref() else {
            return Err(MetalError::InvalidShape(
                "non-empty i8 buffer copy requires a destination Metal buffer".to_owned(),
            ));
        };
        let byte_len = u64::try_from(len).map_err(|err| {
            MetalError::InvalidShape(format!(
                "i8 buffer copy byte length does not fit u64: {err}"
            ))
        })?;
        let source_offset = u64::try_from(source_start).map_err(|err| {
            MetalError::InvalidShape(format!(
                "source i8 buffer copy offset does not fit u64: {err}"
            ))
        })?;
        let destination_offset = u64::try_from(destination_start).map_err(|err| {
            MetalError::InvalidShape(format!(
                "destination i8 buffer copy offset does not fit u64: {err}"
            ))
        })?;

        let command_buffer = self.transfer_queue().new_command_buffer();
        let encoder = command_buffer.new_blit_command_encoder();
        encoder.copy_from_buffer(
            source_buffer,
            source_offset,
            destination_buffer,
            destination_offset,
            byte_len,
        );
        encoder.end_encoding();
        finish_command_buffer_async(
            &[&source.hazard, &destination.hazard],
            command_buffer,
            "copy_i8_buffer_range",
        )
        .await
    }

    pub fn write_f16_buffer_range_from_f32(
        &self,
        buffer: &F16Buffer,
        start: usize,
        values: &[f32],
    ) -> Result<(), MetalError> {
        let end = start.checked_add(values.len()).ok_or_else(|| {
            MetalError::InvalidShape("f16 buffer write range overflows usize".to_owned())
        })?;
        if end > buffer.len {
            return Err(MetalError::InvalidShape(format!(
                "f16 buffer write range {start}..{end} exceeds buffer length {}",
                buffer.len
            )));
        }
        if values.is_empty() {
            return Ok(());
        }
        let Some(metal_buffer) = buffer.buffer.as_ref() else {
            return Err(MetalError::InvalidShape(
                "non-empty f16 buffer range write requires a Metal buffer".to_owned(),
            ));
        };
        let bits = values
            .iter()
            .copied()
            .map(f32_to_f16_bits)
            .collect::<Vec<_>>();
        let _cpu_access = buffer.hazard.begin_cpu_access();
        // SAFETY: the destination range is bounds-checked above against the
        // f16 element length used to allocate the StorageModeShared buffer. The
        // buffer hazard guard waits for in-flight GPU commands touching this
        // buffer and prevents same-buffer submissions during this copy.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bits.as_ptr(),
                metal_buffer.contents().cast::<u16>().add(start),
                bits.len(),
            );
        }
        Ok(())
    }

    pub async fn copy_f16_buffer_range(
        &self,
        source: &F16Buffer,
        source_start: usize,
        destination: &F16Buffer,
        destination_start: usize,
        len: usize,
    ) -> Result<(), MetalError> {
        let source_end = source_start.checked_add(len).ok_or_else(|| {
            MetalError::InvalidShape("source f16 buffer copy range overflows usize".to_owned())
        })?;
        if source_end > source.len {
            return Err(MetalError::InvalidShape(format!(
                "source f16 buffer copy range {source_start}..{source_end} exceeds buffer length {}",
                source.len
            )));
        }
        let destination_end = destination_start.checked_add(len).ok_or_else(|| {
            MetalError::InvalidShape("destination f16 buffer copy range overflows usize".to_owned())
        })?;
        if destination_end > destination.len {
            return Err(MetalError::InvalidShape(format!(
                "destination f16 buffer copy range {destination_start}..{destination_end} exceeds buffer length {}",
                destination.len
            )));
        }
        if len == 0 {
            return Ok(());
        }
        let Some(source_buffer) = source.buffer.as_ref() else {
            return Err(MetalError::InvalidShape(
                "non-empty f16 buffer copy requires a source Metal buffer".to_owned(),
            ));
        };
        let Some(destination_buffer) = destination.buffer.as_ref() else {
            return Err(MetalError::InvalidShape(
                "non-empty f16 buffer copy requires a destination Metal buffer".to_owned(),
            ));
        };
        let element_size = std::mem::size_of::<u16>();
        let byte_len = len.checked_mul(element_size).ok_or_else(|| {
            MetalError::InvalidShape("f16 buffer copy byte length overflows usize".to_owned())
        })?;
        let source_offset = source_start.checked_mul(element_size).ok_or_else(|| {
            MetalError::InvalidShape("source f16 buffer copy offset overflows usize".to_owned())
        })?;
        let destination_offset = destination_start.checked_mul(element_size).ok_or_else(|| {
            MetalError::InvalidShape(
                "destination f16 buffer copy offset overflows usize".to_owned(),
            )
        })?;
        let byte_len = u64::try_from(byte_len).map_err(|err| {
            MetalError::InvalidShape(format!(
                "f16 buffer copy byte length does not fit u64: {err}"
            ))
        })?;
        let source_offset = u64::try_from(source_offset).map_err(|err| {
            MetalError::InvalidShape(format!(
                "source f16 buffer copy offset does not fit u64: {err}"
            ))
        })?;
        let destination_offset = u64::try_from(destination_offset).map_err(|err| {
            MetalError::InvalidShape(format!(
                "destination f16 buffer copy offset does not fit u64: {err}"
            ))
        })?;

        let command_buffer = self.transfer_queue().new_command_buffer();
        let encoder = command_buffer.new_blit_command_encoder();
        encoder.copy_from_buffer(
            source_buffer,
            source_offset,
            destination_buffer,
            destination_offset,
            byte_len,
        );
        encoder.end_encoding();
        finish_command_buffer_async(
            &[&source.hazard, &destination.hazard],
            command_buffer,
            "copy_f16_buffer_range",
        )
        .await
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
        let _cpu_access = buffer.hazard.begin_cpu_access();
        // SAFETY: metal_buffer was allocated with byte_len bytes for len f32
        // values. The buffer hazard guard above waits for in-flight GPU
        // commands touching this buffer and prevents same-buffer submissions
        // during this copy.
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

    pub fn write_f32_buffer_range(
        &self,
        buffer: &F32Buffer,
        start: usize,
        values: &[f32],
    ) -> Result<(), MetalError> {
        let end = start.checked_add(values.len()).ok_or_else(|| {
            MetalError::InvalidShape("f32 buffer write range overflows usize".to_owned())
        })?;
        if end > buffer.len {
            return Err(MetalError::InvalidShape(format!(
                "f32 buffer write range {start}..{end} exceeds buffer length {}",
                buffer.len
            )));
        }
        if values.is_empty() {
            return Ok(());
        }
        let Some(metal_buffer) = buffer.buffer.as_ref() else {
            return Err(MetalError::InvalidShape(
                "non-empty f32 buffer range write requires a Metal buffer".to_owned(),
            ));
        };
        let _cpu_access = buffer.hazard.begin_cpu_access();
        // SAFETY: the destination range is bounds-checked above against the
        // f32 length used to allocate the StorageModeShared buffer. The buffer
        // hazard guard waits for in-flight GPU commands touching this buffer
        // and prevents same-buffer submissions during this copy. Both pointers
        // remain valid for values.len() f32 values.
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                metal_buffer.contents().cast::<f32>().add(start),
                values.len(),
            );
        }
        Ok(())
    }

    pub fn read_f32_buffer(&self, buffer: &F32Buffer) -> Result<Vec<f32>, MetalError> {
        self.read_f32_buffer_range(buffer, 0, buffer.len)
    }

    pub fn read_f32_transient_buffer(
        &self,
        buffer: &F32TransientBuffer,
    ) -> Result<Vec<f32>, MetalError> {
        let mut output = vec![0.0; buffer.len];
        self.read_f32_transient_buffer_in_place(buffer, &mut output)?;
        Ok(output)
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

    pub fn read_f32_transient_buffer_in_place(
        &self,
        buffer: &F32TransientBuffer,
        output: &mut [f32],
    ) -> Result<(), MetalError> {
        if output.len() < buffer.len {
            return Err(MetalError::InvalidShape(format!(
                "output length {} is smaller than transient f32 buffer length {}",
                output.len(),
                buffer.len
            )));
        }
        if buffer.len == 0 {
            return Ok(());
        }
        let Some(metal_buffer) = buffer.buffer.as_ref() else {
            return Err(MetalError::InvalidShape(
                "non-empty transient f32 buffer read requires a Metal buffer".to_owned(),
            ));
        };
        let _cpu_access = buffer.hazard.begin_cpu_access();
        // SAFETY: metal_buffer was allocated with byte_len bytes for len f32
        // values. The buffer hazard waits for in-flight GPU commands touching
        // this transient buffer before exposing it to the CPU.
        unsafe {
            let ptr = metal_buffer.contents().cast::<f32>();
            let values = std::slice::from_raw_parts(ptr, buffer.len);
            output[..buffer.len].copy_from_slice(values);
        };
        Ok(())
    }

    pub(crate) fn zero_f32_transient_buffer(
        &self,
        buffer: &F32TransientBuffer,
    ) -> Result<(), MetalError> {
        if buffer.len == 0 {
            return Ok(());
        }
        let Some(metal_buffer) = buffer.buffer.as_ref() else {
            return Err(MetalError::InvalidShape(
                "non-empty transient f32 buffer zero fill requires a Metal buffer".to_owned(),
            ));
        };
        let _cpu_access = buffer.hazard.begin_cpu_access();
        // SAFETY: metal_buffer was allocated with byte_len bytes for len f32
        // values. Filling bytewise zero is valid for f32 zero.
        unsafe {
            std::ptr::write_bytes(metal_buffer.contents().cast::<f32>(), 0, buffer.len);
        }
        Ok(())
    }

    pub async fn copy_f32_buffer_range(
        &self,
        source: &F32Buffer,
        source_start: usize,
        destination: &F32Buffer,
        destination_start: usize,
        len: usize,
    ) -> Result<(), MetalError> {
        let source_end = source_start.checked_add(len).ok_or_else(|| {
            MetalError::InvalidShape("source f32 buffer copy range overflows usize".to_owned())
        })?;
        if source_end > source.len {
            return Err(MetalError::InvalidShape(format!(
                "source f32 buffer copy range {source_start}..{source_end} exceeds buffer length {}",
                source.len
            )));
        }
        let destination_end = destination_start.checked_add(len).ok_or_else(|| {
            MetalError::InvalidShape("destination f32 buffer copy range overflows usize".to_owned())
        })?;
        if destination_end > destination.len {
            return Err(MetalError::InvalidShape(format!(
                "destination f32 buffer copy range {destination_start}..{destination_end} exceeds buffer length {}",
                destination.len
            )));
        }
        if len == 0 {
            return Ok(());
        }
        let Some(source_buffer) = source.buffer.as_ref() else {
            return Err(MetalError::InvalidShape(
                "non-empty f32 buffer copy requires a source Metal buffer".to_owned(),
            ));
        };
        let Some(destination_buffer) = destination.buffer.as_ref() else {
            return Err(MetalError::InvalidShape(
                "non-empty f32 buffer copy requires a destination Metal buffer".to_owned(),
            ));
        };
        let element_size = std::mem::size_of::<f32>();
        let byte_len = len.checked_mul(element_size).ok_or_else(|| {
            MetalError::InvalidShape("f32 buffer copy byte length overflows usize".to_owned())
        })?;
        let source_offset = source_start.checked_mul(element_size).ok_or_else(|| {
            MetalError::InvalidShape("source f32 buffer copy offset overflows usize".to_owned())
        })?;
        let destination_offset = destination_start.checked_mul(element_size).ok_or_else(|| {
            MetalError::InvalidShape(
                "destination f32 buffer copy offset overflows usize".to_owned(),
            )
        })?;
        let byte_len = u64::try_from(byte_len).map_err(|err| {
            MetalError::InvalidShape(format!(
                "f32 buffer copy byte length does not fit u64: {err}"
            ))
        })?;
        let source_offset = u64::try_from(source_offset).map_err(|err| {
            MetalError::InvalidShape(format!(
                "source f32 buffer copy offset does not fit u64: {err}"
            ))
        })?;
        let destination_offset = u64::try_from(destination_offset).map_err(|err| {
            MetalError::InvalidShape(format!(
                "destination f32 buffer copy offset does not fit u64: {err}"
            ))
        })?;

        let command_buffer = self.transfer_queue().new_command_buffer();
        let encoder = command_buffer.new_blit_command_encoder();
        encoder.copy_from_buffer(
            source_buffer,
            source_offset,
            destination_buffer,
            destination_offset,
            byte_len,
        );
        encoder.end_encoding();
        finish_command_buffer_async(
            &[&source.hazard, &destination.hazard],
            command_buffer,
            "copy_f32_buffer_range",
        )
        .await
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
        let _cpu_access = buffer.hazard.begin_cpu_access();
        // SAFETY: the requested range is bounds-checked above against the f32
        // length used to allocate the StorageModeShared buffer. The buffer
        // hazard guard above waits for in-flight GPU commands touching this
        // buffer and prevents same-buffer submissions during this copy.
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

    fn f32_to_bf16_bits(value: f32) -> u16 {
        (value.to_bits() >> 16) as u16
    }

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

    #[tokio::test]
    async fn matvec_bf16_f32_buffered_reuses_transient_scratch_buffers() {
        let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
        else {
            eprintln!("no Metal device available; skipping smoke test");
            return;
        };

        let matrix = [1.0, 2.0, 3.0, 4.0, -1.0, 0.5].map(f32_to_bf16_bits);
        let matrix_buffer = device
            .new_bf16_matrix_buffer(&matrix, 2, 3)
            .expect("BF16 matrix buffer uploads");
        let mut output = vec![0.0; 2];

        device
            .matvec_bf16_f32_buffered(&matrix_buffer, &[0.5, -2.0, 4.0], &mut output)
            .await
            .expect("first buffered BF16 matvec succeeds");
        device
            .matvec_bf16_f32_buffered(&matrix_buffer, &[1.0, 0.0, -1.0], &mut output)
            .await
            .expect("second buffered BF16 matvec succeeds");

        assert_eq!(
            device.scratch_buffer_count_for_test(3 * std::mem::size_of::<f32>() as u64),
            1
        );
        assert_eq!(
            device.scratch_buffer_count_for_test(2 * std::mem::size_of::<f32>() as u64),
            1
        );
        assert!((output[0] + 2.0).abs() < 1e-6);
        assert!((output[1] - 3.5).abs() < 1e-6);
    }

    #[tokio::test]
    async fn matvec_bf16_f32_reuses_transient_matrix_scratch_buffer() {
        let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
        else {
            eprintln!("no Metal device available; skipping smoke test");
            return;
        };

        let matrix = [
            1.0, 2.0, 3.0, 4.0, -1.0, 0.5, -0.5, 1.5, 2.0, 0.25, -2.0, 1.0, 3.5, -1.5, 0.75,
        ]
        .map(f32_to_bf16_bits);
        let vector = [0.5, -2.0, 4.0];
        let mut output = vec![0.0; 5];

        device
            .matvec_bf16_f32(&matrix, 5, 3, &vector, &mut output)
            .await
            .expect("first BF16 matvec succeeds");
        device
            .matvec_bf16_f32(&matrix, 5, 3, &vector, &mut output)
            .await
            .expect("second BF16 matvec succeeds");

        assert_eq!(
            device.scratch_buffer_count_for_test(15 * std::mem::size_of::<u16>() as u64),
            1
        );
        assert_eq!(
            device.scratch_buffer_count_for_test(3 * std::mem::size_of::<f32>() as u64),
            1
        );
        assert_eq!(
            device.scratch_buffer_count_for_test(5 * std::mem::size_of::<f32>() as u64),
            1
        );
        assert!((output[0] - 8.5).abs() < 1e-6);
        assert!((output[1] - 6.0).abs() < 1e-6);
        assert!((output[2] - 4.75).abs() < 1e-6);
        assert!((output[3] - 8.125).abs() < 1e-6);
        assert!((output[4] - 7.75).abs() < 1e-6);
    }

    #[tokio::test]
    async fn batched_matvec_bf16_f32_buffered_reuses_transient_scratch_buffers() {
        let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
        else {
            eprintln!("no Metal device available; skipping smoke test");
            return;
        };

        let matrix = [1.0, 2.0, 3.0, 4.0, -1.0, 0.5].map(f32_to_bf16_bits);
        let matrix_buffer = device
            .new_bf16_matrix_buffer(&matrix, 2, 3)
            .expect("BF16 matrix buffer uploads");
        let vectors = [0.5, -2.0, 4.0, 1.0, 0.0, -1.0];
        let mut output = vec![0.0; 4];

        device
            .batched_matvec_bf16_f32_buffered(&matrix_buffer, &vectors, 2, &mut output)
            .await
            .expect("first buffered batched BF16 matvec succeeds");
        device
            .batched_matvec_bf16_f32_buffered(&matrix_buffer, &vectors, 2, &mut output)
            .await
            .expect("second buffered batched BF16 matvec succeeds");

        assert_eq!(
            device.scratch_buffer_count_for_test(6 * std::mem::size_of::<f32>() as u64),
            1
        );
        assert_eq!(
            device.scratch_buffer_count_for_test(4 * std::mem::size_of::<f32>() as u64),
            1
        );
        assert!((output[0] - 8.5).abs() < 1e-6);
        assert!((output[1] - 6.0).abs() < 1e-6);
        assert!((output[2] + 2.0).abs() < 1e-6);
        assert!((output[3] - 3.5).abs() < 1e-6);
    }

    #[tokio::test]
    async fn batched_matvec_bf16_f32_reuses_transient_matrix_scratch_buffer() {
        let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
        else {
            eprintln!("no Metal device available; skipping smoke test");
            return;
        };

        let matrix = [
            1.0, 2.0, 3.0, 4.0, -1.0, 0.5, -0.5, 1.5, 2.0, 0.25, -2.0, 1.0, 3.5, -1.5, 0.75,
        ]
        .map(f32_to_bf16_bits);
        let vectors = [0.5, -2.0, 4.0, 1.0, 0.0, -1.0];
        let mut output = vec![0.0; 10];

        device
            .batched_matvec_bf16_f32(&matrix, 5, 3, &vectors, 2, &mut output)
            .await
            .expect("first batched BF16 matvec succeeds");
        device
            .batched_matvec_bf16_f32(&matrix, 5, 3, &vectors, 2, &mut output)
            .await
            .expect("second batched BF16 matvec succeeds");

        assert_eq!(
            device.scratch_buffer_count_for_test(15 * std::mem::size_of::<u16>() as u64),
            1
        );
        assert_eq!(
            device.scratch_buffer_count_for_test(6 * std::mem::size_of::<f32>() as u64),
            1
        );
        assert_eq!(
            device.scratch_buffer_count_for_test(10 * std::mem::size_of::<f32>() as u64),
            1
        );
        assert!((output[0] - 8.5).abs() < 1e-6);
        assert!((output[1] - 6.0).abs() < 1e-6);
        assert!((output[2] - 4.75).abs() < 1e-6);
        assert!((output[3] - 8.125).abs() < 1e-6);
        assert!((output[4] - 7.75).abs() < 1e-6);
        assert!((output[5] + 2.0).abs() < 1e-6);
        assert!((output[6] - 3.5).abs() < 1e-6);
        assert!((output[7] + 2.5).abs() < 1e-6);
        assert!((output[8] + 0.75).abs() < 1e-6);
        assert!((output[9] - 2.75).abs() < 1e-6);
    }

    #[tokio::test]
    async fn scratch_buffer_pool_groups_buffers_by_byte_len() {
        let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
        else {
            eprintln!("no Metal device available; skipping smoke test");
            return;
        };

        let small = device.take_scratch_buffer(16);
        let large = device.take_scratch_buffer(32);
        let other_small = device.take_scratch_buffer(16);

        device.return_scratch_buffer(16, small);
        device.return_scratch_buffer(32, large);
        device.return_scratch_buffer(16, other_small);

        assert_eq!(device.scratch_buffer_bucket_count_for_test(), 2);
        assert_eq!(device.scratch_buffer_count_for_test(16), 2);
        assert_eq!(device.scratch_buffer_count_for_test(32), 1);

        let reused_small = device.take_scratch_buffer(16);
        assert_eq!(device.scratch_buffer_count_for_test(16), 1);
        device.return_scratch_buffer(16, reused_small);
    }

    #[tokio::test]
    async fn copy_f16_buffer_range_copies_between_existing_buffers() {
        let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
        else {
            eprintln!("no Metal device available; skipping f16 copy test");
            return;
        };

        let source = device
            .new_f16_buffer_from_f32(&[1.0, 2.0, 3.0, 4.0])
            .expect("source buffer");
        let destination = device.new_f16_buffer_len(4).expect("destination buffer");

        device
            .copy_f16_buffer_range(&source, 1, &destination, 0, 2)
            .await
            .expect("buffer copy succeeds");

        let mut output = vec![0.0; 2];
        device
            .select_head_rows_f16_buffered(&destination, 2, 1, 0, 1, &mut output)
            .await
            .expect("copied f16 rows are readable");
        assert_eq!(output, vec![2.0, 3.0]);
    }
}
