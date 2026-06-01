use super::TensorLoadError;

/// Number of quantized weights carried by one GGML Q8_0 block.
pub const Q8_0_BLOCK_SIZE: usize = 32;
/// Byte width of one GGML Q8_0 block: f16 scale plus 32 signed quantized weights.
pub const Q8_0_BLOCK_BYTE_LEN: usize = 2 + Q8_0_BLOCK_SIZE;
/// Number of quantized weights carried by one GGML Q4_0 block.
pub const Q4_0_BLOCK_SIZE: usize = 32;
const Q4_0_PACKED_QUANTS_PER_BLOCK: usize = Q4_0_BLOCK_SIZE / 2;
/// Byte width of one GGML Q4_0 block: f16 scale plus 16 packed 4-bit weights.
pub const Q4_0_BLOCK_BYTE_LEN: usize = 2 + Q4_0_PACKED_QUANTS_PER_BLOCK;

/// Borrowed row-major Q4_0 weight matrix.
///
/// The matrix keeps GGML-compatible Q4_0 block bytes borrowed from the caller
/// and dequantizes each packed nibble only while computing a dot product.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Q4RowMajorMatrix<'a> {
    rows: usize,
    columns: usize,
    blocks_per_row: usize,
    blocks: &'a [u8],
}

impl<'a> Q4RowMajorMatrix<'a> {
    pub fn from_blocks(
        rows: usize,
        columns: usize,
        blocks: &'a [u8],
    ) -> Result<Self, TensorLoadError> {
        let blocks_per_row = columns.div_ceil(Q4_0_BLOCK_SIZE);
        let expected_len = rows
            .checked_mul(blocks_per_row)
            .and_then(|block_count| block_count.checked_mul(Q4_0_BLOCK_BYTE_LEN))
            .ok_or_else(|| {
                TensorLoadError::integrity("Q4_0 row-major matrix byte length overflow")
            })?;
        if blocks.len() != expected_len {
            return Err(TensorLoadError::integrity(format!(
                "Q4_0 row-major matrix byte length {} does not match rows {rows} * blocks_per_row {blocks_per_row} * block bytes {Q4_0_BLOCK_BYTE_LEN}",
                blocks.len()
            )));
        }
        Ok(Self {
            rows,
            columns,
            blocks_per_row,
            blocks,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn blocks(&self) -> &'a [u8] {
        self.blocks
    }

    pub fn matvec_f32(&self, input: &[f32]) -> Result<Vec<f32>, TensorLoadError> {
        let mut output = vec![0.0; self.rows];
        self.matvec_f32_in_place(input, &mut output)?;
        Ok(output)
    }

    pub fn matvec_f32_in_place(
        &self,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), TensorLoadError> {
        if input.len() != self.columns {
            return Err(TensorLoadError::integrity(format!(
                "Q4_0 matvec input length {} does not match columns {}",
                input.len(),
                self.columns
            )));
        }
        if output.len() < self.rows {
            return Err(TensorLoadError::integrity(
                "output buffer too small for Q4_0 matvec",
            ));
        }
        for (row, out) in output.iter_mut().take(self.rows).enumerate() {
            *out = self.dot_row(row, input)?;
        }
        Ok(())
    }

    fn dot_row(&self, row: usize, input: &[f32]) -> Result<f32, TensorLoadError> {
        let row_block_start = row
            .checked_mul(self.blocks_per_row)
            .ok_or_else(|| TensorLoadError::integrity("Q4_0 row block offset overflow"))?;
        let mut sum = 0.0_f32;
        for block_idx in 0..self.blocks_per_row {
            let block_start = row_block_start
                .checked_add(block_idx)
                .and_then(|block_index| block_index.checked_mul(Q4_0_BLOCK_BYTE_LEN))
                .ok_or_else(|| TensorLoadError::integrity("Q4_0 block offset overflow"))?;
            let block_end = block_start
                .checked_add(Q4_0_BLOCK_BYTE_LEN)
                .ok_or_else(|| TensorLoadError::integrity("Q4_0 block range overflow"))?;
            let block = self
                .blocks
                .get(block_start..block_end)
                .ok_or_else(|| TensorLoadError::integrity("Q4_0 block range is invalid"))?;
            let scale = q4_0_block_scale(block)?;
            let column_start = block_idx
                .checked_mul(Q4_0_BLOCK_SIZE)
                .ok_or_else(|| TensorLoadError::integrity("Q4_0 column offset overflow"))?;
            let remaining_columns = self
                .columns
                .checked_sub(column_start)
                .ok_or_else(|| TensorLoadError::integrity("Q4_0 column range is invalid"))?;
            let active_columns = Q4_0_BLOCK_SIZE.min(remaining_columns);
            let column_end = column_start
                .checked_add(active_columns)
                .ok_or_else(|| TensorLoadError::integrity("Q4_0 input range overflow"))?;
            let input_values = input
                .get(column_start..column_end)
                .ok_or_else(|| TensorLoadError::integrity("Q4_0 input range is invalid"))?;
            for (column_offset, value) in input_values.iter().enumerate() {
                sum += scale * f32::from(q4_0_quant(block, column_offset)) * value;
            }
        }
        Ok(sum)
    }
}

/// Borrowed row-major Q8_0 weight matrix.
///
/// The matrix keeps GGML-compatible Q8_0 block bytes borrowed from the caller
/// and dequantizes each block only while computing a dot product.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Q8RowMajorMatrix<'a> {
    rows: usize,
    columns: usize,
    blocks_per_row: usize,
    blocks: &'a [u8],
}

impl<'a> Q8RowMajorMatrix<'a> {
    pub fn from_blocks(
        rows: usize,
        columns: usize,
        blocks: &'a [u8],
    ) -> Result<Self, TensorLoadError> {
        let blocks_per_row = columns.div_ceil(Q8_0_BLOCK_SIZE);
        let expected_len = rows
            .checked_mul(blocks_per_row)
            .and_then(|block_count| block_count.checked_mul(Q8_0_BLOCK_BYTE_LEN))
            .ok_or_else(|| {
                TensorLoadError::integrity("Q8_0 row-major matrix byte length overflow")
            })?;
        if blocks.len() != expected_len {
            return Err(TensorLoadError::integrity(format!(
                "Q8_0 row-major matrix byte length {} does not match rows {rows} * blocks_per_row {blocks_per_row} * block bytes {Q8_0_BLOCK_BYTE_LEN}",
                blocks.len()
            )));
        }
        Ok(Self {
            rows,
            columns,
            blocks_per_row,
            blocks,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn blocks(&self) -> &'a [u8] {
        self.blocks
    }

    pub fn matvec_f32(&self, input: &[f32]) -> Result<Vec<f32>, TensorLoadError> {
        let mut output = vec![0.0; self.rows];
        self.matvec_f32_in_place(input, &mut output)?;
        Ok(output)
    }

    pub fn matvec_f32_in_place(
        &self,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), TensorLoadError> {
        if input.len() != self.columns {
            return Err(TensorLoadError::integrity(format!(
                "Q8_0 matvec input length {} does not match columns {}",
                input.len(),
                self.columns
            )));
        }
        if output.len() < self.rows {
            return Err(TensorLoadError::integrity(
                "output buffer too small for Q8_0 matvec",
            ));
        }
        for (row, out) in output.iter_mut().take(self.rows).enumerate() {
            *out = self.dot_row(row, input)?;
        }
        Ok(())
    }

    fn dot_row(&self, row: usize, input: &[f32]) -> Result<f32, TensorLoadError> {
        let row_block_start = row
            .checked_mul(self.blocks_per_row)
            .ok_or_else(|| TensorLoadError::integrity("Q8_0 row block offset overflow"))?;
        let mut sum = 0.0_f32;
        for block_idx in 0..self.blocks_per_row {
            let block_start = row_block_start
                .checked_add(block_idx)
                .and_then(|block_index| block_index.checked_mul(Q8_0_BLOCK_BYTE_LEN))
                .ok_or_else(|| TensorLoadError::integrity("Q8_0 block offset overflow"))?;
            let block_end = block_start
                .checked_add(Q8_0_BLOCK_BYTE_LEN)
                .ok_or_else(|| TensorLoadError::integrity("Q8_0 block range overflow"))?;
            let block = self
                .blocks
                .get(block_start..block_end)
                .ok_or_else(|| TensorLoadError::integrity("Q8_0 block range is invalid"))?;
            let scale = q8_0_block_scale(block)?;
            let column_start = block_idx
                .checked_mul(Q8_0_BLOCK_SIZE)
                .ok_or_else(|| TensorLoadError::integrity("Q8_0 column offset overflow"))?;
            let remaining_columns = self
                .columns
                .checked_sub(column_start)
                .ok_or_else(|| TensorLoadError::integrity("Q8_0 column range is invalid"))?;
            let active_columns = Q8_0_BLOCK_SIZE.min(remaining_columns);
            let quants = &block[2..2 + active_columns];
            let column_end = column_start
                .checked_add(active_columns)
                .ok_or_else(|| TensorLoadError::integrity("Q8_0 input range overflow"))?;
            let input_values = input
                .get(column_start..column_end)
                .ok_or_else(|| TensorLoadError::integrity("Q8_0 input range is invalid"))?;
            for (quant, value) in quants.iter().zip(input_values) {
                sum += scale * f32::from(*quant as i8) * value;
            }
        }
        Ok(sum)
    }
}

fn q8_0_block_scale(block: &[u8]) -> Result<f32, TensorLoadError> {
    finite_f16_block_scale(block, "Q8_0")
}

fn q4_0_block_scale(block: &[u8]) -> Result<f32, TensorLoadError> {
    finite_f16_block_scale(block, "Q4_0")
}

fn finite_f16_block_scale(block: &[u8], format: &str) -> Result<f32, TensorLoadError> {
    let scale = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
    if !scale.is_finite() {
        return Err(TensorLoadError::integrity(format!(
            "{format} block scale must be finite"
        )));
    }
    Ok(scale)
}

fn q4_0_quant(block: &[u8], column_offset: usize) -> i8 {
    let byte = block[2 + column_offset % Q4_0_PACKED_QUANTS_PER_BLOCK];
    let nibble = if column_offset < Q4_0_PACKED_QUANTS_PER_BLOCK {
        byte & 0x0f
    } else {
        byte >> 4
    };
    (nibble as i8) - 8
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = bits & 0x03ff;
    match exponent {
        0 => {
            if fraction == 0 {
                return f32::from_bits(sign);
            }
            let mut mantissa = u32::from(fraction);
            let mut exponent = -14_i32;
            while mantissa & 0x0400 == 0 {
                mantissa <<= 1;
                exponent -= 1;
            }
            mantissa &= 0x03ff;
            f32::from_bits(sign | (((exponent + 127) as u32) << 23) | (mantissa << 13))
        }
        0x1f => f32::from_bits(sign | 0x7f80_0000 | (u32::from(fraction) << 13)),
        _ => f32::from_bits(
            sign | (((i32::from(exponent) - 15 + 127) as u32) << 23) | (u32::from(fraction) << 13),
        ),
    }
}
