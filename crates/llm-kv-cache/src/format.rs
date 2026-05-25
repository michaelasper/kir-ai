use crate::KvCacheError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum KvCacheFormat {
    F32,
    F16,
    Int8,
    AsymmetricVq,
}

impl std::fmt::Display for KvCacheFormat {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F32 => formatter.write_str("f32"),
            Self::F16 => formatter.write_str("f16"),
            Self::Int8 => formatter.write_str("int8"),
            Self::AsymmetricVq => formatter.write_str("asymmetric_vq"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum KvCacheValueQuantizationBits {
    Three,
    Four,
    Eight,
}

impl KvCacheValueQuantizationBits {
    pub fn bit_width(self) -> usize {
        match self {
            Self::Three => 3,
            Self::Four => 4,
            Self::Eight => 8,
        }
    }

    pub fn level_count(self) -> usize {
        1_usize << self.bit_width()
    }

    pub(crate) fn payload_bytes(self, value_count: usize) -> Result<usize, KvCacheError> {
        value_count
            .checked_mul(self.bit_width())
            .and_then(|bits| bits.checked_add(7))
            .map(|bits| bits / 8)
            .ok_or(KvCacheError::InvalidShape)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AsymmetricVqCacheConfig {
    value_bits: KvCacheValueQuantizationBits,
}

impl AsymmetricVqCacheConfig {
    pub fn new(value_bits: KvCacheValueQuantizationBits) -> Self {
        Self { value_bits }
    }

    pub fn value_bits(self) -> KvCacheValueQuantizationBits {
        self.value_bits
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCacheConfig {
    format: KvCacheFormat,
    asymmetric_vq: Option<AsymmetricVqCacheConfig>,
}

impl KvCacheConfig {
    pub fn f32() -> Self {
        Self {
            format: KvCacheFormat::F32,
            asymmetric_vq: None,
        }
    }

    pub fn f16() -> Self {
        Self {
            format: KvCacheFormat::F16,
            asymmetric_vq: None,
        }
    }

    pub fn int8() -> Self {
        Self {
            format: KvCacheFormat::Int8,
            asymmetric_vq: None,
        }
    }

    /// Opts into the asymmetric/codebook quantization prototype.
    ///
    /// This remains the phase-3 TurboQuant-style track deferred to #334. It is
    /// kept separate from the symmetric INT8 CPU and Metal KV cache path.
    pub fn asymmetric_vq(config: AsymmetricVqCacheConfig) -> Self {
        Self {
            format: KvCacheFormat::AsymmetricVq,
            asymmetric_vq: Some(config),
        }
    }

    pub fn format(self) -> KvCacheFormat {
        self.format
    }

    pub fn asymmetric_vq_config(self) -> Option<AsymmetricVqCacheConfig> {
        self.asymmetric_vq
    }
}

impl Default for KvCacheConfig {
    fn default() -> Self {
        Self::f32()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KvCacheReconstructionError {
    mse: f64,
    max_abs: f32,
}

impl KvCacheReconstructionError {
    pub(crate) fn new(mse: f64, max_abs: f32) -> Self {
        Self { mse, max_abs }
    }

    pub fn mse(self) -> f64 {
        self.mse
    }

    pub fn max_abs(self) -> f32 {
        self.max_abs
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KvCacheFormatMetrics {
    active_format: KvCacheFormat,
    phase3_value_bits: Option<KvCacheValueQuantizationBits>,
    f32_resident_bytes: u64,
    f16_resident_bytes: u64,
    int8_resident_bytes: u64,
    f32_uploaded_bytes: u64,
    f16_uploaded_bytes: u64,
    int8_uploaded_bytes: u64,
    phase3_resident_bytes: u64,
    phase3_value_payload_bytes: u64,
    phase3_value_metadata_bytes: u64,
    phase3_uploaded_bytes: u64,
    phase3_reconstruction_error: Option<KvCacheReconstructionError>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct KvCacheFormatMetricParts {
    pub(crate) active_format: KvCacheFormat,
    pub(crate) phase3_value_bits: Option<KvCacheValueQuantizationBits>,
    pub(crate) f32_resident_bytes: u64,
    pub(crate) f16_resident_bytes: u64,
    pub(crate) int8_resident_bytes: u64,
    pub(crate) f32_uploaded_bytes: u64,
    pub(crate) f16_uploaded_bytes: u64,
    pub(crate) int8_uploaded_bytes: u64,
    pub(crate) phase3_resident_bytes: u64,
    pub(crate) phase3_value_payload_bytes: u64,
    pub(crate) phase3_value_metadata_bytes: u64,
    pub(crate) phase3_uploaded_bytes: u64,
    pub(crate) phase3_reconstruction_error: Option<KvCacheReconstructionError>,
}

impl KvCacheFormatMetrics {
    pub(crate) fn from_parts(parts: KvCacheFormatMetricParts) -> Self {
        Self {
            active_format: parts.active_format,
            phase3_value_bits: parts.phase3_value_bits,
            f32_resident_bytes: parts.f32_resident_bytes,
            f16_resident_bytes: parts.f16_resident_bytes,
            int8_resident_bytes: parts.int8_resident_bytes,
            f32_uploaded_bytes: parts.f32_uploaded_bytes,
            f16_uploaded_bytes: parts.f16_uploaded_bytes,
            int8_uploaded_bytes: parts.int8_uploaded_bytes,
            phase3_resident_bytes: parts.phase3_resident_bytes,
            phase3_value_payload_bytes: parts.phase3_value_payload_bytes,
            phase3_value_metadata_bytes: parts.phase3_value_metadata_bytes,
            phase3_uploaded_bytes: parts.phase3_uploaded_bytes,
            phase3_reconstruction_error: parts.phase3_reconstruction_error,
        }
    }

    pub fn active_format(self) -> KvCacheFormat {
        self.active_format
    }

    pub fn phase3_value_bits(self) -> Option<KvCacheValueQuantizationBits> {
        self.phase3_value_bits
    }

    pub fn f32_resident_bytes(self) -> u64 {
        self.f32_resident_bytes
    }

    pub fn f16_resident_bytes(self) -> u64 {
        self.f16_resident_bytes
    }

    pub fn int8_resident_bytes(self) -> u64 {
        self.int8_resident_bytes
    }

    pub fn f32_uploaded_bytes(self) -> u64 {
        self.f32_uploaded_bytes
    }

    pub fn f16_uploaded_bytes(self) -> u64 {
        self.f16_uploaded_bytes
    }

    pub fn int8_uploaded_bytes(self) -> u64 {
        self.int8_uploaded_bytes
    }

    pub fn phase3_resident_bytes(self) -> u64 {
        self.phase3_resident_bytes
    }

    pub fn phase3_value_payload_bytes(self) -> u64 {
        self.phase3_value_payload_bytes
    }

    pub fn phase3_value_metadata_bytes(self) -> u64 {
        self.phase3_value_metadata_bytes
    }

    pub fn phase3_uploaded_bytes(self) -> u64 {
        self.phase3_uploaded_bytes
    }

    pub fn phase3_reconstruction_error(self) -> Option<KvCacheReconstructionError> {
        self.phase3_reconstruction_error
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LayerInt8KvStore {
    vector_len: usize,
    blocks: Vec<Option<Int8KvBlock>>,
}

impl LayerInt8KvStore {
    pub(crate) fn new(block_count: usize, vector_len: usize) -> Result<Self, KvCacheError> {
        if block_count == 0 || vector_len == 0 {
            return Err(KvCacheError::InvalidShape);
        }
        Ok(Self {
            vector_len,
            blocks: vec![None; block_count],
        })
    }

    pub(crate) fn update_block(
        &mut self,
        block_index: usize,
        keys: &[f32],
        values: &[f32],
    ) -> Result<(), KvCacheError> {
        if block_index >= self.blocks.len()
            || keys.is_empty()
            || keys.len() != values.len()
            || !keys.len().is_multiple_of(self.vector_len)
        {
            return Err(KvCacheError::InvalidShape);
        }
        self.blocks[block_index] = Some(Int8KvBlock::quantize(keys, values, self.vector_len)?);
        Ok(())
    }

    pub(crate) fn clear(&mut self) {
        for block in &mut self.blocks {
            *block = None;
        }
    }

    pub(crate) fn block(&self, block_index: usize) -> Result<&Int8KvBlock, KvCacheError> {
        self.blocks
            .get(block_index)
            .ok_or(KvCacheError::InvalidShape)?
            .as_ref()
            .ok_or(KvCacheError::InvalidShape)
    }

    pub(crate) fn dequantized_key_block(
        &self,
        block_index: usize,
    ) -> Result<Vec<f32>, KvCacheError> {
        self.block(block_index)?.dequantize_keys()
    }

    pub(crate) fn dequantized_value_block(
        &self,
        block_index: usize,
    ) -> Result<Vec<f32>, KvCacheError> {
        self.block(block_index)?.dequantize_values()
    }

    pub(crate) fn resident_bytes(&self) -> u64 {
        self.payload_bytes().saturating_add(self.metadata_bytes())
    }

    pub(crate) fn payload_bytes(&self) -> u64 {
        self.blocks
            .iter()
            .filter_map(Option::as_ref)
            .map(Int8KvBlock::payload_bytes)
            .sum()
    }

    pub(crate) fn metadata_bytes(&self) -> u64 {
        self.blocks
            .iter()
            .filter_map(Option::as_ref)
            .map(Int8KvBlock::metadata_bytes)
            .sum()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Int8KvBlock {
    vector_len: usize,
    token_count: usize,
    key_scales: Vec<f32>,
    value_scales: Vec<f32>,
    key_codes: Vec<i8>,
    value_codes: Vec<i8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LayerInt8KvToken {
    key_codes: Vec<i8>,
    value_codes: Vec<i8>,
    key_scales: Vec<f32>,
    value_scales: Vec<f32>,
}

impl LayerInt8KvToken {
    pub(crate) fn quantize(
        key: &[f32],
        value: &[f32],
        vector_len: usize,
    ) -> Result<Self, KvCacheError> {
        if key.len() != vector_len || value.len() != vector_len {
            return Err(KvCacheError::InvalidShape);
        }
        let (key_codes, key_scales) = quantize_int8_rows(key, vector_len)?;
        let (value_codes, value_scales) = quantize_int8_rows(value, vector_len)?;
        Ok(Self {
            key_codes,
            value_codes,
            key_scales,
            value_scales,
        })
    }

    pub fn key_codes(&self) -> &[i8] {
        &self.key_codes
    }

    pub fn value_codes(&self) -> &[i8] {
        &self.value_codes
    }

    pub fn key_scales(&self) -> &[f32] {
        &self.key_scales
    }

    pub fn value_scales(&self) -> &[f32] {
        &self.value_scales
    }
}

impl Int8KvBlock {
    fn quantize(keys: &[f32], values: &[f32], vector_len: usize) -> Result<Self, KvCacheError> {
        if keys.is_empty()
            || vector_len == 0
            || keys.len() != values.len()
            || !keys.len().is_multiple_of(vector_len)
        {
            return Err(KvCacheError::InvalidShape);
        }
        let (key_codes, key_scales) = quantize_int8_rows(keys, vector_len)?;
        let (value_codes, value_scales) = quantize_int8_rows(values, vector_len)?;
        Ok(Self {
            vector_len,
            token_count: keys.len() / vector_len,
            key_scales,
            value_scales,
            key_codes,
            value_codes,
        })
    }

    pub(crate) fn key_codes(&self) -> &[i8] {
        &self.key_codes
    }

    pub(crate) fn value_codes(&self) -> &[i8] {
        &self.value_codes
    }

    pub(crate) fn key_scales(&self) -> &[f32] {
        &self.key_scales
    }

    pub(crate) fn value_scales(&self) -> &[f32] {
        &self.value_scales
    }

    fn dequantize_keys(&self) -> Result<Vec<f32>, KvCacheError> {
        dequantize_int8_rows(&self.key_codes, &self.key_scales, self.vector_len)
    }

    fn dequantize_values(&self) -> Result<Vec<f32>, KvCacheError> {
        dequantize_int8_rows(&self.value_codes, &self.value_scales, self.vector_len)
    }

    fn payload_bytes(&self) -> u64 {
        self.key_codes.len().saturating_add(self.value_codes.len()) as u64
    }

    fn metadata_bytes(&self) -> u64 {
        let scale_bytes = self
            .key_scales
            .len()
            .saturating_add(self.value_scales.len())
            .saturating_mul(std::mem::size_of::<f32>());
        scale_bytes.saturating_add(2_usize.saturating_mul(std::mem::size_of::<usize>())) as u64
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LayerQuantizedValueStore {
    config: AsymmetricVqCacheConfig,
    vector_len: usize,
    blocks: Vec<Option<QuantizedValueBlock>>,
}

impl LayerQuantizedValueStore {
    pub(crate) fn new(
        block_count: usize,
        vector_len: usize,
        config: AsymmetricVqCacheConfig,
    ) -> Result<Self, KvCacheError> {
        if block_count == 0 || vector_len == 0 {
            return Err(KvCacheError::InvalidShape);
        }
        Ok(Self {
            config,
            vector_len,
            blocks: vec![None; block_count],
        })
    }

    pub(crate) fn value_bits(&self) -> KvCacheValueQuantizationBits {
        self.config.value_bits()
    }

    pub(crate) fn update_block(
        &mut self,
        block_index: usize,
        values: &[f32],
    ) -> Result<(), KvCacheError> {
        if block_index >= self.blocks.len()
            || values.is_empty()
            || !values.len().is_multiple_of(self.vector_len)
        {
            return Err(KvCacheError::InvalidShape);
        }
        self.blocks[block_index] = Some(QuantizedValueBlock::quantize(
            self.config.value_bits(),
            values,
            self.vector_len,
        )?);
        Ok(())
    }

    pub(crate) fn clear(&mut self) {
        for block in &mut self.blocks {
            *block = None;
        }
    }

    pub(crate) fn dequantized_block(&self, block_index: usize) -> Result<Vec<f32>, KvCacheError> {
        let block = self
            .blocks
            .get(block_index)
            .ok_or(KvCacheError::InvalidShape)?
            .as_ref()
            .ok_or(KvCacheError::InvalidShape)?;
        block.dequantize()
    }

    pub(crate) fn resident_bytes(&self) -> u64 {
        self.payload_bytes().saturating_add(self.metadata_bytes())
    }

    pub(crate) fn payload_bytes(&self) -> u64 {
        self.blocks
            .iter()
            .filter_map(Option::as_ref)
            .map(QuantizedValueBlock::payload_bytes)
            .sum()
    }

    pub(crate) fn metadata_bytes(&self) -> u64 {
        self.blocks
            .iter()
            .filter_map(Option::as_ref)
            .map(QuantizedValueBlock::metadata_bytes)
            .sum()
    }
}

#[derive(Debug, Clone, PartialEq)]
struct QuantizedValueBlock {
    bits: KvCacheValueQuantizationBits,
    vector_len: usize,
    value_count: usize,
    rows: Vec<QuantizedValueRow>,
    payload: Vec<u8>,
}

impl QuantizedValueBlock {
    fn quantize(
        bits: KvCacheValueQuantizationBits,
        values: &[f32],
        vector_len: usize,
    ) -> Result<Self, KvCacheError> {
        if values.is_empty() || vector_len == 0 || !values.len().is_multiple_of(vector_len) {
            return Err(KvCacheError::InvalidShape);
        }
        let mut rows = Vec::with_capacity(values.len() / vector_len);
        let mut payload = Vec::with_capacity(bits.payload_bytes(values.len())?);
        for row in values.chunks_exact(vector_len) {
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            for value in row {
                if !value.is_finite() {
                    return Err(KvCacheError::NonFiniteValue);
                }
                min = min.min(*value);
                max = max.max(*value);
            }

            let max_code = bits.level_count().saturating_sub(1) as u16;
            let scale = if min == max {
                0.0
            } else {
                (max - min) / f32::from(max_code)
            };
            let mut codes = Vec::with_capacity(row.len());
            for value in row {
                let code = if scale == 0.0 {
                    0
                } else {
                    ((*value - min) / scale)
                        .round()
                        .clamp(0.0, f32::from(max_code)) as u16
                };
                codes.push(code);
            }
            payload.extend_from_slice(&pack_codes(bits, &codes)?);
            rows.push(QuantizedValueRow {
                zero_point: min,
                scale,
            });
        }

        Ok(Self {
            bits,
            vector_len,
            value_count: values.len(),
            rows,
            payload,
        })
    }

    fn dequantize(&self) -> Result<Vec<f32>, KvCacheError> {
        if self.vector_len == 0 || !self.value_count.is_multiple_of(self.vector_len) {
            return Err(KvCacheError::InvalidShape);
        }
        let row_payload_bytes = self.bits.payload_bytes(self.vector_len)?;
        let expected_payload_bytes = row_payload_bytes
            .checked_mul(self.rows.len())
            .ok_or(KvCacheError::InvalidShape)?;
        if self.payload.len() != expected_payload_bytes {
            return Err(KvCacheError::InvalidShape);
        }
        let mut values = Vec::with_capacity(self.value_count);
        for (row_index, row) in self.rows.iter().enumerate() {
            let payload_start = row_index
                .checked_mul(row_payload_bytes)
                .ok_or(KvCacheError::InvalidShape)?;
            let payload_end = payload_start
                .checked_add(row_payload_bytes)
                .ok_or(KvCacheError::InvalidShape)?;
            let codes = unpack_codes(
                self.bits,
                self.vector_len,
                self.payload
                    .get(payload_start..payload_end)
                    .ok_or(KvCacheError::InvalidShape)?,
            )?;
            values.extend(
                codes
                    .iter()
                    .map(|code| row.zero_point + f32::from(*code) * row.scale),
            );
        }
        Ok(values)
    }

    fn payload_bytes(&self) -> u64 {
        self.payload.len() as u64
    }

    fn metadata_bytes(&self) -> u64 {
        let row_metadata_bytes = self
            .rows
            .len()
            .saturating_mul(std::mem::size_of::<QuantizedValueRow>());
        row_metadata_bytes
            .saturating_add(std::mem::size_of::<usize>() * 2)
            .saturating_add(std::mem::size_of::<KvCacheValueQuantizationBits>()) as u64
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct QuantizedValueRow {
    zero_point: f32,
    scale: f32,
}

fn pack_codes(bits: KvCacheValueQuantizationBits, codes: &[u16]) -> Result<Vec<u8>, KvCacheError> {
    let bit_width = bits.bit_width();
    let max_code = bits.level_count().saturating_sub(1) as u16;
    let mut bytes = vec![0_u8; bits.payload_bytes(codes.len())?];
    for (value_index, code) in codes.iter().copied().enumerate() {
        if code > max_code {
            return Err(KvCacheError::InvalidShape);
        }
        let base_bit = value_index
            .checked_mul(bit_width)
            .ok_or(KvCacheError::InvalidShape)?;
        for code_bit in 0..bit_width {
            if ((code >> code_bit) & 1) == 1 {
                let bit_index = base_bit
                    .checked_add(code_bit)
                    .ok_or(KvCacheError::InvalidShape)?;
                let byte = bytes
                    .get_mut(bit_index / 8)
                    .ok_or(KvCacheError::InvalidShape)?;
                *byte |= 1_u8 << (bit_index % 8);
            }
        }
    }
    Ok(bytes)
}

fn unpack_codes(
    bits: KvCacheValueQuantizationBits,
    value_count: usize,
    bytes: &[u8],
) -> Result<Vec<u16>, KvCacheError> {
    if bytes.len() != bits.payload_bytes(value_count)? {
        return Err(KvCacheError::InvalidShape);
    }
    let bit_width = bits.bit_width();
    let mut codes = Vec::with_capacity(value_count);
    for value_index in 0..value_count {
        let base_bit = value_index
            .checked_mul(bit_width)
            .ok_or(KvCacheError::InvalidShape)?;
        let mut code = 0_u16;
        for code_bit in 0..bit_width {
            let bit_index = base_bit
                .checked_add(code_bit)
                .ok_or(KvCacheError::InvalidShape)?;
            let byte = bytes.get(bit_index / 8).ok_or(KvCacheError::InvalidShape)?;
            if ((byte >> (bit_index % 8)) & 1) == 1 {
                code |= 1_u16 << code_bit;
            }
        }
        codes.push(code);
    }
    Ok(codes)
}

fn quantize_int8_rows(
    values: &[f32],
    vector_len: usize,
) -> Result<(Vec<i8>, Vec<f32>), KvCacheError> {
    if values.is_empty() || vector_len == 0 || !values.len().is_multiple_of(vector_len) {
        return Err(KvCacheError::InvalidShape);
    }
    let mut codes = Vec::with_capacity(values.len());
    let mut scales = Vec::with_capacity(values.len() / vector_len);
    for row in values.chunks_exact(vector_len) {
        let mut max_abs = 0.0_f32;
        for value in row {
            if !value.is_finite() {
                return Err(KvCacheError::NonFiniteValue);
            }
            max_abs = max_abs.max(value.abs());
        }
        let scale = if max_abs == 0.0 {
            0.0
        } else {
            max_abs / f32::from(i8::MAX)
        };
        for value in row {
            let code = if scale == 0.0 {
                0
            } else {
                (*value / scale)
                    .round()
                    .clamp(f32::from(-i8::MAX), f32::from(i8::MAX)) as i8
            };
            codes.push(code);
        }
        scales.push(scale);
    }
    Ok((codes, scales))
}

fn dequantize_int8_rows(
    codes: &[i8],
    scales: &[f32],
    vector_len: usize,
) -> Result<Vec<f32>, KvCacheError> {
    if vector_len == 0 || !codes.len().is_multiple_of(vector_len) {
        return Err(KvCacheError::InvalidShape);
    }
    let token_count = codes.len() / vector_len;
    if scales.len() != token_count {
        return Err(KvCacheError::ShapeMismatch {
            expected: token_count,
            actual: scales.len(),
        });
    }
    let mut values = Vec::with_capacity(codes.len());
    for (row, scale) in codes.chunks_exact(vector_len).zip(scales) {
        values.extend(row.iter().map(|code| f32::from(*code) * *scale));
    }
    Ok(values)
}
