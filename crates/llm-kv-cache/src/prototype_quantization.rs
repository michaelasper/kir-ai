//! Offline KV value-cache quantization prototype for COR-292.
//!
//! This module is intentionally disconnected from serving defaults. It exists
//! to compare random orthogonal rotation plus Lloyd-Max/codebook quantization
//! against simple uniform INT8/INT4/3-bit baselines on deterministic fixtures.
//! Asymmetric/TurboQuant-style cache quantization remains deferred to #334; the
//! COR-296 serving path only enables symmetric per-token INT8 KV storage.

use std::{fmt, hint::black_box, time::Instant};

const DECODE_BENCH_REPETITIONS: u64 = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum KvQuantizationBits {
    Three,
    Four,
    Eight,
}

impl KvQuantizationBits {
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

    pub fn payload_bytes(self, value_count: usize) -> u64 {
        value_count
            .saturating_mul(self.bit_width())
            .div_ceil(8)
            .try_into()
            .unwrap_or(u64::MAX)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum KvQuantizationScheme {
    UniformAffine,
    LloydMaxCodebook,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum KvQuantizationScope {
    ModelFamily,
    Layer,
    LayerHead,
    Block,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvQuantizationIdentity {
    model_family: String,
    layer_index: usize,
    head_index: usize,
    block_index: usize,
    vector_len: usize,
}

impl KvQuantizationIdentity {
    pub fn new(
        model_family: impl Into<String>,
        layer_index: usize,
        head_index: usize,
        block_index: usize,
        vector_len: usize,
    ) -> Self {
        Self {
            model_family: model_family.into(),
            layer_index,
            head_index,
            block_index,
            vector_len,
        }
    }

    pub fn model_family(&self) -> &str {
        &self.model_family
    }

    pub fn layer_index(&self) -> usize {
        self.layer_index
    }

    pub fn head_index(&self) -> usize {
        self.head_index
    }

    pub fn block_index(&self) -> usize {
        self.block_index
    }

    pub fn vector_len(&self) -> usize {
        self.vector_len
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvRotationMetadata {
    scope: KvQuantizationScope,
    seed: u64,
    vector_len: usize,
    fingerprint: u64,
}

impl KvRotationMetadata {
    pub fn scope(&self) -> KvQuantizationScope {
        self.scope
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn vector_len(&self) -> usize {
        self.vector_len
    }

    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvCodebookMetadata {
    scope: KvQuantizationScope,
    bits: KvQuantizationBits,
    entry_count: usize,
    fingerprint: u64,
}

impl KvCodebookMetadata {
    pub fn scope(&self) -> KvQuantizationScope {
        self.scope
    }

    pub fn bits(&self) -> KvQuantizationBits {
        self.bits
    }

    pub fn entry_count(&self) -> usize {
        self.entry_count
    }

    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct KvQuantizedBlockMetadata {
    identity: KvQuantizationIdentity,
    scheme: KvQuantizationScheme,
    bits: KvQuantizationBits,
    value_count: usize,
    payload_bytes: u64,
    rotation: Option<KvRotationMetadata>,
    codebook: Option<KvCodebookMetadata>,
}

impl KvQuantizedBlockMetadata {
    pub fn identity(&self) -> &KvQuantizationIdentity {
        &self.identity
    }

    pub fn scheme(&self) -> KvQuantizationScheme {
        self.scheme
    }

    pub fn bits(&self) -> KvQuantizationBits {
        self.bits
    }

    pub fn value_count(&self) -> usize {
        self.value_count
    }

    pub fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }

    pub fn rotation(&self) -> Option<&KvRotationMetadata> {
        self.rotation.as_ref()
    }

    pub fn codebook(&self) -> Option<&KvCodebookMetadata> {
        self.codebook.as_ref()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct KvValueQuantizerPrototype {
    identity: KvQuantizationIdentity,
    bits: KvQuantizationBits,
    scheme: KvQuantizationScheme,
    rotation: Option<RandomOrthogonalRotation>,
    codebook: Option<LloydMaxCodebook>,
}

impl KvValueQuantizerPrototype {
    pub fn uniform(
        identity: KvQuantizationIdentity,
        bits: KvQuantizationBits,
        rotation_seed: Option<u64>,
    ) -> Result<Self, KvQuantizationPrototypeError> {
        validate_identity(&identity)?;
        let rotation = rotation_seed
            .map(|seed| RandomOrthogonalRotation::new(identity.vector_len, seed))
            .transpose()?;
        Ok(Self {
            identity,
            bits,
            scheme: KvQuantizationScheme::UniformAffine,
            rotation,
            codebook: None,
        })
    }

    pub fn train_lloyd_max(
        identity: KvQuantizationIdentity,
        bits: KvQuantizationBits,
        rotation_seed: Option<u64>,
        calibration_values: &[f32],
        iterations: usize,
    ) -> Result<Self, KvQuantizationPrototypeError> {
        validate_identity(&identity)?;
        validate_values(calibration_values, identity.vector_len)?;
        let rotation = rotation_seed
            .map(|seed| RandomOrthogonalRotation::new(identity.vector_len, seed))
            .transpose()?;
        let training_values = if let Some(rotation) = rotation.as_ref() {
            rotation.rotate_rows(calibration_values)?
        } else {
            calibration_values.to_vec()
        };
        let codebook = LloydMaxCodebook::train(
            KvQuantizationScope::LayerHead,
            identity.model_family.as_str(),
            identity.layer_index,
            identity.head_index,
            bits,
            &training_values,
            iterations,
        )?;
        Ok(Self {
            identity,
            bits,
            scheme: KvQuantizationScheme::LloydMaxCodebook,
            rotation,
            codebook: Some(codebook),
        })
    }

    pub fn quantize(
        &self,
        values: &[f32],
    ) -> Result<KvQuantizedValueBlock, KvQuantizationPrototypeError> {
        validate_values(values, self.identity.vector_len)?;
        let transformed = if let Some(rotation) = self.rotation.as_ref() {
            rotation.rotate_rows(values)?
        } else {
            values.to_vec()
        };
        let (payload, uniform) = match self.scheme {
            KvQuantizationScheme::UniformAffine => {
                let (codes, uniform) = uniform_quantize(&transformed, self.bits)?;
                (PackedKvPayload::pack(self.bits, &codes)?, Some(uniform))
            }
            KvQuantizationScheme::LloydMaxCodebook => {
                let codebook = self
                    .codebook
                    .as_ref()
                    .ok_or(KvQuantizationPrototypeError::MissingCodebook)?;
                let codes = codebook.quantize(&transformed)?;
                (PackedKvPayload::pack(self.bits, &codes)?, None)
            }
        };
        let metadata = KvQuantizedBlockMetadata {
            identity: self.identity.clone(),
            scheme: self.scheme,
            bits: self.bits,
            value_count: values.len(),
            payload_bytes: payload.bytes.len().try_into().unwrap_or(u64::MAX),
            rotation: self
                .rotation
                .as_ref()
                .map(|rotation| rotation.metadata.clone()),
            codebook: self
                .codebook
                .as_ref()
                .map(|codebook| codebook.metadata.clone()),
        };
        Ok(KvQuantizedValueBlock {
            metadata,
            payload,
            uniform,
        })
    }

    fn expected_rotation_metadata(&self) -> Option<&KvRotationMetadata> {
        self.rotation.as_ref().map(|rotation| &rotation.metadata)
    }

    fn expected_codebook_metadata(&self) -> Option<&KvCodebookMetadata> {
        self.codebook.as_ref().map(|codebook| &codebook.metadata)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct KvQuantizedValueBlock {
    metadata: KvQuantizedBlockMetadata,
    payload: PackedKvPayload,
    uniform: Option<UniformAffineBlock>,
}

impl KvQuantizedValueBlock {
    pub fn metadata(&self) -> &KvQuantizedBlockMetadata {
        &self.metadata
    }

    pub fn payload_bytes(&self) -> u64 {
        self.metadata.payload_bytes
    }

    pub fn dequantize(
        &self,
        quantizer: &KvValueQuantizerPrototype,
    ) -> Result<Vec<f32>, KvQuantizationPrototypeError> {
        self.validate_quantizer_metadata(quantizer)?;
        let codes = self.payload.unpack()?;
        let mut transformed = match self.metadata.scheme {
            KvQuantizationScheme::UniformAffine => {
                let uniform = self
                    .uniform
                    .as_ref()
                    .ok_or(KvQuantizationPrototypeError::MissingUniformAffineMetadata)?;
                uniform.dequantize(&codes)
            }
            KvQuantizationScheme::LloydMaxCodebook => {
                let codebook = quantizer
                    .codebook
                    .as_ref()
                    .ok_or(KvQuantizationPrototypeError::MissingCodebook)?;
                codebook.dequantize(&codes)?
            }
        };
        if let Some(rotation) = quantizer.rotation.as_ref() {
            transformed = rotation.inverse_rotate_rows(&transformed)?;
        }
        Ok(transformed)
    }

    pub fn decode_estimated_ops(&self) -> u64 {
        let scalar_ops = match self.metadata.scheme {
            KvQuantizationScheme::UniformAffine => self.metadata.value_count.saturating_mul(2),
            KvQuantizationScheme::LloydMaxCodebook => self.metadata.value_count,
        };
        let rotation_ops = if self.metadata.rotation.is_some() {
            let rows = self
                .metadata
                .value_count
                .checked_div(self.metadata.identity.vector_len.max(1))
                .unwrap_or(0);
            rows.saturating_mul(
                self.metadata
                    .identity
                    .vector_len
                    .saturating_mul(self.metadata.identity.vector_len)
                    .saturating_mul(2),
            )
        } else {
            0
        };
        scalar_ops
            .saturating_add(rotation_ops)
            .try_into()
            .unwrap_or(u64::MAX)
    }

    fn validate_quantizer_metadata(
        &self,
        quantizer: &KvValueQuantizerPrototype,
    ) -> Result<(), KvQuantizationPrototypeError> {
        if self.metadata.identity != quantizer.identity {
            return Err(KvQuantizationPrototypeError::MetadataMismatch { field: "identity" });
        }
        if self.metadata.scheme != quantizer.scheme {
            return Err(KvQuantizationPrototypeError::MetadataMismatch { field: "scheme" });
        }
        if self.metadata.bits != quantizer.bits {
            return Err(KvQuantizationPrototypeError::MetadataMismatch { field: "bits" });
        }
        if self.metadata.rotation() != quantizer.expected_rotation_metadata() {
            return Err(KvQuantizationPrototypeError::MetadataMismatch { field: "rotation" });
        }
        if self.metadata.codebook() != quantizer.expected_codebook_metadata() {
            return Err(KvQuantizationPrototypeError::MetadataMismatch { field: "codebook" });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct KvQuantizationEvaluationFixture {
    model_family: &'static str,
    vector_len: usize,
    queries: Vec<f32>,
    keys: Vec<f32>,
    values: Vec<f32>,
    score_scale: f32,
}

impl KvQuantizationEvaluationFixture {
    pub fn model_family(&self) -> &str {
        self.model_family
    }

    pub fn vector_len(&self) -> usize {
        self.vector_len
    }

    pub fn queries(&self) -> &[f32] {
        &self.queries
    }

    pub fn keys(&self) -> &[f32] {
        &self.keys
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn score_scale(&self) -> f32 {
        self.score_scale
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct KvQuantizationEvaluationReport {
    fixture_model_family: String,
    original_bytes: u64,
    rows: Vec<KvQuantizationEvaluationRow>,
}

impl KvQuantizationEvaluationReport {
    pub fn fixture_model_family(&self) -> &str {
        &self.fixture_model_family
    }

    pub fn original_bytes(&self) -> u64 {
        self.original_bytes
    }

    pub fn rows(&self) -> &[KvQuantizationEvaluationRow] {
        &self.rows
    }

    pub fn row(&self, name: &str) -> Option<&KvQuantizationEvaluationRow> {
        self.rows.iter().find(|row| row.name == name)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct KvQuantizationEvaluationRow {
    name: &'static str,
    scheme: KvQuantizationScheme,
    bits: KvQuantizationBits,
    rotation: bool,
    payload_bytes: u64,
    payload_memory_ratio: f64,
    reconstruction_mse: f64,
    reconstruction_max_abs: f32,
    attention_output_mse: f64,
    attention_output_max_abs: f32,
    decode_estimated_ops: u64,
    decode_average_nanos: u64,
}

impl KvQuantizationEvaluationRow {
    pub fn name(&self) -> &str {
        self.name
    }

    pub fn scheme(&self) -> KvQuantizationScheme {
        self.scheme
    }

    pub fn bits(&self) -> KvQuantizationBits {
        self.bits
    }

    pub fn uses_rotation(&self) -> bool {
        self.rotation
    }

    pub fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }

    pub fn payload_memory_ratio(&self) -> f64 {
        self.payload_memory_ratio
    }

    pub fn reconstruction_mse(&self) -> f64 {
        self.reconstruction_mse
    }

    pub fn reconstruction_max_abs(&self) -> f32 {
        self.reconstruction_max_abs
    }

    pub fn attention_output_mse(&self) -> f64 {
        self.attention_output_mse
    }

    pub fn attention_output_max_abs(&self) -> f32 {
        self.attention_output_max_abs
    }

    pub fn decode_estimated_ops(&self) -> u64 {
        self.decode_estimated_ops
    }

    pub fn decode_average_nanos(&self) -> u64 {
        self.decode_average_nanos
    }
}

pub fn tiny_qwen_value_fixture() -> KvQuantizationEvaluationFixture {
    tiny_value_fixture("qwen3-tiny", 0.35355338, 0.17)
}

pub fn tiny_gemma_value_fixture() -> KvQuantizationEvaluationFixture {
    tiny_value_fixture("gemma4-tiny", 0.5, -0.11)
}

pub fn evaluate_kv_quantization_fixture(
    fixture: &KvQuantizationEvaluationFixture,
) -> Result<KvQuantizationEvaluationReport, KvQuantizationPrototypeError> {
    validate_values(&fixture.queries, fixture.vector_len)?;
    validate_values(&fixture.keys, fixture.vector_len)?;
    validate_values(&fixture.values, fixture.vector_len)?;
    let original_attention = causal_attention_outputs(
        &fixture.queries,
        &fixture.keys,
        &fixture.values,
        fixture.vector_len,
        fixture.score_scale,
    )?;
    let original_bytes = fixture
        .values
        .len()
        .saturating_mul(std::mem::size_of::<f32>())
        .try_into()
        .unwrap_or(u64::MAX);
    let plans = [
        QuantizationPlan::uniform("uniform-int8", KvQuantizationBits::Eight),
        QuantizationPlan::uniform("uniform-int4", KvQuantizationBits::Four),
        QuantizationPlan::uniform("uniform-int3", KvQuantizationBits::Three),
        QuantizationPlan::lloyd("lloyd-max-int4", KvQuantizationBits::Four, None),
        QuantizationPlan::lloyd(
            "rotated-lloyd-max-int4",
            KvQuantizationBits::Four,
            Some(0xC0_0292),
        ),
        QuantizationPlan::lloyd("lloyd-max-int3", KvQuantizationBits::Three, None),
        QuantizationPlan::lloyd(
            "rotated-lloyd-max-int3",
            KvQuantizationBits::Three,
            Some(0xC0_3292),
        ),
    ];
    let mut rows = Vec::with_capacity(plans.len());
    for plan in plans {
        let identity =
            KvQuantizationIdentity::new(fixture.model_family, 0, 0, 0, fixture.vector_len);
        let quantizer = match plan.scheme {
            KvQuantizationScheme::UniformAffine => {
                KvValueQuantizerPrototype::uniform(identity, plan.bits, plan.rotation_seed)?
            }
            KvQuantizationScheme::LloydMaxCodebook => KvValueQuantizerPrototype::train_lloyd_max(
                identity,
                plan.bits,
                plan.rotation_seed,
                &fixture.values,
                16,
            )?,
        };
        let block = quantizer.quantize(&fixture.values)?;
        let decoded = block.dequantize(&quantizer)?;
        let decode_average_nanos = average_decode_nanos(&block, &quantizer)?;
        let reconstruction = error_metrics(&fixture.values, &decoded)?;
        let decoded_attention = causal_attention_outputs(
            &fixture.queries,
            &fixture.keys,
            &decoded,
            fixture.vector_len,
            fixture.score_scale,
        )?;
        let attention = error_metrics(&original_attention, &decoded_attention)?;
        let payload_memory_ratio = if original_bytes == 0 {
            0.0
        } else {
            block.payload_bytes() as f64 / original_bytes as f64
        };
        rows.push(KvQuantizationEvaluationRow {
            name: plan.name,
            scheme: plan.scheme,
            bits: plan.bits,
            rotation: plan.rotation_seed.is_some(),
            payload_bytes: block.payload_bytes(),
            payload_memory_ratio,
            reconstruction_mse: reconstruction.mse,
            reconstruction_max_abs: reconstruction.max_abs,
            attention_output_mse: attention.mse,
            attention_output_max_abs: attention.max_abs,
            decode_estimated_ops: block.decode_estimated_ops(),
            decode_average_nanos,
        });
    }
    Ok(KvQuantizationEvaluationReport {
        fixture_model_family: fixture.model_family.to_owned(),
        original_bytes,
        rows,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum KvQuantizationPrototypeError {
    EmptyInput,
    NonFiniteValue,
    ShapeMismatch { expected: usize, actual: usize },
    InvalidPayload,
    InvalidCodebook,
    InvalidRotation,
    MissingCodebook,
    MissingUniformAffineMetadata,
    MetadataMismatch { field: &'static str },
}

impl fmt::Display for KvQuantizationPrototypeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(formatter, "KV quantization input must not be empty"),
            Self::NonFiniteValue => write!(
                formatter,
                "KV quantization prototype only accepts finite f32 values"
            ),
            Self::ShapeMismatch { expected, actual } => {
                write!(
                    formatter,
                    "KV quantization shape mismatch: expected {expected}, got {actual}"
                )
            }
            Self::InvalidPayload => write!(formatter, "KV quantization payload is invalid"),
            Self::InvalidCodebook => write!(formatter, "KV quantization codebook is invalid"),
            Self::InvalidRotation => write!(formatter, "KV quantization rotation is invalid"),
            Self::MissingCodebook => write!(formatter, "KV quantization codebook is missing"),
            Self::MissingUniformAffineMetadata => {
                write!(
                    formatter,
                    "KV quantization uniform affine metadata is missing"
                )
            }
            Self::MetadataMismatch { field } => {
                write!(formatter, "KV quantization {field} metadata does not match")
            }
        }
    }
}

impl std::error::Error for KvQuantizationPrototypeError {}

#[derive(Debug, Clone, PartialEq)]
struct QuantizationPlan {
    name: &'static str,
    scheme: KvQuantizationScheme,
    bits: KvQuantizationBits,
    rotation_seed: Option<u64>,
}

impl QuantizationPlan {
    fn uniform(name: &'static str, bits: KvQuantizationBits) -> Self {
        Self {
            name,
            scheme: KvQuantizationScheme::UniformAffine,
            bits,
            rotation_seed: None,
        }
    }

    fn lloyd(name: &'static str, bits: KvQuantizationBits, rotation_seed: Option<u64>) -> Self {
        Self {
            name,
            scheme: KvQuantizationScheme::LloydMaxCodebook,
            bits,
            rotation_seed,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct UniformAffineBlock {
    min: f32,
    scale: f32,
}

impl UniformAffineBlock {
    fn dequantize(&self, codes: &[u16]) -> Vec<f32> {
        codes
            .iter()
            .map(|code| self.min + f32::from(*code) * self.scale)
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PackedKvPayload {
    bits: KvQuantizationBits,
    value_count: usize,
    bytes: Vec<u8>,
}

impl PackedKvPayload {
    fn pack(bits: KvQuantizationBits, codes: &[u16]) -> Result<Self, KvQuantizationPrototypeError> {
        let bit_width = bits.bit_width();
        let max_code = bits.level_count().saturating_sub(1) as u16;
        let mut bytes = vec![
            0_u8;
            bits.payload_bytes(codes.len())
                .try_into()
                .unwrap_or(usize::MAX)
        ];
        for (value_index, code) in codes.iter().copied().enumerate() {
            if code > max_code {
                return Err(KvQuantizationPrototypeError::InvalidPayload);
            }
            let base_bit = value_index.saturating_mul(bit_width);
            for code_bit in 0..bit_width {
                if ((code >> code_bit) & 1) == 1 {
                    let bit_index = base_bit + code_bit;
                    let Some(byte) = bytes.get_mut(bit_index / 8) else {
                        return Err(KvQuantizationPrototypeError::InvalidPayload);
                    };
                    *byte |= 1_u8 << (bit_index % 8);
                }
            }
        }
        Ok(Self {
            bits,
            value_count: codes.len(),
            bytes,
        })
    }

    fn unpack(&self) -> Result<Vec<u16>, KvQuantizationPrototypeError> {
        let expected_bytes: usize = self
            .bits
            .payload_bytes(self.value_count)
            .try_into()
            .unwrap_or(usize::MAX);
        if self.bytes.len() != expected_bytes {
            return Err(KvQuantizationPrototypeError::InvalidPayload);
        }
        let bit_width = self.bits.bit_width();
        let mut codes = Vec::with_capacity(self.value_count);
        for value_index in 0..self.value_count {
            let base_bit = value_index.saturating_mul(bit_width);
            let mut code = 0_u16;
            for code_bit in 0..bit_width {
                let bit_index = base_bit + code_bit;
                let Some(byte) = self.bytes.get(bit_index / 8) else {
                    return Err(KvQuantizationPrototypeError::InvalidPayload);
                };
                if ((byte >> (bit_index % 8)) & 1) == 1 {
                    code |= 1_u16 << code_bit;
                }
            }
            codes.push(code);
        }
        Ok(codes)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct LloydMaxCodebook {
    entries: Vec<f32>,
    metadata: KvCodebookMetadata,
}

impl LloydMaxCodebook {
    fn train(
        scope: KvQuantizationScope,
        model_family: &str,
        layer_index: usize,
        head_index: usize,
        bits: KvQuantizationBits,
        values: &[f32],
        iterations: usize,
    ) -> Result<Self, KvQuantizationPrototypeError> {
        if values.is_empty() {
            return Err(KvQuantizationPrototypeError::EmptyInput);
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(KvQuantizationPrototypeError::NonFiniteValue);
        }
        let entry_count = bits.level_count();
        let mut sorted = values.to_vec();
        sorted.sort_by(f32::total_cmp);
        let mut entries = (0..entry_count)
            .map(|index| {
                let quantile_index = index
                    .saturating_mul(sorted.len())
                    .saturating_add(sorted.len() / 2)
                    / entry_count;
                sorted[quantile_index.min(sorted.len() - 1)]
            })
            .collect::<Vec<_>>();

        for _ in 0..iterations.max(1) {
            let mut sums = vec![0.0_f64; entry_count];
            let mut counts = vec![0_usize; entry_count];
            for value in values {
                let index = nearest_entry_index(&entries, *value)?;
                sums[index] += f64::from(*value);
                counts[index] += 1;
            }
            for (index, entry) in entries.iter_mut().enumerate() {
                if counts[index] > 0 {
                    *entry = (sums[index] / counts[index] as f64) as f32;
                }
            }
            entries.sort_by(f32::total_cmp);
        }

        let metadata = KvCodebookMetadata {
            scope,
            bits,
            entry_count,
            fingerprint: codebook_fingerprint(
                scope,
                model_family,
                layer_index,
                head_index,
                bits,
                &entries,
            ),
        };
        Ok(Self { entries, metadata })
    }

    fn quantize(&self, values: &[f32]) -> Result<Vec<u16>, KvQuantizationPrototypeError> {
        values
            .iter()
            .map(|value| {
                nearest_entry_index(&self.entries, *value).and_then(|index| {
                    index
                        .try_into()
                        .map_err(|_| KvQuantizationPrototypeError::InvalidCodebook)
                })
            })
            .collect()
    }

    fn dequantize(&self, codes: &[u16]) -> Result<Vec<f32>, KvQuantizationPrototypeError> {
        codes
            .iter()
            .map(|code| {
                self.entries
                    .get(usize::from(*code))
                    .copied()
                    .ok_or(KvQuantizationPrototypeError::InvalidPayload)
            })
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq)]
struct RandomOrthogonalRotation {
    metadata: KvRotationMetadata,
    matrix: Vec<f32>,
}

impl RandomOrthogonalRotation {
    fn new(vector_len: usize, seed: u64) -> Result<Self, KvQuantizationPrototypeError> {
        if vector_len == 0 {
            return Err(KvQuantizationPrototypeError::InvalidRotation);
        }
        let matrix = deterministic_orthonormal_matrix(vector_len, seed)?;
        let metadata = KvRotationMetadata {
            scope: KvQuantizationScope::LayerHead,
            seed,
            vector_len,
            fingerprint: rotation_fingerprint(KvQuantizationScope::LayerHead, seed, vector_len),
        };
        Ok(Self { metadata, matrix })
    }

    fn rotate_rows(&self, values: &[f32]) -> Result<Vec<f32>, KvQuantizationPrototypeError> {
        self.validate_values(values)?;
        let mut output = vec![0.0; values.len()];
        for (row_index, row) in values.chunks_exact(self.metadata.vector_len).enumerate() {
            let output_row_start = row_index * self.metadata.vector_len;
            for rotated_col in 0..self.metadata.vector_len {
                let matrix_start = rotated_col * self.metadata.vector_len;
                let mut sum = 0.0_f32;
                for (source_col, value) in row.iter().enumerate() {
                    sum += self.matrix[matrix_start + source_col] * value;
                }
                output[output_row_start + rotated_col] = sum;
            }
        }
        Ok(output)
    }

    fn inverse_rotate_rows(
        &self,
        values: &[f32],
    ) -> Result<Vec<f32>, KvQuantizationPrototypeError> {
        self.validate_values(values)?;
        let mut output = vec![0.0; values.len()];
        for (row_index, row) in values.chunks_exact(self.metadata.vector_len).enumerate() {
            let output_row_start = row_index * self.metadata.vector_len;
            for source_col in 0..self.metadata.vector_len {
                let mut sum = 0.0_f32;
                for (rotated_col, value) in row.iter().enumerate() {
                    sum += self.matrix[rotated_col * self.metadata.vector_len + source_col] * value;
                }
                output[output_row_start + source_col] = sum;
            }
        }
        Ok(output)
    }

    fn validate_values(&self, values: &[f32]) -> Result<(), KvQuantizationPrototypeError> {
        validate_values(values, self.metadata.vector_len)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ErrorMetrics {
    mse: f64,
    max_abs: f32,
}

#[derive(Debug)]
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    fn next_f32(&mut self) -> f32 {
        self.state ^= self.state << 7;
        self.state ^= self.state >> 9;
        self.state = self.state.wrapping_mul(0xA24B_AED4_963E_E407);
        let mantissa = (self.state >> 41) as u32;
        (mantissa as f32 / ((1_u32 << 23) as f32)) * 2.0 - 1.0
    }
}

fn validate_identity(
    identity: &KvQuantizationIdentity,
) -> Result<(), KvQuantizationPrototypeError> {
    if identity.model_family.is_empty() || identity.vector_len == 0 {
        return Err(KvQuantizationPrototypeError::InvalidCodebook);
    }
    Ok(())
}

fn validate_values(values: &[f32], vector_len: usize) -> Result<(), KvQuantizationPrototypeError> {
    if values.is_empty() {
        return Err(KvQuantizationPrototypeError::EmptyInput);
    }
    if vector_len == 0 || !values.len().is_multiple_of(vector_len) {
        return Err(KvQuantizationPrototypeError::ShapeMismatch {
            expected: vector_len,
            actual: values.len(),
        });
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(KvQuantizationPrototypeError::NonFiniteValue);
    }
    Ok(())
}

fn uniform_quantize(
    values: &[f32],
    bits: KvQuantizationBits,
) -> Result<(Vec<u16>, UniformAffineBlock), KvQuantizationPrototypeError> {
    if values.is_empty() {
        return Err(KvQuantizationPrototypeError::EmptyInput);
    }
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !min.is_finite() || !max.is_finite() {
        return Err(KvQuantizationPrototypeError::NonFiniteValue);
    }
    let max_code = bits.level_count().saturating_sub(1) as u16;
    let scale = if min == max {
        0.0
    } else {
        (max - min) / f32::from(max_code)
    };
    let codes = values
        .iter()
        .map(|value| {
            if scale == 0.0 {
                0
            } else {
                ((*value - min) / scale)
                    .round()
                    .clamp(0.0, f32::from(max_code)) as u16
            }
        })
        .collect();
    Ok((codes, UniformAffineBlock { min, scale }))
}

fn nearest_entry_index(entries: &[f32], value: f32) -> Result<usize, KvQuantizationPrototypeError> {
    if entries.is_empty() {
        return Err(KvQuantizationPrototypeError::InvalidCodebook);
    }
    if !value.is_finite() {
        return Err(KvQuantizationPrototypeError::NonFiniteValue);
    }
    let mut nearest = 0_usize;
    let mut nearest_distance = (value - entries[0]).abs();
    for (index, entry) in entries.iter().copied().enumerate().skip(1) {
        let distance = (value - entry).abs();
        if distance < nearest_distance {
            nearest = index;
            nearest_distance = distance;
        }
    }
    Ok(nearest)
}

fn deterministic_orthonormal_matrix(
    dim: usize,
    seed: u64,
) -> Result<Vec<f32>, KvQuantizationPrototypeError> {
    let mut rng = DeterministicRng::new(seed);
    let mut matrix = Vec::with_capacity(dim.saturating_mul(dim));
    for row_index in 0..dim {
        let mut row = Vec::new();
        let mut accepted = false;
        for _ in 0..32 {
            row = (0..dim).map(|_| rng.next_f32()).collect::<Vec<_>>();
            orthogonalize_candidate(&mut row, &matrix, row_index, dim);
            if normalize_candidate(&mut row) {
                accepted = true;
                break;
            }
        }
        if !accepted {
            row = fallback_basis_row(&matrix, row_index, dim)?;
        }
        matrix.extend_from_slice(&row);
    }
    Ok(matrix)
}

fn orthogonalize_candidate(
    candidate: &mut [f32],
    matrix: &[f32],
    previous_rows: usize,
    dim: usize,
) {
    for previous_row in 0..previous_rows {
        let previous = &matrix[previous_row * dim..previous_row * dim + dim];
        let projection = dot(candidate, previous);
        for (value, previous_value) in candidate.iter_mut().zip(previous) {
            *value -= projection * previous_value;
        }
    }
}

fn normalize_candidate(candidate: &mut [f32]) -> bool {
    let norm = dot(candidate, candidate).sqrt();
    if norm <= 1e-6 {
        return false;
    }
    for value in candidate {
        *value /= norm;
    }
    true
}

fn fallback_basis_row(
    matrix: &[f32],
    row_index: usize,
    dim: usize,
) -> Result<Vec<f32>, KvQuantizationPrototypeError> {
    for basis_index in 0..dim {
        let mut row = vec![0.0; dim];
        row[basis_index] = 1.0;
        orthogonalize_candidate(&mut row, matrix, row_index, dim);
        if normalize_candidate(&mut row) {
            return Ok(row);
        }
    }
    Err(KvQuantizationPrototypeError::InvalidRotation)
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn error_metrics(
    expected: &[f32],
    actual: &[f32],
) -> Result<ErrorMetrics, KvQuantizationPrototypeError> {
    if expected.len() != actual.len() {
        return Err(KvQuantizationPrototypeError::ShapeMismatch {
            expected: expected.len(),
            actual: actual.len(),
        });
    }
    if expected.is_empty() {
        return Err(KvQuantizationPrototypeError::EmptyInput);
    }
    let mut squared_error = 0.0_f64;
    let mut max_abs = 0.0_f32;
    for (expected, actual) in expected.iter().zip(actual) {
        let delta = expected - actual;
        squared_error += f64::from(delta * delta);
        max_abs = max_abs.max(delta.abs());
    }
    Ok(ErrorMetrics {
        mse: squared_error / expected.len() as f64,
        max_abs,
    })
}

fn causal_attention_outputs(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    vector_len: usize,
    score_scale: f32,
) -> Result<Vec<f32>, KvQuantizationPrototypeError> {
    validate_values(queries, vector_len)?;
    validate_values(keys, vector_len)?;
    validate_values(values, vector_len)?;
    if queries.len() != keys.len() || queries.len() != values.len() {
        return Err(KvQuantizationPrototypeError::ShapeMismatch {
            expected: queries.len(),
            actual: keys.len().min(values.len()),
        });
    }
    let row_count = queries.len() / vector_len;
    let mut output = vec![0.0; queries.len()];
    let mut scores = Vec::with_capacity(row_count);
    let mut weights = Vec::with_capacity(row_count);
    for row_index in 0..row_count {
        scores.clear();
        weights.clear();
        let query = row_slice(queries, row_index, vector_len);
        for key_index in 0..=row_index {
            scores.push(dot(query, row_slice(keys, key_index, vector_len)) * score_scale);
        }
        softmax(&scores, &mut weights);
        let output_row_start = row_index * vector_len;
        for value_col in 0..vector_len {
            let mut sum = 0.0_f32;
            for (source_row, weight) in weights.iter().copied().enumerate() {
                sum += weight * row_slice(values, source_row, vector_len)[value_col];
            }
            output[output_row_start + value_col] = sum;
        }
    }
    Ok(output)
}

fn row_slice(values: &[f32], row_index: usize, vector_len: usize) -> &[f32] {
    let start = row_index * vector_len;
    &values[start..start + vector_len]
}

fn softmax(scores: &[f32], output: &mut Vec<f32>) {
    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for score in scores {
        let value = (*score - max_score).exp();
        output.push(value);
        sum += value;
    }
    if sum != 0.0 {
        for value in output {
            *value /= sum;
        }
    }
}

fn average_decode_nanos(
    block: &KvQuantizedValueBlock,
    quantizer: &KvValueQuantizerPrototype,
) -> Result<u64, KvQuantizationPrototypeError> {
    let started = Instant::now();
    let mut checksum = 0.0_f32;
    for _ in 0..DECODE_BENCH_REPETITIONS {
        let decoded = block.dequantize(quantizer)?;
        checksum += decoded.iter().copied().sum::<f32>();
        black_box(&decoded);
    }
    black_box(checksum);
    let nanos = started.elapsed().as_nanos() / u128::from(DECODE_BENCH_REPETITIONS);
    Ok(nanos.try_into().unwrap_or(u64::MAX))
}

fn tiny_value_fixture(
    model_family: &'static str,
    score_scale: f32,
    phase: f32,
) -> KvQuantizationEvaluationFixture {
    let vector_len = 8;
    let row_count = 8;
    let centroids = [-1.25, -0.55, -0.10, 0.32, 0.84, 1.12, 1.62, 2.05];
    let mut queries = Vec::with_capacity(row_count * vector_len);
    let mut keys = Vec::with_capacity(row_count * vector_len);
    let mut values = Vec::with_capacity(row_count * vector_len);
    for row in 0..row_count {
        for col in 0..vector_len {
            let x = row as f32 + phase;
            let y = col as f32 - phase;
            queries.push((x * 0.37 + y * 0.11).sin());
            keys.push((x * 0.19 - y * 0.23).cos());
            let centroid = centroids[(row + col * 3) % centroids.len()];
            let jitter = (x * 0.29 + y * 0.41).sin() * 0.035 + (x * 0.13 - y * 0.17).cos() * 0.015;
            values.push(centroid + jitter);
        }
    }
    KvQuantizationEvaluationFixture {
        model_family,
        vector_len,
        queries,
        keys,
        values,
        score_scale,
    }
}

fn rotation_fingerprint(scope: KvQuantizationScope, seed: u64, vector_len: usize) -> u64 {
    let mut hash = StableHash::new();
    hash.update_str("kir-ai-kv-random-orthogonal-rotation/v1");
    hash.update_u64(scope_tag(scope));
    hash.update_u64(seed);
    hash.update_u64(vector_len as u64);
    hash.finish()
}

fn codebook_fingerprint(
    scope: KvQuantizationScope,
    model_family: &str,
    layer_index: usize,
    head_index: usize,
    bits: KvQuantizationBits,
    entries: &[f32],
) -> u64 {
    let mut hash = StableHash::new();
    hash.update_str("kir-ai-kv-lloyd-max-codebook/v1");
    hash.update_u64(scope_tag(scope));
    hash.update_str(model_family);
    hash.update_u64(layer_index as u64);
    hash.update_u64(head_index as u64);
    hash.update_u64(bits.bit_width() as u64);
    hash.update_u64(entries.len() as u64);
    for entry in entries {
        hash.update_u64(u64::from(entry.to_bits()));
    }
    hash.finish()
}

fn scope_tag(scope: KvQuantizationScope) -> u64 {
    match scope {
        KvQuantizationScope::ModelFamily => 1,
        KvQuantizationScope::Layer => 2,
        KvQuantizationScope::LayerHead => 3,
        KvQuantizationScope::Block => 4,
    }
}

#[derive(Debug, Clone, Copy)]
struct StableHash {
    value: u64,
}

impl StableHash {
    fn new() -> Self {
        Self {
            value: 0xcbf2_9ce4_8422_2325,
        }
    }

    fn update_str(&mut self, value: &str) {
        self.update_u64(value.len() as u64);
        for byte in value.as_bytes() {
            self.update_byte(*byte);
        }
    }

    fn update_u64(&mut self, value: u64) {
        for byte in value.to_le_bytes() {
            self.update_byte(byte);
        }
    }

    fn update_byte(&mut self, byte: u8) {
        self.value ^= u64::from(byte);
        self.value = self.value.wrapping_mul(0x0000_0100_0000_01B3);
    }

    fn finish(self) -> u64 {
        self.value
    }
}
