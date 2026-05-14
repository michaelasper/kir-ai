#![allow(dead_code)]
// Attention helpers include crate-private reference paths used by backend tests
// and native parity probes, not only the production call graph.

use super::math::{MathError, require_len, sigmoid_f32};
use super::{
    LayerKvCache, NativeBatchedMatvecOutput, NativeKvCacheTensor, NativeMatvecBackend,
    SafeTensorShardStore,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NativeFullAttentionDims {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
}

impl NativeFullAttentionDims {
    pub fn attention_dim(self) -> Result<usize, MathError> {
        self.num_attention_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| MathError::InvalidShape("attention dimension overflow".to_owned()))
    }

    pub fn key_value_dim(self) -> Result<usize, MathError> {
        self.num_key_value_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| MathError::InvalidShape("KV dimension overflow".to_owned()))
    }
}

#[derive(Debug, Clone, Copy)]
struct NativeFullAttentionShape {
    attention_dim: usize,
    key_value_dim: usize,
    groups: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct NativeF32Rows<'a>(NativeF32RowsInner<'a>);

#[derive(Debug, Clone, Copy)]
enum NativeF32RowsInner<'a> {
    Rows(&'a [Vec<f32>]),
    Flat { values: &'a [f32], row_len: usize },
}

impl<'a> NativeF32Rows<'a> {
    pub fn from_rows(rows: &'a [Vec<f32>]) -> Self {
        Self(NativeF32RowsInner::Rows(rows))
    }

    pub fn flat(values: &'a [f32], row_len: usize) -> Result<Self, MathError> {
        if row_len == 0 {
            if values.is_empty() {
                return Ok(Self(NativeF32RowsInner::Flat { values, row_len }));
            }
            return Err(MathError::InvalidShape(
                "flat row length must be non-zero for non-empty values".to_owned(),
            ));
        }
        if !values.len().is_multiple_of(row_len) {
            return Err(MathError::InvalidShape(format!(
                "flat row values length {} must be divisible by row length {row_len}",
                values.len()
            )));
        }
        Ok(Self(NativeF32RowsInner::Flat { values, row_len }))
    }

    pub fn from_batched_matvec(output: &'a NativeBatchedMatvecOutput) -> Result<Self, MathError> {
        Self::flat(output.values(), output.row_len())
    }

    pub fn len(self) -> usize {
        match self.0 {
            NativeF32RowsInner::Rows(rows) => rows.len(),
            NativeF32RowsInner::Flat { values, row_len } => {
                values.len().checked_div(row_len).unwrap_or(0)
            }
        }
    }

    pub fn is_empty(self) -> bool {
        self.len() == 0
    }

    pub fn row(self, index: usize) -> &'a [f32] {
        match self.0 {
            NativeF32RowsInner::Rows(rows) => rows[index].as_slice(),
            NativeF32RowsInner::Flat { values, row_len } => {
                let start = index * row_len;
                &values[start..start + row_len]
            }
        }
    }
}

pub(crate) struct NativeFullAttentionSequenceParts<'a> {
    pub queries: NativeF32Rows<'a>,
    pub keys: NativeF32Rows<'a>,
    pub values: NativeF32Rows<'a>,
    pub gates: Option<NativeF32Rows<'a>>,
    pub output_projection: NativeOutputProjection<'a>,
    pub score_scale: f32,
}

pub(crate) struct NativeFullAttentionStepParts<'a> {
    pub query: &'a [f32],
    pub key: &'a [f32],
    pub value: &'a [f32],
    pub gate: Option<&'a [f32]>,
    pub output_projection: NativeOutputProjection<'a>,
    pub score_scale: f32,
}

pub(crate) struct NativeFullAttentionCacheSequenceParts<'a> {
    pub queries: NativeF32Rows<'a>,
    pub gates: Option<NativeF32Rows<'a>>,
    pub source_counts: &'a [usize],
    pub output_projection: NativeOutputProjection<'a>,
    pub score_scale: f32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum NativeOutputProjection<'a> {
    F32(&'a [f32]),
    Bf16Tensor {
        store: &'a SafeTensorShardStore,
        tensor: &'a str,
    },
}

struct NativeFullAttentionInlineSource<'a> {
    keys: NativeF32Rows<'a>,
    values: NativeF32Rows<'a>,
    count: usize,
}

#[derive(Debug, Clone, Copy)]
struct NativeFullAttentionMixInput<'a> {
    query: &'a [f32],
    gate: Option<&'a [f32]>,
    score_scale: f32,
}

#[derive(Debug, Clone, Copy)]
struct NativeFullAttentionCacheSource<'a> {
    cache: &'a LayerKvCache,
    count: usize,
}

pub(crate) async fn native_full_attention_sequence_from_parts(
    dims: NativeFullAttentionDims,
    parts: &NativeFullAttentionSequenceParts<'_>,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<Vec<f32>>, MathError> {
    native_full_attention_sequence_impl(dims, parts, None, matvec).await
}

pub(crate) async fn native_full_attention_sequence_with_cache_from_parts(
    dims: NativeFullAttentionDims,
    parts: &NativeFullAttentionSequenceParts<'_>,
    cache: &mut LayerKvCache,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<Vec<f32>>, MathError> {
    native_full_attention_sequence_impl(dims, parts, Some(cache), matvec).await
}

pub(crate) async fn native_full_attention_sequence_from_cache_parts(
    dims: NativeFullAttentionDims,
    parts: &NativeFullAttentionCacheSequenceParts<'_>,
    cache: &LayerKvCache,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<Vec<f32>>, MathError> {
    if parts.queries.is_empty() {
        return Ok(Vec::new());
    }
    let shape = validate_full_attention_shape(dims)?;
    require_full_attention_cache_shape(dims, cache)?;
    let seq_len = parts.queries.len();
    if parts.source_counts.len() != seq_len {
        return Err(MathError::InvalidShape(
            "full attention cache source counts must match queries".to_owned(),
        ));
    }
    if let Some(gates) = parts.gates
        && gates.len() != seq_len
    {
        return Err(MathError::InvalidShape(
            "full attention gate sequence length must match queries".to_owned(),
        ));
    }
    require_native_output_projection_shape(
        parts.output_projection,
        dims.hidden_size,
        shape.attention_dim,
    )?;
    for token_idx in 0..seq_len {
        require_len(
            "query projection",
            parts.queries.row(token_idx).len(),
            shape.attention_dim,
        )?;
        if let Some(gates) = parts.gates {
            require_len(
                "gate projection",
                gates.row(token_idx).len(),
                shape.attention_dim,
            )?;
        }
    }

    let mut outputs = Vec::with_capacity(seq_len);
    for token_idx in 0..seq_len {
        let source_count = parts.source_counts[token_idx];
        if source_count > cache.token_count() {
            return Err(MathError::InvalidShape(format!(
                "full attention source count {source_count} exceeds cache token count {}",
                cache.token_count()
            )));
        }
        let attended = native_full_attention_mix_from_cache(
            dims,
            shape,
            NativeFullAttentionMixInput {
                query: parts.queries.row(token_idx),
                gate: parts.gates.map(|gates| gates.row(token_idx)),
                score_scale: parts.score_scale,
            },
            NativeFullAttentionCacheSource {
                cache,
                count: source_count,
            },
            matvec,
        )
        .await?;
        outputs.push(
            native_output_projection_unchecked(
                matvec,
                parts.output_projection,
                &attended,
                dims.hidden_size,
                shape.attention_dim,
            )
            .await?,
        );
    }
    Ok(outputs)
}

pub(crate) async fn native_full_attention_step_with_cache_from_parts(
    dims: NativeFullAttentionDims,
    parts: &NativeFullAttentionStepParts<'_>,
    cache: &mut LayerKvCache,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, MathError> {
    let mut output = vec![0.0; dims.hidden_size];
    native_full_attention_step_with_cache_from_parts_in_place(
        dims,
        parts,
        cache,
        matvec,
        &mut output,
    )
    .await?;
    Ok(output)
}

pub(crate) async fn native_full_attention_step_with_cache_from_parts_in_place(
    dims: NativeFullAttentionDims,
    parts: &NativeFullAttentionStepParts<'_>,
    cache: &mut LayerKvCache,
    matvec: &impl NativeMatvecBackend,
    output: &mut [f32],
) -> Result<(), MathError> {
    let shape = validate_full_attention_shape(dims)?;
    require_full_attention_cache_shape(dims, cache)?;
    require_full_attention_token_parts(
        dims,
        shape,
        parts.query,
        parts.key,
        parts.value,
        parts.gate,
    )?;
    require_native_output_projection_shape(
        parts.output_projection,
        dims.hidden_size,
        shape.attention_dim,
    )?;

    cache
        .append_sliding(parts.key, parts.value)
        .map_err(|err| MathError::InvalidShape(format!("KV cache append failed: {err}")))?;
    let attended = native_full_attention_mix_from_cache(
        dims,
        shape,
        NativeFullAttentionMixInput {
            query: parts.query,
            gate: parts.gate,
            score_scale: parts.score_scale,
        },
        NativeFullAttentionCacheSource {
            cache,
            count: cache.token_count(),
        },
        matvec,
    )
    .await?;
    native_output_projection_in_place_unchecked(
        matvec,
        parts.output_projection,
        &attended,
        dims.hidden_size,
        shape.attention_dim,
        output,
    )
    .await
}

async fn native_full_attention_sequence_impl(
    dims: NativeFullAttentionDims,
    parts: &NativeFullAttentionSequenceParts<'_>,
    mut cache: Option<&mut LayerKvCache>,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<Vec<f32>>, MathError> {
    if parts.queries.is_empty() {
        return Ok(Vec::new());
    }
    let shape = validate_full_attention_shape(dims)?;
    let seq_len = parts.queries.len();
    if parts.keys.len() != seq_len || parts.values.len() != seq_len {
        return Err(MathError::InvalidShape(
            "full attention sequence queries, keys, and values must have the same length"
                .to_owned(),
        ));
    }
    if let Some(gates) = parts.gates
        && gates.len() != seq_len
    {
        return Err(MathError::InvalidShape(
            "full attention gate sequence length must match queries".to_owned(),
        ));
    }
    if let Some(cache) = cache.as_ref() {
        require_full_attention_cache_shape(dims, cache)?;
    }
    require_native_output_projection_shape(
        parts.output_projection,
        dims.hidden_size,
        shape.attention_dim,
    )?;
    for token_idx in 0..seq_len {
        require_full_attention_token_parts(
            dims,
            shape,
            parts.queries.row(token_idx),
            parts.keys.row(token_idx),
            parts.values.row(token_idx),
            parts.gates.map(|gates| gates.row(token_idx)),
        )?;
    }

    let mut outputs = Vec::with_capacity(seq_len);
    for token_idx in 0..seq_len {
        let query = parts.queries.row(token_idx);
        let gate = parts.gates.map(|gates| gates.row(token_idx));
        let attended = if let Some(cache) = cache.as_deref_mut() {
            cache
                .append_sliding(parts.keys.row(token_idx), parts.values.row(token_idx))
                .map_err(|err| MathError::InvalidShape(format!("KV cache append failed: {err}")))?;
            native_full_attention_mix_from_cache(
                dims,
                shape,
                NativeFullAttentionMixInput {
                    query,
                    gate,
                    score_scale: parts.score_scale,
                },
                NativeFullAttentionCacheSource {
                    cache,
                    count: cache.token_count(),
                },
                matvec,
            )
            .await?
        } else {
            native_full_attention_mix_from_inline(
                dims,
                shape,
                NativeFullAttentionMixInput {
                    query,
                    gate,
                    score_scale: parts.score_scale,
                },
                NativeFullAttentionInlineSource {
                    keys: parts.keys,
                    values: parts.values,
                    count: token_idx + 1,
                },
                matvec,
            )
            .await?
        };
        outputs.push(
            native_output_projection_unchecked(
                matvec,
                parts.output_projection,
                &attended,
                dims.hidden_size,
                shape.attention_dim,
            )
            .await?,
        );
    }

    Ok(outputs)
}

pub(crate) async fn native_output_projection(
    matvec: &impl NativeMatvecBackend,
    projection: NativeOutputProjection<'_>,
    input: &[f32],
    rows: usize,
    columns: usize,
) -> Result<Vec<f32>, MathError> {
    let mut output = vec![0.0; rows];
    native_output_projection_in_place(matvec, projection, input, rows, columns, &mut output)
        .await?;
    Ok(output)
}

pub(crate) async fn native_output_projection_in_place(
    matvec: &impl NativeMatvecBackend,
    projection: NativeOutputProjection<'_>,
    input: &[f32],
    rows: usize,
    columns: usize,
    output: &mut [f32],
) -> Result<(), MathError> {
    require_native_output_projection_shape(projection, rows, columns)?;
    native_output_projection_in_place_unchecked(matvec, projection, input, rows, columns, output)
        .await
}

pub(crate) async fn native_output_projection_unchecked(
    matvec: &impl NativeMatvecBackend,
    projection: NativeOutputProjection<'_>,
    input: &[f32],
    rows: usize,
    columns: usize,
) -> Result<Vec<f32>, MathError> {
    let mut output = vec![0.0; rows];
    native_output_projection_in_place_unchecked(
        matvec,
        projection,
        input,
        rows,
        columns,
        &mut output,
    )
    .await?;
    Ok(output)
}

pub(crate) async fn native_output_projection_in_place_unchecked(
    matvec: &impl NativeMatvecBackend,
    projection: NativeOutputProjection<'_>,
    input: &[f32],
    rows: usize,
    columns: usize,
    output: &mut [f32],
) -> Result<(), MathError> {
    match projection {
        NativeOutputProjection::F32(weights) => matvec
            .matvec_row_major_f32_in_place(input, weights, rows, columns, output)
            .await
            .map_err(|err| MathError::InvalidShape(format!("output projection failed: {err}"))),
        NativeOutputProjection::Bf16Tensor { store, tensor } => matvec
            .bf16_matvec_row_major_f32_in_place(store, tensor, input, output)
            .await
            .map_err(|err| MathError::InvalidShape(format!("output projection failed: {err}"))),
    }
}

pub(crate) fn require_native_output_projection_shape(
    projection: NativeOutputProjection<'_>,
    rows: usize,
    columns: usize,
) -> Result<(), MathError> {
    match projection {
        NativeOutputProjection::F32(weights) => require_len(
            "output projection weight",
            weights.len(),
            rows.checked_mul(columns).ok_or_else(|| {
                MathError::InvalidShape("output projection shape overflow".to_owned())
            })?,
        ),
        NativeOutputProjection::Bf16Tensor { store, tensor } => {
            let metadata = store.tensor_metadata(tensor).map_err(|err| {
                MathError::InvalidShape(format!("output projection failed: {err}"))
            })?;
            if metadata.dtype != "BF16" {
                return Err(MathError::InvalidShape(format!(
                    "output projection failed: tensor `{tensor}` has dtype {}, expected BF16",
                    metadata.dtype
                )));
            }
            if metadata.shape.as_slice() != [rows, columns] {
                return Err(MathError::InvalidShape(format!(
                    "output projection failed: tensor `{tensor}` shape {:?} must be [{rows}, {columns}]",
                    metadata.shape
                )));
            }
            Ok(())
        }
    }
}

fn validate_full_attention_shape(
    dims: NativeFullAttentionDims,
) -> Result<NativeFullAttentionShape, MathError> {
    if dims.num_attention_heads == 0
        || dims.num_key_value_heads == 0
        || dims.head_dim == 0
        || dims.hidden_size == 0
    {
        return Err(MathError::InvalidShape(
            "full attention dimensions must be non-zero".to_owned(),
        ));
    }
    if !dims
        .num_attention_heads
        .is_multiple_of(dims.num_key_value_heads)
    {
        return Err(MathError::InvalidShape(
            "attention heads must be divisible by key/value heads".to_owned(),
        ));
    }
    Ok(NativeFullAttentionShape {
        attention_dim: dims.attention_dim()?,
        key_value_dim: dims.key_value_dim()?,
        groups: dims.num_attention_heads / dims.num_key_value_heads,
    })
}

fn require_full_attention_token_parts(
    dims: NativeFullAttentionDims,
    shape: NativeFullAttentionShape,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    gate: Option<&[f32]>,
) -> Result<(), MathError> {
    require_len("query projection", query.len(), shape.attention_dim)?;
    require_len("key projection", key.len(), shape.key_value_dim)?;
    require_len("value projection", value.len(), shape.key_value_dim)?;
    if let Some(gate) = gate {
        require_len("gate projection", gate.len(), shape.attention_dim)?;
    }
    if dims.hidden_size == 0 {
        return Err(MathError::InvalidShape(
            "full attention hidden size must be non-zero".to_owned(),
        ));
    }
    Ok(())
}

fn require_full_attention_cache_shape(
    dims: NativeFullAttentionDims,
    cache: &LayerKvCache,
) -> Result<(), MathError> {
    if cache.key_value_heads() != dims.num_key_value_heads || cache.head_dim() != dims.head_dim {
        return Err(MathError::InvalidShape(format!(
            "full attention cache shape does not match dims: cache key_value_heads={}, head_dim={}; dims key_value_heads={}, head_dim={}",
            cache.key_value_heads(),
            cache.head_dim(),
            dims.num_key_value_heads,
            dims.head_dim
        )));
    }
    Ok(())
}

async fn native_full_attention_mix_from_cache(
    dims: NativeFullAttentionDims,
    shape: NativeFullAttentionShape,
    input: NativeFullAttentionMixInput<'_>,
    source: NativeFullAttentionCacheSource<'_>,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, MathError> {
    let mut attended = vec![0.0; shape.attention_dim];
    if matvec
        .full_attention_cache_mix_f32_in_place(
            source.cache,
            input.query,
            source.count,
            dims.num_attention_heads,
            dims.num_key_value_heads,
            dims.head_dim,
            input.score_scale,
            &mut attended,
        )
        .await?
    {
        apply_attention_gate(input.gate, &mut attended);
        return Ok(attended);
    }
    for head in 0..dims.num_attention_heads {
        let kv_head = head / shape.groups;
        let q_start = head * dims.head_dim;
        let kv_start = kv_head * dims.head_dim;
        let key_rows = matvec
            .select_kv_cache_head_rows_f32(
                source.cache,
                NativeKvCacheTensor::Key,
                source.count,
                kv_start,
                dims.head_dim,
            )
            .await?;
        let scores = scaled_full_attention_scores(
            &input.query[q_start..q_start + dims.head_dim],
            &key_rows,
            source.count,
            input.score_scale,
            matvec,
        )
        .await?;
        let weights = matvec.softmax_f32(&scores).await?;
        let value_rows = matvec
            .select_kv_cache_head_rows_f32(
                source.cache,
                NativeKvCacheTensor::Value,
                source.count,
                kv_start,
                dims.head_dim,
            )
            .await?;
        let mixed = matvec
            .weighted_sum_f32(&value_rows, &weights, dims.head_dim)
            .await?;
        for offset in 0..dims.head_dim {
            let gate = input
                .gate
                .map_or(1.0, |gate| sigmoid_f32(gate[q_start + offset]));
            attended[q_start + offset] = mixed[offset] * gate;
        }
    }
    Ok(attended)
}

fn apply_attention_gate(gate: Option<&[f32]>, attended: &mut [f32]) {
    if let Some(gate) = gate {
        for (value, gate) in attended.iter_mut().zip(gate) {
            *value *= sigmoid_f32(*gate);
        }
    }
}

async fn native_full_attention_mix_from_inline(
    dims: NativeFullAttentionDims,
    shape: NativeFullAttentionShape,
    input: NativeFullAttentionMixInput<'_>,
    source: NativeFullAttentionInlineSource<'_>,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, MathError> {
    let mut attended = vec![0.0; shape.attention_dim];
    for head in 0..dims.num_attention_heads {
        let kv_head = head / shape.groups;
        let q_start = head * dims.head_dim;
        let kv_start = kv_head * dims.head_dim;
        let mut key_rows = Vec::with_capacity(source.count * dims.head_dim);
        for source_idx in 0..source.count {
            let key = source.keys.row(source_idx);
            key_rows.extend_from_slice(&key[kv_start..kv_start + dims.head_dim]);
        }
        let scores = scaled_full_attention_scores(
            &input.query[q_start..q_start + dims.head_dim],
            &key_rows,
            source.count,
            input.score_scale,
            matvec,
        )
        .await?;
        let weights = matvec.softmax_f32(&scores).await?;
        let mut value_rows = Vec::with_capacity(source.count * dims.head_dim);
        for source_idx in 0..source.count {
            let value = source.values.row(source_idx);
            value_rows.extend_from_slice(&value[kv_start..kv_start + dims.head_dim]);
        }
        let mixed = matvec
            .weighted_sum_f32(&value_rows, &weights, dims.head_dim)
            .await?;
        for offset in 0..dims.head_dim {
            let gate = input
                .gate
                .map_or(1.0, |gate| sigmoid_f32(gate[q_start + offset]));
            attended[q_start + offset] = mixed[offset] * gate;
        }
    }
    Ok(attended)
}

async fn scaled_full_attention_scores(
    query_head: &[f32],
    key_rows: &[f32],
    row_count: usize,
    scale: f32,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, MathError> {
    let mut scores = matvec
        .matvec_row_major_f32(query_head, key_rows, row_count, query_head.len())
        .await?;
    for score in &mut scores {
        *score *= scale;
    }
    Ok(scores)
}

#[cfg(test)]
mod tests {
    use super::super::{CpuNativeMatvecBackend, TensorLoadError};
    use super::*;
    use std::{
        path::PathBuf,
        sync::atomic::{AtomicUsize, Ordering},
    };

    fn assert_close(left: f32, right: f32) {
        assert!(
            (left - right).abs() < 1e-5,
            "expected {left} to be close to {right}"
        );
    }

    #[derive(Debug)]
    struct DeletingBf16ProjectionBackend {
        root: PathBuf,
        calls: AtomicUsize,
    }

    impl NativeMatvecBackend for DeletingBf16ProjectionBackend {
        async fn bf16_matvec_row_major_f32_in_place(
            &self,
            _store: &SafeTensorShardStore,
            _tensor: &str,
            input: &[f32],
            output: &mut [f32],
        ) -> Result<(), TensorLoadError> {
            output[..input.len()].copy_from_slice(input);
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                std::fs::remove_dir_all(&self.root).expect("remove snapshot root");
            }
            Ok(())
        }

        async fn bf16_matvec_rows_f32_in_place(
            &self,
            store: &SafeTensorShardStore,
            tensor: &str,
            input: &[f32],
            chunk_rows: usize,
            output: &mut [f32],
        ) -> Result<(), TensorLoadError> {
            CpuNativeMatvecBackend
                .bf16_matvec_rows_f32_in_place(store, tensor, input, chunk_rows, output)
                .await
        }

        async fn matvec_row_major_f32_in_place(
            &self,
            input: &[f32],
            weights: &[f32],
            rows: usize,
            columns: usize,
            output: &mut [f32],
        ) -> Result<(), MathError> {
            CpuNativeMatvecBackend
                .matvec_row_major_f32_in_place(input, weights, rows, columns, output)
                .await
        }

        async fn rms_norm_one_centered_f32_in_place(
            &self,
            input: &[f32],
            weight: &[f32],
            eps: f32,
            output: &mut [f32],
        ) -> Result<(), MathError> {
            CpuNativeMatvecBackend
                .rms_norm_one_centered_f32_in_place(input, weight, eps, output)
                .await
        }

        async fn softmax_f32_in_place(
            &self,
            scores: &[f32],
            output: &mut [f32],
        ) -> Result<(), MathError> {
            CpuNativeMatvecBackend
                .softmax_f32_in_place(scores, output)
                .await
        }

        async fn linear_attention_conv1d_silu_f32_in_place(
            &self,
            window: &[f32],
            weights: &[f32],
            conv_dim: usize,
            kernel_size: usize,
            output: &mut [f32],
        ) -> Result<(), MathError> {
            CpuNativeMatvecBackend
                .linear_attention_conv1d_silu_f32_in_place(
                    window,
                    weights,
                    conv_dim,
                    kernel_size,
                    output,
                )
                .await
        }

        async fn weighted_sum_f32_in_place(
            &self,
            values: &[f32],
            weights: &[f32],
            vector_len: usize,
            output: &mut [f32],
        ) -> Result<(), MathError> {
            CpuNativeMatvecBackend
                .weighted_sum_f32_in_place(values, weights, vector_len, output)
                .await
        }

        async fn linear_attention_recurrent_update_f32_in_place(
            &self,
            state: &[f32],
            key: &[f32],
            value: &[f32],
            memory: &[f32],
            beta: f32,
            decay: f32,
            key_head_dim: usize,
            value_head_dim: usize,
            output: &mut [f32],
        ) -> Result<(), MathError> {
            CpuNativeMatvecBackend
                .linear_attention_recurrent_update_f32_in_place(
                    state,
                    key,
                    value,
                    memory,
                    beta,
                    decay,
                    key_head_dim,
                    value_head_dim,
                    output,
                )
                .await
        }

        async fn select_head_rows_f32_in_place(
            &self,
            values: &[f32],
            row_count: usize,
            row_len: usize,
            head_start: usize,
            head_len: usize,
            output: &mut [f32],
        ) -> Result<(), MathError> {
            CpuNativeMatvecBackend
                .select_head_rows_f32_in_place(
                    values, row_count, row_len, head_start, head_len, output,
                )
                .await
        }
    }

    fn tiny_safetensors_bf16(name: &str, shape: &[usize], values: &[f32]) -> Vec<u8> {
        let mut data = Vec::new();
        for value in values {
            data.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
        }
        let header = serde_json::json!({
            name: {
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [0, data.len()]
            }
        })
        .to_string();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&data);
        bytes
    }

    #[tokio::test]
    async fn bf16_output_projection_sequence_validates_shape_once_before_hot_loop() {
        let root =
            std::env::temp_dir().join(format!("kir-ai-native-attn-test-{}", std::process::id()));
        std::fs::remove_dir_all(&root).ok();
        std::fs::create_dir_all(&root).expect("tempdir");
        std::fs::write(
            root.join("model.safetensors"),
            tiny_safetensors_bf16("o_proj.weight", &[1, 1], &[1.0]),
        )
        .expect("write projection tensor");
        let store = SafeTensorShardStore::open(&root).expect("store opens");
        let dims = NativeFullAttentionDims {
            hidden_size: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 1,
        };
        let queries = vec![vec![1.0], vec![1.0]];
        let keys = vec![vec![1.0], vec![1.0]];
        let values = vec![vec![10.0], vec![20.0]];
        let backend = DeletingBf16ProjectionBackend {
            root,
            calls: AtomicUsize::new(0),
        };

        let output = native_full_attention_sequence_from_parts(
            dims,
            &NativeFullAttentionSequenceParts {
                queries: NativeF32Rows::from_rows(&queries),
                keys: NativeF32Rows::from_rows(&keys),
                values: NativeF32Rows::from_rows(&values),
                gates: None,
                output_projection: NativeOutputProjection::Bf16Tensor {
                    store: &store,
                    tensor: "o_proj.weight",
                },
                score_scale: 0.0,
            },
            &backend,
        )
        .await
        .expect("sequence should not re-read projection metadata after upfront validation");

        assert_eq!(backend.calls.load(Ordering::SeqCst), 2);
        assert_close(output[0][0], 10.0);
        assert_close(output[1][0], 15.0);
    }

    #[tokio::test]
    async fn native_full_attention_sequence_is_causal_without_cache() {
        let dims = NativeFullAttentionDims {
            hidden_size: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 1,
        };
        let queries = vec![vec![1.0], vec![1.0]];
        let keys = vec![vec![1.0], vec![3.0]];
        let values = vec![vec![10.0], vec![30.0]];
        let output_projection = vec![1.0];

        let output = native_full_attention_sequence_from_parts(
            dims,
            &NativeFullAttentionSequenceParts {
                queries: NativeF32Rows::from_rows(&queries),
                keys: NativeF32Rows::from_rows(&keys),
                values: NativeF32Rows::from_rows(&values),
                gates: None,
                output_projection: NativeOutputProjection::F32(&output_projection),
                score_scale: 1.0,
            },
            &CpuNativeMatvecBackend,
        )
        .await
        .expect("attention succeeds");

        assert_close(output[0][0], 10.0);
        assert!(output[1][0] > 27.0, "second token should attend to new key");
    }

    #[tokio::test]
    async fn native_full_attention_sequence_accepts_flat_projection_rows() {
        let dims = NativeFullAttentionDims {
            hidden_size: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 1,
        };
        let queries = vec![1.0, 1.0];
        let keys = vec![1.0, 3.0];
        let values = vec![10.0, 30.0];
        let output_projection = vec![1.0];

        let output = native_full_attention_sequence_from_parts(
            dims,
            &NativeFullAttentionSequenceParts {
                queries: NativeF32Rows::flat(&queries, 1).expect("query rows"),
                keys: NativeF32Rows::flat(&keys, 1).expect("key rows"),
                values: NativeF32Rows::flat(&values, 1).expect("value rows"),
                gates: None,
                output_projection: NativeOutputProjection::F32(&output_projection),
                score_scale: 1.0,
            },
            &CpuNativeMatvecBackend,
        )
        .await
        .expect("attention succeeds");

        assert_close(output[0][0], 10.0);
        assert!(output[1][0] > 27.0, "second token should attend to new key");
    }

    #[tokio::test]
    async fn native_full_attention_cache_matches_uncached_sequence() {
        let dims = NativeFullAttentionDims {
            hidden_size: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 1,
        };
        let queries = vec![vec![1.0, 0.5], vec![0.25, 1.0]];
        let keys = vec![vec![1.0], vec![0.5]];
        let values = vec![vec![4.0], vec![8.0]];
        let gates = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let output_projection = vec![1.0, 0.0, 0.0, 1.0];
        let parts = NativeFullAttentionSequenceParts {
            queries: NativeF32Rows::from_rows(&queries),
            keys: NativeF32Rows::from_rows(&keys),
            values: NativeF32Rows::from_rows(&values),
            gates: Some(NativeF32Rows::from_rows(&gates)),
            output_projection: NativeOutputProjection::F32(&output_projection),
            score_scale: 1.0,
        };

        let uncached =
            native_full_attention_sequence_from_parts(dims, &parts, &CpuNativeMatvecBackend)
                .await
                .expect("uncached succeeds");
        let mut cache = LayerKvCache::new(8, 1, 1).expect("cache shape");
        let cached = native_full_attention_sequence_with_cache_from_parts(
            dims,
            &parts,
            &mut cache,
            &CpuNativeMatvecBackend,
        )
        .await
        .expect("cached succeeds");

        assert_eq!(uncached.len(), cached.len());
        for (uncached, cached) in uncached.iter().zip(cached) {
            for (uncached, cached) in uncached.iter().zip(cached) {
                assert_close(*uncached, cached);
            }
        }
    }

    #[tokio::test]
    async fn native_full_attention_uses_caller_score_scale() {
        let dims = NativeFullAttentionDims {
            hidden_size: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 1,
        };
        let queries = vec![vec![1.0], vec![1.0]];
        let keys = vec![vec![0.0], vec![10.0]];
        let values = vec![vec![0.0], vec![100.0]];
        let output_projection = vec![1.0];

        let flat_scaled = native_full_attention_sequence_from_parts(
            dims,
            &NativeFullAttentionSequenceParts {
                queries: NativeF32Rows::from_rows(&queries),
                keys: NativeF32Rows::from_rows(&keys),
                values: NativeF32Rows::from_rows(&values),
                gates: None,
                output_projection: NativeOutputProjection::F32(&output_projection),
                score_scale: 0.0,
            },
            &CpuNativeMatvecBackend,
        )
        .await
        .expect("flat scale attention succeeds");
        let sharp_scaled = native_full_attention_sequence_from_parts(
            dims,
            &NativeFullAttentionSequenceParts {
                queries: NativeF32Rows::from_rows(&queries),
                keys: NativeF32Rows::from_rows(&keys),
                values: NativeF32Rows::from_rows(&values),
                gates: None,
                output_projection: NativeOutputProjection::F32(&output_projection),
                score_scale: 1.0,
            },
            &CpuNativeMatvecBackend,
        )
        .await
        .expect("sharp scale attention succeeds");

        assert_close(flat_scaled[1][0], 50.0);
        assert!(
            sharp_scaled[1][0] > 99.0,
            "positive score scale should strongly prefer the matching key"
        );
    }

    #[tokio::test]
    async fn native_full_attention_cache_sequence_reads_without_appending() {
        let dims = NativeFullAttentionDims {
            hidden_size: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 1,
        };
        let mut cache = LayerKvCache::new(8, 1, 1).expect("cache shape");
        cache
            .append_sliding(&[1.0], &[10.0])
            .expect("first cache append");
        cache
            .append_sliding(&[3.0], &[30.0])
            .expect("second cache append");
        let queries = vec![vec![1.0], vec![1.0]];
        let source_counts = vec![1, 2];
        let output_projection = vec![1.0];

        let output = native_full_attention_sequence_from_cache_parts(
            dims,
            &NativeFullAttentionCacheSequenceParts {
                queries: NativeF32Rows::from_rows(&queries),
                gates: None,
                source_counts: &source_counts,
                output_projection: NativeOutputProjection::F32(&output_projection),
                score_scale: 1.0,
            },
            &cache,
            &CpuNativeMatvecBackend,
        )
        .await
        .expect("cache-only sequence attention succeeds");

        assert_eq!(cache.token_count(), 2);
        assert_close(output[0][0], 10.0);
        assert!(
            output[1][0] > 27.0,
            "second token should read the larger source window"
        );
    }

    #[derive(Debug, Default)]
    struct FusedCacheMixBackend {
        fused_calls: AtomicUsize,
        per_head_calls: AtomicUsize,
    }

    impl NativeMatvecBackend for FusedCacheMixBackend {
        async fn bf16_matvec_row_major_f32_in_place(
            &self,
            store: &SafeTensorShardStore,
            tensor: &str,
            input: &[f32],
            output: &mut [f32],
        ) -> Result<(), TensorLoadError> {
            CpuNativeMatvecBackend
                .bf16_matvec_row_major_f32_in_place(store, tensor, input, output)
                .await
        }

        async fn bf16_matvec_rows_f32_in_place(
            &self,
            store: &SafeTensorShardStore,
            tensor: &str,
            input: &[f32],
            chunk_rows: usize,
            output: &mut [f32],
        ) -> Result<(), TensorLoadError> {
            CpuNativeMatvecBackend
                .bf16_matvec_rows_f32_in_place(store, tensor, input, chunk_rows, output)
                .await
        }

        async fn matvec_row_major_f32_in_place(
            &self,
            input: &[f32],
            weights: &[f32],
            rows: usize,
            columns: usize,
            output: &mut [f32],
        ) -> Result<(), MathError> {
            CpuNativeMatvecBackend
                .matvec_row_major_f32_in_place(input, weights, rows, columns, output)
                .await
        }

        async fn rms_norm_one_centered_f32_in_place(
            &self,
            input: &[f32],
            weight: &[f32],
            eps: f32,
            output: &mut [f32],
        ) -> Result<(), MathError> {
            CpuNativeMatvecBackend
                .rms_norm_one_centered_f32_in_place(input, weight, eps, output)
                .await
        }

        async fn softmax_f32_in_place(
            &self,
            scores: &[f32],
            output: &mut [f32],
        ) -> Result<(), MathError> {
            self.per_head_calls.fetch_add(1, Ordering::SeqCst);
            CpuNativeMatvecBackend
                .softmax_f32_in_place(scores, output)
                .await
        }

        async fn linear_attention_conv1d_silu_f32_in_place(
            &self,
            window: &[f32],
            weights: &[f32],
            conv_dim: usize,
            kernel_size: usize,
            output: &mut [f32],
        ) -> Result<(), MathError> {
            CpuNativeMatvecBackend
                .linear_attention_conv1d_silu_f32_in_place(
                    window,
                    weights,
                    conv_dim,
                    kernel_size,
                    output,
                )
                .await
        }

        async fn weighted_sum_f32_in_place(
            &self,
            values: &[f32],
            weights: &[f32],
            vector_len: usize,
            output: &mut [f32],
        ) -> Result<(), MathError> {
            self.per_head_calls.fetch_add(1, Ordering::SeqCst);
            CpuNativeMatvecBackend
                .weighted_sum_f32_in_place(values, weights, vector_len, output)
                .await
        }

        async fn linear_attention_recurrent_update_f32_in_place(
            &self,
            state: &[f32],
            key: &[f32],
            value: &[f32],
            memory: &[f32],
            beta: f32,
            decay: f32,
            key_head_dim: usize,
            value_head_dim: usize,
            output: &mut [f32],
        ) -> Result<(), MathError> {
            CpuNativeMatvecBackend
                .linear_attention_recurrent_update_f32_in_place(
                    state,
                    key,
                    value,
                    memory,
                    beta,
                    decay,
                    key_head_dim,
                    value_head_dim,
                    output,
                )
                .await
        }

        async fn select_head_rows_f32_in_place(
            &self,
            values: &[f32],
            row_count: usize,
            row_len: usize,
            head_start: usize,
            head_len: usize,
            output: &mut [f32],
        ) -> Result<(), MathError> {
            self.per_head_calls.fetch_add(1, Ordering::SeqCst);
            CpuNativeMatvecBackend
                .select_head_rows_f32_in_place(
                    values, row_count, row_len, head_start, head_len, output,
                )
                .await
        }

        async fn full_attention_cache_mix_f32_in_place(
            &self,
            _cache: &LayerKvCache,
            _query: &[f32],
            row_count: usize,
            num_attention_heads: usize,
            num_key_value_heads: usize,
            head_dim: usize,
            _score_scale: f32,
            output: &mut [f32],
        ) -> Result<bool, MathError> {
            assert_eq!(row_count, 2);
            assert_eq!(num_attention_heads, 2);
            assert_eq!(num_key_value_heads, 1);
            assert_eq!(head_dim, 1);
            self.fused_calls.fetch_add(1, Ordering::SeqCst);
            output[..2].copy_from_slice(&[11.0, 22.0]);
            Ok(true)
        }
    }

    #[tokio::test]
    async fn native_full_attention_cache_mix_uses_backend_fused_path() {
        let dims = NativeFullAttentionDims {
            hidden_size: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 1,
        };
        let mut cache = LayerKvCache::new(8, 1, 1).expect("cache shape");
        cache
            .append_sliding(&[1.0], &[10.0])
            .expect("first cache append");
        cache
            .append_sliding(&[2.0], &[20.0])
            .expect("second cache append");
        let queries = vec![vec![1.0, 2.0]];
        let source_counts = vec![2];
        let output_projection = vec![1.0, 0.0, 0.0, 1.0];
        let backend = FusedCacheMixBackend::default();

        let output = native_full_attention_sequence_from_cache_parts(
            dims,
            &NativeFullAttentionCacheSequenceParts {
                queries: NativeF32Rows::from_rows(&queries),
                gates: None,
                source_counts: &source_counts,
                output_projection: NativeOutputProjection::F32(&output_projection),
                score_scale: 1.0,
            },
            &cache,
            &backend,
        )
        .await
        .expect("cache-only attention succeeds");

        assert_eq!(backend.fused_calls.load(Ordering::SeqCst), 1);
        assert_eq!(backend.per_head_calls.load(Ordering::SeqCst), 0);
        assert_close(output[0][0], 11.0);
        assert_close(output[0][1], 22.0);
    }
}
