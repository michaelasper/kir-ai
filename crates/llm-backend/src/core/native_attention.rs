use super::math::{MathError, require_len, sigmoid_f32};
use super::{CpuNativeMatvecBackend, LayerKvCache, NativeKvCacheTensor, NativeMatvecBackend};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeFullAttentionDims {
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

pub struct NativeFullAttentionSequenceParts<'a> {
    pub queries: &'a [Vec<f32>],
    pub keys: &'a [Vec<f32>],
    pub values: &'a [Vec<f32>],
    pub gates: Option<&'a [Vec<f32>]>,
    pub output_projection: &'a [f32],
    pub score_scale: f32,
}

pub struct NativeFullAttentionStepParts<'a> {
    pub query: &'a [f32],
    pub key: &'a [f32],
    pub value: &'a [f32],
    pub gate: Option<&'a [f32]>,
    pub output_projection: &'a [f32],
    pub score_scale: f32,
}

pub struct NativeFullAttentionCacheSequenceParts<'a> {
    pub queries: &'a [Vec<f32>],
    pub gates: Option<&'a [Vec<f32>]>,
    pub source_counts: &'a [usize],
    pub output_projection: &'a [f32],
    pub score_scale: f32,
}

struct NativeFullAttentionInlineSource<'a> {
    keys: &'a [Vec<f32>],
    values: &'a [Vec<f32>],
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

pub async fn native_full_attention_sequence_from_parts(
    dims: NativeFullAttentionDims,
    parts: &NativeFullAttentionSequenceParts<'_>,
) -> Result<Vec<Vec<f32>>, MathError> {
    native_full_attention_sequence_from_parts_with_matvec(dims, parts, &CpuNativeMatvecBackend)
        .await
}

pub async fn native_full_attention_sequence_from_parts_with_matvec(
    dims: NativeFullAttentionDims,
    parts: &NativeFullAttentionSequenceParts<'_>,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<Vec<f32>>, MathError> {
    native_full_attention_sequence_impl(dims, parts, None, matvec).await
}

pub async fn native_full_attention_sequence_with_cache_from_parts(
    dims: NativeFullAttentionDims,
    parts: &NativeFullAttentionSequenceParts<'_>,
    cache: &mut LayerKvCache,
) -> Result<Vec<Vec<f32>>, MathError> {
    native_full_attention_sequence_with_cache_from_parts_with_matvec(
        dims,
        parts,
        cache,
        &CpuNativeMatvecBackend,
    )
    .await
}

pub async fn native_full_attention_sequence_with_cache_from_parts_with_matvec(
    dims: NativeFullAttentionDims,
    parts: &NativeFullAttentionSequenceParts<'_>,
    cache: &mut LayerKvCache,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<Vec<f32>>, MathError> {
    native_full_attention_sequence_impl(dims, parts, Some(cache), matvec).await
}

pub async fn native_full_attention_step_with_cache_from_parts(
    dims: NativeFullAttentionDims,
    parts: &NativeFullAttentionStepParts<'_>,
    cache: &mut LayerKvCache,
) -> Result<Vec<f32>, MathError> {
    native_full_attention_step_with_cache_from_parts_with_matvec(
        dims,
        parts,
        cache,
        &CpuNativeMatvecBackend,
    )
    .await
}

pub async fn native_full_attention_sequence_from_cache_parts_with_matvec(
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
    require_len(
        "output projection weight",
        parts.output_projection.len(),
        dims.hidden_size
            .checked_mul(shape.attention_dim)
            .ok_or_else(|| {
                MathError::InvalidShape("output projection shape overflow".to_owned())
            })?,
    )?;
    for token_idx in 0..seq_len {
        require_len(
            "query projection",
            parts.queries[token_idx].len(),
            shape.attention_dim,
        )?;
        if let Some(gates) = parts.gates {
            require_len(
                "gate projection",
                gates[token_idx].len(),
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
                query: &parts.queries[token_idx],
                gate: parts.gates.map(|gates| gates[token_idx].as_slice()),
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
            matvec
                .matvec_row_major_f32(
                    &attended,
                    parts.output_projection,
                    dims.hidden_size,
                    shape.attention_dim,
                )
                .await?,
        );
    }
    Ok(outputs)
}

pub async fn native_full_attention_step_with_cache_from_parts_with_matvec(
    dims: NativeFullAttentionDims,
    parts: &NativeFullAttentionStepParts<'_>,
    cache: &mut LayerKvCache,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, MathError> {
    let mut output = vec![0.0; dims.hidden_size];
    native_full_attention_step_with_cache_from_parts_with_matvec_in_place(
        dims,
        parts,
        cache,
        matvec,
        &mut output,
    )
    .await?;
    Ok(output)
}

pub async fn native_full_attention_step_with_cache_from_parts_with_matvec_in_place(
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
    require_len(
        "output projection weight",
        parts.output_projection.len(),
        dims.hidden_size
            .checked_mul(shape.attention_dim)
            .ok_or_else(|| {
                MathError::InvalidShape("output projection shape overflow".to_owned())
            })?,
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
    matvec
        .matvec_row_major_f32_in_place(
            &attended,
            parts.output_projection,
            dims.hidden_size,
            shape.attention_dim,
            output,
        )
        .await
        .map_err(|err| MathError::InvalidShape(format!("output projection failed: {err}")))
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
    require_len(
        "output projection weight",
        parts.output_projection.len(),
        dims.hidden_size
            .checked_mul(shape.attention_dim)
            .ok_or_else(|| {
                MathError::InvalidShape("output projection shape overflow".to_owned())
            })?,
    )?;
    for token_idx in 0..seq_len {
        require_full_attention_token_parts(
            dims,
            shape,
            &parts.queries[token_idx],
            &parts.keys[token_idx],
            &parts.values[token_idx],
            parts.gates.map(|gates| gates[token_idx].as_slice()),
        )?;
    }

    let mut outputs = Vec::with_capacity(seq_len);
    for token_idx in 0..seq_len {
        let query = &parts.queries[token_idx];
        let gate = parts.gates.map(|gates| gates[token_idx].as_slice());
        let attended = if let Some(cache) = cache.as_deref_mut() {
            cache
                .append_sliding(&parts.keys[token_idx], &parts.values[token_idx])
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
            matvec
                .matvec_row_major_f32(
                    &attended,
                    parts.output_projection,
                    dims.hidden_size,
                    shape.attention_dim,
                )
                .await?,
        );
    }

    Ok(outputs)
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
        let scores = scaled_full_attention_scores_with_matvec(
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
        for key in source.keys.iter().take(source.count) {
            key_rows.extend_from_slice(&key[kv_start..kv_start + dims.head_dim]);
        }
        let scores = scaled_full_attention_scores_with_matvec(
            &input.query[q_start..q_start + dims.head_dim],
            &key_rows,
            source.count,
            input.score_scale,
            matvec,
        )
        .await?;
        let weights = matvec.softmax_f32(&scores).await?;
        let mut value_rows = Vec::with_capacity(source.count * dims.head_dim);
        for value in source.values.iter().take(source.count) {
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

async fn scaled_full_attention_scores_with_matvec(
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
    use super::*;

    fn assert_close(left: f32, right: f32) {
        assert!(
            (left - right).abs() < 1e-5,
            "expected {left} to be close to {right}"
        );
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
                queries: &queries,
                keys: &keys,
                values: &values,
                gates: None,
                output_projection: &output_projection,
                score_scale: 1.0,
            },
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
            queries: &queries,
            keys: &keys,
            values: &values,
            gates: Some(&gates),
            output_projection: &output_projection,
            score_scale: 1.0,
        };

        let uncached = native_full_attention_sequence_from_parts(dims, &parts)
            .await
            .expect("uncached succeeds");
        let mut cache = LayerKvCache::new(8, 1, 1).expect("cache shape");
        let cached = native_full_attention_sequence_with_cache_from_parts(dims, &parts, &mut cache)
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
                queries: &queries,
                keys: &keys,
                values: &values,
                gates: None,
                output_projection: &output_projection,
                score_scale: 0.0,
            },
        )
        .await
        .expect("flat scale attention succeeds");
        let sharp_scaled = native_full_attention_sequence_from_parts(
            dims,
            &NativeFullAttentionSequenceParts {
                queries: &queries,
                keys: &keys,
                values: &values,
                gates: None,
                output_projection: &output_projection,
                score_scale: 1.0,
            },
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

        let output = native_full_attention_sequence_from_cache_parts_with_matvec(
            dims,
            &NativeFullAttentionCacheSequenceParts {
                queries: &queries,
                gates: None,
                source_counts: &source_counts,
                output_projection: &output_projection,
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
}
