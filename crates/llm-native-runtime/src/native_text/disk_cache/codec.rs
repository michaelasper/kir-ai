use super::{
    NativeTextDiskCacheError, NativeTextDiskCacheStateBlock, NativeTextDiskCacheTensorArchive,
    NativeTextDiskCacheTensorSink, NativeTextDiskCacheValue,
};
use llm_backend::native::{
    GemmaLayerCache, GemmaLayerCachePrefixState, LayerKvCache, LayerKvCachePrefixState,
    LayerKvCacheSnapshot, LinearAttentionCacheSnapshot, QwenLayerCache, QwenLayerCachePrefixState,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum NativeTextDiskCacheLayerLayout {
    #[serde(rename = "qwen_full_attention")]
    QwenFull(NativeTextDiskFullAttentionLayout),
    #[serde(rename = "qwen_linear_attention")]
    QwenLinear(NativeTextDiskLinearAttentionLayout),
    #[serde(rename = "gemma_attention")]
    GemmaFull(NativeTextDiskFullAttentionLayout),
    #[cfg(test)]
    TestMarkerTensor { tensor: String },
}

impl NativeTextDiskCacheLayerLayout {
    #[cfg(test)]
    pub(crate) fn test_marker_tensor(tensor: &str) -> Self {
        Self::TestMarkerTensor {
            tensor: tensor.to_owned(),
        }
    }

    #[cfg(test)]
    pub(crate) fn test_marker_tensor_name(&self) -> Option<&str> {
        match self {
            Self::TestMarkerTensor { tensor } => Some(tensor),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct NativeTextDiskFullAttentionLayout {
    revision: u64,
    max_tokens: usize,
    key_value_heads: usize,
    head_dim: usize,
    token_count: usize,
    tokens_seen: usize,
    cache_format: String,
    key_tensor: String,
    value_tensor: String,
    shape: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct NativeTextDiskLinearAttentionLayout {
    revision: u64,
    conv_kernel_size: usize,
    conv_dim: usize,
    num_value_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    token_count: usize,
    conv_window_tensor: String,
    recurrent_state_tensor: String,
    conv_window_shape: Vec<usize>,
    recurrent_state_shape: Vec<usize>,
}

pub(crate) fn encode_layer_kv_snapshot(
    prefix: &str,
    snapshot: LayerKvCacheSnapshot,
    sink: &mut NativeTextDiskCacheTensorSink,
) -> Result<NativeTextDiskFullAttentionLayout, NativeTextDiskCacheError> {
    if snapshot.config.format().to_string() != "f32" {
        return Err(NativeTextDiskCacheError::integrity(
            "only f32 KV cache snapshots can be written to disk",
        ));
    }
    let shape = vec![
        snapshot.token_count,
        snapshot.key_value_heads,
        snapshot.head_dim,
    ];
    let key_tensor = format!("{prefix}.keys");
    let value_tensor = format!("{prefix}.values");
    sink.push_f32(&key_tensor, shape.clone(), snapshot.keys.clone())?;
    sink.push_f32(&value_tensor, shape.clone(), snapshot.values.clone())?;
    Ok(NativeTextDiskFullAttentionLayout {
        revision: snapshot.revision,
        max_tokens: snapshot.max_tokens,
        key_value_heads: snapshot.key_value_heads,
        head_dim: snapshot.head_dim,
        token_count: snapshot.token_count,
        tokens_seen: snapshot.tokens_seen,
        cache_format: "f32".to_owned(),
        key_tensor,
        value_tensor,
        shape,
    })
}

pub(crate) fn decode_layer_kv_snapshot(
    layout: &NativeTextDiskFullAttentionLayout,
    archive: &NativeTextDiskCacheTensorArchive<'_>,
) -> Result<LayerKvCacheSnapshot, NativeTextDiskCacheError> {
    if layout.cache_format != "f32" {
        return Err(NativeTextDiskCacheError::integrity(
            "unsupported KV cache disk format",
        ));
    }
    let keys = archive.f32_tensor(&layout.key_tensor)?;
    let values = archive.f32_tensor(&layout.value_tensor)?;
    validate_tensor_shape(archive, &layout.key_tensor, &layout.shape)?;
    validate_tensor_shape(archive, &layout.value_tensor, &layout.shape)?;
    let config = LayerKvCache::new(layout.max_tokens, layout.key_value_heads, layout.head_dim)?
        .snapshot()
        .config;
    Ok(LayerKvCacheSnapshot {
        revision: layout.revision,
        config,
        max_tokens: layout.max_tokens,
        key_value_heads: layout.key_value_heads,
        head_dim: layout.head_dim,
        token_count: layout.token_count,
        tokens_seen: layout.tokens_seen,
        keys,
        values,
    })
}

pub(crate) fn encode_linear_attention_snapshot(
    prefix: &str,
    snapshot: LinearAttentionCacheSnapshot,
    sink: &mut NativeTextDiskCacheTensorSink,
) -> Result<NativeTextDiskLinearAttentionLayout, NativeTextDiskCacheError> {
    let conv_window_shape = vec![snapshot.conv_kernel_size, snapshot.conv_dim];
    let recurrent_state_shape = vec![
        snapshot.num_value_heads,
        snapshot.key_head_dim,
        snapshot.value_head_dim,
    ];
    let conv_window_tensor = format!("{prefix}.conv_window");
    let recurrent_state_tensor = format!("{prefix}.recurrent_state");
    sink.push_f32(
        &conv_window_tensor,
        conv_window_shape.clone(),
        snapshot.conv_window.clone(),
    )?;
    sink.push_f32(
        &recurrent_state_tensor,
        recurrent_state_shape.clone(),
        snapshot.recurrent_state.clone(),
    )?;
    Ok(NativeTextDiskLinearAttentionLayout {
        revision: snapshot.revision,
        conv_kernel_size: snapshot.conv_kernel_size,
        conv_dim: snapshot.conv_dim,
        num_value_heads: snapshot.num_value_heads,
        key_head_dim: snapshot.key_head_dim,
        value_head_dim: snapshot.value_head_dim,
        token_count: snapshot.token_count,
        conv_window_tensor,
        recurrent_state_tensor,
        conv_window_shape,
        recurrent_state_shape,
    })
}

pub(crate) fn decode_linear_attention_snapshot(
    layout: &NativeTextDiskLinearAttentionLayout,
    archive: &NativeTextDiskCacheTensorArchive<'_>,
) -> Result<LinearAttentionCacheSnapshot, NativeTextDiskCacheError> {
    validate_tensor_shape(
        archive,
        &layout.conv_window_tensor,
        &layout.conv_window_shape,
    )?;
    validate_tensor_shape(
        archive,
        &layout.recurrent_state_tensor,
        &layout.recurrent_state_shape,
    )?;
    Ok(LinearAttentionCacheSnapshot {
        revision: layout.revision,
        conv_kernel_size: layout.conv_kernel_size,
        conv_dim: layout.conv_dim,
        num_value_heads: layout.num_value_heads,
        key_head_dim: layout.key_head_dim,
        value_head_dim: layout.value_head_dim,
        token_count: layout.token_count,
        conv_window: archive.f32_tensor(&layout.conv_window_tensor)?,
        recurrent_state: archive.f32_tensor(&layout.recurrent_state_tensor)?,
    })
}

fn validate_tensor_shape(
    archive: &NativeTextDiskCacheTensorArchive<'_>,
    tensor: &str,
    expected: &[usize],
) -> Result<(), NativeTextDiskCacheError> {
    let actual = archive.f32_tensor_shape(tensor)?;
    if actual != expected {
        return Err(NativeTextDiskCacheError::integrity(format!(
            "tensor `{tensor}` shape mismatch"
        )));
    }
    Ok(())
}

fn encode_layer_kv_block_from_prefix_state(
    prefix: &str,
    state: &LayerKvCachePrefixState,
    block_start: usize,
    block_token_count: usize,
    sink: &mut NativeTextDiskCacheTensorSink,
) -> Result<NativeTextDiskFullAttentionLayout, NativeTextDiskCacheError> {
    let cache = LayerKvCache::from_prefix_cache_state(state)?;
    let snapshot = cache.snapshot();
    let prefix_end = block_start
        .checked_add(block_token_count)
        .ok_or_else(|| NativeTextDiskCacheError::integrity("disk cache block range overflow"))?;
    if prefix_end > snapshot.token_count {
        return Err(NativeTextDiskCacheError::integrity(
            "KV block range exceeds retained prefix state",
        ));
    }
    let vector_len = snapshot
        .key_value_heads
        .checked_mul(snapshot.head_dim)
        .ok_or_else(|| NativeTextDiskCacheError::integrity("KV block vector shape overflow"))?;
    let start = block_start
        .checked_mul(vector_len)
        .ok_or_else(|| NativeTextDiskCacheError::integrity("KV block start overflow"))?;
    let end = prefix_end
        .checked_mul(vector_len)
        .ok_or_else(|| NativeTextDiskCacheError::integrity("KV block end overflow"))?;
    let block_snapshot = LayerKvCacheSnapshot {
        revision: snapshot.revision,
        config: snapshot.config,
        max_tokens: snapshot.max_tokens,
        key_value_heads: snapshot.key_value_heads,
        head_dim: snapshot.head_dim,
        token_count: block_token_count,
        tokens_seen: prefix_end,
        keys: snapshot.keys[start..end].to_vec(),
        values: snapshot.values[start..end].to_vec(),
    };
    encode_layer_kv_snapshot(prefix, block_snapshot, sink)
}

fn assemble_layer_kv_prefix_state_blocks<'a>(
    states: impl IntoIterator<Item = &'a LayerKvCachePrefixState>,
) -> Result<LayerKvCachePrefixState, NativeTextDiskCacheError> {
    let mut snapshots = states
        .into_iter()
        .map(|state| LayerKvCache::from_prefix_cache_state(state).map(|cache| cache.snapshot()));
    let Some(first) = snapshots.next().transpose()? else {
        return Err(NativeTextDiskCacheError::integrity(
            "missing KV block state",
        ));
    };
    let mut revision = first.revision;
    let config = first.config;
    let max_tokens = first.max_tokens;
    let key_value_heads = first.key_value_heads;
    let head_dim = first.head_dim;
    let mut token_count = first.token_count;
    let mut keys = first.keys;
    let mut values = first.values;

    for snapshot in snapshots {
        let snapshot = snapshot?;
        if snapshot.config != config
            || snapshot.max_tokens != max_tokens
            || snapshot.key_value_heads != key_value_heads
            || snapshot.head_dim != head_dim
        {
            return Err(NativeTextDiskCacheError::integrity(
                "incompatible KV block shapes",
            ));
        }
        revision = snapshot.revision;
        token_count = token_count
            .checked_add(snapshot.token_count)
            .ok_or_else(|| NativeTextDiskCacheError::integrity("KV block token count overflow"))?;
        keys.extend_from_slice(&snapshot.keys);
        values.extend_from_slice(&snapshot.values);
    }
    if token_count > max_tokens {
        return Err(NativeTextDiskCacheError::integrity(
            "assembled KV prefix exceeds cache capacity",
        ));
    }
    Ok(LayerKvCache::from_snapshot(LayerKvCacheSnapshot {
        revision,
        config,
        max_tokens,
        key_value_heads,
        head_dim,
        token_count,
        tokens_seen: token_count,
        keys,
        values,
    })?
    .prefix_cache_state())
}

fn validate_contiguous_disk_blocks<S>(
    blocks: &[NativeTextDiskCacheStateBlock<S>],
) -> Result<(), NativeTextDiskCacheError> {
    let mut expected_start = 0_usize;
    for block in blocks {
        if block.block_start != expected_start {
            return Err(NativeTextDiskCacheError::integrity(
                "disk cache blocks are not contiguous",
            ));
        }
        expected_start = expected_start
            .checked_add(block.token_count)
            .ok_or_else(|| {
                NativeTextDiskCacheError::integrity("disk cache block range overflow")
            })?;
    }
    Ok(())
}

impl NativeTextDiskCacheValue for QwenLayerCache {
    fn encode_disk_block_states(
        states: &[Self::PrefixCacheState],
        block_start: usize,
        block_token_count: usize,
        sink: &mut NativeTextDiskCacheTensorSink,
    ) -> Result<Vec<NativeTextDiskCacheLayerLayout>, NativeTextDiskCacheError> {
        states
            .iter()
            .enumerate()
            .map(|(layer_idx, state)| match state {
                QwenLayerCachePrefixState::Full(state) => encode_layer_kv_block_from_prefix_state(
                    &format!("layers.{layer_idx}.full"),
                    state,
                    block_start,
                    block_token_count,
                    sink,
                )
                .map(NativeTextDiskCacheLayerLayout::QwenFull),
                QwenLayerCachePrefixState::Linear(snapshot) => {
                    let prefix_end =
                        block_start.checked_add(block_token_count).ok_or_else(|| {
                            NativeTextDiskCacheError::integrity("linear block range overflow")
                        })?;
                    if snapshot.token_count != prefix_end {
                        return Err(NativeTextDiskCacheError::integrity(
                            "linear attention snapshot is not at the block boundary",
                        ));
                    }
                    encode_linear_attention_snapshot(
                        &format!("layers.{layer_idx}.linear"),
                        snapshot.clone(),
                        sink,
                    )
                    .map(NativeTextDiskCacheLayerLayout::QwenLinear)
                }
            })
            .collect()
    }

    fn decode_disk_states(
        layouts: &[NativeTextDiskCacheLayerLayout],
        archive: &NativeTextDiskCacheTensorArchive<'_>,
    ) -> Result<Vec<Self::PrefixCacheState>, NativeTextDiskCacheError> {
        layouts
            .iter()
            .map(|layout| match layout {
                NativeTextDiskCacheLayerLayout::QwenFull(layout) => {
                    let snapshot = decode_layer_kv_snapshot(layout, archive)?;
                    Ok(QwenLayerCachePrefixState::Full(
                        LayerKvCache::from_snapshot(snapshot)?.prefix_cache_state(),
                    ))
                }
                NativeTextDiskCacheLayerLayout::QwenLinear(layout) => {
                    Ok(QwenLayerCachePrefixState::Linear(
                        decode_linear_attention_snapshot(layout, archive)?,
                    ))
                }
                _ => Err(NativeTextDiskCacheError::integrity(
                    "non-Qwen layer layout in Qwen disk cache block",
                )),
            })
            .collect()
    }

    fn assemble_disk_block_states(
        blocks: &[NativeTextDiskCacheStateBlock<Self::PrefixCacheState>],
    ) -> Result<Vec<Self::PrefixCacheState>, NativeTextDiskCacheError> {
        validate_contiguous_disk_blocks(blocks)?;
        let Some(first) = blocks.first() else {
            return Ok(Vec::new());
        };
        let layer_count = first.states.len();
        let mut assembled = Vec::with_capacity(layer_count);
        for layer_idx in 0..layer_count {
            match first.states.get(layer_idx).ok_or_else(|| {
                NativeTextDiskCacheError::integrity("missing Qwen disk block layer")
            })? {
                QwenLayerCachePrefixState::Full(_) => {
                    let layer_states = blocks
                        .iter()
                        .map(|block| {
                            if block.states.len() != layer_count {
                                return Err(NativeTextDiskCacheError::integrity(
                                    "inconsistent Qwen disk block layer count",
                                ));
                            }
                            match block.states.get(layer_idx) {
                                Some(QwenLayerCachePrefixState::Full(state)) => Ok(state),
                                _ => Err(NativeTextDiskCacheError::integrity(
                                    "mixed Qwen disk block layer layout",
                                )),
                            }
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    assembled.push(QwenLayerCachePrefixState::Full(
                        assemble_layer_kv_prefix_state_blocks(layer_states)?,
                    ));
                }
                QwenLayerCachePrefixState::Linear(_) => {
                    let Some(QwenLayerCachePrefixState::Linear(snapshot)) =
                        blocks.last().and_then(|block| block.states.get(layer_idx))
                    else {
                        return Err(NativeTextDiskCacheError::integrity(
                            "missing Qwen linear disk block terminal state",
                        ));
                    };
                    assembled.push(QwenLayerCachePrefixState::Linear(snapshot.clone()));
                }
            }
        }
        Ok(assembled)
    }
}

impl NativeTextDiskCacheValue for GemmaLayerCache {
    fn encode_disk_block_states(
        states: &[Self::PrefixCacheState],
        block_start: usize,
        block_token_count: usize,
        sink: &mut NativeTextDiskCacheTensorSink,
    ) -> Result<Vec<NativeTextDiskCacheLayerLayout>, NativeTextDiskCacheError> {
        states
            .iter()
            .enumerate()
            .map(|(layer_idx, state)| match state {
                GemmaLayerCachePrefixState::Attention(state) => {
                    encode_layer_kv_block_from_prefix_state(
                        &format!("layers.{layer_idx}.attention"),
                        state,
                        block_start,
                        block_token_count,
                        sink,
                    )
                    .map(NativeTextDiskCacheLayerLayout::GemmaFull)
                }
            })
            .collect()
    }

    fn decode_disk_states(
        layouts: &[NativeTextDiskCacheLayerLayout],
        archive: &NativeTextDiskCacheTensorArchive<'_>,
    ) -> Result<Vec<Self::PrefixCacheState>, NativeTextDiskCacheError> {
        layouts
            .iter()
            .map(|layout| match layout {
                NativeTextDiskCacheLayerLayout::GemmaFull(layout) => {
                    let snapshot = decode_layer_kv_snapshot(layout, archive)?;
                    Ok(GemmaLayerCachePrefixState::Attention(
                        LayerKvCache::from_snapshot(snapshot)?.prefix_cache_state(),
                    ))
                }
                _ => Err(NativeTextDiskCacheError::integrity(
                    "non-Gemma layer layout in Gemma disk cache block",
                )),
            })
            .collect()
    }

    fn assemble_disk_block_states(
        blocks: &[NativeTextDiskCacheStateBlock<Self::PrefixCacheState>],
    ) -> Result<Vec<Self::PrefixCacheState>, NativeTextDiskCacheError> {
        validate_contiguous_disk_blocks(blocks)?;
        let Some(first) = blocks.first() else {
            return Ok(Vec::new());
        };
        let layer_count = first.states.len();
        let mut assembled = Vec::with_capacity(layer_count);
        for layer_idx in 0..layer_count {
            let layer_states = blocks
                .iter()
                .map(|block| {
                    if block.states.len() != layer_count {
                        return Err(NativeTextDiskCacheError::integrity(
                            "inconsistent Gemma disk block layer count",
                        ));
                    }
                    match block.states.get(layer_idx) {
                        Some(GemmaLayerCachePrefixState::Attention(state)) => Ok(state),
                        _ => Err(NativeTextDiskCacheError::integrity(
                            "mixed Gemma disk block layer layout",
                        )),
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            assembled.push(GemmaLayerCachePrefixState::Attention(
                assemble_layer_kv_prefix_state_blocks(layer_states)?,
            ));
        }
        Ok(assembled)
    }
}
