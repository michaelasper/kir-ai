use llm_backend::native::{BlockId, LayerKvCache, LayerKvCacheBlock, LayerKvCacheInt8Block};
use std::collections::{HashMap, HashSet};

#[derive(Debug)]
pub(crate) struct MetalBlockKvMirror {
    pub(crate) block_id: BlockId,
    pub(crate) keys: llm_metal::F16Buffer,
    pub(crate) values: llm_metal::F16Buffer,
    pub(crate) revision_at_last_sync: u64,
}

#[derive(Debug)]
pub(crate) struct MetalBlockInt8KvMirror {
    pub(crate) block_id: BlockId,
    pub(crate) keys: llm_metal::I8Buffer,
    pub(crate) key_scales: llm_metal::F32Buffer,
    pub(crate) values: llm_metal::I8Buffer,
    pub(crate) value_scales: llm_metal::F32Buffer,
    pub(crate) revision_at_last_sync: u64,
}

#[derive(Debug)]
pub(crate) struct MetalLayerKvStageMirror {
    pub(crate) cache_id: u64,
    pub(crate) keys: llm_metal::F16Buffer,
    pub(crate) values: llm_metal::F16Buffer,
    pub(crate) revision_at_last_sync: Option<u64>,
    pub(crate) tokens_seen_at_last_sync: usize,
    pub(crate) token_count_at_last_sync: usize,
    pub(crate) max_tokens: usize,
    pub(crate) vector_len: usize,
}

#[derive(Debug)]
pub(crate) struct MetalLayerInt8KvStageMirror {
    pub(crate) cache_id: u64,
    pub(crate) keys: llm_metal::I8Buffer,
    pub(crate) key_scales: llm_metal::F32Buffer,
    pub(crate) values: llm_metal::I8Buffer,
    pub(crate) value_scales: llm_metal::F32Buffer,
    pub(crate) revision_at_last_sync: Option<u64>,
    pub(crate) tokens_seen_at_last_sync: usize,
    pub(crate) token_count_at_last_sync: usize,
    pub(crate) max_tokens: usize,
    pub(crate) vector_len: usize,
}

#[derive(Debug)]
pub(crate) struct MetalBlockCopy {
    pub(crate) source_keys: llm_metal::F16Buffer,
    pub(crate) source_values: llm_metal::F16Buffer,
    pub(crate) source_start: usize,
    pub(crate) destination_start: usize,
    pub(crate) element_count: usize,
}

#[derive(Debug)]
pub(crate) struct MetalStageWrite<'a> {
    pub(crate) source_keys: &'a [f32],
    pub(crate) source_values: &'a [f32],
    pub(crate) destination_start: usize,
    pub(crate) element_count: usize,
}

#[derive(Debug)]
pub(crate) struct MetalBlockInt8Copy {
    pub(crate) source_keys: llm_metal::I8Buffer,
    pub(crate) source_key_scales: llm_metal::F32Buffer,
    pub(crate) source_values: llm_metal::I8Buffer,
    pub(crate) source_value_scales: llm_metal::F32Buffer,
    pub(crate) source_start: usize,
    pub(crate) source_scale_start: usize,
    pub(crate) destination_start: usize,
    pub(crate) destination_scale_start: usize,
    pub(crate) element_count: usize,
    pub(crate) token_count: usize,
}

#[derive(Debug)]
pub(crate) struct MetalInt8StageWrite<'a> {
    pub(crate) source_keys: &'a [i8],
    pub(crate) source_key_scales: &'a [f32],
    pub(crate) source_values: &'a [i8],
    pub(crate) source_value_scales: &'a [f32],
    pub(crate) destination_start: usize,
    pub(crate) destination_scale_start: usize,
    pub(crate) element_count: usize,
    pub(crate) token_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MetalStageSyncPlan {
    Clean,
    Write {
        logical_start: usize,
        logical_end: usize,
        full_rebuild: bool,
    },
}

pub(crate) fn kv_cache_shape_error(
    err: llm_backend::native::KvCacheError,
) -> llm_metal::MetalError {
    llm_metal::MetalError::InvalidShape(format!("invalid block KV cache shape: {err}"))
}

pub(crate) fn kv_cache_block_pair_mirror_byte_len(
    block: LayerKvCacheBlock<'_>,
) -> Result<u64, llm_metal::MetalError> {
    let elements = block
        .key_storage()
        .len()
        .checked_add(block.value_storage().len())
        .ok_or_else(|| {
            llm_metal::MetalError::InvalidShape(
                "Metal KV block mirror byte length overflows usize".to_owned(),
            )
        })?;
    cache_resident_mirror_byte_len(elements)
}

pub(crate) fn int8_kv_cache_block_pair_mirror_byte_len(
    block: LayerKvCacheInt8Block<'_>,
) -> Result<u64, llm_metal::MetalError> {
    let code_bytes = block
        .key_codes_storage()
        .len()
        .checked_add(block.value_codes_storage().len())
        .ok_or_else(|| {
            llm_metal::MetalError::InvalidShape(
                "Metal INT8 KV block code byte length overflows usize".to_owned(),
            )
        })?;
    let scale_count = block
        .key_scales_storage()
        .len()
        .checked_add(block.value_scales_storage().len())
        .ok_or_else(|| {
            llm_metal::MetalError::InvalidShape(
                "Metal INT8 KV block scale count overflows usize".to_owned(),
            )
        })?;
    let scale_bytes = scale_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            llm_metal::MetalError::InvalidShape(
                "Metal INT8 KV block scale byte length overflows usize".to_owned(),
            )
        })?;
    code_bytes
        .checked_add(scale_bytes)
        .map(|bytes| bytes as u64)
        .ok_or_else(|| {
            llm_metal::MetalError::InvalidShape(
                "Metal INT8 KV block mirror byte length overflows usize".to_owned(),
            )
        })
}

pub(crate) fn kv_cache_blocks_needing_sync_from_active<'a>(
    active_blocks: impl IntoIterator<Item = LayerKvCacheBlock<'a>>,
    synced_revisions: &HashMap<BlockId, u64>,
) -> Vec<LayerKvCacheBlock<'a>> {
    let mut seen = HashSet::new();
    let mut sync_blocks = Vec::new();
    for block in active_blocks {
        if !seen.insert(block.block_id()) {
            continue;
        }
        if synced_revisions.get(&block.block_id()).copied() != Some(block.revision()) {
            sync_blocks.push(block);
        }
    }
    sync_blocks
}

pub(crate) fn int8_kv_cache_blocks_needing_sync_from_active<'a>(
    active_blocks: impl IntoIterator<Item = LayerKvCacheInt8Block<'a>>,
    synced_revisions: &HashMap<BlockId, u64>,
) -> Vec<LayerKvCacheInt8Block<'a>> {
    let mut seen = HashSet::new();
    let mut sync_blocks = Vec::new();
    for block in active_blocks {
        if !seen.insert(block.block_id()) {
            continue;
        }
        if synced_revisions.get(&block.block_id()).copied() != Some(block.revision()) {
            sync_blocks.push(block);
        }
    }
    sync_blocks
}

pub(crate) fn kv_cache_stage_element_len(
    cache: &LayerKvCache,
) -> Result<usize, llm_metal::MetalError> {
    cache
        .max_tokens()
        .checked_mul(cache.vector_len())
        .and_then(|len| len.checked_mul(2))
        .ok_or_else(|| {
            llm_metal::MetalError::InvalidShape(
                "Metal KV cache stage length overflows usize".to_owned(),
            )
        })
}

pub(crate) fn kv_stage_sync_plan(
    revision_at_last_sync: Option<u64>,
    tokens_seen_at_last_sync: usize,
    token_count_at_last_sync: usize,
    cache_revision: u64,
    cache_tokens_seen: usize,
    cache_token_count: usize,
) -> MetalStageSyncPlan {
    if revision_at_last_sync == Some(cache_revision) {
        return MetalStageSyncPlan::Clean;
    }

    let full_rebuild = revision_at_last_sync.is_none()
        || cache_tokens_seen < tokens_seen_at_last_sync
        || cache_token_count < token_count_at_last_sync;
    if full_rebuild {
        return MetalStageSyncPlan::Write {
            logical_start: 0,
            logical_end: cache_token_count,
            full_rebuild: true,
        };
    }

    let dirty_tokens = cache_tokens_seen.saturating_sub(tokens_seen_at_last_sync);
    if dirty_tokens == 0 || dirty_tokens > cache_token_count {
        return MetalStageSyncPlan::Write {
            logical_start: 0,
            logical_end: cache_token_count,
            full_rebuild: true,
        };
    }

    MetalStageSyncPlan::Write {
        logical_start: cache_token_count - dirty_tokens,
        logical_end: cache_token_count,
        full_rebuild: false,
    }
}

pub(crate) fn kv_stage_writes_from_active_blocks<'a>(
    active_blocks: &'a [LayerKvCacheBlock<'a>],
    logical_start: usize,
    logical_end: usize,
    vector_len: usize,
    max_tokens: usize,
) -> Result<Vec<MetalStageWrite<'a>>, llm_metal::MetalError> {
    let mut writes = Vec::new();
    for block in active_blocks {
        let Some(range) = intersect_stage_logical_range(
            block.logical_token_start(),
            block.physical_token_start(),
            block.block_token_start(),
            block.token_count(),
            logical_start,
            logical_end,
        )?
        else {
            continue;
        };
        let source_start = range
            .block_token_start
            .checked_mul(vector_len)
            .ok_or_else(|| {
                llm_metal::MetalError::InvalidShape(
                    "KV cache stage source start overflows usize".to_owned(),
                )
            })?;
        let element_count = range.token_count.checked_mul(vector_len).ok_or_else(|| {
            llm_metal::MetalError::InvalidShape(
                "KV cache stage copy length overflows usize".to_owned(),
            )
        })?;
        let source_end = source_start.checked_add(element_count).ok_or_else(|| {
            llm_metal::MetalError::InvalidShape(
                "KV cache stage source range overflows usize".to_owned(),
            )
        })?;
        let source_keys = block
            .key_storage()
            .get(source_start..source_end)
            .ok_or_else(|| {
                llm_metal::MetalError::InvalidShape(
                    "KV cache stage source key range exceeds block storage".to_owned(),
                )
            })?;
        let source_values = block
            .value_storage()
            .get(source_start..source_end)
            .ok_or_else(|| {
                llm_metal::MetalError::InvalidShape(
                    "KV cache stage source value range exceeds block storage".to_owned(),
                )
            })?;
        let destination_starts = kv_stage_destination_starts(
            range.physical_token_start,
            range.token_count,
            vector_len,
            max_tokens,
            "KV cache stage",
        )?;
        for destination_start in destination_starts {
            writes.push(MetalStageWrite {
                source_keys,
                source_values,
                destination_start,
                element_count,
            });
        }
    }
    Ok(writes)
}

pub(crate) fn int8_kv_stage_writes_from_active_blocks<'a>(
    active_blocks: &'a [LayerKvCacheInt8Block<'a>],
    logical_start: usize,
    logical_end: usize,
    vector_len: usize,
    max_tokens: usize,
) -> Result<Vec<MetalInt8StageWrite<'a>>, llm_metal::MetalError> {
    let mut writes = Vec::new();
    for block in active_blocks {
        let Some(range) = intersect_stage_logical_range(
            block.logical_token_start(),
            block.physical_token_start(),
            block.block_token_start(),
            block.token_count(),
            logical_start,
            logical_end,
        )?
        else {
            continue;
        };
        let source_start = range
            .block_token_start
            .checked_mul(vector_len)
            .ok_or_else(|| {
                llm_metal::MetalError::InvalidShape(
                    "INT8 KV cache stage source start overflows usize".to_owned(),
                )
            })?;
        let element_count = range.token_count.checked_mul(vector_len).ok_or_else(|| {
            llm_metal::MetalError::InvalidShape(
                "INT8 KV cache stage copy length overflows usize".to_owned(),
            )
        })?;
        let source_end = source_start.checked_add(element_count).ok_or_else(|| {
            llm_metal::MetalError::InvalidShape(
                "INT8 KV cache stage source range overflows usize".to_owned(),
            )
        })?;
        let scale_end = range
            .block_token_start
            .checked_add(range.token_count)
            .ok_or_else(|| {
                llm_metal::MetalError::InvalidShape(
                    "INT8 KV cache stage scale source range overflows usize".to_owned(),
                )
            })?;
        let source_keys = block
            .key_codes_storage()
            .get(source_start..source_end)
            .ok_or_else(|| {
                llm_metal::MetalError::InvalidShape(
                    "INT8 KV cache stage source key range exceeds block storage".to_owned(),
                )
            })?;
        let source_values = block
            .value_codes_storage()
            .get(source_start..source_end)
            .ok_or_else(|| {
                llm_metal::MetalError::InvalidShape(
                    "INT8 KV cache stage source value range exceeds block storage".to_owned(),
                )
            })?;
        let source_key_scales = block
            .key_scales_storage()
            .get(range.block_token_start..scale_end)
            .ok_or_else(|| {
                llm_metal::MetalError::InvalidShape(
                    "INT8 KV cache stage key scale range exceeds block storage".to_owned(),
                )
            })?;
        let source_value_scales = block
            .value_scales_storage()
            .get(range.block_token_start..scale_end)
            .ok_or_else(|| {
                llm_metal::MetalError::InvalidShape(
                    "INT8 KV cache stage value scale range exceeds block storage".to_owned(),
                )
            })?;
        let destination_starts = kv_stage_destination_starts(
            range.physical_token_start,
            range.token_count,
            vector_len,
            max_tokens,
            "INT8 KV cache stage",
        )?;
        let scale_destination_starts = [
            range.physical_token_start,
            range
                .physical_token_start
                .checked_add(max_tokens)
                .ok_or_else(|| {
                    llm_metal::MetalError::InvalidShape(
                        "INT8 KV cache stage mirror scale destination overflows usize".to_owned(),
                    )
                })?,
        ];
        for (destination_start, destination_scale_start) in
            destination_starts.into_iter().zip(scale_destination_starts)
        {
            writes.push(MetalInt8StageWrite {
                source_keys,
                source_key_scales,
                source_values,
                source_value_scales,
                destination_start,
                destination_scale_start,
                element_count,
                token_count: range.token_count,
            });
        }
    }
    Ok(writes)
}

pub(crate) fn f16_stage_copy_bytes(element_count: usize) -> u64 {
    (element_count as u64)
        .saturating_mul(std::mem::size_of::<u16>() as u64)
        .saturating_mul(2)
}

pub(crate) fn int8_stage_copy_bytes(element_count: usize, token_count: usize) -> u64 {
    let code_bytes = (element_count as u64).saturating_mul(2);
    let scale_bytes = (token_count as u64)
        .saturating_mul(std::mem::size_of::<f32>() as u64)
        .saturating_mul(2);
    code_bytes.saturating_add(scale_bytes)
}

fn cache_resident_mirror_byte_len(elements: usize) -> Result<u64, llm_metal::MetalError> {
    elements
        .checked_mul(std::mem::size_of::<u16>())
        .map(|bytes| bytes as u64)
        .ok_or_else(|| {
            llm_metal::MetalError::InvalidShape(
                "Metal resident cache byte length overflows usize".to_owned(),
            )
        })
}

#[derive(Debug, Clone, Copy)]
struct StageLogicalRange {
    block_token_start: usize,
    physical_token_start: usize,
    token_count: usize,
}

fn intersect_stage_logical_range(
    block_logical_start: usize,
    block_physical_start: usize,
    block_token_start: usize,
    block_token_count: usize,
    logical_start: usize,
    logical_end: usize,
) -> Result<Option<StageLogicalRange>, llm_metal::MetalError> {
    let block_logical_end = block_logical_start
        .checked_add(block_token_count)
        .ok_or_else(|| {
            llm_metal::MetalError::InvalidShape(
                "KV cache stage logical block range overflows usize".to_owned(),
            )
        })?;
    let copy_logical_start = block_logical_start.max(logical_start);
    let copy_logical_end = block_logical_end.min(logical_end);
    if copy_logical_start >= copy_logical_end {
        return Ok(None);
    }
    let token_offset = copy_logical_start - block_logical_start;
    Ok(Some(StageLogicalRange {
        block_token_start: block_token_start.checked_add(token_offset).ok_or_else(|| {
            llm_metal::MetalError::InvalidShape(
                "KV cache stage block token start overflows usize".to_owned(),
            )
        })?,
        physical_token_start: block_physical_start
            .checked_add(token_offset)
            .ok_or_else(|| {
                llm_metal::MetalError::InvalidShape(
                    "KV cache stage physical token start overflows usize".to_owned(),
                )
            })?,
        token_count: copy_logical_end - copy_logical_start,
    }))
}

pub(crate) fn kv_stage_destination_starts(
    physical_token_start: usize,
    token_count: usize,
    vector_len: usize,
    max_tokens: usize,
    label: &'static str,
) -> Result<[usize; 2], llm_metal::MetalError> {
    let physical_end = physical_token_start
        .checked_add(token_count)
        .ok_or_else(|| {
            llm_metal::MetalError::InvalidShape(format!("{label} physical range overflows usize"))
        })?;
    if physical_end > max_tokens {
        return Err(llm_metal::MetalError::InvalidShape(format!(
            "{label} physical range {physical_token_start}..{physical_end} exceeds max_tokens {max_tokens}"
        )));
    }
    let destination_start = physical_token_start
        .checked_mul(vector_len)
        .ok_or_else(|| {
            llm_metal::MetalError::InvalidShape(format!(
                "{label} destination start overflows usize"
            ))
        })?;
    let mirror_destination_start = physical_token_start
        .checked_add(max_tokens)
        .and_then(|token| token.checked_mul(vector_len))
        .ok_or_else(|| {
            llm_metal::MetalError::InvalidShape(format!(
                "{label} mirror destination start overflows usize"
            ))
        })?;
    Ok([destination_start, mirror_destination_start])
}

#[cfg(test)]
fn kv_cache_blocks_needing_sync<'a>(
    cache: &'a LayerKvCache,
    synced_revisions: &HashMap<BlockId, u64>,
) -> Result<Vec<LayerKvCacheBlock<'a>>, llm_metal::MetalError> {
    let active_blocks = cache.active_blocks().map_err(kv_cache_shape_error)?;
    Ok(kv_cache_blocks_needing_sync_from_active(
        active_blocks,
        synced_revisions,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_KV_CACHE_MIRROR_BLOCK_TOKENS: usize = 256;

    #[test]
    fn kv_cache_block_mirror_byte_len_uses_f16_block_storage() {
        let mut cache = LayerKvCache::new(10, 1, 2).expect("cache shape is valid");

        cache
            .append(&[1.0, 2.0], &[3.0, 4.0])
            .expect("first token fits");
        let block = cache
            .active_blocks()
            .expect("active block view is valid")
            .remove(0);

        assert_eq!(
            kv_cache_block_pair_mirror_byte_len(block).expect("mirror byte length fits"),
            80
        );
    }

    #[test]
    fn int8_kv_cache_block_mirror_byte_len_includes_codes_and_scales() {
        let mut cache =
            LayerKvCache::new_with_config(10, 1, 2, llm_backend::native::KvCacheConfig::int8())
                .expect("cache shape is valid");

        cache
            .append(&[1.0, 2.0], &[3.0, 4.0])
            .expect("first token fits");
        let block = cache
            .active_int8_blocks()
            .expect("active int8 blocks are valid")
            .expect("int8 blocks exist")
            .remove(0);

        assert_eq!(
            int8_kv_cache_block_pair_mirror_byte_len(block).expect("mirror byte length fits"),
            12
        );
    }

    #[test]
    fn kv_cache_block_sync_plan_syncs_only_missing_or_changed_blocks() {
        let mut cache = LayerKvCache::new(TEST_KV_CACHE_MIRROR_BLOCK_TOKENS + 1, 1, 1)
            .expect("cache shape is valid");
        for token in 0..cache.max_tokens() {
            cache
                .append(&[token as f32], &[1000.0 + token as f32])
                .expect("token appends");
        }
        let active_blocks = cache.active_blocks().expect("active block view is valid");
        let synced_revisions = active_blocks
            .iter()
            .map(|block| (block.block_id(), block.revision()))
            .collect::<HashMap<_, _>>();

        let cold_plan =
            kv_cache_blocks_needing_sync(&cache, &HashMap::new()).expect("cold sync plan is valid");
        assert_eq!(cold_plan.len(), 2);

        let reused_prefix = cache.clone();
        let reused_plan = kv_cache_blocks_needing_sync(&reused_prefix, &synced_revisions)
            .expect("reused prefix plan is valid");
        assert!(
            reused_plan.is_empty(),
            "prefix blocks with unchanged revisions should not sync"
        );

        cache
            .append_sliding(&[999.0], &[1999.0])
            .expect("sliding append overwrites one physical block");
        let dirty_block_id = cache
            .active_blocks()
            .expect("dirty active block view is valid")[0]
            .block_id();
        let dirty_plan = kv_cache_blocks_needing_sync(&cache, &synced_revisions)
            .expect("dirty sync plan is valid");
        assert_eq!(dirty_plan.len(), 1);
        assert_eq!(dirty_plan[0].block_id(), dirty_block_id);
    }
}
