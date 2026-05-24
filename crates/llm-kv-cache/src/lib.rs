mod block;
mod block_id;
mod block_pool;
mod block_table;
mod format;
pub mod prototype_quantization;

pub use block::{CacheBlock, CacheBlockHash, cache_block_chain_hash};
pub use block_id::BlockId;
pub use block_pool::BlockPool;
pub use block_table::{BlockTable, SessionBlockTable, SessionId};
pub use format::{
    AsymmetricVqCacheConfig, KvCacheConfig, KvCacheFormat, KvCacheFormatMetrics,
    KvCacheReconstructionError, KvCacheValueQuantizationBits,
};

use std::{
    collections::HashSet,
    fmt,
    sync::atomic::{AtomicU64, Ordering},
};

use block::RetainedCacheBlock;
use format::{KvCacheFormatMetricParts, LayerInt8KvStore, LayerQuantizedValueStore};

static NEXT_CACHE_ID: AtomicU64 = AtomicU64::new(1);

fn next_cache_id() -> u64 {
    NEXT_CACHE_ID.fetch_add(1, Ordering::Relaxed)
}

fn f32_resident_bytes(values: &[f32]) -> u64 {
    (values.len() as u64).saturating_mul(std::mem::size_of::<f32>() as u64)
}

fn uploaded_bytes(elements: usize, bytes_per_element: u64) -> u64 {
    (elements as u64).saturating_mul(bytes_per_element)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCacheBudget {
    max_tokens: usize,
    used_tokens: usize,
}

impl KvCacheBudget {
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            used_tokens: 0,
        }
    }

    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    pub fn used_tokens(&self) -> usize {
        self.used_tokens
    }

    pub fn remaining_tokens(&self) -> usize {
        self.max_tokens - self.used_tokens
    }

    pub fn reserve(&mut self, tokens: usize) -> Result<(), KvCacheError> {
        let next = self
            .used_tokens
            .checked_add(tokens)
            .ok_or(KvCacheError::CapacityExceeded {
                requested: tokens,
                available: self.remaining_tokens(),
            })?;
        if next > self.max_tokens {
            return Err(KvCacheError::CapacityExceeded {
                requested: tokens,
                available: self.remaining_tokens(),
            });
        }
        self.used_tokens = next;
        Ok(())
    }

    pub fn release(&mut self, tokens: usize) {
        self.used_tokens = self.used_tokens.saturating_sub(tokens);
    }
}

const LAYER_KV_BLOCK_TOKENS: usize = 256;

#[derive(Debug, PartialEq)]
pub struct LayerKvCache {
    id: u64,
    revision: u64,
    config: KvCacheConfig,
    max_tokens: usize,
    key_value_heads: usize,
    head_dim: usize,
    token_count: usize,
    tokens_seen: usize,
    // Physical slot of the oldest logical token in the circular window.
    window_start: usize,
    block_table: BlockTable,
    blocks: Vec<CacheBlock>,
    // Primary ring storage followed by mirrored slots for contiguous logical views.
    key_stage: Vec<f32>,
    value_stage: Vec<f32>,
    int8_storage: Option<LayerInt8KvStore>,
    quantized_values: Option<LayerQuantizedValueStore>,
}

/// Owned layer KV cache state, excluding runtime cache identity.
///
/// `keys` and `values` contain only the used token rows. Restoring a snapshot
/// allocates fresh backing storage and a fresh cache id.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerKvCacheSnapshot {
    pub revision: u64,
    pub config: KvCacheConfig,
    pub max_tokens: usize,
    pub key_value_heads: usize,
    pub head_dim: usize,
    pub token_count: usize,
    pub tokens_seen: usize,
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LayerKvCachePrefixState {
    revision: u64,
    config: KvCacheConfig,
    max_tokens: usize,
    key_value_heads: usize,
    head_dim: usize,
    token_count: usize,
    tokens_seen: usize,
    window_start: usize,
    blocks: Vec<LayerKvCachePrefixBlock>,
}

impl LayerKvCachePrefixState {
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    pub fn token_count(&self) -> usize {
        self.token_count
    }

    pub fn block_ids(&self) -> Vec<BlockId> {
        self.blocks
            .iter()
            .map(|block| block.retained.block_id())
            .collect()
    }

    pub fn retained_block_ref_count(&self, block_id: BlockId) -> Option<usize> {
        self.blocks
            .iter()
            .find(|block| block.retained.block_id() == block_id)
            .map(|block| block.retained.storage_ref_count())
    }

    pub fn retained_block_payload_bytes(&self) -> u64 {
        self.blocks.iter().fold(0_u64, |total, block| {
            total.saturating_add(block.retained.payload_bytes())
        })
    }

    pub fn metadata_bytes(&self) -> u64 {
        let scalar_bytes = std::mem::size_of::<u64>()
            .saturating_add(6_usize.saturating_mul(std::mem::size_of::<usize>()));
        let block_metadata_bytes = self.blocks.len().saturating_mul(
            std::mem::size_of::<BlockId>()
                .saturating_add(std::mem::size_of::<u64>())
                .saturating_add(4_usize.saturating_mul(std::mem::size_of::<usize>())),
        );
        (scalar_bytes as u64)
            .saturating_add(block_metadata_bytes as u64)
            .saturating_add(self.retained_block_payload_bytes())
    }
}

#[derive(Debug, Clone, PartialEq)]
struct LayerKvCachePrefixBlock {
    block_index: usize,
    retained: RetainedCacheBlock,
}

#[derive(Debug, Clone, Copy)]
pub struct LayerKvCacheBlock<'a> {
    block_id: BlockId,
    revision: u64,
    logical_token_start: usize,
    block_token_start: usize,
    token_count: usize,
    vector_len: usize,
    key_storage: &'a [f32],
    value_storage: &'a [f32],
}

#[derive(Debug, Clone, Copy)]
pub struct LayerKvCacheInt8Block<'a> {
    block_id: BlockId,
    revision: u64,
    logical_token_start: usize,
    block_token_start: usize,
    token_count: usize,
    vector_len: usize,
    key_codes: &'a [i8],
    value_codes: &'a [i8],
    key_scales: &'a [f32],
    value_scales: &'a [f32],
}

impl<'a> LayerKvCacheInt8Block<'a> {
    pub fn block_id(&self) -> BlockId {
        self.block_id
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn logical_token_start(&self) -> usize {
        self.logical_token_start
    }

    pub fn block_token_start(&self) -> usize {
        self.block_token_start
    }

    pub fn token_count(&self) -> usize {
        self.token_count
    }

    pub fn vector_len(&self) -> usize {
        self.vector_len
    }

    pub fn key_codes_storage(&self) -> &'a [i8] {
        self.key_codes
    }

    pub fn value_codes_storage(&self) -> &'a [i8] {
        self.value_codes
    }

    pub fn key_scales_storage(&self) -> &'a [f32] {
        self.key_scales
    }

    pub fn value_scales_storage(&self) -> &'a [f32] {
        self.value_scales
    }
}

impl<'a> LayerKvCacheBlock<'a> {
    pub fn block_id(&self) -> BlockId {
        self.block_id
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn logical_token_start(&self) -> usize {
        self.logical_token_start
    }

    pub fn block_token_start(&self) -> usize {
        self.block_token_start
    }

    pub fn token_count(&self) -> usize {
        self.token_count
    }

    pub fn vector_len(&self) -> usize {
        self.vector_len
    }

    pub fn keys(&self) -> &'a [f32] {
        self.row_slice(self.key_storage)
    }

    pub fn values(&self) -> &'a [f32] {
        self.row_slice(self.value_storage)
    }

    pub fn key_storage(&self) -> &'a [f32] {
        self.key_storage
    }

    pub fn value_storage(&self) -> &'a [f32] {
        self.value_storage
    }

    fn row_slice(&self, storage: &'a [f32]) -> &'a [f32] {
        let start = self.block_token_start * self.vector_len;
        let end = start + self.token_count * self.vector_len;
        &storage[start..end]
    }
}

impl Clone for LayerKvCache {
    fn clone(&self) -> Self {
        Self {
            id: next_cache_id(),
            revision: self.revision,
            config: self.config,
            max_tokens: self.max_tokens,
            key_value_heads: self.key_value_heads,
            head_dim: self.head_dim,
            token_count: self.token_count,
            tokens_seen: self.tokens_seen,
            window_start: self.window_start,
            block_table: self.block_table.clone(),
            blocks: self.blocks.clone(),
            key_stage: self.key_stage.clone(),
            value_stage: self.value_stage.clone(),
            int8_storage: self.int8_storage.clone(),
            quantized_values: self.quantized_values.clone(),
        }
    }
}

impl LayerKvCache {
    pub fn new(
        max_tokens: usize,
        key_value_heads: usize,
        head_dim: usize,
    ) -> Result<Self, KvCacheError> {
        Self::new_with_config(max_tokens, key_value_heads, head_dim, KvCacheConfig::f32())
    }

    pub fn new_with_config(
        max_tokens: usize,
        key_value_heads: usize,
        head_dim: usize,
        config: KvCacheConfig,
    ) -> Result<Self, KvCacheError> {
        if max_tokens == 0 || key_value_heads == 0 || head_dim == 0 {
            return Err(KvCacheError::InvalidShape);
        }
        if matches!(config.format(), KvCacheFormat::F16) {
            return Err(KvCacheError::UnsupportedFormat {
                format: config.format(),
            });
        }
        let vector_len = key_value_heads
            .checked_mul(head_dim)
            .ok_or(KvCacheError::InvalidShape)?;
        let storage_len = max_tokens
            .checked_mul(vector_len)
            .ok_or(KvCacheError::InvalidShape)?;
        let stage_len = storage_len
            .checked_mul(2)
            .ok_or(KvCacheError::InvalidShape)?;
        let block_count = max_tokens.div_ceil(LAYER_KV_BLOCK_TOKENS);
        let mut block_table = BlockTable::with_capacity(block_count);
        let mut blocks = Vec::with_capacity(block_count);
        let mut remaining_tokens = max_tokens;
        while remaining_tokens > 0 {
            let capacity_tokens = remaining_tokens.min(LAYER_KV_BLOCK_TOKENS);
            let block = CacheBlock::new(capacity_tokens, vector_len)?;
            block_table.append(block.id())?;
            blocks.push(block);
            remaining_tokens -= capacity_tokens;
        }
        let int8_storage = if config.format() == KvCacheFormat::Int8 {
            Some(LayerInt8KvStore::new(block_count, vector_len)?)
        } else {
            None
        };
        let quantized_values = config
            .asymmetric_vq_config()
            .map(|phase3| LayerQuantizedValueStore::new(block_count, vector_len, phase3))
            .transpose()?;
        Ok(Self {
            id: next_cache_id(),
            revision: 0,
            config,
            max_tokens,
            key_value_heads,
            head_dim,
            token_count: 0,
            tokens_seen: 0,
            window_start: 0,
            block_table,
            blocks,
            key_stage: vec![0.0; stage_len],
            value_stage: vec![0.0; stage_len],
            int8_storage,
            quantized_values,
        })
    }

    pub fn snapshot(&self) -> LayerKvCacheSnapshot {
        LayerKvCacheSnapshot {
            revision: self.revision,
            config: self.config,
            max_tokens: self.max_tokens,
            key_value_heads: self.key_value_heads,
            head_dim: self.head_dim,
            token_count: self.token_count,
            tokens_seen: self.tokens_seen,
            keys: self.keys().to_vec(),
            values: self.values().to_vec(),
        }
    }

    pub fn from_snapshot(snapshot: LayerKvCacheSnapshot) -> Result<Self, KvCacheError> {
        if snapshot.token_count > snapshot.max_tokens || snapshot.tokens_seen < snapshot.token_count
        {
            return Err(KvCacheError::InvalidShape);
        }

        let mut cache = Self::new_with_config(
            snapshot.max_tokens,
            snapshot.key_value_heads,
            snapshot.head_dim,
            snapshot.config,
        )?;
        let vector_len = cache.vector_len();
        let used_len = snapshot
            .token_count
            .checked_mul(vector_len)
            .ok_or(KvCacheError::InvalidShape)?;
        if snapshot.keys.len() != used_len {
            return Err(KvCacheError::ShapeMismatch {
                expected: used_len,
                actual: snapshot.keys.len(),
            });
        }
        if snapshot.values.len() != used_len {
            return Err(KvCacheError::ShapeMismatch {
                expected: used_len,
                actual: snapshot.values.len(),
            });
        }

        for (key, value) in snapshot
            .keys
            .chunks_exact(vector_len)
            .zip(snapshot.values.chunks_exact(vector_len))
        {
            cache.append(key, value)?;
        }
        cache.revision = snapshot.revision;
        cache.tokens_seen = snapshot.tokens_seen;
        Ok(cache)
    }

    pub fn prefix_cache_state(&self) -> LayerKvCachePrefixState {
        let mut seen = HashSet::new();
        let mut blocks = Vec::new();
        for logical_token_index in 0..self.token_count {
            let physical_token_index = (self.window_start + logical_token_index) % self.max_tokens;
            let Some((block_index, _)) = self.physical_block_position(physical_token_index) else {
                continue;
            };
            let Some(block_id) = self.block_table.get(block_index) else {
                continue;
            };
            if !seen.insert(block_id) {
                continue;
            }
            let Some(block) = self.blocks.get(block_index) else {
                continue;
            };
            if block.id() != block_id {
                continue;
            }
            blocks.push(LayerKvCachePrefixBlock {
                block_index,
                retained: block.retain_storage(),
            });
        }
        LayerKvCachePrefixState {
            revision: self.revision,
            config: self.config,
            max_tokens: self.max_tokens,
            key_value_heads: self.key_value_heads,
            head_dim: self.head_dim,
            token_count: self.token_count,
            tokens_seen: self.tokens_seen,
            window_start: self.window_start,
            blocks,
        }
    }

    pub fn from_prefix_cache_state(state: &LayerKvCachePrefixState) -> Result<Self, KvCacheError> {
        if state.max_tokens == 0
            || state.key_value_heads == 0
            || state.head_dim == 0
            || state.token_count > state.max_tokens
            || state.tokens_seen < state.token_count
            || state.window_start >= state.max_tokens
        {
            return Err(KvCacheError::InvalidShape);
        }
        let mut cache = Self::new_with_config(
            state.max_tokens,
            state.key_value_heads,
            state.head_dim,
            state.config,
        )?;
        for prefix_block in &state.blocks {
            let block = CacheBlock::from_retained(&prefix_block.retained)?;
            if prefix_block.block_index >= cache.blocks.len()
                || block.vector_len() != cache.vector_len()
                || block.capacity_tokens()
                    != cache.blocks[prefix_block.block_index].capacity_tokens()
            {
                return Err(KvCacheError::InvalidShape);
            }
            cache
                .block_table
                .replace(prefix_block.block_index, block.id())?;
            cache.blocks[prefix_block.block_index] = block;
        }
        cache.revision = state.revision;
        cache.token_count = state.token_count;
        cache.tokens_seen = state.tokens_seen;
        cache.window_start = state.window_start;
        cache.rebuild_stage_from_blocks()?;
        Ok(cache)
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn config(&self) -> KvCacheConfig {
        self.config
    }

    pub fn format(&self) -> KvCacheFormat {
        self.config.format()
    }

    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    pub fn key_value_heads(&self) -> usize {
        self.key_value_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn vector_len(&self) -> usize {
        self.key_value_heads * self.head_dim
    }

    pub fn token_count(&self) -> usize {
        self.token_count
    }

    pub fn next_position(&self) -> usize {
        self.tokens_seen
    }

    pub fn remaining_tokens(&self) -> usize {
        self.max_tokens - self.token_count
    }

    /// Returns bytes retained by compatibility staging and block key/value storage.
    pub fn resident_bytes(&self) -> u64 {
        let f32_bytes = self.blocks.iter().fold(
            f32_resident_bytes(&self.key_stage)
                .saturating_add(f32_resident_bytes(&self.value_stage)),
            |resident_bytes, block| {
                resident_bytes
                    .saturating_add(f32_resident_bytes(block.key_storage()))
                    .saturating_add(f32_resident_bytes(block.value_storage()))
            },
        );
        f32_bytes
            .saturating_add(
                self.int8_storage
                    .as_ref()
                    .map_or(0, LayerInt8KvStore::resident_bytes),
            )
            .saturating_add(
                self.quantized_values
                    .as_ref()
                    .map_or(0, LayerQuantizedValueStore::resident_bytes),
            )
    }

    pub fn int8_dequantized_keys(&self) -> Result<Option<Vec<f32>>, KvCacheError> {
        self.int8_dequantized_tensor(Int8Tensor::Key)
    }

    pub fn int8_dequantized_values(&self) -> Result<Option<Vec<f32>>, KvCacheError> {
        self.int8_dequantized_tensor(Int8Tensor::Value)
    }

    pub fn phase3_dequantized_values(&self) -> Result<Option<Vec<f32>>, KvCacheError> {
        let Some(quantized_values) = self.quantized_values.as_ref() else {
            return Ok(None);
        };
        let mut values = Vec::with_capacity(self.used_len());
        for logical_token_index in 0..self.token_count {
            let physical_token_index = (self.window_start + logical_token_index) % self.max_tokens;
            let (block_index, block_token_index) = self
                .physical_block_position(physical_token_index)
                .ok_or(KvCacheError::InvalidShape)?;
            let block_values = quantized_values.dequantized_block(block_index)?;
            let start = block_token_index
                .checked_mul(self.vector_len())
                .ok_or(KvCacheError::InvalidShape)?;
            let end = start
                .checked_add(self.vector_len())
                .ok_or(KvCacheError::InvalidShape)?;
            values.extend_from_slice(
                block_values
                    .get(start..end)
                    .ok_or(KvCacheError::InvalidShape)?,
            );
        }
        Ok(Some(values))
    }

    pub fn format_metrics(&self) -> Result<KvCacheFormatMetrics, KvCacheError> {
        let f32_residency = self.blocks.iter().fold(
            f32_resident_bytes(&self.key_stage)
                .saturating_add(f32_resident_bytes(&self.value_stage)),
            |resident_bytes, block| {
                resident_bytes
                    .saturating_add(f32_resident_bytes(block.key_storage()))
                    .saturating_add(f32_resident_bytes(block.value_storage()))
            },
        );
        let active_key_value_elements = self.keys().len().saturating_add(self.values().len());
        let f32_uploaded_bytes = uploaded_bytes(active_key_value_elements, 4);
        let f16_uploaded_bytes = uploaded_bytes(active_key_value_elements, 2);
        let int8_uploaded_bytes = uploaded_bytes(active_key_value_elements, 1);
        let f16_resident_bytes = f32_residency / 2;
        let int8_resident_bytes = self
            .int8_storage
            .as_ref()
            .map_or(0, LayerInt8KvStore::resident_bytes);
        let (phase3_value_bits, phase3_resident_bytes, phase3_payload_bytes, phase3_metadata_bytes) =
            self.quantized_values
                .as_ref()
                .map_or((None, 0, 0, 0), |quantized_values| {
                    (
                        Some(quantized_values.value_bits()),
                        quantized_values.resident_bytes(),
                        quantized_values.payload_bytes(),
                        quantized_values.metadata_bytes(),
                    )
                });
        let phase3_uploaded_bytes = if self.quantized_values.is_some() {
            uploaded_bytes(self.keys().len(), 2)
                .saturating_add(phase3_payload_bytes)
                .saturating_add(phase3_metadata_bytes)
        } else {
            0
        };
        let phase3_reconstruction_error =
            self.phase3_reconstruction_error(phase3_value_bits.is_some())?;
        Ok(KvCacheFormatMetrics::from_parts(KvCacheFormatMetricParts {
            active_format: self.format(),
            phase3_value_bits,
            f32_resident_bytes: f32_residency,
            f16_resident_bytes,
            int8_resident_bytes,
            f32_uploaded_bytes,
            f16_uploaded_bytes,
            int8_uploaded_bytes,
            phase3_resident_bytes,
            phase3_value_payload_bytes: phase3_payload_bytes,
            phase3_value_metadata_bytes: phase3_metadata_bytes,
            phase3_uploaded_bytes,
            phase3_reconstruction_error,
        }))
    }

    pub fn append(&mut self, key: &[f32], value: &[f32]) -> Result<usize, KvCacheError> {
        self.validate_token_shape(key, value)?;
        self.validate_compressed_token_payload(key, value)?;
        if self.token_count == self.max_tokens {
            return Err(KvCacheError::CapacityExceeded {
                requested: 1,
                available: 0,
            });
        }
        let tokens_seen = self
            .tokens_seen
            .checked_add(1)
            .ok_or(KvCacheError::InvalidShape)?;
        let token_index = self.token_count;
        self.append_block_token(token_index, key, value)?;
        self.write_stage_token(token_index, key, value);
        self.token_count += 1;
        self.tokens_seen = tokens_seen;
        self.revision = self.revision.saturating_add(1);
        tracing::trace!(
            operation = "layer_kv_cache_append",
            cache_id = self.id,
            revision = self.revision,
            token_index,
            token_count = self.token_count,
            max_tokens = self.max_tokens,
            "layer KV cache token appended"
        );
        Ok(token_index)
    }

    pub fn append_sliding(&mut self, key: &[f32], value: &[f32]) -> Result<usize, KvCacheError> {
        self.validate_token_shape(key, value)?;
        if self.token_count < self.max_tokens {
            return self.append(key, value);
        }
        self.validate_compressed_token_payload(key, value)?;
        let tokens_seen = self
            .tokens_seen
            .checked_add(1)
            .ok_or(KvCacheError::InvalidShape)?;
        let physical_token_index = self.window_start;
        let token_index = self.max_tokens - 1;
        self.write_block_token(physical_token_index, key, value)?;
        self.write_stage_token(physical_token_index, key, value);
        self.window_start = (self.window_start + 1) % self.max_tokens;
        self.tokens_seen = tokens_seen;
        self.revision = self.revision.saturating_add(1);
        tracing::trace!(
            operation = "layer_kv_cache_append_sliding",
            cache_id = self.id,
            revision = self.revision,
            token_index,
            physical_token_index,
            token_count = self.token_count,
            window_start = self.window_start,
            max_tokens = self.max_tokens,
            "layer KV cache sliding token appended"
        );
        Ok(token_index)
    }

    pub fn key(&self, token_index: usize) -> Option<&[f32]> {
        let Some((block_index, block_token_index)) = self.block_position(token_index) else {
            tracing::trace!(
                operation = "layer_kv_cache_key_lookup",
                cache_id = self.id,
                revision = self.revision,
                token_index,
                cache_hit = false,
                miss_reason = "token_out_of_window",
                "layer KV cache key lookup missed"
            );
            return None;
        };
        let Some(block_id) = self.block_table.get(block_index) else {
            tracing::trace!(
                operation = "layer_kv_cache_key_lookup",
                cache_id = self.id,
                revision = self.revision,
                token_index,
                block_index,
                cache_hit = false,
                miss_reason = "missing_block_table_entry",
                "layer KV cache key lookup missed"
            );
            return None;
        };
        let Some(block) = self.blocks.get(block_index) else {
            tracing::trace!(
                operation = "layer_kv_cache_key_lookup",
                cache_id = self.id,
                revision = self.revision,
                token_index,
                block_index,
                cache_hit = false,
                miss_reason = "missing_block",
                "layer KV cache key lookup missed"
            );
            return None;
        };
        if block.id() != block_id {
            tracing::trace!(
                operation = "layer_kv_cache_key_lookup",
                cache_id = self.id,
                revision = self.revision,
                token_index,
                block_index,
                expected_block_id = ?block_id,
                actual_block_id = ?block.id(),
                cache_hit = false,
                miss_reason = "stale_block_table_entry",
                "layer KV cache key lookup missed"
            );
            return None;
        }
        let key = block.key(block_token_index);
        tracing::trace!(
            operation = "layer_kv_cache_key_lookup",
            cache_id = self.id,
            revision = self.revision,
            token_index,
            block_index,
            block_token_index,
            block_id = ?block_id,
            cache_hit = key.is_some(),
            "layer KV cache key lookup complete"
        );
        key
    }

    pub fn value(&self, token_index: usize) -> Option<&[f32]> {
        let (block_index, block_token_index) = self.block_position(token_index)?;
        let block_id = self.block_table.get(block_index)?;
        let block = self.blocks.get(block_index)?;
        if block.id() != block_id {
            return None;
        }
        block.value(block_token_index)
    }

    pub fn block_ids(&self) -> &[BlockId] {
        self.block_table.as_slice()
    }

    pub fn retained_block_ref_count(&self, block_id: BlockId) -> Option<usize> {
        self.blocks
            .iter()
            .find(|block| block.id() == block_id)
            .map(CacheBlock::retained_storage_ref_count)
    }

    pub fn active_blocks(&self) -> Result<Vec<LayerKvCacheBlock<'_>>, KvCacheError> {
        let mut active_blocks: Vec<LayerKvCacheBlock<'_>> = Vec::new();
        for logical_token_index in 0..self.token_count {
            let (block_index, block_token_index) = self
                .block_position(logical_token_index)
                .ok_or(KvCacheError::InvalidShape)?;
            let block_id = self
                .block_table
                .get(block_index)
                .ok_or(KvCacheError::InvalidShape)?;
            let block = self
                .blocks
                .get(block_index)
                .ok_or(KvCacheError::InvalidShape)?;
            if block.id() != block_id {
                return Err(KvCacheError::InvalidShape);
            }
            if let Some(previous) = active_blocks.last_mut()
                && previous.block_id == block_id
                && previous.block_token_start + previous.token_count == block_token_index
            {
                previous.token_count += 1;
                continue;
            }
            active_blocks.push(LayerKvCacheBlock {
                block_id,
                revision: block.revision(),
                logical_token_start: logical_token_index,
                block_token_start: block_token_index,
                token_count: 1,
                vector_len: self.vector_len(),
                key_storage: block.key_storage(),
                value_storage: block.value_storage(),
            });
        }
        Ok(active_blocks)
    }

    pub fn active_int8_blocks(
        &self,
    ) -> Result<Option<Vec<LayerKvCacheInt8Block<'_>>>, KvCacheError> {
        let Some(int8_storage) = self.int8_storage.as_ref() else {
            return Ok(None);
        };
        let mut active_blocks: Vec<LayerKvCacheInt8Block<'_>> = Vec::new();
        for logical_token_index in 0..self.token_count {
            let (block_index, block_token_index) = self
                .block_position(logical_token_index)
                .ok_or(KvCacheError::InvalidShape)?;
            let block_id = self
                .block_table
                .get(block_index)
                .ok_or(KvCacheError::InvalidShape)?;
            let block = self
                .blocks
                .get(block_index)
                .ok_or(KvCacheError::InvalidShape)?;
            if block.id() != block_id {
                return Err(KvCacheError::InvalidShape);
            }
            let int8_block = int8_storage.block(block_index)?;
            if let Some(previous) = active_blocks.last_mut()
                && previous.block_id == block_id
                && previous.block_token_start + previous.token_count == block_token_index
            {
                previous.token_count += 1;
                continue;
            }
            active_blocks.push(LayerKvCacheInt8Block {
                block_id,
                revision: block.revision(),
                logical_token_start: logical_token_index,
                block_token_start: block_token_index,
                token_count: 1,
                vector_len: self.vector_len(),
                key_codes: int8_block.key_codes(),
                value_codes: int8_block.value_codes(),
                key_scales: int8_block.key_scales(),
                value_scales: int8_block.value_scales(),
            });
        }
        Ok(Some(active_blocks))
    }

    pub fn keys(&self) -> &[f32] {
        let start = self.window_start * self.vector_len();
        &self.key_stage[start..start + self.used_len()]
    }

    pub fn values(&self) -> &[f32] {
        let start = self.window_start * self.vector_len();
        &self.value_stage[start..start + self.used_len()]
    }

    pub fn key_storage(&self) -> &[f32] {
        &self.key_stage[..self.stage_storage_len()]
    }

    pub fn value_storage(&self) -> &[f32] {
        &self.value_stage[..self.stage_storage_len()]
    }

    pub fn clear(&mut self) {
        self.token_count = 0;
        self.tokens_seen = 0;
        self.window_start = 0;
        for block in &mut self.blocks {
            block.clear();
        }
        if let Some(quantized_values) = self.quantized_values.as_mut() {
            quantized_values.clear();
        }
        if let Some(int8_storage) = self.int8_storage.as_mut() {
            int8_storage.clear();
        }
        self.key_stage.fill(0.0);
        self.value_stage.fill(0.0);
        self.revision = self.revision.saturating_add(1);
    }

    fn block_position(&self, token_index: usize) -> Option<(usize, usize)> {
        if token_index >= self.token_count {
            return None;
        }
        self.physical_block_position((self.window_start + token_index) % self.max_tokens)
    }

    fn physical_block_position(&self, token_index: usize) -> Option<(usize, usize)> {
        if token_index >= self.max_tokens {
            return None;
        }
        Some((
            token_index / LAYER_KV_BLOCK_TOKENS,
            token_index % LAYER_KV_BLOCK_TOKENS,
        ))
    }

    fn append_block_token(
        &mut self,
        token_index: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<(), KvCacheError> {
        let (block_index, expected_block_token_index) = self
            .physical_block_position(token_index)
            .ok_or(KvCacheError::InvalidShape)?;
        let block_id = self
            .block_table
            .get(block_index)
            .ok_or(KvCacheError::InvalidShape)?;
        let write_block_id = {
            let block = self
                .blocks
                .get_mut(block_index)
                .ok_or(KvCacheError::InvalidShape)?;
            if block.id() != block_id {
                return Err(KvCacheError::InvalidShape);
            }
            block.ensure_exclusive_identity()?;
            block.id()
        };
        if write_block_id != block_id {
            self.block_table.replace(block_index, write_block_id)?;
        }
        let block = self
            .blocks
            .get_mut(block_index)
            .ok_or(KvCacheError::InvalidShape)?;
        let block_token_index = block.append(key, value)?;
        if block_token_index != expected_block_token_index {
            return Err(KvCacheError::InvalidShape);
        }
        self.refresh_int8_block(block_index)?;
        self.refresh_quantized_block(block_index)?;
        Ok(())
    }

    fn write_block_token(
        &mut self,
        token_index: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<(), KvCacheError> {
        let (block_index, block_token_index) = self
            .physical_block_position(token_index)
            .ok_or(KvCacheError::InvalidShape)?;
        let block_id = self
            .block_table
            .get(block_index)
            .ok_or(KvCacheError::InvalidShape)?;
        let write_block_id = {
            let block = self
                .blocks
                .get_mut(block_index)
                .ok_or(KvCacheError::InvalidShape)?;
            if block.id() != block_id {
                return Err(KvCacheError::InvalidShape);
            }
            block.ensure_exclusive_identity()?;
            block.id()
        };
        if write_block_id != block_id {
            self.block_table.replace(block_index, write_block_id)?;
        }
        let block = self
            .blocks
            .get_mut(block_index)
            .ok_or(KvCacheError::InvalidShape)?;
        block.write_at(block_token_index, key, value)?;
        self.refresh_int8_block(block_index)?;
        self.refresh_quantized_block(block_index)
    }

    fn write_stage_token(&mut self, token_index: usize, key: &[f32], value: &[f32]) {
        let vector_len = self.vector_len();
        let start = token_index * vector_len;
        let end = start + vector_len;
        self.key_stage[start..end].copy_from_slice(key);
        self.value_stage[start..end].copy_from_slice(value);
        let mirror_start = (token_index + self.max_tokens) * vector_len;
        let mirror_end = mirror_start + vector_len;
        self.key_stage[mirror_start..mirror_end].copy_from_slice(key);
        self.value_stage[mirror_start..mirror_end].copy_from_slice(value);
    }

    fn rebuild_stage_from_blocks(&mut self) -> Result<(), KvCacheError> {
        self.key_stage.fill(0.0);
        self.value_stage.fill(0.0);
        for logical_token_index in 0..self.token_count {
            let physical_token_index = (self.window_start + logical_token_index) % self.max_tokens;
            let (block_index, block_token_index) = self
                .physical_block_position(physical_token_index)
                .ok_or(KvCacheError::InvalidShape)?;
            let block = self
                .blocks
                .get(block_index)
                .ok_or(KvCacheError::InvalidShape)?;
            let key = block
                .key(block_token_index)
                .ok_or(KvCacheError::InvalidShape)?
                .to_vec();
            let value = block
                .value(block_token_index)
                .ok_or(KvCacheError::InvalidShape)?
                .to_vec();
            self.write_stage_token(physical_token_index, &key, &value);
        }
        self.rebuild_int8_storage()?;
        self.rebuild_quantized_values()
    }

    fn rebuild_int8_storage(&mut self) -> Result<(), KvCacheError> {
        let Some(int8_storage) = self.int8_storage.as_mut() else {
            return Ok(());
        };
        int8_storage.clear();
        for block_index in 0..self.blocks.len() {
            let Some(block) = self.blocks.get(block_index) else {
                return Err(KvCacheError::InvalidShape);
            };
            if block.token_count() == 0 {
                continue;
            }
            int8_storage.update_block(block_index, block.keys(), block.values())?;
        }
        Ok(())
    }

    fn rebuild_quantized_values(&mut self) -> Result<(), KvCacheError> {
        let Some(quantized_values) = self.quantized_values.as_mut() else {
            return Ok(());
        };
        quantized_values.clear();
        for block_index in 0..self.blocks.len() {
            let Some(block) = self.blocks.get(block_index) else {
                return Err(KvCacheError::InvalidShape);
            };
            if block.token_count() == 0 {
                continue;
            }
            quantized_values.update_block(block_index, block.values())?;
        }
        Ok(())
    }

    fn refresh_quantized_block(&mut self, block_index: usize) -> Result<(), KvCacheError> {
        let Some(quantized_values) = self.quantized_values.as_mut() else {
            return Ok(());
        };
        let block = self
            .blocks
            .get(block_index)
            .ok_or(KvCacheError::InvalidShape)?;
        if block.token_count() == 0 {
            return Ok(());
        }
        quantized_values.update_block(block_index, block.values())
    }

    fn refresh_int8_block(&mut self, block_index: usize) -> Result<(), KvCacheError> {
        let Some(int8_storage) = self.int8_storage.as_mut() else {
            return Ok(());
        };
        let block = self
            .blocks
            .get(block_index)
            .ok_or(KvCacheError::InvalidShape)?;
        if block.token_count() == 0 {
            return Ok(());
        }
        int8_storage.update_block(block_index, block.keys(), block.values())
    }

    fn int8_dequantized_tensor(
        &self,
        tensor: Int8Tensor,
    ) -> Result<Option<Vec<f32>>, KvCacheError> {
        let Some(int8_storage) = self.int8_storage.as_ref() else {
            return Ok(None);
        };
        let mut values = Vec::with_capacity(self.used_len());
        for logical_token_index in 0..self.token_count {
            let physical_token_index = (self.window_start + logical_token_index) % self.max_tokens;
            let (block_index, block_token_index) = self
                .physical_block_position(physical_token_index)
                .ok_or(KvCacheError::InvalidShape)?;
            let block_values = match tensor {
                Int8Tensor::Key => int8_storage.dequantized_key_block(block_index)?,
                Int8Tensor::Value => int8_storage.dequantized_value_block(block_index)?,
            };
            let start = block_token_index
                .checked_mul(self.vector_len())
                .ok_or(KvCacheError::InvalidShape)?;
            let end = start
                .checked_add(self.vector_len())
                .ok_or(KvCacheError::InvalidShape)?;
            values.extend_from_slice(
                block_values
                    .get(start..end)
                    .ok_or(KvCacheError::InvalidShape)?,
            );
        }
        Ok(Some(values))
    }

    fn phase3_reconstruction_error(
        &self,
        enabled: bool,
    ) -> Result<Option<KvCacheReconstructionError>, KvCacheError> {
        if !enabled {
            return Ok(None);
        }
        let Some(decoded) = self.phase3_dequantized_values()? else {
            return Ok(None);
        };
        if decoded.is_empty() {
            return Ok(Some(KvCacheReconstructionError::new(0.0, 0.0)));
        }
        if decoded.len() != self.values().len() {
            return Err(KvCacheError::ShapeMismatch {
                expected: self.values().len(),
                actual: decoded.len(),
            });
        }
        let mut squared_error = 0.0_f64;
        let mut max_abs = 0.0_f32;
        for (expected, actual) in self.values().iter().zip(decoded) {
            let delta = expected - actual;
            squared_error += f64::from(delta * delta);
            max_abs = max_abs.max(delta.abs());
        }
        Ok(Some(KvCacheReconstructionError::new(
            squared_error / self.values().len() as f64,
            max_abs,
        )))
    }

    fn validate_token_shape(&self, key: &[f32], value: &[f32]) -> Result<(), KvCacheError> {
        let vector_len = self.vector_len();
        if key.len() != vector_len {
            return Err(KvCacheError::ShapeMismatch {
                expected: vector_len,
                actual: key.len(),
            });
        }
        if value.len() != vector_len {
            return Err(KvCacheError::ShapeMismatch {
                expected: vector_len,
                actual: value.len(),
            });
        }
        Ok(())
    }

    fn validate_compressed_token_payload(
        &self,
        key: &[f32],
        value: &[f32],
    ) -> Result<(), KvCacheError> {
        if self.quantized_values.is_none() && self.int8_storage.is_none() {
            return Ok(());
        }
        if key.iter().chain(value).any(|value| !value.is_finite()) {
            return Err(KvCacheError::NonFiniteValue);
        }
        Ok(())
    }

    fn used_len(&self) -> usize {
        self.token_count * self.vector_len()
    }

    fn stage_storage_len(&self) -> usize {
        self.max_tokens * self.vector_len()
    }
}

#[derive(Debug, Clone, Copy)]
enum Int8Tensor {
    Key,
    Value,
}

#[derive(Debug, PartialEq)]
pub struct LinearAttentionCache {
    id: u64,
    revision: u64,
    conv_kernel_size: usize,
    conv_dim: usize,
    num_value_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    token_count: usize,
    conv_window: Vec<f32>,
    recurrent_state: Vec<f32>,
}

/// Owned linear attention cache state, excluding runtime cache identity.
///
/// Restoring a snapshot allocates fresh backing storage and a fresh cache id.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearAttentionCacheSnapshot {
    pub revision: u64,
    pub conv_kernel_size: usize,
    pub conv_dim: usize,
    pub num_value_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub token_count: usize,
    pub conv_window: Vec<f32>,
    pub recurrent_state: Vec<f32>,
}

impl Clone for LinearAttentionCache {
    fn clone(&self) -> Self {
        Self {
            id: next_cache_id(),
            revision: self.revision,
            conv_kernel_size: self.conv_kernel_size,
            conv_dim: self.conv_dim,
            num_value_heads: self.num_value_heads,
            key_head_dim: self.key_head_dim,
            value_head_dim: self.value_head_dim,
            token_count: self.token_count,
            conv_window: self.conv_window.clone(),
            recurrent_state: self.recurrent_state.clone(),
        }
    }
}

impl LinearAttentionCache {
    pub fn new(
        conv_kernel_size: usize,
        conv_dim: usize,
        num_value_heads: usize,
        key_head_dim: usize,
        value_head_dim: usize,
    ) -> Result<Self, KvCacheError> {
        if conv_kernel_size == 0
            || conv_dim == 0
            || num_value_heads == 0
            || key_head_dim == 0
            || value_head_dim == 0
        {
            return Err(KvCacheError::InvalidShape);
        }
        let conv_len = conv_kernel_size
            .checked_mul(conv_dim)
            .ok_or(KvCacheError::InvalidShape)?;
        let recurrent_state_len = num_value_heads
            .checked_mul(key_head_dim)
            .and_then(|len| len.checked_mul(value_head_dim))
            .ok_or(KvCacheError::InvalidShape)?;
        Ok(Self {
            id: next_cache_id(),
            revision: 0,
            conv_kernel_size,
            conv_dim,
            num_value_heads,
            key_head_dim,
            value_head_dim,
            token_count: 0,
            conv_window: vec![0.0; conv_len],
            recurrent_state: vec![0.0; recurrent_state_len],
        })
    }

    pub fn snapshot(&self) -> LinearAttentionCacheSnapshot {
        LinearAttentionCacheSnapshot {
            revision: self.revision,
            conv_kernel_size: self.conv_kernel_size,
            conv_dim: self.conv_dim,
            num_value_heads: self.num_value_heads,
            key_head_dim: self.key_head_dim,
            value_head_dim: self.value_head_dim,
            token_count: self.token_count,
            conv_window: self.conv_window.clone(),
            recurrent_state: self.recurrent_state.clone(),
        }
    }

    pub fn from_snapshot(snapshot: LinearAttentionCacheSnapshot) -> Result<Self, KvCacheError> {
        let mut cache = Self::new(
            snapshot.conv_kernel_size,
            snapshot.conv_dim,
            snapshot.num_value_heads,
            snapshot.key_head_dim,
            snapshot.value_head_dim,
        )?;
        if snapshot.conv_window.len() != cache.conv_window.len() {
            return Err(KvCacheError::ShapeMismatch {
                expected: cache.conv_window.len(),
                actual: snapshot.conv_window.len(),
            });
        }
        if snapshot.recurrent_state.len() != cache.recurrent_state.len() {
            return Err(KvCacheError::ShapeMismatch {
                expected: cache.recurrent_state.len(),
                actual: snapshot.recurrent_state.len(),
            });
        }

        cache.revision = snapshot.revision;
        cache.token_count = snapshot.token_count;
        cache.conv_window = snapshot.conv_window;
        cache.recurrent_state = snapshot.recurrent_state;
        Ok(cache)
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn conv_kernel_size(&self) -> usize {
        self.conv_kernel_size
    }

    pub fn conv_dim(&self) -> usize {
        self.conv_dim
    }

    pub fn num_value_heads(&self) -> usize {
        self.num_value_heads
    }

    pub fn key_head_dim(&self) -> usize {
        self.key_head_dim
    }

    pub fn value_head_dim(&self) -> usize {
        self.value_head_dim
    }

    pub fn recurrent_state_len(&self) -> usize {
        self.recurrent_state.len()
    }

    pub fn token_count(&self) -> usize {
        self.token_count
    }

    /// Returns bytes retained by the allocated convolution and recurrent state storage.
    pub fn resident_bytes(&self) -> u64 {
        f32_resident_bytes(&self.conv_window)
            .saturating_add(f32_resident_bytes(&self.recurrent_state))
    }

    pub fn conv_window(&self) -> &[f32] {
        &self.conv_window
    }

    pub fn recurrent_state(&self) -> &[f32] {
        &self.recurrent_state
    }

    pub fn recurrent_state_mut(&mut self) -> &mut [f32] {
        &mut self.recurrent_state
    }

    pub fn push_conv_input(&mut self, input: &[f32]) -> Result<(), KvCacheError> {
        if input.len() != self.conv_dim {
            return Err(KvCacheError::ShapeMismatch {
                expected: self.conv_dim,
                actual: input.len(),
            });
        }
        self.conv_window.copy_within(self.conv_dim.., 0);
        let start = self.conv_window.len() - self.conv_dim;
        self.conv_window[start..].copy_from_slice(input);
        self.token_count = self.token_count.saturating_add(1);
        self.revision = self.revision.saturating_add(1);
        Ok(())
    }

    pub fn replace_recurrent_state(&mut self, state: &[f32]) -> Result<(), KvCacheError> {
        if state.len() != self.recurrent_state.len() {
            return Err(KvCacheError::ShapeMismatch {
                expected: self.recurrent_state.len(),
                actual: state.len(),
            });
        }
        self.recurrent_state.copy_from_slice(state);
        self.revision = self.revision.saturating_add(1);
        Ok(())
    }

    pub fn replace_recurrent_state_range(
        &mut self,
        start: usize,
        state: &[f32],
    ) -> Result<(), KvCacheError> {
        let end = start
            .checked_add(state.len())
            .ok_or(KvCacheError::InvalidShape)?;
        if end > self.recurrent_state.len() {
            return Err(KvCacheError::ShapeMismatch {
                expected: self.recurrent_state.len().saturating_sub(start),
                actual: state.len(),
            });
        }
        self.recurrent_state[start..end].copy_from_slice(state);
        self.revision = self.revision.saturating_add(1);
        Ok(())
    }

    pub fn clear(&mut self) {
        self.token_count = 0;
        self.conv_window.fill(0.0);
        self.recurrent_state.fill(0.0);
        self.revision = self.revision.saturating_add(1);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvCacheError {
    CapacityExceeded { requested: usize, available: usize },
    ShapeMismatch { expected: usize, actual: usize },
    UnsupportedFormat { format: KvCacheFormat },
    NonFiniteValue,
    InvalidShape,
}

impl fmt::Display for KvCacheError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CapacityExceeded {
                requested,
                available,
            } => write!(
                formatter,
                "KV cache capacity exceeded: requested {requested} tokens, {available} available"
            ),
            Self::ShapeMismatch { expected, actual } => write!(
                formatter,
                "KV cache shape mismatch: expected {expected} values, got {actual}"
            ),
            Self::UnsupportedFormat { format } => {
                write!(formatter, "KV cache format {format} is not supported")
            }
            Self::NonFiniteValue => write!(formatter, "KV cache values must be finite"),
            Self::InvalidShape => {
                write!(formatter, "KV cache shape must be non-zero and fit usize")
            }
        }
    }
}

impl std::error::Error for KvCacheError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt;
    use std::sync::{Arc, Mutex, OnceLock};
    use tracing::field::{Field, Visit};
    use tracing::{Event, Id, Metadata, Subscriber, span};

    #[test]
    fn budget_tracks_reserved_and_released_tokens() {
        let mut budget = KvCacheBudget::new(8);

        budget.reserve(3).expect("reserve fits");
        assert_eq!(budget.max_tokens(), 8);
        assert_eq!(budget.used_tokens(), 3);
        assert_eq!(budget.remaining_tokens(), 5);

        budget.release(2);
        assert_eq!(budget.used_tokens(), 1);
        assert_eq!(budget.remaining_tokens(), 7);
    }

    #[test]
    fn budget_rejects_over_capacity_reservation() {
        let mut budget = KvCacheBudget::new(4);
        budget.reserve(3).expect("first reserve fits");

        let err = budget.reserve(2).expect_err("capacity is enforced");

        assert_eq!(
            err,
            KvCacheError::CapacityExceeded {
                requested: 2,
                available: 1
            }
        );
        assert_eq!(budget.used_tokens(), 3);
    }

    #[test]
    fn release_saturates_at_zero() {
        let mut budget = KvCacheBudget::new(4);
        budget.reserve(1).expect("reserve fits");

        budget.release(99);

        assert_eq!(budget.used_tokens(), 0);
        assert_eq!(budget.remaining_tokens(), 4);
    }

    #[test]
    fn layer_kv_cache_appends_and_reads_token_slices() {
        let mut cache = LayerKvCache::new(3, 2, 2).expect("cache shape is valid");

        let initial_revision = cache.revision();
        assert!(cache.id() > 0);
        assert_eq!(cache.max_tokens(), 3);
        assert_eq!(cache.token_count(), 0);
        assert_eq!(cache.vector_len(), 4);

        assert_eq!(
            cache
                .append(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0])
                .expect("first token fits"),
            0
        );
        assert_eq!(
            cache
                .append(&[5.0, 6.0, 7.0, 8.0], &[50.0, 60.0, 70.0, 80.0])
                .expect("second token fits"),
            1
        );

        assert_eq!(cache.token_count(), 2);
        assert_eq!(cache.key(0), Some(&[1.0, 2.0, 3.0, 4.0][..]));
        assert_eq!(cache.value(1), Some(&[50.0, 60.0, 70.0, 80.0][..]));
        assert_eq!(cache.keys(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(
            cache.values(),
            &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
        );
        assert!(cache.revision() > initial_revision);

        cache.clear();
        assert_eq!(cache.token_count(), 0);
        assert_eq!(cache.key(0), None);
        assert!(cache.revision() > initial_revision);
    }

    #[test]
    fn layer_kv_cache_key_lookup_emits_trace_metadata() {
        let mut cache = LayerKvCache::new(2, 1, 1).expect("cache shape is valid");
        cache.append(&[1.0], &[10.0]).expect("token fits");

        let capture = TraceCapture::start();
        assert_eq!(cache.key(0), Some(&[1.0][..]));
        assert_eq!(cache.key(1), None);
        let events = capture.events();

        assert!(
            events.iter().any(|event| {
                event.has_field("operation", "layer_kv_cache_key_lookup")
                    && event.has_field("cache_hit", "true")
                    && event.has_field("token_index", "0")
            }),
            "key hit should emit structured trace metadata, got {events:?}"
        );
        assert!(
            events.iter().any(|event| {
                event.has_field("operation", "layer_kv_cache_key_lookup")
                    && event.has_field("cache_hit", "false")
                    && event.has_field("token_index", "1")
            }),
            "key miss should emit structured trace metadata, got {events:?}"
        );
    }

    #[test]
    fn layer_kv_cache_clone_preserves_state_with_fresh_identity() {
        let mut cache = LayerKvCache::new(3, 1, 2).expect("cache shape is valid");
        cache.append(&[1.0, 2.0], &[3.0, 4.0]).expect("token fits");

        let clone = cache.clone();

        assert_ne!(clone.id(), cache.id());
        assert_eq!(clone.revision(), cache.revision());
        assert_eq!(clone.token_count(), cache.token_count());
        assert_eq!(clone.keys(), cache.keys());
        assert_eq!(clone.values(), cache.values());
    }

    #[test]
    fn layer_kv_cache_snapshot_round_trips_used_storage_and_shape() {
        let mut cache = LayerKvCache::new(4, 2, 3).expect("cache shape is valid");
        cache
            .append(
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            )
            .expect("first token fits");
        cache
            .append(
                &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                &[70.0, 80.0, 90.0, 100.0, 110.0, 120.0],
            )
            .expect("second token fits");

        let snapshot = cache.snapshot();
        let restored = LayerKvCache::from_snapshot(snapshot).expect("snapshot restores");

        assert_ne!(restored.id(), cache.id());
        assert_eq!(restored.revision(), cache.revision());
        assert_eq!(restored.max_tokens(), cache.max_tokens());
        assert_eq!(restored.key_value_heads(), cache.key_value_heads());
        assert_eq!(restored.head_dim(), cache.head_dim());
        assert_eq!(restored.token_count(), cache.token_count());
        assert_eq!(restored.next_position(), cache.next_position());
        assert_eq!(restored.keys(), cache.keys());
        assert_eq!(restored.values(), cache.values());
        assert_eq!(
            &restored.key_storage()[restored.keys().len()..],
            vec![0.0; restored.key_storage().len() - restored.keys().len()]
        );
    }

    #[test]
    fn layer_kv_cache_restore_rejects_bad_storage_len() {
        let mut cache = LayerKvCache::new(2, 1, 2).expect("cache shape is valid");
        cache.append(&[1.0, 2.0], &[3.0, 4.0]).expect("token fits");
        let mut snapshot = cache.snapshot();
        snapshot.keys.pop();

        let err = LayerKvCache::from_snapshot(snapshot).expect_err("bad storage length fails");

        assert_eq!(
            err,
            KvCacheError::ShapeMismatch {
                expected: 2,
                actual: 1
            }
        );
    }

    #[test]
    fn layer_kv_cache_rejects_shape_mismatch_and_capacity_overflow() {
        let mut cache = LayerKvCache::new(1, 1, 2).expect("cache shape is valid");

        let err = cache
            .append(&[1.0], &[2.0, 3.0])
            .expect_err("key shape mismatch fails");
        assert_eq!(
            err,
            KvCacheError::ShapeMismatch {
                expected: 2,
                actual: 1
            }
        );

        cache
            .append(&[1.0, 2.0], &[3.0, 4.0])
            .expect("first token fits");
        let err = cache
            .append(&[5.0, 6.0], &[7.0, 8.0])
            .expect_err("capacity is enforced");
        assert_eq!(
            err,
            KvCacheError::CapacityExceeded {
                requested: 1,
                available: 0
            }
        );

        let err = LayerKvCache::new(1, 0, 2).expect_err("zero heads are invalid");
        assert_eq!(err, KvCacheError::InvalidShape);
    }

    #[test]
    fn layer_kv_cache_sliding_append_evicts_oldest_token_when_full() {
        let mut cache = LayerKvCache::new(2, 1, 2).expect("cache shape is valid");

        assert_eq!(
            cache
                .append_sliding(&[1.0, 2.0], &[10.0, 20.0])
                .expect("first token fits"),
            0
        );
        assert_eq!(cache.next_position(), 1);
        assert_eq!(
            cache
                .append_sliding(&[3.0, 4.0], &[30.0, 40.0])
                .expect("second token fits"),
            1
        );
        assert_eq!(cache.next_position(), 2);
        assert_eq!(
            cache
                .append_sliding(&[5.0, 6.0], &[50.0, 60.0])
                .expect("full cache evicts oldest token"),
            1
        );
        assert_eq!(cache.next_position(), 3);

        assert_eq!(cache.token_count(), 2);
        assert_eq!(cache.key(0), Some(&[3.0, 4.0][..]));
        assert_eq!(cache.value(0), Some(&[30.0, 40.0][..]));
        assert_eq!(cache.key(1), Some(&[5.0, 6.0][..]));
        assert_eq!(cache.value(1), Some(&[50.0, 60.0][..]));
        assert_eq!(cache.keys(), &[3.0, 4.0, 5.0, 6.0]);
        assert_eq!(cache.values(), &[30.0, 40.0, 50.0, 60.0]);
        assert_eq!(cache.remaining_tokens(), 0);
    }

    #[test]
    fn layer_kv_cache_sliding_append_reuses_ring_slot_without_shifting_storage() {
        let mut cache = LayerKvCache::new(3, 1, 1).expect("cache shape is valid");

        cache.append_sliding(&[1.0], &[10.0]).expect("token fits");
        cache.append_sliding(&[2.0], &[20.0]).expect("token fits");
        cache.append_sliding(&[3.0], &[30.0]).expect("token fits");
        cache
            .append_sliding(&[4.0], &[40.0])
            .expect("full cache recycles oldest slot");

        assert_eq!(cache.keys(), &[2.0, 3.0, 4.0]);
        assert_eq!(cache.values(), &[20.0, 30.0, 40.0]);
        assert_eq!(cache.key(0), Some(&[2.0][..]));
        assert_eq!(cache.value(2), Some(&[40.0][..]));
        assert_eq!(cache.key_storage(), &[4.0, 2.0, 3.0]);
        assert_eq!(cache.value_storage(), &[40.0, 20.0, 30.0]);
    }

    #[test]
    fn layer_kv_cache_sliding_append_keeps_block_storage_bounded_across_cycles() {
        let mut cache = LayerKvCache::new(3, 1, 1).expect("cache shape is valid");
        let initial_resident_bytes = cache.resident_bytes();
        let block_ids = cache.blocks.iter().map(CacheBlock::id).collect::<Vec<_>>();

        for token in 1..=12 {
            let key = [token as f32];
            let value = [(token as f32) * 10.0];
            cache
                .append_sliding(&key, &value)
                .expect("sliding append succeeds");
        }

        assert_eq!(cache.keys(), &[10.0, 11.0, 12.0]);
        assert_eq!(cache.values(), &[100.0, 110.0, 120.0]);
        assert_eq!(cache.token_count(), 3);
        assert_eq!(cache.next_position(), 12);
        assert_eq!(cache.resident_bytes(), initial_resident_bytes);
        assert_eq!(
            cache.blocks.iter().map(CacheBlock::id).collect::<Vec<_>>(),
            block_ids
        );
    }

    #[test]
    fn layer_kv_cache_stores_rows_in_blocks_while_preserving_contiguous_views() {
        let max_tokens = LAYER_KV_BLOCK_TOKENS + 1;
        let mut cache = LayerKvCache::new(max_tokens, 1, 1).expect("cache shape is valid");

        assert_eq!(cache.blocks.len(), 2);
        assert_eq!(cache.block_table.block_count(), 2);
        for (index, block) in cache.blocks.iter().enumerate() {
            assert_eq!(cache.block_table.get(index), Some(block.id()));
        }

        for token in 0..max_tokens {
            let key = [token as f32];
            let value = [(token as f32) * 10.0];
            assert_eq!(cache.append(&key, &value).expect("token appends"), token);
        }

        assert_eq!(cache.blocks[0].token_count(), LAYER_KV_BLOCK_TOKENS);
        assert_eq!(cache.blocks[1].token_count(), 1);
        assert_eq!(
            cache.blocks[0].key(LAYER_KV_BLOCK_TOKENS - 1),
            Some(&[(LAYER_KV_BLOCK_TOKENS - 1) as f32][..])
        );
        assert_eq!(
            cache.blocks[1].key(0),
            Some(&[LAYER_KV_BLOCK_TOKENS as f32][..])
        );
        assert_eq!(cache.keys().len(), max_tokens);
        assert_eq!(
            cache.values()[max_tokens - 1],
            (LAYER_KV_BLOCK_TOKENS as f32) * 10.0
        );

        cache
            .append_sliding(&[999.0], &[9990.0])
            .expect("sliding append evicts oldest token");

        assert_eq!(cache.keys()[0], 1.0);
        assert_eq!(cache.keys()[max_tokens - 1], 999.0);
        assert_eq!(cache.key(0), Some(&[1.0][..]));
        assert_eq!(cache.key(max_tokens - 1), Some(&[999.0][..]));
        assert_eq!(cache.key_storage()[0], 999.0);
        assert_eq!(cache.blocks[0].key(0), Some(&[999.0][..]));
        assert_eq!(
            cache.blocks[0].key(LAYER_KV_BLOCK_TOKENS - 1),
            Some(&[(LAYER_KV_BLOCK_TOKENS - 1) as f32][..])
        );
        assert_eq!(
            cache.blocks[1].key(0),
            Some(&[LAYER_KV_BLOCK_TOKENS as f32][..])
        );
    }

    #[test]
    fn layer_kv_cache_resident_bytes_count_stage_and_block_storage() {
        let mut cache = LayerKvCache::new(3, 2, 2).expect("cache shape is valid");

        assert_eq!(cache.resident_bytes(), 288);

        cache
            .append(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0])
            .expect("token fits");
        assert_eq!(cache.token_count(), 1);
        assert_eq!(cache.resident_bytes(), 288);
    }

    #[test]
    fn layer_kv_cache_active_blocks_expose_logical_rows_and_revisions() {
        let max_tokens = LAYER_KV_BLOCK_TOKENS + 1;
        let mut cache = LayerKvCache::new(max_tokens, 1, 1).expect("cache shape is valid");

        for token in 0..max_tokens {
            cache
                .append(&[token as f32], &[1000.0 + token as f32])
                .expect("token appends");
        }

        let active_blocks = cache.active_blocks().expect("active block view is valid");
        assert_eq!(active_blocks.len(), 2);
        assert_eq!(active_blocks[0].logical_token_start(), 0);
        assert_eq!(active_blocks[0].block_token_start(), 0);
        assert_eq!(active_blocks[0].token_count(), LAYER_KV_BLOCK_TOKENS);
        assert_eq!(active_blocks[0].keys()[0], 0.0);
        assert_eq!(
            active_blocks[0].keys()[LAYER_KV_BLOCK_TOKENS - 1],
            (LAYER_KV_BLOCK_TOKENS - 1) as f32
        );
        assert_eq!(
            active_blocks[1].logical_token_start(),
            LAYER_KV_BLOCK_TOKENS
        );
        assert_eq!(active_blocks[1].block_token_start(), 0);
        assert_eq!(active_blocks[1].token_count(), 1);
        assert_eq!(active_blocks[1].keys(), &[LAYER_KV_BLOCK_TOKENS as f32]);

        let first_revision = active_blocks[0].revision();
        let second_revision = active_blocks[1].revision();
        cache
            .append_sliding(&[999.0], &[1999.0])
            .expect("sliding append overwrites the first physical block row");
        let active_blocks = cache.active_blocks().expect("active block view is valid");

        assert_eq!(
            active_blocks[0].block_id(),
            active_blocks[2].block_id(),
            "sliding wrap should split the same physical block into logical runs"
        );
        assert!(active_blocks[0].revision() > first_revision);
        assert_eq!(active_blocks[1].revision(), second_revision);
        assert_eq!(active_blocks[0].keys()[0], 1.0);
        assert_eq!(
            active_blocks[2].keys()[active_blocks[2].token_count() - 1],
            999.0
        );
    }

    #[test]
    fn cloned_layer_kv_cache_forks_block_identity_on_suffix_write() {
        let mut cache = LayerKvCache::new(4, 1, 1).expect("cache shape is valid");
        cache.append(&[1.0], &[10.0]).expect("prefix token fits");
        let prefix_block_id = cache.block_ids()[0];
        let prefix_revision = cache.active_blocks().expect("active blocks")[0].revision();

        let mut first_suffix = cache.clone();
        let mut second_suffix = cache.clone();
        first_suffix
            .append(&[2.0], &[20.0])
            .expect("first suffix token fits");
        second_suffix
            .append(&[3.0], &[30.0])
            .expect("second suffix token fits");

        assert_ne!(first_suffix.block_ids()[0], prefix_block_id);
        assert_ne!(second_suffix.block_ids()[0], prefix_block_id);
        assert_ne!(first_suffix.block_ids()[0], second_suffix.block_ids()[0]);
        assert_eq!(
            cache.active_blocks().expect("active blocks")[0].revision(),
            prefix_revision,
            "the cached prefix block remains unchanged"
        );
    }

    #[test]
    fn layer_kv_cache_prefix_state_retains_active_blocks_without_stage_payloads() {
        let max_tokens = LAYER_KV_BLOCK_TOKENS + 1;
        let mut cache = LayerKvCache::new(max_tokens, 1, 2).expect("cache shape is valid");
        cache
            .append(&[1.0, 2.0], &[10.0, 20.0])
            .expect("prefix token fits");

        let prefix_block_id = cache.block_ids()[0];
        let state = cache.prefix_cache_state();

        assert_eq!(state.block_ids(), vec![prefix_block_id]);
        assert_eq!(
            state.retained_block_payload_bytes(),
            (LAYER_KV_BLOCK_TOKENS * cache.vector_len() * 2 * std::mem::size_of::<f32>()) as u64
        );
        assert!(
            state.metadata_bytes() >= state.retained_block_payload_bytes(),
            "entry sizing must include the retained block payload it keeps alive"
        );
        assert!(
            state.metadata_bytes() < cache.resident_bytes(),
            "prefix state should not retain full cache staging or inactive block storage"
        );
        assert_eq!(cache.retained_block_ref_count(prefix_block_id), Some(2));
    }

    #[test]
    fn layer_kv_cache_prefix_hit_rebuilds_local_stage_and_cows_only_blocks() {
        let mut cache = LayerKvCache::new(4, 1, 1).expect("cache shape is valid");
        cache.append(&[1.0], &[10.0]).expect("prefix token fits");
        let prefix_block_id = cache.block_ids()[0];
        let prefix_block_key_ptr = cache.active_blocks().expect("active blocks")[0]
            .key_storage()
            .as_ptr();

        let state = cache.prefix_cache_state();
        let mut hit_cache =
            LayerKvCache::from_prefix_cache_state(&state).expect("prefix state restores");
        let hit_stage_ptr = hit_cache.key_storage().as_ptr();

        assert_eq!(hit_cache.block_ids()[0], prefix_block_id);
        assert_eq!(
            hit_cache.active_blocks().expect("active blocks")[0]
                .key_storage()
                .as_ptr(),
            prefix_block_key_ptr,
            "prefix hit should share retained block storage"
        );
        assert_eq!(cache.retained_block_ref_count(prefix_block_id), Some(3));

        hit_cache
            .append(&[2.0], &[20.0])
            .expect("suffix token fits");

        assert_eq!(
            hit_cache.key_storage().as_ptr(),
            hit_stage_ptr,
            "suffix writes should not COW-clone a shared full staging buffer"
        );
        assert_ne!(
            hit_cache.block_ids()[0],
            prefix_block_id,
            "suffix write should fork only the touched prefix block"
        );
        assert_eq!(cache.retained_block_ref_count(prefix_block_id), Some(2));
        assert_eq!(cache.key(0), Some(&[1.0][..]));
        assert_eq!(hit_cache.key(0), Some(&[1.0][..]));
        assert_eq!(hit_cache.key(1), Some(&[2.0][..]));
    }

    #[test]
    fn layer_kv_cache_clear_with_retained_prefix_forks_reused_block_identity() {
        let mut cache = LayerKvCache::new(4, 1, 1).expect("cache shape is valid");
        cache.append(&[1.0], &[10.0]).expect("prefix token fits");
        let prefix_block_id = cache.block_ids()[0];
        let state = cache.prefix_cache_state();

        cache.clear();
        cache.append(&[2.0], &[20.0]).expect("new token fits");

        assert_ne!(
            cache.block_ids()[0],
            prefix_block_id,
            "reusing a block whose storage is retained by a prefix entry must get a new id"
        );
        let restored =
            LayerKvCache::from_prefix_cache_state(&state).expect("prefix state restores");
        assert_eq!(restored.block_ids()[0], prefix_block_id);
        assert_eq!(restored.key(0), Some(&[1.0][..]));
    }

    #[test]
    fn linear_attention_cache_tracks_conv_window_and_recurrent_state() {
        let mut cache = LinearAttentionCache::new(2, 3, 1, 2, 2).expect("cache shape is valid");

        let initial_revision = cache.revision();
        assert!(cache.id() > 0);
        assert_eq!(cache.conv_kernel_size(), 2);
        assert_eq!(cache.conv_dim(), 3);
        assert_eq!(cache.recurrent_state_len(), 4);
        assert_eq!(cache.token_count(), 0);
        assert_eq!(cache.conv_window(), &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(cache.recurrent_state(), &[0.0, 0.0, 0.0, 0.0]);

        cache
            .push_conv_input(&[1.0, 2.0, 3.0])
            .expect("first conv input fits");
        assert_eq!(cache.token_count(), 1);
        assert_eq!(cache.conv_window(), &[0.0, 0.0, 0.0, 1.0, 2.0, 3.0]);

        cache
            .push_conv_input(&[4.0, 5.0, 6.0])
            .expect("second conv input fits");
        assert_eq!(cache.token_count(), 2);
        assert_eq!(cache.conv_window(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        cache
            .replace_recurrent_state(&[0.5, 1.5, 2.5, 3.5])
            .expect("state shape fits");
        assert_eq!(cache.recurrent_state(), &[0.5, 1.5, 2.5, 3.5]);
        cache
            .replace_recurrent_state_range(2, &[8.5, 9.5])
            .expect("state range fits");
        assert_eq!(cache.recurrent_state(), &[0.5, 1.5, 8.5, 9.5]);
        assert!(cache.revision() > initial_revision);

        cache.clear();
        assert_eq!(cache.token_count(), 0);
        assert_eq!(cache.conv_window(), &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(cache.recurrent_state(), &[0.0, 0.0, 0.0, 0.0]);
        assert!(cache.revision() > initial_revision);
    }

    #[test]
    fn linear_attention_cache_clone_preserves_state_with_fresh_identity() {
        let mut cache = LinearAttentionCache::new(2, 3, 1, 2, 2).expect("cache shape is valid");
        cache
            .push_conv_input(&[1.0, 2.0, 3.0])
            .expect("conv input fits");
        cache
            .replace_recurrent_state(&[0.5, 1.5, 2.5, 3.5])
            .expect("state shape fits");

        let clone = cache.clone();

        assert_ne!(clone.id(), cache.id());
        assert_eq!(clone.revision(), cache.revision());
        assert_eq!(clone.token_count(), cache.token_count());
        assert_eq!(clone.conv_window(), cache.conv_window());
        assert_eq!(clone.recurrent_state(), cache.recurrent_state());
    }

    #[test]
    fn linear_attention_cache_snapshot_round_trips_window_state_and_shape() {
        let mut cache = LinearAttentionCache::new(2, 3, 1, 2, 2).expect("cache shape is valid");
        cache
            .push_conv_input(&[1.0, 2.0, 3.0])
            .expect("first conv input fits");
        cache
            .push_conv_input(&[4.0, 5.0, 6.0])
            .expect("second conv input fits");
        cache
            .replace_recurrent_state(&[0.5, 1.5, 2.5, 3.5])
            .expect("state shape fits");

        let snapshot = cache.snapshot();
        let restored = LinearAttentionCache::from_snapshot(snapshot).expect("snapshot restores");

        assert_ne!(restored.id(), cache.id());
        assert_eq!(restored.revision(), cache.revision());
        assert_eq!(restored.conv_kernel_size(), cache.conv_kernel_size());
        assert_eq!(restored.conv_dim(), cache.conv_dim());
        assert_eq!(restored.num_value_heads(), cache.num_value_heads());
        assert_eq!(restored.key_head_dim(), cache.key_head_dim());
        assert_eq!(restored.value_head_dim(), cache.value_head_dim());
        assert_eq!(restored.token_count(), cache.token_count());
        assert_eq!(restored.conv_window(), cache.conv_window());
        assert_eq!(restored.recurrent_state(), cache.recurrent_state());
    }

    #[test]
    fn linear_attention_cache_restore_rejects_bad_state_len() {
        let mut cache = LinearAttentionCache::new(2, 3, 1, 2, 2).expect("cache shape is valid");
        cache
            .replace_recurrent_state(&[0.5, 1.5, 2.5, 3.5])
            .expect("state shape fits");
        let mut snapshot = cache.snapshot();
        snapshot.recurrent_state.push(4.5);

        let err =
            LinearAttentionCache::from_snapshot(snapshot).expect_err("bad state length fails");

        assert_eq!(
            err,
            KvCacheError::ShapeMismatch {
                expected: 4,
                actual: 5
            }
        );
    }

    #[test]
    fn linear_attention_cache_rejects_invalid_shapes() {
        let mut cache = LinearAttentionCache::new(2, 3, 1, 2, 2).expect("cache shape is valid");

        let err = cache
            .push_conv_input(&[1.0, 2.0])
            .expect_err("conv input shape mismatch fails");
        assert_eq!(
            err,
            KvCacheError::ShapeMismatch {
                expected: 3,
                actual: 2
            }
        );

        let err = cache
            .replace_recurrent_state(&[1.0, 2.0, 3.0])
            .expect_err("state shape mismatch fails");
        assert_eq!(
            err,
            KvCacheError::ShapeMismatch {
                expected: 4,
                actual: 3
            }
        );

        let err = LinearAttentionCache::new(0, 3, 1, 2, 2).expect_err("zero kernel is invalid");
        assert_eq!(err, KvCacheError::InvalidShape);
    }

    #[test]
    fn linear_attention_cache_resident_bytes_use_allocated_storage() {
        let mut cache = LinearAttentionCache::new(2, 3, 1, 2, 2).expect("cache shape is valid");

        assert_eq!(cache.resident_bytes(), 40);

        cache
            .push_conv_input(&[1.0, 2.0, 3.0])
            .expect("conv input fits");
        assert_eq!(cache.token_count(), 1);
        assert_eq!(cache.resident_bytes(), 40);
    }

    #[derive(Clone, Debug)]
    struct RecordedEvent {
        fields: Vec<(String, String)>,
    }

    impl RecordedEvent {
        fn has_field(&self, name: &str, value: &str) -> bool {
            self.fields
                .iter()
                .any(|(field, recorded)| field == name && recorded == value)
        }
    }

    static TRACE_EVENTS: OnceLock<Arc<Mutex<Vec<RecordedEvent>>>> = OnceLock::new();

    struct TraceCapture {
        events: Arc<Mutex<Vec<RecordedEvent>>>,
    }

    impl TraceCapture {
        fn start() -> Self {
            let events = Arc::clone(TRACE_EVENTS.get_or_init(|| {
                let events = Arc::new(Mutex::new(Vec::new()));
                let subscriber = RecordingSubscriber {
                    events: Arc::clone(&events),
                };
                tracing::subscriber::set_global_default(subscriber)
                    .expect("trace test subscriber installs once");
                events
            }));
            events.lock().expect("recorded events lock").clear();
            tracing::callsite::rebuild_interest_cache();
            Self { events }
        }

        fn events(&self) -> Vec<RecordedEvent> {
            self.events.lock().expect("recorded events lock").clone()
        }
    }

    struct RecordingSubscriber {
        events: Arc<Mutex<Vec<RecordedEvent>>>,
    }

    impl Subscriber for RecordingSubscriber {
        fn enabled(&self, _metadata: &Metadata<'_>) -> bool {
            true
        }

        fn register_callsite(
            &self,
            _metadata: &'static Metadata<'static>,
        ) -> tracing::subscriber::Interest {
            tracing::subscriber::Interest::always()
        }

        fn max_level_hint(&self) -> Option<tracing::level_filters::LevelFilter> {
            Some(tracing::level_filters::LevelFilter::TRACE)
        }

        fn new_span(&self, _span: &span::Attributes<'_>) -> Id {
            Id::from_u64(1)
        }

        fn record(&self, _span: &Id, _values: &span::Record<'_>) {}

        fn record_follows_from(&self, _span: &Id, _follows: &Id) {}

        fn event(&self, event: &Event<'_>) {
            let mut visitor = FieldRecorder::default();
            event.record(&mut visitor);
            self.events
                .lock()
                .expect("recorded events lock")
                .push(RecordedEvent {
                    fields: visitor.fields,
                });
        }

        fn enter(&self, _span: &Id) {}

        fn exit(&self, _span: &Id) {}
    }

    #[derive(Default)]
    struct FieldRecorder {
        fields: Vec<(String, String)>,
    }

    impl FieldRecorder {
        fn record_value(&mut self, field: &Field, value: String) {
            self.fields.push((field.name().to_owned(), value));
        }
    }

    impl Visit for FieldRecorder {
        fn record_bool(&mut self, field: &Field, value: bool) {
            self.record_value(field, value.to_string());
        }

        fn record_i64(&mut self, field: &Field, value: i64) {
            self.record_value(field, value.to_string());
        }

        fn record_u64(&mut self, field: &Field, value: u64) {
            self.record_value(field, value.to_string());
        }

        fn record_str(&mut self, field: &Field, value: &str) {
            self.record_value(field, value.to_owned());
        }

        fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
            self.record_value(field, format!("{value:?}"));
        }
    }
}
