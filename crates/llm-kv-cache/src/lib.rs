mod block;
mod block_id;
mod block_pool;
mod block_table;

pub use block::{CacheBlock, CacheBlockHash};
pub use block_id::BlockId;
pub use block_pool::BlockPool;
pub use block_table::BlockTable;

use std::{
    fmt,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_CACHE_ID: AtomicU64 = AtomicU64::new(1);

fn next_cache_id() -> u64 {
    NEXT_CACHE_ID.fetch_add(1, Ordering::Relaxed)
}

fn f32_resident_bytes(values: &[f32]) -> u64 {
    (values.len() as u64).saturating_mul(std::mem::size_of::<f32>() as u64)
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

#[derive(Debug, PartialEq)]
pub struct LayerKvCache {
    id: u64,
    revision: u64,
    max_tokens: usize,
    key_value_heads: usize,
    head_dim: usize,
    token_count: usize,
    tokens_seen: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
}

/// Owned layer KV cache state, excluding runtime cache identity.
///
/// `keys` and `values` contain only the used token rows. Restoring a snapshot
/// allocates fresh backing storage and a fresh cache id.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerKvCacheSnapshot {
    pub revision: u64,
    pub max_tokens: usize,
    pub key_value_heads: usize,
    pub head_dim: usize,
    pub token_count: usize,
    pub tokens_seen: usize,
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
}

impl Clone for LayerKvCache {
    fn clone(&self) -> Self {
        Self {
            id: next_cache_id(),
            revision: self.revision,
            max_tokens: self.max_tokens,
            key_value_heads: self.key_value_heads,
            head_dim: self.head_dim,
            token_count: self.token_count,
            tokens_seen: self.tokens_seen,
            keys: self.keys.clone(),
            values: self.values.clone(),
        }
    }
}

impl LayerKvCache {
    pub fn new(
        max_tokens: usize,
        key_value_heads: usize,
        head_dim: usize,
    ) -> Result<Self, KvCacheError> {
        if max_tokens == 0 || key_value_heads == 0 || head_dim == 0 {
            return Err(KvCacheError::InvalidShape);
        }
        let vector_len = key_value_heads
            .checked_mul(head_dim)
            .ok_or(KvCacheError::InvalidShape)?;
        let storage_len = max_tokens
            .checked_mul(vector_len)
            .ok_or(KvCacheError::InvalidShape)?;
        Ok(Self {
            id: next_cache_id(),
            revision: 0,
            max_tokens,
            key_value_heads,
            head_dim,
            token_count: 0,
            tokens_seen: 0,
            keys: vec![0.0; storage_len],
            values: vec![0.0; storage_len],
        })
    }

    pub fn snapshot(&self) -> LayerKvCacheSnapshot {
        LayerKvCacheSnapshot {
            revision: self.revision,
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

        let mut cache = Self::new(
            snapshot.max_tokens,
            snapshot.key_value_heads,
            snapshot.head_dim,
        )?;
        let used_len = snapshot
            .token_count
            .checked_mul(cache.vector_len())
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

        cache.revision = snapshot.revision;
        cache.token_count = snapshot.token_count;
        cache.tokens_seen = snapshot.tokens_seen;
        cache.keys[..used_len].copy_from_slice(&snapshot.keys);
        cache.values[..used_len].copy_from_slice(&snapshot.values);
        Ok(cache)
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn revision(&self) -> u64 {
        self.revision
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

    /// Returns bytes retained by the allocated key/value backing storage.
    pub fn resident_bytes(&self) -> u64 {
        f32_resident_bytes(&self.keys).saturating_add(f32_resident_bytes(&self.values))
    }

    pub fn append(&mut self, key: &[f32], value: &[f32]) -> Result<usize, KvCacheError> {
        self.validate_token_shape(key, value)?;
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
        let vector_len = self.vector_len();
        let token_index = self.token_count;
        let start = token_index * vector_len;
        let end = start + vector_len;
        self.keys[start..end].copy_from_slice(key);
        self.values[start..end].copy_from_slice(value);
        self.token_count += 1;
        self.tokens_seen = tokens_seen;
        self.revision = self.revision.saturating_add(1);
        Ok(token_index)
    }

    pub fn append_sliding(&mut self, key: &[f32], value: &[f32]) -> Result<usize, KvCacheError> {
        self.validate_token_shape(key, value)?;
        if self.token_count < self.max_tokens {
            return self.append(key, value);
        }
        let tokens_seen = self
            .tokens_seen
            .checked_add(1)
            .ok_or(KvCacheError::InvalidShape)?;
        let vector_len = self.vector_len();
        let used_len = self.used_len();
        self.keys.copy_within(vector_len..used_len, 0);
        self.values.copy_within(vector_len..used_len, 0);
        let token_index = self.max_tokens - 1;
        let start = token_index * vector_len;
        let end = start + vector_len;
        self.keys[start..end].copy_from_slice(key);
        self.values[start..end].copy_from_slice(value);
        self.tokens_seen = tokens_seen;
        self.revision = self.revision.saturating_add(1);
        Ok(token_index)
    }

    pub fn key(&self, token_index: usize) -> Option<&[f32]> {
        self.token_slice(&self.keys, token_index)
    }

    pub fn value(&self, token_index: usize) -> Option<&[f32]> {
        self.token_slice(&self.values, token_index)
    }

    pub fn keys(&self) -> &[f32] {
        &self.keys[..self.used_len()]
    }

    pub fn values(&self) -> &[f32] {
        &self.values[..self.used_len()]
    }

    pub fn key_storage(&self) -> &[f32] {
        &self.keys
    }

    pub fn value_storage(&self) -> &[f32] {
        &self.values
    }

    pub fn clear(&mut self) {
        self.token_count = 0;
        self.tokens_seen = 0;
        self.revision = self.revision.saturating_add(1);
    }

    fn token_slice<'a>(&self, storage: &'a [f32], token_index: usize) -> Option<&'a [f32]> {
        if token_index >= self.token_count {
            return None;
        }
        let vector_len = self.vector_len();
        let start = token_index * vector_len;
        Some(&storage[start..start + vector_len])
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

    fn used_len(&self) -> usize {
        self.token_count * self.vector_len()
    }
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
    fn layer_kv_cache_resident_bytes_use_allocated_storage() {
        let mut cache = LayerKvCache::new(3, 2, 2).expect("cache shape is valid");

        assert_eq!(cache.resident_bytes(), 96);

        cache
            .append(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0])
            .expect("token fits");
        assert_eq!(cache.token_count(), 1);
        assert_eq!(cache.resident_bytes(), 96);
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
}
