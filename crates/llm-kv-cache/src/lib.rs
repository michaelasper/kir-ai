use std::fmt;

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

#[derive(Debug, Clone, PartialEq)]
pub struct LayerKvCache {
    max_tokens: usize,
    key_value_heads: usize,
    head_dim: usize,
    token_count: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
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
            max_tokens,
            key_value_heads,
            head_dim,
            token_count: 0,
            keys: vec![0.0; storage_len],
            values: vec![0.0; storage_len],
        })
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

    pub fn remaining_tokens(&self) -> usize {
        self.max_tokens - self.token_count
    }

    pub fn append(&mut self, key: &[f32], value: &[f32]) -> Result<usize, KvCacheError> {
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
        if self.token_count == self.max_tokens {
            return Err(KvCacheError::CapacityExceeded {
                requested: 1,
                available: 0,
            });
        }
        let token_index = self.token_count;
        let start = token_index * vector_len;
        let end = start + vector_len;
        self.keys[start..end].copy_from_slice(key);
        self.values[start..end].copy_from_slice(value);
        self.token_count += 1;
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

    pub fn clear(&mut self) {
        self.token_count = 0;
    }

    fn token_slice<'a>(&self, storage: &'a [f32], token_index: usize) -> Option<&'a [f32]> {
        if token_index >= self.token_count {
            return None;
        }
        let vector_len = self.vector_len();
        let start = token_index * vector_len;
        Some(&storage[start..start + vector_len])
    }

    fn used_len(&self) -> usize {
        self.token_count * self.vector_len()
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

        cache.clear();
        assert_eq!(cache.token_count(), 0);
        assert_eq!(cache.key(0), None);
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
}
