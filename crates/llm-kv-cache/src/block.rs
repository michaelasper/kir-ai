use crate::{BlockId, KvCacheError};

pub type CacheBlockHash = [u8; 32];

#[derive(Debug, Clone, PartialEq)]
pub struct CacheBlock {
    id: BlockId,
    capacity_tokens: usize,
    vector_len: usize,
    token_count: usize,
    ref_count: usize,
    content_hash: Option<CacheBlockHash>,
    last_access: u64,
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl CacheBlock {
    pub fn new(capacity_tokens: usize, vector_len: usize) -> Result<Self, KvCacheError> {
        if capacity_tokens == 0 || vector_len == 0 {
            return Err(KvCacheError::InvalidShape);
        }
        let storage_len = capacity_tokens
            .checked_mul(vector_len)
            .ok_or(KvCacheError::InvalidShape)?;
        Ok(Self {
            id: BlockId::next()?,
            capacity_tokens,
            vector_len,
            token_count: 0,
            ref_count: 0,
            content_hash: None,
            last_access: 0,
            keys: vec![0.0; storage_len],
            values: vec![0.0; storage_len],
        })
    }

    pub fn id(&self) -> BlockId {
        self.id
    }

    pub fn capacity_tokens(&self) -> usize {
        self.capacity_tokens
    }

    pub fn vector_len(&self) -> usize {
        self.vector_len
    }

    pub fn token_count(&self) -> usize {
        self.token_count
    }

    pub fn remaining_tokens(&self) -> usize {
        self.capacity_tokens - self.token_count
    }

    pub fn is_full(&self) -> bool {
        self.token_count == self.capacity_tokens
    }

    pub fn ref_count(&self) -> usize {
        self.ref_count
    }

    pub fn increment_ref_count(&mut self) -> usize {
        self.ref_count = self.ref_count.saturating_add(1);
        self.ref_count
    }

    pub fn decrement_ref_count(&mut self) -> usize {
        self.ref_count = self.ref_count.saturating_sub(1);
        self.ref_count
    }

    pub fn content_hash(&self) -> Option<&CacheBlockHash> {
        self.content_hash.as_ref()
    }

    pub fn set_content_hash(&mut self, content_hash: Option<CacheBlockHash>) {
        self.content_hash = content_hash;
    }

    pub fn last_access(&self) -> u64 {
        self.last_access
    }

    pub fn touch(&mut self, last_access: u64) {
        self.last_access = last_access;
    }

    pub fn append(&mut self, key: &[f32], value: &[f32]) -> Result<usize, KvCacheError> {
        self.validate_token_shape(key, value)?;
        if self.is_full() {
            return Err(KvCacheError::CapacityExceeded {
                requested: 1,
                available: 0,
            });
        }
        let token_index = self.token_count;
        let start = token_index * self.vector_len;
        let end = start + self.vector_len;
        self.keys[start..end].copy_from_slice(key);
        self.values[start..end].copy_from_slice(value);
        self.token_count += 1;
        self.content_hash = None;
        Ok(token_index)
    }

    pub(crate) fn write_at(
        &mut self,
        token_index: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<(), KvCacheError> {
        self.validate_token_shape(key, value)?;
        if token_index >= self.token_count {
            return Err(KvCacheError::InvalidShape);
        }
        let start = token_index * self.vector_len;
        let end = start + self.vector_len;
        self.keys[start..end].copy_from_slice(key);
        self.values[start..end].copy_from_slice(value);
        self.content_hash = None;
        Ok(())
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
        self.ref_count = 0;
        self.content_hash = None;
        self.last_access = 0;
        self.keys.fill(0.0);
        self.values.fill(0.0);
    }

    pub(crate) fn reset_for_allocation(&mut self, last_access: u64) {
        self.clear();
        self.ref_count = 1;
        self.last_access = last_access;
    }

    pub(crate) fn copy_contents_from(
        &mut self,
        source: &Self,
        last_access: u64,
    ) -> Result<(), KvCacheError> {
        if self.capacity_tokens != source.capacity_tokens || self.vector_len != source.vector_len {
            return Err(KvCacheError::InvalidShape);
        }
        self.token_count = source.token_count;
        self.ref_count = 1;
        self.content_hash = source.content_hash;
        self.last_access = last_access;
        self.keys.copy_from_slice(&source.keys);
        self.values.copy_from_slice(&source.values);
        Ok(())
    }

    fn token_slice<'a>(&self, storage: &'a [f32], token_index: usize) -> Option<&'a [f32]> {
        if token_index >= self.token_count {
            return None;
        }
        let start = token_index * self.vector_len;
        Some(&storage[start..start + self.vector_len])
    }

    fn validate_token_shape(&self, key: &[f32], value: &[f32]) -> Result<(), KvCacheError> {
        if key.len() != self.vector_len {
            return Err(KvCacheError::ShapeMismatch {
                expected: self.vector_len,
                actual: key.len(),
            });
        }
        if value.len() != self.vector_len {
            return Err(KvCacheError::ShapeMismatch {
                expected: self.vector_len,
                actual: value.len(),
            });
        }
        Ok(())
    }

    fn used_len(&self) -> usize {
        self.token_count * self.vector_len
    }
}
