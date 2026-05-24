use crate::{KvCacheError, f32_resident_bytes, next_cache_id};

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

#[cfg(test)]
mod tests {
    use super::*;

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
