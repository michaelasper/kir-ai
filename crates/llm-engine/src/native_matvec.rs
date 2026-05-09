use crate::sync_ext::RecoverPoisonedMutex;
use llm_backend::{
    CpuNativeMatvecBackend, LayerKvCache, LinearAttentionCache, MathError, NativeMatvecBackend,
    QwenKvCacheTensor, QwenLayerCache, SafeTensorShardStore, TensorLoadError, TopKLogit,
    TopKWeight,
};
use serde_json::{Value, json};
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex, OnceLock},
};

pub(crate) const DEFAULT_NATIVE_TEXT_METAL_WEIGHT_CACHE_BYTES: u64 = 8 * 1024 * 1024 * 1024;

pub(crate) struct NativeTextMetalState {
    pub(crate) device: llm_metal::MetalDevice,
    bf16_matrices: Mutex<Bf16MatrixBufferCache<Arc<llm_metal::Bf16MatrixBuffer>>>,
    kv_caches: Mutex<HashMap<u64, MetalLayerKvCacheMirror>>,
    linear_caches: Mutex<HashMap<u64, MetalLinearAttentionCacheMirror>>,
}

#[derive(Clone)]
pub(crate) enum NativeTextMatvecBackend {
    Cpu,
    Metal(Arc<NativeTextMetalState>),
}

#[derive(Debug)]
struct MetalLayerKvCacheMirror {
    keys: llm_metal::F32Buffer,
    values: llm_metal::F32Buffer,
    revision: u64,
}

#[derive(Debug)]
struct MetalLinearAttentionCacheMirror {
    recurrent_state: llm_metal::F32Buffer,
    revision: u64,
}

type NativeTextMetalStateRegistry =
    Mutex<HashMap<NativeTextMetalStateKey, Arc<NativeTextMetalState>>>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct NativeTextMetalStateKey {
    cache_namespace: String,
    weight_cache_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct Bf16MatrixCacheKey {
    pub(crate) tensor: String,
    pub(crate) element_offset: usize,
    pub(crate) rows: usize,
    pub(crate) columns: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NativeTextWarmableBf16MatrixTensor {
    pub(crate) name: String,
    pub(crate) rows: usize,
    pub(crate) columns: usize,
    pub(crate) byte_len: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct NativeTextWeightWarmOrder {
    stage: u8,
    layer: usize,
    item: u8,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct NativeTextMetalWarmup {
    pub(crate) candidates: u64,
    pub(crate) warmed: u64,
    pub(crate) already_resident: u64,
    pub(crate) skipped_budget: u64,
    pub(crate) skipped_non_metal: u64,
}

#[derive(Debug)]
pub(crate) struct Bf16MatrixBufferCache<T> {
    max_bytes: u64,
    used_bytes: u64,
    next_access: u64,
    entries: HashMap<Bf16MatrixCacheKey, CachedBf16MatrixBuffer<T>>,
}

#[derive(Debug)]
struct CachedBf16MatrixBuffer<T> {
    value: T,
    byte_len: u64,
    last_used: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Bf16MatrixBufferCacheInsert {
    pub(crate) inserted: bool,
    pub(crate) evicted_count: u64,
    pub(crate) evicted_bytes: u64,
}

impl<T: Clone> Bf16MatrixBufferCache<T> {
    pub(crate) fn new(max_bytes: u64) -> Self {
        Self {
            max_bytes,
            used_bytes: 0,
            next_access: 0,
            entries: HashMap::new(),
        }
    }

    pub(crate) fn get(&mut self, key: &Bf16MatrixCacheKey) -> Option<T> {
        let access = self.next_access();
        self.entries.get_mut(key).map(|entry| {
            entry.last_used = access;
            entry.value.clone()
        })
    }

    pub(crate) fn insert(
        &mut self,
        key: Bf16MatrixCacheKey,
        value: T,
        byte_len: u64,
    ) -> Bf16MatrixBufferCacheInsert {
        if byte_len > self.max_bytes {
            return Bf16MatrixBufferCacheInsert::default();
        }
        if let Some(existing) = self.entries.remove(&key) {
            self.used_bytes = self.used_bytes.saturating_sub(existing.byte_len);
        }
        let mut result = Bf16MatrixBufferCacheInsert::default();
        while self.used_bytes.saturating_add(byte_len) > self.max_bytes {
            let Some(lru_key) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            let Some(evicted) = self.entries.remove(&lru_key) else {
                break;
            };
            self.used_bytes = self.used_bytes.saturating_sub(evicted.byte_len);
            result.evicted_count += 1;
            result.evicted_bytes += evicted.byte_len;
        }
        let access = self.next_access();
        self.entries.insert(
            key,
            CachedBf16MatrixBuffer {
                value,
                byte_len,
                last_used: access,
            },
        );
        self.used_bytes = self.used_bytes.saturating_add(byte_len);
        result.inserted = true;
        result
    }

    #[cfg(test)]
    pub(crate) fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    fn resident_bytes(&self) -> u64 {
        self.used_bytes
    }

    fn resident_buffers(&self) -> u64 {
        self.entries.len() as u64
    }

    fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    fn can_insert_without_eviction(&self, byte_len: u64) -> bool {
        byte_len <= self.max_bytes && self.used_bytes.saturating_add(byte_len) <= self.max_bytes
    }

    fn next_access(&mut self) -> u64 {
        let access = self.next_access;
        self.next_access = self.next_access.saturating_add(1);
        access
    }
}

#[derive(Debug)]
pub(crate) enum NativeTextMetalBufferError {
    Shape(String),
    Tensor(TensorLoadError),
    Metal(llm_metal::MetalError),
}

impl std::fmt::Display for NativeTextMetalBufferError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shape(message) => formatter.write_str(message),
            Self::Tensor(err) => write!(formatter, "{err}"),
            Self::Metal(err) => write!(formatter, "{err}"),
        }
    }
}

impl NativeTextMetalState {
    fn new(device: llm_metal::MetalDevice, weight_cache_bytes: u64) -> Self {
        native_text_metal_metrics().record_bf16_matrix_cache_residency(0, 0, weight_cache_bytes);
        Self {
            device,
            bf16_matrices: Mutex::new(Bf16MatrixBufferCache::new(weight_cache_bytes)),
            kv_caches: Mutex::new(HashMap::new()),
            linear_caches: Mutex::new(HashMap::new()),
        }
    }

    fn bf16_matrix_buffer(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        element_offset: usize,
        rows: usize,
        columns: usize,
    ) -> Result<Arc<llm_metal::Bf16MatrixBuffer>, NativeTextMetalBufferError> {
        let key = Bf16MatrixCacheKey {
            tensor: tensor.to_owned(),
            element_offset,
            rows,
            columns,
        };
        if let Some(buffer) = self
            .bf16_matrices
            .lock_or_recover("BF16 matrix buffer cache")
            .get(&key)
        {
            native_text_metal_metrics().record_bf16_matrix_cache_hit();
            return Ok(buffer);
        }
        native_text_metal_metrics().record_bf16_matrix_cache_miss();
        let element_count = rows.checked_mul(columns).ok_or_else(|| {
            NativeTextMetalBufferError::Shape("BF16 matrix element count overflow".to_owned())
        })?;
        let weights = store
            .bf16_tensor_bits_range(tensor, element_offset, element_count)
            .map_err(NativeTextMetalBufferError::Tensor)?;
        let buffer = Arc::new(
            self.device
                .new_bf16_matrix_buffer(&weights, rows, columns)
                .map_err(NativeTextMetalBufferError::Metal)?,
        );
        let mut matrices = self
            .bf16_matrices
            .lock_or_recover("BF16 matrix buffer cache");
        if let Some(existing) = matrices.get(&key) {
            native_text_metal_metrics().record_bf16_matrix_cache_hit();
            return Ok(existing);
        }
        let byte_len = buffer.byte_len() as u64;
        let insert = matrices.insert(key, Arc::clone(&buffer), byte_len);
        let metrics = native_text_metal_metrics();
        metrics.record_bf16_matrix_cache_upload(byte_len);
        if insert.evicted_count > 0 {
            metrics.record_bf16_matrix_cache_eviction(insert.evicted_count, insert.evicted_bytes);
        }
        metrics.record_bf16_matrix_cache_residency(
            matrices.resident_bytes(),
            matrices.resident_buffers(),
            matrices.max_bytes(),
        );
        Ok(buffer)
    }

    fn warm_bf16_matrix_cache(
        &self,
        store: &SafeTensorShardStore,
    ) -> Result<NativeTextMetalWarmup, NativeTextMetalBufferError> {
        let tensors = native_text_warmable_bf16_matrix_tensors(store)
            .map_err(NativeTextMetalBufferError::Tensor)?;
        let mut warmup = NativeTextMetalWarmup {
            candidates: tensors.len() as u64,
            ..NativeTextMetalWarmup::default()
        };
        for tensor in tensors {
            let key = Bf16MatrixCacheKey {
                tensor: tensor.name.clone(),
                element_offset: 0,
                rows: tensor.rows,
                columns: tensor.columns,
            };
            {
                let mut matrices = self
                    .bf16_matrices
                    .lock_or_recover("BF16 matrix buffer cache");
                if matrices.get(&key).is_some() {
                    warmup.already_resident += 1;
                    continue;
                }
                if !matrices.can_insert_without_eviction(tensor.byte_len) {
                    warmup.skipped_budget += 1;
                    continue;
                }
            }
            self.bf16_matrix_buffer(store, &tensor.name, 0, tensor.rows, tensor.columns)?;
            warmup.warmed += 1;
        }
        Ok(warmup)
    }

    fn sync_kv_cache(&self, cache: &LayerKvCache) -> Result<(), llm_metal::MetalError> {
        let byte_len =
            cache_resident_byte_len(cache.key_storage().len() + cache.value_storage().len())?;
        let mut caches = self.kv_caches.lock_or_recover("Metal KV cache mirror");
        match caches.get_mut(&cache.id()) {
            Some(mirror) if mirror.revision == cache.revision() => Ok(()),
            Some(mirror) => {
                self.device
                    .write_f32_buffer(&mirror.keys, cache.key_storage())?;
                self.device
                    .write_f32_buffer(&mirror.values, cache.value_storage())?;
                mirror.revision = cache.revision();
                native_text_metal_metrics().record_kv_cache_sync(byte_len);
                Ok(())
            }
            None => {
                let keys = self.device.new_f32_buffer(cache.key_storage())?;
                let values = self.device.new_f32_buffer(cache.value_storage())?;
                caches.insert(
                    cache.id(),
                    MetalLayerKvCacheMirror {
                        keys,
                        values,
                        revision: cache.revision(),
                    },
                );
                native_text_metal_metrics().record_kv_cache_allocation(byte_len);
                self.record_kv_cache_residency_locked(&caches);
                Ok(())
            }
        }
    }

    fn select_kv_cache_head_rows(
        &self,
        cache: &LayerKvCache,
        tensor: QwenKvCacheTensor,
        row_count: usize,
        head_start: usize,
        head_len: usize,
    ) -> Result<Vec<f32>, llm_metal::MetalError> {
        self.sync_kv_cache(cache)?;
        let caches = self.kv_caches.lock_or_recover("Metal KV cache mirror");
        let mirror = caches.get(&cache.id()).ok_or_else(|| {
            llm_metal::MetalError::InvalidShape(format!(
                "missing Metal KV cache mirror for cache {}",
                cache.id()
            ))
        })?;
        let values = match tensor {
            QwenKvCacheTensor::Key => &mirror.keys,
            QwenKvCacheTensor::Value => &mirror.values,
        };
        self.device.select_head_rows_f32_buffered(
            values,
            row_count,
            cache.vector_len(),
            head_start,
            head_len,
        )
    }

    fn sync_linear_cache(&self, cache: &LinearAttentionCache) -> Result<(), llm_metal::MetalError> {
        let byte_len = cache_resident_byte_len(cache.recurrent_state().len())?;
        let mut caches = self
            .linear_caches
            .lock_or_recover("Metal linear attention cache mirror");
        match caches.get_mut(&cache.id()) {
            Some(mirror) if mirror.revision == cache.revision() => Ok(()),
            Some(mirror) => {
                self.device
                    .write_f32_buffer(&mirror.recurrent_state, cache.recurrent_state())?;
                mirror.revision = cache.revision();
                native_text_metal_metrics().record_linear_cache_sync(byte_len);
                Ok(())
            }
            None => {
                let recurrent_state = self.device.new_f32_buffer(cache.recurrent_state())?;
                caches.insert(
                    cache.id(),
                    MetalLinearAttentionCacheMirror {
                        recurrent_state,
                        revision: cache.revision(),
                    },
                );
                native_text_metal_metrics().record_linear_cache_allocation(byte_len);
                self.record_linear_cache_residency_locked(&caches);
                Ok(())
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_attention_recurrent_cache_update(
        &self,
        cache: &LinearAttentionCache,
        state_start: usize,
        key: &[f32],
        value: &[f32],
        memory: &[f32],
        beta: f32,
        decay: f32,
        key_head_dim: usize,
        value_head_dim: usize,
    ) -> Result<Vec<f32>, llm_metal::MetalError> {
        self.sync_linear_cache(cache)?;
        let mut caches = self
            .linear_caches
            .lock_or_recover("Metal linear attention cache mirror");
        let mirror = caches.get_mut(&cache.id()).ok_or_else(|| {
            llm_metal::MetalError::InvalidShape(format!(
                "missing Metal linear attention cache mirror for cache {}",
                cache.id()
            ))
        })?;
        let updated = self
            .device
            .linear_attention_recurrent_update_f32_buffered_state(
                &mirror.recurrent_state,
                state_start,
                key,
                value,
                memory,
                beta,
                decay,
                key_head_dim,
                value_head_dim,
            )?;
        mirror.revision = cache.revision().saturating_add(1);
        Ok(updated)
    }

    pub(crate) fn remove_cache_mirrors(&self, caches: &[QwenLayerCache]) {
        let mut kv_removed = Vec::new();
        let mut linear_removed = Vec::new();
        for cache in caches {
            match cache {
                QwenLayerCache::Full(cache) => kv_removed.push(cache.id()),
                QwenLayerCache::Linear(cache) => linear_removed.push(cache.id()),
            }
        }
        if !kv_removed.is_empty() {
            let mut mirrors = self.kv_caches.lock_or_recover("Metal KV cache mirror");
            let mut bytes = 0_u64;
            let mut count = 0_u64;
            for id in kv_removed {
                if let Some(mirror) = mirrors.remove(&id) {
                    bytes = bytes
                        .saturating_add((mirror.keys.byte_len() + mirror.values.byte_len()) as u64);
                    count += 2;
                }
            }
            if count > 0 {
                native_text_metal_metrics().record_kv_cache_eviction(count, bytes);
                self.record_kv_cache_residency_locked(&mirrors);
            }
        }
        if !linear_removed.is_empty() {
            let mut mirrors = self
                .linear_caches
                .lock_or_recover("Metal linear attention cache mirror");
            let mut bytes = 0_u64;
            let mut count = 0_u64;
            for id in linear_removed {
                if let Some(mirror) = mirrors.remove(&id) {
                    bytes = bytes.saturating_add(mirror.recurrent_state.byte_len() as u64);
                    count += 1;
                }
            }
            if count > 0 {
                native_text_metal_metrics().record_linear_cache_eviction(count, bytes);
                self.record_linear_cache_residency_locked(&mirrors);
            }
        }
    }

    fn record_kv_cache_residency_locked(&self, caches: &HashMap<u64, MetalLayerKvCacheMirror>) {
        let resident_bytes = caches
            .values()
            .map(|mirror| mirror.keys.byte_len() as u64 + mirror.values.byte_len() as u64)
            .sum();
        native_text_metal_metrics()
            .record_kv_cache_residency(resident_bytes, caches.len() as u64 * 2);
    }

    fn record_linear_cache_residency_locked(
        &self,
        caches: &HashMap<u64, MetalLinearAttentionCacheMirror>,
    ) {
        let resident_bytes = caches
            .values()
            .map(|mirror| mirror.recurrent_state.byte_len() as u64)
            .sum();
        native_text_metal_metrics()
            .record_linear_cache_residency(resident_bytes, caches.len() as u64);
    }
}

fn cache_resident_byte_len(elements: usize) -> Result<u64, llm_metal::MetalError> {
    elements
        .checked_mul(std::mem::size_of::<f32>())
        .map(|bytes| bytes as u64)
        .ok_or_else(|| {
            llm_metal::MetalError::InvalidShape(
                "Metal resident cache byte length overflows usize".to_owned(),
            )
        })
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct MetalKernelCounters {
    attempts: u64,
    successes: u64,
    fallbacks: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct MetalBf16MatrixCacheCounters {
    hits: u64,
    misses: u64,
    uploads: u64,
    bytes_uploaded: u64,
    evictions: u64,
    bytes_evicted: u64,
    resident_bytes: u64,
    resident_buffers: u64,
    budget_bytes: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct MetalCacheCounters {
    allocations: u64,
    syncs: u64,
    evictions: u64,
    bytes_uploaded: u64,
    bytes_evicted: u64,
    resident_bytes: u64,
    resident_buffers: u64,
}

#[derive(Debug, Default)]
pub(crate) struct MetalBackendMetrics {
    counters: Mutex<HashMap<&'static str, MetalKernelCounters>>,
    bf16_matrix_cache: Mutex<MetalBf16MatrixCacheCounters>,
    kv_cache: Mutex<MetalCacheCounters>,
    linear_cache: Mutex<MetalCacheCounters>,
    warned_fallbacks: Mutex<HashSet<String>>,
}

impl MetalBackendMetrics {
    pub(crate) fn record_attempt(&self, kernel: &'static str) {
        self.update_counter(kernel, |counters| counters.attempts += 1);
    }

    pub(crate) fn record_success(&self, kernel: &'static str) {
        self.update_counter(kernel, |counters| counters.successes += 1);
    }

    pub(crate) fn record_fallback(
        &self,
        kernel: &'static str,
        bucket: impl Into<String>,
        error: impl std::fmt::Display,
    ) {
        self.update_counter(kernel, |counters| counters.fallbacks += 1);
        let bucket = bucket.into();
        let error = error.to_string();
        let warning_key = format!("{kernel}:{bucket}");
        let should_warn = self
            .warned_fallbacks
            .lock_or_recover("Metal fallback warning")
            .insert(warning_key);
        if should_warn {
            tracing::warn!(
                target: "native_text_metal",
                kernel,
                shape_bucket = %bucket,
                error = %error,
                "native text Metal kernel fell back to CPU"
            );
        } else {
            tracing::debug!(
                target: "native_text_metal",
                kernel,
                shape_bucket = %bucket,
                error = %error,
                "native text Metal kernel fell back to CPU"
            );
        }
    }

    pub(crate) fn record_bf16_matrix_cache_hit(&self) {
        let mut cache = self
            .bf16_matrix_cache
            .lock_or_recover("Metal BF16 matrix cache metrics");
        cache.hits += 1;
    }

    pub(crate) fn record_bf16_matrix_cache_miss(&self) {
        let mut cache = self
            .bf16_matrix_cache
            .lock_or_recover("Metal BF16 matrix cache metrics");
        cache.misses += 1;
    }

    pub(crate) fn record_bf16_matrix_cache_upload(&self, byte_len: u64) {
        let mut cache = self
            .bf16_matrix_cache
            .lock_or_recover("Metal BF16 matrix cache metrics");
        cache.uploads += 1;
        cache.bytes_uploaded += byte_len;
    }

    pub(crate) fn record_bf16_matrix_cache_eviction(&self, count: u64, byte_len: u64) {
        let mut cache = self
            .bf16_matrix_cache
            .lock_or_recover("Metal BF16 matrix cache metrics");
        cache.evictions += count;
        cache.bytes_evicted += byte_len;
    }

    pub(crate) fn record_bf16_matrix_cache_residency(
        &self,
        resident_bytes: u64,
        resident_buffers: u64,
        budget_bytes: u64,
    ) {
        let mut cache = self
            .bf16_matrix_cache
            .lock_or_recover("Metal BF16 matrix cache metrics");
        cache.resident_bytes = resident_bytes;
        cache.resident_buffers = resident_buffers;
        cache.budget_bytes = budget_bytes;
    }

    pub(crate) fn record_kv_cache_allocation(&self, byte_len: u64) {
        self.update_cache_counter(CacheMetricKind::Kv, |cache| {
            cache.allocations += 1;
            cache.bytes_uploaded += byte_len;
        });
    }

    pub(crate) fn record_kv_cache_sync(&self, byte_len: u64) {
        self.update_cache_counter(CacheMetricKind::Kv, |cache| {
            cache.syncs += 1;
            cache.bytes_uploaded += byte_len;
        });
    }

    pub(crate) fn record_kv_cache_eviction(&self, count: u64, byte_len: u64) {
        self.update_cache_counter(CacheMetricKind::Kv, |cache| {
            cache.evictions += count;
            cache.bytes_evicted += byte_len;
        });
    }

    pub(crate) fn record_kv_cache_residency(&self, resident_bytes: u64, resident_buffers: u64) {
        self.update_cache_counter(CacheMetricKind::Kv, |cache| {
            cache.resident_bytes = resident_bytes;
            cache.resident_buffers = resident_buffers;
        });
    }

    pub(crate) fn record_linear_cache_allocation(&self, byte_len: u64) {
        self.update_cache_counter(CacheMetricKind::Linear, |cache| {
            cache.allocations += 1;
            cache.bytes_uploaded += byte_len;
        });
    }

    pub(crate) fn record_linear_cache_sync(&self, byte_len: u64) {
        self.update_cache_counter(CacheMetricKind::Linear, |cache| {
            cache.syncs += 1;
            cache.bytes_uploaded += byte_len;
        });
    }

    pub(crate) fn record_linear_cache_eviction(&self, count: u64, byte_len: u64) {
        self.update_cache_counter(CacheMetricKind::Linear, |cache| {
            cache.evictions += count;
            cache.bytes_evicted += byte_len;
        });
    }

    pub(crate) fn record_linear_cache_residency(&self, resident_bytes: u64, resident_buffers: u64) {
        self.update_cache_counter(CacheMetricKind::Linear, |cache| {
            cache.resident_bytes = resident_bytes;
            cache.resident_buffers = resident_buffers;
        });
    }

    pub(crate) fn snapshot(&self) -> Value {
        let counters = self.counters.lock_or_recover("Metal metrics");
        let bf16_matrix_cache = *self
            .bf16_matrix_cache
            .lock_or_recover("Metal BF16 matrix cache metrics");
        let kv_cache = *self.kv_cache.lock_or_recover("Metal KV cache metrics");
        let linear_cache = *self
            .linear_cache
            .lock_or_recover("Metal linear cache metrics");
        let mut kernels = serde_json::Map::new();
        let mut kernel_names = counters.keys().copied().collect::<Vec<_>>();
        kernel_names.sort_unstable();
        for kernel in kernel_names {
            let counters = counters.get(kernel).copied().unwrap_or_default();
            kernels.insert(
                kernel.to_owned(),
                json!({
                    "attempts": counters.attempts,
                    "successes": counters.successes,
                    "fallbacks": counters.fallbacks,
                }),
            );
        }
        json!({
            "kernels": kernels,
            "bf16_matrix_cache": {
                "hits": bf16_matrix_cache.hits,
                "misses": bf16_matrix_cache.misses,
                "uploads": bf16_matrix_cache.uploads,
                "bytes_uploaded": bf16_matrix_cache.bytes_uploaded,
                "evictions": bf16_matrix_cache.evictions,
                "bytes_evicted": bf16_matrix_cache.bytes_evicted,
                "resident_bytes": bf16_matrix_cache.resident_bytes,
                "resident_buffers": bf16_matrix_cache.resident_buffers,
                "budget_bytes": bf16_matrix_cache.budget_bytes,
            },
            "kv_cache": cache_counters_json(kv_cache),
            "linear_attention_cache": cache_counters_json(linear_cache),
        })
    }

    fn update_cache_counter(
        &self,
        kind: CacheMetricKind,
        update: impl FnOnce(&mut MetalCacheCounters),
    ) {
        let cache = match kind {
            CacheMetricKind::Kv => &self.kv_cache,
            CacheMetricKind::Linear => &self.linear_cache,
        };
        let mut cache = cache.lock_or_recover("Metal resident cache metrics");
        update(&mut cache);
    }

    fn update_counter(&self, kernel: &'static str, update: impl FnOnce(&mut MetalKernelCounters)) {
        let mut counters = self.counters.lock_or_recover("Metal metrics");
        update(counters.entry(kernel).or_default());
    }
}

#[derive(Debug, Clone, Copy)]
enum CacheMetricKind {
    Kv,
    Linear,
}

fn cache_counters_json(counters: MetalCacheCounters) -> Value {
    json!({
        "allocations": counters.allocations,
        "syncs": counters.syncs,
        "evictions": counters.evictions,
        "bytes_uploaded": counters.bytes_uploaded,
        "bytes_evicted": counters.bytes_evicted,
        "resident_bytes": counters.resident_bytes,
        "resident_buffers": counters.resident_buffers,
    })
}

pub(crate) fn native_text_metal_metrics() -> &'static MetalBackendMetrics {
    static METRICS: OnceLock<MetalBackendMetrics> = OnceLock::new();
    METRICS.get_or_init(MetalBackendMetrics::default)
}

fn native_text_metal_state_registry() -> &'static NativeTextMetalStateRegistry {
    static REGISTRY: OnceLock<NativeTextMetalStateRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn native_text_shared_metal_state(
    weight_cache_bytes: u64,
    cache_namespace: &str,
) -> Result<Option<Arc<NativeTextMetalState>>, llm_metal::MetalError> {
    let key = NativeTextMetalStateKey {
        cache_namespace: cache_namespace.to_owned(),
        weight_cache_bytes,
    };
    let registry = native_text_metal_state_registry();
    if let Some(state) = registry
        .lock_or_recover("native text Metal state registry")
        .get(&key)
        .cloned()
    {
        return Ok(Some(state));
    }
    let Some(device) = llm_metal::MetalDevice::system_default_result()? else {
        return Ok(None);
    };
    let mut states = registry.lock_or_recover("native text Metal state registry");
    if let Some(state) = states.get(&key).cloned() {
        return Ok(Some(state));
    }
    let state = Arc::new(NativeTextMetalState::new(device, weight_cache_bytes));
    states.insert(key, Arc::clone(&state));
    Ok(Some(state))
}

impl NativeTextMatvecBackend {
    pub(crate) fn system_default(weight_cache_bytes: u64, cache_namespace: &str) -> Self {
        match native_text_shared_metal_state(weight_cache_bytes, cache_namespace) {
            Ok(Some(state)) => Self::Metal(state),
            Ok(None) => Self::Cpu,
            Err(err) => {
                tracing::warn!("native text Metal matvec backend unavailable: {err}");
                Self::Cpu
            }
        }
    }

    fn cpu() -> CpuNativeMatvecBackend {
        CpuNativeMatvecBackend
    }

    pub(crate) fn metal_state(&self) -> Option<Arc<NativeTextMetalState>> {
        match self {
            Self::Cpu => None,
            Self::Metal(state) => Some(Arc::clone(state)),
        }
    }

    pub(crate) fn warm_bf16_matrix_cache(
        &self,
        store: &SafeTensorShardStore,
    ) -> Result<NativeTextMetalWarmup, NativeTextMetalBufferError> {
        let candidates = native_text_warmable_bf16_matrix_tensors(store)
            .map_err(NativeTextMetalBufferError::Tensor)?
            .len() as u64;
        match self {
            Self::Cpu => Ok(NativeTextMetalWarmup {
                candidates,
                skipped_non_metal: candidates,
                ..NativeTextMetalWarmup::default()
            }),
            Self::Metal(metal) => metal.warm_bf16_matrix_cache(store),
        }
    }

    fn bf16_matrix_shape(
        store: &SafeTensorShardStore,
        tensor: &str,
        input: &[f32],
    ) -> Option<(usize, usize)> {
        let metadata = store.tensor_metadata(tensor).ok()?;
        if metadata.dtype != "BF16" || metadata.shape.len() != 2 {
            return None;
        }
        let rows = metadata.shape[0];
        let columns = metadata.shape[1];
        (input.len() == columns).then_some((rows, columns))
    }

    fn flattened_inputs(inputs: &[Vec<f32>], columns: usize) -> Option<Vec<f32>> {
        let mut flattened = Vec::with_capacity(inputs.len().checked_mul(columns)?);
        for input in inputs {
            if input.len() != columns {
                return None;
            }
            flattened.extend_from_slice(input);
        }
        Some(flattened)
    }

    fn record_metal_fallback(
        kernel: &'static str,
        bucket: impl Into<String>,
        error: impl std::fmt::Display,
    ) {
        native_text_metal_metrics().record_fallback(kernel, bucket, error);
    }

    fn run_metal_math<T>(
        kernel: &'static str,
        bucket: impl Into<String>,
        metal: impl FnOnce() -> Result<T, llm_metal::MetalError>,
        cpu: impl FnOnce() -> Result<T, MathError>,
    ) -> Result<T, MathError> {
        let metrics = native_text_metal_metrics();
        metrics.record_attempt(kernel);
        match metal() {
            Ok(value) => {
                metrics.record_success(kernel);
                Ok(value)
            }
            Err(err) => {
                metrics.record_fallback(kernel, bucket, err);
                cpu()
            }
        }
    }

    fn run_metal_tensor<T>(
        kernel: &'static str,
        bucket: impl Into<String>,
        metal: impl FnOnce() -> Result<T, llm_metal::MetalError>,
        cpu: impl FnOnce() -> Result<T, TensorLoadError>,
    ) -> Result<T, TensorLoadError> {
        let metrics = native_text_metal_metrics();
        metrics.record_attempt(kernel);
        match metal() {
            Ok(value) => {
                metrics.record_success(kernel);
                Ok(value)
            }
            Err(err) => {
                metrics.record_fallback(kernel, bucket, err);
                cpu()
            }
        }
    }
}

impl NativeMatvecBackend for NativeTextMatvecBackend {
    fn bf16_matvec_row_major_f32(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        input: &[f32],
    ) -> Result<Vec<f32>, TensorLoadError> {
        let Self::Metal(state) = self else {
            return Self::cpu().bf16_matvec_row_major_f32(store, tensor, input);
        };
        let Some((rows, columns)) = Self::bf16_matrix_shape(store, tensor, input) else {
            Self::record_metal_fallback(
                "matvec_bf16_f32",
                format!("tensor={tensor},input_len={}", input.len()),
                "unsupported BF16 matrix shape or input length",
            );
            return Self::cpu().bf16_matvec_row_major_f32(store, tensor, input);
        };
        let matrix = match state.bf16_matrix_buffer(store, tensor, 0, rows, columns) {
            Ok(matrix) => matrix,
            Err(err) => {
                Self::record_metal_fallback(
                    "matvec_bf16_f32",
                    format!("tensor={tensor},rows={rows},cols={columns}"),
                    err,
                );
                return Self::cpu().bf16_matvec_row_major_f32(store, tensor, input);
            }
        };
        Self::run_metal_tensor(
            "matvec_bf16_f32",
            format!("tensor={tensor},rows={rows},cols={columns}"),
            || state.device.matvec_bf16_f32_buffered(&matrix, input),
            || Self::cpu().bf16_matvec_row_major_f32(store, tensor, input),
        )
    }

    fn bf16_matvecs_row_major_f32(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        inputs: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, TensorLoadError> {
        let Self::Metal(state) = self else {
            return Self::cpu().bf16_matvecs_row_major_f32(store, tensor, inputs);
        };
        let Some(first_input) = inputs.first() else {
            return Ok(Vec::new());
        };
        let Some((rows, columns)) = Self::bf16_matrix_shape(store, tensor, first_input) else {
            Self::record_metal_fallback(
                "batched_matvec_bf16_f32",
                format!(
                    "tensor={tensor},inputs={},first_input_len={}",
                    inputs.len(),
                    first_input.len()
                ),
                "unsupported BF16 matrix shape or input length",
            );
            return Self::cpu().bf16_matvecs_row_major_f32(store, tensor, inputs);
        };
        let Some(flattened) = Self::flattened_inputs(inputs, columns) else {
            Self::record_metal_fallback(
                "batched_matvec_bf16_f32",
                format!("tensor={tensor},inputs={},cols={columns}", inputs.len()),
                "batched input width mismatch",
            );
            return Self::cpu().bf16_matvecs_row_major_f32(store, tensor, inputs);
        };
        let matrix = match state.bf16_matrix_buffer(store, tensor, 0, rows, columns) {
            Ok(matrix) => matrix,
            Err(err) => {
                Self::record_metal_fallback(
                    "batched_matvec_bf16_f32",
                    format!("tensor={tensor},rows={rows},cols={columns}"),
                    err,
                );
                return Self::cpu().bf16_matvecs_row_major_f32(store, tensor, inputs);
            }
        };
        Self::run_metal_tensor(
            "batched_matvec_bf16_f32",
            format!(
                "tensor={tensor},rows={rows},cols={columns},inputs={}",
                inputs.len()
            ),
            || {
                state
                    .device
                    .batched_matvec_bf16_f32_buffered(&matrix, &flattened, inputs.len())
                    .map(|values| {
                        values
                            .chunks_exact(rows)
                            .map(|chunk| chunk.to_vec())
                            .collect()
                    })
            },
            || Self::cpu().bf16_matvecs_row_major_f32(store, tensor, inputs),
        )
    }

    fn bf16_matvec_rows_f32(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        input: &[f32],
        chunk_rows: usize,
    ) -> Result<Vec<f32>, TensorLoadError> {
        let Self::Metal(state) = self else {
            return Self::cpu().bf16_matvec_rows_f32(store, tensor, input, chunk_rows);
        };
        if chunk_rows == 0 {
            Self::record_metal_fallback(
                "matvec_bf16_f32",
                format!("tensor={tensor},input_len={},chunk_rows=0", input.len()),
                "zero chunk rows",
            );
            return Self::cpu().bf16_matvec_rows_f32(store, tensor, input, chunk_rows);
        }
        let Some((rows, columns)) = Self::bf16_matrix_shape(store, tensor, input) else {
            Self::record_metal_fallback(
                "matvec_bf16_f32",
                format!(
                    "tensor={tensor},input_len={},chunk_rows={chunk_rows}",
                    input.len()
                ),
                "unsupported BF16 matrix shape or input length",
            );
            return Self::cpu().bf16_matvec_rows_f32(store, tensor, input, chunk_rows);
        };
        let mut output = Vec::with_capacity(rows);
        for row_start in (0..rows).step_by(chunk_rows) {
            let rows_in_chunk = chunk_rows.min(rows - row_start);
            let Some(element_offset) = row_start.checked_mul(columns) else {
                Self::record_metal_fallback(
                    "matvec_bf16_f32",
                    format!("tensor={tensor},row_start={row_start},rows={rows},cols={columns}"),
                    "BF16 row offset overflow",
                );
                return Self::cpu().bf16_matvec_rows_f32(store, tensor, input, chunk_rows);
            };
            let matrix = match state.bf16_matrix_buffer(
                store,
                tensor,
                element_offset,
                rows_in_chunk,
                columns,
            ) {
                Ok(matrix) => matrix,
                Err(err) => {
                    Self::record_metal_fallback(
                        "matvec_bf16_f32",
                        format!(
                            "tensor={tensor},row_start={row_start},rows_in_chunk={rows_in_chunk},cols={columns}"
                        ),
                        err,
                    );
                    return Self::cpu().bf16_matvec_rows_f32(store, tensor, input, chunk_rows);
                }
            };
            let metrics = native_text_metal_metrics();
            metrics.record_attempt("matvec_bf16_f32");
            let logits = match state.device.matvec_bf16_f32_buffered(&matrix, input) {
                Ok(logits) => {
                    metrics.record_success("matvec_bf16_f32");
                    logits
                }
                Err(err) => {
                    metrics.record_fallback(
                        "matvec_bf16_f32",
                        format!(
                            "tensor={tensor},row_start={row_start},rows_in_chunk={rows_in_chunk},cols={columns}"
                        ),
                        err,
                    );
                    return Self::cpu().bf16_matvec_rows_f32(store, tensor, input, chunk_rows);
                }
            };
            output.extend(logits);
        }
        Ok(output)
    }

    fn bf16_matvec_range_row_major_f32(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        element_offset: usize,
        rows: usize,
        columns: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, TensorLoadError> {
        let Self::Metal(state) = self else {
            return Self::cpu().bf16_matvec_range_row_major_f32(
                store,
                tensor,
                element_offset,
                rows,
                columns,
                input,
            );
        };
        if input.len() != columns {
            Self::record_metal_fallback(
                "matvec_bf16_f32",
                format!(
                    "tensor={tensor},offset={element_offset},rows={rows},cols={columns},input_len={}",
                    input.len()
                ),
                "BF16 range input width mismatch",
            );
            return Self::cpu().bf16_matvec_range_row_major_f32(
                store,
                tensor,
                element_offset,
                rows,
                columns,
                input,
            );
        }
        let matrix = match state.bf16_matrix_buffer(store, tensor, element_offset, rows, columns) {
            Ok(matrix) => matrix,
            Err(err) => {
                Self::record_metal_fallback(
                    "matvec_bf16_f32",
                    format!("tensor={tensor},offset={element_offset},rows={rows},cols={columns}"),
                    err,
                );
                return Self::cpu().bf16_matvec_range_row_major_f32(
                    store,
                    tensor,
                    element_offset,
                    rows,
                    columns,
                    input,
                );
            }
        };
        Self::run_metal_tensor(
            "matvec_bf16_f32",
            format!("tensor={tensor},offset={element_offset},rows={rows},cols={columns}"),
            || state.device.matvec_bf16_f32_buffered(&matrix, input),
            || {
                Self::cpu().bf16_matvec_range_row_major_f32(
                    store,
                    tensor,
                    element_offset,
                    rows,
                    columns,
                    input,
                )
            },
        )
    }

    fn bf16_matvec_top_k_rows_f32(
        &self,
        store: &SafeTensorShardStore,
        tensor: &str,
        input: &[f32],
        top_k: usize,
        chunk_rows: usize,
    ) -> Result<Vec<TopKLogit>, TensorLoadError> {
        let Self::Metal(state) = self else {
            return Self::cpu().bf16_matvec_top_k_rows_f32(store, tensor, input, top_k, chunk_rows);
        };
        if chunk_rows == 0 {
            Self::record_metal_fallback(
                "matvec_bf16_f32",
                format!(
                    "tensor={tensor},input_len={},top_k={top_k},chunk_rows=0",
                    input.len()
                ),
                "zero chunk rows",
            );
            return Self::cpu().bf16_matvec_top_k_rows_f32(store, tensor, input, top_k, chunk_rows);
        }
        let Some((rows, columns)) = Self::bf16_matrix_shape(store, tensor, input) else {
            Self::record_metal_fallback(
                "matvec_bf16_f32",
                format!(
                    "tensor={tensor},input_len={},top_k={top_k},chunk_rows={chunk_rows}",
                    input.len()
                ),
                "unsupported BF16 matrix shape or input length",
            );
            return Self::cpu().bf16_matvec_top_k_rows_f32(store, tensor, input, top_k, chunk_rows);
        };
        if top_k == 0 || top_k > rows {
            Self::record_metal_fallback(
                "top_k_f32",
                format!("tensor={tensor},rows={rows},top_k={top_k}"),
                "unsupported top-k request",
            );
            return Self::cpu().bf16_matvec_top_k_rows_f32(store, tensor, input, top_k, chunk_rows);
        }
        let mut top = Vec::new();
        for row_start in (0..rows).step_by(chunk_rows) {
            let rows_in_chunk = chunk_rows.min(rows - row_start);
            let Some(element_offset) = row_start.checked_mul(columns) else {
                Self::record_metal_fallback(
                    "matvec_bf16_f32",
                    format!("tensor={tensor},row_start={row_start},rows={rows},cols={columns}"),
                    "BF16 row offset overflow",
                );
                return Self::cpu()
                    .bf16_matvec_top_k_rows_f32(store, tensor, input, top_k, chunk_rows);
            };
            let matrix = match state.bf16_matrix_buffer(
                store,
                tensor,
                element_offset,
                rows_in_chunk,
                columns,
            ) {
                Ok(matrix) => matrix,
                Err(err) => {
                    Self::record_metal_fallback(
                        "matvec_bf16_f32",
                        format!(
                            "tensor={tensor},row_start={row_start},rows_in_chunk={rows_in_chunk},cols={columns}"
                        ),
                        err,
                    );
                    return Self::cpu()
                        .bf16_matvec_top_k_rows_f32(store, tensor, input, top_k, chunk_rows);
                }
            };
            let metrics = native_text_metal_metrics();
            metrics.record_attempt("matvec_bf16_f32");
            let logits = match state.device.matvec_bf16_f32_buffered(&matrix, input) {
                Ok(logits) => {
                    metrics.record_success("matvec_bf16_f32");
                    logits
                }
                Err(err) => {
                    metrics.record_fallback(
                        "matvec_bf16_f32",
                        format!(
                            "tensor={tensor},row_start={row_start},rows_in_chunk={rows_in_chunk},cols={columns}"
                        ),
                        err,
                    );
                    return Self::cpu()
                        .bf16_matvec_top_k_rows_f32(store, tensor, input, top_k, chunk_rows);
                }
            };
            metrics.record_attempt("top_k_f32");
            let chunk_top = match state.device.top_k_f32(&logits, top_k.min(rows_in_chunk)) {
                Ok(chunk_top) => {
                    metrics.record_success("top_k_f32");
                    chunk_top
                }
                Err(err) => {
                    metrics.record_fallback(
                        "top_k_f32",
                        format!(
                            "tensor={tensor},row_start={row_start},rows_in_chunk={rows_in_chunk},top_k={top_k}"
                        ),
                        err,
                    );
                    return Self::cpu()
                        .bf16_matvec_top_k_rows_f32(store, tensor, input, top_k, chunk_rows);
                }
            };
            top.extend(chunk_top.into_iter().map(|item| TopKLogit {
                index: row_start + item.index,
                logit: item.value,
            }));
        }
        top.sort_by(|left, right| {
            right
                .logit
                .total_cmp(&left.logit)
                .then_with(|| left.index.cmp(&right.index))
        });
        top.truncate(top_k);
        Ok(top)
    }

    fn matvec_row_major_f32(
        &self,
        input: &[f32],
        weights: &[f32],
        rows: usize,
        columns: usize,
    ) -> Result<Vec<f32>, MathError> {
        match self {
            Self::Cpu => Self::cpu().matvec_row_major_f32(input, weights, rows, columns),
            Self::Metal(metal) => Self::run_metal_math(
                "matvec_f32",
                format!("rows={rows},cols={columns},input_len={}", input.len()),
                || metal.device.matvec_f32(weights, rows, columns, input),
                || Self::cpu().matvec_row_major_f32(input, weights, rows, columns),
            ),
        }
    }

    fn qwen_rms_norm_f32(
        &self,
        input: &[f32],
        weight: &[f32],
        eps: f32,
    ) -> Result<Vec<f32>, MathError> {
        match self {
            Self::Cpu => Self::cpu().qwen_rms_norm_f32(input, weight, eps),
            Self::Metal(metal) => Self::run_metal_math(
                "qwen_rms_norm",
                format!("len={},weight_len={}", input.len(), weight.len()),
                || metal.device.qwen_rms_norm_f32(input, weight, eps),
                || Self::cpu().qwen_rms_norm_f32(input, weight, eps),
            ),
        }
    }

    fn softmax_f32(&self, scores: &[f32]) -> Result<Vec<f32>, MathError> {
        match self {
            Self::Cpu => Self::cpu().softmax_f32(scores),
            Self::Metal(metal) => Self::run_metal_math(
                "softmax_f32",
                format!("len={}", scores.len()),
                || metal.device.softmax_f32(scores),
                || Self::cpu().softmax_f32(scores),
            ),
        }
    }

    fn linear_attention_conv1d_silu_f32(
        &self,
        window: &[f32],
        weights: &[f32],
        conv_dim: usize,
        kernel_size: usize,
    ) -> Result<Vec<f32>, MathError> {
        match self {
            Self::Cpu => {
                Self::cpu().linear_attention_conv1d_silu_f32(window, weights, conv_dim, kernel_size)
            }
            Self::Metal(metal) => Self::run_metal_math(
                "linear_attention_conv1d_silu_f32",
                format!(
                    "window_len={},weight_len={},conv_dim={conv_dim},kernel_size={kernel_size}",
                    window.len(),
                    weights.len()
                ),
                || {
                    metal.device.linear_attention_conv1d_silu_f32(
                        window,
                        weights,
                        conv_dim,
                        kernel_size,
                    )
                },
                || {
                    Self::cpu().linear_attention_conv1d_silu_f32(
                        window,
                        weights,
                        conv_dim,
                        kernel_size,
                    )
                },
            ),
        }
    }

    fn softmax_top_k_f32(
        &self,
        logits: &[f32],
        top_k: usize,
    ) -> Result<Vec<TopKWeight>, MathError> {
        match self {
            Self::Cpu => Self::cpu().softmax_top_k_f32(logits, top_k),
            Self::Metal(metal) => {
                if top_k == 0
                    || top_k > logits.len()
                    || logits.iter().any(|value| !value.is_finite())
                {
                    Self::record_metal_fallback(
                        "top_k_f32",
                        format!("logits_len={},top_k={top_k}", logits.len()),
                        "unsupported top-k softmax request",
                    );
                    return Self::cpu().softmax_top_k_f32(logits, top_k);
                }
                let metrics = native_text_metal_metrics();
                metrics.record_attempt("top_k_f32");
                let top = match metal.device.top_k_f32(logits, top_k) {
                    Ok(top) => top,
                    Err(err) => {
                        metrics.record_fallback(
                            "top_k_f32",
                            format!("logits_len={},top_k={top_k}", logits.len()),
                            err,
                        );
                        return Self::cpu().softmax_top_k_f32(logits, top_k);
                    }
                };
                match softmax_metal_top_k(top) {
                    Ok(weights) => {
                        metrics.record_success("top_k_f32");
                        Ok(weights)
                    }
                    Err(()) => {
                        metrics.record_fallback(
                            "top_k_f32",
                            format!("logits_len={},top_k={top_k}", logits.len()),
                            "Metal top-k softmax normalization failed",
                        );
                        Self::cpu().softmax_top_k_f32(logits, top_k)
                    }
                }
            }
        }
    }

    fn weighted_sum_f32(
        &self,
        values: &[f32],
        weights: &[f32],
        vector_len: usize,
    ) -> Result<Vec<f32>, MathError> {
        match self {
            Self::Cpu => Self::cpu().weighted_sum_f32(values, weights, vector_len),
            Self::Metal(metal) => Self::run_metal_math(
                "weighted_sum_f32",
                format!(
                    "values_len={},weights_len={},vector_len={vector_len}",
                    values.len(),
                    weights.len()
                ),
                || metal.device.weighted_sum_f32(values, weights, vector_len),
                || Self::cpu().weighted_sum_f32(values, weights, vector_len),
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_attention_recurrent_update_f32(
        &self,
        state: &[f32],
        key: &[f32],
        value: &[f32],
        memory: &[f32],
        beta: f32,
        decay: f32,
        key_head_dim: usize,
        value_head_dim: usize,
    ) -> Result<Vec<f32>, MathError> {
        match self {
            Self::Cpu => Self::cpu().linear_attention_recurrent_update_f32(
                state,
                key,
                value,
                memory,
                beta,
                decay,
                key_head_dim,
                value_head_dim,
            ),
            Self::Metal(metal) => Self::run_metal_math(
                "linear_attention_recurrent_update_f32",
                format!(
                    "state_len={},key_len={},value_len={},memory_len={},key_head_dim={key_head_dim},value_head_dim={value_head_dim}",
                    state.len(),
                    key.len(),
                    value.len(),
                    memory.len()
                ),
                || {
                    metal.device.linear_attention_recurrent_update_f32(
                        state,
                        key,
                        value,
                        memory,
                        beta,
                        decay,
                        key_head_dim,
                        value_head_dim,
                    )
                },
                || {
                    Self::cpu().linear_attention_recurrent_update_f32(
                        state,
                        key,
                        value,
                        memory,
                        beta,
                        decay,
                        key_head_dim,
                        value_head_dim,
                    )
                },
            ),
        }
    }
    fn select_head_rows_f32(
        &self,
        values: &[f32],
        row_count: usize,
        row_len: usize,
        head_start: usize,
        head_len: usize,
    ) -> Result<Vec<f32>, MathError> {
        match self {
            Self::Cpu => {
                Self::cpu().select_head_rows_f32(values, row_count, row_len, head_start, head_len)
            }
            Self::Metal(metal) => Self::run_metal_math(
                "select_head_rows_f32",
                format!(
                    "values_len={},row_count={row_count},row_len={row_len},head_start={head_start},head_len={head_len}",
                    values.len()
                ),
                || {
                    metal
                        .device
                        .select_head_rows_f32(values, row_count, row_len, head_start, head_len)
                },
                || {
                    Self::cpu()
                        .select_head_rows_f32(values, row_count, row_len, head_start, head_len)
                },
            ),
        }
    }

    fn select_kv_cache_head_rows_f32(
        &self,
        cache: &LayerKvCache,
        tensor: QwenKvCacheTensor,
        row_count: usize,
        head_start: usize,
        head_len: usize,
    ) -> Result<Vec<f32>, MathError> {
        match self {
            Self::Cpu => Self::cpu()
                .select_kv_cache_head_rows_f32(cache, tensor, row_count, head_start, head_len),
            Self::Metal(metal) => Self::run_metal_math(
                "select_head_rows_f32",
                format!(
                    "cache_id={},tensor={tensor:?},row_count={row_count},row_len={},head_start={head_start},head_len={head_len}",
                    cache.id(),
                    cache.vector_len()
                ),
                || metal.select_kv_cache_head_rows(cache, tensor, row_count, head_start, head_len),
                || {
                    Self::cpu().select_kv_cache_head_rows_f32(
                        cache, tensor, row_count, head_start, head_len,
                    )
                },
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_attention_recurrent_cache_update_f32(
        &self,
        cache: &LinearAttentionCache,
        state_start: usize,
        key: &[f32],
        value: &[f32],
        memory: &[f32],
        beta: f32,
        decay: f32,
        key_head_dim: usize,
        value_head_dim: usize,
    ) -> Result<Vec<f32>, MathError> {
        match self {
            Self::Cpu => Self::cpu().linear_attention_recurrent_cache_update_f32(
                cache,
                state_start,
                key,
                value,
                memory,
                beta,
                decay,
                key_head_dim,
                value_head_dim,
            ),
            Self::Metal(metal) => Self::run_metal_math(
                "linear_attention_recurrent_update_state_f32",
                format!(
                    "cache_id={},state_start={state_start},key_head_dim={key_head_dim},value_head_dim={value_head_dim}",
                    cache.id()
                ),
                || {
                    metal.linear_attention_recurrent_cache_update(
                        cache,
                        state_start,
                        key,
                        value,
                        memory,
                        beta,
                        decay,
                        key_head_dim,
                        value_head_dim,
                    )
                },
                || {
                    Self::cpu().linear_attention_recurrent_cache_update_f32(
                        cache,
                        state_start,
                        key,
                        value,
                        memory,
                        beta,
                        decay,
                        key_head_dim,
                        value_head_dim,
                    )
                },
            ),
        }
    }
}

fn softmax_metal_top_k(top: Vec<llm_metal::TopKResult>) -> Result<Vec<TopKWeight>, ()> {
    let max = top
        .iter()
        .map(|item| item.value)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut exp_values = top
        .iter()
        .map(|item| (item.value - max).exp())
        .collect::<Vec<_>>();
    let sum = exp_values.iter().sum::<f32>();
    if sum == 0.0 || !sum.is_finite() {
        return Err(());
    }
    Ok(top
        .iter()
        .zip(exp_values.iter_mut())
        .map(|(item, value)| TopKWeight {
            index: item.index,
            weight: *value / sum,
        })
        .collect())
}
pub(crate) fn native_text_metal_weight_cache_bytes(configured: Option<u64>) -> u64 {
    configured.unwrap_or(DEFAULT_NATIVE_TEXT_METAL_WEIGHT_CACHE_BYTES)
}

pub(crate) fn native_text_warmable_bf16_matrix_tensors(
    store: &SafeTensorShardStore,
) -> Result<Vec<NativeTextWarmableBf16MatrixTensor>, TensorLoadError> {
    let mut tensors = Vec::new();
    for name in store.tensor_names() {
        let metadata = store.tensor_metadata(name)?;
        if metadata.dtype == "BF16" && metadata.shape.len() == 2 {
            tensors.push(NativeTextWarmableBf16MatrixTensor {
                name: name.to_owned(),
                rows: metadata.shape[0],
                columns: metadata.shape[1],
                byte_len: metadata.byte_len as u64,
            });
        }
    }
    tensors.sort_by(|left, right| {
        native_text_bf16_matrix_warm_order(&left.name)
            .cmp(&native_text_bf16_matrix_warm_order(&right.name))
            .then_with(|| left.name.cmp(&right.name))
    });
    Ok(tensors)
}

fn native_text_bf16_matrix_warm_order(name: &str) -> NativeTextWeightWarmOrder {
    let root = name
        .strip_prefix("model.language_model.")
        .or_else(|| name.strip_prefix("model."));
    if matches!(
        root,
        Some("embed_tokens.weight" | "embed_tokens_per_layer.weight")
    ) {
        return NativeTextWeightWarmOrder {
            stage: 0,
            layer: 0,
            item: 0,
        };
    }
    if matches!(root, Some("norm.weight")) || name == "lm_head.weight" {
        return NativeTextWeightWarmOrder {
            stage: 3,
            layer: 0,
            item: 0,
        };
    }
    if matches!(
        root,
        Some("per_layer_model_projection.weight" | "per_layer_projection_norm.weight")
    ) {
        return NativeTextWeightWarmOrder {
            stage: 0,
            layer: 0,
            item: 1,
        };
    }
    let Some(layer_suffix) = root.and_then(|root| root.strip_prefix("layers.")) else {
        return native_text_unknown_weight_warm_order();
    };
    let Some((layer, suffix)) = layer_suffix.split_once('.') else {
        return native_text_unknown_weight_warm_order();
    };
    let Ok(layer) = layer.parse::<usize>() else {
        return native_text_unknown_weight_warm_order();
    };
    let Some((stage, item)) = native_text_layer_bf16_matrix_warm_order(suffix) else {
        return native_text_unknown_weight_warm_order();
    };
    NativeTextWeightWarmOrder { stage, layer, item }
}

fn native_text_layer_bf16_matrix_warm_order(suffix: &str) -> Option<(u8, u8)> {
    let item = match suffix {
        "self_attn.q_proj.weight" | "linear_attn.in_proj_qkv.weight" => 0,
        "self_attn.k_proj.weight" | "linear_attn.in_proj_z.weight" => 1,
        "self_attn.v_proj.weight" | "linear_attn.in_proj_b.weight" => 2,
        "self_attn.o_proj.weight" | "linear_attn.in_proj_a.weight" => 3,
        "linear_attn.out_proj.weight" => 4,
        "input_layernorm.weight" => 5,
        "post_attention_layernorm.weight" => 6,
        "pre_feedforward_layernorm.weight" => 7,
        "post_feedforward_layernorm.weight" => 8,
        "mlp.gate.weight" => 10,
        "mlp.gate_proj.weight" => 10,
        "mlp.up_proj.weight" => 11,
        "mlp.down_proj.weight" => 12,
        "mlp.shared_expert.gate_proj.weight" => 11,
        "mlp.shared_expert.up_proj.weight" => 12,
        "mlp.shared_expert.down_proj.weight" => 13,
        "mlp.shared_expert_gate.weight" => 14,
        "input_gate.weight" => 20,
        "post_per_layer_input_norm.weight" => 21,
        _ => return None,
    };
    Some((1, item))
}

fn native_text_unknown_weight_warm_order() -> NativeTextWeightWarmOrder {
    NativeTextWeightWarmOrder {
        stage: 4,
        layer: usize::MAX,
        item: u8::MAX,
    }
}

pub(crate) fn native_text_metal_metrics_snapshot() -> Value {
    native_text_metal_metrics().snapshot()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_text_warm_order_recognizes_gemma_text_roots() {
        let mut names = [
            "zz.unclassified.weight",
            "model.norm.weight",
            "model.layers.2.mlp.down_proj.weight",
            "model.layers.2.self_attn.q_proj.weight",
            "model.layers.2.input_gate.weight",
            "model.embed_tokens_per_layer.weight",
            "model.embed_tokens.weight",
            "model.per_layer_model_projection.weight",
            "model.language_model.layers.1.self_attn.o_proj.weight",
            "model.language_model.layers.1.mlp.gate_proj.weight",
        ];

        names.sort_by(|left, right| {
            native_text_bf16_matrix_warm_order(left)
                .cmp(&native_text_bf16_matrix_warm_order(right))
                .then_with(|| left.cmp(right))
        });

        assert_eq!(
            names,
            [
                "model.embed_tokens.weight",
                "model.embed_tokens_per_layer.weight",
                "model.per_layer_model_projection.weight",
                "model.language_model.layers.1.self_attn.o_proj.weight",
                "model.language_model.layers.1.mlp.gate_proj.weight",
                "model.layers.2.self_attn.q_proj.weight",
                "model.layers.2.mlp.down_proj.weight",
                "model.layers.2.input_gate.weight",
                "model.norm.weight",
                "zz.unclassified.weight",
            ]
        );
    }
}
