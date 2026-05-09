use crate::{
    native_text::{
        NativeTextAdapter, NativeTextCandidateDecision, NativeTextDriver, NativeTextPrefixCache,
        NativeTextPrefixCacheMetrics, NativeTextPrefixCacheNamespace, NativeTextPrefixCacheValue,
        sample_token_id_with_draw as native_text_sample_token_id_with_draw,
    },
    sync_ext::RecoverPoisonedMutex,
};
use async_trait::async_trait;
use futures::stream::BoxStream;
use llm_backend::{
    BackendCacheContext, BackendError, BackendModelMetadata, BackendOutput, BackendRequest,
    BackendStreamChunk, CpuQwenMatvecBackend, LayerKvCache, LinearAttentionCache, MathError,
    ModelBackend, QwenKvCacheTensor, QwenLayerCache, QwenMatvecBackend, SafeTensorShardStore,
    SamplingConfig, TensorLoadError, TopKLogit, TopKWeight,
    native_decode_token_with_cache_with_matvec, native_final_norm_for_spec_with_matvec,
    native_layer_caches_for_spec, native_lm_head_logits_for_spec_with_matvec,
    native_lm_head_top_k_for_spec_with_matvec, native_prefill_sequence_with_cache_with_matvec,
};
use llm_hub::SnapshotManifest;
use llm_models::{ModelFamily, NativeTextModelSpec};
use llm_tokenizer::HuggingFaceTokenizer;
use serde_json::{Value, json};
use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};
use tokio_util::sync::CancellationToken;

pub const DEFAULT_NATIVE_QWEN_MAX_NEW_TOKENS: u32 = 256;

#[derive(Clone)]
pub struct NativeQwenBackend {
    driver: NativeTextDriver<NativeQwenAdapter>,
}

#[derive(Clone)]
pub(crate) struct NativeQwenAdapter {
    model_id: String,
    metadata: BackendModelMetadata,
    spec: NativeTextModelSpec,
    store: SafeTensorShardStore,
    matvec: NativeQwenMatvecBackend,
    max_prefill_tokens: usize,
    top_k: usize,
    chunk_rows: usize,
    prefix_cache: Arc<NativeQwenPrefixCache>,
}

#[derive(Clone)]
enum NativeQwenMatvecBackend {
    Cpu,
    Metal(Arc<NativeQwenMetalState>),
}

const DEFAULT_NATIVE_QWEN_METAL_WEIGHT_CACHE_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const DEFAULT_NATIVE_QWEN_PREFIX_CACHE_BYTES: u64 = 512 * 1024 * 1024;
const NATIVE_QWEN_PREFIX_CACHE_LAYOUT_VERSION: u32 = 1;

type NativeQwenPrefixCache = NativeTextPrefixCache<QwenLayerCache>;
type NativeQwenPrefixCacheNamespace = NativeTextPrefixCacheNamespace;
type NativeQwenPrefixCacheMetrics = NativeTextPrefixCacheMetrics;

fn native_qwen_prefix_cache_metrics() -> &'static NativeQwenPrefixCacheMetrics {
    static METRICS: OnceLock<NativeQwenPrefixCacheMetrics> = OnceLock::new();
    METRICS.get_or_init(NativeQwenPrefixCacheMetrics::default)
}

fn native_qwen_prefix_entry_bytes(hidden: &[f32], caches: &[QwenLayerCache]) -> u64 {
    let hidden_bytes = std::mem::size_of_val(hidden) as u64;
    caches.iter().fold(hidden_bytes, |total, cache| {
        total.saturating_add(match cache {
            QwenLayerCache::Full(cache) => {
                ((cache.key_storage().len() + cache.value_storage().len())
                    * std::mem::size_of::<f32>()) as u64
            }
            QwenLayerCache::Linear(cache) => {
                ((cache.conv_window().len() + cache.recurrent_state().len())
                    * std::mem::size_of::<f32>()) as u64
            }
        })
    })
}

impl NativeTextPrefixCacheValue for QwenLayerCache {
    fn prefix_cache_entry_bytes(hidden: &[f32], caches: &[Self]) -> u64 {
        native_qwen_prefix_entry_bytes(hidden, caches)
    }
}

fn native_qwen_prefix_namespace(
    adapter: &NativeQwenAdapter,
    request: &BackendRequest,
    cache_tokens: usize,
) -> NativeQwenPrefixCacheNamespace {
    NativeQwenPrefixCacheNamespace {
        model_id: adapter.model_id.clone(),
        backend: adapter.metadata.backend.clone(),
        family: adapter.metadata.family.clone(),
        loader: adapter.metadata.loader.clone(),
        quantization: adapter.metadata.quantization.clone(),
        repo_id: adapter.metadata.repo_id.clone(),
        resolved_commit: adapter.metadata.resolved_commit.clone(),
        profile: adapter.metadata.profile.clone(),
        manifest_digest: adapter.metadata.manifest_digest.clone(),
        prompt_template: backend_request_cache_prompt_template(request),
        tool_schema: request.cache_context.tool_schema.clone(),
        request_mode: native_qwen_prefix_request_mode(request),
        sampling: native_qwen_prefix_sampling_key(request.sampling),
        cache_layout_version: NATIVE_QWEN_PREFIX_CACHE_LAYOUT_VERSION,
        cache_tokens,
        max_prefill_tokens: adapter.max_prefill_tokens,
    }
}

fn native_qwen_prefix_request_mode(request: &BackendRequest) -> String {
    format!(
        "conversation={},json_object={},required_tool={:?}",
        request.conversation_mode, request.json_object_mode, request.required_tool_choice
    )
}

fn backend_request_cache_prompt_template(request: &BackendRequest) -> String {
    if request.cache_context.prompt_template.is_empty() {
        BackendCacheContext::raw_prompt().prompt_template
    } else {
        request.cache_context.prompt_template.clone()
    }
}

fn native_qwen_prefix_sampling_key(sampling: SamplingConfig) -> String {
    match sampling {
        SamplingConfig::Greedy => "greedy".to_owned(),
        SamplingConfig::TopP { temperature, top_p } => {
            format!(
                "top_p:{:08x}:{:08x}",
                temperature.to_bits(),
                top_p.to_bits()
            )
        }
    }
}

struct NativeQwenMetalState {
    device: llm_metal::MetalDevice,
    bf16_matrices: Mutex<Bf16MatrixBufferCache<Arc<llm_metal::Bf16MatrixBuffer>>>,
    kv_caches: Mutex<HashMap<u64, MetalLayerKvCacheMirror>>,
    linear_caches: Mutex<HashMap<u64, MetalLinearAttentionCacheMirror>>,
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

type NativeQwenMetalStateRegistry =
    Mutex<HashMap<NativeQwenMetalStateKey, Arc<NativeQwenMetalState>>>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct NativeQwenMetalStateKey {
    cache_namespace: String,
    weight_cache_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Bf16MatrixCacheKey {
    tensor: String,
    element_offset: usize,
    rows: usize,
    columns: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WarmableBf16MatrixTensor {
    name: String,
    rows: usize,
    columns: usize,
    byte_len: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct NativeQwenWeightWarmOrder {
    stage: u8,
    layer: usize,
    item: u8,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct NativeQwenMetalWarmup {
    candidates: u64,
    warmed: u64,
    already_resident: u64,
    skipped_budget: u64,
    skipped_non_metal: u64,
}

#[derive(Debug)]
struct Bf16MatrixBufferCache<T> {
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
struct Bf16MatrixBufferCacheInsert {
    inserted: bool,
    evicted_count: u64,
    evicted_bytes: u64,
}

impl<T: Clone> Bf16MatrixBufferCache<T> {
    fn new(max_bytes: u64) -> Self {
        Self {
            max_bytes,
            used_bytes: 0,
            next_access: 0,
            entries: HashMap::new(),
        }
    }

    fn get(&mut self, key: &Bf16MatrixCacheKey) -> Option<T> {
        let access = self.next_access();
        self.entries.get_mut(key).map(|entry| {
            entry.last_used = access;
            entry.value.clone()
        })
    }

    fn insert(
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
    fn used_bytes(&self) -> u64 {
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
enum NativeQwenMetalBufferError {
    Shape(String),
    Tensor(TensorLoadError),
    Metal(llm_metal::MetalError),
}

impl std::fmt::Display for NativeQwenMetalBufferError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shape(message) => formatter.write_str(message),
            Self::Tensor(err) => write!(formatter, "{err}"),
            Self::Metal(err) => write!(formatter, "{err}"),
        }
    }
}

impl NativeQwenMetalState {
    fn new(device: llm_metal::MetalDevice, weight_cache_bytes: u64) -> Self {
        native_qwen_metal_metrics().record_bf16_matrix_cache_residency(0, 0, weight_cache_bytes);
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
    ) -> Result<Arc<llm_metal::Bf16MatrixBuffer>, NativeQwenMetalBufferError> {
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
            native_qwen_metal_metrics().record_bf16_matrix_cache_hit();
            return Ok(buffer);
        }
        native_qwen_metal_metrics().record_bf16_matrix_cache_miss();
        let element_count = rows.checked_mul(columns).ok_or_else(|| {
            NativeQwenMetalBufferError::Shape("BF16 matrix element count overflow".to_owned())
        })?;
        let weights = store
            .bf16_tensor_bits_range(tensor, element_offset, element_count)
            .map_err(NativeQwenMetalBufferError::Tensor)?;
        let buffer = Arc::new(
            self.device
                .new_bf16_matrix_buffer(&weights, rows, columns)
                .map_err(NativeQwenMetalBufferError::Metal)?,
        );
        let mut matrices = self
            .bf16_matrices
            .lock_or_recover("BF16 matrix buffer cache");
        if let Some(existing) = matrices.get(&key) {
            native_qwen_metal_metrics().record_bf16_matrix_cache_hit();
            return Ok(existing);
        }
        let byte_len = buffer.byte_len() as u64;
        let insert = matrices.insert(key, Arc::clone(&buffer), byte_len);
        let metrics = native_qwen_metal_metrics();
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
    ) -> Result<NativeQwenMetalWarmup, NativeQwenMetalBufferError> {
        let tensors = native_qwen_warmable_bf16_matrix_tensors(store)
            .map_err(NativeQwenMetalBufferError::Tensor)?;
        let mut warmup = NativeQwenMetalWarmup {
            candidates: tensors.len() as u64,
            ..NativeQwenMetalWarmup::default()
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
                native_qwen_metal_metrics().record_kv_cache_sync(byte_len);
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
                native_qwen_metal_metrics().record_kv_cache_allocation(byte_len);
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
                native_qwen_metal_metrics().record_linear_cache_sync(byte_len);
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
                native_qwen_metal_metrics().record_linear_cache_allocation(byte_len);
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

    fn remove_cache_mirrors(&self, caches: &[QwenLayerCache]) {
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
                native_qwen_metal_metrics().record_kv_cache_eviction(count, bytes);
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
                native_qwen_metal_metrics().record_linear_cache_eviction(count, bytes);
                self.record_linear_cache_residency_locked(&mirrors);
            }
        }
    }

    fn record_kv_cache_residency_locked(&self, caches: &HashMap<u64, MetalLayerKvCacheMirror>) {
        let resident_bytes = caches
            .values()
            .map(|mirror| mirror.keys.byte_len() as u64 + mirror.values.byte_len() as u64)
            .sum();
        native_qwen_metal_metrics()
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
        native_qwen_metal_metrics()
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
struct MetalBackendMetrics {
    counters: Mutex<HashMap<&'static str, MetalKernelCounters>>,
    bf16_matrix_cache: Mutex<MetalBf16MatrixCacheCounters>,
    kv_cache: Mutex<MetalCacheCounters>,
    linear_cache: Mutex<MetalCacheCounters>,
    warned_fallbacks: Mutex<HashSet<String>>,
}

impl MetalBackendMetrics {
    fn record_attempt(&self, kernel: &'static str) {
        self.update_counter(kernel, |counters| counters.attempts += 1);
    }

    fn record_success(&self, kernel: &'static str) {
        self.update_counter(kernel, |counters| counters.successes += 1);
    }

    fn record_fallback(
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
                target: "native_qwen_metal",
                kernel,
                shape_bucket = %bucket,
                error = %error,
                "native Qwen Metal kernel fell back to CPU"
            );
        } else {
            tracing::debug!(
                target: "native_qwen_metal",
                kernel,
                shape_bucket = %bucket,
                error = %error,
                "native Qwen Metal kernel fell back to CPU"
            );
        }
    }

    fn record_bf16_matrix_cache_hit(&self) {
        let mut cache = self
            .bf16_matrix_cache
            .lock_or_recover("Metal BF16 matrix cache metrics");
        cache.hits += 1;
    }

    fn record_bf16_matrix_cache_miss(&self) {
        let mut cache = self
            .bf16_matrix_cache
            .lock_or_recover("Metal BF16 matrix cache metrics");
        cache.misses += 1;
    }

    fn record_bf16_matrix_cache_upload(&self, byte_len: u64) {
        let mut cache = self
            .bf16_matrix_cache
            .lock_or_recover("Metal BF16 matrix cache metrics");
        cache.uploads += 1;
        cache.bytes_uploaded += byte_len;
    }

    fn record_bf16_matrix_cache_eviction(&self, count: u64, byte_len: u64) {
        let mut cache = self
            .bf16_matrix_cache
            .lock_or_recover("Metal BF16 matrix cache metrics");
        cache.evictions += count;
        cache.bytes_evicted += byte_len;
    }

    fn record_bf16_matrix_cache_residency(
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

    fn record_kv_cache_allocation(&self, byte_len: u64) {
        self.update_cache_counter(CacheMetricKind::Kv, |cache| {
            cache.allocations += 1;
            cache.bytes_uploaded += byte_len;
        });
    }

    fn record_kv_cache_sync(&self, byte_len: u64) {
        self.update_cache_counter(CacheMetricKind::Kv, |cache| {
            cache.syncs += 1;
            cache.bytes_uploaded += byte_len;
        });
    }

    fn record_kv_cache_eviction(&self, count: u64, byte_len: u64) {
        self.update_cache_counter(CacheMetricKind::Kv, |cache| {
            cache.evictions += count;
            cache.bytes_evicted += byte_len;
        });
    }

    fn record_kv_cache_residency(&self, resident_bytes: u64, resident_buffers: u64) {
        self.update_cache_counter(CacheMetricKind::Kv, |cache| {
            cache.resident_bytes = resident_bytes;
            cache.resident_buffers = resident_buffers;
        });
    }

    fn record_linear_cache_allocation(&self, byte_len: u64) {
        self.update_cache_counter(CacheMetricKind::Linear, |cache| {
            cache.allocations += 1;
            cache.bytes_uploaded += byte_len;
        });
    }

    fn record_linear_cache_sync(&self, byte_len: u64) {
        self.update_cache_counter(CacheMetricKind::Linear, |cache| {
            cache.syncs += 1;
            cache.bytes_uploaded += byte_len;
        });
    }

    fn record_linear_cache_eviction(&self, count: u64, byte_len: u64) {
        self.update_cache_counter(CacheMetricKind::Linear, |cache| {
            cache.evictions += count;
            cache.bytes_evicted += byte_len;
        });
    }

    fn record_linear_cache_residency(&self, resident_bytes: u64, resident_buffers: u64) {
        self.update_cache_counter(CacheMetricKind::Linear, |cache| {
            cache.resident_bytes = resident_bytes;
            cache.resident_buffers = resident_buffers;
        });
    }

    fn snapshot(&self) -> Value {
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

fn native_qwen_metal_metrics() -> &'static MetalBackendMetrics {
    static METRICS: OnceLock<MetalBackendMetrics> = OnceLock::new();
    METRICS.get_or_init(MetalBackendMetrics::default)
}

fn native_qwen_metal_state_registry() -> &'static NativeQwenMetalStateRegistry {
    static REGISTRY: OnceLock<NativeQwenMetalStateRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn native_qwen_shared_metal_state(
    weight_cache_bytes: u64,
    cache_namespace: &str,
) -> Result<Option<Arc<NativeQwenMetalState>>, llm_metal::MetalError> {
    let key = NativeQwenMetalStateKey {
        cache_namespace: cache_namespace.to_owned(),
        weight_cache_bytes,
    };
    let registry = native_qwen_metal_state_registry();
    if let Some(state) = registry
        .lock_or_recover("native Qwen Metal state registry")
        .get(&key)
        .cloned()
    {
        return Ok(Some(state));
    }
    let Some(device) = llm_metal::MetalDevice::system_default_result()? else {
        return Ok(None);
    };
    let mut states = registry.lock_or_recover("native Qwen Metal state registry");
    if let Some(state) = states.get(&key).cloned() {
        return Ok(Some(state));
    }
    let state = Arc::new(NativeQwenMetalState::new(device, weight_cache_bytes));
    states.insert(key, Arc::clone(&state));
    Ok(Some(state))
}

impl NativeQwenMatvecBackend {
    fn system_default(weight_cache_bytes: u64, cache_namespace: &str) -> Self {
        match native_qwen_shared_metal_state(weight_cache_bytes, cache_namespace) {
            Ok(Some(state)) => Self::Metal(state),
            Ok(None) => Self::Cpu,
            Err(err) => {
                tracing::warn!("Metal Qwen matvec backend unavailable: {err}");
                Self::Cpu
            }
        }
    }

    fn cpu() -> CpuQwenMatvecBackend {
        CpuQwenMatvecBackend
    }

    fn metal_state(&self) -> Option<Arc<NativeQwenMetalState>> {
        match self {
            Self::Cpu => None,
            Self::Metal(state) => Some(Arc::clone(state)),
        }
    }

    fn warm_bf16_matrix_cache(
        &self,
        store: &SafeTensorShardStore,
    ) -> Result<NativeQwenMetalWarmup, NativeQwenMetalBufferError> {
        let candidates = native_qwen_warmable_bf16_matrix_tensors(store)
            .map_err(NativeQwenMetalBufferError::Tensor)?
            .len() as u64;
        match self {
            Self::Cpu => Ok(NativeQwenMetalWarmup {
                candidates,
                skipped_non_metal: candidates,
                ..NativeQwenMetalWarmup::default()
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
        native_qwen_metal_metrics().record_fallback(kernel, bucket, error);
    }

    fn run_metal_math<T>(
        kernel: &'static str,
        bucket: impl Into<String>,
        metal: impl FnOnce() -> Result<T, llm_metal::MetalError>,
        cpu: impl FnOnce() -> Result<T, MathError>,
    ) -> Result<T, MathError> {
        let metrics = native_qwen_metal_metrics();
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
        let metrics = native_qwen_metal_metrics();
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

impl QwenMatvecBackend for NativeQwenMatvecBackend {
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
            let metrics = native_qwen_metal_metrics();
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
            let metrics = native_qwen_metal_metrics();
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
                let metrics = native_qwen_metal_metrics();
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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativeQwenLoadOptions {
    pub eager_materialize_shards: bool,
    pub metal_weight_cache_bytes: Option<u64>,
    pub warm_metal_weight_cache: bool,
}
impl NativeQwenBackend {
    pub fn open(
        model_id: impl Into<String>,
        snapshot_path: impl AsRef<Path>,
    ) -> anyhow::Result<Self> {
        Self::open_with_options(model_id, snapshot_path, NativeQwenLoadOptions::default())
    }

    pub fn open_with_options(
        model_id: impl Into<String>,
        snapshot_path: impl AsRef<Path>,
        options: NativeQwenLoadOptions,
    ) -> anyhow::Result<Self> {
        let model_id = model_id.into();
        let snapshot_path = snapshot_path.as_ref();
        let cache_namespace = snapshot_path.canonicalize()?.to_string_lossy().into_owned();
        let config_json = std::fs::read_to_string(snapshot_path.join("config.json"))?;
        let metadata = native_qwen_metadata(&model_id, snapshot_path)?;
        let store = SafeTensorShardStore::open(snapshot_path)?;
        if options.eager_materialize_shards {
            let materialized_bytes = store.materialize_all_shards()?;
            tracing::info!(
                materialized_bytes,
                "materialized native Qwen safetensors shards"
            );
        }
        let matvec = NativeQwenMatvecBackend::system_default(
            native_qwen_metal_weight_cache_bytes(options.metal_weight_cache_bytes),
            &cache_namespace,
        );
        if options.warm_metal_weight_cache {
            let warmup = matvec.warm_bf16_matrix_cache(&store).map_err(|err| {
                anyhow::anyhow!("native Qwen Metal weight cache warm-up failed: {err}")
            })?;
            tracing::info!(
                candidates = warmup.candidates,
                warmed = warmup.warmed,
                already_resident = warmup.already_resident,
                skipped_budget = warmup.skipped_budget,
                skipped_non_metal = warmup.skipped_non_metal,
                "native Qwen Metal BF16 weight cache warm-up complete"
            );
        }
        let tokenizer = HuggingFaceTokenizer::from_file(snapshot_path.join("tokenizer.json"))?;
        let adapter = NativeQwenAdapter {
            model_id: model_id.clone(),
            metadata: metadata.clone(),
            spec: NativeTextModelSpec::from_config_json(ModelFamily::Qwen, &config_json)?,
            store,
            matvec,
            max_prefill_tokens: 32,
            top_k: 16,
            chunk_rows: 2048,
            prefix_cache: Arc::new(NativeQwenPrefixCache::new(
                DEFAULT_NATIVE_QWEN_PREFIX_CACHE_BYTES,
            )),
        };
        Ok(Self {
            driver: NativeTextDriver::new(
                model_id,
                metadata,
                tokenizer,
                adapter,
                DEFAULT_NATIVE_QWEN_MAX_NEW_TOKENS,
            ),
        })
    }

    pub fn with_max_new_tokens(mut self, max_new_tokens: u32) -> Self {
        self.driver = self.driver.with_max_new_tokens(max_new_tokens);
        self
    }

    pub fn with_max_prefill_tokens(mut self, max_prefill_tokens: usize) -> Self {
        self.driver = self.driver.with_max_prefill_tokens(max_prefill_tokens);
        self
    }

    pub(crate) fn into_driver(self) -> NativeTextDriver<NativeQwenAdapter> {
        self.driver
    }

    #[cfg(test)]
    fn generate_blocking_stream(
        &self,
        request: BackendRequest,
        tx: tokio::sync::mpsc::Sender<Result<BackendStreamChunk, BackendError>>,
        cancellation: CancellationToken,
    ) -> Result<(), BackendError> {
        self.driver
            .generate_blocking_stream(request, tx, cancellation)
    }

    #[cfg(test)]
    fn start_decode_session(
        &self,
        context_tokens: &[usize],
        max_new_tokens: u32,
        request: &BackendRequest,
        cancellation: &CancellationToken,
    ) -> Result<NativeQwenDecodeSession, BackendError> {
        self.driver
            .start_decode_session(context_tokens, max_new_tokens, request, cancellation)
    }

    #[cfg(test)]
    fn next_token_from_hidden(
        &self,
        hidden: &[f32],
        sampling: SamplingConfig,
    ) -> Result<NativeQwenCandidate, BackendError> {
        Ok(NativeQwenCandidate {
            token_id: self
                .driver
                .adapter
                .next_token_from_hidden(hidden, sampling)?,
        })
    }
}

impl NativeTextAdapter for NativeQwenAdapter {
    type DecodeSession = NativeQwenDecodeSession;
    type LayerCache = QwenLayerCache;

    fn family_display_name(&self) -> &'static str {
        "Qwen"
    }

    fn worker_label(&self) -> &'static str {
        "native Qwen"
    }

    fn set_max_prefill_tokens(&mut self, max_prefill_tokens: usize) {
        self.max_prefill_tokens = max_prefill_tokens.max(1);
    }

    fn encode_prompt(
        &self,
        tokenizer: &HuggingFaceTokenizer,
        request: &BackendRequest,
    ) -> Result<Vec<u32>, BackendError> {
        tokenizer
            .encode(&request.prompt, false)
            .map_err(|err| BackendError::Other(err.to_string()))
    }

    fn decode_output(
        &self,
        tokenizer: &HuggingFaceTokenizer,
        output_ids: &[u32],
    ) -> Result<String, BackendError> {
        tokenizer
            .decode(output_ids, false)
            .map_err(|err| BackendError::Other(err.to_string()))
    }

    fn observe_candidate(
        &self,
        tokenizer: &HuggingFaceTokenizer,
        _emitted_tokens: &[u32],
        token_id: usize,
    ) -> Result<NativeTextCandidateDecision, BackendError> {
        if tokenizer
            .token_to_id("<|im_end|>")
            .is_some_and(|stop_id| token_id == stop_id as usize)
        {
            return Ok(NativeTextCandidateDecision::Stop);
        }
        Ok(NativeTextCandidateDecision::Emit(token_id))
    }

    fn cache_token_capacity(
        &self,
        context_tokens: usize,
        max_new_tokens: u32,
    ) -> Result<usize, BackendError> {
        native_qwen_cache_token_capacity(
            context_tokens,
            max_new_tokens,
            self.max_prefill_tokens,
            self.spec.max_position_embeddings(),
        )
    }

    fn prefix_cache(&self) -> &NativeTextPrefixCache<QwenLayerCache> {
        &self.prefix_cache
    }

    fn prefix_cache_metrics(&self) -> &NativeTextPrefixCacheMetrics {
        native_qwen_prefix_cache_metrics()
    }

    fn prefix_cache_namespace(
        &self,
        request: &BackendRequest,
        cache_tokens: usize,
    ) -> NativeTextPrefixCacheNamespace {
        native_qwen_prefix_namespace(self, request, cache_tokens)
    }

    fn layer_count(&self) -> usize {
        self.spec.num_hidden_layers() as usize
    }

    fn allocate_caches(&self, cache_tokens: usize) -> Result<Vec<QwenLayerCache>, BackendError> {
        native_layer_caches_for_spec(&self.spec, cache_tokens)
            .map_err(|err| BackendError::Other(err.to_string()))
    }

    fn prefill_context_with_cache(
        &self,
        context_tokens: &[usize],
        caches: &mut [QwenLayerCache],
        cancellation: &CancellationToken,
    ) -> Result<Vec<f32>, BackendError> {
        native_qwen_prefill_context_with_cache(
            &self.store,
            &self.spec,
            context_tokens,
            caches,
            &self.matvec,
            self.max_prefill_tokens,
            cancellation,
        )
    }

    fn make_decode_session(
        &self,
        hidden: Vec<f32>,
        caches: Vec<QwenLayerCache>,
    ) -> NativeQwenDecodeSession {
        NativeQwenDecodeSession {
            hidden,
            caches,
            metal_state: self.matvec.metal_state(),
        }
    }

    fn hidden<'a>(&self, session: &'a NativeQwenDecodeSession) -> &'a [f32] {
        session.hidden()
    }

    fn step(
        &self,
        session: &mut NativeQwenDecodeSession,
        token_id: usize,
    ) -> Result<(), BackendError> {
        session.step(&self.store, &self.spec, &self.matvec, token_id)
    }

    fn next_token_from_hidden(
        &self,
        hidden: &[f32],
        sampling: SamplingConfig,
    ) -> Result<usize, BackendError> {
        let final_norm =
            native_final_norm_for_spec_with_matvec(&self.store, &self.spec, hidden, &self.matvec)
                .map_err(|err| BackendError::Other(err.to_string()))?;
        if !sampling.is_greedy() {
            let logits = native_lm_head_logits_for_spec_with_matvec(
                &self.store,
                &self.spec,
                &final_norm,
                self.chunk_rows,
                &self.matvec,
            )
            .map_err(|err| BackendError::Other(err.to_string()))?;
            let sampled_token_id =
                sample_token_id_with_draw(&logits, sampling, native_sampling_draw())?;
            u32::try_from(sampled_token_id).map_err(|err| {
                BackendError::Other(format!("Qwen token id does not fit u32: {err}"))
            })?;
            return Ok(sampled_token_id);
        }

        let top_logits = native_lm_head_top_k_for_spec_with_matvec(
            &self.store,
            &self.spec,
            &final_norm,
            self.top_k,
            self.chunk_rows,
            &self.matvec,
        )
        .map_err(|err| BackendError::Other(err.to_string()))?;

        let item = top_logits
            .into_iter()
            .next()
            .ok_or_else(|| BackendError::Other("Qwen lm head returned no logits".to_owned()))?;
        u32::try_from(item.index)
            .map_err(|err| BackendError::Other(format!("Qwen token id does not fit u32: {err}")))?;
        Ok(item.index)
    }
}

fn native_qwen_prefill_context_with_cache(
    store: &SafeTensorShardStore,
    spec: &NativeTextModelSpec,
    context_tokens: &[usize],
    caches: &mut [QwenLayerCache],
    matvec: &impl QwenMatvecBackend,
    prefill_chunk_tokens: usize,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, BackendError> {
    if cancellation.is_cancelled() {
        return Err(BackendError::Cancelled);
    }
    let mut hidden = None;
    for chunk in context_tokens.chunks(prefill_chunk_tokens.max(1)) {
        if cancellation.is_cancelled() {
            return Err(BackendError::Cancelled);
        }
        let hidden_states =
            native_prefill_sequence_with_cache_with_matvec(store, spec, chunk, caches, matvec)
                .map_err(|err| BackendError::Other(err.to_string()))?;
        if cancellation.is_cancelled() {
            return Err(BackendError::Cancelled);
        }
        hidden = hidden_states.last().cloned();
    }
    hidden.ok_or_else(|| BackendError::Other("Qwen prefill returned no hidden states".to_owned()))
}

pub(crate) struct NativeQwenDecodeSession {
    hidden: Vec<f32>,
    caches: Vec<QwenLayerCache>,
    metal_state: Option<Arc<NativeQwenMetalState>>,
}

impl NativeQwenDecodeSession {
    fn hidden(&self) -> &[f32] {
        &self.hidden
    }

    fn step(
        &mut self,
        store: &SafeTensorShardStore,
        spec: &NativeTextModelSpec,
        matvec: &impl QwenMatvecBackend,
        token_id: usize,
    ) -> Result<(), BackendError> {
        self.hidden = native_decode_token_with_cache_with_matvec(
            store,
            spec,
            token_id,
            &mut self.caches,
            matvec,
        )
        .map_err(|err| BackendError::Other(err.to_string()))?;
        Ok(())
    }
}

impl Drop for NativeQwenDecodeSession {
    fn drop(&mut self) {
        if let Some(state) = &self.metal_state {
            state.remove_cache_mirrors(&self.caches);
        }
    }
}

#[cfg(test)]
fn resolve_native_max_tokens(
    requested: Option<u32>,
    configured_max: u32,
) -> Result<u32, BackendError> {
    crate::native_text::resolve_native_text_max_tokens(requested, configured_max, "Qwen")
}

fn native_qwen_cache_token_capacity(
    context_tokens: usize,
    max_new_tokens: u32,
    min_cache_tokens: usize,
    max_position_embeddings: u32,
) -> Result<usize, BackendError> {
    let max_position_embeddings = usize::try_from(max_position_embeddings).map_err(|err| {
        BackendError::Other(format!(
            "native Qwen max_position_embeddings does not fit usize: {err}"
        ))
    })?;
    if max_position_embeddings == 0 {
        return Err(BackendError::UnsupportedRequest(
            "native Qwen model declares zero max_position_embeddings".to_owned(),
        ));
    }
    let max_new_tokens = usize::try_from(max_new_tokens).map_err(|err| {
        BackendError::Other(format!(
            "native Qwen max_new_tokens does not fit usize: {err}"
        ))
    })?;
    let requested_context = context_tokens.checked_add(max_new_tokens).ok_or_else(|| {
        BackendError::UnsupportedRequest(
            "native Qwen context length plus generation budget overflows usize".to_owned(),
        )
    })?;
    if requested_context > max_position_embeddings {
        return Err(BackendError::UnsupportedRequest(format!(
            "native Qwen request needs {context_tokens} prompt tokens plus {max_new_tokens} generation tokens, exceeding model context limit {max_position_embeddings}"
        )));
    }
    let required = requested_context.max(min_cache_tokens.max(1));
    Ok(required
        .checked_next_power_of_two()
        .unwrap_or(max_position_embeddings)
        .min(max_position_embeddings))
}

fn native_qwen_metal_weight_cache_bytes(configured: Option<u64>) -> u64 {
    configured.unwrap_or(DEFAULT_NATIVE_QWEN_METAL_WEIGHT_CACHE_BYTES)
}

fn native_qwen_warmable_bf16_matrix_tensors(
    store: &SafeTensorShardStore,
) -> Result<Vec<WarmableBf16MatrixTensor>, TensorLoadError> {
    let mut tensors = Vec::new();
    for name in store.tensor_names() {
        let metadata = store.tensor_metadata(name)?;
        if metadata.dtype == "BF16" && metadata.shape.len() == 2 {
            tensors.push(WarmableBf16MatrixTensor {
                name: name.to_owned(),
                rows: metadata.shape[0],
                columns: metadata.shape[1],
                byte_len: metadata.byte_len as u64,
            });
        }
    }
    tensors.sort_by(|left, right| {
        native_qwen_bf16_matrix_warm_order(&left.name)
            .cmp(&native_qwen_bf16_matrix_warm_order(&right.name))
            .then_with(|| left.name.cmp(&right.name))
    });
    Ok(tensors)
}

fn native_qwen_bf16_matrix_warm_order(name: &str) -> NativeQwenWeightWarmOrder {
    if name == "model.language_model.embed_tokens.weight" {
        return NativeQwenWeightWarmOrder {
            stage: 0,
            layer: 0,
            item: 0,
        };
    }
    if name == "lm_head.weight" {
        return NativeQwenWeightWarmOrder {
            stage: 3,
            layer: 0,
            item: 0,
        };
    }
    let Some(layer_suffix) = name.strip_prefix("model.language_model.layers.") else {
        return native_qwen_unknown_weight_warm_order();
    };
    let Some((layer, suffix)) = layer_suffix.split_once('.') else {
        return native_qwen_unknown_weight_warm_order();
    };
    let Ok(layer) = layer.parse::<usize>() else {
        return native_qwen_unknown_weight_warm_order();
    };
    let Some((stage, item)) = native_qwen_layer_bf16_matrix_warm_order(suffix) else {
        return native_qwen_unknown_weight_warm_order();
    };
    NativeQwenWeightWarmOrder { stage, layer, item }
}

fn native_qwen_layer_bf16_matrix_warm_order(suffix: &str) -> Option<(u8, u8)> {
    let item = match suffix {
        "self_attn.q_proj.weight" | "linear_attn.in_proj_qkv.weight" => 0,
        "self_attn.k_proj.weight" | "linear_attn.in_proj_z.weight" => 1,
        "self_attn.v_proj.weight" | "linear_attn.in_proj_b.weight" => 2,
        "self_attn.o_proj.weight" | "linear_attn.in_proj_a.weight" => 3,
        "linear_attn.out_proj.weight" => 4,
        "mlp.gate.weight" => 10,
        "mlp.shared_expert.gate_proj.weight" => 11,
        "mlp.shared_expert.up_proj.weight" => 12,
        "mlp.shared_expert.down_proj.weight" => 13,
        "mlp.shared_expert_gate.weight" => 14,
        _ => return None,
    };
    Some((1, item))
}

fn native_qwen_unknown_weight_warm_order() -> NativeQwenWeightWarmOrder {
    NativeQwenWeightWarmOrder {
        stage: 4,
        layer: usize::MAX,
        item: u8::MAX,
    }
}

#[cfg(test)]
#[derive(Debug, Clone)]
struct NativeQwenCandidate {
    token_id: usize,
}

fn sample_token_id_with_draw(
    logits: &[f32],
    sampling: SamplingConfig,
    draw: f32,
) -> Result<usize, BackendError> {
    native_text_sample_token_id_with_draw(logits, sampling, draw, "Qwen")
}

static NATIVE_SAMPLING_COUNTER: AtomicU64 = AtomicU64::new(0);

fn native_sampling_draw() -> f32 {
    let time_seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0);
    let counter = NATIVE_SAMPLING_COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut value = time_seed ^ counter.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    let bits = value.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40;
    (bits as f32) / ((1_u32 << 24) as f32)
}

#[async_trait]
impl ModelBackend for NativeQwenBackend {
    fn model_id(&self) -> &str {
        self.driver.model_id()
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        self.driver.model_metadata()
    }

    async fn generate(&self, request: BackendRequest) -> Result<BackendOutput, BackendError> {
        self.generate_with_cancel(request, CancellationToken::new())
            .await
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        self.driver
            .generate_with_cancel(request, cancellation)
            .await
    }

    fn generate_stream<'a>(
        &'a self,
        request: BackendRequest,
    ) -> BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        self.generate_stream_with_cancel(request, CancellationToken::new())
    }

    fn generate_stream_with_cancel<'a>(
        &'a self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        self.driver
            .generate_stream_with_cancel(request, cancellation)
    }
}

#[cfg(test)]
fn native_qwen_worker_stream(
    rx: tokio::sync::mpsc::Receiver<Result<BackendStreamChunk, BackendError>>,
    worker: tokio::task::JoinHandle<()>,
) -> BoxStream<'static, Result<BackendStreamChunk, BackendError>> {
    crate::native_text::native_text_worker_stream("native Qwen", rx, worker)
}
fn native_qwen_metadata(
    model_id: &str,
    snapshot_path: &Path,
) -> anyhow::Result<BackendModelMetadata> {
    let manifest_path = snapshot_path.join("llm-engine-manifest.json");
    let mut metadata =
        BackendModelMetadata::new(model_id.to_owned(), "native-qwen").with_family("qwen");
    metadata.loader = Some("native-metal".to_owned());
    metadata.snapshot_path = Some(PathBuf::from(snapshot_path));
    let manifest_bytes = match std::fs::read(&manifest_path) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(metadata),
        Err(err) => return Err(err.into()),
    };
    let manifest = serde_json::from_slice::<SnapshotManifest>(&manifest_bytes)?;
    if manifest.family != "qwen" {
        anyhow::bail!(
            "native Qwen backend only supports family `qwen`, not `{}`",
            manifest.family
        );
    }
    if manifest.loader != "native-metal" {
        anyhow::bail!(
            "native Qwen backend only supports loader `native-metal`, not `{}`",
            manifest.loader
        );
    }
    metadata.family = Some(manifest.family.clone());
    metadata.loader = Some(manifest.loader.clone());
    metadata.quantization = Some(manifest.quantization.clone());
    metadata.repo_id = Some(manifest.repo_id.clone());
    metadata.resolved_commit = Some(manifest.resolved_commit.clone());
    metadata.profile = Some(manifest.profile.clone());
    metadata.manifest_digest = Some(manifest.digest());
    Ok(metadata)
}

pub(crate) fn native_qwen_metal_metrics_snapshot() -> Value {
    native_qwen_metal_metrics().snapshot()
}

pub(crate) fn native_qwen_prefix_cache_metrics_snapshot() -> Value {
    native_qwen_prefix_cache_metrics().snapshot()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_text::NativeStreamTextDeltas;
    use futures::StreamExt;
    use llm_backend::qwen_layer_caches_for_spec;
    use llm_models::QwenModelSpec;
    use llm_models::{ModelFamilyAdapter, QwenFamilyAdapter};

    #[test]
    fn metal_backend_metrics_records_attempt_success_and_fallback_by_kernel() {
        let metrics = MetalBackendMetrics::default();

        metrics.record_attempt("matvec_bf16_f32");
        metrics.record_success("matvec_bf16_f32");
        metrics.record_attempt("matvec_bf16_f32");
        metrics.record_fallback("matvec_bf16_f32", "rows=2,cols=3", "execution failed");

        let snapshot = metrics.snapshot();
        let matvec = &snapshot["kernels"]["matvec_bf16_f32"];
        assert_eq!(matvec["attempts"], 2);
        assert_eq!(matvec["successes"], 1);
        assert_eq!(matvec["fallbacks"], 1);
    }

    #[test]
    fn metal_backend_metrics_records_bf16_matrix_cache_activity() {
        let metrics = MetalBackendMetrics::default();

        metrics.record_bf16_matrix_cache_miss();
        metrics.record_bf16_matrix_cache_upload(12);
        metrics.record_bf16_matrix_cache_eviction(2, 8);
        metrics.record_bf16_matrix_cache_residency(10, 3, 16);
        metrics.record_bf16_matrix_cache_hit();

        let snapshot = metrics.snapshot();
        let cache = &snapshot["bf16_matrix_cache"];
        assert_eq!(cache["hits"], 1);
        assert_eq!(cache["misses"], 1);
        assert_eq!(cache["uploads"], 1);
        assert_eq!(cache["bytes_uploaded"], 12);
        assert_eq!(cache["evictions"], 2);
        assert_eq!(cache["bytes_evicted"], 8);
        assert_eq!(cache["resident_bytes"], 10);
        assert_eq!(cache["resident_buffers"], 3);
        assert_eq!(cache["budget_bytes"], 16);
    }

    #[test]
    fn metal_backend_metrics_records_resident_attention_cache_activity() {
        let metrics = MetalBackendMetrics::default();

        metrics.record_kv_cache_allocation(16);
        metrics.record_kv_cache_sync(8);
        metrics.record_kv_cache_residency(16, 2);
        metrics.record_kv_cache_eviction(2, 16);
        metrics.record_kv_cache_residency(0, 0);
        metrics.record_linear_cache_allocation(12);
        metrics.record_linear_cache_sync(4);
        metrics.record_linear_cache_residency(12, 1);
        metrics.record_linear_cache_eviction(1, 12);
        metrics.record_linear_cache_residency(0, 0);

        let snapshot = metrics.snapshot();
        let kv = &snapshot["kv_cache"];
        assert_eq!(kv["allocations"], 1);
        assert_eq!(kv["syncs"], 1);
        assert_eq!(kv["evictions"], 2);
        assert_eq!(kv["bytes_uploaded"], 24);
        assert_eq!(kv["bytes_evicted"], 16);
        assert_eq!(kv["resident_bytes"], 0);
        assert_eq!(kv["resident_buffers"], 0);
        let linear = &snapshot["linear_attention_cache"];
        assert_eq!(linear["allocations"], 1);
        assert_eq!(linear["syncs"], 1);
        assert_eq!(linear["evictions"], 1);
        assert_eq!(linear["bytes_uploaded"], 16);
        assert_eq!(linear["bytes_evicted"], 12);
        assert_eq!(linear["resident_bytes"], 0);
        assert_eq!(linear["resident_buffers"], 0);
    }

    #[test]
    fn native_qwen_prefix_cache_reuses_longest_compatible_prefix() {
        let cache = NativeQwenPrefixCache::new(10_000);
        let metrics = NativeQwenPrefixCacheMetrics::default();
        let namespace = native_qwen_test_prefix_namespace("base");
        let mut layer_cache = LayerKvCache::new(4, 1, 2).expect("cache shape is valid");
        layer_cache
            .append(&[1.0, 2.0], &[3.0, 4.0])
            .expect("token fits");
        let original_cache_id = layer_cache.id();
        let caches = vec![QwenLayerCache::Full(layer_cache)];

        cache.store(namespace.clone(), &[1, 2], &[0.25, 0.75], &caches, &metrics);

        let hit = cache
            .lookup(&namespace, &[1, 2, 3], &metrics)
            .expect("compatible longer prompt reuses stored prefix");
        assert_eq!(hit.token_count, 2);
        assert_eq!(hit.hidden, vec![0.25, 0.75]);
        match &hit.caches[0] {
            QwenLayerCache::Full(cache) => {
                assert_ne!(cache.id(), original_cache_id);
                assert_eq!(cache.token_count(), 1);
            }
            QwenLayerCache::Linear(_) => panic!("expected full-attention cache"),
        }

        let incompatible_namespace = NativeQwenPrefixCacheNamespace {
            tool_schema: Some("different-tool-schema".to_owned()),
            ..namespace.clone()
        };
        assert!(
            cache
                .lookup(&incompatible_namespace, &[1, 2], &metrics)
                .is_none(),
            "tool schema changes must not reuse prefix state"
        );
    }

    #[test]
    fn native_qwen_prefix_cache_evicts_lru_entries_to_fit_budget() {
        let cache = NativeQwenPrefixCache::new(40);
        let metrics = NativeQwenPrefixCacheMetrics::default();
        let namespace = native_qwen_test_prefix_namespace("eviction");
        let hidden = vec![1.0; 8];

        cache.store(namespace.clone(), &[1], &hidden, &[], &metrics);
        cache.store(namespace.clone(), &[2], &hidden, &[], &metrics);

        assert!(
            cache.lookup(&namespace, &[1], &metrics).is_none(),
            "oldest entry should be evicted"
        );
        assert!(
            cache.lookup(&namespace, &[2], &metrics).is_some(),
            "newest entry should remain resident"
        );
        let inner = cache.inner.lock_or_recover("native Qwen prefix cache");
        assert_eq!(inner.entries.len(), 1);
        assert_eq!(inner.used_bytes, 32);
    }

    #[test]
    fn native_qwen_prefix_cache_metrics_expose_hits_misses_and_evictions() {
        let metrics = NativeQwenPrefixCacheMetrics::default();

        metrics.record_hit(3);
        metrics.record_miss();
        metrics.record_store(32);
        metrics.record_eviction(16);
        metrics.record_rejected();
        metrics.record_residency(32, 1);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot["hits"], 1);
        assert_eq!(snapshot["misses"], 1);
        assert_eq!(snapshot["stores"], 1);
        assert_eq!(snapshot["evictions"], 1);
        assert_eq!(snapshot["rejected"], 1);
        assert_eq!(snapshot["reused_tokens"], 3);
        assert_eq!(snapshot["bytes_stored"], 32);
        assert_eq!(snapshot["bytes_evicted"], 16);
        assert_eq!(snapshot["resident_bytes"], 32);
        assert_eq!(snapshot["resident_entries"], 1);
    }

    #[test]
    fn bf16_matrix_buffer_cache_evicts_lru_entries_to_fit_budget() {
        let mut cache = Bf16MatrixBufferCache::new(10);
        let first = Bf16MatrixCacheKey {
            tensor: "first.weight".to_owned(),
            element_offset: 0,
            rows: 2,
            columns: 1,
        };
        let second = Bf16MatrixCacheKey {
            tensor: "second.weight".to_owned(),
            element_offset: 0,
            rows: 2,
            columns: 1,
        };
        let third = Bf16MatrixCacheKey {
            tensor: "third.weight".to_owned(),
            element_offset: 0,
            rows: 3,
            columns: 1,
        };

        assert!(cache.get(&first).is_none());
        assert!(cache.insert(first.clone(), "first", 4).inserted);
        assert!(cache.insert(second.clone(), "second", 4).inserted);
        assert_eq!(cache.get(&first), Some("first"));

        let result = cache.insert(third.clone(), "third", 6);

        assert!(result.inserted);
        assert_eq!(result.evicted_count, 1);
        assert_eq!(result.evicted_bytes, 4);
        assert_eq!(cache.used_bytes(), 10);
        assert_eq!(cache.get(&second), None);
        assert_eq!(cache.get(&first), Some("first"));
        assert_eq!(cache.get(&third), Some("third"));
    }

    #[test]
    fn bf16_matrix_buffer_cache_skips_entries_larger_than_budget() {
        let mut cache = Bf16MatrixBufferCache::new(4);
        let key = Bf16MatrixCacheKey {
            tensor: "large.weight".to_owned(),
            element_offset: 0,
            rows: 3,
            columns: 1,
        };

        let result = cache.insert(key.clone(), "large", 6);

        assert!(!result.inserted);
        assert_eq!(result.evicted_count, 0);
        assert_eq!(cache.used_bytes(), 0);
        assert_eq!(cache.get(&key), None);
    }

    #[test]
    fn native_qwen_metal_weight_cache_bytes_uses_default_or_configured_value() {
        assert_eq!(
            native_qwen_metal_weight_cache_bytes(None),
            DEFAULT_NATIVE_QWEN_METAL_WEIGHT_CACHE_BYTES
        );
        assert_eq!(native_qwen_metal_weight_cache_bytes(Some(0)), 0);
        assert_eq!(native_qwen_metal_weight_cache_bytes(Some(4096)), 4096);
    }

    #[test]
    fn native_qwen_warmable_bf16_matrix_tensors_filters_rank2_bf16() {
        let snapshot = temp_snapshot_dir("warmable-bf16-matrices");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        let tensors = vec![
            ("z.bias", vec![2], vec![1.0, 2.0]),
            ("b.weight", vec![2, 1], vec![3.0, 4.0]),
            ("a.weight", vec![1, 2], vec![5.0, 6.0]),
        ];
        let safetensors = tiny_owned_multi_safetensors_bf16(&tensors);
        std::fs::write(snapshot.join("model.safetensors"), &safetensors).expect("write shard");
        std::fs::write(
            snapshot.join("model.safetensors.index.json"),
            serde_json::json!({
                "metadata": { "total_size": safetensors.len() },
                "weight_map": {
                    "z.bias": "model.safetensors",
                    "b.weight": "model.safetensors",
                    "a.weight": "model.safetensors"
                }
            })
            .to_string(),
        )
        .expect("write index");
        let store = SafeTensorShardStore::open(&snapshot).expect("store opens");

        let warmable = native_qwen_warmable_bf16_matrix_tensors(&store).expect("warmable tensors");

        assert_eq!(
            warmable
                .iter()
                .map(|tensor| (
                    tensor.name.as_str(),
                    tensor.rows,
                    tensor.columns,
                    tensor.byte_len
                ))
                .collect::<Vec<_>>(),
            vec![("a.weight", 1, 2, 4), ("b.weight", 2, 1, 4)]
        );
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn native_qwen_warmable_bf16_matrix_tensors_orders_qwen_execution_weights() {
        let snapshot = temp_snapshot_dir("warmable-qwen-order");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        let tensors = vec![
            ("zz.unclassified.weight", vec![1, 1], vec![1.0]),
            ("lm_head.weight", vec![1, 1], vec![2.0]),
            (
                "model.language_model.layers.10.self_attn.o_proj.weight",
                vec![1, 1],
                vec![3.0],
            ),
            (
                "model.language_model.layers.2.mlp.shared_expert.down_proj.weight",
                vec![1, 1],
                vec![4.0],
            ),
            (
                "model.language_model.layers.2.self_attn.q_proj.weight",
                vec![1, 1],
                vec![5.0],
            ),
            (
                "model.language_model.embed_tokens.weight",
                vec![1, 1],
                vec![6.0],
            ),
            (
                "model.language_model.layers.2.self_attn.k_proj.weight",
                vec![1, 1],
                vec![7.0],
            ),
            (
                "model.language_model.layers.2.mlp.gate.weight",
                vec![1, 1],
                vec![8.0],
            ),
        ];
        let safetensors = tiny_owned_multi_safetensors_bf16(&tensors);
        std::fs::write(snapshot.join("model.safetensors"), &safetensors).expect("write shard");
        std::fs::write(
            snapshot.join("model.safetensors.index.json"),
            serde_json::json!({
                "metadata": { "total_size": safetensors.len() },
                "weight_map": tensors
                    .iter()
                    .map(|(name, _, _)| {
                        (
                            (*name).to_owned(),
                            serde_json::Value::String("model.safetensors".to_owned()),
                        )
                    })
                    .collect::<serde_json::Map<_, _>>()
            })
            .to_string(),
        )
        .expect("write index");
        let store = SafeTensorShardStore::open(&snapshot).expect("store opens");

        let warmable = native_qwen_warmable_bf16_matrix_tensors(&store).expect("warmable tensors");

        assert_eq!(
            warmable
                .iter()
                .map(|tensor| tensor.name.as_str())
                .collect::<Vec<_>>(),
            vec![
                "model.language_model.embed_tokens.weight",
                "model.language_model.layers.2.self_attn.q_proj.weight",
                "model.language_model.layers.2.self_attn.k_proj.weight",
                "model.language_model.layers.2.mlp.gate.weight",
                "model.language_model.layers.2.mlp.shared_expert.down_proj.weight",
                "model.language_model.layers.10.self_attn.o_proj.weight",
                "lm_head.weight",
                "zz.unclassified.weight",
            ]
        );
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn native_qwen_cpu_backend_warmup_reports_non_metal_skip() {
        let snapshot = temp_snapshot_dir("cpu-warmup");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        let safetensors = tiny_owned_multi_safetensors_bf16(&[
            ("a.weight", vec![1, 2], vec![1.0, 2.0]),
            ("b.bias", vec![2], vec![3.0, 4.0]),
        ]);
        std::fs::write(snapshot.join("model.safetensors"), &safetensors).expect("write shard");
        std::fs::write(
            snapshot.join("model.safetensors.index.json"),
            serde_json::json!({
                "metadata": { "total_size": safetensors.len() },
                "weight_map": {
                    "a.weight": "model.safetensors",
                    "b.bias": "model.safetensors"
                }
            })
            .to_string(),
        )
        .expect("write index");
        let store = SafeTensorShardStore::open(&snapshot).expect("store opens");

        let warmup = NativeQwenMatvecBackend::Cpu
            .warm_bf16_matrix_cache(&store)
            .expect("cpu warmup reports stats");

        assert_eq!(
            warmup,
            NativeQwenMetalWarmup {
                candidates: 1,
                skipped_non_metal: 1,
                ..NativeQwenMetalWarmup::default()
            }
        );
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn native_qwen_system_default_reuses_shared_metal_state_for_same_model_budget() {
        let first = NativeQwenMatvecBackend::system_default(1_234_567, "test-shared-model");
        let second = NativeQwenMatvecBackend::system_default(1_234_567, "test-shared-model");
        let other_model = NativeQwenMatvecBackend::system_default(1_234_567, "test-other-model");

        match (&first, &second, &other_model) {
            (
                NativeQwenMatvecBackend::Metal(first),
                NativeQwenMatvecBackend::Metal(second),
                NativeQwenMatvecBackend::Metal(other_model),
            ) => {
                assert!(Arc::ptr_eq(first, second));
                assert!(!Arc::ptr_eq(first, other_model));
            }
            (
                NativeQwenMatvecBackend::Cpu,
                NativeQwenMatvecBackend::Cpu,
                NativeQwenMatvecBackend::Cpu,
            ) => {
                eprintln!("no Metal device available; skipping shared state test");
            }
            _ => panic!("Metal backend availability changed between calls"),
        }
    }

    #[test]
    fn native_max_tokens_defaults_to_configured_cache_limit() {
        assert_eq!(
            resolve_native_max_tokens(None, 4).expect("omitted max tokens uses configured cap"),
            4
        );
    }

    #[test]
    fn native_qwen_default_max_new_tokens_is_interactive_budget() {
        assert_eq!(DEFAULT_NATIVE_QWEN_MAX_NEW_TOKENS, 256);
        assert_eq!(
            resolve_native_max_tokens(None, DEFAULT_NATIVE_QWEN_MAX_NEW_TOKENS)
                .expect("omitted max tokens uses native default"),
            256
        );
        assert_eq!(
            resolve_native_max_tokens(Some(128), DEFAULT_NATIVE_QWEN_MAX_NEW_TOKENS)
                .expect("requests below native default are accepted"),
            128
        );
    }

    #[test]
    fn native_max_tokens_accepts_multi_token_decode_with_cache() {
        assert_eq!(
            resolve_native_max_tokens(Some(2), 4).expect("multi-token decode uses cache"),
            2
        );
    }

    #[test]
    fn native_max_tokens_rejects_requests_above_configured_limit() {
        let err = resolve_native_max_tokens(Some(5), 4)
            .expect_err("request above configured limit fails closed");

        assert!(matches!(err, BackendError::UnsupportedRequest(_)));
        assert!(err.to_string().contains("configured native Qwen limit"));
    }

    #[test]
    fn native_qwen_cache_capacity_preserves_prompt_and_generation_budget() {
        let capacity = native_qwen_cache_token_capacity(40, 8, 32, 64)
            .expect("prompt plus generation budget fits context");
        let spec = QwenModelSpec {
            family: llm_models::ModelFamily::Qwen,
            architecture: "Qwen3_5MoeForConditionalGeneration".to_owned(),
            model_type: "qwen3_5_moe".to_owned(),
            text_model_type: "qwen3_5_moe_text".to_owned(),
            hidden_size: 2,
            rms_norm_eps: 0.0,
            tie_word_embeddings: false,
            rope_theta: 1_000_000.0,
            partial_rotary_factor: 1.0,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 2,
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_key_head_dim: 1,
            linear_value_head_dim: 1,
            linear_conv_kernel_dim: 1,
            num_experts: 1,
            num_experts_per_tok: 1,
            moe_intermediate_size: 1,
            shared_expert_intermediate_size: 1,
            max_position_embeddings: 32,
            vocab_size: 16,
            layer_kinds: vec![llm_models::AttentionKind::FullAttention],
        };

        let caches = qwen_layer_caches_for_spec(&spec, capacity).expect("cache allocates");
        match &caches[0] {
            QwenLayerCache::Full(cache) => assert_eq!(cache.max_tokens(), 64),
            QwenLayerCache::Linear(_) => panic!("expected full-attention cache"),
        }
    }

    #[test]
    fn native_qwen_cache_capacity_rejects_context_beyond_position_limit() {
        let err = native_qwen_cache_token_capacity(60, 8, 32, 64)
            .expect_err("context beyond model position limit fails closed");

        assert!(matches!(err, BackendError::UnsupportedRequest(_)));
        assert!(
            err.to_string().contains("model context limit"),
            "error should name context limit: {err}"
        );
    }

    #[test]
    fn native_qwen_start_decode_session_prefills_full_context_with_bounded_cache() {
        let snapshot = temp_snapshot_dir("full-context-prefill");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));
        write_tiny_linear_decoder_snapshot(&snapshot);
        let backend = native_qwen_test_backend(
            &snapshot,
            "local-qwen36",
            NativeTextModelSpec::Qwen(tiny_engine_qwen_spec(
                llm_models::AttentionKind::LinearAttention,
            )),
            8,
            16,
            2,
            64,
        );

        let decode = backend
            .start_decode_session(
                &[0, 1, 0],
                8,
                &native_qwen_test_request("local-qwen36"),
                &CancellationToken::new(),
            )
            .expect("decode session starts");

        match &decode.caches[0] {
            QwenLayerCache::Linear(cache) => assert_eq!(cache.token_count(), 3),
            QwenLayerCache::Full(_) => panic!("layer 0 should be linear attention"),
        }
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn native_qwen_start_decode_session_reuses_shared_prefix_across_requests() {
        let snapshot = temp_snapshot_dir("shared-prefix-prefill");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));
        write_tiny_linear_decoder_snapshot(&snapshot);
        let backend = native_qwen_test_backend(
            &snapshot,
            "local-qwen36",
            NativeTextModelSpec::Qwen(tiny_engine_qwen_spec(
                llm_models::AttentionKind::LinearAttention,
            )),
            8,
            1,
            2,
            64,
        );
        let request = native_qwen_test_request("local-qwen36");
        let before_hits = native_prefix_metric_counter("hits");

        let first = backend
            .start_decode_session(&[0, 1], 8, &request, &CancellationToken::new())
            .expect("first decode session starts");
        drop(first);
        let second = backend
            .start_decode_session(&[0, 1, 0], 8, &request, &CancellationToken::new())
            .expect("second decode session starts");

        assert!(
            native_prefix_metric_counter("hits") > before_hits,
            "second request should hit the shared prefix cache"
        );
        match &second.caches[0] {
            QwenLayerCache::Linear(cache) => assert_eq!(cache.token_count(), 3),
            QwenLayerCache::Full(_) => panic!("layer 0 should be linear attention"),
        }

        let mut expected_caches = native_layer_caches_for_spec(
            &backend.driver.adapter.spec,
            native_qwen_cache_token_capacity(
                3,
                8,
                backend.driver.adapter.max_prefill_tokens,
                backend.driver.adapter.spec.max_position_embeddings(),
            )
            .expect("expected cache capacity"),
        )
        .expect("expected caches allocate");
        let expected_hidden = native_qwen_prefill_context_with_cache(
            &backend.driver.adapter.store,
            &backend.driver.adapter.spec,
            &[0, 1, 0],
            &mut expected_caches,
            &NativeQwenMatvecBackend::Cpu,
            1,
            &CancellationToken::new(),
        )
        .expect("fresh prefill succeeds");
        assert_close_vec(second.hidden(), &expected_hidden);
        match (&second.caches[0], &expected_caches[0]) {
            (QwenLayerCache::Linear(actual), QwenLayerCache::Linear(expected)) => {
                assert_eq!(actual.token_count(), expected.token_count());
                assert_eq!(actual.conv_window(), expected.conv_window());
                assert_eq!(actual.recurrent_state(), expected.recurrent_state());
            }
            _ => panic!("expected linear attention caches"),
        }
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn native_qwen_prefill_context_uses_sequence_cache_path_for_full_context() {
        let snapshot = temp_snapshot_dir("sequence-prefill");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        write_tiny_linear_decoder_snapshot(&snapshot);
        let spec = tiny_engine_qwen_spec(llm_models::AttentionKind::LinearAttention);
        let native_spec = NativeTextModelSpec::Qwen(spec.clone());
        let store = SafeTensorShardStore::open(&snapshot).expect("store opens");
        let mut caches = qwen_layer_caches_for_spec(&spec, 1).expect("caches allocate");

        let hidden = native_qwen_prefill_context_with_cache(
            &store,
            &native_spec,
            &[0, 1, 0],
            &mut caches,
            &NativeQwenMatvecBackend::Cpu,
            1,
            &CancellationToken::new(),
        )
        .expect("sequence prefill succeeds");

        assert_eq!(hidden.len(), 2);
        match &caches[0] {
            QwenLayerCache::Linear(cache) => assert_eq!(cache.token_count(), 3),
            QwenLayerCache::Full(_) => panic!("layer 0 should be linear attention"),
        }
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn native_qwen_prefill_context_checks_cancellation_between_chunks() {
        let snapshot = temp_snapshot_dir("sequence-prefill-cancel");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        write_tiny_linear_decoder_snapshot(&snapshot);
        let spec = tiny_engine_qwen_spec(llm_models::AttentionKind::LinearAttention);
        let native_spec = NativeTextModelSpec::Qwen(spec.clone());
        let store = SafeTensorShardStore::open(&snapshot).expect("store opens");
        let mut caches = qwen_layer_caches_for_spec(&spec, 1).expect("caches allocate");
        let cancellation = CancellationToken::new();
        let matvec = CancelAfterFirstConv {
            cancellation: cancellation.clone(),
            conv_calls: std::cell::Cell::new(0),
        };

        let err = native_qwen_prefill_context_with_cache(
            &store,
            &native_spec,
            &[0, 1, 0],
            &mut caches,
            &matvec,
            1,
            &cancellation,
        )
        .expect_err("cancelled after first chunk");

        assert!(matches!(err, BackendError::Cancelled));
        match &caches[0] {
            QwenLayerCache::Linear(cache) => assert_eq!(cache.token_count(), 1),
            QwenLayerCache::Full(_) => panic!("layer 0 should be linear attention"),
        }
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn native_qwen_backend_opens_snapshot_without_engine_manifest() {
        let snapshot = temp_snapshot_dir("no-manifest");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        copy_fixture("config.json", snapshot.join("config.json"));
        copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));
        copy_fixture(
            "model.safetensors.index.json",
            snapshot.join("model.safetensors.index.json"),
        );

        let backend =
            NativeQwenBackend::open("local-qwen36", &snapshot).expect("backend opens snapshot");
        let metadata = backend.model_metadata();

        assert_eq!(
            backend.driver.max_new_tokens,
            DEFAULT_NATIVE_QWEN_MAX_NEW_TOKENS
        );
        assert_eq!(metadata.id, "local-qwen36");
        assert_eq!(metadata.backend, "native-qwen");
        assert_eq!(metadata.snapshot_path.as_deref(), Some(snapshot.as_path()));
        assert!(metadata.manifest_digest.is_none());
        assert!(metadata.repo_id.is_none());
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn native_qwen_backend_runs_qwen3_dense_single_file_prefill() {
        let snapshot = temp_snapshot_dir("qwen3-dense-single-file");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        write_tiny_qwen3_dense_single_file_decoder_snapshot(&snapshot);
        copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));

        let mut backend =
            NativeQwenBackend::open("local-qwen3", &snapshot).expect("backend opens snapshot");
        backend.driver.adapter.top_k = 2;
        let decode = backend
            .start_decode_session(
                &[0, 1],
                4,
                &native_qwen_test_request("local-qwen3"),
                &CancellationToken::new(),
            )
            .expect("dense single-file prefill runs");
        let candidate = backend
            .next_token_from_hidden(decode.hidden(), SamplingConfig::Greedy)
            .expect("dense tied lm head can select a token");

        assert!(backend.driver.adapter.spec.is_qwen3_dense());
        assert!(candidate.token_id < 2);
        match &decode.caches[0] {
            QwenLayerCache::Full(cache) => assert_eq!(cache.token_count(), 2),
            QwenLayerCache::Linear(_) => panic!("dense Qwen3 should use full attention cache"),
        }
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn native_qwen_full_attention_prefill_keeps_context_beyond_chunk_size() {
        let snapshot = temp_snapshot_dir("qwen3-dense-long-prefill");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        write_tiny_qwen3_dense_single_file_decoder_snapshot(&snapshot);
        copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));

        let mut backend =
            NativeQwenBackend::open("local-qwen3", &snapshot).expect("backend opens snapshot");
        backend.driver.adapter.max_prefill_tokens = 1;
        let context = [0, 1].repeat(6);
        let decode = backend
            .start_decode_session(
                &context,
                4,
                &native_qwen_test_request("local-qwen3"),
                &CancellationToken::new(),
            )
            .expect("dense full-attention prefill keeps the accepted context");

        match &decode.caches[0] {
            QwenLayerCache::Full(cache) => {
                assert_eq!(cache.max_tokens(), 16);
                assert_eq!(cache.token_count(), context.len());
                assert!(cache.key(0).is_some(), "oldest prompt token must remain");
                assert!(
                    cache.key(context.len() - 1).is_some(),
                    "latest prompt token must remain"
                );
            }
            QwenLayerCache::Linear(_) => panic!("dense Qwen3 should use full attention cache"),
        }
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn native_qwen_backend_can_eagerly_materialize_indexed_shards_on_open() {
        let snapshot = temp_snapshot_dir("eager-materialize");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        copy_fixture("config.json", snapshot.join("config.json"));
        copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));
        std::fs::write(
            snapshot.join("model.safetensors.index.json"),
            serde_json::json!({
                "metadata": { "total_size": 2 },
                "weight_map": { "dummy.weight": "dummy.safetensors" }
            })
            .to_string(),
        )
        .expect("index");
        std::fs::write(
            snapshot.join("dummy.safetensors"),
            tiny_safetensors_bf16("dummy.weight", &[1], &[1.0]),
        )
        .expect("dummy shard");

        let backend = NativeQwenBackend::open_with_options(
            "local-qwen36",
            &snapshot,
            NativeQwenLoadOptions {
                eager_materialize_shards: true,
                ..NativeQwenLoadOptions::default()
            },
        )
        .expect("backend opens and materializes shards");

        assert_eq!(backend.driver.adapter.store.materialized_shard_count(), 1);
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[tokio::test]
    async fn native_qwen_generate_with_cancel_observes_pre_cancelled_token() {
        let snapshot = temp_snapshot_dir("cancelled-generate");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        copy_fixture("config.json", snapshot.join("config.json"));
        copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));
        copy_fixture(
            "model.safetensors.index.json",
            snapshot.join("model.safetensors.index.json"),
        );
        let backend =
            NativeQwenBackend::open("local-qwen36", &snapshot).expect("backend opens snapshot");
        let cancellation = CancellationToken::new();
        cancellation.cancel();

        let err = backend
            .generate_with_cancel(
                BackendRequest {
                    model: "local-qwen36".to_owned(),
                    prompt: "say hi".to_owned(),
                    max_tokens: Some(1),
                    sampling: SamplingConfig::Greedy,
                    required_tool_choice: None,
                    json_object_mode: false,
                    conversation_mode: false,
                    cache_context: BackendCacheContext::default(),
                },
                cancellation,
            )
            .await
            .expect_err("pre-cancelled generation fails before decode");

        assert!(err.to_string().contains("cancelled"));
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn native_qwen_stream_with_cancel_observes_pre_cancelled_token() {
        let snapshot = temp_snapshot_dir("cancelled-stream");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        copy_fixture("config.json", snapshot.join("config.json"));
        copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));
        copy_fixture(
            "model.safetensors.index.json",
            snapshot.join("model.safetensors.index.json"),
        );
        let backend =
            NativeQwenBackend::open("local-qwen36", &snapshot).expect("backend opens snapshot");
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let (tx, _rx) = tokio::sync::mpsc::channel(1);

        let err = backend
            .generate_blocking_stream(
                BackendRequest {
                    model: "local-qwen36".to_owned(),
                    prompt: "say hi".to_owned(),
                    max_tokens: Some(1),
                    sampling: SamplingConfig::Greedy,
                    required_tool_choice: None,
                    json_object_mode: false,
                    conversation_mode: false,
                    cache_context: BackendCacheContext::default(),
                },
                tx,
                cancellation,
            )
            .expect_err("pre-cancelled stream fails before normal EOF");

        assert!(matches!(err, BackendError::Cancelled));
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[tokio::test]
    async fn native_qwen_worker_stream_reports_join_failure_after_channel_close() {
        let (_tx, rx) = tokio::sync::mpsc::channel(1);
        let worker = tokio::task::spawn_blocking(|| panic!("stream worker panic"));
        let mut stream = native_qwen_worker_stream(rx, worker);

        let err = stream
            .next()
            .await
            .expect("join failure event")
            .expect_err("worker panic is surfaced");

        assert!(
            err.to_string()
                .contains("native Qwen streaming worker failed")
        );
        assert!(stream.next().await.is_none());
    }

    #[test]
    fn native_qwen_start_decode_session_observes_pre_cancelled_token() {
        let snapshot = temp_snapshot_dir("cancelled-start-decode");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        copy_fixture("config.json", snapshot.join("config.json"));
        copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));
        copy_fixture(
            "model.safetensors.index.json",
            snapshot.join("model.safetensors.index.json"),
        );
        let backend =
            NativeQwenBackend::open("local-qwen36", &snapshot).expect("backend opens snapshot");
        let cancellation = CancellationToken::new();
        cancellation.cancel();

        match backend.start_decode_session(
            &[0],
            1,
            &native_qwen_test_request("local-qwen36"),
            &cancellation,
        ) {
            Err(BackendError::Cancelled) => {}
            Err(err) => panic!("expected cancellation before prefill, got {err}"),
            Ok(_) => panic!("pre-cancelled decode startup should fail before prefill"),
        }
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn native_qwen_greedy_returns_top_logit_even_when_it_decodes_to_whitespace() {
        let snapshot = temp_snapshot_dir("greedy-whitespace");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));

        let norm_shape = [1_usize];
        let norm = [1.0_f32];
        let lm_head_shape = [221_usize, 1_usize];
        let mut lm_head = vec![0.0_f32; 221];
        lm_head[32] = 1.0;
        lm_head[220] = 2.0;
        let safetensors = tiny_multi_safetensors_bf16(&[
            (
                "model.language_model.norm.weight",
                &norm_shape,
                norm.as_slice(),
            ),
            ("lm_head.weight", &lm_head_shape, lm_head.as_slice()),
        ]);
        std::fs::write(snapshot.join("model.safetensors"), &safetensors)
            .expect("write greedy fixture shard");
        std::fs::write(
            snapshot.join("model.safetensors.index.json"),
            serde_json::json!({
                "metadata": { "total_size": safetensors.len() },
                "weight_map": {
                    "model.language_model.norm.weight": "model.safetensors",
                    "lm_head.weight": "model.safetensors"
                }
            })
            .to_string(),
        )
        .expect("write greedy fixture index");

        let backend = native_qwen_test_backend(
            &snapshot,
            "local-qwen36",
            NativeTextModelSpec::Qwen(QwenModelSpec {
                family: llm_models::ModelFamily::Qwen,
                architecture: "Qwen3_5MoeForConditionalGeneration".to_owned(),
                model_type: "qwen3_5_moe".to_owned(),
                text_model_type: "qwen3_5_moe_text".to_owned(),
                hidden_size: 1,
                rms_norm_eps: 0.0,
                tie_word_embeddings: false,
                rope_theta: 1_000_000.0,
                partial_rotary_factor: 1.0,
                num_hidden_layers: 0,
                num_attention_heads: 1,
                num_key_value_heads: 1,
                head_dim: 1,
                linear_num_key_heads: 1,
                linear_num_value_heads: 1,
                linear_key_head_dim: 1,
                linear_value_head_dim: 1,
                linear_conv_kernel_dim: 1,
                num_experts: 1,
                num_experts_per_tok: 1,
                moe_intermediate_size: 1,
                shared_expert_intermediate_size: 1,
                max_position_embeddings: 1,
                vocab_size: 221,
                layer_kinds: Vec::new(),
            }),
            1,
            1,
            2,
            64,
        );

        let candidate = backend
            .next_token_from_hidden(&[1.0], SamplingConfig::Greedy)
            .expect("greedy candidate");

        assert_eq!(candidate.token_id, 220);
        let decoded = backend
            .driver
            .tokenizer
            .decode(&[candidate.token_id as u32], false)
            .expect("candidate decodes");
        assert!(decoded.trim().is_empty());
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn native_stream_text_deltas_withhold_unstable_prefix_until_finish() {
        let mut deltas = NativeStreamTextDeltas::default();

        assert_eq!(deltas.observe("�".to_owned()).expect("observe"), None);
        assert_eq!(deltas.observe("é".to_owned()).expect("observe"), None);

        assert_eq!(
            deltas.finish("é".to_owned()).expect("finish"),
            Some("é".to_owned())
        );
    }

    #[test]
    fn native_stream_text_deltas_emit_stable_prefix_with_one_token_delay() {
        let mut deltas = NativeStreamTextDeltas::default();

        assert_eq!(deltas.observe("a".to_owned()).expect("observe"), None);
        assert_eq!(
            deltas.observe("ab".to_owned()).expect("observe"),
            Some("a".to_owned())
        );
        assert_eq!(
            deltas.observe("abc".to_owned()).expect("observe"),
            Some("b".to_owned())
        );
        assert_eq!(
            deltas.finish("abc".to_owned()).expect("finish"),
            Some("c".to_owned())
        );
    }

    #[test]
    fn native_stream_text_deltas_fail_closed_after_emitted_prefix_changes() {
        let mut deltas = NativeStreamTextDeltas::default();

        assert_eq!(deltas.observe("a".to_owned()).expect("observe"), None);
        assert_eq!(
            deltas.observe("ab".to_owned()).expect("observe"),
            Some("a".to_owned())
        );

        let err = deltas
            .observe("xb".to_owned())
            .expect_err("emitted prefix mismatch fails closed");
        assert!(err.to_string().contains("non-prefix"));
    }

    #[test]
    fn native_top_p_sampling_selects_full_vocab_token_from_draw() {
        let token_id = sample_token_id_with_draw(
            &[2.0, 1.0, 0.0],
            SamplingConfig::TopP {
                temperature: 1.0,
                top_p: 0.9,
            },
            0.8,
        )
        .expect("sampling succeeds");

        assert_eq!(token_id, 1);
    }
    fn copy_fixture(name: &str, destination: impl AsRef<Path>) {
        let source = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/qwen36")
            .join(name);
        std::fs::copy(&source, destination).expect("copy fixture");
    }

    fn write_tiny_qwen3_dense_single_file_decoder_snapshot(root: &Path) {
        std::fs::write(
            root.join("config.json"),
            serde_json::json!({
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "attention_bias": false,
                "hidden_act": "silu",
                "hidden_size": 2,
                "intermediate_size": 1,
                "max_position_embeddings": 16,
                "num_attention_heads": 1,
                "num_hidden_layers": 1,
                "num_key_value_heads": 1,
                "head_dim": 2,
                "rms_norm_eps": 1e-6,
                "rope_scaling": null,
                "rope_theta": 1_000_000,
                "sliding_window": null,
                "tie_word_embeddings": true,
                "use_sliding_window": false,
                "vocab_size": 2
            })
            .to_string(),
        )
        .expect("config");
        std::fs::write(
            root.join("model.safetensors"),
            tiny_multi_safetensors_bf16(&[
                ("model.embed_tokens.weight", &[2, 2], &[1.0, 0.0, 0.0, 1.0]),
                ("model.norm.weight", &[2], &[1.0, 1.0]),
                ("model.layers.0.input_layernorm.weight", &[2], &[1.0, 1.0]),
                (
                    "model.layers.0.self_attn.q_proj.weight",
                    &[2, 2],
                    &[1.0, 0.0, 0.0, 1.0],
                ),
                (
                    "model.layers.0.self_attn.k_proj.weight",
                    &[2, 2],
                    &[1.0, 0.0, 0.0, 1.0],
                ),
                (
                    "model.layers.0.self_attn.v_proj.weight",
                    &[2, 2],
                    &[1.0, 0.0, 0.0, 1.0],
                ),
                ("model.layers.0.self_attn.q_norm.weight", &[2], &[1.0, 1.0]),
                ("model.layers.0.self_attn.k_norm.weight", &[2], &[1.0, 1.0]),
                (
                    "model.layers.0.self_attn.o_proj.weight",
                    &[2, 2],
                    &[1.0, 0.0, 0.0, 1.0],
                ),
                (
                    "model.layers.0.post_attention_layernorm.weight",
                    &[2],
                    &[1.0, 1.0],
                ),
                ("model.layers.0.mlp.gate_proj.weight", &[1, 2], &[0.0, 0.0]),
                ("model.layers.0.mlp.up_proj.weight", &[1, 2], &[0.0, 0.0]),
                ("model.layers.0.mlp.down_proj.weight", &[2, 1], &[0.0, 0.0]),
            ]),
        )
        .expect("single safetensors");
    }

    fn tiny_multi_safetensors_bf16(tensors: &[(&str, &[usize], &[f32])]) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape, values) in tensors {
            let start = data.len();
            for value in *values {
                data.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
            }
            let end = data.len();
            header.insert(
                (*name).to_owned(),
                serde_json::json!({
                    "dtype": "BF16",
                    "shape": shape,
                    "data_offsets": [start, end]
                }),
            );
        }
        let header = serde_json::Value::Object(header).to_string();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&data);
        bytes
    }

    fn tiny_safetensors_bf16(name: &str, shape: &[usize], values: &[f32]) -> Vec<u8> {
        let mut data = Vec::with_capacity(values.len() * 2);
        for value in values {
            data.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
        }
        let data_len = data.len();
        let header = serde_json::json!({
            name: {
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [0, data_len]
            }
        })
        .to_string();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&data);
        bytes
    }

    fn tiny_owned_multi_safetensors_bf16(tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape, values) in tensors {
            let start = data.len();
            for value in values {
                data.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
            }
            let end = data.len();
            header.insert(
                (*name).to_owned(),
                serde_json::json!({
                    "dtype": "BF16",
                    "shape": shape,
                    "data_offsets": [start, end]
                }),
            );
        }
        let header = serde_json::Value::Object(header).to_string();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&data);
        bytes
    }

    fn write_tiny_linear_decoder_snapshot(root: &Path) {
        let tensors = vec![
            (
                "model.language_model.embed_tokens.weight",
                vec![2, 2],
                vec![1.0, 0.0, 0.0, 1.0],
            ),
            (
                "model.language_model.layers.0.input_layernorm.weight",
                vec![2],
                vec![0.0, 0.0],
            ),
            (
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                vec![4, 2],
                vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 4.0],
            ),
            (
                "model.language_model.layers.0.linear_attn.in_proj_z.weight",
                vec![2, 2],
                vec![1.0, 0.0, 0.0, 1.0],
            ),
            (
                "model.language_model.layers.0.linear_attn.in_proj_b.weight",
                vec![1, 2],
                vec![0.0, 0.0],
            ),
            (
                "model.language_model.layers.0.linear_attn.in_proj_a.weight",
                vec![1, 2],
                vec![0.0, 0.0],
            ),
            (
                "model.language_model.layers.0.linear_attn.dt_bias",
                vec![1],
                vec![0.0],
            ),
            (
                "model.language_model.layers.0.linear_attn.A_log",
                vec![1],
                vec![0.0],
            ),
            (
                "model.language_model.layers.0.linear_attn.conv1d.weight",
                vec![4, 1],
                vec![1.0, 1.0, 1.0, 1.0],
            ),
            (
                "model.language_model.layers.0.linear_attn.norm.weight",
                vec![2],
                vec![1.0, 1.0],
            ),
            (
                "model.language_model.layers.0.linear_attn.out_proj.weight",
                vec![2, 2],
                vec![1.0, 0.0, 0.0, 1.0],
            ),
            (
                "model.language_model.layers.0.post_attention_layernorm.weight",
                vec![2],
                vec![0.0, 0.0],
            ),
            (
                "model.language_model.layers.0.mlp.gate.weight",
                vec![1, 2],
                vec![0.0, 0.0],
            ),
            (
                "model.language_model.layers.0.mlp.experts.gate_up_proj",
                vec![2, 2],
                vec![0.0, 0.0, 0.0, 0.0],
            ),
            (
                "model.language_model.layers.0.mlp.experts.down_proj",
                vec![2, 1],
                vec![0.0, 0.0],
            ),
            (
                "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
                vec![1, 2],
                vec![0.0, 0.0],
            ),
            (
                "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
                vec![1, 2],
                vec![0.0, 0.0],
            ),
            (
                "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
                vec![2, 1],
                vec![0.0, 0.0],
            ),
            (
                "model.language_model.layers.0.mlp.shared_expert_gate.weight",
                vec![1, 2],
                vec![0.0, 0.0],
            ),
        ];
        let mut weight_map = serde_json::Map::new();
        for (name, _, _) in &tensors {
            weight_map.insert(
                (*name).to_owned(),
                serde_json::Value::String("model.safetensors".to_owned()),
            );
        }
        let safetensors = tiny_owned_multi_safetensors_bf16(&tensors);
        std::fs::write(snapshot_path(root, "model.safetensors"), &safetensors)
            .expect("write tiny decoder shard");
        std::fs::write(
            snapshot_path(root, "model.safetensors.index.json"),
            serde_json::json!({
                "metadata": { "total_size": safetensors.len() },
                "weight_map": serde_json::Value::Object(weight_map)
            })
            .to_string(),
        )
        .expect("write tiny decoder index");
    }

    fn snapshot_path(root: &Path, name: &str) -> PathBuf {
        root.join(name)
    }

    fn tiny_engine_qwen_spec(kind: llm_models::AttentionKind) -> QwenModelSpec {
        QwenModelSpec {
            family: llm_models::ModelFamily::Qwen,
            architecture: "Qwen3_5MoeForConditionalGeneration".to_owned(),
            model_type: "qwen3_5_moe".to_owned(),
            text_model_type: "qwen3_5_moe_text".to_owned(),
            hidden_size: 2,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: false,
            rope_theta: 1_000_000.0,
            partial_rotary_factor: 1.0,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 2,
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_key_head_dim: 1,
            linear_value_head_dim: 2,
            linear_conv_kernel_dim: 1,
            num_experts: 1,
            num_experts_per_tok: 1,
            moe_intermediate_size: 1,
            shared_expert_intermediate_size: 1,
            max_position_embeddings: 32,
            vocab_size: 2,
            layer_kinds: vec![kind],
        }
    }

    fn native_qwen_test_backend(
        snapshot: &Path,
        model_id: &str,
        spec: NativeTextModelSpec,
        max_new_tokens: u32,
        max_prefill_tokens: usize,
        top_k: usize,
        chunk_rows: usize,
    ) -> NativeQwenBackend {
        let metadata = BackendModelMetadata::new(model_id.to_owned(), "native-qwen");
        let tokenizer = HuggingFaceTokenizer::from_file(snapshot.join("tokenizer.json"))
            .expect("tokenizer loads");
        let adapter = NativeQwenAdapter {
            model_id: model_id.to_owned(),
            metadata: metadata.clone(),
            spec,
            store: SafeTensorShardStore::open(snapshot).expect("store opens"),
            matvec: NativeQwenMatvecBackend::Cpu,
            max_prefill_tokens,
            top_k,
            chunk_rows,
            prefix_cache: Arc::new(NativeQwenPrefixCache::new(
                DEFAULT_NATIVE_QWEN_PREFIX_CACHE_BYTES,
            )),
        };
        NativeQwenBackend {
            driver: NativeTextDriver::new(
                model_id.to_owned(),
                metadata,
                tokenizer,
                adapter,
                max_new_tokens,
            ),
        }
    }

    fn native_qwen_test_request(model: &str) -> BackendRequest {
        BackendRequest {
            model: model.to_owned(),
            prompt: "test".to_owned(),
            max_tokens: Some(1),
            sampling: SamplingConfig::Greedy,
            required_tool_choice: None,
            json_object_mode: false,
            conversation_mode: false,
            cache_context: BackendCacheContext::default(),
        }
    }
    fn native_qwen_test_prefix_namespace(label: &str) -> NativeQwenPrefixCacheNamespace {
        NativeQwenPrefixCacheNamespace {
            model_id: format!("model-{label}"),
            backend: "native-qwen".to_owned(),
            family: Some("qwen".to_owned()),
            loader: Some("safetensors".to_owned()),
            quantization: Some("bf16".to_owned()),
            repo_id: Some("local/test".to_owned()),
            resolved_commit: Some("0123456789abcdef0123456789abcdef01234567".to_owned()),
            profile: Some("qwen-test".to_owned()),
            manifest_digest: Some(format!("digest-{label}")),
            prompt_template: QwenFamilyAdapter.cache_template_id().to_owned(),
            tool_schema: Some("tool-schema-v1".to_owned()),
            request_mode: "conversation=true,json_object=false,required_tool=None".to_owned(),
            sampling: "greedy".to_owned(),
            cache_layout_version: NATIVE_QWEN_PREFIX_CACHE_LAYOUT_VERSION,
            cache_tokens: 8,
            max_prefill_tokens: 8,
        }
    }

    fn native_prefix_metric_counter(name: &str) -> u64 {
        native_qwen_prefix_cache_metrics().snapshot()[name]
            .as_u64()
            .unwrap_or_else(|| panic!("prefix metric `{name}` is an unsigned integer"))
    }

    fn assert_close_vec(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "value {index} differed: actual={actual}, expected={expected}"
            );
        }
    }

    struct CancelAfterFirstConv {
        cancellation: CancellationToken,
        conv_calls: std::cell::Cell<usize>,
    }

    impl QwenMatvecBackend for CancelAfterFirstConv {
        fn linear_attention_conv1d_silu_f32(
            &self,
            window: &[f32],
            weights: &[f32],
            conv_dim: usize,
            kernel_size: usize,
        ) -> Result<Vec<f32>, MathError> {
            self.conv_calls.set(self.conv_calls.get() + 1);
            if self.conv_calls.get() == 1 {
                self.cancellation.cancel();
            }
            CpuQwenMatvecBackend.linear_attention_conv1d_silu_f32(
                window,
                weights,
                conv_dim,
                kernel_size,
            )
        }
    }
    fn temp_snapshot_dir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!("llm-engine-{label}-{}", std::process::id()))
    }
}
