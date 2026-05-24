use crate::sync_ext::FailPoisonedMutex;
use serde_json::{Value, json};
use std::{
    collections::{HashMap, HashSet},
    sync::{Mutex, OnceLock},
};

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
    skipped_syncs: u64,
    evictions: u64,
    stage_rebuilds: u64,
    bytes_uploaded: u64,
    bytes_evicted: u64,
    stage_bytes_copied: u64,
    resident_bytes: u64,
    resident_buffers: u64,
    stage_resident_bytes: u64,
    stage_resident_buffers: u64,
    f32_bytes_uploaded: u64,
    f16_bytes_uploaded: u64,
    int8_bytes_uploaded: u64,
    f32_bytes_evicted: u64,
    f16_bytes_evicted: u64,
    int8_bytes_evicted: u64,
    f32_resident_bytes: u64,
    f16_resident_bytes: u64,
    int8_resident_bytes: u64,
    f32_resident_buffers: u64,
    f16_resident_buffers: u64,
    int8_resident_buffers: u64,
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
            .lock_or_panic("Metal fallback warning")
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
            .lock_or_panic("Metal BF16 matrix cache metrics");
        cache.hits += 1;
    }

    pub(crate) fn record_bf16_matrix_cache_miss(&self) {
        let mut cache = self
            .bf16_matrix_cache
            .lock_or_panic("Metal BF16 matrix cache metrics");
        cache.misses += 1;
    }

    pub(crate) fn record_bf16_matrix_cache_upload(&self, byte_len: u64) {
        let mut cache = self
            .bf16_matrix_cache
            .lock_or_panic("Metal BF16 matrix cache metrics");
        cache.uploads += 1;
        cache.bytes_uploaded += byte_len;
    }

    pub(crate) fn record_bf16_matrix_cache_eviction(&self, count: u64, byte_len: u64) {
        let mut cache = self
            .bf16_matrix_cache
            .lock_or_panic("Metal BF16 matrix cache metrics");
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
            .lock_or_panic("Metal BF16 matrix cache metrics");
        cache.resident_bytes = resident_bytes;
        cache.resident_buffers = resident_buffers;
        cache.budget_bytes = budget_bytes;
    }

    pub(crate) fn record_kv_cache_allocation(&self, byte_len: u64) {
        self.update_cache_counter(CacheMetricKind::Kv, |cache| {
            cache.allocations += 1;
            cache.record_upload(CachePrecisionMetric::F16, byte_len);
        });
    }

    pub(crate) fn record_int8_kv_cache_allocation(&self, byte_len: u64) {
        self.update_cache_counter(CacheMetricKind::Kv, |cache| {
            cache.allocations += 1;
            cache.record_upload(CachePrecisionMetric::Int8, byte_len);
        });
    }

    pub(crate) fn record_kv_cache_sync(&self, byte_len: u64) {
        self.update_cache_counter(CacheMetricKind::Kv, |cache| {
            cache.syncs += 1;
            cache.record_upload(CachePrecisionMetric::F16, byte_len);
        });
    }

    pub(crate) fn record_int8_kv_cache_sync(&self, byte_len: u64) {
        self.update_cache_counter(CacheMetricKind::Kv, |cache| {
            cache.syncs += 1;
            cache.record_upload(CachePrecisionMetric::Int8, byte_len);
        });
    }

    pub(crate) fn record_kv_cache_skipped_syncs(&self, count: u64) {
        self.update_cache_counter(CacheMetricKind::Kv, |cache| {
            cache.skipped_syncs = cache.skipped_syncs.saturating_add(count);
        });
    }

    pub(crate) fn record_kv_cache_eviction(&self, count: u64, byte_len: u64) {
        self.update_cache_counter(CacheMetricKind::Kv, |cache| {
            cache.evictions += count;
            cache.record_eviction(CachePrecisionMetric::F16, byte_len);
        });
    }

    pub(crate) fn record_int8_kv_cache_eviction(&self, count: u64, byte_len: u64) {
        self.update_cache_counter(CacheMetricKind::Kv, |cache| {
            cache.evictions += count;
            cache.record_eviction(CachePrecisionMetric::Int8, byte_len);
        });
    }

    pub(crate) fn record_kv_cache_residency(&self, resident_bytes: u64, resident_buffers: u64) {
        self.update_cache_counter(CacheMetricKind::Kv, |cache| {
            cache.record_residency(CachePrecisionMetric::F16, resident_bytes, resident_buffers);
        });
    }

    pub(crate) fn record_int8_kv_cache_residency(
        &self,
        resident_bytes: u64,
        resident_buffers: u64,
    ) {
        self.update_cache_counter(CacheMetricKind::Kv, |cache| {
            cache.record_residency(CachePrecisionMetric::Int8, resident_bytes, resident_buffers);
        });
    }

    pub(crate) fn record_kv_cache_stage_rebuild(&self, byte_len: u64) {
        self.update_cache_counter(CacheMetricKind::Kv, |cache| {
            cache.stage_rebuilds = cache.stage_rebuilds.saturating_add(1);
            cache.stage_bytes_copied = cache.stage_bytes_copied.saturating_add(byte_len);
        });
    }

    pub(crate) fn record_kv_cache_stage_sync(&self, byte_len: u64) {
        self.update_cache_counter(CacheMetricKind::Kv, |cache| {
            cache.stage_bytes_copied = cache.stage_bytes_copied.saturating_add(byte_len);
        });
    }

    pub(crate) fn record_kv_cache_stage_residency(
        &self,
        resident_bytes: u64,
        resident_buffers: u64,
    ) {
        self.update_cache_counter(CacheMetricKind::Kv, |cache| {
            cache.stage_resident_bytes = resident_bytes;
            cache.stage_resident_buffers = resident_buffers;
            cache.recompute_total_residency();
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
        let counters = self.counters.lock_or_panic("Metal metrics");
        let bf16_matrix_cache = *self
            .bf16_matrix_cache
            .lock_or_panic("Metal BF16 matrix cache metrics");
        let kv_cache = *self.kv_cache.lock_or_panic("Metal KV cache metrics");
        let linear_cache = *self
            .linear_cache
            .lock_or_panic("Metal linear cache metrics");
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
        let mut cache = cache.lock_or_panic("Metal resident cache metrics");
        update(&mut cache);
    }

    fn update_counter(&self, kernel: &'static str, update: impl FnOnce(&mut MetalKernelCounters)) {
        let mut counters = self.counters.lock_or_panic("Metal metrics");
        update(counters.entry(kernel).or_default());
    }
}

#[derive(Debug, Clone, Copy)]
enum CacheMetricKind {
    Kv,
    Linear,
}

#[derive(Debug, Clone, Copy)]
enum CachePrecisionMetric {
    F16,
    Int8,
}

impl MetalCacheCounters {
    fn record_upload(&mut self, precision: CachePrecisionMetric, byte_len: u64) {
        self.bytes_uploaded = self.bytes_uploaded.saturating_add(byte_len);
        match precision {
            CachePrecisionMetric::F16 => {
                self.f16_bytes_uploaded = self.f16_bytes_uploaded.saturating_add(byte_len);
            }
            CachePrecisionMetric::Int8 => {
                self.int8_bytes_uploaded = self.int8_bytes_uploaded.saturating_add(byte_len);
            }
        }
    }

    fn record_eviction(&mut self, precision: CachePrecisionMetric, byte_len: u64) {
        self.bytes_evicted = self.bytes_evicted.saturating_add(byte_len);
        match precision {
            CachePrecisionMetric::F16 => {
                self.f16_bytes_evicted = self.f16_bytes_evicted.saturating_add(byte_len);
            }
            CachePrecisionMetric::Int8 => {
                self.int8_bytes_evicted = self.int8_bytes_evicted.saturating_add(byte_len);
            }
        }
    }

    fn record_residency(
        &mut self,
        precision: CachePrecisionMetric,
        resident_bytes: u64,
        resident_buffers: u64,
    ) {
        match precision {
            CachePrecisionMetric::F16 => {
                self.f16_resident_bytes = resident_bytes;
                self.f16_resident_buffers = resident_buffers;
            }
            CachePrecisionMetric::Int8 => {
                self.int8_resident_bytes = resident_bytes;
                self.int8_resident_buffers = resident_buffers;
            }
        }
        self.resident_bytes = self
            .f32_resident_bytes
            .saturating_add(self.f16_resident_bytes)
            .saturating_add(self.int8_resident_bytes)
            .saturating_add(self.stage_resident_bytes);
        self.resident_buffers = self
            .f32_resident_buffers
            .saturating_add(self.f16_resident_buffers)
            .saturating_add(self.int8_resident_buffers)
            .saturating_add(self.stage_resident_buffers);
    }

    fn recompute_total_residency(&mut self) {
        self.resident_bytes = self
            .f32_resident_bytes
            .saturating_add(self.f16_resident_bytes)
            .saturating_add(self.int8_resident_bytes)
            .saturating_add(self.stage_resident_bytes);
        self.resident_buffers = self
            .f32_resident_buffers
            .saturating_add(self.f16_resident_buffers)
            .saturating_add(self.int8_resident_buffers)
            .saturating_add(self.stage_resident_buffers);
    }
}

fn cache_counters_json(counters: MetalCacheCounters) -> Value {
    json!({
        "allocations": counters.allocations,
        "syncs": counters.syncs,
        "skipped_syncs": counters.skipped_syncs,
        "evictions": counters.evictions,
        "stage_rebuilds": counters.stage_rebuilds,
        "bytes_uploaded": counters.bytes_uploaded,
        "bytes_evicted": counters.bytes_evicted,
        "stage_bytes_copied": counters.stage_bytes_copied,
        "resident_bytes": counters.resident_bytes,
        "resident_buffers": counters.resident_buffers,
        "stage_resident_bytes": counters.stage_resident_bytes,
        "stage_resident_buffers": counters.stage_resident_buffers,
        "f32_bytes_uploaded": counters.f32_bytes_uploaded,
        "f16_bytes_uploaded": counters.f16_bytes_uploaded,
        "int8_bytes_uploaded": counters.int8_bytes_uploaded,
        "f32_bytes_evicted": counters.f32_bytes_evicted,
        "f16_bytes_evicted": counters.f16_bytes_evicted,
        "int8_bytes_evicted": counters.int8_bytes_evicted,
        "f32_resident_bytes": counters.f32_resident_bytes,
        "f16_resident_bytes": counters.f16_resident_bytes,
        "int8_resident_bytes": counters.int8_resident_bytes,
        "f32_resident_buffers": counters.f32_resident_buffers,
        "f16_resident_buffers": counters.f16_resident_buffers,
        "int8_resident_buffers": counters.int8_resident_buffers,
    })
}

pub(crate) fn native_text_metal_metrics() -> &'static MetalBackendMetrics {
    static METRICS: OnceLock<MetalBackendMetrics> = OnceLock::new();
    METRICS.get_or_init(MetalBackendMetrics::default)
}

pub(crate) fn native_text_metal_metrics_snapshot() -> Value {
    native_text_metal_metrics().snapshot()
}

#[cfg(test)]
mod tests {
    use super::*;

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
        metrics.record_kv_cache_skipped_syncs(3);
        metrics.record_kv_cache_residency(16, 2);
        metrics.record_int8_kv_cache_allocation(10);
        metrics.record_int8_kv_cache_sync(6);
        metrics.record_int8_kv_cache_residency(12, 4);
        let active_snapshot = metrics.snapshot();
        let active_kv = &active_snapshot["kv_cache"];
        assert_eq!(active_kv["bytes_uploaded"], 40);
        assert_eq!(active_kv["f32_bytes_uploaded"], 0);
        assert_eq!(active_kv["f16_bytes_uploaded"], 24);
        assert_eq!(active_kv["int8_bytes_uploaded"], 16);
        assert_eq!(active_kv["resident_bytes"], 28);
        assert_eq!(active_kv["f32_resident_bytes"], 0);
        assert_eq!(active_kv["f16_resident_bytes"], 16);
        assert_eq!(active_kv["int8_resident_bytes"], 12);
        assert_eq!(active_kv["resident_buffers"], 6);
        assert_eq!(active_kv["f16_resident_buffers"], 2);
        assert_eq!(active_kv["int8_resident_buffers"], 4);

        metrics.record_kv_cache_eviction(2, 16);
        metrics.record_int8_kv_cache_eviction(4, 12);
        metrics.record_kv_cache_residency(0, 0);
        metrics.record_int8_kv_cache_residency(0, 0);
        metrics.record_linear_cache_allocation(12);
        metrics.record_linear_cache_sync(4);
        metrics.record_linear_cache_residency(12, 1);
        metrics.record_linear_cache_eviction(1, 12);
        metrics.record_linear_cache_residency(0, 0);

        let snapshot = metrics.snapshot();
        let kv = &snapshot["kv_cache"];
        assert_eq!(kv["allocations"], 2);
        assert_eq!(kv["syncs"], 2);
        assert_eq!(kv["skipped_syncs"], 3);
        assert_eq!(kv["evictions"], 6);
        assert_eq!(kv["bytes_uploaded"], 40);
        assert_eq!(kv["bytes_evicted"], 28);
        assert_eq!(kv["f16_bytes_evicted"], 16);
        assert_eq!(kv["int8_bytes_evicted"], 12);
        assert_eq!(kv["resident_bytes"], 0);
        assert_eq!(kv["resident_buffers"], 0);
        assert_eq!(kv["f16_resident_bytes"], 0);
        assert_eq!(kv["int8_resident_bytes"], 0);
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
    fn metal_metrics_snapshot_exposes_staged_kv_cache_residency() {
        let snapshot = MetalBackendMetrics::default().snapshot();
        let kv_cache = &snapshot["kv_cache"];

        assert!(kv_cache["stage_rebuilds"].is_number());
        assert!(kv_cache["stage_bytes_copied"].is_number());
        assert!(kv_cache["stage_resident_bytes"].is_number());
        assert!(kv_cache["stage_resident_buffers"].is_number());
    }
}
