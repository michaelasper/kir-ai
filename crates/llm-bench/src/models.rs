use crate::cli::{parse_u64_flag, parse_usize_flag};
use crate::{
    CACHE_LAYOUT, CaseRun, cache_lookup_result, counter_delta, detect_cpu_name, flag_value,
    gauge_delta, hit_rate, time_to_first_token_ms, uncached_tokens,
};
use llm_models::{ModelFamilyAdapter, QwenFamilyAdapter};
use llm_server::EngineOptions;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub(crate) struct BenchLaneConfig {
    pub(crate) name: String,
    pub(crate) endpoint: Option<String>,
    pub(crate) model_id: String,
    pub(crate) snapshot_path: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
pub(crate) struct BenchReport {
    pub(crate) gate: &'static str,
    pub(crate) status: String,
    pub(crate) generated_at_unix_ms: u128,
    pub(crate) trace_output_path: Option<String>,
    pub(crate) model: ModelIdentityReport,
    pub(crate) hardware: HardwareReport,
    pub(crate) cache_policy: CachePolicyReport,
    pub(crate) run_controls: BenchRunControlsReport,
    pub(crate) scheduler: SchedulerSettingsReport,
    pub(crate) baseline: BaselineReport,
    pub(crate) profiles: Vec<BenchProfileReport>,
    pub(crate) lanes: Vec<BenchLaneReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) failure_classification: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) comparison: Option<BenchLaneComparisonReport>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub(crate) struct BenchRunControlsReport {
    pub(crate) warmup_count: u32,
    pub(crate) repetitions: u32,
    pub(crate) timeout_ms: u64,
    pub(crate) connect_timeout_ms: u64,
    pub(crate) max_tokens: u32,
    pub(crate) latency_regression_threshold: f64,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct SchedulerSettingsReport {
    pub(crate) source: &'static str,
    pub(crate) concurrency_limit: usize,
    pub(crate) queue_limit: usize,
    pub(crate) queue_timeout_ms: u64,
    pub(crate) prefill_threshold_chars: usize,
    pub(crate) prefill_burst: usize,
}

impl SchedulerSettingsReport {
    pub(crate) fn from_args(args: &[String]) -> anyhow::Result<Self> {
        let defaults = EngineOptions::default();
        let source = if [
            "--scheduler-concurrency",
            "--scheduler-queue-limit",
            "--scheduler-queue-timeout-ms",
            "--scheduler-prefill-threshold-chars",
            "--scheduler-prefill-burst",
        ]
        .iter()
        .any(|flag| flag_value(args, flag).is_some())
        {
            "serve_defaults_with_bench_cli_overrides"
        } else {
            "serve_defaults"
        };
        Ok(Self {
            source,
            concurrency_limit: parse_usize_flag(
                args,
                "--scheduler-concurrency",
                defaults.concurrency_limit.max(1),
            )?,
            queue_limit: parse_usize_flag(
                args,
                "--scheduler-queue-limit",
                defaults.scheduler_queue_limit,
            )?,
            queue_timeout_ms: parse_u64_flag(
                args,
                "--scheduler-queue-timeout-ms",
                defaults
                    .scheduler_queue_timeout
                    .unwrap_or_default()
                    .as_millis()
                    .try_into()
                    .unwrap_or(u64::MAX),
            )?,
            prefill_threshold_chars: parse_usize_flag(
                args,
                "--scheduler-prefill-threshold-chars",
                defaults.scheduler_prefill_threshold_chars,
            )?,
            prefill_burst: parse_usize_flag(
                args,
                "--scheduler-prefill-burst",
                defaults.scheduler_prefill_burst.max(1),
            )?,
        })
    }
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct BenchLaneReport {
    pub(crate) name: String,
    pub(crate) status: String,
    pub(crate) model: ModelIdentityReport,
    pub(crate) profiles: Vec<BenchProfileReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cache_metrics: Option<BenchCacheMetricsReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) admin_metrics: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) admin_metrics_error: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct BenchLaneComparisonReport {
    pub(crate) status: String,
    pub(crate) artifact_identity_match: bool,
    pub(crate) cases: Vec<BenchLaneCaseComparisonReport>,
}

#[derive(Debug, Serialize)]
pub(crate) struct BenchLaneCaseComparisonReport {
    pub(crate) profile: &'static str,
    pub(crate) case: &'static str,
    pub(crate) lanes: Vec<BenchLaneCaseMetricReport>,
    pub(crate) fastest_lane: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct BenchLaneCaseMetricReport {
    pub(crate) lane: String,
    pub(crate) status: String,
    pub(crate) latency_ms: Option<u128>,
    #[serde(flatten)]
    pub(crate) stream_timing: StreamTimingReport,
    pub(crate) tokens_per_second: Option<f64>,
    pub(crate) prompt_tokens: Option<u64>,
    pub(crate) completion_tokens: Option<u64>,
    pub(crate) total_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cached_tokens_status: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cached_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cache_memory: Option<BenchLaneCacheMemoryReport>,
    pub(crate) classification: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct BenchLaneCacheMemoryReport {
    pub(crate) bytes_uploaded: u64,
    pub(crate) bytes_uploaded_delta_vs_first_lane: i64,
    pub(crate) resident_bytes: u64,
    pub(crate) resident_bytes_delta_vs_first_lane: i64,
    pub(crate) f32_bytes_uploaded: u64,
    pub(crate) f16_bytes_uploaded: u64,
    pub(crate) int8_bytes_uploaded: u64,
    pub(crate) f32_resident_bytes: u64,
    pub(crate) f16_resident_bytes: u64,
    pub(crate) int8_resident_bytes: u64,
}

impl BenchLaneCacheMemoryReport {
    pub(crate) fn from_kv_cache(
        cache: &ResidentCacheMetricsReport,
        first_lane: Option<&ResidentCacheMetricsReport>,
    ) -> Self {
        let first_lane = first_lane.unwrap_or(cache);
        Self {
            bytes_uploaded: cache.bytes_uploaded,
            bytes_uploaded_delta_vs_first_lane: gauge_delta(
                first_lane.bytes_uploaded,
                cache.bytes_uploaded,
            ),
            resident_bytes: cache.resident_bytes,
            resident_bytes_delta_vs_first_lane: gauge_delta(
                first_lane.resident_bytes,
                cache.resident_bytes,
            ),
            f32_bytes_uploaded: cache.f32_bytes_uploaded,
            f16_bytes_uploaded: cache.f16_bytes_uploaded,
            int8_bytes_uploaded: cache.int8_bytes_uploaded,
            f32_resident_bytes: cache.f32_resident_bytes,
            f16_resident_bytes: cache.f16_resident_bytes,
            int8_resident_bytes: cache.int8_resident_bytes,
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub(crate) struct BenchCacheMetricsReport {
    pub(crate) prefix_cache: PrefixCacheMetricsReport,
    pub(crate) weight_cache: WeightCacheMetricsReport,
    pub(crate) kv_cache: ResidentCacheMetricsReport,
    pub(crate) linear_attention_cache: ResidentCacheMetricsReport,
    pub(crate) readiness: CacheReadinessReport,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub(crate) struct PrefixCacheMetricsReport {
    pub(crate) hits: u64,
    pub(crate) misses: u64,
    pub(crate) hit_rate: Option<f64>,
    pub(crate) stores: u64,
    pub(crate) evictions: u64,
    pub(crate) rejected: u64,
    pub(crate) reused_tokens: u64,
    pub(crate) hit_tokens: u64,
    pub(crate) miss_tokens: u64,
    pub(crate) avoided_prefill_tokens: u64,
    pub(crate) resident_bytes: u64,
    pub(crate) resident_entries: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub(crate) struct WeightCacheMetricsReport {
    pub(crate) hits: u64,
    pub(crate) misses: u64,
    pub(crate) hit_rate: Option<f64>,
    pub(crate) uploads: u64,
    pub(crate) evictions: u64,
    pub(crate) bytes_uploaded: u64,
    pub(crate) bytes_evicted: u64,
    pub(crate) resident_bytes: u64,
    pub(crate) resident_buffers: u64,
    pub(crate) budget_bytes: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub(crate) struct ResidentCacheMetricsReport {
    pub(crate) allocations: u64,
    pub(crate) syncs: u64,
    pub(crate) evictions: u64,
    pub(crate) bytes_uploaded: u64,
    pub(crate) bytes_evicted: u64,
    pub(crate) resident_bytes: u64,
    pub(crate) resident_buffers: u64,
    pub(crate) f32_bytes_uploaded: u64,
    pub(crate) f16_bytes_uploaded: u64,
    pub(crate) int8_bytes_uploaded: u64,
    pub(crate) f32_resident_bytes: u64,
    pub(crate) f16_resident_bytes: u64,
    pub(crate) int8_resident_bytes: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub(crate) struct CacheReadinessReport {
    pub(crate) status: &'static str,
    pub(crate) observed_signals: Vec<&'static str>,
    pub(crate) missing_signals: Vec<&'static str>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct BenchProfileReport {
    pub(crate) name: &'static str,
    pub(crate) target_tokens: usize,
    pub(crate) release_blocking: bool,
    pub(crate) status: String,
    pub(crate) cases: Vec<BenchCaseReport>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct BenchCaseReport {
    pub(crate) name: &'static str,
    pub(crate) mode: &'static str,
    pub(crate) target_tokens: usize,
    pub(crate) stream: bool,
    pub(crate) response_contract: &'static str,
    pub(crate) request_count: usize,
    pub(crate) marker: String,
    pub(crate) prompt_identity: PromptIdentityReport,
    pub(crate) status: String,
    pub(crate) classification: String,
    pub(crate) prefill: BenchPrefillReport,
    pub(crate) decode: BenchDecodeReport,
    pub(crate) cache: BenchCacheBehaviorReport,
    pub(crate) summary: BenchCaseSummaryReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) planned_prompt_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) latency_ms: Option<u128>,
    #[serde(flatten)]
    pub(crate) stream_timing: StreamTimingReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tokens_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) prompt_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) completion_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) total_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cached_tokens_status: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cached_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) http_status: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) admin_metrics: Option<BenchCaseAdminMetricsReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) baseline: Option<BaselineComparisonReport>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct BenchCaseAdminMetricsReport {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) prefix_cache: Option<BenchCasePrefixCacheMetricsReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) error: Option<String>,
}

impl BenchCaseAdminMetricsReport {
    pub(crate) fn from_prefix_cache_snapshots(
        before: PrefixCacheMetricsReport,
        after: PrefixCacheMetricsReport,
    ) -> Self {
        Self {
            prefix_cache: Some(BenchCasePrefixCacheMetricsReport {
                delta: PrefixCacheMetricsDeltaReport::between(&before, &after),
                before,
                after,
            }),
            error: None,
        }
    }

    pub(crate) fn error(error: String) -> Self {
        Self {
            prefix_cache: None,
            error: Some(error),
        }
    }

    pub(crate) fn prefix_cache_delta(&self) -> Option<&PrefixCacheMetricsDeltaReport> {
        self.prefix_cache.as_ref().map(|metrics| &metrics.delta)
    }
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct BenchCasePrefixCacheMetricsReport {
    pub(crate) before: PrefixCacheMetricsReport,
    pub(crate) after: PrefixCacheMetricsReport,
    pub(crate) delta: PrefixCacheMetricsDeltaReport,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct PrefixCacheMetricsDeltaReport {
    pub(crate) hits: u64,
    pub(crate) misses: u64,
    pub(crate) hit_rate: Option<f64>,
    pub(crate) stores: u64,
    pub(crate) evictions: u64,
    pub(crate) rejected: u64,
    pub(crate) reused_tokens: u64,
    pub(crate) hit_tokens: u64,
    pub(crate) miss_tokens: u64,
    pub(crate) avoided_prefill_tokens: u64,
    pub(crate) resident_bytes: i64,
    pub(crate) resident_entries: i64,
}

impl PrefixCacheMetricsDeltaReport {
    pub(crate) fn between(
        before: &PrefixCacheMetricsReport,
        after: &PrefixCacheMetricsReport,
    ) -> Self {
        let hits = counter_delta(before.hits, after.hits);
        let misses = counter_delta(before.misses, after.misses);
        Self {
            hits,
            misses,
            hit_rate: hit_rate(hits, misses),
            stores: counter_delta(before.stores, after.stores),
            evictions: counter_delta(before.evictions, after.evictions),
            rejected: counter_delta(before.rejected, after.rejected),
            reused_tokens: counter_delta(before.reused_tokens, after.reused_tokens),
            hit_tokens: counter_delta(before.hit_tokens, after.hit_tokens),
            miss_tokens: counter_delta(before.miss_tokens, after.miss_tokens),
            avoided_prefill_tokens: counter_delta(
                before.avoided_prefill_tokens,
                after.avoided_prefill_tokens,
            ),
            resident_bytes: gauge_delta(before.resident_bytes, after.resident_bytes),
            resident_entries: gauge_delta(before.resident_entries, after.resident_entries),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct PromptIdentityReport {
    pub(crate) profile: &'static str,
    pub(crate) case: &'static str,
    pub(crate) context_tokens: usize,
    pub(crate) marker: String,
    pub(crate) prompt_hash: String,
    pub(crate) prompt_hash_source: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct BenchPrefillReport {
    pub(crate) planned_prompt_tokens: Option<usize>,
    pub(crate) prompt_tokens: Option<u64>,
    pub(crate) cached_tokens: Option<u64>,
    pub(crate) uncached_tokens: Option<u64>,
    pub(crate) time_to_first_token_ms: Option<u128>,
}

impl BenchPrefillReport {
    pub(crate) fn planned() -> Self {
        Self {
            planned_prompt_tokens: None,
            prompt_tokens: None,
            cached_tokens: None,
            uncached_tokens: None,
            time_to_first_token_ms: None,
        }
    }

    pub(crate) fn from_run(run: &CaseRun) -> Self {
        Self {
            planned_prompt_tokens: Some(run.planned_prompt_tokens),
            prompt_tokens: run.prompt_tokens,
            cached_tokens: run.cached_tokens,
            uncached_tokens: uncached_tokens(run.prompt_tokens, run.cached_tokens),
            time_to_first_token_ms: time_to_first_token_ms(run.stream_timing),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct BenchDecodeReport {
    pub(crate) max_tokens: u32,
    pub(crate) completion_tokens: Option<u64>,
    pub(crate) total_latency_ms: Option<u128>,
    pub(crate) time_to_first_token_ms: Option<u128>,
    pub(crate) tokens_per_second: Option<f64>,
}

impl BenchDecodeReport {
    pub(crate) fn planned(max_tokens: u32) -> Self {
        Self {
            max_tokens,
            completion_tokens: None,
            total_latency_ms: None,
            time_to_first_token_ms: None,
            tokens_per_second: None,
        }
    }

    pub(crate) fn from_run(max_tokens: u32, run: &CaseRun) -> Self {
        Self {
            max_tokens,
            completion_tokens: run.completion_tokens,
            total_latency_ms: run.latency_ms,
            time_to_first_token_ms: time_to_first_token_ms(run.stream_timing),
            tokens_per_second: run.tokens_per_second,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct BenchCacheBehaviorReport {
    pub(crate) cached_tokens_status: Option<&'static str>,
    pub(crate) cached_tokens: Option<u64>,
    pub(crate) reused_tokens: Option<u64>,
    pub(crate) lookup_result: Option<&'static str>,
}

impl BenchCacheBehaviorReport {
    pub(crate) fn planned() -> Self {
        Self {
            cached_tokens_status: None,
            cached_tokens: None,
            reused_tokens: None,
            lookup_result: None,
        }
    }

    pub(crate) fn from_run(run: &CaseRun) -> Self {
        Self {
            cached_tokens_status: run.cached_tokens_status,
            cached_tokens: run.cached_tokens,
            reused_tokens: run.cached_tokens,
            lookup_result: cache_lookup_result(run.cached_tokens_status, run.cached_tokens),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct BenchCaseSummaryReport {
    pub(crate) sample_count: usize,
    pub(crate) latency_ms_p50: Option<u128>,
    pub(crate) latency_ms_p95: Option<u128>,
    pub(crate) tokens_per_second_p50: Option<f64>,
    pub(crate) tokens_per_second_p95: Option<f64>,
    pub(crate) ttft_ms_p50: Option<u128>,
    pub(crate) ttft_ms_p95: Option<u128>,
}

impl BenchCaseSummaryReport {
    pub(crate) fn planned() -> Self {
        Self {
            sample_count: 0,
            latency_ms_p50: None,
            latency_ms_p95: None,
            tokens_per_second_p50: None,
            tokens_per_second_p95: None,
            ttft_ms_p50: None,
            ttft_ms_p95: None,
        }
    }

    pub(crate) fn from_run(run: &CaseRun) -> Self {
        let ttft_ms = time_to_first_token_ms(run.stream_timing);
        let sample_count = usize::from(
            run.latency_ms.is_some() || run.tokens_per_second.is_some() || ttft_ms.is_some(),
        );
        Self {
            sample_count,
            latency_ms_p50: run.latency_ms,
            latency_ms_p95: run.latency_ms,
            tokens_per_second_p50: run.tokens_per_second,
            tokens_per_second_p95: run.tokens_per_second,
            ttft_ms_p50: ttft_ms,
            ttft_ms_p95: ttft_ms,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ModelIdentityReport {
    pub(crate) id: String,
    pub(crate) endpoint: Option<String>,
    pub(crate) snapshot_path: Option<String>,
    pub(crate) repo_id: Option<String>,
    pub(crate) requested_revision: Option<String>,
    pub(crate) resolved_commit: Option<String>,
    pub(crate) profile: Option<String>,
    pub(crate) family: Option<String>,
    pub(crate) loader: Option<String>,
    pub(crate) quantization: Option<String>,
    pub(crate) manifest_digest: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct HardwareReport {
    pub(crate) os: String,
    pub(crate) arch: String,
    pub(crate) cpu: Option<String>,
}

impl HardwareReport {
    pub(crate) fn detect() -> Self {
        Self {
            os: std::env::consts::OS.to_owned(),
            arch: std::env::consts::ARCH.to_owned(),
            cpu: detect_cpu_name(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct CachePolicyReport {
    pub(crate) cache_layout: &'static str,
    pub(crate) prompt_template: &'static str,
    pub(crate) namespace_fields: Vec<&'static str>,
    pub(crate) benchmark_metrics: Vec<&'static str>,
    pub(crate) env: BTreeMap<String, String>,
}

impl CachePolicyReport {
    pub(crate) fn from_env() -> Self {
        let env = [
            "LLM_MODEL_HOME",
            "LLM_ENGINE_PREFIX_CACHE_BYTES",
            "LLM_ENGINE_NATIVE_CACHE_BYTES",
            "LLM_ENGINE_METAL_WEIGHT_CACHE_BYTES",
        ]
        .into_iter()
        .filter_map(|key| std::env::var(key).ok().map(|value| (key.to_owned(), value)))
        .collect();
        Self {
            cache_layout: CACHE_LAYOUT,
            prompt_template: QwenFamilyAdapter.cache_template_id(),
            namespace_fields: vec![
                "model_id",
                "backend",
                "family",
                "quantization",
                "repo_id",
                "resolved_commit",
                "profile",
                "prompt_cache_key",
                "tool_schema",
                "request_mode",
                "cache_layout_version",
                "cache_capacity_bucket",
                "max_prefill_tokens",
            ],
            benchmark_metrics: vec![
                "prefix_cache_hit_rate",
                "prefix_cache_hit_tokens",
                "prefix_cache_miss_tokens",
                "prefix_cache_residency",
                "weight_cache_hit_rate",
                "weight_cache_residency",
                "kv_cache_residency",
                "kv_cache_precision_residency",
                "kv_cache_precision_uploads",
                "linear_attention_cache_residency",
                "eviction_churn",
            ],
            env,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct BaselineReport {
    pub(crate) path: Option<String>,
    pub(crate) status: String,
    pub(crate) latency_regression_threshold: f64,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct BaselineComparisonReport {
    pub(crate) status: String,
    pub(crate) baseline_status: Option<String>,
    pub(crate) baseline_latency_ms: Option<u128>,
    pub(crate) baseline_tokens_per_second: Option<f64>,
    pub(crate) hardware_match: bool,
    pub(crate) model_class_match: bool,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq)]
pub(crate) struct StreamTimingReport {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) first_byte_latency_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) first_sse_data_latency_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) first_content_delta_latency_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) first_tool_delta_latency_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_finish_latency_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) first_semantic_delta_latency_ms: Option<u128>,
}
