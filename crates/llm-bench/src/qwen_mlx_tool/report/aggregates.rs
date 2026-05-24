use super::*;

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedComparisonReport {
    pub(in crate::qwen_mlx_tool) status: String,
    pub(in crate::qwen_mlx_tool) fastest_successful_lanes: Vec<NormalizedFastestLaneReport>,
}

impl NormalizedComparisonReport {
    pub(in crate::qwen_mlx_tool) fn dry_run() -> Self {
        Self {
            status: "dry_run".to_owned(),
            fastest_successful_lanes: Vec::new(),
        }
    }
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedFastestLaneReport {
    pub(in crate::qwen_mlx_tool) case: &'static str,
    pub(in crate::qwen_mlx_tool) schema_variant: &'static str,
    pub(in crate::qwen_mlx_tool) tool_choice_variant: &'static str,
    pub(in crate::qwen_mlx_tool) max_tokens: u32,
    pub(in crate::qwen_mlx_tool) cache_phase: &'static str,
    pub(in crate::qwen_mlx_tool) fastest_lane: Option<String>,
    pub(in crate::qwen_mlx_tool) fastest_latency_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) lanes: Vec<NormalizedComparisonLaneMetric>,
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedComparisonLaneMetric {
    pub(in crate::qwen_mlx_tool) lane: String,
    pub(in crate::qwen_mlx_tool) status: String,
    pub(in crate::qwen_mlx_tool) best_latency_ms: Option<u128>,
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedAgenticGateReport {
    pub(in crate::qwen_mlx_tool) status: String,
    pub(in crate::qwen_mlx_tool) rows: Vec<NormalizedAgenticGateRow>,
}

impl NormalizedAgenticGateReport {
    pub(in crate::qwen_mlx_tool) fn dry_run() -> Self {
        Self {
            status: "dry_run".to_owned(),
            rows: Vec::new(),
        }
    }
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedAgenticGateRow {
    pub(in crate::qwen_mlx_tool) case: &'static str,
    pub(in crate::qwen_mlx_tool) schema_variant: &'static str,
    pub(in crate::qwen_mlx_tool) tool_choice_variant: &'static str,
    pub(in crate::qwen_mlx_tool) max_tokens: u32,
    pub(in crate::qwen_mlx_tool) cache_phase: &'static str,
    pub(in crate::qwen_mlx_tool) run_mode: &'static str,
    pub(in crate::qwen_mlx_tool) fastest_lane: Option<String>,
    pub(in crate::qwen_mlx_tool) lanes: Vec<NormalizedAgenticGateLaneMetric>,
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedAgenticGateLaneMetric {
    pub(in crate::qwen_mlx_tool) lane: String,
    pub(in crate::qwen_mlx_tool) pass_count: usize,
    pub(in crate::qwen_mlx_tool) p50_latency_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) latency_delta_vs_fastest_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) p50_first_byte_latency_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) p50_first_semantic_delta_latency_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) p50_first_tool_delta_latency_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) avg_tokens_per_second: Option<f64>,
    pub(in crate::qwen_mlx_tool) avg_cached_tokens: Option<f64>,
    pub(in crate::qwen_mlx_tool) avg_prompt_tokens: Option<f64>,
    pub(in crate::qwen_mlx_tool) avg_completion_tokens: Option<f64>,
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedAgenticStreamingFastPathAbReport {
    pub(in crate::qwen_mlx_tool) status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::qwen_mlx_tool) baseline_path: Option<String>,
    pub(in crate::qwen_mlx_tool) case: &'static str,
    pub(in crate::qwen_mlx_tool) schema_variant: &'static str,
    pub(in crate::qwen_mlx_tool) tool_choice_variant: &'static str,
    pub(in crate::qwen_mlx_tool) rows: Vec<NormalizedAgenticStreamingFastPathAbRow>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(in crate::qwen_mlx_tool) failure_reasons: Vec<String>,
}

impl NormalizedAgenticStreamingFastPathAbReport {
    pub(in crate::qwen_mlx_tool) fn dry_run(baseline_path: Option<&Path>) -> Self {
        match baseline_path {
            Some(path) => Self {
                status: "dry_run".to_owned(),
                baseline_path: Some(path.display().to_string()),
                case: AGENTIC_AB_CASE,
                schema_variant: AGENTIC_AB_SCHEMA_VARIANT,
                tool_choice_variant: AGENTIC_AB_TOOL_CHOICE_VARIANT,
                rows: Vec::new(),
                failure_reasons: Vec::new(),
            },
            None => Self::not_configured(),
        }
    }

    pub(in crate::qwen_mlx_tool) fn not_configured() -> Self {
        Self {
            status: "not_configured".to_owned(),
            baseline_path: None,
            case: AGENTIC_AB_CASE,
            schema_variant: AGENTIC_AB_SCHEMA_VARIANT,
            tool_choice_variant: AGENTIC_AB_TOOL_CHOICE_VARIANT,
            rows: Vec::new(),
            failure_reasons: Vec::new(),
        }
    }
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedAgenticStreamingFastPathAbRow {
    pub(in crate::qwen_mlx_tool) lane: String,
    pub(in crate::qwen_mlx_tool) kind: String,
    pub(in crate::qwen_mlx_tool) assertion_role: &'static str,
    pub(in crate::qwen_mlx_tool) cache_phase: String,
    pub(in crate::qwen_mlx_tool) run_mode: String,
    pub(in crate::qwen_mlx_tool) baseline_sample_count: usize,
    pub(in crate::qwen_mlx_tool) candidate_sample_count: usize,
    pub(in crate::qwen_mlx_tool) baseline_pass_count: usize,
    pub(in crate::qwen_mlx_tool) candidate_pass_count: usize,
    pub(in crate::qwen_mlx_tool) baseline_fail_count: usize,
    pub(in crate::qwen_mlx_tool) candidate_fail_count: usize,
    pub(in crate::qwen_mlx_tool) baseline_p50_first_tool_delta_latency_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) candidate_p50_first_tool_delta_latency_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) first_tool_delta_delta_ms: Option<i64>,
    pub(in crate::qwen_mlx_tool) first_tool_delta_advanced: Option<bool>,
    pub(in crate::qwen_mlx_tool) baseline_p50_tool_finish_latency_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) candidate_p50_tool_finish_latency_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) tool_finish_delta_ms: Option<i64>,
    pub(in crate::qwen_mlx_tool) final_validation_unchanged: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(in crate::qwen_mlx_tool) failure_reasons: Vec<String>,
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedPrefillConcurrencyReport {
    pub(in crate::qwen_mlx_tool) status: String,
    pub(in crate::qwen_mlx_tool) scenarios: Vec<NormalizedPrefillConcurrencyScenarioReport>,
}

impl NormalizedPrefillConcurrencyReport {
    pub(in crate::qwen_mlx_tool) fn dry_run() -> Self {
        Self {
            status: "dry_run".to_owned(),
            scenarios: prefill_concurrency_scenarios()
                .into_iter()
                .map(|scenario| NormalizedPrefillConcurrencyScenarioReport {
                    scenario: scenario.scenario,
                    objective: scenario.objective,
                    cache_phase: scenario.phase.name(),
                    run_mode: scenario.run_mode.name(),
                    lanes: Vec::new(),
                })
                .collect(),
        }
    }
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedPrefillConcurrencyScenarioReport {
    pub(in crate::qwen_mlx_tool) scenario: &'static str,
    pub(in crate::qwen_mlx_tool) objective: &'static str,
    pub(in crate::qwen_mlx_tool) cache_phase: &'static str,
    pub(in crate::qwen_mlx_tool) run_mode: &'static str,
    pub(in crate::qwen_mlx_tool) lanes: Vec<NormalizedPrefillConcurrencyLaneMetric>,
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedPrefillConcurrencyLaneMetric {
    pub(in crate::qwen_mlx_tool) lane: String,
    pub(in crate::qwen_mlx_tool) lane_kind: &'static str,
    pub(in crate::qwen_mlx_tool) experimental: bool,
    pub(in crate::qwen_mlx_tool) prefill_step_size: DefaultOrU64,
    pub(in crate::qwen_mlx_tool) cache_phase: &'static str,
    pub(in crate::qwen_mlx_tool) run_mode: &'static str,
    pub(in crate::qwen_mlx_tool) sample_count: usize,
    pub(in crate::qwen_mlx_tool) request_count: usize,
    pub(in crate::qwen_mlx_tool) pass_count: usize,
    pub(in crate::qwen_mlx_tool) fail_count: usize,
    pub(in crate::qwen_mlx_tool) p50_first_semantic_delta_latency_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) p50_elapsed_latency_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) avg_prompt_tokens: Option<f64>,
    pub(in crate::qwen_mlx_tool) avg_cached_tokens: Option<f64>,
    pub(in crate::qwen_mlx_tool) avg_uncached_tokens: Option<f64>,
    pub(in crate::qwen_mlx_tool) scheduler_prefill: NormalizedSchedulerPrefillCountersReport,
    pub(in crate::qwen_mlx_tool) checkpoint_reuse: NormalizedCheckpointReuseCountersReport,
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedSchedulerPrefillCountersReport {
    pub(in crate::qwen_mlx_tool) prefill_yields_delta: Option<i64>,
    pub(in crate::qwen_mlx_tool) prefill_yields_after: Option<u64>,
    pub(in crate::qwen_mlx_tool) prefill_yields_to_decode_delta: Option<i64>,
    pub(in crate::qwen_mlx_tool) prefill_yields_to_decode_after: Option<u64>,
    pub(in crate::qwen_mlx_tool) prefill_yield_reacquire_waits_delta: Option<i64>,
    pub(in crate::qwen_mlx_tool) prefill_yield_reacquire_waits_after: Option<u64>,
    pub(in crate::qwen_mlx_tool) prefill_yield_reacquire_wait_ms_total_delta: Option<f64>,
    pub(in crate::qwen_mlx_tool) prefill_yield_reacquire_wait_ms_total_after: Option<f64>,
    pub(in crate::qwen_mlx_tool) prefill_yield_reacquire_wait_ms_max_after: Option<f64>,
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedCheckpointReuseCountersReport {
    pub(in crate::qwen_mlx_tool) checkpoint_reuse_hits_delta: Option<i64>,
    pub(in crate::qwen_mlx_tool) checkpoint_reuse_hits_after: Option<u64>,
    pub(in crate::qwen_mlx_tool) checkpoint_reused_tokens_delta: Option<i64>,
    pub(in crate::qwen_mlx_tool) checkpoint_reused_tokens_after: Option<u64>,
    pub(in crate::qwen_mlx_tool) avoided_prefill_tokens_delta: Option<i64>,
    pub(in crate::qwen_mlx_tool) avoided_prefill_tokens_after: Option<u64>,
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedPrefillSweepReport {
    pub(in crate::qwen_mlx_tool) status: String,
    pub(in crate::qwen_mlx_tool) rows: Vec<NormalizedPrefillSweepRow>,
}

impl NormalizedPrefillSweepReport {
    pub(in crate::qwen_mlx_tool) fn dry_run() -> Self {
        Self {
            status: "dry_run".to_owned(),
            rows: Vec::new(),
        }
    }
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedPrefillSweepRow {
    pub(in crate::qwen_mlx_tool) case: &'static str,
    pub(in crate::qwen_mlx_tool) schema_variant: &'static str,
    pub(in crate::qwen_mlx_tool) tool_choice_variant: &'static str,
    pub(in crate::qwen_mlx_tool) max_tokens: u32,
    pub(in crate::qwen_mlx_tool) cache_phase: &'static str,
    pub(in crate::qwen_mlx_tool) run_mode: &'static str,
    pub(in crate::qwen_mlx_tool) fastest_lane: Option<String>,
    pub(in crate::qwen_mlx_tool) lanes: Vec<NormalizedPrefillSweepLaneMetric>,
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedPrefillSweepLaneMetric {
    pub(in crate::qwen_mlx_tool) lane: String,
    pub(in crate::qwen_mlx_tool) lane_kind: &'static str,
    pub(in crate::qwen_mlx_tool) experimental: bool,
    pub(in crate::qwen_mlx_tool) prefill_step_size: DefaultOrU64,
    pub(in crate::qwen_mlx_tool) valid: bool,
    pub(in crate::qwen_mlx_tool) failure_classifications: BTreeMap<String, usize>,
    pub(in crate::qwen_mlx_tool) invalid_reasons: Vec<String>,
    pub(in crate::qwen_mlx_tool) sample_count: usize,
    pub(in crate::qwen_mlx_tool) pass_count: usize,
    pub(in crate::qwen_mlx_tool) fail_count: usize,
    pub(in crate::qwen_mlx_tool) p50_first_response_byte_latency_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) p50_first_parsed_sse_chunk_latency_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) p50_first_semantic_delta_latency_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) p50_first_tool_delta_latency_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) p50_elapsed_latency_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) latency_delta_vs_fastest_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) avg_tokens_per_second: Option<f64>,
    pub(in crate::qwen_mlx_tool) avg_cached_tokens: Option<f64>,
    pub(in crate::qwen_mlx_tool) avg_uncached_tokens: Option<f64>,
    pub(in crate::qwen_mlx_tool) avg_prompt_tokens: Option<f64>,
    pub(in crate::qwen_mlx_tool) avg_completion_tokens: Option<f64>,
    pub(in crate::qwen_mlx_tool) avg_total_tokens: Option<f64>,
    pub(in crate::qwen_mlx_tool) response_headers: Vec<BTreeMap<String, String>>,
    pub(in crate::qwen_mlx_tool) admin_mlx_upstream_timing:
        Option<NormalizedPrefillSweepAdminMlxTiming>,
    pub(in crate::qwen_mlx_tool) process_rss_bytes_after: Option<u64>,
    pub(in crate::qwen_mlx_tool) stream_stalled_requests_delta: Option<i64>,
    pub(in crate::qwen_mlx_tool) no_progress_failures_delta: Option<i64>,
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedPrefillSweepAdminMlxTiming {
    pub(in crate::qwen_mlx_tool) stream_first_upstream_byte_ms: NormalizedAdminLatencyMetricReport,
    pub(in crate::qwen_mlx_tool) stream_first_parsed_chunk_ms: NormalizedAdminLatencyMetricReport,
    pub(in crate::qwen_mlx_tool) stream_first_tool_delta_ms: NormalizedAdminLatencyMetricReport,
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedStablePrefixReport {
    pub(in crate::qwen_mlx_tool) status: String,
    pub(in crate::qwen_mlx_tool) rows: Vec<NormalizedStablePrefixRow>,
}

impl NormalizedStablePrefixReport {
    pub(in crate::qwen_mlx_tool) fn dry_run() -> Self {
        Self {
            status: "dry_run".to_owned(),
            rows: Vec::new(),
        }
    }
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedStablePrefixRow {
    pub(in crate::qwen_mlx_tool) case: &'static str,
    pub(in crate::qwen_mlx_tool) schema_variant: &'static str,
    pub(in crate::qwen_mlx_tool) tool_choice_variant: &'static str,
    pub(in crate::qwen_mlx_tool) max_tokens: u32,
    pub(in crate::qwen_mlx_tool) cache_phase: &'static str,
    pub(in crate::qwen_mlx_tool) run_mode: &'static str,
    pub(in crate::qwen_mlx_tool) fastest_lane: Option<String>,
    pub(in crate::qwen_mlx_tool) lanes: Vec<NormalizedStablePrefixLaneMetric>,
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedStablePrefixLaneMetric {
    pub(in crate::qwen_mlx_tool) lane: String,
    pub(in crate::qwen_mlx_tool) lane_kind: &'static str,
    pub(in crate::qwen_mlx_tool) sample_count: usize,
    pub(in crate::qwen_mlx_tool) pass_count: usize,
    pub(in crate::qwen_mlx_tool) fail_count: usize,
    pub(in crate::qwen_mlx_tool) p50_first_semantic_delta_latency_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) p50_first_tool_delta_latency_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) p50_elapsed_latency_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) latency_delta_vs_fastest_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) avg_prompt_tokens: Option<f64>,
    pub(in crate::qwen_mlx_tool) avg_cached_tokens: Option<f64>,
    pub(in crate::qwen_mlx_tool) avg_uncached_tokens: Option<f64>,
    pub(in crate::qwen_mlx_tool) cache_status_counts: BTreeMap<String, usize>,
    pub(in crate::qwen_mlx_tool) request_cache_observations:
        Vec<NormalizedStablePrefixRequestCacheObservation>,
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedStablePrefixRequestCacheObservation {
    pub(in crate::qwen_mlx_tool) request_id: String,
    pub(in crate::qwen_mlx_tool) model: String,
    pub(in crate::qwen_mlx_tool) streamed: bool,
    pub(in crate::qwen_mlx_tool) prompt_tokens: u64,
    pub(in crate::qwen_mlx_tool) cached_tokens: Option<u64>,
    pub(in crate::qwen_mlx_tool) uncached_tokens: Option<u64>,
    pub(in crate::qwen_mlx_tool) cache_status: String,
    pub(in crate::qwen_mlx_tool) latency_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub(in crate::qwen_mlx_tool) struct EngineDbBaselineExport {
    #[serde(default)]
    pub(in crate::qwen_mlx_tool) source: Option<String>,
    #[serde(default)]
    pub(in crate::qwen_mlx_tool) rows: Vec<EngineDbBaselineRow>,
}

#[derive(Debug, Clone, Deserialize)]
pub(in crate::qwen_mlx_tool) struct EngineDbBaselineRow {
    pub(in crate::qwen_mlx_tool) engine: String,
    pub(in crate::qwen_mlx_tool) profile: String,
    #[serde(default)]
    pub(in crate::qwen_mlx_tool) model: Option<String>,
    pub(in crate::qwen_mlx_tool) probe: String,
    #[serde(default, alias = "ttft_ms", alias = "first_semantic_delta_ms")]
    pub(in crate::qwen_mlx_tool) ttfi_ms: Option<f64>,
    #[serde(default, alias = "first_tool_event_ms")]
    pub(in crate::qwen_mlx_tool) first_tool_delta_ms: Option<f64>,
    #[serde(default)]
    pub(in crate::qwen_mlx_tool) validated_tool_call_ms: Option<f64>,
    #[serde(default, alias = "latency_ms")]
    pub(in crate::qwen_mlx_tool) total_latency_ms: Option<f64>,
    #[serde(default, alias = "tok_s", alias = "toks_per_second")]
    pub(in crate::qwen_mlx_tool) tokens_per_second: Option<f64>,
    #[serde(default, alias = "cold_latency_ms")]
    pub(in crate::qwen_mlx_tool) cache_cold_latency_ms: Option<f64>,
    #[serde(default, alias = "warm_latency_ms")]
    pub(in crate::qwen_mlx_tool) cache_warm_latency_ms: Option<f64>,
    #[serde(default, alias = "speedup")]
    pub(in crate::qwen_mlx_tool) cache_speedup: Option<f64>,
    #[serde(default)]
    pub(in crate::qwen_mlx_tool) cached_tokens: Option<u64>,
    #[serde(default)]
    pub(in crate::qwen_mlx_tool) notes: Option<String>,
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedLatestPerformanceComparisonReport {
    pub(in crate::qwen_mlx_tool) status: String,
    pub(in crate::qwen_mlx_tool) engine_db_baseline_source: Option<String>,
    pub(in crate::qwen_mlx_tool) evidence: NormalizedLatestPerformanceEvidence,
    pub(in crate::qwen_mlx_tool) rows: Vec<NormalizedLatestPerformanceComparisonRow>,
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedLatestPerformanceEvidence {
    pub(in crate::qwen_mlx_tool) has_kir_latest: bool,
    pub(in crate::qwen_mlx_tool) has_direct_mlx_latest: bool,
    pub(in crate::qwen_mlx_tool) has_engine_db_baselines: bool,
    pub(in crate::qwen_mlx_tool) has_ttfi_ms: bool,
    pub(in crate::qwen_mlx_tool) has_cache_metrics: bool,
    pub(in crate::qwen_mlx_tool) has_tokens_per_second: bool,
}

impl NormalizedLatestPerformanceEvidence {
    pub(in crate::qwen_mlx_tool) fn from_rows(
        rows: &[NormalizedLatestPerformanceComparisonRow],
    ) -> Self {
        Self {
            has_kir_latest: rows.iter().any(|row| row.source_kind == "latest_kir"),
            has_direct_mlx_latest: rows.iter().any(|row| row.source_kind == "direct_mlx"),
            has_engine_db_baselines: rows
                .iter()
                .any(|row| row.source_kind == "engine_db_baseline"),
            has_ttfi_ms: rows
                .iter()
                .any(|row| row.ttfi_ms.is_some() || row.first_tool_delta_ms.is_some()),
            has_cache_metrics: rows.iter().any(|row| {
                row.cache_cold_latency_ms.is_some()
                    || row.cache_warm_latency_ms.is_some()
                    || row.cache_speedup.is_some()
                    || row.cached_tokens.is_some()
            }),
            has_tokens_per_second: rows.iter().any(|row| row.tokens_per_second.is_some()),
        }
    }
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedLatestPerformanceComparisonRow {
    pub(in crate::qwen_mlx_tool) source_kind: String,
    pub(in crate::qwen_mlx_tool) lane: Option<String>,
    pub(in crate::qwen_mlx_tool) kind: Option<String>,
    pub(in crate::qwen_mlx_tool) engine: Option<String>,
    pub(in crate::qwen_mlx_tool) profile: Option<String>,
    pub(in crate::qwen_mlx_tool) model: Option<String>,
    pub(in crate::qwen_mlx_tool) probe: String,
    pub(in crate::qwen_mlx_tool) ttfi_ms: Option<f64>,
    pub(in crate::qwen_mlx_tool) first_tool_delta_ms: Option<f64>,
    pub(in crate::qwen_mlx_tool) validated_tool_call_ms: Option<f64>,
    pub(in crate::qwen_mlx_tool) total_latency_ms: Option<f64>,
    pub(in crate::qwen_mlx_tool) tokens_per_second: Option<f64>,
    pub(in crate::qwen_mlx_tool) cache_cold_latency_ms: Option<f64>,
    pub(in crate::qwen_mlx_tool) cache_warm_latency_ms: Option<f64>,
    pub(in crate::qwen_mlx_tool) cache_speedup: Option<f64>,
    pub(in crate::qwen_mlx_tool) cached_tokens: Option<u64>,
    pub(in crate::qwen_mlx_tool) notes: Option<String>,
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedRequiredToolTtftMatrixReport {
    pub(in crate::qwen_mlx_tool) status: String,
    pub(in crate::qwen_mlx_tool) rows: Vec<NormalizedRequiredToolTtftMatrixRow>,
}

impl NormalizedRequiredToolTtftMatrixReport {
    pub(in crate::qwen_mlx_tool) fn dry_run() -> Self {
        Self {
            status: "dry_run".to_owned(),
            rows: Vec::new(),
        }
    }
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedRequiredToolTtftMatrixRow {
    pub(in crate::qwen_mlx_tool) lane: String,
    pub(in crate::qwen_mlx_tool) kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::qwen_mlx_tool) tool_parser: Option<&'static str>,
    pub(in crate::qwen_mlx_tool) schema_variant: &'static str,
    pub(in crate::qwen_mlx_tool) tool_choice_variant: &'static str,
    pub(in crate::qwen_mlx_tool) max_tokens: u32,
    pub(in crate::qwen_mlx_tool) cache_phase: &'static str,
    pub(in crate::qwen_mlx_tool) run_mode: &'static str,
    pub(in crate::qwen_mlx_tool) sample_index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::qwen_mlx_tool) request_index: Option<usize>,
    pub(in crate::qwen_mlx_tool) status: String,
    pub(in crate::qwen_mlx_tool) classification: String,
    pub(in crate::qwen_mlx_tool) first_response_byte_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) first_parsed_sse_chunk_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) first_tool_delta_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) tool_finish_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) validated_tool_call_ms: Option<f64>,
    pub(in crate::qwen_mlx_tool) latency_delta_vs_fastest_lane_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::qwen_mlx_tool) finish_reason: Option<String>,
    pub(in crate::qwen_mlx_tool) stream_stalled_requests_delta: Option<i64>,
    pub(in crate::qwen_mlx_tool) no_progress_failures_delta: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::qwen_mlx_tool) error: Option<String>,
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedToolRequiredStreamTimingReport {
    pub(in crate::qwen_mlx_tool) status: String,
    pub(in crate::qwen_mlx_tool) attribution: NormalizedToolRequiredStreamAttributionReport,
    pub(in crate::qwen_mlx_tool) lanes: Vec<NormalizedToolRequiredStreamLaneTimingReport>,
}

impl NormalizedToolRequiredStreamTimingReport {
    pub(in crate::qwen_mlx_tool) fn dry_run(lanes: &[NormalizedLaneReport]) -> Self {
        Self {
            status: "dry_run".to_owned(),
            attribution: NormalizedToolRequiredStreamAttributionReport::dry_run(),
            lanes: lanes
                .iter()
                .map(|lane| NormalizedToolRequiredStreamLaneTimingReport {
                    lane: lane.name.clone(),
                    kind: lane.kind,
                    pass_count: 0,
                    p50_first_byte_latency_ms: None,
                    p50_first_sse_data_latency_ms: None,
                    p50_first_tool_delta_latency_ms: None,
                    p50_tool_finish_latency_ms: None,
                    p50_first_semantic_delta_latency_ms: None,
                    admin_metrics: None,
                    admin_metrics_error: None,
                    tool_stream_observations: Vec::new(),
                })
                .collect(),
        }
    }
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedToolRequiredStreamAttributionReport {
    pub(in crate::qwen_mlx_tool) status: String,
    pub(in crate::qwen_mlx_tool) rows: Vec<NormalizedToolRequiredStreamAttributionRow>,
}

impl NormalizedToolRequiredStreamAttributionReport {
    pub(in crate::qwen_mlx_tool) fn dry_run() -> Self {
        Self {
            status: "dry_run".to_owned(),
            rows: Vec::new(),
        }
    }
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedToolRequiredStreamAttributionRow {
    pub(in crate::qwen_mlx_tool) lane: String,
    pub(in crate::qwen_mlx_tool) kind: &'static str,
    pub(in crate::qwen_mlx_tool) pass_count: usize,
    pub(in crate::qwen_mlx_tool) client: NormalizedToolRequiredStreamClientTiming,
    pub(in crate::qwen_mlx_tool) admin_metrics_scope: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::qwen_mlx_tool) admin_metrics: Option<NormalizedToolRequiredStreamAdminMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::qwen_mlx_tool) admin_metrics_error: Option<String>,
    pub(in crate::qwen_mlx_tool) first_tool_delta_gap_ms: NormalizedToolRequiredStreamGap,
    pub(in crate::qwen_mlx_tool) decision: &'static str,
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedToolRequiredStreamClientTiming {
    pub(in crate::qwen_mlx_tool) first_byte_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) first_sse_data_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) first_tool_delta_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) tool_finish_ms: Option<u128>,
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedToolRequiredStreamGap {
    pub(in crate::qwen_mlx_tool) mlx_stream_to_client_ms: Option<f64>,
    pub(in crate::qwen_mlx_tool) kir_first_tool_delta_to_client_ms: Option<f64>,
    pub(in crate::qwen_mlx_tool) validated_tool_call_to_tool_finish_ms: Option<f64>,
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedToolRequiredStreamLaneTimingReport {
    pub(in crate::qwen_mlx_tool) lane: String,
    pub(in crate::qwen_mlx_tool) kind: &'static str,
    pub(in crate::qwen_mlx_tool) pass_count: usize,
    pub(in crate::qwen_mlx_tool) p50_first_byte_latency_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) p50_first_sse_data_latency_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) p50_first_tool_delta_latency_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) p50_tool_finish_latency_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) p50_first_semantic_delta_latency_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) admin_metrics: Option<NormalizedToolRequiredStreamAdminMetrics>,
    pub(in crate::qwen_mlx_tool) admin_metrics_error: Option<String>,
    pub(in crate::qwen_mlx_tool) tool_stream_observations: Vec<NormalizedToolStreamObservation>,
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedToolStreamObservation {
    pub(in crate::qwen_mlx_tool) request_id: String,
    pub(in crate::qwen_mlx_tool) model: String,
    pub(in crate::qwen_mlx_tool) streamed: bool,
    pub(in crate::qwen_mlx_tool) client_first_byte_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) client_first_sse_data_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) client_visible_first_tool_delta_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) client_tool_finish_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) kir_first_tool_delta_ms: Option<u64>,
    pub(in crate::qwen_mlx_tool) kir_first_tool_delta_after_ttft_ms: Option<u64>,
    pub(in crate::qwen_mlx_tool) tool_argument_assembly_ms: Option<u64>,
    pub(in crate::qwen_mlx_tool) tool_intent_fill_ms: Option<u64>,
    pub(in crate::qwen_mlx_tool) tool_schema_validation_ms: Option<u64>,
    pub(in crate::qwen_mlx_tool) tool_finish_ms: Option<u64>,
    pub(in crate::qwen_mlx_tool) validated_tool_call_ms: Option<u64>,
    pub(in crate::qwen_mlx_tool) mlx_response_headers_ms: Option<u64>,
    pub(in crate::qwen_mlx_tool) mlx_first_upstream_byte_ms: Option<u64>,
    pub(in crate::qwen_mlx_tool) mlx_first_parsed_chunk_ms: Option<u64>,
    pub(in crate::qwen_mlx_tool) mlx_first_tool_delta_ms: Option<u64>,
    pub(in crate::qwen_mlx_tool) mlx_upstream_complete_ms: Option<u64>,
    pub(in crate::qwen_mlx_tool) latency_ms: u64,
}

#[derive(Clone, Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedToolRequiredStreamAdminMetrics {
    pub(in crate::qwen_mlx_tool) first_tool_delta_ms: NormalizedAdminLatencyMetricReport,
    pub(in crate::qwen_mlx_tool) first_tool_delta_after_ttft_ms: NormalizedAdminLatencyMetricReport,
    pub(in crate::qwen_mlx_tool) tool_argument_assembly_ms: NormalizedAdminLatencyMetricReport,
    pub(in crate::qwen_mlx_tool) tool_intent_fill_ms: NormalizedAdminLatencyMetricReport,
    pub(in crate::qwen_mlx_tool) tool_schema_validation_ms: NormalizedAdminLatencyMetricReport,
    pub(in crate::qwen_mlx_tool) tool_finish_ms: NormalizedAdminLatencyMetricReport,
    pub(in crate::qwen_mlx_tool) validated_tool_call_ms: NormalizedAdminLatencyMetricReport,
    pub(in crate::qwen_mlx_tool) mlx_stream_first_upstream_byte_ms:
        NormalizedAdminLatencyMetricReport,
    pub(in crate::qwen_mlx_tool) mlx_stream_first_parsed_chunk_ms:
        NormalizedAdminLatencyMetricReport,
    pub(in crate::qwen_mlx_tool) mlx_stream_first_tool_delta_ms: NormalizedAdminLatencyMetricReport,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedAdminLatencyMetricReport {
    pub(in crate::qwen_mlx_tool) count_delta: Option<i64>,
    pub(in crate::qwen_mlx_tool) count_after: Option<u64>,
    pub(in crate::qwen_mlx_tool) min_ms_after: Option<f64>,
    pub(in crate::qwen_mlx_tool) max_ms_after: Option<f64>,
    pub(in crate::qwen_mlx_tool) avg_ms_after: Option<f64>,
    pub(in crate::qwen_mlx_tool) window_avg_ms: Option<f64>,
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedAggregateSummaryRow {
    pub(in crate::qwen_mlx_tool) lane: String,
    pub(in crate::qwen_mlx_tool) case: &'static str,
    pub(in crate::qwen_mlx_tool) schema_variant: &'static str,
    pub(in crate::qwen_mlx_tool) tool_choice_variant: &'static str,
    pub(in crate::qwen_mlx_tool) max_tokens: u32,
    pub(in crate::qwen_mlx_tool) cache_phase: &'static str,
    pub(in crate::qwen_mlx_tool) run_mode: &'static str,
    pub(in crate::qwen_mlx_tool) pass_count: usize,
    pub(in crate::qwen_mlx_tool) fail_count: usize,
    pub(in crate::qwen_mlx_tool) p50_latency_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) p95_latency_ms: Option<u128>,
    pub(in crate::qwen_mlx_tool) avg_cached_tokens: Option<f64>,
    pub(in crate::qwen_mlx_tool) avg_prompt_tokens: Option<f64>,
    pub(in crate::qwen_mlx_tool) avg_completion_tokens: Option<f64>,
    pub(in crate::qwen_mlx_tool) avg_total_tokens: Option<f64>,
    pub(in crate::qwen_mlx_tool) fastest_lane: Option<String>,
}
