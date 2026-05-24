use super::*;

#[derive(Debug, Clone, Copy)]
pub(in crate::qwen_mlx_tool) struct PrefillConcurrencyScenario {
    pub(in crate::qwen_mlx_tool) scenario: &'static str,
    pub(in crate::qwen_mlx_tool) objective: &'static str,
    pub(in crate::qwen_mlx_tool) phase: CachePhase,
    pub(in crate::qwen_mlx_tool) run_mode: RunMode,
}

pub(in crate::qwen_mlx_tool) fn prefill_concurrency_scenarios() -> [PrefillConcurrencyScenario; 3] {
    [
        PrefillConcurrencyScenario {
            scenario: "cold_long_context_prefill",
            objective: "cold long-context prefill latency and scheduler counter baseline",
            phase: CachePhase::Cold,
            run_mode: RunMode::Sequential,
        },
        PrefillConcurrencyScenario {
            scenario: "warm_checkpoint_reuse",
            objective: "warm prefix or checkpoint reuse after a compatible long-context prefill",
            phase: CachePhase::WarmSamePrompt,
            run_mode: RunMode::Sequential,
        },
        PrefillConcurrencyScenario {
            scenario: "mixed_long_prefill_short_decode_concurrency",
            objective: "concurrent long-prefill pressure with decode admission interleaving",
            phase: CachePhase::Cold,
            run_mode: RunMode::Concurrent,
        },
    ]
}

pub(in crate::qwen_mlx_tool) fn prefill_concurrency_report(
    lanes: &[NormalizedLaneReport],
    probes: &[NormalizedProbePlan],
) -> NormalizedPrefillConcurrencyReport {
    let scenarios = prefill_concurrency_scenarios()
        .into_iter()
        .map(|scenario| {
            let mut lane_metrics = lanes
                .iter()
                .filter_map(|lane| prefill_concurrency_lane_metric(lane, probes, scenario))
                .collect::<Vec<_>>();
            lane_metrics.sort_by(|left, right| {
                prefill_concurrency_metric_sort_key(left)
                    .cmp(&prefill_concurrency_metric_sort_key(right))
            });
            NormalizedPrefillConcurrencyScenarioReport {
                scenario: scenario.scenario,
                objective: scenario.objective,
                cache_phase: scenario.phase.name(),
                run_mode: scenario.run_mode.name(),
                lanes: lane_metrics,
            }
        })
        .collect::<Vec<_>>();
    let status = if scenarios.iter().any(|scenario| !scenario.lanes.is_empty()) {
        "reported"
    } else {
        "no_samples"
    };
    NormalizedPrefillConcurrencyReport {
        status: status.to_owned(),
        scenarios,
    }
}

pub(in crate::qwen_mlx_tool) fn prefill_concurrency_lane_metric(
    lane: &NormalizedLaneReport,
    probes: &[NormalizedProbePlan],
    scenario: PrefillConcurrencyScenario,
) -> Option<NormalizedPrefillConcurrencyLaneMetric> {
    let samples = lane_samples(lane)
        .filter(|sample| {
            sample.cache_phase == scenario.phase.name()
                && sample.run_mode == scenario.run_mode.name()
                && probes
                    .iter()
                    .any(|probe| sample_matches_probe(sample, *probe))
        })
        .collect::<Vec<_>>();
    if samples.is_empty() {
        return None;
    }
    let passed = samples
        .iter()
        .copied()
        .filter(|sample| sample.status == "passed")
        .collect::<Vec<_>>();
    Some(NormalizedPrefillConcurrencyLaneMetric {
        lane: lane.name.clone(),
        lane_kind: lane.kind,
        experimental: lane.experimental,
        prefill_step_size: lane.mlx_lm_settings.prefill_step_size,
        cache_phase: scenario.phase.name(),
        run_mode: scenario.run_mode.name(),
        sample_count: samples.len(),
        request_count: sample_request_count(&samples),
        pass_count: passed.len(),
        fail_count: samples
            .iter()
            .filter(|sample| sample.status == "failed")
            .count(),
        p50_first_semantic_delta_latency_ms: percentile_for_samples(&passed, |sample| {
            sample.stream_timing.first_semantic_delta_latency_ms
        }),
        p50_elapsed_latency_ms: percentile_for_samples(&passed, |sample| sample.latency_ms),
        avg_prompt_tokens: average_u64(passed.iter().filter_map(|sample| sample.prompt_tokens)),
        avg_cached_tokens: average_u64(passed.iter().filter_map(|sample| sample.cached_tokens)),
        avg_uncached_tokens: average_u64(
            passed
                .iter()
                .filter_map(|sample| sample_direct_uncached_tokens(sample)),
        ),
        scheduler_prefill: scheduler_prefill_counters(&lane.admin_metrics),
        checkpoint_reuse: checkpoint_reuse_counters(&lane.admin_metrics),
    })
}

pub(in crate::qwen_mlx_tool) fn sample_request_count(samples: &[&NormalizedSampleReport]) -> usize {
    let request_indexes = samples
        .iter()
        .filter_map(|sample| sample.request_index)
        .collect::<BTreeSet<_>>();
    if request_indexes.is_empty() {
        samples.len()
    } else {
        request_indexes.len()
    }
}

pub(in crate::qwen_mlx_tool) fn prefill_concurrency_metric_sort_key(
    metric: &NormalizedPrefillConcurrencyLaneMetric,
) -> (u128, String) {
    (
        metric
            .p50_first_semantic_delta_latency_ms
            .unwrap_or(u128::MAX),
        metric.lane.clone(),
    )
}

#[cfg(test)]
pub(in crate::qwen_mlx_tool) fn prefill_sweep_report(
    lanes: &[NormalizedLaneReport],
    probes: &[NormalizedProbePlan],
) -> NormalizedPrefillSweepReport {
    prefill_sweep_report_for_phases(lanes, probes, &CachePhase::all())
}

pub(in crate::qwen_mlx_tool) fn prefill_sweep_report_for_phases(
    lanes: &[NormalizedLaneReport],
    probes: &[NormalizedProbePlan],
    phases: &[CachePhase],
) -> NormalizedPrefillSweepReport {
    let mut rows = Vec::new();
    for &probe in probes {
        for &phase in phases {
            for run_mode in [RunMode::Sequential, RunMode::Concurrent] {
                let mut lane_metrics = lanes
                    .iter()
                    .filter_map(|lane| prefill_sweep_lane_metric(lane, probe, phase, run_mode))
                    .collect::<Vec<_>>();
                if lane_metrics.is_empty() {
                    continue;
                }
                let fastest_latency = lane_metrics
                    .iter()
                    .filter(|metric| metric.valid)
                    .filter_map(|metric| metric.p50_first_semantic_delta_latency_ms)
                    .min();
                let fastest_lane = fastest_latency.and_then(|fastest| {
                    lane_metrics
                        .iter()
                        .find(|metric| {
                            metric.valid
                                && metric.p50_first_semantic_delta_latency_ms == Some(fastest)
                        })
                        .map(|metric| metric.lane.clone())
                });
                if let Some(fastest) = fastest_latency {
                    for metric in &mut lane_metrics {
                        metric.latency_delta_vs_fastest_ms = metric
                            .p50_first_semantic_delta_latency_ms
                            .map(|latency| latency.saturating_sub(fastest));
                    }
                }
                lane_metrics.sort_by(|left, right| {
                    prefill_metric_sort_key(left).cmp(&prefill_metric_sort_key(right))
                });
                rows.push(NormalizedPrefillSweepRow {
                    case: probe.case.name(),
                    schema_variant: probe.schema_variant.name(),
                    tool_choice_variant: probe.tool_choice_variant.name(),
                    max_tokens: probe.max_tokens,
                    cache_phase: phase.name(),
                    run_mode: run_mode.name(),
                    fastest_lane,
                    lanes: lane_metrics,
                });
            }
        }
    }
    NormalizedPrefillSweepReport {
        status: if rows.is_empty() {
            "no_samples"
        } else {
            "reported"
        }
        .to_owned(),
        rows,
    }
}

pub(in crate::qwen_mlx_tool) fn prefill_sweep_lane_metric(
    lane: &NormalizedLaneReport,
    probe: NormalizedProbePlan,
    phase: CachePhase,
    run_mode: RunMode,
) -> Option<NormalizedPrefillSweepLaneMetric> {
    let samples = lane_samples(lane)
        .filter(|sample| {
            sample_matches_probe(sample, probe)
                && sample.cache_phase == phase.name()
                && sample.run_mode == run_mode.name()
        })
        .collect::<Vec<_>>();
    if samples.is_empty() {
        return None;
    }
    let passed = samples
        .iter()
        .copied()
        .filter(|sample| sample.status == "passed")
        .collect::<Vec<_>>();
    let pass_count = passed.len();
    let fail_count = samples
        .iter()
        .filter(|sample| sample.status == "failed")
        .count();
    let p50_first_semantic_delta_latency_ms = percentile_for_samples(&passed, |sample| {
        sample.stream_timing.first_semantic_delta_latency_ms
    });
    let p50_first_tool_delta_latency_ms = percentile_for_samples(&passed, |sample| {
        sample.stream_timing.first_tool_delta_latency_ms
    });
    let stream_stalled_requests_delta =
        admin_counter_delta(&lane.admin_metrics, &["stream_stalled_requests"]);
    let no_progress_failures_delta =
        admin_counter_delta(&lane.admin_metrics, &["no_progress_failures"]);
    let failure_classifications = sample_failure_classification_counts(&samples);
    let mut invalid_reasons = Vec::new();
    if fail_count > 0 {
        invalid_reasons.push("sample_failed".to_owned());
    }
    if p50_first_semantic_delta_latency_ms.is_none() {
        invalid_reasons.push("missing_ttft".to_owned());
        invalid_reasons.push("missing_stream_delta".to_owned());
    }
    if probe.case.requires_tool_delta() && p50_first_tool_delta_latency_ms.is_none() {
        invalid_reasons.push("missing_tool_delta".to_owned());
    }
    if stream_stalled_requests_delta.is_some_and(|delta| delta > 0) {
        invalid_reasons.push("admin_stalled_request_delta".to_owned());
    }
    if no_progress_failures_delta.is_some_and(|delta| delta > 0) {
        invalid_reasons.push("admin_no_progress_delta".to_owned());
    }
    invalid_reasons.extend(failure_classifications.keys().cloned());
    invalid_reasons.sort();
    invalid_reasons.dedup();

    Some(NormalizedPrefillSweepLaneMetric {
        lane: lane.name.clone(),
        lane_kind: lane.kind,
        experimental: lane.experimental,
        prefill_step_size: lane.mlx_lm_settings.prefill_step_size,
        valid: invalid_reasons.is_empty(),
        failure_classifications,
        invalid_reasons,
        sample_count: samples.len(),
        pass_count,
        fail_count,
        p50_first_response_byte_latency_ms: percentile_for_samples(&passed, |sample| {
            sample.stream_timing.first_byte_latency_ms
        }),
        p50_first_parsed_sse_chunk_latency_ms: percentile_for_samples(&passed, |sample| {
            sample.stream_timing.first_sse_data_latency_ms
        }),
        p50_first_semantic_delta_latency_ms,
        p50_first_tool_delta_latency_ms,
        p50_elapsed_latency_ms: percentile_for_samples(&passed, |sample| sample.latency_ms),
        latency_delta_vs_fastest_ms: None,
        avg_tokens_per_second: average_f64(
            passed.iter().filter_map(|sample| sample.tokens_per_second),
        ),
        avg_cached_tokens: average_u64(passed.iter().filter_map(|sample| sample.cached_tokens)),
        avg_uncached_tokens: average_u64(
            passed
                .iter()
                .filter_map(|sample| sample_direct_uncached_tokens(sample)),
        ),
        avg_prompt_tokens: average_u64(passed.iter().filter_map(|sample| sample.prompt_tokens)),
        avg_completion_tokens: average_u64(
            passed.iter().filter_map(|sample| sample.completion_tokens),
        ),
        avg_total_tokens: average_u64(passed.iter().filter_map(|sample| sample.total_tokens)),
        response_headers: samples
            .iter()
            .filter_map(|sample| sample.response_headers.clone())
            .collect(),
        admin_mlx_upstream_timing: normalized_prefill_admin_mlx_timing(&lane.admin_metrics),
        process_rss_bytes_after: admin_counter_after(&lane.admin_metrics, &["process_rss_bytes"]),
        stream_stalled_requests_delta,
        no_progress_failures_delta,
    })
}

pub(in crate::qwen_mlx_tool) fn prefill_metric_sort_key(
    metric: &NormalizedPrefillSweepLaneMetric,
) -> (bool, u128, String) {
    (
        !metric.valid,
        metric
            .p50_first_semantic_delta_latency_ms
            .unwrap_or(u128::MAX),
        metric.lane.clone(),
    )
}

#[cfg(test)]
pub(in crate::qwen_mlx_tool) fn stable_prefix_report(
    lanes: &[NormalizedLaneReport],
    probes: &[NormalizedProbePlan],
) -> NormalizedStablePrefixReport {
    stable_prefix_report_for_phases(lanes, probes, &CachePhase::all())
}

pub(in crate::qwen_mlx_tool) fn stable_prefix_report_for_phases(
    lanes: &[NormalizedLaneReport],
    probes: &[NormalizedProbePlan],
    phases: &[CachePhase],
) -> NormalizedStablePrefixReport {
    let mut rows = Vec::new();
    for &probe in probes {
        for &phase in phases {
            for run_mode in [RunMode::Sequential, RunMode::Concurrent] {
                let mut lane_metrics = lanes
                    .iter()
                    .filter_map(|lane| stable_prefix_lane_metric(lane, probe, phase, run_mode))
                    .collect::<Vec<_>>();
                if lane_metrics.is_empty() {
                    continue;
                }
                let fastest_latency = lane_metrics
                    .iter()
                    .filter_map(|metric| metric.p50_elapsed_latency_ms)
                    .min();
                let fastest_lane = fastest_latency.and_then(|fastest| {
                    lane_metrics
                        .iter()
                        .find(|metric| metric.p50_elapsed_latency_ms == Some(fastest))
                        .map(|metric| metric.lane.clone())
                });
                if let Some(fastest) = fastest_latency {
                    for metric in &mut lane_metrics {
                        metric.latency_delta_vs_fastest_ms = metric
                            .p50_elapsed_latency_ms
                            .map(|latency| latency.saturating_sub(fastest));
                    }
                }
                lane_metrics.sort_by(|left, right| {
                    stable_prefix_metric_sort_key(left).cmp(&stable_prefix_metric_sort_key(right))
                });
                rows.push(NormalizedStablePrefixRow {
                    case: probe.case.name(),
                    schema_variant: probe.schema_variant.name(),
                    tool_choice_variant: probe.tool_choice_variant.name(),
                    max_tokens: probe.max_tokens,
                    cache_phase: phase.name(),
                    run_mode: run_mode.name(),
                    fastest_lane,
                    lanes: lane_metrics,
                });
            }
        }
    }
    NormalizedStablePrefixReport {
        status: if rows.is_empty() {
            "no_samples"
        } else {
            "reported"
        }
        .to_owned(),
        rows,
    }
}

pub(in crate::qwen_mlx_tool) fn stable_prefix_lane_metric(
    lane: &NormalizedLaneReport,
    probe: NormalizedProbePlan,
    phase: CachePhase,
    run_mode: RunMode,
) -> Option<NormalizedStablePrefixLaneMetric> {
    let samples = lane_samples(lane)
        .filter(|sample| {
            sample_matches_probe(sample, probe)
                && sample.cache_phase == phase.name()
                && sample.run_mode == run_mode.name()
        })
        .collect::<Vec<_>>();
    if samples.is_empty() {
        return None;
    }
    let passed = samples
        .iter()
        .copied()
        .filter(|sample| sample.status == "passed")
        .collect::<Vec<_>>();
    let request_cache_observations = matching_request_cache_observations(lane, &samples);
    Some(NormalizedStablePrefixLaneMetric {
        lane: lane.name.clone(),
        lane_kind: lane.kind,
        sample_count: samples.len(),
        pass_count: passed.len(),
        fail_count: samples
            .iter()
            .filter(|sample| sample.status == "failed")
            .count(),
        p50_first_semantic_delta_latency_ms: percentile_for_samples(&passed, |sample| {
            sample.stream_timing.first_semantic_delta_latency_ms
        }),
        p50_first_tool_delta_latency_ms: percentile_for_samples(&passed, |sample| {
            sample.stream_timing.first_tool_delta_latency_ms
        }),
        p50_elapsed_latency_ms: percentile_for_samples(&passed, |sample| sample.latency_ms),
        latency_delta_vs_fastest_ms: None,
        avg_prompt_tokens: average_u64(passed.iter().filter_map(|sample| sample.prompt_tokens)),
        avg_cached_tokens: average_u64(
            passed
                .iter()
                .filter_map(|sample| sample_cached_tokens(sample, &request_cache_observations)),
        ),
        avg_uncached_tokens: average_u64(
            passed
                .iter()
                .filter_map(|sample| sample_uncached_tokens(sample, &request_cache_observations)),
        ),
        cache_status_counts: cache_status_counts(&passed, &request_cache_observations),
        request_cache_observations,
    })
}

pub(in crate::qwen_mlx_tool) fn stable_prefix_metric_sort_key(
    metric: &NormalizedStablePrefixLaneMetric,
) -> (u128, String) {
    (
        metric.p50_elapsed_latency_ms.unwrap_or(u128::MAX),
        metric.lane.clone(),
    )
}
