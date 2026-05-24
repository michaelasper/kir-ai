use super::*;

pub(in crate::qwen_mlx_tool) fn normalized_plan_summary(
    lanes: &[NormalizedLaneConfig],
    probes: &[NormalizedProbePlan],
    run_config: &NormalizedRunConfig,
) -> NormalizedPlanSummaryReport {
    let sequential_plan = phase_plan(
        &run_config.cache_phases,
        run_config.warmups,
        run_config.samples,
    );
    let concurrent_plan = concurrent_phase_plan(
        &run_config.cache_phases,
        run_config.concurrent_requests,
        run_config.effective_concurrent_samples,
    );
    let per_probe_warmups = sequential_plan
        .iter()
        .filter(|run| run.kind == PlannedRunKind::Warmup)
        .count();
    let per_probe_sequential_measured = sequential_plan
        .iter()
        .filter(|run| run.kind == PlannedRunKind::Measured)
        .count();
    let per_probe_concurrent_measured = concurrent_plan.len();
    let plan_multiplier = lanes.len().saturating_mul(probes.len());
    let warmup_requests = per_probe_warmups.saturating_mul(plan_multiplier);
    let sequential_measured_requests =
        per_probe_sequential_measured.saturating_mul(plan_multiplier);
    let concurrent_measured_requests =
        per_probe_concurrent_measured.saturating_mul(plan_multiplier);
    let measured_requests =
        sequential_measured_requests.saturating_add(concurrent_measured_requests);
    let total_http_requests = warmup_requests.saturating_add(measured_requests);

    NormalizedPlanSummaryReport {
        probe_count: probes.len(),
        lane_count: lanes.len(),
        warmups_per_warm_phase: run_config.warmups,
        samples_per_phase: run_config.samples,
        concurrent_requests: run_config.concurrent_requests,
        concurrent_samples: run_config.concurrent_samples,
        effective_concurrent_samples: run_config.effective_concurrent_samples,
        cache_phases: run_config
            .cache_phases
            .iter()
            .map(|phase| phase.name())
            .collect(),
        probes: probes
            .iter()
            .map(|probe| NormalizedPlanProbeReport {
                case: probe.case.name(),
                schema_variant: probe.schema_variant.name(),
                tool_choice_variant: probe.tool_choice_variant.name(),
                max_tokens: probe.max_tokens,
            })
            .collect(),
        lanes: lanes.iter().map(|lane| lane.name.clone()).collect(),
        warmup_requests,
        measured_requests,
        sequential_measured_requests,
        concurrent_measured_requests,
        total_http_requests,
        planned_prompt_token_budget: total_http_requests.saturating_mul(run_config.context_tokens),
    }
}

pub(in crate::qwen_mlx_tool) fn enforce_plan_budget(
    summary: &NormalizedPlanSummaryReport,
    max_requests: Option<usize>,
    max_planned_prompt_tokens: Option<usize>,
) -> anyhow::Result<()> {
    if let Some(max_requests) = max_requests
        && summary.total_http_requests > max_requests
    {
        anyhow::bail!(
            "selected benchmark plan requires {} HTTP requests, exceeding --max-requests {max_requests}",
            summary.total_http_requests
        );
    }
    if let Some(max_planned_prompt_tokens) = max_planned_prompt_tokens
        && summary.planned_prompt_token_budget > max_planned_prompt_tokens
    {
        anyhow::bail!(
            "selected benchmark plan requires {} planned prompt tokens, exceeding --max-planned-prompt-tokens {max_planned_prompt_tokens}",
            summary.planned_prompt_token_budget
        );
    }
    Ok(())
}

pub(in crate::qwen_mlx_tool) fn planned_requests_for(
    probes: &[NormalizedProbePlan],
    run_config: &NormalizedRunConfig,
) -> Vec<NormalizedPlannedRequestReport> {
    let mut planned_requests = Vec::new();
    for &probe in probes {
        for planned in phase_plan(
            &run_config.cache_phases,
            run_config.warmups,
            run_config.samples,
        ) {
            planned_requests.push(NormalizedPlannedRequestReport::new(
                probe, planned, run_config,
            ));
        }
        for planned in concurrent_phase_plan(
            &run_config.cache_phases,
            run_config.concurrent_requests,
            run_config.effective_concurrent_samples,
        ) {
            planned_requests.push(NormalizedPlannedRequestReport::new(
                probe, planned, run_config,
            ));
        }
    }
    planned_requests
}

pub(in crate::qwen_mlx_tool) fn compare_normalized_lanes_for_phases(
    lanes: &[NormalizedLaneReport],
    probes: &[NormalizedProbePlan],
    phases: &[CachePhase],
) -> NormalizedComparisonReport {
    let mut fastest = Vec::new();
    for &probe in probes {
        for &phase in phases {
            let mut fastest_lane = None;
            let mut fastest_latency_ms = None;
            let mut lane_metrics = Vec::new();
            for lane in lanes {
                let best_latency_ms = lane
                    .samples
                    .iter()
                    .filter(|sample| {
                        sample_matches_probe(sample, probe)
                            && sample.cache_phase == phase.name()
                            && sample.status == "passed"
                    })
                    .filter_map(|sample| sample.latency_ms)
                    .min();
                if let Some(latency) = best_latency_ms
                    && fastest_latency_ms.is_none_or(|fastest| latency < fastest)
                {
                    fastest_latency_ms = Some(latency);
                    fastest_lane = Some(lane.name.clone());
                }
                lane_metrics.push(NormalizedComparisonLaneMetric {
                    lane: lane.name.clone(),
                    status: lane.status.clone(),
                    best_latency_ms,
                });
            }
            fastest.push(NormalizedFastestLaneReport {
                case: probe.case.name(),
                schema_variant: probe.schema_variant.name(),
                tool_choice_variant: probe.tool_choice_variant.name(),
                max_tokens: probe.max_tokens,
                cache_phase: phase.name(),
                fastest_lane,
                fastest_latency_ms,
                lanes: lane_metrics,
            });
        }
    }
    NormalizedComparisonReport {
        status: if lanes.len() > 1 {
            "comparable"
        } else {
            "single_lane"
        }
        .to_owned(),
        fastest_successful_lanes: fastest,
    }
}

#[cfg(test)]
pub(in crate::qwen_mlx_tool) fn aggregate_normalized_summary(
    lanes: &[NormalizedLaneReport],
    probes: &[NormalizedProbePlan],
) -> Vec<NormalizedAggregateSummaryRow> {
    aggregate_normalized_summary_for_phases(lanes, probes, &CachePhase::all())
}

pub(in crate::qwen_mlx_tool) fn aggregate_normalized_summary_for_phases(
    lanes: &[NormalizedLaneReport],
    probes: &[NormalizedProbePlan],
    phases: &[CachePhase],
) -> Vec<NormalizedAggregateSummaryRow> {
    let mut rows = Vec::new();
    for lane in lanes {
        for &probe in probes {
            for &phase in phases {
                for run_mode in [RunMode::Sequential, RunMode::Concurrent] {
                    let samples = lane_samples(lane)
                        .filter(|sample| {
                            sample_matches_probe(sample, probe)
                                && sample.cache_phase == phase.name()
                                && sample.run_mode == run_mode.name()
                        })
                        .collect::<Vec<_>>();
                    if samples.is_empty() {
                        continue;
                    }
                    let pass_count = samples
                        .iter()
                        .filter(|sample| sample.status == "passed")
                        .count();
                    let fail_count = samples
                        .iter()
                        .filter(|sample| sample.status == "failed")
                        .count();
                    let mut passed_latencies = samples
                        .iter()
                        .filter(|sample| sample.status == "passed")
                        .filter_map(|sample| sample.latency_ms)
                        .collect::<Vec<_>>();
                    passed_latencies.sort_unstable();
                    rows.push(NormalizedAggregateSummaryRow {
                        lane: lane.name.clone(),
                        case: probe.case.name(),
                        schema_variant: probe.schema_variant.name(),
                        tool_choice_variant: probe.tool_choice_variant.name(),
                        max_tokens: probe.max_tokens,
                        cache_phase: phase.name(),
                        run_mode: run_mode.name(),
                        pass_count,
                        fail_count,
                        p50_latency_ms: percentile_latency(&passed_latencies, 0.50),
                        p95_latency_ms: percentile_latency(&passed_latencies, 0.95),
                        avg_cached_tokens: average_u64(
                            samples.iter().filter_map(|sample| sample.cached_tokens),
                        ),
                        avg_prompt_tokens: average_u64(
                            samples.iter().filter_map(|sample| sample.prompt_tokens),
                        ),
                        avg_completion_tokens: average_u64(
                            samples.iter().filter_map(|sample| sample.completion_tokens),
                        ),
                        avg_total_tokens: average_u64(
                            samples.iter().filter_map(|sample| sample.total_tokens),
                        ),
                        fastest_lane: fastest_lane_for(lanes, probe, phase, run_mode),
                    });
                }
            }
        }
    }
    rows
}

#[cfg(test)]
pub(in crate::qwen_mlx_tool) fn agentic_gate_report(
    lanes: &[NormalizedLaneReport],
) -> NormalizedAgenticGateReport {
    agentic_gate_report_for_phases(lanes, &CachePhase::all())
}

pub(in crate::qwen_mlx_tool) fn agentic_gate_report_for_phases(
    lanes: &[NormalizedLaneReport],
    phases: &[CachePhase],
) -> NormalizedAgenticGateReport {
    let mut rows = Vec::new();
    for probe in NormalizedProbePlan::focused_agentic_gate() {
        for &phase in phases {
            for run_mode in [RunMode::Sequential, RunMode::Concurrent] {
                let mut lane_metrics = Vec::new();
                let mut fastest_lane = None;
                let mut fastest_latency = None;
                for lane in lanes {
                    let samples = lane_samples(lane)
                        .filter(|sample| {
                            sample_matches_probe(sample, probe)
                                && sample.cache_phase == phase.name()
                                && sample.run_mode == run_mode.name()
                                && sample.status == "passed"
                        })
                        .collect::<Vec<_>>();
                    if samples.is_empty() {
                        continue;
                    }
                    let p50_latency_ms =
                        percentile_for_samples(&samples, |sample| sample.latency_ms);
                    if let Some(latency) = p50_latency_ms
                        && fastest_latency.is_none_or(|fastest| latency < fastest)
                    {
                        fastest_latency = Some(latency);
                        fastest_lane = Some(lane.name.clone());
                    }
                    lane_metrics.push(NormalizedAgenticGateLaneMetric {
                        lane: lane.name.clone(),
                        pass_count: samples.len(),
                        p50_latency_ms,
                        latency_delta_vs_fastest_ms: None,
                        p50_first_byte_latency_ms: percentile_for_samples(&samples, |sample| {
                            sample.stream_timing.first_byte_latency_ms
                        }),
                        p50_first_semantic_delta_latency_ms: percentile_for_samples(
                            &samples,
                            |sample| sample.stream_timing.first_semantic_delta_latency_ms,
                        ),
                        p50_first_tool_delta_latency_ms: percentile_for_samples(
                            &samples,
                            |sample| sample.stream_timing.first_tool_delta_latency_ms,
                        ),
                        avg_tokens_per_second: average_f64(
                            samples.iter().filter_map(|sample| sample.tokens_per_second),
                        ),
                        avg_cached_tokens: average_u64(
                            samples.iter().filter_map(|sample| sample.cached_tokens),
                        ),
                        avg_prompt_tokens: average_u64(
                            samples.iter().filter_map(|sample| sample.prompt_tokens),
                        ),
                        avg_completion_tokens: average_u64(
                            samples.iter().filter_map(|sample| sample.completion_tokens),
                        ),
                    });
                }
                if lane_metrics.is_empty() {
                    continue;
                }
                if let Some(fastest) = fastest_latency {
                    for metric in &mut lane_metrics {
                        metric.latency_delta_vs_fastest_ms = metric
                            .p50_latency_ms
                            .map(|latency| latency.saturating_sub(fastest));
                    }
                }
                rows.push(NormalizedAgenticGateRow {
                    case: probe.case.name(),
                    schema_variant: probe.schema_variant.name(),
                    tool_choice_variant: probe.tool_choice_variant.name(),
                    max_tokens: probe.max_tokens,
                    cache_phase: phase.name(),
                    run_mode: run_mode.name(),
                    fastest_lane: fastest_lane.clone(),
                    lanes: lane_metrics,
                });
            }
        }
    }
    NormalizedAgenticGateReport {
        status: if rows.is_empty() {
            "no_samples"
        } else {
            "reported"
        }
        .to_owned(),
        rows,
    }
}
