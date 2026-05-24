use super::*;

pub(in crate::qwen_mlx_tool) fn latest_performance_comparison_report(
    lanes: &[NormalizedLaneReport],
    engine_db_baselines: Option<&EngineDbBaselineExport>,
) -> NormalizedLatestPerformanceComparisonReport {
    let mut rows = Vec::new();
    for lane in lanes {
        if let Some(row) = latest_plain_stream_row(lane) {
            rows.push(row);
        }
        if let Some(row) = latest_tool_stream_row(lane) {
            rows.push(row);
        }
        if let Some(row) = latest_prefix_cache_row(lane) {
            rows.push(row);
        }
    }
    if let Some(export) = engine_db_baselines {
        rows.extend(export.rows.iter().map(engine_db_baseline_comparison_row));
    }
    let evidence = NormalizedLatestPerformanceEvidence::from_rows(&rows);
    let status = if rows.is_empty() {
        "no_samples"
    } else if evidence.has_kir_latest
        && evidence.has_direct_mlx_latest
        && evidence.has_engine_db_baselines
        && evidence.has_ttfi_ms
        && evidence.has_cache_metrics
        && evidence.has_tokens_per_second
    {
        "reported"
    } else {
        "partial"
    };
    NormalizedLatestPerformanceComparisonReport {
        status: status.to_owned(),
        engine_db_baseline_source: engine_db_baselines.and_then(|export| export.source.clone()),
        evidence,
        rows,
    }
}

pub(in crate::qwen_mlx_tool) fn latest_plain_stream_row(
    lane: &NormalizedLaneReport,
) -> Option<NormalizedLatestPerformanceComparisonRow> {
    let source_kind = latest_source_kind(lane.kind)?;
    let samples =
        latest_passed_samples_prefer_phase(lane, NormalizedCaseKind::ChatStream, CachePhase::Cold);
    if samples.is_empty() {
        return None;
    }
    let mut row = latest_live_comparison_row(lane, source_kind, "plain_stream");
    row.ttfi_ms = optional_u128_as_f64(percentile_for_samples(&samples, |sample| {
        sample.stream_timing.first_semantic_delta_latency_ms
    }));
    row.total_latency_ms =
        optional_u128_as_f64(percentile_for_samples(&samples, |sample| sample.latency_ms));
    row.tokens_per_second =
        average_f64(samples.iter().filter_map(|sample| sample.tokens_per_second));
    Some(row)
}

pub(in crate::qwen_mlx_tool) fn latest_tool_stream_row(
    lane: &NormalizedLaneReport,
) -> Option<NormalizedLatestPerformanceComparisonRow> {
    let source_kind = latest_source_kind(lane.kind)?;
    let samples = latest_passed_samples_prefer_phase(
        lane,
        NormalizedCaseKind::ToolRequiredStream,
        CachePhase::Cold,
    );
    if samples.is_empty() {
        return None;
    }
    let mut row = latest_live_comparison_row(lane, source_kind, "required_tool_stream");
    row.ttfi_ms = optional_u128_as_f64(percentile_for_samples(&samples, |sample| {
        sample.stream_timing.first_semantic_delta_latency_ms
    }));
    row.first_tool_delta_ms = optional_u128_as_f64(percentile_for_samples(&samples, |sample| {
        sample.stream_timing.first_tool_delta_latency_ms
    }));
    row.validated_tool_call_ms =
        optional_u128_as_f64(percentile_for_samples(&samples, |sample| sample.latency_ms));
    row.total_latency_ms = row.validated_tool_call_ms;
    row.tokens_per_second =
        average_f64(samples.iter().filter_map(|sample| sample.tokens_per_second));
    Some(row)
}

pub(in crate::qwen_mlx_tool) fn latest_prefix_cache_row(
    lane: &NormalizedLaneReport,
) -> Option<NormalizedLatestPerformanceComparisonRow> {
    let source_kind = latest_source_kind(lane.kind)?;
    let case = latest_cache_case(lane)?;
    let cold = latest_passed_samples(lane, case, Some(CachePhase::Cold));
    let warm = latest_warm_cache_samples(lane, case);
    if cold.is_empty() && warm.is_empty() {
        return None;
    }
    let metric_samples = if warm.is_empty() { &cold } else { &warm };
    let cold_latency =
        optional_u128_as_f64(percentile_for_samples(&cold, |sample| sample.latency_ms));
    let warm_latency =
        optional_u128_as_f64(percentile_for_samples(&warm, |sample| sample.latency_ms));
    let mut row = latest_live_comparison_row(lane, source_kind, "prefix_cache");
    row.ttfi_ms = optional_u128_as_f64(percentile_for_samples(metric_samples, |sample| {
        sample.stream_timing.first_semantic_delta_latency_ms
    }));
    row.total_latency_ms = optional_u128_as_f64(percentile_for_samples(metric_samples, |sample| {
        sample.latency_ms
    }));
    row.tokens_per_second = average_f64(
        metric_samples
            .iter()
            .filter_map(|sample| sample.tokens_per_second),
    );
    row.cache_cold_latency_ms = cold_latency;
    row.cache_warm_latency_ms = warm_latency;
    row.cache_speedup = cache_speedup(cold_latency, warm_latency, None);
    row.cached_tokens = metric_samples
        .iter()
        .filter_map(|sample| sample.cached_tokens)
        .max();
    Some(row)
}

pub(in crate::qwen_mlx_tool) fn latest_live_comparison_row(
    lane: &NormalizedLaneReport,
    source_kind: &str,
    probe: &str,
) -> NormalizedLatestPerformanceComparisonRow {
    NormalizedLatestPerformanceComparisonRow {
        source_kind: source_kind.to_owned(),
        lane: Some(lane.name.clone()),
        kind: Some(lane.kind.to_owned()),
        engine: None,
        profile: None,
        model: Some(lane.effective_request_model_id.clone()),
        probe: probe.to_owned(),
        ttfi_ms: None,
        first_tool_delta_ms: None,
        validated_tool_call_ms: None,
        total_latency_ms: None,
        tokens_per_second: None,
        cache_cold_latency_ms: None,
        cache_warm_latency_ms: None,
        cache_speedup: None,
        cached_tokens: None,
        notes: None,
    }
}

pub(in crate::qwen_mlx_tool) fn engine_db_baseline_comparison_row(
    baseline: &EngineDbBaselineRow,
) -> NormalizedLatestPerformanceComparisonRow {
    NormalizedLatestPerformanceComparisonRow {
        source_kind: "engine_db_baseline".to_owned(),
        lane: None,
        kind: None,
        engine: Some(baseline.engine.clone()),
        profile: Some(baseline.profile.clone()),
        model: baseline.model.clone(),
        probe: baseline.probe.clone(),
        ttfi_ms: baseline.ttfi_ms,
        first_tool_delta_ms: baseline.first_tool_delta_ms,
        validated_tool_call_ms: baseline.validated_tool_call_ms,
        total_latency_ms: baseline.total_latency_ms,
        tokens_per_second: baseline.tokens_per_second,
        cache_cold_latency_ms: baseline.cache_cold_latency_ms,
        cache_warm_latency_ms: baseline.cache_warm_latency_ms,
        cache_speedup: cache_speedup(
            baseline.cache_cold_latency_ms,
            baseline.cache_warm_latency_ms,
            baseline.cache_speedup,
        ),
        cached_tokens: baseline.cached_tokens,
        notes: baseline.notes.clone(),
    }
}

pub(in crate::qwen_mlx_tool) fn latest_source_kind(kind: &str) -> Option<&'static str> {
    match kind {
        "kir_ai_proxy" => Some("latest_kir"),
        "direct_mlx" => Some("direct_mlx"),
        _ => None,
    }
}

pub(in crate::qwen_mlx_tool) fn latest_passed_samples_prefer_phase(
    lane: &NormalizedLaneReport,
    case: NormalizedCaseKind,
    preferred_phase: CachePhase,
) -> Vec<&NormalizedSampleReport> {
    let preferred = latest_passed_samples(lane, case, Some(preferred_phase));
    if preferred.is_empty() {
        latest_passed_samples(lane, case, None)
    } else {
        preferred
    }
}

pub(in crate::qwen_mlx_tool) fn latest_passed_samples(
    lane: &NormalizedLaneReport,
    case: NormalizedCaseKind,
    phase: Option<CachePhase>,
) -> Vec<&NormalizedSampleReport> {
    lane_samples(lane)
        .filter(|sample| {
            sample.status == "passed"
                && sample.case == case.name()
                && phase.is_none_or(|phase| sample.cache_phase == phase.name())
        })
        .collect()
}

pub(in crate::qwen_mlx_tool) fn latest_cache_case(
    lane: &NormalizedLaneReport,
) -> Option<NormalizedCaseKind> {
    [
        NormalizedCaseKind::WarmPrefixRepeatedTurnStream,
        NormalizedCaseKind::OmpRepeatedPrefix,
    ]
    .into_iter()
    .find(|case| !latest_passed_samples(lane, *case, None).is_empty())
}

pub(in crate::qwen_mlx_tool) fn latest_warm_cache_samples(
    lane: &NormalizedLaneReport,
    case: NormalizedCaseKind,
) -> Vec<&NormalizedSampleReport> {
    let warm_same_prompt = latest_passed_samples(lane, case, Some(CachePhase::WarmSamePrompt));
    if warm_same_prompt.is_empty() {
        latest_passed_samples(lane, case, Some(CachePhase::WarmSameToolSchema))
    } else {
        warm_same_prompt
    }
}

pub(in crate::qwen_mlx_tool) fn optional_u128_as_f64(value: Option<u128>) -> Option<f64> {
    value.map(|value| value as f64)
}

pub(in crate::qwen_mlx_tool) fn cache_speedup(
    cold_latency_ms: Option<f64>,
    warm_latency_ms: Option<f64>,
    explicit_speedup: Option<f64>,
) -> Option<f64> {
    if explicit_speedup.is_some() {
        return explicit_speedup;
    }
    match (cold_latency_ms, warm_latency_ms) {
        (Some(cold), Some(warm)) if warm > 0.0 => Some(cold / warm),
        _ => None,
    }
}
