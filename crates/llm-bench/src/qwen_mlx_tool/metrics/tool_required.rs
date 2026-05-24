use super::*;

pub(in crate::qwen_mlx_tool) fn required_tool_ttft_matrix_report(
    lanes: &[NormalizedLaneReport],
    probes: &[NormalizedProbePlan],
    phases: &[CachePhase],
) -> NormalizedRequiredToolTtftMatrixReport {
    let rows = lanes
        .iter()
        .flat_map(|lane| {
            lane_samples(lane)
                .filter(|sample| required_tool_ttft_sample_selected(sample, probes, phases))
                .map(|sample| required_tool_ttft_matrix_row(lane, sample, lanes))
        })
        .collect::<Vec<_>>();
    let has_samples = !rows.is_empty();
    let has_admin = rows.iter().any(|row| row.validated_tool_call_ms.is_some());
    let status = match (has_samples, has_admin) {
        (false, _) => "no_samples",
        (true, true) => "reported",
        (true, false) => "client_only",
    };
    NormalizedRequiredToolTtftMatrixReport {
        status: status.to_owned(),
        rows,
    }
}

pub(in crate::qwen_mlx_tool) fn required_tool_ttft_sample_selected(
    sample: &NormalizedSampleReport,
    probes: &[NormalizedProbePlan],
    phases: &[CachePhase],
) -> bool {
    sample.case == NormalizedCaseKind::ToolRequiredStream.name()
        && phases
            .iter()
            .any(|phase| sample.cache_phase == phase.name())
        && probes
            .iter()
            .any(|&probe| sample_matches_probe(sample, probe))
}

pub(in crate::qwen_mlx_tool) fn required_tool_ttft_matrix_row(
    lane: &NormalizedLaneReport,
    sample: &NormalizedSampleReport,
    lanes: &[NormalizedLaneReport],
) -> NormalizedRequiredToolTtftMatrixRow {
    let fastest_tool_delta_ms = fastest_required_tool_ttft_delta(lanes, sample);
    let latency_delta_vs_fastest_lane_ms = sample
        .stream_timing
        .first_tool_delta_latency_ms
        .zip(fastest_tool_delta_ms)
        .map(|(latency, fastest)| latency.saturating_sub(fastest));
    let admin_metrics = sample
        .tool_required_stream_admin_metrics
        .as_ref()
        .and_then(normalized_tool_stream_admin_metrics);
    let stream_stalled_requests_delta = sample
        .tool_required_stream_admin_metrics
        .as_ref()
        .and_then(|capture| admin_counter_delta(capture, &["stream_stalled_requests"]));
    let no_progress_failures_delta = sample
        .tool_required_stream_admin_metrics
        .as_ref()
        .and_then(|capture| admin_counter_delta(capture, &["no_progress_failures"]));

    NormalizedRequiredToolTtftMatrixRow {
        lane: lane.name.clone(),
        kind: lane.kind,
        tool_parser: lane.tool_parser,
        schema_variant: sample.schema_variant,
        tool_choice_variant: sample.tool_choice_variant,
        max_tokens: sample.max_tokens,
        cache_phase: sample.cache_phase,
        run_mode: sample.run_mode,
        sample_index: sample.sample_index,
        request_index: sample.request_index,
        status: sample.status.clone(),
        classification: sample.classification.clone(),
        first_response_byte_ms: sample.stream_timing.first_byte_latency_ms,
        first_parsed_sse_chunk_ms: sample.stream_timing.first_sse_data_latency_ms,
        first_tool_delta_ms: sample.stream_timing.first_tool_delta_latency_ms,
        tool_finish_ms: sample.stream_timing.tool_finish_latency_ms,
        validated_tool_call_ms: admin_metrics.and_then(|metrics| {
            metrics
                .validated_tool_call_ms
                .window_avg_ms
                .or(metrics.validated_tool_call_ms.avg_ms_after)
        }),
        latency_delta_vs_fastest_lane_ms,
        finish_reason: sample.finish_reason.clone(),
        stream_stalled_requests_delta,
        no_progress_failures_delta,
        error: sample.error.clone(),
    }
}

pub(in crate::qwen_mlx_tool) fn fastest_required_tool_ttft_delta(
    lanes: &[NormalizedLaneReport],
    target: &NormalizedSampleReport,
) -> Option<u128> {
    lane_samples_for_all_lanes(lanes)
        .filter(|sample| {
            sample.case == target.case
                && sample.schema_variant == target.schema_variant
                && sample.tool_choice_variant == target.tool_choice_variant
                && sample.max_tokens == target.max_tokens
                && sample.cache_phase == target.cache_phase
                && sample.run_mode == target.run_mode
                && sample.status == "passed"
        })
        .filter_map(|sample| sample.stream_timing.first_tool_delta_latency_ms)
        .min()
}

pub(in crate::qwen_mlx_tool) fn lane_samples_for_all_lanes(
    lanes: &[NormalizedLaneReport],
) -> impl Iterator<Item = &NormalizedSampleReport> {
    lanes.iter().flat_map(lane_samples)
}

pub(in crate::qwen_mlx_tool) fn tool_required_stream_timing_report(
    lanes: &[NormalizedLaneReport],
) -> NormalizedToolRequiredStreamTimingReport {
    let lane_reports = lanes
        .iter()
        .map(tool_required_stream_lane_timing_report)
        .collect::<Vec<_>>();
    let has_admin = lane_reports.iter().any(|lane| lane.admin_metrics.is_some());
    let has_admin_error = lane_reports
        .iter()
        .any(|lane| lane.admin_metrics_error.is_some());
    let has_samples = lane_reports.iter().any(|lane| lane.pass_count > 0);
    let status = match (has_admin, has_admin_error, has_samples) {
        (true, true, _) => "partial_admin_metrics",
        (true, false, _) => "reported",
        (false, true, _) => "admin_metrics_unavailable",
        (false, false, true) => "client_only",
        (false, false, false) => "no_samples",
    };
    let attribution = tool_required_stream_attribution_report(lanes, &lane_reports);
    NormalizedToolRequiredStreamTimingReport {
        status: status.to_owned(),
        attribution,
        lanes: lane_reports,
    }
}

pub(in crate::qwen_mlx_tool) fn tool_required_stream_lane_timing_report(
    lane: &NormalizedLaneReport,
) -> NormalizedToolRequiredStreamLaneTimingReport {
    let samples = lane_samples(lane)
        .filter(|sample| {
            sample.case == NormalizedCaseKind::ToolRequiredStream.name()
                && sample.status == "passed"
        })
        .collect::<Vec<_>>();
    NormalizedToolRequiredStreamLaneTimingReport {
        lane: lane.name.clone(),
        kind: lane.kind,
        pass_count: samples.len(),
        p50_first_byte_latency_ms: percentile_for_samples(&samples, |sample| {
            sample.stream_timing.first_byte_latency_ms
        }),
        p50_first_sse_data_latency_ms: percentile_for_samples(&samples, |sample| {
            sample.stream_timing.first_sse_data_latency_ms
        }),
        p50_first_tool_delta_latency_ms: percentile_for_samples(&samples, |sample| {
            sample.stream_timing.first_tool_delta_latency_ms
        }),
        p50_tool_finish_latency_ms: percentile_for_samples(&samples, |sample| {
            sample.stream_timing.tool_finish_latency_ms
        }),
        p50_first_semantic_delta_latency_ms: percentile_for_samples(&samples, |sample| {
            sample.stream_timing.first_semantic_delta_latency_ms
        }),
        admin_metrics: normalized_tool_stream_admin_metrics(&lane.admin_metrics),
        admin_metrics_error: lane.admin_metrics.error.clone(),
        tool_stream_observations: matching_tool_stream_observations(lane, &samples),
    }
}

pub(in crate::qwen_mlx_tool) fn tool_required_stream_attribution_report(
    lanes: &[NormalizedLaneReport],
    lane_reports: &[NormalizedToolRequiredStreamLaneTimingReport],
) -> NormalizedToolRequiredStreamAttributionReport {
    let rows = lanes
        .iter()
        .zip(lane_reports)
        .map(tool_required_stream_attribution_row)
        .collect::<Vec<_>>();
    let has_admin = rows.iter().any(|row| row.admin_metrics.is_some());
    let has_admin_error = rows.iter().any(|row| row.admin_metrics_error.is_some());
    let has_samples = rows.iter().any(|row| row.pass_count > 0);
    let status = match (has_admin, has_admin_error, has_samples) {
        (true, true, _) => "partial_admin_metrics",
        (true, false, _) => "reported",
        (false, true, _) => "admin_metrics_unavailable",
        (false, false, true) => "client_only",
        (false, false, false) => "no_samples",
    };
    NormalizedToolRequiredStreamAttributionReport {
        status: status.to_owned(),
        rows,
    }
}

pub(in crate::qwen_mlx_tool) fn tool_required_stream_attribution_row(
    (lane, lane_report): (
        &NormalizedLaneReport,
        &NormalizedToolRequiredStreamLaneTimingReport,
    ),
) -> NormalizedToolRequiredStreamAttributionRow {
    let samples = lane_samples(lane)
        .filter(|sample| {
            sample.case == NormalizedCaseKind::ToolRequiredStream.name()
                && sample.status == "passed"
        })
        .collect::<Vec<_>>();
    let client = NormalizedToolRequiredStreamClientTiming {
        first_byte_ms: lane_report.p50_first_byte_latency_ms,
        first_sse_data_ms: lane_report.p50_first_sse_data_latency_ms,
        first_tool_delta_ms: lane_report.p50_first_tool_delta_latency_ms,
        tool_finish_ms: lane_report.p50_tool_finish_latency_ms,
    };
    let (admin_metrics_scope, admin_metrics, admin_metrics_error) =
        tool_required_stream_attribution_admin_metrics(lane, &samples);
    let first_tool_delta_gap_ms = tool_required_stream_gap(&client, admin_metrics.as_ref());
    let decision = tool_required_stream_attribution_decision(
        client.first_tool_delta_ms,
        admin_metrics.as_ref(),
        admin_metrics_error.as_deref(),
    );

    NormalizedToolRequiredStreamAttributionRow {
        lane: lane.name.clone(),
        kind: lane.kind,
        pass_count: samples.len(),
        client,
        admin_metrics_scope,
        admin_metrics,
        admin_metrics_error,
        first_tool_delta_gap_ms,
        decision,
    }
}

pub(in crate::qwen_mlx_tool) fn tool_required_stream_attribution_admin_metrics(
    lane: &NormalizedLaneReport,
    samples: &[&NormalizedSampleReport],
) -> (
    &'static str,
    Option<NormalizedToolRequiredStreamAdminMetrics>,
    Option<String>,
) {
    let sample_captures = samples
        .iter()
        .filter_map(|sample| sample.tool_required_stream_admin_metrics.as_ref())
        .collect::<Vec<_>>();
    let sample_metrics = sample_captures
        .iter()
        .filter_map(|capture| normalized_tool_stream_admin_metrics(capture))
        .collect::<Vec<_>>();
    if !sample_metrics.is_empty() {
        return (
            "per_sample",
            Some(aggregate_tool_stream_admin_metrics(&sample_metrics)),
            join_admin_metric_errors(&sample_captures),
        );
    }
    if !sample_captures.is_empty() {
        return (
            "per_sample_unavailable",
            None,
            join_admin_metric_errors(&sample_captures),
        );
    }

    (
        "lane_window",
        normalized_tool_stream_admin_metrics(&lane.admin_metrics),
        lane.admin_metrics.error.clone(),
    )
}

pub(in crate::qwen_mlx_tool) fn join_admin_metric_errors(
    captures: &[&NormalizedAdminMetricsCapture],
) -> Option<String> {
    let errors = captures
        .iter()
        .filter_map(|capture| capture.error.as_deref())
        .collect::<Vec<_>>();
    (!errors.is_empty()).then(|| errors.join("; "))
}

pub(in crate::qwen_mlx_tool) fn aggregate_tool_stream_admin_metrics(
    metrics: &[NormalizedToolRequiredStreamAdminMetrics],
) -> NormalizedToolRequiredStreamAdminMetrics {
    NormalizedToolRequiredStreamAdminMetrics {
        first_tool_delta_ms: aggregate_admin_latency_metric(
            metrics.iter().map(|metric| metric.first_tool_delta_ms),
        ),
        first_tool_delta_after_ttft_ms: aggregate_admin_latency_metric(
            metrics
                .iter()
                .map(|metric| metric.first_tool_delta_after_ttft_ms),
        ),
        tool_argument_assembly_ms: aggregate_admin_latency_metric(
            metrics
                .iter()
                .map(|metric| metric.tool_argument_assembly_ms),
        ),
        tool_intent_fill_ms: aggregate_admin_latency_metric(
            metrics.iter().map(|metric| metric.tool_intent_fill_ms),
        ),
        tool_schema_validation_ms: aggregate_admin_latency_metric(
            metrics
                .iter()
                .map(|metric| metric.tool_schema_validation_ms),
        ),
        tool_finish_ms: aggregate_admin_latency_metric(
            metrics.iter().map(|metric| metric.tool_finish_ms),
        ),
        validated_tool_call_ms: aggregate_admin_latency_metric(
            metrics.iter().map(|metric| metric.validated_tool_call_ms),
        ),
        mlx_stream_first_upstream_byte_ms: aggregate_admin_latency_metric(
            metrics
                .iter()
                .map(|metric| metric.mlx_stream_first_upstream_byte_ms),
        ),
        mlx_stream_first_parsed_chunk_ms: aggregate_admin_latency_metric(
            metrics
                .iter()
                .map(|metric| metric.mlx_stream_first_parsed_chunk_ms),
        ),
        mlx_stream_first_tool_delta_ms: aggregate_admin_latency_metric(
            metrics
                .iter()
                .map(|metric| metric.mlx_stream_first_tool_delta_ms),
        ),
    }
}

pub(in crate::qwen_mlx_tool) fn aggregate_admin_latency_metric(
    metrics: impl Iterator<Item = NormalizedAdminLatencyMetricReport>,
) -> NormalizedAdminLatencyMetricReport {
    let metrics = metrics.collect::<Vec<_>>();
    NormalizedAdminLatencyMetricReport {
        count_delta: sum_present_i64(metrics.iter().map(|metric| metric.count_delta)),
        count_after: max_present_u64(metrics.iter().map(|metric| metric.count_after)),
        min_ms_after: min_present_f64(metrics.iter().map(|metric| metric.min_ms_after)),
        max_ms_after: max_present_f64(metrics.iter().map(|metric| metric.max_ms_after)),
        avg_ms_after: avg_present_f64(metrics.iter().map(|metric| metric.avg_ms_after)),
        window_avg_ms: avg_present_f64(metrics.iter().map(|metric| metric.window_avg_ms)),
    }
}

pub(in crate::qwen_mlx_tool) fn tool_required_stream_gap(
    client: &NormalizedToolRequiredStreamClientTiming,
    admin_metrics: Option<&NormalizedToolRequiredStreamAdminMetrics>,
) -> NormalizedToolRequiredStreamGap {
    let client_tool = client.first_tool_delta_ms.map(|value| value as f64);
    let client_finish = client.tool_finish_ms.map(|value| value as f64);
    NormalizedToolRequiredStreamGap {
        mlx_stream_to_client_ms: metric_gap_ms(
            client_tool,
            admin_metrics.and_then(|metrics| metrics.mlx_stream_first_tool_delta_ms.window_avg_ms),
        ),
        kir_first_tool_delta_to_client_ms: metric_gap_ms(
            client_tool,
            admin_metrics.and_then(|metrics| metrics.first_tool_delta_ms.window_avg_ms),
        ),
        validated_tool_call_to_tool_finish_ms: metric_gap_ms(
            client_finish,
            admin_metrics.and_then(|metrics| metrics.validated_tool_call_ms.window_avg_ms),
        ),
    }
}

pub(in crate::qwen_mlx_tool) fn tool_required_stream_attribution_decision(
    client_first_tool_delta_ms: Option<u128>,
    admin_metrics: Option<&NormalizedToolRequiredStreamAdminMetrics>,
    admin_metrics_error: Option<&str>,
) -> &'static str {
    let Some(client_ms) = client_first_tool_delta_ms else {
        return "missing_client_first_tool_delta";
    };
    let Some(upstream_ms) =
        admin_metrics.and_then(|metrics| metrics.mlx_stream_first_tool_delta_ms.window_avg_ms)
    else {
        return if admin_metrics_error.is_some() {
            "admin_metrics_unavailable"
        } else {
            "missing_mlx_stream_first_tool_delta"
        };
    };

    let client_ms = client_ms as f64;
    let gap_ms = client_ms - upstream_ms;
    let aligned_threshold_ms = (client_ms * 0.10).max(50.0);
    if gap_ms.abs() <= aligned_threshold_ms {
        "upstream_aligned_with_client"
    } else if gap_ms > aligned_threshold_ms {
        "kir_buffering_or_validation_gap"
    } else {
        "upstream_slower_than_client"
    }
}

pub(in crate::qwen_mlx_tool) fn metric_gap_ms(
    client_ms: Option<f64>,
    metric_ms: Option<f64>,
) -> Option<f64> {
    Some(client_ms? - metric_ms?)
}

pub(in crate::qwen_mlx_tool) fn sum_present_i64(
    values: impl Iterator<Item = Option<i64>>,
) -> Option<i64> {
    let mut found = false;
    let mut sum = 0;
    for value in values.flatten() {
        found = true;
        sum += value;
    }
    found.then_some(sum)
}

pub(in crate::qwen_mlx_tool) fn max_present_u64(
    values: impl Iterator<Item = Option<u64>>,
) -> Option<u64> {
    values.flatten().max()
}

pub(in crate::qwen_mlx_tool) fn min_present_f64(
    values: impl Iterator<Item = Option<f64>>,
) -> Option<f64> {
    values.flatten().reduce(f64::min)
}

pub(in crate::qwen_mlx_tool) fn max_present_f64(
    values: impl Iterator<Item = Option<f64>>,
) -> Option<f64> {
    values.flatten().reduce(f64::max)
}

pub(in crate::qwen_mlx_tool) fn avg_present_f64(
    values: impl Iterator<Item = Option<f64>>,
) -> Option<f64> {
    let mut count = 0;
    let mut sum = 0.0;
    for value in values.flatten() {
        count += 1;
        sum += value;
    }
    (count > 0).then_some(sum / f64::from(count))
}

pub(in crate::qwen_mlx_tool) fn matching_tool_stream_observations(
    lane: &NormalizedLaneReport,
    samples: &[&NormalizedSampleReport],
) -> Vec<NormalizedToolStreamObservation> {
    if lane.kind != NormalizedLaneKind::KirAiProxy.as_str() {
        return Vec::new();
    }
    let request_ids = samples
        .iter()
        .filter_map(|sample| sample.request_id.as_deref())
        .collect::<Vec<_>>();
    if request_ids.is_empty() {
        return Vec::new();
    }
    lane.admin_metrics
        .after
        .as_ref()
        .and_then(|metrics| metrics.pointer("/tool_stream/recent"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|value| tool_stream_observation(value, &request_ids, samples))
        .collect()
}

pub(in crate::qwen_mlx_tool) fn tool_stream_observation(
    value: &Value,
    request_ids: &[&str],
    samples: &[&NormalizedSampleReport],
) -> Option<NormalizedToolStreamObservation> {
    let request_id = value.get("request_id")?.as_str()?;
    if !request_ids.contains(&request_id) {
        return None;
    }
    let sample = samples
        .iter()
        .find(|sample| sample.request_id.as_deref() == Some(request_id))?;
    Some(NormalizedToolStreamObservation {
        request_id: request_id.to_owned(),
        model: value.get("model")?.as_str()?.to_owned(),
        streamed: value.get("streamed")?.as_bool()?,
        client_first_byte_ms: sample.stream_timing.first_byte_latency_ms,
        client_first_sse_data_ms: sample.stream_timing.first_sse_data_latency_ms,
        client_visible_first_tool_delta_ms: sample.stream_timing.first_tool_delta_latency_ms,
        client_tool_finish_ms: sample.stream_timing.tool_finish_latency_ms,
        kir_first_tool_delta_ms: value.get("kir_first_tool_delta_ms").and_then(Value::as_u64),
        kir_first_tool_delta_after_ttft_ms: value
            .get("kir_first_tool_delta_after_ttft_ms")
            .and_then(Value::as_u64),
        tool_argument_assembly_ms: value
            .get("tool_argument_assembly_ms")
            .and_then(Value::as_u64),
        tool_intent_fill_ms: value.get("tool_intent_fill_ms").and_then(Value::as_u64),
        tool_schema_validation_ms: value
            .get("tool_schema_validation_ms")
            .and_then(Value::as_u64),
        tool_finish_ms: value.get("tool_finish_ms").and_then(Value::as_u64),
        validated_tool_call_ms: value.get("validated_tool_call_ms").and_then(Value::as_u64),
        mlx_response_headers_ms: value.get("mlx_response_headers_ms").and_then(Value::as_u64),
        mlx_first_upstream_byte_ms: value
            .get("mlx_first_upstream_byte_ms")
            .and_then(Value::as_u64),
        mlx_first_parsed_chunk_ms: value
            .get("mlx_first_parsed_chunk_ms")
            .and_then(Value::as_u64),
        mlx_first_tool_delta_ms: value.get("mlx_first_tool_delta_ms").and_then(Value::as_u64),
        mlx_upstream_complete_ms: value
            .get("mlx_upstream_complete_ms")
            .and_then(Value::as_u64),
        latency_ms: value.get("latency_ms")?.as_u64()?,
    })
}

pub(in crate::qwen_mlx_tool) fn normalized_tool_stream_admin_metrics(
    capture: &NormalizedAdminMetricsCapture,
) -> Option<NormalizedToolRequiredStreamAdminMetrics> {
    let after = capture.after.as_ref()?;
    Some(NormalizedToolRequiredStreamAdminMetrics {
        first_tool_delta_ms: admin_latency_metric(
            capture.before.as_ref(),
            after,
            &["first_tool_delta_ms"],
        ),
        first_tool_delta_after_ttft_ms: admin_latency_metric(
            capture.before.as_ref(),
            after,
            &["first_tool_delta_after_ttft_ms"],
        ),
        tool_argument_assembly_ms: admin_latency_metric(
            capture.before.as_ref(),
            after,
            &["tool_argument_assembly_ms"],
        ),
        tool_intent_fill_ms: admin_latency_metric(
            capture.before.as_ref(),
            after,
            &["tool_intent_fill_ms"],
        ),
        tool_schema_validation_ms: admin_latency_metric(
            capture.before.as_ref(),
            after,
            &["tool_schema_validation_ms"],
        ),
        tool_finish_ms: admin_latency_metric(capture.before.as_ref(), after, &["tool_finish_ms"]),
        validated_tool_call_ms: admin_latency_metric(
            capture.before.as_ref(),
            after,
            &["validated_tool_call_ms"],
        ),
        mlx_stream_first_upstream_byte_ms: admin_latency_metric(
            capture.before.as_ref(),
            after,
            &["mlx", "stream_first_upstream_byte_ms"],
        ),
        mlx_stream_first_parsed_chunk_ms: admin_latency_metric(
            capture.before.as_ref(),
            after,
            &["mlx", "stream_first_parsed_chunk_ms"],
        ),
        mlx_stream_first_tool_delta_ms: admin_latency_metric(
            capture.before.as_ref(),
            after,
            &["mlx", "stream_first_tool_delta_ms"],
        ),
    })
}

pub(in crate::qwen_mlx_tool) fn admin_latency_metric(
    before: Option<&Value>,
    after: &Value,
    path: &[&str],
) -> NormalizedAdminLatencyMetricReport {
    let before_summary = before.and_then(|value| value_path(value, path));
    let after_summary = value_path(after, path);
    let before_count = before_summary.and_then(metric_count);
    let after_count = after_summary.and_then(metric_count);
    let before_avg = before_summary
        .and_then(|summary| summary.get("avg"))
        .and_then(Value::as_f64);
    let after_avg = after_summary
        .and_then(|summary| summary.get("avg"))
        .and_then(Value::as_f64);
    NormalizedAdminLatencyMetricReport {
        count_delta: match (before_count, after_count) {
            (Some(before_count), Some(after_count)) => Some(after_count - before_count),
            _ => None,
        },
        count_after: after_count.and_then(|count| u64::try_from(count).ok()),
        min_ms_after: after_summary
            .and_then(|summary| summary.get("min"))
            .and_then(Value::as_f64),
        max_ms_after: after_summary
            .and_then(|summary| summary.get("max"))
            .and_then(Value::as_f64),
        avg_ms_after: after_avg,
        window_avg_ms: admin_latency_window_avg_ms(
            before_count,
            before_avg,
            after_count,
            after_avg,
        ),
    }
}

pub(in crate::qwen_mlx_tool) fn admin_latency_window_avg_ms(
    before_count: Option<i64>,
    before_avg_ms: Option<f64>,
    after_count: Option<i64>,
    after_avg_ms: Option<f64>,
) -> Option<f64> {
    let before_count = before_count?;
    let after_count = after_count?;
    let count_delta = after_count.checked_sub(before_count)?;
    if count_delta <= 0 {
        return None;
    }
    let before_total = before_avg_ms? * before_count as f64;
    let after_total = after_avg_ms? * after_count as f64;
    let window_total = after_total - before_total;
    (window_total >= 0.0).then_some(window_total / count_delta as f64)
}

pub(in crate::qwen_mlx_tool) fn value_path<'a>(
    value: &'a Value,
    path: &[&str],
) -> Option<&'a Value> {
    if let Some(found) = value_path_direct(value, path) {
        return Some(found);
    }
    let (first, rest) = path.split_first()?;
    (*first == "mlx").then_some(())?;
    value_path_direct(value.get("backend_metrics")?.get("mlx")?, rest)
}

pub(in crate::qwen_mlx_tool) fn value_path_direct<'a>(
    mut value: &'a Value,
    path: &[&str],
) -> Option<&'a Value> {
    for segment in path {
        value = value.get(*segment)?;
    }
    Some(value)
}

pub(in crate::qwen_mlx_tool) fn metric_count(value: &Value) -> Option<i64> {
    value.get("count").and_then(Value::as_i64).or_else(|| {
        value
            .get("count")
            .and_then(Value::as_u64)
            .and_then(|count| i64::try_from(count).ok())
    })
}
