use super::*;

pub(in crate::qwen_mlx_tool) fn cache_status_counts(
    samples: &[&NormalizedSampleReport],
    observations: &[NormalizedStablePrefixRequestCacheObservation],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for sample in samples {
        *counts
            .entry(cache_status_for_sample(sample, observations))
            .or_insert(0) += 1;
    }
    counts
}

pub(in crate::qwen_mlx_tool) fn sample_failure_classification_counts(
    samples: &[&NormalizedSampleReport],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for sample in samples {
        if let Some(classification) = &sample.failure_classification {
            *counts.entry(classification.clone()).or_insert(0) += 1;
        }
    }
    counts
}

pub(in crate::qwen_mlx_tool) fn cache_status_for_sample(
    sample: &NormalizedSampleReport,
    observations: &[NormalizedStablePrefixRequestCacheObservation],
) -> String {
    if let Some(status) = cache_status_from_sample(sample) {
        return status.to_owned();
    }
    request_cache_observation_for_sample(sample, observations)
        .map(|observation| observation.cache_status.clone())
        .unwrap_or_else(|| "unknown".to_owned())
}

pub(in crate::qwen_mlx_tool) fn cache_status_from_sample(
    sample: &NormalizedSampleReport,
) -> Option<&'static str> {
    if sample.cached_tokens_status != "present" {
        return None;
    }
    Some(match (sample.prompt_tokens, sample.cached_tokens) {
        (_, Some(0)) => "miss",
        (Some(prompt), Some(cached)) if cached >= prompt => "hit",
        (Some(_), Some(_)) => "partial",
        _ => "unknown",
    })
}

pub(in crate::qwen_mlx_tool) fn sample_cached_tokens(
    sample: &NormalizedSampleReport,
    observations: &[NormalizedStablePrefixRequestCacheObservation],
) -> Option<u64> {
    sample.cached_tokens.or_else(|| {
        request_cache_observation_for_sample(sample, observations)
            .and_then(|observation| observation.cached_tokens)
    })
}

pub(in crate::qwen_mlx_tool) fn sample_direct_uncached_tokens(
    sample: &NormalizedSampleReport,
) -> Option<u64> {
    Some(sample.prompt_tokens?.saturating_sub(sample.cached_tokens?))
}

pub(in crate::qwen_mlx_tool) fn sample_uncached_tokens(
    sample: &NormalizedSampleReport,
    observations: &[NormalizedStablePrefixRequestCacheObservation],
) -> Option<u64> {
    if let Some(cached_tokens) = sample.cached_tokens {
        return Some(sample.prompt_tokens?.saturating_sub(cached_tokens));
    }
    request_cache_observation_for_sample(sample, observations)
        .and_then(|observation| observation.uncached_tokens)
}

pub(in crate::qwen_mlx_tool) fn request_cache_observation_for_sample<'a>(
    sample: &NormalizedSampleReport,
    observations: &'a [NormalizedStablePrefixRequestCacheObservation],
) -> Option<&'a NormalizedStablePrefixRequestCacheObservation> {
    let request_id = sample.request_id.as_deref()?;
    observations
        .iter()
        .find(|observation| observation.request_id == request_id)
}

pub(in crate::qwen_mlx_tool) fn matching_request_cache_observations(
    lane: &NormalizedLaneReport,
    samples: &[&NormalizedSampleReport],
) -> Vec<NormalizedStablePrefixRequestCacheObservation> {
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
        .and_then(|metrics| metrics.pointer("/request_cache/recent"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|value| stable_prefix_request_cache_observation(value, &request_ids))
        .collect()
}

pub(in crate::qwen_mlx_tool) fn stable_prefix_request_cache_observation(
    value: &Value,
    request_ids: &[&str],
) -> Option<NormalizedStablePrefixRequestCacheObservation> {
    let request_id = value.get("request_id")?.as_str()?;
    if !request_ids.contains(&request_id) {
        return None;
    }
    Some(NormalizedStablePrefixRequestCacheObservation {
        request_id: request_id.to_owned(),
        model: value.get("model")?.as_str()?.to_owned(),
        streamed: value.get("streamed")?.as_bool()?,
        prompt_tokens: value.get("prompt_tokens")?.as_u64()?,
        cached_tokens: value.get("cached_tokens").and_then(Value::as_u64),
        uncached_tokens: value.get("uncached_tokens").and_then(Value::as_u64),
        cache_status: value.get("cache_status")?.as_str()?.to_owned(),
        latency_ms: value.get("latency_ms")?.as_u64()?,
    })
}

pub(in crate::qwen_mlx_tool) fn normalized_prefill_admin_mlx_timing(
    capture: &NormalizedAdminMetricsCapture,
) -> Option<NormalizedPrefillSweepAdminMlxTiming> {
    let after = capture.after.as_ref()?;
    Some(NormalizedPrefillSweepAdminMlxTiming {
        stream_first_upstream_byte_ms: admin_latency_metric(
            capture.before.as_ref(),
            after,
            &["mlx", "stream_first_upstream_byte_ms"],
        ),
        stream_first_parsed_chunk_ms: admin_latency_metric(
            capture.before.as_ref(),
            after,
            &["mlx", "stream_first_parsed_chunk_ms"],
        ),
        stream_first_tool_delta_ms: admin_latency_metric(
            capture.before.as_ref(),
            after,
            &["mlx", "stream_first_tool_delta_ms"],
        ),
    })
}

pub(in crate::qwen_mlx_tool) fn admin_counter_delta(
    capture: &NormalizedAdminMetricsCapture,
    path: &[&str],
) -> Option<i64> {
    let before = capture
        .before
        .as_ref()
        .and_then(|value| value_path(value, path))
        .and_then(value_i64);
    let after = capture
        .after
        .as_ref()
        .and_then(|value| value_path(value, path))
        .and_then(value_i64);
    match (before, after) {
        (Some(before), Some(after)) => Some(after - before),
        _ => None,
    }
}

pub(in crate::qwen_mlx_tool) fn admin_counter_after(
    capture: &NormalizedAdminMetricsCapture,
    path: &[&str],
) -> Option<u64> {
    capture
        .after
        .as_ref()
        .and_then(|value| value_path(value, path))
        .and_then(Value::as_u64)
}

pub(in crate::qwen_mlx_tool) fn admin_number_delta(
    capture: &NormalizedAdminMetricsCapture,
    path: &[&str],
) -> Option<f64> {
    let before = capture
        .before
        .as_ref()
        .and_then(|value| value_path(value, path))
        .and_then(Value::as_f64);
    let after = capture
        .after
        .as_ref()
        .and_then(|value| value_path(value, path))
        .and_then(Value::as_f64);
    Some(after? - before?)
}

pub(in crate::qwen_mlx_tool) fn admin_number_after(
    capture: &NormalizedAdminMetricsCapture,
    path: &[&str],
) -> Option<f64> {
    capture
        .after
        .as_ref()
        .and_then(|value| value_path(value, path))
        .and_then(Value::as_f64)
}

pub(in crate::qwen_mlx_tool) fn scheduler_prefill_counters(
    capture: &NormalizedAdminMetricsCapture,
) -> NormalizedSchedulerPrefillCountersReport {
    NormalizedSchedulerPrefillCountersReport {
        prefill_yields_delta: admin_counter_delta(capture, &["scheduler_prefill_yields"]),
        prefill_yields_after: admin_counter_after(capture, &["scheduler_prefill_yields"]),
        prefill_yields_to_decode_delta: admin_counter_delta(
            capture,
            &["scheduler_prefill_yields_to_decode"],
        ),
        prefill_yields_to_decode_after: admin_counter_after(
            capture,
            &["scheduler_prefill_yields_to_decode"],
        ),
        prefill_yield_reacquire_waits_delta: admin_counter_delta(
            capture,
            &["scheduler_prefill_yield_reacquire_waits"],
        ),
        prefill_yield_reacquire_waits_after: admin_counter_after(
            capture,
            &["scheduler_prefill_yield_reacquire_waits"],
        ),
        prefill_yield_reacquire_wait_ms_total_delta: admin_number_delta(
            capture,
            &["scheduler_prefill_yield_reacquire_wait_ms_total"],
        ),
        prefill_yield_reacquire_wait_ms_total_after: admin_number_after(
            capture,
            &["scheduler_prefill_yield_reacquire_wait_ms_total"],
        ),
        prefill_yield_reacquire_wait_ms_max_after: admin_number_after(
            capture,
            &["scheduler_prefill_yield_reacquire_wait_ms_max"],
        ),
    }
}

pub(in crate::qwen_mlx_tool) fn checkpoint_reuse_counters(
    capture: &NormalizedAdminMetricsCapture,
) -> NormalizedCheckpointReuseCountersReport {
    NormalizedCheckpointReuseCountersReport {
        checkpoint_reuse_hits_delta: prefix_cache_counter_delta(capture, "checkpoint_reuse_hits"),
        checkpoint_reuse_hits_after: prefix_cache_counter_after(capture, "checkpoint_reuse_hits"),
        checkpoint_reused_tokens_delta: prefix_cache_counter_delta(
            capture,
            "checkpoint_reused_tokens",
        ),
        checkpoint_reused_tokens_after: prefix_cache_counter_after(
            capture,
            "checkpoint_reused_tokens",
        ),
        avoided_prefill_tokens_delta: prefix_cache_counter_delta(capture, "avoided_prefill_tokens"),
        avoided_prefill_tokens_after: prefix_cache_counter_after(capture, "avoided_prefill_tokens"),
    }
}

pub(in crate::qwen_mlx_tool) fn prefix_cache_counter_delta(
    capture: &NormalizedAdminMetricsCapture,
    field: &str,
) -> Option<i64> {
    let before = capture
        .before
        .as_ref()
        .and_then(prefix_cache_metrics_value)
        .and_then(|value| value.get(field))
        .and_then(value_i64);
    let after = capture
        .after
        .as_ref()
        .and_then(prefix_cache_metrics_value)
        .and_then(|value| value.get(field))
        .and_then(value_i64);
    Some(after? - before?)
}

pub(in crate::qwen_mlx_tool) fn prefix_cache_counter_after(
    capture: &NormalizedAdminMetricsCapture,
    field: &str,
) -> Option<u64> {
    capture
        .after
        .as_ref()
        .and_then(prefix_cache_metrics_value)
        .and_then(|value| value.get(field))
        .and_then(Value::as_u64)
}

pub(in crate::qwen_mlx_tool) fn prefix_cache_metrics_value(metrics: &Value) -> Option<&Value> {
    let backend_metrics = metrics.get("backend_metrics").unwrap_or(metrics);
    backend_metrics
        .get("native_text_prefix_cache")
        .and_then(|native_text| native_text.get("qwen"))
        .or_else(|| backend_metrics.get("native_qwen_prefix_cache"))
}

pub(in crate::qwen_mlx_tool) fn value_i64(value: &Value) -> Option<i64> {
    value
        .as_i64()
        .or_else(|| value.as_u64().and_then(|value| i64::try_from(value).ok()))
}
