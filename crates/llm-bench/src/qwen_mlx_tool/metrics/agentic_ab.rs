use super::*;

pub(in crate::qwen_mlx_tool) async fn load_agentic_streaming_fast_path_ab_report(
    baseline_path: Option<&Path>,
    candidate_lanes: &[NormalizedLaneReport],
) -> anyhow::Result<NormalizedAgenticStreamingFastPathAbReport> {
    let Some(path) = baseline_path else {
        return Ok(NormalizedAgenticStreamingFastPathAbReport::not_configured());
    };
    let bytes = tokio::fs::read(path)
        .await
        .with_context(|| format!("read A/B baseline trace {}", path.display()))?;
    let baseline = serde_json::from_slice::<ComparableBenchReport>(&bytes)
        .with_context(|| format!("parse A/B baseline trace {}", path.display()))?;
    let candidate = comparable_lanes_from_normalized(candidate_lanes);
    Ok(agentic_streaming_fast_path_ab_report(
        Some(path.display().to_string()),
        &baseline.lanes,
        &candidate,
    ))
}

pub(in crate::qwen_mlx_tool) fn agentic_streaming_fast_path_ab_report(
    baseline_path: Option<String>,
    baseline_lanes: &[ComparableLaneReport],
    candidate_lanes: &[ComparableLaneReport],
) -> NormalizedAgenticStreamingFastPathAbReport {
    let baseline_by_name = baseline_lanes
        .iter()
        .map(|lane| (lane.name.as_str(), lane))
        .collect::<BTreeMap<_, _>>();
    let mut rows = Vec::new();
    let mut top_level_failures = Vec::new();

    for candidate_lane in candidate_lanes {
        let Some(baseline_lane) = baseline_by_name.get(candidate_lane.name.as_str()).copied()
        else {
            top_level_failures.push(format!("missing_baseline_lane:{}", candidate_lane.name));
            continue;
        };
        for (cache_phase, run_mode) in agentic_ab_group_keys(baseline_lane, candidate_lane) {
            rows.push(agentic_streaming_fast_path_ab_row(
                baseline_lane,
                candidate_lane,
                &cache_phase,
                &run_mode,
            ));
        }
    }

    if rows.is_empty() {
        top_level_failures.push("missing_comparison_samples".to_owned());
    }
    if !rows
        .iter()
        .any(|row| row.assertion_role == "fast_path_candidate")
    {
        top_level_failures.push("missing_fast_path_candidate_lane".to_owned());
    }

    let mut failure_reasons = top_level_failures;
    for row in &rows {
        for reason in &row.failure_reasons {
            failure_reasons.push(format!(
                "{}:{}:{}:{reason}",
                row.lane, row.cache_phase, row.run_mode
            ));
        }
    }
    failure_reasons.sort();
    failure_reasons.dedup();

    NormalizedAgenticStreamingFastPathAbReport {
        status: if failure_reasons.is_empty() {
            "passed"
        } else {
            "failed"
        }
        .to_owned(),
        baseline_path,
        case: AGENTIC_AB_CASE,
        schema_variant: AGENTIC_AB_SCHEMA_VARIANT,
        tool_choice_variant: AGENTIC_AB_TOOL_CHOICE_VARIANT,
        rows,
        failure_reasons,
    }
}

pub(in crate::qwen_mlx_tool) fn agentic_streaming_fast_path_ab_row(
    baseline_lane: &ComparableLaneReport,
    candidate_lane: &ComparableLaneReport,
    cache_phase: &str,
    run_mode: &str,
) -> NormalizedAgenticStreamingFastPathAbRow {
    let baseline_samples = agentic_ab_samples(baseline_lane, cache_phase, run_mode);
    let candidate_samples = agentic_ab_samples(candidate_lane, cache_phase, run_mode);
    let baseline_first_tool_delta =
        percentile_for_comparable_samples(&baseline_samples, |sample| {
            sample.stream_timing.first_tool_delta_latency_ms
        });
    let candidate_first_tool_delta =
        percentile_for_comparable_samples(&candidate_samples, |sample| {
            sample.stream_timing.first_tool_delta_latency_ms
        });
    let baseline_tool_finish = percentile_for_comparable_samples(&baseline_samples, |sample| {
        sample.stream_timing.tool_finish_latency_ms
    });
    let candidate_tool_finish = percentile_for_comparable_samples(&candidate_samples, |sample| {
        sample.stream_timing.tool_finish_latency_ms
    });

    let assertion_role = if candidate_lane.kind == AGENTIC_AB_FAST_PATH_KIND {
        "fast_path_candidate"
    } else {
        "control"
    };
    let first_tool_delta_advanced = if assertion_role == "fast_path_candidate" {
        Some(matches!(
            (baseline_first_tool_delta, candidate_first_tool_delta),
            (Some(baseline), Some(candidate)) if candidate < baseline
        ))
    } else {
        None
    };
    let baseline_validation = ToolValidationSignature::from_samples(&baseline_samples);
    let candidate_validation = ToolValidationSignature::from_samples(&candidate_samples);
    let final_validation_unchanged = baseline_validation == candidate_validation
        && baseline_validation.successful_tool_stream()
        && candidate_validation.successful_tool_stream();

    let mut failure_reasons = Vec::new();
    if baseline_samples.is_empty() {
        failure_reasons.push("missing_baseline_samples".to_owned());
    }
    if candidate_samples.is_empty() {
        failure_reasons.push("missing_candidate_samples".to_owned());
    }
    if assertion_role == "fast_path_candidate" && first_tool_delta_advanced != Some(true) {
        failure_reasons.push("first_tool_delta_not_advanced".to_owned());
    }
    if !final_validation_unchanged {
        failure_reasons.push("final_validation_changed".to_owned());
    }

    NormalizedAgenticStreamingFastPathAbRow {
        lane: candidate_lane.name.clone(),
        kind: candidate_lane.kind.clone(),
        assertion_role,
        cache_phase: cache_phase.to_owned(),
        run_mode: run_mode.to_owned(),
        baseline_sample_count: baseline_samples.len(),
        candidate_sample_count: candidate_samples.len(),
        baseline_pass_count: baseline_validation.pass_count,
        candidate_pass_count: candidate_validation.pass_count,
        baseline_fail_count: baseline_validation.fail_count,
        candidate_fail_count: candidate_validation.fail_count,
        baseline_p50_first_tool_delta_latency_ms: baseline_first_tool_delta,
        candidate_p50_first_tool_delta_latency_ms: candidate_first_tool_delta,
        first_tool_delta_delta_ms: metric_delta_ms(
            candidate_first_tool_delta,
            baseline_first_tool_delta,
        ),
        first_tool_delta_advanced,
        baseline_p50_tool_finish_latency_ms: baseline_tool_finish,
        candidate_p50_tool_finish_latency_ms: candidate_tool_finish,
        tool_finish_delta_ms: metric_delta_ms(candidate_tool_finish, baseline_tool_finish),
        final_validation_unchanged,
        failure_reasons,
    }
}

pub(in crate::qwen_mlx_tool) fn agentic_ab_group_keys(
    baseline_lane: &ComparableLaneReport,
    candidate_lane: &ComparableLaneReport,
) -> BTreeSet<(String, String)> {
    baseline_lane
        .all_samples()
        .chain(candidate_lane.all_samples())
        .filter(|sample| sample.matches_agentic_ab_probe())
        .map(|sample| (sample.cache_phase.clone(), sample.run_mode.clone()))
        .collect()
}

pub(in crate::qwen_mlx_tool) fn agentic_ab_samples<'a>(
    lane: &'a ComparableLaneReport,
    cache_phase: &str,
    run_mode: &str,
) -> Vec<&'a ComparableSampleReport> {
    lane.all_samples()
        .filter(|sample| {
            sample.matches_agentic_ab_probe()
                && sample.cache_phase == cache_phase
                && sample.run_mode == run_mode
        })
        .collect()
}

pub(in crate::qwen_mlx_tool) fn percentile_for_comparable_samples(
    samples: &[&ComparableSampleReport],
    value: impl Fn(&ComparableSampleReport) -> Option<u128>,
) -> Option<u128> {
    let mut values = samples
        .iter()
        .filter_map(|sample| value(sample))
        .collect::<Vec<_>>();
    values.sort_unstable();
    percentile_latency(&values, 0.50)
}

pub(in crate::qwen_mlx_tool) fn metric_delta_ms(
    candidate: Option<u128>,
    baseline: Option<u128>,
) -> Option<i64> {
    let candidate = i64::try_from(candidate?).ok()?;
    let baseline = i64::try_from(baseline?).ok()?;
    Some(candidate - baseline)
}

pub(in crate::qwen_mlx_tool) fn comparable_lanes_from_normalized(
    lanes: &[NormalizedLaneReport],
) -> Vec<ComparableLaneReport> {
    lanes
        .iter()
        .map(|lane| ComparableLaneReport {
            name: lane.name.clone(),
            kind: lane.kind.to_owned(),
            samples: lane
                .samples
                .iter()
                .map(comparable_sample_from_normalized)
                .collect(),
            concurrent_samples: lane
                .concurrent_samples
                .iter()
                .map(comparable_sample_from_normalized)
                .collect(),
        })
        .collect()
}

pub(in crate::qwen_mlx_tool) fn comparable_sample_from_normalized(
    sample: &NormalizedSampleReport,
) -> ComparableSampleReport {
    ComparableSampleReport {
        case: sample.case.to_owned(),
        schema_variant: sample.schema_variant.to_owned(),
        tool_choice_variant: sample.tool_choice_variant.to_owned(),
        cache_phase: sample.cache_phase.to_owned(),
        run_mode: sample.run_mode.to_owned(),
        status: sample.status.clone(),
        classification: sample.classification.clone(),
        stream_timing: sample.stream_timing,
        finish_reason: sample.finish_reason.clone(),
    }
}

#[derive(Debug, Deserialize)]
pub(in crate::qwen_mlx_tool) struct ComparableBenchReport {
    #[serde(default)]
    pub(in crate::qwen_mlx_tool) lanes: Vec<ComparableLaneReport>,
}

#[derive(Debug, Clone, Deserialize)]
pub(in crate::qwen_mlx_tool) struct ComparableLaneReport {
    #[serde(default)]
    pub(in crate::qwen_mlx_tool) name: String,
    #[serde(default)]
    pub(in crate::qwen_mlx_tool) kind: String,
    #[serde(default)]
    pub(in crate::qwen_mlx_tool) samples: Vec<ComparableSampleReport>,
    #[serde(default)]
    pub(in crate::qwen_mlx_tool) concurrent_samples: Vec<ComparableSampleReport>,
}

impl ComparableLaneReport {
    pub(in crate::qwen_mlx_tool) fn all_samples(
        &self,
    ) -> impl Iterator<Item = &ComparableSampleReport> {
        self.samples.iter().chain(self.concurrent_samples.iter())
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(in crate::qwen_mlx_tool) struct ComparableSampleReport {
    #[serde(default)]
    pub(in crate::qwen_mlx_tool) case: String,
    #[serde(default)]
    pub(in crate::qwen_mlx_tool) schema_variant: String,
    #[serde(default)]
    pub(in crate::qwen_mlx_tool) tool_choice_variant: String,
    #[serde(default)]
    pub(in crate::qwen_mlx_tool) cache_phase: String,
    #[serde(default)]
    pub(in crate::qwen_mlx_tool) run_mode: String,
    #[serde(default)]
    pub(in crate::qwen_mlx_tool) status: String,
    #[serde(default)]
    pub(in crate::qwen_mlx_tool) classification: String,
    #[serde(default, flatten)]
    pub(in crate::qwen_mlx_tool) stream_timing: StreamTimingReport,
    #[serde(default)]
    pub(in crate::qwen_mlx_tool) finish_reason: Option<String>,
}

impl ComparableSampleReport {
    pub(in crate::qwen_mlx_tool) fn matches_agentic_ab_probe(&self) -> bool {
        self.case == AGENTIC_AB_CASE
            && self.schema_variant == AGENTIC_AB_SCHEMA_VARIANT
            && self.tool_choice_variant == AGENTIC_AB_TOOL_CHOICE_VARIANT
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(in crate::qwen_mlx_tool) struct ToolValidationSignature {
    pub(in crate::qwen_mlx_tool) sample_count: usize,
    pub(in crate::qwen_mlx_tool) pass_count: usize,
    pub(in crate::qwen_mlx_tool) fail_count: usize,
    pub(in crate::qwen_mlx_tool) status_counts: BTreeMap<String, usize>,
    pub(in crate::qwen_mlx_tool) classification_counts: BTreeMap<String, usize>,
    pub(in crate::qwen_mlx_tool) finish_reason_counts: BTreeMap<String, usize>,
}

impl ToolValidationSignature {
    pub(in crate::qwen_mlx_tool) fn from_samples(samples: &[&ComparableSampleReport]) -> Self {
        let mut signature = Self {
            sample_count: samples.len(),
            ..Self::default()
        };
        for sample in samples {
            if sample.status == "passed" {
                signature.pass_count += 1;
            }
            if sample.status == "failed" {
                signature.fail_count += 1;
            }
            increment_count(&mut signature.status_counts, &sample.status);
            increment_count(&mut signature.classification_counts, &sample.classification);
            increment_count(
                &mut signature.finish_reason_counts,
                sample.finish_reason.as_deref().unwrap_or("<missing>"),
            );
        }
        signature
    }

    pub(in crate::qwen_mlx_tool) fn successful_tool_stream(&self) -> bool {
        self.sample_count > 0
            && self.pass_count == self.sample_count
            && self.fail_count == 0
            && self
                .finish_reason_counts
                .get("tool_calls")
                .is_some_and(|count| *count == self.sample_count)
    }
}

pub(in crate::qwen_mlx_tool) fn increment_count(counts: &mut BTreeMap<String, usize>, value: &str) {
    *counts.entry(value.to_owned()).or_insert(0) += 1;
}
