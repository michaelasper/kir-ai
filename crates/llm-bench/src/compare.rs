use crate::{
    BaselineComparisonReport, BenchCaseReport, BenchLaneCacheMemoryReport,
    BenchLaneCaseComparisonReport, BenchLaneCaseMetricReport, BenchLaneComparisonReport,
    BenchLaneReport, BenchReport, HardwareReport, ModelIdentityReport,
};
use serde_json::Value;

pub(crate) fn compare_bench_lanes(lanes: &[BenchLaneReport]) -> BenchLaneComparisonReport {
    let artifact_identity_match = lane_artifact_identity_matches(lanes);
    let mut comparisons = Vec::new();
    let Some(first_lane) = lanes.first() else {
        return BenchLaneComparisonReport {
            status: "no_lanes".to_owned(),
            artifact_identity_match,
            cases: comparisons,
        };
    };
    for profile in &first_lane.profiles {
        for case in &profile.cases {
            let mut lane_metrics = Vec::new();
            let mut fastest_lane = None;
            let mut fastest_latency = None;
            let first_lane_kv_cache = first_lane
                .cache_metrics
                .as_ref()
                .map(|metrics| &metrics.kv_cache);
            for lane in lanes {
                let Some(lane_case) = find_lane_case(lane, profile.name, case.name) else {
                    continue;
                };
                if lane_case.status == "passed"
                    && let Some(latency) = lane_case.latency_ms
                    && fastest_latency.is_none_or(|fastest| latency < fastest)
                {
                    fastest_latency = Some(latency);
                    fastest_lane = Some(lane.name.clone());
                }
                lane_metrics.push(BenchLaneCaseMetricReport {
                    lane: lane.name.clone(),
                    status: lane_case.status.clone(),
                    latency_ms: lane_case.latency_ms,
                    stream_timing: lane_case.stream_timing,
                    tokens_per_second: lane_case.tokens_per_second,
                    prompt_tokens: lane_case.prompt_tokens,
                    completion_tokens: lane_case.completion_tokens,
                    total_tokens: lane_case.total_tokens,
                    cached_tokens_status: lane_case.cached_tokens_status,
                    cached_tokens: lane_case.cached_tokens,
                    cache_memory: lane.cache_metrics.as_ref().map(|metrics| {
                        BenchLaneCacheMemoryReport::from_kv_cache(
                            &metrics.kv_cache,
                            first_lane_kv_cache,
                        )
                    }),
                    classification: lane_case.classification.clone(),
                });
            }
            comparisons.push(BenchLaneCaseComparisonReport {
                profile: profile.name,
                case: case.name,
                lanes: lane_metrics,
                fastest_lane,
            });
        }
    }
    BenchLaneComparisonReport {
        status: if lanes.len() > 1 {
            if artifact_identity_match {
                "comparable".to_owned()
            } else {
                "artifact_identity_mismatch".to_owned()
            }
        } else {
            "single_lane".to_owned()
        },
        artifact_identity_match,
        cases: comparisons,
    }
}

pub(crate) fn bench_gate_status(
    release_blocking_failed: bool,
    comparison: &BenchLaneComparisonReport,
) -> &'static str {
    if bench_gate_failure_classification(release_blocking_failed, comparison).is_some() {
        "failed"
    } else {
        "passed"
    }
}

pub(crate) fn bench_gate_failure_classification(
    release_blocking_failed: bool,
    comparison: &BenchLaneComparisonReport,
) -> Option<&'static str> {
    if release_blocking_failed {
        Some("release_blocking_case_failed")
    } else if comparison.status == "artifact_identity_mismatch" {
        Some("lane_artifact_identity_mismatch")
    } else {
        None
    }
}

fn lane_artifact_identity_matches(lanes: &[BenchLaneReport]) -> bool {
    let Some(first) = lanes.first() else {
        return false;
    };
    lanes.iter().all(|lane| {
        lane.model.repo_id == first.model.repo_id
            && lane.model.resolved_commit == first.model.resolved_commit
            && lane.model.profile == first.model.profile
            && lane.model.quantization == first.model.quantization
    })
}

fn find_lane_case<'a>(
    lane: &'a BenchLaneReport,
    profile_name: &str,
    case_name: &str,
) -> Option<&'a BenchCaseReport> {
    lane.profiles
        .iter()
        .find(|profile| profile.name == profile_name)?
        .cases
        .iter()
        .find(|case| case.name == case_name)
}

pub(crate) fn baseline_status(report: &BenchReport, baseline: Option<&Value>) -> String {
    let Some(baseline) = baseline else {
        return report.baseline.status.clone();
    };
    let hardware_match = baseline
        .get("hardware")
        .and_then(|hardware| {
            Some(
                hardware.get("os")?.as_str()? == report.hardware.os
                    && hardware.get("arch")?.as_str()? == report.hardware.arch,
            )
        })
        .unwrap_or(false);
    let model_class_match = baseline_model_class_matches(baseline, &report.model);
    match (hardware_match, model_class_match) {
        (true, true) => "loaded".to_owned(),
        (false, true) => "loaded_hardware_mismatch".to_owned(),
        (true, false) => "loaded_model_class_mismatch".to_owned(),
        (false, false) => "loaded_hardware_and_model_class_mismatch".to_owned(),
    }
}

pub(crate) fn apply_baseline_comparison(
    case: &mut BenchCaseReport,
    baseline: Option<&Value>,
    profile_name: &str,
    hardware: &HardwareReport,
    model: &ModelIdentityReport,
    latency_regression_threshold: f64,
) {
    let Some(baseline) = baseline else {
        return;
    };
    let Some(baseline_case) = find_baseline_case(baseline, profile_name, case.name) else {
        case.baseline = Some(BaselineComparisonReport {
            status: "missing_case".to_owned(),
            baseline_status: None,
            baseline_latency_ms: None,
            baseline_tokens_per_second: None,
            hardware_match: baseline_hardware_matches(baseline, hardware),
            model_class_match: baseline_model_class_matches(baseline, model),
        });
        return;
    };
    let baseline_status = baseline_case
        .get("status")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let baseline_latency_ms = baseline_case.get("latency_ms").and_then(Value::as_u64);
    let baseline_tps = baseline_case
        .get("tokens_per_second")
        .and_then(Value::as_f64);
    let hardware_match = baseline_hardware_matches(baseline, hardware);
    let model_class_match = baseline_model_class_matches(baseline, model);
    let status = if !hardware_match {
        "hardware_mismatch"
    } else if !model_class_match {
        "model_class_mismatch"
    } else if baseline_status.as_deref() == Some("passed") && case.status != "passed" {
        "regression"
    } else if let (Some(baseline_latency), Some(current_latency)) = (
        baseline_latency_ms,
        case.latency_ms
            .and_then(|latency| u64::try_from(latency).ok()),
    ) {
        let allowed = baseline_latency as f64 * (1.0 + latency_regression_threshold);
        if current_latency as f64 > allowed {
            "latency_regression"
        } else {
            "within_baseline"
        }
    } else {
        "not_comparable"
    };
    case.baseline = Some(BaselineComparisonReport {
        status: status.to_owned(),
        baseline_status,
        baseline_latency_ms: baseline_latency_ms.map(u128::from),
        baseline_tokens_per_second: baseline_tps,
        hardware_match,
        model_class_match,
    });
}

fn find_baseline_case<'a>(
    baseline: &'a Value,
    profile_name: &str,
    case_name: &str,
) -> Option<&'a Value> {
    baseline
        .get("profiles")?
        .as_array()?
        .iter()
        .find(|profile| profile.get("name").and_then(Value::as_str) == Some(profile_name))?
        .get("cases")?
        .as_array()?
        .iter()
        .find(|case| case.get("name").and_then(Value::as_str) == Some(case_name))
}

fn baseline_hardware_matches(baseline: &Value, hardware: &HardwareReport) -> bool {
    baseline
        .get("hardware")
        .and_then(|value| {
            Some(
                value.get("os")?.as_str()? == hardware.os
                    && value.get("arch")?.as_str()? == hardware.arch,
            )
        })
        .unwrap_or(false)
}

fn baseline_model_class_matches(baseline: &Value, model: &ModelIdentityReport) -> bool {
    let Some(baseline_model) = baseline.get("model") else {
        return false;
    };
    let baseline_family = baseline_model.get("family").and_then(Value::as_str);
    let baseline_profile = baseline_model.get("profile").and_then(Value::as_str);
    match (
        baseline_family,
        model.family.as_deref(),
        baseline_profile,
        model.profile.as_deref(),
    ) {
        (Some(left_family), Some(right_family), Some(left_profile), Some(right_profile)) => {
            left_family == right_family && left_profile == right_profile
        }
        _ => baseline_model.get("id").and_then(Value::as_str) == Some(model.id.as_str()),
    }
}
