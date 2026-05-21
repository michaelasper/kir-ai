use anyhow::{Context, anyhow};
use futures::StreamExt;
use llm_hub::ModelStore;
use llm_models::{ModelFamilyAdapter, QwenFamilyAdapter};
use llm_server::EngineOptions;
use llm_tokenizer::HuggingFaceTokenizer;
pub use llm_util::defaults::DEFAULT_MODEL_ID;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

mod cli;
mod qwen_mlx_tool;

use cli::{
    flag_values, normalize_endpoint, parse_f64_flag, parse_u32_flag, parse_u64_flag,
    parse_usize_flag, print_bench_help,
};

const GATE_NAME: &str = "qwen-long-context";
const CACHE_LAYOUT: &str = "shared-prefix-v1";
const DEFAULT_TIMEOUT_MS: u64 = 30 * 60 * 1000;
const DEFAULT_CONNECT_TIMEOUT_MS: u64 = 10 * 1000;
const DEFAULT_MAX_TOKENS: u32 = 128;
const DEFAULT_LATENCY_REGRESSION_THRESHOLD: f64 = 0.20;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum MlxToolParserMode {
    #[default]
    Auto,
    Json,
    QwenXml,
}

impl MlxToolParserMode {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "auto" => Some(Self::Auto),
            "json" => Some(Self::Json),
            "qwen-xml" => Some(Self::QwenXml),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Json => "json",
            Self::QwenXml => "qwen-xml",
        }
    }
}

pub(crate) fn flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find_map(|window| (window[0] == flag).then_some(window[1].as_str()))
}

pub(crate) fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|arg| arg == flag)
}

pub async fn run_bench_command(args: Vec<String>) -> anyhow::Result<()> {
    let Some(subcommand) = args.first() else {
        print_bench_help();
        return Ok(());
    };
    if subcommand == "--help" || subcommand == "-h" {
        print_bench_help();
        return Ok(());
    }
    match subcommand.as_str() {
        "qwen-long-context" => run_qwen_long_context_bench(&args[1..]).await,
        "qwen-mlx-tool-normalized" => {
            qwen_mlx_tool::run_qwen_mlx_tool_normalized_bench(&args[1..]).await
        }
        other => anyhow::bail!("unknown bench subcommand `{other}`"),
    }
}

async fn run_qwen_long_context_bench(args: &[String]) -> anyhow::Result<()> {
    if has_flag(args, "--help") || has_flag(args, "-h") {
        print_bench_help();
        return Ok(());
    }

    let dry_run = has_flag(args, "--dry-run");
    let selected_profiles = selected_profiles(flag_value(args, "--profile").unwrap_or("135k"))?;
    let model_id = flag_value(args, "--model")
        .or_else(|| flag_value(args, "--model-id"))
        .unwrap_or(DEFAULT_MODEL_ID)
        .to_owned();
    let endpoint = flag_value(args, "--endpoint").map(normalize_endpoint);
    let snapshot_path = flag_value(args, "--snapshot").map(PathBuf::from);
    let lanes = parse_lane_configs(args, &model_id, endpoint, snapshot_path)?;
    let baseline_path = flag_value(args, "--baseline").map(PathBuf::from);
    let output_path = flag_value(args, "--output").map(PathBuf::from);
    let admin_token = flag_value(args, "--admin-token").map(str::to_owned);
    let timeout_ms = parse_u64_flag(args, "--timeout-ms", DEFAULT_TIMEOUT_MS)?;
    let connect_timeout_ms =
        parse_u64_flag(args, "--connect-timeout-ms", DEFAULT_CONNECT_TIMEOUT_MS)?;
    let max_tokens = parse_u32_flag(args, "--max-tokens", DEFAULT_MAX_TOKENS)?;
    let latency_regression_threshold = parse_f64_flag(
        args,
        "--latency-regression-threshold",
        DEFAULT_LATENCY_REGRESSION_THRESHOLD,
    )?;
    let run_controls = BenchRunControlsReport {
        warmup_count: 0,
        repetitions: 1,
        timeout_ms,
        connect_timeout_ms,
        max_tokens,
        latency_regression_threshold,
    };
    let scheduler = SchedulerSettingsReport::from_args(args)?;

    if !dry_run {
        for lane in &lanes {
            if lane.endpoint.is_none() {
                anyhow::bail!(
                    "benchmark lane `{}` requires endpoint=<url> or top-level --endpoint <url>",
                    lane.name
                );
            }
            if lane.snapshot_path.is_none() {
                anyhow::bail!(
                    "benchmark lane `{}` requires snapshot=<path> or top-level --snapshot <path>",
                    lane.name
                );
            }
        }
    }

    let mut lane_reports = Vec::with_capacity(lanes.len());
    for lane in &lanes {
        lane_reports.push(BenchLaneReport {
            name: lane.name.clone(),
            status: if dry_run { "dry_run" } else { "planned" }.to_owned(),
            model: load_model_identity(
                &lane.model_id,
                lane.endpoint.as_deref(),
                lane.snapshot_path.as_deref(),
                dry_run,
            )
            .await?,
            profiles: selected_profiles
                .iter()
                .map(|profile| profile_report_with_max_tokens(*profile, max_tokens))
                .collect(),
            cache_metrics: None,
            admin_metrics: None,
            admin_metrics_error: None,
        });
    }
    let primary_model = lane_reports
        .first()
        .map(|lane| lane.model.clone())
        .ok_or_else(|| anyhow!("benchmark requires at least one lane"))?;
    let primary_profiles = lane_reports
        .first()
        .map(|lane| lane.profiles.clone())
        .unwrap_or_default();

    let mut report = BenchReport {
        gate: GATE_NAME,
        status: if dry_run { "dry_run" } else { "running" }.to_owned(),
        generated_at_unix_ms: unix_now_ms(),
        trace_output_path: output_path.as_ref().map(|path| path.display().to_string()),
        model: primary_model,
        hardware: HardwareReport::detect(),
        cache_policy: CachePolicyReport::from_env(),
        run_controls,
        scheduler,
        baseline: BaselineReport {
            path: baseline_path
                .as_ref()
                .map(|path| path.display().to_string()),
            status: if baseline_path.is_some() {
                if dry_run {
                    "pending".to_owned()
                } else {
                    "configured".to_owned()
                }
            } else {
                "not_configured".to_owned()
            },
            latency_regression_threshold,
        },
        profiles: primary_profiles,
        lanes: lane_reports,
        failure_classification: None,
        comparison: None,
    };

    if dry_run {
        write_and_print_report(&report, output_path.as_deref()).await?;
        return Ok(());
    }

    let baseline_trace = load_baseline_trace(baseline_path.as_deref()).await?;
    report.baseline.status = baseline_status(&report, baseline_trace.as_ref());

    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_millis(connect_timeout_ms))
        .timeout(Duration::from_millis(timeout_ms))
        .build()
        .context("build benchmark HTTP client")?;

    let mut release_blocking_failed = false;
    for (lane_config, lane_report) in lanes.iter().zip(&mut report.lanes) {
        let endpoint = lane_config
            .endpoint
            .as_deref()
            .context("lane endpoint is missing")?;
        let snapshot_path = lane_config
            .snapshot_path
            .as_deref()
            .context("lane snapshot path is missing")?;
        let tokenizer = load_qwen_tokenizer(snapshot_path)?;
        let run_context = BenchExecutionContext {
            client: &client,
            baseline_trace: baseline_trace.as_ref(),
            hardware: &report.hardware,
            latency_regression_threshold,
            max_tokens,
            admin_token: admin_token.as_deref(),
        };
        let lane_failed = run_lane_profiles(
            lane_report,
            endpoint,
            &lane_config.model_id,
            &tokenizer,
            &run_context,
        )
        .await?;
        release_blocking_failed |= lane_failed;
        capture_lane_admin_metrics(lane_report, &client, endpoint, admin_token.as_deref()).await;
    }
    if let Some(primary_lane) = report.lanes.first() {
        report.model = primary_lane.model.clone();
        report.profiles = primary_lane.profiles.clone();
    }
    let comparison = compare_bench_lanes(&report.lanes);
    let failure_classification =
        bench_gate_failure_classification(release_blocking_failed, &comparison);
    report.failure_classification = failure_classification.map(str::to_owned);
    report.status = bench_gate_status(release_blocking_failed, &comparison).to_owned();
    report.comparison = Some(comparison);

    write_and_print_report(&report, output_path.as_deref()).await?;
    if let Some(classification) = failure_classification {
        anyhow::bail!("qwen long-context promotion gate failed: {classification}");
    }
    Ok(())
}

async fn run_lane_profiles(
    lane: &mut BenchLaneReport,
    endpoint: &str,
    model_id: &str,
    tokenizer: &HuggingFaceTokenizer,
    context: &BenchExecutionContext<'_>,
) -> anyhow::Result<bool> {
    let mut release_blocking_failed = false;
    for profile in &mut lane.profiles {
        let profile_kind = BenchProfileKind::from_name(profile.name)
            .ok_or_else(|| anyhow!("unknown profile in report: {}", profile.name))?;
        let mut profile_failed = false;
        for case in &mut profile.cases {
            let case_kind = BenchCaseKind::from_name(case.name)
                .ok_or_else(|| anyhow!("unknown case in report: {}", case.name))?;
            let before_metrics =
                capture_case_prefix_metrics(context.client, endpoint, context.admin_token).await;
            let case_run = run_case(
                context.client,
                endpoint,
                model_id,
                tokenizer,
                profile_kind,
                case_kind,
                context.max_tokens,
            )
            .await;
            apply_case_run(case, case_run);
            let after_metrics =
                capture_case_prefix_metrics(context.client, endpoint, context.admin_token).await;
            apply_case_admin_metrics(case, before_metrics, after_metrics);
            apply_case_cache_metric_validation(case, case_kind);
            apply_baseline_comparison(
                case,
                context.baseline_trace,
                profile.name,
                context.hardware,
                &lane.model,
                context.latency_regression_threshold,
            );
            if profile.release_blocking && case.status != "passed" {
                profile_failed = true;
            }
        }
        profile.status = if profile.release_blocking {
            if profile_failed {
                release_blocking_failed = true;
                "failed".to_owned()
            } else {
                "passed".to_owned()
            }
        } else {
            "characterized".to_owned()
        };
    }
    lane.status = if release_blocking_failed {
        "failed".to_owned()
    } else {
        "passed".to_owned()
    };
    Ok(release_blocking_failed)
}

fn parse_lane_configs(
    args: &[String],
    default_model_id: &str,
    default_endpoint: Option<String>,
    default_snapshot_path: Option<PathBuf>,
) -> anyhow::Result<Vec<BenchLaneConfig>> {
    let lane_specs = flag_values(args, "--lane");
    if lane_specs.is_empty() {
        return Ok(vec![BenchLaneConfig {
            name: "primary".to_owned(),
            endpoint: default_endpoint,
            model_id: default_model_id.to_owned(),
            snapshot_path: default_snapshot_path,
        }]);
    }

    lane_specs
        .into_iter()
        .map(|spec| parse_lane_config(spec, default_model_id))
        .collect()
}

fn parse_lane_config(spec: &str, default_model_id: &str) -> anyhow::Result<BenchLaneConfig> {
    let mut values = BTreeMap::new();
    for part in spec.split(',') {
        let Some((key, value)) = part.split_once('=') else {
            anyhow::bail!("invalid --lane spec `{spec}`; expected comma-separated key=value pairs");
        };
        let key = key.trim();
        let value = value.trim();
        if key.is_empty() || value.is_empty() {
            anyhow::bail!("invalid --lane spec `{spec}`; keys and values must be non-empty");
        }
        values.insert(key.to_owned(), value.to_owned());
    }
    let name = values
        .remove("name")
        .ok_or_else(|| anyhow!("--lane spec `{spec}` is missing name=<id>"))?;
    let endpoint = values
        .remove("endpoint")
        .map(|value| normalize_endpoint(&value));
    let model_id = values
        .remove("model")
        .or_else(|| values.remove("model_id"))
        .unwrap_or_else(|| default_model_id.to_owned());
    let snapshot_path = values.remove("snapshot").map(PathBuf::from);
    if !values.is_empty() {
        let unknown = values.keys().cloned().collect::<Vec<_>>().join(", ");
        anyhow::bail!("--lane spec `{spec}` contains unknown keys: {unknown}");
    }
    Ok(BenchLaneConfig {
        name,
        endpoint,
        model_id,
        snapshot_path,
    })
}

async fn capture_lane_admin_metrics(
    lane: &mut BenchLaneReport,
    client: &reqwest::Client,
    endpoint: &str,
    admin_token: Option<&str>,
) {
    match fetch_admin_metrics(client, endpoint, admin_token).await {
        Ok(metrics) => {
            lane.cache_metrics = cache_metrics_from_admin(&metrics);
            lane.admin_metrics = Some(metrics);
        }
        Err(err) => {
            lane.admin_metrics_error = Some(err);
        }
    }
}

async fn capture_case_prefix_metrics(
    client: &reqwest::Client,
    endpoint: &str,
    admin_token: Option<&str>,
) -> Result<Option<PrefixCacheMetricsReport>, String> {
    fetch_admin_metrics(client, endpoint, admin_token)
        .await
        .map(|metrics| prefix_cache_metrics_from_admin(&metrics))
}

async fn fetch_admin_metrics(
    client: &reqwest::Client,
    endpoint: &str,
    admin_token: Option<&str>,
) -> Result<Value, String> {
    let url = format!("{endpoint}/admin/metrics");
    let mut request = client.get(url);
    if let Some(token) = admin_token {
        request = request.bearer_auth(token);
    }
    match request.send().await {
        Ok(response) if response.status().is_success() => match response.json::<Value>().await {
            Ok(metrics) => Ok(metrics),
            Err(err) => Err(format!("parse admin metrics: {err}")),
        },
        Ok(response) => Err(format!("admin metrics HTTP {}", response.status())),
        Err(err) => Err(format!("admin metrics request failed: {err}")),
    }
}

fn cache_metrics_from_admin(metrics: &Value) -> Option<BenchCacheMetricsReport> {
    let backend_metrics = metrics.get("backend_metrics").unwrap_or(metrics);
    let prefix = prefix_cache_metrics_value(metrics)?;
    let metal = backend_metrics
        .get("native_text_metal")
        .or_else(|| backend_metrics.get("native_qwen_metal"))?;
    let weight = metal.get("bf16_matrix_cache")?;
    let kv = metal.get("kv_cache")?;
    let linear = metal.get("linear_attention_cache")?;
    Some(BenchCacheMetricsReport {
        prefix_cache: prefix_cache_metrics_report(prefix),
        weight_cache: WeightCacheMetricsReport {
            hits: metric_u64(weight, "hits"),
            misses: metric_u64(weight, "misses"),
            hit_rate: hit_rate(metric_u64(weight, "hits"), metric_u64(weight, "misses")),
            uploads: metric_u64(weight, "uploads"),
            evictions: metric_u64(weight, "evictions"),
            bytes_uploaded: metric_u64(weight, "bytes_uploaded"),
            bytes_evicted: metric_u64(weight, "bytes_evicted"),
            resident_bytes: metric_u64(weight, "resident_bytes"),
            resident_buffers: metric_u64(weight, "resident_buffers"),
            budget_bytes: metric_u64(weight, "budget_bytes"),
        },
        kv_cache: resident_cache_metrics(kv),
        linear_attention_cache: resident_cache_metrics(linear),
        readiness: cache_readiness(prefix, weight, kv, linear),
    })
}

fn prefix_cache_metrics_from_admin(metrics: &Value) -> Option<PrefixCacheMetricsReport> {
    prefix_cache_metrics_value(metrics).map(prefix_cache_metrics_report)
}

fn prefix_cache_metrics_value(metrics: &Value) -> Option<&Value> {
    let backend_metrics = metrics.get("backend_metrics").unwrap_or(metrics);
    backend_metrics
        .get("native_text_prefix_cache")
        .and_then(|native_text| native_text.get("qwen"))
        .or_else(|| backend_metrics.get("native_qwen_prefix_cache"))
}

fn prefix_cache_metrics_report(prefix: &Value) -> PrefixCacheMetricsReport {
    PrefixCacheMetricsReport {
        hits: metric_u64(prefix, "hits"),
        misses: metric_u64(prefix, "misses"),
        hit_rate: hit_rate(metric_u64(prefix, "hits"), metric_u64(prefix, "misses")),
        stores: metric_u64(prefix, "stores"),
        evictions: metric_u64(prefix, "evictions"),
        rejected: metric_u64(prefix, "rejected"),
        reused_tokens: metric_u64(prefix, "reused_tokens"),
        hit_tokens: metric_u64(prefix, "hit_tokens"),
        miss_tokens: metric_u64(prefix, "miss_tokens"),
        avoided_prefill_tokens: metric_u64(prefix, "avoided_prefill_tokens"),
        resident_bytes: metric_u64(prefix, "resident_bytes"),
        resident_entries: metric_u64(prefix, "resident_entries"),
    }
}

fn resident_cache_metrics(cache: &Value) -> ResidentCacheMetricsReport {
    ResidentCacheMetricsReport {
        allocations: metric_u64(cache, "allocations"),
        syncs: metric_u64(cache, "syncs"),
        evictions: metric_u64(cache, "evictions"),
        bytes_uploaded: metric_u64(cache, "bytes_uploaded"),
        bytes_evicted: metric_u64(cache, "bytes_evicted"),
        resident_bytes: metric_u64(cache, "resident_bytes"),
        resident_buffers: metric_u64(cache, "resident_buffers"),
    }
}

fn cache_readiness(
    prefix: &Value,
    weight: &Value,
    kv: &Value,
    linear: &Value,
) -> CacheReadinessReport {
    let signals = [
        (
            "prefix_cache_hit_rate",
            has_metric(prefix, "hits") && has_metric(prefix, "misses"),
        ),
        (
            "prefix_cache_residency",
            has_metric(prefix, "resident_bytes") && has_metric(prefix, "resident_entries"),
        ),
        ("prefix_cache_hit_tokens", has_metric(prefix, "hit_tokens")),
        (
            "prefix_cache_miss_tokens",
            has_metric(prefix, "miss_tokens"),
        ),
        (
            "weight_cache_hit_rate",
            has_metric(weight, "hits") && has_metric(weight, "misses"),
        ),
        (
            "weight_cache_residency",
            has_metric(weight, "resident_bytes") && has_metric(weight, "budget_bytes"),
        ),
        (
            "kv_cache_residency",
            has_metric(kv, "resident_bytes") && has_metric(kv, "resident_buffers"),
        ),
        (
            "linear_attention_cache_residency",
            has_metric(linear, "resident_bytes") && has_metric(linear, "resident_buffers"),
        ),
        (
            "eviction_churn",
            has_metric(prefix, "evictions")
                && has_metric(weight, "evictions")
                && has_metric(kv, "evictions")
                && has_metric(linear, "evictions"),
        ),
    ];
    let observed_signals = signals
        .iter()
        .filter_map(|(name, present)| present.then_some(*name))
        .collect::<Vec<_>>();
    let missing_signals = signals
        .iter()
        .filter_map(|(name, present)| (!present).then_some(*name))
        .collect::<Vec<_>>();
    CacheReadinessReport {
        status: if missing_signals.is_empty() {
            "observable"
        } else {
            "partial"
        },
        observed_signals,
        missing_signals,
    }
}

fn metric_u64(value: &Value, key: &str) -> u64 {
    value.get(key).and_then(Value::as_u64).unwrap_or_default()
}

fn has_metric(value: &Value, key: &str) -> bool {
    value.get(key).and_then(Value::as_u64).is_some()
}

fn hit_rate(hits: u64, misses: u64) -> Option<f64> {
    let total = hits + misses;
    (total > 0).then_some(hits as f64 / total as f64)
}

fn counter_delta(before: u64, after: u64) -> u64 {
    after.saturating_sub(before)
}

fn gauge_delta(before: u64, after: u64) -> i64 {
    let before = i128::from(before);
    let after = i128::from(after);
    let delta = after - before;
    delta.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
}

fn compare_bench_lanes(lanes: &[BenchLaneReport]) -> BenchLaneComparisonReport {
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

fn bench_gate_status(
    release_blocking_failed: bool,
    comparison: &BenchLaneComparisonReport,
) -> &'static str {
    if bench_gate_failure_classification(release_blocking_failed, comparison).is_some() {
        "failed"
    } else {
        "passed"
    }
}

fn bench_gate_failure_classification(
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

#[derive(Debug, Clone, Copy)]
enum BenchProfileKind {
    Promotion135k,
    Characterization200k,
    Characterization256k,
}

impl BenchProfileKind {
    fn name(self) -> &'static str {
        match self {
            Self::Promotion135k => "qwen-135k-promotion",
            Self::Characterization200k => "qwen-200k-characterization",
            Self::Characterization256k => "qwen-256k-characterization",
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        match name {
            "qwen-135k-promotion" => Some(Self::Promotion135k),
            "qwen-200k-characterization" => Some(Self::Characterization200k),
            "qwen-256k-characterization" => Some(Self::Characterization256k),
            _ => None,
        }
    }

    fn target_tokens(self) -> usize {
        match self {
            Self::Promotion135k => 135_000,
            Self::Characterization200k => 200_000,
            Self::Characterization256k => 256_000,
        }
    }

    fn release_blocking(self) -> bool {
        matches!(self, Self::Promotion135k)
    }

    fn short_label(self) -> &'static str {
        match self {
            Self::Promotion135k => "135k",
            Self::Characterization200k => "200k",
            Self::Characterization256k => "256k",
        }
    }
}

fn selected_profiles(profile: &str) -> anyhow::Result<Vec<BenchProfileKind>> {
    match profile {
        "135k" | "135K" | "qwen-135k-promotion" => Ok(vec![BenchProfileKind::Promotion135k]),
        "200k" | "200K" | "qwen-200k-characterization" => {
            Ok(vec![BenchProfileKind::Characterization200k])
        }
        "256k" | "256K" | "qwen-256k-characterization" => {
            Ok(vec![BenchProfileKind::Characterization256k])
        }
        "all" => Ok(vec![
            BenchProfileKind::Promotion135k,
            BenchProfileKind::Characterization200k,
            BenchProfileKind::Characterization256k,
        ]),
        other => anyhow::bail!("unknown qwen long-context profile `{other}`"),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchCaseKind {
    PlainRecall,
    JsonObjectRecall,
    RequiredToolRecall,
    StreamedRequiredToolRecall,
    MultiTurnLifecycle,
    SameLongPromptTwice,
    SharedPrefixShortSuffixVariation,
}

impl BenchCaseKind {
    fn all() -> [Self; 7] {
        [
            Self::PlainRecall,
            Self::JsonObjectRecall,
            Self::RequiredToolRecall,
            Self::StreamedRequiredToolRecall,
            Self::MultiTurnLifecycle,
            Self::SameLongPromptTwice,
            Self::SharedPrefixShortSuffixVariation,
        ]
    }

    fn name(self) -> &'static str {
        match self {
            Self::PlainRecall => "plain-recall",
            Self::JsonObjectRecall => "json-object-recall",
            Self::RequiredToolRecall => "required-tool-recall",
            Self::StreamedRequiredToolRecall => "streamed-required-tool-recall",
            Self::MultiTurnLifecycle => "multi-turn-lifecycle",
            Self::SameLongPromptTwice => "same-long-prompt-twice",
            Self::SharedPrefixShortSuffixVariation => "shared-prefix-short-suffix-variation",
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        match name {
            "plain-recall" => Some(Self::PlainRecall),
            "json-object-recall" => Some(Self::JsonObjectRecall),
            "required-tool-recall" => Some(Self::RequiredToolRecall),
            "streamed-required-tool-recall" => Some(Self::StreamedRequiredToolRecall),
            "multi-turn-lifecycle" => Some(Self::MultiTurnLifecycle),
            "same-long-prompt-twice" => Some(Self::SameLongPromptTwice),
            "shared-prefix-short-suffix-variation" => Some(Self::SharedPrefixShortSuffixVariation),
            _ => None,
        }
    }

    fn mode(self) -> &'static str {
        match self {
            Self::PlainRecall => "chat",
            Self::JsonObjectRecall => "chat-json-object",
            Self::RequiredToolRecall => "chat-required-tool",
            Self::StreamedRequiredToolRecall => "chat-stream-required-tool",
            Self::MultiTurnLifecycle => "chat-multi-turn",
            Self::SameLongPromptTwice => "chat-warm-prefix-repeat",
            Self::SharedPrefixShortSuffixVariation => "chat-shared-prefix-short-suffix",
        }
    }

    fn response_contract(self) -> &'static str {
        match self {
            Self::PlainRecall => "assistant content must contain the target marker",
            Self::JsonObjectRecall => {
                "assistant content must be a JSON object with marker, profile, and case"
            }
            Self::RequiredToolRecall => {
                "assistant must finish with tool_calls and call report_long_context_recall with marker, profile, and case arguments"
            }
            Self::StreamedRequiredToolRecall => {
                "SSE deltas must finish with tool_calls and assemble to report_long_context_recall with marker, profile, and case arguments"
            }
            Self::MultiTurnLifecycle => {
                "multi-message chat response must recall the target marker from the first turn"
            }
            Self::SameLongPromptTwice => {
                "two identical long-prompt chat requests must recall the marker and increase prefix cache hit tokens"
            }
            Self::SharedPrefixShortSuffixVariation => {
                "two chat requests with a shared long prefix and short suffix variation must recall the marker and increase prefix cache hit tokens"
            }
        }
    }

    fn streams(self) -> bool {
        matches!(self, Self::StreamedRequiredToolRecall)
    }

    fn request_count(self) -> usize {
        match self {
            Self::SameLongPromptTwice | Self::SharedPrefixShortSuffixVariation => 2,
            _ => 1,
        }
    }

    fn requires_prefix_cache_validation(self) -> bool {
        matches!(
            self,
            Self::SameLongPromptTwice | Self::SharedPrefixShortSuffixVariation
        )
    }
}

#[derive(Debug, Clone)]
struct BenchLaneConfig {
    name: String,
    endpoint: Option<String>,
    model_id: String,
    snapshot_path: Option<PathBuf>,
}

struct BenchExecutionContext<'a> {
    client: &'a reqwest::Client,
    baseline_trace: Option<&'a Value>,
    hardware: &'a HardwareReport,
    latency_regression_threshold: f64,
    max_tokens: u32,
    admin_token: Option<&'a str>,
}

#[derive(Debug, Serialize)]
struct BenchReport {
    gate: &'static str,
    status: String,
    generated_at_unix_ms: u128,
    trace_output_path: Option<String>,
    model: ModelIdentityReport,
    hardware: HardwareReport,
    cache_policy: CachePolicyReport,
    run_controls: BenchRunControlsReport,
    scheduler: SchedulerSettingsReport,
    baseline: BaselineReport,
    profiles: Vec<BenchProfileReport>,
    lanes: Vec<BenchLaneReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    failure_classification: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    comparison: Option<BenchLaneComparisonReport>,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct BenchRunControlsReport {
    warmup_count: u32,
    repetitions: u32,
    timeout_ms: u64,
    connect_timeout_ms: u64,
    max_tokens: u32,
    latency_regression_threshold: f64,
}

#[derive(Debug, Clone, Serialize)]
struct SchedulerSettingsReport {
    source: &'static str,
    concurrency_limit: usize,
    queue_limit: usize,
    queue_timeout_ms: u64,
    prefill_threshold_chars: usize,
    prefill_burst: usize,
}

impl SchedulerSettingsReport {
    fn from_args(args: &[String]) -> anyhow::Result<Self> {
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
struct BenchLaneReport {
    name: String,
    status: String,
    model: ModelIdentityReport,
    profiles: Vec<BenchProfileReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_metrics: Option<BenchCacheMetricsReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    admin_metrics: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    admin_metrics_error: Option<String>,
}

#[derive(Debug, Serialize)]
struct BenchLaneComparisonReport {
    status: String,
    artifact_identity_match: bool,
    cases: Vec<BenchLaneCaseComparisonReport>,
}

#[derive(Debug, Serialize)]
struct BenchLaneCaseComparisonReport {
    profile: &'static str,
    case: &'static str,
    lanes: Vec<BenchLaneCaseMetricReport>,
    fastest_lane: Option<String>,
}

#[derive(Debug, Serialize)]
struct BenchLaneCaseMetricReport {
    lane: String,
    status: String,
    latency_ms: Option<u128>,
    #[serde(flatten)]
    stream_timing: StreamTimingReport,
    tokens_per_second: Option<f64>,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    total_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cached_tokens_status: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cached_tokens: Option<u64>,
    classification: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
struct BenchCacheMetricsReport {
    prefix_cache: PrefixCacheMetricsReport,
    weight_cache: WeightCacheMetricsReport,
    kv_cache: ResidentCacheMetricsReport,
    linear_attention_cache: ResidentCacheMetricsReport,
    readiness: CacheReadinessReport,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
struct PrefixCacheMetricsReport {
    hits: u64,
    misses: u64,
    hit_rate: Option<f64>,
    stores: u64,
    evictions: u64,
    rejected: u64,
    reused_tokens: u64,
    hit_tokens: u64,
    miss_tokens: u64,
    avoided_prefill_tokens: u64,
    resident_bytes: u64,
    resident_entries: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
struct WeightCacheMetricsReport {
    hits: u64,
    misses: u64,
    hit_rate: Option<f64>,
    uploads: u64,
    evictions: u64,
    bytes_uploaded: u64,
    bytes_evicted: u64,
    resident_bytes: u64,
    resident_buffers: u64,
    budget_bytes: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
struct ResidentCacheMetricsReport {
    allocations: u64,
    syncs: u64,
    evictions: u64,
    bytes_uploaded: u64,
    bytes_evicted: u64,
    resident_bytes: u64,
    resident_buffers: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
struct CacheReadinessReport {
    status: &'static str,
    observed_signals: Vec<&'static str>,
    missing_signals: Vec<&'static str>,
}

#[derive(Debug, Clone, Serialize)]
struct BenchProfileReport {
    name: &'static str,
    target_tokens: usize,
    release_blocking: bool,
    status: String,
    cases: Vec<BenchCaseReport>,
}

#[derive(Debug, Clone, Serialize)]
struct BenchCaseReport {
    name: &'static str,
    mode: &'static str,
    target_tokens: usize,
    stream: bool,
    response_contract: &'static str,
    request_count: usize,
    marker: String,
    prompt_identity: PromptIdentityReport,
    status: String,
    classification: String,
    prefill: BenchPrefillReport,
    decode: BenchDecodeReport,
    cache: BenchCacheBehaviorReport,
    summary: BenchCaseSummaryReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    planned_prompt_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    latency_ms: Option<u128>,
    #[serde(flatten)]
    stream_timing: StreamTimingReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokens_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    completion_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cached_tokens_status: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cached_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    http_status: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    admin_metrics: Option<BenchCaseAdminMetricsReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    baseline: Option<BaselineComparisonReport>,
}

#[derive(Debug, Clone, Serialize)]
struct BenchCaseAdminMetricsReport {
    #[serde(skip_serializing_if = "Option::is_none")]
    prefix_cache: Option<BenchCasePrefixCacheMetricsReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

impl BenchCaseAdminMetricsReport {
    fn from_prefix_cache_snapshots(
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

    fn error(error: String) -> Self {
        Self {
            prefix_cache: None,
            error: Some(error),
        }
    }

    fn prefix_cache_delta(&self) -> Option<&PrefixCacheMetricsDeltaReport> {
        self.prefix_cache.as_ref().map(|metrics| &metrics.delta)
    }
}

#[derive(Debug, Clone, Serialize)]
struct BenchCasePrefixCacheMetricsReport {
    before: PrefixCacheMetricsReport,
    after: PrefixCacheMetricsReport,
    delta: PrefixCacheMetricsDeltaReport,
}

#[derive(Debug, Clone, Serialize)]
struct PrefixCacheMetricsDeltaReport {
    hits: u64,
    misses: u64,
    hit_rate: Option<f64>,
    stores: u64,
    evictions: u64,
    rejected: u64,
    reused_tokens: u64,
    hit_tokens: u64,
    miss_tokens: u64,
    avoided_prefill_tokens: u64,
    resident_bytes: i64,
    resident_entries: i64,
}

impl PrefixCacheMetricsDeltaReport {
    fn between(before: &PrefixCacheMetricsReport, after: &PrefixCacheMetricsReport) -> Self {
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
struct PromptIdentityReport {
    profile: &'static str,
    case: &'static str,
    context_tokens: usize,
    marker: String,
    prompt_hash: String,
    prompt_hash_source: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct BenchPrefillReport {
    planned_prompt_tokens: Option<usize>,
    prompt_tokens: Option<u64>,
    cached_tokens: Option<u64>,
    uncached_tokens: Option<u64>,
    time_to_first_token_ms: Option<u128>,
}

impl BenchPrefillReport {
    fn planned() -> Self {
        Self {
            planned_prompt_tokens: None,
            prompt_tokens: None,
            cached_tokens: None,
            uncached_tokens: None,
            time_to_first_token_ms: None,
        }
    }

    fn from_run(run: &CaseRun) -> Self {
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
struct BenchDecodeReport {
    max_tokens: u32,
    completion_tokens: Option<u64>,
    total_latency_ms: Option<u128>,
    time_to_first_token_ms: Option<u128>,
    tokens_per_second: Option<f64>,
}

impl BenchDecodeReport {
    fn planned(max_tokens: u32) -> Self {
        Self {
            max_tokens,
            completion_tokens: None,
            total_latency_ms: None,
            time_to_first_token_ms: None,
            tokens_per_second: None,
        }
    }

    fn from_run(max_tokens: u32, run: &CaseRun) -> Self {
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
struct BenchCacheBehaviorReport {
    cached_tokens_status: Option<&'static str>,
    cached_tokens: Option<u64>,
    reused_tokens: Option<u64>,
    lookup_result: Option<&'static str>,
}

impl BenchCacheBehaviorReport {
    fn planned() -> Self {
        Self {
            cached_tokens_status: None,
            cached_tokens: None,
            reused_tokens: None,
            lookup_result: None,
        }
    }

    fn from_run(run: &CaseRun) -> Self {
        Self {
            cached_tokens_status: run.cached_tokens_status,
            cached_tokens: run.cached_tokens,
            reused_tokens: run.cached_tokens,
            lookup_result: cache_lookup_result(run.cached_tokens_status, run.cached_tokens),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct BenchCaseSummaryReport {
    sample_count: usize,
    latency_ms_p50: Option<u128>,
    latency_ms_p95: Option<u128>,
    tokens_per_second_p50: Option<f64>,
    tokens_per_second_p95: Option<f64>,
    ttft_ms_p50: Option<u128>,
    ttft_ms_p95: Option<u128>,
}

impl BenchCaseSummaryReport {
    fn planned() -> Self {
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

    fn from_run(run: &CaseRun) -> Self {
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
struct ModelIdentityReport {
    id: String,
    endpoint: Option<String>,
    snapshot_path: Option<String>,
    repo_id: Option<String>,
    requested_revision: Option<String>,
    resolved_commit: Option<String>,
    profile: Option<String>,
    family: Option<String>,
    loader: Option<String>,
    quantization: Option<String>,
    manifest_digest: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct HardwareReport {
    os: String,
    arch: String,
    cpu: Option<String>,
}

impl HardwareReport {
    fn detect() -> Self {
        Self {
            os: std::env::consts::OS.to_owned(),
            arch: std::env::consts::ARCH.to_owned(),
            cpu: detect_cpu_name(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct CachePolicyReport {
    cache_layout: &'static str,
    prompt_template: &'static str,
    namespace_fields: Vec<&'static str>,
    benchmark_metrics: Vec<&'static str>,
    env: BTreeMap<String, String>,
}

impl CachePolicyReport {
    fn from_env() -> Self {
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
                "linear_attention_cache_residency",
                "eviction_churn",
            ],
            env,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct BaselineReport {
    path: Option<String>,
    status: String,
    latency_regression_threshold: f64,
}

#[derive(Debug, Clone, Serialize)]
struct BaselineComparisonReport {
    status: String,
    baseline_status: Option<String>,
    baseline_latency_ms: Option<u128>,
    baseline_tokens_per_second: Option<f64>,
    hardware_match: bool,
    model_class_match: bool,
}

#[derive(Debug)]
struct CaseRun {
    status: &'static str,
    classification: String,
    planned_prompt_tokens: usize,
    latency_ms: Option<u128>,
    stream_timing: StreamTimingReport,
    tokens_per_second: Option<f64>,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    total_tokens: Option<u64>,
    cached_tokens_status: Option<&'static str>,
    cached_tokens: Option<u64>,
    prompt_hash: Option<String>,
    http_status: Option<u16>,
    finish_reason: Option<String>,
    error: Option<String>,
}

#[derive(Debug)]
struct PromptBuild {
    marker: String,
    body: String,
    token_count: usize,
}

#[derive(Debug, Clone)]
struct UsageMetrics {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    total_tokens: Option<u64>,
    cached_tokens_status: Option<&'static str>,
    cached_tokens: Option<u64>,
}

impl Default for UsageMetrics {
    fn default() -> Self {
        Self {
            prompt_tokens: None,
            completion_tokens: None,
            total_tokens: None,
            cached_tokens_status: Some("missing"),
            cached_tokens: None,
        }
    }
}

impl UsageMetrics {
    fn merge(&mut self, next: Self) {
        self.prompt_tokens = max_optional_u64(self.prompt_tokens, next.prompt_tokens);
        self.completion_tokens = max_optional_u64(self.completion_tokens, next.completion_tokens);
        self.total_tokens = max_optional_u64(self.total_tokens, next.total_tokens);
        self.cached_tokens = max_optional_u64(self.cached_tokens, next.cached_tokens);
        self.cached_tokens_status =
            merge_cached_tokens_status(self.cached_tokens_status, next.cached_tokens_status);
        if self.total_tokens.is_none()
            && let (Some(prompt), Some(completion)) = (self.prompt_tokens, self.completion_tokens)
        {
            self.total_tokens = prompt.checked_add(completion);
        }
    }
}

fn max_optional_u64(current: Option<u64>, next: Option<u64>) -> Option<u64> {
    match (current, next) {
        (Some(current), Some(next)) => Some(current.max(next)),
        (Some(current), None) => Some(current),
        (None, Some(next)) => Some(next),
        (None, None) => None,
    }
}

fn sum_optional_u64(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.saturating_add(right)),
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (None, None) => None,
    }
}

fn sum_optional_u128(left: Option<u128>, right: Option<u128>) -> Option<u128> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.saturating_add(right)),
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (None, None) => None,
    }
}

fn merge_cached_tokens_status(
    current: Option<&'static str>,
    next: Option<&'static str>,
) -> Option<&'static str> {
    match (
        cached_tokens_status_rank(current),
        cached_tokens_status_rank(next),
    ) {
        (Some((current_rank, current)), Some((next_rank, next))) => {
            if next_rank > current_rank {
                Some(next)
            } else {
                Some(current)
            }
        }
        (Some((_rank, current)), None) => Some(current),
        (None, Some((_rank, next))) => Some(next),
        (None, None) => None,
    }
}

fn cached_tokens_status_rank(status: Option<&'static str>) -> Option<(u8, &'static str)> {
    status.map(|status| {
        let rank = match status {
            "present" => 4,
            "invalid" => 3,
            "null" => 2,
            "missing" => 1,
            _ => 0,
        };
        (rank, status)
    })
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq)]
struct StreamTimingReport {
    #[serde(skip_serializing_if = "Option::is_none")]
    first_byte_latency_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    first_sse_data_latency_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    first_content_delta_latency_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    first_tool_delta_latency_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_finish_latency_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    first_semantic_delta_latency_ms: Option<u128>,
}

#[derive(Debug, Default)]
struct StreamTimingTracker {
    first_byte_latency: Option<Duration>,
    first_sse_data_latency: Option<Duration>,
    first_content_delta_latency: Option<Duration>,
    first_tool_delta_latency: Option<Duration>,
    tool_finish_latency: Option<Duration>,
    first_semantic_delta_latency: Option<Duration>,
}

impl StreamTimingTracker {
    fn record_first_byte(&mut self, elapsed: Duration) {
        if self.first_byte_latency.is_none() {
            self.first_byte_latency = Some(elapsed);
        }
    }

    fn record_sse_frame(&mut self, elapsed: Duration, delta: StreamFrameDelta) {
        if self.first_sse_data_latency.is_none() {
            self.first_sse_data_latency = Some(elapsed);
        }
        if delta.content && self.first_content_delta_latency.is_none() {
            self.first_content_delta_latency = Some(elapsed);
        }
        if delta.tool && self.first_tool_delta_latency.is_none() {
            self.first_tool_delta_latency = Some(elapsed);
        }
        if delta.tool_finish && self.tool_finish_latency.is_none() {
            self.tool_finish_latency = Some(elapsed);
        }
        if delta.semantic() && self.first_semantic_delta_latency.is_none() {
            self.first_semantic_delta_latency = Some(elapsed);
        }
    }

    fn to_report(&self) -> StreamTimingReport {
        StreamTimingReport {
            first_byte_latency_ms: self.first_byte_latency.map(|duration| duration.as_millis()),
            first_sse_data_latency_ms: self
                .first_sse_data_latency
                .map(|duration| duration.as_millis()),
            first_content_delta_latency_ms: self
                .first_content_delta_latency
                .map(|duration| duration.as_millis()),
            first_tool_delta_latency_ms: self
                .first_tool_delta_latency
                .map(|duration| duration.as_millis()),
            tool_finish_latency_ms: self
                .tool_finish_latency
                .map(|duration| duration.as_millis()),
            first_semantic_delta_latency_ms: self
                .first_semantic_delta_latency
                .map(|duration| duration.as_millis()),
        }
    }
}

#[derive(Debug, Default)]
struct StreamFrameDelta {
    content: bool,
    tool: bool,
    tool_finish: bool,
}

impl StreamFrameDelta {
    fn semantic(&self) -> bool {
        self.content || self.tool
    }
}

#[derive(Debug, Clone, Default)]
struct StreamAssembly {
    content: String,
    tool_name: Option<String>,
    tool_arguments: String,
    finish_reason: Option<String>,
    usage: UsageMetrics,
}

#[cfg(test)]
fn profile_report(profile: BenchProfileKind) -> BenchProfileReport {
    profile_report_with_max_tokens(profile, DEFAULT_MAX_TOKENS)
}

fn profile_report_with_max_tokens(
    profile: BenchProfileKind,
    max_tokens: u32,
) -> BenchProfileReport {
    BenchProfileReport {
        name: profile.name(),
        target_tokens: profile.target_tokens(),
        release_blocking: profile.release_blocking(),
        status: "planned".to_owned(),
        cases: BenchCaseKind::all()
            .iter()
            .copied()
            .map(|case| case_report(profile, case, max_tokens))
            .collect(),
    }
}

fn case_report(profile: BenchProfileKind, case: BenchCaseKind, max_tokens: u32) -> BenchCaseReport {
    let marker = marker_for_case(profile, case);
    BenchCaseReport {
        name: case.name(),
        mode: case.mode(),
        target_tokens: profile.target_tokens(),
        stream: case.streams(),
        response_contract: case.response_contract(),
        request_count: case.request_count(),
        marker: marker.clone(),
        prompt_identity: PromptIdentityReport {
            profile: profile.name(),
            case: case.name(),
            context_tokens: profile.target_tokens(),
            marker: marker.clone(),
            prompt_hash: prompt_identity_hash(profile, case, &marker),
            prompt_hash_source: "planned_identity",
        },
        status: "planned".to_owned(),
        classification: if profile.release_blocking() {
            "release-blocking".to_owned()
        } else {
            "frontier-characterization".to_owned()
        },
        prefill: BenchPrefillReport::planned(),
        decode: BenchDecodeReport::planned(max_tokens),
        cache: BenchCacheBehaviorReport::planned(),
        summary: BenchCaseSummaryReport::planned(),
        planned_prompt_tokens: None,
        latency_ms: None,
        stream_timing: StreamTimingReport::default(),
        tokens_per_second: None,
        prompt_tokens: None,
        completion_tokens: None,
        total_tokens: None,
        cached_tokens_status: None,
        cached_tokens: None,
        http_status: None,
        finish_reason: None,
        error: None,
        admin_metrics: None,
        baseline: None,
    }
}

fn prompt_identity_hash(profile: BenchProfileKind, case: BenchCaseKind, marker: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(profile.name().as_bytes());
    hasher.update([0]);
    hasher.update(case.name().as_bytes());
    hasher.update([0]);
    hasher.update(profile.target_tokens().to_le_bytes());
    hasher.update(marker.as_bytes());
    let digest = hasher.finalize();
    format!("sha256:{digest:x}")
}

fn prompt_body_hash(prompt: &str) -> String {
    let digest = Sha256::digest(prompt.as_bytes());
    format!("sha256:{digest:x}")
}

async fn load_model_identity(
    model_id: &str,
    endpoint: Option<&str>,
    snapshot_path: Option<&Path>,
    dry_run: bool,
) -> anyhow::Result<ModelIdentityReport> {
    let mut report = ModelIdentityReport {
        id: model_id.to_owned(),
        endpoint: endpoint.map(str::to_owned),
        snapshot_path: snapshot_path.map(|path| path.display().to_string()),
        repo_id: None,
        requested_revision: None,
        resolved_commit: None,
        profile: None,
        family: None,
        loader: None,
        quantization: None,
        manifest_digest: None,
    };

    let Some(snapshot_path) = snapshot_path else {
        return Ok(report);
    };
    if dry_run && !snapshot_path.join("llm-engine-manifest.json").is_file() {
        return Ok(report);
    }

    let snapshot = ModelStore::inspect_snapshot(snapshot_path)
        .await
        .with_context(|| {
            format!(
                "inspect benchmark snapshot manifest at `{}`",
                snapshot_path.display()
            )
        })?;
    report.repo_id = Some(snapshot.manifest.repo_id);
    report.requested_revision = Some(snapshot.manifest.requested_revision);
    report.resolved_commit = Some(snapshot.manifest.resolved_commit);
    report.profile = Some(snapshot.manifest.profile);
    report.family = Some(snapshot.manifest.family);
    report.loader = Some(snapshot.manifest.loader);
    report.quantization = Some(snapshot.manifest.quantization);
    report.manifest_digest = Some(snapshot.manifest_digest);
    Ok(report)
}

fn load_qwen_tokenizer(snapshot_path: &Path) -> anyhow::Result<HuggingFaceTokenizer> {
    let tokenizer_path = snapshot_path.join("tokenizer.json");
    HuggingFaceTokenizer::from_file(&tokenizer_path)
        .with_context(|| format!("load Qwen tokenizer `{}`", tokenizer_path.display()))
}

async fn load_baseline_trace(path: Option<&Path>) -> anyhow::Result<Option<Value>> {
    let Some(path) = path else {
        return Ok(None);
    };
    let bytes = tokio::fs::read(path)
        .await
        .with_context(|| format!("read baseline trace `{}`", path.display()))?;
    let value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse baseline trace `{}`", path.display()))?;
    Ok(Some(value))
}

fn baseline_status(report: &BenchReport, baseline: Option<&Value>) -> String {
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

async fn run_case(
    client: &reqwest::Client,
    endpoint: &str,
    model_id: &str,
    tokenizer: &HuggingFaceTokenizer,
    profile: BenchProfileKind,
    case: BenchCaseKind,
    max_tokens: u32,
) -> CaseRun {
    let prompt = match build_prompt(tokenizer, profile, case) {
        Ok(prompt) => prompt,
        Err(err) => {
            return CaseRun {
                status: "failed",
                classification: "prompt_build_failed".to_owned(),
                planned_prompt_tokens: 0,
                latency_ms: None,
                stream_timing: StreamTimingReport::default(),
                tokens_per_second: None,
                prompt_tokens: None,
                completion_tokens: None,
                total_tokens: None,
                cached_tokens_status: None,
                cached_tokens: None,
                prompt_hash: None,
                http_status: None,
                finish_reason: None,
                error: Some(err.to_string()),
            };
        }
    };
    let prompt_hash = prompt_body_hash(&prompt.body);
    if case == BenchCaseKind::SameLongPromptTwice {
        let mut run = run_same_long_prompt_twice_case(
            client, endpoint, model_id, profile, case, &prompt, max_tokens,
        )
        .await;
        run.prompt_hash = Some(prompt_hash);
        return run;
    }
    if case == BenchCaseKind::SharedPrefixShortSuffixVariation {
        let mut run = run_shared_prefix_short_suffix_case(
            client, endpoint, model_id, profile, case, &prompt, max_tokens,
        )
        .await;
        run.prompt_hash = Some(prompt_hash);
        return run;
    }
    let body = request_body(model_id, profile, case, &prompt, max_tokens);
    let mut run = if case.streams() {
        run_streaming_case(client, endpoint, profile, case, &prompt, body).await
    } else {
        run_buffered_case(client, endpoint, profile, case, &prompt, body).await
    };
    run.prompt_hash = Some(prompt_hash);
    run
}

async fn run_same_long_prompt_twice_case(
    client: &reqwest::Client,
    endpoint: &str,
    model_id: &str,
    profile: BenchProfileKind,
    case: BenchCaseKind,
    prompt: &PromptBuild,
    max_tokens: u32,
) -> CaseRun {
    let body = cache_probe_request_body(
        model_id,
        prompt,
        max_tokens,
        "Return only the target_marker value.",
    );
    let first = run_buffered_case(client, endpoint, profile, case, prompt, body.clone()).await;
    let second = run_buffered_case(client, endpoint, profile, case, prompt, body).await;
    merge_case_runs(first, second, "second_identical_prompt")
}

async fn run_shared_prefix_short_suffix_case(
    client: &reqwest::Client,
    endpoint: &str,
    model_id: &str,
    profile: BenchProfileKind,
    case: BenchCaseKind,
    prompt: &PromptBuild,
    max_tokens: u32,
) -> CaseRun {
    let first = run_buffered_case(
        client,
        endpoint,
        profile,
        case,
        prompt,
        cache_probe_request_body(
            model_id,
            prompt,
            max_tokens,
            "Short suffix A: prime shared-prefix reuse and return only the target_marker value.",
        ),
    )
    .await;
    let second = run_buffered_case(
        client,
        endpoint,
        profile,
        case,
        prompt,
        cache_probe_request_body(
            model_id,
            prompt,
            max_tokens,
            "Short suffix B: vary only this suffix and return only the target_marker value.",
        ),
    )
    .await;
    merge_case_runs(first, second, "second_shared_prefix_prompt")
}

async fn run_buffered_case(
    client: &reqwest::Client,
    endpoint: &str,
    profile: BenchProfileKind,
    case: BenchCaseKind,
    prompt: &PromptBuild,
    body: Value,
) -> CaseRun {
    let url = format!("{endpoint}/v1/chat/completions");
    let started = Instant::now();
    let response = match client.post(&url).json(&body).send().await {
        Ok(response) => response,
        Err(err) => {
            return failed_case(
                "http_request_failed",
                prompt.token_count,
                started.elapsed(),
                None,
                err.to_string(),
            );
        }
    };
    let status = response.status();
    let http_status = Some(status.as_u16());
    let text = match response.text().await {
        Ok(text) => text,
        Err(err) => {
            return failed_case(
                "http_body_failed",
                prompt.token_count,
                started.elapsed(),
                http_status,
                err.to_string(),
            );
        }
    };
    let latency = started.elapsed();
    if !status.is_success() {
        return failed_case(
            "http_status_failed",
            prompt.token_count,
            latency,
            http_status,
            text,
        );
    }
    let value = match serde_json::from_str::<Value>(&text) {
        Ok(value) => value,
        Err(err) => {
            return failed_case(
                "response_json_failed",
                prompt.token_count,
                latency,
                http_status,
                err.to_string(),
            );
        }
    };
    let usage = usage_from_value(value.get("usage"));
    let finish_reason = value
        .pointer("/choices/0/finish_reason")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let validation = validate_buffered_case(profile, case, &prompt.marker, &value);
    case_from_validation(
        validation,
        prompt.token_count,
        latency,
        StreamTimingReport::default(),
        http_status,
        finish_reason,
        usage,
    )
}

async fn run_streaming_case(
    client: &reqwest::Client,
    endpoint: &str,
    profile: BenchProfileKind,
    case: BenchCaseKind,
    prompt: &PromptBuild,
    body: Value,
) -> CaseRun {
    let url = format!("{endpoint}/v1/chat/completions");
    let started = Instant::now();
    let response = match client.post(&url).json(&body).send().await {
        Ok(response) => response,
        Err(err) => {
            return failed_case(
                "stream_http_request_failed",
                prompt.token_count,
                started.elapsed(),
                None,
                err.to_string(),
            );
        }
    };
    let status = response.status();
    let http_status = Some(status.as_u16());
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        return failed_case(
            "stream_http_status_failed",
            prompt.token_count,
            started.elapsed(),
            http_status,
            text,
        );
    }

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();
    let mut assembly = StreamAssembly::default();
    let mut timings = StreamTimingTracker::default();
    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(chunk) => chunk,
            Err(err) => {
                return failed_case_with_stream_timing(
                    "stream_body_failed",
                    prompt.token_count,
                    started.elapsed(),
                    http_status,
                    err.to_string(),
                    timings.to_report(),
                );
            }
        };
        if chunk.is_empty() {
            continue;
        }
        let chunk_elapsed = started.elapsed();
        timings.record_first_byte(chunk_elapsed);
        buffer.push_str(&String::from_utf8_lossy(&chunk));
        consume_sse_buffer(&mut buffer, &mut assembly, &mut timings, chunk_elapsed);
    }
    if !buffer.is_empty() {
        buffer.push('\n');
        consume_sse_buffer(&mut buffer, &mut assembly, &mut timings, started.elapsed());
    }
    let latency = started.elapsed();
    let validation = validate_streaming_case(profile, case, &prompt.marker, &assembly);
    case_from_validation(
        validation,
        prompt.token_count,
        latency,
        timings.to_report(),
        http_status,
        assembly.finish_reason,
        assembly.usage,
    )
}

fn build_prompt(
    tokenizer: &HuggingFaceTokenizer,
    profile: BenchProfileKind,
    case: BenchCaseKind,
) -> anyhow::Result<PromptBuild> {
    let marker = marker_for_case(profile, case);
    let target_tokens = profile.target_tokens();
    let mut body = format!(
        "\
Long-context benchmark profile: {profile_name}
Scenario: {case_name}
Target marker name: target_marker
Target marker value: {marker}

Only the marker value above is correct. Later context rows are distractors and must not replace it.

",
        profile_name = profile.name(),
        case_name = case.name(),
    );
    let footer = "\nEnd of benchmark context. Use the target_marker value from the first section when answering.\n";
    let row_template = "Context row 000000: repository module notes, scheduler counters, cache namespace fields, parser states, and trace identifiers. This row is distractor material only.\n";
    let row_tokens = tokenizer.encode(row_template, false)?.len().max(1);
    let base_tokens = tokenizer.encode(&(body.clone() + footer), false)?.len();
    let estimated_rows = target_tokens
        .saturating_sub(base_tokens)
        .div_ceil(row_tokens)
        .saturating_add(8);
    for row in 0..estimated_rows {
        body.push_str(&format!(
            "Context row {row:06}: repository module notes, scheduler counters, cache namespace fields, parser states, and trace identifiers. This row is distractor material only.\n"
        ));
    }
    body.push_str(footer);
    let mut token_count = tokenizer.encode(&body, false)?.len();
    while token_count < target_tokens {
        let row = token_count;
        body.push_str(&format!(
            "Context extension {row:06}: additional non-target diagnostics for long-window pressure.\n"
        ));
        token_count = tokenizer.encode(&body, false)?.len();
    }
    Ok(PromptBuild {
        marker,
        body,
        token_count,
    })
}

fn request_body(
    model_id: &str,
    profile: BenchProfileKind,
    case: BenchCaseKind,
    prompt: &PromptBuild,
    max_tokens: u32,
) -> Value {
    let mut body = serde_json::json!({
        "model": model_id,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1
    });
    match case {
        BenchCaseKind::PlainRecall => {
            body["messages"] = serde_json::json!([
                {"role": "system", "content": "You are a long-context recall evaluator. Return the requested marker exactly."},
                {"role": "user", "content": format!("{}\nReturn only the target_marker value.", prompt.body)}
            ]);
        }
        BenchCaseKind::JsonObjectRecall => {
            body["response_format"] = serde_json::json!({"type": "json_object"});
            body["messages"] = serde_json::json!([
                {"role": "system", "content": "You are a long-context JSON recall evaluator. Return one JSON object and no prose."},
                {"role": "user", "content": format!("{}\nReturn exactly this JSON shape with the recalled marker value: {{\"marker\":\"...\",\"profile\":\"{}\",\"case\":\"{}\"}}.", prompt.body, profile.name(), case.name())}
            ]);
        }
        BenchCaseKind::RequiredToolRecall | BenchCaseKind::StreamedRequiredToolRecall => {
            body["tools"] = serde_json::json!([recall_tool_schema()]);
            body["tool_choice"] = serde_json::json!("required");
            if case.streams() {
                body["stream"] = serde_json::json!(true);
                body["stream_options"] = serde_json::json!({"include_usage": true});
            }
            body["messages"] = serde_json::json!([
                {"role": "system", "content": "You are a long-context tool-call evaluator. Use the provided function to report the recalled marker."},
                {"role": "user", "content": format!("{}\nCall report_long_context_recall with marker, profile, and case.", prompt.body)}
            ]);
        }
        BenchCaseKind::MultiTurnLifecycle => {
            body["messages"] = serde_json::json!([
                {"role": "system", "content": "You are a long-context multi-turn lifecycle evaluator. Return the requested marker exactly."},
                {"role": "user", "content": prompt.body},
                {"role": "assistant", "content": "I have processed the long context and will wait for the recall request."},
                {"role": "user", "content": "Now answer with only the target_marker value from the first user turn."}
            ]);
        }
        BenchCaseKind::SameLongPromptTwice | BenchCaseKind::SharedPrefixShortSuffixVariation => {
            body = cache_probe_request_body(
                model_id,
                prompt,
                max_tokens,
                "Return only the target_marker value.",
            );
        }
    }
    body
}

fn cache_probe_request_body(
    model_id: &str,
    prompt: &PromptBuild,
    max_tokens: u32,
    suffix: &str,
) -> Value {
    serde_json::json!({
        "model": model_id,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "messages": [
            {"role": "system", "content": "You are a long-context prefix-cache evaluator. Return the requested marker exactly."},
            {"role": "user", "content": format!("{}\n{suffix}", prompt.body)}
        ]
    })
}

fn recall_tool_schema() -> Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "report_long_context_recall",
            "description": "Report a recalled long-context benchmark marker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "marker": {"type": "string"},
                    "profile": {"type": "string"},
                    "case": {"type": "string"}
                },
                "required": ["marker", "profile", "case"],
                "additionalProperties": false
            }
        }
    })
}

fn validate_buffered_case(
    profile: BenchProfileKind,
    case: BenchCaseKind,
    marker: &str,
    value: &Value,
) -> Result<(), String> {
    match case {
        BenchCaseKind::PlainRecall
        | BenchCaseKind::MultiTurnLifecycle
        | BenchCaseKind::SameLongPromptTwice
        | BenchCaseKind::SharedPrefixShortSuffixVariation => {
            let content = value
                .pointer("/choices/0/message/content")
                .and_then(Value::as_str)
                .ok_or_else(|| "missing assistant content".to_owned())?;
            if content.contains(marker) {
                Ok(())
            } else {
                Err(format!(
                    "assistant content did not contain marker `{marker}`"
                ))
            }
        }
        BenchCaseKind::JsonObjectRecall => {
            let content = value
                .pointer("/choices/0/message/content")
                .and_then(Value::as_str)
                .ok_or_else(|| "missing assistant JSON content".to_owned())?;
            let parsed = serde_json::from_str::<Value>(content)
                .map_err(|err| format!("assistant content was not valid JSON: {err}"))?;
            parsed
                .as_object()
                .ok_or_else(|| "assistant JSON content was not an object".to_owned())?;
            validate_recall_arguments(&parsed, profile, case, marker, "JSON")
        }
        BenchCaseKind::RequiredToolRecall => {
            let finish_reason = value
                .pointer("/choices/0/finish_reason")
                .and_then(Value::as_str);
            validate_tool_finish_reason(finish_reason, "tool call")?;
            let tool_call = value
                .pointer("/choices/0/message/tool_calls/0")
                .ok_or_else(|| "missing required tool call".to_owned())?;
            validate_tool_call(tool_call, profile, case, marker)
        }
        BenchCaseKind::StreamedRequiredToolRecall => {
            Err("streamed tool case was routed through buffered validator".to_owned())
        }
    }
}

fn validate_streaming_case(
    profile: BenchProfileKind,
    case: BenchCaseKind,
    marker: &str,
    assembly: &StreamAssembly,
) -> Result<(), String> {
    if !case.streams() {
        return Err("non-streaming case was routed through streaming validator".to_owned());
    }
    let name = assembly
        .tool_name
        .as_deref()
        .ok_or_else(|| "missing streamed tool name".to_owned())?;
    if name != "report_long_context_recall" {
        return Err(format!(
            "streamed tool name `{name}` did not match expected"
        ));
    }
    validate_tool_finish_reason(assembly.finish_reason.as_deref(), "streamed tool call")?;
    let args = serde_json::from_str::<Value>(&assembly.tool_arguments)
        .map_err(|err| format!("streamed tool arguments were not JSON: {err}"))?;
    validate_recall_arguments(&args, profile, case, marker, "streamed tool")
}

fn validate_tool_call(
    tool_call: &Value,
    profile: BenchProfileKind,
    case: BenchCaseKind,
    marker: &str,
) -> Result<(), String> {
    let name = tool_call
        .pointer("/function/name")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing tool function name".to_owned())?;
    if name != "report_long_context_recall" {
        return Err(format!("tool function `{name}` did not match expected"));
    }
    let args_text = tool_call
        .pointer("/function/arguments")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing tool function arguments".to_owned())?;
    let args = serde_json::from_str::<Value>(args_text)
        .map_err(|err| format!("tool arguments were not JSON: {err}"))?;
    validate_recall_arguments(&args, profile, case, marker, "tool")
}

fn validate_tool_finish_reason(finish_reason: Option<&str>, label: &str) -> Result<(), String> {
    match finish_reason {
        Some("tool_calls") => Ok(()),
        Some(other) => Err(format!(
            "{label} finish_reason `{other}` did not equal `tool_calls`"
        )),
        None => Err(format!("{label} response was missing finish_reason")),
    }
}

fn validate_recall_arguments(
    args: &Value,
    profile: BenchProfileKind,
    case: BenchCaseKind,
    marker: &str,
    label: &str,
) -> Result<(), String> {
    let object = args
        .as_object()
        .ok_or_else(|| format!("{label} arguments were not a JSON object"))?;
    validate_recall_argument(object.get("marker"), marker, label, "marker")?;
    validate_recall_argument(object.get("profile"), profile.name(), label, "profile")?;
    validate_recall_argument(object.get("case"), case.name(), label, "case")?;
    if object.len() != 3 {
        return Err(format!(
            "{label} arguments must contain exactly marker, profile, and case"
        ));
    }
    Ok(())
}

fn validate_recall_argument(
    value: Option<&Value>,
    expected: &str,
    label: &str,
    key: &str,
) -> Result<(), String> {
    match value.and_then(Value::as_str) {
        Some(actual) if actual == expected => Ok(()),
        Some(actual) => Err(format!(
            "{label} {key} `{actual}` did not equal `{expected}`"
        )),
        None => Err(format!("{label} arguments missing string `{key}`")),
    }
}

fn consume_sse_buffer(
    buffer: &mut String,
    assembly: &mut StreamAssembly,
    timings: &mut StreamTimingTracker,
    elapsed: Duration,
) {
    while let Some(index) = buffer.find('\n') {
        let mut line = buffer[..index].trim_end_matches('\r').to_owned();
        buffer.drain(..=index);
        if !line.starts_with("data:") {
            continue;
        }
        line.drain(..5);
        let data = line.trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        let delta = apply_sse_frame(&value, assembly);
        timings.record_sse_frame(elapsed, delta);
    }
}

fn apply_sse_frame(value: &Value, assembly: &mut StreamAssembly) -> StreamFrameDelta {
    let mut delta = StreamFrameDelta::default();
    if let Some(usage) = value.get("usage") {
        assembly.usage.merge(usage_from_value(Some(usage)));
    }
    if let Some(choice) = value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
    {
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            if reason == "tool_calls" {
                delta.tool_finish = true;
            }
            assembly.finish_reason = Some(reason.to_owned());
        }
        if let Some(content) = choice.pointer("/delta/content").and_then(Value::as_str) {
            if !content.is_empty() {
                delta.content = true;
            }
            assembly.content.push_str(content);
        }
        if let Some(tool_calls) = choice
            .pointer("/delta/tool_calls")
            .and_then(Value::as_array)
        {
            for tool_call in tool_calls {
                if tool_call
                    .get("id")
                    .and_then(Value::as_str)
                    .is_some_and(|id| !id.is_empty())
                {
                    delta.tool = true;
                }
                if let Some(name) = tool_call.pointer("/function/name").and_then(Value::as_str) {
                    if !name.is_empty() {
                        delta.tool = true;
                    }
                    assembly.tool_name = Some(name.to_owned());
                }
                if let Some(arguments) = tool_call
                    .pointer("/function/arguments")
                    .and_then(Value::as_str)
                {
                    if !arguments.is_empty() {
                        delta.tool = true;
                    }
                    assembly.tool_arguments.push_str(arguments);
                }
            }
        }
    }
    delta
}

fn usage_from_value(value: Option<&Value>) -> UsageMetrics {
    let Some(value) = value else {
        return UsageMetrics::default();
    };
    UsageMetrics {
        prompt_tokens: value.get("prompt_tokens").and_then(Value::as_u64),
        completion_tokens: value.get("completion_tokens").and_then(Value::as_u64),
        total_tokens: value.get("total_tokens").and_then(Value::as_u64),
        cached_tokens_status: cached_tokens_status(value),
        cached_tokens: value
            .pointer("/prompt_tokens_details/cached_tokens")
            .and_then(Value::as_u64),
    }
}

fn cached_tokens_status(value: &Value) -> Option<&'static str> {
    match value.pointer("/prompt_tokens_details/cached_tokens") {
        Some(Value::Number(number)) if number.as_u64().is_some() => Some("present"),
        Some(Value::Null) => Some("null"),
        Some(_) => Some("invalid"),
        None => Some("missing"),
    }
}

fn case_from_validation(
    validation: Result<(), String>,
    planned_prompt_tokens: usize,
    latency: Duration,
    stream_timing: StreamTimingReport,
    http_status: Option<u16>,
    finish_reason: Option<String>,
    usage: UsageMetrics,
) -> CaseRun {
    let latency_ms = latency.as_millis();
    let tokens_per_second = usage.completion_tokens.and_then(|tokens| {
        (latency.as_secs_f64() > 0.0).then_some(tokens as f64 / latency.as_secs_f64())
    });
    match validation {
        Ok(()) => CaseRun {
            status: "passed",
            classification: "passed".to_owned(),
            planned_prompt_tokens,
            latency_ms: Some(latency_ms),
            stream_timing,
            tokens_per_second,
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
            cached_tokens_status: usage.cached_tokens_status,
            cached_tokens: usage.cached_tokens,
            prompt_hash: None,
            http_status,
            finish_reason,
            error: None,
        },
        Err(err) => CaseRun {
            status: "failed",
            classification: "response_validation_failed".to_owned(),
            planned_prompt_tokens,
            latency_ms: Some(latency_ms),
            stream_timing,
            tokens_per_second,
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
            cached_tokens_status: usage.cached_tokens_status,
            cached_tokens: usage.cached_tokens,
            prompt_hash: None,
            http_status,
            finish_reason,
            error: Some(err),
        },
    }
}

fn merge_case_runs(first: CaseRun, second: CaseRun, second_attempt_label: &str) -> CaseRun {
    let latency_ms = sum_optional_u128(first.latency_ms, second.latency_ms);
    let completion_tokens = sum_optional_u64(first.completion_tokens, second.completion_tokens);
    let tokens_per_second = match (latency_ms, completion_tokens) {
        (Some(latency_ms), Some(tokens)) if latency_ms > 0 => {
            Some(tokens as f64 / (latency_ms as f64 / 1000.0))
        }
        _ => None,
    };
    let (status, classification, error) = if first.status != "passed" {
        (
            first.status,
            first.classification.clone(),
            first.error.clone(),
        )
    } else if second.status != "passed" {
        (
            second.status,
            format!("{second_attempt_label}_{}", second.classification),
            second.error.clone(),
        )
    } else {
        ("passed", "passed".to_owned(), None)
    };
    CaseRun {
        status,
        classification,
        planned_prompt_tokens: first
            .planned_prompt_tokens
            .saturating_add(second.planned_prompt_tokens),
        latency_ms,
        stream_timing: second.stream_timing,
        tokens_per_second,
        prompt_tokens: sum_optional_u64(first.prompt_tokens, second.prompt_tokens),
        completion_tokens,
        total_tokens: sum_optional_u64(first.total_tokens, second.total_tokens),
        cached_tokens_status: merge_cached_tokens_status(
            first.cached_tokens_status,
            second.cached_tokens_status,
        ),
        cached_tokens: sum_optional_u64(first.cached_tokens, second.cached_tokens),
        prompt_hash: None,
        http_status: second.http_status.or(first.http_status),
        finish_reason: second.finish_reason.or(first.finish_reason),
        error,
    }
}

fn failed_case(
    classification: impl Into<String>,
    planned_prompt_tokens: usize,
    latency: Duration,
    http_status: Option<u16>,
    error: String,
) -> CaseRun {
    failed_case_with_stream_timing(
        classification,
        planned_prompt_tokens,
        latency,
        http_status,
        error,
        StreamTimingReport::default(),
    )
}

fn failed_case_with_stream_timing(
    classification: impl Into<String>,
    planned_prompt_tokens: usize,
    latency: Duration,
    http_status: Option<u16>,
    error: String,
    stream_timing: StreamTimingReport,
) -> CaseRun {
    CaseRun {
        status: "failed",
        classification: classification.into(),
        planned_prompt_tokens,
        latency_ms: Some(latency.as_millis()),
        stream_timing,
        tokens_per_second: None,
        prompt_tokens: None,
        completion_tokens: None,
        total_tokens: None,
        cached_tokens_status: None,
        cached_tokens: None,
        prompt_hash: None,
        http_status,
        finish_reason: None,
        error: Some(error),
    }
}

fn apply_case_run(case: &mut BenchCaseReport, run: CaseRun) {
    let prefill = BenchPrefillReport::from_run(&run);
    let decode = BenchDecodeReport::from_run(case.decode.max_tokens, &run);
    let cache = BenchCacheBehaviorReport::from_run(&run);
    let summary = BenchCaseSummaryReport::from_run(&run);
    if let Some(prompt_hash) = &run.prompt_hash {
        case.prompt_identity.prompt_hash = prompt_hash.clone();
        case.prompt_identity.prompt_hash_source = "prompt_body";
    }
    case.status = run.status.to_owned();
    case.classification = run.classification;
    case.prefill = prefill;
    case.decode = decode;
    case.cache = cache;
    case.summary = summary;
    case.planned_prompt_tokens = Some(run.planned_prompt_tokens);
    case.latency_ms = run.latency_ms;
    case.stream_timing = run.stream_timing;
    case.tokens_per_second = run.tokens_per_second;
    case.prompt_tokens = run.prompt_tokens;
    case.completion_tokens = run.completion_tokens;
    case.total_tokens = run.total_tokens;
    case.cached_tokens_status = run.cached_tokens_status;
    case.cached_tokens = run.cached_tokens;
    case.http_status = run.http_status;
    case.finish_reason = run.finish_reason;
    case.error = run.error;
}

fn apply_case_admin_metrics(
    case: &mut BenchCaseReport,
    before: Result<Option<PrefixCacheMetricsReport>, String>,
    after: Result<Option<PrefixCacheMetricsReport>, String>,
) {
    case.admin_metrics = Some(match (before, after) {
        (Ok(Some(before)), Ok(Some(after))) => {
            BenchCaseAdminMetricsReport::from_prefix_cache_snapshots(before, after)
        }
        (Err(before), Err(after)) => BenchCaseAdminMetricsReport::error(format!(
            "before admin metrics failed: {before}; after admin metrics failed: {after}"
        )),
        (Err(before), _) => {
            BenchCaseAdminMetricsReport::error(format!("before admin metrics failed: {before}"))
        }
        (_, Err(after)) => {
            BenchCaseAdminMetricsReport::error(format!("after admin metrics failed: {after}"))
        }
        (Ok(None), Ok(None)) => {
            BenchCaseAdminMetricsReport::error("prefix cache admin metrics missing".to_owned())
        }
        (Ok(None), Ok(Some(_))) => BenchCaseAdminMetricsReport::error(
            "prefix cache admin metrics missing before case".to_owned(),
        ),
        (Ok(Some(_)), Ok(None)) => BenchCaseAdminMetricsReport::error(
            "prefix cache admin metrics missing after case".to_owned(),
        ),
    });
}

fn apply_case_cache_metric_validation(case: &mut BenchCaseReport, case_kind: BenchCaseKind) {
    if !case_kind.requires_prefix_cache_validation() || case.status != "passed" {
        return;
    }
    let Some(delta) = case
        .admin_metrics
        .as_ref()
        .and_then(BenchCaseAdminMetricsReport::prefix_cache_delta)
    else {
        case.status = "failed".to_owned();
        case.classification = "prefix_cache_metrics_missing".to_owned();
        case.error =
            Some("prefix cache admin metrics were not available for cache probe".to_owned());
        return;
    };
    if delta.hit_tokens == 0 {
        case.status = "failed".to_owned();
        case.classification = "prefix_cache_hit_tokens_missing".to_owned();
        case.error =
            Some("prefix cache hit_tokens did not increase during cache probe case".to_owned());
    }
}

fn time_to_first_token_ms(stream_timing: StreamTimingReport) -> Option<u128> {
    stream_timing.first_semantic_delta_latency_ms
}

fn uncached_tokens(prompt_tokens: Option<u64>, cached_tokens: Option<u64>) -> Option<u64> {
    Some(prompt_tokens?.saturating_sub(cached_tokens?))
}

fn cache_lookup_result(
    cached_tokens_status: Option<&'static str>,
    cached_tokens: Option<u64>,
) -> Option<&'static str> {
    match (cached_tokens_status, cached_tokens) {
        (_, Some(0)) => Some("miss"),
        (_, Some(_)) => Some("hit"),
        (Some("missing"), None) => Some("not_reported"),
        (Some("null"), None) => Some("null"),
        (Some("invalid"), None) => Some("invalid"),
        _ => None,
    }
}

fn apply_baseline_comparison(
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

async fn write_and_print_report(
    report: &BenchReport,
    output_path: Option<&Path>,
) -> anyhow::Result<()> {
    let json = serde_json::to_string_pretty(report)?;
    if let Some(path) = output_path {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            tokio::fs::create_dir_all(parent)
                .await
                .with_context(|| format!("create trace output directory `{}`", parent.display()))?;
        }
        tokio::fs::write(path, json.as_bytes())
            .await
            .with_context(|| format!("write benchmark trace `{}`", path.display()))?;
    }
    println!("{json}");
    Ok(())
}

fn marker_for_case(profile: BenchProfileKind, case: BenchCaseKind) -> String {
    format!(
        "KIR_LONG_CONTEXT_{}_{}_QUARTZ_2741",
        profile.short_label().to_ascii_uppercase(),
        case.name().replace('-', "_").to_ascii_uppercase()
    )
}

fn unix_now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn detect_cpu_name() -> Option<String> {
    if std::env::consts::OS == "macos" {
        command_output("sysctl", &["-n", "machdep.cpu.brand_string"])
    } else if std::env::consts::OS == "linux" {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|cpuinfo| {
                cpuinfo.lines().find_map(|line| {
                    line.strip_prefix("model name").and_then(|rest| {
                        rest.split_once(':')
                            .map(|(_, value)| value.trim().to_owned())
                    })
                })
            })
    } else {
        None
    }
}

fn command_output(command: &str, args: &[&str]) -> Option<String> {
    let output = std::process::Command::new(command)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let text = text.trim();
    (!text.is_empty()).then_some(text.to_owned())
}

#[cfg(test)]
mod tests;
