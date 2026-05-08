use crate::{flag_value, has_flag};
use anyhow::{Context, anyhow};
use futures::StreamExt;
use llm_hub::ModelStore;
use llm_models::{ModelFamilyAdapter, QwenFamilyAdapter};
use llm_tokenizer::HuggingFaceTokenizer;
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const GATE_NAME: &str = "qwen-long-context";
const CACHE_LAYOUT: &str = "shared-prefix-v1";
const DEFAULT_MODEL_ID: &str = "local-qwen36";
const DEFAULT_TIMEOUT_MS: u64 = 30 * 60 * 1000;
const DEFAULT_CONNECT_TIMEOUT_MS: u64 = 10 * 1000;
const DEFAULT_MAX_TOKENS: u32 = 128;
const DEFAULT_LATENCY_REGRESSION_THRESHOLD: f64 = 0.20;

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
        other => anyhow::bail!("unknown bench subcommand `{other}`"),
    }
}

fn print_bench_help() {
    println!(
        "\
Usage: llm-engine bench qwen-long-context [OPTIONS]

Options:
  --endpoint <url>                    OpenAI-compatible server base URL
  --model <id>                        Model id to send in requests [default: local-qwen36]
  --snapshot <path>                   Qwen snapshot path with tokenizer.json and manifest
  --lane <spec>                       Named lane: name=<id>,endpoint=<url>,snapshot=<path>[,model=<id>]
  --profile <135k|200k|all>           Benchmark profile [default: 135k]
  --baseline <path>                   Previous trace JSON for same hardware/model comparison
  --output <path>                     Write the trace JSON to a file as well as stdout
  --max-tokens <n>                    Completion token limit per request [default: 128]
  --admin-token <token>               Optional bearer token for lane /admin/metrics snapshots
  --timeout-ms <n>                    Whole request timeout [default: 1800000]
  --connect-timeout-ms <n>            HTTP connect timeout [default: 10000]
  --latency-regression-threshold <f>  Allowed latency increase over baseline [default: 0.20]
  --dry-run                           Print the exact gate plan without HTTP requests
  -h, --help                          Print help"
    );
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
                .map(|profile| profile_report(*profile))
                .collect(),
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
            .expect("lane endpoint is checked above");
        let snapshot_path = lane_config
            .snapshot_path
            .as_deref()
            .expect("lane snapshot is checked above");
        let tokenizer = load_qwen_tokenizer(snapshot_path)?;
        let run_context = BenchExecutionContext {
            client: &client,
            baseline_trace: baseline_trace.as_ref(),
            hardware: &report.hardware,
            latency_regression_threshold,
            max_tokens,
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
    report.comparison = Some(compare_bench_lanes(&report.lanes));
    report.status = if release_blocking_failed {
        "failed".to_owned()
    } else {
        "passed".to_owned()
    };

    write_and_print_report(&report, output_path.as_deref()).await?;
    if release_blocking_failed {
        anyhow::bail!("qwen long-context promotion gate failed");
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

fn flag_values<'a>(args: &'a [String], flag: &str) -> Vec<&'a str> {
    args.windows(2)
        .filter_map(|window| (window[0] == flag).then_some(window[1].as_str()))
        .collect()
}

async fn capture_lane_admin_metrics(
    lane: &mut BenchLaneReport,
    client: &reqwest::Client,
    endpoint: &str,
    admin_token: Option<&str>,
) {
    let url = format!("{endpoint}/admin/metrics");
    let mut request = client.get(url);
    if let Some(token) = admin_token {
        request = request.bearer_auth(token);
    }
    match request.send().await {
        Ok(response) if response.status().is_success() => match response.json::<Value>().await {
            Ok(metrics) => lane.admin_metrics = Some(metrics),
            Err(err) => lane.admin_metrics_error = Some(format!("parse admin metrics: {err}")),
        },
        Ok(response) => {
            lane.admin_metrics_error = Some(format!("admin metrics HTTP {}", response.status()));
        }
        Err(err) => {
            lane.admin_metrics_error = Some(format!("admin metrics request failed: {err}"));
        }
    }
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
                    ttft_ms: lane_case.ttft_ms,
                    tokens_per_second: lane_case.tokens_per_second,
                    prompt_tokens: lane_case.prompt_tokens,
                    completion_tokens: lane_case.completion_tokens,
                    total_tokens: lane_case.total_tokens,
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
}

impl BenchProfileKind {
    fn name(self) -> &'static str {
        match self {
            Self::Promotion135k => "qwen-135k-promotion",
            Self::Characterization200k => "qwen-200k-characterization",
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        match name {
            "qwen-135k-promotion" => Some(Self::Promotion135k),
            "qwen-200k-characterization" => Some(Self::Characterization200k),
            _ => None,
        }
    }

    fn target_tokens(self) -> usize {
        match self {
            Self::Promotion135k => 135_000,
            Self::Characterization200k => 200_000,
        }
    }

    fn release_blocking(self) -> bool {
        matches!(self, Self::Promotion135k)
    }

    fn short_label(self) -> &'static str {
        match self {
            Self::Promotion135k => "135k",
            Self::Characterization200k => "200k",
        }
    }
}

fn selected_profiles(profile: &str) -> anyhow::Result<Vec<BenchProfileKind>> {
    match profile {
        "135k" | "135K" | "qwen-135k-promotion" => Ok(vec![BenchProfileKind::Promotion135k]),
        "200k" | "200K" | "qwen-200k-characterization" => {
            Ok(vec![BenchProfileKind::Characterization200k])
        }
        "all" => Ok(vec![
            BenchProfileKind::Promotion135k,
            BenchProfileKind::Characterization200k,
        ]),
        other => anyhow::bail!("unknown qwen long-context profile `{other}`"),
    }
}

#[derive(Debug, Clone, Copy)]
enum BenchCaseKind {
    PlainRecall,
    JsonObjectRecall,
    RequiredToolRecall,
    StreamedRequiredToolRecall,
    MultiTurnLifecycle,
}

impl BenchCaseKind {
    fn all() -> [Self; 5] {
        [
            Self::PlainRecall,
            Self::JsonObjectRecall,
            Self::RequiredToolRecall,
            Self::StreamedRequiredToolRecall,
            Self::MultiTurnLifecycle,
        ]
    }

    fn name(self) -> &'static str {
        match self {
            Self::PlainRecall => "plain-recall",
            Self::JsonObjectRecall => "json-object-recall",
            Self::RequiredToolRecall => "required-tool-recall",
            Self::StreamedRequiredToolRecall => "streamed-required-tool-recall",
            Self::MultiTurnLifecycle => "multi-turn-lifecycle",
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        match name {
            "plain-recall" => Some(Self::PlainRecall),
            "json-object-recall" => Some(Self::JsonObjectRecall),
            "required-tool-recall" => Some(Self::RequiredToolRecall),
            "streamed-required-tool-recall" => Some(Self::StreamedRequiredToolRecall),
            "multi-turn-lifecycle" => Some(Self::MultiTurnLifecycle),
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
        }
    }

    fn response_contract(self) -> &'static str {
        match self {
            Self::PlainRecall => "assistant content must contain the target marker",
            Self::JsonObjectRecall => "assistant content must be a JSON object with marker",
            Self::RequiredToolRecall => {
                "assistant must call report_long_context_recall with marker arguments"
            }
            Self::StreamedRequiredToolRecall => {
                "SSE deltas must assemble to report_long_context_recall with marker arguments"
            }
            Self::MultiTurnLifecycle => {
                "multi-message chat response must recall the target marker from the first turn"
            }
        }
    }

    fn streams(self) -> bool {
        matches!(self, Self::StreamedRequiredToolRecall)
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
    baseline: BaselineReport,
    profiles: Vec<BenchProfileReport>,
    lanes: Vec<BenchLaneReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    comparison: Option<BenchLaneComparisonReport>,
}

#[derive(Debug, Clone, Serialize)]
struct BenchLaneReport {
    name: String,
    status: String,
    model: ModelIdentityReport,
    profiles: Vec<BenchProfileReport>,
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
    ttft_ms: Option<u128>,
    tokens_per_second: Option<f64>,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    total_tokens: Option<u64>,
    classification: String,
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
    marker: String,
    status: String,
    classification: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    planned_prompt_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    latency_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ttft_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokens_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    completion_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    http_status: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    baseline: Option<BaselineComparisonReport>,
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
                "snapshot_manifest_digest",
                "prompt_template",
                "tool_schema",
                "request_mode",
                "sampling",
                "cache_layout",
                "cache_capacity",
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
    ttft_ms: Option<u128>,
    tokens_per_second: Option<f64>,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    total_tokens: Option<u64>,
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

#[derive(Debug, Default)]
struct UsageMetrics {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    total_tokens: Option<u64>,
}

#[derive(Debug, Default)]
struct StreamAssembly {
    content: String,
    tool_name: Option<String>,
    tool_arguments: String,
    finish_reason: Option<String>,
    usage: UsageMetrics,
}

fn profile_report(profile: BenchProfileKind) -> BenchProfileReport {
    BenchProfileReport {
        name: profile.name(),
        target_tokens: profile.target_tokens(),
        release_blocking: profile.release_blocking(),
        status: "planned".to_owned(),
        cases: BenchCaseKind::all()
            .iter()
            .copied()
            .map(|case| case_report(profile, case))
            .collect(),
    }
}

fn case_report(profile: BenchProfileKind, case: BenchCaseKind) -> BenchCaseReport {
    BenchCaseReport {
        name: case.name(),
        mode: case.mode(),
        target_tokens: profile.target_tokens(),
        stream: case.streams(),
        response_contract: case.response_contract(),
        marker: marker_for_case(profile, case),
        status: "planned".to_owned(),
        classification: if profile.release_blocking() {
            "release-blocking".to_owned()
        } else {
            "frontier-characterization".to_owned()
        },
        planned_prompt_tokens: None,
        latency_ms: None,
        ttft_ms: None,
        tokens_per_second: None,
        prompt_tokens: None,
        completion_tokens: None,
        total_tokens: None,
        http_status: None,
        finish_reason: None,
        error: None,
        baseline: None,
    }
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
                ttft_ms: None,
                tokens_per_second: None,
                prompt_tokens: None,
                completion_tokens: None,
                total_tokens: None,
                http_status: None,
                finish_reason: None,
                error: Some(err.to_string()),
            };
        }
    };
    let body = request_body(model_id, profile, case, &prompt, max_tokens);
    if case.streams() {
        run_streaming_case(client, endpoint, case, prompt, body).await
    } else {
        run_buffered_case(client, endpoint, case, prompt, body).await
    }
}

async fn run_buffered_case(
    client: &reqwest::Client,
    endpoint: &str,
    case: BenchCaseKind,
    prompt: PromptBuild,
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
    let validation = validate_buffered_case(case, &prompt.marker, &value);
    case_from_validation(
        validation,
        prompt.token_count,
        latency,
        None,
        http_status,
        finish_reason,
        usage,
    )
}

async fn run_streaming_case(
    client: &reqwest::Client,
    endpoint: &str,
    case: BenchCaseKind,
    prompt: PromptBuild,
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
    let mut ttft = None;
    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(chunk) => chunk,
            Err(err) => {
                return failed_case(
                    "stream_body_failed",
                    prompt.token_count,
                    started.elapsed(),
                    http_status,
                    err.to_string(),
                );
            }
        };
        if ttft.is_none() {
            ttft = Some(started.elapsed());
        }
        buffer.push_str(&String::from_utf8_lossy(&chunk));
        consume_sse_buffer(&mut buffer, &mut assembly);
    }
    if !buffer.is_empty() {
        buffer.push('\n');
        consume_sse_buffer(&mut buffer, &mut assembly);
    }
    let latency = started.elapsed();
    let validation = validate_streaming_case(case, &prompt.marker, &assembly);
    case_from_validation(
        validation,
        prompt.token_count,
        latency,
        ttft,
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
    }
    body
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

fn validate_buffered_case(case: BenchCaseKind, marker: &str, value: &Value) -> Result<(), String> {
    match case {
        BenchCaseKind::PlainRecall | BenchCaseKind::MultiTurnLifecycle => {
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
            let object = parsed
                .as_object()
                .ok_or_else(|| "assistant JSON content was not an object".to_owned())?;
            if object.get("marker").and_then(Value::as_str) == Some(marker) {
                Ok(())
            } else {
                Err(format!("JSON marker did not equal `{marker}`"))
            }
        }
        BenchCaseKind::RequiredToolRecall => {
            let tool_call = value
                .pointer("/choices/0/message/tool_calls/0")
                .ok_or_else(|| "missing required tool call".to_owned())?;
            validate_tool_call(tool_call, marker)
        }
        BenchCaseKind::StreamedRequiredToolRecall => {
            Err("streamed tool case was routed through buffered validator".to_owned())
        }
    }
}

fn validate_streaming_case(
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
    let args = serde_json::from_str::<Value>(&assembly.tool_arguments)
        .map_err(|err| format!("streamed tool arguments were not JSON: {err}"))?;
    if args.get("marker").and_then(Value::as_str) == Some(marker) {
        Ok(())
    } else {
        Err(format!("streamed tool marker did not equal `{marker}`"))
    }
}

fn validate_tool_call(tool_call: &Value, marker: &str) -> Result<(), String> {
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
    if args.get("marker").and_then(Value::as_str) == Some(marker) {
        Ok(())
    } else {
        Err(format!("tool marker did not equal `{marker}`"))
    }
}

fn consume_sse_buffer(buffer: &mut String, assembly: &mut StreamAssembly) {
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
        if let Some(usage) = value.get("usage") {
            assembly.usage = usage_from_value(Some(usage));
        }
        if let Some(choice) = value
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
        {
            if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
                assembly.finish_reason = Some(reason.to_owned());
            }
            if let Some(content) = choice.pointer("/delta/content").and_then(Value::as_str) {
                assembly.content.push_str(content);
            }
            if let Some(tool_calls) = choice
                .pointer("/delta/tool_calls")
                .and_then(Value::as_array)
            {
                for tool_call in tool_calls {
                    if let Some(name) = tool_call.pointer("/function/name").and_then(Value::as_str)
                    {
                        assembly.tool_name = Some(name.to_owned());
                    }
                    if let Some(arguments) = tool_call
                        .pointer("/function/arguments")
                        .and_then(Value::as_str)
                    {
                        assembly.tool_arguments.push_str(arguments);
                    }
                }
            }
        }
    }
}

fn usage_from_value(value: Option<&Value>) -> UsageMetrics {
    let Some(value) = value else {
        return UsageMetrics::default();
    };
    UsageMetrics {
        prompt_tokens: value.get("prompt_tokens").and_then(Value::as_u64),
        completion_tokens: value.get("completion_tokens").and_then(Value::as_u64),
        total_tokens: value.get("total_tokens").and_then(Value::as_u64),
    }
}

fn case_from_validation(
    validation: Result<(), String>,
    planned_prompt_tokens: usize,
    latency: Duration,
    ttft: Option<Duration>,
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
            ttft_ms: ttft.map(|duration| duration.as_millis()),
            tokens_per_second,
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
            http_status,
            finish_reason,
            error: None,
        },
        Err(err) => CaseRun {
            status: "failed",
            classification: "response_validation_failed".to_owned(),
            planned_prompt_tokens,
            latency_ms: Some(latency_ms),
            ttft_ms: ttft.map(|duration| duration.as_millis()),
            tokens_per_second,
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
            http_status,
            finish_reason,
            error: Some(err),
        },
    }
}

fn failed_case(
    classification: impl Into<String>,
    planned_prompt_tokens: usize,
    latency: Duration,
    http_status: Option<u16>,
    error: String,
) -> CaseRun {
    CaseRun {
        status: "failed",
        classification: classification.into(),
        planned_prompt_tokens,
        latency_ms: Some(latency.as_millis()),
        ttft_ms: None,
        tokens_per_second: None,
        prompt_tokens: None,
        completion_tokens: None,
        total_tokens: None,
        http_status,
        finish_reason: None,
        error: Some(error),
    }
}

fn apply_case_run(case: &mut BenchCaseReport, run: CaseRun) {
    case.status = run.status.to_owned();
    case.classification = run.classification;
    case.planned_prompt_tokens = Some(run.planned_prompt_tokens);
    case.latency_ms = run.latency_ms;
    case.ttft_ms = run.ttft_ms;
    case.tokens_per_second = run.tokens_per_second;
    case.prompt_tokens = run.prompt_tokens;
    case.completion_tokens = run.completion_tokens;
    case.total_tokens = run.total_tokens;
    case.http_status = run.http_status;
    case.finish_reason = run.finish_reason;
    case.error = run.error;
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

fn normalize_endpoint(endpoint: &str) -> String {
    endpoint.trim_end_matches('/').to_owned()
}

fn parse_u64_flag(args: &[String], flag: &str, default: u64) -> anyhow::Result<u64> {
    flag_value(args, flag)
        .map(str::parse::<u64>)
        .transpose()
        .with_context(|| format!("parse {flag}"))?
        .map_or(Ok(default), |value| {
            if value == 0 {
                anyhow::bail!("{flag} must be greater than zero");
            }
            Ok(value)
        })
}

fn parse_u32_flag(args: &[String], flag: &str, default: u32) -> anyhow::Result<u32> {
    flag_value(args, flag)
        .map(str::parse::<u32>)
        .transpose()
        .with_context(|| format!("parse {flag}"))?
        .map_or(Ok(default), |value| {
            if value == 0 {
                anyhow::bail!("{flag} must be greater than zero");
            }
            Ok(value)
        })
}

fn parse_f64_flag(args: &[String], flag: &str, default: f64) -> anyhow::Result<f64> {
    let value = flag_value(args, flag)
        .map(str::parse::<f64>)
        .transpose()
        .with_context(|| format!("parse {flag}"))?
        .unwrap_or(default);
    if !value.is_finite() || value < 0.0 {
        anyhow::bail!("{flag} must be a finite non-negative number");
    }
    Ok(value)
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
