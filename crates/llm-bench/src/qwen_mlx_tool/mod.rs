use super::{
    DEFAULT_CONNECT_TIMEOUT_MS, DEFAULT_TIMEOUT_MS, HardwareReport, ModelIdentityReport,
    StreamAssembly, StreamTimingReport, StreamTimingTracker, cli::flag_values, consume_sse_buffer,
    load_model_identity, load_qwen_tokenizer, unix_now_ms, usage_from_value,
};
use crate::{flag_value, has_flag};
use anyhow::{Context, anyhow};
use futures::StreamExt;
use futures::future::join_all;
use llm_api::canonicalize_json_value;
use llm_tokenizer::HuggingFaceTokenizer;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

mod config;
mod metrics;
mod report;

use metrics::*;
use report::*;

use config::{
    NormalizedLaneConfig, NormalizedLaneKind, NormalizedRunConfig, NormalizedSweepProfile,
    QWEN_MLX_PREFILL_135K_PROFILE, default_run_config_for_probe_suite, parse_cache_phases_flag,
    parse_count_flag, parse_lane_specs, parse_millis_flag, parse_optional_count_flag,
    parse_probe_suite_flag, parse_sweep_profile_flag, sweep_profile_requires_exact_token_prompt,
};

#[cfg(test)]
use config::{
    DefaultOrU32, DefaultOrU64, NormalizedModelAddressing, NormalizedTemplatePolicy, UnsetOrU64,
    effective_concurrent_samples, parse_lane_spec,
};

const BENCHMARK_NAME: &str = "qwen-mlx-tool-normalized";
const DEFAULT_MAX_TOKENS: u32 = 512;
const REQUIRED_TOOL_TTFT_MAX_TOKENS: [u32; 3] = [24, 48, 96];
const PREFILL_SWEEP_135K_PROFILE_NAME: &str = "qwen-prefill-sweep-135k";
const CHAT_STREAM_MARKER: &str = "KIR_QWEN_MLX_PREFILL_135K_CHAT_STREAM_QUARTZ_2741";
const CONTEXT_RECALL_STREAM_135K_MARKER: &str =
    "KIR_LONG_CONTEXT_135K_CONTEXT_RECALL_STREAM_135K_QUARTZ_2741";
const BENCH_REPO_DIR_ENV: &str = "LLM_ENGINE_BENCH_REPO_DIR";
const BENCH_REPO_COMMIT_ENV: &str = "LLM_ENGINE_BENCH_REPO_COMMIT";
const BENCH_REPO_BRANCH_ENV: &str = "LLM_ENGINE_BENCH_REPO_BRANCH";
const BENCH_REPO_DIRTY_ENV: &str = "LLM_ENGINE_BENCH_REPO_DIRTY";
const BENCH_REPO_ORIGIN_FILE: &str = ".kir-ai-origin.json";
const ADMIN_METRICS_TIMEOUT_MS: u64 = 250;
const AGENTIC_AB_CASE: &str = "tool_required_stream";
const AGENTIC_AB_SCHEMA_VARIANT: &str = "canonical_current";
const AGENTIC_AB_TOOL_CHOICE_VARIANT: &str = "required";
const AGENTIC_AB_FAST_PATH_KIND: &str = "kir_ai_proxy";

pub(super) async fn run_qwen_mlx_tool_normalized_bench(args: &[String]) -> anyhow::Result<()> {
    if has_flag(args, "--help") || has_flag(args, "-h") {
        print_help();
        return Ok(());
    }

    let dry_run = has_flag(args, "--dry-run");
    let sweep_profile = parse_sweep_profile_flag(args)?;
    let probe_suite = parse_probe_suite_flag(args, sweep_profile)?;
    let default_run_config = default_run_config_for_probe_suite(probe_suite);
    let warmups = parse_count_flag(args, "--warmups", default_run_config.warmups, true)?;
    let samples = parse_count_flag(args, "--samples", default_run_config.samples, false)?;
    let context_tokens = parse_count_flag(
        args,
        "--context-tokens",
        default_run_config.context_tokens,
        false,
    )?;
    let concurrent_requests = parse_count_flag(
        args,
        "--concurrent-requests",
        default_run_config.concurrent_requests,
        false,
    )?;
    let concurrent_samples = parse_count_flag(
        args,
        "--concurrent-samples",
        default_run_config.concurrent_samples,
        true,
    )?;
    let cache_phases = if flag_values(args, "--cache-phases").is_empty() {
        default_run_config.cache_phases
    } else {
        parse_cache_phases_flag(args)?
    };
    let run_config = NormalizedRunConfig::new(
        warmups,
        samples,
        context_tokens,
        concurrent_requests,
        concurrent_samples,
    )
    .with_cache_phases(cache_phases);
    let timeout_ms = parse_millis_flag(args, "--timeout-ms", DEFAULT_TIMEOUT_MS)?;
    let connect_timeout_ms =
        parse_millis_flag(args, "--connect-timeout-ms", DEFAULT_CONNECT_TIMEOUT_MS)?;
    let output_path = flag_value(args, "--output").map(PathBuf::from);
    let ab_baseline_path = flag_value(args, "--ab-baseline").map(PathBuf::from);
    let engine_db_baseline_path = flag_value(args, "--engine-db-baselines").map(PathBuf::from);
    let engine_db_baselines =
        load_engine_db_baseline_export(engine_db_baseline_path.as_deref()).await?;
    let admin_token = flag_value(args, "--admin-token").map(str::to_owned);
    let max_requests = parse_optional_count_flag(args, "--max-requests")?;
    let max_planned_prompt_tokens = parse_optional_count_flag(args, "--max-planned-prompt-tokens")?;
    let probes = probe_suite.probes();
    let lanes = parse_lane_specs(args)?;
    let plan_summary = normalized_plan_summary(&lanes, &probes, &run_config);

    let mut lane_reports = Vec::with_capacity(lanes.len());
    for lane in &lanes {
        let snapshot_identity = load_lane_snapshot_identity(lane, dry_run).await?;
        lane_reports.push(if dry_run {
            NormalizedLaneReport::dry_run(lane, &run_config, snapshot_identity, &probes)
        } else {
            NormalizedLaneReport::planned_with_requests(
                lane,
                run_config.warmups,
                run_config.samples,
                &run_config,
                snapshot_identity,
                &probes,
            )
        });
    }
    let latest_performance_comparison =
        latest_performance_comparison_report(&lane_reports, engine_db_baselines.as_ref());

    let mut report = NormalizedBenchReport {
        benchmark: BENCHMARK_NAME,
        sweep_profile: sweep_profile.map(NormalizedSweepProfile::as_str),
        status: if dry_run { "dry_run" } else { "running" }.to_owned(),
        generated_at_unix_ms: unix_now_ms(),
        trace_output_path: output_path.as_ref().map(|path| path.display().to_string()),
        warmups: run_config.warmups,
        samples: run_config.samples,
        context_tokens: run_config.context_tokens,
        concurrent_requests: run_config.concurrent_requests,
        concurrent_samples: run_config.concurrent_samples,
        effective_concurrent_samples: run_config.effective_concurrent_samples,
        timeout_ms,
        connect_timeout_ms,
        probe_suite: probe_suite.name(),
        repo_revision: RepoRevisionReport::detect(),
        cases: probe_suite.case_names(&probes),
        schema_variants: probe_suite.schema_variant_names(&probes),
        tool_choice_variants: probe_suite.tool_choice_variant_names(&probes),
        cache_phases: run_config
            .cache_phases
            .iter()
            .map(|phase| phase.name())
            .collect(),
        plan_summary,
        summary: aggregate_normalized_summary_for_phases(
            &lane_reports,
            &probes,
            &run_config.cache_phases,
        ),
        tool_required_stream: NormalizedToolRequiredStreamTimingReport::dry_run(&lane_reports),
        required_tool_ttft_matrix: NormalizedRequiredToolTtftMatrixReport::dry_run(),
        lanes: lane_reports,
        hardware: HardwareReport::detect(),
        comparison: NormalizedComparisonReport::dry_run(),
        agentic_gate: NormalizedAgenticGateReport::dry_run(),
        agentic_streaming_fast_path_ab: NormalizedAgenticStreamingFastPathAbReport::dry_run(
            ab_baseline_path.as_deref(),
        ),
        prefill_concurrency: NormalizedPrefillConcurrencyReport::dry_run(),
        prefill_sweep: NormalizedPrefillSweepReport::dry_run(),
        stable_prefix: NormalizedStablePrefixReport::dry_run(),
        latest_performance_comparison,
    };

    if dry_run {
        write_and_print_normalized_report(&report, output_path.as_deref()).await?;
        return Ok(());
    }

    enforce_plan_budget(
        &report.plan_summary,
        max_requests,
        max_planned_prompt_tokens,
    )?;

    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_millis(connect_timeout_ms))
        .timeout(Duration::from_millis(timeout_ms))
        .build()
        .context("build qwen mlx tool normalized benchmark HTTP client")?;

    let progress = NormalizedProgress::new(report.plan_summary.total_http_requests);
    for (lane, lane_report) in lanes.iter().zip(&mut report.lanes) {
        let tokenizer = if sweep_profile_requires_exact_token_prompt(sweep_profile) {
            Some(load_lane_tokenizer(lane)?)
        } else {
            None
        };
        run_lane(
            lane,
            lane_report,
            LaneRunContext {
                client: &client,
                run_config: &run_config,
                probes: &probes,
                admin_token: admin_token.as_deref(),
                prompt_tokenizer: tokenizer.as_ref(),
                progress: &progress,
            },
        )
        .await;
    }
    report.summary =
        aggregate_normalized_summary_for_phases(&report.lanes, &probes, &run_config.cache_phases);
    report.tool_required_stream = tool_required_stream_timing_report(&report.lanes);
    report.required_tool_ttft_matrix =
        required_tool_ttft_matrix_report(&report.lanes, &probes, &run_config.cache_phases);
    report.comparison =
        compare_normalized_lanes_for_phases(&report.lanes, &probes, &run_config.cache_phases);
    report.agentic_gate = agentic_gate_report_for_phases(&report.lanes, &run_config.cache_phases);
    report.agentic_streaming_fast_path_ab =
        load_agentic_streaming_fast_path_ab_report(ab_baseline_path.as_deref(), &report.lanes)
            .await?;
    report.prefill_concurrency = prefill_concurrency_report(&report.lanes, &probes);
    report.prefill_sweep =
        prefill_sweep_report_for_phases(&report.lanes, &probes, &run_config.cache_phases);
    report.stable_prefix =
        stable_prefix_report_for_phases(&report.lanes, &probes, &run_config.cache_phases);
    report.latest_performance_comparison =
        latest_performance_comparison_report(&report.lanes, engine_db_baselines.as_ref());
    report.status = if report.lanes.iter().all(|lane| lane.status == "passed")
        && report.agentic_streaming_fast_path_ab.status != "failed"
    {
        "passed"
    } else {
        "failed"
    }
    .to_owned();

    write_and_print_normalized_report(&report, output_path.as_deref()).await?;
    if report.status != "passed" {
        anyhow::bail!("qwen mlx tool normalized benchmark failed");
    }
    Ok(())
}

fn print_help() {
    println!(
        "\
Usage:
  llm-bench qwen-mlx-tool-normalized [OPTIONS]
  llm-engine bench qwen-mlx-tool-normalized [OPTIONS]

Options:
  --sweep-profile <name>        Built-in lane matrix: qwen-mlx-cache-prefill, qwen-mlx-prefill-135k, qwen-mlx-prefill-135k-experimental, or qwen-mlx-stable-prefix (requires --snapshot)
  --probe-suite <name>          Probe suite: full-matrix, focused-agentic-gate, required-tool-ttft-matrix, prefill-sweep-135k, prefill-sweep-135k-context-recall, stable-agent-prefix, or stable-prefix-smoke
  --snapshot <path>             Raw Hugging Face snapshot path for built-in sweep profiles
  --cache-phases <csv>          Cache phases to run: cold, warm_same_prompt, warm_same_tool_schema [default: all]
  --only-lanes <csv>            Keep only the named lanes after built-in profile expansion
  --profile-lanes <csv>         Alias for --only-lanes
  --lane <spec>                 Lane: name=<id>,endpoint=<url>,model=<id>[,launched_model_id=<id-or-path>][,snapshot=<path>][,kind=direct_mlx|kir_ai_proxy|other][,model_addressing=loaded_model_id|default_model|server_default|custom][,template=qwen-no-thinking|sidecar-chat-template-args|none][,tool_parser=auto|json|qwen-xml][,mlx_prompt_cache_size=default|<n>][,mlx_prompt_cache_bytes=unset|<n>][,mlx_prefill_step_size=default|<n>][,mlx_prompt_concurrency=default|<n>][,mlx_decode_concurrency=default|<n>]
  --warmups <n>                 Warmup requests for warm phases [default: 1]
  --samples <n>                 Measured samples per case and phase [default: 1]
  --context-tokens <n>          Stable long-context prompt target [default: 135000]
  --concurrent-requests <n>     Requests to issue together during the concurrent pass [default: 1]
  --concurrent-samples <n>      Concurrent sample batches per case and phase; 0 disables unless concurrent requests > 1 [default: 0]
  --max-requests <n>            Fail before live HTTP requests if the selected plan exceeds this many requests
  --max-planned-prompt-tokens <n>
                                Fail before live HTTP requests if planned prompt-token budget exceeds this value
  --focused-agentic-gate        Compatibility alias for --probe-suite focused-agentic-gate
  --ab-baseline <path>          Compare against a prior qwen-mlx-tool-normalized trace; fails if Kir proxy first tool delta does not advance or final validation changes
  --output <path>               Write the trace JSON to a file as well as stdout
  --engine-db-baselines <path>  JSON export of benchmark DB baseline rows for latest Kir/direct comparison
  --admin-token <token>         Optional bearer token for lane /admin/metrics snapshots
  --timeout-ms <n>              Whole request timeout [default: 1800000]
  --connect-timeout-ms <n>      HTTP connect timeout [default: 10000]
  --dry-run                     Print the exact probe plan without HTTP requests
  -h, --help                    Print help"
    );
}

#[derive(Debug)]
struct NormalizedProgress {
    total_requests: usize,
    started_requests: AtomicUsize,
    started_at: Instant,
}

impl NormalizedProgress {
    fn new(total_requests: usize) -> Self {
        Self {
            total_requests,
            started_requests: AtomicUsize::new(0),
            started_at: Instant::now(),
        }
    }

    fn record_request_start(
        &self,
        lane: &NormalizedLaneConfig,
        probe: NormalizedProbePlan,
        planned: PlannedRun,
    ) {
        if self.total_requests == 0 {
            return;
        }
        let started = self.started_requests.fetch_add(1, Ordering::Relaxed) + 1;
        let elapsed = self.started_at.elapsed();
        let remaining = self.total_requests.saturating_sub(started);
        let eta = if started > 0 && remaining > 0 {
            elapsed.mul_f64(remaining as f64 / started as f64)
        } else {
            Duration::ZERO
        };
        eprintln!(
            "qwen-mlx-tool-normalized progress: request {started}/{} lane={} case={} schema_variant={} tool_choice_variant={} max_tokens={} cache_phase={} request_kind={} run_mode={} eta_seconds={:.1}",
            self.total_requests,
            lane.name,
            probe.case.name(),
            probe.schema_variant.name(),
            probe.tool_choice_variant.name(),
            probe.max_tokens,
            planned.phase.name(),
            planned.kind.name(),
            planned.run_mode.name(),
            eta.as_secs_f64()
        );
    }
}

async fn load_lane_snapshot_identity(
    lane: &NormalizedLaneConfig,
    dry_run: bool,
) -> anyhow::Result<Option<ModelIdentityReport>> {
    let Some(snapshot_path) = lane.snapshot_path.as_deref() else {
        return Ok(None);
    };
    if !snapshot_path.join("llm-engine-manifest.json").is_file() {
        return Ok(Some(raw_snapshot_identity(lane, snapshot_path)));
    }
    let identity_model_id = lane.identity_model_id();
    load_model_identity(
        &identity_model_id,
        Some(&lane.endpoint),
        Some(snapshot_path),
        dry_run,
    )
    .await
    .map(Some)
}

fn load_lane_tokenizer(lane: &NormalizedLaneConfig) -> anyhow::Result<HuggingFaceTokenizer> {
    let snapshot_path = lane.snapshot_path.as_deref().ok_or_else(|| {
        anyhow!(
            "profile {} requires snapshot-backed lanes to build exact-token recall prompts",
            QWEN_MLX_PREFILL_135K_PROFILE
        )
    })?;
    load_qwen_tokenizer(snapshot_path)
}

fn raw_snapshot_identity(lane: &NormalizedLaneConfig, snapshot_path: &Path) -> ModelIdentityReport {
    let mut report = ModelIdentityReport {
        id: lane.identity_model_id(),
        endpoint: Some(lane.endpoint.clone()),
        snapshot_path: Some(snapshot_path.display().to_string()),
        repo_id: None,
        requested_revision: None,
        resolved_commit: None,
        profile: None,
        family: Some("qwen".to_owned()),
        loader: matches!(
            lane.kind,
            NormalizedLaneKind::DirectMlx | NormalizedLaneKind::KirAiProxy
        )
        .then(|| "mlx".to_owned()),
        quantization: None,
        manifest_digest: None,
    };
    let inferred = infer_huggingface_snapshot_identity(snapshot_path);
    report.repo_id = inferred.repo_id;
    report.resolved_commit = inferred.resolved_commit;
    report
}

#[derive(Debug, Default, PartialEq, Eq)]
struct InferredSnapshotIdentity {
    repo_id: Option<String>,
    resolved_commit: Option<String>,
}

fn infer_huggingface_snapshot_identity(snapshot_path: &Path) -> InferredSnapshotIdentity {
    let Some(snapshot_dir_name) = snapshot_path.file_name().and_then(|name| name.to_str()) else {
        return InferredSnapshotIdentity::default();
    };
    let Some(snapshots_dir_name) = snapshot_path
        .parent()
        .and_then(|path| path.file_name())
        .and_then(|name| name.to_str())
    else {
        return InferredSnapshotIdentity::default();
    };
    if snapshots_dir_name != "snapshots" {
        return InferredSnapshotIdentity::default();
    }
    let repo_id = snapshot_path
        .parent()
        .and_then(Path::parent)
        .and_then(|path| path.file_name())
        .and_then(|name| name.to_str())
        .and_then(huggingface_cache_repo_id);
    InferredSnapshotIdentity {
        repo_id,
        resolved_commit: Some(snapshot_dir_name.to_owned()),
    }
}

fn huggingface_cache_repo_id(directory_name: &str) -> Option<String> {
    let encoded = directory_name.strip_prefix("models--")?;
    let (owner, repo) = encoded.split_once("--")?;
    if owner.is_empty() || repo.is_empty() {
        return None;
    }
    Some(format!("{owner}/{}", repo.replace("--", "/")))
}

async fn capture_normalized_admin_metrics(
    client: &reqwest::Client,
    lane: &NormalizedLaneConfig,
    admin_token: Option<&str>,
) -> Result<Value, String> {
    let mut request = client
        .get(admin_metrics_url(&lane.endpoint))
        .timeout(admin_metrics_timeout());
    if let Some(token) = admin_token {
        request = request.bearer_auth(token);
    }
    match request.send().await {
        Ok(response) if response.status().is_success() => response
            .json::<Value>()
            .await
            .map_err(|err| format!("parse admin metrics: {err}")),
        Ok(response) => Err(format!("admin metrics HTTP {}", response.status())),
        Err(err) => Err(format!("admin metrics request failed: {err}")),
    }
}

fn should_capture_admin_metrics(lane: &NormalizedLaneConfig) -> bool {
    matches!(lane.kind, NormalizedLaneKind::KirAiProxy)
}

fn should_capture_tool_stream_attribution_metrics(
    lane: &NormalizedLaneConfig,
    probe: NormalizedProbePlan,
    planned: PlannedRun,
) -> bool {
    should_capture_admin_metrics(lane)
        && probe.case == NormalizedCaseKind::ToolRequiredStream
        && planned.kind == PlannedRunKind::Measured
        && planned.run_mode == RunMode::Sequential
}

fn admin_metrics_timeout() -> Duration {
    Duration::from_millis(ADMIN_METRICS_TIMEOUT_MS)
}

fn admin_metrics_url(endpoint: &str) -> String {
    let root = endpoint
        .trim_end_matches('/')
        .strip_suffix("/v1")
        .unwrap_or_else(|| endpoint.trim_end_matches('/'));
    format!("{root}/admin/metrics")
}

struct LaneRunContext<'a> {
    client: &'a reqwest::Client,
    run_config: &'a NormalizedRunConfig,
    probes: &'a [NormalizedProbePlan],
    admin_token: Option<&'a str>,
    prompt_tokenizer: Option<&'a HuggingFaceTokenizer>,
    progress: &'a NormalizedProgress,
}

async fn run_lane(
    lane: &NormalizedLaneConfig,
    lane_report: &mut NormalizedLaneReport,
    context: LaneRunContext<'_>,
) {
    if should_capture_admin_metrics(lane) {
        lane_report.admin_metrics.record_before(
            capture_normalized_admin_metrics(context.client, lane, context.admin_token).await,
        );
    }
    for &probe in context.probes {
        for planned in phase_plan(
            &context.run_config.cache_phases,
            context.run_config.warmups,
            context.run_config.samples,
        ) {
            match planned.kind {
                PlannedRunKind::Warmup => {
                    let result = execute_probe(lane, probe, planned, &context).await;
                    if result.status != "passed" {
                        lane_report.warmup_failures.push(NormalizedWarmupFailure {
                            case: probe.case.name(),
                            schema_variant: probe.schema_variant.name(),
                            tool_choice_variant: probe.tool_choice_variant.name(),
                            max_tokens: probe.max_tokens,
                            cache_phase: planned.phase.name(),
                            warmup_index: planned.warmup_index.unwrap_or_default(),
                            classification: result.classification,
                            http_status: result.http_status,
                            error: result.error,
                        });
                    }
                }
                PlannedRunKind::Measured => {
                    let sample = execute_probe(lane, probe, planned, &context).await;
                    lane_report.samples.push(sample);
                }
            }
        }
    }
    for &probe in context.probes {
        for &phase in &context.run_config.cache_phases {
            for sample_index in 0..context.run_config.effective_concurrent_samples {
                let requests = (0..context.run_config.concurrent_requests).map(|request_index| {
                    let planned = PlannedRun {
                        phase,
                        kind: PlannedRunKind::Measured,
                        run_mode: RunMode::Concurrent,
                        sample_index: Some(sample_index),
                        request_index: Some(request_index),
                        warmup_index: None,
                    };
                    execute_probe(lane, probe, planned, &context)
                });
                lane_report
                    .concurrent_samples
                    .extend(join_all(requests).await);
            }
        }
    }
    lane_report.status = if lane_report
        .samples
        .iter()
        .chain(&lane_report.concurrent_samples)
        .all(|sample| sample.status == "passed")
    {
        "passed"
    } else {
        "failed"
    }
    .to_owned();
    if should_capture_admin_metrics(lane) {
        lane_report.admin_metrics.record_after(
            capture_normalized_admin_metrics(context.client, lane, context.admin_token).await,
        );
    }
}

async fn execute_probe(
    lane: &NormalizedLaneConfig,
    probe: NormalizedProbePlan,
    planned: PlannedRun,
    context: &LaneRunContext<'_>,
) -> NormalizedSampleReport {
    let prompt = match planned.prompt(
        context.run_config.context_tokens,
        probe.case,
        context.prompt_tokenizer,
    ) {
        Ok(prompt) => prompt,
        Err(err) => {
            let sample_context = SampleContext {
                probe,
                phase: planned.phase,
                run_mode: planned.run_mode,
                sample_index: planned.sample_index.unwrap_or_default(),
                request_index: planned.request_index,
                planned_prompt_tokens: 0,
                prewarmed: planned.phase.warms_before_samples() && context.run_config.warmups > 0,
                expected_probe_id: probe.case.probe_id().to_owned(),
                expected_marker: None,
            };
            return failed_sample(
                sample_context,
                "prompt_build_failed",
                Duration::from_millis(0),
                None,
                None,
                err.to_string(),
                StreamTimingReport::default(),
            );
        }
    };
    let expected_probe_id = prompt.probe_id(probe.case);
    let expected_marker = prompt.expected_marker(probe.case);
    let sample_context = SampleContext {
        probe,
        phase: planned.phase,
        run_mode: planned.run_mode,
        sample_index: planned.sample_index.unwrap_or_default(),
        request_index: planned.request_index,
        planned_prompt_tokens: prompt.planned_prompt_tokens(),
        prewarmed: planned.phase.warms_before_samples() && context.run_config.warmups > 0,
        expected_probe_id,
        expected_marker,
    };
    let body = probe_request_body(lane, probe, prompt);
    let mut attribution_admin_metrics =
        should_capture_tool_stream_attribution_metrics(lane, probe, planned)
            .then(NormalizedAdminMetricsCapture::default);
    if let Some(capture) = &mut attribution_admin_metrics {
        capture.record_before(
            capture_normalized_admin_metrics(context.client, lane, context.admin_token).await,
        );
    }
    context.progress.record_request_start(lane, probe, planned);
    let mut sample = if probe.case.streams() {
        run_streaming_probe(context.client, lane, sample_context, body).await
    } else {
        run_buffered_probe(context.client, lane, sample_context, body).await
    };
    if let Some(mut capture) = attribution_admin_metrics {
        capture.record_after(
            capture_normalized_admin_metrics(context.client, lane, context.admin_token).await,
        );
        sample.tool_required_stream_admin_metrics = Some(capture);
    }
    sample
}

async fn run_buffered_probe(
    client: &reqwest::Client,
    lane: &NormalizedLaneConfig,
    context: SampleContext,
    body: Value,
) -> NormalizedSampleReport {
    let url = chat_completions_url(&lane.endpoint);
    let started = Instant::now();
    let response = match client.post(&url).json(&body).send().await {
        Ok(response) => response,
        Err(err) => {
            return failed_sample(
                context,
                "http_request_failed",
                started.elapsed(),
                None,
                None,
                err.to_string(),
                StreamTimingReport::default(),
            );
        }
    };
    let status = response.status();
    let http_status = Some(status.as_u16());
    let response_headers = response_headers_map(response.headers());
    let text = match response.text().await {
        Ok(text) => text,
        Err(err) => {
            return failed_sample(
                context,
                "http_body_failed",
                started.elapsed(),
                http_status,
                Some(response_headers),
                err.to_string(),
                StreamTimingReport::default(),
            );
        }
    };
    let latency = started.elapsed();
    if !status.is_success() {
        return failed_sample(
            context,
            "http_status_failed",
            latency,
            http_status,
            Some(response_headers),
            text,
            StreamTimingReport::default(),
        );
    }
    let value = match serde_json::from_str::<Value>(&text) {
        Ok(value) => value,
        Err(err) => {
            return failed_sample(
                context,
                "response_json_failed",
                latency,
                http_status,
                Some(response_headers),
                err.to_string(),
                StreamTimingReport::default(),
            );
        }
    };
    let usage = usage_from_value(value.get("usage"));
    let finish_reason = value
        .pointer("/choices/0/finish_reason")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let validation =
        validate_buffered_probe(context.probe.case, &value, &context.expected_probe_id);
    sample_from_validation(
        context,
        validation,
        ProbeResponseMetadata {
            latency,
            stream_timing: StreamTimingReport::default(),
            http_status,
            response_headers: Some(response_headers),
            finish_reason,
            usage,
        },
    )
}

async fn run_streaming_probe(
    client: &reqwest::Client,
    lane: &NormalizedLaneConfig,
    context: SampleContext,
    body: Value,
) -> NormalizedSampleReport {
    let url = chat_completions_url(&lane.endpoint);
    let started = Instant::now();
    let response = match client.post(&url).json(&body).send().await {
        Ok(response) => response,
        Err(err) => {
            return failed_sample(
                context,
                "stream_http_request_failed",
                started.elapsed(),
                None,
                None,
                err.to_string(),
                StreamTimingReport::default(),
            );
        }
    };
    let status = response.status();
    let http_status = Some(status.as_u16());
    let response_headers = response_headers_map(response.headers());
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        return failed_sample(
            context,
            "stream_http_status_failed",
            started.elapsed(),
            http_status,
            Some(response_headers),
            text,
            StreamTimingReport::default(),
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
                return failed_sample(
                    context,
                    "stream_body_failed",
                    started.elapsed(),
                    http_status,
                    Some(response_headers),
                    err.to_string(),
                    timings.to_report(),
                );
            }
        };
        if chunk.is_empty() {
            continue;
        }
        let elapsed = started.elapsed();
        timings.record_first_byte(elapsed);
        buffer.push_str(&String::from_utf8_lossy(&chunk));
        consume_sse_buffer(&mut buffer, &mut assembly, &mut timings, elapsed);
    }
    if !buffer.is_empty() {
        buffer.push('\n');
        consume_sse_buffer(&mut buffer, &mut assembly, &mut timings, started.elapsed());
    }

    let validation = validate_streaming_probe(
        context.probe.case,
        &assembly,
        &context.expected_probe_id,
        context.expected_marker.as_deref(),
    );
    sample_from_validation(
        context,
        validation,
        ProbeResponseMetadata {
            latency: started.elapsed(),
            stream_timing: timings.to_report(),
            http_status,
            response_headers: Some(response_headers),
            finish_reason: assembly.finish_reason,
            usage: assembly.usage,
        },
    )
}

struct ProbeResponseMetadata {
    latency: Duration,
    stream_timing: StreamTimingReport,
    http_status: Option<u16>,
    response_headers: Option<BTreeMap<String, String>>,
    finish_reason: Option<String>,
    usage: super::UsageMetrics,
}

fn sample_from_validation(
    context: SampleContext,
    validation: Result<(), String>,
    response: ProbeResponseMetadata,
) -> NormalizedSampleReport {
    let tokens_per_second = response.usage.completion_tokens.and_then(|tokens| {
        (response.latency.as_secs_f64() > 0.0)
            .then_some(tokens as f64 / response.latency.as_secs_f64())
    });
    let mut sample = NormalizedSampleReport::base(
        context.probe,
        context.phase,
        context.run_mode,
        context.sample_index,
        context.request_index,
        context.prewarmed,
        context.planned_prompt_tokens,
    );
    sample.latency_ms = Some(response.latency.as_millis());
    sample.stream_timing = response.stream_timing;
    sample.tokens_per_second = tokens_per_second;
    sample.prompt_tokens = response.usage.prompt_tokens;
    sample.completion_tokens = response.usage.completion_tokens;
    sample.total_tokens = response.usage.total_tokens;
    sample.cached_tokens_status = response.usage.cached_tokens_status.unwrap_or("missing");
    sample.cached_tokens = response.usage.cached_tokens;
    sample.request_id = request_id_from_response_headers(&response.response_headers);
    sample.http_status = response.http_status;
    sample.response_headers = response.response_headers;
    sample.finish_reason = response.finish_reason;
    match validation {
        Ok(()) => {
            sample.status = "passed".to_owned();
            sample.classification = "passed".to_owned();
        }
        Err(err) => {
            sample.status = "failed".to_owned();
            sample.classification = "response_validation_failed".to_owned();
            sample.failure_classification =
                classify_sample_failure("response_validation_failed", response.http_status, &err)
                    .map(str::to_owned);
            sample.error = Some(err);
        }
    }
    sample
}

fn failed_sample(
    context: SampleContext,
    classification: impl Into<String>,
    latency: Duration,
    http_status: Option<u16>,
    response_headers: Option<BTreeMap<String, String>>,
    error: String,
    stream_timing: StreamTimingReport,
) -> NormalizedSampleReport {
    let mut sample = NormalizedSampleReport::base(
        context.probe,
        context.phase,
        context.run_mode,
        context.sample_index,
        context.request_index,
        context.prewarmed,
        context.planned_prompt_tokens,
    );
    sample.status = "failed".to_owned();
    let classification = classification.into();
    sample.failure_classification =
        classify_sample_failure(&classification, http_status, &error).map(str::to_owned);
    sample.classification = classification;
    sample.latency_ms = Some(latency.as_millis());
    sample.stream_timing = stream_timing;
    sample.http_status = http_status;
    sample.request_id = request_id_from_response_headers(&response_headers);
    sample.response_headers = response_headers;
    sample.error = Some(error);
    sample
}

fn classify_sample_failure(
    classification: &str,
    http_status: Option<u16>,
    error: &str,
) -> Option<&'static str> {
    let error = error.to_ascii_lowercase();
    if error.contains("out of memory") || error.contains("out-of-memory") || error.contains("oom") {
        return Some("oom");
    }
    if error.contains("metal")
        || error.contains("mtlcommandbuffer")
        || error.contains("command buffer")
    {
        return Some("metal_failure");
    }
    if classification == "response_validation_failed" {
        return Some("progress_validation_failed");
    }
    if matches!(http_status, Some(408 | 413 | 429 | 503 | 507))
        || error.contains("timed out")
        || error.contains("timeout")
        || error.contains("deadline")
        || error.contains("resource exhausted")
        || error.contains("memory pressure")
    {
        return Some("resource_limit_exceeded");
    }
    None
}

fn chat_completions_url(endpoint: &str) -> String {
    if endpoint.ends_with("/v1") {
        format!("{endpoint}/chat/completions")
    } else {
        format!("{endpoint}/v1/chat/completions")
    }
}

fn response_headers_map(headers: &reqwest::header::HeaderMap) -> BTreeMap<String, String> {
    headers
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.as_str().to_owned(), value.to_owned()))
        })
        .collect()
}

fn request_id_from_response_headers(headers: &Option<BTreeMap<String, String>>) -> Option<String> {
    let headers = headers.as_ref()?;
    headers
        .iter()
        .find(|(name, _)| {
            name.eq_ignore_ascii_case("x-request-id")
                || name.eq_ignore_ascii_case("x-llm-request-id")
        })
        .map(|(_, value)| value.clone())
}

fn probe_request_body(
    lane: &NormalizedLaneConfig,
    probe: NormalizedProbePlan,
    prompt: ProbePrompt,
) -> Value {
    let mut body = json!({
        "max_tokens": probe.max_tokens,
        "temperature": 0,
        "top_p": 1,
        "messages": probe_messages(probe.case, prompt)
    });
    if let Some(model_id) = lane.request_model_id() {
        body["model"] = json!(model_id);
    }
    match probe.case {
        NormalizedCaseKind::ToolRequired
        | NormalizedCaseKind::ToolRequiredStream
        | NormalizedCaseKind::ContextRecallStream135k
        | NormalizedCaseKind::OmpRepeatedPrefix
        | NormalizedCaseKind::WarmPrefixRepeatedTurnStream => {
            body["tools"] = json!([probe_tool_schema(probe)]);
            body["tool_choice"] = probe.tool_choice_variant.request_value(probe.case);
            if probe.case.streams() {
                body["stream"] = json!(true);
                body["stream_options"] = json!({"include_usage": true});
            }
        }
        NormalizedCaseKind::ChatStream => {
            body["stream"] = json!(true);
            body["stream_options"] = json!({"include_usage": true});
        }
        NormalizedCaseKind::JsonObject => {
            body["response_format"] = json!({"type": "json_object"});
        }
    }
    lane.template.apply_request_kwargs(&mut body);
    body
}

fn probe_messages(case: NormalizedCaseKind, prompt: ProbePrompt) -> Value {
    if matches!(
        case,
        NormalizedCaseKind::OmpRepeatedPrefix | NormalizedCaseKind::WarmPrefixRepeatedTurnStream
    ) {
        let history_probe_id = format!("{}_HISTORY", case.probe_id());
        let history_arguments =
            json!({"probe_id": history_probe_id.clone(), "case": case.name()}).to_string();
        return json!([
            {"role": "system", "content": case.system_prompt()},
            {"role": "user", "content": stable_context_prefix(prompt.context_tokens, case)},
            {
                "role": "assistant",
                "content": Value::Null,
                "tool_calls": [{
                    "id": "call_qwen_tool_probe_history",
                    "type": "function",
                    "function": {
                        "name": "record_qwen_tool_probe",
                        "arguments": history_arguments
                    }
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_qwen_tool_probe_history",
                "content": json!({"status": "recorded", "probe_id": history_probe_id}).to_string()
            },
            {"role": "user", "content": prompt.user_prompt(case)}
        ]);
    }
    json!([
        {"role": "system", "content": case.system_prompt()},
        {"role": "user", "content": prompt.user_prompt(case)}
    ])
}

fn probe_tool_schema(probe: NormalizedProbePlan) -> Value {
    match probe.case {
        NormalizedCaseKind::ContextRecallStream135k => {
            recall_probe_tool_schema(probe.schema_variant)
        }
        _ => qwen_probe_tool_schema(probe.schema_variant),
    }
}

fn qwen_probe_tool_schema(variant: SchemaVariant) -> Value {
    let minimal = json!({
        "type": "function",
        "function": {
            "name": "record_qwen_tool_probe",
            "parameters": {
                "type": "object",
                "properties": {
                    "probe_id": {"type": "string"},
                    "case": {"type": "string"}
                },
                "required": ["probe_id", "case"]
            }
        }
    });
    let current = json!({
        "type": "function",
        "function": {
            "name": "record_qwen_tool_probe",
            "description": "Record the normalized Qwen tool benchmark probe.",
            "parameters": {
                "type": "object",
                "properties": {
                    "probe_id": {"type": "string"},
                    "case": {"type": "string"}
                },
                "required": ["probe_id", "case"],
                "additionalProperties": false
            }
        }
    });
    let permuted = json!({
        "function": {
            "parameters": {
                "additionalProperties": false,
                "required": ["case", "probe_id"],
                "properties": {
                    "case": {"type": "string"},
                    "probe_id": {"type": "string"}
                },
                "type": "object"
            },
            "description": "Record the normalized Qwen tool benchmark probe.",
            "name": "record_qwen_tool_probe"
        },
        "type": "function"
    });
    let omp_style_i = json!({
        "type": "function",
        "function": {
            "name": "record_qwen_tool_probe",
            "description": "Record an OpenManus-style Qwen tool probe with a call index field.",
            "parameters": {
                "type": "object",
                "properties": {
                    "_i": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Optional tool-call index emitted by OMP-style agents."
                    },
                    "probe_id": {"type": "string"},
                    "case": {"type": "string"}
                },
                "required": ["probe_id", "case"],
                "additionalProperties": false
            }
        }
    });
    match variant {
        SchemaVariant::MinimalShallow => minimal,
        SchemaVariant::BaselineCurrent => current,
        SchemaVariant::CanonicalCurrent => canonicalize_json_value(&current),
        SchemaVariant::BaselinePermutedEquivalent => permuted,
        SchemaVariant::CanonicalPermutedEquivalent => canonicalize_json_value(&permuted),
        SchemaVariant::OmpStyleI => omp_style_i,
        SchemaVariant::LargeStress => large_stress_qwen_probe_tool_schema(),
        SchemaVariant::None => Value::Null,
    }
}

fn large_stress_qwen_probe_tool_schema() -> Value {
    let mut properties = Map::new();
    properties.insert("probe_id".to_owned(), json!({"type": "string"}));
    properties.insert("case".to_owned(), json!({"type": "string"}));
    properties.insert(
        "agent_context".to_owned(),
        json!({
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "step": {"type": "integer"},
                "source": {"type": "string"}
            },
            "additionalProperties": false
        }),
    );
    properties.insert(
        "evidence".to_owned(),
        json!({
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                    "score": {"type": "number"}
                },
                "required": ["key", "value"],
                "additionalProperties": false
            },
            "maxItems": 8
        }),
    );
    for index in 0..32 {
        properties.insert(
            format!("stress_field_{index:02}"),
            json!({
                "type": ["string", "null"],
                "description": format!("Optional distractor schema field {index:02} for required-tool TTFT stress.")
            }),
        );
    }

    let mut parameters = Map::new();
    parameters.insert("type".to_owned(), json!("object"));
    parameters.insert("properties".to_owned(), Value::Object(properties));
    parameters.insert("required".to_owned(), json!(["probe_id", "case"]));
    parameters.insert("additionalProperties".to_owned(), Value::Bool(false));

    json!({
        "type": "function",
        "function": {
            "name": "record_qwen_tool_probe",
            "description": "Record the normalized Qwen tool benchmark probe with a deliberately large schema.",
            "parameters": Value::Object(parameters)
        }
    })
}

fn recall_probe_tool_schema(variant: SchemaVariant) -> Value {
    let current = json!({
        "type": "function",
        "function": {
            "name": "report_long_context_recall",
            "description": "Report a recalled long-context benchmark marker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case": {"type": "string"},
                    "marker": {"type": "string"},
                    "profile": {"type": "string"}
                },
                "required": ["case", "marker", "profile"],
                "additionalProperties": false
            }
        }
    });
    let permuted = json!({
        "function": {
            "parameters": {
                "additionalProperties": false,
                "required": ["profile", "marker", "case"],
                "properties": {
                    "profile": {"type": "string"},
                    "marker": {"type": "string"},
                    "case": {"type": "string"}
                },
                "type": "object"
            },
            "description": "Report a recalled long-context benchmark marker.",
            "name": "report_long_context_recall"
        },
        "type": "function"
    });
    match variant {
        SchemaVariant::MinimalShallow | SchemaVariant::OmpStyleI | SchemaVariant::LargeStress => {
            current
        }
        SchemaVariant::BaselineCurrent => current,
        SchemaVariant::CanonicalCurrent => canonicalize_json_value(&current),
        SchemaVariant::BaselinePermutedEquivalent => permuted,
        SchemaVariant::CanonicalPermutedEquivalent => canonicalize_json_value(&permuted),
        SchemaVariant::None => Value::Null,
    }
}

fn tool_schema_metadata(probe: NormalizedProbePlan) -> ToolSchemaMetadata {
    if probe.schema_variant == SchemaVariant::None {
        return ToolSchemaMetadata {
            sha256: None,
            bytes: None,
        };
    }
    let schema_json = serde_json::to_string(&json!([probe_tool_schema(probe)]))
        .expect("benchmark tool schema serializes");
    let digest = Sha256::digest(schema_json.as_bytes());
    ToolSchemaMetadata {
        sha256: Some(format!("{digest:x}")),
        bytes: Some(schema_json.len()),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ToolSchemaMetadata {
    sha256: Option<String>,
    bytes: Option<usize>,
}

fn validate_buffered_probe(
    case: NormalizedCaseKind,
    value: &Value,
    expected_probe_id: &str,
) -> Result<(), String> {
    match case {
        NormalizedCaseKind::ToolRequired | NormalizedCaseKind::OmpRepeatedPrefix => {
            let finish_reason = value
                .pointer("/choices/0/finish_reason")
                .and_then(Value::as_str);
            validate_tool_finish_reason(finish_reason, "tool call")?;
            let tool_call = value
                .pointer("/choices/0/message/tool_calls/0")
                .ok_or_else(|| "missing required tool call".to_owned())?;
            validate_probe_tool_call(tool_call, case, expected_probe_id)
        }
        NormalizedCaseKind::JsonObject => {
            let content = value
                .pointer("/choices/0/message/content")
                .and_then(Value::as_str)
                .ok_or_else(|| "missing assistant JSON content".to_owned())?;
            let parsed = serde_json::from_str::<Value>(content)
                .map_err(|err| format!("assistant content was not valid JSON: {err}"))?;
            validate_probe_arguments(&parsed, case, expected_probe_id, "JSON")
        }
        NormalizedCaseKind::ToolRequiredStream
        | NormalizedCaseKind::ChatStream
        | NormalizedCaseKind::ContextRecallStream135k
        | NormalizedCaseKind::WarmPrefixRepeatedTurnStream => {
            Err("streamed tool case was routed through buffered validator".to_owned())
        }
    }
}

fn validate_streaming_probe(
    case: NormalizedCaseKind,
    assembly: &StreamAssembly,
    expected_probe_id: &str,
    expected_marker: Option<&str>,
) -> Result<(), String> {
    if !case.streams() {
        return Err("non-streaming case was routed through streaming validator".to_owned());
    }
    if case == NormalizedCaseKind::ChatStream {
        let marker = expected_marker
            .ok_or_else(|| "chat stream validation was missing expected marker".to_owned())?;
        return if assembly.content.contains(marker) {
            Ok(())
        } else {
            Err(format!(
                "streamed assistant content did not contain marker `{marker}`"
            ))
        };
    }
    if case == NormalizedCaseKind::ContextRecallStream135k {
        let marker = expected_marker
            .ok_or_else(|| "recall stream validation was missing expected marker".to_owned())?;
        return validate_streaming_recall_probe(case, assembly, marker);
    }
    let name = assembly
        .tool_name
        .as_deref()
        .ok_or_else(|| "missing streamed tool name".to_owned())?;
    if name != "record_qwen_tool_probe" {
        return Err(format!(
            "streamed tool name `{name}` did not match expected"
        ));
    }
    validate_tool_finish_reason(assembly.finish_reason.as_deref(), "streamed tool call")?;
    let args = serde_json::from_str::<Value>(&assembly.tool_arguments)
        .map_err(|err| format!("streamed tool arguments were not JSON: {err}"))?;
    validate_probe_arguments(&args, case, expected_probe_id, "streamed tool")
}

fn validate_streaming_recall_probe(
    case: NormalizedCaseKind,
    assembly: &StreamAssembly,
    expected_marker: &str,
) -> Result<(), String> {
    let name = assembly
        .tool_name
        .as_deref()
        .ok_or_else(|| "missing streamed recall tool name".to_owned())?;
    if name != "report_long_context_recall" {
        return Err(format!(
            "streamed recall tool name `{name}` did not match expected"
        ));
    }
    validate_tool_finish_reason(
        assembly.finish_reason.as_deref(),
        "streamed recall tool call",
    )?;
    let args = serde_json::from_str::<Value>(&assembly.tool_arguments)
        .map_err(|err| format!("streamed recall tool arguments were not JSON: {err}"))?;
    validate_recall_arguments(&args, case, expected_marker, "streamed recall tool")
}

fn validate_probe_tool_call(
    tool_call: &Value,
    case: NormalizedCaseKind,
    expected_probe_id: &str,
) -> Result<(), String> {
    let name = tool_call
        .pointer("/function/name")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing tool function name".to_owned())?;
    if name != "record_qwen_tool_probe" {
        return Err(format!("tool function `{name}` did not match expected"));
    }
    let args_text = tool_call
        .pointer("/function/arguments")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing tool function arguments".to_owned())?;
    let args = serde_json::from_str::<Value>(args_text)
        .map_err(|err| format!("tool arguments were not JSON: {err}"))?;
    validate_probe_arguments(&args, case, expected_probe_id, "tool")
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

fn validate_probe_arguments(
    args: &Value,
    case: NormalizedCaseKind,
    expected_probe_id: &str,
    label: &str,
) -> Result<(), String> {
    let object = args
        .as_object()
        .ok_or_else(|| format!("{label} arguments were not a JSON object"))?;
    let probe_id = object
        .get("probe_id")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label} arguments missing string `probe_id`"))?;
    if probe_id != expected_probe_id {
        return Err(format!(
            "{label} probe_id `{probe_id}` did not equal `{expected_probe_id}`"
        ));
    }
    let actual_case = object
        .get("case")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label} arguments missing string `case`"))?;
    if actual_case != case.name() {
        return Err(format!(
            "{label} case `{actual_case}` did not equal `{}`",
            case.name()
        ));
    }
    Ok(())
}

fn validate_recall_arguments(
    args: &Value,
    case: NormalizedCaseKind,
    expected_marker: &str,
    label: &str,
) -> Result<(), String> {
    let object = args
        .as_object()
        .ok_or_else(|| format!("{label} arguments were not a JSON object"))?;
    let marker = object
        .get("marker")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label} arguments missing string `marker`"))?;
    if marker != expected_marker {
        return Err(format!(
            "{label} marker `{marker}` did not equal `{expected_marker}`"
        ));
    }
    let profile = object
        .get("profile")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label} arguments missing string `profile`"))?;
    if profile != PREFILL_SWEEP_135K_PROFILE_NAME {
        return Err(format!(
            "{label} profile `{profile}` did not equal `{PREFILL_SWEEP_135K_PROFILE_NAME}`"
        ));
    }
    let actual_case = object
        .get("case")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label} arguments missing string `case`"))?;
    if actual_case != case.name() {
        return Err(format!(
            "{label} case `{actual_case}` did not equal `{}`",
            case.name()
        ));
    }
    if object.len() != 3 {
        return Err(format!(
            "{label} arguments must contain exactly marker, profile, and case"
        ));
    }
    Ok(())
}

fn phase_plan(phases: &[CachePhase], warmups: usize, samples: usize) -> Vec<PlannedRun> {
    let mut runs = Vec::new();
    for &phase in phases {
        if phase.warms_before_samples() {
            for warmup_index in 0..warmups {
                runs.push(PlannedRun {
                    phase,
                    kind: PlannedRunKind::Warmup,
                    run_mode: RunMode::Sequential,
                    sample_index: None,
                    request_index: None,
                    warmup_index: Some(warmup_index),
                });
            }
        }
        for sample_index in 0..samples {
            runs.push(PlannedRun {
                phase,
                kind: PlannedRunKind::Measured,
                run_mode: RunMode::Sequential,
                sample_index: Some(sample_index),
                request_index: None,
                warmup_index: None,
            });
        }
    }
    runs
}

fn concurrent_phase_plan(
    phases: &[CachePhase],
    concurrent_requests: usize,
    concurrent_samples: usize,
) -> Vec<PlannedRun> {
    let mut runs = Vec::new();
    for &phase in phases {
        for sample_index in 0..concurrent_samples {
            for request_index in 0..concurrent_requests {
                runs.push(PlannedRun {
                    phase,
                    kind: PlannedRunKind::Measured,
                    run_mode: RunMode::Concurrent,
                    sample_index: Some(sample_index),
                    request_index: Some(request_index),
                    warmup_index: None,
                });
            }
        }
    }
    runs
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NormalizedCaseKind {
    ToolRequired,
    ToolRequiredStream,
    JsonObject,
    OmpRepeatedPrefix,
    ChatStream,
    ContextRecallStream135k,
    WarmPrefixRepeatedTurnStream,
}

impl NormalizedCaseKind {
    fn all() -> [Self; 4] {
        [
            Self::ToolRequired,
            Self::ToolRequiredStream,
            Self::JsonObject,
            Self::OmpRepeatedPrefix,
        ]
    }

    fn name(self) -> &'static str {
        match self {
            Self::ToolRequired => "tool_required",
            Self::ToolRequiredStream => "tool_required_stream",
            Self::JsonObject => "json_object",
            Self::OmpRepeatedPrefix => "omp_repeated_prefix",
            Self::ChatStream => "chat_stream",
            Self::ContextRecallStream135k => "context_recall_stream_135k",
            Self::WarmPrefixRepeatedTurnStream => "warm_prefix_repeated_turn_stream",
        }
    }

    fn probe_id(self) -> &'static str {
        match self {
            Self::ToolRequired => "KIR_QWEN_MLX_TOOL_NORMALIZED_TOOL_REQUIRED",
            Self::ToolRequiredStream => "KIR_QWEN_MLX_TOOL_NORMALIZED_TOOL_REQUIRED_STREAM",
            Self::JsonObject => "KIR_QWEN_MLX_TOOL_NORMALIZED_JSON_OBJECT",
            Self::OmpRepeatedPrefix => "KIR_QWEN_MLX_TOOL_NORMALIZED_OMP_REPEATED_PREFIX",
            Self::ChatStream => "KIR_QWEN_MLX_TOOL_NORMALIZED_CHAT_STREAM",
            Self::ContextRecallStream135k => {
                "KIR_QWEN_MLX_TOOL_NORMALIZED_CONTEXT_RECALL_STREAM_135K"
            }
            Self::WarmPrefixRepeatedTurnStream => {
                "KIR_QWEN_MLX_TOOL_NORMALIZED_WARM_PREFIX_REPEATED_TURN_STREAM"
            }
        }
    }

    fn system_prompt(self) -> &'static str {
        match self {
            Self::ToolRequired | Self::ToolRequiredStream => {
                "You are a tool-call conformance probe. Use the provided function exactly once."
            }
            Self::ChatStream => {
                "You are a streaming chat latency probe. Return the requested marker in assistant content."
            }
            Self::ContextRecallStream135k => {
                "You are a long-context streaming tool-call evaluator. Use the provided function to report the recalled marker."
            }
            Self::JsonObject => {
                "You are a JSON conformance probe. Return one JSON object and no prose."
            }
            Self::OmpRepeatedPrefix => {
                "You are an OMP-style repeated-prefix workflow probe. Continue the tool workflow and use the provided function exactly once."
            }
            Self::WarmPrefixRepeatedTurnStream => {
                "You are a warm-prefix repeated-turn streaming workflow probe. Continue the tool workflow and use the provided function exactly once."
            }
        }
    }

    fn streams(self) -> bool {
        matches!(
            self,
            Self::ToolRequiredStream
                | Self::ChatStream
                | Self::ContextRecallStream135k
                | Self::WarmPrefixRepeatedTurnStream
        )
    }

    fn tool_function_name(self) -> &'static str {
        match self {
            Self::ContextRecallStream135k => "report_long_context_recall",
            _ => "record_qwen_tool_probe",
        }
    }

    fn requires_tool_delta(self) -> bool {
        matches!(
            self,
            Self::ToolRequiredStream
                | Self::ContextRecallStream135k
                | Self::WarmPrefixRepeatedTurnStream
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SchemaVariant {
    None,
    MinimalShallow,
    BaselineCurrent,
    CanonicalCurrent,
    BaselinePermutedEquivalent,
    CanonicalPermutedEquivalent,
    OmpStyleI,
    LargeStress,
}

impl SchemaVariant {
    fn all() -> [Self; 4] {
        [
            Self::BaselineCurrent,
            Self::CanonicalCurrent,
            Self::BaselinePermutedEquivalent,
            Self::CanonicalPermutedEquivalent,
        ]
    }

    fn required_tool_ttft_matrix() -> [Self; 4] {
        [
            Self::MinimalShallow,
            Self::CanonicalCurrent,
            Self::OmpStyleI,
            Self::LargeStress,
        ]
    }

    fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::MinimalShallow => "minimal_shallow",
            Self::BaselineCurrent => "baseline_current",
            Self::CanonicalCurrent => "canonical_current",
            Self::BaselinePermutedEquivalent => "baseline_permuted_equivalent",
            Self::CanonicalPermutedEquivalent => "canonical_permuted_equivalent",
            Self::OmpStyleI => "omp_style_i",
            Self::LargeStress => "large_stress",
        }
    }

    fn canonicalized(self) -> bool {
        matches!(
            self,
            Self::CanonicalCurrent | Self::CanonicalPermutedEquivalent
        )
    }

    fn permuted(self) -> bool {
        matches!(
            self,
            Self::BaselinePermutedEquivalent | Self::CanonicalPermutedEquivalent
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolChoiceVariant {
    None,
    Required,
    Function,
}

impl ToolChoiceVariant {
    fn all() -> [Self; 2] {
        [Self::Required, Self::Function]
    }

    fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Required => "required",
            Self::Function => "function",
        }
    }

    fn request_value(self, case: NormalizedCaseKind) -> Value {
        match self {
            Self::Required => json!("required"),
            Self::Function => {
                json!({"type": "function", "function": {"name": case.tool_function_name()}})
            }
            Self::None => Value::Null,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NormalizedProbeSuite {
    FullMatrix,
    FocusedAgenticGate,
    RequiredToolTtftMatrix,
    PrefillSweep135k,
    PrefillSweep135kContextRecall,
    StableAgentPrefix,
    StablePrefixSmoke,
}

impl NormalizedProbeSuite {
    fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "full-matrix" | "full_matrix" => Ok(Self::FullMatrix),
            "focused-agentic-gate" | "focused_agentic_gate" => Ok(Self::FocusedAgenticGate),
            "required-tool-ttft-matrix" | "required_tool_ttft_matrix" => {
                Ok(Self::RequiredToolTtftMatrix)
            }
            "prefill-sweep-135k" | "prefill_sweep_135k" => Ok(Self::PrefillSweep135k),
            "prefill-sweep-135k-context-recall" | "prefill_sweep_135k_context_recall" => {
                Ok(Self::PrefillSweep135kContextRecall)
            }
            "stable-agent-prefix" | "stable_agent_prefix" => Ok(Self::StableAgentPrefix),
            "stable-prefix-smoke" | "stable_prefix_smoke" => Ok(Self::StablePrefixSmoke),
            other => anyhow::bail!(
                "unknown --probe-suite `{other}`; expected full-matrix, focused-agentic-gate, required-tool-ttft-matrix, prefill-sweep-135k, prefill-sweep-135k-context-recall, stable-agent-prefix, or stable-prefix-smoke"
            ),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::FullMatrix => "full_matrix",
            Self::FocusedAgenticGate => "focused_agentic_gate",
            Self::RequiredToolTtftMatrix => "required_tool_ttft_matrix",
            Self::PrefillSweep135k => "prefill_sweep_135k",
            Self::PrefillSweep135kContextRecall => "prefill_sweep_135k_context_recall",
            Self::StableAgentPrefix => "stable_agent_prefix",
            Self::StablePrefixSmoke => "stable_prefix_smoke",
        }
    }

    fn probes(self) -> Vec<NormalizedProbePlan> {
        match self {
            Self::FullMatrix => NormalizedProbePlan::all(),
            Self::FocusedAgenticGate => NormalizedProbePlan::focused_agentic_gate(),
            Self::RequiredToolTtftMatrix => NormalizedProbePlan::required_tool_ttft_matrix(),
            Self::PrefillSweep135k => NormalizedProbePlan::prefill_sweep_135k(),
            Self::PrefillSweep135kContextRecall => {
                NormalizedProbePlan::prefill_sweep_135k_context_recall()
            }
            Self::StableAgentPrefix => NormalizedProbePlan::stable_agent_prefix(),
            Self::StablePrefixSmoke => NormalizedProbePlan::stable_prefix_smoke(),
        }
    }

    fn case_names(self, probes: &[NormalizedProbePlan]) -> Vec<&'static str> {
        match self {
            Self::FullMatrix => NormalizedCaseKind::all()
                .iter()
                .map(|case| case.name())
                .collect(),
            Self::FocusedAgenticGate
            | Self::RequiredToolTtftMatrix
            | Self::PrefillSweep135k
            | Self::PrefillSweep135kContextRecall
            | Self::StableAgentPrefix
            | Self::StablePrefixSmoke => probe_case_names(probes),
        }
    }

    fn schema_variant_names(self, probes: &[NormalizedProbePlan]) -> Vec<&'static str> {
        match self {
            Self::FullMatrix => SchemaVariant::all()
                .iter()
                .map(|variant| variant.name())
                .collect(),
            Self::FocusedAgenticGate
            | Self::RequiredToolTtftMatrix
            | Self::PrefillSweep135k
            | Self::PrefillSweep135kContextRecall
            | Self::StableAgentPrefix
            | Self::StablePrefixSmoke => probe_schema_variant_names(probes),
        }
    }

    fn tool_choice_variant_names(self, probes: &[NormalizedProbePlan]) -> Vec<&'static str> {
        match self {
            Self::FullMatrix => ToolChoiceVariant::all()
                .iter()
                .map(|variant| variant.name())
                .collect(),
            Self::FocusedAgenticGate
            | Self::RequiredToolTtftMatrix
            | Self::PrefillSweep135k
            | Self::PrefillSweep135kContextRecall
            | Self::StableAgentPrefix
            | Self::StablePrefixSmoke => probe_tool_choice_variant_names(probes),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NormalizedProbePlan {
    case: NormalizedCaseKind,
    schema_variant: SchemaVariant,
    tool_choice_variant: ToolChoiceVariant,
    max_tokens: u32,
}

impl NormalizedProbePlan {
    fn new(
        case: NormalizedCaseKind,
        schema_variant: SchemaVariant,
        tool_choice_variant: ToolChoiceVariant,
    ) -> Self {
        Self {
            case,
            schema_variant,
            tool_choice_variant,
            max_tokens: DEFAULT_MAX_TOKENS,
        }
    }

    fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    fn all() -> Vec<Self> {
        let mut probes = Vec::new();
        for case in NormalizedCaseKind::all() {
            if case == NormalizedCaseKind::JsonObject {
                probes.push(Self::new(
                    case,
                    SchemaVariant::None,
                    ToolChoiceVariant::None,
                ));
                continue;
            }
            for schema_variant in SchemaVariant::all() {
                for tool_choice_variant in ToolChoiceVariant::all() {
                    probes.push(Self::new(case, schema_variant, tool_choice_variant));
                }
            }
        }
        probes
    }

    fn focused_agentic_gate() -> Vec<Self> {
        vec![
            Self::new(
                NormalizedCaseKind::ToolRequired,
                SchemaVariant::CanonicalCurrent,
                ToolChoiceVariant::Required,
            ),
            Self::new(
                NormalizedCaseKind::ToolRequiredStream,
                SchemaVariant::CanonicalCurrent,
                ToolChoiceVariant::Required,
            ),
            Self::new(
                NormalizedCaseKind::OmpRepeatedPrefix,
                SchemaVariant::CanonicalCurrent,
                ToolChoiceVariant::Required,
            ),
        ]
    }

    fn required_tool_ttft_matrix() -> Vec<Self> {
        let mut probes = Vec::new();
        for schema_variant in SchemaVariant::required_tool_ttft_matrix() {
            for tool_choice_variant in ToolChoiceVariant::all() {
                for max_tokens in REQUIRED_TOOL_TTFT_MAX_TOKENS {
                    probes.push(
                        Self::new(
                            NormalizedCaseKind::ToolRequiredStream,
                            schema_variant,
                            tool_choice_variant,
                        )
                        .with_max_tokens(max_tokens),
                    );
                }
            }
        }
        probes
    }

    fn prefill_sweep_135k() -> Vec<Self> {
        vec![
            Self::new(
                NormalizedCaseKind::ChatStream,
                SchemaVariant::None,
                ToolChoiceVariant::None,
            ),
            Self::new(
                NormalizedCaseKind::ToolRequiredStream,
                SchemaVariant::CanonicalCurrent,
                ToolChoiceVariant::Required,
            ),
            Self::new(
                NormalizedCaseKind::ContextRecallStream135k,
                SchemaVariant::CanonicalCurrent,
                ToolChoiceVariant::Required,
            ),
            Self::new(
                NormalizedCaseKind::WarmPrefixRepeatedTurnStream,
                SchemaVariant::CanonicalCurrent,
                ToolChoiceVariant::Required,
            ),
        ]
    }

    fn prefill_sweep_135k_context_recall() -> Vec<Self> {
        vec![Self::new(
            NormalizedCaseKind::ContextRecallStream135k,
            SchemaVariant::CanonicalCurrent,
            ToolChoiceVariant::Required,
        )]
    }

    fn stable_agent_prefix() -> Vec<Self> {
        vec![
            Self::new(
                NormalizedCaseKind::ChatStream,
                SchemaVariant::None,
                ToolChoiceVariant::None,
            ),
            Self::new(
                NormalizedCaseKind::ToolRequiredStream,
                SchemaVariant::CanonicalCurrent,
                ToolChoiceVariant::Required,
            ),
            Self::new(
                NormalizedCaseKind::WarmPrefixRepeatedTurnStream,
                SchemaVariant::CanonicalCurrent,
                ToolChoiceVariant::Required,
            ),
        ]
    }

    fn stable_prefix_smoke() -> Vec<Self> {
        vec![Self::new(
            NormalizedCaseKind::WarmPrefixRepeatedTurnStream,
            SchemaVariant::CanonicalCurrent,
            ToolChoiceVariant::Required,
        )]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CachePhase {
    Cold,
    WarmSamePrompt,
    WarmSameToolSchema,
}

impl CachePhase {
    fn all() -> [Self; 3] {
        [Self::Cold, Self::WarmSamePrompt, Self::WarmSameToolSchema]
    }

    fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "cold" => Ok(Self::Cold),
            "warm_same_prompt" => Ok(Self::WarmSamePrompt),
            "warm_same_tool_schema" => Ok(Self::WarmSameToolSchema),
            other => anyhow::bail!(
                "unknown cache phase `{other}`; expected cold, warm_same_prompt, or warm_same_tool_schema"
            ),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Cold => "cold",
            Self::WarmSamePrompt => "warm_same_prompt",
            Self::WarmSameToolSchema => "warm_same_tool_schema",
        }
    }

    fn warms_before_samples(self) -> bool {
        !matches!(self, Self::Cold)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlannedRunKind {
    Warmup,
    Measured,
}

impl PlannedRunKind {
    fn name(self) -> &'static str {
        match self {
            Self::Warmup => "warmup",
            Self::Measured => "measured",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RunMode {
    Sequential,
    Concurrent,
}

impl RunMode {
    fn name(self) -> &'static str {
        match self {
            Self::Sequential => "sequential",
            Self::Concurrent => "concurrent",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct PlannedRun {
    phase: CachePhase,
    kind: PlannedRunKind,
    run_mode: RunMode,
    sample_index: Option<usize>,
    request_index: Option<usize>,
    warmup_index: Option<usize>,
}

impl PlannedRun {
    fn prompt(
        self,
        context_tokens: usize,
        case: NormalizedCaseKind,
        tokenizer: Option<&HuggingFaceTokenizer>,
    ) -> anyhow::Result<ProbePrompt> {
        let long_context = if case == NormalizedCaseKind::ContextRecallStream135k {
            tokenizer
                .map(|tokenizer| build_context_recall_prompt(tokenizer, context_tokens))
                .transpose()?
        } else {
            None
        };
        match (self.kind, self.phase) {
            (PlannedRunKind::Warmup, CachePhase::WarmSameToolSchema) => {
                Ok(ProbePrompt::schema_warmup(
                    context_tokens,
                    self.warmup_index.unwrap_or_default(),
                    long_context,
                ))
            }
            _ => Ok(ProbePrompt::measured_with_long_context(
                context_tokens,
                self.sample_index.unwrap_or_default(),
                self.request_index,
                long_context,
            )),
        }
    }
}

#[derive(Debug, Clone)]
struct SampleContext {
    probe: NormalizedProbePlan,
    phase: CachePhase,
    run_mode: RunMode,
    sample_index: usize,
    request_index: Option<usize>,
    planned_prompt_tokens: usize,
    prewarmed: bool,
    expected_probe_id: String,
    expected_marker: Option<String>,
}

#[derive(Debug, Clone)]
struct ProbePrompt {
    variant: ProbePromptVariant,
    context_tokens: usize,
    sample_index: usize,
    request_index: Option<usize>,
    long_context: Option<LongContextPrompt>,
}

impl ProbePrompt {
    #[cfg(test)]
    fn measured(context_tokens: usize, sample_index: usize, request_index: Option<usize>) -> Self {
        Self::measured_with_long_context(context_tokens, sample_index, request_index, None)
    }

    fn measured_with_long_context(
        context_tokens: usize,
        sample_index: usize,
        request_index: Option<usize>,
        long_context: Option<LongContextPrompt>,
    ) -> Self {
        Self {
            variant: ProbePromptVariant::Measured,
            context_tokens,
            sample_index,
            request_index,
            long_context,
        }
    }

    fn schema_warmup(
        context_tokens: usize,
        index: usize,
        long_context: Option<LongContextPrompt>,
    ) -> Self {
        Self {
            variant: ProbePromptVariant::SchemaWarmup(index),
            context_tokens,
            sample_index: 0,
            request_index: None,
            long_context,
        }
    }

    fn planned_prompt_tokens(&self) -> usize {
        self.long_context
            .as_ref()
            .map(|prompt| prompt.token_count)
            .unwrap_or(self.context_tokens)
    }

    fn probe_id(&self, case: NormalizedCaseKind) -> String {
        match self.variant {
            ProbePromptVariant::Measured => case.probe_id().to_owned(),
            ProbePromptVariant::SchemaWarmup(index) => {
                format!("{}_SCHEMA_WARMUP_{index}", case.probe_id())
            }
        }
    }

    fn expected_marker(&self, case: NormalizedCaseKind) -> Option<String> {
        match case {
            NormalizedCaseKind::ChatStream => Some(CHAT_STREAM_MARKER.to_owned()),
            NormalizedCaseKind::ContextRecallStream135k => Some(
                self.long_context
                    .as_ref()
                    .map(|prompt| prompt.marker.as_str())
                    .unwrap_or(CONTEXT_RECALL_STREAM_135K_MARKER)
                    .to_owned(),
            ),
            _ => None,
        }
    }

    fn user_prompt(&self, case: NormalizedCaseKind) -> String {
        let probe_id = self.probe_id(case);
        let prefix = stable_context_prefix(self.context_tokens, case);
        match case {
            NormalizedCaseKind::ToolRequired | NormalizedCaseKind::ToolRequiredStream => {
                format!(
                    "{prefix}\nCall record_qwen_tool_probe with probe_id `{probe_id}` and case `{}`.",
                    case.name()
                )
            }
            NormalizedCaseKind::JsonObject => {
                format!(
                    "{prefix}\nReturn exactly this JSON shape with probe_id `{probe_id}` and case `{}`: {{\"probe_id\":\"...\",\"case\":\"...\"}}",
                    case.name()
                )
            }
            NormalizedCaseKind::ChatStream => {
                format!(
                    "{prefix}\nReturn exactly this marker in assistant content and no tool call: {CHAT_STREAM_MARKER}"
                )
            }
            NormalizedCaseKind::ContextRecallStream135k => {
                let body = self
                    .long_context
                    .as_ref()
                    .map(|prompt| prompt.body.as_str())
                    .map(str::to_owned)
                    .unwrap_or_else(|| approximate_context_recall_prompt(self.context_tokens));
                format!("{body}\nCall report_long_context_recall with marker, profile, and case.")
            }
            NormalizedCaseKind::OmpRepeatedPrefix => {
                let request = self
                    .request_index
                    .map(|index| index.to_string())
                    .unwrap_or_else(|| "sequential".to_owned());
                format!(
                    "OMP final delta: sample={} request={request}. Call record_qwen_tool_probe with probe_id `{probe_id}` and case `{}`.",
                    self.sample_index,
                    case.name()
                )
            }
            NormalizedCaseKind::WarmPrefixRepeatedTurnStream => {
                let request = self
                    .request_index
                    .map(|index| index.to_string())
                    .unwrap_or_else(|| "sequential".to_owned());
                format!(
                    "Warm-prefix final delta: sample={} request={request}. Call record_qwen_tool_probe with probe_id `{probe_id}` and case `{}`.",
                    self.sample_index,
                    case.name()
                )
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum ProbePromptVariant {
    Measured,
    SchemaWarmup(usize),
}

#[derive(Debug, Clone)]
struct LongContextPrompt {
    marker: String,
    body: String,
    token_count: usize,
}

fn build_context_recall_prompt(
    tokenizer: &HuggingFaceTokenizer,
    target_tokens: usize,
) -> anyhow::Result<LongContextPrompt> {
    let marker = CONTEXT_RECALL_STREAM_135K_MARKER.to_owned();
    let mut body = context_recall_prompt_header(&marker);
    let footer = "\nEnd of benchmark context. Use the target_marker value from the first section when calling the tool.\n";
    let row_template = "Context row 000000: MLX scheduler counters, prefill chunk sizes, cache namespace fields, parser states, and trace identifiers. This row is distractor material only.\n";
    let row_tokens = tokenizer.encode(row_template, false)?.len().max(1);
    let base_tokens = tokenizer.encode(&(body.clone() + footer), false)?.len();
    let estimated_rows = target_tokens
        .saturating_sub(base_tokens)
        .div_ceil(row_tokens)
        .saturating_add(8);
    for row in 0..estimated_rows {
        body.push_str(&format!(
            "Context row {row:06}: MLX scheduler counters, prefill chunk sizes, cache namespace fields, parser states, and trace identifiers. This row is distractor material only.\n"
        ));
    }
    body.push_str(footer);
    let mut token_count = tokenizer.encode(&body, false)?.len();
    while token_count < target_tokens {
        let row = token_count;
        body.push_str(&format!(
            "Context extension {row:06}: additional non-target diagnostics for Qwen MLX prefill pressure.\n"
        ));
        token_count = tokenizer.encode(&body, false)?.len();
    }
    Ok(LongContextPrompt {
        marker,
        body,
        token_count,
    })
}

fn approximate_context_recall_prompt(context_tokens: usize) -> String {
    let marker = CONTEXT_RECALL_STREAM_135K_MARKER;
    let mut body = context_recall_prompt_header(marker);
    body.push_str(&stable_context_prefix(
        context_tokens,
        NormalizedCaseKind::ContextRecallStream135k,
    ));
    body.push_str(
        "\nEnd of benchmark context. Use the target_marker value from the first section when calling the tool.",
    );
    body
}

fn context_recall_prompt_header(marker: &str) -> String {
    format!(
        "\
Long-context benchmark profile: {PREFILL_SWEEP_135K_PROFILE_NAME}
Scenario: context_recall_stream_135k
Target marker name: target_marker
Target marker value: {marker}

Only the marker value above is correct. Later context rows are distractors and must not replace it.

"
    )
}

fn stable_context_prefix(context_tokens: usize, case: NormalizedCaseKind) -> String {
    let mut body = format!(
        "\
Qwen MLX-LM tool sweep long-context payload.
Declared context token target: {context_tokens}
Case: {case_name}
This shared prefix is stable across measured requests for cache and prefill pressure.
For OMP repeated-prefix probes, only the final user delta changes after the shared history.

",
        case_name = case.name()
    );
    let estimated_tokens_per_row = 32usize;
    let fixed_token_estimate = 80usize;
    let rows = context_tokens
        .saturating_sub(fixed_token_estimate)
        .div_ceil(estimated_tokens_per_row);
    for row in 0..rows {
        body.push_str(&format!(
            "Stable context row {row:06}: scheduler trace fields, tool schemas, repository paths, prompt-cache keys, decode counters, and parser states are distractor material.\n"
        ));
    }
    body
}

#[cfg(test)]
mod tests;
