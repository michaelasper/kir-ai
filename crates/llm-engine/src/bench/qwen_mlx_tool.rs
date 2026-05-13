use super::{
    DEFAULT_CONNECT_TIMEOUT_MS, DEFAULT_TIMEOUT_MS, HardwareReport, ModelIdentityReport,
    StreamAssembly, StreamTimingReport, StreamTimingTracker, cli::flag_values, consume_sse_buffer,
    load_model_identity, normalize_endpoint, unix_now_ms, usage_from_value,
};
use crate::{DEFAULT_MODEL_ID, flag_value, has_flag};
use anyhow::{Context, anyhow};
use futures::StreamExt;
use futures::future::join_all;
use llm_api::canonicalize_json_value;
use serde::{Serialize, Serializer};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

const BENCHMARK_NAME: &str = "qwen-mlx-tool-normalized";
const DEFAULT_WARMUPS: usize = 1;
const DEFAULT_SAMPLES: usize = 1;
const DEFAULT_CONTEXT_TOKENS: usize = 135_000;
const DEFAULT_CONCURRENT_REQUESTS: usize = 1;
const DEFAULT_CONCURRENT_SAMPLES: usize = 0;
const DEFAULT_MAX_TOKENS: u32 = 96;
const QWEN_MLX_CACHE_PREFILL_PROFILE: &str = "qwen-mlx-cache-prefill";
const PROFILE_PROXY_MODEL_ID: &str = "local-qwen36-mlx";
const PROFILE_CACHE_BYTES_1G: u64 = 1_073_741_824;
const BENCH_REPO_DIR_ENV: &str = "LLM_ENGINE_BENCH_REPO_DIR";
const BENCH_REPO_COMMIT_ENV: &str = "LLM_ENGINE_BENCH_REPO_COMMIT";
const BENCH_REPO_BRANCH_ENV: &str = "LLM_ENGINE_BENCH_REPO_BRANCH";
const BENCH_REPO_DIRTY_ENV: &str = "LLM_ENGINE_BENCH_REPO_DIRTY";
const BENCH_REPO_ORIGIN_FILE: &str = ".kir-ai-origin.json";

pub(super) async fn run_qwen_mlx_tool_normalized_bench(args: &[String]) -> anyhow::Result<()> {
    if has_flag(args, "--help") || has_flag(args, "-h") {
        print_help();
        return Ok(());
    }

    let dry_run = has_flag(args, "--dry-run");
    let warmups = parse_count_flag(args, "--warmups", DEFAULT_WARMUPS, true)?;
    let samples = parse_count_flag(args, "--samples", DEFAULT_SAMPLES, false)?;
    let context_tokens = parse_count_flag(args, "--context-tokens", DEFAULT_CONTEXT_TOKENS, false)?;
    let concurrent_requests = parse_count_flag(
        args,
        "--concurrent-requests",
        DEFAULT_CONCURRENT_REQUESTS,
        false,
    )?;
    let concurrent_samples = parse_count_flag(
        args,
        "--concurrent-samples",
        DEFAULT_CONCURRENT_SAMPLES,
        true,
    )?;
    let run_config = NormalizedRunConfig::new(
        warmups,
        samples,
        context_tokens,
        concurrent_requests,
        concurrent_samples,
    );
    let timeout_ms = parse_millis_flag(args, "--timeout-ms", DEFAULT_TIMEOUT_MS)?;
    let connect_timeout_ms =
        parse_millis_flag(args, "--connect-timeout-ms", DEFAULT_CONNECT_TIMEOUT_MS)?;
    let output_path = flag_value(args, "--output").map(PathBuf::from);
    let sweep_profile = parse_sweep_profile_flag(args)?;
    let probe_suite = parse_probe_suite_flag(args);
    let probes = probe_suite.probes();
    let lanes = parse_lane_specs(args)?;

    let mut lane_reports = Vec::with_capacity(lanes.len());
    for lane in &lanes {
        let snapshot_identity = load_lane_snapshot_identity(lane, dry_run).await?;
        lane_reports.push(if dry_run {
            NormalizedLaneReport::dry_run(lane, run_config, snapshot_identity, &probes)
        } else {
            NormalizedLaneReport::planned(
                lane,
                run_config.warmups,
                run_config.samples,
                snapshot_identity,
            )
        });
    }

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
        cache_phases: CachePhase::all().iter().map(|phase| phase.name()).collect(),
        summary: aggregate_normalized_summary(&lane_reports, &probes),
        lanes: lane_reports,
        hardware: HardwareReport::detect(),
        comparison: NormalizedComparisonReport::dry_run(),
        agentic_gate: NormalizedAgenticGateReport::dry_run(),
    };

    if dry_run {
        write_and_print_normalized_report(&report, output_path.as_deref()).await?;
        return Ok(());
    }

    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_millis(connect_timeout_ms))
        .timeout(Duration::from_millis(timeout_ms))
        .build()
        .context("build qwen mlx tool normalized benchmark HTTP client")?;

    for (lane, lane_report) in lanes.iter().zip(&mut report.lanes) {
        run_lane(&client, lane, lane_report, run_config, &probes).await;
    }
    report.summary = aggregate_normalized_summary(&report.lanes, &probes);
    report.comparison = compare_normalized_lanes(&report.lanes, &probes);
    report.agentic_gate = agentic_gate_report(&report.lanes);
    report.status = if report.lanes.iter().all(|lane| lane.status == "passed") {
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
Usage: llm-engine bench qwen-mlx-tool-normalized [OPTIONS]

Options:
  --sweep-profile <name>        Built-in lane matrix: qwen-mlx-cache-prefill (requires --snapshot)
  --snapshot <path>             Raw Hugging Face snapshot path for built-in sweep profiles
  --lane <spec>                 Lane: name=<id>,endpoint=<url>,model=<id>[,launched_model_id=<id-or-path>][,snapshot=<path>][,kind=direct_mlx|kir_ai_proxy|other][,model_addressing=loaded_model_id|default_model|server_default|custom][,template=qwen-no-thinking|sidecar-chat-template-args|none][,mlx_prompt_cache_size=default|<n>][,mlx_prompt_cache_bytes=unset|<n>][,mlx_prefill_step_size=default|<n>][,mlx_prompt_concurrency=default|<n>][,mlx_decode_concurrency=default|<n>]
  --warmups <n>                 Warmup requests for warm phases [default: 1]
  --samples <n>                 Measured samples per case and phase [default: 1]
  --context-tokens <n>          Stable long-context prompt target [default: 135000]
  --concurrent-requests <n>     Requests to issue together during the concurrent pass [default: 1]
  --concurrent-samples <n>      Concurrent sample batches per case and phase; 0 disables unless concurrent requests > 1 [default: 0]
  --focused-agentic-gate        Run the small agentic gate instead of the full schema/tool matrix
  --output <path>               Write the trace JSON to a file as well as stdout
  --timeout-ms <n>              Whole request timeout [default: 1800000]
  --connect-timeout-ms <n>      HTTP connect timeout [default: 10000]
  --dry-run                     Print the exact probe plan without HTTP requests
  -h, --help                    Print help"
    );
}

fn parse_lane_specs(args: &[String]) -> anyhow::Result<Vec<NormalizedLaneConfig>> {
    let lane_specs = flag_values(args, "--lane");
    if let Some(profile) = parse_sweep_profile_flag(args)? {
        if !lane_specs.is_empty() {
            anyhow::bail!("--sweep-profile cannot be combined with explicit --lane specs");
        }
        return expand_sweep_profile(profile, args);
    }
    if lane_specs.is_empty() {
        anyhow::bail!("qwen mlx tool normalized benchmark requires at least one --lane <spec>");
    }
    lane_specs.into_iter().map(parse_lane_spec).collect()
}

fn parse_sweep_profile_flag(args: &[String]) -> anyhow::Result<Option<NormalizedSweepProfile>> {
    let profiles = flag_values(args, "--sweep-profile");
    match profiles.as_slice() {
        [] => Ok(None),
        [profile] => NormalizedSweepProfile::parse(profile).map(Some),
        _ => anyhow::bail!("--sweep-profile may only be provided once"),
    }
}

fn parse_probe_suite_flag(args: &[String]) -> NormalizedProbeSuite {
    let focused_agentic_gate = has_flag(args, "--focused-agentic-gate");
    if focused_agentic_gate {
        NormalizedProbeSuite::FocusedAgenticGate
    } else {
        NormalizedProbeSuite::FullMatrix
    }
}

fn expand_sweep_profile(
    profile: NormalizedSweepProfile,
    args: &[String],
) -> anyhow::Result<Vec<NormalizedLaneConfig>> {
    let snapshot = required_profile_snapshot(args)?;
    Ok(match profile {
        NormalizedSweepProfile::QwenMlxCachePrefill => qwen_mlx_cache_prefill_lanes(snapshot),
    })
}

fn required_profile_snapshot(args: &[String]) -> anyhow::Result<&str> {
    let snapshots = flag_values(args, "--snapshot");
    match snapshots.as_slice() {
        [snapshot] => Ok(snapshot),
        [] => anyhow::bail!(
            "--sweep-profile {QWEN_MLX_CACHE_PREFILL_PROFILE} requires --snapshot <path>"
        ),
        _ => anyhow::bail!("--snapshot may only be provided once for --sweep-profile"),
    }
}

fn qwen_mlx_cache_prefill_lanes(snapshot: &str) -> Vec<NormalizedLaneConfig> {
    vec![
        profile_direct_lane("mlx-default", 8080, MlxLmSettings::default(), snapshot),
        profile_direct_lane(
            "mlx-cache-size-4096",
            8081,
            MlxLmSettings {
                prompt_cache_size: DefaultOrU64::Value(4096),
                ..MlxLmSettings::default()
            },
            snapshot,
        ),
        profile_direct_lane(
            "mlx-cache-bytes-1g",
            8082,
            MlxLmSettings {
                prompt_cache_bytes: UnsetOrU64::Value(PROFILE_CACHE_BYTES_1G),
                ..MlxLmSettings::default()
            },
            snapshot,
        ),
        profile_direct_lane(
            "mlx-prefill-2048",
            8083,
            MlxLmSettings {
                prefill_step_size: DefaultOrU64::Value(2048),
                ..MlxLmSettings::default()
            },
            snapshot,
        ),
        profile_direct_lane(
            "mlx-prefill-4096",
            8084,
            MlxLmSettings {
                prefill_step_size: DefaultOrU64::Value(4096),
                ..MlxLmSettings::default()
            },
            snapshot,
        ),
        profile_direct_lane(
            "mlx-prefill-8192",
            8085,
            MlxLmSettings {
                prefill_step_size: DefaultOrU64::Value(8192),
                ..MlxLmSettings::default()
            },
            snapshot,
        ),
        profile_direct_lane(
            "mlx-concurrent-4x2",
            8086,
            MlxLmSettings {
                prompt_concurrency: DefaultOrU32::Value(4),
                decode_concurrency: DefaultOrU32::Value(2),
                ..MlxLmSettings::default()
            },
            snapshot,
        ),
        NormalizedLaneConfig {
            name: "kir-proxy".to_owned(),
            endpoint: "http://127.0.0.1:3000".to_owned(),
            declared_model_id: PROFILE_PROXY_MODEL_ID.to_owned(),
            launched_model_id: Some(snapshot.to_owned()),
            snapshot_path: Some(PathBuf::from(snapshot)),
            kind: NormalizedLaneKind::KirAiProxy,
            model_addressing: NormalizedModelAddressing::DefaultModel,
            template: NormalizedTemplatePolicy::SidecarChatTemplateArgs,
            mlx_lm_settings: MlxLmSettings::default(),
        },
    ]
}

fn profile_direct_lane(
    name: &str,
    port: u16,
    mlx_lm_settings: MlxLmSettings,
    snapshot: &str,
) -> NormalizedLaneConfig {
    NormalizedLaneConfig {
        name: name.to_owned(),
        endpoint: format!("http://127.0.0.1:{port}/v1"),
        declared_model_id: snapshot.to_owned(),
        launched_model_id: Some(snapshot.to_owned()),
        snapshot_path: Some(PathBuf::from(snapshot)),
        kind: NormalizedLaneKind::DirectMlx,
        model_addressing: NormalizedModelAddressing::ServerDefault,
        template: NormalizedTemplatePolicy::SidecarChatTemplateArgs,
        mlx_lm_settings,
    }
}

fn parse_lane_spec(spec: &str) -> anyhow::Result<NormalizedLaneConfig> {
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
        .map(|value| normalize_endpoint(&value))
        .ok_or_else(|| anyhow!("--lane spec `{spec}` is missing endpoint=<url>"))?;
    let declared_model_id = values
        .remove("model")
        .or_else(|| values.remove("model_id"))
        .ok_or_else(|| anyhow!("--lane spec `{spec}` is missing model=<id>"))?;
    let launched_model_id = values
        .remove("launched_model_id")
        .or_else(|| values.remove("launch_model_id"));
    let snapshot_path = values.remove("snapshot").map(PathBuf::from);
    let kind = values
        .remove("kind")
        .map(|value| NormalizedLaneKind::parse(&value))
        .transpose()?
        .unwrap_or(NormalizedLaneKind::Other);
    let model_addressing = values
        .remove("model_addressing")
        .map(|value| NormalizedModelAddressing::parse(&value))
        .transpose()?
        .unwrap_or(NormalizedModelAddressing::LoadedModelId);
    let template = values
        .remove("template")
        .map(|value| NormalizedTemplatePolicy::parse(&value))
        .transpose()?
        .unwrap_or(NormalizedTemplatePolicy::QwenNoThinking);
    let mlx_lm_settings = MlxLmSettings::parse(&mut values)?;

    if !values.is_empty() {
        let unknown = values.keys().cloned().collect::<Vec<_>>().join(", ");
        anyhow::bail!("--lane spec `{spec}` contains unknown keys: {unknown}");
    }

    Ok(NormalizedLaneConfig {
        name,
        endpoint,
        declared_model_id,
        launched_model_id,
        snapshot_path,
        kind,
        model_addressing,
        template,
        mlx_lm_settings,
    })
}

fn parse_count_flag(
    args: &[String],
    flag: &str,
    default: usize,
    allow_zero: bool,
) -> anyhow::Result<usize> {
    let value = flag_value(args, flag)
        .map(str::parse::<usize>)
        .transpose()
        .with_context(|| format!("parse {flag}"))?
        .unwrap_or(default);
    if !allow_zero && value == 0 {
        anyhow::bail!("{flag} must be greater than zero");
    }
    Ok(value)
}

fn parse_millis_flag(args: &[String], flag: &str, default: u64) -> anyhow::Result<u64> {
    let value = flag_value(args, flag)
        .map(str::parse::<u64>)
        .transpose()
        .with_context(|| format!("parse {flag}"))?
        .unwrap_or(default);
    if value == 0 {
        anyhow::bail!("{flag} must be greater than zero");
    }
    Ok(value)
}

#[derive(Debug, Clone, Copy)]
struct NormalizedRunConfig {
    warmups: usize,
    samples: usize,
    context_tokens: usize,
    concurrent_requests: usize,
    concurrent_samples: usize,
    effective_concurrent_samples: usize,
}

impl NormalizedRunConfig {
    fn new(
        warmups: usize,
        samples: usize,
        context_tokens: usize,
        concurrent_requests: usize,
        concurrent_samples: usize,
    ) -> Self {
        Self {
            warmups,
            samples,
            context_tokens,
            concurrent_requests,
            concurrent_samples,
            effective_concurrent_samples: effective_concurrent_samples(
                concurrent_requests,
                samples,
                concurrent_samples,
            ),
        }
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

async fn run_lane(
    client: &reqwest::Client,
    lane: &NormalizedLaneConfig,
    lane_report: &mut NormalizedLaneReport,
    run_config: NormalizedRunConfig,
    probes: &[NormalizedProbePlan],
) {
    for &probe in probes {
        for planned in phase_plan(run_config.warmups, run_config.samples) {
            let prompt = planned.prompt(run_config.context_tokens);
            match planned.kind {
                PlannedRunKind::Warmup => {
                    let result =
                        execute_probe(client, lane, probe, planned, prompt, run_config).await;
                    if result.status != "passed" {
                        lane_report.warmup_failures.push(NormalizedWarmupFailure {
                            case: probe.case.name(),
                            schema_variant: probe.schema_variant.name(),
                            tool_choice_variant: probe.tool_choice_variant.name(),
                            cache_phase: planned.phase.name(),
                            warmup_index: planned.warmup_index.unwrap_or_default(),
                            classification: result.classification,
                            http_status: result.http_status,
                            error: result.error,
                        });
                    }
                }
                PlannedRunKind::Measured => {
                    let sample =
                        execute_probe(client, lane, probe, planned, prompt, run_config).await;
                    lane_report.samples.push(sample);
                }
            }
        }
    }
    for &probe in probes {
        for phase in CachePhase::all() {
            for sample_index in 0..run_config.effective_concurrent_samples {
                let requests = (0..run_config.concurrent_requests).map(|request_index| {
                    let planned = PlannedRun {
                        phase,
                        kind: PlannedRunKind::Measured,
                        run_mode: RunMode::Concurrent,
                        sample_index: Some(sample_index),
                        request_index: Some(request_index),
                        warmup_index: None,
                    };
                    let prompt = planned.prompt(run_config.context_tokens);
                    execute_probe(client, lane, probe, planned, prompt, run_config)
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
}

async fn execute_probe(
    client: &reqwest::Client,
    lane: &NormalizedLaneConfig,
    probe: NormalizedProbePlan,
    planned: PlannedRun,
    prompt: ProbePrompt,
    run_config: NormalizedRunConfig,
) -> NormalizedSampleReport {
    let expected_probe_id = prompt.probe_id(probe.case);
    let context = SampleContext {
        probe,
        phase: planned.phase,
        run_mode: planned.run_mode,
        sample_index: planned.sample_index.unwrap_or_default(),
        request_index: planned.request_index,
        planned_prompt_tokens: prompt.planned_prompt_tokens(),
        prewarmed: planned.phase.warms_before_samples() && run_config.warmups > 0,
        expected_probe_id,
    };
    let body = probe_request_body(lane, probe, prompt);
    if probe.case.streams() {
        run_streaming_probe(client, lane, context, body).await
    } else {
        run_buffered_probe(client, lane, context, body).await
    }
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
                err.to_string(),
                StreamTimingReport::default(),
            );
        }
    };
    let status = response.status();
    let http_status = Some(status.as_u16());
    let text = match response.text().await {
        Ok(text) => text,
        Err(err) => {
            return failed_sample(
                context,
                "http_body_failed",
                started.elapsed(),
                http_status,
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
        latency,
        StreamTimingReport::default(),
        http_status,
        finish_reason,
        usage,
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
                err.to_string(),
                StreamTimingReport::default(),
            );
        }
    };
    let status = response.status();
    let http_status = Some(status.as_u16());
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        return failed_sample(
            context,
            "stream_http_status_failed",
            started.elapsed(),
            http_status,
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

    let validation =
        validate_streaming_probe(context.probe.case, &assembly, &context.expected_probe_id);
    sample_from_validation(
        context,
        validation,
        started.elapsed(),
        timings.to_report(),
        http_status,
        assembly.finish_reason,
        assembly.usage,
    )
}

fn sample_from_validation(
    context: SampleContext,
    validation: Result<(), String>,
    latency: Duration,
    stream_timing: StreamTimingReport,
    http_status: Option<u16>,
    finish_reason: Option<String>,
    usage: super::UsageMetrics,
) -> NormalizedSampleReport {
    let tokens_per_second = usage.completion_tokens.and_then(|tokens| {
        (latency.as_secs_f64() > 0.0).then_some(tokens as f64 / latency.as_secs_f64())
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
    sample.latency_ms = Some(latency.as_millis());
    sample.stream_timing = stream_timing;
    sample.tokens_per_second = tokens_per_second;
    sample.prompt_tokens = usage.prompt_tokens;
    sample.completion_tokens = usage.completion_tokens;
    sample.total_tokens = usage.total_tokens;
    sample.cached_tokens_status = usage.cached_tokens_status.unwrap_or("missing");
    sample.cached_tokens = usage.cached_tokens;
    sample.http_status = http_status;
    sample.finish_reason = finish_reason;
    match validation {
        Ok(()) => {
            sample.status = "passed".to_owned();
            sample.classification = "passed".to_owned();
        }
        Err(err) => {
            sample.status = "failed".to_owned();
            sample.classification = "response_validation_failed".to_owned();
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
    sample.classification = classification.into();
    sample.latency_ms = Some(latency.as_millis());
    sample.stream_timing = stream_timing;
    sample.http_status = http_status;
    sample.error = Some(error);
    sample
}

fn chat_completions_url(endpoint: &str) -> String {
    if endpoint.ends_with("/v1") {
        format!("{endpoint}/chat/completions")
    } else {
        format!("{endpoint}/v1/chat/completions")
    }
}

fn probe_request_body(
    lane: &NormalizedLaneConfig,
    probe: NormalizedProbePlan,
    prompt: ProbePrompt,
) -> Value {
    let mut body = json!({
        "max_tokens": DEFAULT_MAX_TOKENS,
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
        | NormalizedCaseKind::OmpRepeatedPrefix => {
            body["tools"] = json!([probe_tool_schema(probe.schema_variant)]);
            body["tool_choice"] = probe.tool_choice_variant.request_value();
            if probe.case.streams() {
                body["stream"] = json!(true);
                body["stream_options"] = json!({"include_usage": true});
            }
        }
        NormalizedCaseKind::JsonObject => {
            body["response_format"] = json!({"type": "json_object"});
        }
    }
    lane.template.apply_request_kwargs(&mut body);
    body
}

fn probe_messages(case: NormalizedCaseKind, prompt: ProbePrompt) -> Value {
    if case == NormalizedCaseKind::OmpRepeatedPrefix {
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

fn probe_tool_schema(variant: SchemaVariant) -> Value {
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
    match variant {
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
    let schema_json = serde_json::to_string(&json!([probe_tool_schema(probe.schema_variant)]))
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
        NormalizedCaseKind::ToolRequiredStream => {
            Err("streamed tool case was routed through buffered validator".to_owned())
        }
    }
}

fn validate_streaming_probe(
    case: NormalizedCaseKind,
    assembly: &StreamAssembly,
    expected_probe_id: &str,
) -> Result<(), String> {
    if !case.streams() {
        return Err("non-streaming case was routed through streaming validator".to_owned());
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

fn phase_plan(warmups: usize, samples: usize) -> Vec<PlannedRun> {
    let mut runs = Vec::new();
    for phase in CachePhase::all() {
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

fn concurrent_phase_plan(concurrent_requests: usize, concurrent_samples: usize) -> Vec<PlannedRun> {
    let mut runs = Vec::new();
    for phase in CachePhase::all() {
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

fn effective_concurrent_samples(
    concurrent_requests: usize,
    samples: usize,
    concurrent_samples: usize,
) -> usize {
    if concurrent_samples > 0 {
        concurrent_samples
    } else if concurrent_requests > 1 {
        samples
    } else {
        0
    }
}

fn compare_normalized_lanes(
    lanes: &[NormalizedLaneReport],
    probes: &[NormalizedProbePlan],
) -> NormalizedComparisonReport {
    let mut fastest = Vec::new();
    for &probe in probes {
        for phase in CachePhase::all() {
            let mut fastest_lane = None;
            let mut fastest_latency_ms = None;
            let mut lane_metrics = Vec::new();
            for lane in lanes {
                let best_latency_ms = lane
                    .samples
                    .iter()
                    .filter(|sample| {
                        sample.case == probe.case.name()
                            && sample.schema_variant == probe.schema_variant.name()
                            && sample.tool_choice_variant == probe.tool_choice_variant.name()
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

fn aggregate_normalized_summary(
    lanes: &[NormalizedLaneReport],
    probes: &[NormalizedProbePlan],
) -> Vec<NormalizedAggregateSummaryRow> {
    let mut rows = Vec::new();
    for lane in lanes {
        for &probe in probes {
            for phase in CachePhase::all() {
                for run_mode in [RunMode::Sequential, RunMode::Concurrent] {
                    let samples = lane_samples(lane)
                        .filter(|sample| {
                            sample.case == probe.case.name()
                                && sample.schema_variant == probe.schema_variant.name()
                                && sample.tool_choice_variant == probe.tool_choice_variant.name()
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

fn agentic_gate_report(lanes: &[NormalizedLaneReport]) -> NormalizedAgenticGateReport {
    let mut rows = Vec::new();
    for probe in NormalizedProbePlan::focused_agentic_gate() {
        for phase in CachePhase::all() {
            for run_mode in [RunMode::Sequential, RunMode::Concurrent] {
                let mut lane_metrics = Vec::new();
                let mut fastest_lane = None;
                let mut fastest_latency = None;
                for lane in lanes {
                    let samples = lane_samples(lane)
                        .filter(|sample| {
                            sample.case == probe.case.name()
                                && sample.schema_variant == probe.schema_variant.name()
                                && sample.tool_choice_variant == probe.tool_choice_variant.name()
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

fn percentile_for_samples(
    samples: &[&NormalizedSampleReport],
    value: impl Fn(&NormalizedSampleReport) -> Option<u128>,
) -> Option<u128> {
    let mut values = samples
        .iter()
        .filter_map(|sample| value(sample))
        .collect::<Vec<_>>();
    values.sort_unstable();
    percentile_latency(&values, 0.50)
}

fn lane_samples(lane: &NormalizedLaneReport) -> impl Iterator<Item = &NormalizedSampleReport> {
    lane.samples.iter().chain(lane.concurrent_samples.iter())
}

fn fastest_lane_for(
    lanes: &[NormalizedLaneReport],
    probe: NormalizedProbePlan,
    phase: CachePhase,
    run_mode: RunMode,
) -> Option<String> {
    let mut fastest_lane = None;
    let mut fastest_latency = None;
    for lane in lanes {
        let mut latencies = lane_samples(lane)
            .filter(|sample| {
                sample.case == probe.case.name()
                    && sample.schema_variant == probe.schema_variant.name()
                    && sample.tool_choice_variant == probe.tool_choice_variant.name()
                    && sample.cache_phase == phase.name()
                    && sample.run_mode == run_mode.name()
                    && sample.status == "passed"
            })
            .filter_map(|sample| sample.latency_ms)
            .collect::<Vec<_>>();
        latencies.sort_unstable();
        if let Some(p50) = percentile_latency(&latencies, 0.50)
            && fastest_latency.is_none_or(|fastest| p50 < fastest)
        {
            fastest_latency = Some(p50);
            fastest_lane = Some(lane.name.clone());
        }
    }
    fastest_lane
}

fn percentile_latency(sorted_values: &[u128], percentile: f64) -> Option<u128> {
    if sorted_values.is_empty() {
        return None;
    }
    let last_index = sorted_values.len() - 1;
    let index = ((last_index as f64) * percentile).round() as usize;
    sorted_values.get(index).copied()
}

fn average_u64(values: impl Iterator<Item = u64>) -> Option<f64> {
    let mut count = 0u64;
    let mut total = 0u64;
    for value in values {
        count += 1;
        total += value;
    }
    (count > 0).then_some(total as f64 / count as f64)
}

fn average_f64(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut count = 0u64;
    let mut total = 0.0;
    for value in values {
        count += 1;
        total += value;
    }
    (count > 0).then_some(total / count as f64)
}

fn probe_case_names(probes: &[NormalizedProbePlan]) -> Vec<&'static str> {
    unique_probe_names(probes.iter().map(|probe| probe.case.name()))
}

fn probe_schema_variant_names(probes: &[NormalizedProbePlan]) -> Vec<&'static str> {
    unique_probe_names(probes.iter().map(|probe| probe.schema_variant.name()))
}

fn probe_tool_choice_variant_names(probes: &[NormalizedProbePlan]) -> Vec<&'static str> {
    unique_probe_names(probes.iter().map(|probe| probe.tool_choice_variant.name()))
}

fn unique_probe_names(names: impl Iterator<Item = &'static str>) -> Vec<&'static str> {
    let mut unique = Vec::new();
    for name in names {
        if !unique.contains(&name) {
            unique.push(name);
        }
    }
    unique
}

fn benchmark_repo_dir() -> PathBuf {
    std::env::var_os(BENCH_REPO_DIR_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .and_then(Path::parent)
                .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")))
                .to_path_buf()
        })
}

fn env_string(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn env_bool(name: &str) -> Option<bool> {
    let value = env_string(name)?;
    parse_bool_text(&value)
}

fn origin_string(value: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| value.get(*key).and_then(Value::as_str))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn origin_bool(value: &Value, keys: &[&str]) -> Option<bool> {
    for key in keys {
        let Some(value) = value.get(*key) else {
            continue;
        };
        if let Some(boolean) = value.as_bool() {
            return Some(boolean);
        }
        if let Some(text) = value.as_str()
            && let Some(boolean) = parse_bool_text(text)
        {
            return Some(boolean);
        }
    }
    None
}

fn parse_bool_text(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "dirty" => Some(true),
        "0" | "false" | "no" | "clean" => Some(false),
        _ => None,
    }
}

fn git_output_in_dir(dir: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    Some(text.trim().to_owned())
}

fn benchmark_repo_git_root(dir: &Path) -> Option<PathBuf> {
    let top_level = git_output_in_dir(dir, &["rev-parse", "--show-toplevel"])?;
    PathBuf::from(top_level).canonicalize().ok()
}

fn is_benchmark_git_root(dir: &Path) -> bool {
    let Ok(dir) = dir.canonicalize() else {
        return false;
    };
    benchmark_repo_git_root(&dir).is_some_and(|root| root == dir)
}

fn git_output(args: &[&str]) -> Option<String> {
    let dir = benchmark_repo_dir();
    if !is_benchmark_git_root(&dir) {
        return None;
    }
    git_output_in_dir(&dir, args)
}

fn git_dirty() -> bool {
    let dir = benchmark_repo_dir();
    if !is_benchmark_git_root(&dir) {
        return false;
    }
    Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["status", "--porcelain", "--untracked-files=all"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .is_some_and(|output| !output.stdout.is_empty())
}

async fn write_and_print_normalized_report(
    report: &NormalizedBenchReport,
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

#[derive(Debug, Clone)]
struct NormalizedLaneConfig {
    name: String,
    endpoint: String,
    declared_model_id: String,
    launched_model_id: Option<String>,
    snapshot_path: Option<PathBuf>,
    kind: NormalizedLaneKind,
    model_addressing: NormalizedModelAddressing,
    template: NormalizedTemplatePolicy,
    mlx_lm_settings: MlxLmSettings,
}

impl NormalizedLaneConfig {
    fn effective_request_model_id(&self) -> &str {
        match self.model_addressing {
            NormalizedModelAddressing::LoadedModelId | NormalizedModelAddressing::Custom => {
                &self.declared_model_id
            }
            NormalizedModelAddressing::DefaultModel => DEFAULT_MODEL_ID,
            NormalizedModelAddressing::ServerDefault => self
                .launched_model_id
                .as_deref()
                .or_else(|| self.snapshot_path.as_deref().and_then(Path::to_str))
                .unwrap_or(&self.declared_model_id),
        }
    }

    fn request_model_id(&self) -> Option<&str> {
        match self.model_addressing {
            NormalizedModelAddressing::ServerDefault => None,
            NormalizedModelAddressing::LoadedModelId
            | NormalizedModelAddressing::DefaultModel
            | NormalizedModelAddressing::Custom => Some(self.effective_request_model_id()),
        }
    }

    fn identity_model_id(&self) -> String {
        self.launched_model_id
            .clone()
            .or_else(|| {
                self.snapshot_path
                    .as_ref()
                    .map(|path| path.display().to_string())
            })
            .unwrap_or_else(|| self.effective_request_model_id().to_owned())
    }

    fn model_identity_source(&self) -> &'static str {
        if self.launched_model_id.is_some() {
            "lane_launched_model_id"
        } else if self.snapshot_path.is_some() {
            "lane_snapshot_path"
        } else {
            "effective_request_model_id"
        }
    }

    fn thinking_policy_report(&self) -> Value {
        self.template.thinking_policy_report()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NormalizedSweepProfile {
    QwenMlxCachePrefill,
}

impl NormalizedSweepProfile {
    fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            QWEN_MLX_CACHE_PREFILL_PROFILE => Ok(Self::QwenMlxCachePrefill),
            other => anyhow::bail!(
                "unknown --sweep-profile `{other}`; expected {QWEN_MLX_CACHE_PREFILL_PROFILE}"
            ),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::QwenMlxCachePrefill => QWEN_MLX_CACHE_PREFILL_PROFILE,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
struct MlxLmSettings {
    #[serde(rename = "mlx_prompt_cache_size")]
    prompt_cache_size: DefaultOrU64,
    #[serde(rename = "mlx_prompt_cache_bytes")]
    prompt_cache_bytes: UnsetOrU64,
    #[serde(rename = "mlx_prefill_step_size")]
    prefill_step_size: DefaultOrU64,
    #[serde(rename = "mlx_prompt_concurrency")]
    prompt_concurrency: DefaultOrU32,
    #[serde(rename = "mlx_decode_concurrency")]
    decode_concurrency: DefaultOrU32,
}

impl MlxLmSettings {
    fn parse(values: &mut BTreeMap<String, String>) -> anyhow::Result<Self> {
        Ok(Self {
            prompt_cache_size: parse_default_or_u64(values, "mlx_prompt_cache_size")?,
            prompt_cache_bytes: parse_unset_or_u64(values, "mlx_prompt_cache_bytes")?,
            prefill_step_size: parse_default_or_u64(values, "mlx_prefill_step_size")?,
            prompt_concurrency: parse_default_or_u32(values, "mlx_prompt_concurrency")?,
            decode_concurrency: parse_default_or_u32(values, "mlx_decode_concurrency")?,
        })
    }
}

impl Default for MlxLmSettings {
    fn default() -> Self {
        Self {
            prompt_cache_size: DefaultOrU64::Default,
            prompt_cache_bytes: UnsetOrU64::Unset,
            prefill_step_size: DefaultOrU64::Default,
            prompt_concurrency: DefaultOrU32::Default,
            decode_concurrency: DefaultOrU32::Default,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DefaultOrU64 {
    Default,
    Value(u64),
}

impl Serialize for DefaultOrU64 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Default => serializer.serialize_str("default"),
            Self::Value(value) => serializer.serialize_u64(*value),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UnsetOrU64 {
    Unset,
    Value(u64),
}

impl Serialize for UnsetOrU64 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Unset => serializer.serialize_str("unset"),
            Self::Value(value) => serializer.serialize_u64(*value),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DefaultOrU32 {
    Default,
    Value(u32),
}

impl Serialize for DefaultOrU32 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Default => serializer.serialize_str("default"),
            Self::Value(value) => serializer.serialize_u32(*value),
        }
    }
}

fn parse_default_or_u64(
    values: &mut BTreeMap<String, String>,
    key: &'static str,
) -> anyhow::Result<DefaultOrU64> {
    let Some(value) = values.remove(key) else {
        return Ok(DefaultOrU64::Default);
    };
    if value == "default" {
        return Ok(DefaultOrU64::Default);
    }
    parse_positive_u64(key, &value).map(DefaultOrU64::Value)
}

fn parse_unset_or_u64(
    values: &mut BTreeMap<String, String>,
    key: &'static str,
) -> anyhow::Result<UnsetOrU64> {
    let Some(value) = values.remove(key) else {
        return Ok(UnsetOrU64::Unset);
    };
    if value == "unset" {
        return Ok(UnsetOrU64::Unset);
    }
    parse_positive_u64(key, &value).map(UnsetOrU64::Value)
}

fn parse_default_or_u32(
    values: &mut BTreeMap<String, String>,
    key: &'static str,
) -> anyhow::Result<DefaultOrU32> {
    let Some(value) = values.remove(key) else {
        return Ok(DefaultOrU32::Default);
    };
    if value == "default" {
        return Ok(DefaultOrU32::Default);
    }
    parse_positive_u32(key, &value).map(DefaultOrU32::Value)
}

fn parse_positive_u64(key: &str, value: &str) -> anyhow::Result<u64> {
    let parsed = value
        .parse::<u64>()
        .with_context(|| format!("parse {key}; expected default/unset or a positive integer"))?;
    if parsed == 0 {
        anyhow::bail!("{key} must be greater than zero");
    }
    Ok(parsed)
}

fn parse_positive_u32(key: &str, value: &str) -> anyhow::Result<u32> {
    let parsed = value
        .parse::<u32>()
        .with_context(|| format!("parse {key}; expected default or a positive integer"))?;
    if parsed == 0 {
        anyhow::bail!("{key} must be greater than zero");
    }
    Ok(parsed)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NormalizedLaneKind {
    DirectMlx,
    KirAiProxy,
    Other,
}

impl NormalizedLaneKind {
    fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "direct_mlx" => Ok(Self::DirectMlx),
            "kir_ai_proxy" => Ok(Self::KirAiProxy),
            "other" => Ok(Self::Other),
            other => anyhow::bail!(
                "unknown lane kind `{other}`; expected direct_mlx, kir_ai_proxy, or other"
            ),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::DirectMlx => "direct_mlx",
            Self::KirAiProxy => "kir_ai_proxy",
            Self::Other => "other",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NormalizedModelAddressing {
    LoadedModelId,
    DefaultModel,
    ServerDefault,
    Custom,
}

impl NormalizedModelAddressing {
    fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "loaded_model_id" => Ok(Self::LoadedModelId),
            "default_model" => Ok(Self::DefaultModel),
            "server_default" => Ok(Self::ServerDefault),
            "custom" => Ok(Self::Custom),
            other => anyhow::bail!(
                "unknown model_addressing `{other}`; expected loaded_model_id, default_model, server_default, or custom"
            ),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::LoadedModelId => "loaded_model_id",
            Self::DefaultModel => "default_model",
            Self::ServerDefault => "server_default",
            Self::Custom => "custom",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NormalizedTemplatePolicy {
    QwenNoThinking,
    SidecarChatTemplateArgs,
    None,
}

impl NormalizedTemplatePolicy {
    fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "qwen-no-thinking" => Ok(Self::QwenNoThinking),
            "sidecar-chat-template-args" => Ok(Self::SidecarChatTemplateArgs),
            "none" => Ok(Self::None),
            other => anyhow::bail!(
                "unknown template `{other}`; expected qwen-no-thinking, sidecar-chat-template-args, or none"
            ),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::QwenNoThinking => "qwen-no-thinking",
            Self::SidecarChatTemplateArgs => "sidecar-chat-template-args",
            Self::None => "none",
        }
    }

    fn apply_request_kwargs(self, body: &mut Value) {
        if matches!(self, Self::QwenNoThinking) {
            body["chat_template_kwargs"] = json!({"enable_thinking": false});
        }
    }

    fn thinking_policy_report(self) -> Value {
        match self {
            Self::QwenNoThinking => json!({
                "template": self.as_str(),
                "enable_thinking": false,
                "source": "request_chat_template_kwargs",
                "request_chat_template_kwargs": {"enable_thinking": false}
            }),
            Self::SidecarChatTemplateArgs => json!({
                "template": self.as_str(),
                "enable_thinking": false,
                "source": "sidecar_chat_template_args_declared_by_lane"
            }),
            Self::None => json!({
                "template": self.as_str(),
                "enable_thinking": Value::Null,
                "source": "not_configured"
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NormalizedCaseKind {
    ToolRequired,
    ToolRequiredStream,
    JsonObject,
    OmpRepeatedPrefix,
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
        }
    }

    fn probe_id(self) -> &'static str {
        match self {
            Self::ToolRequired => "KIR_QWEN_MLX_TOOL_NORMALIZED_TOOL_REQUIRED",
            Self::ToolRequiredStream => "KIR_QWEN_MLX_TOOL_NORMALIZED_TOOL_REQUIRED_STREAM",
            Self::JsonObject => "KIR_QWEN_MLX_TOOL_NORMALIZED_JSON_OBJECT",
            Self::OmpRepeatedPrefix => "KIR_QWEN_MLX_TOOL_NORMALIZED_OMP_REPEATED_PREFIX",
        }
    }

    fn system_prompt(self) -> &'static str {
        match self {
            Self::ToolRequired | Self::ToolRequiredStream => {
                "You are a tool-call conformance probe. Use the provided function exactly once."
            }
            Self::JsonObject => {
                "You are a JSON conformance probe. Return one JSON object and no prose."
            }
            Self::OmpRepeatedPrefix => {
                "You are an OMP-style repeated-prefix workflow probe. Continue the tool workflow and use the provided function exactly once."
            }
        }
    }

    fn streams(self) -> bool {
        matches!(self, Self::ToolRequiredStream)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SchemaVariant {
    None,
    BaselineCurrent,
    CanonicalCurrent,
    BaselinePermutedEquivalent,
    CanonicalPermutedEquivalent,
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

    fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::BaselineCurrent => "baseline_current",
            Self::CanonicalCurrent => "canonical_current",
            Self::BaselinePermutedEquivalent => "baseline_permuted_equivalent",
            Self::CanonicalPermutedEquivalent => "canonical_permuted_equivalent",
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

    fn request_value(self) -> Value {
        match self {
            Self::Required => json!("required"),
            Self::Function => {
                json!({"type": "function", "function": {"name": "record_qwen_tool_probe"}})
            }
            Self::None => Value::Null,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NormalizedProbeSuite {
    FullMatrix,
    FocusedAgenticGate,
}

impl NormalizedProbeSuite {
    fn name(self) -> &'static str {
        match self {
            Self::FullMatrix => "full_matrix",
            Self::FocusedAgenticGate => "focused_agentic_gate",
        }
    }

    fn probes(self) -> Vec<NormalizedProbePlan> {
        match self {
            Self::FullMatrix => NormalizedProbePlan::all(),
            Self::FocusedAgenticGate => NormalizedProbePlan::focused_agentic_gate(),
        }
    }

    fn case_names(self, probes: &[NormalizedProbePlan]) -> Vec<&'static str> {
        match self {
            Self::FullMatrix => NormalizedCaseKind::all()
                .iter()
                .map(|case| case.name())
                .collect(),
            Self::FocusedAgenticGate => probe_case_names(probes),
        }
    }

    fn schema_variant_names(self, probes: &[NormalizedProbePlan]) -> Vec<&'static str> {
        match self {
            Self::FullMatrix => SchemaVariant::all()
                .iter()
                .map(|variant| variant.name())
                .collect(),
            Self::FocusedAgenticGate => probe_schema_variant_names(probes),
        }
    }

    fn tool_choice_variant_names(self, probes: &[NormalizedProbePlan]) -> Vec<&'static str> {
        match self {
            Self::FullMatrix => ToolChoiceVariant::all()
                .iter()
                .map(|variant| variant.name())
                .collect(),
            Self::FocusedAgenticGate => probe_tool_choice_variant_names(probes),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NormalizedProbePlan {
    case: NormalizedCaseKind,
    schema_variant: SchemaVariant,
    tool_choice_variant: ToolChoiceVariant,
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
        }
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
    fn prompt(self, context_tokens: usize) -> ProbePrompt {
        match (self.kind, self.phase) {
            (PlannedRunKind::Warmup, CachePhase::WarmSameToolSchema) => {
                ProbePrompt::schema_warmup(context_tokens, self.warmup_index.unwrap_or_default())
            }
            _ => ProbePrompt::measured(
                context_tokens,
                self.sample_index.unwrap_or_default(),
                self.request_index,
            ),
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
}

#[derive(Debug, Clone, Copy)]
struct ProbePrompt {
    variant: ProbePromptVariant,
    context_tokens: usize,
    sample_index: usize,
    request_index: Option<usize>,
}

impl ProbePrompt {
    fn measured(context_tokens: usize, sample_index: usize, request_index: Option<usize>) -> Self {
        Self {
            variant: ProbePromptVariant::Measured,
            context_tokens,
            sample_index,
            request_index,
        }
    }

    fn schema_warmup(context_tokens: usize, index: usize) -> Self {
        Self {
            variant: ProbePromptVariant::SchemaWarmup(index),
            context_tokens,
            sample_index: 0,
            request_index: None,
        }
    }

    fn planned_prompt_tokens(self) -> usize {
        self.context_tokens
    }

    fn probe_id(self, case: NormalizedCaseKind) -> String {
        match self.variant {
            ProbePromptVariant::Measured => case.probe_id().to_owned(),
            ProbePromptVariant::SchemaWarmup(index) => {
                format!("{}_SCHEMA_WARMUP_{index}", case.probe_id())
            }
        }
    }

    fn user_prompt(self, case: NormalizedCaseKind) -> String {
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
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum ProbePromptVariant {
    Measured,
    SchemaWarmup(usize),
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

#[derive(Debug, Serialize)]
struct NormalizedBenchReport {
    benchmark: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    sweep_profile: Option<&'static str>,
    status: String,
    generated_at_unix_ms: u128,
    trace_output_path: Option<String>,
    warmups: usize,
    samples: usize,
    context_tokens: usize,
    concurrent_requests: usize,
    concurrent_samples: usize,
    effective_concurrent_samples: usize,
    timeout_ms: u64,
    connect_timeout_ms: u64,
    probe_suite: &'static str,
    repo_revision: RepoRevisionReport,
    cases: Vec<&'static str>,
    schema_variants: Vec<&'static str>,
    tool_choice_variants: Vec<&'static str>,
    cache_phases: Vec<&'static str>,
    summary: Vec<NormalizedAggregateSummaryRow>,
    lanes: Vec<NormalizedLaneReport>,
    hardware: HardwareReport,
    comparison: NormalizedComparisonReport,
    agentic_gate: NormalizedAgenticGateReport,
}

#[derive(Debug, Serialize)]
struct RepoRevisionReport {
    branch: Option<String>,
    commit_sha: Option<String>,
    dirty: bool,
}

impl RepoRevisionReport {
    fn detect() -> Self {
        if let Some(report) = Self::from_env() {
            return report;
        }
        if let Some(report) = Self::from_origin_file() {
            return report;
        }
        Self {
            branch: git_output(&["branch", "--show-current"]).filter(|branch| !branch.is_empty()),
            commit_sha: git_output(&["rev-parse", "HEAD"]),
            dirty: git_dirty(),
        }
    }

    fn from_env() -> Option<Self> {
        let branch = env_string(BENCH_REPO_BRANCH_ENV);
        let commit_sha = env_string(BENCH_REPO_COMMIT_ENV);
        let dirty = env_bool(BENCH_REPO_DIRTY_ENV);
        if branch.is_none() && commit_sha.is_none() && dirty.is_none() {
            return None;
        }
        Some(Self {
            branch,
            commit_sha,
            dirty: dirty.unwrap_or(false),
        })
    }

    fn from_origin_file() -> Option<Self> {
        let path = benchmark_repo_dir().join(BENCH_REPO_ORIGIN_FILE);
        let value = serde_json::from_slice::<Value>(&std::fs::read(path).ok()?).ok()?;
        let revision = value.get("repo_revision").unwrap_or(&value);
        let branch = origin_string(revision, &["branch", "git_branch", "ref"]);
        let commit_sha = origin_string(
            revision,
            &[
                "commit_sha",
                "commit",
                "git_commit",
                "revision",
                "source_commit",
            ],
        );
        let dirty = origin_bool(revision, &["dirty", "git_dirty"]);
        if branch.is_none() && commit_sha.is_none() && dirty.is_none() {
            return None;
        }
        Some(Self {
            branch,
            commit_sha,
            dirty: dirty.unwrap_or(false),
        })
    }
}

#[derive(Debug, Serialize)]
struct NormalizedLaneReport {
    name: String,
    status: String,
    endpoint: String,
    kind: &'static str,
    declared_model_id: String,
    effective_request_model_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    launched_model_id: Option<String>,
    model_identity_source: &'static str,
    model_addressing: &'static str,
    mlx_lm_settings: MlxLmSettings,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot_identity: Option<ModelIdentityReport>,
    qwen_thinking_policy: Value,
    warmups: usize,
    sample_count: usize,
    samples: Vec<NormalizedSampleReport>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    concurrent_samples: Vec<NormalizedSampleReport>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    warmup_failures: Vec<NormalizedWarmupFailure>,
}

impl NormalizedLaneReport {
    fn planned(
        lane: &NormalizedLaneConfig,
        warmups: usize,
        samples: usize,
        snapshot_identity: Option<ModelIdentityReport>,
    ) -> Self {
        Self {
            name: lane.name.clone(),
            status: "planned".to_owned(),
            endpoint: lane.endpoint.clone(),
            kind: lane.kind.as_str(),
            declared_model_id: lane.declared_model_id.clone(),
            effective_request_model_id: lane.effective_request_model_id().to_owned(),
            launched_model_id: lane.launched_model_id.clone(),
            model_identity_source: lane.model_identity_source(),
            model_addressing: lane.model_addressing.as_str(),
            mlx_lm_settings: lane.mlx_lm_settings,
            snapshot_path: lane
                .snapshot_path
                .as_ref()
                .map(|path| path.display().to_string()),
            snapshot_identity,
            qwen_thinking_policy: lane.thinking_policy_report(),
            warmups,
            sample_count: samples,
            samples: Vec::new(),
            concurrent_samples: Vec::new(),
            warmup_failures: Vec::new(),
        }
    }

    fn dry_run(
        lane: &NormalizedLaneConfig,
        run_config: NormalizedRunConfig,
        snapshot_identity: Option<ModelIdentityReport>,
        probes: &[NormalizedProbePlan],
    ) -> Self {
        let mut report = Self::planned(
            lane,
            run_config.warmups,
            run_config.samples,
            snapshot_identity,
        );
        report.status = "dry_run".to_owned();
        for &probe in probes {
            for planned in phase_plan(run_config.warmups, run_config.samples) {
                if planned.kind == PlannedRunKind::Measured {
                    let mut sample = NormalizedSampleReport::base(
                        probe,
                        planned.phase,
                        planned.run_mode,
                        planned.sample_index.unwrap_or_default(),
                        planned.request_index,
                        planned.phase.warms_before_samples() && run_config.warmups > 0,
                        run_config.context_tokens,
                    );
                    sample.status = "dry_run".to_owned();
                    sample.classification = "planned".to_owned();
                    sample.cached_tokens_status = "not_measured";
                    report.samples.push(sample);
                }
            }
            for planned in concurrent_phase_plan(
                run_config.concurrent_requests,
                run_config.effective_concurrent_samples,
            ) {
                let mut sample = NormalizedSampleReport::base(
                    probe,
                    planned.phase,
                    planned.run_mode,
                    planned.sample_index.unwrap_or_default(),
                    planned.request_index,
                    planned.phase.warms_before_samples() && run_config.warmups > 0,
                    run_config.context_tokens,
                );
                sample.status = "dry_run".to_owned();
                sample.classification = "planned".to_owned();
                sample.cached_tokens_status = "not_measured";
                report.concurrent_samples.push(sample);
            }
        }
        report
    }
}

#[derive(Debug, Serialize)]
struct NormalizedWarmupFailure {
    case: &'static str,
    schema_variant: &'static str,
    tool_choice_variant: &'static str,
    cache_phase: &'static str,
    warmup_index: usize,
    classification: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    http_status: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct NormalizedSampleReport {
    case: &'static str,
    schema_variant: &'static str,
    tool_choice_variant: &'static str,
    schema_canonicalized: bool,
    schema_permuted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_schema_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_schema_bytes: Option<usize>,
    cache_phase: &'static str,
    run_mode: &'static str,
    sample_index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_index: Option<usize>,
    planned_prompt_tokens: usize,
    prewarmed: bool,
    status: String,
    classification: String,
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
    cached_tokens_status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    cached_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    http_status: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

impl NormalizedSampleReport {
    fn base(
        probe: NormalizedProbePlan,
        phase: CachePhase,
        run_mode: RunMode,
        sample_index: usize,
        request_index: Option<usize>,
        prewarmed: bool,
        planned_prompt_tokens: usize,
    ) -> Self {
        let tool_schema = tool_schema_metadata(probe);
        Self {
            case: probe.case.name(),
            schema_variant: probe.schema_variant.name(),
            tool_choice_variant: probe.tool_choice_variant.name(),
            schema_canonicalized: probe.schema_variant.canonicalized(),
            schema_permuted: probe.schema_variant.permuted(),
            tool_schema_sha256: tool_schema.sha256,
            tool_schema_bytes: tool_schema.bytes,
            cache_phase: phase.name(),
            run_mode: run_mode.name(),
            sample_index,
            request_index,
            planned_prompt_tokens,
            prewarmed,
            status: "planned".to_owned(),
            classification: "planned".to_owned(),
            latency_ms: None,
            stream_timing: StreamTimingReport::default(),
            tokens_per_second: None,
            prompt_tokens: None,
            completion_tokens: None,
            total_tokens: None,
            cached_tokens_status: "missing",
            cached_tokens: None,
            http_status: None,
            finish_reason: None,
            error: None,
        }
    }
}

#[derive(Debug, Serialize)]
struct NormalizedComparisonReport {
    status: String,
    fastest_successful_lanes: Vec<NormalizedFastestLaneReport>,
}

impl NormalizedComparisonReport {
    fn dry_run() -> Self {
        Self {
            status: "dry_run".to_owned(),
            fastest_successful_lanes: Vec::new(),
        }
    }
}

#[derive(Debug, Serialize)]
struct NormalizedFastestLaneReport {
    case: &'static str,
    schema_variant: &'static str,
    tool_choice_variant: &'static str,
    cache_phase: &'static str,
    fastest_lane: Option<String>,
    fastest_latency_ms: Option<u128>,
    lanes: Vec<NormalizedComparisonLaneMetric>,
}

#[derive(Debug, Serialize)]
struct NormalizedComparisonLaneMetric {
    lane: String,
    status: String,
    best_latency_ms: Option<u128>,
}

#[derive(Debug, Serialize)]
struct NormalizedAgenticGateReport {
    status: String,
    rows: Vec<NormalizedAgenticGateRow>,
}

impl NormalizedAgenticGateReport {
    fn dry_run() -> Self {
        Self {
            status: "dry_run".to_owned(),
            rows: Vec::new(),
        }
    }
}

#[derive(Debug, Serialize)]
struct NormalizedAgenticGateRow {
    case: &'static str,
    schema_variant: &'static str,
    tool_choice_variant: &'static str,
    cache_phase: &'static str,
    run_mode: &'static str,
    fastest_lane: Option<String>,
    lanes: Vec<NormalizedAgenticGateLaneMetric>,
}

#[derive(Debug, Serialize)]
struct NormalizedAgenticGateLaneMetric {
    lane: String,
    pass_count: usize,
    p50_latency_ms: Option<u128>,
    latency_delta_vs_fastest_ms: Option<u128>,
    p50_first_byte_latency_ms: Option<u128>,
    p50_first_semantic_delta_latency_ms: Option<u128>,
    p50_first_tool_delta_latency_ms: Option<u128>,
    avg_tokens_per_second: Option<f64>,
    avg_cached_tokens: Option<f64>,
    avg_prompt_tokens: Option<f64>,
    avg_completion_tokens: Option<f64>,
}

#[derive(Debug, Serialize)]
struct NormalizedAggregateSummaryRow {
    lane: String,
    case: &'static str,
    schema_variant: &'static str,
    tool_choice_variant: &'static str,
    cache_phase: &'static str,
    run_mode: &'static str,
    pass_count: usize,
    fail_count: usize,
    p50_latency_ms: Option<u128>,
    p95_latency_ms: Option<u128>,
    avg_cached_tokens: Option<f64>,
    avg_prompt_tokens: Option<f64>,
    avg_completion_tokens: Option<f64>,
    avg_total_tokens: Option<f64>,
    fastest_lane: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::super::{StreamAssembly, StreamTimingReport, apply_sse_frame, usage_from_value};
    use super::*;
    use crate::DEFAULT_MODEL_ID;
    use serde_json::json;

    fn lane(spec: &str) -> NormalizedLaneConfig {
        parse_lane_spec(spec).expect("lane spec parses")
    }

    fn args(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|part| (*part).to_owned()).collect()
    }

    #[test]
    fn qwen_mlx_tool_normalized_lane_spec_defaults_to_qwen_no_thinking_and_rejects_unknown_keys() {
        let lane = lane("name=direct,endpoint=http://127.0.0.1:8080/v1/,model=qwen-loaded");

        assert_eq!(lane.name, "direct");
        assert_eq!(lane.endpoint, "http://127.0.0.1:8080/v1");
        assert_eq!(lane.kind, NormalizedLaneKind::Other);
        assert_eq!(
            lane.model_addressing,
            NormalizedModelAddressing::LoadedModelId
        );
        assert_eq!(lane.template, NormalizedTemplatePolicy::QwenNoThinking);
        assert_eq!(lane.effective_request_model_id(), "qwen-loaded");

        let err = parse_lane_spec(
            "name=direct,endpoint=http://127.0.0.1:8080/v1,model=qwen,unknown=value",
        )
        .expect_err("unknown keys fail");
        assert!(
            err.to_string().contains("unknown keys: unknown"),
            "error: {err}"
        );
    }

    #[test]
    fn qwen_mlx_tool_normalized_cache_prefill_profile_expands_default_lanes() {
        let snapshot = "/tmp/huggingface/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/abcdef1234567890";
        let lanes = parse_lane_specs(&args(&[
            "--sweep-profile",
            "qwen-mlx-cache-prefill",
            "--snapshot",
            snapshot,
        ]))
        .expect("profile expands");

        assert_eq!(
            lanes
                .iter()
                .map(|lane| lane.name.as_str())
                .collect::<Vec<_>>(),
            [
                "mlx-default",
                "mlx-cache-size-4096",
                "mlx-cache-bytes-1g",
                "mlx-prefill-2048",
                "mlx-prefill-4096",
                "mlx-prefill-8192",
                "mlx-concurrent-4x2",
                "kir-proxy",
            ]
        );
        assert_eq!(
            lanes
                .iter()
                .map(|lane| lane.endpoint.as_str())
                .collect::<Vec<_>>(),
            [
                "http://127.0.0.1:8080/v1",
                "http://127.0.0.1:8081/v1",
                "http://127.0.0.1:8082/v1",
                "http://127.0.0.1:8083/v1",
                "http://127.0.0.1:8084/v1",
                "http://127.0.0.1:8085/v1",
                "http://127.0.0.1:8086/v1",
                "http://127.0.0.1:3000",
            ]
        );
        assert!(
            lanes
                .iter()
                .all(|lane| lane.launched_model_id.as_deref() == Some(snapshot))
        );
        assert!(
            lanes
                .iter()
                .all(|lane| lane.snapshot_path.as_deref() == Some(Path::new(snapshot)))
        );

        let default = &lanes[0];
        assert_eq!(default.kind, NormalizedLaneKind::DirectMlx);
        assert_eq!(default.declared_model_id, snapshot);
        assert_eq!(
            default.model_addressing,
            NormalizedModelAddressing::ServerDefault
        );
        let direct_body = probe_request_body(
            default,
            NormalizedProbePlan::new(
                NormalizedCaseKind::JsonObject,
                SchemaVariant::None,
                ToolChoiceVariant::None,
            ),
            ProbePrompt::measured(128, 0, None),
        );
        assert!(
            direct_body.get("model").is_none(),
            "plain mlx_lm.server treats unknown model ids as Hugging Face repos"
        );
        assert_eq!(
            default.template,
            NormalizedTemplatePolicy::SidecarChatTemplateArgs
        );
        assert_eq!(
            default.mlx_lm_settings.prompt_cache_size,
            DefaultOrU64::Default
        );
        assert_eq!(
            lanes[1].mlx_lm_settings.prompt_cache_size,
            DefaultOrU64::Value(4096)
        );
        assert_eq!(
            lanes[2].mlx_lm_settings.prompt_cache_bytes,
            UnsetOrU64::Value(1_073_741_824)
        );
        assert_eq!(
            lanes[3].mlx_lm_settings.prefill_step_size,
            DefaultOrU64::Value(2048)
        );
        assert_eq!(
            lanes[4].mlx_lm_settings.prefill_step_size,
            DefaultOrU64::Value(4096)
        );
        assert_eq!(
            lanes[5].mlx_lm_settings.prefill_step_size,
            DefaultOrU64::Value(8192)
        );
        assert_eq!(
            lanes[6].mlx_lm_settings.prompt_concurrency,
            DefaultOrU32::Value(4)
        );
        assert_eq!(
            lanes[6].mlx_lm_settings.decode_concurrency,
            DefaultOrU32::Value(2)
        );

        let proxy = &lanes[7];
        assert_eq!(proxy.kind, NormalizedLaneKind::KirAiProxy);
        assert_eq!(proxy.declared_model_id, "local-qwen36-mlx");
        assert_eq!(proxy.effective_request_model_id(), DEFAULT_MODEL_ID);
        assert_eq!(
            proxy.model_addressing,
            NormalizedModelAddressing::DefaultModel
        );
        assert_eq!(
            proxy.template,
            NormalizedTemplatePolicy::SidecarChatTemplateArgs
        );
    }

    #[test]
    fn qwen_mlx_tool_normalized_cache_prefill_profile_requires_snapshot() {
        let err = parse_lane_specs(&args(&["--sweep-profile", "qwen-mlx-cache-prefill"]))
            .expect_err("profile requires snapshot");

        assert!(
            err.to_string().contains("--snapshot"),
            "error should mention --snapshot: {err}"
        );
    }

    #[test]
    fn qwen_mlx_tool_normalized_explicit_lane_mode_remains_available() {
        let lanes = parse_lane_specs(&args(&[
            "--lane",
            "name=custom,endpoint=http://127.0.0.1:9090/v1,model=qwen-custom,kind=direct_mlx",
        ]))
        .expect("explicit lane mode parses");

        assert_eq!(lanes.len(), 1);
        assert_eq!(lanes[0].name, "custom");
        assert_eq!(lanes[0].endpoint, "http://127.0.0.1:9090/v1");
        assert_eq!(lanes[0].declared_model_id, "qwen-custom");
    }

    #[test]
    fn qwen_mlx_tool_normalized_lane_spec_parses_mlx_lm_sweep_knobs_and_serializes_metadata() {
        let parsed_lane = lane(
            "name=direct,endpoint=http://127.0.0.1:8080/v1,model=qwen,mlx_prompt_cache_size=4096,mlx_prompt_cache_bytes=unset,mlx_prefill_step_size=8192,mlx_prompt_concurrency=4,mlx_decode_concurrency=2",
        );

        assert_eq!(
            parsed_lane.mlx_lm_settings.prompt_cache_size,
            DefaultOrU64::Value(4096)
        );
        assert_eq!(
            parsed_lane.mlx_lm_settings.prompt_cache_bytes,
            UnsetOrU64::Unset
        );
        assert_eq!(
            parsed_lane.mlx_lm_settings.prefill_step_size,
            DefaultOrU64::Value(8192)
        );
        assert_eq!(
            parsed_lane.mlx_lm_settings.prompt_concurrency,
            DefaultOrU32::Value(4)
        );
        assert_eq!(
            parsed_lane.mlx_lm_settings.decode_concurrency,
            DefaultOrU32::Value(2)
        );

        let defaulted = lane("name=defaulted,endpoint=http://127.0.0.1:8081/v1,model=qwen-default");
        assert_eq!(
            defaulted.mlx_lm_settings.prompt_cache_size,
            DefaultOrU64::Default
        );
        assert_eq!(
            defaulted.mlx_lm_settings.prompt_cache_bytes,
            UnsetOrU64::Unset
        );
        assert_eq!(
            defaulted.mlx_lm_settings.prefill_step_size,
            DefaultOrU64::Default
        );
        assert_eq!(
            defaulted.mlx_lm_settings.prompt_concurrency,
            DefaultOrU32::Default
        );
        assert_eq!(
            defaulted.mlx_lm_settings.decode_concurrency,
            DefaultOrU32::Default
        );

        let report = NormalizedLaneReport::dry_run(
            &parsed_lane,
            NormalizedRunConfig::new(1, 1, 128, 1, 0),
            None,
            &NormalizedProbePlan::all(),
        );
        let value = serde_json::to_value(report).expect("lane report serializes");
        assert_eq!(value["mlx_lm_settings"]["mlx_prompt_cache_size"], 4096);
        assert_eq!(value["mlx_lm_settings"]["mlx_prompt_cache_bytes"], "unset");
        assert_eq!(value["mlx_lm_settings"]["mlx_prefill_step_size"], 8192);
        assert_eq!(value["mlx_lm_settings"]["mlx_prompt_concurrency"], 4);
        assert_eq!(value["mlx_lm_settings"]["mlx_decode_concurrency"], 2);
    }

    #[test]
    fn qwen_mlx_tool_normalized_lane_spec_rejects_invalid_mlx_lm_sweep_knobs() {
        for spec in [
            "name=bad,endpoint=http://127.0.0.1:8080/v1,model=qwen,mlx_prompt_cache_size=0",
            "name=bad,endpoint=http://127.0.0.1:8080/v1,model=qwen,mlx_prompt_cache_size=-1",
            "name=bad,endpoint=http://127.0.0.1:8080/v1,model=qwen,mlx_prompt_cache_bytes=default",
            "name=bad,endpoint=http://127.0.0.1:8080/v1,model=qwen,mlx_prefill_step_size=0",
            "name=bad,endpoint=http://127.0.0.1:8080/v1,model=qwen,mlx_prompt_concurrency=0",
            "name=bad,endpoint=http://127.0.0.1:8080/v1,model=qwen,mlx_decode_concurrency=-2",
        ] {
            let err = parse_lane_spec(spec).expect_err("invalid MLX knob should fail");
            assert!(
                err.to_string().contains("mlx_"),
                "error should name MLX knob for `{spec}`: {err}"
            );
        }
    }

    #[test]
    fn qwen_mlx_tool_normalized_sidecar_template_policy_declares_assumption_without_request_kwargs()
    {
        let lane = lane(
            "name=sidecar,endpoint=http://127.0.0.1:8080/v1,model=qwen,template=sidecar-chat-template-args",
        );

        let body = probe_request_body(
            &lane,
            NormalizedProbePlan::new(
                NormalizedCaseKind::ToolRequired,
                SchemaVariant::BaselineCurrent,
                ToolChoiceVariant::Required,
            ),
            ProbePrompt::measured(128, 0, None),
        );
        assert!(
            body.get("chat_template_kwargs").is_none(),
            "sidecar template policy must not inject request kwargs: {body}"
        );

        let policy = lane.thinking_policy_report();
        assert_eq!(policy["template"], "sidecar-chat-template-args");
        assert_eq!(policy["enable_thinking"], false);
        assert_eq!(
            policy["source"],
            "sidecar_chat_template_args_declared_by_lane"
        );
    }

    #[test]
    fn qwen_mlx_tool_normalized_model_addressing_controls_effective_request_model_id_and_serializes()
     {
        let loaded = lane(
            "name=loaded,endpoint=http://127.0.0.1:8080/v1,model=qwen-loaded,model_addressing=loaded_model_id",
        );
        let default_model = lane(
            "name=default,endpoint=http://127.0.0.1:8081/v1,model=qwen-loaded,model_addressing=default_model",
        );
        let custom = lane(
            "name=custom,endpoint=http://127.0.0.1:8082/v1,model=qwen-custom,model_addressing=custom",
        );
        let server_default = lane(
            "name=server-default,endpoint=http://127.0.0.1:8083/v1,model=qwen-loaded,snapshot=/models/qwen-snapshot,model_addressing=server_default",
        );

        assert_eq!(loaded.effective_request_model_id(), "qwen-loaded");
        assert_eq!(default_model.effective_request_model_id(), DEFAULT_MODEL_ID);
        assert_eq!(custom.effective_request_model_id(), "qwen-custom");
        assert_eq!(
            server_default.effective_request_model_id(),
            "/models/qwen-snapshot"
        );
        assert_eq!(server_default.request_model_id(), None);

        let report = NormalizedLaneReport::dry_run(
            &default_model,
            NormalizedRunConfig::new(1, 1, 128, 1, 0),
            None,
            &NormalizedProbePlan::all(),
        );
        let value = serde_json::to_value(report).expect("lane report serializes");
        assert_eq!(value["declared_model_id"], "qwen-loaded");
        assert_eq!(value["effective_request_model_id"], DEFAULT_MODEL_ID);
        assert_eq!(value["model_addressing"], "default_model");

        let body = probe_request_body(
            &server_default,
            NormalizedProbePlan::new(
                NormalizedCaseKind::JsonObject,
                SchemaVariant::None,
                ToolChoiceVariant::None,
            ),
            ProbePrompt::measured(128, 0, None),
        );
        assert!(body.get("model").is_none());
    }

    #[test]
    fn qwen_mlx_tool_normalized_lane_can_pin_launched_model_identity() {
        let lane = lane(
            "name=direct,endpoint=http://127.0.0.1:8080/v1,model=default_model,launched_model_id=/models/qwen-snapshot,kind=direct_mlx,model_addressing=loaded_model_id",
        );

        assert_eq!(lane.effective_request_model_id(), "default_model");
        assert_eq!(lane.identity_model_id(), "/models/qwen-snapshot");

        let report = NormalizedLaneReport::dry_run(
            &lane,
            NormalizedRunConfig::new(0, 1, 128, 1, 0),
            None,
            &NormalizedProbePlan::all(),
        );
        let value = serde_json::to_value(report).expect("lane report serializes");
        assert_eq!(value["declared_model_id"], "default_model");
        assert_eq!(value["effective_request_model_id"], "default_model");
        assert_eq!(value["launched_model_id"], "/models/qwen-snapshot");
        assert_eq!(value["model_identity_source"], "lane_launched_model_id");
    }

    #[tokio::test]
    async fn qwen_mlx_tool_normalized_raw_hf_snapshot_identity_does_not_require_kir_manifest() {
        let temp = tempfile::tempdir().expect("tempdir");
        let snapshot = temp
            .path()
            .join("huggingface")
            .join("models--mlx-community--Qwen3.6-35B-A3B-4bit")
            .join("snapshots")
            .join("abcdef1234567890");
        tokio::fs::create_dir_all(&snapshot)
            .await
            .expect("raw snapshot dir");
        tokio::fs::write(snapshot.join("config.json"), "{}")
            .await
            .expect("config");

        let lane = lane(&format!(
            "name=direct,endpoint=http://127.0.0.1:8080/v1,model=qwen-loaded,snapshot={},kind=direct_mlx",
            snapshot.display()
        ));
        let identity = load_lane_snapshot_identity(&lane, false)
            .await
            .expect("raw HF snapshot identity should not require llm-engine-manifest.json")
            .expect("snapshot identity");

        let snapshot_display = snapshot.display().to_string();
        assert_eq!(identity.id, snapshot_display);
        assert_eq!(
            identity.snapshot_path.as_deref(),
            Some(snapshot_display.as_str())
        );
        assert_eq!(
            identity.repo_id.as_deref(),
            Some("mlx-community/Qwen3.6-35B-A3B-4bit")
        );
        assert_eq!(
            identity.resolved_commit.as_deref(),
            Some("abcdef1234567890")
        );
        assert_eq!(identity.manifest_digest, None);
    }

    #[test]
    fn qwen_mlx_tool_normalized_probe_plan_expands_schema_and_tool_choice_variants() {
        let probes = NormalizedProbePlan::all();

        assert_eq!(probes.len(), 25);
        assert_eq!(
            probes
                .iter()
                .filter(|probe| probe.case == NormalizedCaseKind::JsonObject)
                .collect::<Vec<_>>(),
            vec![&NormalizedProbePlan::new(
                NormalizedCaseKind::JsonObject,
                SchemaVariant::None,
                ToolChoiceVariant::None,
            )]
        );
        assert_eq!(
            probes
                .iter()
                .filter(|probe| {
                    probe.case == NormalizedCaseKind::OmpRepeatedPrefix
                        && probe.schema_variant == SchemaVariant::CanonicalPermutedEquivalent
                        && probe.tool_choice_variant == ToolChoiceVariant::Function
                })
                .count(),
            1
        );
        assert!(
            probes.iter().any(|probe| {
                probe.case == NormalizedCaseKind::ToolRequiredStream
                    && probe.schema_variant == SchemaVariant::BaselineCurrent
                    && probe.tool_choice_variant == ToolChoiceVariant::Required
            }),
            "streamed tool probes should participate in the schema/tool-choice matrix"
        );
    }

    #[test]
    fn qwen_mlx_tool_normalized_canonical_and_permuted_schema_hashes_capture_equivalence() {
        let baseline = tool_schema_metadata(NormalizedProbePlan::new(
            NormalizedCaseKind::ToolRequired,
            SchemaVariant::BaselineCurrent,
            ToolChoiceVariant::Required,
        ));
        let baseline_permuted = tool_schema_metadata(NormalizedProbePlan::new(
            NormalizedCaseKind::ToolRequired,
            SchemaVariant::BaselinePermutedEquivalent,
            ToolChoiceVariant::Required,
        ));
        let canonical = tool_schema_metadata(NormalizedProbePlan::new(
            NormalizedCaseKind::ToolRequired,
            SchemaVariant::CanonicalCurrent,
            ToolChoiceVariant::Required,
        ));
        let canonical_permuted = tool_schema_metadata(NormalizedProbePlan::new(
            NormalizedCaseKind::ToolRequired,
            SchemaVariant::CanonicalPermutedEquivalent,
            ToolChoiceVariant::Required,
        ));

        assert_ne!(baseline.sha256, baseline_permuted.sha256);
        assert_eq!(canonical.sha256, canonical_permuted.sha256);
        assert_ne!(baseline.sha256, canonical.sha256);
        assert!(canonical.bytes.expect("canonical bytes") > 0);
    }

    #[test]
    fn qwen_mlx_tool_normalized_request_bodies_cover_tool_stream_and_json_with_default_no_thinking_kwargs()
     {
        let lane = lane("name=direct,endpoint=http://127.0.0.1:8080/v1,model=qwen,kind=direct_mlx");

        let tool = probe_request_body(
            &lane,
            NormalizedProbePlan::new(
                NormalizedCaseKind::ToolRequired,
                SchemaVariant::BaselineCurrent,
                ToolChoiceVariant::Required,
            ),
            ProbePrompt::measured(128, 0, None),
        );
        assert_eq!(tool["model"], "qwen");
        assert_eq!(tool["tool_choice"], "required");
        assert_eq!(
            tool["tools"][0]["function"]["name"],
            "record_qwen_tool_probe"
        );
        assert_eq!(tool["chat_template_kwargs"]["enable_thinking"], false);
        assert!(tool.get("stream").is_none());

        let stream = probe_request_body(
            &lane,
            NormalizedProbePlan::new(
                NormalizedCaseKind::ToolRequiredStream,
                SchemaVariant::CanonicalPermutedEquivalent,
                ToolChoiceVariant::Function,
            ),
            ProbePrompt::measured(128, 0, None),
        );
        assert_eq!(stream["stream"], true);
        assert_eq!(stream["stream_options"]["include_usage"], true);
        assert_eq!(stream["chat_template_kwargs"]["enable_thinking"], false);
        assert_eq!(
            stream["tool_choice"],
            json!({"type":"function","function":{"name":"record_qwen_tool_probe"}})
        );
        assert_eq!(
            stream["tools"][0]["function"]["parameters"]["required"],
            json!(["case", "probe_id"])
        );

        let json_body = probe_request_body(
            &lane,
            NormalizedProbePlan::new(
                NormalizedCaseKind::JsonObject,
                SchemaVariant::None,
                ToolChoiceVariant::None,
            ),
            ProbePrompt::measured(128, 0, None),
        );
        assert_eq!(json_body["response_format"]["type"], "json_object");
        assert_eq!(json_body["chat_template_kwargs"]["enable_thinking"], false);
        assert!(
            json_body["messages"]
                .as_array()
                .expect("messages array")
                .iter()
                .any(|message| message["content"]
                    .as_str()
                    .unwrap_or_default()
                    .contains("KIR_QWEN_MLX_TOOL_NORMALIZED_JSON_OBJECT"))
        );

        let synthetic = probe_request_body(
            &lane,
            NormalizedProbePlan::new(
                NormalizedCaseKind::OmpRepeatedPrefix,
                SchemaVariant::BaselinePermutedEquivalent,
                ToolChoiceVariant::Function,
            ),
            ProbePrompt::measured(512, 7, Some(2)),
        );
        let messages = synthetic["messages"].as_array().expect("OMP messages");
        assert_eq!(
            messages
                .iter()
                .map(|message| message["role"].as_str().expect("message role"))
                .collect::<Vec<_>>(),
            ["system", "user", "assistant", "tool", "user"]
        );
        assert_eq!(messages[2]["tool_calls"][0]["type"], "function");
        assert_eq!(
            messages[2]["tool_calls"][0]["function"]["name"],
            "record_qwen_tool_probe"
        );
        assert_eq!(
            messages[3]["tool_call_id"],
            messages[2]["tool_calls"][0]["id"]
        );
        let final_user = messages[4]["content"].as_str().expect("final OMP user");
        assert!(final_user.contains("OMP final delta"));
        assert!(final_user.contains("sample=7 request=2"));
        assert_eq!(
            synthetic["tool_choice"],
            json!({"type":"function","function":{"name":"record_qwen_tool_probe"}})
        );
    }

    #[test]
    fn qwen_mlx_tool_normalized_chat_completions_url_accepts_openai_base_with_or_without_v1() {
        assert_eq!(
            chat_completions_url("http://127.0.0.1:8080/v1"),
            "http://127.0.0.1:8080/v1/chat/completions"
        );
        assert_eq!(
            chat_completions_url("http://127.0.0.1:3000"),
            "http://127.0.0.1:3000/v1/chat/completions"
        );
    }

    #[test]
    fn qwen_mlx_tool_normalized_cache_phase_plan_excludes_warmups_from_measured_samples() {
        let plan = phase_plan(2, 3);
        let measured = plan
            .iter()
            .filter(|run| run.kind == PlannedRunKind::Measured)
            .collect::<Vec<_>>();
        let warmups = plan
            .iter()
            .filter(|run| run.kind == PlannedRunKind::Warmup)
            .collect::<Vec<_>>();

        assert_eq!(measured.len(), 9);
        assert_eq!(warmups.len(), 4);
        assert!(warmups.iter().all(|run| run.phase != CachePhase::Cold));
        assert_eq!(
            measured
                .iter()
                .map(|run| (run.run_mode, run.phase, run.sample_index, run.request_index))
                .collect::<Vec<_>>(),
            vec![
                (RunMode::Sequential, CachePhase::Cold, Some(0), None),
                (RunMode::Sequential, CachePhase::Cold, Some(1), None),
                (RunMode::Sequential, CachePhase::Cold, Some(2), None),
                (
                    RunMode::Sequential,
                    CachePhase::WarmSamePrompt,
                    Some(0),
                    None
                ),
                (
                    RunMode::Sequential,
                    CachePhase::WarmSamePrompt,
                    Some(1),
                    None
                ),
                (
                    RunMode::Sequential,
                    CachePhase::WarmSamePrompt,
                    Some(2),
                    None
                ),
                (
                    RunMode::Sequential,
                    CachePhase::WarmSameToolSchema,
                    Some(0),
                    None
                ),
                (
                    RunMode::Sequential,
                    CachePhase::WarmSameToolSchema,
                    Some(1),
                    None
                ),
                (
                    RunMode::Sequential,
                    CachePhase::WarmSameToolSchema,
                    Some(2),
                    None
                ),
            ]
        );
    }

    #[test]
    fn qwen_mlx_tool_normalized_concurrent_phase_plan_preserves_sample_and_request_indexes() {
        assert_eq!(effective_concurrent_samples(1, 2, 0), 0);
        assert_eq!(effective_concurrent_samples(3, 2, 0), 2);
        assert_eq!(effective_concurrent_samples(1, 2, 4), 4);

        let plan = concurrent_phase_plan(3, 2);

        assert_eq!(plan.len(), 18);
        assert!(plan.iter().all(|run| run.kind == PlannedRunKind::Measured));
        assert!(plan.iter().all(|run| run.run_mode == RunMode::Concurrent));
        assert_eq!(
            plan.iter()
                .filter(|run| run.phase == CachePhase::Cold)
                .map(|run| (run.sample_index, run.request_index))
                .collect::<Vec<_>>(),
            vec![
                (Some(0), Some(0)),
                (Some(0), Some(1)),
                (Some(0), Some(2)),
                (Some(1), Some(0)),
                (Some(1), Some(1)),
                (Some(1), Some(2)),
            ]
        );
    }

    #[test]
    fn qwen_mlx_tool_normalized_focused_agentic_gate_uses_small_probe_plan() {
        let suite = parse_probe_suite_flag(&args(&["--focused-agentic-gate"]));
        let probes = suite.probes();

        assert_eq!(suite.name(), "focused_agentic_gate");
        assert_eq!(
            probes,
            vec![
                NormalizedProbePlan::new(
                    NormalizedCaseKind::ToolRequired,
                    SchemaVariant::CanonicalCurrent,
                    ToolChoiceVariant::Required,
                ),
                NormalizedProbePlan::new(
                    NormalizedCaseKind::ToolRequiredStream,
                    SchemaVariant::CanonicalCurrent,
                    ToolChoiceVariant::Required,
                ),
                NormalizedProbePlan::new(
                    NormalizedCaseKind::OmpRepeatedPrefix,
                    SchemaVariant::CanonicalCurrent,
                    ToolChoiceVariant::Required,
                ),
            ]
        );

        let lane = lane("name=direct,endpoint=http://127.0.0.1:8080/v1,model=qwen");
        let report = NormalizedLaneReport::dry_run(
            &lane,
            NormalizedRunConfig::new(0, 1, 128, 1, 0),
            None,
            &probes,
        );
        assert_eq!(report.samples.len(), 9);
        assert!(
            report
                .samples
                .iter()
                .all(|sample| sample.case != "json_object")
        );
    }

    #[test]
    fn qwen_mlx_tool_normalized_aggregate_summary_rows_group_by_lane_case_phase_and_run_mode() {
        let lane_a = lane("name=a,endpoint=http://127.0.0.1:8080/v1,model=qwen-a");
        let lane_b = lane("name=b,endpoint=http://127.0.0.1:8081/v1,model=qwen-b");
        let mut report_a = NormalizedLaneReport::planned(&lane_a, 0, 0, None);
        let mut report_b = NormalizedLaneReport::planned(&lane_b, 0, 0, None);

        report_a.samples = vec![
            passed_sample(
                NormalizedCaseKind::ToolRequired,
                CachePhase::Cold,
                RunMode::Sequential,
                0,
                None,
                100,
                10,
            ),
            passed_sample(
                NormalizedCaseKind::ToolRequired,
                CachePhase::Cold,
                RunMode::Sequential,
                1,
                None,
                200,
                20,
            ),
            passed_sample(
                NormalizedCaseKind::ToolRequired,
                CachePhase::Cold,
                RunMode::Sequential,
                2,
                None,
                400,
                30,
            ),
            failed_summary_sample(
                NormalizedCaseKind::ToolRequired,
                CachePhase::Cold,
                RunMode::Sequential,
                3,
                None,
            ),
        ];
        report_b.samples = vec![passed_sample(
            NormalizedCaseKind::ToolRequired,
            CachePhase::Cold,
            RunMode::Sequential,
            0,
            None,
            50,
            5,
        )];

        let probes = NormalizedProbePlan::all();
        let summary = aggregate_normalized_summary(&[report_a, report_b], &probes);
        let a_row = summary
            .iter()
            .find(|row| {
                row.lane == "a"
                    && row.case == "tool_required"
                    && row.cache_phase == "cold"
                    && row.run_mode == "sequential"
            })
            .expect("lane a summary row");

        assert_eq!(a_row.pass_count, 3);
        assert_eq!(a_row.schema_variant, "baseline_current");
        assert_eq!(a_row.tool_choice_variant, "required");
        assert_eq!(a_row.fail_count, 1);
        assert_eq!(a_row.p50_latency_ms, Some(200));
        assert_eq!(a_row.p95_latency_ms, Some(400));
        assert_eq!(a_row.avg_cached_tokens, Some(20.0));
        assert_eq!(a_row.avg_prompt_tokens, Some(1000.0));
        assert_eq!(a_row.avg_completion_tokens, Some(10.0));
        assert_eq!(a_row.avg_total_tokens, Some(1010.0));
        assert_eq!(a_row.fastest_lane.as_deref(), Some("b"));
    }

    #[test]
    fn qwen_mlx_tool_normalized_agentic_gate_reports_warm_stream_cache_and_lane_deltas() {
        let lane_a = lane("name=direct,endpoint=http://127.0.0.1:8080/v1,model=qwen-a");
        let lane_b = lane("name=proxy,endpoint=http://127.0.0.1:3000,model=qwen-b");
        let mut report_a = NormalizedLaneReport::planned(&lane_a, 0, 0, None);
        let mut report_b = NormalizedLaneReport::planned(&lane_b, 0, 0, None);

        let mut direct = passed_sample(
            NormalizedCaseKind::ToolRequiredStream,
            CachePhase::WarmSamePrompt,
            RunMode::Sequential,
            0,
            None,
            1000,
            64,
        );
        direct.schema_variant = SchemaVariant::CanonicalCurrent.name();
        direct.schema_canonicalized = true;
        direct.stream_timing = StreamTimingReport {
            first_byte_latency_ms: Some(120),
            first_sse_data_latency_ms: Some(125),
            first_content_delta_latency_ms: None,
            first_tool_delta_latency_ms: Some(700),
            first_semantic_delta_latency_ms: Some(700),
        };
        direct.tokens_per_second = Some(33.0);
        report_a.samples = vec![direct];

        let mut proxy = passed_sample(
            NormalizedCaseKind::ToolRequiredStream,
            CachePhase::WarmSamePrompt,
            RunMode::Sequential,
            0,
            None,
            1125,
            60,
        );
        proxy.schema_variant = SchemaVariant::CanonicalCurrent.name();
        proxy.schema_canonicalized = true;
        proxy.stream_timing = StreamTimingReport {
            first_byte_latency_ms: Some(150),
            first_sse_data_latency_ms: Some(155),
            first_content_delta_latency_ms: None,
            first_tool_delta_latency_ms: Some(760),
            first_semantic_delta_latency_ms: Some(760),
        };
        proxy.tokens_per_second = Some(31.0);
        report_b.samples = vec![proxy];

        let gate = agentic_gate_report(&[report_a, report_b]);
        let row = gate
            .rows
            .iter()
            .find(|row| {
                row.case == "tool_required_stream"
                    && row.cache_phase == "warm_same_prompt"
                    && row.run_mode == "sequential"
            })
            .expect("warm stream gate row");

        assert_eq!(gate.status, "reported");
        assert_eq!(row.fastest_lane.as_deref(), Some("direct"));
        assert_eq!(row.lanes[0].p50_first_byte_latency_ms, Some(120));
        assert_eq!(row.lanes[0].p50_first_semantic_delta_latency_ms, Some(700));
        assert_eq!(row.lanes[0].p50_first_tool_delta_latency_ms, Some(700));
        assert_eq!(row.lanes[0].avg_cached_tokens, Some(64.0));
        assert_eq!(row.lanes[1].latency_delta_vs_fastest_ms, Some(125));
    }

    #[test]
    fn qwen_mlx_tool_normalized_cached_tokens_usage_parses_present_null_and_missing_shapes() {
        let present = usage_from_value(Some(&json!({
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "prompt_tokens_details": {"cached_tokens": 7}
        })));
        assert_eq!(present.cached_tokens, Some(7));
        assert_eq!(present.cached_tokens_status, Some("present"));

        let null = usage_from_value(Some(&json!({
            "prompt_tokens_details": {"cached_tokens": null}
        })));
        assert_eq!(null.cached_tokens, None);
        assert_eq!(null.cached_tokens_status, Some("null"));

        let missing = usage_from_value(Some(&json!({
            "prompt_tokens": 10
        })));
        assert_eq!(missing.cached_tokens, None);
        assert_eq!(missing.cached_tokens_status, Some("missing"));
    }

    #[test]
    fn qwen_mlx_tool_normalized_stream_usage_merges_across_frames() {
        let mut assembly = StreamAssembly::default();
        apply_sse_frame(
            &json!({
                "choices": [{"delta": {"role": "assistant"}, "finish_reason": null}],
                "usage": {
                    "prompt_tokens": 100,
                    "prompt_tokens_details": {"cached_tokens": 80}
                }
            }),
            &mut assembly,
        );
        apply_sse_frame(
            &json!({
                "choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "record_qwen_tool_probe", "arguments": "{}"}}]}, "finish_reason": "tool_calls"}],
                "usage": {"completion_tokens": 12}
            }),
            &mut assembly,
        );

        assert_eq!(assembly.usage.prompt_tokens, Some(100));
        assert_eq!(assembly.usage.cached_tokens_status, Some("present"));
        assert_eq!(assembly.usage.cached_tokens, Some(80));
        assert_eq!(assembly.usage.completion_tokens, Some(12));
        assert_eq!(assembly.usage.total_tokens, Some(112));
    }

    #[test]
    fn qwen_mlx_tool_normalized_validation_classifies_buffered_tool_json_and_stream_responses() {
        let tool = json!({
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "tool_calls": [{
                        "function": {
                            "name": "record_qwen_tool_probe",
                            "arguments": "{\"probe_id\":\"KIR_QWEN_MLX_TOOL_NORMALIZED_TOOL_REQUIRED\",\"case\":\"tool_required\"}"
                        }
                    }]
                }
            }]
        });
        assert_eq!(
            validate_buffered_probe(
                NormalizedCaseKind::ToolRequired,
                &tool,
                NormalizedCaseKind::ToolRequired.probe_id()
            ),
            Ok(())
        );

        let json_response = json!({
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "content": "{\"probe_id\":\"KIR_QWEN_MLX_TOOL_NORMALIZED_JSON_OBJECT\",\"case\":\"json_object\"}"
                }
            }]
        });
        assert_eq!(
            validate_buffered_probe(
                NormalizedCaseKind::JsonObject,
                &json_response,
                NormalizedCaseKind::JsonObject.probe_id()
            ),
            Ok(())
        );

        let assembly = StreamAssembly {
            tool_name: Some("record_qwen_tool_probe".to_owned()),
            tool_arguments:
                "{\"probe_id\":\"KIR_QWEN_MLX_TOOL_NORMALIZED_TOOL_REQUIRED_STREAM\",\"case\":\"tool_required_stream\"}".to_owned(),
            finish_reason: Some("tool_calls".to_owned()),
            ..StreamAssembly::default()
        };
        assert_eq!(
            validate_streaming_probe(
                NormalizedCaseKind::ToolRequiredStream,
                &assembly,
                NormalizedCaseKind::ToolRequiredStream.probe_id()
            ),
            Ok(())
        );
    }

    fn passed_sample(
        case: NormalizedCaseKind,
        phase: CachePhase,
        run_mode: RunMode,
        sample_index: usize,
        request_index: Option<usize>,
        latency_ms: u128,
        cached_tokens: u64,
    ) -> NormalizedSampleReport {
        let mut sample = NormalizedSampleReport::base(
            NormalizedProbePlan::new(
                case,
                SchemaVariant::BaselineCurrent,
                ToolChoiceVariant::Required,
            ),
            phase,
            run_mode,
            sample_index,
            request_index,
            false,
            128,
        );
        sample.status = "passed".to_owned();
        sample.classification = "passed".to_owned();
        sample.latency_ms = Some(latency_ms);
        sample.prompt_tokens = Some(1000);
        sample.completion_tokens = Some(10);
        sample.total_tokens = Some(1010);
        sample.cached_tokens_status = "present";
        sample.cached_tokens = Some(cached_tokens);
        sample
    }

    fn failed_summary_sample(
        case: NormalizedCaseKind,
        phase: CachePhase,
        run_mode: RunMode,
        sample_index: usize,
        request_index: Option<usize>,
    ) -> NormalizedSampleReport {
        let mut sample = NormalizedSampleReport::base(
            NormalizedProbePlan::new(
                case,
                SchemaVariant::BaselineCurrent,
                ToolChoiceVariant::Required,
            ),
            phase,
            run_mode,
            sample_index,
            request_index,
            false,
            128,
        );
        sample.status = "failed".to_owned();
        sample.classification = "http_status_failed".to_owned();
        sample.latency_ms = Some(900);
        sample
    }
}
