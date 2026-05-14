use super::{
    DEFAULT_CONNECT_TIMEOUT_MS, DEFAULT_TIMEOUT_MS, HardwareReport, ModelIdentityReport,
    StreamAssembly, StreamTimingReport, StreamTimingTracker, cli::flag_values, consume_sse_buffer,
    load_model_identity, load_qwen_tokenizer, normalize_endpoint, unix_now_ms, usage_from_value,
};
use crate::{DEFAULT_MODEL_ID, MlxToolParserMode, flag_value, has_flag};
use anyhow::{Context, anyhow};
use futures::StreamExt;
use futures::future::join_all;
use llm_api::canonicalize_json_value;
use llm_tokenizer::HuggingFaceTokenizer;
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
const QWEN_MLX_PREFILL_135K_PROFILE: &str = "qwen-mlx-prefill-135k";
const QWEN_MLX_STABLE_PREFIX_PROFILE: &str = "qwen-mlx-stable-prefix";
const PROFILE_PROXY_MODEL_ID: &str = "local-qwen36-mlx";
const PROFILE_CACHE_BYTES_1G: u64 = 1_073_741_824;
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
    let admin_token = flag_value(args, "--admin-token").map(str::to_owned);
    let sweep_profile = parse_sweep_profile_flag(args)?;
    let probe_suite = parse_probe_suite_flag(args, sweep_profile)?;
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
        tool_required_stream: NormalizedToolRequiredStreamTimingReport::dry_run(&lane_reports),
        lanes: lane_reports,
        hardware: HardwareReport::detect(),
        comparison: NormalizedComparisonReport::dry_run(),
        agentic_gate: NormalizedAgenticGateReport::dry_run(),
        prefill_sweep: NormalizedPrefillSweepReport::dry_run(),
        stable_prefix: NormalizedStablePrefixReport::dry_run(),
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
        let tokenizer = if sweep_profile_requires_exact_token_prompt(sweep_profile) {
            Some(load_lane_tokenizer(lane)?)
        } else {
            None
        };
        run_lane(
            &client,
            lane,
            lane_report,
            run_config,
            &probes,
            admin_token.as_deref(),
            tokenizer.as_ref(),
        )
        .await;
    }
    report.summary = aggregate_normalized_summary(&report.lanes, &probes);
    report.tool_required_stream = tool_required_stream_timing_report(&report.lanes);
    report.comparison = compare_normalized_lanes(&report.lanes, &probes);
    report.agentic_gate = agentic_gate_report(&report.lanes);
    report.prefill_sweep = prefill_sweep_report(&report.lanes, &probes);
    report.stable_prefix = stable_prefix_report(&report.lanes, &probes);
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
  --sweep-profile <name>        Built-in lane matrix: qwen-mlx-cache-prefill, qwen-mlx-prefill-135k, or qwen-mlx-stable-prefix (requires --snapshot)
  --probe-suite <name>          Probe suite: full-matrix, focused-agentic-gate, prefill-sweep-135k, or stable-agent-prefix
  --snapshot <path>             Raw Hugging Face snapshot path for built-in sweep profiles
  --lane <spec>                 Lane: name=<id>,endpoint=<url>,model=<id>[,launched_model_id=<id-or-path>][,snapshot=<path>][,kind=direct_mlx|kir_ai_proxy|other][,model_addressing=loaded_model_id|default_model|server_default|custom][,template=qwen-no-thinking|sidecar-chat-template-args|none][,tool_parser=auto|json|qwen-xml][,mlx_prompt_cache_size=default|<n>][,mlx_prompt_cache_bytes=unset|<n>][,mlx_prefill_step_size=default|<n>][,mlx_prompt_concurrency=default|<n>][,mlx_decode_concurrency=default|<n>]
  --warmups <n>                 Warmup requests for warm phases [default: 1]
  --samples <n>                 Measured samples per case and phase [default: 1]
  --context-tokens <n>          Stable long-context prompt target [default: 135000]
  --concurrent-requests <n>     Requests to issue together during the concurrent pass [default: 1]
  --concurrent-samples <n>      Concurrent sample batches per case and phase; 0 disables unless concurrent requests > 1 [default: 0]
  --focused-agentic-gate        Compatibility alias for --probe-suite focused-agentic-gate
  --output <path>               Write the trace JSON to a file as well as stdout
  --admin-token <token>         Optional bearer token for lane /admin/metrics snapshots
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

fn parse_probe_suite_flag(
    args: &[String],
    sweep_profile: Option<NormalizedSweepProfile>,
) -> anyhow::Result<NormalizedProbeSuite> {
    let suites = flag_values(args, "--probe-suite");
    let focused_agentic_gate = has_flag(args, "--focused-agentic-gate");
    let explicit = match suites.as_slice() {
        [] => None,
        [suite] => Some(NormalizedProbeSuite::parse(suite)?),
        _ => anyhow::bail!("--probe-suite may only be provided once"),
    };
    if focused_agentic_gate && explicit.is_some() {
        anyhow::bail!("--focused-agentic-gate cannot be combined with --probe-suite");
    }
    if focused_agentic_gate {
        return Ok(NormalizedProbeSuite::FocusedAgenticGate);
    }
    Ok(explicit.unwrap_or_else(|| {
        sweep_profile
            .map(NormalizedSweepProfile::default_probe_suite)
            .unwrap_or(NormalizedProbeSuite::FullMatrix)
    }))
}

fn expand_sweep_profile(
    profile: NormalizedSweepProfile,
    args: &[String],
) -> anyhow::Result<Vec<NormalizedLaneConfig>> {
    let snapshot = required_profile_snapshot(args)?;
    Ok(match profile {
        NormalizedSweepProfile::QwenMlxCachePrefill => qwen_mlx_cache_prefill_lanes(snapshot),
        NormalizedSweepProfile::QwenMlxPrefill135k => qwen_mlx_prefill_135k_lanes(snapshot),
        NormalizedSweepProfile::QwenMlxStablePrefix => qwen_mlx_stable_prefix_lanes(snapshot),
    })
}

fn required_profile_snapshot(args: &[String]) -> anyhow::Result<&str> {
    let snapshots = flag_values(args, "--snapshot");
    match snapshots.as_slice() {
        [snapshot] => Ok(snapshot),
        [] => anyhow::bail!("--sweep-profile requires --snapshot <path>"),
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
            tool_parser: MlxToolParserMode::Auto,
            mlx_lm_settings: MlxLmSettings::default(),
        },
    ]
}

fn qwen_mlx_prefill_135k_lanes(snapshot: &str) -> Vec<NormalizedLaneConfig> {
    let steps = [
        ("default", DefaultOrU64::Default),
        ("512", DefaultOrU64::Value(512)),
        ("1024", DefaultOrU64::Value(1024)),
        ("2048", DefaultOrU64::Value(2048)),
        ("4096", DefaultOrU64::Value(4096)),
        ("8192", DefaultOrU64::Value(8192)),
    ];
    let mut lanes = Vec::with_capacity(steps.len() * 2);
    for (index, (label, prefill_step_size)) in steps.into_iter().enumerate() {
        let port_offset = index as u16;
        let settings = MlxLmSettings {
            prefill_step_size,
            ..MlxLmSettings::default()
        };
        lanes.push(profile_direct_lane(
            &format!("mlx-prefill-{label}"),
            8080 + port_offset,
            settings,
            snapshot,
        ));
        lanes.push(profile_proxy_lane(
            &format!("kir-prefill-{label}"),
            3000 + port_offset,
            settings,
            snapshot,
        ));
    }
    lanes
}

fn qwen_mlx_stable_prefix_lanes(snapshot: &str) -> Vec<NormalizedLaneConfig> {
    vec![
        NormalizedLaneConfig {
            name: "mlx-stable-prefix".to_owned(),
            endpoint: "http://127.0.0.1:8080/v1".to_owned(),
            declared_model_id: snapshot.to_owned(),
            launched_model_id: Some(snapshot.to_owned()),
            snapshot_path: Some(PathBuf::from(snapshot)),
            kind: NormalizedLaneKind::DirectMlx,
            model_addressing: NormalizedModelAddressing::ServerDefault,
            template: NormalizedTemplatePolicy::QwenNoThinking,
            tool_parser: MlxToolParserMode::Auto,
            mlx_lm_settings: MlxLmSettings::default(),
        },
        NormalizedLaneConfig {
            name: "kir-stable-prefix".to_owned(),
            endpoint: "http://127.0.0.1:3000".to_owned(),
            declared_model_id: PROFILE_PROXY_MODEL_ID.to_owned(),
            launched_model_id: Some(snapshot.to_owned()),
            snapshot_path: Some(PathBuf::from(snapshot)),
            kind: NormalizedLaneKind::KirAiProxy,
            model_addressing: NormalizedModelAddressing::DefaultModel,
            template: NormalizedTemplatePolicy::SidecarChatTemplateArgs,
            tool_parser: MlxToolParserMode::Auto,
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
        tool_parser: MlxToolParserMode::Auto,
        mlx_lm_settings,
    }
}

fn profile_proxy_lane(
    name: &str,
    port: u16,
    mlx_lm_settings: MlxLmSettings,
    snapshot: &str,
) -> NormalizedLaneConfig {
    NormalizedLaneConfig {
        name: name.to_owned(),
        endpoint: format!("http://127.0.0.1:{port}"),
        declared_model_id: PROFILE_PROXY_MODEL_ID.to_owned(),
        launched_model_id: Some(snapshot.to_owned()),
        snapshot_path: Some(PathBuf::from(snapshot)),
        kind: NormalizedLaneKind::KirAiProxy,
        model_addressing: NormalizedModelAddressing::DefaultModel,
        template: NormalizedTemplatePolicy::SidecarChatTemplateArgs,
        tool_parser: MlxToolParserMode::Auto,
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
    let tool_parser = values
        .remove("tool_parser")
        .map(|value| parse_mlx_tool_parser_mode(&value))
        .transpose()?
        .unwrap_or(MlxToolParserMode::Auto);
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
        tool_parser,
        mlx_lm_settings,
    })
}

fn parse_mlx_tool_parser_mode(value: &str) -> anyhow::Result<MlxToolParserMode> {
    MlxToolParserMode::parse(value)
        .ok_or_else(|| anyhow!("unknown tool_parser `{value}`; expected auto, json, or qwen-xml"))
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

fn sweep_profile_requires_exact_token_prompt(profile: Option<NormalizedSweepProfile>) -> bool {
    matches!(profile, Some(NormalizedSweepProfile::QwenMlxPrefill135k))
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

async fn run_lane(
    client: &reqwest::Client,
    lane: &NormalizedLaneConfig,
    lane_report: &mut NormalizedLaneReport,
    run_config: NormalizedRunConfig,
    probes: &[NormalizedProbePlan],
    admin_token: Option<&str>,
    prompt_tokenizer: Option<&HuggingFaceTokenizer>,
) {
    if should_capture_admin_metrics(lane) {
        lane_report
            .admin_metrics
            .record_before(capture_normalized_admin_metrics(client, lane, admin_token).await);
    }
    for &probe in probes {
        for planned in phase_plan(run_config.warmups, run_config.samples) {
            match planned.kind {
                PlannedRunKind::Warmup => {
                    let result =
                        execute_probe(client, lane, probe, planned, prompt_tokenizer, run_config)
                            .await;
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
                        execute_probe(client, lane, probe, planned, prompt_tokenizer, run_config)
                            .await;
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
                    execute_probe(client, lane, probe, planned, prompt_tokenizer, run_config)
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
        lane_report
            .admin_metrics
            .record_after(capture_normalized_admin_metrics(client, lane, admin_token).await);
    }
}

async fn execute_probe(
    client: &reqwest::Client,
    lane: &NormalizedLaneConfig,
    probe: NormalizedProbePlan,
    planned: PlannedRun,
    prompt_tokenizer: Option<&HuggingFaceTokenizer>,
    run_config: NormalizedRunConfig,
) -> NormalizedSampleReport {
    let prompt = match planned.prompt(run_config.context_tokens, probe.case, prompt_tokenizer) {
        Ok(prompt) => prompt,
        Err(err) => {
            let context = SampleContext {
                probe,
                phase: planned.phase,
                run_mode: planned.run_mode,
                sample_index: planned.sample_index.unwrap_or_default(),
                request_index: planned.request_index,
                planned_prompt_tokens: 0,
                prewarmed: planned.phase.warms_before_samples() && run_config.warmups > 0,
                expected_probe_id: probe.case.probe_id().to_owned(),
                expected_marker: None,
            };
            return failed_sample(
                context,
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
    let context = SampleContext {
        probe,
        phase: planned.phase,
        run_mode: planned.run_mode,
        sample_index: planned.sample_index.unwrap_or_default(),
        request_index: planned.request_index,
        planned_prompt_tokens: prompt.planned_prompt_tokens(),
        prewarmed: planned.phase.warms_before_samples() && run_config.warmups > 0,
        expected_probe_id,
        expected_marker,
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
    sample.classification = classification.into();
    sample.latency_ms = Some(latency.as_millis());
    sample.stream_timing = stream_timing;
    sample.http_status = http_status;
    sample.request_id = request_id_from_response_headers(&response_headers);
    sample.response_headers = response_headers;
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

fn prefill_sweep_report(
    lanes: &[NormalizedLaneReport],
    probes: &[NormalizedProbePlan],
) -> NormalizedPrefillSweepReport {
    let mut rows = Vec::new();
    for &probe in probes {
        for phase in CachePhase::all() {
            for run_mode in [RunMode::Sequential, RunMode::Concurrent] {
                let mut lane_metrics = lanes
                    .iter()
                    .filter_map(|lane| prefill_sweep_lane_metric(lane, probe, phase, run_mode))
                    .collect::<Vec<_>>();
                if lane_metrics.is_empty() {
                    continue;
                }
                let fastest_latency = lane_metrics
                    .iter()
                    .filter(|metric| metric.valid)
                    .filter_map(|metric| metric.p50_first_semantic_delta_latency_ms)
                    .min();
                let fastest_lane = fastest_latency.and_then(|fastest| {
                    lane_metrics
                        .iter()
                        .find(|metric| {
                            metric.valid
                                && metric.p50_first_semantic_delta_latency_ms == Some(fastest)
                        })
                        .map(|metric| metric.lane.clone())
                });
                if let Some(fastest) = fastest_latency {
                    for metric in &mut lane_metrics {
                        metric.latency_delta_vs_fastest_ms = metric
                            .p50_first_semantic_delta_latency_ms
                            .map(|latency| latency.saturating_sub(fastest));
                    }
                }
                lane_metrics.sort_by(|left, right| {
                    prefill_metric_sort_key(left).cmp(&prefill_metric_sort_key(right))
                });
                rows.push(NormalizedPrefillSweepRow {
                    case: probe.case.name(),
                    schema_variant: probe.schema_variant.name(),
                    tool_choice_variant: probe.tool_choice_variant.name(),
                    cache_phase: phase.name(),
                    run_mode: run_mode.name(),
                    fastest_lane,
                    lanes: lane_metrics,
                });
            }
        }
    }
    NormalizedPrefillSweepReport {
        status: if rows.is_empty() {
            "no_samples"
        } else {
            "reported"
        }
        .to_owned(),
        rows,
    }
}

fn prefill_sweep_lane_metric(
    lane: &NormalizedLaneReport,
    probe: NormalizedProbePlan,
    phase: CachePhase,
    run_mode: RunMode,
) -> Option<NormalizedPrefillSweepLaneMetric> {
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
        return None;
    }
    let passed = samples
        .iter()
        .copied()
        .filter(|sample| sample.status == "passed")
        .collect::<Vec<_>>();
    let pass_count = passed.len();
    let fail_count = samples
        .iter()
        .filter(|sample| sample.status == "failed")
        .count();
    let p50_first_semantic_delta_latency_ms = percentile_for_samples(&passed, |sample| {
        sample.stream_timing.first_semantic_delta_latency_ms
    });
    let p50_first_tool_delta_latency_ms = percentile_for_samples(&passed, |sample| {
        sample.stream_timing.first_tool_delta_latency_ms
    });
    let stream_stalled_requests_delta =
        admin_counter_delta(&lane.admin_metrics, &["stream_stalled_requests"]);
    let no_progress_failures_delta =
        admin_counter_delta(&lane.admin_metrics, &["no_progress_failures"]);
    let mut invalid_reasons = Vec::new();
    if fail_count > 0 {
        invalid_reasons.push("sample_failed".to_owned());
    }
    if p50_first_semantic_delta_latency_ms.is_none() {
        invalid_reasons.push("missing_ttft".to_owned());
        invalid_reasons.push("missing_stream_delta".to_owned());
    }
    if probe.case.requires_tool_delta() && p50_first_tool_delta_latency_ms.is_none() {
        invalid_reasons.push("missing_tool_delta".to_owned());
    }
    if stream_stalled_requests_delta.is_some_and(|delta| delta > 0) {
        invalid_reasons.push("admin_stalled_request_delta".to_owned());
    }
    if no_progress_failures_delta.is_some_and(|delta| delta > 0) {
        invalid_reasons.push("admin_no_progress_delta".to_owned());
    }
    invalid_reasons.sort();
    invalid_reasons.dedup();

    Some(NormalizedPrefillSweepLaneMetric {
        lane: lane.name.clone(),
        lane_kind: lane.kind,
        prefill_step_size: lane.mlx_lm_settings.prefill_step_size,
        valid: invalid_reasons.is_empty(),
        invalid_reasons,
        sample_count: samples.len(),
        pass_count,
        fail_count,
        p50_first_response_byte_latency_ms: percentile_for_samples(&passed, |sample| {
            sample.stream_timing.first_byte_latency_ms
        }),
        p50_first_parsed_sse_chunk_latency_ms: percentile_for_samples(&passed, |sample| {
            sample.stream_timing.first_sse_data_latency_ms
        }),
        p50_first_semantic_delta_latency_ms,
        p50_first_tool_delta_latency_ms,
        p50_elapsed_latency_ms: percentile_for_samples(&passed, |sample| sample.latency_ms),
        latency_delta_vs_fastest_ms: None,
        avg_tokens_per_second: average_f64(
            passed.iter().filter_map(|sample| sample.tokens_per_second),
        ),
        avg_cached_tokens: average_u64(passed.iter().filter_map(|sample| sample.cached_tokens)),
        avg_prompt_tokens: average_u64(passed.iter().filter_map(|sample| sample.prompt_tokens)),
        avg_completion_tokens: average_u64(
            passed.iter().filter_map(|sample| sample.completion_tokens),
        ),
        avg_total_tokens: average_u64(passed.iter().filter_map(|sample| sample.total_tokens)),
        response_headers: samples
            .iter()
            .filter_map(|sample| sample.response_headers.clone())
            .collect(),
        admin_mlx_upstream_timing: normalized_prefill_admin_mlx_timing(&lane.admin_metrics),
        process_rss_bytes_after: admin_counter_after(&lane.admin_metrics, &["process_rss_bytes"]),
        stream_stalled_requests_delta,
        no_progress_failures_delta,
    })
}

fn prefill_metric_sort_key(metric: &NormalizedPrefillSweepLaneMetric) -> (bool, u128, String) {
    (
        !metric.valid,
        metric
            .p50_first_semantic_delta_latency_ms
            .unwrap_or(u128::MAX),
        metric.lane.clone(),
    )
}

fn stable_prefix_report(
    lanes: &[NormalizedLaneReport],
    probes: &[NormalizedProbePlan],
) -> NormalizedStablePrefixReport {
    let mut rows = Vec::new();
    for &probe in probes {
        for phase in CachePhase::all() {
            for run_mode in [RunMode::Sequential, RunMode::Concurrent] {
                let mut lane_metrics = lanes
                    .iter()
                    .filter_map(|lane| stable_prefix_lane_metric(lane, probe, phase, run_mode))
                    .collect::<Vec<_>>();
                if lane_metrics.is_empty() {
                    continue;
                }
                let fastest_latency = lane_metrics
                    .iter()
                    .filter_map(|metric| metric.p50_elapsed_latency_ms)
                    .min();
                let fastest_lane = fastest_latency.and_then(|fastest| {
                    lane_metrics
                        .iter()
                        .find(|metric| metric.p50_elapsed_latency_ms == Some(fastest))
                        .map(|metric| metric.lane.clone())
                });
                if let Some(fastest) = fastest_latency {
                    for metric in &mut lane_metrics {
                        metric.latency_delta_vs_fastest_ms = metric
                            .p50_elapsed_latency_ms
                            .map(|latency| latency.saturating_sub(fastest));
                    }
                }
                lane_metrics.sort_by(|left, right| {
                    stable_prefix_metric_sort_key(left).cmp(&stable_prefix_metric_sort_key(right))
                });
                rows.push(NormalizedStablePrefixRow {
                    case: probe.case.name(),
                    schema_variant: probe.schema_variant.name(),
                    tool_choice_variant: probe.tool_choice_variant.name(),
                    cache_phase: phase.name(),
                    run_mode: run_mode.name(),
                    fastest_lane,
                    lanes: lane_metrics,
                });
            }
        }
    }
    NormalizedStablePrefixReport {
        status: if rows.is_empty() {
            "no_samples"
        } else {
            "reported"
        }
        .to_owned(),
        rows,
    }
}

fn stable_prefix_lane_metric(
    lane: &NormalizedLaneReport,
    probe: NormalizedProbePlan,
    phase: CachePhase,
    run_mode: RunMode,
) -> Option<NormalizedStablePrefixLaneMetric> {
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
        return None;
    }
    let passed = samples
        .iter()
        .copied()
        .filter(|sample| sample.status == "passed")
        .collect::<Vec<_>>();
    Some(NormalizedStablePrefixLaneMetric {
        lane: lane.name.clone(),
        lane_kind: lane.kind,
        sample_count: samples.len(),
        pass_count: passed.len(),
        fail_count: samples
            .iter()
            .filter(|sample| sample.status == "failed")
            .count(),
        p50_first_semantic_delta_latency_ms: percentile_for_samples(&passed, |sample| {
            sample.stream_timing.first_semantic_delta_latency_ms
        }),
        p50_first_tool_delta_latency_ms: percentile_for_samples(&passed, |sample| {
            sample.stream_timing.first_tool_delta_latency_ms
        }),
        p50_elapsed_latency_ms: percentile_for_samples(&passed, |sample| sample.latency_ms),
        latency_delta_vs_fastest_ms: None,
        avg_prompt_tokens: average_u64(passed.iter().filter_map(|sample| sample.prompt_tokens)),
        avg_cached_tokens: average_u64(passed.iter().filter_map(|sample| sample.cached_tokens)),
        avg_uncached_tokens: average_u64(
            passed
                .iter()
                .filter_map(|sample| sample_uncached_tokens(sample)),
        ),
        cache_status_counts: cache_status_counts(&passed),
        request_cache_observations: matching_request_cache_observations(lane, &samples),
    })
}

fn stable_prefix_metric_sort_key(metric: &NormalizedStablePrefixLaneMetric) -> (u128, String) {
    (
        metric.p50_elapsed_latency_ms.unwrap_or(u128::MAX),
        metric.lane.clone(),
    )
}

fn cache_status_counts(samples: &[&NormalizedSampleReport]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for sample in samples {
        *counts
            .entry(cache_status_from_sample(sample).to_owned())
            .or_insert(0) += 1;
    }
    counts
}

fn cache_status_from_sample(sample: &NormalizedSampleReport) -> &'static str {
    if sample.cached_tokens_status != "present" {
        return "unknown";
    }
    match (sample.prompt_tokens, sample.cached_tokens) {
        (_, Some(0)) => "miss",
        (Some(prompt), Some(cached)) if cached >= prompt => "hit",
        (Some(_), Some(_)) => "partial",
        _ => "unknown",
    }
}

fn sample_uncached_tokens(sample: &NormalizedSampleReport) -> Option<u64> {
    Some(sample.prompt_tokens?.saturating_sub(sample.cached_tokens?))
}

fn matching_request_cache_observations(
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

fn stable_prefix_request_cache_observation(
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

fn normalized_prefill_admin_mlx_timing(
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

fn admin_counter_delta(capture: &NormalizedAdminMetricsCapture, path: &[&str]) -> Option<i64> {
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

fn admin_counter_after(capture: &NormalizedAdminMetricsCapture, path: &[&str]) -> Option<u64> {
    capture
        .after
        .as_ref()
        .and_then(|value| value_path(value, path))
        .and_then(Value::as_u64)
}

fn value_i64(value: &Value) -> Option<i64> {
    value
        .as_i64()
        .or_else(|| value.as_u64().and_then(|value| i64::try_from(value).ok()))
}

fn tool_required_stream_timing_report(
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
    NormalizedToolRequiredStreamTimingReport {
        status: status.to_owned(),
        lanes: lane_reports,
    }
}

fn tool_required_stream_lane_timing_report(
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
    }
}

fn normalized_tool_stream_admin_metrics(
    capture: &NormalizedAdminMetricsCapture,
) -> Option<NormalizedToolRequiredStreamAdminMetrics> {
    let after = capture.after.as_ref()?;
    Some(NormalizedToolRequiredStreamAdminMetrics {
        first_tool_delta_ms: admin_latency_metric(
            capture.before.as_ref(),
            after,
            &["first_tool_delta_ms"],
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

fn admin_latency_metric(
    before: Option<&Value>,
    after: &Value,
    path: &[&str],
) -> NormalizedAdminLatencyMetricReport {
    let after_summary = value_path(after, path);
    let before_count = before
        .and_then(|value| value_path(value, path))
        .and_then(metric_count);
    let after_count = after_summary.and_then(metric_count);
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
        avg_ms_after: after_summary
            .and_then(|summary| summary.get("avg"))
            .and_then(Value::as_f64),
    }
}

fn value_path<'a>(mut value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    for segment in path {
        value = value.get(*segment)?;
    }
    Some(value)
}

fn metric_count(value: &Value) -> Option<i64> {
    value.get("count").and_then(Value::as_i64).or_else(|| {
        value
            .get("count")
            .and_then(Value::as_u64)
            .and_then(|count| i64::try_from(count).ok())
    })
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
    tool_parser: MlxToolParserMode,
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

    fn tool_parser_report(&self) -> Option<&'static str> {
        (self.tool_parser != MlxToolParserMode::Auto).then(|| self.tool_parser.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NormalizedSweepProfile {
    QwenMlxCachePrefill,
    QwenMlxPrefill135k,
    QwenMlxStablePrefix,
}

impl NormalizedSweepProfile {
    fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            QWEN_MLX_CACHE_PREFILL_PROFILE => Ok(Self::QwenMlxCachePrefill),
            QWEN_MLX_PREFILL_135K_PROFILE => Ok(Self::QwenMlxPrefill135k),
            QWEN_MLX_STABLE_PREFIX_PROFILE => Ok(Self::QwenMlxStablePrefix),
            other => anyhow::bail!(
                "unknown --sweep-profile `{other}`; expected {QWEN_MLX_CACHE_PREFILL_PROFILE}, {QWEN_MLX_PREFILL_135K_PROFILE}, or {QWEN_MLX_STABLE_PREFIX_PROFILE}"
            ),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::QwenMlxCachePrefill => QWEN_MLX_CACHE_PREFILL_PROFILE,
            Self::QwenMlxPrefill135k => QWEN_MLX_PREFILL_135K_PROFILE,
            Self::QwenMlxStablePrefix => QWEN_MLX_STABLE_PREFIX_PROFILE,
        }
    }

    fn default_probe_suite(self) -> NormalizedProbeSuite {
        match self {
            Self::QwenMlxCachePrefill => NormalizedProbeSuite::FullMatrix,
            Self::QwenMlxPrefill135k => NormalizedProbeSuite::PrefillSweep135k,
            Self::QwenMlxStablePrefix => NormalizedProbeSuite::StableAgentPrefix,
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
    PrefillSweep135k,
    StableAgentPrefix,
}

impl NormalizedProbeSuite {
    fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "full-matrix" | "full_matrix" => Ok(Self::FullMatrix),
            "focused-agentic-gate" | "focused_agentic_gate" => Ok(Self::FocusedAgenticGate),
            "prefill-sweep-135k" | "prefill_sweep_135k" => Ok(Self::PrefillSweep135k),
            "stable-agent-prefix" | "stable_agent_prefix" => Ok(Self::StableAgentPrefix),
            other => anyhow::bail!(
                "unknown --probe-suite `{other}`; expected full-matrix, focused-agentic-gate, prefill-sweep-135k, or stable-agent-prefix"
            ),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::FullMatrix => "full_matrix",
            Self::FocusedAgenticGate => "focused_agentic_gate",
            Self::PrefillSweep135k => "prefill_sweep_135k",
            Self::StableAgentPrefix => "stable_agent_prefix",
        }
    }

    fn probes(self) -> Vec<NormalizedProbePlan> {
        match self {
            Self::FullMatrix => NormalizedProbePlan::all(),
            Self::FocusedAgenticGate => NormalizedProbePlan::focused_agentic_gate(),
            Self::PrefillSweep135k => NormalizedProbePlan::prefill_sweep_135k(),
            Self::StableAgentPrefix => NormalizedProbePlan::stable_agent_prefix(),
        }
    }

    fn case_names(self, probes: &[NormalizedProbePlan]) -> Vec<&'static str> {
        match self {
            Self::FullMatrix => NormalizedCaseKind::all()
                .iter()
                .map(|case| case.name())
                .collect(),
            Self::FocusedAgenticGate | Self::PrefillSweep135k | Self::StableAgentPrefix => {
                probe_case_names(probes)
            }
        }
    }

    fn schema_variant_names(self, probes: &[NormalizedProbePlan]) -> Vec<&'static str> {
        match self {
            Self::FullMatrix => SchemaVariant::all()
                .iter()
                .map(|variant| variant.name())
                .collect(),
            Self::FocusedAgenticGate | Self::PrefillSweep135k | Self::StableAgentPrefix => {
                probe_schema_variant_names(probes)
            }
        }
    }

    fn tool_choice_variant_names(self, probes: &[NormalizedProbePlan]) -> Vec<&'static str> {
        match self {
            Self::FullMatrix => ToolChoiceVariant::all()
                .iter()
                .map(|variant| variant.name())
                .collect(),
            Self::FocusedAgenticGate | Self::PrefillSweep135k | Self::StableAgentPrefix => {
                probe_tool_choice_variant_names(probes)
            }
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
    tool_required_stream: NormalizedToolRequiredStreamTimingReport,
    lanes: Vec<NormalizedLaneReport>,
    hardware: HardwareReport,
    comparison: NormalizedComparisonReport,
    agentic_gate: NormalizedAgenticGateReport,
    prefill_sweep: NormalizedPrefillSweepReport,
    stable_prefix: NormalizedStablePrefixReport,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_parser: Option<&'static str>,
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
    #[serde(skip)]
    admin_metrics: NormalizedAdminMetricsCapture,
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
            tool_parser: lane.tool_parser_report(),
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
            admin_metrics: NormalizedAdminMetricsCapture::default(),
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

#[derive(Debug, Default)]
struct NormalizedAdminMetricsCapture {
    before: Option<Value>,
    after: Option<Value>,
    error: Option<String>,
}

impl NormalizedAdminMetricsCapture {
    fn record_before(&mut self, result: Result<Value, String>) {
        match result {
            Ok(metrics) => self.before = Some(metrics),
            Err(err) => self.push_error(format!("before {err}")),
        }
    }

    fn record_after(&mut self, result: Result<Value, String>) {
        match result {
            Ok(metrics) => self.after = Some(metrics),
            Err(err) => self.push_error(format!("after {err}")),
        }
    }

    fn push_error(&mut self, err: String) {
        match &mut self.error {
            Some(existing) => {
                existing.push_str("; ");
                existing.push_str(&err);
            }
            None => self.error = Some(err),
        }
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
    request_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_headers: Option<BTreeMap<String, String>>,
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
            request_id: None,
            response_headers: None,
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
struct NormalizedPrefillSweepReport {
    status: String,
    rows: Vec<NormalizedPrefillSweepRow>,
}

impl NormalizedPrefillSweepReport {
    fn dry_run() -> Self {
        Self {
            status: "dry_run".to_owned(),
            rows: Vec::new(),
        }
    }
}

#[derive(Debug, Serialize)]
struct NormalizedPrefillSweepRow {
    case: &'static str,
    schema_variant: &'static str,
    tool_choice_variant: &'static str,
    cache_phase: &'static str,
    run_mode: &'static str,
    fastest_lane: Option<String>,
    lanes: Vec<NormalizedPrefillSweepLaneMetric>,
}

#[derive(Debug, Serialize)]
struct NormalizedPrefillSweepLaneMetric {
    lane: String,
    lane_kind: &'static str,
    prefill_step_size: DefaultOrU64,
    valid: bool,
    invalid_reasons: Vec<String>,
    sample_count: usize,
    pass_count: usize,
    fail_count: usize,
    p50_first_response_byte_latency_ms: Option<u128>,
    p50_first_parsed_sse_chunk_latency_ms: Option<u128>,
    p50_first_semantic_delta_latency_ms: Option<u128>,
    p50_first_tool_delta_latency_ms: Option<u128>,
    p50_elapsed_latency_ms: Option<u128>,
    latency_delta_vs_fastest_ms: Option<u128>,
    avg_tokens_per_second: Option<f64>,
    avg_cached_tokens: Option<f64>,
    avg_prompt_tokens: Option<f64>,
    avg_completion_tokens: Option<f64>,
    avg_total_tokens: Option<f64>,
    response_headers: Vec<BTreeMap<String, String>>,
    admin_mlx_upstream_timing: Option<NormalizedPrefillSweepAdminMlxTiming>,
    process_rss_bytes_after: Option<u64>,
    stream_stalled_requests_delta: Option<i64>,
    no_progress_failures_delta: Option<i64>,
}

#[derive(Debug, Serialize)]
struct NormalizedPrefillSweepAdminMlxTiming {
    stream_first_upstream_byte_ms: NormalizedAdminLatencyMetricReport,
    stream_first_parsed_chunk_ms: NormalizedAdminLatencyMetricReport,
    stream_first_tool_delta_ms: NormalizedAdminLatencyMetricReport,
}

#[derive(Debug, Serialize)]
struct NormalizedStablePrefixReport {
    status: String,
    rows: Vec<NormalizedStablePrefixRow>,
}

impl NormalizedStablePrefixReport {
    fn dry_run() -> Self {
        Self {
            status: "dry_run".to_owned(),
            rows: Vec::new(),
        }
    }
}

#[derive(Debug, Serialize)]
struct NormalizedStablePrefixRow {
    case: &'static str,
    schema_variant: &'static str,
    tool_choice_variant: &'static str,
    cache_phase: &'static str,
    run_mode: &'static str,
    fastest_lane: Option<String>,
    lanes: Vec<NormalizedStablePrefixLaneMetric>,
}

#[derive(Debug, Serialize)]
struct NormalizedStablePrefixLaneMetric {
    lane: String,
    lane_kind: &'static str,
    sample_count: usize,
    pass_count: usize,
    fail_count: usize,
    p50_first_semantic_delta_latency_ms: Option<u128>,
    p50_first_tool_delta_latency_ms: Option<u128>,
    p50_elapsed_latency_ms: Option<u128>,
    latency_delta_vs_fastest_ms: Option<u128>,
    avg_prompt_tokens: Option<f64>,
    avg_cached_tokens: Option<f64>,
    avg_uncached_tokens: Option<f64>,
    cache_status_counts: BTreeMap<String, usize>,
    request_cache_observations: Vec<NormalizedStablePrefixRequestCacheObservation>,
}

#[derive(Debug, Serialize)]
struct NormalizedStablePrefixRequestCacheObservation {
    request_id: String,
    model: String,
    streamed: bool,
    prompt_tokens: u64,
    cached_tokens: Option<u64>,
    uncached_tokens: Option<u64>,
    cache_status: String,
    latency_ms: u64,
}

#[derive(Debug, Serialize)]
struct NormalizedToolRequiredStreamTimingReport {
    status: String,
    lanes: Vec<NormalizedToolRequiredStreamLaneTimingReport>,
}

impl NormalizedToolRequiredStreamTimingReport {
    fn dry_run(lanes: &[NormalizedLaneReport]) -> Self {
        Self {
            status: "dry_run".to_owned(),
            lanes: lanes
                .iter()
                .map(|lane| NormalizedToolRequiredStreamLaneTimingReport {
                    lane: lane.name.clone(),
                    kind: lane.kind,
                    pass_count: 0,
                    p50_first_byte_latency_ms: None,
                    p50_first_sse_data_latency_ms: None,
                    p50_first_tool_delta_latency_ms: None,
                    p50_tool_finish_latency_ms: None,
                    p50_first_semantic_delta_latency_ms: None,
                    admin_metrics: None,
                    admin_metrics_error: None,
                })
                .collect(),
        }
    }
}

#[derive(Debug, Serialize)]
struct NormalizedToolRequiredStreamLaneTimingReport {
    lane: String,
    kind: &'static str,
    pass_count: usize,
    p50_first_byte_latency_ms: Option<u128>,
    p50_first_sse_data_latency_ms: Option<u128>,
    p50_first_tool_delta_latency_ms: Option<u128>,
    p50_tool_finish_latency_ms: Option<u128>,
    p50_first_semantic_delta_latency_ms: Option<u128>,
    admin_metrics: Option<NormalizedToolRequiredStreamAdminMetrics>,
    admin_metrics_error: Option<String>,
}

#[derive(Debug, Serialize)]
struct NormalizedToolRequiredStreamAdminMetrics {
    first_tool_delta_ms: NormalizedAdminLatencyMetricReport,
    tool_argument_assembly_ms: NormalizedAdminLatencyMetricReport,
    tool_intent_fill_ms: NormalizedAdminLatencyMetricReport,
    tool_schema_validation_ms: NormalizedAdminLatencyMetricReport,
    tool_finish_ms: NormalizedAdminLatencyMetricReport,
    validated_tool_call_ms: NormalizedAdminLatencyMetricReport,
    mlx_stream_first_upstream_byte_ms: NormalizedAdminLatencyMetricReport,
    mlx_stream_first_parsed_chunk_ms: NormalizedAdminLatencyMetricReport,
    mlx_stream_first_tool_delta_ms: NormalizedAdminLatencyMetricReport,
}

#[derive(Debug, Serialize)]
struct NormalizedAdminLatencyMetricReport {
    count_delta: Option<i64>,
    count_after: Option<u64>,
    min_ms_after: Option<f64>,
    max_ms_after: Option<f64>,
    avg_ms_after: Option<f64>,
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
mod tests;
