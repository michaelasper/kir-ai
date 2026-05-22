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
use serde::{Deserialize, Serialize, Serializer};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

const BENCHMARK_NAME: &str = "qwen-mlx-tool-normalized";
const DEFAULT_WARMUPS: usize = 1;
const DEFAULT_SAMPLES: usize = 1;
const DEFAULT_CONTEXT_TOKENS: usize = 135_000;
const DEFAULT_CONCURRENT_REQUESTS: usize = 1;
const DEFAULT_CONCURRENT_SAMPLES: usize = 0;
const DEFAULT_MAX_TOKENS: u32 = 512;
const REQUIRED_TOOL_TTFT_MAX_TOKENS: [u32; 3] = [24, 48, 96];
const QWEN_MLX_CACHE_PREFILL_PROFILE: &str = "qwen-mlx-cache-prefill";
const QWEN_MLX_PREFILL_135K_PROFILE: &str = "qwen-mlx-prefill-135k";
const QWEN_MLX_PREFILL_135K_EXPERIMENTAL_PROFILE: &str = "qwen-mlx-prefill-135k-experimental";
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
Usage: llm-engine bench qwen-mlx-tool-normalized [OPTIONS]

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

fn parse_lane_specs(args: &[String]) -> anyhow::Result<Vec<NormalizedLaneConfig>> {
    let lane_specs = flag_values(args, "--lane");
    let lanes = if let Some(profile) = parse_sweep_profile_flag(args)? {
        if !lane_specs.is_empty() {
            anyhow::bail!("--sweep-profile cannot be combined with explicit --lane specs");
        }
        expand_sweep_profile(profile, args)?
    } else {
        if lane_specs.is_empty() {
            anyhow::bail!("qwen mlx tool normalized benchmark requires at least one --lane <spec>");
        }
        lane_specs
            .into_iter()
            .map(parse_lane_spec)
            .collect::<anyhow::Result<Vec<_>>>()?
    };
    filter_lanes_by_flag(args, lanes)
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

fn default_run_config_for_probe_suite(suite: NormalizedProbeSuite) -> NormalizedRunConfig {
    let mut config = NormalizedRunConfig::new(
        if suite == NormalizedProbeSuite::RequiredToolTtftMatrix {
            0
        } else {
            DEFAULT_WARMUPS
        },
        DEFAULT_SAMPLES,
        DEFAULT_CONTEXT_TOKENS,
        DEFAULT_CONCURRENT_REQUESTS,
        DEFAULT_CONCURRENT_SAMPLES,
    );
    if suite == NormalizedProbeSuite::RequiredToolTtftMatrix {
        config = config.with_cache_phases(vec![CachePhase::Cold]);
    }
    config
}

fn filter_lanes_by_flag(
    args: &[String],
    lanes: Vec<NormalizedLaneConfig>,
) -> anyhow::Result<Vec<NormalizedLaneConfig>> {
    let only_lanes = flag_values(args, "--only-lanes");
    let profile_lanes = flag_values(args, "--profile-lanes");
    let (flag, values) = match (only_lanes.as_slice(), profile_lanes.as_slice()) {
        ([], []) => return Ok(lanes),
        ([value], []) => ("--only-lanes", *value),
        ([], [value]) => ("--profile-lanes", *value),
        _ => anyhow::bail!(
            "--only-lanes and --profile-lanes may only be provided once and cannot be combined"
        ),
    };
    let selected = parse_csv_names(flag, values)?;
    let available = lanes
        .iter()
        .map(|lane| lane.name.as_str())
        .collect::<BTreeSet<_>>();
    let missing = selected
        .iter()
        .filter(|name| !available.contains(name.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        anyhow::bail!(
            "{flag} referenced unknown lanes: {}; available lanes: {}",
            missing.join(", "),
            available.into_iter().collect::<Vec<_>>().join(", ")
        );
    }
    let selected = selected.into_iter().collect::<BTreeSet<_>>();
    Ok(lanes
        .into_iter()
        .filter(|lane| selected.contains(&lane.name))
        .collect())
}

fn expand_sweep_profile(
    profile: NormalizedSweepProfile,
    args: &[String],
) -> anyhow::Result<Vec<NormalizedLaneConfig>> {
    let snapshot = required_profile_snapshot(args)?;
    Ok(match profile {
        NormalizedSweepProfile::QwenMlxCachePrefill => qwen_mlx_cache_prefill_lanes(snapshot),
        NormalizedSweepProfile::QwenMlxPrefill135k => qwen_mlx_prefill_135k_lanes(snapshot),
        NormalizedSweepProfile::QwenMlxPrefill135kExperimental => {
            qwen_mlx_prefill_135k_experimental_lanes(snapshot)
        }
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
            experimental: false,
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

fn qwen_mlx_prefill_135k_experimental_lanes(snapshot: &str) -> Vec<NormalizedLaneConfig> {
    let steps = [
        ("8192-control", DefaultOrU64::Value(8192), false),
        ("experimental-12288", DefaultOrU64::Value(12288), true),
        ("experimental-16384", DefaultOrU64::Value(16384), true),
        ("experimental-32768", DefaultOrU64::Value(32768), true),
    ];
    let mut lanes = Vec::with_capacity(steps.len() * 2);
    for (index, (label, prefill_step_size, experimental)) in steps.into_iter().enumerate() {
        let port_offset = index as u16;
        let settings = MlxLmSettings {
            prefill_step_size,
            ..MlxLmSettings::default()
        };
        let mut direct = profile_direct_lane(
            &format!("mlx-prefill-{label}"),
            8080 + port_offset,
            settings,
            snapshot,
        );
        direct.experimental = experimental;
        lanes.push(direct);

        let mut proxy = profile_proxy_lane(
            &format!("kir-prefill-{label}"),
            3000 + port_offset,
            settings,
            snapshot,
        );
        proxy.experimental = experimental;
        lanes.push(proxy);
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
            experimental: false,
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
            experimental: false,
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
        experimental: false,
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
        experimental: false,
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
        experimental: false,
    })
}

fn parse_mlx_tool_parser_mode(value: &str) -> anyhow::Result<MlxToolParserMode> {
    MlxToolParserMode::parse(value)
        .ok_or_else(|| anyhow!("unknown tool_parser `{value}`; expected auto, json, or qwen-xml"))
}

fn parse_cache_phases_flag(args: &[String]) -> anyhow::Result<Vec<CachePhase>> {
    let values = flag_values(args, "--cache-phases");
    let Some(value) = values.first() else {
        return Ok(CachePhase::all().to_vec());
    };
    if values.len() > 1 {
        anyhow::bail!("--cache-phases may only be provided once");
    }
    parse_csv_names("--cache-phases", value)?
        .into_iter()
        .map(|name| CachePhase::parse(&name))
        .collect()
}

fn parse_csv_names(flag: &str, value: &str) -> anyhow::Result<Vec<String>> {
    let mut names = Vec::new();
    for name in value.split(',').map(str::trim) {
        if name.is_empty() {
            anyhow::bail!("{flag} contains an empty value");
        }
        let name = name.to_owned();
        if names.contains(&name) {
            anyhow::bail!("{flag} contains duplicate value `{name}`");
        }
        names.push(name);
    }
    if names.is_empty() {
        anyhow::bail!("{flag} requires at least one value");
    }
    Ok(names)
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

fn parse_optional_count_flag(args: &[String], flag: &str) -> anyhow::Result<Option<usize>> {
    let values = flag_values(args, flag);
    let Some(value) = values.first() else {
        return Ok(None);
    };
    if values.len() > 1 {
        anyhow::bail!("{flag} may only be provided once");
    }
    let parsed = value
        .parse::<usize>()
        .with_context(|| format!("parse {flag}"))?;
    if parsed == 0 {
        anyhow::bail!("{flag} must be greater than zero");
    }
    Ok(Some(parsed))
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

#[derive(Debug, Clone)]
struct NormalizedRunConfig {
    warmups: usize,
    samples: usize,
    context_tokens: usize,
    concurrent_requests: usize,
    concurrent_samples: usize,
    effective_concurrent_samples: usize,
    cache_phases: Vec<CachePhase>,
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
            cache_phases: CachePhase::all().to_vec(),
        }
    }

    fn with_cache_phases(mut self, cache_phases: Vec<CachePhase>) -> Self {
        self.cache_phases = cache_phases;
        self
    }
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

fn sweep_profile_requires_exact_token_prompt(profile: Option<NormalizedSweepProfile>) -> bool {
    matches!(
        profile,
        Some(
            NormalizedSweepProfile::QwenMlxPrefill135k
                | NormalizedSweepProfile::QwenMlxPrefill135kExperimental
        )
    )
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

fn normalized_plan_summary(
    lanes: &[NormalizedLaneConfig],
    probes: &[NormalizedProbePlan],
    run_config: &NormalizedRunConfig,
) -> NormalizedPlanSummaryReport {
    let sequential_plan = phase_plan(
        &run_config.cache_phases,
        run_config.warmups,
        run_config.samples,
    );
    let concurrent_plan = concurrent_phase_plan(
        &run_config.cache_phases,
        run_config.concurrent_requests,
        run_config.effective_concurrent_samples,
    );
    let per_probe_warmups = sequential_plan
        .iter()
        .filter(|run| run.kind == PlannedRunKind::Warmup)
        .count();
    let per_probe_sequential_measured = sequential_plan
        .iter()
        .filter(|run| run.kind == PlannedRunKind::Measured)
        .count();
    let per_probe_concurrent_measured = concurrent_plan.len();
    let plan_multiplier = lanes.len().saturating_mul(probes.len());
    let warmup_requests = per_probe_warmups.saturating_mul(plan_multiplier);
    let sequential_measured_requests =
        per_probe_sequential_measured.saturating_mul(plan_multiplier);
    let concurrent_measured_requests =
        per_probe_concurrent_measured.saturating_mul(plan_multiplier);
    let measured_requests =
        sequential_measured_requests.saturating_add(concurrent_measured_requests);
    let total_http_requests = warmup_requests.saturating_add(measured_requests);

    NormalizedPlanSummaryReport {
        probe_count: probes.len(),
        lane_count: lanes.len(),
        warmups_per_warm_phase: run_config.warmups,
        samples_per_phase: run_config.samples,
        concurrent_requests: run_config.concurrent_requests,
        concurrent_samples: run_config.concurrent_samples,
        effective_concurrent_samples: run_config.effective_concurrent_samples,
        cache_phases: run_config
            .cache_phases
            .iter()
            .map(|phase| phase.name())
            .collect(),
        probes: probes
            .iter()
            .map(|probe| NormalizedPlanProbeReport {
                case: probe.case.name(),
                schema_variant: probe.schema_variant.name(),
                tool_choice_variant: probe.tool_choice_variant.name(),
                max_tokens: probe.max_tokens,
            })
            .collect(),
        lanes: lanes.iter().map(|lane| lane.name.clone()).collect(),
        warmup_requests,
        measured_requests,
        sequential_measured_requests,
        concurrent_measured_requests,
        total_http_requests,
        planned_prompt_token_budget: total_http_requests.saturating_mul(run_config.context_tokens),
    }
}

fn enforce_plan_budget(
    summary: &NormalizedPlanSummaryReport,
    max_requests: Option<usize>,
    max_planned_prompt_tokens: Option<usize>,
) -> anyhow::Result<()> {
    if let Some(max_requests) = max_requests
        && summary.total_http_requests > max_requests
    {
        anyhow::bail!(
            "selected benchmark plan requires {} HTTP requests, exceeding --max-requests {max_requests}",
            summary.total_http_requests
        );
    }
    if let Some(max_planned_prompt_tokens) = max_planned_prompt_tokens
        && summary.planned_prompt_token_budget > max_planned_prompt_tokens
    {
        anyhow::bail!(
            "selected benchmark plan requires {} planned prompt tokens, exceeding --max-planned-prompt-tokens {max_planned_prompt_tokens}",
            summary.planned_prompt_token_budget
        );
    }
    Ok(())
}

fn planned_requests_for(
    probes: &[NormalizedProbePlan],
    run_config: &NormalizedRunConfig,
) -> Vec<NormalizedPlannedRequestReport> {
    let mut planned_requests = Vec::new();
    for &probe in probes {
        for planned in phase_plan(
            &run_config.cache_phases,
            run_config.warmups,
            run_config.samples,
        ) {
            planned_requests.push(NormalizedPlannedRequestReport::new(
                probe, planned, run_config,
            ));
        }
        for planned in concurrent_phase_plan(
            &run_config.cache_phases,
            run_config.concurrent_requests,
            run_config.effective_concurrent_samples,
        ) {
            planned_requests.push(NormalizedPlannedRequestReport::new(
                probe, planned, run_config,
            ));
        }
    }
    planned_requests
}

fn compare_normalized_lanes_for_phases(
    lanes: &[NormalizedLaneReport],
    probes: &[NormalizedProbePlan],
    phases: &[CachePhase],
) -> NormalizedComparisonReport {
    let mut fastest = Vec::new();
    for &probe in probes {
        for &phase in phases {
            let mut fastest_lane = None;
            let mut fastest_latency_ms = None;
            let mut lane_metrics = Vec::new();
            for lane in lanes {
                let best_latency_ms = lane
                    .samples
                    .iter()
                    .filter(|sample| {
                        sample_matches_probe(sample, probe)
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
                max_tokens: probe.max_tokens,
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

#[cfg(test)]
fn aggregate_normalized_summary(
    lanes: &[NormalizedLaneReport],
    probes: &[NormalizedProbePlan],
) -> Vec<NormalizedAggregateSummaryRow> {
    aggregate_normalized_summary_for_phases(lanes, probes, &CachePhase::all())
}

fn aggregate_normalized_summary_for_phases(
    lanes: &[NormalizedLaneReport],
    probes: &[NormalizedProbePlan],
    phases: &[CachePhase],
) -> Vec<NormalizedAggregateSummaryRow> {
    let mut rows = Vec::new();
    for lane in lanes {
        for &probe in probes {
            for &phase in phases {
                for run_mode in [RunMode::Sequential, RunMode::Concurrent] {
                    let samples = lane_samples(lane)
                        .filter(|sample| {
                            sample_matches_probe(sample, probe)
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
                        max_tokens: probe.max_tokens,
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

#[cfg(test)]
fn agentic_gate_report(lanes: &[NormalizedLaneReport]) -> NormalizedAgenticGateReport {
    agentic_gate_report_for_phases(lanes, &CachePhase::all())
}

fn agentic_gate_report_for_phases(
    lanes: &[NormalizedLaneReport],
    phases: &[CachePhase],
) -> NormalizedAgenticGateReport {
    let mut rows = Vec::new();
    for probe in NormalizedProbePlan::focused_agentic_gate() {
        for &phase in phases {
            for run_mode in [RunMode::Sequential, RunMode::Concurrent] {
                let mut lane_metrics = Vec::new();
                let mut fastest_lane = None;
                let mut fastest_latency = None;
                for lane in lanes {
                    let samples = lane_samples(lane)
                        .filter(|sample| {
                            sample_matches_probe(sample, probe)
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
                    max_tokens: probe.max_tokens,
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

async fn load_agentic_streaming_fast_path_ab_report(
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

fn agentic_streaming_fast_path_ab_report(
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

fn agentic_streaming_fast_path_ab_row(
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

fn agentic_ab_group_keys(
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

fn agentic_ab_samples<'a>(
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

fn percentile_for_comparable_samples(
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

fn metric_delta_ms(candidate: Option<u128>, baseline: Option<u128>) -> Option<i64> {
    let candidate = i64::try_from(candidate?).ok()?;
    let baseline = i64::try_from(baseline?).ok()?;
    Some(candidate - baseline)
}

fn comparable_lanes_from_normalized(lanes: &[NormalizedLaneReport]) -> Vec<ComparableLaneReport> {
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

fn comparable_sample_from_normalized(sample: &NormalizedSampleReport) -> ComparableSampleReport {
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
struct ComparableBenchReport {
    #[serde(default)]
    lanes: Vec<ComparableLaneReport>,
}

#[derive(Debug, Clone, Deserialize)]
struct ComparableLaneReport {
    #[serde(default)]
    name: String,
    #[serde(default)]
    kind: String,
    #[serde(default)]
    samples: Vec<ComparableSampleReport>,
    #[serde(default)]
    concurrent_samples: Vec<ComparableSampleReport>,
}

impl ComparableLaneReport {
    fn all_samples(&self) -> impl Iterator<Item = &ComparableSampleReport> {
        self.samples.iter().chain(self.concurrent_samples.iter())
    }
}

#[derive(Debug, Clone, Deserialize)]
struct ComparableSampleReport {
    #[serde(default)]
    case: String,
    #[serde(default)]
    schema_variant: String,
    #[serde(default)]
    tool_choice_variant: String,
    #[serde(default)]
    cache_phase: String,
    #[serde(default)]
    run_mode: String,
    #[serde(default)]
    status: String,
    #[serde(default)]
    classification: String,
    #[serde(default, flatten)]
    stream_timing: StreamTimingReport,
    #[serde(default)]
    finish_reason: Option<String>,
}

impl ComparableSampleReport {
    fn matches_agentic_ab_probe(&self) -> bool {
        self.case == AGENTIC_AB_CASE
            && self.schema_variant == AGENTIC_AB_SCHEMA_VARIANT
            && self.tool_choice_variant == AGENTIC_AB_TOOL_CHOICE_VARIANT
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
struct ToolValidationSignature {
    sample_count: usize,
    pass_count: usize,
    fail_count: usize,
    status_counts: BTreeMap<String, usize>,
    classification_counts: BTreeMap<String, usize>,
    finish_reason_counts: BTreeMap<String, usize>,
}

impl ToolValidationSignature {
    fn from_samples(samples: &[&ComparableSampleReport]) -> Self {
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

    fn successful_tool_stream(&self) -> bool {
        self.sample_count > 0
            && self.pass_count == self.sample_count
            && self.fail_count == 0
            && self
                .finish_reason_counts
                .get("tool_calls")
                .is_some_and(|count| *count == self.sample_count)
    }
}

fn increment_count(counts: &mut BTreeMap<String, usize>, value: &str) {
    *counts.entry(value.to_owned()).or_insert(0) += 1;
}

#[cfg(test)]
fn prefill_sweep_report(
    lanes: &[NormalizedLaneReport],
    probes: &[NormalizedProbePlan],
) -> NormalizedPrefillSweepReport {
    prefill_sweep_report_for_phases(lanes, probes, &CachePhase::all())
}

fn prefill_sweep_report_for_phases(
    lanes: &[NormalizedLaneReport],
    probes: &[NormalizedProbePlan],
    phases: &[CachePhase],
) -> NormalizedPrefillSweepReport {
    let mut rows = Vec::new();
    for &probe in probes {
        for &phase in phases {
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
                    max_tokens: probe.max_tokens,
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
            sample_matches_probe(sample, probe)
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
    let failure_classifications = sample_failure_classification_counts(&samples);
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
    invalid_reasons.extend(failure_classifications.keys().cloned());
    invalid_reasons.sort();
    invalid_reasons.dedup();

    Some(NormalizedPrefillSweepLaneMetric {
        lane: lane.name.clone(),
        lane_kind: lane.kind,
        experimental: lane.experimental,
        prefill_step_size: lane.mlx_lm_settings.prefill_step_size,
        valid: invalid_reasons.is_empty(),
        failure_classifications,
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
        avg_uncached_tokens: average_u64(
            passed
                .iter()
                .filter_map(|sample| sample_direct_uncached_tokens(sample)),
        ),
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

#[cfg(test)]
fn stable_prefix_report(
    lanes: &[NormalizedLaneReport],
    probes: &[NormalizedProbePlan],
) -> NormalizedStablePrefixReport {
    stable_prefix_report_for_phases(lanes, probes, &CachePhase::all())
}

fn stable_prefix_report_for_phases(
    lanes: &[NormalizedLaneReport],
    probes: &[NormalizedProbePlan],
    phases: &[CachePhase],
) -> NormalizedStablePrefixReport {
    let mut rows = Vec::new();
    for &probe in probes {
        for &phase in phases {
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
                    max_tokens: probe.max_tokens,
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
            sample_matches_probe(sample, probe)
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
    let request_cache_observations = matching_request_cache_observations(lane, &samples);
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
        avg_cached_tokens: average_u64(
            passed
                .iter()
                .filter_map(|sample| sample_cached_tokens(sample, &request_cache_observations)),
        ),
        avg_uncached_tokens: average_u64(
            passed
                .iter()
                .filter_map(|sample| sample_uncached_tokens(sample, &request_cache_observations)),
        ),
        cache_status_counts: cache_status_counts(&passed, &request_cache_observations),
        request_cache_observations,
    })
}

fn stable_prefix_metric_sort_key(metric: &NormalizedStablePrefixLaneMetric) -> (u128, String) {
    (
        metric.p50_elapsed_latency_ms.unwrap_or(u128::MAX),
        metric.lane.clone(),
    )
}

fn latest_performance_comparison_report(
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

fn latest_plain_stream_row(
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

fn latest_tool_stream_row(
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

fn latest_prefix_cache_row(
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

fn latest_live_comparison_row(
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

fn engine_db_baseline_comparison_row(
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

fn latest_source_kind(kind: &str) -> Option<&'static str> {
    match kind {
        "kir_ai_proxy" => Some("latest_kir"),
        "direct_mlx" => Some("direct_mlx"),
        _ => None,
    }
}

fn latest_passed_samples_prefer_phase(
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

fn latest_passed_samples(
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

fn latest_cache_case(lane: &NormalizedLaneReport) -> Option<NormalizedCaseKind> {
    [
        NormalizedCaseKind::WarmPrefixRepeatedTurnStream,
        NormalizedCaseKind::OmpRepeatedPrefix,
    ]
    .into_iter()
    .find(|case| !latest_passed_samples(lane, *case, None).is_empty())
}

fn latest_warm_cache_samples(
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

fn optional_u128_as_f64(value: Option<u128>) -> Option<f64> {
    value.map(|value| value as f64)
}

fn cache_speedup(
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

fn cache_status_counts(
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

fn sample_failure_classification_counts(
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

fn cache_status_for_sample(
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

fn cache_status_from_sample(sample: &NormalizedSampleReport) -> Option<&'static str> {
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

fn sample_cached_tokens(
    sample: &NormalizedSampleReport,
    observations: &[NormalizedStablePrefixRequestCacheObservation],
) -> Option<u64> {
    sample.cached_tokens.or_else(|| {
        request_cache_observation_for_sample(sample, observations)
            .and_then(|observation| observation.cached_tokens)
    })
}

fn sample_direct_uncached_tokens(sample: &NormalizedSampleReport) -> Option<u64> {
    Some(sample.prompt_tokens?.saturating_sub(sample.cached_tokens?))
}

fn sample_uncached_tokens(
    sample: &NormalizedSampleReport,
    observations: &[NormalizedStablePrefixRequestCacheObservation],
) -> Option<u64> {
    if let Some(cached_tokens) = sample.cached_tokens {
        return Some(sample.prompt_tokens?.saturating_sub(cached_tokens));
    }
    request_cache_observation_for_sample(sample, observations)
        .and_then(|observation| observation.uncached_tokens)
}

fn request_cache_observation_for_sample<'a>(
    sample: &NormalizedSampleReport,
    observations: &'a [NormalizedStablePrefixRequestCacheObservation],
) -> Option<&'a NormalizedStablePrefixRequestCacheObservation> {
    let request_id = sample.request_id.as_deref()?;
    observations
        .iter()
        .find(|observation| observation.request_id == request_id)
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

fn required_tool_ttft_matrix_report(
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

fn required_tool_ttft_sample_selected(
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

fn required_tool_ttft_matrix_row(
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

fn fastest_required_tool_ttft_delta(
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

fn lane_samples_for_all_lanes(
    lanes: &[NormalizedLaneReport],
) -> impl Iterator<Item = &NormalizedSampleReport> {
    lanes.iter().flat_map(lane_samples)
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
    let attribution = tool_required_stream_attribution_report(lanes, &lane_reports);
    NormalizedToolRequiredStreamTimingReport {
        status: status.to_owned(),
        attribution,
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
        tool_stream_observations: matching_tool_stream_observations(lane, &samples),
    }
}

fn tool_required_stream_attribution_report(
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

fn tool_required_stream_attribution_row(
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

fn tool_required_stream_attribution_admin_metrics(
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

fn join_admin_metric_errors(captures: &[&NormalizedAdminMetricsCapture]) -> Option<String> {
    let errors = captures
        .iter()
        .filter_map(|capture| capture.error.as_deref())
        .collect::<Vec<_>>();
    (!errors.is_empty()).then(|| errors.join("; "))
}

fn aggregate_tool_stream_admin_metrics(
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

fn aggregate_admin_latency_metric(
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

fn tool_required_stream_gap(
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

fn tool_required_stream_attribution_decision(
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

fn metric_gap_ms(client_ms: Option<f64>, metric_ms: Option<f64>) -> Option<f64> {
    Some(client_ms? - metric_ms?)
}

fn sum_present_i64(values: impl Iterator<Item = Option<i64>>) -> Option<i64> {
    let mut found = false;
    let mut sum = 0;
    for value in values.flatten() {
        found = true;
        sum += value;
    }
    found.then_some(sum)
}

fn max_present_u64(values: impl Iterator<Item = Option<u64>>) -> Option<u64> {
    values.flatten().max()
}

fn min_present_f64(values: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    values.flatten().reduce(f64::min)
}

fn max_present_f64(values: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    values.flatten().reduce(f64::max)
}

fn avg_present_f64(values: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    let mut count = 0;
    let mut sum = 0.0;
    for value in values.flatten() {
        count += 1;
        sum += value;
    }
    (count > 0).then_some(sum / f64::from(count))
}

fn matching_tool_stream_observations(
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

fn tool_stream_observation(
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

fn admin_latency_metric(
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

fn admin_latency_window_avg_ms(
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

fn value_path<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    if let Some(found) = value_path_direct(value, path) {
        return Some(found);
    }
    let (first, rest) = path.split_first()?;
    (*first == "mlx").then_some(())?;
    value_path_direct(value.get("backend_metrics")?.get("mlx")?, rest)
}

fn value_path_direct<'a>(mut value: &'a Value, path: &[&str]) -> Option<&'a Value> {
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

fn sample_matches_probe(sample: &NormalizedSampleReport, probe: NormalizedProbePlan) -> bool {
    sample.case == probe.case.name()
        && sample.schema_variant == probe.schema_variant.name()
        && sample.tool_choice_variant == probe.tool_choice_variant.name()
        && sample.max_tokens == probe.max_tokens
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
                sample_matches_probe(sample, probe)
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

#[cfg(test)]
fn unique_probe_max_tokens(probes: &[NormalizedProbePlan]) -> Vec<u32> {
    let mut unique = Vec::new();
    for max_tokens in probes.iter().map(|probe| probe.max_tokens) {
        if !unique.contains(&max_tokens) {
            unique.push(max_tokens);
        }
    }
    unique
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

async fn load_engine_db_baseline_export(
    path: Option<&Path>,
) -> anyhow::Result<Option<EngineDbBaselineExport>> {
    let Some(path) = path else {
        return Ok(None);
    };
    let bytes = tokio::fs::read(path)
        .await
        .with_context(|| format!("read engine DB baseline export `{}`", path.display()))?;
    let mut export: EngineDbBaselineExport = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse engine DB baseline export `{}`", path.display()))?;
    if export.source.is_none() {
        export.source = Some(path.display().to_string());
    }
    Ok(Some(export))
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
    experimental: bool,
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
    QwenMlxPrefill135kExperimental,
    QwenMlxStablePrefix,
}

impl NormalizedSweepProfile {
    fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            QWEN_MLX_CACHE_PREFILL_PROFILE => Ok(Self::QwenMlxCachePrefill),
            QWEN_MLX_PREFILL_135K_PROFILE => Ok(Self::QwenMlxPrefill135k),
            QWEN_MLX_PREFILL_135K_EXPERIMENTAL_PROFILE => Ok(Self::QwenMlxPrefill135kExperimental),
            QWEN_MLX_STABLE_PREFIX_PROFILE => Ok(Self::QwenMlxStablePrefix),
            other => anyhow::bail!(
                "unknown --sweep-profile `{other}`; expected {QWEN_MLX_CACHE_PREFILL_PROFILE}, {QWEN_MLX_PREFILL_135K_PROFILE}, {QWEN_MLX_PREFILL_135K_EXPERIMENTAL_PROFILE}, or {QWEN_MLX_STABLE_PREFIX_PROFILE}"
            ),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::QwenMlxCachePrefill => QWEN_MLX_CACHE_PREFILL_PROFILE,
            Self::QwenMlxPrefill135k => QWEN_MLX_PREFILL_135K_PROFILE,
            Self::QwenMlxPrefill135kExperimental => QWEN_MLX_PREFILL_135K_EXPERIMENTAL_PROFILE,
            Self::QwenMlxStablePrefix => QWEN_MLX_STABLE_PREFIX_PROFILE,
        }
    }

    fn default_probe_suite(self) -> NormalizedProbeSuite {
        match self {
            Self::QwenMlxCachePrefill => NormalizedProbeSuite::FullMatrix,
            Self::QwenMlxPrefill135k => NormalizedProbeSuite::PrefillSweep135k,
            Self::QwenMlxPrefill135kExperimental => {
                NormalizedProbeSuite::PrefillSweep135kContextRecall
            }
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

#[derive(Debug, Serialize)]
struct NormalizedPlanSummaryReport {
    probe_count: usize,
    lane_count: usize,
    warmups_per_warm_phase: usize,
    samples_per_phase: usize,
    concurrent_requests: usize,
    concurrent_samples: usize,
    effective_concurrent_samples: usize,
    cache_phases: Vec<&'static str>,
    probes: Vec<NormalizedPlanProbeReport>,
    lanes: Vec<String>,
    warmup_requests: usize,
    measured_requests: usize,
    sequential_measured_requests: usize,
    concurrent_measured_requests: usize,
    total_http_requests: usize,
    planned_prompt_token_budget: usize,
}

#[derive(Debug, Serialize)]
struct NormalizedPlanProbeReport {
    case: &'static str,
    schema_variant: &'static str,
    tool_choice_variant: &'static str,
    max_tokens: u32,
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
    plan_summary: NormalizedPlanSummaryReport,
    summary: Vec<NormalizedAggregateSummaryRow>,
    tool_required_stream: NormalizedToolRequiredStreamTimingReport,
    required_tool_ttft_matrix: NormalizedRequiredToolTtftMatrixReport,
    lanes: Vec<NormalizedLaneReport>,
    hardware: HardwareReport,
    comparison: NormalizedComparisonReport,
    agentic_gate: NormalizedAgenticGateReport,
    agentic_streaming_fast_path_ab: NormalizedAgenticStreamingFastPathAbReport,
    prefill_sweep: NormalizedPrefillSweepReport,
    stable_prefix: NormalizedStablePrefixReport,
    latest_performance_comparison: NormalizedLatestPerformanceComparisonReport,
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
    experimental: bool,
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
    planned_requests: Vec<NormalizedPlannedRequestReport>,
    samples: Vec<NormalizedSampleReport>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    concurrent_samples: Vec<NormalizedSampleReport>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    warmup_failures: Vec<NormalizedWarmupFailure>,
    #[serde(skip)]
    admin_metrics: NormalizedAdminMetricsCapture,
}

impl NormalizedLaneReport {
    #[cfg(test)]
    fn planned(
        lane: &NormalizedLaneConfig,
        warmups: usize,
        samples: usize,
        snapshot_identity: Option<ModelIdentityReport>,
    ) -> Self {
        let run_config = NormalizedRunConfig::new(
            warmups,
            samples,
            DEFAULT_CONTEXT_TOKENS,
            DEFAULT_CONCURRENT_REQUESTS,
            DEFAULT_CONCURRENT_SAMPLES,
        );
        Self::planned_with_requests(lane, warmups, samples, &run_config, snapshot_identity, &[])
    }

    fn planned_with_requests(
        lane: &NormalizedLaneConfig,
        warmups: usize,
        samples: usize,
        run_config: &NormalizedRunConfig,
        snapshot_identity: Option<ModelIdentityReport>,
        probes: &[NormalizedProbePlan],
    ) -> Self {
        Self {
            name: lane.name.clone(),
            status: "planned".to_owned(),
            endpoint: lane.endpoint.clone(),
            kind: lane.kind.as_str(),
            experimental: lane.experimental,
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
            planned_requests: planned_requests_for(probes, run_config),
            samples: Vec::new(),
            concurrent_samples: Vec::new(),
            warmup_failures: Vec::new(),
            admin_metrics: NormalizedAdminMetricsCapture::default(),
        }
    }

    fn dry_run(
        lane: &NormalizedLaneConfig,
        run_config: &NormalizedRunConfig,
        snapshot_identity: Option<ModelIdentityReport>,
        probes: &[NormalizedProbePlan],
    ) -> Self {
        let mut report = Self::planned_with_requests(
            lane,
            run_config.warmups,
            run_config.samples,
            run_config,
            snapshot_identity,
            probes,
        );
        report.status = "dry_run".to_owned();
        for &probe in probes {
            for planned in phase_plan(
                &run_config.cache_phases,
                run_config.warmups,
                run_config.samples,
            ) {
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
                &run_config.cache_phases,
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
    max_tokens: u32,
    cache_phase: &'static str,
    warmup_index: usize,
    classification: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    http_status: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct NormalizedPlannedRequestReport {
    case: &'static str,
    schema_variant: &'static str,
    tool_choice_variant: &'static str,
    max_tokens: u32,
    cache_phase: &'static str,
    run_mode: &'static str,
    request_kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    sample_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    warmup_index: Option<usize>,
    planned_prompt_tokens: usize,
    prewarmed: bool,
}

impl NormalizedPlannedRequestReport {
    fn new(
        probe: NormalizedProbePlan,
        planned: PlannedRun,
        run_config: &NormalizedRunConfig,
    ) -> Self {
        Self {
            case: probe.case.name(),
            schema_variant: probe.schema_variant.name(),
            tool_choice_variant: probe.tool_choice_variant.name(),
            max_tokens: probe.max_tokens,
            cache_phase: planned.phase.name(),
            run_mode: planned.run_mode.name(),
            request_kind: planned.kind.name(),
            sample_index: planned.sample_index,
            request_index: planned.request_index,
            warmup_index: planned.warmup_index,
            planned_prompt_tokens: run_config.context_tokens,
            prewarmed: planned.phase.warms_before_samples() && run_config.warmups > 0,
        }
    }
}

#[derive(Debug, Serialize)]
struct NormalizedSampleReport {
    case: &'static str,
    schema_variant: &'static str,
    tool_choice_variant: &'static str,
    max_tokens: u32,
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
    failure_classification: Option<String>,
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
    #[serde(skip)]
    tool_required_stream_admin_metrics: Option<NormalizedAdminMetricsCapture>,
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
            max_tokens: probe.max_tokens,
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
            failure_classification: None,
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
            tool_required_stream_admin_metrics: None,
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
    max_tokens: u32,
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
    max_tokens: u32,
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
struct NormalizedAgenticStreamingFastPathAbReport {
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    baseline_path: Option<String>,
    case: &'static str,
    schema_variant: &'static str,
    tool_choice_variant: &'static str,
    rows: Vec<NormalizedAgenticStreamingFastPathAbRow>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    failure_reasons: Vec<String>,
}

impl NormalizedAgenticStreamingFastPathAbReport {
    fn dry_run(baseline_path: Option<&Path>) -> Self {
        match baseline_path {
            Some(path) => Self {
                status: "dry_run".to_owned(),
                baseline_path: Some(path.display().to_string()),
                case: AGENTIC_AB_CASE,
                schema_variant: AGENTIC_AB_SCHEMA_VARIANT,
                tool_choice_variant: AGENTIC_AB_TOOL_CHOICE_VARIANT,
                rows: Vec::new(),
                failure_reasons: Vec::new(),
            },
            None => Self::not_configured(),
        }
    }

    fn not_configured() -> Self {
        Self {
            status: "not_configured".to_owned(),
            baseline_path: None,
            case: AGENTIC_AB_CASE,
            schema_variant: AGENTIC_AB_SCHEMA_VARIANT,
            tool_choice_variant: AGENTIC_AB_TOOL_CHOICE_VARIANT,
            rows: Vec::new(),
            failure_reasons: Vec::new(),
        }
    }
}

#[derive(Debug, Serialize)]
struct NormalizedAgenticStreamingFastPathAbRow {
    lane: String,
    kind: String,
    assertion_role: &'static str,
    cache_phase: String,
    run_mode: String,
    baseline_sample_count: usize,
    candidate_sample_count: usize,
    baseline_pass_count: usize,
    candidate_pass_count: usize,
    baseline_fail_count: usize,
    candidate_fail_count: usize,
    baseline_p50_first_tool_delta_latency_ms: Option<u128>,
    candidate_p50_first_tool_delta_latency_ms: Option<u128>,
    first_tool_delta_delta_ms: Option<i64>,
    first_tool_delta_advanced: Option<bool>,
    baseline_p50_tool_finish_latency_ms: Option<u128>,
    candidate_p50_tool_finish_latency_ms: Option<u128>,
    tool_finish_delta_ms: Option<i64>,
    final_validation_unchanged: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    failure_reasons: Vec<String>,
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
    max_tokens: u32,
    cache_phase: &'static str,
    run_mode: &'static str,
    fastest_lane: Option<String>,
    lanes: Vec<NormalizedPrefillSweepLaneMetric>,
}

#[derive(Debug, Serialize)]
struct NormalizedPrefillSweepLaneMetric {
    lane: String,
    lane_kind: &'static str,
    experimental: bool,
    prefill_step_size: DefaultOrU64,
    valid: bool,
    failure_classifications: BTreeMap<String, usize>,
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
    avg_uncached_tokens: Option<f64>,
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
    max_tokens: u32,
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

#[derive(Debug, Clone, Deserialize)]
struct EngineDbBaselineExport {
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    rows: Vec<EngineDbBaselineRow>,
}

#[derive(Debug, Clone, Deserialize)]
struct EngineDbBaselineRow {
    engine: String,
    profile: String,
    #[serde(default)]
    model: Option<String>,
    probe: String,
    #[serde(default, alias = "ttft_ms", alias = "first_semantic_delta_ms")]
    ttfi_ms: Option<f64>,
    #[serde(default, alias = "first_tool_event_ms")]
    first_tool_delta_ms: Option<f64>,
    #[serde(default)]
    validated_tool_call_ms: Option<f64>,
    #[serde(default, alias = "latency_ms")]
    total_latency_ms: Option<f64>,
    #[serde(default, alias = "tok_s", alias = "toks_per_second")]
    tokens_per_second: Option<f64>,
    #[serde(default, alias = "cold_latency_ms")]
    cache_cold_latency_ms: Option<f64>,
    #[serde(default, alias = "warm_latency_ms")]
    cache_warm_latency_ms: Option<f64>,
    #[serde(default, alias = "speedup")]
    cache_speedup: Option<f64>,
    #[serde(default)]
    cached_tokens: Option<u64>,
    #[serde(default)]
    notes: Option<String>,
}

#[derive(Debug, Serialize)]
struct NormalizedLatestPerformanceComparisonReport {
    status: String,
    engine_db_baseline_source: Option<String>,
    evidence: NormalizedLatestPerformanceEvidence,
    rows: Vec<NormalizedLatestPerformanceComparisonRow>,
}

#[derive(Debug, Serialize)]
struct NormalizedLatestPerformanceEvidence {
    has_kir_latest: bool,
    has_direct_mlx_latest: bool,
    has_engine_db_baselines: bool,
    has_ttfi_ms: bool,
    has_cache_metrics: bool,
    has_tokens_per_second: bool,
}

impl NormalizedLatestPerformanceEvidence {
    fn from_rows(rows: &[NormalizedLatestPerformanceComparisonRow]) -> Self {
        Self {
            has_kir_latest: rows.iter().any(|row| row.source_kind == "latest_kir"),
            has_direct_mlx_latest: rows.iter().any(|row| row.source_kind == "direct_mlx"),
            has_engine_db_baselines: rows
                .iter()
                .any(|row| row.source_kind == "engine_db_baseline"),
            has_ttfi_ms: rows
                .iter()
                .any(|row| row.ttfi_ms.is_some() || row.first_tool_delta_ms.is_some()),
            has_cache_metrics: rows.iter().any(|row| {
                row.cache_cold_latency_ms.is_some()
                    || row.cache_warm_latency_ms.is_some()
                    || row.cache_speedup.is_some()
                    || row.cached_tokens.is_some()
            }),
            has_tokens_per_second: rows.iter().any(|row| row.tokens_per_second.is_some()),
        }
    }
}

#[derive(Debug, Serialize)]
struct NormalizedLatestPerformanceComparisonRow {
    source_kind: String,
    lane: Option<String>,
    kind: Option<String>,
    engine: Option<String>,
    profile: Option<String>,
    model: Option<String>,
    probe: String,
    ttfi_ms: Option<f64>,
    first_tool_delta_ms: Option<f64>,
    validated_tool_call_ms: Option<f64>,
    total_latency_ms: Option<f64>,
    tokens_per_second: Option<f64>,
    cache_cold_latency_ms: Option<f64>,
    cache_warm_latency_ms: Option<f64>,
    cache_speedup: Option<f64>,
    cached_tokens: Option<u64>,
    notes: Option<String>,
}

#[derive(Debug, Serialize)]
struct NormalizedRequiredToolTtftMatrixReport {
    status: String,
    rows: Vec<NormalizedRequiredToolTtftMatrixRow>,
}

impl NormalizedRequiredToolTtftMatrixReport {
    fn dry_run() -> Self {
        Self {
            status: "dry_run".to_owned(),
            rows: Vec::new(),
        }
    }
}

#[derive(Debug, Serialize)]
struct NormalizedRequiredToolTtftMatrixRow {
    lane: String,
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_parser: Option<&'static str>,
    schema_variant: &'static str,
    tool_choice_variant: &'static str,
    max_tokens: u32,
    cache_phase: &'static str,
    run_mode: &'static str,
    sample_index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_index: Option<usize>,
    status: String,
    classification: String,
    first_response_byte_ms: Option<u128>,
    first_parsed_sse_chunk_ms: Option<u128>,
    first_tool_delta_ms: Option<u128>,
    tool_finish_ms: Option<u128>,
    validated_tool_call_ms: Option<f64>,
    latency_delta_vs_fastest_lane_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
    stream_stalled_requests_delta: Option<i64>,
    no_progress_failures_delta: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct NormalizedToolRequiredStreamTimingReport {
    status: String,
    attribution: NormalizedToolRequiredStreamAttributionReport,
    lanes: Vec<NormalizedToolRequiredStreamLaneTimingReport>,
}

impl NormalizedToolRequiredStreamTimingReport {
    fn dry_run(lanes: &[NormalizedLaneReport]) -> Self {
        Self {
            status: "dry_run".to_owned(),
            attribution: NormalizedToolRequiredStreamAttributionReport::dry_run(),
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
                    tool_stream_observations: Vec::new(),
                })
                .collect(),
        }
    }
}

#[derive(Debug, Serialize)]
struct NormalizedToolRequiredStreamAttributionReport {
    status: String,
    rows: Vec<NormalizedToolRequiredStreamAttributionRow>,
}

impl NormalizedToolRequiredStreamAttributionReport {
    fn dry_run() -> Self {
        Self {
            status: "dry_run".to_owned(),
            rows: Vec::new(),
        }
    }
}

#[derive(Debug, Serialize)]
struct NormalizedToolRequiredStreamAttributionRow {
    lane: String,
    kind: &'static str,
    pass_count: usize,
    client: NormalizedToolRequiredStreamClientTiming,
    admin_metrics_scope: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    admin_metrics: Option<NormalizedToolRequiredStreamAdminMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    admin_metrics_error: Option<String>,
    first_tool_delta_gap_ms: NormalizedToolRequiredStreamGap,
    decision: &'static str,
}

#[derive(Debug, Serialize)]
struct NormalizedToolRequiredStreamClientTiming {
    first_byte_ms: Option<u128>,
    first_sse_data_ms: Option<u128>,
    first_tool_delta_ms: Option<u128>,
    tool_finish_ms: Option<u128>,
}

#[derive(Debug, Serialize)]
struct NormalizedToolRequiredStreamGap {
    mlx_stream_to_client_ms: Option<f64>,
    kir_first_tool_delta_to_client_ms: Option<f64>,
    validated_tool_call_to_tool_finish_ms: Option<f64>,
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
    tool_stream_observations: Vec<NormalizedToolStreamObservation>,
}

#[derive(Debug, Serialize)]
struct NormalizedToolStreamObservation {
    request_id: String,
    model: String,
    streamed: bool,
    client_first_byte_ms: Option<u128>,
    client_first_sse_data_ms: Option<u128>,
    client_visible_first_tool_delta_ms: Option<u128>,
    client_tool_finish_ms: Option<u128>,
    kir_first_tool_delta_ms: Option<u64>,
    kir_first_tool_delta_after_ttft_ms: Option<u64>,
    tool_argument_assembly_ms: Option<u64>,
    tool_intent_fill_ms: Option<u64>,
    tool_schema_validation_ms: Option<u64>,
    tool_finish_ms: Option<u64>,
    validated_tool_call_ms: Option<u64>,
    mlx_response_headers_ms: Option<u64>,
    mlx_first_upstream_byte_ms: Option<u64>,
    mlx_first_parsed_chunk_ms: Option<u64>,
    mlx_first_tool_delta_ms: Option<u64>,
    mlx_upstream_complete_ms: Option<u64>,
    latency_ms: u64,
}

#[derive(Clone, Debug, Serialize)]
struct NormalizedToolRequiredStreamAdminMetrics {
    first_tool_delta_ms: NormalizedAdminLatencyMetricReport,
    first_tool_delta_after_ttft_ms: NormalizedAdminLatencyMetricReport,
    tool_argument_assembly_ms: NormalizedAdminLatencyMetricReport,
    tool_intent_fill_ms: NormalizedAdminLatencyMetricReport,
    tool_schema_validation_ms: NormalizedAdminLatencyMetricReport,
    tool_finish_ms: NormalizedAdminLatencyMetricReport,
    validated_tool_call_ms: NormalizedAdminLatencyMetricReport,
    mlx_stream_first_upstream_byte_ms: NormalizedAdminLatencyMetricReport,
    mlx_stream_first_parsed_chunk_ms: NormalizedAdminLatencyMetricReport,
    mlx_stream_first_tool_delta_ms: NormalizedAdminLatencyMetricReport,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct NormalizedAdminLatencyMetricReport {
    count_delta: Option<i64>,
    count_after: Option<u64>,
    min_ms_after: Option<f64>,
    max_ms_after: Option<f64>,
    avg_ms_after: Option<f64>,
    window_avg_ms: Option<f64>,
}

#[derive(Debug, Serialize)]
struct NormalizedAggregateSummaryRow {
    lane: String,
    case: &'static str,
    schema_variant: &'static str,
    tool_choice_variant: &'static str,
    max_tokens: u32,
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
