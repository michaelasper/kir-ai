use super::super::super::{cli::flag_values, cli::normalize_endpoint};
use super::super::{CachePhase, NormalizedProbeSuite};
use super::profiles::expand_sweep_profile;
use super::{
    DEFAULT_CONCURRENT_REQUESTS, DEFAULT_CONCURRENT_SAMPLES, DEFAULT_CONTEXT_TOKENS,
    DEFAULT_SAMPLES, DEFAULT_WARMUPS, MlxLmSettings, NormalizedLaneConfig, NormalizedLaneKind,
    NormalizedModelAddressing, NormalizedRunConfig, NormalizedSweepProfile,
    NormalizedTemplatePolicy,
};
use crate::{MlxToolParserMode, flag_value, has_flag};
use anyhow::{Context, anyhow};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

pub(in crate::qwen_mlx_tool) fn parse_lane_specs(
    args: &[String],
) -> anyhow::Result<Vec<NormalizedLaneConfig>> {
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

pub(in crate::qwen_mlx_tool) fn parse_sweep_profile_flag(
    args: &[String],
) -> anyhow::Result<Option<NormalizedSweepProfile>> {
    let profiles = flag_values(args, "--sweep-profile");
    match profiles.as_slice() {
        [] => Ok(None),
        [profile] => NormalizedSweepProfile::parse(profile).map(Some),
        _ => anyhow::bail!("--sweep-profile may only be provided once"),
    }
}

pub(in crate::qwen_mlx_tool) fn parse_probe_suite_flag(
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

pub(in crate::qwen_mlx_tool) fn default_run_config_for_probe_suite(
    suite: NormalizedProbeSuite,
) -> NormalizedRunConfig {
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
pub(in crate::qwen_mlx_tool) fn parse_lane_spec(
    spec: &str,
) -> anyhow::Result<NormalizedLaneConfig> {
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

pub(in crate::qwen_mlx_tool) fn parse_cache_phases_flag(
    args: &[String],
) -> anyhow::Result<Vec<CachePhase>> {
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

pub(in crate::qwen_mlx_tool) fn parse_count_flag(
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

pub(in crate::qwen_mlx_tool) fn parse_optional_count_flag(
    args: &[String],
    flag: &str,
) -> anyhow::Result<Option<usize>> {
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

pub(in crate::qwen_mlx_tool) fn parse_millis_flag(
    args: &[String],
    flag: &str,
    default: u64,
) -> anyhow::Result<u64> {
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
