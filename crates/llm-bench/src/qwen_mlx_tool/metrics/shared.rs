use super::*;

pub(in crate::qwen_mlx_tool) fn percentile_for_samples(
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

pub(in crate::qwen_mlx_tool) fn lane_samples(
    lane: &NormalizedLaneReport,
) -> impl Iterator<Item = &NormalizedSampleReport> {
    lane.samples.iter().chain(lane.concurrent_samples.iter())
}

pub(in crate::qwen_mlx_tool) fn sample_matches_probe(
    sample: &NormalizedSampleReport,
    probe: NormalizedProbePlan,
) -> bool {
    sample.case == probe.case.name()
        && sample.schema_variant == probe.schema_variant.name()
        && sample.tool_choice_variant == probe.tool_choice_variant.name()
        && sample.max_tokens == probe.max_tokens
}

pub(in crate::qwen_mlx_tool) fn fastest_lane_for(
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

pub(in crate::qwen_mlx_tool) fn percentile_latency(
    sorted_values: &[u128],
    percentile: f64,
) -> Option<u128> {
    if sorted_values.is_empty() {
        return None;
    }
    let last_index = sorted_values.len() - 1;
    let index = ((last_index as f64) * percentile).round() as usize;
    sorted_values.get(index).copied()
}

pub(in crate::qwen_mlx_tool) fn average_u64(values: impl Iterator<Item = u64>) -> Option<f64> {
    let mut count = 0u64;
    let mut total = 0u64;
    for value in values {
        count += 1;
        total += value;
    }
    (count > 0).then_some(total as f64 / count as f64)
}

pub(in crate::qwen_mlx_tool) fn average_f64(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut count = 0u64;
    let mut total = 0.0;
    for value in values {
        count += 1;
        total += value;
    }
    (count > 0).then_some(total / count as f64)
}

pub(in crate::qwen_mlx_tool) fn probe_case_names(
    probes: &[NormalizedProbePlan],
) -> Vec<&'static str> {
    unique_probe_names(probes.iter().map(|probe| probe.case.name()))
}

pub(in crate::qwen_mlx_tool) fn probe_schema_variant_names(
    probes: &[NormalizedProbePlan],
) -> Vec<&'static str> {
    unique_probe_names(probes.iter().map(|probe| probe.schema_variant.name()))
}

pub(in crate::qwen_mlx_tool) fn probe_tool_choice_variant_names(
    probes: &[NormalizedProbePlan],
) -> Vec<&'static str> {
    unique_probe_names(probes.iter().map(|probe| probe.tool_choice_variant.name()))
}

#[cfg(test)]
pub(in crate::qwen_mlx_tool) fn unique_probe_max_tokens(
    probes: &[NormalizedProbePlan],
) -> Vec<u32> {
    let mut unique = Vec::new();
    for max_tokens in probes.iter().map(|probe| probe.max_tokens) {
        if !unique.contains(&max_tokens) {
            unique.push(max_tokens);
        }
    }
    unique
}

pub(in crate::qwen_mlx_tool) fn unique_probe_names(
    names: impl Iterator<Item = &'static str>,
) -> Vec<&'static str> {
    let mut unique = Vec::new();
    for name in names {
        if !unique.contains(&name) {
            unique.push(name);
        }
    }
    unique
}

pub(in crate::qwen_mlx_tool) async fn load_engine_db_baseline_export(
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

pub(in crate::qwen_mlx_tool) fn benchmark_repo_dir() -> PathBuf {
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

pub(in crate::qwen_mlx_tool) fn env_string(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

pub(in crate::qwen_mlx_tool) fn env_bool(name: &str) -> Option<bool> {
    let value = env_string(name)?;
    parse_bool_text(&value)
}

pub(in crate::qwen_mlx_tool) fn origin_string(value: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| value.get(*key).and_then(Value::as_str))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

pub(in crate::qwen_mlx_tool) fn origin_bool(value: &Value, keys: &[&str]) -> Option<bool> {
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

pub(in crate::qwen_mlx_tool) fn parse_bool_text(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "dirty" => Some(true),
        "0" | "false" | "no" | "clean" => Some(false),
        _ => None,
    }
}

pub(in crate::qwen_mlx_tool) fn git_output_in_dir(dir: &Path, args: &[&str]) -> Option<String> {
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

pub(in crate::qwen_mlx_tool) fn benchmark_repo_git_root(dir: &Path) -> Option<PathBuf> {
    let top_level = git_output_in_dir(dir, &["rev-parse", "--show-toplevel"])?;
    PathBuf::from(top_level).canonicalize().ok()
}

pub(in crate::qwen_mlx_tool) fn is_benchmark_git_root(dir: &Path) -> bool {
    let Ok(dir) = dir.canonicalize() else {
        return false;
    };
    benchmark_repo_git_root(&dir).is_some_and(|root| root == dir)
}

pub(in crate::qwen_mlx_tool) fn git_output(args: &[&str]) -> Option<String> {
    let dir = benchmark_repo_dir();
    if !is_benchmark_git_root(&dir) {
        return None;
    }
    git_output_in_dir(&dir, args)
}

pub(in crate::qwen_mlx_tool) fn git_dirty() -> bool {
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

pub(in crate::qwen_mlx_tool) async fn write_and_print_normalized_report(
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
