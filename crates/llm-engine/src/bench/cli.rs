pub(super) fn print_bench_help() {
    println!(
        "\
Usage: llm-engine bench qwen-long-context [OPTIONS]

Options:
  --endpoint <url>                    OpenAI-compatible server base URL
  --model <id>                        Model id to send in requests [default: {}]
  --snapshot <path>                   Qwen snapshot path with tokenizer.json and manifest
  --lane <spec>                       Named lane: name=<id>,endpoint=<url>,snapshot=<path>[,model=<id>]
  --profile <135k|200k|256k|all>      Benchmark profile [default: 135k]
  --baseline <path>                   Previous trace JSON for same hardware/model comparison
  --output <path>                     Write the trace JSON to a file as well as stdout
  --max-tokens <n>                    Completion token limit per request [default: 128]
  --admin-token <token>               Optional bearer token for lane /admin/metrics snapshots
  --timeout-ms <n>                    Whole request timeout [default: 1800000]
  --connect-timeout-ms <n>            HTTP connect timeout [default: 10000]
  --latency-regression-threshold <f>  Allowed latency increase over baseline [default: 0.20]
  --dry-run                           Print the exact gate plan without HTTP requests
  -h, --help                          Print help",
        crate::DEFAULT_MODEL_ID
    );
}

pub(super) fn flag_values<'a>(args: &'a [String], flag: &str) -> Vec<&'a str> {
    args.windows(2)
        .filter_map(|window| (window[0] == flag).then_some(window[1].as_str()))
        .collect()
}

pub(super) fn normalize_endpoint(endpoint: &str) -> String {
    endpoint.trim_end_matches('/').to_owned()
}

pub(super) fn parse_u64_flag(args: &[String], flag: &str, default: u64) -> anyhow::Result<u64> {
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

pub(super) fn parse_u32_flag(args: &[String], flag: &str, default: u32) -> anyhow::Result<u32> {
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

pub(super) fn parse_f64_flag(args: &[String], flag: &str, default: f64) -> anyhow::Result<f64> {
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
use crate::flag_value;
use anyhow::Context;
