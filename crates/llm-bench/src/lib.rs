#[cfg(not(feature = "bench-server"))]
pub use llm_util::defaults::DEFAULT_MODEL_ID;

#[cfg(not(feature = "bench-server"))]
#[allow(dead_code)]
mod cli;

#[cfg(feature = "bench-server")]
include!("bench_server.rs");

#[cfg(not(feature = "bench-server"))]
#[allow(dead_code)]
pub(crate) fn flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find_map(|window| (window[0] == flag).then_some(window[1].as_str()))
}

#[cfg(not(feature = "bench-server"))]
#[allow(dead_code)]
pub(crate) fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|arg| arg == flag)
}

#[cfg(not(feature = "bench-server"))]
pub async fn run_bench_command(args: Vec<String>) -> anyhow::Result<()> {
    let Some(subcommand) = args.first() else {
        cli::print_bench_help();
        return Ok(());
    };
    if subcommand == "--help" || subcommand == "-h" {
        cli::print_bench_help();
        return Ok(());
    }
    match subcommand.as_str() {
        "qwen-long-context" | "qwen-mlx-tool-normalized" => {
            anyhow::bail!("llm-bench subcommand `{subcommand}` requires the `bench-server` feature")
        }
        other => anyhow::bail!("unknown bench subcommand `{other}`"),
    }
}
