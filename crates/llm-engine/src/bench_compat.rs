use anyhow::Context;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};

const BENCH_BIN_ENV: &str = "LLM_ENGINE_BENCH_BIN";

pub(crate) fn run(args: Vec<String>) -> anyhow::Result<()> {
    let status = if let Some(bench_bin) = std::env::var_os(BENCH_BIN_ENV) {
        let mut command = Command::new(bench_bin);
        command.args(&args);
        run_child(command, "llm-bench")
    } else if let Some(workspace_root) = workspace_root() {
        run_with_cargo(&workspace_root, &args)
    } else if let Some(bench_bin) = sibling_bench_binary()? {
        let mut command = Command::new(bench_bin);
        command.args(&args);
        run_child(command, "llm-bench")
    } else {
        anyhow::bail!(
            "locate kir-ai workspace root for llm-bench; set {BENCH_BIN_ENV} to a built llm-bench binary"
        )
    }?;

    if status.success() {
        return Ok(());
    }

    if let Some(code) = status.code() {
        std::process::exit(code);
    }

    anyhow::bail!("llm-bench terminated without an exit code");
}

fn sibling_bench_binary() -> anyhow::Result<Option<PathBuf>> {
    let current_exe = std::env::current_exe().context("resolve llm-engine executable path")?;
    let Some(bin_dir) = current_exe.parent() else {
        return Ok(None);
    };
    let candidate = bin_dir.join(format!("llm-bench{}", std::env::consts::EXE_SUFFIX));
    Ok(candidate.is_file().then_some(candidate))
}

fn run_with_cargo(workspace_root: &Path, args: &[String]) -> anyhow::Result<ExitStatus> {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let mut command = Command::new(cargo);
    command
        .current_dir(workspace_root)
        .args([
            "run",
            "--quiet",
            "-p",
            "llm-bench",
            "--features",
            "bench-server",
            "--",
        ])
        .args(args);
    run_child(command, "cargo run -p llm-bench --features bench-server")
}

fn run_child(mut command: Command, program: &str) -> anyhow::Result<ExitStatus> {
    command
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .with_context(|| format!("run {program}"))
}

fn workspace_root() -> Option<PathBuf> {
    std::env::current_dir()
        .ok()
        .and_then(|cwd| find_workspace_root(&cwd))
        .or_else(|| find_workspace_root(Path::new(env!("CARGO_MANIFEST_DIR"))))
}

fn find_workspace_root(start: &Path) -> Option<PathBuf> {
    start
        .ancestors()
        .find(|candidate| candidate.join("crates/llm-bench/Cargo.toml").is_file())
        .map(Path::to_path_buf)
}
