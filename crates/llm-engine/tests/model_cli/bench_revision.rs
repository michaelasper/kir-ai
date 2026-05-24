use super::*;

#[test]
#[cfg(feature = "bench")]
fn qwen_mlx_tool_normalized_prefill_135k_profile_dry_run_defaults_to_prefill_suite() {
    let temp = tempfile::tempdir().expect("tempdir");
    let snapshot = temp
        .path()
        .join("huggingface")
        .join("models--mlx-community--Qwen3.6-35B-A3B-4bit")
        .join("snapshots")
        .join("abcdef1234567890");
    std::fs::create_dir_all(&snapshot).expect("raw HF snapshot dir");

    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args([
            "bench",
            "qwen-mlx-tool-normalized",
            "--dry-run",
            "--sweep-profile",
            "qwen-mlx-prefill-135k",
            "--warmups",
            "0",
            "--samples",
            "1",
            "--context-tokens",
            "128",
            "--snapshot",
        ])
        .arg(&snapshot)
        .output()
        .expect("run qwen mlx prefill 135k profile dry-run");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).expect("json output");
    assert_eq!(value["sweep_profile"], "qwen-mlx-prefill-135k");
    assert_eq!(value["probe_suite"], "prefill_sweep_135k");
    assert_eq!(value["prefill_sweep"]["status"], "dry_run");
    assert_eq!(
        value["cases"]
            .as_array()
            .expect("cases")
            .iter()
            .map(|case| case.as_str().expect("case"))
            .collect::<Vec<_>>(),
        [
            "chat_stream",
            "tool_required_stream",
            "context_recall_stream_135k",
            "warm_prefix_repeated_turn_stream"
        ]
    );
    let lanes = value["lanes"].as_array().expect("lanes array");
    assert_eq!(lanes.len(), 12);
    assert_eq!(lanes[0]["name"], "mlx-prefill-default");
    assert_eq!(lanes[1]["name"], "kir-prefill-default");
    assert_eq!(lanes[10]["endpoint"], "http://127.0.0.1:8085/v1");
    assert_eq!(lanes[11]["endpoint"], "http://127.0.0.1:3005");
    assert_eq!(lanes[2]["mlx_lm_settings"]["mlx_prefill_step_size"], 512);
    assert_eq!(lanes[0]["samples"].as_array().expect("samples").len(), 12);
}

#[test]
#[cfg(feature = "bench")]
fn qwen_mlx_tool_normalized_stable_prefix_smoke_dry_run_reports_selected_plan() {
    let temp = tempfile::tempdir().expect("tempdir");
    let snapshot = temp
        .path()
        .join("huggingface")
        .join("models--mlx-community--Qwen3.6-35B-A3B-4bit")
        .join("snapshots")
        .join("abcdef1234567890");
    std::fs::create_dir_all(&snapshot).expect("raw HF snapshot dir");

    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args([
            "bench",
            "qwen-mlx-tool-normalized",
            "--dry-run",
            "--sweep-profile",
            "qwen-mlx-stable-prefix",
            "--probe-suite",
            "stable-prefix-smoke",
            "--cache-phases",
            "warm_same_prompt",
            "--only-lanes",
            "kir-stable-prefix",
            "--warmups",
            "1",
            "--samples",
            "1",
            "--context-tokens",
            "128",
            "--snapshot",
        ])
        .arg(&snapshot)
        .output()
        .expect("run qwen mlx stable-prefix smoke dry-run");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).expect("json output");
    assert_eq!(value["sweep_profile"], "qwen-mlx-stable-prefix");
    assert_eq!(value["probe_suite"], "stable_prefix_smoke");
    assert_eq!(value["cache_phases"], json!(["warm_same_prompt"]));
    assert_eq!(value["plan_summary"]["probe_count"], 1);
    assert_eq!(value["plan_summary"]["lane_count"], 1);
    assert_eq!(value["plan_summary"]["warmup_requests"], 1);
    assert_eq!(value["plan_summary"]["measured_requests"], 1);
    assert_eq!(value["plan_summary"]["total_http_requests"], 2);
    assert_eq!(value["plan_summary"]["planned_prompt_token_budget"], 256);
    assert_eq!(
        value["plan_summary"]["probes"][0]["case"],
        "warm_prefix_repeated_turn_stream"
    );
    assert_eq!(value["plan_summary"]["lanes"], json!(["kir-stable-prefix"]));

    let lanes = value["lanes"].as_array().expect("lanes array");
    assert_eq!(lanes.len(), 1);
    assert_eq!(lanes[0]["name"], "kir-stable-prefix");
    let planned_requests = lanes[0]["planned_requests"]
        .as_array()
        .expect("planned requests");
    assert_eq!(planned_requests.len(), 2);
    assert_eq!(planned_requests[0]["request_kind"], "warmup");
    assert_eq!(planned_requests[0]["cache_phase"], "warm_same_prompt");
    assert_eq!(planned_requests[0]["warmup_index"], 0);
    assert_eq!(planned_requests[1]["request_kind"], "measured");
    assert_eq!(planned_requests[1]["sample_index"], 0);
    assert_eq!(lanes[0]["samples"].as_array().expect("samples").len(), 1);
}

#[test]
#[cfg(feature = "bench")]
fn qwen_mlx_tool_normalized_max_requests_fails_before_live_http() {
    let temp = tempfile::tempdir().expect("tempdir");
    let snapshot = temp
        .path()
        .join("huggingface")
        .join("models--mlx-community--Qwen3.6-35B-A3B-4bit")
        .join("snapshots")
        .join("abcdef1234567890");
    std::fs::create_dir_all(&snapshot).expect("raw HF snapshot dir");

    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args([
            "bench",
            "qwen-mlx-tool-normalized",
            "--sweep-profile",
            "qwen-mlx-stable-prefix",
            "--probe-suite",
            "stable-prefix-smoke",
            "--cache-phases",
            "warm_same_prompt",
            "--only-lanes",
            "kir-stable-prefix",
            "--warmups",
            "1",
            "--samples",
            "1",
            "--context-tokens",
            "128",
            "--max-requests",
            "1",
            "--connect-timeout-ms",
            "1",
            "--timeout-ms",
            "1",
            "--snapshot",
        ])
        .arg(&snapshot)
        .output()
        .expect("run qwen mlx stable-prefix smoke with request guard");

    assert!(
        !output.status.success(),
        "guard should fail before live run"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--max-requests"),
        "stderr should mention request guard: {stderr}"
    );
    assert!(
        !stderr.contains("Connection refused") && !stderr.contains("error sending request"),
        "guard should fail before HTTP attempt: {stderr}"
    );
}

#[test]
#[cfg(feature = "bench")]
fn qwen_mlx_tool_normalized_repo_revision_uses_kir_ai_checkout_when_run_from_harness_repo() {
    let temp = tempfile::tempdir().expect("tempdir");
    let harness_repo = temp.path().join("llm-server");
    std::fs::create_dir_all(&harness_repo).expect("harness repo dir");
    std::fs::write(harness_repo.join("README.md"), "harness\n").expect("harness file");
    assert!(
        Command::new("git")
            .arg("-C")
            .arg(&harness_repo)
            .arg("init")
            .status()
            .expect("git init")
            .success()
    );
    assert!(
        Command::new("git")
            .arg("-C")
            .arg(&harness_repo)
            .args(["add", "."])
            .status()
            .expect("git add")
            .success()
    );
    assert!(
        Command::new("git")
            .arg("-C")
            .arg(&harness_repo)
            .args([
                "-c",
                "user.name=Benchmark Test",
                "-c",
                "user.email=benchmark@example.com",
                "commit",
                "-m",
                "init harness",
            ])
            .status()
            .expect("git commit")
            .success()
    );
    let harness_sha = String::from_utf8(
        Command::new("git")
            .arg("-C")
            .arg(&harness_repo)
            .args(["rev-parse", "HEAD"])
            .output()
            .expect("harness rev-parse")
            .stdout,
    )
    .expect("harness sha utf8")
    .trim()
    .to_owned();
    let source_repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");
    let source_sha = String::from_utf8(
        Command::new("git")
            .arg("-C")
            .arg(source_repo)
            .args(["rev-parse", "HEAD"])
            .output()
            .expect("source rev-parse")
            .stdout,
    )
    .expect("source sha utf8")
    .trim()
    .to_owned();

    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .current_dir(&harness_repo)
        .args([
            "bench",
            "qwen-mlx-tool-normalized",
            "--dry-run",
            "--warmups",
            "0",
            "--samples",
            "1",
            "--context-tokens",
            "128",
            "--lane",
            "name=direct,endpoint=http://127.0.0.1:8080/v1,model=qwen-loaded,kind=direct_mlx",
        ])
        .output()
        .expect("run qwen mlx tool normalized dry-run from harness repo");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).expect("json output");
    assert_eq!(value["repo_revision"]["commit_sha"], source_sha);
    assert_ne!(value["repo_revision"]["commit_sha"], harness_sha);
}

#[test]
#[cfg(feature = "bench")]
fn qwen_mlx_tool_normalized_repo_revision_accepts_exported_source_metadata() {
    let temp = tempfile::tempdir().expect("tempdir");
    let harness_repo = temp.path().join("llm-server");
    std::fs::create_dir_all(&harness_repo).expect("harness repo dir");
    std::fs::write(harness_repo.join("README.md"), "harness\n").expect("harness file");
    assert!(
        Command::new("git")
            .arg("-C")
            .arg(&harness_repo)
            .arg("init")
            .status()
            .expect("git init")
            .success()
    );
    assert!(
        Command::new("git")
            .arg("-C")
            .arg(&harness_repo)
            .args(["add", "."])
            .status()
            .expect("git add")
            .success()
    );
    assert!(
        Command::new("git")
            .arg("-C")
            .arg(&harness_repo)
            .args([
                "-c",
                "user.name=Benchmark Test",
                "-c",
                "user.email=benchmark@example.com",
                "commit",
                "-m",
                "init harness",
            ])
            .status()
            .expect("git commit")
            .success()
    );
    let harness_sha = String::from_utf8(
        Command::new("git")
            .arg("-C")
            .arg(&harness_repo)
            .args(["rev-parse", "HEAD"])
            .output()
            .expect("harness rev-parse")
            .stdout,
    )
    .expect("harness sha utf8")
    .trim()
    .to_owned();
    let source_repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");
    let source_sha = String::from_utf8(
        Command::new("git")
            .arg("-C")
            .arg(source_repo)
            .args(["rev-parse", "HEAD"])
            .output()
            .expect("source rev-parse")
            .stdout,
    )
    .expect("source sha utf8")
    .trim()
    .to_owned();
    let exported_workspace = harness_repo
        .join(".cache")
        .join("kir-ai-export")
        .join("worktree");
    std::fs::create_dir_all(&exported_workspace).expect("exported workspace dir");

    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .current_dir(&exported_workspace)
        .env("LLM_ENGINE_BENCH_REPO_DIR", &exported_workspace)
        .env("LLM_ENGINE_BENCH_REPO_COMMIT", &source_sha)
        .env("LLM_ENGINE_BENCH_REPO_BRANCH", "main")
        .env("LLM_ENGINE_BENCH_REPO_DIRTY", "false")
        .args([
            "bench",
            "qwen-mlx-tool-normalized",
            "--dry-run",
            "--warmups",
            "0",
            "--samples",
            "1",
            "--context-tokens",
            "128",
            "--lane",
            "name=direct,endpoint=http://127.0.0.1:8080/v1,model=qwen-loaded,kind=direct_mlx",
        ])
        .output()
        .expect("run qwen mlx tool normalized dry-run from exported workspace");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).expect("json output");
    assert_eq!(value["repo_revision"]["commit_sha"], source_sha);
    assert_ne!(value["repo_revision"]["commit_sha"], harness_sha);
    assert_eq!(value["repo_revision"]["branch"], "main");
    assert_eq!(value["repo_revision"]["dirty"], false);
}

#[test]
#[cfg(feature = "bench")]
fn qwen_mlx_tool_normalized_repo_revision_reads_kir_ai_origin_json() {
    let temp = tempfile::tempdir().expect("tempdir");
    let harness_repo = temp.path().join("llm-server");
    std::fs::create_dir_all(&harness_repo).expect("harness repo dir");
    std::fs::write(harness_repo.join("README.md"), "harness\n").expect("harness file");
    assert!(
        Command::new("git")
            .arg("-C")
            .arg(&harness_repo)
            .arg("init")
            .status()
            .expect("git init")
            .success()
    );
    assert!(
        Command::new("git")
            .arg("-C")
            .arg(&harness_repo)
            .args(["add", "."])
            .status()
            .expect("git add")
            .success()
    );
    assert!(
        Command::new("git")
            .arg("-C")
            .arg(&harness_repo)
            .args([
                "-c",
                "user.name=Benchmark Test",
                "-c",
                "user.email=benchmark@example.com",
                "commit",
                "-m",
                "init harness",
            ])
            .status()
            .expect("git commit")
            .success()
    );
    let harness_sha = String::from_utf8(
        Command::new("git")
            .arg("-C")
            .arg(&harness_repo)
            .args(["rev-parse", "HEAD"])
            .output()
            .expect("harness rev-parse")
            .stdout,
    )
    .expect("harness sha utf8")
    .trim()
    .to_owned();
    let source_repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");
    let source_sha = String::from_utf8(
        Command::new("git")
            .arg("-C")
            .arg(source_repo)
            .args(["rev-parse", "HEAD"])
            .output()
            .expect("source rev-parse")
            .stdout,
    )
    .expect("source sha utf8")
    .trim()
    .to_owned();
    let exported_workspace = harness_repo
        .join(".cache")
        .join("kir-ai-export")
        .join("worktree");
    std::fs::create_dir_all(&exported_workspace).expect("exported workspace dir");
    std::fs::write(
        exported_workspace.join(".kir-ai-origin.json"),
        json!({
            "repo_revision": {
                "commit_sha": source_sha,
                "branch": "main",
                "dirty": false
            }
        })
        .to_string(),
    )
    .expect("origin metadata");

    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .current_dir(&exported_workspace)
        .env("LLM_ENGINE_BENCH_REPO_DIR", &exported_workspace)
        .args([
            "bench",
            "qwen-mlx-tool-normalized",
            "--dry-run",
            "--warmups",
            "0",
            "--samples",
            "1",
            "--context-tokens",
            "128",
            "--lane",
            "name=direct,endpoint=http://127.0.0.1:8080/v1,model=qwen-loaded,kind=direct_mlx",
        ])
        .output()
        .expect("run qwen mlx tool normalized dry-run from origin metadata workspace");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).expect("json output");
    assert_eq!(value["repo_revision"]["commit_sha"], source_sha);
    assert_ne!(value["repo_revision"]["commit_sha"], harness_sha);
    assert_eq!(value["repo_revision"]["branch"], "main");
    assert_eq!(value["repo_revision"]["dirty"], false);
}

#[test]
#[cfg(feature = "bench")]
fn qwen_mlx_tool_normalized_repo_revision_does_not_walk_into_parent_harness_repo() {
    let temp = tempfile::tempdir().expect("tempdir");
    let harness_repo = temp.path().join("llm-server");
    std::fs::create_dir_all(&harness_repo).expect("harness repo dir");
    std::fs::write(harness_repo.join("README.md"), "harness\n").expect("harness file");
    assert!(
        Command::new("git")
            .arg("-C")
            .arg(&harness_repo)
            .arg("init")
            .status()
            .expect("git init")
            .success()
    );
    assert!(
        Command::new("git")
            .arg("-C")
            .arg(&harness_repo)
            .args(["add", "."])
            .status()
            .expect("git add")
            .success()
    );
    assert!(
        Command::new("git")
            .arg("-C")
            .arg(&harness_repo)
            .args([
                "-c",
                "user.name=Benchmark Test",
                "-c",
                "user.email=benchmark@example.com",
                "commit",
                "-m",
                "init harness",
            ])
            .status()
            .expect("git commit")
            .success()
    );
    let harness_sha = String::from_utf8(
        Command::new("git")
            .arg("-C")
            .arg(&harness_repo)
            .args(["rev-parse", "HEAD"])
            .output()
            .expect("harness rev-parse")
            .stdout,
    )
    .expect("harness sha utf8")
    .trim()
    .to_owned();
    let exported_workspace = harness_repo
        .join(".cache")
        .join("kir-ai-export")
        .join("worktree");
    std::fs::create_dir_all(&exported_workspace).expect("exported workspace dir");

    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .current_dir(&exported_workspace)
        .env("LLM_ENGINE_BENCH_REPO_DIR", &exported_workspace)
        .args([
            "bench",
            "qwen-mlx-tool-normalized",
            "--dry-run",
            "--warmups",
            "0",
            "--samples",
            "1",
            "--context-tokens",
            "128",
            "--lane",
            "name=direct,endpoint=http://127.0.0.1:8080/v1,model=qwen-loaded,kind=direct_mlx",
        ])
        .output()
        .expect("run qwen mlx tool normalized dry-run from exported workspace");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).expect("json output");
    assert_ne!(value["repo_revision"]["commit_sha"], harness_sha);
}
