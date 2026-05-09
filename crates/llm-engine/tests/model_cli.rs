use llm_hub::{HubFile, HubRepoId, ModelProfile, ModelStore, build_download_plan};
use serde_json::Value;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

fn runnable_qwen_files() -> Vec<HubFile> {
    vec![
        HubFile::new("config.json", 2, Some("\"cfg\"")),
        HubFile::new("tokenizer.json", 2, Some("\"tok\"")),
        HubFile::new("model.safetensors", 4, Some("\"weights\"")),
    ]
}

async fn write_runnable_qwen_files(snapshot_path: &Path) {
    tokio::fs::write(snapshot_path.join("config.json"), "{}")
        .await
        .expect("config");
    tokio::fs::write(snapshot_path.join("tokenizer.json"), "{}")
        .await
        .expect("tokenizer");
    tokio::fs::write(snapshot_path.join("model.safetensors"), b"data")
        .await
        .expect("weights");
}

#[test]
fn long_context_bench_dry_run_defines_qwen_promotion_profiles() {
    let temp = tempfile::tempdir().expect("tempdir");
    let snapshot = temp.path().join("snapshot");
    let baseline = temp.path().join("baseline.json");
    let trace = temp.path().join("trace.json");

    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args([
            "bench",
            "qwen-long-context",
            "--dry-run",
            "--profile",
            "all",
            "--model",
            "local-qwen36",
            "--snapshot",
        ])
        .arg(&snapshot)
        .args(["--baseline"])
        .arg(&baseline)
        .args(["--output"])
        .arg(&trace)
        .output()
        .expect("run qwen long-context bench dry-run");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).expect("json output");
    let trace_value: Value =
        serde_json::from_slice(&std::fs::read(&trace).expect("trace output file"))
            .expect("trace JSON output");
    assert_eq!(trace_value["gate"], "qwen-long-context");
    assert_eq!(value["gate"], "qwen-long-context");
    assert_eq!(value["status"], "dry_run");
    assert_eq!(value["model"]["id"], "local-qwen36");
    assert_eq!(
        value["model"]["snapshot_path"],
        snapshot.display().to_string()
    );
    assert_eq!(value["baseline"]["path"], baseline.display().to_string());
    assert_eq!(value["trace_output_path"], trace.display().to_string());
    assert_eq!(value["hardware"]["os"], std::env::consts::OS);
    assert_eq!(value["hardware"]["arch"], std::env::consts::ARCH);
    assert_eq!(value["cache_policy"]["cache_layout"], "shared-prefix-v1");
    let lanes = value["lanes"].as_array().expect("lanes array");
    assert_eq!(lanes.len(), 1, "lanes: {lanes:?}");
    assert_eq!(lanes[0]["name"], "primary");
    assert_eq!(lanes[0]["status"], "dry_run");
    assert_eq!(lanes[0]["model"]["id"], "local-qwen36");
    assert_eq!(
        lanes[0]["model"]["snapshot_path"],
        snapshot.display().to_string()
    );

    let profiles = value["profiles"].as_array().expect("profiles array");
    assert_eq!(profiles.len(), 2, "profiles: {profiles:?}");
    let promotion = profiles
        .iter()
        .find(|profile| profile["name"] == "qwen-135k-promotion")
        .expect("135K promotion profile");
    assert_eq!(promotion["target_tokens"], 135_000);
    assert_eq!(promotion["release_blocking"], true);
    let frontier = profiles
        .iter()
        .find(|profile| profile["name"] == "qwen-200k-characterization")
        .expect("200K characterization profile");
    assert_eq!(frontier["target_tokens"], 200_000);
    assert_eq!(frontier["release_blocking"], false);

    let case_names = promotion["cases"]
        .as_array()
        .expect("promotion cases")
        .iter()
        .map(|case| case["name"].as_str().expect("case name"))
        .collect::<Vec<_>>();
    assert_eq!(
        case_names,
        [
            "plain-recall",
            "json-object-recall",
            "required-tool-recall",
            "streamed-required-tool-recall",
            "multi-turn-lifecycle"
        ]
    );
    let required_tool = promotion["cases"]
        .as_array()
        .expect("promotion cases")
        .iter()
        .find(|case| case["name"] == "required-tool-recall")
        .expect("required tool case");
    let streamed_tool = promotion["cases"]
        .as_array()
        .expect("promotion cases")
        .iter()
        .find(|case| case["name"] == "streamed-required-tool-recall")
        .expect("streamed required tool case");
    assert!(
        required_tool["response_contract"]
            .as_str()
            .expect("required tool response contract")
            .contains("tool_calls")
    );
    assert!(
        streamed_tool["response_contract"]
            .as_str()
            .expect("streamed tool response contract")
            .contains("tool_calls")
    );
}

#[test]
fn long_context_bench_dry_run_accepts_named_backend_lanes() {
    let temp = tempfile::tempdir().expect("tempdir");
    let native_snapshot = temp.path().join("native");
    let mlx_snapshot = temp.path().join("mlx");

    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args([
            "bench",
            "qwen-long-context",
            "--dry-run",
            "--profile",
            "135k",
            "--lane",
        ])
        .arg(format!(
            "name=native-metal,endpoint=http://127.0.0.1:3101,snapshot={},model=local-native",
            native_snapshot.display()
        ))
        .args(["--lane"])
        .arg(format!(
            "name=mlx,endpoint=http://127.0.0.1:3102,snapshot={},model=local-mlx",
            mlx_snapshot.display()
        ))
        .output()
        .expect("run qwen long-context lane dry-run");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).expect("json output");
    let lanes = value["lanes"].as_array().expect("lanes array");
    assert_eq!(lanes.len(), 2, "lanes: {lanes:?}");
    assert_eq!(lanes[0]["name"], "native-metal");
    assert_eq!(lanes[0]["model"]["id"], "local-native");
    assert_eq!(lanes[0]["model"]["endpoint"], "http://127.0.0.1:3101");
    assert_eq!(
        lanes[0]["model"]["snapshot_path"],
        native_snapshot.display().to_string()
    );
    assert_eq!(lanes[1]["name"], "mlx");
    assert_eq!(lanes[1]["model"]["id"], "local-mlx");
    assert_eq!(value["model"]["id"], "local-native");
}

#[test]
fn serve_help_prints_without_backend_validation() {
    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args(["serve", "--help"])
        .output()
        .expect("run serve help");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--addr"), "stdout: {stdout}");
    assert!(stdout.contains("--snapshot"), "stdout: {stdout}");
    assert!(stdout.contains("--snapshot-alias"), "stdout: {stdout}");
    assert!(stdout.contains("--model-alias"), "stdout: {stdout}");
    assert!(stdout.contains("--model-id"), "stdout: {stdout}");
    assert!(
        stdout.contains("--loader <native-metal|mlx>"),
        "stdout: {stdout}"
    );
    assert!(
        stdout.contains("--backend <native-metal|mlx>"),
        "stdout: {stdout}"
    );
    assert!(
        stdout.contains("--family <qwen|deep_seek|gemma>"),
        "stdout: {stdout}"
    );
    assert!(
        stdout.contains(
            "Qwen and Gemma are serveable through MLX; DeepSeek is recognized but deferred"
        ),
        "stdout: {stdout}"
    );
    assert!(
        stdout.contains("--deterministic-test-backend"),
        "stdout: {stdout}"
    );
    assert!(stdout.contains("--max-new-tokens <n>"), "stdout: {stdout}");
    assert!(
        stdout.contains("[default: 256]"),
        "stdout should document the usable native generation default: {stdout}"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("requires --snapshot"),
        "stderr unexpectedly validated backend: {stderr}"
    );
}

#[tokio::test]
async fn serve_without_snapshot_requires_explicit_backend() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args(["serve", "--addr", "127.0.0.1:0"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn serve");
    let deadline = std::time::Instant::now() + Duration::from_secs(1);
    while std::time::Instant::now() < deadline {
        if child.try_wait().expect("poll serve").is_some() {
            let output = child.wait_with_output().expect("collect serve output");
            assert!(!output.status.success());
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(stderr.contains("--snapshot"), "stderr: {stderr}");
            assert!(
                stderr.contains("--deterministic-test-backend"),
                "stderr: {stderr}"
            );
            return;
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    child.kill().expect("kill hanging serve");
    let _ = child.wait();
    panic!("serve bound the deterministic backend instead of failing without --snapshot");
}

#[tokio::test]
async fn serve_with_missing_snapshot_alias_fails_before_binding() {
    let temp = tempfile::tempdir().expect("tempdir");
    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args([
            "serve",
            "--addr",
            "127.0.0.1:0",
            "--snapshot-alias",
            "missing-model",
            "--loader",
            "mlx",
            "--model-home",
        ])
        .arg(temp.path())
        .output()
        .expect("run serve with missing snapshot alias");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("model alias `missing-model`"),
        "stderr: {stderr}"
    );
}

#[tokio::test]
async fn serve_rejects_raw_mlx_snapshot_without_family_before_binding() {
    let temp = tempfile::tempdir().expect("tempdir");
    let snapshot = temp.path().join("raw-mlx");
    tokio::fs::create_dir_all(&snapshot)
        .await
        .expect("snapshot dir");

    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args(["serve", "--addr", "127.0.0.1:0", "--snapshot"])
        .arg(&snapshot)
        .args(["--loader", "mlx"])
        .output()
        .expect("run serve with manifestless MLX snapshot");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("snapshot loader `mlx` requires model family metadata"),
        "stderr: {stderr}"
    );
}

#[tokio::test]
async fn serve_rejects_deferred_deepseek_mlx_family_before_binding() {
    let temp = tempfile::tempdir().expect("tempdir");
    let snapshot = temp.path().join("raw-deepseek");
    tokio::fs::create_dir_all(&snapshot)
        .await
        .expect("snapshot dir");

    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args(["serve", "--addr", "127.0.0.1:0", "--snapshot"])
        .arg(&snapshot)
        .args(["--loader", "mlx", "--family", "deep_seek"])
        .output()
        .expect("run serve with deferred MLX family");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("model family `deep_seek` is recognized but not serveable yet"),
        "stderr: {stderr}"
    );
}

#[tokio::test]
async fn serve_rejects_invalid_hub_endpoint_without_panic() {
    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args([
            "serve",
            "--addr",
            "127.0.0.1:0",
            "--deterministic-test-backend",
            "--hub-endpoint",
            "not a url",
        ])
        .output()
        .expect("run serve with invalid hub endpoint");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("invalid hub endpoint"), "stderr: {stderr}");
    assert!(!stderr.contains("panicked"), "stderr: {stderr}");
}

#[tokio::test]
async fn model_list_outputs_promoted_snapshots() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = ModelStore::new(temp.path());
    let plan = build_download_plan(
        HubRepoId::model("Qwen/Qwen3.6-35B-A3B").expect("repo id"),
        "main",
        "0123456789abcdef0123456789abcdef01234567",
        ModelProfile::qwen36_safetensors_bf16(),
        runnable_qwen_files(),
        &[],
    )
    .expect("plan builds");
    let snapshot_path = store.snapshot_path(&plan);
    tokio::fs::create_dir_all(&snapshot_path)
        .await
        .expect("snapshot dir");
    write_runnable_qwen_files(&snapshot_path).await;
    store
        .verify_existing_snapshot(&plan)
        .await
        .expect("snapshot verifies");

    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args(["model", "list", "--model-home"])
        .arg(temp.path())
        .output()
        .expect("run model list");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).expect("json output");
    assert_eq!(value["snapshots"][0]["repo_id"], "Qwen/Qwen3.6-35B-A3B");
    assert_eq!(
        value["snapshots"][0]["resolved_commit"],
        "0123456789abcdef0123456789abcdef01234567"
    );
    assert_eq!(
        value["snapshots"][0]["manifest_digest"]
            .as_str()
            .unwrap()
            .len(),
        64
    );
}

#[tokio::test]
async fn model_inspect_outputs_snapshot_manifest_summary() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = ModelStore::new(temp.path());
    let plan = build_download_plan(
        HubRepoId::model("Qwen/Qwen3.6-35B-A3B").expect("repo id"),
        "main",
        "0123456789abcdef0123456789abcdef01234567",
        ModelProfile::qwen36_safetensors_bf16(),
        runnable_qwen_files(),
        &[],
    )
    .expect("plan builds");
    let snapshot_path = store.snapshot_path(&plan);
    tokio::fs::create_dir_all(&snapshot_path)
        .await
        .expect("snapshot dir");
    write_runnable_qwen_files(&snapshot_path).await;
    store
        .verify_existing_snapshot(&plan)
        .await
        .expect("snapshot verifies");

    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args(["model", "inspect"])
        .arg(&snapshot_path)
        .output()
        .expect("run model inspect");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).expect("json output");
    assert_eq!(value["status"], "ready");
    assert_eq!(value["repo_id"], "Qwen/Qwen3.6-35B-A3B");
    assert_eq!(value["files"], 3);
    assert_eq!(value["total_bytes"], 8);
}

#[tokio::test]
async fn model_verify_outputs_snapshot_integrity_summary() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = ModelStore::new(temp.path());
    let plan = build_download_plan(
        HubRepoId::model("Qwen/Qwen3.6-35B-A3B").expect("repo id"),
        "main",
        "0123456789abcdef0123456789abcdef01234567",
        ModelProfile::qwen36_safetensors_bf16(),
        runnable_qwen_files(),
        &[],
    )
    .expect("plan builds");
    let snapshot_path = store.snapshot_path(&plan);
    tokio::fs::create_dir_all(&snapshot_path)
        .await
        .expect("snapshot dir");
    write_runnable_qwen_files(&snapshot_path).await;
    store
        .verify_existing_snapshot(&plan)
        .await
        .expect("snapshot verifies");

    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args(["model", "verify"])
        .arg(&snapshot_path)
        .output()
        .expect("run model verify");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).expect("json output");
    assert_eq!(value["status"], "ok");
    assert_eq!(value["verified_files"], 3);
    assert_eq!(value["verified_bytes"], 8);
}

#[tokio::test]
async fn model_prune_dry_run_outputs_snapshot_usage_without_deleting() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = ModelStore::new(temp.path());
    let plan = build_download_plan(
        HubRepoId::model("Qwen/Qwen3.6-35B-A3B").expect("repo id"),
        "main",
        "0123456789abcdef0123456789abcdef01234567",
        ModelProfile::qwen36_safetensors_bf16(),
        vec![HubFile::new("config.json", 2, Some("\"cfg\""))],
        &[],
    )
    .expect("plan builds");
    let snapshot_path = store.snapshot_path(&plan);
    tokio::fs::create_dir_all(&snapshot_path)
        .await
        .expect("snapshot dir");
    tokio::fs::write(snapshot_path.join("config.json"), "{}")
        .await
        .expect("config");
    store
        .verify_existing_snapshot(&plan)
        .await
        .expect("snapshot verifies");

    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args(["model", "prune", "--dry-run", "--model-home"])
        .arg(temp.path())
        .output()
        .expect("run model prune dry-run");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).expect("json output");
    assert_eq!(value["dry_run"], true);
    assert_eq!(value["snapshots"], 1);
    assert_eq!(value["total_bytes"], 2);
    assert_eq!(value["reclaimable_bytes"], 0);
    assert!(snapshot_path.join("config.json").is_file());
}

#[tokio::test]
async fn model_prune_confirm_deletes_same_candidates_as_dry_run() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = ModelStore::new(temp.path());
    let plan = build_download_plan(
        HubRepoId::model("Qwen/Qwen3.6-35B-A3B").expect("repo id"),
        "main",
        "abcdefabcdefabcdefabcdefabcdefabcdefabcd",
        ModelProfile::qwen36_safetensors_bf16(),
        vec![HubFile::new("config.json", 2, Some("\"cfg\""))],
        &[],
    )
    .expect("plan builds");
    let snapshot_path = store.snapshot_path(&plan);
    tokio::fs::create_dir_all(&snapshot_path)
        .await
        .expect("snapshot dir");
    tokio::fs::write(snapshot_path.join("config.json"), "{}")
        .await
        .expect("config");
    store
        .verify_existing_snapshot(&plan)
        .await
        .expect("snapshot verifies");
    let old = chrono::DateTime::parse_from_rfc3339("2026-04-01T00:00:00Z")
        .expect("fixed time")
        .with_timezone(&chrono::Utc);
    ModelStore::mark_snapshot_used_at(&snapshot_path, old)
        .await
        .expect("usage recorded");

    let dry_run = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args([
            "model",
            "prune",
            "--dry-run",
            "--older-than-days",
            "7",
            "--keep-min-per-profile",
            "0",
            "--now",
            "2026-05-08T00:00:00Z",
            "--model-home",
        ])
        .arg(temp.path())
        .output()
        .expect("run model prune dry-run");

    assert!(
        dry_run.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&dry_run.stderr)
    );
    let dry_run_value: Value = serde_json::from_slice(&dry_run.stdout).expect("json output");
    let dry_run_candidates = dry_run_value["candidates"]
        .as_array()
        .expect("candidate array");
    assert_eq!(dry_run_value["dry_run"], true);
    assert_eq!(dry_run_candidates.len(), 1);
    assert_eq!(
        dry_run_candidates[0]["path"],
        snapshot_path.display().to_string()
    );
    assert!(snapshot_path.join("config.json").is_file());

    let destructive = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args([
            "model",
            "prune",
            "--confirm-delete",
            "--older-than-days",
            "7",
            "--keep-min-per-profile",
            "0",
            "--now",
            "2026-05-08T00:00:00Z",
            "--model-home",
        ])
        .arg(temp.path())
        .output()
        .expect("run model prune destructive");

    assert!(
        destructive.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&destructive.stderr)
    );
    let destructive_value: Value =
        serde_json::from_slice(&destructive.stdout).expect("json output");
    let destructive_candidates = destructive_value["candidates"]
        .as_array()
        .expect("candidate array");
    assert_eq!(destructive_value["dry_run"], false);
    assert_eq!(destructive_candidates.len(), dry_run_candidates.len());
    assert_eq!(
        destructive_candidates[0]["path"],
        dry_run_candidates[0]["path"]
    );
    assert_eq!(
        destructive_value["deleted"]
            .as_array()
            .expect("deleted")
            .len(),
        1
    );
    assert!(!snapshot_path.exists());
}

#[tokio::test]
async fn model_list_and_inspect_show_quarantined_snapshots() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = ModelStore::new(temp.path());
    let plan = build_download_plan(
        HubRepoId::model("Qwen/Qwen3.6-35B-A3B").expect("repo id"),
        "main",
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        ModelProfile::qwen36_safetensors_bf16(),
        vec![HubFile::new("config.json", 2, Some("\"cfg\""))],
        &[],
    )
    .expect("plan builds");
    let snapshot_path = store.snapshot_path(&plan);
    tokio::fs::create_dir_all(&snapshot_path)
        .await
        .expect("snapshot dir");
    tokio::fs::write(snapshot_path.join("config.json"), "{}")
        .await
        .expect("config");
    store
        .verify_existing_snapshot(&plan)
        .await
        .expect("snapshot verifies");
    let quarantined = store
        .quarantine_snapshot(&snapshot_path, "test corruption")
        .await
        .expect("quarantined");

    let list = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args(["model", "list", "--model-home"])
        .arg(temp.path())
        .output()
        .expect("run model list");

    assert!(
        list.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&list.stderr)
    );
    let list_value: Value = serde_json::from_slice(&list.stdout).expect("json output");
    assert_eq!(
        list_value["snapshots"].as_array().expect("snapshots").len(),
        0
    );
    assert_eq!(
        list_value["quarantined_snapshots"]
            .as_array()
            .expect("quarantined")
            .len(),
        1
    );
    assert_eq!(
        list_value["quarantined_snapshots"][0]["path"],
        quarantined.path.display().to_string()
    );

    let inspect = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args(["model", "inspect"])
        .arg(&quarantined.path)
        .output()
        .expect("run model inspect");

    assert!(
        inspect.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&inspect.stderr)
    );
    let inspect_value: Value = serde_json::from_slice(&inspect.stdout).expect("json output");
    assert_eq!(inspect_value["status"], "quarantined");
    assert_eq!(
        inspect_value["original_path"],
        snapshot_path.display().to_string()
    );
}
