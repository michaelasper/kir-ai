use llm_hub::{HubFile, HubRepoId, ModelProfile, ModelStore, build_download_plan};
use serde_json::Value;
use std::process::{Command, Stdio};
use std::time::Duration;

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
async fn model_list_outputs_promoted_snapshots() {
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
    assert_eq!(value["repo_id"], "Qwen/Qwen3.6-35B-A3B");
    assert_eq!(value["files"], 1);
    assert_eq!(value["total_bytes"], 2);
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
    assert_eq!(value["verified_files"], 1);
    assert_eq!(value["verified_bytes"], 2);
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
