use super::*;

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
async fn model_verify_outputs_runnable_mode() {
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
    assert_eq!(value["verification_mode"], "runnable");
    assert_eq!(value["verified_files"], 3);
    assert_eq!(value["verified_bytes"], 8);
    assert_eq!(value["snapshot_path"], snapshot_path.display().to_string());
}

#[tokio::test]
async fn model_verify_rejects_metadata_only_snapshot() {
    let temp = tempfile::tempdir().expect("tempdir");
    let snapshot_path = write_verified_metadata_only_snapshot(temp.path()).await;

    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args(["model", "verify"])
        .arg(&snapshot_path)
        .output()
        .expect("run model verify");

    assert!(!output.status.success(), "verify unexpectedly succeeded");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("contains no weight files"),
        "stderr: {stderr}"
    );
    assert!(stderr.contains("--metadata-only"), "stderr: {stderr}");
}

#[tokio::test]
async fn model_list_uses_fast_readiness_without_rehashing_weights() {
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
    let weight_path = snapshot_path.join("model.safetensors");
    let weight_modified_at = std::fs::metadata(&weight_path)
        .expect("weight metadata")
        .modified()
        .expect("weight modified time");
    tokio::fs::write(snapshot_path.join("model.safetensors"), b"xxxx")
        .await
        .expect("same-size weight corruption");
    std::fs::OpenOptions::new()
        .write(true)
        .open(&weight_path)
        .expect("reopen corrupted weight")
        .set_times(std::fs::FileTimes::new().set_modified(weight_modified_at))
        .expect("restore weight modified time");

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
    assert_eq!(value["snapshots"].as_array().expect("snapshots").len(), 1);
    assert_eq!(
        value["snapshots"][0]["path"],
        snapshot_path.display().to_string()
    );
    assert_eq!(
        value["quarantined_snapshots"]
            .as_array()
            .expect("quarantined")
            .len(),
        0
    );

    let verify = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args(["model", "verify"])
        .arg(&snapshot_path)
        .output()
        .expect("run model verify");
    assert!(!verify.status.success(), "verify unexpectedly succeeded");
    let stderr = String::from_utf8_lossy(&verify.stderr);
    assert!(stderr.contains("sha256"), "stderr: {stderr}");
}

#[tokio::test]
async fn model_list_deep_readiness_hashes_and_quarantines_corrupt_snapshot() {
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
    tokio::fs::write(snapshot_path.join("model.safetensors"), b"xxxx")
        .await
        .expect("same-size weight corruption");

    let output = Command::new(env!("CARGO_BIN_EXE_llm-engine"))
        .args([
            "model",
            "list",
            "--snapshot-readiness",
            "deep",
            "--model-home",
        ])
        .arg(temp.path())
        .output()
        .expect("run deep model list");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).expect("json output");
    assert_eq!(value["snapshots"].as_array().expect("snapshots").len(), 0);
    assert_eq!(
        value["quarantined_snapshots"]
            .as_array()
            .expect("quarantined")
            .len(),
        1
    );
    assert!(
        value["quarantined_snapshots"][0]["reason"]
            .as_str()
            .expect("quarantine reason")
            .contains("sha256")
    );
    assert!(!snapshot_path.exists());
}
