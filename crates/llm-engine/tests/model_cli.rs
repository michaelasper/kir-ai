use llm_hub::{HubFile, HubRepoId, ModelProfile, ModelStore, build_download_plan};
use serde_json::Value;
use std::process::Command;

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
