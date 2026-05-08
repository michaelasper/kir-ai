use llm_hub::{HubFile, HubRepoId, ModelProfile, ModelStore, build_download_plan};

#[tokio::test]
async fn promotes_staged_snapshot_with_manifest() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = ModelStore::new(temp.path());
    let plan = build_download_plan(
        HubRepoId::model("Qwen/Qwen3.6-35B-A3B").expect("repo id"),
        "main",
        "0123456789abcdef0123456789abcdef01234567",
        ModelProfile::qwen36_mlx_4bit(),
        vec![HubFile::new("config.json", 100, Some("\"cfg\""))],
        &[],
    )
    .expect("plan builds");

    let staging = store.create_staging_dir(&plan).await.expect("staging dir");
    tokio::fs::write(staging.join("config.json"), "{}")
        .await
        .expect("write staged file");

    let snapshot = store
        .promote_staging(&plan, staging)
        .await
        .expect("snapshot promoted");

    assert!(snapshot.path.join("config.json").is_file());
    assert!(snapshot.path.join("llm-engine-manifest.json").is_file());
    assert_eq!(snapshot.manifest.repo_id, "Qwen/Qwen3.6-35B-A3B");
    assert_eq!(snapshot.manifest.resolved_commit, plan.resolved_commit);
    assert_eq!(snapshot.manifest_digest.len(), 64);
}

#[tokio::test]
async fn verifies_existing_snapshot_and_refreshes_manifest_profile() {
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
        .expect("existing config");

    let snapshot = store
        .verify_existing_snapshot(&plan)
        .await
        .expect("snapshot verifies")
        .expect("snapshot exists");

    assert_eq!(snapshot.path, snapshot_path);
    assert_eq!(snapshot.manifest.profile, "qwen36-safetensors-bf16");
    assert_eq!(snapshot.manifest.loader, "native-metal");
    let manifest: serde_json::Value = serde_json::from_slice(
        &tokio::fs::read(snapshot_path.join("llm-engine-manifest.json"))
            .await
            .expect("manifest bytes"),
    )
    .expect("manifest json");
    assert_eq!(manifest["profile"], "qwen36-safetensors-bf16");
}

#[tokio::test]
async fn lists_promoted_snapshots_from_model_store() {
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
        .expect("existing config");
    store
        .verify_existing_snapshot(&plan)
        .await
        .expect("snapshot verifies");

    let snapshots = store.list_snapshots().await.expect("snapshots list");

    assert_eq!(snapshots.len(), 1);
    assert_eq!(snapshots[0].path, snapshot_path);
    assert_eq!(snapshots[0].manifest.repo_id, "Qwen/Qwen3.6-35B-A3B");
    assert_eq!(snapshots[0].manifest_digest.len(), 64);
}

#[tokio::test]
async fn rejects_existing_snapshot_with_wrong_sha256_digest() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = ModelStore::new(temp.path());
    let plan = build_download_plan(
        HubRepoId::model("Qwen/Qwen3.6-35B-A3B").expect("repo id"),
        "main",
        "0123456789abcdef0123456789abcdef01234567",
        ModelProfile::qwen36_safetensors_bf16(),
        vec![HubFile::new(
            "config.json",
            2,
            Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
        )],
        &[],
    )
    .expect("plan builds");
    let snapshot_path = store.snapshot_path(&plan);
    tokio::fs::create_dir_all(&snapshot_path)
        .await
        .expect("snapshot dir");
    tokio::fs::write(snapshot_path.join("config.json"), "{}")
        .await
        .expect("existing config");

    let err = store
        .verify_existing_snapshot(&plan)
        .await
        .expect_err("digest mismatch fails");

    assert_eq!(err.code(), "model_integrity_failed");
}

#[test]
fn metadata_only_snapshot_path_does_not_collide_with_full_snapshot() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = ModelStore::new(temp.path());
    let full = build_download_plan(
        HubRepoId::model("Qwen/Qwen3.6-35B-A3B").expect("repo id"),
        "main",
        "0123456789abcdef0123456789abcdef01234567",
        ModelProfile::qwen36_safetensors_bf16(),
        vec![
            HubFile::new("config.json", 2, Some("\"cfg\"")),
            HubFile::new("model.safetensors", 2, Some("\"weights\"")),
        ],
        &[],
    )
    .expect("plan builds");

    assert_ne!(
        store.snapshot_path(&full),
        store.snapshot_path(&full.metadata_only())
    );
}
