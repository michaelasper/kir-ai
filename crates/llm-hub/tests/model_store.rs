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
