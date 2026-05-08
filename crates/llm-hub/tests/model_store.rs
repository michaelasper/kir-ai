use llm_hub::{
    HubFile, HubRepoId, ModelProfile, ModelStore, SnapshotManifest, build_download_plan,
};

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
async fn staging_dirs_are_unique_for_same_plan() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = ModelStore::new(temp.path());
    let plan = build_download_plan(
        HubRepoId::model("Qwen/Qwen3.6-35B-A3B").expect("repo id"),
        "main",
        "0123456789abcdef0123456789abcdef01234567",
        ModelProfile::qwen36_mlx_4bit(),
        vec![HubFile::new("config.json", 2, Some("\"cfg\""))],
        &[],
    )
    .expect("plan builds");

    let first = store
        .create_staging_dir(&plan)
        .await
        .expect("first staging");
    let second = store
        .create_staging_dir(&plan)
        .await
        .expect("second staging");

    assert_ne!(first, second);
    assert!(first.is_dir());
    assert!(second.is_dir());
}

#[tokio::test]
async fn promoting_when_snapshot_exists_reuses_snapshot_and_cleans_staging() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = ModelStore::new(temp.path());
    let plan = build_download_plan(
        HubRepoId::model("Qwen/Qwen3.6-35B-A3B").expect("repo id"),
        "main",
        "0123456789abcdef0123456789abcdef01234567",
        ModelProfile::qwen36_mlx_4bit(),
        vec![HubFile::new("config.json", 2, Some("\"cfg\""))],
        &[],
    )
    .expect("plan builds");

    let winner = store
        .create_staging_dir(&plan)
        .await
        .expect("winner staging");
    tokio::fs::write(winner.join("config.json"), "{}")
        .await
        .expect("write winner file");
    let promoted = store
        .promote_staging(&plan, winner)
        .await
        .expect("winner promoted");

    let loser = store
        .create_staging_dir(&plan)
        .await
        .expect("loser staging");
    tokio::fs::write(loser.join("config.json"), "{}")
        .await
        .expect("write loser file");
    let reused = store
        .promote_staging(&plan, loser.clone())
        .await
        .expect("existing snapshot reused");

    assert_eq!(reused.path, promoted.path);
    assert!(!loser.exists());
    assert!(promoted.path.join("config.json").is_file());
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
async fn verifying_existing_snapshot_twice_preserves_manifest_digest() {
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

    let first = store
        .verify_existing_snapshot(&plan)
        .await
        .expect("first verification succeeds")
        .expect("snapshot exists");
    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
    let second = store
        .verify_existing_snapshot(&plan)
        .await
        .expect("second verification succeeds")
        .expect("snapshot exists");

    assert_eq!(first.manifest_digest, second.manifest_digest);
    assert_eq!(first.manifest.created_at, second.manifest.created_at);
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
async fn inspects_promoted_snapshot_from_manifest() {
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

    let snapshot = ModelStore::inspect_snapshot(&snapshot_path)
        .await
        .expect("snapshot inspects");

    assert_eq!(snapshot.path, snapshot_path);
    assert_eq!(snapshot.manifest.repo_id, "Qwen/Qwen3.6-35B-A3B");
    assert_eq!(snapshot.manifest.files.len(), 1);
    assert_eq!(snapshot.manifest_digest.len(), 64);
}

#[tokio::test]
async fn verifies_promoted_snapshot_from_manifest() {
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

    let verification = ModelStore::verify_snapshot(&snapshot_path)
        .await
        .expect("snapshot verifies from manifest");

    assert_eq!(verification.snapshot.path, snapshot_path);
    assert_eq!(verification.verified_files, 1);
    assert_eq!(verification.verified_bytes, 2);
}

#[tokio::test]
async fn rejects_corrupt_promoted_snapshot_from_manifest() {
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
    tokio::fs::write(snapshot_path.join("config.json"), "wrong")
        .await
        .expect("corrupt config");

    let err = ModelStore::verify_snapshot(&snapshot_path)
        .await
        .expect_err("size mismatch fails");

    assert_eq!(err.code(), "model_integrity_failed");
}

#[cfg(unix)]
#[tokio::test]
async fn rejects_promoted_snapshot_manifest_symlinked_file() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = ModelStore::new(temp.path());
    let plan = build_download_plan(
        HubRepoId::model("Qwen/Qwen3.6-35B-A3B").expect("repo id"),
        "main",
        "0123456789abcdef0123456789abcdef01234567",
        ModelProfile::qwen36_safetensors_bf16(),
        vec![HubFile::new("config.json", 2, None)],
        &[],
    )
    .expect("plan builds");
    let snapshot_path = store.snapshot_path(&plan);
    tokio::fs::create_dir_all(&snapshot_path)
        .await
        .expect("snapshot dir");
    let outside = temp.path().join("outside-config.json");
    tokio::fs::write(&outside, "{}")
        .await
        .expect("outside config");
    std::os::unix::fs::symlink(&outside, snapshot_path.join("config.json"))
        .expect("symlink config");
    let manifest = SnapshotManifest::from_plan(&plan, snapshot_path.display().to_string());
    tokio::fs::write(
        snapshot_path.join("llm-engine-manifest.json"),
        serde_json::to_vec_pretty(&manifest).expect("manifest json"),
    )
    .await
    .expect("manifest");

    let err = ModelStore::verify_snapshot(&snapshot_path)
        .await
        .expect_err("manifest symlink fails");

    assert_eq!(err.code(), "model_integrity_failed");
    assert!(err.to_string().contains("symlink"), "err: {err}");
}

#[cfg(unix)]
#[tokio::test]
async fn rejects_existing_snapshot_with_nested_symlinked_artifact() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = ModelStore::new(temp.path());
    let plan = build_download_plan(
        HubRepoId::model("Qwen/Qwen3.6-35B-A3B").expect("repo id"),
        "main",
        "0123456789abcdef0123456789abcdef01234567",
        ModelProfile::qwen36_safetensors_bf16(),
        vec![HubFile::new("nested/config.json", 2, None)],
        &[],
    )
    .expect("plan builds");
    let snapshot_path = store.snapshot_path(&plan);
    tokio::fs::create_dir_all(snapshot_path.join("nested"))
        .await
        .expect("nested snapshot dir");
    let outside = temp.path().join("outside-nested-config.json");
    tokio::fs::write(&outside, "{}")
        .await
        .expect("outside nested config");
    std::os::unix::fs::symlink(&outside, snapshot_path.join("nested/config.json"))
        .expect("nested symlink config");

    let err = store
        .verify_existing_snapshot(&plan)
        .await
        .expect_err("nested symlink fails");

    assert_eq!(err.code(), "model_integrity_failed");
    assert!(err.to_string().contains("symlink"), "err: {err}");
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

#[test]
fn full_snapshot_paths_do_not_collide_across_profiles() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = ModelStore::new(temp.path());
    let repo = HubRepoId::model("Qwen/Qwen3.6-35B-A3B").expect("repo id");
    let files = vec![HubFile::new("config.json", 2, Some("\"cfg\""))];
    let mlx = build_download_plan(
        repo.clone(),
        "main",
        "0123456789abcdef0123456789abcdef01234567",
        ModelProfile::qwen36_mlx_4bit(),
        files.clone(),
        &[],
    )
    .expect("mlx plan builds");
    let native = build_download_plan(
        repo,
        "main",
        "0123456789abcdef0123456789abcdef01234567",
        ModelProfile::qwen36_safetensors_bf16(),
        files,
        &[],
    )
    .expect("native plan builds");

    assert_ne!(store.snapshot_path(&mlx), store.snapshot_path(&native));
}
