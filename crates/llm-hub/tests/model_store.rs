use llm_hub::{
    HubFile, HubRepoId, ModelProfile, ModelStore, PrunePolicy, SnapshotManifest,
    build_download_plan,
};
use std::path::Path;
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
async fn snapshot_inventory_quarantines_stale_builtin_profile_metadata() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = ModelStore::new(temp.path());
    let plan = build_download_plan(
        HubRepoId::model("mlx-community/Qwen3.6-35B-A3B-4bit").expect("repo id"),
        "main",
        "0123456789abcdef0123456789abcdef01234567",
        ModelProfile::qwen36_mlx_4bit(),
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

    let manifest_path = snapshot_path.join("llm-engine-manifest.json");
    let mut manifest = serde_json::from_slice::<SnapshotManifest>(
        &tokio::fs::read(&manifest_path)
            .await
            .expect("manifest bytes"),
    )
    .expect("manifest json");
    manifest.loader = "native-metal".to_owned();
    tokio::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).expect("manifest serializes"),
    )
    .await
    .expect("write stale manifest");

    let inventory = store.snapshot_inventory().await.expect("inventory");

    assert!(inventory.ready_snapshots.is_empty());
    assert!(inventory.metadata_only_snapshots.is_empty());
    assert_eq!(inventory.quarantined_snapshots.len(), 1);
    assert!(
        inventory.quarantined_snapshots[0]
            .metadata
            .reason
            .contains("loader `native-metal`, expected `mlx`"),
        "reason: {}",
        inventory.quarantined_snapshots[0].metadata.reason
    );
    assert!(!snapshot_path.exists());
}

#[tokio::test]
async fn snapshot_inventory_reports_metadata_only_snapshots_without_quarantine() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = ModelStore::new(temp.path());
    let full_plan = build_download_plan(
        HubRepoId::model("mlx-community/Qwen3.6-35B-A3B-4bit").expect("repo id"),
        "main",
        "0123456789abcdef0123456789abcdef01234567",
        ModelProfile::qwen36_mlx_4bit(),
        runnable_qwen_files(),
        &[],
    )
    .expect("plan builds");
    let metadata_plan = full_plan.metadata_only();
    let snapshot_path = store.snapshot_path(&metadata_plan);
    tokio::fs::create_dir_all(&snapshot_path)
        .await
        .expect("snapshot dir");
    tokio::fs::write(snapshot_path.join("config.json"), "{}")
        .await
        .expect("config");
    tokio::fs::write(snapshot_path.join("tokenizer.json"), "{}")
        .await
        .expect("tokenizer");
    store
        .verify_existing_snapshot(&metadata_plan)
        .await
        .expect("snapshot verifies");

    let inventory = store.snapshot_inventory().await.expect("inventory");

    assert!(inventory.ready_snapshots.is_empty());
    assert_eq!(inventory.metadata_only_snapshots.len(), 1);
    assert_eq!(
        inventory.metadata_only_snapshots[0].readiness.status(),
        "metadata_only"
    );
    assert!(
        inventory.metadata_only_snapshots[0]
            .readiness
            .reason()
            .expect("reason")
            .contains("contains no weight files")
    );
    assert!(inventory.quarantined_snapshots.is_empty());
    assert!(snapshot_path.exists());
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

    let snapshot = store
        .verify_existing_snapshot(&plan)
        .await
        .expect("nested symlink is quarantined");

    assert!(snapshot.is_none());
    assert!(!snapshot_path.exists());
    let quarantined = store
        .list_quarantined_snapshots()
        .await
        .expect("quarantine list");
    assert_eq!(quarantined.len(), 1);
    assert!(
        quarantined[0].metadata.reason.contains("symlink"),
        "reason: {}",
        quarantined[0].metadata.reason
    );
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

    let snapshot = store
        .verify_existing_snapshot(&plan)
        .await
        .expect("digest mismatch is quarantined");

    assert!(snapshot.is_none());
    assert!(!snapshot_path.exists());
    assert_eq!(
        store
            .list_quarantined_snapshots()
            .await
            .expect("quarantine list")
            .len(),
        1
    );
}

#[tokio::test]
async fn corrupt_existing_snapshot_is_quarantined_before_redownload() {
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
    tokio::fs::write(snapshot_path.join("config.json"), "bad")
        .await
        .expect("corrupt config");

    let snapshot = store
        .verify_existing_snapshot(&plan)
        .await
        .expect("corrupt snapshot is quarantined");

    assert!(snapshot.is_none());
    assert!(!snapshot_path.exists());
    let quarantined = store
        .list_quarantined_snapshots()
        .await
        .expect("quarantine list");
    assert_eq!(quarantined.len(), 1);
    assert_eq!(quarantined[0].metadata.original_path, snapshot_path);
    assert!(
        quarantined[0].metadata.reason.contains("size"),
        "reason: {}",
        quarantined[0].metadata.reason
    );
}

#[tokio::test]
async fn prune_plan_protects_aliases_recent_usage_and_minimum_per_profile() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = ModelStore::new(temp.path());
    let now = chrono::DateTime::parse_from_rfc3339("2026-05-08T00:00:00Z")
        .expect("fixed time")
        .with_timezone(&chrono::Utc);
    let old = now - chrono::Duration::days(30);
    let recent = now - chrono::Duration::days(1);

    let candidate = write_verified_snapshot(
        &store,
        "1111111111111111111111111111111111111111",
        ModelProfile::qwen36_safetensors_bf16(),
        old,
    )
    .await;
    let aliased = write_verified_snapshot(
        &store,
        "2222222222222222222222222222222222222222",
        ModelProfile::qwen36_safetensors_bf16(),
        old,
    )
    .await;
    let recent_path = write_verified_snapshot(
        &store,
        "3333333333333333333333333333333333333333",
        ModelProfile::qwen36_safetensors_bf16(),
        recent,
    )
    .await;
    let minimum_profile = write_verified_snapshot(
        &store,
        "4444444444444444444444444444444444444444",
        ModelProfile::qwen36_mlx_4bit(),
        old,
    )
    .await;
    store
        .record_snapshot_alias("local-qwen36", &aliased)
        .await
        .expect("alias recorded");

    let plan = store
        .prune_plan(PrunePolicy {
            now,
            keep_recent: Some(Duration::from_secs(7 * 24 * 60 * 60)),
            keep_min_per_profile: 1,
            profile: None,
        })
        .await
        .expect("prune plan");

    assert_eq!(plan.candidates.len(), 1);
    assert_eq!(plan.candidates[0].path, candidate);
    assert!(plan.protected.iter().any(|entry| {
        entry.path == aliased
            && entry
                .reasons
                .iter()
                .any(|reason| reason == "active_alias:local-qwen36")
    }));
    assert!(plan.protected.iter().any(|entry| {
        entry.path == recent_path && entry.reasons.iter().any(|reason| reason == "recently_used")
    }));
    assert!(plan.protected.iter().any(|entry| {
        entry.path == minimum_profile
            && entry
                .reasons
                .iter()
                .any(|reason| reason == "minimum_retained_for_profile")
    }));

    let report = store.apply_prune_plan(&plan).await.expect("apply prune");

    assert_eq!(report.candidates.len(), 1);
    assert_eq!(report.candidates[0].path, candidate);
    assert_eq!(report.deleted.len(), 1);
    assert!(!candidate.exists());
    assert!(aliased.exists());
    assert!(recent_path.exists());
    assert!(minimum_profile.exists());
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

#[tokio::test]
async fn resolves_snapshot_alias_and_checks_manifest_digest() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = ModelStore::new(temp.path());
    let snapshot_path = write_verified_snapshot(
        &store,
        "0123456789abcdef0123456789abcdef01234567",
        ModelProfile::qwen35_4b_mlx_4bit(),
        chrono::Utc::now(),
    )
    .await;
    store
        .record_snapshot_alias("local-qwen35-4b", &snapshot_path)
        .await
        .expect("alias recorded");

    let resolved = store
        .resolve_snapshot_alias("local-qwen35-4b")
        .await
        .expect("alias resolves");

    assert_eq!(resolved.path, snapshot_path);
    assert_eq!(resolved.manifest.profile, "qwen35-4b-mlx-4bit");
    assert_eq!(resolved.manifest.loader, "mlx");
}

#[tokio::test]
async fn resolving_missing_snapshot_alias_fails_as_model_not_found() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = ModelStore::new(temp.path());

    let err = store
        .resolve_snapshot_alias("missing-model")
        .await
        .expect_err("missing alias fails");

    assert_eq!(err.code(), "model_not_found");
}

async fn write_verified_snapshot(
    store: &ModelStore,
    commit: &str,
    profile: ModelProfile,
    last_used_at: chrono::DateTime<chrono::Utc>,
) -> std::path::PathBuf {
    let plan = build_download_plan(
        HubRepoId::model("Qwen/Qwen3.6-35B-A3B").expect("repo id"),
        "main",
        commit,
        profile,
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
        .expect("snapshot verifies")
        .expect("snapshot exists");
    ModelStore::mark_snapshot_used_at(&snapshot_path, last_used_at)
        .await
        .expect("usage recorded");
    snapshot_path
}
