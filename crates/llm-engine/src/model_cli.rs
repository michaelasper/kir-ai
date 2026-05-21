use llm_hub::{
    DeletedSnapshot, HubRepoId, ModelProfile, ModelStore, PromotedSnapshot, ProtectedSnapshot,
    PruneCandidate, PrunePlan, PrunePolicy, PruneReport, QuarantinedSnapshot,
    SnapshotReadinessMode, SnapshotRecord,
};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelPlanOptions {
    pub repo_id: HubRepoId,
    pub revision: String,
    pub profile: ModelProfile,
    pub metadata_only: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PruneMode {
    DryRun,
    ConfirmDelete,
}

pub fn model_plan_options_from_args(
    subcommand: &str,
    args: &[String],
) -> anyhow::Result<ModelPlanOptions> {
    let repo = args
        .get(1)
        .ok_or_else(|| anyhow::anyhow!("usage: llm-engine model {subcommand} <repo>"))?;
    let revision = flag_value(args, "--revision").unwrap_or("main").to_owned();
    let profile_name = flag_value(args, "--profile").unwrap_or("qwen36-safetensors-bf16");
    let profile = ModelProfile::builtin(profile_name)
        .ok_or_else(|| anyhow::anyhow!("unknown model profile `{profile_name}`"))?;
    let repo_id = HubRepoId::model(repo)?;
    let metadata_only = has_flag(args, "--metadata-only");
    Ok(ModelPlanOptions {
        repo_id,
        revision,
        profile,
        metadata_only,
    })
}

pub fn prune_mode_from_args(args: &[String]) -> anyhow::Result<PruneMode> {
    let dry_run = has_flag(args, "--dry-run");
    let confirmed = has_flag(args, "--confirm-delete");
    match (dry_run, confirmed) {
        (true, false) => Ok(PruneMode::DryRun),
        (false, true) => Ok(PruneMode::ConfirmDelete),
        (false, false) => {
            anyhow::bail!("llm-engine model prune requires --dry-run or --confirm-delete")
        }
        (true, true) => anyhow::bail!(
            "llm-engine model prune accepts only one of --dry-run or --confirm-delete"
        ),
    }
}

pub fn prune_policy_from_args(args: &[String]) -> anyhow::Result<PrunePolicy> {
    let mut policy = PrunePolicy::default();
    if let Some(days) = flag_value(args, "--older-than-days") {
        let days = days.parse::<u64>()?;
        let seconds = days
            .checked_mul(24 * 60 * 60)
            .ok_or_else(|| anyhow::anyhow!("--older-than-days is too large"))?;
        policy.keep_recent = Some(std::time::Duration::from_secs(seconds));
    }
    if let Some(count) = flag_value(args, "--keep-min-per-profile") {
        policy.keep_min_per_profile = count.parse::<usize>()?;
    }
    if let Some(profile) = flag_value(args, "--profile") {
        policy.profile = Some(profile.to_owned());
    }
    if let Some(now) = flag_value(args, "--now") {
        policy.now = chrono::DateTime::parse_from_rfc3339(now)?.with_timezone(&chrono::Utc);
    }
    Ok(policy)
}

pub fn snapshot_readiness_mode_from_args(args: &[String]) -> anyhow::Result<SnapshotReadinessMode> {
    flag_value(args, "--snapshot-readiness")
        .map(SnapshotReadinessMode::parse)
        .transpose()
        .map_err(anyhow::Error::msg)
        .map(|mode| mode.unwrap_or(SnapshotReadinessMode::Fast))
}

pub async fn model_list_json(root: impl AsRef<Path>) -> anyhow::Result<Value> {
    model_list_json_with_mode(root, SnapshotReadinessMode::Fast).await
}

pub async fn model_list_json_with_mode(
    root: impl AsRef<Path>,
    readiness_mode: SnapshotReadinessMode,
) -> anyhow::Result<Value> {
    let store = ModelStore::new(root);
    let aliases = store.list_aliases().await?;
    let mut aliases_by_path: HashMap<std::path::PathBuf, Vec<String>> = HashMap::new();
    for alias in aliases {
        aliases_by_path
            .entry(alias.snapshot_path)
            .or_default()
            .push(alias.alias);
    }
    let inventory = store.snapshot_inventory_with_mode(readiness_mode).await?;
    let snapshots = inventory
        .ready_snapshots
        .into_iter()
        .map(|snapshot| {
            let aliases = aliases_by_path.remove(&snapshot.path).unwrap_or_default();
            promoted_snapshot_json(snapshot, "ready", None, aliases)
        })
        .collect::<Vec<_>>();
    let metadata_only = inventory
        .metadata_only_snapshots
        .into_iter()
        .map(|record| {
            let aliases = aliases_by_path
                .remove(&record.snapshot.path)
                .unwrap_or_default();
            snapshot_record_json(record, aliases)
        })
        .collect::<Vec<_>>();
    let quarantined = inventory
        .quarantined_snapshots
        .into_iter()
        .map(quarantined_snapshot_json)
        .collect::<Vec<_>>();
    Ok(serde_json::json!({
        "snapshots": snapshots,
        "metadata_only_snapshots": metadata_only,
        "quarantined_snapshots": quarantined,
    }))
}

pub async fn model_prune_json(
    root: impl AsRef<Path>,
    policy: PrunePolicy,
    mode: PruneMode,
) -> anyhow::Result<Value> {
    let store = ModelStore::new(root);
    let plan = store.prune_plan(policy).await?;
    let report = match mode {
        PruneMode::DryRun => None,
        PruneMode::ConfirmDelete => Some(store.apply_prune_plan(&plan).await?),
    };
    Ok(prune_output_json(
        mode == PruneMode::DryRun,
        &plan,
        report.as_ref(),
    ))
}

pub async fn model_inspect_json(snapshot_path: impl AsRef<Path>) -> anyhow::Result<Value> {
    let snapshot_path = snapshot_path.as_ref();
    if let Ok(quarantine) = ModelStore::inspect_quarantined_snapshot(snapshot_path).await {
        return Ok(quarantined_snapshot_json(quarantine));
    }
    let record = ModelStore::inspect_snapshot_readiness(snapshot_path).await?;
    let snapshot = record.snapshot;
    let total_bytes = snapshot
        .manifest
        .files
        .iter()
        .map(|file| file.size)
        .sum::<u64>();
    Ok(serde_json::json!({
        "status": record.readiness.status(),
        "readiness_reason": record.readiness.reason(),
        "snapshot_path": snapshot.path,
        "repo_id": snapshot.manifest.repo_id,
        "requested_revision": snapshot.manifest.requested_revision,
        "resolved_commit": snapshot.manifest.resolved_commit,
        "profile": snapshot.manifest.profile,
        "family": snapshot.manifest.family,
        "loader": snapshot.manifest.loader,
        "quantization": snapshot.manifest.quantization,
        "manifest_digest": snapshot.manifest_digest,
        "files": snapshot.manifest.files.len(),
        "total_bytes": total_bytes,
    }))
}

pub async fn model_verify_json(snapshot_path: impl AsRef<Path>) -> anyhow::Result<Value> {
    let snapshot_path = snapshot_path.as_ref();
    let verification = ModelStore::verify_runnable_snapshot(snapshot_path).await?;
    ModelStore::mark_snapshot_used(snapshot_path).await?;
    Ok(serde_json::json!({
        "status": "ok",
        "snapshot_path": verification.snapshot.path,
        "repo_id": verification.snapshot.manifest.repo_id,
        "resolved_commit": verification.snapshot.manifest.resolved_commit,
        "manifest_digest": verification.snapshot.manifest_digest,
        "verified_files": verification.verified_files,
        "verified_bytes": verification.verified_bytes,
    }))
}

fn flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find_map(|window| (window[0] == flag).then_some(window[1].as_str()))
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|arg| arg == flag)
}

fn snapshot_record_json(record: SnapshotRecord, aliases: Vec<String>) -> Value {
    let reason = record.readiness.reason().map(str::to_owned);
    let status = record.readiness.status();
    promoted_snapshot_json(record.snapshot, status, reason, aliases)
}

fn prune_output_json(dry_run: bool, plan: &PrunePlan, report: Option<&PruneReport>) -> Value {
    let candidates = plan
        .candidates
        .iter()
        .map(prune_candidate_json)
        .collect::<Vec<_>>();
    let protected = plan
        .protected
        .iter()
        .map(protected_snapshot_json)
        .collect::<Vec<_>>();
    let deleted = report
        .map(|report| {
            report
                .deleted
                .iter()
                .map(deleted_snapshot_json)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let quarantined = report
        .map(|report| {
            report
                .quarantined
                .iter()
                .cloned()
                .map(quarantined_snapshot_json)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let deleted_bytes = report.map_or(0, |report| report.deleted_bytes);
    serde_json::json!({
        "dry_run": dry_run,
        "confirmed": report.is_some(),
        "snapshots": plan.scanned_snapshots,
        "total_bytes": plan.total_bytes,
        "reclaimable_bytes": plan.reclaimable_bytes,
        "deleted_bytes": deleted_bytes,
        "candidates": candidates,
        "protected": protected,
        "deleted": deleted,
        "quarantined": quarantined,
    })
}

fn prune_candidate_json(candidate: &PruneCandidate) -> Value {
    serde_json::json!({
        "path": path_string(&candidate.path),
        "repo_id": &candidate.repo_id,
        "resolved_commit": &candidate.resolved_commit,
        "profile": &candidate.profile,
        "manifest_digest": &candidate.manifest_digest,
        "bytes": candidate.bytes,
        "last_used_at": candidate.last_used_at,
        "aliases": &candidate.aliases,
        "would_delete": true,
    })
}

fn protected_snapshot_json(snapshot: &ProtectedSnapshot) -> Value {
    serde_json::json!({
        "path": path_string(&snapshot.path),
        "repo_id": &snapshot.repo_id,
        "resolved_commit": &snapshot.resolved_commit,
        "profile": &snapshot.profile,
        "manifest_digest": &snapshot.manifest_digest,
        "bytes": snapshot.bytes,
        "last_used_at": snapshot.last_used_at,
        "aliases": &snapshot.aliases,
        "reasons": &snapshot.reasons,
        "would_delete": false,
    })
}

fn deleted_snapshot_json(snapshot: &DeletedSnapshot) -> Value {
    serde_json::json!({
        "path": path_string(&snapshot.path),
        "bytes": snapshot.bytes,
    })
}

fn promoted_snapshot_json(
    snapshot: PromotedSnapshot,
    status: &str,
    reason: Option<String>,
    aliases: Vec<String>,
) -> Value {
    serde_json::json!({
        "status": status,
        "path": path_string(&snapshot.path),
        "repo_id": snapshot.manifest.repo_id,
        "requested_revision": snapshot.manifest.requested_revision,
        "resolved_commit": snapshot.manifest.resolved_commit,
        "profile": snapshot.manifest.profile,
        "family": snapshot.manifest.family,
        "loader": snapshot.manifest.loader,
        "quantization": snapshot.manifest.quantization,
        "manifest_digest": snapshot.manifest_digest,
        "files": snapshot.manifest.files.len(),
        "readiness_reason": reason,
        "aliases": aliases,
    })
}

fn quarantined_snapshot_json(snapshot: QuarantinedSnapshot) -> Value {
    serde_json::json!({
        "status": "quarantined",
        "path": path_string(&snapshot.path),
        "original_path": path_string(&snapshot.metadata.original_path),
        "reason": snapshot.metadata.reason,
        "quarantined_at": snapshot.metadata.quarantined_at,
        "manifest_digest": snapshot.metadata.manifest_digest,
        "bytes": snapshot.bytes,
    })
}

fn path_string(path: &Path) -> String {
    path.display().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_hub::{HubFile, build_download_plan};

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    fn runnable_qwen_files() -> Vec<HubFile> {
        vec![
            HubFile::new("config.json", 2, Some("\"cfg\"")),
            HubFile::new("tokenizer.json", 2, Some("\"tok\"")),
            HubFile::new(
                "model.safetensors",
                4,
                Some("3a6eb0790f39ac87c94f3856b2dd2c5d110e6811602261a9a923d3bb23adc8b7"),
            ),
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

    async fn verified_runnable_snapshot(temp: &tempfile::TempDir) -> std::path::PathBuf {
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
        snapshot_path
    }

    async fn verified_config_only_snapshot(
        temp: &tempfile::TempDir,
        resolved_commit: &str,
    ) -> std::path::PathBuf {
        let store = ModelStore::new(temp.path());
        let plan = build_download_plan(
            HubRepoId::model("Qwen/Qwen3.6-35B-A3B").expect("repo id"),
            "main",
            resolved_commit,
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
        snapshot_path
    }

    #[test]
    fn plan_options_parse_defaults_and_metadata_only() {
        let options =
            model_plan_options_from_args("plan", &args(&["plan", "Qwen/Qwen3.6-35B-A3B"]))
                .expect("default plan options parse");
        assert_eq!(options.repo_id.as_str(), "Qwen/Qwen3.6-35B-A3B");
        assert_eq!(options.revision, "main");
        assert_eq!(options.profile.name, "qwen36-safetensors-bf16");
        assert!(!options.metadata_only);

        let options = model_plan_options_from_args(
            "plan",
            &args(&[
                "plan",
                "Qwen/Qwen3.6-35B-A3B",
                "--revision",
                "refs/pr/1",
                "--profile",
                "qwen36-mlx-4bit",
                "--metadata-only",
            ]),
        )
        .expect("custom plan options parse");
        assert_eq!(options.revision, "refs/pr/1");
        assert_eq!(options.profile.name, "qwen36-mlx-4bit");
        assert!(options.metadata_only);
    }

    #[test]
    fn snapshot_readiness_mode_parses_default_and_overrides() {
        assert_eq!(
            snapshot_readiness_mode_from_args(&args(&["list"]))
                .expect("default readiness mode parses"),
            SnapshotReadinessMode::Fast
        );
        assert_eq!(
            snapshot_readiness_mode_from_args(&args(&["list", "--snapshot-readiness", "deep"]))
                .expect("deep readiness mode parses"),
            SnapshotReadinessMode::Deep
        );
        let err =
            snapshot_readiness_mode_from_args(&args(&["list", "--snapshot-readiness", "full"]))
                .expect_err("invalid readiness mode fails");
        assert!(err.to_string().contains("fast or deep"));
    }

    #[tokio::test]
    async fn list_json_outputs_promoted_snapshots() {
        let temp = tempfile::tempdir().expect("tempdir");
        verified_runnable_snapshot(&temp).await;

        let value = model_list_json(temp.path()).await.expect("model list json");

        assert_eq!(value["snapshots"][0]["repo_id"], "Qwen/Qwen3.6-35B-A3B");
        assert_eq!(
            value["snapshots"][0]["resolved_commit"],
            "0123456789abcdef0123456789abcdef01234567"
        );
        assert_eq!(
            value["snapshots"][0]["manifest_digest"]
                .as_str()
                .expect("manifest digest")
                .len(),
            64
        );
    }

    #[tokio::test]
    async fn inspect_json_outputs_snapshot_manifest_summary() {
        let temp = tempfile::tempdir().expect("tempdir");
        let snapshot_path = verified_runnable_snapshot(&temp).await;

        let value = model_inspect_json(&snapshot_path)
            .await
            .expect("model inspect json");

        assert_eq!(value["status"], "ready");
        assert_eq!(value["repo_id"], "Qwen/Qwen3.6-35B-A3B");
        assert_eq!(value["files"], 3);
        assert_eq!(value["total_bytes"], 8);
    }

    #[tokio::test]
    async fn verify_json_outputs_snapshot_integrity_summary() {
        let temp = tempfile::tempdir().expect("tempdir");
        let snapshot_path = verified_runnable_snapshot(&temp).await;

        let value = model_verify_json(&snapshot_path)
            .await
            .expect("model verify json");

        assert_eq!(value["status"], "ok");
        assert_eq!(value["verified_files"], 3);
        assert_eq!(value["verified_bytes"], 8);
    }

    #[tokio::test]
    async fn prune_dry_run_json_outputs_usage_without_deleting() {
        let temp = tempfile::tempdir().expect("tempdir");
        let snapshot_path =
            verified_config_only_snapshot(&temp, "0123456789abcdef0123456789abcdef01234567").await;

        let value = model_prune_json(temp.path(), PrunePolicy::default(), PruneMode::DryRun)
            .await
            .expect("model prune dry run json");

        assert_eq!(value["dry_run"], true);
        assert_eq!(value["snapshots"], 1);
        assert_eq!(value["total_bytes"], 2);
        assert_eq!(value["reclaimable_bytes"], 0);
        assert!(snapshot_path.join("config.json").is_file());
    }

    #[tokio::test]
    async fn prune_confirm_json_deletes_same_candidates_as_dry_run() {
        let temp = tempfile::tempdir().expect("tempdir");
        let snapshot_path =
            verified_config_only_snapshot(&temp, "abcdefabcdefabcdefabcdefabcdefabcdefabcd").await;
        let old = chrono::DateTime::parse_from_rfc3339("2026-04-01T00:00:00Z")
            .expect("fixed time")
            .with_timezone(&chrono::Utc);
        ModelStore::mark_snapshot_used_at(&snapshot_path, old)
            .await
            .expect("usage recorded");
        let policy = prune_policy_from_args(&args(&[
            "prune",
            "--older-than-days",
            "7",
            "--keep-min-per-profile",
            "0",
            "--now",
            "2026-05-08T00:00:00Z",
        ]))
        .expect("prune policy");

        let dry_run = model_prune_json(temp.path(), policy.clone(), PruneMode::DryRun)
            .await
            .expect("model prune dry run json");
        let dry_run_candidates = dry_run["candidates"].as_array().expect("candidate array");
        assert_eq!(dry_run["dry_run"], true);
        assert_eq!(dry_run_candidates.len(), 1);
        assert_eq!(
            dry_run_candidates[0]["path"],
            snapshot_path.display().to_string()
        );
        assert!(snapshot_path.join("config.json").is_file());

        let destructive = model_prune_json(temp.path(), policy, PruneMode::ConfirmDelete)
            .await
            .expect("model prune destructive json");
        let destructive_candidates = destructive["candidates"]
            .as_array()
            .expect("candidate array");
        assert_eq!(destructive["dry_run"], false);
        assert_eq!(destructive_candidates.len(), dry_run_candidates.len());
        assert_eq!(
            destructive_candidates[0]["path"],
            dry_run_candidates[0]["path"]
        );
        assert_eq!(destructive["deleted"].as_array().expect("deleted").len(), 1);
        assert!(!snapshot_path.exists());
    }

    #[tokio::test]
    async fn list_and_inspect_json_show_quarantined_snapshots() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = ModelStore::new(temp.path());
        let snapshot_path =
            verified_config_only_snapshot(&temp, "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb").await;
        let quarantined = store
            .quarantine_snapshot(&snapshot_path, "test corruption")
            .await
            .expect("quarantined");

        let list = model_list_json(temp.path()).await.expect("model list json");
        assert_eq!(list["snapshots"].as_array().expect("snapshots").len(), 0);
        assert_eq!(
            list["quarantined_snapshots"]
                .as_array()
                .expect("quarantined")
                .len(),
            1
        );
        assert_eq!(
            list["quarantined_snapshots"][0]["path"],
            quarantined.path.display().to_string()
        );

        let inspect = model_inspect_json(&quarantined.path)
            .await
            .expect("model inspect quarantined json");
        assert_eq!(inspect["status"], "quarantined");
        assert_eq!(
            inspect["original_path"],
            snapshot_path.display().to_string()
        );
    }
}
