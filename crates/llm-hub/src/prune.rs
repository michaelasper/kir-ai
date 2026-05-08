use crate::manifest::SnapshotManifest;
use crate::store::{
    ModelAlias, ModelStore, SNAPSHOT_USAGE_FILE, SnapshotUsage, remove_snapshot_dir,
};
use crate::{HubError, QuarantinedSnapshot};
use chrono::{DateTime, Utc};
use std::{
    collections::{HashMap, HashSet},
    io::ErrorKind,
    path::{Path, PathBuf},
    time::Duration,
};

#[derive(Debug, Clone)]
pub struct PrunePolicy {
    pub now: DateTime<Utc>,
    pub keep_recent: Option<Duration>,
    pub keep_min_per_profile: usize,
    pub profile: Option<String>,
}

impl Default for PrunePolicy {
    fn default() -> Self {
        Self {
            now: Utc::now(),
            keep_recent: Some(Duration::from_secs(7 * 24 * 60 * 60)),
            keep_min_per_profile: 1,
            profile: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PruneCandidate {
    pub path: PathBuf,
    pub repo_id: String,
    pub resolved_commit: String,
    pub profile: String,
    pub manifest_digest: String,
    pub bytes: u64,
    pub last_used_at: DateTime<Utc>,
    pub aliases: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProtectedSnapshot {
    pub path: PathBuf,
    pub repo_id: String,
    pub resolved_commit: String,
    pub profile: String,
    pub manifest_digest: String,
    pub bytes: u64,
    pub last_used_at: DateTime<Utc>,
    pub aliases: Vec<String>,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrunePlan {
    pub scanned_snapshots: usize,
    pub total_bytes: u64,
    pub reclaimable_bytes: u64,
    pub candidates: Vec<PruneCandidate>,
    pub protected: Vec<ProtectedSnapshot>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeletedSnapshot {
    pub path: PathBuf,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PruneReport {
    pub candidates: Vec<PruneCandidate>,
    pub protected: Vec<ProtectedSnapshot>,
    pub deleted: Vec<DeletedSnapshot>,
    pub quarantined: Vec<QuarantinedSnapshot>,
    pub deleted_bytes: u64,
}

impl ModelStore {
    pub async fn prune_plan(&self, policy: PrunePolicy) -> Result<PrunePlan, HubError> {
        let snapshots = self.list_snapshots().await?;
        let aliases = self.list_aliases().await?;
        let aliases_by_path = aliases_by_snapshot_path(aliases);
        let mut entries = Vec::with_capacity(snapshots.len());
        for snapshot in snapshots {
            if policy
                .profile
                .as_ref()
                .is_some_and(|profile| profile != &snapshot.manifest.profile)
            {
                continue;
            }
            let aliases = aliases_by_path
                .get(&snapshot.path)
                .cloned()
                .unwrap_or_default();
            let bytes = snapshot.manifest.files.iter().map(|file| file.size).sum();
            let last_used_at = snapshot_last_used_at(&snapshot.path, &snapshot.manifest).await?;
            entries.push(PruneSnapshotEntry {
                path: snapshot.path,
                repo_id: snapshot.manifest.repo_id,
                resolved_commit: snapshot.manifest.resolved_commit,
                profile: snapshot.manifest.profile,
                manifest_digest: snapshot.manifest_digest,
                bytes,
                last_used_at,
                aliases,
            });
        }

        let retained_minimum_paths =
            minimum_retained_snapshot_paths(&entries, policy.keep_min_per_profile);
        let mut candidates = Vec::new();
        let mut protected = Vec::new();
        for entry in entries {
            let mut reasons = Vec::new();
            for alias in &entry.aliases {
                reasons.push(format!("active_alias:{alias}"));
            }
            if retained_minimum_paths.contains(&entry.path) {
                reasons.push("minimum_retained_for_profile".to_owned());
            }
            if let Some(keep_recent) = policy.keep_recent {
                if let Ok(age) = (policy.now - entry.last_used_at).to_std() {
                    if age <= keep_recent {
                        reasons.push("recently_used".to_owned());
                    }
                } else {
                    reasons.push("recently_used".to_owned());
                }
            }
            if reasons.is_empty() {
                candidates.push(entry.candidate());
            } else {
                protected.push(entry.protected(reasons));
            }
        }
        candidates.sort_by(|left, right| left.path.cmp(&right.path));
        protected.sort_by(|left, right| left.path.cmp(&right.path));
        let total_bytes = candidates
            .iter()
            .map(|snapshot| snapshot.bytes)
            .chain(protected.iter().map(|snapshot| snapshot.bytes))
            .sum();
        let reclaimable_bytes = candidates.iter().map(|snapshot| snapshot.bytes).sum();
        Ok(PrunePlan {
            scanned_snapshots: candidates.len() + protected.len(),
            total_bytes,
            reclaimable_bytes,
            candidates,
            protected,
        })
    }

    pub async fn apply_prune_plan(&self, plan: &PrunePlan) -> Result<PruneReport, HubError> {
        let mut deleted = Vec::new();
        let mut quarantined = Vec::new();
        let mut deleted_bytes = 0_u64;
        for candidate in &plan.candidates {
            match Self::verify_snapshot(&candidate.path).await {
                Ok(_) => {
                    remove_snapshot_dir(&candidate.path).await?;
                    deleted_bytes = deleted_bytes.saturating_add(candidate.bytes);
                    deleted.push(DeletedSnapshot {
                        path: candidate.path.clone(),
                        bytes: candidate.bytes,
                    });
                }
                Err(err) => {
                    quarantined.push(
                        self.quarantine_snapshot(&candidate.path, err.to_string())
                            .await?,
                    );
                }
            }
        }
        Ok(PruneReport {
            candidates: plan.candidates.clone(),
            protected: plan.protected.clone(),
            deleted,
            quarantined,
            deleted_bytes,
        })
    }
}

#[derive(Debug, Clone)]
struct PruneSnapshotEntry {
    path: PathBuf,
    repo_id: String,
    resolved_commit: String,
    profile: String,
    manifest_digest: String,
    bytes: u64,
    last_used_at: DateTime<Utc>,
    aliases: Vec<String>,
}

impl PruneSnapshotEntry {
    fn candidate(self) -> PruneCandidate {
        PruneCandidate {
            path: self.path,
            repo_id: self.repo_id,
            resolved_commit: self.resolved_commit,
            profile: self.profile,
            manifest_digest: self.manifest_digest,
            bytes: self.bytes,
            last_used_at: self.last_used_at,
            aliases: self.aliases,
        }
    }

    fn protected(self, reasons: Vec<String>) -> ProtectedSnapshot {
        ProtectedSnapshot {
            path: self.path,
            repo_id: self.repo_id,
            resolved_commit: self.resolved_commit,
            profile: self.profile,
            manifest_digest: self.manifest_digest,
            bytes: self.bytes,
            last_used_at: self.last_used_at,
            aliases: self.aliases,
            reasons,
        }
    }
}

fn aliases_by_snapshot_path(aliases: Vec<ModelAlias>) -> HashMap<PathBuf, Vec<String>> {
    let mut aliases_by_path: HashMap<PathBuf, Vec<String>> = HashMap::new();
    for alias in aliases {
        aliases_by_path
            .entry(alias.snapshot_path)
            .or_default()
            .push(alias.alias);
    }
    for aliases in aliases_by_path.values_mut() {
        aliases.sort();
    }
    aliases_by_path
}

fn minimum_retained_snapshot_paths(
    entries: &[PruneSnapshotEntry],
    keep_min_per_profile: usize,
) -> HashSet<PathBuf> {
    if keep_min_per_profile == 0 {
        return HashSet::new();
    }
    let mut by_profile: HashMap<&str, Vec<&PruneSnapshotEntry>> = HashMap::new();
    for entry in entries {
        by_profile.entry(&entry.profile).or_default().push(entry);
    }
    let mut retained = HashSet::new();
    for snapshots in by_profile.values_mut() {
        snapshots.sort_by(|left, right| {
            right
                .last_used_at
                .cmp(&left.last_used_at)
                .then_with(|| left.path.cmp(&right.path))
        });
        for snapshot in snapshots.iter().take(keep_min_per_profile) {
            retained.insert(snapshot.path.clone());
        }
    }
    retained
}

async fn snapshot_last_used_at(
    snapshot: &Path,
    manifest: &SnapshotManifest,
) -> Result<DateTime<Utc>, HubError> {
    let usage_path = snapshot.join(SNAPSHOT_USAGE_FILE);
    match tokio::fs::read(&usage_path).await {
        Ok(bytes) => {
            let usage = serde_json::from_slice::<SnapshotUsage>(&bytes).map_err(|err| {
                HubError::integrity_failed(format!(
                    "invalid snapshot usage `{}`: {err}",
                    usage_path.display()
                ))
            })?;
            Ok(usage.last_used_at)
        }
        Err(err) if err.kind() == ErrorKind::NotFound => Ok(manifest.created_at),
        Err(err) => Err(HubError::io(err)),
    }
}
