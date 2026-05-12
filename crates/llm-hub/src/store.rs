use crate::client::HubDownloadFileRequest;
use crate::manifest::{
    PromotedSnapshot, SNAPSHOT_MANIFEST_FILE, SnapshotManifest, SnapshotVerification,
    canonicalize_snapshot_root, manifest_matches_plan, read_promoted_snapshot,
    verify_snapshot_file,
};
use crate::plan::{snapshot_dir_name, validate_artifact_path};
use crate::{DownloadPlan, HubClient, HubError, HubRepoId};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    io::ErrorKind,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

static STAGING_COUNTER: AtomicU64 = AtomicU64::new(0);
pub(crate) const SNAPSHOT_USAGE_FILE: &str = "llm-engine-usage.json";
const QUARANTINE_MANIFEST_FILE: &str = "llm-engine-quarantine.json";

#[derive(Debug, Clone)]
pub struct ModelStore {
    root: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotUsage {
    pub schema_version: u32,
    pub last_used_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelAlias {
    pub schema_version: u32,
    pub alias: String,
    pub snapshot_path: PathBuf,
    pub manifest_digest: Option<String>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuarantineMetadata {
    pub schema_version: u32,
    pub original_path: PathBuf,
    pub quarantined_path: PathBuf,
    pub reason: String,
    pub quarantined_at: DateTime<Utc>,
    pub manifest_digest: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuarantinedSnapshot {
    pub path: PathBuf,
    pub metadata: QuarantineMetadata,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SnapshotReadiness {
    Ready,
    MetadataOnly { reason: String },
    Invalid { reason: String },
}

impl SnapshotReadiness {
    pub fn status(&self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::MetadataOnly { .. } => "metadata_only",
            Self::Invalid { .. } => "invalid",
        }
    }

    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Ready => None,
            Self::MetadataOnly { reason } | Self::Invalid { reason } => Some(reason),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SnapshotRecord {
    pub snapshot: PromotedSnapshot,
    pub readiness: SnapshotReadiness,
}

#[derive(Debug, Clone)]
pub struct SnapshotInventory {
    pub ready_snapshots: Vec<PromotedSnapshot>,
    pub metadata_only_snapshots: Vec<SnapshotRecord>,
    pub quarantined_snapshots: Vec<QuarantinedSnapshot>,
}

impl ModelStore {
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
        }
    }

    pub fn snapshot_path(&self, plan: &DownloadPlan) -> PathBuf {
        self.repo_root(&plan.repo_id)
            .join("snapshots")
            .join(snapshot_dir_name(plan))
    }

    pub async fn create_staging_dir(&self, plan: &DownloadPlan) -> Result<PathBuf, HubError> {
        let staging_root = self.repo_root(&plan.repo_id).join("staging");
        tokio::fs::create_dir_all(&staging_root)
            .await
            .map_err(HubError::io)?;
        let snapshot_name = snapshot_dir_name(plan);
        for _ in 0..16 {
            let counter = STAGING_COUNTER.fetch_add(1, Ordering::Relaxed);
            let timestamp = Utc::now().timestamp_nanos_opt().unwrap_or_default();
            let staging = staging_root.join(format!(
                "{snapshot_name}.partial.{}.{timestamp}.{counter}",
                std::process::id()
            ));
            match tokio::fs::create_dir(&staging).await {
                Ok(()) => return Ok(staging),
                Err(err) if err.kind() == ErrorKind::AlreadyExists => continue,
                Err(err) => return Err(HubError::io(err)),
            }
        }
        Err(HubError::invalid_request(
            "failed to allocate unique model staging directory",
        ))
    }

    pub async fn promote_staging(
        &self,
        plan: &DownloadPlan,
        staging: PathBuf,
    ) -> Result<PromotedSnapshot, HubError> {
        let snapshot = self.snapshot_path(plan);
        if tokio::fs::try_exists(&snapshot)
            .await
            .map_err(HubError::io)?
        {
            match self.verify_snapshot_files(plan, &snapshot).await {
                Ok(()) => match self
                    .reuse_or_write_snapshot_manifest(plan, snapshot.clone())
                    .await
                {
                    Ok(existing) => {
                        remove_staging_dir(&staging).await?;
                        return Ok(existing);
                    }
                    Err(err) => {
                        self.quarantine_snapshot(&snapshot, err.to_string()).await?;
                    }
                },
                Err(err) => {
                    self.quarantine_snapshot(&snapshot, err.to_string()).await?;
                }
            }
        }
        let manifest = SnapshotManifest::from_plan(plan, snapshot.display().to_string());
        let manifest_digest = manifest.digest();
        let manifest_bytes = serde_json::to_vec_pretty(&manifest)
            .map_err(|err| HubError::invalid_response(format!("manifest JSON failed: {err}")))?;
        tokio::fs::write(staging.join(SNAPSHOT_MANIFEST_FILE), manifest_bytes)
            .await
            .map_err(HubError::io)?;
        if let Some(parent) = snapshot.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(HubError::io)?;
        }
        tokio::fs::rename(&staging, &snapshot)
            .await
            .map_err(HubError::io)?;
        Ok(PromotedSnapshot {
            path: snapshot,
            manifest,
            manifest_digest,
        })
    }

    pub async fn verify_existing_snapshot(
        &self,
        plan: &DownloadPlan,
    ) -> Result<Option<PromotedSnapshot>, HubError> {
        let snapshot = self.snapshot_path(plan);
        if !tokio::fs::try_exists(&snapshot)
            .await
            .map_err(HubError::io)?
        {
            return Ok(None);
        }
        if let Err(err) = self.verify_snapshot_files(plan, &snapshot).await {
            self.quarantine_snapshot(&snapshot, err.to_string()).await?;
            return Ok(None);
        }
        match self
            .reuse_or_write_snapshot_manifest(plan, snapshot.clone())
            .await
        {
            Ok(snapshot) => Ok(Some(snapshot)),
            Err(err) => {
                self.quarantine_snapshot(&snapshot, err.to_string()).await?;
                Ok(None)
            }
        }
    }

    pub async fn pull_plan(
        &self,
        client: &HubClient,
        plan: &DownloadPlan,
        token: Option<&str>,
    ) -> Result<PromotedSnapshot, HubError> {
        if let Some(snapshot) = self.verify_existing_snapshot(plan).await? {
            return Ok(snapshot);
        }
        let staging = self.create_staging_dir(plan).await?;
        let result = async {
            for file in &plan.files_to_download {
                client
                    .download_file_to(HubDownloadFileRequest {
                        repo_id: &plan.repo_id,
                        resolved_commit: &plan.resolved_commit,
                        path: &file.path,
                        destination: &staging.join(&file.path),
                        expected_size: file.size,
                        expected_sha256: file.sha256.as_deref(),
                        artifact_class: file.class,
                        token,
                    })
                    .await?;
            }
            self.promote_staging(plan, staging.clone()).await
        }
        .await;
        if let Err(err) = result {
            let _ = remove_staging_dir(&staging).await;
            return Err(err);
        }
        result
    }

    pub async fn list_snapshots(&self) -> Result<Vec<PromotedSnapshot>, HubError> {
        if !tokio::fs::try_exists(&self.root)
            .await
            .map_err(HubError::io)?
        {
            return Ok(Vec::new());
        }
        let mut snapshots = Vec::new();
        let repos_root = self.root.join("huggingface");
        if !tokio::fs::try_exists(&repos_root)
            .await
            .map_err(HubError::io)?
        {
            return Ok(snapshots);
        }
        let mut repos = tokio::fs::read_dir(&repos_root)
            .await
            .map_err(HubError::io)?;
        while let Some(repo) = repos.next_entry().await.map_err(HubError::io)? {
            if !repo.file_type().await.map_err(HubError::io)?.is_dir() {
                continue;
            }
            let snapshots_dir = repo.path().join("snapshots");
            if !tokio::fs::try_exists(&snapshots_dir)
                .await
                .map_err(HubError::io)?
            {
                continue;
            }
            let mut entries = tokio::fs::read_dir(&snapshots_dir)
                .await
                .map_err(HubError::io)?;
            while let Some(entry) = entries.next_entry().await.map_err(HubError::io)? {
                if !entry.file_type().await.map_err(HubError::io)?.is_dir() {
                    continue;
                }
                let path = entry.path();
                let manifest_path = path.join(SNAPSHOT_MANIFEST_FILE);
                if !tokio::fs::try_exists(&manifest_path)
                    .await
                    .map_err(HubError::io)?
                {
                    continue;
                }
                let bytes = tokio::fs::read(&manifest_path)
                    .await
                    .map_err(HubError::io)?;
                let manifest =
                    serde_json::from_slice::<SnapshotManifest>(&bytes).map_err(|err| {
                        HubError::integrity_failed(format!(
                            "invalid snapshot manifest `{}`: {err}",
                            manifest_path.display()
                        ))
                    })?;
                let manifest_digest = manifest.digest();
                snapshots.push(PromotedSnapshot {
                    path,
                    manifest,
                    manifest_digest,
                });
            }
        }
        snapshots.sort_by(|left, right| {
            left.manifest
                .repo_id
                .cmp(&right.manifest.repo_id)
                .then_with(|| {
                    left.manifest
                        .resolved_commit
                        .cmp(&right.manifest.resolved_commit)
                })
                .then_with(|| left.path.cmp(&right.path))
        });
        Ok(snapshots)
    }

    pub async fn snapshot_inventory(&self) -> Result<SnapshotInventory, HubError> {
        let mut ready_snapshots = Vec::new();
        let mut metadata_only_snapshots = Vec::new();
        for snapshot in self.list_snapshots().await? {
            let readiness = snapshot_readiness(&snapshot).await;
            match readiness {
                SnapshotReadiness::Ready => ready_snapshots.push(snapshot),
                SnapshotReadiness::MetadataOnly { .. } => {
                    metadata_only_snapshots.push(SnapshotRecord {
                        snapshot,
                        readiness,
                    });
                }
                SnapshotReadiness::Invalid { reason } => {
                    self.quarantine_snapshot(&snapshot.path, reason).await?;
                }
            }
        }
        Ok(SnapshotInventory {
            ready_snapshots,
            metadata_only_snapshots,
            quarantined_snapshots: self.list_quarantined_snapshots().await?,
        })
    }

    pub async fn inspect_snapshot(
        snapshot: impl AsRef<Path>,
    ) -> Result<PromotedSnapshot, HubError> {
        read_promoted_snapshot(snapshot.as_ref().to_path_buf()).await
    }

    pub async fn inspect_snapshot_readiness(
        snapshot: impl AsRef<Path>,
    ) -> Result<SnapshotRecord, HubError> {
        let snapshot = Self::inspect_snapshot(snapshot).await?;
        let readiness = snapshot_readiness(&snapshot).await;
        Ok(SnapshotRecord {
            snapshot,
            readiness,
        })
    }

    pub async fn verify_snapshot(
        snapshot: impl AsRef<Path>,
    ) -> Result<SnapshotVerification, HubError> {
        let snapshot = Self::inspect_snapshot(snapshot).await?;
        let canonical_snapshot_root = canonicalize_snapshot_root(&snapshot.path).await?;
        let mut verified_files = 0_u64;
        let mut verified_bytes = 0_u64;
        for file in &snapshot.manifest.files {
            verify_snapshot_file(
                &snapshot.path,
                &canonical_snapshot_root,
                &file.path,
                file.size,
                file.sha256.as_deref(),
                file.class,
            )
            .await?;
            verified_files += 1;
            verified_bytes += file.size;
        }
        Ok(SnapshotVerification {
            snapshot,
            verified_files,
            verified_bytes,
        })
    }

    pub async fn verify_runnable_snapshot(
        snapshot: impl AsRef<Path>,
    ) -> Result<SnapshotVerification, HubError> {
        let verification = Self::verify_snapshot(snapshot).await?;
        if let Err(reason) = validate_runnable_snapshot(&verification.snapshot).await {
            return Err(HubError::integrity_failed(reason));
        }
        Ok(verification)
    }

    pub async fn mark_snapshot_used(snapshot: impl AsRef<Path>) -> Result<SnapshotUsage, HubError> {
        Self::mark_snapshot_used_at(snapshot, Utc::now()).await
    }

    pub async fn mark_snapshot_used_at(
        snapshot: impl AsRef<Path>,
        last_used_at: DateTime<Utc>,
    ) -> Result<SnapshotUsage, HubError> {
        let usage = SnapshotUsage {
            schema_version: 1,
            last_used_at,
        };
        let bytes = serde_json::to_vec_pretty(&usage)
            .map_err(|err| HubError::invalid_response(format!("usage JSON failed: {err}")))?;
        tokio::fs::write(snapshot.as_ref().join(SNAPSHOT_USAGE_FILE), bytes)
            .await
            .map_err(HubError::io)?;
        Ok(usage)
    }

    pub async fn record_snapshot_alias(
        &self,
        alias: &str,
        snapshot: impl AsRef<Path>,
    ) -> Result<ModelAlias, HubError> {
        validate_alias(alias)?;
        let snapshot_path = snapshot.as_ref().to_path_buf();
        let manifest_digest = read_promoted_snapshot(snapshot_path.clone())
            .await
            .ok()
            .map(|snapshot| snapshot.manifest_digest);
        let record = ModelAlias {
            schema_version: 1,
            alias: alias.to_owned(),
            snapshot_path,
            manifest_digest,
            updated_at: Utc::now(),
        };
        let aliases_root = self.aliases_root();
        tokio::fs::create_dir_all(&aliases_root)
            .await
            .map_err(HubError::io)?;
        let bytes = serde_json::to_vec_pretty(&record)
            .map_err(|err| HubError::invalid_response(format!("alias JSON failed: {err}")))?;
        tokio::fs::write(aliases_root.join(alias_file_name(alias)), bytes)
            .await
            .map_err(HubError::io)?;
        Ok(record)
    }

    pub async fn resolve_snapshot_alias(&self, alias: &str) -> Result<PromotedSnapshot, HubError> {
        validate_alias(alias)?;
        let alias_path = self.aliases_root().join(alias_file_name(alias));
        let bytes = tokio::fs::read(&alias_path).await.map_err(|err| {
            if err.kind() == ErrorKind::NotFound {
                HubError::model_not_found(format!("model alias `{alias}` was not found"))
            } else {
                HubError::io(err)
            }
        })?;
        let record = serde_json::from_slice::<ModelAlias>(&bytes).map_err(|err| {
            HubError::integrity_failed(format!(
                "invalid model alias record `{}`: {err}",
                alias_path.display()
            ))
        })?;
        if record.alias != alias {
            return Err(HubError::integrity_failed(format!(
                "model alias record `{}` points at alias `{}` instead of `{alias}`",
                alias_path.display(),
                record.alias
            )));
        }
        let snapshot = read_promoted_snapshot(record.snapshot_path).await?;
        if let Some(expected_digest) = record.manifest_digest.as_deref()
            && expected_digest != snapshot.manifest_digest
        {
            return Err(HubError::integrity_failed(format!(
                "model alias `{alias}` manifest digest mismatch: alias recorded {expected_digest}, snapshot has {}",
                snapshot.manifest_digest
            )));
        }
        Ok(snapshot)
    }

    pub async fn list_aliases(&self) -> Result<Vec<ModelAlias>, HubError> {
        let aliases_root = self.aliases_root();
        if !tokio::fs::try_exists(&aliases_root)
            .await
            .map_err(HubError::io)?
        {
            return Ok(Vec::new());
        }
        let mut aliases = Vec::new();
        let mut entries = tokio::fs::read_dir(&aliases_root)
            .await
            .map_err(HubError::io)?;
        while let Some(entry) = entries.next_entry().await.map_err(HubError::io)? {
            if !entry.file_type().await.map_err(HubError::io)?.is_file() {
                continue;
            }
            let bytes = tokio::fs::read(entry.path()).await.map_err(HubError::io)?;
            aliases.push(serde_json::from_slice::<ModelAlias>(&bytes).map_err(|err| {
                HubError::integrity_failed(format!(
                    "invalid model alias record `{}`: {err}",
                    entry.path().display()
                ))
            })?);
        }
        aliases.sort_by(|left, right| left.alias.cmp(&right.alias));
        Ok(aliases)
    }

    pub async fn quarantine_snapshot(
        &self,
        snapshot: impl AsRef<Path>,
        reason: impl Into<String>,
    ) -> Result<QuarantinedSnapshot, HubError> {
        let original_path = snapshot.as_ref().to_path_buf();
        let manifest_digest = read_promoted_snapshot(original_path.clone())
            .await
            .ok()
            .map(|snapshot| snapshot.manifest_digest);
        let quarantine_root = quarantine_root_for_snapshot(&self.root, &original_path);
        tokio::fs::create_dir_all(&quarantine_root)
            .await
            .map_err(HubError::io)?;
        let snapshot_name = original_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("snapshot");
        let timestamp = Utc::now().timestamp_nanos_opt().unwrap_or_default();
        let counter = STAGING_COUNTER.fetch_add(1, Ordering::Relaxed);
        let quarantined_path =
            quarantine_root.join(format!("{snapshot_name}.quarantined.{timestamp}.{counter}"));
        let bytes = snapshot_manifest_bytes(&original_path).await.unwrap_or(0);
        tokio::fs::rename(&original_path, &quarantined_path)
            .await
            .map_err(HubError::io)?;
        let metadata = QuarantineMetadata {
            schema_version: 1,
            original_path,
            quarantined_path: quarantined_path.clone(),
            reason: reason.into(),
            quarantined_at: Utc::now(),
            manifest_digest,
        };
        let metadata_bytes = serde_json::to_vec_pretty(&metadata).map_err(|err| {
            HubError::invalid_response(format!("quarantine metadata JSON failed: {err}"))
        })?;
        tokio::fs::write(
            quarantined_path.join(QUARANTINE_MANIFEST_FILE),
            metadata_bytes,
        )
        .await
        .map_err(HubError::io)?;
        Ok(QuarantinedSnapshot {
            path: quarantined_path,
            metadata,
            bytes,
        })
    }

    pub async fn list_quarantined_snapshots(&self) -> Result<Vec<QuarantinedSnapshot>, HubError> {
        let mut snapshots = Vec::new();
        let repos_root = self.root.join("huggingface");
        if !tokio::fs::try_exists(&repos_root)
            .await
            .map_err(HubError::io)?
        {
            return Ok(snapshots);
        }
        let mut repos = tokio::fs::read_dir(&repos_root)
            .await
            .map_err(HubError::io)?;
        while let Some(repo) = repos.next_entry().await.map_err(HubError::io)? {
            if !repo.file_type().await.map_err(HubError::io)?.is_dir() {
                continue;
            }
            let quarantine_dir = repo.path().join("quarantine");
            if !tokio::fs::try_exists(&quarantine_dir)
                .await
                .map_err(HubError::io)?
            {
                continue;
            }
            let mut entries = tokio::fs::read_dir(&quarantine_dir)
                .await
                .map_err(HubError::io)?;
            while let Some(entry) = entries.next_entry().await.map_err(HubError::io)? {
                if !entry.file_type().await.map_err(HubError::io)?.is_dir() {
                    continue;
                }
                let path = entry.path();
                let metadata_path = path.join(QUARANTINE_MANIFEST_FILE);
                if !tokio::fs::try_exists(&metadata_path)
                    .await
                    .map_err(HubError::io)?
                {
                    continue;
                }
                let bytes = tokio::fs::read(&metadata_path)
                    .await
                    .map_err(HubError::io)?;
                let metadata =
                    serde_json::from_slice::<QuarantineMetadata>(&bytes).map_err(|err| {
                        HubError::integrity_failed(format!(
                            "invalid quarantine metadata `{}`: {err}",
                            metadata_path.display()
                        ))
                    })?;
                let bytes = snapshot_manifest_bytes(&path).await.unwrap_or(0);
                snapshots.push(QuarantinedSnapshot {
                    path,
                    metadata,
                    bytes,
                });
            }
        }
        snapshots.sort_by(|left, right| left.path.cmp(&right.path));
        Ok(snapshots)
    }

    pub async fn inspect_quarantined_snapshot(
        snapshot: impl AsRef<Path>,
    ) -> Result<QuarantinedSnapshot, HubError> {
        let path = snapshot.as_ref().to_path_buf();
        let metadata_path = path.join(QUARANTINE_MANIFEST_FILE);
        let bytes = tokio::fs::read(&metadata_path).await.map_err(|err| {
            HubError::integrity_failed(format!(
                "quarantine metadata `{}` is missing or unreadable: {err}",
                metadata_path.display()
            ))
        })?;
        let metadata = serde_json::from_slice::<QuarantineMetadata>(&bytes).map_err(|err| {
            HubError::integrity_failed(format!(
                "invalid quarantine metadata `{}`: {err}",
                metadata_path.display()
            ))
        })?;
        let bytes = snapshot_manifest_bytes(&path).await.unwrap_or(0);
        Ok(QuarantinedSnapshot {
            path,
            metadata,
            bytes,
        })
    }

    async fn verify_snapshot_files(
        &self,
        plan: &DownloadPlan,
        snapshot: &Path,
    ) -> Result<(), HubError> {
        let canonical_snapshot_root = canonicalize_snapshot_root(snapshot).await?;
        for file in &plan.files_to_download {
            verify_snapshot_file(
                snapshot,
                &canonical_snapshot_root,
                &file.path,
                file.size,
                file.sha256.as_deref(),
                file.class,
            )
            .await?;
        }
        Ok(())
    }

    async fn write_snapshot_manifest(
        &self,
        plan: &DownloadPlan,
        snapshot: PathBuf,
    ) -> Result<PromotedSnapshot, HubError> {
        let manifest = SnapshotManifest::from_plan(plan, snapshot.display().to_string());
        let manifest_digest = manifest.digest();
        let manifest_bytes = serde_json::to_vec_pretty(&manifest)
            .map_err(|err| HubError::invalid_response(format!("manifest JSON failed: {err}")))?;
        tokio::fs::write(snapshot.join(SNAPSHOT_MANIFEST_FILE), manifest_bytes)
            .await
            .map_err(HubError::io)?;
        Ok(PromotedSnapshot {
            path: snapshot,
            manifest,
            manifest_digest,
        })
    }

    async fn reuse_or_write_snapshot_manifest(
        &self,
        plan: &DownloadPlan,
        snapshot: PathBuf,
    ) -> Result<PromotedSnapshot, HubError> {
        let manifest_path = snapshot.join(SNAPSHOT_MANIFEST_FILE);
        if tokio::fs::try_exists(&manifest_path)
            .await
            .map_err(HubError::io)?
        {
            let existing = read_promoted_snapshot(snapshot.clone()).await?;
            if manifest_matches_plan(&existing.manifest, plan, &snapshot) {
                return Ok(existing);
            }
        }
        self.write_snapshot_manifest(plan, snapshot).await
    }

    fn repo_root(&self, repo_id: &HubRepoId) -> PathBuf {
        self.root
            .join("huggingface")
            .join(format!("models--{}", repo_id.as_str().replace('/', "--")))
    }

    fn aliases_root(&self) -> PathBuf {
        self.root.join("aliases")
    }
}

async fn remove_staging_dir(path: &Path) -> Result<(), HubError> {
    match tokio::fs::remove_dir_all(path).await {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == ErrorKind::NotFound => Ok(()),
        Err(err) => Err(HubError::io(err)),
    }
}

pub(crate) async fn remove_snapshot_dir(path: &Path) -> Result<(), HubError> {
    match tokio::fs::remove_dir_all(path).await {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == ErrorKind::NotFound => Ok(()),
        Err(err) => Err(HubError::io(err)),
    }
}

async fn snapshot_manifest_bytes(snapshot: &Path) -> Result<u64, HubError> {
    let manifest_path = snapshot.join(SNAPSHOT_MANIFEST_FILE);
    let bytes = tokio::fs::read(&manifest_path)
        .await
        .map_err(HubError::io)?;
    let manifest = serde_json::from_slice::<SnapshotManifest>(&bytes)
        .map_err(|err| HubError::integrity_failed(format!("invalid snapshot manifest: {err}")))?;
    Ok(manifest.files.iter().map(|file| file.size).sum())
}

fn quarantine_root_for_snapshot(model_home: &Path, snapshot: &Path) -> PathBuf {
    if snapshot
        .parent()
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        == Some("snapshots")
        && let Some(repo_root) = snapshot.parent().and_then(Path::parent)
    {
        return repo_root.join("quarantine");
    }
    model_home.join("quarantine")
}

fn validate_alias(alias: &str) -> Result<(), HubError> {
    if alias.is_empty()
        || alias == "."
        || alias == ".."
        || alias.bytes().any(|byte| byte == 0 || byte == b'/')
    {
        return Err(HubError::invalid_request(
            "model alias must be non-empty and must not contain path separators",
        ));
    }
    Ok(())
}

fn alias_file_name(alias: &str) -> String {
    let sanitized = alias
        .bytes()
        .map(|byte| {
            if byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b':') {
                byte as char
            } else {
                '_'
            }
        })
        .collect::<String>();
    let digest = hex::encode(Sha256::digest(alias.as_bytes()));
    format!("{sanitized}.{}.json", &digest[..16])
}

async fn snapshot_readiness(snapshot: &PromotedSnapshot) -> SnapshotReadiness {
    match validate_runnable_snapshot(snapshot).await {
        Ok(()) => SnapshotReadiness::Ready,
        Err(reason) if is_metadata_only_snapshot(snapshot) && missing_weights_reason(&reason) => {
            SnapshotReadiness::MetadataOnly { reason }
        }
        Err(reason) => SnapshotReadiness::Invalid { reason },
    }
}

async fn validate_runnable_snapshot(snapshot: &PromotedSnapshot) -> Result<(), String> {
    validate_manifest_files(snapshot).await?;
    validate_builtin_profile_metadata(&snapshot.manifest)?;
    validate_manifest_file_classes(&snapshot.manifest, &snapshot.path)?;
    validate_safetensors_index_shards(snapshot).await?;
    Ok(())
}

async fn validate_manifest_files(snapshot: &PromotedSnapshot) -> Result<(), String> {
    let canonical_snapshot_root = canonicalize_snapshot_root(&snapshot.path)
        .await
        .map_err(|err| err.to_string())?;
    for file in &snapshot.manifest.files {
        verify_snapshot_file(
            &snapshot.path,
            &canonical_snapshot_root,
            &file.path,
            file.size,
            file.sha256.as_deref(),
            file.class,
        )
        .await
        .map_err(|err| err.to_string())?;
    }
    Ok(())
}

fn validate_builtin_profile_metadata(manifest: &SnapshotManifest) -> Result<(), String> {
    let Some(profile) = crate::ModelProfile::builtin(&manifest.profile) else {
        return Ok(());
    };
    let mut mismatches = Vec::new();
    if manifest.family != profile.family {
        mismatches.push(format!(
            "family `{}`, expected `{}`",
            manifest.family, profile.family
        ));
    }
    if manifest.loader != profile.loader {
        mismatches.push(format!(
            "loader `{}`, expected `{}`",
            manifest.loader, profile.loader
        ));
    }
    if manifest.quantization != profile.quantization {
        mismatches.push(format!(
            "quantization `{}`, expected `{}`",
            manifest.quantization, profile.quantization
        ));
    }
    if mismatches.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "snapshot manifest profile `{}` is stale or inconsistent: {}",
            manifest.profile,
            mismatches.join(", ")
        ))
    }
}

fn validate_manifest_file_classes(manifest: &SnapshotManifest, path: &Path) -> Result<(), String> {
    let has_config = manifest
        .files
        .iter()
        .any(|file| file.class == crate::ArtifactClass::Config);
    let has_tokenizer = manifest
        .files
        .iter()
        .any(|file| file.class == crate::ArtifactClass::Tokenizer);
    let has_weights = manifest
        .files
        .iter()
        .any(|file| file.class == crate::ArtifactClass::Weights);
    if !has_config {
        return Err(format!(
            "snapshot `{}` is missing config artifacts",
            path.display()
        ));
    }
    if !has_tokenizer {
        return Err(format!(
            "snapshot `{}` is missing tokenizer artifacts",
            path.display()
        ));
    }
    if !has_weights {
        return Err(format!(
            "snapshot `{}` contains no weight files; pull without --metadata-only before serving",
            path.display()
        ));
    }
    Ok(())
}

async fn validate_safetensors_index_shards(snapshot: &PromotedSnapshot) -> Result<(), String> {
    for index in snapshot
        .manifest
        .files
        .iter()
        .filter(|file| file.path.ends_with(".safetensors.index.json"))
    {
        let index_path = snapshot.path.join(&index.path);
        let bytes = tokio::fs::read(&index_path).await.map_err(|err| {
            format!(
                "could not read safetensors index `{}`: {err}",
                index_path.display()
            )
        })?;
        let index = serde_json::from_slice::<RawSafetensorsIndex>(&bytes)
            .map_err(|err| format!("invalid safetensors index JSON: {err}"))?;
        let shards = index
            .weight_map
            .values()
            .map(|shard| {
                validate_artifact_path(shard).map_err(|err| err.to_string())?;
                Ok(shard.as_str())
            })
            .collect::<Result<BTreeSet<_>, String>>()?;
        if shards.is_empty() {
            return Err("safetensors index does not reference any weight shards".to_owned());
        }
        for shard in shards {
            let Some(file) = snapshot
                .manifest
                .files
                .iter()
                .find(|file| file.path == shard)
            else {
                return Err(format!(
                    "safetensors index references shard `{shard}` that is not recorded in the snapshot manifest"
                ));
            };
            if file.class != crate::ArtifactClass::Weights {
                return Err(format!(
                    "safetensors index references shard `{shard}` recorded as {:?}, expected weights",
                    file.class
                ));
            }
        }
    }
    Ok(())
}

fn is_metadata_only_snapshot(snapshot: &PromotedSnapshot) -> bool {
    snapshot
        .path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".metadata-only"))
}

fn missing_weights_reason(reason: &str) -> bool {
    reason.contains("contains no weight files")
}

#[derive(Debug, Deserialize)]
struct RawSafetensorsIndex {
    weight_map: std::collections::BTreeMap<String, String>,
}
