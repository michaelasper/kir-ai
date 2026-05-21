use crate::plan::validate_artifact_path;
use crate::{ArtifactClass, DownloadPlan, HubError, RepoType};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::AsyncReadExt;

pub const SNAPSHOT_MANIFEST_FILE: &str = "llm-engine-manifest.json";
#[cfg(test)]
static SNAPSHOT_FILE_VERIFICATION_COUNT: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotManifest {
    pub schema_version: u32,
    pub source: String,
    pub repo_type: RepoType,
    pub repo_id: String,
    pub requested_revision: String,
    pub resolved_commit: String,
    pub profile: String,
    pub family: String,
    pub loader: String,
    pub quantization: String,
    pub created_at: DateTime<Utc>,
    pub snapshot_path: String,
    pub files: Vec<ManifestFile>,
    pub allow_patterns: Vec<String>,
    pub ignore_patterns: Vec<String>,
}

impl SnapshotManifest {
    pub fn from_plan(plan: &DownloadPlan, snapshot_path: impl Into<String>) -> Self {
        Self {
            schema_version: 1,
            source: "huggingface".to_owned(),
            repo_type: RepoType::Model,
            repo_id: plan.repo_id.as_str().to_owned(),
            requested_revision: plan.requested_revision.clone(),
            resolved_commit: plan.resolved_commit.clone(),
            profile: plan.profile.name.clone(),
            family: plan.profile.family.clone(),
            loader: plan.profile.loader.clone(),
            quantization: plan.profile.quantization.clone(),
            created_at: Utc::now(),
            snapshot_path: snapshot_path.into(),
            files: plan
                .files_to_download
                .iter()
                .map(|file| ManifestFile {
                    path: file.path.clone(),
                    size: file.size,
                    etag: file.etag.clone(),
                    sha256: file.sha256.clone(),
                    class: file.class,
                })
                .collect(),
            allow_patterns: plan.profile.allow_patterns.clone(),
            ignore_patterns: plan.profile.ignore_patterns.clone(),
        }
    }

    pub fn digest(&self) -> String {
        let bytes = serde_json::to_vec(self).expect("snapshot manifest serializes");
        hex::encode(Sha256::digest(bytes))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestFile {
    pub path: String,
    pub size: u64,
    pub etag: Option<String>,
    pub sha256: Option<String>,
    pub class: ArtifactClass,
}

#[derive(Debug, Clone)]
pub struct PromotedSnapshot {
    pub path: PathBuf,
    pub manifest: SnapshotManifest,
    pub manifest_digest: String,
}

#[derive(Debug, Clone)]
pub struct SnapshotVerification {
    pub snapshot: PromotedSnapshot,
    pub verified_files: u64,
    pub verified_bytes: u64,
}

pub(crate) async fn read_promoted_snapshot(path: PathBuf) -> Result<PromotedSnapshot, HubError> {
    let manifest_path = path.join(SNAPSHOT_MANIFEST_FILE);
    let bytes = tokio::fs::read(&manifest_path).await.map_err(|err| {
        HubError::integrity_failed(format!(
            "snapshot manifest `{}` is missing or unreadable: {err}",
            manifest_path.display()
        ))
    })?;
    let manifest = serde_json::from_slice::<SnapshotManifest>(&bytes).map_err(|err| {
        HubError::integrity_failed(format!(
            "invalid snapshot manifest `{}`: {err}",
            manifest_path.display()
        ))
    })?;
    let manifest_digest = manifest.digest();
    Ok(PromotedSnapshot {
        path,
        manifest,
        manifest_digest,
    })
}

pub(crate) async fn verify_promoted_snapshot(
    snapshot: PromotedSnapshot,
) -> Result<SnapshotVerification, HubError> {
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

pub(crate) fn manifest_matches_plan(
    manifest: &SnapshotManifest,
    plan: &DownloadPlan,
    snapshot: &Path,
) -> bool {
    let mut expected = SnapshotManifest::from_plan(plan, snapshot.display().to_string());
    expected.created_at = manifest.created_at;
    manifest == &expected
}

pub(crate) async fn canonicalize_snapshot_root(snapshot: &Path) -> Result<PathBuf, HubError> {
    tokio::fs::canonicalize(snapshot).await.map_err(|err| {
        HubError::integrity_failed(format!(
            "snapshot root `{}` is missing or unreadable: {err}",
            snapshot.display()
        ))
    })
}

pub(crate) async fn verify_snapshot_file(
    snapshot_root: &Path,
    canonical_snapshot_root: &Path,
    relative_path: &str,
    expected_size: u64,
    expected_sha256: Option<&str>,
    artifact_class: ArtifactClass,
) -> Result<(), HubError> {
    #[cfg(test)]
    SNAPSHOT_FILE_VERIFICATION_COUNT.fetch_add(1, Ordering::Relaxed);
    validate_artifact_path(relative_path)?;
    let path = snapshot_root.join(relative_path);
    let metadata = tokio::fs::symlink_metadata(&path).await.map_err(|err| {
        HubError::integrity_failed(format!(
            "snapshot file `{}` is missing or unreadable: {err}",
            path.display()
        ))
    })?;
    if metadata.file_type().is_symlink() {
        return Err(HubError::integrity_failed(format!(
            "snapshot path `{}` is a symlink",
            path.display()
        )));
    }
    let canonical_path = tokio::fs::canonicalize(&path).await.map_err(|err| {
        HubError::integrity_failed(format!(
            "snapshot file `{}` is missing or unreadable: {err}",
            path.display()
        ))
    })?;
    if !canonical_path.starts_with(canonical_snapshot_root) {
        return Err(HubError::integrity_failed(format!(
            "snapshot file `{}` resolves outside snapshot root `{}`",
            path.display(),
            snapshot_root.display()
        )));
    }
    if !metadata.is_file() {
        return Err(HubError::integrity_failed(format!(
            "snapshot path `{}` is not a file",
            path.display()
        )));
    }
    if metadata.len() != expected_size {
        return Err(HubError::integrity_failed(format!(
            "snapshot file `{}` has size {}, expected {}",
            path.display(),
            metadata.len(),
            expected_size
        )));
    }
    verify_file_sha256_for_artifact(&canonical_path, expected_sha256, artifact_class).await
}

#[cfg(test)]
pub(crate) fn reset_snapshot_file_verification_count_for_tests() {
    SNAPSHOT_FILE_VERIFICATION_COUNT.store(0, Ordering::Relaxed);
}

#[cfg(test)]
pub(crate) fn snapshot_file_verification_count_for_tests() -> u64 {
    SNAPSHOT_FILE_VERIFICATION_COUNT.load(Ordering::Relaxed)
}

pub(crate) async fn verify_file_sha256(path: &Path, expected_sha256: &str) -> Result<(), HubError> {
    let mut file = tokio::fs::File::open(path).await.map_err(HubError::io)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer).await.map_err(HubError::io)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let actual = hex::encode(hasher.finalize());
    if actual != expected_sha256 {
        return Err(HubError::integrity_failed(format!(
            "snapshot file `{}` has sha256 {actual}, expected {expected_sha256}",
            path.display()
        )));
    }
    Ok(())
}

pub(crate) async fn verify_file_sha256_for_artifact(
    path: &Path,
    expected_sha256: Option<&str>,
    artifact_class: ArtifactClass,
) -> Result<(), HubError> {
    match expected_sha256 {
        Some(expected_sha256) => verify_file_sha256(path, expected_sha256).await,
        None if artifact_class == ArtifactClass::Weights => {
            Err(HubError::integrity_failed(format!(
                "snapshot weight file `{}` is missing sha256 digest",
                path.display()
            )))
        }
        None => Ok(()),
    }
}
