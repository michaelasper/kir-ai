use chrono::{DateTime, Utc};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use thiserror::Error;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use url::Url;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HubRepoId {
    repo_type: RepoType,
    id: String,
}

impl HubRepoId {
    pub fn model(id: impl Into<String>) -> Result<Self, HubError> {
        let id = id.into();
        if !id.contains('/') || id.starts_with('/') || id.ends_with('/') {
            return Err(HubError::invalid_request("repo id must be org/name"));
        }
        Ok(Self {
            repo_type: RepoType::Model,
            id,
        })
    }

    pub fn as_str(&self) -> &str {
        &self.id
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RepoType {
    Model,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HubFile {
    pub path: String,
    pub size: u64,
    pub etag: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HubModelInfo {
    pub repo_id: String,
    pub resolved_commit: String,
    pub files: Vec<HubFile>,
}

impl HubModelInfo {
    pub fn from_api_json(value: Value) -> Result<Self, HubError> {
        let repo_id = value
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| HubError::invalid_response("Hugging Face model info missing id"))?
            .to_owned();
        let resolved_commit = value
            .get("sha")
            .and_then(Value::as_str)
            .ok_or_else(|| HubError::invalid_response("Hugging Face model info missing sha"))?
            .to_owned();
        if !is_commit_hash(&resolved_commit) {
            return Err(HubError::model_revision_unresolved(
                "Hugging Face model info sha was not an immutable commit",
            ));
        }
        let siblings = value
            .get("siblings")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                HubError::invalid_response("Hugging Face model info missing siblings")
            })?;
        let mut files = Vec::with_capacity(siblings.len());
        for sibling in siblings {
            let path = sibling
                .get("rfilename")
                .or_else(|| sibling.get("path"))
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    HubError::invalid_response("Hugging Face sibling missing rfilename")
                })?;
            let lfs = sibling.get("lfs");
            let size = sibling
                .get("size")
                .and_then(Value::as_u64)
                .or_else(|| lfs.and_then(|lfs| lfs.get("size")).and_then(Value::as_u64))
                .unwrap_or(0);
            let etag = sibling
                .get("blobId")
                .or_else(|| sibling.get("blob_id"))
                .or_else(|| lfs.and_then(|lfs| lfs.get("oid")))
                .and_then(Value::as_str);
            files.push(HubFile::new(path, size, etag));
        }
        Ok(Self {
            repo_id,
            resolved_commit,
            files,
        })
    }
}

#[derive(Debug, Clone)]
pub struct HubClient {
    endpoint: Url,
    client: reqwest::Client,
}

impl Default for HubClient {
    fn default() -> Self {
        Self {
            endpoint: Url::parse("https://huggingface.co").expect("static Hugging Face URL"),
            client: reqwest::Client::new(),
        }
    }
}

impl HubClient {
    pub fn new(endpoint: Url) -> Self {
        Self {
            endpoint,
            client: reqwest::Client::new(),
        }
    }

    pub async fn model_info(
        &self,
        repo_id: &HubRepoId,
        revision: &str,
        token: Option<&str>,
    ) -> Result<HubModelInfo, HubError> {
        let mut url = self.endpoint.clone();
        url.set_path(&format!(
            "/api/models/{}/revision/{}",
            repo_id.as_str(),
            revision
        ));
        let mut request = self
            .client
            .get(url)
            .query(&[("blobs", "true"), ("securityStatus", "true")]);
        if let Some(token) = token {
            request = request.bearer_auth(token);
        }
        let response = request.send().await.map_err(HubError::network)?;
        let status = response.status();
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            return Err(HubError::auth_failed("Hugging Face authentication failed"));
        }
        if status == reqwest::StatusCode::NOT_FOUND {
            return Err(HubError::model_not_found(format!(
                "model repo `{}` revision `{revision}` was not found",
                repo_id.as_str()
            )));
        }
        if !status.is_success() {
            return Err(HubError::network(format!(
                "Hugging Face API returned HTTP {status}"
            )));
        }
        let value = response.json::<Value>().await.map_err(HubError::network)?;
        HubModelInfo::from_api_json(value)
    }

    pub async fn plan_model(
        &self,
        repo_id: HubRepoId,
        revision: &str,
        profile: ModelProfile,
        token: Option<&str>,
    ) -> Result<DownloadPlan, HubError> {
        let info = self.model_info(&repo_id, revision, token).await?;
        build_download_plan(
            repo_id,
            revision,
            info.resolved_commit,
            profile,
            info.files,
            &[],
        )
    }

    async fn download_file_to(&self, request: HubDownloadFileRequest<'_>) -> Result<(), HubError> {
        let HubDownloadFileRequest {
            repo_id,
            resolved_commit,
            path,
            destination,
            expected_size,
            expected_sha256,
            token,
        } = request;
        validate_artifact_path(path)?;
        if let Some(parent) = destination.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(HubError::io)?;
        }
        let existing_len = match tokio::fs::metadata(destination).await {
            Ok(metadata) if metadata.len() == expected_size => {
                verify_file_sha256(destination, expected_sha256).await?;
                return Ok(());
            }
            Ok(metadata) if metadata.len() < expected_size => metadata.len(),
            Ok(_) => {
                tokio::fs::remove_file(destination)
                    .await
                    .map_err(HubError::io)?;
                0
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => 0,
            Err(err) => return Err(HubError::io(err)),
        };
        let mut url = self.endpoint.clone();
        url.set_path(&format!(
            "/{}/resolve/{}/{}",
            repo_id.as_str(),
            resolved_commit,
            path
        ));
        let mut request = self.client.get(url);
        if let Some(token) = token {
            request = request.bearer_auth(token);
        }
        if existing_len > 0 {
            request = request.header(reqwest::header::RANGE, format!("bytes={existing_len}-"));
        }
        let response = request.send().await.map_err(HubError::network)?;
        let status = response.status();
        if !(status.is_success() || status == reqwest::StatusCode::PARTIAL_CONTENT) {
            return Err(HubError::network(format!(
                "download for `{path}` returned HTTP {status}"
            )));
        }
        let mut file = if existing_len > 0 && status == reqwest::StatusCode::PARTIAL_CONTENT {
            tokio::fs::OpenOptions::new()
                .append(true)
                .open(destination)
                .await
                .map_err(HubError::io)?
        } else {
            tokio::fs::File::create(destination)
                .await
                .map_err(HubError::io)?
        };
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(HubError::network)?;
            file.write_all(&chunk).await.map_err(HubError::io)?;
        }
        file.flush().await.map_err(HubError::io)?;
        let final_len = tokio::fs::metadata(destination)
            .await
            .map_err(HubError::io)?
            .len();
        if final_len != expected_size {
            return Err(HubError::integrity_failed(format!(
                "downloaded `{path}` size {final_len} did not match expected {expected_size}"
            )));
        }
        verify_file_sha256(destination, expected_sha256).await?;
        Ok(())
    }
}

struct HubDownloadFileRequest<'a> {
    repo_id: &'a HubRepoId,
    resolved_commit: &'a str,
    path: &'a str,
    destination: &'a Path,
    expected_size: u64,
    expected_sha256: Option<&'a str>,
    token: Option<&'a str>,
}

impl HubFile {
    pub fn new(path: impl Into<String>, size: u64, etag: Option<&str>) -> Self {
        Self {
            path: path.into(),
            size,
            etag: etag.map(str::to_owned),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelProfile {
    pub name: String,
    pub family: String,
    pub loader: String,
    pub quantization: String,
    pub allow_patterns: Vec<String>,
    pub ignore_patterns: Vec<String>,
}

impl ModelProfile {
    pub fn qwen36_mlx_4bit() -> Self {
        Self {
            name: "qwen36-mlx-4bit".to_owned(),
            family: "qwen".to_owned(),
            loader: "mlx".to_owned(),
            quantization: "4bit".to_owned(),
            allow_patterns: qwen_static_and_safetensors_patterns(),
            ignore_patterns: qwen_ignore_patterns(),
        }
    }

    pub fn qwen36_safetensors_bf16() -> Self {
        Self {
            name: "qwen36-safetensors-bf16".to_owned(),
            family: "qwen".to_owned(),
            loader: "native-metal".to_owned(),
            quantization: "bf16".to_owned(),
            allow_patterns: qwen_static_and_safetensors_patterns(),
            ignore_patterns: qwen_ignore_patterns(),
        }
    }
}

fn qwen_static_and_safetensors_patterns() -> Vec<String> {
    vec![
        "*.json".to_owned(),
        "*.jinja".to_owned(),
        "*.txt".to_owned(),
        "tokenizer*".to_owned(),
        "README.md".to_owned(),
        "LICENSE*".to_owned(),
        "*.safetensors".to_owned(),
        "*.safetensors.index.json".to_owned(),
    ]
}

fn qwen_ignore_patterns() -> Vec<String> {
    vec![
        "*.bin".to_owned(),
        "*.pt".to_owned(),
        "optimizer*".to_owned(),
        "training_args.bin".to_owned(),
    ]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactClass {
    Config,
    Tokenizer,
    Weights,
    Quantization,
    License,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlannedFile {
    pub path: String,
    pub size: u64,
    pub etag: Option<String>,
    pub sha256: Option<String>,
    pub class: ArtifactClass,
    pub cached: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DownloadPlan {
    pub repo_id: HubRepoId,
    pub requested_revision: String,
    pub resolved_commit: String,
    pub profile: ModelProfile,
    pub files_to_download: Vec<PlannedFile>,
    pub skipped_files: Vec<String>,
    pub total_bytes_to_download: u64,
    pub total_final_disk_bytes: u64,
    pub metadata_only: bool,
}

impl DownloadPlan {
    pub fn metadata_only(&self) -> Self {
        let mut plan = self.clone();
        plan.files_to_download
            .retain(|file| file.class != ArtifactClass::Weights);
        plan.metadata_only = true;
        plan.recompute_totals();
        plan
    }

    fn recompute_totals(&mut self) {
        self.total_bytes_to_download = self
            .files_to_download
            .iter()
            .filter(|file| !file.cached)
            .map(|file| file.size)
            .sum();
        self.total_final_disk_bytes = self.files_to_download.iter().map(|file| file.size).sum();
    }
}

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
pub struct ModelStore {
    root: PathBuf,
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
            .join(if plan.metadata_only {
                format!("{}.metadata-only", plan.resolved_commit)
            } else {
                plan.resolved_commit.clone()
            })
    }

    pub async fn create_staging_dir(&self, plan: &DownloadPlan) -> Result<PathBuf, HubError> {
        let staging = self
            .repo_root(&plan.repo_id)
            .join("staging")
            .join(format!("{}.partial", plan.resolved_commit));
        tokio::fs::create_dir_all(&staging)
            .await
            .map_err(HubError::io)?;
        Ok(staging)
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
            return Err(HubError::invalid_request(format!(
                "snapshot already exists at {}",
                snapshot.display()
            )));
        }
        let manifest = SnapshotManifest::from_plan(plan, snapshot.display().to_string());
        let manifest_digest = manifest.digest();
        let manifest_bytes = serde_json::to_vec_pretty(&manifest)
            .map_err(|err| HubError::invalid_response(format!("manifest JSON failed: {err}")))?;
        tokio::fs::write(staging.join("llm-engine-manifest.json"), manifest_bytes)
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
        self.verify_snapshot_files(plan, &snapshot).await?;
        Ok(Some(self.write_snapshot_manifest(plan, snapshot).await?))
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
        for file in &plan.files_to_download {
            client
                .download_file_to(HubDownloadFileRequest {
                    repo_id: &plan.repo_id,
                    resolved_commit: &plan.resolved_commit,
                    path: &file.path,
                    destination: &staging.join(&file.path),
                    expected_size: file.size,
                    expected_sha256: file.sha256.as_deref(),
                    token,
                })
                .await?;
        }
        self.promote_staging(plan, staging).await
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
                let manifest_path = path.join("llm-engine-manifest.json");
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

    async fn verify_snapshot_files(
        &self,
        plan: &DownloadPlan,
        snapshot: &Path,
    ) -> Result<(), HubError> {
        for file in &plan.files_to_download {
            let path = snapshot.join(&file.path);
            let metadata = tokio::fs::metadata(&path).await.map_err(|err| {
                HubError::integrity_failed(format!(
                    "snapshot file `{}` is missing or unreadable: {err}",
                    path.display()
                ))
            })?;
            if !metadata.is_file() {
                return Err(HubError::integrity_failed(format!(
                    "snapshot path `{}` is not a file",
                    path.display()
                )));
            }
            if metadata.len() != file.size {
                return Err(HubError::integrity_failed(format!(
                    "snapshot file `{}` has size {}, expected {}",
                    path.display(),
                    metadata.len(),
                    file.size
                )));
            }
            verify_file_sha256(&path, file.sha256.as_deref()).await?;
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
        tokio::fs::write(snapshot.join("llm-engine-manifest.json"), manifest_bytes)
            .await
            .map_err(HubError::io)?;
        Ok(PromotedSnapshot {
            path: snapshot,
            manifest,
            manifest_digest,
        })
    }

    fn repo_root(&self, repo_id: &HubRepoId) -> PathBuf {
        self.root
            .join("huggingface")
            .join(format!("models--{}", repo_id.as_str().replace('/', "--")))
    }
}

#[derive(Debug, Clone)]
pub struct PromotedSnapshot {
    pub path: PathBuf,
    pub manifest: SnapshotManifest,
    pub manifest_digest: String,
}

pub fn build_download_plan(
    repo_id: HubRepoId,
    requested_revision: impl Into<String>,
    resolved_commit: impl Into<String>,
    profile: ModelProfile,
    files: Vec<HubFile>,
    cached_paths: &[String],
) -> Result<DownloadPlan, HubError> {
    let requested_revision = requested_revision.into();
    let resolved_commit = resolved_commit.into();
    if !is_commit_hash(&resolved_commit) {
        return Err(HubError::model_revision_unresolved(
            "resolved commit must be a 40-character immutable SHA",
        ));
    }

    let mut selected = Vec::new();
    let mut skipped = Vec::new();
    for file in files {
        validate_artifact_path(&file.path)?;
        if matches_any(&profile.ignore_patterns, &file.path)
            || !matches_any(&profile.allow_patterns, &file.path)
        {
            skipped.push(file.path);
            continue;
        }
        let cached = cached_paths.iter().any(|path| path == &file.path);
        selected.push(PlannedFile {
            class: classify_artifact(&file.path),
            path: file.path,
            size: file.size,
            sha256: file.etag.as_deref().and_then(normalize_sha256),
            etag: file.etag,
            cached,
        });
    }
    selected.sort_by(|a, b| {
        artifact_order(a.class)
            .cmp(&artifact_order(b.class))
            .then(a.path.cmp(&b.path))
    });
    skipped.sort();
    let total_bytes_to_download = selected
        .iter()
        .filter(|file| !file.cached)
        .map(|file| file.size)
        .sum();
    let total_final_disk_bytes = selected.iter().map(|file| file.size).sum();
    Ok(DownloadPlan {
        repo_id,
        requested_revision,
        resolved_commit,
        profile,
        files_to_download: selected,
        skipped_files: skipped,
        total_bytes_to_download,
        total_final_disk_bytes,
        metadata_only: false,
    })
}

#[derive(Debug, Error)]
#[error("{code}: {message}")]
pub struct HubError {
    code: &'static str,
    message: String,
}

impl HubError {
    pub fn code(&self) -> &'static str {
        self.code
    }

    fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            code: "invalid_request",
            message: message.into(),
        }
    }

    fn invalid_response(message: impl Into<String>) -> Self {
        Self {
            code: "model_integrity_failed",
            message: message.into(),
        }
    }

    fn integrity_failed(message: impl Into<String>) -> Self {
        Self {
            code: "model_integrity_failed",
            message: message.into(),
        }
    }

    fn auth_failed(message: impl Into<String>) -> Self {
        Self {
            code: "model_auth_failed",
            message: message.into(),
        }
    }

    fn model_not_found(message: impl Into<String>) -> Self {
        Self {
            code: "model_not_found",
            message: message.into(),
        }
    }

    fn network(message: impl ToString) -> Self {
        Self {
            code: "model_download_interrupted",
            message: message.to_string(),
        }
    }

    fn io(message: impl ToString) -> Self {
        Self {
            code: "model_download_interrupted",
            message: message.to_string(),
        }
    }

    fn model_revision_unresolved(message: impl Into<String>) -> Self {
        Self {
            code: "model_revision_unresolved",
            message: message.into(),
        }
    }
}

fn is_commit_hash(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn normalize_sha256(value: &str) -> Option<String> {
    let trimmed = value.trim_matches('"');
    (trimmed.len() == 64 && trimmed.bytes().all(|b| b.is_ascii_hexdigit()))
        .then(|| trimmed.to_ascii_lowercase())
}

fn validate_artifact_path(path: &str) -> Result<(), HubError> {
    if path.is_empty()
        || path.starts_with('/')
        || path.contains('\\')
        || path.bytes().any(|byte| byte == 0)
        || path
            .split('/')
            .any(|component| component.is_empty() || component == "." || component == "..")
    {
        return Err(HubError::invalid_request(format!(
            "unsafe Hugging Face artifact path `{path}`"
        )));
    }
    Ok(())
}

async fn verify_file_sha256(path: &Path, expected_sha256: Option<&str>) -> Result<(), HubError> {
    let Some(expected_sha256) = expected_sha256 else {
        return Ok(());
    };
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

fn matches_any(patterns: &[String], path: &str) -> bool {
    patterns
        .iter()
        .any(|pattern| matches_pattern(pattern, path))
}

fn matches_pattern(pattern: &str, path: &str) -> bool {
    if pattern == path {
        return true;
    }
    if let Some(suffix) = pattern.strip_prefix('*') {
        return path.ends_with(suffix);
    }
    if let Some(prefix) = pattern.strip_suffix('*') {
        return path.starts_with(prefix);
    }
    false
}

fn classify_artifact(path: &str) -> ArtifactClass {
    match path {
        "config.json" | "generation_config.json" => ArtifactClass::Config,
        "tokenizer.json" | "tokenizer_config.json" => ArtifactClass::Tokenizer,
        "README.md" | "LICENSE" | "LICENSE.txt" => ArtifactClass::License,
        _ if path.starts_with("tokenizer") => ArtifactClass::Tokenizer,
        _ if path.ends_with(".jinja") || path == "merges.txt" || path == "vocab.json" => {
            ArtifactClass::Tokenizer
        }
        _ if path.ends_with(".safetensors") || path.ends_with(".gguf") => ArtifactClass::Weights,
        _ if path.contains("quant") => ArtifactClass::Quantization,
        _ => ArtifactClass::Other,
    }
}

fn artifact_order(class: ArtifactClass) -> u8 {
    match class {
        ArtifactClass::Config => 0,
        ArtifactClass::Tokenizer => 1,
        ArtifactClass::Quantization => 2,
        ArtifactClass::Weights => 3,
        ArtifactClass::License => 4,
        ArtifactClass::Other => 5,
    }
}
