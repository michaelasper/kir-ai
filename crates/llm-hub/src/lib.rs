use serde::{Deserialize, Serialize};
use thiserror::Error;

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
            loader: "native-metal".to_owned(),
            quantization: "4bit".to_owned(),
            allow_patterns: vec![
                "*.json".to_owned(),
                "tokenizer*".to_owned(),
                "*.safetensors".to_owned(),
                "*.safetensors.index.json".to_owned(),
            ],
            ignore_patterns: vec![
                "*.bin".to_owned(),
                "*.pt".to_owned(),
                "optimizer*".to_owned(),
                "training_args.bin".to_owned(),
            ],
        }
    }
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
