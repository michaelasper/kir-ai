use crate::{HubError, HubFile, HubRepoId, ModelProfile};
use serde::{Deserialize, Serialize};

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

pub(crate) fn snapshot_dir_name(plan: &DownloadPlan) -> String {
    let mut name = format!(
        "{}.{}",
        plan.resolved_commit,
        safe_path_component(&plan.profile.name)
    );
    if plan.metadata_only {
        name.push_str(".metadata-only");
    }
    name
}

fn safe_path_component(value: &str) -> String {
    let component = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    if component.is_empty() {
        "profile".to_owned()
    } else {
        component
    }
}

pub(crate) fn is_commit_hash(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn normalize_sha256(value: &str) -> Option<String> {
    let trimmed = value.trim_matches('"');
    (trimmed.len() == 64 && trimmed.bytes().all(|b| b.is_ascii_hexdigit()))
        .then(|| trimmed.to_ascii_lowercase())
}

pub(crate) fn validate_artifact_path(path: &str) -> Result<(), HubError> {
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
