use crate::HubError;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub struct HubRepoId {
    repo_type: RepoType,
    id: String,
}

impl HubRepoId {
    pub fn model(id: impl Into<String>) -> Result<Self, HubError> {
        let id = id.into();
        let Some((namespace, name)) = id.split_once('/') else {
            return Err(HubError::invalid_request("repo id must be org/name"));
        };
        if name.contains('/') || !is_safe_repo_component(namespace) || !is_safe_repo_component(name)
        {
            return Err(HubError::invalid_request(
                "repo id must be exactly two safe path components",
            ));
        }
        Ok(Self {
            repo_type: RepoType::Model,
            id,
        })
    }

    pub fn as_str(&self) -> &str {
        &self.id
    }

    pub(crate) fn components(&self) -> (&str, &str) {
        self.id
            .split_once('/')
            .expect("HubRepoId is validated as two components")
    }
}

fn is_safe_repo_component(component: &str) -> bool {
    !component.is_empty()
        && component != "."
        && component != ".."
        && component
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
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
