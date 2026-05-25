use crate::plan::is_commit_hash;
use crate::{HubError, HubFile};
use serde::{Deserialize, Serialize};
use serde_json::Value;

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
            let etag = lfs
                .and_then(|lfs| lfs.get("oid"))
                .or_else(|| sibling.get("blobId"))
                .or_else(|| sibling.get("blob_id"))
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
