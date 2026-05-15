use crate::ModelSpecError;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetensorsIndex {
    pub total_size_bytes: u64,
    weight_map: BTreeMap<String, String>,
}

impl SafetensorsIndex {
    pub fn from_json(json: &str) -> Result<Self, ModelSpecError> {
        let raw: RawSafetensorsIndex = serde_json::from_str(json)
            .map_err(|err| ModelSpecError::invalid_request(format!("invalid index JSON: {err}")))?;
        for shard_path in raw.weight_map.values() {
            validate_safetensors_shard_path(shard_path)?;
        }
        let total_size_bytes = raw.metadata.total_size.round() as u64;
        Ok(Self {
            total_size_bytes,
            weight_map: raw.weight_map,
        })
    }

    pub fn single_file(
        total_size_bytes: u64,
        shard_path: impl Into<String>,
        tensor_names: impl IntoIterator<Item = String>,
    ) -> Result<Self, ModelSpecError> {
        let shard_path = shard_path.into();
        validate_safetensors_shard_path(&shard_path)?;
        let weight_map = tensor_names
            .into_iter()
            .map(|name| (name, shard_path.clone()))
            .collect::<BTreeMap<_, _>>();
        if weight_map.is_empty() {
            return Err(ModelSpecError::invalid_request(
                "safetensors file does not contain tensors",
            ));
        }
        Ok(Self {
            total_size_bytes,
            weight_map,
        })
    }

    pub fn tensor_count(&self) -> usize {
        self.weight_map.len()
    }

    pub fn shard_count(&self) -> usize {
        self.weight_map.values().collect::<BTreeSet<_>>().len()
    }

    pub fn shard_paths(&self) -> Vec<&str> {
        self.weight_map
            .values()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.weight_map.keys().map(String::as_str)
    }

    pub fn contains(&self, tensor: &str) -> bool {
        self.weight_map.contains_key(tensor)
    }

    pub fn shard_for(&self, tensor: &str) -> Option<&str> {
        self.weight_map.get(tensor).map(String::as_str)
    }

    pub(crate) fn require(&self, tensor: impl AsRef<str>) -> Result<(), ModelSpecError> {
        let tensor = tensor.as_ref();
        if self.contains(tensor) {
            Ok(())
        } else {
            Err(ModelSpecError::invalid_request(format!(
                "safetensors index missing required tensor `{tensor}`"
            )))
        }
    }
}

fn validate_safetensors_shard_path(path: &str) -> Result<(), ModelSpecError> {
    if path.is_empty()
        || path.starts_with('/')
        || path.contains('\\')
        || path.bytes().any(|byte| byte == 0)
        || path
            .split('/')
            .any(|component| component.is_empty() || component == "." || component == "..")
    {
        return Err(ModelSpecError::invalid_request(format!(
            "unsafe safetensors shard path `{path}`"
        )));
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
struct RawSafetensorsIndex {
    metadata: RawSafetensorsMetadata,
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct RawSafetensorsMetadata {
    total_size: f64,
}
