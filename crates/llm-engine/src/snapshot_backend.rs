use crate::{
    DEFAULT_NATIVE_QWEN_MAX_NEW_TOKENS, MlxBackend, MlxBackendOptions, NativeQwenBackend,
    NativeQwenLoadOptions,
};
use llm_backend::ModelBackend;
use llm_hub::SnapshotManifest;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct SnapshotBackendOptions {
    pub native_qwen: NativeQwenLoadOptions,
    pub mlx: MlxBackendOptions,
    pub max_new_tokens: u32,
    pub max_prefill_tokens: usize,
}

impl Default for SnapshotBackendOptions {
    fn default() -> Self {
        Self {
            native_qwen: NativeQwenLoadOptions::default(),
            mlx: MlxBackendOptions::default(),
            max_new_tokens: DEFAULT_NATIVE_QWEN_MAX_NEW_TOKENS,
            max_prefill_tokens: 32,
        }
    }
}

pub fn open_snapshot_backend(
    model_id: impl Into<String>,
    snapshot_path: impl AsRef<Path>,
    options: SnapshotBackendOptions,
) -> anyhow::Result<Box<dyn ModelBackend>> {
    let model_id = model_id.into();
    let snapshot_path = snapshot_path.as_ref();
    match snapshot_backend_loader(snapshot_path)?.as_deref() {
        Some("mlx") => Ok(Box::new(MlxBackend::open_with_options(
            model_id,
            snapshot_path,
            options.mlx,
        )?)),
        Some("native-metal") | None => Ok(Box::new(
            NativeQwenBackend::open_with_options(model_id, snapshot_path, options.native_qwen)?
                .with_max_new_tokens(options.max_new_tokens)
                .with_max_prefill_tokens(options.max_prefill_tokens),
        )),
        Some(other) => Err(anyhow::anyhow!(
            "unsupported snapshot loader `{other}` in llm-engine manifest"
        )),
    }
}

fn snapshot_backend_loader(snapshot_path: &Path) -> anyhow::Result<Option<String>> {
    let manifest_path = snapshot_path.join("llm-engine-manifest.json");
    let manifest_bytes = match std::fs::read(&manifest_path) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err.into()),
    };
    let manifest = serde_json::from_slice::<SnapshotManifest>(&manifest_bytes)?;
    Ok(Some(manifest.loader))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_backend_factory_selects_mlx_from_manifest_loader() {
        let snapshot = temp_snapshot_dir("mlx-loader-selection");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        write_manifest(&snapshot, "mlx");

        let backend = open_snapshot_backend(
            "local-mlx",
            &snapshot,
            SnapshotBackendOptions {
                mlx: crate::MlxBackendOptions {
                    endpoint: url::Url::parse("http://127.0.0.1:18080/v1").expect("url"),
                },
                ..SnapshotBackendOptions::default()
            },
        )
        .expect("mlx backend opens");
        let metadata = backend.model_metadata();

        assert_eq!(metadata.backend, "mlx");
        assert_eq!(metadata.loader.as_deref(), Some("mlx"));
        assert_eq!(metadata.profile.as_deref(), Some("qwen36-mlx-4bit"));
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn snapshot_backend_factory_rejects_unknown_manifest_loader() {
        let snapshot = temp_snapshot_dir("unknown-loader-selection");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        write_manifest(&snapshot, "llama-cpp");

        let err = match open_snapshot_backend(
            "local-unknown",
            &snapshot,
            SnapshotBackendOptions::default(),
        ) {
            Ok(_) => panic!("unknown loader should fail closed"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("unsupported snapshot loader"));
        std::fs::remove_dir_all(snapshot).ok();
    }

    fn write_manifest(root: &Path, loader: &str) {
        std::fs::write(
            root.join("llm-engine-manifest.json"),
            serde_json::json!({
                "schema_version": 1,
                "source": "huggingface",
                "repo_type": "model",
                "repo_id": "mlx-community/Qwen3.6-35B-A3B-4bit",
                "requested_revision": "main",
                "resolved_commit": "0123456789abcdef0123456789abcdef01234567",
                "profile": "qwen36-mlx-4bit",
                "family": "qwen",
                "loader": loader,
                "quantization": "4bit",
                "created_at": "2026-05-08T00:00:00Z",
                "snapshot_path": root.display().to_string(),
                "files": [],
                "allow_patterns": [],
                "ignore_patterns": []
            })
            .to_string(),
        )
        .expect("manifest");
    }

    fn temp_snapshot_dir(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("llm-engine-{label}-{}", std::process::id()))
    }
}
