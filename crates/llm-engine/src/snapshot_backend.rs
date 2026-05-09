use crate::{
    DEFAULT_NATIVE_QWEN_MAX_NEW_TOKENS, MlxBackend, MlxBackendOptions, NativeQwenBackend,
    NativeQwenLoadOptions,
};
use llm_backend::ModelBackend;
use llm_hub::SnapshotManifest;
use llm_models::{BackendKind, ModelFamily};
use std::path::Path;

pub type SnapshotBackendLoader = BackendKind;

pub fn parse_snapshot_model_family(value: &str) -> anyhow::Result<ModelFamily> {
    ModelFamily::parse_slug(value).map_err(|_| {
        anyhow::anyhow!(
            "unsupported snapshot family `{value}`; expected `qwen`, `deep_seek`, or `gemma`"
        )
    })
}

#[derive(Debug, Clone)]
pub struct SnapshotBackendOptions {
    pub loader: Option<SnapshotBackendLoader>,
    pub family: Option<ModelFamily>,
    pub native_qwen: NativeQwenLoadOptions,
    pub mlx: MlxBackendOptions,
    pub max_new_tokens: u32,
    pub max_prefill_tokens: usize,
}

impl Default for SnapshotBackendOptions {
    fn default() -> Self {
        Self {
            loader: None,
            family: None,
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
    let manifest = snapshot_manifest(snapshot_path)?;
    let requested_family = options.family.or(options.mlx.family);
    let manifest_family = snapshot_manifest_family(manifest.as_ref())?;
    let loader = select_snapshot_backend_loader(manifest.as_ref(), options.loader)?;
    validate_snapshot_family(manifest_family, requested_family)?;
    validate_snapshot_loader_has_family(loader, manifest_family, requested_family)?;
    validate_snapshot_serving_family(manifest_family.or(requested_family))?;
    validate_snapshot_loader_family(loader, manifest_family, requested_family)?;
    match loader {
        SnapshotBackendLoader::Mlx => {
            let mut mlx_options = options.mlx;
            mlx_options.family = requested_family.or(manifest_family);
            Ok(Box::new(MlxBackend::open_with_options(
                model_id,
                snapshot_path,
                mlx_options,
            )?))
        }
        SnapshotBackendLoader::NativeMetal => Ok(Box::new(
            NativeQwenBackend::open_with_options(model_id, snapshot_path, options.native_qwen)?
                .with_max_new_tokens(options.max_new_tokens)
                .with_max_prefill_tokens(options.max_prefill_tokens),
        )),
    }
}

fn select_snapshot_backend_loader(
    manifest: Option<&SnapshotManifest>,
    requested: Option<SnapshotBackendLoader>,
) -> anyhow::Result<SnapshotBackendLoader> {
    let manifest_loader = snapshot_manifest_loader(manifest)?;
    if let (Some(requested), Some(manifest_loader)) = (requested, manifest_loader)
        && manifest_loader != requested
    {
        anyhow::bail!(
            "requested snapshot loader `{}` does not match manifest loader `{manifest_loader}`",
            requested.canonical_slug()
        );
    }
    if let Some(requested) = requested {
        return Ok(requested);
    }
    match manifest_loader {
        Some(loader) => Ok(loader),
        None => Ok(SnapshotBackendLoader::NativeMetal),
    }
}

fn validate_snapshot_family(
    manifest_family: Option<ModelFamily>,
    requested_family: Option<ModelFamily>,
) -> anyhow::Result<()> {
    if let (Some(manifest_family), Some(requested_family)) = (manifest_family, requested_family)
        && manifest_family != requested_family
    {
        anyhow::bail!(
            "requested snapshot family `{requested_family}` does not match manifest family `{}`",
            manifest_family.canonical_slug()
        );
    }
    Ok(())
}

fn validate_snapshot_loader_has_family(
    loader: SnapshotBackendLoader,
    manifest_family: Option<ModelFamily>,
    requested_family: Option<ModelFamily>,
) -> anyhow::Result<()> {
    if loader == SnapshotBackendLoader::Mlx && manifest_family.or(requested_family).is_none() {
        anyhow::bail!(
            "snapshot loader `mlx` requires model family metadata; add --family qwen for raw MLX snapshots or promote the snapshot with an llm-engine manifest"
        );
    }
    Ok(())
}

fn validate_snapshot_loader_family(
    loader: SnapshotBackendLoader,
    manifest_family: Option<ModelFamily>,
    requested_family: Option<ModelFamily>,
) -> anyhow::Result<()> {
    let effective_family = requested_family.or(manifest_family);
    if let Some(family) = effective_family
        && !family.adapter().production_backends().contains(&loader)
    {
        let supported = family
            .adapter()
            .production_backends()
            .iter()
            .map(|backend| backend.canonical_slug())
            .collect::<Vec<_>>()
            .join(", ");
        anyhow::bail!(
            "snapshot loader `{}` is not supported for family `{}`; supported loaders: {supported}",
            loader.canonical_slug(),
            family.canonical_slug()
        );
    }
    Ok(())
}

fn validate_snapshot_serving_family(family: Option<ModelFamily>) -> anyhow::Result<()> {
    let Some(family) = family else {
        return Ok(());
    };
    if !family.adapter().capabilities().backend_execution {
        anyhow::bail!(
            "model family `{}` is recognized but not serveable yet; {} serving is deferred until Qwen production parity",
            family.canonical_slug(),
            family.display_name()
        );
    }
    Ok(())
}

fn snapshot_manifest_loader(
    manifest: Option<&SnapshotManifest>,
) -> anyhow::Result<Option<SnapshotBackendLoader>> {
    manifest
        .map(|manifest| {
            BackendKind::parse_slug(&manifest.loader).map_err(|_| {
                anyhow::anyhow!(
                    "unsupported snapshot loader `{}` in llm-engine manifest",
                    manifest.loader
                )
            })
        })
        .transpose()
}

fn snapshot_manifest_family(
    manifest: Option<&SnapshotManifest>,
) -> anyhow::Result<Option<ModelFamily>> {
    manifest
        .map(|manifest| ModelFamily::parse_slug(&manifest.family).map_err(anyhow::Error::new))
        .transpose()
}

fn snapshot_manifest(snapshot_path: &Path) -> anyhow::Result<Option<SnapshotManifest>> {
    let manifest_path = snapshot_path.join("llm-engine-manifest.json");
    let manifest_bytes = match std::fs::read(&manifest_path) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err.into()),
    };
    let manifest = serde_json::from_slice::<SnapshotManifest>(&manifest_bytes)?;
    Ok(Some(manifest))
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
                family: Some(ModelFamily::Qwen),
                mlx: crate::MlxBackendOptions {
                    endpoint: url::Url::parse("http://127.0.0.1:18080/v1").expect("url"),
                    ..crate::MlxBackendOptions::default()
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
    fn snapshot_backend_factory_allows_explicit_mlx_loader_without_manifest() {
        let snapshot = temp_snapshot_dir("mlx-loader-override");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");

        let backend = open_snapshot_backend(
            "local-mlx",
            &snapshot,
            SnapshotBackendOptions {
                loader: Some(SnapshotBackendLoader::Mlx),
                family: Some(ModelFamily::Qwen),
                mlx: crate::MlxBackendOptions {
                    endpoint: url::Url::parse("http://127.0.0.1:18080/v1").expect("url"),
                    ..crate::MlxBackendOptions::default()
                },
                ..SnapshotBackendOptions::default()
            },
        )
        .expect("mlx backend opens");
        let metadata = backend.model_metadata();

        assert_eq!(metadata.backend, "mlx");
        assert_eq!(metadata.loader.as_deref(), Some("mlx"));
        assert_eq!(metadata.family.as_deref(), Some("qwen"));
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn snapshot_backend_factory_rejects_manifestless_mlx_without_family() {
        let snapshot = temp_snapshot_dir("mlx-loader-missing-family");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");

        let err = match open_snapshot_backend(
            "local-mlx",
            &snapshot,
            SnapshotBackendOptions {
                loader: Some(SnapshotBackendLoader::Mlx),
                mlx: crate::MlxBackendOptions {
                    endpoint: url::Url::parse("http://127.0.0.1:18080/v1").expect("url"),
                    ..crate::MlxBackendOptions::default()
                },
                ..SnapshotBackendOptions::default()
            },
        ) {
            Ok(_) => panic!("manifestless MLX snapshot without family should fail closed"),
            Err(err) => err,
        };

        assert!(
            err.to_string()
                .contains("snapshot loader `mlx` requires model family metadata")
        );
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn snapshot_backend_factory_rejects_loader_override_manifest_mismatch() {
        let snapshot = temp_snapshot_dir("loader-selection-mismatch");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        write_manifest(&snapshot, "native-metal");

        let err = match open_snapshot_backend(
            "local-mlx",
            &snapshot,
            SnapshotBackendOptions {
                loader: Some(SnapshotBackendLoader::Mlx),
                mlx: crate::MlxBackendOptions {
                    endpoint: url::Url::parse("http://127.0.0.1:18080/v1").expect("url"),
                    ..crate::MlxBackendOptions::default()
                },
                ..SnapshotBackendOptions::default()
            },
        ) {
            Ok(_) => panic!("loader mismatch should fail closed"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("does not match manifest loader"));
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn snapshot_backend_factory_rejects_family_override_manifest_mismatch() {
        let snapshot = temp_snapshot_dir("family-selection-mismatch");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        write_manifest(&snapshot, "mlx");

        let err = match open_snapshot_backend(
            "local-mlx",
            &snapshot,
            SnapshotBackendOptions {
                loader: Some(SnapshotBackendLoader::Mlx),
                family: Some(ModelFamily::Gemma),
                mlx: crate::MlxBackendOptions {
                    endpoint: url::Url::parse("http://127.0.0.1:18080/v1").expect("url"),
                    ..crate::MlxBackendOptions::default()
                },
                ..SnapshotBackendOptions::default()
            },
        ) {
            Ok(_) => panic!("family mismatch should fail closed"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("does not match manifest family"));
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn snapshot_backend_factory_rejects_deferred_family_for_serving() {
        let snapshot = temp_snapshot_dir("mlx-deferred-family");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");

        let err = match open_snapshot_backend(
            "local-gemma",
            &snapshot,
            SnapshotBackendOptions {
                loader: Some(SnapshotBackendLoader::Mlx),
                family: Some(ModelFamily::Gemma),
                mlx: crate::MlxBackendOptions {
                    endpoint: url::Url::parse("http://127.0.0.1:18080/v1").expect("url"),
                    ..crate::MlxBackendOptions::default()
                },
                ..SnapshotBackendOptions::default()
            },
        ) {
            Ok(_) => panic!("Gemma serving should fail closed until adapter support lands"),
            Err(err) => err,
        };

        assert!(
            err.to_string()
                .contains("model family `gemma` is recognized but not serveable yet")
        );
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn snapshot_backend_factory_rejects_non_qwen_family_for_raw_native_metal_snapshot() {
        let snapshot = temp_snapshot_dir("native-family-override-mismatch");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");

        let err = match open_snapshot_backend(
            "local-native",
            &snapshot,
            SnapshotBackendOptions {
                loader: Some(SnapshotBackendLoader::NativeMetal),
                family: Some(ModelFamily::Gemma),
                ..SnapshotBackendOptions::default()
            },
        ) {
            Ok(_) => panic!("native-metal non-Qwen family should fail closed"),
            Err(err) => err,
        };

        assert!(
            err.to_string()
                .contains("model family `gemma` is recognized but not serveable yet")
        );
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn snapshot_backend_factory_rejects_non_qwen_family_for_native_metal_manifest() {
        let snapshot = temp_snapshot_dir("native-family-manifest-mismatch");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        write_manifest_with_family(&snapshot, "native-metal", "gemma");

        let err = match open_snapshot_backend(
            "local-native",
            &snapshot,
            SnapshotBackendOptions::default(),
        ) {
            Ok(_) => panic!("native-metal Gemma manifest should fail closed"),
            Err(err) => err,
        };

        assert!(
            err.to_string()
                .contains("model family `gemma` is recognized but not serveable yet")
        );
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn snapshot_backend_factory_rejects_unknown_manifest_family_before_opening_mlx() {
        let snapshot = temp_snapshot_dir("unknown-family-manifest");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        write_manifest_with_family(&snapshot, "mlx", "llama");

        let err = match open_snapshot_backend(
            "local-unknown-family",
            &snapshot,
            SnapshotBackendOptions {
                mlx: crate::MlxBackendOptions {
                    endpoint: url::Url::parse("http://127.0.0.1:18080/v1").expect("url"),
                    ..crate::MlxBackendOptions::default()
                },
                ..SnapshotBackendOptions::default()
            },
        ) {
            Ok(_) => panic!("unknown manifest family should fail closed"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("unsupported model family `llama`"));
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
        write_manifest_with_family(root, loader, "qwen");
    }

    fn write_manifest_with_family(root: &Path, loader: &str, family: &str) {
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
                "family": family,
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
