use llm_backend::BackendModelMetadata;
use llm_hub::SnapshotManifest;
use llm_models::{BackendKind, ModelFamily};
use std::path::{Path, PathBuf};

pub(super) async fn mlx_metadata(
    model_id: &str,
    snapshot_path: &Path,
    requested_family: Option<ModelFamily>,
) -> anyhow::Result<BackendModelMetadata> {
    let mut metadata = BackendModelMetadata::new(model_id.to_owned(), "mlx");
    metadata.snapshot_path = Some(PathBuf::from(snapshot_path));
    let manifest_path = snapshot_path.join("llm-engine-manifest.json");
    match crate::fs_util::read_optional_bytes(&manifest_path).await? {
        None => {
            let family = requested_family.ok_or_else(|| {
                anyhow::anyhow!(
                    "MLX backend requires model family metadata; add --family qwen, deep_seek, gemma, or llama for raw MLX snapshots or promote the snapshot with an llm-engine manifest"
                )
            })?;
            validate_mlx_serving_family(family)?;
            metadata.loader = Some("mlx".to_owned());
            metadata.family = Some(family.canonical_slug().to_owned());
            Ok(metadata)
        }
        Some(manifest_bytes) => {
            let manifest = serde_json::from_slice::<SnapshotManifest>(&manifest_bytes)?;
            let manifest_loader = BackendKind::parse_slug(&manifest.loader)?;
            if manifest_loader != BackendKind::Mlx {
                anyhow::bail!(
                    "MLX backend requires manifest loader `mlx`, not `{}`",
                    manifest_loader.canonical_slug()
                );
            }
            let manifest_family = ModelFamily::parse_slug(&manifest.family)?;
            if let Some(requested_family) = requested_family
                && manifest_family != requested_family
            {
                anyhow::bail!(
                    "requested snapshot family `{}` does not match manifest family `{}`",
                    requested_family.canonical_slug(),
                    manifest_family.canonical_slug()
                );
            }
            validate_mlx_serving_family(manifest_family)?;
            metadata.family = Some(manifest_family.canonical_slug().to_owned());
            metadata.loader = Some(manifest_loader.canonical_slug().to_owned());
            metadata.quantization = Some(manifest.quantization.clone());
            metadata.repo_id = Some(manifest.repo_id.clone());
            metadata.resolved_commit = Some(manifest.resolved_commit.clone());
            metadata.profile = Some(manifest.profile.clone());
            metadata.manifest_digest = Some(manifest.digest());
            Ok(metadata)
        }
    }
}

fn validate_mlx_serving_family(family: ModelFamily) -> anyhow::Result<()> {
    if !family
        .adapter()
        .production_backends()
        .contains(&BackendKind::Mlx)
    {
        let supported = family
            .adapter()
            .production_backends()
            .iter()
            .map(|backend| backend.canonical_slug())
            .collect::<Vec<_>>()
            .join(", ");
        anyhow::bail!(
            "MLX backend is not supported for family `{}`; supported loaders: {supported}",
            family.canonical_slug(),
        );
    }
    Ok(())
}
