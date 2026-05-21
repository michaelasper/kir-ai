use crate::snapshot_backend::ResolvedSnapshotBackend;
use llm_backend::BackendModelMetadata;
use llm_models::{BackendKind, ModelFamily};

pub(super) fn mlx_metadata(
    model_id: &str,
    identity: &ResolvedSnapshotBackend,
) -> anyhow::Result<BackendModelMetadata> {
    if identity.loader() != BackendKind::Mlx {
        anyhow::bail!(
            "MLX backend requires manifest loader `mlx`, not `{}`",
            identity.loader().canonical_slug()
        );
    }
    let family = identity.family().ok_or_else(|| {
        anyhow::anyhow!(
            "MLX backend requires model family metadata; add --family qwen, deep_seek, gemma, or llama for raw MLX snapshots or promote the snapshot with an llm-engine manifest"
        )
    })?;
    validate_mlx_serving_family(family)?;
    Ok(identity.backend_metadata(model_id.to_owned(), "mlx", None))
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
