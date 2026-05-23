#[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
use crate::native_text::infer_native_text_family;
use llm_backend_contracts::BackendModelMetadata;
use llm_hub::{ModelStore, PromotedSnapshot, SNAPSHOT_MANIFEST_FILE, SnapshotManifest};
use llm_models::{BackendKind, ModelFamily};
use std::path::Path;

pub type SnapshotBackendLoader = BackendKind;

pub fn parse_snapshot_model_family(value: &str) -> anyhow::Result<ModelFamily> {
    ModelFamily::parse_slug(value).map_err(|_| {
        anyhow::anyhow!(
            "unsupported snapshot family `{value}`; expected `qwen`, `deep_seek`, `gemma`, or `llama`"
        )
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedSnapshotBackend {
    loader: SnapshotBackendLoader,
    family: Option<ModelFamily>,
    manifest_digest: Option<String>,
    quantization: Option<String>,
    repo_id: Option<String>,
    resolved_commit: Option<String>,
    profile: Option<String>,
}

impl ResolvedSnapshotBackend {
    pub async fn resolve(
        snapshot_path: &Path,
        requested_loader: Option<SnapshotBackendLoader>,
        requested_family: Option<ModelFamily>,
        default_loader: SnapshotBackendLoader,
        detect_native_family: bool,
        require_mlx_family: bool,
    ) -> anyhow::Result<Self> {
        let promoted_snapshot = inspect_snapshot_manifest(snapshot_path).await?;
        let manifest = promoted_snapshot
            .as_ref()
            .map(|snapshot| &snapshot.manifest);
        let manifest_family = snapshot_manifest_family(manifest)?;
        let loader = select_snapshot_backend_loader(manifest, requested_loader, default_loader)?;
        let detected_family = detect_snapshot_family(
            snapshot_path,
            loader,
            manifest_family,
            requested_family,
            detect_native_family,
        )
        .await?;
        let family = requested_family.or(manifest_family).or(detected_family);
        validate_snapshot_family(manifest_family, requested_family)?;
        if require_mlx_family {
            validate_snapshot_loader_has_family(loader, family)?;
        }
        validate_snapshot_loader_family(loader, family)?;
        Ok(Self::from_parts(promoted_snapshot, loader, family))
    }

    fn from_parts(
        promoted_snapshot: Option<PromotedSnapshot>,
        loader: SnapshotBackendLoader,
        family: Option<ModelFamily>,
    ) -> Self {
        let manifest_digest = promoted_snapshot
            .as_ref()
            .map(|snapshot| snapshot.manifest_digest.clone());
        let manifest = promoted_snapshot
            .as_ref()
            .map(|snapshot| &snapshot.manifest);
        Self {
            loader,
            family,
            manifest_digest,
            quantization: manifest.map(|manifest| manifest.quantization.clone()),
            repo_id: manifest.map(|manifest| manifest.repo_id.clone()),
            resolved_commit: manifest.map(|manifest| manifest.resolved_commit.clone()),
            profile: manifest.map(|manifest| manifest.profile.clone()),
        }
    }

    pub fn loader(&self) -> SnapshotBackendLoader {
        self.loader
    }

    pub fn family(&self) -> Option<ModelFamily> {
        self.family
    }

    pub fn manifest_digest(&self) -> Option<&str> {
        self.manifest_digest.as_deref()
    }

    pub fn quantization(&self) -> Option<&str> {
        self.quantization.as_deref()
    }

    pub fn repo_id(&self) -> Option<&str> {
        self.repo_id.as_deref()
    }

    pub fn resolved_commit(&self) -> Option<&str> {
        self.resolved_commit.as_deref()
    }

    pub fn profile(&self) -> Option<&str> {
        self.profile.as_deref()
    }

    pub fn backend_metadata(
        &self,
        model_id: impl Into<String>,
        backend: impl Into<String>,
        fallback_family: Option<ModelFamily>,
    ) -> BackendModelMetadata {
        let mut metadata = BackendModelMetadata::new(model_id, backend);
        if let Some(family) = self.family.or(fallback_family) {
            metadata.family = Some(family.canonical_slug().to_owned());
        }
        metadata.quantization = self.quantization.clone();
        metadata.repo_id = self.repo_id.clone();
        metadata.resolved_commit = self.resolved_commit.clone();
        metadata.profile = self.profile.clone();
        metadata
    }
}

async fn detect_snapshot_family(
    snapshot_path: &Path,
    loader: SnapshotBackendLoader,
    manifest_family: Option<ModelFamily>,
    requested_family: Option<ModelFamily>,
    enabled: bool,
) -> anyhow::Result<Option<ModelFamily>> {
    if enabled
        && loader == SnapshotBackendLoader::NativeMetal
        && manifest_family.is_none()
        && requested_family.is_none()
    {
        #[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
        return Ok(Some(infer_native_text_family(snapshot_path).await?));
    }
    let _ = snapshot_path;
    Ok(None)
}

fn select_snapshot_backend_loader(
    manifest: Option<&SnapshotManifest>,
    requested: Option<SnapshotBackendLoader>,
    default_loader: SnapshotBackendLoader,
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
        None => Ok(default_loader),
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
    family: Option<ModelFamily>,
) -> anyhow::Result<()> {
    if loader == SnapshotBackendLoader::Mlx && family.is_none() {
        anyhow::bail!(
            "snapshot loader `mlx` requires model family metadata; add --family qwen, deep_seek, gemma, or llama for raw MLX snapshots or promote the snapshot with an llm-engine manifest"
        );
    }
    Ok(())
}

fn validate_snapshot_loader_family(
    loader: SnapshotBackendLoader,
    effective_family: Option<ModelFamily>,
) -> anyhow::Result<()> {
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

async fn inspect_snapshot_manifest(
    snapshot_path: &Path,
) -> anyhow::Result<Option<PromotedSnapshot>> {
    let manifest_path = snapshot_path.join(SNAPSHOT_MANIFEST_FILE);
    if !tokio::fs::try_exists(&manifest_path).await? {
        return Ok(None);
    }
    Ok(Some(ModelStore::inspect_snapshot(snapshot_path).await?))
}
