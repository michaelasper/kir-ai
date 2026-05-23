#[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
use crate::{
    DEFAULT_NATIVE_TEXT_MAX_NEW_TOKENS, DEFAULT_NATIVE_TEXT_MAX_PREFILL_TOKENS, NativeTextBackend,
    NativeTextLoadOptions, native_text::infer_native_text_family,
};
#[cfg(feature = "mlx")]
use crate::{MlxBackend, MlxBackendOptions};
use llm_backend_contracts::{BackendModelMetadata, ModelBackend};
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
    pub(crate) async fn resolve(
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

    pub(crate) fn backend_metadata(
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

#[derive(Debug, Clone)]
pub struct SnapshotBackendOptions {
    pub loader: Option<SnapshotBackendLoader>,
    pub family: Option<ModelFamily>,
    #[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
    pub native_text: NativeTextLoadOptions,
    #[cfg(feature = "mlx")]
    pub mlx: MlxBackendOptions,
    #[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
    pub max_new_tokens: u32,
    #[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
    pub max_prefill_tokens: usize,
}

impl Default for SnapshotBackendOptions {
    fn default() -> Self {
        Self {
            loader: None,
            family: None,
            #[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
            native_text: NativeTextLoadOptions::default(),
            #[cfg(feature = "mlx")]
            mlx: MlxBackendOptions::default(),
            #[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
            max_new_tokens: DEFAULT_NATIVE_TEXT_MAX_NEW_TOKENS,
            #[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
            max_prefill_tokens: DEFAULT_NATIVE_TEXT_MAX_PREFILL_TOKENS,
        }
    }
}

pub async fn open_snapshot_backend(
    model_id: impl Into<String>,
    snapshot_path: impl AsRef<Path>,
    options: SnapshotBackendOptions,
) -> anyhow::Result<Box<dyn ModelBackend>> {
    let snapshot_path = snapshot_path.as_ref();
    let requested_family = options.family;
    #[cfg(feature = "mlx")]
    let requested_family = requested_family.or(options.mlx.family);
    #[cfg(not(any(feature = "mlx", feature = "native-qwen", feature = "native-gemma")))]
    let _ = &model_id;
    let identity = ResolvedSnapshotBackend::resolve(
        snapshot_path,
        options.loader,
        requested_family,
        SnapshotBackendLoader::NativeMetal,
        true,
        true,
    )
    .await?;
    match identity.loader() {
        SnapshotBackendLoader::Mlx => {
            #[cfg(feature = "mlx")]
            {
                let model_id = model_id.into();
                Ok(Box::new(
                    MlxBackend::open_with_snapshot_identity(
                        model_id,
                        snapshot_path,
                        options.mlx,
                        identity,
                    )
                    .await?,
                ))
            }
            #[cfg(not(feature = "mlx"))]
            {
                anyhow::bail!(
                    "snapshot loader `mlx` is disabled; rebuild llm-engine with --features mlx"
                );
            }
        }
        SnapshotBackendLoader::NativeMetal => {
            #[cfg(any(feature = "native-qwen", feature = "native-gemma"))]
            {
                let model_id = model_id.into();
                Ok(Box::new(
                    NativeTextBackend::open_with_snapshot_identity(
                        model_id,
                        snapshot_path,
                        options.native_text,
                        identity,
                    )
                    .await?
                    .with_max_new_tokens(options.max_new_tokens)
                    .with_max_prefill_tokens(options.max_prefill_tokens),
                ))
            }
            #[cfg(not(any(feature = "native-qwen", feature = "native-gemma")))]
            {
                anyhow::bail!(
                    "snapshot loader `native-metal` is disabled; rebuild llm-engine with --features native-qwen and/or native-gemma"
                );
            }
        }
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

#[cfg(all(
    test,
    feature = "mlx",
    feature = "native-qwen",
    feature = "native-gemma"
))]
mod tests {
    use super::*;

    type TinyBf16Tensor = (String, Vec<usize>, Vec<f32>);
    type TinyBf16ShardMap = std::collections::BTreeMap<String, Vec<TinyBf16Tensor>>;

    fn open_blocking(
        model_id: impl Into<String>,
        snapshot_path: impl AsRef<Path>,
        options: SnapshotBackendOptions,
    ) -> Result<Box<dyn ModelBackend>, anyhow::Error> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime");
        rt.block_on(open_snapshot_backend(model_id, snapshot_path, options))
    }

    fn resolve_identity_blocking(
        snapshot_path: &Path,
        requested_loader: Option<SnapshotBackendLoader>,
        requested_family: Option<ModelFamily>,
        default_loader: SnapshotBackendLoader,
        detect_native_family: bool,
    ) -> anyhow::Result<ResolvedSnapshotBackend> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime");
        rt.block_on(ResolvedSnapshotBackend::resolve(
            snapshot_path,
            requested_loader,
            requested_family,
            default_loader,
            detect_native_family,
            true,
        ))
    }

    #[test]
    fn snapshot_backend_default_prefill_tokens_match_long_context_default() {
        assert_eq!(
            SnapshotBackendOptions::default().max_prefill_tokens,
            crate::DEFAULT_NATIVE_TEXT_MAX_PREFILL_TOKENS
        );
        assert_eq!(crate::DEFAULT_NATIVE_TEXT_MAX_PREFILL_TOKENS, 2048);
    }

    #[test]
    fn resolved_snapshot_backend_includes_manifest_identity_fields() {
        let snapshot = temp_snapshot_dir("resolved-snapshot-identity");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        write_manifest(&snapshot, "mlx");

        let identity = resolve_identity_blocking(
            &snapshot,
            None,
            None,
            SnapshotBackendLoader::NativeMetal,
            false,
        )
        .expect("identity resolves");

        assert_eq!(identity.loader(), SnapshotBackendLoader::Mlx);
        assert_eq!(identity.family(), Some(ModelFamily::Qwen));
        assert_eq!(identity.manifest_digest().map(str::len), Some(64));
        assert_eq!(
            identity.repo_id(),
            Some("mlx-community/Qwen3.6-35B-A3B-4bit")
        );
        assert_eq!(identity.profile(), Some("qwen36-mlx-4bit"));
        assert_eq!(identity.quantization(), Some("4bit"));
        assert_eq!(
            identity.resolved_commit(),
            Some("0123456789abcdef0123456789abcdef01234567")
        );
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn snapshot_backend_factory_selects_mlx_from_manifest_loader() {
        let snapshot = temp_snapshot_dir("mlx-loader-selection");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        write_manifest(&snapshot, "mlx");

        let backend = open_blocking(
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
        assert_eq!(metadata.profile.as_deref(), Some("qwen36-mlx-4bit"));
        assert_manifest_metadata(&metadata);
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn snapshot_backend_factory_allows_explicit_mlx_loader_without_manifest() {
        let snapshot = temp_snapshot_dir("mlx-loader-override");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");

        let backend = open_blocking(
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
        assert_eq!(metadata.family.as_deref(), Some("qwen"));
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn snapshot_backend_factory_opens_native_text_backend_from_raw_qwen_snapshot() {
        let snapshot = temp_snapshot_dir("native-text-raw-qwen");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        copy_qwen36_fixture("config.json", snapshot.join("config.json"));
        copy_qwen36_fixture("tokenizer.json", snapshot.join("tokenizer.json"));
        copy_qwen36_fixture(
            "model.safetensors.index.json",
            snapshot.join("model.safetensors.index.json"),
        );

        let backend = open_blocking(
            crate::DEFAULT_MODEL_ID,
            &snapshot,
            SnapshotBackendOptions::default(),
        )
        .expect("native text backend opens raw Qwen snapshot");
        let metadata = backend.model_metadata();

        assert_eq!(metadata.backend, "native-qwen");
        assert_eq!(metadata.family.as_deref(), Some("qwen"));
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn snapshot_backend_factory_passes_manifest_identity_to_native_qwen() {
        let snapshot = temp_snapshot_dir("native-qwen-manifest-identity");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        write_manifest_with_family(&snapshot, "native-metal", "qwen");
        copy_qwen36_fixture("config.json", snapshot.join("config.json"));
        copy_qwen36_fixture("tokenizer.json", snapshot.join("tokenizer.json"));
        copy_qwen36_fixture(
            "model.safetensors.index.json",
            snapshot.join("model.safetensors.index.json"),
        );

        let backend = open_blocking(
            crate::DEFAULT_MODEL_ID,
            &snapshot,
            SnapshotBackendOptions::default(),
        )
        .expect("native Qwen backend opens promoted snapshot");
        let metadata = backend.model_metadata();

        assert_eq!(metadata.backend, "native-qwen");
        assert_eq!(metadata.family.as_deref(), Some("qwen"));
        assert_manifest_metadata(&metadata);
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn snapshot_backend_factory_rejects_manifestless_mlx_without_family() {
        let snapshot = temp_snapshot_dir("mlx-loader-missing-family");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");

        let err = match open_blocking(
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

        let err = match open_blocking(
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

        let err = match open_blocking(
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
    fn snapshot_backend_factory_opens_gemma_mlx_snapshot() {
        let snapshot = temp_snapshot_dir("mlx-gemma-family");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");

        let backend = open_blocking(
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
        )
        .expect("Gemma MLX snapshot opens");

        assert_eq!(backend.model_metadata().family.as_deref(), Some("gemma"));
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn snapshot_backend_factory_opens_deepseek_mlx_snapshot() {
        let snapshot = temp_snapshot_dir("mlx-deepseek-family");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");

        let backend = open_blocking(
            "local-deepseek",
            &snapshot,
            SnapshotBackendOptions {
                loader: Some(SnapshotBackendLoader::Mlx),
                family: Some(ModelFamily::DeepSeek),
                mlx: crate::MlxBackendOptions {
                    endpoint: url::Url::parse("http://127.0.0.1:18080/v1").expect("url"),
                    ..crate::MlxBackendOptions::default()
                },
                ..SnapshotBackendOptions::default()
            },
        )
        .expect("DeepSeek MLX snapshot opens");

        assert_eq!(
            backend.model_metadata().family.as_deref(),
            Some("deep_seek")
        );
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn snapshot_backend_factory_opens_llama_mlx_snapshot() {
        let snapshot = temp_snapshot_dir("mlx-llama-family");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");

        let backend = open_blocking(
            "local-llama",
            &snapshot,
            SnapshotBackendOptions {
                loader: Some(SnapshotBackendLoader::Mlx),
                family: Some(ModelFamily::Llama),
                mlx: crate::MlxBackendOptions {
                    endpoint: url::Url::parse("http://127.0.0.1:18080/v1").expect("url"),
                    ..crate::MlxBackendOptions::default()
                },
                ..SnapshotBackendOptions::default()
            },
        )
        .expect("Llama MLX snapshot opens");

        assert_eq!(backend.model_metadata().family.as_deref(), Some("llama"));
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn snapshot_backend_factory_rejects_deepseek_for_raw_native_metal_snapshot() {
        let snapshot = temp_snapshot_dir("native-family-override-mismatch");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");

        let err = match open_blocking(
            "local-native",
            &snapshot,
            SnapshotBackendOptions {
                loader: Some(SnapshotBackendLoader::NativeMetal),
                family: Some(ModelFamily::DeepSeek),
                ..SnapshotBackendOptions::default()
            },
        ) {
            Ok(_) => panic!("native-metal DeepSeek family should fail closed"),
            Err(err) => err,
        };

        assert!(
            err.to_string()
                .contains("snapshot loader `native-metal` is not supported for family `deep_seek`")
        );
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn snapshot_backend_factory_rejects_llama_for_raw_native_metal_snapshot() {
        let snapshot = temp_snapshot_dir("native-llama-override-mismatch");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");

        let err = match open_blocking(
            "local-native",
            &snapshot,
            SnapshotBackendOptions {
                loader: Some(SnapshotBackendLoader::NativeMetal),
                family: Some(ModelFamily::Llama),
                ..SnapshotBackendOptions::default()
            },
        ) {
            Ok(_) => panic!("native-metal Llama family should fail closed"),
            Err(err) => err,
        };

        assert!(
            err.to_string()
                .contains("snapshot loader `native-metal` is not supported for family `llama`")
        );
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn snapshot_backend_factory_rejects_deepseek_for_native_metal_manifest() {
        let snapshot = temp_snapshot_dir("native-family-manifest-mismatch");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        write_manifest_with_family(&snapshot, "native-metal", "deep_seek");

        let err = match open_blocking("local-native", &snapshot, SnapshotBackendOptions::default())
        {
            Ok(_) => panic!("native-metal DeepSeek manifest should fail closed"),
            Err(err) => err,
        };

        assert!(
            err.to_string()
                .contains("snapshot loader `native-metal` is not supported for family `deep_seek`")
        );
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn snapshot_backend_factory_reports_gemma_native_layout_errors_before_execution_gate() {
        let snapshot = temp_snapshot_dir("native-gemma-invalid-layout");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        write_manifest_with_family(&snapshot, "native-metal", "gemma");
        write_gemma4_native_config(&snapshot);
        write_gemma4_native_index(&snapshot, false);

        let err = match open_blocking("local-gemma", &snapshot, SnapshotBackendOptions::default()) {
            Ok(_) => panic!("native Gemma snapshot missing text tensors should fail closed"),
            Err(err) => err,
        };

        assert!(
            err.to_string()
                .contains("model.language_model.layers.0.self_attn.k_proj.weight"),
            "expected missing Gemma text tensor error, got {err}"
        );
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn snapshot_backend_factory_opens_gemma_native_snapshot_after_layout_validation() {
        let snapshot = temp_snapshot_dir("native-gemma-open");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        write_manifest_with_family(&snapshot, "native-metal", "gemma");
        write_gemma4_native_config(&snapshot);
        write_gemma4_native_index(&snapshot, true);
        copy_qwen36_fixture("tokenizer.json", snapshot.join("tokenizer.json"));

        let backend = open_blocking("local-gemma", &snapshot, SnapshotBackendOptions::default())
            .expect("native Gemma backend opens");

        assert_eq!(backend.model_metadata().backend, "native-gemma");
        assert_eq!(backend.model_metadata().family.as_deref(), Some("gemma"));
        assert_manifest_metadata(&backend.model_metadata());
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn snapshot_backend_factory_infers_gemma_for_raw_native_text_snapshot() {
        let snapshot = temp_snapshot_dir("native-gemma-raw-open");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        write_gemma4_native_config(&snapshot);
        write_gemma4_native_index(&snapshot, true);
        copy_qwen36_fixture("tokenizer.json", snapshot.join("tokenizer.json"));

        let backend = open_blocking("local-gemma", &snapshot, SnapshotBackendOptions::default())
            .expect("raw native Gemma backend opens by config detection");

        assert_eq!(backend.model_metadata().backend, "native-gemma");
        assert_eq!(backend.model_metadata().family.as_deref(), Some("gemma"));
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn snapshot_backend_factory_rejects_unknown_manifest_family_before_opening_mlx() {
        let snapshot = temp_snapshot_dir("unknown-family-manifest");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        write_manifest_with_family(&snapshot, "mlx", "glm");

        let err = match open_blocking(
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

        assert!(err.to_string().contains("unsupported model family `glm`"));
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn snapshot_backend_factory_rejects_unknown_manifest_loader() {
        let snapshot = temp_snapshot_dir("unknown-loader-selection");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        write_manifest(&snapshot, "llama-cpp");

        let err = match open_blocking(
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

    fn assert_manifest_metadata(metadata: &BackendModelMetadata) {
        assert_eq!(
            metadata.repo_id.as_deref(),
            Some("mlx-community/Qwen3.6-35B-A3B-4bit")
        );
        assert_eq!(metadata.profile.as_deref(), Some("qwen36-mlx-4bit"));
        assert_eq!(metadata.quantization.as_deref(), Some("4bit"));
        assert_eq!(
            metadata.resolved_commit.as_deref(),
            Some("0123456789abcdef0123456789abcdef01234567")
        );
    }

    fn copy_qwen36_fixture(name: &str, destination: impl AsRef<Path>) {
        let source = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/qwen36")
            .join(name);
        let destination = destination.as_ref();
        std::fs::copy(&source, destination).expect("copy Qwen fixture");
        if name == "model.safetensors.index.json"
            && let Some(root) = destination.parent()
        {
            write_qwen36_static_f32_fixture_shards(root);
        }
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

    fn write_gemma4_native_config(root: &Path) {
        std::fs::write(
            root.join("config.json"),
            serde_json::json!({
                "architectures": ["Gemma4ForConditionalGeneration"],
                "model_type": "gemma4",
                "text_config": {
                    "attention_bias": false,
                    "attention_dropout": 0.0,
                    "attention_k_eq_v": true,
                    "dtype": "bfloat16",
                    "enable_moe_block": false,
                    "final_logit_softcapping": 30.0,
                    "global_head_dim": 512,
                    "head_dim": 256,
                    "hidden_activation": "gelu_pytorch_tanh",
                    "hidden_size": 5376,
                    "hidden_size_per_layer_input": 0,
                    "intermediate_size": 21504,
                    "layer_types": ["sliding_attention"],
                    "max_position_embeddings": 262144,
                    "model_type": "gemma4_text",
                    "num_attention_heads": 32,
                    "num_global_key_value_heads": 4,
                    "num_hidden_layers": 1,
                    "num_key_value_heads": 16,
                    "num_kv_shared_layers": 0,
                    "rms_norm_eps": 1e-6,
                    "rope_parameters": {
                        "full_attention": {
                            "partial_rotary_factor": 0.25,
                            "rope_theta": 1000000.0,
                            "rope_type": "proportional"
                        },
                        "sliding_attention": {
                            "rope_theta": 10000.0,
                            "rope_type": "default"
                        }
                    },
                    "sliding_window": 1024,
                    "tie_word_embeddings": true,
                    "use_cache": true,
                    "use_double_wide_mlp": false,
                    "vocab_size": 262144,
                    "vocab_size_per_layer_input": 262144
                },
                "tie_word_embeddings": true,
                "vision_config": {"model_type": "gemma4_vision"}
            })
            .to_string(),
        )
        .expect("Gemma config");
    }

    fn write_gemma4_native_index(root: &Path, include_key_projection: bool) {
        let mut weight_map = serde_json::Map::from_iter([
            (
                "model.embed_vision.embedding_projection.weight".to_owned(),
                serde_json::json!("model.safetensors"),
            ),
            (
                "model.language_model.embed_tokens.weight".to_owned(),
                serde_json::json!("model.safetensors"),
            ),
            (
                "model.language_model.norm.weight".to_owned(),
                serde_json::json!("model.safetensors"),
            ),
            (
                "model.language_model.layers.0.input_layernorm.weight".to_owned(),
                serde_json::json!("model.safetensors"),
            ),
            (
                "model.language_model.layers.0.layer_scalar".to_owned(),
                serde_json::json!("model.safetensors"),
            ),
            (
                "model.language_model.layers.0.mlp.down_proj.weight".to_owned(),
                serde_json::json!("model.safetensors"),
            ),
            (
                "model.language_model.layers.0.mlp.gate_proj.weight".to_owned(),
                serde_json::json!("model.safetensors"),
            ),
            (
                "model.language_model.layers.0.mlp.up_proj.weight".to_owned(),
                serde_json::json!("model.safetensors"),
            ),
            (
                "model.language_model.layers.0.post_attention_layernorm.weight".to_owned(),
                serde_json::json!("model.safetensors"),
            ),
            (
                "model.language_model.layers.0.post_feedforward_layernorm.weight".to_owned(),
                serde_json::json!("model.safetensors"),
            ),
            (
                "model.language_model.layers.0.pre_feedforward_layernorm.weight".to_owned(),
                serde_json::json!("model.safetensors"),
            ),
            (
                "model.language_model.layers.0.self_attn.k_norm.weight".to_owned(),
                serde_json::json!("model.safetensors"),
            ),
            (
                "model.language_model.layers.0.self_attn.o_proj.weight".to_owned(),
                serde_json::json!("model.safetensors"),
            ),
            (
                "model.language_model.layers.0.self_attn.q_norm.weight".to_owned(),
                serde_json::json!("model.safetensors"),
            ),
            (
                "model.language_model.layers.0.self_attn.q_proj.weight".to_owned(),
                serde_json::json!("model.safetensors"),
            ),
            (
                "model.language_model.layers.0.self_attn.v_proj.weight".to_owned(),
                serde_json::json!("model.safetensors"),
            ),
        ]);
        if include_key_projection {
            weight_map.insert(
                "model.language_model.layers.0.self_attn.k_proj.weight".to_owned(),
                serde_json::json!("model.safetensors"),
            );
        }
        std::fs::write(
            root.join("model.safetensors.index.json"),
            serde_json::json!({
                "metadata": {"total_size": 1},
                "weight_map": weight_map
            })
            .to_string(),
        )
        .expect("Gemma safetensors index");
        if include_key_projection {
            write_gemma4_static_f32_fixture_shard(root);
        }
    }

    fn write_qwen36_static_f32_fixture_shards(root: &Path) {
        let config_json = std::fs::read_to_string(root.join("config.json")).expect("Qwen config");
        let spec = llm_models::QwenModelSpec::from_config_json(&config_json).expect("Qwen spec");
        let index_json =
            std::fs::read_to_string(root.join("model.safetensors.index.json")).expect("Qwen index");
        let index =
            llm_models::SafetensorsIndex::from_json(&index_json).expect("Qwen index parses");
        let mut shards: TinyBf16ShardMap = std::collections::BTreeMap::new();
        for tensor in llm_backend::native::qwen_static_f32_tensors_for_spec(&spec) {
            let Some(shard) = index.shard_for(&tensor) else {
                continue;
            };
            let shape = qwen_static_f32_tensor_shape(&spec, &tensor);
            let element_count = shape.iter().product();
            shards.entry(shard.to_owned()).or_default().push((
                tensor,
                shape,
                vec![0.0; element_count],
            ));
        }
        for (shard, tensors) in shards {
            let path = root.join(shard);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).expect("Qwen shard parent");
            }
            std::fs::write(path, tiny_named_safetensors_bf16(&tensors))
                .expect("Qwen static f32 fixture shard");
        }
    }

    fn qwen_static_f32_tensor_shape(spec: &llm_models::QwenModelSpec, tensor: &str) -> Vec<usize> {
        if tensor == spec.final_norm_weight()
            || tensor.ends_with("input_layernorm.weight")
            || tensor.ends_with("post_attention_layernorm.weight")
        {
            return vec![spec.hidden_size as usize];
        }
        if tensor.ends_with("self_attn.q_norm.weight")
            || tensor.ends_with("self_attn.k_norm.weight")
        {
            return vec![spec.head_dim as usize];
        }
        if tensor.ends_with("linear_attn.dt_bias") || tensor.ends_with("linear_attn.A_log") {
            return vec![spec.linear_num_value_heads as usize];
        }
        if tensor.ends_with("linear_attn.norm.weight") {
            return vec![spec.linear_value_head_dim as usize];
        }
        if tensor.ends_with("linear_attn.conv1d.weight") {
            let key_dim =
                (spec.linear_num_key_heads as usize) * (spec.linear_key_head_dim as usize);
            let value_dim =
                (spec.linear_num_value_heads as usize) * (spec.linear_value_head_dim as usize);
            return vec![
                key_dim * 2 + value_dim,
                spec.linear_conv_kernel_dim as usize,
            ];
        }
        panic!("unknown Qwen static f32 tensor `{tensor}`");
    }

    fn write_gemma4_static_f32_fixture_shard(root: &Path) {
        let config_json = std::fs::read_to_string(root.join("config.json")).expect("Gemma config");
        let spec = llm_models::GemmaModelSpec::from_config_json(&config_json).expect("Gemma spec");
        let tensors = llm_backend::native::gemma_static_f32_tensors_for_spec(&spec)
            .into_iter()
            .map(|tensor| {
                let shape = gemma_static_f32_tensor_shape(&spec, &tensor);
                let element_count = shape.iter().product();
                (tensor, shape, vec![0.0; element_count])
            })
            .collect::<Vec<_>>();
        std::fs::write(
            root.join("model.safetensors"),
            tiny_named_safetensors_bf16(&tensors),
        )
        .expect("Gemma static f32 fixture shard");
    }

    fn gemma_static_f32_tensor_shape(
        spec: &llm_models::GemmaModelSpec,
        tensor: &str,
    ) -> Vec<usize> {
        if tensor == spec.final_norm_weight()
            || tensor.ends_with("input_layernorm.weight")
            || tensor.ends_with("post_attention_layernorm.weight")
            || tensor.ends_with("pre_feedforward_layernorm.weight")
            || tensor.ends_with("post_feedforward_layernorm.weight")
            || tensor.ends_with("post_per_layer_input_norm.weight")
            || tensor.ends_with("pre_feedforward_layernorm_2.weight")
            || tensor.ends_with("post_feedforward_layernorm_1.weight")
            || tensor.ends_with("post_feedforward_layernorm_2.weight")
        {
            return vec![spec.hidden_size as usize];
        }
        if tensor == spec.per_layer_projection_norm_weight() {
            return vec![spec.hidden_size_per_layer_input as usize];
        }
        if tensor.ends_with("self_attn.q_norm.weight")
            || tensor.ends_with("self_attn.k_norm.weight")
        {
            return vec![spec.head_dim as usize];
        }
        if tensor.ends_with("layer_scalar") || tensor.ends_with("router.scale") {
            return vec![1];
        }
        if tensor.ends_with("router.per_expert_scale") {
            return vec![spec.num_experts.unwrap_or(1) as usize];
        }
        panic!("unknown Gemma static f32 tensor `{tensor}`");
    }

    fn tiny_named_safetensors_bf16(tensors: &[TinyBf16Tensor]) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape, values) in tensors {
            let start = data.len();
            for value in values {
                data.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
            }
            let end = data.len();
            header.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": "BF16",
                    "shape": shape,
                    "data_offsets": [start, end]
                }),
            );
        }
        let header = serde_json::Value::Object(header).to_string();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&data);
        bytes
    }

    fn temp_snapshot_dir(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("llm-engine-{label}-{}", std::process::id()))
    }
}
