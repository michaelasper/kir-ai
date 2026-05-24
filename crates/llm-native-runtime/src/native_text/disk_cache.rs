use super::{
    NativeTextPrefixCache, NativeTextPrefixCacheHit, NativeTextPrefixCacheMetrics,
    NativeTextPrefixCacheNamespace, NativeTextPrefixCacheValue,
};
use crate::sync_ext::FailPoisonedMutex;
use llm_backend::native::{
    GemmaLayerCache, GemmaLayerCachePrefixState, KvCacheError, LayerKvCache,
    LayerKvCachePrefixState, LayerKvCacheSnapshot, LinearAttentionCacheSnapshot, QwenLayerCache,
    QwenLayerCachePrefixState, TensorLoadError,
};
use llm_backend_contracts::BackendModelMetadata;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fmt,
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};
use tokio::sync::mpsc;
#[cfg(test)]
use tokio::sync::oneshot;

const NATIVE_TEXT_DISK_CACHE_CODEC: &str = "kir-ai-native-text-prefix-block";
const NATIVE_TEXT_DISK_CACHE_LAYOUT_VERSION: u32 = 1;
const DEFAULT_WRITER_QUEUE_DEPTH: usize = 8;
const DEFAULT_BLOCK_TOKEN_COUNT: usize = 256;
const MAX_SAFETENSORS_HEADER_LEN: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeTextDiskCacheConfig {
    pub root: PathBuf,
    pub writer_queue_depth: usize,
    pub block_token_count: usize,
}

impl NativeTextDiskCacheConfig {
    pub fn for_root(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
            writer_queue_depth: DEFAULT_WRITER_QUEUE_DEPTH,
            block_token_count: DEFAULT_BLOCK_TOKEN_COUNT,
        }
    }

    pub fn default_root() -> PathBuf {
        if let Some(cache_home) = std::env::var_os("XDG_CACHE_HOME") {
            return PathBuf::from(cache_home).join("kir-ai").join("kv-cache");
        }
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home)
                .join(".cache")
                .join("kir-ai")
                .join("kv-cache");
        }
        std::env::temp_dir().join("kir-ai").join("kv-cache")
    }

    pub fn with_writer_queue_depth(mut self, writer_queue_depth: usize) -> Self {
        self.writer_queue_depth = writer_queue_depth.max(1);
        self
    }

    pub fn with_block_token_count(mut self, block_token_count: usize) -> Self {
        self.block_token_count = block_token_count.max(1);
        self
    }
}

impl Default for NativeTextDiskCacheConfig {
    fn default() -> Self {
        Self::for_root(Self::default_root())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NativeTextDiskCacheIdentity {
    model_hash: String,
    snapshot_hash: String,
    model_family: String,
    allowed_namespace_hash: Option<String>,
}

impl NativeTextDiskCacheIdentity {
    #[cfg(test)]
    pub(crate) fn from_namespace(
        namespace: &NativeTextPrefixCacheNamespace,
        model_family: &str,
    ) -> Self {
        Self {
            model_hash: native_text_disk_model_hash_from_namespace(namespace),
            snapshot_hash: native_text_disk_snapshot_hash(Some("test-snapshot")),
            model_family: model_family.to_owned(),
            allowed_namespace_hash: Some(native_text_disk_namespace_hash(namespace)),
        }
    }

    pub(crate) fn from_model_metadata(
        metadata: &BackendModelMetadata,
        model_family: &str,
        snapshot_identity: Option<&str>,
    ) -> Self {
        let snapshot_hash = native_text_disk_snapshot_hash(snapshot_identity);
        Self {
            model_hash: native_text_disk_model_hash(NativeTextDiskModelHashParts {
                model_id: &metadata.id,
                backend: &metadata.backend,
                family: metadata.family.as_deref(),
                quantization: metadata.quantization.as_deref(),
                repo_id: metadata.repo_id.as_deref(),
                resolved_commit: metadata.resolved_commit.as_deref(),
                profile: metadata.profile.as_deref(),
                snapshot_hash: &snapshot_hash,
            }),
            snapshot_hash,
            model_family: model_family.to_owned(),
            allowed_namespace_hash: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn for_test(model_hash: &str, model_family: &str) -> Self {
        Self {
            model_hash: model_hash.to_owned(),
            snapshot_hash: native_text_disk_snapshot_hash(Some("test-snapshot")),
            model_family: model_family.to_owned(),
            allowed_namespace_hash: None,
        }
    }

    pub(crate) fn model_hash(&self) -> &str {
        &self.model_hash
    }

    #[cfg(test)]
    pub(crate) fn snapshot_hash(&self) -> &str {
        &self.snapshot_hash
    }
}

pub(crate) fn native_text_disk_cache_snapshot_identity(
    snapshot_path: &Path,
    manifest_digest: Option<&str>,
) -> String {
    if let Some(manifest_digest) = manifest_digest {
        return format!("manifest:{manifest_digest}");
    }
    let canonical =
        std::fs::canonicalize(snapshot_path).unwrap_or_else(|_| snapshot_path.to_path_buf());
    format!("raw-path:{}", canonical.display())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NativeTextDiskCacheBlockDescriptor {
    model_hash: String,
    snapshot_hash: String,
    model_family: String,
    namespace_hash: String,
    block_hash: String,
    block_start: usize,
    token_count: usize,
    cache_layout_version: u32,
}

impl NativeTextDiskCacheBlockDescriptor {
    pub(crate) fn new(
        identity: &NativeTextDiskCacheIdentity,
        namespace: &NativeTextPrefixCacheNamespace,
        block_start: usize,
        tokens: &[usize],
    ) -> Self {
        let namespace_hash = native_text_disk_namespace_hash(namespace);
        Self {
            model_hash: identity.model_hash.clone(),
            snapshot_hash: identity.snapshot_hash.clone(),
            model_family: identity.model_family.clone(),
            block_hash: native_text_disk_block_hash(
                &identity.model_hash,
                &namespace_hash,
                block_start,
                tokens,
            ),
            namespace_hash,
            block_start,
            token_count: tokens.len(),
            cache_layout_version: namespace.cache_layout_version,
        }
    }

    #[cfg(test)]
    pub(crate) fn namespace_hash(&self) -> &str {
        &self.namespace_hash
    }

    pub(crate) fn block_hash(&self) -> &str {
        &self.block_hash
    }

    fn validate_for_identity(
        &self,
        identity: &NativeTextDiskCacheIdentity,
    ) -> Result<(), NativeTextDiskCacheError> {
        if self.model_hash != identity.model_hash {
            return Err(NativeTextDiskCacheError::integrity("model hash mismatch"));
        }
        if self.snapshot_hash != identity.snapshot_hash {
            return Err(NativeTextDiskCacheError::integrity(
                "snapshot hash mismatch",
            ));
        }
        if self.model_family != identity.model_family {
            return Err(NativeTextDiskCacheError::integrity("model family mismatch"));
        }
        if let Some(namespace_hash) = identity.allowed_namespace_hash.as_deref()
            && self.namespace_hash != namespace_hash
        {
            return Err(NativeTextDiskCacheError::integrity(
                "namespace hash mismatch",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NativeTextDiskCacheStoreStatus {
    Queued,
    Dropped,
    Skipped,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NativeTextDiskCacheError {
    message: String,
}

impl NativeTextDiskCacheError {
    pub(crate) fn integrity(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    fn io(context: &str, err: impl fmt::Display) -> Self {
        Self::integrity(format!("{context}: {err}"))
    }
}

impl fmt::Display for NativeTextDiskCacheError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for NativeTextDiskCacheError {}

impl From<KvCacheError> for NativeTextDiskCacheError {
    fn from(err: KvCacheError) -> Self {
        Self::integrity(err.to_string())
    }
}

impl From<TensorLoadError> for NativeTextDiskCacheError {
    fn from(err: TensorLoadError) -> Self {
        Self::integrity(err.to_string())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct NativeTextDiskCacheStateBlock<S> {
    pub(crate) block_start: usize,
    pub(crate) token_count: usize,
    pub(crate) states: Vec<S>,
}

pub(crate) trait NativeTextDiskCacheValue: NativeTextPrefixCacheValue + Sized {
    fn encode_disk_block_states(
        states: &[Self::PrefixCacheState],
        block_start: usize,
        block_token_count: usize,
        sink: &mut NativeTextDiskCacheTensorSink,
    ) -> Result<Vec<NativeTextDiskCacheLayerLayout>, NativeTextDiskCacheError>;

    fn decode_disk_states(
        layouts: &[NativeTextDiskCacheLayerLayout],
        archive: &NativeTextDiskCacheTensorArchive<'_>,
    ) -> Result<Vec<Self::PrefixCacheState>, NativeTextDiskCacheError>;

    fn assemble_disk_block_states(
        blocks: &[NativeTextDiskCacheStateBlock<Self::PrefixCacheState>],
    ) -> Result<Vec<Self::PrefixCacheState>, NativeTextDiskCacheError>;
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum NativeTextDiskCacheLayerLayout {
    #[serde(rename = "qwen_full_attention")]
    QwenFull(NativeTextDiskFullAttentionLayout),
    #[serde(rename = "qwen_linear_attention")]
    QwenLinear(NativeTextDiskLinearAttentionLayout),
    #[serde(rename = "gemma_attention")]
    GemmaFull(NativeTextDiskFullAttentionLayout),
    #[cfg(test)]
    TestMarkerTensor { tensor: String },
}

impl NativeTextDiskCacheLayerLayout {
    #[cfg(test)]
    pub(crate) fn test_marker_tensor(tensor: &str) -> Self {
        Self::TestMarkerTensor {
            tensor: tensor.to_owned(),
        }
    }

    #[cfg(test)]
    pub(crate) fn test_marker_tensor_name(&self) -> Option<&str> {
        match self {
            Self::TestMarkerTensor { tensor } => Some(tensor),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct NativeTextDiskFullAttentionLayout {
    revision: u64,
    max_tokens: usize,
    key_value_heads: usize,
    head_dim: usize,
    token_count: usize,
    tokens_seen: usize,
    cache_format: String,
    key_tensor: String,
    value_tensor: String,
    shape: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct NativeTextDiskLinearAttentionLayout {
    revision: u64,
    conv_kernel_size: usize,
    conv_dim: usize,
    num_value_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    token_count: usize,
    conv_window_tensor: String,
    recurrent_state_tensor: String,
    conv_window_shape: Vec<usize>,
    recurrent_state_shape: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NativeTextDiskCacheBlockMetadata {
    codec: String,
    layout_version: u32,
    cache_layout_version: u32,
    model_family: String,
    model_hash: String,
    snapshot_hash: String,
    namespace_hash: String,
    block_hash: String,
    block_start: usize,
    token_count: usize,
    hidden_shape: Vec<usize>,
    layers: Vec<NativeTextDiskCacheLayerLayout>,
}

#[derive(Debug)]
pub(crate) struct NativeTextDiskCacheBlock<C: NativeTextDiskCacheValue> {
    pub(crate) block_start: usize,
    pub(crate) token_count: usize,
    pub(crate) hidden: Vec<f32>,
    pub(crate) states: Vec<C::PrefixCacheState>,
}

impl<C> NativeTextDiskCacheBlock<C>
where
    C: NativeTextDiskCacheValue,
{
    pub(crate) fn encode(
        descriptor: &NativeTextDiskCacheBlockDescriptor,
        hidden: &[f32],
        states: &[C::PrefixCacheState],
    ) -> Result<Vec<u8>, NativeTextDiskCacheError> {
        let mut sink = NativeTextDiskCacheTensorSink::default();
        sink.push_f32("hidden", vec![hidden.len()], hidden.to_vec())?;
        let layers = C::encode_disk_block_states(
            states,
            descriptor.block_start,
            descriptor.token_count,
            &mut sink,
        )?;
        let metadata = NativeTextDiskCacheBlockMetadata {
            codec: NATIVE_TEXT_DISK_CACHE_CODEC.to_owned(),
            layout_version: NATIVE_TEXT_DISK_CACHE_LAYOUT_VERSION,
            cache_layout_version: descriptor.cache_layout_version,
            model_family: descriptor.model_family.clone(),
            model_hash: descriptor.model_hash.clone(),
            snapshot_hash: descriptor.snapshot_hash.clone(),
            namespace_hash: descriptor.namespace_hash.clone(),
            block_hash: descriptor.block_hash.clone(),
            block_start: descriptor.block_start,
            token_count: descriptor.token_count,
            hidden_shape: vec![hidden.len()],
            layers,
        };
        encode_safetensors(metadata, sink.into_tensors())
    }

    pub(crate) fn decode(
        bytes: &[u8],
        identity: &NativeTextDiskCacheIdentity,
        descriptor: &NativeTextDiskCacheBlockDescriptor,
    ) -> Result<Self, NativeTextDiskCacheError> {
        let (metadata, archive) = NativeTextDiskCacheTensorArchive::from_bytes(bytes)?;
        if metadata.namespace_hash != descriptor.namespace_hash
            || metadata.block_hash != descriptor.block_hash
            || metadata.block_start != descriptor.block_start
            || metadata.token_count != descriptor.token_count
        {
            return Err(NativeTextDiskCacheError::integrity(
                "disk cache block descriptor mismatch",
            ));
        }
        descriptor.validate_for_identity(identity)?;
        validate_metadata_for_identity(&metadata, identity)?;
        if metadata.cache_layout_version != descriptor.cache_layout_version {
            return Err(NativeTextDiskCacheError::integrity(
                "cache layout version mismatch",
            ));
        }
        let hidden = archive.f32_tensor("hidden")?;
        if metadata.hidden_shape.as_slice() != [hidden.len()] {
            return Err(NativeTextDiskCacheError::integrity(
                "hidden tensor shape does not match metadata",
            ));
        }
        let states = C::decode_disk_states(&metadata.layers, &archive)?;
        Ok(Self {
            block_start: metadata.block_start,
            token_count: metadata.token_count,
            hidden,
            states,
        })
    }

    fn decode_descriptor(
        bytes: &[u8],
    ) -> Result<NativeTextDiskCacheBlockDescriptor, NativeTextDiskCacheError> {
        let metadata = decode_safetensors_metadata(bytes)?;
        validate_metadata_codec_and_layout(&metadata)?;
        Ok(NativeTextDiskCacheBlockDescriptor {
            model_hash: metadata.model_hash,
            snapshot_hash: metadata.snapshot_hash,
            model_family: metadata.model_family,
            namespace_hash: metadata.namespace_hash,
            block_hash: metadata.block_hash,
            block_start: metadata.block_start,
            token_count: metadata.token_count,
            cache_layout_version: metadata.cache_layout_version,
        })
    }

    #[cfg(test)]
    pub(crate) fn rewrite_layout_version_for_test(
        bytes: &mut Vec<u8>,
        layout_version: u32,
    ) -> Result<(), NativeTextDiskCacheError> {
        let (mut metadata, archive) = NativeTextDiskCacheTensorArchive::from_bytes(bytes)?;
        metadata.layout_version = layout_version;
        *bytes = encode_safetensors(metadata, archive.tensors.clone())?;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn rewrite_cache_layout_version_for_test(
        bytes: &mut Vec<u8>,
        cache_layout_version: u32,
    ) -> Result<(), NativeTextDiskCacheError> {
        let (mut metadata, archive) = NativeTextDiskCacheTensorArchive::from_bytes(bytes)?;
        metadata.cache_layout_version = cache_layout_version;
        *bytes = encode_safetensors(metadata, archive.tensors.clone())?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct NativeTextDiskCache<C: NativeTextDiskCacheValue> {
    config: NativeTextDiskCacheConfig,
    identity: NativeTextDiskCacheIdentity,
    index: NativeTextDiskCacheIndex,
    writer: NativeTextDiskCacheWriter,
    _cache: PhantomData<C>,
}

impl<C> NativeTextDiskCache<C>
where
    C: NativeTextDiskCacheValue + Send + Sync + 'static,
{
    pub(crate) async fn open(
        config: NativeTextDiskCacheConfig,
        identity: NativeTextDiskCacheIdentity,
    ) -> Result<Self, NativeTextDiskCacheError> {
        let writer_queue_depth = config.writer_queue_depth;
        let block_token_count = config.block_token_count;
        let config = config
            .with_writer_queue_depth(writer_queue_depth)
            .with_block_token_count(block_token_count);
        let index = NativeTextDiskCacheIndex::default();
        reindex_disk_cache_root(&config, &identity, &index).await?;
        let writer = NativeTextDiskCacheWriter::spawn(index.clone(), config.writer_queue_depth);
        Ok(Self {
            config,
            identity,
            index,
            writer,
            _cache: PhantomData,
        })
    }

    pub(crate) fn queue_store(
        &self,
        namespace: &NativeTextPrefixCacheNamespace,
        tokens: &[usize],
        hidden: &[f32],
        states: &[C::PrefixCacheState],
    ) -> NativeTextDiskCacheStoreStatus {
        if tokens.is_empty() || !tokens.len().is_multiple_of(self.config.block_token_count) {
            return NativeTextDiskCacheStoreStatus::Skipped;
        }
        let Some(block_start) = tokens.len().checked_sub(self.config.block_token_count) else {
            return NativeTextDiskCacheStoreStatus::Skipped;
        };
        let descriptor = NativeTextDiskCacheBlockDescriptor::new(
            &self.identity,
            namespace,
            block_start,
            &tokens[block_start..],
        );
        let Some(permit) = self.writer.try_reserve() else {
            return NativeTextDiskCacheStoreStatus::Dropped;
        };
        let bytes = match NativeTextDiskCacheBlock::<C>::encode(&descriptor, hidden, states) {
            Ok(bytes) => bytes,
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    model_hash = %descriptor.model_hash,
                    namespace_hash = %descriptor.namespace_hash,
                    block_hash = %descriptor.block_hash,
                    "failed to encode native text SSD prefix cache block"
                );
                return NativeTextDiskCacheStoreStatus::Dropped;
            }
        };
        let path = self.path_for_descriptor(&descriptor);
        let entry = NativeTextDiskCacheIndexedEntry {
            path: path.clone(),
            block_start: descriptor.block_start,
            token_count: descriptor.token_count,
        };
        permit.send(NativeTextDiskCacheWriterMessage::Write(
            NativeTextDiskCacheWriteJob {
                path,
                bytes,
                index: Some(self.index.clone()),
                index_key: Some(NativeTextDiskCacheIndexKey::from_descriptor(&descriptor)),
                index_entry: Some(entry),
            },
        ));
        NativeTextDiskCacheStoreStatus::Queued
    }

    pub(crate) async fn lookup(
        &self,
        namespace: &NativeTextPrefixCacheNamespace,
        tokens: &[usize],
        mut is_compatible: impl FnMut(&[C::PrefixCacheState]) -> bool,
    ) -> Result<Option<NativeTextPrefixCacheHit<C>>, NativeTextDiskCacheError> {
        if tokens.is_empty() {
            return Ok(None);
        }
        let max_prefix_len =
            (tokens.len() / self.config.block_token_count) * self.config.block_token_count;
        if max_prefix_len == 0 {
            return Ok(None);
        }
        'prefixes: for prefix_len in (self.config.block_token_count..=max_prefix_len)
            .rev()
            .step_by(self.config.block_token_count)
        {
            let mut blocks = Vec::new();
            for block_start in (0..prefix_len).step_by(self.config.block_token_count) {
                let block_end = block_start + self.config.block_token_count;
                let descriptor = NativeTextDiskCacheBlockDescriptor::new(
                    &self.identity,
                    namespace,
                    block_start,
                    &tokens[block_start..block_end],
                );
                let key = NativeTextDiskCacheIndexKey::from_descriptor(&descriptor);
                let Some(entry) = self.index.get(&key) else {
                    continue 'prefixes;
                };
                let bytes = match tokio::fs::read(&entry.path).await {
                    Ok(bytes) => bytes,
                    Err(err) => {
                        tracing::warn!(
                            error = %err,
                            path = %entry.path.display(),
                            "native text SSD prefix cache indexed file could not be read"
                        );
                        self.index.remove(&key);
                        continue 'prefixes;
                    }
                };
                let block = match NativeTextDiskCacheBlock::<C>::decode(
                    &bytes,
                    &self.identity,
                    &descriptor,
                ) {
                    Ok(block) => block,
                    Err(err) => {
                        tracing::warn!(
                            error = %err,
                            path = %entry.path.display(),
                            "native text SSD prefix cache block failed validation"
                        );
                        self.index.remove(&key);
                        continue 'prefixes;
                    }
                };
                if block.block_start != entry.block_start || block.token_count != entry.token_count
                {
                    continue 'prefixes;
                }
                blocks.push(block);
            }
            let state_blocks = blocks
                .iter()
                .map(|block| NativeTextDiskCacheStateBlock {
                    block_start: block.block_start,
                    token_count: block.token_count,
                    states: block.states.clone(),
                })
                .collect::<Vec<_>>();
            let states = C::assemble_disk_block_states(&state_blocks)?;
            if !is_compatible(&states) {
                continue;
            }
            let Some(caches) = C::prefix_cache_from_state(&states) else {
                continue;
            };
            let Some(last_block) = blocks.last() else {
                continue;
            };
            return Ok(Some(NativeTextPrefixCacheHit {
                token_count: prefix_len,
                hidden: last_block.hidden.clone(),
                caches,
            }));
        }
        Ok(None)
    }

    pub(crate) fn promote_hit(
        &self,
        memory: &NativeTextPrefixCache<C>,
        namespace: NativeTextPrefixCacheNamespace,
        tokens: &[usize],
        metrics: &NativeTextPrefixCacheMetrics,
        hit: &NativeTextPrefixCacheHit<C>,
    ) {
        memory.store(namespace, tokens, &hit.hidden, &hit.caches, metrics);
    }

    pub(crate) fn block_token_count(&self) -> usize {
        self.config.block_token_count
    }

    fn path_for_descriptor(&self, descriptor: &NativeTextDiskCacheBlockDescriptor) -> PathBuf {
        self.config
            .root
            .join(&descriptor.model_hash)
            .join(&descriptor.namespace_hash)
            .join(descriptor.block_hash.get(..2).unwrap_or("00"))
            .join(format!("{}.safetensors", descriptor.block_hash))
    }

    #[cfg(test)]
    pub(crate) async fn flush_for_test(&self) -> Result<(), NativeTextDiskCacheError> {
        self.writer.flush().await
    }

    #[cfg(test)]
    pub(crate) fn indexed_entry_count_for_test(&self) -> usize {
        self.index.len()
    }

    #[cfg(test)]
    pub(crate) fn root_for_test(&self) -> &Path {
        &self.config.root
    }

    #[cfg(test)]
    pub(crate) fn path_for_descriptor_for_test(
        &self,
        descriptor: &NativeTextDiskCacheBlockDescriptor,
    ) -> PathBuf {
        let path = self.path_for_descriptor(descriptor);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("test disk cache parent creates");
        }
        path
    }
}

#[derive(Debug, Clone, Default)]
struct NativeTextDiskCacheIndex {
    inner: Arc<Mutex<HashMap<NativeTextDiskCacheIndexKey, NativeTextDiskCacheIndexedEntry>>>,
}

impl NativeTextDiskCacheIndex {
    fn insert(&self, key: NativeTextDiskCacheIndexKey, entry: NativeTextDiskCacheIndexedEntry) {
        self.inner
            .lock_or_panic("native text disk cache index")
            .insert(key, entry);
    }

    fn get(&self, key: &NativeTextDiskCacheIndexKey) -> Option<NativeTextDiskCacheIndexedEntry> {
        self.inner
            .lock_or_panic("native text disk cache index")
            .get(key)
            .cloned()
    }

    fn remove(&self, key: &NativeTextDiskCacheIndexKey) {
        self.inner
            .lock_or_panic("native text disk cache index")
            .remove(key);
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.inner
            .lock_or_panic("native text disk cache index")
            .len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct NativeTextDiskCacheIndexKey {
    namespace_hash: String,
    block_start: usize,
    block_hash: String,
}

impl NativeTextDiskCacheIndexKey {
    fn from_descriptor(descriptor: &NativeTextDiskCacheBlockDescriptor) -> Self {
        Self {
            namespace_hash: descriptor.namespace_hash.clone(),
            block_start: descriptor.block_start,
            block_hash: descriptor.block_hash.clone(),
        }
    }
}

#[derive(Debug, Clone)]
struct NativeTextDiskCacheIndexedEntry {
    path: PathBuf,
    block_start: usize,
    token_count: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct NativeTextDiskCacheWriter {
    tx: mpsc::Sender<NativeTextDiskCacheWriterMessage>,
}

impl NativeTextDiskCacheWriter {
    fn spawn(index: NativeTextDiskCacheIndex, queue_depth: usize) -> Self {
        let (tx, rx) = mpsc::channel(queue_depth.max(1));
        tokio::spawn(native_text_disk_cache_writer_loop(rx, index));
        Self { tx }
    }

    #[cfg(test)]
    fn try_enqueue(&self, job: NativeTextDiskCacheWriteJob) -> NativeTextDiskCacheStoreStatus {
        match self
            .tx
            .try_send(NativeTextDiskCacheWriterMessage::Write(job))
        {
            Ok(()) => NativeTextDiskCacheStoreStatus::Queued,
            Err(mpsc::error::TrySendError::Full(_)) | Err(mpsc::error::TrySendError::Closed(_)) => {
                NativeTextDiskCacheStoreStatus::Dropped
            }
        }
    }

    fn try_reserve(&self) -> Option<mpsc::Permit<'_, NativeTextDiskCacheWriterMessage>> {
        self.tx.try_reserve().ok()
    }

    #[cfg(test)]
    fn detached_for_test(
        queue_depth: usize,
    ) -> (Self, mpsc::Receiver<NativeTextDiskCacheWriterMessage>) {
        let (tx, rx) = mpsc::channel(queue_depth.max(1));
        (Self { tx }, rx)
    }

    #[cfg(test)]
    async fn flush(&self) -> Result<(), NativeTextDiskCacheError> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(NativeTextDiskCacheWriterMessage::Flush(tx))
            .await
            .map_err(|err| NativeTextDiskCacheError::io("disk cache writer flush failed", err))?;
        rx.await
            .map_err(|err| NativeTextDiskCacheError::io("disk cache writer flush dropped", err))
    }
}

#[derive(Debug)]
enum NativeTextDiskCacheWriterMessage {
    Write(NativeTextDiskCacheWriteJob),
    #[cfg(test)]
    Flush(oneshot::Sender<()>),
}

#[derive(Debug, Clone)]
pub(crate) struct NativeTextDiskCacheWriteJob {
    path: PathBuf,
    bytes: Vec<u8>,
    index: Option<NativeTextDiskCacheIndex>,
    index_key: Option<NativeTextDiskCacheIndexKey>,
    index_entry: Option<NativeTextDiskCacheIndexedEntry>,
}

impl NativeTextDiskCacheWriteJob {
    #[cfg(test)]
    fn for_test(path: &str, bytes: Vec<u8>) -> Self {
        Self {
            path: PathBuf::from(path),
            bytes,
            index: None,
            index_key: None,
            index_entry: None,
        }
    }
}

async fn native_text_disk_cache_writer_loop(
    mut rx: mpsc::Receiver<NativeTextDiskCacheWriterMessage>,
    fallback_index: NativeTextDiskCacheIndex,
) {
    while let Some(message) = rx.recv().await {
        match message {
            NativeTextDiskCacheWriterMessage::Write(job) => {
                let index = job.index.clone().unwrap_or_else(|| fallback_index.clone());
                let index_key = job.index_key.clone();
                let index_entry = job.index_entry.clone();
                if let Err(err) = write_disk_cache_job(job).await {
                    tracing::warn!(error = %err, "native text SSD prefix cache write failed");
                    continue;
                }
                if let (Some(key), Some(entry)) = (index_key, index_entry) {
                    index.insert(key, entry);
                }
            }
            #[cfg(test)]
            NativeTextDiskCacheWriterMessage::Flush(done) => {
                let _ = done.send(());
            }
        }
    }
}

async fn write_disk_cache_job(
    job: NativeTextDiskCacheWriteJob,
) -> Result<(), NativeTextDiskCacheError> {
    if let Some(parent) = job.path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|err| NativeTextDiskCacheError::io("create disk cache parent", err))?;
    }
    static NEXT_TMP_ID: AtomicU64 = AtomicU64::new(1);
    let tmp_id = NEXT_TMP_ID.fetch_add(1, Ordering::Relaxed);
    let tmp_path = job
        .path
        .with_extension(format!("safetensors.tmp-{}-{tmp_id}", std::process::id()));
    tokio::fs::write(&tmp_path, &job.bytes)
        .await
        .map_err(|err| NativeTextDiskCacheError::io("write disk cache temp file", err))?;
    tokio::fs::rename(&tmp_path, &job.path)
        .await
        .map_err(|err| NativeTextDiskCacheError::io("promote disk cache temp file", err))?;
    Ok(())
}

#[derive(Debug, Default)]
pub(crate) struct NativeTextDiskCacheTensorSink {
    tensors: Vec<NativeTextDiskCacheTensor>,
}

impl NativeTextDiskCacheTensorSink {
    pub(crate) fn push_f32(
        &mut self,
        name: impl Into<String>,
        shape: Vec<usize>,
        values: Vec<f32>,
    ) -> Result<(), NativeTextDiskCacheError> {
        let expected = shape.iter().try_fold(1_usize, |acc, dim| {
            acc.checked_mul(*dim)
                .ok_or_else(|| NativeTextDiskCacheError::integrity("tensor shape overflow"))
        })?;
        if expected != values.len() {
            return Err(NativeTextDiskCacheError::integrity(
                "tensor shape does not match value count",
            ));
        }
        self.tensors.push(NativeTextDiskCacheTensor {
            name: name.into(),
            shape,
            values,
        });
        Ok(())
    }

    fn into_tensors(self) -> Vec<NativeTextDiskCacheTensor> {
        self.tensors
    }
}

#[derive(Debug, Clone)]
struct NativeTextDiskCacheTensor {
    name: String,
    shape: Vec<usize>,
    values: Vec<f32>,
}

#[derive(Debug)]
pub(crate) struct NativeTextDiskCacheTensorArchive<'a> {
    tensors: Vec<NativeTextDiskCacheTensor>,
    bytes: &'a [u8],
}

impl<'a> NativeTextDiskCacheTensorArchive<'a> {
    fn from_bytes(
        bytes: &'a [u8],
    ) -> Result<(NativeTextDiskCacheBlockMetadata, Self), NativeTextDiskCacheError> {
        let header = safetensors_header_json(bytes)?;
        let metadata = metadata_from_header(&header)?;
        let tensors = tensors_from_header(bytes, &header)?;
        let archive = Self { tensors, bytes };
        Ok((metadata, archive))
    }

    pub(crate) fn f32_tensor(&self, name: &str) -> Result<Vec<f32>, NativeTextDiskCacheError> {
        let tensor = self
            .tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .ok_or_else(|| {
                NativeTextDiskCacheError::integrity(format!("tensor `{name}` not found"))
            })?;
        Ok(tensor.values.clone())
    }

    pub(crate) fn f32_tensor_shape(
        &self,
        name: &str,
    ) -> Result<&[usize], NativeTextDiskCacheError> {
        self.tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .map(|tensor| tensor.shape.as_slice())
            .ok_or_else(|| {
                NativeTextDiskCacheError::integrity(format!("tensor `{name}` not found"))
            })
    }

    #[allow(dead_code)]
    fn bytes(&self) -> &[u8] {
        self.bytes
    }
}

pub(crate) fn encode_layer_kv_snapshot(
    prefix: &str,
    snapshot: LayerKvCacheSnapshot,
    sink: &mut NativeTextDiskCacheTensorSink,
) -> Result<NativeTextDiskFullAttentionLayout, NativeTextDiskCacheError> {
    if snapshot.config.format().to_string() != "f32" {
        return Err(NativeTextDiskCacheError::integrity(
            "only f32 KV cache snapshots can be written to disk",
        ));
    }
    let shape = vec![
        snapshot.token_count,
        snapshot.key_value_heads,
        snapshot.head_dim,
    ];
    let key_tensor = format!("{prefix}.keys");
    let value_tensor = format!("{prefix}.values");
    sink.push_f32(&key_tensor, shape.clone(), snapshot.keys.clone())?;
    sink.push_f32(&value_tensor, shape.clone(), snapshot.values.clone())?;
    Ok(NativeTextDiskFullAttentionLayout {
        revision: snapshot.revision,
        max_tokens: snapshot.max_tokens,
        key_value_heads: snapshot.key_value_heads,
        head_dim: snapshot.head_dim,
        token_count: snapshot.token_count,
        tokens_seen: snapshot.tokens_seen,
        cache_format: "f32".to_owned(),
        key_tensor,
        value_tensor,
        shape,
    })
}

pub(crate) fn decode_layer_kv_snapshot(
    layout: &NativeTextDiskFullAttentionLayout,
    archive: &NativeTextDiskCacheTensorArchive<'_>,
) -> Result<LayerKvCacheSnapshot, NativeTextDiskCacheError> {
    if layout.cache_format != "f32" {
        return Err(NativeTextDiskCacheError::integrity(
            "unsupported KV cache disk format",
        ));
    }
    let keys = archive.f32_tensor(&layout.key_tensor)?;
    let values = archive.f32_tensor(&layout.value_tensor)?;
    validate_tensor_shape(archive, &layout.key_tensor, &layout.shape)?;
    validate_tensor_shape(archive, &layout.value_tensor, &layout.shape)?;
    let config = LayerKvCache::new(layout.max_tokens, layout.key_value_heads, layout.head_dim)?
        .snapshot()
        .config;
    Ok(LayerKvCacheSnapshot {
        revision: layout.revision,
        config,
        max_tokens: layout.max_tokens,
        key_value_heads: layout.key_value_heads,
        head_dim: layout.head_dim,
        token_count: layout.token_count,
        tokens_seen: layout.tokens_seen,
        keys,
        values,
    })
}

pub(crate) fn encode_linear_attention_snapshot(
    prefix: &str,
    snapshot: LinearAttentionCacheSnapshot,
    sink: &mut NativeTextDiskCacheTensorSink,
) -> Result<NativeTextDiskLinearAttentionLayout, NativeTextDiskCacheError> {
    let conv_window_shape = vec![snapshot.conv_kernel_size, snapshot.conv_dim];
    let recurrent_state_shape = vec![
        snapshot.num_value_heads,
        snapshot.key_head_dim,
        snapshot.value_head_dim,
    ];
    let conv_window_tensor = format!("{prefix}.conv_window");
    let recurrent_state_tensor = format!("{prefix}.recurrent_state");
    sink.push_f32(
        &conv_window_tensor,
        conv_window_shape.clone(),
        snapshot.conv_window.clone(),
    )?;
    sink.push_f32(
        &recurrent_state_tensor,
        recurrent_state_shape.clone(),
        snapshot.recurrent_state.clone(),
    )?;
    Ok(NativeTextDiskLinearAttentionLayout {
        revision: snapshot.revision,
        conv_kernel_size: snapshot.conv_kernel_size,
        conv_dim: snapshot.conv_dim,
        num_value_heads: snapshot.num_value_heads,
        key_head_dim: snapshot.key_head_dim,
        value_head_dim: snapshot.value_head_dim,
        token_count: snapshot.token_count,
        conv_window_tensor,
        recurrent_state_tensor,
        conv_window_shape,
        recurrent_state_shape,
    })
}

pub(crate) fn decode_linear_attention_snapshot(
    layout: &NativeTextDiskLinearAttentionLayout,
    archive: &NativeTextDiskCacheTensorArchive<'_>,
) -> Result<LinearAttentionCacheSnapshot, NativeTextDiskCacheError> {
    validate_tensor_shape(
        archive,
        &layout.conv_window_tensor,
        &layout.conv_window_shape,
    )?;
    validate_tensor_shape(
        archive,
        &layout.recurrent_state_tensor,
        &layout.recurrent_state_shape,
    )?;
    Ok(LinearAttentionCacheSnapshot {
        revision: layout.revision,
        conv_kernel_size: layout.conv_kernel_size,
        conv_dim: layout.conv_dim,
        num_value_heads: layout.num_value_heads,
        key_head_dim: layout.key_head_dim,
        value_head_dim: layout.value_head_dim,
        token_count: layout.token_count,
        conv_window: archive.f32_tensor(&layout.conv_window_tensor)?,
        recurrent_state: archive.f32_tensor(&layout.recurrent_state_tensor)?,
    })
}

fn validate_tensor_shape(
    archive: &NativeTextDiskCacheTensorArchive<'_>,
    tensor: &str,
    expected: &[usize],
) -> Result<(), NativeTextDiskCacheError> {
    let actual = archive.f32_tensor_shape(tensor)?;
    if actual != expected {
        return Err(NativeTextDiskCacheError::integrity(format!(
            "tensor `{tensor}` shape mismatch"
        )));
    }
    Ok(())
}

fn encode_layer_kv_block_from_prefix_state(
    prefix: &str,
    state: &LayerKvCachePrefixState,
    block_start: usize,
    block_token_count: usize,
    sink: &mut NativeTextDiskCacheTensorSink,
) -> Result<NativeTextDiskFullAttentionLayout, NativeTextDiskCacheError> {
    let cache = LayerKvCache::from_prefix_cache_state(state)?;
    let snapshot = cache.snapshot();
    let prefix_end = block_start
        .checked_add(block_token_count)
        .ok_or_else(|| NativeTextDiskCacheError::integrity("disk cache block range overflow"))?;
    if prefix_end > snapshot.token_count {
        return Err(NativeTextDiskCacheError::integrity(
            "KV block range exceeds retained prefix state",
        ));
    }
    let vector_len = snapshot
        .key_value_heads
        .checked_mul(snapshot.head_dim)
        .ok_or_else(|| NativeTextDiskCacheError::integrity("KV block vector shape overflow"))?;
    let start = block_start
        .checked_mul(vector_len)
        .ok_or_else(|| NativeTextDiskCacheError::integrity("KV block start overflow"))?;
    let end = prefix_end
        .checked_mul(vector_len)
        .ok_or_else(|| NativeTextDiskCacheError::integrity("KV block end overflow"))?;
    let block_snapshot = LayerKvCacheSnapshot {
        revision: snapshot.revision,
        config: snapshot.config,
        max_tokens: snapshot.max_tokens,
        key_value_heads: snapshot.key_value_heads,
        head_dim: snapshot.head_dim,
        token_count: block_token_count,
        tokens_seen: prefix_end,
        keys: snapshot.keys[start..end].to_vec(),
        values: snapshot.values[start..end].to_vec(),
    };
    encode_layer_kv_snapshot(prefix, block_snapshot, sink)
}

fn assemble_layer_kv_prefix_state_blocks<'a>(
    states: impl IntoIterator<Item = &'a LayerKvCachePrefixState>,
) -> Result<LayerKvCachePrefixState, NativeTextDiskCacheError> {
    let mut snapshots = states
        .into_iter()
        .map(|state| LayerKvCache::from_prefix_cache_state(state).map(|cache| cache.snapshot()));
    let Some(first) = snapshots.next().transpose()? else {
        return Err(NativeTextDiskCacheError::integrity(
            "missing KV block state",
        ));
    };
    let mut revision = first.revision;
    let config = first.config;
    let max_tokens = first.max_tokens;
    let key_value_heads = first.key_value_heads;
    let head_dim = first.head_dim;
    let mut token_count = first.token_count;
    let mut keys = first.keys;
    let mut values = first.values;

    for snapshot in snapshots {
        let snapshot = snapshot?;
        if snapshot.config != config
            || snapshot.max_tokens != max_tokens
            || snapshot.key_value_heads != key_value_heads
            || snapshot.head_dim != head_dim
        {
            return Err(NativeTextDiskCacheError::integrity(
                "incompatible KV block shapes",
            ));
        }
        revision = snapshot.revision;
        token_count = token_count
            .checked_add(snapshot.token_count)
            .ok_or_else(|| NativeTextDiskCacheError::integrity("KV block token count overflow"))?;
        keys.extend_from_slice(&snapshot.keys);
        values.extend_from_slice(&snapshot.values);
    }
    if token_count > max_tokens {
        return Err(NativeTextDiskCacheError::integrity(
            "assembled KV prefix exceeds cache capacity",
        ));
    }
    Ok(LayerKvCache::from_snapshot(LayerKvCacheSnapshot {
        revision,
        config,
        max_tokens,
        key_value_heads,
        head_dim,
        token_count,
        tokens_seen: token_count,
        keys,
        values,
    })?
    .prefix_cache_state())
}

fn validate_contiguous_disk_blocks<S>(
    blocks: &[NativeTextDiskCacheStateBlock<S>],
) -> Result<(), NativeTextDiskCacheError> {
    let mut expected_start = 0_usize;
    for block in blocks {
        if block.block_start != expected_start {
            return Err(NativeTextDiskCacheError::integrity(
                "disk cache blocks are not contiguous",
            ));
        }
        expected_start = expected_start
            .checked_add(block.token_count)
            .ok_or_else(|| {
                NativeTextDiskCacheError::integrity("disk cache block range overflow")
            })?;
    }
    Ok(())
}

impl NativeTextDiskCacheValue for QwenLayerCache {
    fn encode_disk_block_states(
        states: &[Self::PrefixCacheState],
        block_start: usize,
        block_token_count: usize,
        sink: &mut NativeTextDiskCacheTensorSink,
    ) -> Result<Vec<NativeTextDiskCacheLayerLayout>, NativeTextDiskCacheError> {
        states
            .iter()
            .enumerate()
            .map(|(layer_idx, state)| match state {
                QwenLayerCachePrefixState::Full(state) => encode_layer_kv_block_from_prefix_state(
                    &format!("layers.{layer_idx}.full"),
                    state,
                    block_start,
                    block_token_count,
                    sink,
                )
                .map(NativeTextDiskCacheLayerLayout::QwenFull),
                QwenLayerCachePrefixState::Linear(snapshot) => {
                    let prefix_end =
                        block_start.checked_add(block_token_count).ok_or_else(|| {
                            NativeTextDiskCacheError::integrity("linear block range overflow")
                        })?;
                    if snapshot.token_count != prefix_end {
                        return Err(NativeTextDiskCacheError::integrity(
                            "linear attention snapshot is not at the block boundary",
                        ));
                    }
                    encode_linear_attention_snapshot(
                        &format!("layers.{layer_idx}.linear"),
                        snapshot.clone(),
                        sink,
                    )
                    .map(NativeTextDiskCacheLayerLayout::QwenLinear)
                }
            })
            .collect()
    }

    fn decode_disk_states(
        layouts: &[NativeTextDiskCacheLayerLayout],
        archive: &NativeTextDiskCacheTensorArchive<'_>,
    ) -> Result<Vec<Self::PrefixCacheState>, NativeTextDiskCacheError> {
        layouts
            .iter()
            .map(|layout| match layout {
                NativeTextDiskCacheLayerLayout::QwenFull(layout) => {
                    let snapshot = decode_layer_kv_snapshot(layout, archive)?;
                    Ok(QwenLayerCachePrefixState::Full(
                        LayerKvCache::from_snapshot(snapshot)?.prefix_cache_state(),
                    ))
                }
                NativeTextDiskCacheLayerLayout::QwenLinear(layout) => {
                    Ok(QwenLayerCachePrefixState::Linear(
                        decode_linear_attention_snapshot(layout, archive)?,
                    ))
                }
                _ => Err(NativeTextDiskCacheError::integrity(
                    "non-Qwen layer layout in Qwen disk cache block",
                )),
            })
            .collect()
    }

    fn assemble_disk_block_states(
        blocks: &[NativeTextDiskCacheStateBlock<Self::PrefixCacheState>],
    ) -> Result<Vec<Self::PrefixCacheState>, NativeTextDiskCacheError> {
        validate_contiguous_disk_blocks(blocks)?;
        let Some(first) = blocks.first() else {
            return Ok(Vec::new());
        };
        let layer_count = first.states.len();
        let mut assembled = Vec::with_capacity(layer_count);
        for layer_idx in 0..layer_count {
            match first.states.get(layer_idx).ok_or_else(|| {
                NativeTextDiskCacheError::integrity("missing Qwen disk block layer")
            })? {
                QwenLayerCachePrefixState::Full(_) => {
                    let layer_states = blocks
                        .iter()
                        .map(|block| {
                            if block.states.len() != layer_count {
                                return Err(NativeTextDiskCacheError::integrity(
                                    "inconsistent Qwen disk block layer count",
                                ));
                            }
                            match block.states.get(layer_idx) {
                                Some(QwenLayerCachePrefixState::Full(state)) => Ok(state),
                                _ => Err(NativeTextDiskCacheError::integrity(
                                    "mixed Qwen disk block layer layout",
                                )),
                            }
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    assembled.push(QwenLayerCachePrefixState::Full(
                        assemble_layer_kv_prefix_state_blocks(layer_states)?,
                    ));
                }
                QwenLayerCachePrefixState::Linear(_) => {
                    let Some(QwenLayerCachePrefixState::Linear(snapshot)) =
                        blocks.last().and_then(|block| block.states.get(layer_idx))
                    else {
                        return Err(NativeTextDiskCacheError::integrity(
                            "missing Qwen linear disk block terminal state",
                        ));
                    };
                    assembled.push(QwenLayerCachePrefixState::Linear(snapshot.clone()));
                }
            }
        }
        Ok(assembled)
    }
}

impl NativeTextDiskCacheValue for GemmaLayerCache {
    fn encode_disk_block_states(
        states: &[Self::PrefixCacheState],
        block_start: usize,
        block_token_count: usize,
        sink: &mut NativeTextDiskCacheTensorSink,
    ) -> Result<Vec<NativeTextDiskCacheLayerLayout>, NativeTextDiskCacheError> {
        states
            .iter()
            .enumerate()
            .map(|(layer_idx, state)| match state {
                GemmaLayerCachePrefixState::Attention(state) => {
                    encode_layer_kv_block_from_prefix_state(
                        &format!("layers.{layer_idx}.attention"),
                        state,
                        block_start,
                        block_token_count,
                        sink,
                    )
                    .map(NativeTextDiskCacheLayerLayout::GemmaFull)
                }
            })
            .collect()
    }

    fn decode_disk_states(
        layouts: &[NativeTextDiskCacheLayerLayout],
        archive: &NativeTextDiskCacheTensorArchive<'_>,
    ) -> Result<Vec<Self::PrefixCacheState>, NativeTextDiskCacheError> {
        layouts
            .iter()
            .map(|layout| match layout {
                NativeTextDiskCacheLayerLayout::GemmaFull(layout) => {
                    let snapshot = decode_layer_kv_snapshot(layout, archive)?;
                    Ok(GemmaLayerCachePrefixState::Attention(
                        LayerKvCache::from_snapshot(snapshot)?.prefix_cache_state(),
                    ))
                }
                _ => Err(NativeTextDiskCacheError::integrity(
                    "non-Gemma layer layout in Gemma disk cache block",
                )),
            })
            .collect()
    }

    fn assemble_disk_block_states(
        blocks: &[NativeTextDiskCacheStateBlock<Self::PrefixCacheState>],
    ) -> Result<Vec<Self::PrefixCacheState>, NativeTextDiskCacheError> {
        validate_contiguous_disk_blocks(blocks)?;
        let Some(first) = blocks.first() else {
            return Ok(Vec::new());
        };
        let layer_count = first.states.len();
        let mut assembled = Vec::with_capacity(layer_count);
        for layer_idx in 0..layer_count {
            let layer_states = blocks
                .iter()
                .map(|block| {
                    if block.states.len() != layer_count {
                        return Err(NativeTextDiskCacheError::integrity(
                            "inconsistent Gemma disk block layer count",
                        ));
                    }
                    match block.states.get(layer_idx) {
                        Some(GemmaLayerCachePrefixState::Attention(state)) => Ok(state),
                        _ => Err(NativeTextDiskCacheError::integrity(
                            "mixed Gemma disk block layer layout",
                        )),
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            assembled.push(GemmaLayerCachePrefixState::Attention(
                assemble_layer_kv_prefix_state_blocks(layer_states)?,
            ));
        }
        Ok(assembled)
    }
}

async fn reindex_disk_cache_root(
    config: &NativeTextDiskCacheConfig,
    identity: &NativeTextDiskCacheIdentity,
    index: &NativeTextDiskCacheIndex,
) -> Result<(), NativeTextDiskCacheError> {
    let model_root = config.root.join(identity.model_hash());
    if !tokio::fs::try_exists(&model_root)
        .await
        .map_err(|err| NativeTextDiskCacheError::io("check disk cache model root", err))?
    {
        return Ok(());
    }
    let mut stack = vec![model_root];
    while let Some(dir) = stack.pop() {
        let mut entries = match tokio::fs::read_dir(&dir).await {
            Ok(entries) => entries,
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    path = %dir.display(),
                    "could not read native text SSD prefix cache directory"
                );
                continue;
            }
        };
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|err| NativeTextDiskCacheError::io("read disk cache directory entry", err))?
        {
            let path = entry.path();
            let file_type = match entry.file_type().await {
                Ok(file_type) => file_type,
                Err(err) => {
                    tracing::warn!(
                        error = %err,
                        path = %path.display(),
                        "could not stat native text SSD prefix cache path"
                    );
                    continue;
                }
            };
            if file_type.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().and_then(|ext| ext.to_str()) != Some("safetensors") {
                continue;
            }
            reindex_disk_cache_file(identity, index, path).await;
        }
    }
    Ok(())
}

async fn reindex_disk_cache_file(
    identity: &NativeTextDiskCacheIdentity,
    index: &NativeTextDiskCacheIndex,
    path: PathBuf,
) {
    let bytes = match tokio::fs::read(&path).await {
        Ok(bytes) => bytes,
        Err(err) => {
            tracing::warn!(
                error = %err,
                path = %path.display(),
                "could not read native text SSD prefix cache block"
            );
            return;
        }
    };
    let descriptor = match NativeTextDiskCacheBlock::<QwenLayerCache>::decode_descriptor(&bytes) {
        Ok(descriptor) => descriptor,
        Err(err) => {
            tracing::warn!(
                error = %err,
                path = %path.display(),
                "ignoring invalid native text SSD prefix cache block"
            );
            return;
        }
    };
    if let Err(err) = descriptor.validate_for_identity(identity) {
        tracing::debug!(
            error = %err,
            path = %path.display(),
            "ignoring native text SSD prefix cache block for a different identity"
        );
        return;
    }
    if path.file_stem().and_then(|stem| stem.to_str()) != Some(descriptor.block_hash()) {
        tracing::debug!(
            path = %path.display(),
            block_hash = %descriptor.block_hash(),
            "ignoring native text SSD prefix cache block with stale filename"
        );
        return;
    }
    index.insert(
        NativeTextDiskCacheIndexKey::from_descriptor(&descriptor),
        NativeTextDiskCacheIndexedEntry {
            path,
            block_start: descriptor.block_start,
            token_count: descriptor.token_count,
        },
    );
}

fn validate_metadata_for_identity(
    metadata: &NativeTextDiskCacheBlockMetadata,
    identity: &NativeTextDiskCacheIdentity,
) -> Result<(), NativeTextDiskCacheError> {
    validate_metadata_codec_and_layout(metadata)?;
    if metadata.model_hash != identity.model_hash {
        return Err(NativeTextDiskCacheError::integrity("model hash mismatch"));
    }
    if metadata.snapshot_hash != identity.snapshot_hash {
        return Err(NativeTextDiskCacheError::integrity(
            "snapshot hash mismatch",
        ));
    }
    if metadata.model_family != identity.model_family {
        return Err(NativeTextDiskCacheError::integrity("model family mismatch"));
    }
    if let Some(namespace_hash) = identity.allowed_namespace_hash.as_deref()
        && metadata.namespace_hash != namespace_hash
    {
        return Err(NativeTextDiskCacheError::integrity(
            "namespace hash mismatch",
        ));
    }
    Ok(())
}

fn validate_metadata_codec_and_layout(
    metadata: &NativeTextDiskCacheBlockMetadata,
) -> Result<(), NativeTextDiskCacheError> {
    if metadata.codec != NATIVE_TEXT_DISK_CACHE_CODEC {
        return Err(NativeTextDiskCacheError::integrity(
            "unsupported disk cache codec",
        ));
    }
    if metadata.layout_version != NATIVE_TEXT_DISK_CACHE_LAYOUT_VERSION {
        return Err(NativeTextDiskCacheError::integrity(
            "unsupported disk cache layout version",
        ));
    }
    Ok(())
}

fn encode_safetensors(
    metadata: NativeTextDiskCacheBlockMetadata,
    tensors: Vec<NativeTextDiskCacheTensor>,
) -> Result<Vec<u8>, NativeTextDiskCacheError> {
    let mut payload = Vec::new();
    let mut header = serde_json::Map::new();
    header.insert(
        "__metadata__".to_owned(),
        serde_json::json!({
            "kir_ai": serde_json::to_string(&metadata).map_err(|err| {
                NativeTextDiskCacheError::integrity(format!("disk cache metadata encode failed: {err}"))
            })?,
        }),
    );
    let mut sorted = tensors;
    sorted.sort_by(|left, right| left.name.cmp(&right.name));
    for tensor in &sorted {
        let start = payload.len();
        for value in &tensor.values {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        let end = payload.len();
        header.insert(
            tensor.name.clone(),
            serde_json::json!({
                "dtype": "F32",
                "shape": tensor.shape,
                "data_offsets": [start, end],
            }),
        );
    }
    let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).map_err(|err| {
        NativeTextDiskCacheError::integrity(format!("safetensors header encode failed: {err}"))
    })?;
    let header_len: u64 = header_bytes
        .len()
        .try_into()
        .map_err(|_| NativeTextDiskCacheError::integrity("safetensors header too large"))?;
    let mut bytes = Vec::with_capacity(8 + header_bytes.len() + payload.len());
    bytes.extend_from_slice(&header_len.to_le_bytes());
    bytes.extend_from_slice(&header_bytes);
    bytes.extend_from_slice(&payload);
    Ok(bytes)
}

fn decode_safetensors_metadata(
    bytes: &[u8],
) -> Result<NativeTextDiskCacheBlockMetadata, NativeTextDiskCacheError> {
    metadata_from_header(&safetensors_header_json(bytes)?)
}

fn safetensors_header_json(
    bytes: &[u8],
) -> Result<serde_json::Map<String, serde_json::Value>, NativeTextDiskCacheError> {
    let header_len_bytes = bytes.get(0..8).ok_or_else(|| {
        NativeTextDiskCacheError::integrity("safetensors file is missing header length")
    })?;
    let header_len = u64::from_le_bytes(
        header_len_bytes
            .try_into()
            .map_err(|_| NativeTextDiskCacheError::integrity("header length is not 8 bytes"))?,
    );
    let header_len: usize = header_len
        .try_into()
        .map_err(|_| NativeTextDiskCacheError::integrity("safetensors header too large"))?;
    if header_len > MAX_SAFETENSORS_HEADER_LEN {
        return Err(NativeTextDiskCacheError::integrity(
            "safetensors header exceeds limit",
        ));
    }
    let header_end = 8_usize
        .checked_add(header_len)
        .ok_or_else(|| NativeTextDiskCacheError::integrity("safetensors header overflow"))?;
    let header = bytes
        .get(8..header_end)
        .ok_or_else(|| NativeTextDiskCacheError::integrity("safetensors header is truncated"))?;
    let value: serde_json::Value = serde_json::from_slice(header).map_err(|err| {
        NativeTextDiskCacheError::integrity(format!("invalid safetensors header json: {err}"))
    })?;
    value
        .as_object()
        .cloned()
        .ok_or_else(|| NativeTextDiskCacheError::integrity("safetensors header is not an object"))
}

fn metadata_from_header(
    header: &serde_json::Map<String, serde_json::Value>,
) -> Result<NativeTextDiskCacheBlockMetadata, NativeTextDiskCacheError> {
    let metadata = header
        .get("__metadata__")
        .and_then(serde_json::Value::as_object)
        .and_then(|metadata| metadata.get("kir_ai"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| NativeTextDiskCacheError::integrity("missing disk cache metadata"))?;
    serde_json::from_str(metadata).map_err(|err| {
        NativeTextDiskCacheError::integrity(format!("invalid disk cache metadata: {err}"))
    })
}

fn tensors_from_header(
    bytes: &[u8],
    header: &serde_json::Map<String, serde_json::Value>,
) -> Result<Vec<NativeTextDiskCacheTensor>, NativeTextDiskCacheError> {
    let header_len = u64::from_le_bytes(
        bytes
            .get(0..8)
            .ok_or_else(|| NativeTextDiskCacheError::integrity("missing header length"))?
            .try_into()
            .map_err(|_| NativeTextDiskCacheError::integrity("header length is not 8 bytes"))?,
    ) as usize;
    let data_start = 8_usize
        .checked_add(header_len)
        .ok_or_else(|| NativeTextDiskCacheError::integrity("safetensors data start overflow"))?;
    let mut tensors = Vec::new();
    for (name, value) in header {
        if name == "__metadata__" {
            continue;
        }
        let object = value.as_object().ok_or_else(|| {
            NativeTextDiskCacheError::integrity(format!("tensor `{name}` header is not an object"))
        })?;
        if object.get("dtype").and_then(serde_json::Value::as_str) != Some("F32") {
            return Err(NativeTextDiskCacheError::integrity(format!(
                "tensor `{name}` is not F32"
            )));
        }
        let shape = object
            .get("shape")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                NativeTextDiskCacheError::integrity(format!("tensor `{name}` is missing shape"))
            })?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or_else(|| {
                        NativeTextDiskCacheError::integrity(format!(
                            "tensor `{name}` has invalid shape"
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let offsets = object
            .get("data_offsets")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                NativeTextDiskCacheError::integrity(format!(
                    "tensor `{name}` is missing data offsets"
                ))
            })?;
        if offsets.len() != 2 {
            return Err(NativeTextDiskCacheError::integrity(format!(
                "tensor `{name}` data offsets are invalid"
            )));
        }
        let start = offsets[0]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| {
                NativeTextDiskCacheError::integrity(format!("tensor `{name}` offset is invalid"))
            })?;
        let end = offsets[1]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| {
                NativeTextDiskCacheError::integrity(format!("tensor `{name}` offset is invalid"))
            })?;
        if end < start || (end - start) % std::mem::size_of::<f32>() != 0 {
            return Err(NativeTextDiskCacheError::integrity(format!(
                "tensor `{name}` byte range is invalid"
            )));
        }
        let file_start = data_start
            .checked_add(start)
            .ok_or_else(|| NativeTextDiskCacheError::integrity("tensor start overflow"))?;
        let file_end = data_start
            .checked_add(end)
            .ok_or_else(|| NativeTextDiskCacheError::integrity("tensor end overflow"))?;
        let data = bytes.get(file_start..file_end).ok_or_else(|| {
            NativeTextDiskCacheError::integrity(format!("tensor `{name}` payload is truncated"))
        })?;
        let values = data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<_>>();
        tensors.push(NativeTextDiskCacheTensor {
            name: name.clone(),
            shape,
            values,
        });
    }
    if tensors.is_empty() {
        return Err(NativeTextDiskCacheError::integrity(
            "disk cache block has no tensors",
        ));
    }
    Ok(tensors)
}

#[cfg(test)]
fn native_text_disk_model_hash_from_namespace(
    namespace: &NativeTextPrefixCacheNamespace,
) -> String {
    let snapshot_hash = native_text_disk_snapshot_hash(Some("test-snapshot"));
    native_text_disk_model_hash(NativeTextDiskModelHashParts {
        model_id: &namespace.model_id,
        backend: &namespace.backend,
        family: namespace.family.as_deref(),
        quantization: namespace.quantization.as_deref(),
        repo_id: namespace.repo_id.as_deref(),
        resolved_commit: namespace.resolved_commit.as_deref(),
        profile: namespace.profile.as_deref(),
        snapshot_hash: &snapshot_hash,
    })
}

fn native_text_disk_snapshot_hash(snapshot_identity: Option<&str>) -> String {
    hash_components(
        "kir-ai-native-text-disk-snapshot/v1",
        [("snapshot_identity", snapshot_identity)],
    )
}

struct NativeTextDiskModelHashParts<'a> {
    model_id: &'a str,
    backend: &'a str,
    family: Option<&'a str>,
    quantization: Option<&'a str>,
    repo_id: Option<&'a str>,
    resolved_commit: Option<&'a str>,
    profile: Option<&'a str>,
    snapshot_hash: &'a str,
}

fn native_text_disk_model_hash(parts: NativeTextDiskModelHashParts<'_>) -> String {
    hash_components(
        "kir-ai-native-text-disk-model/v1",
        [
            ("model_id", Some(parts.model_id)),
            ("backend", Some(parts.backend)),
            ("family", parts.family),
            ("quantization", parts.quantization),
            ("repo_id", parts.repo_id),
            ("resolved_commit", parts.resolved_commit),
            ("profile", parts.profile),
            ("snapshot_hash", Some(parts.snapshot_hash)),
        ],
    )
}

fn native_text_disk_namespace_hash(namespace: &NativeTextPrefixCacheNamespace) -> String {
    let cache_layout_version = namespace.cache_layout_version.to_string();
    let cache_tokens = namespace.cache_tokens.to_string();
    let max_prefill_tokens = namespace.max_prefill_tokens.to_string();
    hash_components(
        "kir-ai-native-text-disk-namespace/v1",
        [
            ("model_id", Some(namespace.model_id.as_str())),
            ("backend", Some(namespace.backend.as_str())),
            ("family", namespace.family.as_deref()),
            ("quantization", namespace.quantization.as_deref()),
            ("repo_id", namespace.repo_id.as_deref()),
            ("resolved_commit", namespace.resolved_commit.as_deref()),
            ("profile", namespace.profile.as_deref()),
            ("cache_key", Some(namespace.cache_key.as_str())),
            ("tool_schema", namespace.tool_schema.as_deref()),
            ("request_mode", Some(namespace.request_mode.as_str())),
            ("cache_layout_version", Some(cache_layout_version.as_str())),
            ("cache_tokens", Some(cache_tokens.as_str())),
            ("max_prefill_tokens", Some(max_prefill_tokens.as_str())),
        ],
    )
}

fn native_text_disk_block_hash(
    model_hash: &str,
    namespace_hash: &str,
    block_start: usize,
    tokens: &[usize],
) -> String {
    let mut hasher = Sha256::new();
    update_hash_value(&mut hasher, Some("kir-ai-native-text-disk-block/v1"));
    update_hash_value(&mut hasher, Some(model_hash));
    update_hash_value(&mut hasher, Some(namespace_hash));
    hasher.update((block_start as u64).to_le_bytes());
    hasher.update((tokens.len() as u64).to_le_bytes());
    for token in tokens {
        hasher.update((*token as u64).to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn hash_components<'a>(
    label: &str,
    components: impl IntoIterator<Item = (&'static str, Option<&'a str>)>,
) -> String {
    let mut hasher = Sha256::new();
    update_hash_value(&mut hasher, Some(label));
    for (name, value) in components {
        update_hash_value(&mut hasher, Some(name));
        update_hash_value(&mut hasher, value);
    }
    format!("{:x}", hasher.finalize())
}

fn update_hash_value(hasher: &mut Sha256, value: Option<&str>) {
    match value {
        Some(value) => {
            hasher.update([1]);
            hasher.update((value.len() as u64).to_le_bytes());
            hasher.update(value.as_bytes());
        }
        None => hasher.update([0]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_text::{
        NativeTextPrefixCache, NativeTextPrefixCacheMetrics, NativeTextPrefixCacheNamespace,
        NativeTextPrefixCacheValue,
    };
    use llm_backend::native::{
        GemmaLayerCache, LayerKvCache, LinearAttentionCache, QwenLayerCache,
    };
    use llm_backend_contracts::BackendModelMetadata;
    use std::{
        sync::atomic::{AtomicUsize, Ordering},
        time::{Duration, Instant},
    };

    fn namespace(label: &str, family: &str) -> NativeTextPrefixCacheNamespace {
        NativeTextPrefixCacheNamespace {
            model_id: format!("model-{label}"),
            backend: "native-test".to_owned(),
            family: Some(family.to_owned()),
            quantization: Some("bf16".to_owned()),
            repo_id: Some("org/model".to_owned()),
            resolved_commit: Some("abc123".to_owned()),
            profile: Some(label.to_owned()),
            cache_key: format!("cache-key-{label}"),
            tool_schema: None,
            request_mode: "chat,json_object=false,required_tool=None".to_owned(),
            cache_layout_version: 1,
            cache_tokens: 16,
            max_prefill_tokens: 4,
        }
    }

    fn filled_layer_cache(max_tokens: usize) -> LayerKvCache {
        let mut cache = LayerKvCache::new(max_tokens, 2, 3).expect("layer cache shape is valid");
        cache
            .append(
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            )
            .expect("first token appends");
        cache
            .append(
                &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                &[70.0, 80.0, 90.0, 100.0, 110.0, 120.0],
            )
            .expect("second token appends");
        cache
    }

    fn filled_linear_cache() -> LinearAttentionCache {
        let mut cache = LinearAttentionCache::new(2, 3, 2, 2, 3)
            .expect("linear attention cache shape is valid");
        cache
            .push_conv_input(&[1.0, 2.0, 3.0])
            .expect("conv input appends");
        cache
            .push_conv_input(&[4.0, 5.0, 6.0])
            .expect("second conv input appends");
        cache
            .replace_recurrent_state(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
            .expect("recurrent state matches shape");
        cache
    }

    fn round_trip<C>(family: &str, states: Vec<C::PrefixCacheState>)
    where
        C: NativeTextDiskCacheValue + NativeTextPrefixCacheValue,
    {
        let namespace = namespace("round-trip", family);
        let identity = NativeTextDiskCacheIdentity::from_namespace(&namespace, family);
        let descriptor =
            NativeTextDiskCacheBlockDescriptor::new(&identity, &namespace, 0, &[11, 12]);
        let hidden = vec![0.25, 0.5, 0.75];

        let encoded = NativeTextDiskCacheBlock::<C>::encode(&descriptor, &hidden, &states)
            .expect("prefix block encodes");
        let decoded = NativeTextDiskCacheBlock::<C>::decode(&encoded, &identity, &descriptor)
            .expect("prefix block decodes");

        assert_eq!(decoded.token_count, 2);
        assert_eq!(decoded.hidden, hidden);
        assert_eq!(decoded.states.len(), states.len());
        assert_eq!(
            C::prefix_cache_entry_bytes(&hidden, &decoded.states),
            C::prefix_cache_entry_bytes(&hidden, &states)
        );
        assert!(
            C::prefix_cache_from_state(&decoded.states).is_some(),
            "decoded disk states restore into hot cache values"
        );

        let mut wrong_cache_layout = encoded;
        NativeTextDiskCacheBlock::<C>::rewrite_cache_layout_version_for_test(
            &mut wrong_cache_layout,
            descriptor.cache_layout_version + 1,
        )
        .expect("cache layout metadata is rewritten");
        assert!(
            NativeTextDiskCacheBlock::<C>::decode(&wrong_cache_layout, &identity, &descriptor)
                .is_err(),
            "blocks for another cache layout must not decode"
        );
    }

    #[test]
    fn snapshot_codec_round_trips_qwen_full_qwen_linear_and_gemma_attention_blocks() {
        let qwen_full = QwenLayerCache::Full(filled_layer_cache(4)).prefix_cache_state();
        let qwen_linear = QwenLayerCache::Linear(filled_linear_cache()).prefix_cache_state();
        let gemma_attention =
            GemmaLayerCache::Attention(filled_layer_cache(4)).prefix_cache_state();

        round_trip::<QwenLayerCache>("qwen", vec![qwen_full, qwen_linear]);
        round_trip::<GemmaLayerCache>("gemma", vec![gemma_attention]);
    }

    #[tokio::test]
    async fn startup_reindex_ignores_wrong_namespace_model_layout_version_and_corrupt_files() {
        let temp = tempfile::tempdir().expect("temp dir exists");
        let config = NativeTextDiskCacheConfig::for_root(temp.path())
            .with_writer_queue_depth(4)
            .with_block_token_count(2);
        let valid_namespace = namespace("valid", "qwen");
        let identity = NativeTextDiskCacheIdentity::from_namespace(&valid_namespace, "qwen");
        let hidden = vec![0.25, 0.5];
        let states = vec![QwenLayerCache::Full(filled_layer_cache(4)).prefix_cache_state()];

        let cache = NativeTextDiskCache::<QwenLayerCache>::open(config.clone(), identity.clone())
            .await
            .expect("cache opens");
        assert_eq!(
            cache.queue_store(&valid_namespace, &[11, 12], &hidden, &states),
            NativeTextDiskCacheStoreStatus::Queued
        );
        cache.flush_for_test().await.expect("queued write flushes");

        let valid_descriptor =
            NativeTextDiskCacheBlockDescriptor::new(&identity, &valid_namespace, 0, &[11, 12]);
        let valid_bytes =
            NativeTextDiskCacheBlock::<QwenLayerCache>::encode(&valid_descriptor, &hidden, &states)
                .expect("valid block encodes");
        let wrong_namespace_value = namespace("wrong", "qwen");
        let wrong_namespace = NativeTextDiskCacheBlockDescriptor::new(
            &identity,
            &wrong_namespace_value,
            0,
            &[11, 12],
        );
        std::fs::write(
            cache.path_for_descriptor_for_test(&wrong_namespace),
            NativeTextDiskCacheBlock::<QwenLayerCache>::encode(&wrong_namespace, &hidden, &states)
                .expect("wrong namespace block encodes"),
        )
        .expect("wrong namespace file writes");
        let wrong_model = NativeTextDiskCacheIdentity::for_test("wrong-model", "qwen");
        let wrong_model_descriptor =
            NativeTextDiskCacheBlockDescriptor::new(&wrong_model, &valid_namespace, 0, &[11, 12]);
        std::fs::write(
            cache.path_for_descriptor_for_test(&wrong_model_descriptor),
            NativeTextDiskCacheBlock::<QwenLayerCache>::encode(
                &wrong_model_descriptor,
                &hidden,
                &states,
            )
            .expect("wrong model block encodes"),
        )
        .expect("wrong model file writes");
        let mut wrong_version = valid_bytes;
        NativeTextDiskCacheBlock::<QwenLayerCache>::rewrite_layout_version_for_test(
            &mut wrong_version,
            99,
        )
        .expect("layout metadata is rewritten");
        std::fs::write(
            cache
                .path_for_descriptor_for_test(&valid_descriptor)
                .with_file_name("wrong-version.safetensors"),
            wrong_version,
        )
        .expect("wrong version file writes");
        let corrupt_dir = cache
            .root_for_test()
            .join(identity.model_hash())
            .join("stale");
        std::fs::create_dir_all(&corrupt_dir).expect("corrupt dir creates");
        std::fs::write(
            corrupt_dir.join("corrupt.safetensors"),
            b"not a safetensors file",
        )
        .expect("corrupt file writes");
        drop(cache);

        let reindexed = NativeTextDiskCache::<QwenLayerCache>::open(config, identity)
            .await
            .expect("corrupt or mismatched files do not fail startup");

        assert_eq!(reindexed.indexed_entry_count_for_test(), 1);
    }

    #[tokio::test]
    async fn snapshot_identity_partitions_model_hash_and_rejects_wrong_snapshot_metadata() {
        let temp = tempfile::tempdir().expect("temp dir exists");
        let config = NativeTextDiskCacheConfig::for_root(temp.path()).with_block_token_count(2);
        let mut metadata =
            BackendModelMetadata::new("model-shared", "native-test").with_family("qwen");
        metadata.repo_id = Some("org/model".to_owned());
        metadata.resolved_commit = Some("abc123".to_owned());
        metadata.profile = Some("default".to_owned());
        let first_identity = NativeTextDiskCacheIdentity::from_model_metadata(
            &metadata,
            "qwen",
            Some("manifest:sha256:first"),
        );
        let second_identity = NativeTextDiskCacheIdentity::from_model_metadata(
            &metadata,
            "qwen",
            Some("manifest:sha256:second"),
        );
        let namespace = namespace("snapshot", "qwen");
        let hidden = vec![0.25, 0.5];
        let states = vec![QwenLayerCache::Full(filled_layer_cache(4)).prefix_cache_state()];

        assert_ne!(first_identity.model_hash(), second_identity.model_hash());
        assert_ne!(
            first_identity.snapshot_hash(),
            second_identity.snapshot_hash()
        );

        let first_cache =
            NativeTextDiskCache::<QwenLayerCache>::open(config.clone(), first_identity.clone())
                .await
                .expect("first snapshot cache opens");
        assert_eq!(
            first_cache.queue_store(&namespace, &[11, 12], &hidden, &states),
            NativeTextDiskCacheStoreStatus::Queued
        );
        first_cache
            .flush_for_test()
            .await
            .expect("first snapshot write flushes");

        let wrong_descriptor =
            NativeTextDiskCacheBlockDescriptor::new(&first_identity, &namespace, 0, &[11, 12]);
        let wrong_bytes =
            NativeTextDiskCacheBlock::<QwenLayerCache>::encode(&wrong_descriptor, &hidden, &states)
                .expect("wrong snapshot block encodes");
        let second_descriptor =
            NativeTextDiskCacheBlockDescriptor::new(&second_identity, &namespace, 0, &[11, 12]);
        let second_cache =
            NativeTextDiskCache::<QwenLayerCache>::open(config.clone(), second_identity.clone())
                .await
                .expect("second snapshot cache opens");
        std::fs::write(
            second_cache.path_for_descriptor_for_test(&second_descriptor),
            wrong_bytes,
        )
        .expect("wrong snapshot file writes under second model root");
        drop(second_cache);

        let reindexed = NativeTextDiskCache::<QwenLayerCache>::open(config, second_identity)
            .await
            .expect("wrong snapshot metadata does not fail startup");

        assert_eq!(reindexed.indexed_entry_count_for_test(), 0);
        assert!(
            reindexed
                .lookup(&namespace, &[11, 12, 13], |_| true)
                .await
                .expect("lookup succeeds")
                .is_none(),
            "a block encoded for another snapshot must be ignored"
        );
    }

    #[test]
    fn bounded_writer_backpressure_drops_without_blocking_generation() {
        let (writer, _rx) = NativeTextDiskCacheWriter::detached_for_test(1);
        let job = NativeTextDiskCacheWriteJob::for_test("a.safetensors", vec![1, 2, 3]);

        assert_eq!(
            writer.try_enqueue(job.clone()),
            NativeTextDiskCacheStoreStatus::Queued
        );
        let started = Instant::now();
        assert_eq!(
            writer.try_enqueue(job),
            NativeTextDiskCacheStoreStatus::Dropped
        );

        assert!(
            started.elapsed() < Duration::from_millis(25),
            "try_enqueue must not wait for writer capacity"
        );
    }

    #[test]
    fn queue_store_drops_before_encoding_when_writer_queue_is_full() {
        static ENCODE_CALLS: AtomicUsize = AtomicUsize::new(0);

        #[derive(Debug, Clone, PartialEq, Eq)]
        struct EncodingProbeCache {
            marker: u32,
        }

        impl NativeTextPrefixCacheValue for EncodingProbeCache {
            type PrefixCacheState = Self;

            fn prefix_cache_state(caches: &[Self]) -> Vec<Self::PrefixCacheState> {
                caches.to_vec()
            }

            fn prefix_cache_from_state(states: &[Self::PrefixCacheState]) -> Option<Vec<Self>> {
                Some(states.to_vec())
            }

            fn prefix_cache_entry_bytes(hidden: &[f32], states: &[Self::PrefixCacheState]) -> u64 {
                std::mem::size_of_val(hidden) as u64
                    + states.len() as u64 * std::mem::size_of::<Self>() as u64
            }
        }

        impl NativeTextDiskCacheValue for EncodingProbeCache {
            fn encode_disk_block_states(
                states: &[Self::PrefixCacheState],
                block_start: usize,
                block_token_count: usize,
                sink: &mut NativeTextDiskCacheTensorSink,
            ) -> Result<Vec<NativeTextDiskCacheLayerLayout>, NativeTextDiskCacheError> {
                ENCODE_CALLS.fetch_add(1, Ordering::SeqCst);
                let values = states[block_start..block_start + block_token_count]
                    .iter()
                    .map(|state| state.marker as f32)
                    .collect::<Vec<_>>();
                sink.push_f32("probe.markers", vec![values.len()], values)?;
                Ok(vec![NativeTextDiskCacheLayerLayout::test_marker_tensor(
                    "probe.markers",
                )])
            }

            fn decode_disk_states(
                _layouts: &[NativeTextDiskCacheLayerLayout],
                _archive: &NativeTextDiskCacheTensorArchive<'_>,
            ) -> Result<Vec<Self::PrefixCacheState>, NativeTextDiskCacheError> {
                Ok(Vec::new())
            }

            fn assemble_disk_block_states(
                blocks: &[NativeTextDiskCacheStateBlock<Self::PrefixCacheState>],
            ) -> Result<Vec<Self::PrefixCacheState>, NativeTextDiskCacheError> {
                Ok(blocks
                    .iter()
                    .flat_map(|block| block.states.iter().cloned())
                    .collect())
            }
        }

        ENCODE_CALLS.store(0, Ordering::SeqCst);
        let (writer, _rx) = NativeTextDiskCacheWriter::detached_for_test(1);
        let cache = NativeTextDiskCache::<EncodingProbeCache> {
            config: NativeTextDiskCacheConfig::for_root("unused").with_block_token_count(2),
            identity: NativeTextDiskCacheIdentity::for_test("model", "test"),
            index: NativeTextDiskCacheIndex::default(),
            writer,
            _cache: PhantomData,
        };
        let namespace = namespace("encoding-probe", "test");
        let hidden = [1.0, 2.0];
        let states = vec![
            EncodingProbeCache { marker: 1 },
            EncodingProbeCache { marker: 2 },
            EncodingProbeCache { marker: 3 },
            EncodingProbeCache { marker: 4 },
        ];

        assert_eq!(
            cache.queue_store(&namespace, &[1, 2], &hidden, &states[..2]),
            NativeTextDiskCacheStoreStatus::Queued
        );
        assert_eq!(ENCODE_CALLS.load(Ordering::SeqCst), 1);

        assert_eq!(
            cache.queue_store(&namespace, &[1, 2, 3, 4], &hidden, &states),
            NativeTextDiskCacheStoreStatus::Dropped
        );
        assert_eq!(
            ENCODE_CALLS.load(Ordering::SeqCst),
            1,
            "full queue must be detected before disk payload encoding runs"
        );
    }

    #[tokio::test]
    async fn startup_reindex_handles_nested_block_dirs_and_stale_files() {
        let temp = tempfile::tempdir().expect("temp dir exists");
        let config = NativeTextDiskCacheConfig::for_root(temp.path()).with_block_token_count(2);
        let namespace = namespace("nested", "gemma");
        let identity = NativeTextDiskCacheIdentity::from_namespace(&namespace, "gemma");
        let descriptor =
            NativeTextDiskCacheBlockDescriptor::new(&identity, &namespace, 0, &[21, 22]);
        let states = vec![GemmaLayerCache::Attention(filled_layer_cache(4)).prefix_cache_state()];
        let hidden = vec![0.25, 0.5];
        let bytes =
            NativeTextDiskCacheBlock::<GemmaLayerCache>::encode(&descriptor, &hidden, &states)
                .expect("block encodes");
        let nested = temp
            .path()
            .join(identity.model_hash())
            .join(descriptor.namespace_hash())
            .join("aa");
        std::fs::create_dir_all(&nested).expect("nested dirs create");
        std::fs::write(
            nested.join(format!("{}.safetensors", descriptor.block_hash())),
            bytes,
        )
        .expect("nested block writes");
        std::fs::write(nested.join("README.txt"), b"stale").expect("stale file writes");

        let cache = NativeTextDiskCache::<GemmaLayerCache>::open(config, identity)
            .await
            .expect("nested reindex succeeds");

        assert_eq!(cache.indexed_entry_count_for_test(), 1);
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct DummyCache {
        marker: u32,
    }

    impl NativeTextPrefixCacheValue for DummyCache {
        type PrefixCacheState = Self;

        fn prefix_cache_state(caches: &[Self]) -> Vec<Self::PrefixCacheState> {
            caches.to_vec()
        }

        fn prefix_cache_from_state(states: &[Self::PrefixCacheState]) -> Option<Vec<Self>> {
            Some(states.to_vec())
        }

        fn prefix_cache_entry_bytes(hidden: &[f32], states: &[Self::PrefixCacheState]) -> u64 {
            std::mem::size_of_val(hidden) as u64
                + states.len() as u64 * std::mem::size_of::<Self>() as u64
        }
    }

    impl NativeTextDiskCacheValue for DummyCache {
        fn encode_disk_block_states(
            states: &[Self::PrefixCacheState],
            block_start: usize,
            block_token_count: usize,
            sink: &mut NativeTextDiskCacheTensorSink,
        ) -> Result<Vec<NativeTextDiskCacheLayerLayout>, NativeTextDiskCacheError> {
            let values = states[block_start..block_start + block_token_count]
                .iter()
                .map(|state| state.marker as f32)
                .collect::<Vec<_>>();
            sink.push_f32("dummy.markers", vec![values.len()], values)?;
            Ok(vec![NativeTextDiskCacheLayerLayout::test_marker_tensor(
                "dummy.markers",
            )])
        }

        fn decode_disk_states(
            layouts: &[NativeTextDiskCacheLayerLayout],
            archive: &NativeTextDiskCacheTensorArchive<'_>,
        ) -> Result<Vec<Self::PrefixCacheState>, NativeTextDiskCacheError> {
            let Some(layout) = layouts.first() else {
                return Err(NativeTextDiskCacheError::integrity("missing dummy layout"));
            };
            let tensor = layout
                .test_marker_tensor_name()
                .ok_or_else(|| NativeTextDiskCacheError::integrity("wrong dummy layout"))?;
            archive
                .f32_tensor(tensor)?
                .into_iter()
                .map(|marker| {
                    if marker.fract() != 0.0 || marker < 0.0 {
                        return Err(NativeTextDiskCacheError::integrity(
                            "dummy marker must be a non-negative integer",
                        ));
                    }
                    Ok(DummyCache {
                        marker: marker as u32,
                    })
                })
                .collect()
        }

        fn assemble_disk_block_states(
            blocks: &[NativeTextDiskCacheStateBlock<Self::PrefixCacheState>],
        ) -> Result<Vec<Self::PrefixCacheState>, NativeTextDiskCacheError> {
            Ok(blocks
                .iter()
                .flat_map(|block| block.states.iter().cloned())
                .collect())
        }
    }

    #[tokio::test]
    async fn block_store_writes_only_terminal_block_payload_not_accumulated_prefix() {
        let temp = tempfile::tempdir().expect("temp dir exists");
        let config = NativeTextDiskCacheConfig::for_root(temp.path())
            .with_writer_queue_depth(4)
            .with_block_token_count(2);
        let namespace = namespace("block-payload", "test");
        let identity = NativeTextDiskCacheIdentity::from_namespace(&namespace, "test");
        let disk = NativeTextDiskCache::<DummyCache>::open(config, identity.clone())
            .await
            .expect("cache opens");
        let first_hidden = vec![1.0];
        let second_hidden = vec![2.0];
        let first_states = vec![DummyCache { marker: 1 }, DummyCache { marker: 2 }];
        let second_states = vec![
            DummyCache { marker: 1 },
            DummyCache { marker: 2 },
            DummyCache { marker: 3 },
            DummyCache { marker: 4 },
        ];

        assert_eq!(
            disk.queue_store(&namespace, &[31, 32], &first_hidden, &first_states),
            NativeTextDiskCacheStoreStatus::Queued
        );
        assert_eq!(
            disk.queue_store(
                &namespace,
                &[31, 32, 33, 34],
                &second_hidden,
                &second_states
            ),
            NativeTextDiskCacheStoreStatus::Queued
        );
        disk.flush_for_test().await.expect("queued writes flush");

        let first_descriptor =
            NativeTextDiskCacheBlockDescriptor::new(&identity, &namespace, 0, &[31, 32]);
        let second_descriptor =
            NativeTextDiskCacheBlockDescriptor::new(&identity, &namespace, 2, &[33, 34]);
        let first_bytes = std::fs::read(disk.path_for_descriptor_for_test(&first_descriptor))
            .expect("first block exists");
        let second_bytes = std::fs::read(disk.path_for_descriptor_for_test(&second_descriptor))
            .expect("second block exists");
        let first_block = NativeTextDiskCacheBlock::<DummyCache>::decode(
            &first_bytes,
            &identity,
            &first_descriptor,
        )
        .expect("first block decodes");
        let second_block = NativeTextDiskCacheBlock::<DummyCache>::decode(
            &second_bytes,
            &identity,
            &second_descriptor,
        )
        .expect("second block decodes");

        assert_eq!(first_block.block_start, 0);
        assert_eq!(first_block.token_count, 2);
        assert_eq!(first_block.states, first_states);
        assert_eq!(second_block.block_start, 2);
        assert_eq!(second_block.token_count, 2);
        assert_eq!(
            second_block.states,
            vec![DummyCache { marker: 3 }, DummyCache { marker: 4 }],
            "later block files must not duplicate the earlier prefix payload"
        );
    }

    #[tokio::test]
    async fn lookup_assembles_prefix_from_multiple_independent_block_entries() {
        let temp = tempfile::tempdir().expect("temp dir exists");
        let config = NativeTextDiskCacheConfig::for_root(temp.path()).with_block_token_count(2);
        let namespace = namespace("assembled", "test");
        let identity = NativeTextDiskCacheIdentity::from_namespace(&namespace, "test");
        let disk = NativeTextDiskCache::<DummyCache>::open(config.clone(), identity.clone())
            .await
            .expect("cache opens");
        let first_descriptor =
            NativeTextDiskCacheBlockDescriptor::new(&identity, &namespace, 0, &[41, 42]);
        let second_descriptor =
            NativeTextDiskCacheBlockDescriptor::new(&identity, &namespace, 2, &[43, 44]);
        std::fs::write(
            disk.path_for_descriptor_for_test(&first_descriptor),
            NativeTextDiskCacheBlock::<DummyCache>::encode(
                &first_descriptor,
                &[1.0],
                &[DummyCache { marker: 1 }, DummyCache { marker: 2 }],
            )
            .expect("first block encodes"),
        )
        .expect("first block writes");
        std::fs::write(
            disk.path_for_descriptor_for_test(&second_descriptor),
            NativeTextDiskCacheBlock::<DummyCache>::encode(
                &second_descriptor,
                &[2.0],
                &[
                    DummyCache { marker: 1 },
                    DummyCache { marker: 2 },
                    DummyCache { marker: 3 },
                    DummyCache { marker: 4 },
                ],
            )
            .expect("second block encodes"),
        )
        .expect("second block writes");
        drop(disk);

        let reindexed = NativeTextDiskCache::<DummyCache>::open(config, identity)
            .await
            .expect("cache reindexes independent blocks");
        let hit = reindexed
            .lookup(&namespace, &[41, 42, 43, 44, 45], |_| true)
            .await
            .expect("lookup succeeds")
            .expect("assembled prefix hit exists");

        assert_eq!(hit.token_count, 4);
        assert_eq!(hit.hidden, vec![2.0]);
        assert_eq!(
            hit.caches,
            vec![
                DummyCache { marker: 1 },
                DummyCache { marker: 2 },
                DummyCache { marker: 3 },
                DummyCache { marker: 4 },
            ]
        );
    }

    #[tokio::test]
    async fn disk_hits_promote_validated_blocks_into_hot_prefix_cache() {
        let temp = tempfile::tempdir().expect("temp dir exists");
        let config = NativeTextDiskCacheConfig::for_root(temp.path()).with_block_token_count(2);
        let namespace = namespace("promote", "test");
        let identity = NativeTextDiskCacheIdentity::from_namespace(&namespace, "test");
        let disk = NativeTextDiskCache::<DummyCache>::open(config, identity)
            .await
            .expect("cache opens");
        let metrics = NativeTextPrefixCacheMetrics::default();
        let memory = NativeTextPrefixCache::<DummyCache>::new(1024);
        let hidden = vec![1.0, 2.0];
        let states = vec![DummyCache { marker: 41 }, DummyCache { marker: 42 }];

        assert_eq!(
            disk.queue_store(&namespace, &[31, 32], &hidden, &states),
            NativeTextDiskCacheStoreStatus::Queued
        );
        disk.flush_for_test().await.expect("queued write flushes");
        let hit = disk
            .lookup(&namespace, &[31, 32, 33], |_| true)
            .await
            .expect("disk lookup succeeds")
            .expect("disk prefix hit exists");

        disk.promote_hit(&memory, namespace.clone(), &[31, 32], &metrics, &hit);
        let promoted = memory
            .lookup(&namespace, &[31, 32, 33], &metrics)
            .expect("memory cache now has promoted disk hit");

        assert_eq!(promoted.token_count, 2);
        assert_eq!(promoted.hidden, hidden);
        assert_eq!(promoted.caches, states);
    }
}
