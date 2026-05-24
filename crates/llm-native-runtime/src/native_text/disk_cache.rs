mod codec;
mod hashing;
mod index;
mod io;
#[cfg(test)]
mod tests;
mod writer;

#[cfg(test)]
use self::hashing::native_text_disk_model_hash_from_namespace;
pub(crate) use self::{
    codec::NativeTextDiskCacheLayerLayout,
    io::{NativeTextDiskCacheTensorArchive, NativeTextDiskCacheTensorSink},
};
use self::{
    hashing::{
        NativeTextDiskModelHashParts, native_text_disk_block_hash, native_text_disk_model_hash,
        native_text_disk_namespace_hash, native_text_disk_previous_block_hash,
        native_text_disk_snapshot_hash,
    },
    index::{
        NativeTextDiskCacheIndex, NativeTextDiskCacheIndexKey, NativeTextDiskCacheIndexedEntry,
        reindex_disk_cache_root,
    },
    io::NativeTextDiskCacheBlock,
    writer::{
        NativeTextDiskCacheWriteJob, NativeTextDiskCacheWriter, NativeTextDiskCacheWriterMessage,
    },
};
use super::{
    NativeTextPrefixCache, NativeTextPrefixCacheHit, NativeTextPrefixCacheMetrics,
    NativeTextPrefixCacheNamespace, NativeTextPrefixCacheValue,
};
use llm_backend::native::{KvCacheError, TensorLoadError};
use llm_backend_contracts::BackendModelMetadata;
use std::{
    fmt,
    marker::PhantomData,
    path::{Path, PathBuf},
};

const DEFAULT_WRITER_QUEUE_DEPTH: usize = 8;
const DEFAULT_BLOCK_TOKEN_COUNT: usize = 256;

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

pub(crate) async fn native_text_disk_cache_snapshot_identity(
    snapshot_path: &Path,
    manifest_digest: Option<&str>,
) -> String {
    if let Some(manifest_digest) = manifest_digest {
        return format!("manifest:{manifest_digest}");
    }
    let canonical = tokio::fs::canonicalize(snapshot_path)
        .await
        .unwrap_or_else(|_| snapshot_path.to_path_buf());
    format!("raw-path:{}", canonical.display())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NativeTextDiskCacheBlockDescriptor {
    model_hash: String,
    snapshot_hash: String,
    model_family: String,
    namespace_hash: String,
    previous_block_hash: String,
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
        prefix_tokens: &[usize],
    ) -> Result<Self, NativeTextDiskCacheError> {
        let namespace_hash = native_text_disk_namespace_hash(namespace);
        let block_tokens = prefix_tokens.get(block_start..).ok_or_else(|| {
            NativeTextDiskCacheError::integrity("disk cache block start exceeds prefix tokens")
        })?;
        if block_tokens.is_empty() {
            return Err(NativeTextDiskCacheError::integrity(
                "disk cache block has no tokens",
            ));
        }
        let previous_block_hash = native_text_disk_previous_block_hash(
            &identity.model_hash,
            &namespace_hash,
            block_start,
            block_tokens.len(),
            prefix_tokens,
        )?;
        let block_hash = native_text_disk_block_hash(
            &identity.model_hash,
            &namespace_hash,
            &previous_block_hash,
            block_start,
            block_tokens,
        );
        Ok(Self {
            model_hash: identity.model_hash.clone(),
            snapshot_hash: identity.snapshot_hash.clone(),
            model_family: identity.model_family.clone(),
            namespace_hash,
            previous_block_hash,
            block_hash,
            block_start,
            token_count: block_tokens.len(),
            cache_layout_version: namespace.cache_layout_version,
        })
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
        let descriptor = match NativeTextDiskCacheBlockDescriptor::new(
            &self.identity,
            namespace,
            block_start,
            tokens,
        ) {
            Ok(descriptor) => descriptor,
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    model_hash = %self.identity.model_hash,
                    "failed to build native text SSD prefix cache block descriptor"
                );
                return NativeTextDiskCacheStoreStatus::Dropped;
            }
        };
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
                    previous_block_hash = %descriptor.previous_block_hash,
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
                    &tokens[..block_end],
                )?;
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
