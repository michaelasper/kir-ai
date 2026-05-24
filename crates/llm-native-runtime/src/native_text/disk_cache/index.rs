use super::{
    NativeTextDiskCacheBlock, NativeTextDiskCacheBlockDescriptor, NativeTextDiskCacheConfig,
    NativeTextDiskCacheError, NativeTextDiskCacheIdentity,
};
use crate::sync_ext::FailPoisonedMutex;
use llm_backend::native::QwenLayerCache;
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, Mutex},
};

#[derive(Debug, Clone, Default)]
pub(super) struct NativeTextDiskCacheIndex {
    inner: Arc<Mutex<HashMap<NativeTextDiskCacheIndexKey, NativeTextDiskCacheIndexedEntry>>>,
}

impl NativeTextDiskCacheIndex {
    pub(super) fn insert(
        &self,
        key: NativeTextDiskCacheIndexKey,
        entry: NativeTextDiskCacheIndexedEntry,
    ) {
        self.inner
            .lock_or_panic("native text disk cache index")
            .insert(key, entry);
    }

    pub(super) fn get(
        &self,
        key: &NativeTextDiskCacheIndexKey,
    ) -> Option<NativeTextDiskCacheIndexedEntry> {
        self.inner
            .lock_or_panic("native text disk cache index")
            .get(key)
            .cloned()
    }

    pub(super) fn remove(&self, key: &NativeTextDiskCacheIndexKey) {
        self.inner
            .lock_or_panic("native text disk cache index")
            .remove(key);
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.inner
            .lock_or_panic("native text disk cache index")
            .len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct NativeTextDiskCacheIndexKey {
    pub(super) namespace_hash: String,
    pub(super) previous_block_hash: String,
    pub(super) block_start: usize,
    pub(super) block_hash: String,
}

impl NativeTextDiskCacheIndexKey {
    pub(super) fn from_descriptor(descriptor: &NativeTextDiskCacheBlockDescriptor) -> Self {
        Self {
            namespace_hash: descriptor.namespace_hash.clone(),
            previous_block_hash: descriptor.previous_block_hash.clone(),
            block_start: descriptor.block_start,
            block_hash: descriptor.block_hash.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct NativeTextDiskCacheIndexedEntry {
    pub(super) path: PathBuf,
    pub(super) block_start: usize,
    pub(super) token_count: usize,
}

pub(super) async fn reindex_disk_cache_root(
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
