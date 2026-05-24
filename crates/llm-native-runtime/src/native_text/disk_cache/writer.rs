#[cfg(test)]
use super::NativeTextDiskCacheStoreStatus;
use super::{
    NativeTextDiskCacheError, NativeTextDiskCacheIndex, NativeTextDiskCacheIndexKey,
    NativeTextDiskCacheIndexedEntry,
};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
use tokio::sync::mpsc;
#[cfg(test)]
use tokio::sync::oneshot;

#[derive(Debug, Clone)]
pub(crate) struct NativeTextDiskCacheWriter {
    tx: mpsc::Sender<NativeTextDiskCacheWriterMessage>,
}

impl NativeTextDiskCacheWriter {
    pub(super) fn spawn(index: NativeTextDiskCacheIndex, queue_depth: usize) -> Self {
        let (tx, rx) = mpsc::channel(queue_depth.max(1));
        tokio::spawn(native_text_disk_cache_writer_loop(rx, index));
        Self { tx }
    }

    #[cfg(test)]
    pub(super) fn try_enqueue(
        &self,
        job: NativeTextDiskCacheWriteJob,
    ) -> NativeTextDiskCacheStoreStatus {
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

    pub(super) fn try_reserve(&self) -> Option<mpsc::Permit<'_, NativeTextDiskCacheWriterMessage>> {
        self.tx.try_reserve().ok()
    }

    #[cfg(test)]
    pub(super) fn detached_for_test(
        queue_depth: usize,
    ) -> (Self, mpsc::Receiver<NativeTextDiskCacheWriterMessage>) {
        let (tx, rx) = mpsc::channel(queue_depth.max(1));
        (Self { tx }, rx)
    }

    #[cfg(test)]
    pub(super) async fn flush(&self) -> Result<(), NativeTextDiskCacheError> {
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
pub(super) enum NativeTextDiskCacheWriterMessage {
    Write(NativeTextDiskCacheWriteJob),
    #[cfg(test)]
    Flush(oneshot::Sender<()>),
}

#[derive(Debug, Clone)]
pub(crate) struct NativeTextDiskCacheWriteJob {
    pub(super) path: PathBuf,
    pub(super) bytes: Vec<u8>,
    pub(super) index: Option<NativeTextDiskCacheIndex>,
    pub(super) index_key: Option<NativeTextDiskCacheIndexKey>,
    pub(super) index_entry: Option<NativeTextDiskCacheIndexedEntry>,
}

impl NativeTextDiskCacheWriteJob {
    #[cfg(test)]
    pub(super) fn for_test(path: &str, bytes: Vec<u8>) -> Self {
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
