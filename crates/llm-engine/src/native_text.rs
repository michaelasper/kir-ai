use crate::{DEFAULT_NATIVE_QWEN_MAX_NEW_TOKENS, NativeQwenBackend, NativeQwenLoadOptions};
use async_trait::async_trait;
use futures::stream::BoxStream;
use llm_backend::{
    BackendError, BackendModelMetadata, BackendOutput, BackendRequest, BackendStreamChunk,
    ModelBackend,
};
use std::path::Path;
use tokio_util::sync::CancellationToken;

pub const DEFAULT_NATIVE_TEXT_MAX_NEW_TOKENS: u32 = DEFAULT_NATIVE_QWEN_MAX_NEW_TOKENS;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativeTextLoadOptions {
    pub qwen: NativeQwenLoadOptions,
}

impl NativeTextLoadOptions {
    pub fn with_qwen_options(qwen: NativeQwenLoadOptions) -> Self {
        Self { qwen }
    }
}

#[derive(Clone)]
pub struct NativeTextBackend {
    inner: NativeTextBackendInner,
}

#[derive(Clone)]
enum NativeTextBackendInner {
    Qwen(NativeQwenBackend),
}

impl NativeTextBackend {
    pub fn open(
        model_id: impl Into<String>,
        snapshot_path: impl AsRef<Path>,
    ) -> anyhow::Result<Self> {
        Self::open_with_options(model_id, snapshot_path, NativeTextLoadOptions::default())
    }

    pub fn open_with_options(
        model_id: impl Into<String>,
        snapshot_path: impl AsRef<Path>,
        options: NativeTextLoadOptions,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            inner: NativeTextBackendInner::Qwen(NativeQwenBackend::open_with_options(
                model_id,
                snapshot_path,
                options.qwen,
            )?),
        })
    }

    pub fn with_max_new_tokens(mut self, max_new_tokens: u32) -> Self {
        self.inner = match self.inner {
            NativeTextBackendInner::Qwen(backend) => {
                NativeTextBackendInner::Qwen(backend.with_max_new_tokens(max_new_tokens))
            }
        };
        self
    }

    pub fn with_max_prefill_tokens(mut self, max_prefill_tokens: usize) -> Self {
        self.inner = match self.inner {
            NativeTextBackendInner::Qwen(backend) => {
                NativeTextBackendInner::Qwen(backend.with_max_prefill_tokens(max_prefill_tokens))
            }
        };
        self
    }
}

#[async_trait]
impl ModelBackend for NativeTextBackend {
    fn model_id(&self) -> &str {
        match &self.inner {
            NativeTextBackendInner::Qwen(backend) => backend.model_id(),
        }
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        match &self.inner {
            NativeTextBackendInner::Qwen(backend) => backend.model_metadata(),
        }
    }

    async fn generate(&self, request: BackendRequest) -> Result<BackendOutput, BackendError> {
        match &self.inner {
            NativeTextBackendInner::Qwen(backend) => backend.generate(request).await,
        }
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        match &self.inner {
            NativeTextBackendInner::Qwen(backend) => {
                backend.generate_with_cancel(request, cancellation).await
            }
        }
    }

    fn generate_stream<'a>(
        &'a self,
        request: BackendRequest,
    ) -> BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        match &self.inner {
            NativeTextBackendInner::Qwen(backend) => backend.generate_stream(request),
        }
    }

    fn generate_stream_with_cancel<'a>(
        &'a self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        match &self.inner {
            NativeTextBackendInner::Qwen(backend) => {
                backend.generate_stream_with_cancel(request, cancellation)
            }
        }
    }
}
