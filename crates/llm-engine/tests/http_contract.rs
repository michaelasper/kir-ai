use async_trait::async_trait;
use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use futures::StreamExt;
use llm_backend::{
    BackendError, BackendFinishReason, BackendModelMetadata, BackendOutput, BackendRequest,
    BackendStreamChunk, BackendStreamProgress, BackendToolCallDelta, BackendToolCallFunctionDelta,
    BackendToolCallType, ModelBackend, ProtocolTestBackend,
};
use llm_engine::{EngineOptions, build_router, router_builder};
use llm_hub::{HubFile, HubRepoId, ModelProfile, ModelStore, build_download_plan};
use serde_json::{Value, json};
use std::{
    io::{Read, Write},
    net::TcpListener,
    path::{Path, PathBuf},
    sync::{Arc, Condvar, Mutex},
    thread,
    time::Duration,
};
use tokio::sync::{Notify, Semaphore, mpsc};
use tokio_util::sync::CancellationToken;
use tower::ServiceExt;

#[path = "http_contract/admin_contract.rs"]
mod admin_contract;
#[path = "http_contract/chat_contract.rs"]
mod chat_contract;
#[path = "http_contract/completion_contract.rs"]
mod completion_contract;
#[path = "http_contract/core_contract.rs"]
mod core_contract;
#[path = "http_contract/streaming_contract.rs"]
mod streaming_contract;

#[test]
fn public_router_builder_requires_explicit_backend() {
    let Err(err) = build_router() else {
        panic!("router builder without a backend must fail closed");
    };

    assert!(
        err.to_string().contains("explicit backend"),
        "error should explain how to provide a backend: {err}"
    );
}

struct FailingBackend;

fn qwen_test_metadata(model_id: &str, backend: &str) -> BackendModelMetadata {
    BackendModelMetadata::new(model_id, backend).with_family("qwen")
}

fn unauthenticated_admin_options() -> EngineOptions {
    EngineOptions::default()
}

fn build_router_with_unauthenticated_admin(backend: Box<dyn ModelBackend>) -> Router {
    build_router_with_unauthenticated_admin_and_options(backend, unauthenticated_admin_options())
        .expect("unauthenticated admin test router builds")
}

fn build_router_with_backend(backend: Box<dyn ModelBackend>) -> Router {
    router_builder(backend).build().expect("test router builds")
}

fn build_router_with_protocol_test_backend() -> Router {
    router_builder(Box::new(
        ProtocolTestBackend::new(
            llm_engine::DEFAULT_MODEL_ID,
            "hello from rust native backend",
        )
        .with_required_tool_protocol()
        .with_json_object_protocol(),
    ))
    .with_options(EngineOptions::default())
    .allow_unauthenticated_admin()
    .build()
    .expect("protocol test router builds")
}

fn build_router_with_backend_and_options(
    backend: Box<dyn ModelBackend>,
    options: EngineOptions,
) -> Result<Router, llm_engine::EngineConfigError> {
    router_builder(backend).with_options(options).build()
}

fn build_router_with_unauthenticated_admin_and_options(
    backend: Box<dyn ModelBackend>,
    options: EngineOptions,
) -> Result<Router, llm_engine::EngineConfigError> {
    build_router_with_backend_and_options_allowing_unauthenticated_admin(backend, options)
}

fn build_router_with_backend_and_options_allowing_unauthenticated_admin(
    backend: Box<dyn ModelBackend>,
    options: EngineOptions,
) -> Result<Router, llm_engine::EngineConfigError> {
    router_builder(backend)
        .with_options(options)
        .allow_unauthenticated_admin()
        .build()
}

#[async_trait]
impl ModelBackend for FailingBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "failing")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Err(BackendError::other("execution failed".to_owned()))
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }
}

#[test]
fn public_router_builders_with_backend_return_config_results() {
    let _: Result<Router, llm_engine::EngineConfigError> =
        llm_engine::router_builder(Box::new(FailingBackend)).build();
    let _: Result<Router, llm_engine::EngineConfigError> =
        llm_engine::router_builder(Box::new(FailingBackend))
            .with_concurrency(1)
            .build();
}

struct StaticBackend {
    text: String,
}

struct CachedUsageBackend {
    prompt_tokens: u64,
    prompt_cached_tokens: Option<u64>,
    completion_tokens: u64,
}

struct FamilyStaticBackend {
    model_id: &'static str,
    family: &'static str,
    text: &'static str,
}

#[async_trait]
impl ModelBackend for StaticBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "static")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Ok(BackendOutput {
            text: self.text.clone(),
            prompt_tokens: 1,
            prompt_cached_tokens: None,
            completion_tokens: 1,
            finish_reason: BackendFinishReason::Stop,
        })
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }
}

#[async_trait]
impl ModelBackend for CachedUsageBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "cached-usage")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Ok(BackendOutput {
            text: "cached response".to_owned(),
            prompt_tokens: self.prompt_tokens,
            prompt_cached_tokens: self.prompt_cached_tokens,
            completion_tokens: self.completion_tokens,
            finish_reason: BackendFinishReason::Stop,
        })
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }
}

#[async_trait]
impl ModelBackend for FamilyStaticBackend {
    fn model_id(&self) -> &str {
        self.model_id
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        BackendModelMetadata::new(self.model_id, "family-static").with_family(self.family)
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Ok(BackendOutput {
            text: self.text.to_owned(),
            prompt_tokens: 1,
            prompt_cached_tokens: None,
            completion_tokens: 1,
            finish_reason: BackendFinishReason::Stop,
        })
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }
}

struct ScriptedChatBackend;

#[async_trait]
impl ModelBackend for ScriptedChatBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "scripted-chat")
    }

    async fn generate(&self, request: BackendRequest) -> Result<BackendOutput, BackendError> {
        if request.model != self.model_id() {
            return Err(BackendError::model_not_found(
                request.model,
                self.model_id().to_owned(),
            ));
        }
        let text = scripted_chat_response(request.prompt());
        Ok(BackendOutput {
            prompt_tokens: test_token_count(request.prompt()),
            prompt_cached_tokens: None,
            completion_tokens: test_token_count(&text),
            text,
            finish_reason: BackendFinishReason::Stop,
        })
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }
}

struct BlockingBackend {
    entered: Arc<Notify>,
    release: Arc<Notify>,
}

#[async_trait]
impl ModelBackend for BlockingBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "blocking")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        self.entered.notify_waiters();
        self.release.notified().await;
        Ok(BackendOutput {
            text: "released".to_owned(),
            prompt_tokens: 1,
            prompt_cached_tokens: None,
            completion_tokens: 1,
            finish_reason: BackendFinishReason::Stop,
        })
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }
}

struct FairnessBackend {
    order: Arc<Mutex<Vec<String>>>,
    entered: Arc<Notify>,
    release: Arc<Semaphore>,
}

#[async_trait]
impl ModelBackend for FairnessBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "fairness")
    }

    async fn generate(&self, request: BackendRequest) -> Result<BackendOutput, BackendError> {
        let label = if request.prompt().contains("first-long") {
            "first-long"
        } else if request.prompt().contains("second-long") {
            "second-long"
        } else if request.prompt().contains("third-short") {
            "third-short"
        } else {
            "unknown"
        };
        self.order
            .lock()
            .expect("order lock is not poisoned")
            .push(label.to_owned());
        self.entered.notify_waiters();
        let _permit = self
            .release
            .acquire()
            .await
            .expect("release semaphore open");
        Ok(BackendOutput {
            text: label.to_owned(),
            prompt_tokens: 1,
            prompt_cached_tokens: None,
            completion_tokens: 1,
            finish_reason: BackendFinishReason::Stop,
        })
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }
}

struct AdminCancellableBackend {
    entered: Arc<Notify>,
    cancelled: Arc<Notify>,
}

#[async_trait]
impl ModelBackend for AdminCancellableBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "admin-cancellable")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Err(BackendError::other(
            "generate_with_cancel should be used".to_owned(),
        ))
    }

    async fn generate_with_cancel(
        &self,
        _request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        self.entered.notify_waiters();
        cancellation.cancelled().await;
        self.cancelled.notify_waiters();
        Err(BackendError::cancelled())
    }
}

struct AdminLateErrorBackend {
    entered: Arc<Notify>,
    release: Arc<Semaphore>,
}

#[async_trait]
impl ModelBackend for AdminLateErrorBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "admin-late-error")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Err(BackendError::other(
            "generate_with_cancel should be used".to_owned(),
        ))
    }

    async fn generate_with_cancel(
        &self,
        _request: BackendRequest,
        _cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        self.entered.notify_waiters();
        let _permit = self
            .release
            .acquire()
            .await
            .expect("release semaphore open");
        Err(BackendError::other("late backend failure".to_owned()))
    }
}

struct NoProgressBackend;

#[async_trait]
impl ModelBackend for NoProgressBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "no-progress")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Ok(BackendOutput {
            text: String::new(),
            prompt_tokens: 1,
            prompt_cached_tokens: None,
            completion_tokens: 4096,
            finish_reason: BackendFinishReason::Length,
        })
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }
}

struct DelayedStreamBackend {
    release: Arc<Semaphore>,
}

#[async_trait]
impl ModelBackend for DelayedStreamBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "delayed-stream")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        let _permit = self
            .release
            .acquire()
            .await
            .expect("release semaphore open");
        Ok(BackendOutput {
            text: "released".to_owned(),
            prompt_tokens: 1,
            prompt_cached_tokens: None,
            completion_tokens: 1,
            finish_reason: BackendFinishReason::Stop,
        })
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }
}

struct TwoStageStreamBackend {
    first: Arc<Notify>,
    finish: Arc<Notify>,
}

#[async_trait]
impl ModelBackend for TwoStageStreamBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "two-stage-stream")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        self.first.notified().await;
        self.finish.notified().await;
        Ok(BackendOutput {
            text: "first second".to_owned(),
            prompt_tokens: 1,
            prompt_cached_tokens: None,
            completion_tokens: 2,
            finish_reason: BackendFinishReason::Stop,
        })
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }

    fn generate_stream<'a>(
        &'a self,
        _request: BackendRequest,
    ) -> futures::stream::BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        let first = self.first.clone();
        let finish = self.finish.clone();
        async_stream::try_stream! {
            first.notified().await;
            yield BackendStreamChunk {
                text: "first".to_owned(),
                tool_call_deltas: Vec::new(),
                prompt_tokens: 1,
                prompt_cached_tokens: None,
                completion_tokens: 1,
                finish_reason: None,
                progress: None,
            };
            finish.notified().await;
            yield BackendStreamChunk {
                text: " second".to_owned(),
                tool_call_deltas: Vec::new(),
                prompt_tokens: 1,
                prompt_cached_tokens: None,
                completion_tokens: 1,
                finish_reason: Some(BackendFinishReason::Stop),
                progress: None,
            };
        }
        .boxed()
    }

    fn generate_stream_with_cancel<'a>(
        &'a self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> futures::stream::BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        if cancellation.is_cancelled() {
            return futures::stream::once(async { Err(BackendError::cancelled()) }).boxed();
        }
        self.generate_stream(request)
    }
}

struct CancellableStreamBackend {
    cancelled: Arc<Notify>,
}

#[async_trait]
impl ModelBackend for CancellableStreamBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "cancellable-stream")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Ok(BackendOutput {
            text: "first".to_owned(),
            prompt_tokens: 1,
            prompt_cached_tokens: None,
            completion_tokens: 1,
            finish_reason: BackendFinishReason::Stop,
        })
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }

    fn generate_stream_with_cancel<'a>(
        &'a self,
        _request: BackendRequest,
        cancellation: CancellationToken,
    ) -> futures::stream::BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        let cancelled = self.cancelled.clone();
        tokio::spawn(async move {
            cancellation.cancelled().await;
            cancelled.notify_waiters();
        });
        async_stream::try_stream! {
            yield BackendStreamChunk {
                text: "first".to_owned(),
                tool_call_deltas: Vec::new(),
                prompt_tokens: 1,
                prompt_cached_tokens: None,
                completion_tokens: 1,
                finish_reason: None,
                progress: None,
            };
            futures::future::pending::<()>().await;
        }
        .boxed()
    }
}

struct FailingStreamBackend;

#[async_trait]
impl ModelBackend for FailingStreamBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "failing-stream")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Ok(BackendOutput {
            text: "first".to_owned(),
            prompt_tokens: 1,
            prompt_cached_tokens: None,
            completion_tokens: 1,
            finish_reason: BackendFinishReason::Stop,
        })
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }

    fn generate_stream<'a>(
        &'a self,
        _request: BackendRequest,
    ) -> futures::stream::BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        async_stream::try_stream! {
            yield BackendStreamChunk {
                text: "first".to_owned(),
                tool_call_deltas: Vec::new(),
                prompt_tokens: 1,
                prompt_cached_tokens: None,
                completion_tokens: 1,
                finish_reason: None,
                progress: None,
            };
            Err(BackendError::other("stream failed".to_owned()))?;
        }
        .boxed()
    }

    fn generate_stream_with_cancel<'a>(
        &'a self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> futures::stream::BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        if cancellation.is_cancelled() {
            return futures::stream::once(async { Err(BackendError::cancelled()) }).boxed();
        }
        self.generate_stream(request)
    }
}

struct MetadataBackend;

#[async_trait]
impl ModelBackend for MetadataBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        let mut metadata = BackendModelMetadata::new(llm_engine::DEFAULT_MODEL_ID, "native-qwen")
            .with_family("qwen");
        metadata.quantization = Some("bf16".to_owned());
        metadata.repo_id = Some("Qwen/Qwen3.6-35B-A3B".to_owned());
        metadata.resolved_commit = Some("0123456789abcdef0123456789abcdef01234567".to_owned());
        metadata.profile = Some("qwen36-safetensors-bf16".to_owned());
        metadata
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Ok(BackendOutput {
            text: "metadata".to_owned(),
            prompt_tokens: 1,
            prompt_cached_tokens: None,
            completion_tokens: 1,
            finish_reason: BackendFinishReason::Stop,
        })
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }
}

struct MlxMetadataBackend;

#[async_trait]
impl ModelBackend for MlxMetadataBackend {
    fn model_id(&self) -> &str {
        "local-qwen36-mlx"
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        let mut metadata = BackendModelMetadata::new("local-qwen36-mlx", "mlx").with_family("qwen");
        metadata.quantization = Some("4bit".to_owned());
        metadata.repo_id = Some("mlx-community/Qwen3.6-35B-A3B-4bit".to_owned());
        metadata.resolved_commit = Some("0123456789abcdef0123456789abcdef01234567".to_owned());
        metadata.profile = Some("qwen36-mlx-4bit".to_owned());
        metadata
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Ok(BackendOutput {
            text: "mlx metadata".to_owned(),
            prompt_tokens: 1,
            prompt_cached_tokens: None,
            completion_tokens: 1,
            finish_reason: BackendFinishReason::Stop,
        })
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }
}

struct SnapshotMetadataBackend;

#[async_trait]
impl ModelBackend for SnapshotMetadataBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        let mut metadata = BackendModelMetadata::new(llm_engine::DEFAULT_MODEL_ID, "native-qwen")
            .with_family("qwen");
        metadata.quantization = Some("bf16".to_owned());
        metadata.repo_id = Some("Qwen/Qwen3.6-35B-A3B".to_owned());
        metadata.resolved_commit = Some("0123456789abcdef0123456789abcdef01234567".to_owned());
        metadata.profile = Some("qwen36-safetensors-bf16".to_owned());
        metadata
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Ok(BackendOutput {
            text: "metadata".to_owned(),
            prompt_tokens: 1,
            prompt_cached_tokens: None,
            completion_tokens: 1,
            finish_reason: BackendFinishReason::Stop,
        })
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }
}

async fn generate_after_pre_cancel<B: ModelBackend + ?Sized>(
    backend: &B,
    request: BackendRequest,
    cancellation: CancellationToken,
) -> Result<BackendOutput, BackendError> {
    if cancellation.is_cancelled() {
        return Err(BackendError::cancelled());
    }
    backend.generate(request).await
}

async fn write_verified_test_snapshot(root: &Path) -> PathBuf {
    write_verified_test_snapshot_with_profile(
        root,
        "Qwen/Qwen3.6-35B-A3B",
        ModelProfile::qwen36_safetensors_bf16(),
    )
    .await
}

async fn write_verified_mlx_test_snapshot(root: &Path) -> PathBuf {
    write_verified_test_snapshot_with_profile(
        root,
        "mlx-community/Qwen3.6-35B-A3B-4bit",
        ModelProfile::qwen36_mlx_4bit(),
    )
    .await
}

async fn write_verified_test_snapshot_with_profile(
    root: &Path,
    repo_id: &str,
    profile: ModelProfile,
) -> PathBuf {
    let store = ModelStore::new(root);
    let plan = build_download_plan(
        HubRepoId::model(repo_id).expect("repo id"),
        "main",
        "0123456789abcdef0123456789abcdef01234567",
        profile,
        vec![HubFile::new("config.json", 2, Some("\"cfg\""))],
        &[],
    )
    .expect("plan builds");
    let snapshot_path = store.snapshot_path(&plan);
    tokio::fs::create_dir_all(&snapshot_path)
        .await
        .expect("snapshot dir");
    tokio::fs::write(snapshot_path.join("config.json"), "{}")
        .await
        .expect("config");
    store
        .verify_existing_snapshot(&plan)
        .await
        .expect("snapshot verifies");
    snapshot_path
}

fn spawn_fake_hub_server(requests: usize) -> (String, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind fake hub");
    let endpoint = format!("http://{}", listener.local_addr().expect("local addr"));
    let server = thread::spawn(move || {
        for _ in 0..requests {
            let (mut stream, _) = listener.accept().expect("accept fake hub request");
            let mut buffer = [0_u8; 4096];
            let read = stream.read(&mut buffer).expect("read fake hub request");
            let request = String::from_utf8_lossy(&buffer[..read]);
            if request.starts_with(
                "GET /api/models/Qwen/Qwen3.6-35B-A3B/revision/main?blobs=true&securityStatus=true ",
            ) {
                let body = json!({
                    "id": "Qwen/Qwen3.6-35B-A3B",
                    "sha": "0123456789abcdef0123456789abcdef01234567",
                    "siblings": [
                        {"rfilename": "config.json", "size": 2, "blobId": "\"cfg\""},
                        {"rfilename": "model.safetensors", "size": 4, "blobId": "\"weights\""}
                    ]
                })
                .to_string();
                write_http_response(&mut stream, "200 OK", &body);
            } else if request.starts_with(
                "GET /Qwen/Qwen3.6-35B-A3B/resolve/0123456789abcdef0123456789abcdef01234567/config.json ",
            ) {
                write_http_response(&mut stream, "200 OK", "{}");
            } else {
                write_http_response(&mut stream, "404 Not Found", "not found");
            }
        }
    });
    (endpoint, server)
}

struct BlockingFakeHubServer {
    endpoint: String,
    server: thread::JoinHandle<()>,
    download_started: mpsc::UnboundedReceiver<String>,
    release_download: Arc<(Mutex<usize>, Condvar)>,
    max_active_downloads: Arc<Mutex<usize>>,
}

impl BlockingFakeHubServer {
    fn release_one_download(&self) {
        let (release_count, release) = &*self.release_download;
        let mut release_count = release_count.lock().expect("release lock");
        *release_count += 1;
        release.notify_one();
    }

    fn max_active_downloads(&self) -> usize {
        *self
            .max_active_downloads
            .lock()
            .expect("active download lock")
    }
}

fn spawn_blocking_fake_hub_server() -> BlockingFakeHubServer {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind fake hub");
    let endpoint = format!("http://{}", listener.local_addr().expect("local addr"));
    let (download_started, download_started_rx) = mpsc::unbounded_channel();
    let release_download = Arc::new((Mutex::new(0_usize), Condvar::new()));
    let active_downloads = Arc::new(Mutex::new(0_usize));
    let max_active_downloads = Arc::new(Mutex::new(0_usize));
    let server_release_download = release_download.clone();
    let server_active_downloads = active_downloads.clone();
    let server_max_active_downloads = max_active_downloads.clone();
    let server = thread::spawn(move || {
        let mut handlers = Vec::new();
        for _ in 0..4 {
            let (stream, _) = listener.accept().expect("accept fake hub request");
            let handler_download_started = download_started.clone();
            let handler_release_download = server_release_download.clone();
            let handler_active_downloads = server_active_downloads.clone();
            let handler_max_active_downloads = server_max_active_downloads.clone();
            handlers.push(thread::spawn(move || {
                handle_blocking_fake_hub_request(
                    stream,
                    handler_download_started,
                    handler_release_download,
                    handler_active_downloads,
                    handler_max_active_downloads,
                );
            }));
        }
        drop(download_started);
        for handler in handlers {
            handler.join().expect("fake hub handler exits");
        }
    });

    BlockingFakeHubServer {
        endpoint,
        server,
        download_started: download_started_rx,
        release_download,
        max_active_downloads,
    }
}

fn handle_blocking_fake_hub_request(
    mut stream: std::net::TcpStream,
    download_started: mpsc::UnboundedSender<String>,
    release_download: Arc<(Mutex<usize>, Condvar)>,
    active_downloads: Arc<Mutex<usize>>,
    max_active_downloads: Arc<Mutex<usize>>,
) {
    let mut buffer = [0_u8; 4096];
    let read = stream.read(&mut buffer).expect("read fake hub request");
    let request = String::from_utf8_lossy(&buffer[..read]);
    let Some(repo_id) = blocking_fake_hub_repo_id(&request) else {
        write_http_response(&mut stream, "404 Not Found", "not found");
        return;
    };
    if request.starts_with(&format!(
        "GET /api/models/{repo_id}/revision/main?blobs=true&securityStatus=true "
    )) {
        let body = json!({
            "id": repo_id,
            "sha": "0123456789abcdef0123456789abcdef01234567",
            "siblings": [
                {"rfilename": "config.json", "size": 2, "blobId": "\"cfg\""},
                {"rfilename": "model.safetensors", "size": 4, "blobId": "\"weights\""}
            ]
        })
        .to_string();
        write_http_response(&mut stream, "200 OK", &body);
        return;
    }
    if request.starts_with(&format!(
        "GET /{repo_id}/resolve/0123456789abcdef0123456789abcdef01234567/config.json "
    )) {
        record_blocking_fake_hub_download_start(
            &repo_id,
            &download_started,
            &active_downloads,
            &max_active_downloads,
        );
        wait_for_blocking_fake_hub_release(&release_download);
        write_http_response(&mut stream, "200 OK", "{}");
        let mut active_downloads = active_downloads.lock().expect("active download lock");
        *active_downloads -= 1;
        return;
    }
    write_http_response(&mut stream, "404 Not Found", "not found");
}

fn blocking_fake_hub_repo_id(request: &str) -> Option<String> {
    for repo_id in ["TestOrg/FirstModel", "TestOrg/SecondModel"] {
        if request.contains(repo_id) {
            return Some(repo_id.to_owned());
        }
    }
    None
}

fn record_blocking_fake_hub_download_start(
    repo_id: &str,
    download_started: &mpsc::UnboundedSender<String>,
    active_downloads: &Arc<Mutex<usize>>,
    max_active_downloads: &Arc<Mutex<usize>>,
) {
    let active_download_count = {
        let mut active_downloads = active_downloads.lock().expect("active download lock");
        *active_downloads += 1;
        *active_downloads
    };
    let mut max_active_downloads = max_active_downloads.lock().expect("active download lock");
    *max_active_downloads = (*max_active_downloads).max(active_download_count);
    download_started
        .send(repo_id.to_owned())
        .expect("download started receiver is open");
}

fn wait_for_blocking_fake_hub_release(release_download: &Arc<(Mutex<usize>, Condvar)>) {
    let (release_count, release) = &**release_download;
    let mut release_count = release_count.lock().expect("release lock");
    while *release_count == 0 {
        release_count = release.wait(release_count).expect("release lock");
    }
    *release_count -= 1;
}

fn write_http_response(stream: &mut std::net::TcpStream, status: &str, body: &str) {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .expect("write fake hub response");
    stream.flush().expect("flush fake hub response");
}

async fn protocol_chat_content(messages: Value) -> String {
    let response = build_router_with_backend(Box::new(ScriptedChatBackend))
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": llm_engine::DEFAULT_MODEL_ID,
                        "messages": messages
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("chat response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response.into_body()).await;
    body["choices"][0]["message"]["content"]
        .as_str()
        .expect("assistant content")
        .to_owned()
}

async fn protocol_test_chat_content(messages: Value) -> String {
    let response = build_router_with_protocol_test_backend()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": llm_engine::DEFAULT_MODEL_ID,
                        "messages": messages
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("chat response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response.into_body()).await;
    body["choices"][0]["message"]["content"]
        .as_str()
        .expect("assistant content")
        .to_owned()
}

fn scripted_chat_response(prompt: &str) -> String {
    let current = last_user_message(prompt).to_ascii_lowercase();
    let prompt = prompt.to_ascii_lowercase();
    if prompt.contains("miso") {
        if current.contains("memory check") || current.contains("dog's name") {
            return "The dog's name is Miso.".to_owned();
        }
        if current.contains("bedtime") || current.contains("quiet") {
            return "Bedtime version: Miso curled beside the blue sock on the porch, listened to the moon, and fell asleep knowing the house was kind.".to_owned();
        }
        if current.contains("bullet") || current.contains("explain") {
            return "- I kept Miso as the shy dog so the thread has continuity.\n- I added the blue sock and porch so the story has concrete details.".to_owned();
        }
        if current.contains("specific") || current.contains("toy") || current.contains("place") {
            return "Miso carried a blue sock to the porch, peeked at the rain, and wagged when a child sat beside him.".to_owned();
        }
        if current.contains("story") {
            return "Miso was a shy little dog who hid behind a chair until a kind child offered a quiet hello.".to_owned();
        }
    }
    if current.contains("memory check") && prompt.contains("brave hearts") {
        "The avoided phrase was \"brave hearts.\"".to_owned()
    } else if current.contains("explain") && prompt.contains("brave hearts") {
        "The revision changed the one-sentence image into short lines, replaced brave hearts with paws and tails, and made the ending warmer.".to_owned()
    } else if current.contains("bedtime") && prompt.contains("dog") {
        "Bedtime version:\nSoft paws settle by the bed,\nSleepy tails make one last sweep,\nWarm noses rest near open hands,\nDogs turn the quiet house to sleep.".to_owned()
    } else if (current.contains("rewrite")
        || current.contains("revise")
        || current.contains("revised"))
        && prompt.contains("feedback")
    {
        "Revised poem:\nPaws tap softly by the door,\nTails sweep circles on the floor,\nWarm noses nudge the evening in,\nHome begins where dogs have been.".to_owned()
    } else if current.contains("critique") && current.contains("feedback") {
        "Feedback: The dog poem has clear motion; add sharper images and a stronger final line."
            .to_owned()
    } else if current.contains("poem") && current.contains("dog") {
        "Dogs flash through rain-wet grass, brave hearts chasing the sun.".to_owned()
    } else {
        "Unsupported scripted chat test prompt.".to_owned()
    }
}

fn last_user_message(prompt: &str) -> String {
    const USER_START: &str = "<|im_start|>user\n";
    let Some(start) = prompt.rfind(USER_START) else {
        return prompt.to_owned();
    };
    let body_start = start + USER_START.len();
    let body = &prompt[body_start..];
    let end = body.find("<|im_end|>").unwrap_or(body.len());
    body[..end].to_owned()
}

fn test_token_count(text: &str) -> u64 {
    text.split_whitespace().count().max(1) as u64
}

async fn body_json(body: Body) -> Value {
    let bytes = to_bytes(body, usize::MAX).await.expect("body bytes");
    serde_json::from_slice(&bytes).expect("json body")
}

async fn body_text(body: Body) -> String {
    let bytes = to_bytes(body, usize::MAX).await.expect("body bytes");
    String::from_utf8(bytes.to_vec()).expect("utf8 body")
}

fn chat_request_body(content: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": llm_engine::DEFAULT_MODEL_ID,
                "messages": [{"role": "user", "content": content}]
            })
            .to_string(),
        ))
        .expect("request builds")
}

fn chat_request_body_with_id(content: &str, request_id: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("x-request-id", request_id)
        .body(Body::from(
            json!({
                "model": llm_engine::DEFAULT_MODEL_ID,
                "messages": [{"role": "user", "content": content}]
            })
            .to_string(),
        ))
        .expect("request builds")
}

fn completion_request_body_with_id(prompt: &str, request_id: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .header("x-request-id", request_id)
        .body(Body::from(
            json!({
                "model": llm_engine::DEFAULT_MODEL_ID,
                "prompt": prompt
            })
            .to_string(),
        ))
        .expect("request builds")
}
