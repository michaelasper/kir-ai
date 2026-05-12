use async_trait::async_trait;
use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use futures::StreamExt;
use llm_backend::{
    BackendError, BackendModelMetadata, BackendOutput, BackendRequest, BackendStreamChunk,
    ModelBackend,
};
use llm_engine::{
    EngineOptions, build_router, build_router_with_backend, build_router_with_backend_and_options,
    build_router_with_backend_and_options_allowing_unauthenticated_admin,
    build_router_with_protocol_test_backend,
};
use llm_hub::{HubFile, HubRepoId, ModelProfile, ModelStore, build_download_plan};
use serde_json::{Value, json};
use std::{
    io::{Read, Write},
    net::TcpListener,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};
use tokio::sync::{Notify, Semaphore};
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

fn build_router_with_unauthenticated_admin_and_options(
    backend: Box<dyn ModelBackend>,
    options: EngineOptions,
) -> Result<Router, llm_engine::EngineConfigError> {
    build_router_with_backend_and_options_allowing_unauthenticated_admin(backend, options)
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
        Err(BackendError::Other("execution failed".to_owned()))
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }
}

struct StaticBackend {
    text: String,
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
            completion_tokens: 1,
            finish_reason: llm_api::FinishReason::Stop,
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
            completion_tokens: 1,
            finish_reason: llm_api::FinishReason::Stop,
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
            return Err(BackendError::ModelNotFound {
                requested: request.model,
                available: self.model_id().to_owned(),
            });
        }
        let text = scripted_chat_response(&request.prompt);
        Ok(BackendOutput {
            prompt_tokens: test_token_count(&request.prompt),
            completion_tokens: test_token_count(&text),
            text,
            finish_reason: llm_api::FinishReason::Stop,
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
            completion_tokens: 1,
            finish_reason: llm_api::FinishReason::Stop,
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
        let label = if request.prompt.contains("first-long") {
            "first-long"
        } else if request.prompt.contains("second-long") {
            "second-long"
        } else if request.prompt.contains("third-short") {
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
            completion_tokens: 1,
            finish_reason: llm_api::FinishReason::Stop,
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
        Err(BackendError::Other(
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
        Err(BackendError::Cancelled)
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
        Err(BackendError::Other(
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
        Err(BackendError::Other("late backend failure".to_owned()))
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
            completion_tokens: 4096,
            finish_reason: llm_api::FinishReason::Length,
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
            completion_tokens: 1,
            finish_reason: llm_api::FinishReason::Stop,
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
            completion_tokens: 2,
            finish_reason: llm_api::FinishReason::Stop,
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
                prompt_tokens: 1,
                completion_tokens: 1,
                finish_reason: None,
            };
            finish.notified().await;
            yield BackendStreamChunk {
                text: " second".to_owned(),
                prompt_tokens: 1,
                completion_tokens: 1,
                finish_reason: Some(llm_api::FinishReason::Stop),
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
            return futures::stream::once(async { Err(BackendError::Cancelled) }).boxed();
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
            completion_tokens: 1,
            finish_reason: llm_api::FinishReason::Stop,
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
                prompt_tokens: 1,
                completion_tokens: 1,
                finish_reason: None,
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
            completion_tokens: 1,
            finish_reason: llm_api::FinishReason::Stop,
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
                prompt_tokens: 1,
                completion_tokens: 1,
                finish_reason: None,
            };
            Err(BackendError::Other("stream failed".to_owned()))?;
        }
        .boxed()
    }

    fn generate_stream_with_cancel<'a>(
        &'a self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> futures::stream::BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        if cancellation.is_cancelled() {
            return futures::stream::once(async { Err(BackendError::Cancelled) }).boxed();
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
        BackendModelMetadata {
            id: llm_engine::DEFAULT_MODEL_ID.to_owned(),
            backend: "native-qwen".to_owned(),
            family: Some("qwen".to_owned()),
            loader: Some("native-metal".to_owned()),
            quantization: Some("bf16".to_owned()),
            repo_id: Some("Qwen/Qwen3.6-35B-A3B".to_owned()),
            resolved_commit: Some("0123456789abcdef0123456789abcdef01234567".to_owned()),
            profile: Some("qwen36-safetensors-bf16".to_owned()),
            snapshot_path: Some(std::path::PathBuf::from(format!(
                "/tmp/{}",
                llm_engine::DEFAULT_MODEL_ID
            ))),
            manifest_digest: Some(
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
            ),
        }
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Ok(BackendOutput {
            text: "metadata".to_owned(),
            prompt_tokens: 1,
            completion_tokens: 1,
            finish_reason: llm_api::FinishReason::Stop,
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
        BackendModelMetadata {
            id: "local-qwen36-mlx".to_owned(),
            backend: "mlx".to_owned(),
            family: Some("qwen".to_owned()),
            loader: Some("mlx".to_owned()),
            quantization: Some("4bit".to_owned()),
            repo_id: Some("mlx-community/Qwen3.6-35B-A3B-4bit".to_owned()),
            resolved_commit: Some("0123456789abcdef0123456789abcdef01234567".to_owned()),
            profile: Some("qwen36-mlx-4bit".to_owned()),
            snapshot_path: Some(std::path::PathBuf::from("/tmp/local-qwen36-mlx")),
            manifest_digest: Some(
                "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_owned(),
            ),
        }
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Ok(BackendOutput {
            text: "mlx metadata".to_owned(),
            prompt_tokens: 1,
            completion_tokens: 1,
            finish_reason: llm_api::FinishReason::Stop,
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

struct SnapshotMetadataBackend {
    snapshot_path: PathBuf,
}

#[async_trait]
impl ModelBackend for SnapshotMetadataBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        BackendModelMetadata {
            id: llm_engine::DEFAULT_MODEL_ID.to_owned(),
            backend: "native-qwen".to_owned(),
            family: Some("qwen".to_owned()),
            loader: Some("native-metal".to_owned()),
            quantization: Some("bf16".to_owned()),
            repo_id: Some("Qwen/Qwen3.6-35B-A3B".to_owned()),
            resolved_commit: Some("0123456789abcdef0123456789abcdef01234567".to_owned()),
            profile: Some("qwen36-safetensors-bf16".to_owned()),
            snapshot_path: Some(self.snapshot_path.clone()),
            manifest_digest: None,
        }
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Ok(BackendOutput {
            text: "metadata".to_owned(),
            prompt_tokens: 1,
            completion_tokens: 1,
            finish_reason: llm_api::FinishReason::Stop,
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
        return Err(BackendError::Cancelled);
    }
    backend.generate(request).await
}

async fn write_verified_test_snapshot(root: &Path) -> PathBuf {
    let store = ModelStore::new(root);
    let plan = build_download_plan(
        HubRepoId::model("Qwen/Qwen3.6-35B-A3B").expect("repo id"),
        "main",
        "0123456789abcdef0123456789abcdef01234567",
        ModelProfile::qwen36_safetensors_bf16(),
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
