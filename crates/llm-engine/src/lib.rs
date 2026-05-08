use async_trait::async_trait;
use axum::{
    Json, Router,
    extract::{Path as AxumPath, State, rejection::JsonRejection},
    http::{HeaderMap, StatusCode, header},
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::{get, post},
};
use futures::stream;
use llm_api::{
    ApiError, ChatCompletionRequest, CompletionRequest, FinishReason, ModelCard, ModelList, Usage,
};
use llm_backend::{
    BackendError, BackendModelMetadata, BackendOutput, BackendRequest, DeterministicBackend,
    ModelBackend, SafeTensorShardStore, qwen_final_norm, qwen_lm_head_top_k, qwen_prefill_sequence,
};
use llm_hub::{HubError, ModelStore, SnapshotManifest};
use llm_models::QwenModelSpec;
use llm_runtime::{Runtime, RuntimeError};
use llm_telemetry::{ServerMetrics, TokenCounters};
use llm_tokenizer::HuggingFaceTokenizer;
use serde_json::{Value, json};
use std::{
    convert::Infallible,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

type EngineRuntime = Runtime<Box<dyn ModelBackend>>;

#[derive(Clone)]
struct AppState {
    runtime: Arc<EngineRuntime>,
    metrics: Arc<Mutex<ServerMetrics>>,
    model_permits: Arc<Semaphore>,
    admin_token: Option<Arc<str>>,
}

#[derive(Debug, Clone, Default)]
pub struct EngineOptions {
    pub concurrency_limit: usize,
    pub admin_token: Option<String>,
}

pub fn build_router() -> Router {
    build_router_with_backend(Box::new(default_backend()))
}

pub fn build_router_with_backend(backend: Box<dyn ModelBackend>) -> Router {
    build_router_with_backend_and_concurrency(backend, 1)
}

pub fn build_router_with_backend_and_concurrency(
    backend: Box<dyn ModelBackend>,
    concurrency_limit: usize,
) -> Router {
    build_router_with_backend_and_options(
        backend,
        EngineOptions {
            concurrency_limit,
            admin_token: None,
        },
    )
}

pub fn build_router_with_backend_and_options(
    backend: Box<dyn ModelBackend>,
    options: EngineOptions,
) -> Router {
    let runtime = Runtime::new(backend);
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/admin/models", get(admin_models))
        .route("/admin/models/{alias}", get(admin_model))
        .route("/admin/models/{alias}/verify", post(admin_model_verify))
        .route("/admin/metrics", get(admin_metrics))
        .route(
            "/v1/chat/completions",
            axum::routing::post(chat_completions),
        )
        .route("/v1/completions", axum::routing::post(completions))
        .with_state(AppState {
            runtime: Arc::new(runtime),
            metrics: Arc::new(Mutex::new(ServerMetrics::default())),
            model_permits: Arc::new(Semaphore::new(options.concurrency_limit.max(1))),
            admin_token: options.admin_token.map(Arc::from),
        })
}

fn default_backend() -> DeterministicBackend {
    DeterministicBackend::new("local-qwen36", "hello from rust native backend")
}

#[derive(Clone)]
pub struct NativeQwenBackend {
    model_id: String,
    metadata: BackendModelMetadata,
    tokenizer: HuggingFaceTokenizer,
    spec: QwenModelSpec,
    store: SafeTensorShardStore,
    max_new_tokens: u32,
    max_prefill_tokens: usize,
    top_k: usize,
    chunk_rows: usize,
}

impl NativeQwenBackend {
    pub fn open(
        model_id: impl Into<String>,
        snapshot_path: impl AsRef<Path>,
    ) -> anyhow::Result<Self> {
        let model_id = model_id.into();
        let snapshot_path = snapshot_path.as_ref();
        let config_json = std::fs::read_to_string(snapshot_path.join("config.json"))?;
        let metadata = native_qwen_metadata(&model_id, snapshot_path)?;
        Ok(Self {
            model_id,
            metadata,
            tokenizer: HuggingFaceTokenizer::from_file(snapshot_path.join("tokenizer.json"))?,
            spec: QwenModelSpec::from_config_json(&config_json)?,
            store: SafeTensorShardStore::open(snapshot_path)?,
            max_new_tokens: 1,
            max_prefill_tokens: 32,
            top_k: 16,
            chunk_rows: 2048,
        })
    }

    pub fn with_max_new_tokens(mut self, max_new_tokens: u32) -> Self {
        self.max_new_tokens = max_new_tokens.max(1);
        self
    }

    pub fn with_max_prefill_tokens(mut self, max_prefill_tokens: usize) -> Self {
        self.max_prefill_tokens = max_prefill_tokens.max(1);
        self
    }

    fn generate_blocking(&self, request: BackendRequest) -> Result<BackendOutput, BackendError> {
        if request.model != self.model_id {
            return Err(BackendError::ModelNotFound {
                requested: request.model,
                available: self.model_id.clone(),
            });
        }
        let prompt_tokens = self
            .tokenizer
            .encode(&request.prompt, false)
            .map_err(|err| BackendError::Other(err.to_string()))?;
        let mut context_tokens = prompt_tokens
            .iter()
            .map(|token| *token as usize)
            .collect::<Vec<_>>();
        if context_tokens.is_empty() {
            return Err(BackendError::Other(
                "Qwen prompt encoded to zero tokens".to_owned(),
            ));
        }
        let mut output_ids = Vec::new();
        let mut finish_reason = FinishReason::Length;
        let eos_id = self
            .tokenizer
            .token_to_id("<|im_end|>")
            .map(|id| id as usize);
        let requested = resolve_native_max_tokens(request.max_tokens, self.max_new_tokens)?;

        for _ in 0..requested {
            let candidate = self.next_token(&context_tokens)?;
            context_tokens.push(candidate.token_id);
            if Some(candidate.token_id) == eos_id {
                finish_reason = FinishReason::Stop;
                break;
            }
            output_ids.push(u32::try_from(candidate.token_id).map_err(|err| {
                BackendError::Other(format!("Qwen token id does not fit u32: {err}"))
            })?);
        }

        let text = self
            .tokenizer
            .decode(&output_ids, false)
            .map_err(|err| BackendError::Other(err.to_string()))?;
        Ok(BackendOutput {
            text,
            prompt_tokens: prompt_tokens.len() as u64,
            completion_tokens: output_ids.len() as u64,
            finish_reason,
        })
    }

    fn next_token(&self, context_tokens: &[usize]) -> Result<NativeQwenCandidate, BackendError> {
        let start = context_tokens.len().saturating_sub(self.max_prefill_tokens);
        let hidden_states =
            qwen_prefill_sequence(&self.store, &self.spec, &context_tokens[start..])
                .map_err(|err| BackendError::Other(err.to_string()))?;
        let hidden = hidden_states.last().ok_or_else(|| {
            BackendError::Other("Qwen prefill returned no hidden states".to_owned())
        })?;
        let final_norm = qwen_final_norm(
            &self.store,
            hidden,
            self.spec.hidden_size as usize,
            self.spec.rms_norm_eps,
        )
        .map_err(|err| BackendError::Other(err.to_string()))?;
        let top_logits = qwen_lm_head_top_k(&self.store, &final_norm, self.top_k, self.chunk_rows)
            .map_err(|err| BackendError::Other(err.to_string()))?;

        let mut fallback = None;
        for item in top_logits {
            let token_id = u32::try_from(item.index).map_err(|err| {
                BackendError::Other(format!("Qwen token id does not fit u32: {err}"))
            })?;
            let text = self
                .tokenizer
                .decode(&[token_id], false)
                .map_err(|err| BackendError::Other(err.to_string()))?;
            let candidate = NativeQwenCandidate {
                token_id: item.index,
                text,
            };
            if fallback.is_none() {
                fallback = Some(candidate.clone());
            }
            if !candidate.text.trim().is_empty() {
                return Ok(candidate);
            }
        }
        fallback.ok_or_else(|| BackendError::Other("Qwen lm head returned no logits".to_owned()))
    }
}

fn resolve_native_max_tokens(
    requested: Option<u32>,
    configured_max: u32,
) -> Result<u32, BackendError> {
    match requested {
        None => Ok(configured_max),
        Some(0) => Err(BackendError::UnsupportedRequest(
            "max_tokens must be greater than 0".to_owned(),
        )),
        Some(value) if value > configured_max => Err(BackendError::UnsupportedRequest(format!(
            "requested max_tokens {value} exceeds native Qwen max_new_tokens {configured_max}"
        ))),
        Some(value) => Ok(value),
    }
}

#[derive(Debug, Clone)]
struct NativeQwenCandidate {
    token_id: usize,
    text: String,
}

#[async_trait]
impl ModelBackend for NativeQwenBackend {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        self.metadata.clone()
    }

    async fn generate(&self, request: BackendRequest) -> Result<BackendOutput, BackendError> {
        let backend = self.clone();
        tokio::task::spawn_blocking(move || backend.generate_blocking(request))
            .await
            .map_err(|err| BackendError::Other(format!("native Qwen worker failed: {err}")))?
    }
}

async fn health() -> impl IntoResponse {
    Json(json!({
        "status": "ok",
        "runtime": "rust",
        "python_runtime": false
    }))
}

async fn models(State(state): State<AppState>) -> impl IntoResponse {
    Json(ModelList {
        object: "list".to_owned(),
        data: vec![ModelCard {
            id: state.runtime.model_id().to_owned(),
            object: "model".to_owned(),
            owned_by: "local".to_owned(),
        }],
    })
}

async fn admin_models(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Value>, EngineError> {
    require_admin(&state, &headers)?;
    let metadata = state.runtime.model_metadata();
    Ok(Json(json!({
        "object": "list",
        "data": [admin_model_status(&metadata)],
    })))
}

async fn admin_model(
    AxumPath(alias): AxumPath<String>,
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Value>, EngineError> {
    require_admin(&state, &headers)?;
    let metadata = state.runtime.model_metadata();
    if alias != metadata.id {
        return Err(RuntimeError::Backend(BackendError::ModelNotFound {
            requested: alias,
            available: metadata.id,
        })
        .into());
    }
    Ok(Json(admin_model_status(&metadata)))
}

async fn admin_model_verify(
    AxumPath(alias): AxumPath<String>,
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Value>, EngineError> {
    require_admin(&state, &headers)?;
    let metadata = state.runtime.model_metadata();
    if alias != metadata.id {
        return Err(RuntimeError::Backend(BackendError::ModelNotFound {
            requested: alias,
            available: metadata.id,
        })
        .into());
    }
    let snapshot_path = metadata.snapshot_path.ok_or_else(|| {
        RuntimeError::Api(ApiError::unsupported_capability(
            "model verification requires snapshot metadata",
        ))
    })?;
    let verification = ModelStore::verify_snapshot(&snapshot_path)
        .await
        .map_err(EngineError::ModelStore)?;
    Ok(Json(json!({
        "status": "ok",
        "snapshot_path": verification.snapshot.path,
        "repo_id": verification.snapshot.manifest.repo_id,
        "resolved_commit": verification.snapshot.manifest.resolved_commit,
        "manifest_digest": verification.snapshot.manifest_digest,
        "verified_files": verification.verified_files,
        "verified_bytes": verification.verified_bytes,
    })))
}

fn admin_model_status(metadata: &BackendModelMetadata) -> Value {
    json!({
        "id": metadata.id,
        "object": "admin.model",
        "status": "ready",
        "runtime": "rust",
        "python_runtime": false,
        "backend": metadata.backend,
        "family": metadata.family,
        "loader": metadata.loader,
        "quantization": metadata.quantization,
        "repo_id": metadata.repo_id,
        "resolved_commit": metadata.resolved_commit,
        "profile": metadata.profile,
        "snapshot_path": metadata.snapshot_path,
        "manifest_digest": metadata.manifest_digest,
    })
}

fn native_qwen_metadata(
    model_id: &str,
    snapshot_path: &Path,
) -> anyhow::Result<BackendModelMetadata> {
    let manifest_path = snapshot_path.join("llm-engine-manifest.json");
    let manifest = serde_json::from_slice::<SnapshotManifest>(&std::fs::read(&manifest_path)?)?;
    let mut metadata = BackendModelMetadata::new(model_id.to_owned(), "native-qwen");
    metadata.family = Some(manifest.family.clone());
    metadata.loader = Some(manifest.loader.clone());
    metadata.quantization = Some(manifest.quantization.clone());
    metadata.repo_id = Some(manifest.repo_id.clone());
    metadata.resolved_commit = Some(manifest.resolved_commit.clone());
    metadata.profile = Some(manifest.profile.clone());
    metadata.snapshot_path = Some(PathBuf::from(snapshot_path));
    metadata.manifest_digest = Some(manifest.digest());
    Ok(metadata)
}

async fn admin_metrics(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Value>, EngineError> {
    require_admin(&state, &headers)?;
    let metrics = *state.metrics.lock().expect("metrics lock is not poisoned");
    let tokens = metrics.tokens();
    Ok(Json(json!({
        "requests_total": metrics.requests_total(),
        "successful_requests": metrics.successful_requests(),
        "failed_requests": metrics.failed_requests(),
        "streamed_requests": metrics.streamed_requests(),
        "tokens": {
            "prompt_tokens": tokens.prompt_tokens(),
            "completion_tokens": tokens.completion_tokens(),
            "total_tokens": tokens.total_tokens(),
        }
    })))
}

async fn chat_completions(
    State(state): State<AppState>,
    request: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Result<Response, EngineError> {
    let request = parse_json_request(request, &state)?;
    let streamed = request.stream;
    let _permit = acquire_model_permit(&state)?;
    if request.stream {
        let response = match state.runtime.chat_stream(request).await {
            Ok(response) => response,
            Err(err) => {
                record_failure_metrics(&state);
                return Err(err.into());
            }
        };
        record_success_metrics(&state, &response.usage, streamed);
        let mut events: Vec<Result<Event, Infallible>> =
            Vec::with_capacity(response.chunks.len() + 1);
        for chunk in response.chunks {
            let data = serde_json::to_string(&chunk).map_err(|err| {
                record_failure_metrics(&state);
                EngineError::Serialize(err)
            })?;
            events.push(Ok(Event::default().data(data)));
        }
        events.push(Ok(Event::default().data("[DONE]")));
        return Ok(Sse::new(stream::iter(events)).into_response());
    }
    let response = match state.runtime.chat(request).await {
        Ok(response) => response,
        Err(err) => {
            record_failure_metrics(&state);
            return Err(err.into());
        }
    };
    record_success_metrics(&state, &response.usage, streamed);
    Ok(Json(response).into_response())
}

async fn completions(
    State(state): State<AppState>,
    request: Result<Json<CompletionRequest>, JsonRejection>,
) -> Result<Response, EngineError> {
    let request = parse_json_request(request, &state)?;
    let streamed = request.stream;
    let _permit = acquire_model_permit(&state)?;
    if request.stream {
        let response = match state.runtime.completion_stream(request).await {
            Ok(response) => response,
            Err(err) => {
                record_failure_metrics(&state);
                return Err(err.into());
            }
        };
        record_success_metrics(&state, &response.usage, streamed);
        let mut events: Vec<Result<Event, Infallible>> =
            Vec::with_capacity(response.chunks.len() + 1);
        for chunk in response.chunks {
            let data = serde_json::to_string(&chunk).map_err(|err| {
                record_failure_metrics(&state);
                EngineError::Serialize(err)
            })?;
            events.push(Ok(Event::default().data(data)));
        }
        events.push(Ok(Event::default().data("[DONE]")));
        return Ok(Sse::new(stream::iter(events)).into_response());
    }
    let response = match state.runtime.completion(request).await {
        Ok(response) => response,
        Err(err) => {
            record_failure_metrics(&state);
            return Err(err.into());
        }
    };
    record_success_metrics(&state, &response.usage, streamed);
    Ok(Json(response).into_response())
}

fn record_success_metrics(state: &AppState, usage: &Usage, streamed: bool) {
    state
        .metrics
        .lock()
        .expect("metrics lock is not poisoned")
        .record_success(
            TokenCounters::new(usage.prompt_tokens, usage.completion_tokens),
            streamed,
        );
}

fn record_failure_metrics(state: &AppState) {
    state
        .metrics
        .lock()
        .expect("metrics lock is not poisoned")
        .record_failure();
}

fn acquire_model_permit(state: &AppState) -> Result<OwnedSemaphorePermit, EngineError> {
    state
        .model_permits
        .clone()
        .try_acquire_owned()
        .map_err(|_| {
            record_failure_metrics(state);
            EngineError::Overloaded("model is busy; retry the request later".to_owned())
        })
}

fn require_admin(state: &AppState, headers: &HeaderMap) -> Result<(), EngineError> {
    let Some(token) = &state.admin_token else {
        return Ok(());
    };
    let Some(header_value) = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
    else {
        return Err(EngineError::UnauthorizedAdmin);
    };
    if header_value == format!("Bearer {token}") {
        return Ok(());
    }
    Err(EngineError::UnauthorizedAdmin)
}

fn parse_json_request<T>(
    request: Result<Json<T>, JsonRejection>,
    state: &AppState,
) -> Result<T, EngineError> {
    match request {
        Ok(Json(request)) => Ok(request),
        Err(err) => {
            record_failure_metrics(state);
            Err(RuntimeError::Api(ApiError::invalid_request(format!(
                "invalid JSON request body: {err}"
            )))
            .into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_max_tokens_defaults_to_configured_limit_when_omitted() {
        assert_eq!(
            resolve_native_max_tokens(None, 4).expect("omitted max tokens uses backend cap"),
            4
        );
    }

    #[test]
    fn native_max_tokens_rejects_requests_above_configured_limit() {
        let err = resolve_native_max_tokens(Some(8), 4)
            .expect_err("explicit max tokens above backend cap fails closed");

        assert!(matches!(err, BackendError::UnsupportedRequest(_)));
        assert!(err.to_string().contains("max_tokens 8"));
    }
}

#[derive(Debug)]
enum EngineError {
    Runtime(RuntimeError),
    ModelStore(HubError),
    Overloaded(String),
    UnauthorizedAdmin,
    Serialize(serde_json::Error),
}

impl From<RuntimeError> for EngineError {
    fn from(value: RuntimeError) -> Self {
        Self::Runtime(value)
    }
}

impl IntoResponse for EngineError {
    fn into_response(self) -> axum::response::Response {
        let (status, code, phase, retryable, message) = match self {
            Self::Runtime(err) => {
                let (status, code, phase, retryable) = match &err {
                    RuntimeError::Api(api) => (
                        StatusCode::BAD_REQUEST,
                        api.code(),
                        "request_validation",
                        false,
                    ),
                    RuntimeError::Backend(BackendError::ModelNotFound { .. }) => (
                        StatusCode::NOT_FOUND,
                        "model_not_found",
                        "model_resolution",
                        false,
                    ),
                    RuntimeError::Backend(BackendError::UnsupportedRequest(_)) => (
                        StatusCode::BAD_REQUEST,
                        "unsupported_capability",
                        "request_validation",
                        false,
                    ),
                    RuntimeError::Backend(BackendError::Other(_)) => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "backend_execution_failed",
                        "decode",
                        true,
                    ),
                    RuntimeError::Template(_) => (
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "chat_template_failed",
                        "prompt_rendering",
                        false,
                    ),
                    RuntimeError::Parser(err) => (
                        StatusCode::UNPROCESSABLE_ENTITY,
                        err.code(),
                        "response_parsing",
                        false,
                    ),
                    RuntimeError::Json(_) | RuntimeError::JsonMode(_) => (
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "json_validation_failed",
                        "response_validation",
                        false,
                    ),
                    RuntimeError::ToolCallValidation(_) => (
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "tool_call_validation_failed",
                        "response_validation",
                        false,
                    ),
                    RuntimeError::NoProgress(_) => (
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "no_progress",
                        "response_validation",
                        false,
                    ),
                };
                (status, code, phase, retryable, err.to_string())
            }
            Self::Serialize(err) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "response_serialization_failed",
                "response_serialization",
                true,
                err.to_string(),
            ),
            Self::ModelStore(err) => (
                if err.code() == "model_not_found" {
                    StatusCode::NOT_FOUND
                } else {
                    StatusCode::UNPROCESSABLE_ENTITY
                },
                err.code(),
                "model_artifact_verification",
                false,
                err.to_string(),
            ),
            Self::Overloaded(message) => (
                StatusCode::TOO_MANY_REQUESTS,
                "model_overloaded",
                "scheduler",
                true,
                message,
            ),
            Self::UnauthorizedAdmin => (
                StatusCode::UNAUTHORIZED,
                "admin_auth_required",
                "admin_auth",
                false,
                "admin bearer token is required".to_owned(),
            ),
        };
        let body = Json(json!({
            "error": {
                "message": message,
                "code": code,
                "phase": phase,
                "retryable": retryable,
                "type": "llm_engine_error"
            }
        }));
        (status, body).into_response()
    }
}
