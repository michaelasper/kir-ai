use async_trait::async_trait;
use axum::{
    Json, Router,
    extract::{Path as AxumPath, State, rejection::JsonRejection},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use futures::{
    Stream, StreamExt,
    stream::{self, BoxStream},
};
use llm_api::{
    ApiError, ChatCompletionRequest, ChatCompletionStreamResponse, CompletionRequest,
    CompletionStreamResponse, FinishReason, ModelCard, ModelList, Usage, ValidateRequest,
};
use llm_backend::{
    BackendError, BackendModelMetadata, BackendOutput, BackendRequest, BackendStreamChunk,
    DeterministicBackend, ModelBackend, QwenLayerCache, SafeTensorShardStore, SamplingConfig,
    qwen_decode_token_with_cache, qwen_final_norm, qwen_layer_caches_for_spec, qwen_lm_head_logits,
    qwen_lm_head_top_k, qwen_prefill_sequence_with_cache,
};
use llm_hub::{
    DownloadPlan, HubClient, HubError, HubRepoId, ModelProfile, ModelStore, SnapshotManifest,
};
use llm_models::QwenModelSpec;
use llm_runtime::{
    ChatCompletionStreamEvent, CompletionStreamEvent, Runtime, RuntimeError,
    chat_stream_requires_buffering,
};
use llm_sampler::TopPSampler;
use llm_telemetry::{ServerMetrics, TokenCounters};
use llm_tokenizer::HuggingFaceTokenizer;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    convert::Infallible,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_util::sync::CancellationToken;

type EngineRuntime = Runtime<Box<dyn ModelBackend>>;

#[derive(Clone)]
struct AppState {
    runtime: Arc<EngineRuntime>,
    metrics: Arc<Mutex<ServerMetrics>>,
    model_permits: Arc<Semaphore>,
    active_requests: Arc<Mutex<HashMap<String, CancellationToken>>>,
    next_request_id: Arc<AtomicU64>,
    admin_token: Option<Arc<str>>,
    model_home: PathBuf,
    hub_client: HubClient,
    hf_token: Option<Arc<str>>,
    stream_stall_timeout: Option<Duration>,
}

#[derive(Debug)]
struct ActiveRequest {
    id: String,
    cancellation: CancellationToken,
    active_requests: Arc<Mutex<HashMap<String, CancellationToken>>>,
}

impl Drop for ActiveRequest {
    fn drop(&mut self) {
        self.active_requests
            .lock()
            .expect("active request lock is not poisoned")
            .remove(&self.id);
    }
}

#[derive(Debug, Clone)]
pub struct EngineOptions {
    pub concurrency_limit: usize,
    pub admin_token: Option<String>,
    pub model_home: Option<PathBuf>,
    pub hub_endpoint: Option<String>,
    pub hf_token: Option<String>,
    pub stream_stall_timeout: Option<Duration>,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            concurrency_limit: 0,
            admin_token: None,
            model_home: None,
            hub_endpoint: None,
            hf_token: None,
            stream_stall_timeout: Some(DEFAULT_STREAM_STALL_TIMEOUT),
        }
    }
}

const DEFAULT_STREAM_STALL_TIMEOUT: Duration = Duration::from_secs(300);

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
            ..EngineOptions::default()
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
        .route("/admin/models/{alias}/plan", post(admin_model_plan))
        .route("/admin/models/{alias}/pull", post(admin_model_pull))
        .route(
            "/admin/requests/{request_id}/cancel",
            post(admin_cancel_request),
        )
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
            active_requests: Arc::new(Mutex::new(HashMap::new())),
            next_request_id: Arc::new(AtomicU64::new(1)),
            admin_token: options.admin_token.map(Arc::from),
            model_home: options.model_home.unwrap_or_else(default_model_home),
            hub_client: options
                .hub_endpoint
                .map(|endpoint| {
                    HubClient::new(url::Url::parse(&endpoint).expect("hub endpoint URL parses"))
                })
                .unwrap_or_default(),
            hf_token: options.hf_token.map(Arc::from),
            stream_stall_timeout: options.stream_stall_timeout,
        })
}

fn default_backend() -> DeterministicBackend {
    DeterministicBackend::new("local-qwen36", "hello from rust native backend")
        .with_required_tool_protocol()
        .with_json_object_protocol()
        .with_conversation_protocol()
}

fn default_model_home() -> PathBuf {
    std::env::var_os("LLM_MODEL_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".llm-models"))
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativeQwenLoadOptions {
    pub eager_materialize_shards: bool,
}

impl NativeQwenBackend {
    pub fn open(
        model_id: impl Into<String>,
        snapshot_path: impl AsRef<Path>,
    ) -> anyhow::Result<Self> {
        Self::open_with_options(model_id, snapshot_path, NativeQwenLoadOptions::default())
    }

    pub fn open_with_options(
        model_id: impl Into<String>,
        snapshot_path: impl AsRef<Path>,
        options: NativeQwenLoadOptions,
    ) -> anyhow::Result<Self> {
        let model_id = model_id.into();
        let snapshot_path = snapshot_path.as_ref();
        let config_json = std::fs::read_to_string(snapshot_path.join("config.json"))?;
        let metadata = native_qwen_metadata(&model_id, snapshot_path)?;
        let store = SafeTensorShardStore::open(snapshot_path)?;
        if options.eager_materialize_shards {
            let materialized_bytes = store.materialize_all_shards()?;
            tracing::info!(
                materialized_bytes,
                "materialized native Qwen safetensors shards"
            );
        }
        Ok(Self {
            model_id,
            metadata,
            tokenizer: HuggingFaceTokenizer::from_file(snapshot_path.join("tokenizer.json"))?,
            spec: QwenModelSpec::from_config_json(&config_json)?,
            store,
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

    fn generate_blocking(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        if cancellation.is_cancelled() {
            return Err(BackendError::Cancelled);
        }
        if request.model != self.model_id {
            return Err(BackendError::ModelNotFound {
                requested: request.model,
                available: self.model_id.clone(),
            });
        }
        if cancellation.is_cancelled() {
            return Err(BackendError::Cancelled);
        }
        let prompt_tokens = self
            .tokenizer
            .encode(&request.prompt, false)
            .map_err(|err| BackendError::Other(err.to_string()))?;
        let context_tokens = prompt_tokens
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
        let mut decode = self.start_decode_session(&context_tokens, requested)?;
        if cancellation.is_cancelled() {
            return Err(BackendError::Cancelled);
        }

        for step_idx in 0..requested {
            if cancellation.is_cancelled() {
                return Err(BackendError::Cancelled);
            }
            let candidate = self.next_token_from_hidden(decode.hidden(), request.sampling)?;
            if cancellation.is_cancelled() {
                return Err(BackendError::Cancelled);
            }
            if Some(candidate.token_id) == eos_id {
                finish_reason = FinishReason::Stop;
                break;
            }
            output_ids.push(u32::try_from(candidate.token_id).map_err(|err| {
                BackendError::Other(format!("Qwen token id does not fit u32: {err}"))
            })?);
            if step_idx + 1 < requested {
                decode.step(&self.store, &self.spec, candidate.token_id)?;
            }
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

    fn generate_blocking_stream(
        &self,
        request: BackendRequest,
        tx: tokio::sync::mpsc::Sender<Result<BackendStreamChunk, BackendError>>,
        cancellation: CancellationToken,
    ) -> Result<(), BackendError> {
        if cancellation.is_cancelled() {
            return Ok(());
        }
        if request.model != self.model_id {
            return Err(BackendError::ModelNotFound {
                requested: request.model,
                available: self.model_id.clone(),
            });
        }
        if cancellation.is_cancelled() {
            return Ok(());
        }
        let prompt_tokens = self
            .tokenizer
            .encode(&request.prompt, false)
            .map_err(|err| BackendError::Other(err.to_string()))?;
        let context_tokens = prompt_tokens
            .iter()
            .map(|token| *token as usize)
            .collect::<Vec<_>>();
        if context_tokens.is_empty() {
            return Err(BackendError::Other(
                "Qwen prompt encoded to zero tokens".to_owned(),
            ));
        }
        let mut output_ids = Vec::new();
        let mut decoded = String::new();
        let mut finish_reason = FinishReason::Length;
        let eos_id = self
            .tokenizer
            .token_to_id("<|im_end|>")
            .map(|id| id as usize);
        let requested = resolve_native_max_tokens(request.max_tokens, self.max_new_tokens)?;
        let mut decode = self.start_decode_session(&context_tokens, requested)?;
        if cancellation.is_cancelled() {
            return Ok(());
        }

        for step_idx in 0..requested {
            if cancellation.is_cancelled() {
                return Ok(());
            }
            let candidate = self.next_token_from_hidden(decode.hidden(), request.sampling)?;
            if cancellation.is_cancelled() {
                return Ok(());
            }
            if Some(candidate.token_id) == eos_id {
                finish_reason = FinishReason::Stop;
                break;
            }
            output_ids.push(u32::try_from(candidate.token_id).map_err(|err| {
                BackendError::Other(format!("Qwen token id does not fit u32: {err}"))
            })?);
            let next_decoded = self
                .tokenizer
                .decode(&output_ids, false)
                .map_err(|err| BackendError::Other(err.to_string()))?;
            let delta = next_decoded
                .strip_prefix(&decoded)
                .unwrap_or(&next_decoded)
                .to_owned();
            decoded = next_decoded;
            if cancellation.is_cancelled() {
                return Ok(());
            }
            send_backend_stream_chunk(
                &tx,
                BackendStreamChunk {
                    text: delta,
                    prompt_tokens: prompt_tokens.len() as u64,
                    completion_tokens: 1,
                    finish_reason: None,
                },
            )?;
            if step_idx + 1 < requested {
                if cancellation.is_cancelled() {
                    return Ok(());
                }
                decode.step(&self.store, &self.spec, candidate.token_id)?;
            }
        }

        if cancellation.is_cancelled() {
            return Ok(());
        }
        send_backend_stream_chunk(
            &tx,
            BackendStreamChunk {
                text: String::new(),
                prompt_tokens: prompt_tokens.len() as u64,
                completion_tokens: 0,
                finish_reason: Some(finish_reason),
            },
        )
    }

    fn start_decode_session(
        &self,
        context_tokens: &[usize],
        max_new_tokens: u32,
    ) -> Result<NativeQwenDecodeSession, BackendError> {
        let start = context_tokens.len().saturating_sub(self.max_prefill_tokens);
        let prefill_tokens = &context_tokens[start..];
        let cache_tokens = prefill_tokens
            .len()
            .checked_add(max_new_tokens as usize)
            .ok_or_else(|| BackendError::Other("Qwen cache token capacity overflow".to_owned()))?;
        let mut caches = qwen_layer_caches_for_spec(&self.spec, cache_tokens)
            .map_err(|err| BackendError::Other(err.to_string()))?;
        let hidden_states =
            qwen_prefill_sequence_with_cache(&self.store, &self.spec, prefill_tokens, &mut caches)
                .map_err(|err| BackendError::Other(err.to_string()))?;
        let hidden = hidden_states.last().cloned().ok_or_else(|| {
            BackendError::Other("Qwen prefill returned no hidden states".to_owned())
        })?;
        Ok(NativeQwenDecodeSession { hidden, caches })
    }

    fn next_token_from_hidden(
        &self,
        hidden: &[f32],
        sampling: SamplingConfig,
    ) -> Result<NativeQwenCandidate, BackendError> {
        let final_norm = qwen_final_norm(
            &self.store,
            hidden,
            self.spec.hidden_size as usize,
            self.spec.rms_norm_eps,
        )
        .map_err(|err| BackendError::Other(err.to_string()))?;
        if !sampling.is_greedy() {
            let logits = qwen_lm_head_logits(&self.store, &final_norm, self.chunk_rows)
                .map_err(|err| BackendError::Other(err.to_string()))?;
            let sampled_token_id =
                sample_token_id_with_draw(&logits, sampling, native_sampling_draw())?;
            let token_id = u32::try_from(sampled_token_id).map_err(|err| {
                BackendError::Other(format!("Qwen token id does not fit u32: {err}"))
            })?;
            let text = self
                .tokenizer
                .decode(&[token_id], false)
                .map_err(|err| BackendError::Other(err.to_string()))?;
            return Ok(NativeQwenCandidate {
                token_id: sampled_token_id,
                text,
            });
        }

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

struct NativeQwenDecodeSession {
    hidden: Vec<f32>,
    caches: Vec<QwenLayerCache>,
}

impl NativeQwenDecodeSession {
    fn hidden(&self) -> &[f32] {
        &self.hidden
    }

    fn step(
        &mut self,
        store: &SafeTensorShardStore,
        spec: &QwenModelSpec,
        token_id: usize,
    ) -> Result<(), BackendError> {
        self.hidden = qwen_decode_token_with_cache(store, spec, token_id, &mut self.caches)
            .map_err(|err| BackendError::Other(err.to_string()))?;
        Ok(())
    }
}

fn resolve_native_max_tokens(
    requested: Option<u32>,
    configured_max: u32,
) -> Result<u32, BackendError> {
    let configured_max = configured_max.max(1);
    match requested {
        None => Ok(configured_max),
        Some(0) => Err(BackendError::UnsupportedRequest(
            "max_tokens must be greater than 0".to_owned(),
        )),
        Some(value) if value > configured_max => Err(BackendError::UnsupportedRequest(format!(
            "requested max_tokens {value} exceeds configured native Qwen limit {configured_max}"
        ))),
        Some(value) => Ok(value),
    }
}

#[derive(Debug, Clone)]
struct NativeQwenCandidate {
    token_id: usize,
    text: String,
}

fn sample_token_id_with_draw(
    logits: &[f32],
    sampling: SamplingConfig,
    draw: f32,
) -> Result<usize, BackendError> {
    if logits.is_empty() {
        return Err(BackendError::Other(
            "Qwen lm head returned no logits".to_owned(),
        ));
    }
    match sampling {
        SamplingConfig::Greedy => llm_sampler::GreedySampler
            .sample(logits)
            .map_err(|err| BackendError::Other(err.to_string())),
        SamplingConfig::TopP { temperature, top_p } => TopPSampler { temperature, top_p }
            .sample(logits, draw)
            .map_err(|err| BackendError::Other(err.to_string())),
    }
}

static NATIVE_SAMPLING_COUNTER: AtomicU64 = AtomicU64::new(0);

fn native_sampling_draw() -> f32 {
    let time_seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0);
    let counter = NATIVE_SAMPLING_COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut value = time_seed ^ counter.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    let bits = value.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40;
    (bits as f32) / ((1_u32 << 24) as f32)
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
        self.generate_with_cancel(request, CancellationToken::new())
            .await
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        let backend = self.clone();
        tokio::task::spawn_blocking(move || backend.generate_blocking(request, cancellation))
            .await
            .map_err(|err| BackendError::Other(format!("native Qwen worker failed: {err}")))?
    }

    fn generate_stream<'a>(
        &'a self,
        request: BackendRequest,
    ) -> BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        self.generate_stream_with_cancel(request, CancellationToken::new())
    }

    fn generate_stream_with_cancel<'a>(
        &'a self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        let backend = self.clone();
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        tokio::task::spawn_blocking(move || {
            let err_tx = tx.clone();
            if let Err(err) = backend.generate_blocking_stream(request, tx, cancellation) {
                let _ = err_tx.blocking_send(Err(err));
            }
        });
        stream::unfold(rx, |mut rx| async {
            rx.recv().await.map(|item| (item, rx))
        })
        .boxed()
    }
}

fn send_backend_stream_chunk(
    tx: &tokio::sync::mpsc::Sender<Result<BackendStreamChunk, BackendError>>,
    chunk: BackendStreamChunk,
) -> Result<(), BackendError> {
    tx.blocking_send(Ok(chunk))
        .map_err(|_| BackendError::Other("stream receiver dropped".to_owned()))
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

#[derive(Debug, Deserialize)]
struct AdminModelPlanRequest {
    repo_id: String,
    #[serde(default)]
    revision: Option<String>,
    #[serde(default)]
    profile: Option<String>,
    #[serde(default)]
    metadata_only: bool,
}

async fn admin_model_plan(
    AxumPath(alias): AxumPath<String>,
    State(state): State<AppState>,
    headers: HeaderMap,
    request: Result<Json<AdminModelPlanRequest>, JsonRejection>,
) -> Result<Json<DownloadPlan>, EngineError> {
    require_admin(&state, &headers)?;
    require_model_alias(&state, alias)?;
    let request = parse_json_request(request, &state)?;
    let plan = build_admin_download_plan(&state, request).await?;
    Ok(Json(plan))
}

async fn admin_model_pull(
    AxumPath(alias): AxumPath<String>,
    State(state): State<AppState>,
    headers: HeaderMap,
    request: Result<Json<AdminModelPlanRequest>, JsonRejection>,
) -> Result<Json<Value>, EngineError> {
    require_admin(&state, &headers)?;
    require_model_alias(&state, alias)?;
    let request = parse_json_request(request, &state)?;
    let plan = build_admin_download_plan(&state, request).await?;
    let snapshot = ModelStore::new(&state.model_home)
        .pull_plan(&state.hub_client, &plan, state.hf_token.as_deref())
        .await
        .map_err(EngineError::ModelStore)?;
    Ok(Json(json!({
        "snapshot_path": snapshot.path,
        "manifest_digest": snapshot.manifest_digest,
        "repo_id": snapshot.manifest.repo_id,
        "resolved_commit": snapshot.manifest.resolved_commit,
        "profile": snapshot.manifest.profile,
        "files": snapshot.manifest.files.len(),
    })))
}

async fn build_admin_download_plan(
    state: &AppState,
    request: AdminModelPlanRequest,
) -> Result<DownloadPlan, EngineError> {
    let repo_id = HubRepoId::model(request.repo_id).map_err(EngineError::ModelStore)?;
    let revision = request.revision.unwrap_or_else(|| "main".to_owned());
    let profile_name = request
        .profile
        .unwrap_or_else(|| "qwen36-safetensors-bf16".to_owned());
    let profile = model_profile(&profile_name)?;
    let mut plan = state
        .hub_client
        .plan_model(repo_id, &revision, profile, state.hf_token.as_deref())
        .await
        .map_err(EngineError::ModelStore)?;
    if request.metadata_only {
        plan = plan.metadata_only();
    }
    Ok(plan)
}

fn model_profile(name: &str) -> Result<ModelProfile, EngineError> {
    match name {
        "qwen36-mlx-4bit" => Ok(ModelProfile::qwen36_mlx_4bit()),
        "qwen36-safetensors-bf16" => Ok(ModelProfile::qwen36_safetensors_bf16()),
        other => Err(RuntimeError::Api(ApiError::invalid_request(format!(
            "unknown model profile `{other}`"
        )))
        .into()),
    }
}

fn require_model_alias(state: &AppState, alias: String) -> Result<(), EngineError> {
    let model_id = state.runtime.model_id();
    if alias == model_id {
        return Ok(());
    }
    Err(RuntimeError::Backend(BackendError::ModelNotFound {
        requested: alias,
        available: model_id.to_owned(),
    })
    .into())
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
    let mut metadata = BackendModelMetadata::new(model_id.to_owned(), "native-qwen");
    metadata.snapshot_path = Some(PathBuf::from(snapshot_path));
    let manifest_bytes = match std::fs::read(&manifest_path) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(metadata),
        Err(err) => return Err(err.into()),
    };
    let manifest = serde_json::from_slice::<SnapshotManifest>(&manifest_bytes)?;
    metadata.family = Some(manifest.family.clone());
    metadata.loader = Some(manifest.loader.clone());
    metadata.quantization = Some(manifest.quantization.clone());
    metadata.repo_id = Some(manifest.repo_id.clone());
    metadata.resolved_commit = Some(manifest.resolved_commit.clone());
    metadata.profile = Some(manifest.profile.clone());
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
    let request_latency = metrics.request_latency();
    let time_to_first_token = metrics.time_to_first_token();
    let active_requests = state
        .active_requests
        .lock()
        .expect("active request lock is not poisoned")
        .len();
    Ok(Json(json!({
        "requests_total": metrics.requests_total(),
        "successful_requests": metrics.successful_requests(),
        "failed_requests": metrics.failed_requests(),
        "streamed_requests": metrics.streamed_requests(),
        "active_requests": active_requests,
        "queued_requests": 0,
        "prefill_requests": 0,
        "decode_requests": active_requests,
        "cancelled_requests": metrics.cancelled_requests(),
        "no_progress_failures": metrics.no_progress_failures(),
        "tokens_per_second": metrics.tokens_per_second(),
        "request_latency_ms": {
            "count": request_latency.count(),
            "min": request_latency.min_ms(),
            "max": request_latency.max_ms(),
            "avg": request_latency.avg_ms(),
        },
        "time_to_first_token_ms": {
            "count": time_to_first_token.count(),
            "min": time_to_first_token.min_ms(),
            "max": time_to_first_token.max_ms(),
            "avg": time_to_first_token.avg_ms(),
        },
        "tokens": {
            "prompt_tokens": tokens.prompt_tokens(),
            "completion_tokens": tokens.completion_tokens(),
            "total_tokens": tokens.total_tokens(),
        }
    })))
}

async fn admin_cancel_request(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(request_id): AxumPath<String>,
) -> Result<Json<Value>, EngineError> {
    require_admin(&state, &headers)?;
    let cancellation = state
        .active_requests
        .lock()
        .expect("active request lock is not poisoned")
        .get(&request_id)
        .cloned()
        .ok_or_else(|| EngineError::RequestNotFound(request_id.clone()))?;
    cancellation.cancel();
    record_cancellation_metrics(&state);
    Ok(Json(json!({
        "object": "admin.request_cancellation",
        "request_id": request_id,
        "status": "cancelled"
    })))
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    request: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Result<Response, EngineError> {
    let request = parse_json_request(request, &state)?;
    validate_api_request(&request, &state)?;
    let streamed = request.stream;
    if request.stream {
        let permit = acquire_model_permit(&state)?;
        let active_request = register_active_request(&state, &headers)?;
        let request_id = active_request.id.clone();
        let request_started = Instant::now();
        if chat_stream_requires_buffering(&request) {
            let response = match state
                .runtime
                .chat_stream_buffered_with_cancel(request, active_request.cancellation.clone())
                .await
            {
                Ok(response) => response,
                Err(err) => {
                    record_runtime_error_metrics(&state, &err);
                    return Err(err.into());
                }
            };
            let events = async_stream::stream! {
                let _permit = permit;
                let _active_request = active_request;
                let mut events = response.into_events();
                let mut ttft_recorded = false;
                loop {
                    match next_stream_event(&mut events, state.stream_stall_timeout).await {
                        Ok(Some(Ok(ChatCompletionStreamEvent::Chunk(chunk)))) => {
                            if !ttft_recorded && chat_chunk_has_real_delta(&chunk) {
                                record_time_to_first_token_metrics(&state, request_started.elapsed());
                                ttft_recorded = true;
                            }
                            yield sse_json_event(chunk);
                        }
                        Ok(Some(Ok(ChatCompletionStreamEvent::Complete(usage)))) => {
                            record_success_metrics(&state, &usage, streamed, request_started.elapsed());
                            yield Ok(Event::default().data("[DONE]"));
                        }
                        Ok(Some(Err(err))) => {
                            record_runtime_error_metrics(&state, &err);
                            for event in runtime_error_stream_events(err) {
                                yield event;
                            }
                            return;
                        }
                        Ok(None) => break,
                        Err(StreamStalled) => {
                            record_failure_metrics(&state);
                            for event in stream_stalled_stream_events(state.stream_stall_timeout) {
                                yield event;
                            }
                            return;
                        }
                    }
                }
            };
            let mut response = Sse::new(events)
                .keep_alive(engine_sse_keep_alive())
                .into_response();
            insert_request_id_header(&mut response, &request_id);
            return Ok(response);
        }
        let request_started = Instant::now();
        let events = async_stream::stream! {
            let _permit = permit;
            let _active_request = active_request;
            match state
                .runtime
                .chat_stream_with_cancel(request, _active_request.cancellation.clone())
                .await
            {
                Ok(response) => {
                    let mut events = response.into_events();
                    let mut ttft_recorded = false;
                    loop {
                        match next_stream_event(&mut events, state.stream_stall_timeout).await {
                            Ok(Some(Ok(ChatCompletionStreamEvent::Chunk(chunk)))) => {
                                if !ttft_recorded && chat_chunk_has_real_delta(&chunk) {
                                    record_time_to_first_token_metrics(&state, request_started.elapsed());
                                    ttft_recorded = true;
                                }
                                yield sse_json_event(chunk);
                            }
                            Ok(Some(Ok(ChatCompletionStreamEvent::Complete(usage)))) => {
                                record_success_metrics(&state, &usage, streamed, request_started.elapsed());
                                yield Ok(Event::default().data("[DONE]"));
                            }
                            Ok(Some(Err(err))) => {
                                record_runtime_error_metrics(&state, &err);
                                for event in runtime_error_stream_events(err) {
                                    yield event;
                                }
                                return;
                            }
                            Ok(None) => break,
                            Err(StreamStalled) => {
                                record_failure_metrics(&state);
                                for event in stream_stalled_stream_events(state.stream_stall_timeout) {
                                    yield event;
                                }
                                return;
                            }
                        }
                    }
                }
                Err(err) => {
                    record_runtime_error_metrics(&state, &err);
                    for event in runtime_error_stream_events(err) {
                        yield event;
                    }
                }
            }
        };
        let mut response = Sse::new(events)
            .keep_alive(engine_sse_keep_alive())
            .into_response();
        insert_request_id_header(&mut response, &request_id);
        return Ok(response);
    }
    let _permit = acquire_model_permit(&state)?;
    let active_request = register_active_request(&state, &headers)?;
    let request_id = active_request.id.clone();
    let request_started = Instant::now();
    let response = match state
        .runtime
        .chat_with_cancel(request, active_request.cancellation.clone())
        .await
    {
        Ok(response) => response,
        Err(err) => {
            record_runtime_error_metrics(&state, &err);
            return Err(err.into());
        }
    };
    drop(active_request);
    record_success_metrics(&state, &response.usage, streamed, request_started.elapsed());
    let mut response = Json(response).into_response();
    insert_request_id_header(&mut response, &request_id);
    Ok(response)
}

async fn completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    request: Result<Json<CompletionRequest>, JsonRejection>,
) -> Result<Response, EngineError> {
    let request = parse_json_request(request, &state)?;
    validate_api_request(&request, &state)?;
    let streamed = request.stream;
    if request.stream {
        let permit = acquire_model_permit(&state)?;
        let active_request = register_active_request(&state, &headers)?;
        let request_id = active_request.id.clone();
        let request_started = Instant::now();
        let events = async_stream::stream! {
            let _permit = permit;
            let _active_request = active_request;
            match state
                .runtime
                .completion_stream_with_cancel(request, _active_request.cancellation.clone())
                .await
            {
                Ok(response) => {
                    let mut events = response.into_events();
                    let mut ttft_recorded = false;
                    loop {
                        match next_stream_event(&mut events, state.stream_stall_timeout).await {
                            Ok(Some(Ok(CompletionStreamEvent::Chunk(chunk)))) => {
                                if !ttft_recorded && completion_chunk_has_real_delta(&chunk) {
                                    record_time_to_first_token_metrics(&state, request_started.elapsed());
                                    ttft_recorded = true;
                                }
                                yield sse_json_event(chunk);
                            }
                            Ok(Some(Ok(CompletionStreamEvent::Complete(usage)))) => {
                                record_success_metrics(&state, &usage, streamed, request_started.elapsed());
                                yield Ok(Event::default().data("[DONE]"));
                            }
                            Ok(Some(Err(err))) => {
                                record_runtime_error_metrics(&state, &err);
                                for event in runtime_error_stream_events(err) {
                                    yield event;
                                }
                                return;
                            }
                            Ok(None) => break,
                            Err(StreamStalled) => {
                                record_failure_metrics(&state);
                                for event in stream_stalled_stream_events(state.stream_stall_timeout) {
                                    yield event;
                                }
                                return;
                            }
                        }
                    }
                }
                Err(err) => {
                    record_runtime_error_metrics(&state, &err);
                    for event in runtime_error_stream_events(err) {
                        yield event;
                    }
                }
            }
        };
        let mut response = Sse::new(events)
            .keep_alive(engine_sse_keep_alive())
            .into_response();
        insert_request_id_header(&mut response, &request_id);
        return Ok(response);
    }
    let _permit = acquire_model_permit(&state)?;
    let active_request = register_active_request(&state, &headers)?;
    let request_id = active_request.id.clone();
    let request_started = Instant::now();
    let response = match state
        .runtime
        .completion_with_cancel(request, active_request.cancellation.clone())
        .await
    {
        Ok(response) => response,
        Err(err) => {
            record_runtime_error_metrics(&state, &err);
            return Err(err.into());
        }
    };
    drop(active_request);
    record_success_metrics(&state, &response.usage, streamed, request_started.elapsed());
    let mut response = Json(response).into_response();
    insert_request_id_header(&mut response, &request_id);
    Ok(response)
}

fn runtime_error_stream_events(err: RuntimeError) -> Vec<Result<Event, Infallible>> {
    vec![
        sse_json_event(json!({
            "error": {
                "message": err.to_string(),
                "type": "llm_engine_error"
            }
        })),
        Ok(Event::default().data("[DONE]")),
    ]
}

fn stream_stalled_stream_events(timeout: Option<Duration>) -> Vec<Result<Event, Infallible>> {
    let message = match timeout {
        Some(timeout) => format!(
            "stream stalled for {} ms without backend output",
            timeout.as_millis()
        ),
        None => "stream stalled without backend output".to_owned(),
    };
    vec![
        sse_json_event(json!({
            "error": {
                "message": message,
                "code": "stream_stalled",
                "phase": "streaming",
                "retryable": true,
                "type": "llm_engine_error"
            }
        })),
        Ok(Event::default().data("[DONE]")),
    ]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StreamStalled;

async fn next_stream_event<S, T>(
    events: &mut S,
    timeout: Option<Duration>,
) -> Result<Option<Result<T, RuntimeError>>, StreamStalled>
where
    S: Stream<Item = Result<T, RuntimeError>> + Unpin,
{
    match timeout {
        Some(timeout) => tokio::time::timeout(timeout, events.next())
            .await
            .map_err(|_| StreamStalled),
        None => Ok(events.next().await),
    }
}

fn engine_sse_keep_alive() -> KeepAlive {
    KeepAlive::new()
        .interval(Duration::from_millis(100))
        .text("llm-engine-heartbeat")
}

fn sse_json_event(value: impl serde::Serialize) -> Result<Event, Infallible> {
    let data = serde_json::to_string(&value).unwrap_or_else(|err| {
        json!({
            "error": {
                "message": format!("response serialization failed: {err}"),
                "type": "llm_engine_error"
            }
        })
        .to_string()
    });
    Ok(Event::default().data(data))
}

fn chat_chunk_has_real_delta(chunk: &ChatCompletionStreamResponse) -> bool {
    chunk.choices.iter().any(|choice| {
        choice
            .delta
            .content
            .as_ref()
            .is_some_and(|content| !content.is_empty())
            || !choice.delta.tool_calls.is_empty()
    })
}

fn completion_chunk_has_real_delta(chunk: &CompletionStreamResponse) -> bool {
    chunk.choices.iter().any(|choice| !choice.text.is_empty())
}

fn record_success_metrics(state: &AppState, usage: &Usage, streamed: bool, latency: Duration) {
    state
        .metrics
        .lock()
        .expect("metrics lock is not poisoned")
        .record_success(
            TokenCounters::new(usage.prompt_tokens, usage.completion_tokens),
            streamed,
            latency,
        );
}

fn record_failure_metrics(state: &AppState) {
    state
        .metrics
        .lock()
        .expect("metrics lock is not poisoned")
        .record_failure();
}

fn record_runtime_error_metrics(state: &AppState, err: &RuntimeError) {
    let mut metrics = state.metrics.lock().expect("metrics lock is not poisoned");
    if matches!(err, RuntimeError::NoProgress(_)) {
        metrics.record_no_progress_failure();
    }
    metrics.record_failure();
}

fn record_cancellation_metrics(state: &AppState) {
    state
        .metrics
        .lock()
        .expect("metrics lock is not poisoned")
        .record_cancellation();
}

fn record_time_to_first_token_metrics(state: &AppState, latency: Duration) {
    state
        .metrics
        .lock()
        .expect("metrics lock is not poisoned")
        .record_time_to_first_token(latency);
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

fn register_active_request(
    state: &AppState,
    headers: &HeaderMap,
) -> Result<ActiveRequest, EngineError> {
    let id = match request_id_from_headers(state, headers) {
        Ok(id) => id,
        Err(err) => {
            record_failure_metrics(state);
            return Err(err);
        }
    };
    let cancellation = CancellationToken::new();
    let mut active_requests = state
        .active_requests
        .lock()
        .expect("active request lock is not poisoned");
    if active_requests.contains_key(&id) {
        record_failure_metrics(state);
        return Err(EngineError::RequestConflict(id));
    }
    active_requests.insert(id.clone(), cancellation.clone());
    drop(active_requests);
    Ok(ActiveRequest {
        id,
        cancellation,
        active_requests: state.active_requests.clone(),
    })
}

fn request_id_from_headers(state: &AppState, headers: &HeaderMap) -> Result<String, EngineError> {
    let Some(value) = headers
        .get("x-request-id")
        .or_else(|| headers.get("x-llm-request-id"))
    else {
        let next = state.next_request_id.fetch_add(1, Ordering::Relaxed);
        return Ok(format!("req-{next}"));
    };
    let request_id = value
        .to_str()
        .map_err(|_| EngineError::InvalidRequestId("request id must be visible ASCII".to_owned()))?
        .trim();
    if request_id.is_empty() {
        return Err(EngineError::InvalidRequestId(
            "request id must not be empty".to_owned(),
        ));
    }
    if request_id.len() > 128 {
        return Err(EngineError::InvalidRequestId(
            "request id must be at most 128 bytes".to_owned(),
        ));
    }
    Ok(request_id.to_owned())
}

fn insert_request_id_header(response: &mut Response, request_id: &str) {
    let value = HeaderValue::from_str(request_id)
        .expect("registered request id came from a valid header value or generated ASCII");
    response.headers_mut().insert("x-request-id", value);
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

fn validate_api_request<T: ValidateRequest>(
    request: &T,
    state: &AppState,
) -> Result<(), EngineError> {
    request.validate().map_err(|err| {
        record_failure_metrics(state);
        RuntimeError::Api(err).into()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_max_tokens_defaults_to_configured_cache_limit() {
        assert_eq!(
            resolve_native_max_tokens(None, 4).expect("omitted max tokens uses configured cap"),
            4
        );
    }

    #[test]
    fn native_max_tokens_accepts_multi_token_decode_with_cache() {
        assert_eq!(
            resolve_native_max_tokens(Some(2), 4).expect("multi-token decode uses cache"),
            2
        );
    }

    #[test]
    fn native_max_tokens_rejects_requests_above_configured_limit() {
        let err = resolve_native_max_tokens(Some(5), 4)
            .expect_err("request above configured limit fails closed");

        assert!(matches!(err, BackendError::UnsupportedRequest(_)));
        assert!(err.to_string().contains("configured native Qwen limit"));
    }

    #[test]
    fn native_qwen_backend_opens_snapshot_without_engine_manifest() {
        let snapshot = temp_snapshot_dir("no-manifest");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        copy_fixture("config.json", snapshot.join("config.json"));
        copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));
        copy_fixture(
            "model.safetensors.index.json",
            snapshot.join("model.safetensors.index.json"),
        );

        let backend =
            NativeQwenBackend::open("local-qwen36", &snapshot).expect("backend opens snapshot");
        let metadata = backend.model_metadata();

        assert_eq!(metadata.id, "local-qwen36");
        assert_eq!(metadata.backend, "native-qwen");
        assert_eq!(metadata.snapshot_path.as_deref(), Some(snapshot.as_path()));
        assert!(metadata.manifest_digest.is_none());
        assert!(metadata.repo_id.is_none());
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn native_qwen_backend_can_eagerly_materialize_indexed_shards_on_open() {
        let snapshot = temp_snapshot_dir("eager-materialize");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        copy_fixture("config.json", snapshot.join("config.json"));
        copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));
        std::fs::write(
            snapshot.join("model.safetensors.index.json"),
            serde_json::json!({
                "metadata": { "total_size": 2 },
                "weight_map": { "dummy.weight": "dummy.safetensors" }
            })
            .to_string(),
        )
        .expect("index");
        std::fs::write(
            snapshot.join("dummy.safetensors"),
            tiny_safetensors_bf16("dummy.weight", &[1], &[1.0]),
        )
        .expect("dummy shard");

        let backend = NativeQwenBackend::open_with_options(
            "local-qwen36",
            &snapshot,
            NativeQwenLoadOptions {
                eager_materialize_shards: true,
            },
        )
        .expect("backend opens and materializes shards");

        assert_eq!(backend.store.materialized_shard_count(), 1);
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[tokio::test]
    async fn native_qwen_generate_with_cancel_observes_pre_cancelled_token() {
        let snapshot = temp_snapshot_dir("cancelled-generate");
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        copy_fixture("config.json", snapshot.join("config.json"));
        copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));
        copy_fixture(
            "model.safetensors.index.json",
            snapshot.join("model.safetensors.index.json"),
        );
        let backend =
            NativeQwenBackend::open("local-qwen36", &snapshot).expect("backend opens snapshot");
        let cancellation = CancellationToken::new();
        cancellation.cancel();

        let err = backend
            .generate_with_cancel(
                BackendRequest {
                    model: "local-qwen36".to_owned(),
                    prompt: "say hi".to_owned(),
                    max_tokens: Some(1),
                    sampling: SamplingConfig::Greedy,
                    required_tool_choice: None,
                    json_object_mode: false,
                    conversation_mode: false,
                },
                cancellation,
            )
            .await
            .expect_err("pre-cancelled generation fails before decode");

        assert!(err.to_string().contains("cancelled"));
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[test]
    fn native_top_p_sampling_selects_full_vocab_token_from_draw() {
        let token_id = sample_token_id_with_draw(
            &[2.0, 1.0, 0.0],
            SamplingConfig::TopP {
                temperature: 1.0,
                top_p: 0.9,
            },
            0.8,
        )
        .expect("sampling succeeds");

        assert_eq!(token_id, 1);
    }

    fn copy_fixture(name: &str, destination: impl AsRef<Path>) {
        let source = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/qwen36")
            .join(name);
        std::fs::copy(&source, destination).expect("copy fixture");
    }

    fn tiny_safetensors_bf16(name: &str, shape: &[usize], values: &[f32]) -> Vec<u8> {
        let mut data = Vec::with_capacity(values.len() * 2);
        for value in values {
            data.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
        }
        let data_len = data.len();
        let header = serde_json::json!({
            name: {
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [0, data_len]
            }
        })
        .to_string();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&data);
        bytes
    }

    fn temp_snapshot_dir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!("llm-engine-{label}-{}", std::process::id()))
    }
}

#[derive(Debug)]
enum EngineError {
    Runtime(RuntimeError),
    ModelStore(HubError),
    Overloaded(String),
    RequestNotFound(String),
    RequestConflict(String),
    InvalidRequestId(String),
    UnauthorizedAdmin,
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
                    RuntimeError::Backend(BackendError::Cancelled) => {
                        (StatusCode::REQUEST_TIMEOUT, "cancelled", "decode", false)
                    }
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
            Self::RequestNotFound(request_id) => (
                StatusCode::NOT_FOUND,
                "request_not_found",
                "cancellation",
                false,
                format!("request `{request_id}` is not active"),
            ),
            Self::RequestConflict(request_id) => (
                StatusCode::CONFLICT,
                "request_id_conflict",
                "request_validation",
                false,
                format!("request id `{request_id}` is already active"),
            ),
            Self::InvalidRequestId(message) => (
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "request_validation",
                false,
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
