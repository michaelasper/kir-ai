use async_trait::async_trait;
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::get,
};
use futures::stream;
use llm_api::{ChatCompletionRequest, FinishReason, ModelCard, ModelList};
use llm_backend::{
    BackendError, BackendOutput, BackendRequest, DeterministicBackend, ModelBackend,
    SafeTensorShardStore, qwen_final_norm, qwen_lm_head_top_k, qwen_prefill_sequence,
};
use llm_models::QwenModelSpec;
use llm_runtime::{Runtime, RuntimeError};
use llm_tokenizer::HuggingFaceTokenizer;
use serde_json::json;
use std::{convert::Infallible, path::Path, sync::Arc};

type EngineRuntime = Runtime<Box<dyn ModelBackend>>;

#[derive(Clone)]
struct AppState {
    runtime: Arc<EngineRuntime>,
}

pub fn build_router() -> Router {
    build_router_with_backend(Box::new(DeterministicBackend::new(
        "local-qwen36",
        "hello from rust native backend",
    )))
}

pub fn build_router_with_backend(backend: Box<dyn ModelBackend>) -> Router {
    let runtime = Runtime::new(backend);
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route(
            "/v1/chat/completions",
            axum::routing::post(chat_completions),
        )
        .with_state(AppState {
            runtime: Arc::new(runtime),
        })
}

#[derive(Clone)]
pub struct NativeQwenBackend {
    model_id: String,
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
        let snapshot_path = snapshot_path.as_ref();
        let config_json = std::fs::read_to_string(snapshot_path.join("config.json"))?;
        Ok(Self {
            model_id: model_id.into(),
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
        let requested = request.max_tokens.max(1).min(self.max_new_tokens);

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

async fn chat_completions(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, EngineError> {
    if request.stream {
        let response = state.runtime.chat_stream(request).await?;
        let mut events: Vec<Result<Event, Infallible>> =
            Vec::with_capacity(response.chunks.len() + 1);
        for chunk in response.chunks {
            let data = serde_json::to_string(&chunk).map_err(EngineError::Serialize)?;
            events.push(Ok(Event::default().data(data)));
        }
        events.push(Ok(Event::default().data("[DONE]")));
        return Ok(Sse::new(stream::iter(events)).into_response());
    }
    let response = state.runtime.chat(request).await?;
    Ok(Json(response).into_response())
}

#[derive(Debug)]
enum EngineError {
    Runtime(RuntimeError),
    Serialize(serde_json::Error),
}

impl From<RuntimeError> for EngineError {
    fn from(value: RuntimeError) -> Self {
        Self::Runtime(value)
    }
}

impl IntoResponse for EngineError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match self {
            Self::Runtime(err) => {
                let status = match &err {
                    RuntimeError::Api(_) => StatusCode::BAD_REQUEST,
                    RuntimeError::Backend(BackendError::ModelNotFound { .. }) => {
                        StatusCode::NOT_FOUND
                    }
                    RuntimeError::Backend(BackendError::Other(_)) => {
                        StatusCode::INTERNAL_SERVER_ERROR
                    }
                    RuntimeError::Template(_)
                    | RuntimeError::Parser(_)
                    | RuntimeError::NoProgress(_) => StatusCode::UNPROCESSABLE_ENTITY,
                };
                (status, err.to_string())
            }
            Self::Serialize(err) => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()),
        };
        let body = Json(json!({
            "error": {
                "message": message,
                "type": "llm_engine_error"
            }
        }));
        (status, body).into_response()
    }
}
