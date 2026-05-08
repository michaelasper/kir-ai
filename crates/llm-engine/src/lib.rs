use axum::{Json, Router, extract::State, http::StatusCode, response::IntoResponse, routing::get};
use llm_api::{ChatCompletionRequest, ModelCard, ModelList};
use llm_backend::DeterministicBackend;
use llm_runtime::{Runtime, RuntimeError};
use serde_json::json;
use std::sync::Arc;

type EngineRuntime = Runtime<DeterministicBackend>;

#[derive(Clone)]
struct AppState {
    runtime: Arc<EngineRuntime>,
}

pub fn build_router() -> Router {
    let runtime = Runtime::new(DeterministicBackend::new(
        "local-qwen36",
        "hello from rust native backend",
    ));
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
) -> Result<Json<llm_api::ChatCompletionResponse>, EngineError> {
    let response = state.runtime.chat(request).await?;
    Ok(Json(response))
}

#[derive(Debug)]
struct EngineError(RuntimeError);

impl From<RuntimeError> for EngineError {
    fn from(value: RuntimeError) -> Self {
        Self(value)
    }
}

impl IntoResponse for EngineError {
    fn into_response(self) -> axum::response::Response {
        let status = match &self.0 {
            RuntimeError::Api(_) => StatusCode::BAD_REQUEST,
            RuntimeError::Backend(_) => StatusCode::NOT_FOUND,
            RuntimeError::Template(_) | RuntimeError::NoProgress(_) => {
                StatusCode::UNPROCESSABLE_ENTITY
            }
        };
        let body = Json(json!({
            "error": {
                "message": self.0.to_string(),
                "type": "llm_engine_error"
            }
        }));
        (status, body).into_response()
    }
}
