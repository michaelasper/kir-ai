use async_trait::async_trait;
use llm_api::FinishReason;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendRequest {
    pub model: String,
    pub prompt: String,
    pub max_tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendOutput {
    pub text: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub finish_reason: FinishReason,
}

#[async_trait]
pub trait ModelBackend: Send + Sync + 'static {
    fn model_id(&self) -> &str;

    async fn generate(&self, request: BackendRequest) -> Result<BackendOutput, BackendError>;
}

#[derive(Debug, Clone)]
pub struct DeterministicBackend {
    model_id: String,
    text: String,
}

impl DeterministicBackend {
    pub fn new(model_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            text: text.into(),
        }
    }
}

#[async_trait]
impl ModelBackend for DeterministicBackend {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn generate(&self, request: BackendRequest) -> Result<BackendOutput, BackendError> {
        if request.model != self.model_id {
            return Err(BackendError::ModelNotFound {
                requested: request.model,
                available: self.model_id.clone(),
            });
        }
        Ok(BackendOutput {
            text: self.text.clone(),
            prompt_tokens: count_tokens(&request.prompt),
            completion_tokens: count_tokens(&self.text),
            finish_reason: FinishReason::Stop,
        })
    }
}

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("model `{requested}` is not loaded; available model is `{available}`")]
    ModelNotFound {
        requested: String,
        available: String,
    },
    #[error("backend error: {0}")]
    Other(String),
}

fn count_tokens(text: &str) -> u64 {
    let normalized = text
        .replace("<|im_start|>system", " ")
        .replace("<|im_start|>user", " ")
        .replace("<|im_start|>assistant", " ")
        .replace("<|im_start|>tool", " ")
        .replace("<|im_end|>", " ")
        .replace("</think>", " ");
    normalized.split_whitespace().count().max(1) as u64
}
