pub use llm_chat_template::{
    DeepSeekPromptOptions, GemmaPromptOptions, LlamaPromptOptions, QwenPromptOptions,
    TemplateError, render_deepseek_chat_template, render_family_chat_template,
    render_family_chat_template_with_tool_instruction, render_gemma4_chat_template,
    render_llama3_chat_template, render_qwen_chatml,
};
use thiserror::Error;
use tokenizers::Tokenizer;

#[derive(Clone)]
pub struct HuggingFaceTokenizer {
    inner: Tokenizer,
}

impl HuggingFaceTokenizer {
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, TokenizerError> {
        let inner = Tokenizer::from_file(path.as_ref())
            .map_err(|err| TokenizerError::Load(err.to_string()))?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, TokenizerError> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|err| TokenizerError::Encode(err.to_string()))?;
        let ids = encoding.get_ids().to_vec();
        tracing::trace!(
            operation = "encode",
            input_bytes = text.len(),
            token_count = ids.len(),
            add_special_tokens,
            "tokenizer encode complete"
        );
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, TokenizerError> {
        let decoded = self
            .inner
            .decode(ids, skip_special_tokens)
            .map_err(|err| TokenizerError::Decode(err.to_string()))?;
        tracing::trace!(
            operation = "decode",
            token_count = ids.len(),
            output_bytes = decoded.len(),
            skip_special_tokens,
            "tokenizer decode complete"
        );
        Ok(decoded)
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
}

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("failed to load tokenizer: {0}")]
    Load(String),
    #[error("failed to encode text: {0}")]
    Encode(String),
    #[error("failed to decode tokens: {0}")]
    Decode(String),
}
