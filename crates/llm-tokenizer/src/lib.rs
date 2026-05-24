pub use llm_chat_template::{
    DeepSeekPromptOptions, GemmaPromptOptions, LlamaPromptOptions, QwenPromptOptions,
    TemplateError, render_deepseek_chat_template, render_family_chat_template,
    render_family_chat_template_with_tool_instruction, render_gemma4_chat_template,
    render_llama3_chat_template, render_qwen_chatml,
};
use thiserror::Error;
use tokenizers::{
    DecoderWrapper, ModelWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper,
    Tokenizer,
};

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

    pub fn decode_stream(&self, skip_special_tokens: bool) -> HuggingFaceDecodeStream<'_> {
        HuggingFaceDecodeStream {
            inner: self.inner.decode_stream(skip_special_tokens),
        }
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
}

pub struct HuggingFaceDecodeStream<'tokenizer> {
    inner: tokenizers::tokenizer::DecodeStream<
        'tokenizer,
        ModelWrapper,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >,
}

impl HuggingFaceDecodeStream<'_> {
    pub fn step(&mut self, id: u32) -> Result<Option<String>, TokenizerError> {
        self.inner
            .step(id)
            .map_err(|err| TokenizerError::Decode(err.to_string()))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_stream_withholds_partial_utf8_until_token_boundary() {
        let tokenizer_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/qwen36/tokenizer.json");
        let tokenizer = HuggingFaceTokenizer::from_file(tokenizer_path).expect("tokenizer loads");
        let token_ids = tokenizer
            .encode("삥뽕빵", false)
            .expect("fixture text encodes");
        let mut decode_stream = tokenizer.decode_stream(false);
        let mut decoded = String::new();
        let mut withheld_steps = 0;

        for token_id in token_ids {
            match decode_stream
                .step(token_id)
                .expect("stream decode succeeds")
            {
                Some(piece) => decoded.push_str(&piece),
                None => withheld_steps += 1,
            }
        }

        assert_eq!(decoded, "삥뽕빵");
        assert!(withheld_steps > 0);
    }
}
