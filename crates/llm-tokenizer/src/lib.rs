//! Tokenizer and chat-template helpers used by local text inference.
//!
//! The crate wraps Hugging Face `tokenizer.json` files with stable identity
//! metadata for cache keys and re-exports family chat-template renderers used by
//! the runtime prompt adapters.

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

const HUGGINGFACE_TOKENIZER_KIND: &str = "huggingface-tokenizer-json";
const HUGGINGFACE_TOKENIZER_NORMALIZATION: &str = "llm-tokenizer/hf-json/v1";

/// Loaded Hugging Face tokenizer with stable content identity.
#[derive(Clone)]
pub struct HuggingFaceTokenizer {
    inner: Tokenizer,
    identity: HuggingFaceTokenizerIdentity,
}

/// Stable identity for a tokenizer file and normalization contract.
///
/// Cache users should include all fields. `content_hash` changes with the
/// tokenizer JSON bytes, while `normalization` changes if this wrapper changes
/// how the tokenizer is interpreted.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HuggingFaceTokenizerIdentity {
    /// Tokenizer source kind.
    pub kind: String,
    /// SHA-256 hash of the loaded tokenizer JSON bytes.
    pub content_hash: String,
    /// Version string for this wrapper's interpretation of Hugging Face JSON.
    pub normalization: String,
}

impl HuggingFaceTokenizer {
    /// Loads a Hugging Face `tokenizer.json` from disk and records its content hash.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, TokenizerError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|err| {
            TokenizerError::Load(format!("failed to read `{}`: {err}", path.display()))
        })?;
        let inner =
            Tokenizer::from_file(path).map_err(|err| TokenizerError::Load(err.to_string()))?;
        Ok(Self {
            inner,
            identity: HuggingFaceTokenizerIdentity {
                kind: HUGGINGFACE_TOKENIZER_KIND.to_owned(),
                content_hash: hash_bytes(&bytes),
                normalization: HUGGINGFACE_TOKENIZER_NORMALIZATION.to_owned(),
            },
        })
    }

    /// Encodes text into token IDs.
    ///
    /// `add_special_tokens` is passed through to the underlying tokenizer so
    /// prompt renderers can decide whether the template already supplied them.
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

    /// Decodes token IDs into text.
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

    /// Creates an incremental decode stream for token-by-token generation.
    ///
    /// The stream may withhold bytes until a token boundary can be decoded into
    /// valid UTF-8 text.
    pub fn decode_stream(&self, skip_special_tokens: bool) -> HuggingFaceDecodeStream<'_> {
        HuggingFaceDecodeStream {
            inner: self.inner.decode_stream(skip_special_tokens),
        }
    }

    /// Resolves a token string to its token ID if it exists in the vocabulary.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Returns the stable tokenizer identity captured at load time.
    pub fn identity(&self) -> &HuggingFaceTokenizerIdentity {
        &self.identity
    }
}

fn hash_bytes(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};

    let digest = Sha256::digest(bytes);
    format!("sha256:{digest:x}")
}

/// Incremental decoder for streaming generated token IDs.
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
    /// Pushes one token ID and returns newly decodable text, if any.
    pub fn step(&mut self, id: u32) -> Result<Option<String>, TokenizerError> {
        self.inner
            .step(id)
            .map_err(|err| TokenizerError::Decode(err.to_string()))
    }
}

/// Error returned while loading, encoding, or decoding tokenizer data.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TokenizerError {
    /// Tokenizer JSON could not be read or parsed.
    #[error("failed to load tokenizer: {0}")]
    Load(String),
    /// Text could not be encoded.
    #[error("failed to encode text: {0}")]
    Encode(String),
    /// Token IDs could not be decoded.
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

    #[test]
    fn tokenizer_identity_hashes_loaded_json_content() {
        let tokenizer_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/qwen36/tokenizer.json");
        let first = HuggingFaceTokenizer::from_file(&tokenizer_path).expect("tokenizer loads");
        let second = HuggingFaceTokenizer::from_file(tokenizer_path).expect("tokenizer reloads");

        assert_eq!(first.identity(), second.identity());
        assert_eq!(first.identity().kind, "huggingface-tokenizer-json");
        assert!(first.identity().content_hash.starts_with("sha256:"));
        assert_eq!(first.identity().normalization, "llm-tokenizer/hf-json/v1");
    }
}
