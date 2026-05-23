use super::super::super::math::rms_norm_one_centered_f32;
use super::super::super::{SafeTensorShardStore, TensorLoadError};
use super::tensor_names::{QWEN_EMBED_TOKENS_WEIGHT, QWEN_LAYER0_INPUT_NORM_WEIGHT};
use llm_models::QwenModelSpec;

#[derive(Debug, Clone, PartialEq)]
pub struct QwenEmbeddingProbe {
    pub token_id: usize,
    pub embedding: Vec<f32>,
    pub normalized: Vec<f32>,
}

fn qwen_embedding_vocab_size(
    store: &SafeTensorShardStore,
    tensor: &str,
) -> Result<usize, TensorLoadError> {
    let metadata = store.tensor_metadata(tensor)?;
    match metadata.shape.as_slice() {
        [vocab_size, _hidden_size] => Ok(*vocab_size),
        shape => Err(TensorLoadError::integrity(format!(
            "Qwen embedding tensor `{tensor}` must be rank 2, got shape {shape:?}"
        ))),
    }
}

pub(crate) fn validate_qwen_token_id(
    token_id: usize,
    vocab_size: usize,
) -> Result<(), TensorLoadError> {
    if token_id >= vocab_size {
        return Err(TensorLoadError::integrity(format!(
            "Qwen token id {token_id} is outside vocab size {vocab_size}"
        )));
    }
    Ok(())
}

fn validate_qwen_token_ids(token_ids: &[usize], vocab_size: usize) -> Result<(), TensorLoadError> {
    for (position, token_id) in token_ids.iter().enumerate() {
        if *token_id >= vocab_size {
            return Err(TensorLoadError::integrity(format!(
                "Qwen token id {token_id} at position {position} is outside vocab size {vocab_size}"
            )));
        }
    }
    Ok(())
}

pub fn qwen_embedding_and_layer0_norm(
    store: &SafeTensorShardStore,
    token_id: usize,
    hidden_size: usize,
    rms_norm_eps: f32,
) -> Result<QwenEmbeddingProbe, TensorLoadError> {
    let vocab_size = qwen_embedding_vocab_size(store, QWEN_EMBED_TOKENS_WEIGHT)?;
    validate_qwen_token_id(token_id, vocab_size)?;
    let embedding = store.bf16_row_f32(QWEN_EMBED_TOKENS_WEIGHT, token_id)?;
    if embedding.len() != hidden_size {
        return Err(TensorLoadError::integrity(format!(
            "Qwen embedding row has length {}, expected hidden size {hidden_size}",
            embedding.len()
        )));
    }
    let norm_weight = store.bf16_tensor_f32_cached_arc(QWEN_LAYER0_INPUT_NORM_WEIGHT)?;
    let normalized = rms_norm_one_centered_f32(&embedding, norm_weight.as_ref(), rms_norm_eps)
        .map_err(|err| {
            TensorLoadError::integrity(format!("Qwen layer0 input RMSNorm failed: {err}"))
        })?;
    Ok(QwenEmbeddingProbe {
        token_id,
        embedding,
        normalized,
    })
}

pub(crate) fn qwen_embedding_sequence(
    store: &SafeTensorShardStore,
    token_ids: &[usize],
    hidden_size: usize,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    if token_ids.is_empty() {
        return Ok(Vec::new());
    }
    let vocab_size = qwen_embedding_vocab_size(store, QWEN_EMBED_TOKENS_WEIGHT)?;
    validate_qwen_token_ids(token_ids, vocab_size)?;
    token_ids
        .iter()
        .map(|token_id| {
            let embedding = store.bf16_row_f32(QWEN_EMBED_TOKENS_WEIGHT, *token_id)?;
            if embedding.len() != hidden_size {
                return Err(TensorLoadError::integrity(format!(
                    "Qwen embedding row has length {}, expected hidden size {hidden_size}",
                    embedding.len()
                )));
            }
            Ok(embedding)
        })
        .collect()
}

pub(crate) fn qwen_embedding_sequence_for_spec(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    token_ids: &[usize],
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    let hidden_size = spec.hidden_size as usize;
    validate_qwen_token_ids(token_ids, spec.vocab_size as usize)?;
    let tensor = spec.embed_tokens_weight();
    token_ids
        .iter()
        .map(|token_id| {
            let embedding = store.bf16_row_f32(&tensor, *token_id)?;
            if embedding.len() != hidden_size {
                return Err(TensorLoadError::integrity(format!(
                    "Qwen embedding row has length {}, expected hidden size {hidden_size}",
                    embedding.len()
                )));
            }
            Ok(embedding)
        })
        .collect()
}
