#![allow(dead_code)]
// Qwen ops keep layer-level probes and reference entry points crate-private;
// many are validated by backend unit tests rather than the library call graph.

mod attention_full;
mod attention_linear;
mod cache;
mod decoder;
mod embedding;
mod lm_head;
mod moe;
mod norm;
mod tensor_names;

#[cfg(test)]
#[allow(unused_imports)]
pub(crate) use attention_full::{
    QwenFullAttentionDims, QwenFullAttentionSequenceConfig, QwenFullAttentionSequenceParts,
    QwenFullAttentionStepParts, qwen_full_attention_first_token_from_parts,
    qwen_full_attention_sequence_from_parts, qwen_full_attention_sequence_with_cache_from_parts,
    qwen_full_attention_step_with_cache_from_parts, qwen_layer_full_attention_sequence_with_cache,
};
#[cfg(test)]
#[allow(unused_imports)]
pub(crate) use attention_linear::{
    QwenLinearAttentionDims, QwenLinearAttentionFirstTokenParts, QwenLinearAttentionSequenceParts,
    QwenLinearAttentionStepParts, qwen_layer_linear_attention_sequence_with_cache,
    qwen_linear_attention_first_token_from_parts, qwen_linear_attention_sequence_from_parts,
    qwen_linear_attention_sequence_with_cache_from_parts,
    qwen_linear_attention_step_with_cache_from_parts,
};
pub use attention_linear::{
    QwenLinearAttentionProjectionProbe, qwen_layer0_linear_attention_first_token,
    qwen_layer0_linear_attention_projections,
};
pub use cache::{
    QwenLayerCache, QwenLayerCachePrefixState, QwenLayerCacheSnapshot, qwen_layer_caches_for_spec,
};
pub use decoder::{
    qwen_decode_token_with_cache, qwen_decoder_layer_first_token,
    qwen_linear_decoder_layer_first_token, qwen_prefill_sequence_with_cache,
};
pub use embedding::{QwenEmbeddingProbe, qwen_embedding_and_layer0_norm};
#[cfg(test)]
#[allow(unused_imports)]
pub(crate) use embedding::{qwen_embedding_sequence, qwen_embedding_sequence_for_spec};
#[cfg(all(test, feature = "slow-tests"))]
pub(crate) use lm_head::qwen_lm_head_logits;
pub use lm_head::{
    qwen_final_norm, qwen_final_norm_for_spec, qwen_lm_head_logits_for_spec, qwen_lm_head_top_k,
    qwen_lm_head_top_k_for_spec,
};
pub use moe::{
    QwenMoeDims, QwenMoeRouterProbe, qwen_layer_moe_forward, qwen_layer_moe_forward_in_place,
    qwen_layer_moe_router, qwen_layer0_moe_forward, qwen_layer0_moe_router,
};
pub use norm::qwen_layer0_post_attention_norm;
pub use tensor_names::{
    QWEN_EMBED_TOKENS_WEIGHT, QWEN_FINAL_NORM_WEIGHT, QWEN_LAYER0_INPUT_NORM_WEIGHT,
    qwen_static_f32_tensors_for_spec,
};
