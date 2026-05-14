mod core;

#[cfg(feature = "test-utils")]
pub mod protocol_test {
    pub use super::core::ProtocolTestBackend;
}

pub mod traits {
    pub use super::core::{
        BackendCacheKey, BackendModelMetadata, BackendOutput, BackendRequest, BackendStreamChunk,
        BackendToolCallDelta, ModelBackend, SamplingConfig,
    };
}

pub use core::{
    BackendCacheContext, BackendCacheKey, BackendChatContext, BackendChatMessage, BackendChatRole,
    BackendError, BackendModelMetadata, BackendOutput, BackendRequest, BackendStreamChunk,
    BackendToolCall, BackendToolCallDelta, BackendToolCallFunction, BackendToolCallFunctionDelta,
    BackendToolCallType, BackendToolChoice, CpuNativeMatvecBackend, F32TensorCacheWarmup,
    GemmaLayerCache, InferenceScratchpad, LayerKvCache, LinearAttentionCache, MathError,
    ModelBackend, NativeBatchedMatvecOutput, NativeKvCacheTensor, NativeMatvecBackend,
    NativeTextLayerCaches, NativeTextLayerCachesMut, NativeTextModelSpec, NativeTextModelSpecRef,
    QWEN_EMBED_TOKENS_WEIGHT, QWEN_FINAL_NORM_WEIGHT, QWEN_LAYER0_INPUT_NORM_WEIGHT,
    QwenEmbeddingProbe, QwenLayerCache, QwenLinearAttentionProjectionProbe, QwenMoeDims,
    QwenMoeRouterProbe, SafeTensorArchive, SafeTensorFile, SafeTensorHeader, SafeTensorShardStore,
    SamplingConfig, TensorLoadError, TensorMetadata, TopKLogit, TopKWeight,
    gemma_cache_count_for_spec, gemma_decode_token_with_cache_with_matvec,
    gemma_final_norm_for_spec, gemma_layer_caches_for_spec,
    gemma_lm_head_logits_for_spec_with_matvec, gemma_lm_head_top_k_for_spec_with_matvec,
    gemma_prefill_sequence_with_cache_with_matvec, gemma_static_f32_tensors_for_spec,
    native_decode_token_with_cache_for_spec_ref_with_matvec,
    native_final_norm_for_spec_ref_with_matvec, native_layer_caches_for_spec,
    native_lm_head_logits_for_spec_ref_with_matvec, native_lm_head_top_k_for_spec_ref_with_matvec,
    native_prefill_sequence_with_cache_for_spec_ref_with_matvec,
    qwen_decode_token_with_cache_with_matvec, qwen_decoder_layer_first_token,
    qwen_embedding_and_layer0_norm, qwen_final_norm, qwen_final_norm_for_spec_with_matvec,
    qwen_layer_caches_for_spec, qwen_layer_moe_forward_with_matvec_in_place,
    qwen_layer_moe_router_with_matvec, qwen_layer0_linear_attention_first_token,
    qwen_layer0_linear_attention_projections, qwen_layer0_post_attention_norm,
    qwen_linear_decoder_layer_first_token, qwen_lm_head_logits_for_spec_with_matvec,
    qwen_lm_head_top_k, qwen_lm_head_top_k_for_spec_with_matvec,
    qwen_prefill_sequence_with_cache_with_matvec, qwen_static_f32_tensors_for_spec,
};

#[cfg(feature = "test-utils")]
pub use core::ProtocolTestBackend;
