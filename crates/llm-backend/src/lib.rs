mod core;

#[cfg(feature = "test-utils")]
pub mod protocol_test {
    pub use super::core::ProtocolTestBackend;
}

pub mod traits {
    pub use super::core::{
        BackendCacheKey, BackendFinishReason, BackendModelMetadata, BackendOutput, BackendRequest,
        BackendStreamChunk, BackendStreamProgress, BackendToolCallDelta, ModelBackend,
        SamplingConfig,
    };
}

pub use core::{
    BackendCacheContext, BackendCacheKey, BackendChatContext, BackendChatMessage, BackendChatRole,
    BackendError, BackendErrorDomain, BackendFinishReason, BackendModelMetadata, BackendOutput,
    BackendRequest, BackendStreamChunk, BackendStreamProgress, BackendToolCall,
    BackendToolCallDelta, BackendToolCallFunction, BackendToolCallFunctionDelta,
    BackendToolCallType, BackendToolChoice, CpuNativeMatvecBackend, F32TensorCacheWarmup,
    GemmaLayerCache, GemmaLayerCacheSnapshot, InferenceScratchpad, KvCacheError, LayerKvCache,
    LayerKvCacheSnapshot, LinearAttentionCache, LinearAttentionCacheSnapshot, MathError,
    ModelBackend, NativeBatchedMatvecOutput, NativeKvCacheTensor, NativeMatvecBackend,
    NativeTextLayerCaches, NativeTextLayerCachesMut, NativeTextModelSpec, NativeTextModelSpecRef,
    QWEN_EMBED_TOKENS_WEIGHT, QWEN_FINAL_NORM_WEIGHT, QWEN_LAYER0_INPUT_NORM_WEIGHT,
    QwenEmbeddingProbe, QwenLayerCache, QwenLayerCacheSnapshot, QwenLinearAttentionProjectionProbe,
    QwenMoeDims, QwenMoeRouterProbe, SafeTensorArchive, SafeTensorFile, SafeTensorHeader,
    SafeTensorShardStore, SamplingConfig, TensorLoadError, TensorMetadata, TopKLogit, TopKWeight,
    gemma_cache_count_for_spec, gemma_decode_token_with_cache, gemma_final_norm_for_spec,
    gemma_layer_caches_for_spec, gemma_lm_head_logits_for_spec, gemma_lm_head_top_k_for_spec,
    gemma_prefill_sequence_with_cache, gemma_static_f32_tensors_for_spec,
    native_decode_token_with_cache_for_spec_ref, native_final_norm_for_spec_ref,
    native_layer_caches_for_spec, native_lm_head_logits_for_spec_ref,
    native_lm_head_top_k_for_spec_ref, native_prefill_sequence_with_cache_for_spec_ref,
    qwen_decode_token_with_cache, qwen_decoder_layer_first_token, qwen_embedding_and_layer0_norm,
    qwen_final_norm, qwen_final_norm_for_spec, qwen_layer_caches_for_spec,
    qwen_layer_moe_forward_in_place, qwen_layer_moe_router,
    qwen_layer0_linear_attention_first_token, qwen_layer0_linear_attention_projections,
    qwen_layer0_post_attention_norm, qwen_linear_decoder_layer_first_token,
    qwen_lm_head_logits_for_spec, qwen_lm_head_top_k, qwen_lm_head_top_k_for_spec,
    qwen_prefill_sequence_with_cache, qwen_static_f32_tensors_for_spec,
};

#[cfg(feature = "test-utils")]
pub use core::ProtocolTestBackend;
