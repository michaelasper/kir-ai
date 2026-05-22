pub use llm_kv_cache::{
    BlockId, KvCacheError, LayerKvCache, LayerKvCacheBlock, LayerKvCachePrefixState,
    LayerKvCacheSnapshot, LinearAttentionCache, LinearAttentionCacheSnapshot,
};

mod backend;
mod gemma;
mod math;
mod native_attention;
mod native_matvec;
mod native_text;
#[cfg(feature = "test-utils")]
// Security: gated behind non-default feature to prevent production exposure (GH#139).
mod protocol_test;
mod qwen;
mod safetensors;

pub use backend::{
    BackendCacheContext, BackendCacheKey, BackendCapabilities, BackendChatContext,
    BackendChatMessage, BackendChatRequest, BackendChatRole, BackendCompletionRequest,
    BackendError, BackendErrorDomain, BackendFinishReason, BackendHealth, BackendHealthStatus,
    BackendModelMetadata, BackendOutput, BackendRequest, BackendRequestKind, BackendStreamChunk,
    BackendStreamProgress, BackendStreamTimingMilestone, BackendToolCall, BackendToolCallDelta,
    BackendToolCallFunction, BackendToolCallFunctionDelta, BackendToolCallType, BackendToolChoice,
    BackendToolDefinition, BackendToolFunctionDefinition, BackendToolType, ModelBackend,
    SamplingConfig,
};
pub use gemma::{
    GemmaLayerCache, GemmaLayerCachePrefixState, GemmaLayerCacheSnapshot,
    gemma_cache_count_for_spec, gemma_decode_token_with_cache, gemma_final_norm_for_spec,
    gemma_layer_caches_for_spec, gemma_lm_head_logits_for_spec, gemma_lm_head_top_k_for_spec,
    gemma_prefill_sequence_with_cache, gemma_static_f32_tensors_for_spec,
};
pub use math::{InferenceScratchpad, MathError, TopKLogit, TopKWeight};
pub use native_matvec::{
    CpuNativeMatvecBackend, NativeBatchedMatvecOutput, NativeKvCacheTensor, NativeMatvecBackend,
};
pub use native_text::{
    NativeTextLayerCaches, NativeTextLayerCachesMut, NativeTextModelSpec, NativeTextModelSpecRef,
    native_decode_token_with_cache_for_spec_ref, native_final_norm_for_spec_ref,
    native_layer_caches_for_spec, native_lm_head_logits_for_spec_ref,
    native_lm_head_top_k_for_spec_ref, native_prefill_sequence_with_cache_for_spec_ref,
};
#[cfg(feature = "test-utils")]
pub use protocol_test::ProtocolTestBackend;
pub use qwen::{
    QWEN_EMBED_TOKENS_WEIGHT, QWEN_FINAL_NORM_WEIGHT, QWEN_LAYER0_INPUT_NORM_WEIGHT,
    QwenEmbeddingProbe, QwenLayerCache, QwenLayerCachePrefixState, QwenLayerCacheSnapshot,
    QwenLinearAttentionProjectionProbe, QwenMoeDims, QwenMoeRouterProbe,
    qwen_decode_token_with_cache, qwen_decoder_layer_first_token, qwen_embedding_and_layer0_norm,
    qwen_final_norm, qwen_final_norm_for_spec, qwen_layer_caches_for_spec, qwen_layer_moe_forward,
    qwen_layer_moe_forward_in_place, qwen_layer_moe_router,
    qwen_layer0_linear_attention_first_token, qwen_layer0_linear_attention_projections,
    qwen_layer0_moe_forward, qwen_layer0_moe_router, qwen_layer0_post_attention_norm,
    qwen_linear_decoder_layer_first_token, qwen_lm_head_logits_for_spec, qwen_lm_head_top_k,
    qwen_lm_head_top_k_for_spec, qwen_prefill_sequence_with_cache,
    qwen_static_f32_tensors_for_spec,
};
pub use safetensors::{
    F32TensorCacheWarmup, SafeTensorArchive, SafeTensorFile, SafeTensorHeader,
    SafeTensorShardStore, TensorLoadError, TensorMetadata,
};

#[cfg(test)]
mod tests {
    mod cpu_ops;
    mod gemma_native_ops;
    #[cfg(feature = "slow-tests")]
    #[allow(dead_code)]
    mod safetensors_loader;
}
