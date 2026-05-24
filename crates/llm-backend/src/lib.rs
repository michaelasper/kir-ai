mod core;

pub mod contracts {
    pub use llm_backend_contracts::*;
}

#[cfg(feature = "test-utils")]
pub mod protocol_test {
    pub use super::core::ProtocolTestBackend;
}

pub mod traits {
    pub use super::contracts::{
        BackendCacheKey, BackendCapabilities, BackendChatRequest, BackendCompletionRequest,
        BackendFinishReason, BackendHealth, BackendHealthStatus, BackendModelMetadata,
        BackendOutput, BackendRequest, BackendRequestKind, BackendStreamChunk,
        BackendStreamProgress, BackendStreamTimingMilestone, BackendToolCallDelta,
        BackendToolDefinition, BackendToolFunctionDefinition, BackendToolType, ModelBackend,
        SamplingConfig,
    };
}

pub mod native {
    pub use super::core::{
        BlockId, CpuNativeMatvecBackend, F32TensorCacheWarmup, GemmaLayerCache,
        GemmaLayerCachePrefixState, GemmaLayerCacheSnapshot, InferenceScratchpad, KvCacheError,
        LayerKvCache, LayerKvCacheBlock, LayerKvCachePrefixState, LayerKvCacheSnapshot,
        LinearAttentionCache, LinearAttentionCacheSnapshot, MathError,
        NativeBatchedMatvecInputBuffer, NativeBatchedMatvecOutput, NativeBatchedMatvecRows,
        NativeKvCacheTensor, NativeMatvecBackend, NativeTextLayerCaches, NativeTextLayerCachesMut,
        NativeTextModelSpec, NativeTextModelSpecRef, QWEN_EMBED_TOKENS_WEIGHT,
        QWEN_FINAL_NORM_WEIGHT, QWEN_LAYER0_INPUT_NORM_WEIGHT, QwenEmbeddingProbe, QwenLayerCache,
        QwenLayerCachePrefixState, QwenLayerCacheSnapshot, QwenLinearAttentionProjectionProbe,
        QwenMoeDims, QwenMoeRouterProbe, SafeTensorArchive, SafeTensorFile, SafeTensorHeader,
        SafeTensorShardStore, TensorLoadError, TensorMetadata, TopKLogit, TopKWeight,
        gemma_cache_count_for_spec, gemma_decode_token_with_cache, gemma_final_norm_for_spec,
        gemma_layer_caches_for_spec, gemma_lm_head_logits_for_spec, gemma_lm_head_top_k_for_spec,
        gemma_prefill_sequence_with_cache, gemma_static_f32_tensors_for_spec,
        native_decode_token_with_cache_for_spec_ref, native_final_norm_for_spec_ref,
        native_layer_caches_for_spec, native_lm_head_logits_for_spec_ref,
        native_lm_head_top_k_for_spec_ref, native_prefill_sequence_with_cache_for_spec_ref,
        qwen_decode_token_with_cache, qwen_decoder_layer_first_token,
        qwen_embedding_and_layer0_norm, qwen_final_norm, qwen_final_norm_for_spec,
        qwen_layer_caches_for_spec, qwen_layer_moe_forward, qwen_layer_moe_forward_in_place,
        qwen_layer_moe_router, qwen_layer0_linear_attention_first_token,
        qwen_layer0_linear_attention_projections, qwen_layer0_moe_forward, qwen_layer0_moe_router,
        qwen_layer0_post_attention_norm, qwen_linear_decoder_layer_first_token,
        qwen_lm_head_logits_for_spec, qwen_lm_head_top_k, qwen_lm_head_top_k_for_spec,
        qwen_prefill_sequence_with_cache, qwen_static_f32_tensors_for_spec,
    };
}

pub use contracts::*;

#[cfg(feature = "test-utils")]
pub use core::ProtocolTestBackend;
