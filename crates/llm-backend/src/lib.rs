//! Public facade for backend contracts and native execution helpers.
//!
//! `contracts` exposes the backend protocol types shared with runtime crates,
//! `traits` keeps the historical trait-focused import path, and `native`
//! exposes the native tensor/KV-cache helpers used by native runtimes and
//! diagnostics. The root contract re-exports are compatibility shims and are
//! kept explicit so the public API remains auditable.

mod core;

/// Backend protocol contracts shared between runtime and backend crates.
pub mod contracts {
    pub use llm_backend_contracts::{
        BackendCacheContext, BackendCacheKey, BackendCapabilities, BackendChatContext,
        BackendChatMessage, BackendChatRequest, BackendChatRole, BackendCompletionRequest,
        BackendError, BackendErrorDomain, BackendFailureClass, BackendFinishReason, BackendHealth,
        BackendHealthStatus, BackendModelMetadata, BackendOutput, BackendPrefillChunkAdmission,
        BackendPrefillChunkAdmissionHook, BackendRequest, BackendRequestKind, BackendStreamChunk,
        BackendStreamProgress, BackendStreamTimingMilestone, BackendToolCall, BackendToolCallDelta,
        BackendToolCallFunction, BackendToolCallFunctionDelta, BackendToolCallType,
        BackendToolChoice, BackendToolDefinition, BackendToolFunctionDefinition, BackendToolType,
        ModelBackend, SamplingConfig,
    };
}

#[cfg(feature = "test-utils")]
/// Test backend used by runtime and HTTP contract tests.
pub mod protocol_test {
    pub use super::core::ProtocolTestBackend;
}

/// Compatibility import path for the backend trait and request/response types.
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

/// Native tensor, safetensors, and KV-cache helpers for local execution paths.
pub mod native {
    pub use super::core::{
        BlockId, CpuNativeMatvecBackend, F32TensorCacheWarmup, GemmaLayerCache,
        GemmaLayerCachePrefixState, GemmaLayerCacheSnapshot, InferenceScratchpad, KvCacheConfig,
        KvCacheError, KvCacheFormat, LayerInt8KvToken, LayerKvCache, LayerKvCacheAppend,
        LayerKvCacheAppendTarget, LayerKvCacheBlock, LayerKvCacheInt8Block,
        LayerKvCachePrefixState, LayerKvCacheSnapshot, LinearAttentionCache,
        LinearAttentionCacheSnapshot, MathError, NativeBatchedMatvecInputBuffer,
        NativeBatchedMatvecOutput, NativeBatchedMatvecRows, NativeKvCacheTensor,
        NativeMatvecBackend, NativeTextLayerCaches, NativeTextLayerCachesMut, NativeTextModelSpec,
        NativeTextModelSpecRef, QWEN_EMBED_TOKENS_WEIGHT, QWEN_FINAL_NORM_WEIGHT,
        QWEN_LAYER0_INPUT_NORM_WEIGHT, QwenEmbeddingProbe, QwenLayerCache,
        QwenLayerCachePrefixState, QwenLayerCacheSnapshot, QwenLinearAttentionProjectionProbe,
        QwenMoeDims, QwenMoeRouterProbe, SafeTensorArchive, SafeTensorFile, SafeTensorHeader,
        SafeTensorShardStore, TensorLoadError, TensorMetadata, TopKLogit, TopKWeight,
        gemma_cache_count_for_spec, gemma_decode_token_with_cache, gemma_final_norm_for_spec,
        gemma_layer_caches_for_spec, gemma_lm_head_logits_for_spec, gemma_lm_head_top_k_for_spec,
        gemma_prefill_sequence_with_cache, gemma_prefill_sequence_with_cache_with_cancel,
        gemma_static_f32_tensors_for_spec, native_decode_token_with_cache_for_spec_ref,
        native_final_norm_for_spec_ref, native_layer_caches_for_spec,
        native_lm_head_logits_for_spec_ref, native_lm_head_top_k_for_spec_ref,
        native_prefill_sequence_with_cache_for_spec_ref,
        native_prefill_sequence_with_cache_for_spec_ref_with_cancel, qwen_decode_token_with_cache,
        qwen_decoder_layer_first_token, qwen_embedding_and_layer0_norm, qwen_final_norm,
        qwen_final_norm_for_spec, qwen_layer_caches_for_spec, qwen_layer_moe_forward,
        qwen_layer_moe_forward_in_place, qwen_layer_moe_router,
        qwen_layer0_linear_attention_first_token, qwen_layer0_linear_attention_projections,
        qwen_layer0_moe_forward, qwen_layer0_moe_router, qwen_layer0_post_attention_norm,
        qwen_linear_decoder_layer_first_token, qwen_lm_head_logits_for_spec, qwen_lm_head_top_k,
        qwen_lm_head_top_k_for_spec, qwen_prefill_sequence_with_cache,
        qwen_prefill_sequence_with_cache_with_cancel, qwen_static_f32_tensors_for_spec,
    };
}

pub use contracts::{
    BackendCacheContext, BackendCacheKey, BackendCapabilities, BackendChatContext,
    BackendChatMessage, BackendChatRequest, BackendChatRole, BackendCompletionRequest,
    BackendError, BackendErrorDomain, BackendFailureClass, BackendFinishReason, BackendHealth,
    BackendHealthStatus, BackendModelMetadata, BackendOutput, BackendPrefillChunkAdmission,
    BackendPrefillChunkAdmissionHook, BackendRequest, BackendRequestKind, BackendStreamChunk,
    BackendStreamProgress, BackendStreamTimingMilestone, BackendToolCall, BackendToolCallDelta,
    BackendToolCallFunction, BackendToolCallFunctionDelta, BackendToolCallType, BackendToolChoice,
    BackendToolDefinition, BackendToolFunctionDefinition, BackendToolType, ModelBackend,
    SamplingConfig,
};

#[cfg(feature = "test-utils")]
pub use core::ProtocolTestBackend;
