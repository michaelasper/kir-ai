use crate::{
    CpuQwenMatvecBackend, QwenLayerCache, QwenMatvecBackend, SafeTensorShardStore, TensorLoadError,
    TopKLogit, qwen_decode_token_with_cache_with_matvec, qwen_final_norm_for_spec_with_matvec,
    qwen_layer_caches_for_spec, qwen_lm_head_logits_for_spec_with_matvec,
    qwen_lm_head_top_k_for_spec_with_matvec, qwen_prefill_sequence_with_cache_with_matvec,
};
use llm_models::NativeTextModelSpec;

pub fn native_layer_caches_for_spec(
    spec: &NativeTextModelSpec,
    max_tokens: usize,
) -> Result<Vec<QwenLayerCache>, TensorLoadError> {
    match spec {
        NativeTextModelSpec::Qwen(spec) => qwen_layer_caches_for_spec(spec, max_tokens),
    }
}

pub fn native_prefill_sequence_with_cache(
    store: &SafeTensorShardStore,
    spec: &NativeTextModelSpec,
    token_ids: &[usize],
    caches: &mut [QwenLayerCache],
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    native_prefill_sequence_with_cache_with_matvec(
        store,
        spec,
        token_ids,
        caches,
        &CpuQwenMatvecBackend,
    )
}

pub fn native_prefill_sequence_with_cache_with_matvec(
    store: &SafeTensorShardStore,
    spec: &NativeTextModelSpec,
    token_ids: &[usize],
    caches: &mut [QwenLayerCache],
    matvec: &impl QwenMatvecBackend,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    match spec {
        NativeTextModelSpec::Qwen(spec) => {
            qwen_prefill_sequence_with_cache_with_matvec(store, spec, token_ids, caches, matvec)
        }
    }
}

pub fn native_decode_token_with_cache(
    store: &SafeTensorShardStore,
    spec: &NativeTextModelSpec,
    token_id: usize,
    caches: &mut [QwenLayerCache],
) -> Result<Vec<f32>, TensorLoadError> {
    native_decode_token_with_cache_with_matvec(store, spec, token_id, caches, &CpuQwenMatvecBackend)
}

pub fn native_decode_token_with_cache_with_matvec(
    store: &SafeTensorShardStore,
    spec: &NativeTextModelSpec,
    token_id: usize,
    caches: &mut [QwenLayerCache],
    matvec: &impl QwenMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    match spec {
        NativeTextModelSpec::Qwen(spec) => {
            qwen_decode_token_with_cache_with_matvec(store, spec, token_id, caches, matvec)
        }
    }
}

pub fn native_final_norm_for_spec(
    store: &SafeTensorShardStore,
    spec: &NativeTextModelSpec,
    hidden_states: &[f32],
) -> Result<Vec<f32>, TensorLoadError> {
    native_final_norm_for_spec_with_matvec(store, spec, hidden_states, &CpuQwenMatvecBackend)
}

pub fn native_final_norm_for_spec_with_matvec(
    store: &SafeTensorShardStore,
    spec: &NativeTextModelSpec,
    hidden_states: &[f32],
    matvec: &impl QwenMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    match spec {
        NativeTextModelSpec::Qwen(spec) => {
            qwen_final_norm_for_spec_with_matvec(store, spec, hidden_states, matvec)
        }
    }
}

pub fn native_lm_head_top_k_for_spec(
    store: &SafeTensorShardStore,
    spec: &NativeTextModelSpec,
    hidden_states: &[f32],
    top_k: usize,
    chunk_rows: usize,
) -> Result<Vec<TopKLogit>, TensorLoadError> {
    native_lm_head_top_k_for_spec_with_matvec(
        store,
        spec,
        hidden_states,
        top_k,
        chunk_rows,
        &CpuQwenMatvecBackend,
    )
}

pub fn native_lm_head_top_k_for_spec_with_matvec(
    store: &SafeTensorShardStore,
    spec: &NativeTextModelSpec,
    hidden_states: &[f32],
    top_k: usize,
    chunk_rows: usize,
    matvec: &impl QwenMatvecBackend,
) -> Result<Vec<TopKLogit>, TensorLoadError> {
    match spec {
        NativeTextModelSpec::Qwen(spec) => qwen_lm_head_top_k_for_spec_with_matvec(
            store,
            spec,
            hidden_states,
            top_k,
            chunk_rows,
            matvec,
        ),
    }
}

pub fn native_lm_head_logits_for_spec(
    store: &SafeTensorShardStore,
    spec: &NativeTextModelSpec,
    hidden_states: &[f32],
    chunk_rows: usize,
) -> Result<Vec<f32>, TensorLoadError> {
    native_lm_head_logits_for_spec_with_matvec(
        store,
        spec,
        hidden_states,
        chunk_rows,
        &CpuQwenMatvecBackend,
    )
}

pub fn native_lm_head_logits_for_spec_with_matvec(
    store: &SafeTensorShardStore,
    spec: &NativeTextModelSpec,
    hidden_states: &[f32],
    chunk_rows: usize,
    matvec: &impl QwenMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    match spec {
        NativeTextModelSpec::Qwen(spec) => {
            qwen_lm_head_logits_for_spec_with_matvec(store, spec, hidden_states, chunk_rows, matvec)
        }
    }
}
