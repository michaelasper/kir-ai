use crate::{
    CpuNativeMatvecBackend, GemmaLayerCache, NativeMatvecBackend, QwenLayerCache,
    SafeTensorShardStore, TensorLoadError, TopKLogit, gemma_decode_token_with_cache_with_matvec,
    gemma_final_norm_for_spec, gemma_layer_caches_for_spec,
    gemma_lm_head_logits_for_spec_with_matvec, gemma_lm_head_top_k_for_spec_with_matvec,
    gemma_prefill_sequence_with_cache_with_matvec, qwen_decode_token_with_cache_with_matvec,
    qwen_final_norm_for_spec_with_matvec, qwen_layer_caches_for_spec,
    qwen_lm_head_logits_for_spec_with_matvec, qwen_lm_head_top_k_for_spec_with_matvec,
    qwen_prefill_sequence_with_cache_with_matvec,
};
use llm_models::NativeTextModelSpec;

#[derive(Debug, Clone, PartialEq)]
pub enum NativeTextLayerCaches {
    Qwen(Vec<QwenLayerCache>),
    Gemma(Vec<GemmaLayerCache>),
}

impl NativeTextLayerCaches {
    pub fn len(&self) -> usize {
        match self {
            Self::Qwen(caches) => caches.len(),
            Self::Gemma(caches) => caches.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn family_slug(&self) -> &'static str {
        match self {
            Self::Qwen(_) => "qwen",
            Self::Gemma(_) => "gemma",
        }
    }
}

pub fn native_layer_caches_for_spec(
    spec: &NativeTextModelSpec,
    max_tokens: usize,
) -> Result<NativeTextLayerCaches, TensorLoadError> {
    match spec {
        NativeTextModelSpec::Qwen(spec) => {
            qwen_layer_caches_for_spec(spec, max_tokens).map(NativeTextLayerCaches::Qwen)
        }
        NativeTextModelSpec::Gemma(spec) => {
            gemma_layer_caches_for_spec(spec, max_tokens).map(NativeTextLayerCaches::Gemma)
        }
    }
}

pub fn native_prefill_sequence_with_cache(
    store: &SafeTensorShardStore,
    spec: &NativeTextModelSpec,
    token_ids: &[usize],
    caches: &mut NativeTextLayerCaches,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    native_prefill_sequence_with_cache_with_matvec(
        store,
        spec,
        token_ids,
        caches,
        &CpuNativeMatvecBackend,
    )
}

pub fn native_prefill_sequence_with_cache_with_matvec(
    store: &SafeTensorShardStore,
    spec: &NativeTextModelSpec,
    token_ids: &[usize],
    caches: &mut NativeTextLayerCaches,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    match (spec, caches) {
        (NativeTextModelSpec::Qwen(spec), NativeTextLayerCaches::Qwen(caches)) => {
            qwen_prefill_sequence_with_cache_with_matvec(store, spec, token_ids, caches, matvec)
        }
        (NativeTextModelSpec::Gemma(spec), NativeTextLayerCaches::Gemma(caches)) => {
            gemma_prefill_sequence_with_cache_with_matvec(store, spec, token_ids, caches, matvec)
        }
        (_, caches) => Err(cache_family_mismatch("prefill", spec, caches)),
    }
}

pub fn native_decode_token_with_cache(
    store: &SafeTensorShardStore,
    spec: &NativeTextModelSpec,
    token_id: usize,
    caches: &mut NativeTextLayerCaches,
) -> Result<Vec<f32>, TensorLoadError> {
    native_decode_token_with_cache_with_matvec(
        store,
        spec,
        token_id,
        caches,
        &CpuNativeMatvecBackend,
    )
}

pub fn native_decode_token_with_cache_with_matvec(
    store: &SafeTensorShardStore,
    spec: &NativeTextModelSpec,
    token_id: usize,
    caches: &mut NativeTextLayerCaches,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    match (spec, caches) {
        (NativeTextModelSpec::Qwen(spec), NativeTextLayerCaches::Qwen(caches)) => {
            qwen_decode_token_with_cache_with_matvec(store, spec, token_id, caches, matvec)
        }
        (NativeTextModelSpec::Gemma(spec), NativeTextLayerCaches::Gemma(caches)) => {
            gemma_decode_token_with_cache_with_matvec(store, spec, token_id, caches, matvec)
        }
        (_, caches) => Err(cache_family_mismatch("decode", spec, caches)),
    }
}

pub fn native_final_norm_for_spec(
    store: &SafeTensorShardStore,
    spec: &NativeTextModelSpec,
    hidden_states: &[f32],
) -> Result<Vec<f32>, TensorLoadError> {
    native_final_norm_for_spec_with_matvec(store, spec, hidden_states, &CpuNativeMatvecBackend)
}

pub fn native_final_norm_for_spec_with_matvec(
    store: &SafeTensorShardStore,
    spec: &NativeTextModelSpec,
    hidden_states: &[f32],
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    match spec {
        NativeTextModelSpec::Qwen(spec) => {
            qwen_final_norm_for_spec_with_matvec(store, spec, hidden_states, matvec)
        }
        NativeTextModelSpec::Gemma(spec) => gemma_final_norm_for_spec(store, spec, hidden_states),
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
        &CpuNativeMatvecBackend,
    )
}

pub fn native_lm_head_top_k_for_spec_with_matvec(
    store: &SafeTensorShardStore,
    spec: &NativeTextModelSpec,
    hidden_states: &[f32],
    top_k: usize,
    chunk_rows: usize,
    matvec: &impl NativeMatvecBackend,
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
        NativeTextModelSpec::Gemma(spec) => gemma_lm_head_top_k_for_spec_with_matvec(
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
        &CpuNativeMatvecBackend,
    )
}

pub fn native_lm_head_logits_for_spec_with_matvec(
    store: &SafeTensorShardStore,
    spec: &NativeTextModelSpec,
    hidden_states: &[f32],
    chunk_rows: usize,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    match spec {
        NativeTextModelSpec::Qwen(spec) => {
            qwen_lm_head_logits_for_spec_with_matvec(store, spec, hidden_states, chunk_rows, matvec)
        }
        NativeTextModelSpec::Gemma(spec) => gemma_lm_head_logits_for_spec_with_matvec(
            store,
            spec,
            hidden_states,
            chunk_rows,
            matvec,
        ),
    }
}

fn cache_family_mismatch(
    operation: &str,
    spec: &NativeTextModelSpec,
    caches: &NativeTextLayerCaches,
) -> TensorLoadError {
    TensorLoadError::unsupported(format!(
        "native text {operation} received `{}` caches for `{}` spec",
        caches.family_slug(),
        spec.family().canonical_slug(),
    ))
}
