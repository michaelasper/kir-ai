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
use llm_models::{GemmaModelSpec, ModelFamily, NativeTextModelSpec, QwenModelSpec};

#[derive(Debug, Clone, Copy)]
pub enum NativeTextModelSpecRef<'a> {
    Qwen(&'a QwenModelSpec),
    Gemma(&'a GemmaModelSpec),
}

impl NativeTextModelSpecRef<'_> {
    pub fn family(self) -> ModelFamily {
        match self {
            Self::Qwen(spec) => spec.family,
            Self::Gemma(spec) => spec.family,
        }
    }

    pub fn vocab_size(self) -> u32 {
        match self {
            Self::Qwen(spec) => spec.vocab_size,
            Self::Gemma(spec) => spec.vocab_size,
        }
    }
}

impl<'a> From<&'a NativeTextModelSpec> for NativeTextModelSpecRef<'a> {
    fn from(value: &'a NativeTextModelSpec) -> Self {
        match value {
            NativeTextModelSpec::Qwen(spec) => Self::Qwen(spec),
            NativeTextModelSpec::Gemma(spec) => Self::Gemma(spec),
        }
    }
}

impl<'a> From<&'a QwenModelSpec> for NativeTextModelSpecRef<'a> {
    fn from(value: &'a QwenModelSpec) -> Self {
        Self::Qwen(value)
    }
}

impl<'a> From<&'a GemmaModelSpec> for NativeTextModelSpecRef<'a> {
    fn from(value: &'a GemmaModelSpec) -> Self {
        Self::Gemma(value)
    }
}

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

    pub fn as_mut_refs(&mut self) -> NativeTextLayerCachesMut<'_> {
        match self {
            Self::Qwen(caches) => NativeTextLayerCachesMut::Qwen(caches.as_mut_slice()),
            Self::Gemma(caches) => NativeTextLayerCachesMut::Gemma(caches.as_mut_slice()),
        }
    }
}

#[derive(Debug)]
pub enum NativeTextLayerCachesMut<'a> {
    Qwen(&'a mut [QwenLayerCache]),
    Gemma(&'a mut [GemmaLayerCache]),
}

impl NativeTextLayerCachesMut<'_> {
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

pub async fn native_prefill_sequence_with_cache(
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
    .await
}

pub async fn native_prefill_sequence_with_cache_with_matvec(
    store: &SafeTensorShardStore,
    spec: &NativeTextModelSpec,
    token_ids: &[usize],
    caches: &mut NativeTextLayerCaches,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    native_prefill_sequence_with_cache_for_spec_ref_with_matvec(
        store,
        spec.into(),
        token_ids,
        caches.as_mut_refs(),
        matvec,
    )
    .await
}

pub async fn native_prefill_sequence_with_cache_for_spec_ref_with_matvec(
    store: &SafeTensorShardStore,
    spec: NativeTextModelSpecRef<'_>,
    token_ids: &[usize],
    caches: NativeTextLayerCachesMut<'_>,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    let cache_family = caches.family_slug();
    match (spec, caches) {
        (NativeTextModelSpecRef::Qwen(spec), NativeTextLayerCachesMut::Qwen(caches)) => {
            qwen_prefill_sequence_with_cache_with_matvec(store, spec, token_ids, caches, matvec).await
        }
        (NativeTextModelSpecRef::Gemma(spec), NativeTextLayerCachesMut::Gemma(caches)) => {
            gemma_prefill_sequence_with_cache_with_matvec(store, spec, token_ids, caches, matvec)
                .await
        }
        (spec, _) => Err(cache_family_mismatch("prefill", spec, cache_family)),
    }
}

pub async fn native_decode_token_with_cache(
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
    .await
}

pub async fn native_decode_token_with_cache_with_matvec(
    store: &SafeTensorShardStore,
    spec: &NativeTextModelSpec,
    token_id: usize,
    caches: &mut NativeTextLayerCaches,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    native_decode_token_with_cache_for_spec_ref_with_matvec(
        store,
        spec.into(),
        token_id,
        caches.as_mut_refs(),
        matvec,
    )
    .await
}

pub async fn native_decode_token_with_cache_for_spec_ref_with_matvec(
    store: &SafeTensorShardStore,
    spec: NativeTextModelSpecRef<'_>,
    token_id: usize,
    caches: NativeTextLayerCachesMut<'_>,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    let cache_family = caches.family_slug();
    match (spec, caches) {
        (NativeTextModelSpecRef::Qwen(spec), NativeTextLayerCachesMut::Qwen(caches)) => {
            qwen_decode_token_with_cache_with_matvec(store, spec, token_id, caches, matvec).await
        }
        (NativeTextModelSpecRef::Gemma(spec), NativeTextLayerCachesMut::Gemma(caches)) => {
            gemma_decode_token_with_cache_with_matvec(store, spec, token_id, caches, matvec).await
        }
        (spec, _) => Err(cache_family_mismatch("decode", spec, cache_family)),
    }
}

pub async fn native_final_norm_for_spec(
    store: &SafeTensorShardStore,
    spec: &NativeTextModelSpec,
    hidden_states: &[f32],
) -> Result<Vec<f32>, TensorLoadError> {
    native_final_norm_for_spec_with_matvec(store, spec, hidden_states, &CpuNativeMatvecBackend).await
}

pub async fn native_final_norm_for_spec_with_matvec(
    store: &SafeTensorShardStore,
    spec: &NativeTextModelSpec,
    hidden_states: &[f32],
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    native_final_norm_for_spec_ref_with_matvec(store, spec.into(), hidden_states, matvec).await
}

pub async fn native_final_norm_for_spec_ref_with_matvec(
    store: &SafeTensorShardStore,
    spec: NativeTextModelSpecRef<'_>,
    hidden_states: &[f32],
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    match spec {
        NativeTextModelSpecRef::Qwen(spec) => {
            qwen_final_norm_for_spec_with_matvec(store, spec, hidden_states, matvec).await
        }
        NativeTextModelSpecRef::Gemma(spec) => {
            gemma_final_norm_for_spec(store, spec, hidden_states).await
        }
    }
}

pub async fn native_lm_head_top_k_for_spec(
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
    .await
}

pub async fn native_lm_head_top_k_for_spec_with_matvec(
    store: &SafeTensorShardStore,
    spec: &NativeTextModelSpec,
    hidden_states: &[f32],
    top_k: usize,
    chunk_rows: usize,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<TopKLogit>, TensorLoadError> {
    native_lm_head_top_k_for_spec_ref_with_matvec(
        store,
        spec.into(),
        hidden_states,
        top_k,
        chunk_rows,
        matvec,
    )
    .await
}

pub async fn native_lm_head_top_k_for_spec_ref_with_matvec(
    store: &SafeTensorShardStore,
    spec: NativeTextModelSpecRef<'_>,
    hidden_states: &[f32],
    top_k: usize,
    chunk_rows: usize,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<TopKLogit>, TensorLoadError> {
    match spec {
        NativeTextModelSpecRef::Qwen(spec) => {
            qwen_lm_head_top_k_for_spec_with_matvec(store, spec, hidden_states, top_k, chunk_rows, matvec)
                .await
        }
        NativeTextModelSpecRef::Gemma(spec) => {
            gemma_lm_head_top_k_for_spec_with_matvec(store, spec, hidden_states, top_k, chunk_rows, matvec)
                .await
        }
    }
}

pub async fn native_lm_head_logits_for_spec(
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
    .await
}

pub async fn native_lm_head_logits_for_spec_with_matvec(
    store: &SafeTensorShardStore,
    spec: &NativeTextModelSpec,
    hidden_states: &[f32],
    chunk_rows: usize,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    native_lm_head_logits_for_spec_ref_with_matvec(
        store,
        spec.into(),
        hidden_states,
        chunk_rows,
        matvec,
    )
    .await
}

pub async fn native_lm_head_logits_for_spec_ref_with_matvec(
    store: &SafeTensorShardStore,
    spec: NativeTextModelSpecRef<'_>,
    hidden_states: &[f32],
    chunk_rows: usize,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    match spec {
        NativeTextModelSpecRef::Qwen(spec) => {
            qwen_lm_head_logits_for_spec_with_matvec(store, spec, hidden_states, chunk_rows, matvec).await
        }
        NativeTextModelSpecRef::Gemma(spec) => {
            gemma_lm_head_logits_for_spec_with_matvec(store, spec, hidden_states, chunk_rows, matvec).await
        }
    }
}

fn cache_family_mismatch(
    operation: &str,
    spec: NativeTextModelSpecRef<'_>,
    cache_family: &str,
) -> TensorLoadError {
    TensorLoadError::unsupported(format!(
        "native text {operation} received `{}` caches for `{}` spec",
        cache_family,
        spec.family().canonical_slug(),
    ))
}
