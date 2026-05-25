#![allow(dead_code)]
// Family-neutral native text wrappers are retained for cross-family tests and
// diagnostics even when production code enters through concrete family paths.

use super::{
    GemmaLayerCache, InferenceScratchpad, NativeMatvecBackend, QwenLayerCache,
    SafeTensorShardStore, TensorLoadError, TopKLogit, gemma_decode_token_with_cache,
    gemma_final_norm_for_spec, gemma_layer_caches_for_spec, gemma_lm_head_logits_for_spec,
    gemma_lm_head_top_k_for_spec, gemma_prefill_sequence_with_cache, qwen_decode_token_with_cache,
    qwen_final_norm_for_spec, qwen_layer_caches_for_spec, qwen_lm_head_logits_for_spec,
    qwen_lm_head_top_k_for_spec, qwen_prefill_sequence_with_cache,
};
use llm_models::{ModelFamily, ModelSpec};

pub use llm_models::NativeTextModelSpec;

#[non_exhaustive]
pub enum NativeTextLayerCaches {
    Qwen(Vec<QwenLayerCache>),
    Gemma(Vec<GemmaLayerCache>),
}

impl NativeTextLayerCaches {
    pub fn as_mut_refs(&mut self) -> NativeTextLayerCachesMut<'_> {
        match self {
            Self::Qwen(caches) => NativeTextLayerCachesMut::Qwen(caches),
            Self::Gemma(caches) => NativeTextLayerCachesMut::Gemma(caches),
        }
    }
}

#[non_exhaustive]
pub enum NativeTextLayerCachesMut<'a> {
    Qwen(&'a mut [QwenLayerCache]),
    Gemma(&'a mut [GemmaLayerCache]),
}

impl NativeTextLayerCachesMut<'_> {
    pub fn family_slug(&self) -> &'static str {
        match self {
            Self::Qwen(_) => ModelFamily::Qwen.canonical_slug(),
            Self::Gemma(_) => ModelFamily::Gemma.canonical_slug(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum NativeTextModelSpecRef<'a> {
    Qwen(&'a llm_models::QwenModelSpec),
    Gemma(&'a llm_models::GemmaModelSpec),
    Unsupported(ModelFamily),
}

impl<'a> From<&'a NativeTextModelSpec> for NativeTextModelSpecRef<'a> {
    fn from(spec: &'a NativeTextModelSpec) -> Self {
        match spec {
            NativeTextModelSpec::Qwen(spec) => Self::Qwen(spec),
            NativeTextModelSpec::Gemma(spec) => Self::Gemma(spec),
            _ => Self::Unsupported(spec.family()),
        }
    }
}

impl<'a> From<&'a llm_models::QwenModelSpec> for NativeTextModelSpecRef<'a> {
    fn from(spec: &'a llm_models::QwenModelSpec) -> Self {
        Self::Qwen(spec)
    }
}

impl<'a> From<&'a llm_models::GemmaModelSpec> for NativeTextModelSpecRef<'a> {
    fn from(spec: &'a llm_models::GemmaModelSpec) -> Self {
        Self::Gemma(spec)
    }
}

impl NativeTextModelSpecRef<'_> {
    pub fn family(&self) -> ModelFamily {
        match self {
            Self::Qwen(spec) => spec.family(),
            Self::Gemma(spec) => spec.family(),
            Self::Unsupported(family) => *family,
        }
    }

    pub fn vocab_size(&self) -> u32 {
        match self {
            Self::Qwen(spec) => spec.vocab_size(),
            Self::Gemma(spec) => spec.vocab_size(),
            Self::Unsupported(_) => 0,
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
        _ => Err(unsupported_native_text_spec(
            "cache allocation",
            spec.family(),
        )),
    }
}

pub(crate) async fn native_prefill_sequence_with_cache(
    store: &SafeTensorShardStore,
    spec: &NativeTextModelSpec,
    token_ids: &[usize],
    caches: &mut NativeTextLayerCaches,
    matvec: &impl NativeMatvecBackend,
    scratch: &mut InferenceScratchpad,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    native_prefill_sequence_with_cache_for_spec_ref(
        store,
        spec.into(),
        token_ids,
        caches.as_mut_refs(),
        matvec,
        scratch,
    )
    .await
}

pub async fn native_prefill_sequence_with_cache_for_spec_ref(
    store: &SafeTensorShardStore,
    spec: NativeTextModelSpecRef<'_>,
    token_ids: &[usize],
    caches: NativeTextLayerCachesMut<'_>,
    matvec: &impl NativeMatvecBackend,
    scratch: &mut InferenceScratchpad,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    let cache_family = caches.family_slug();
    match (spec, caches) {
        (NativeTextModelSpecRef::Qwen(spec), NativeTextLayerCachesMut::Qwen(caches)) => {
            qwen_prefill_sequence_with_cache(store, spec, token_ids, caches, matvec, scratch).await
        }
        (NativeTextModelSpecRef::Gemma(spec), NativeTextLayerCachesMut::Gemma(caches)) => {
            gemma_prefill_sequence_with_cache(store, spec, token_ids, caches, matvec, scratch).await
        }
        (spec, _) => Err(cache_family_mismatch("prefill", spec, cache_family)),
    }
}

pub(crate) async fn native_decode_token_with_cache(
    store: &SafeTensorShardStore,
    spec: &NativeTextModelSpec,
    token_id: usize,
    caches: &mut NativeTextLayerCaches,
    matvec: &impl NativeMatvecBackend,
    scratch: &mut InferenceScratchpad,
) -> Result<Vec<f32>, TensorLoadError> {
    native_decode_token_with_cache_for_spec_ref(
        store,
        spec.into(),
        token_id,
        caches.as_mut_refs(),
        matvec,
        scratch,
    )
    .await
}

pub async fn native_decode_token_with_cache_for_spec_ref(
    store: &SafeTensorShardStore,
    spec: NativeTextModelSpecRef<'_>,
    token_id: usize,
    caches: NativeTextLayerCachesMut<'_>,
    matvec: &impl NativeMatvecBackend,
    scratch: &mut InferenceScratchpad,
) -> Result<Vec<f32>, TensorLoadError> {
    let cache_family = caches.family_slug();
    match (spec, caches) {
        (NativeTextModelSpecRef::Qwen(spec), NativeTextLayerCachesMut::Qwen(caches)) => {
            qwen_decode_token_with_cache(store, spec, token_id, caches, matvec, scratch).await
        }
        (NativeTextModelSpecRef::Gemma(spec), NativeTextLayerCachesMut::Gemma(caches)) => {
            gemma_decode_token_with_cache(store, spec, token_id, caches, matvec, scratch).await
        }
        (spec, _) => Err(cache_family_mismatch("decode", spec, cache_family)),
    }
}

pub(crate) async fn native_final_norm_for_spec(
    store: &SafeTensorShardStore,
    spec: &NativeTextModelSpec,
    hidden_states: &[f32],
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    native_final_norm_for_spec_ref(store, spec.into(), hidden_states, matvec).await
}

pub async fn native_final_norm_for_spec_ref(
    store: &SafeTensorShardStore,
    spec: NativeTextModelSpecRef<'_>,
    hidden_states: &[f32],
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    match spec {
        NativeTextModelSpecRef::Qwen(spec) => {
            qwen_final_norm_for_spec(store, spec, hidden_states, matvec).await
        }
        NativeTextModelSpecRef::Gemma(spec) => {
            let mut output = vec![0.0; hidden_states.len()];
            gemma_final_norm_for_spec(store, spec, hidden_states, &mut output).await?;
            Ok(output)
        }
        NativeTextModelSpecRef::Unsupported(family) => {
            Err(unsupported_native_text_spec("final norm", family))
        }
    }
}

pub(crate) async fn native_lm_head_top_k_for_spec(
    store: &SafeTensorShardStore,
    spec: &NativeTextModelSpec,
    hidden_states: &[f32],
    top_k: usize,
    chunk_rows: usize,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<TopKLogit>, TensorLoadError> {
    native_lm_head_top_k_for_spec_ref(store, spec.into(), hidden_states, top_k, chunk_rows, matvec)
        .await
}

pub async fn native_lm_head_top_k_for_spec_ref(
    store: &SafeTensorShardStore,
    spec: NativeTextModelSpecRef<'_>,
    hidden_states: &[f32],
    top_k: usize,
    chunk_rows: usize,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<TopKLogit>, TensorLoadError> {
    match spec {
        NativeTextModelSpecRef::Qwen(spec) => {
            qwen_lm_head_top_k_for_spec(store, spec, hidden_states, top_k, chunk_rows, matvec).await
        }
        NativeTextModelSpecRef::Gemma(spec) => {
            gemma_lm_head_top_k_for_spec(store, spec, hidden_states, top_k, chunk_rows, matvec)
                .await
        }
        NativeTextModelSpecRef::Unsupported(family) => {
            Err(unsupported_native_text_spec("lm head top-k", family))
        }
    }
}

pub(crate) async fn native_lm_head_logits_for_spec(
    store: &SafeTensorShardStore,
    spec: &NativeTextModelSpec,
    hidden_states: &[f32],
    chunk_rows: usize,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    native_lm_head_logits_for_spec_ref(store, spec.into(), hidden_states, chunk_rows, matvec).await
}

pub async fn native_lm_head_logits_for_spec_ref(
    store: &SafeTensorShardStore,
    spec: NativeTextModelSpecRef<'_>,
    hidden_states: &[f32],
    chunk_rows: usize,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    match spec {
        NativeTextModelSpecRef::Qwen(spec) => {
            qwen_lm_head_logits_for_spec(store, spec, hidden_states, chunk_rows, matvec).await
        }
        NativeTextModelSpecRef::Gemma(spec) => {
            let mut output = vec![0.0; spec.vocab_size as usize];
            gemma_lm_head_logits_for_spec(
                store,
                spec,
                hidden_states,
                chunk_rows,
                matvec,
                &mut output,
            )
            .await?;
            Ok(output)
        }
        NativeTextModelSpecRef::Unsupported(family) => {
            Err(unsupported_native_text_spec("lm head logits", family))
        }
    }
}

fn unsupported_native_text_spec(operation: &str, family: ModelFamily) -> TensorLoadError {
    TensorLoadError::unsupported(format!(
        "native text {operation} does not support `{}` specs",
        family.canonical_slug()
    ))
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
