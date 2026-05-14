use crate::{
    CpuNativeMatvecBackend, GemmaLayerCache, InferenceScratchpad, NativeMatvecBackend,
    QwenLayerCache, SafeTensorShardStore, TensorLoadError, TopKLogit,
    gemma_decode_token_with_cache_with_matvec, gemma_final_norm_for_spec,
    gemma_layer_caches_for_spec, gemma_lm_head_logits_for_spec_with_matvec,
    gemma_lm_head_top_k_for_spec_with_matvec, gemma_prefill_sequence_with_cache_with_matvec,
    qwen_decode_token_with_cache_with_matvec, qwen_final_norm_for_spec_with_matvec,
    qwen_layer_caches_for_spec, qwen_lm_head_logits_for_spec_with_matvec,
    qwen_lm_head_top_k_for_spec_with_matvec, qwen_prefill_sequence_with_cache_with_matvec,
};
use llm_models::ModelFamily;

pub enum NativeTextModelSpec {
    Qwen(llm_models::QwenModelSpec),
    Gemma(llm_models::GemmaModelSpec),
}

impl NativeTextModelSpec {
    pub fn family(&self) -> ModelFamily {
        match self {
            Self::Qwen(_) => ModelFamily::Qwen,
            Self::Gemma(_) => ModelFamily::Gemma,
        }
    }
}

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
pub enum NativeTextModelSpecRef<'a> {
    Qwen(&'a llm_models::QwenModelSpec),
    Gemma(&'a llm_models::GemmaModelSpec),
}

impl<'a> From<&'a NativeTextModelSpec> for NativeTextModelSpecRef<'a> {
    fn from(spec: &'a NativeTextModelSpec) -> Self {
        match spec {
            NativeTextModelSpec::Qwen(spec) => Self::Qwen(spec),
            NativeTextModelSpec::Gemma(spec) => Self::Gemma(spec),
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
            Self::Qwen(_) => ModelFamily::Qwen,
            Self::Gemma(_) => ModelFamily::Gemma,
        }
    }

    pub fn vocab_size(&self) -> u32 {
        match self {
            Self::Qwen(spec) => spec.vocab_size,
            Self::Gemma(spec) => spec.vocab_size,
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

pub(crate) async fn native_prefill_sequence_with_cache(
    store: &SafeTensorShardStore,
    spec: &NativeTextModelSpec,
    token_ids: &[usize],
    caches: &mut NativeTextLayerCaches,
    scratch: &mut InferenceScratchpad,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    native_prefill_sequence_with_cache_with_matvec(
        store,
        spec,
        token_ids,
        caches,
        &CpuNativeMatvecBackend,
        scratch,
    )
    .await
}

pub(crate) async fn native_prefill_sequence_with_cache_with_matvec(
    store: &SafeTensorShardStore,
    spec: &NativeTextModelSpec,
    token_ids: &[usize],
    caches: &mut NativeTextLayerCaches,
    matvec: &impl NativeMatvecBackend,
    scratch: &mut InferenceScratchpad,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    native_prefill_sequence_with_cache_for_spec_ref_with_matvec(
        store,
        spec.into(),
        token_ids,
        caches.as_mut_refs(),
        matvec,
        scratch,
    )
    .await
}

pub async fn native_prefill_sequence_with_cache_for_spec_ref_with_matvec(
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
            qwen_prefill_sequence_with_cache_with_matvec(
                store, spec, token_ids, caches, matvec, scratch,
            )
            .await
        }
        (NativeTextModelSpecRef::Gemma(spec), NativeTextLayerCachesMut::Gemma(caches)) => {
            gemma_prefill_sequence_with_cache_with_matvec(
                store, spec, token_ids, caches, matvec, scratch,
            )
            .await
        }
        (spec, _) => Err(cache_family_mismatch("prefill", spec, cache_family)),
    }
}

pub(crate) async fn native_decode_token_with_cache(
    store: &SafeTensorShardStore,
    spec: &NativeTextModelSpec,
    token_id: usize,
    caches: &mut NativeTextLayerCaches,
    scratch: &mut InferenceScratchpad,
) -> Result<Vec<f32>, TensorLoadError> {
    native_decode_token_with_cache_with_matvec(
        store,
        spec,
        token_id,
        caches,
        &CpuNativeMatvecBackend,
        scratch,
    )
    .await
}

pub(crate) async fn native_decode_token_with_cache_with_matvec(
    store: &SafeTensorShardStore,
    spec: &NativeTextModelSpec,
    token_id: usize,
    caches: &mut NativeTextLayerCaches,
    matvec: &impl NativeMatvecBackend,
    scratch: &mut InferenceScratchpad,
) -> Result<Vec<f32>, TensorLoadError> {
    native_decode_token_with_cache_for_spec_ref_with_matvec(
        store,
        spec.into(),
        token_id,
        caches.as_mut_refs(),
        matvec,
        scratch,
    )
    .await
}

pub async fn native_decode_token_with_cache_for_spec_ref_with_matvec(
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
            qwen_decode_token_with_cache_with_matvec(store, spec, token_id, caches, matvec, scratch)
                .await
        }
        (NativeTextModelSpecRef::Gemma(spec), NativeTextLayerCachesMut::Gemma(caches)) => {
            gemma_decode_token_with_cache_with_matvec(
                store, spec, token_id, caches, matvec, scratch,
            )
            .await
        }
        (spec, _) => Err(cache_family_mismatch("decode", spec, cache_family)),
    }
}

pub(crate) async fn native_final_norm_for_spec(
    store: &SafeTensorShardStore,
    spec: &NativeTextModelSpec,
    hidden_states: &[f32],
) -> Result<Vec<f32>, TensorLoadError> {
    native_final_norm_for_spec_with_matvec(store, spec, hidden_states, &CpuNativeMatvecBackend)
        .await
}

pub(crate) async fn native_final_norm_for_spec_with_matvec(
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
            let mut output = vec![0.0; hidden_states.len()];
            gemma_final_norm_for_spec(store, spec, hidden_states, &mut output).await?;
            Ok(output)
        }
    }
}

pub(crate) async fn native_lm_head_top_k_for_spec(
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

pub(crate) async fn native_lm_head_top_k_for_spec_with_matvec(
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
            qwen_lm_head_top_k_for_spec_with_matvec(
                store,
                spec,
                hidden_states,
                top_k,
                chunk_rows,
                matvec,
            )
            .await
        }
        NativeTextModelSpecRef::Gemma(spec) => {
            gemma_lm_head_top_k_for_spec_with_matvec(
                store,
                spec,
                hidden_states,
                top_k,
                chunk_rows,
                matvec,
            )
            .await
        }
    }
}

pub(crate) async fn native_lm_head_logits_for_spec(
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

pub(crate) async fn native_lm_head_logits_for_spec_with_matvec(
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
            qwen_lm_head_logits_for_spec_with_matvec(store, spec, hidden_states, chunk_rows, matvec)
                .await
        }
        NativeTextModelSpecRef::Gemma(spec) => {
            let mut output = vec![0.0; spec.vocab_size as usize];
            gemma_lm_head_logits_for_spec_with_matvec(
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
