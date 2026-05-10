use llm_backend::{
    BackendError, NativeMatvecBackend, NativeTextModelSpecRef, SafeTensorShardStore,
    SamplingConfig, native_final_norm_for_spec_ref_with_matvec,
    native_lm_head_logits_for_spec_ref_with_matvec, native_lm_head_top_k_for_spec_ref_with_matvec,
};
use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};
#[cfg(test)]
use tokio_util::sync::CancellationToken;

pub(crate) fn resolve_native_text_max_tokens(
    requested: Option<u32>,
    configured_max: u32,
    family_display_name: &str,
) -> Result<u32, BackendError> {
    let configured_max = configured_max.max(1);
    match requested {
        None => Ok(configured_max),
        Some(0) => Err(BackendError::UnsupportedRequest(
            "max_tokens must be greater than 0".to_owned(),
        )),
        Some(value) if value > configured_max => Err(BackendError::UnsupportedRequest(format!(
            "requested max_tokens {value} exceeds configured native {family_display_name} limit {configured_max}"
        ))),
        Some(value) => Ok(value),
    }
}

pub(crate) fn native_text_cache_token_capacity(
    context_tokens: usize,
    max_new_tokens: u32,
    min_cache_tokens: usize,
    max_position_embeddings: u32,
    family_display_name: &str,
) -> Result<usize, BackendError> {
    let max_position_embeddings = usize::try_from(max_position_embeddings).map_err(|err| {
        BackendError::Other(format!(
            "native {family_display_name} max_position_embeddings does not fit usize: {err}"
        ))
    })?;
    if max_position_embeddings == 0 {
        return Err(BackendError::UnsupportedRequest(format!(
            "native {family_display_name} model declares zero max_position_embeddings"
        )));
    }
    let max_new_tokens = usize::try_from(max_new_tokens).map_err(|err| {
        BackendError::Other(format!(
            "native {family_display_name} max_new_tokens does not fit usize: {err}"
        ))
    })?;
    let requested_context = context_tokens.checked_add(max_new_tokens).ok_or_else(|| {
        BackendError::UnsupportedRequest(format!(
            "native {family_display_name} context length plus generation budget overflows usize"
        ))
    })?;
    if requested_context > max_position_embeddings {
        return Err(BackendError::UnsupportedRequest(format!(
            "native {family_display_name} request needs {context_tokens} prompt tokens plus {max_new_tokens} generation tokens, exceeding model context limit {max_position_embeddings}"
        )));
    }
    let required = requested_context.max(min_cache_tokens.max(1));
    Ok(required
        .checked_next_power_of_two()
        .unwrap_or(max_position_embeddings)
        .min(max_position_embeddings))
}

#[cfg(test)]
use llm_backend::InferenceScratchpad;

#[cfg(test)]
pub(crate) fn native_text_prefill_context_with_cache<C, F>(
    family_display_name: &str,
    prefill_chunk_tokens: usize,
    context_tokens: &[usize],
    caches: &mut [C],
    cancellation: &CancellationToken,
    scratch: &mut InferenceScratchpad,
    mut prefill_chunk: F,
) -> Result<Vec<f32>, BackendError>
where
    F: FnMut(&[usize], &mut [C], &mut InferenceScratchpad) -> Result<Vec<Vec<f32>>, BackendError>,
{
    if cancellation.is_cancelled() {
        return Err(BackendError::Cancelled);
    }
    let mut hidden = None;
    for chunk in context_tokens.chunks(prefill_chunk_tokens.max(1)) {
        if cancellation.is_cancelled() {
            return Err(BackendError::Cancelled);
        }
        let hidden_states = prefill_chunk(chunk, caches, scratch)?;
        if cancellation.is_cancelled() {
            return Err(BackendError::Cancelled);
        }
        hidden = hidden_states.last().cloned();
    }
    hidden.ok_or_else(|| {
        BackendError::Other(format!(
            "{family_display_name} prefill returned no hidden states"
        ))
    })
}

pub(crate) fn sample_token_id_with_draw(
    logits: &[f32],
    sampling: SamplingConfig,
    draw: f32,
    family_display_name: &str,
) -> Result<usize, BackendError> {
    if logits.is_empty() {
        return Err(BackendError::Other(format!(
            "{family_display_name} lm head returned no logits"
        )));
    }
    match sampling {
        SamplingConfig::Greedy => llm_sampler::GreedySampler
            .sample(logits)
            .map_err(|err| BackendError::Other(err.to_string())),
        SamplingConfig::TopP { temperature, top_p } => {
            llm_sampler::TopPSampler { temperature, top_p }
                .sample(logits, draw)
                .map_err(|err| BackendError::Other(err.to_string()))
        }
    }
}

pub(crate) struct NativeTextNextTokenContext<'a, M: NativeMatvecBackend> {
    pub(crate) store: &'a SafeTensorShardStore,
    pub(crate) spec: NativeTextModelSpecRef<'a>,
    pub(crate) top_k: usize,
    pub(crate) chunk_rows: usize,
    pub(crate) matvec: &'a M,
    pub(crate) family_display_name: &'static str,
}

impl<M: NativeMatvecBackend> NativeTextNextTokenContext<'_, M> {
    pub(crate) async fn select_next_token(
        &self,
        hidden: &[f32],
        sampling: SamplingConfig,
    ) -> Result<usize, BackendError> {
        let final_norm =
            native_final_norm_for_spec_ref_with_matvec(self.store, self.spec, hidden, self.matvec)
                .await
                .map_err(|err| BackendError::Other(err.to_string()))?;
        if !sampling.is_greedy() {
            let logits = native_lm_head_logits_for_spec_ref_with_matvec(
                self.store,
                self.spec,
                &final_norm,
                self.chunk_rows,
                self.matvec,
            )
            .await
            .map_err(|err| BackendError::Other(err.to_string()))?;
            let sampled_token_id = sample_token_id_with_draw(
                &logits,
                sampling,
                native_text_sampling_draw(),
                self.family_display_name,
            )?;
            ensure_token_id_fits_u32(sampled_token_id, self.family_display_name)?;
            return Ok(sampled_token_id);
        }

        let top_k = self.top_k.min(self.spec.vocab_size() as usize).max(1);
        let top_logits = native_lm_head_top_k_for_spec_ref_with_matvec(
            self.store,
            self.spec,
            &final_norm,
            top_k,
            self.chunk_rows,
            self.matvec,
        )
        .await
        .map_err(|err| BackendError::Other(err.to_string()))?;
        let item = top_logits.into_iter().next().ok_or_else(|| {
            BackendError::Other(format!(
                "{} lm head returned no logits",
                self.family_display_name
            ))
        })?;
        ensure_token_id_fits_u32(item.index, self.family_display_name)?;
        Ok(item.index)
    }
}

fn ensure_token_id_fits_u32(
    token_id: usize,
    family_display_name: &str,
) -> Result<(), BackendError> {
    u32::try_from(token_id).map_err(|err| {
        BackendError::Other(format!(
            "{family_display_name} token id does not fit u32: {err}"
        ))
    })?;
    Ok(())
}

static NATIVE_TEXT_SAMPLING_COUNTER: AtomicU64 = AtomicU64::new(0);

fn native_text_sampling_draw() -> f32 {
    let time_seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0);
    let counter = NATIVE_TEXT_SAMPLING_COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut value = time_seed ^ counter.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    let bits = value.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40;
    (bits as f32) / ((1_u32 << 24) as f32)
}
