use llm_backend::{
    BackendError, NativeMatvecBackend, NativeTextModelSpecRef, SafeTensorShardStore,
    SamplingConfig, native_final_norm_for_spec_ref_with_matvec,
    native_lm_head_logits_for_spec_ref_with_matvec, native_lm_head_top_k_for_spec_ref_with_matvec,
};
use llm_sampler::TopPSamplerScratch;
use rand::{Rng as _, SeedableRng, rngs::SmallRng};
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

#[cfg(test)]
pub(crate) fn sample_token_id_with_draw(
    logits: &[f32],
    sampling: SamplingConfig,
    draw: f32,
    family_display_name: &str,
) -> Result<usize, BackendError> {
    let mut scratch = TopPSamplerScratch::new();
    sample_token_id_with_draw_with_scratch(
        logits,
        sampling,
        draw,
        family_display_name,
        &mut scratch,
    )
}

pub(crate) fn sample_token_id_with_draw_with_scratch(
    logits: &[f32],
    sampling: SamplingConfig,
    draw: f32,
    family_display_name: &str,
    top_p_scratch: &mut TopPSamplerScratch,
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
                .sample_with_scratch(logits, draw, top_p_scratch)
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
        sampling_draw: Option<f32>,
        top_p_scratch: &mut TopPSamplerScratch,
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
            let sampling_draw = sampling_draw.ok_or_else(|| {
                BackendError::Other(format!(
                    "{} non-greedy sampling requires an RNG draw",
                    self.family_display_name
                ))
            })?;
            let sampled_token_id = sample_token_id_with_draw_with_scratch(
                &logits,
                sampling,
                sampling_draw,
                self.family_display_name,
                top_p_scratch,
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

static NATIVE_TEXT_SAMPLING_SEED_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(crate) struct NativeTextSamplingRng {
    rng: SmallRng,
}

impl NativeTextSamplingRng {
    pub(crate) fn from_entropy() -> Self {
        let mut seed_bytes = [0_u8; 32];
        if getrandom::fill(&mut seed_bytes).is_ok() {
            Self::from_seed_bytes(seed_bytes)
        } else {
            Self::from_seed_words_inner(fallback_entropy_seed_words())
        }
    }

    #[cfg(test)]
    fn from_seed_words(state: [u64; 4]) -> Self {
        Self::from_seed_words_inner(state)
    }

    fn from_seed_words_inner(mut state: [u64; 4]) -> Self {
        if state.iter().all(|word| *word == 0) {
            let mut seed = 0x9E37_79B9_7F4A_7C15;
            state = [
                splitmix64_next(&mut seed),
                splitmix64_next(&mut seed),
                splitmix64_next(&mut seed),
                splitmix64_next(&mut seed),
            ];
        }
        Self::from_seed_bytes(seed_words_to_bytes(state))
    }

    fn from_seed_bytes(seed_bytes: [u8; 32]) -> Self {
        Self {
            rng: SmallRng::from_seed(seed_bytes),
        }
    }

    pub(crate) fn draw_f32(&mut self) -> f32 {
        self.rng.random::<f32>()
    }
}

fn seed_words_to_bytes(words: [u64; 4]) -> [u8; 32] {
    let mut seed_bytes = [0_u8; 32];
    for (chunk, word) in seed_bytes.chunks_exact_mut(8).zip(words) {
        chunk.copy_from_slice(&word.to_le_bytes());
    }
    seed_bytes
}

fn fallback_entropy_seed_words() -> [u64; 4] {
    let time_seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0);
    let counter = NATIVE_TEXT_SAMPLING_SEED_COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut seed = time_seed
        ^ counter.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ u64::from(std::process::id()).rotate_left(32);
    [
        splitmix64_next(&mut seed),
        splitmix64_next(&mut seed),
        splitmix64_next(&mut seed),
        splitmix64_next(&mut seed),
    ]
}

fn splitmix64_next(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *seed;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_text_sampling_rng_uses_independent_seeded_streams() {
        let mut first = NativeTextSamplingRng::from_seed_words([1, 2, 3, 4]);
        let mut first_again = NativeTextSamplingRng::from_seed_words([1, 2, 3, 4]);
        let mut second = NativeTextSamplingRng::from_seed_words([5, 6, 7, 8]);

        let first_draws = (0..8).map(|_| first.draw_f32()).collect::<Vec<_>>();
        let first_again_draws = (0..8).map(|_| first_again.draw_f32()).collect::<Vec<_>>();
        let second_draws = (0..8).map(|_| second.draw_f32()).collect::<Vec<_>>();

        assert_eq!(first_draws, first_again_draws);
        assert_ne!(first_draws, second_draws);
        assert!(
            first_draws
                .iter()
                .chain(second_draws.iter())
                .all(|draw| (0.0..1.0).contains(draw))
        );
    }
}
