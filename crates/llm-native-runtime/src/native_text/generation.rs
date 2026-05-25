use llm_backend::native::{
    NativeMatvecBackend, NativeTextModelSpecRef, SafeTensorShardStore, TopKLogit,
    native_final_norm_for_spec_ref, native_lm_head_top_k_for_spec_ref,
};
use llm_backend_contracts::{BackendError, SamplingConfig};
use llm_sampler::TopPSamplerScratch;
use rand::{Rng as _, SeedableRng, rngs::SmallRng};
use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};
#[cfg(test)]
use tokio_util::sync::CancellationToken;

const NATIVE_TEXT_TOP_P_PREFILTER_TOP_K: usize = 256;

pub(crate) fn resolve_native_text_max_tokens(
    requested: Option<u32>,
    configured_max: u32,
    family_display_name: &str,
) -> Result<u32, BackendError> {
    let configured_max = configured_max.max(1);
    match requested {
        None => Ok(configured_max),
        Some(0) => Err(BackendError::unsupported_request(
            "max_tokens must be greater than 0".to_owned(),
        )),
        Some(value) if value > configured_max => Err(BackendError::unsupported_request(format!(
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
        BackendError::config(format!(
            "native {family_display_name} max_position_embeddings does not fit usize: {err}"
        ))
    })?;
    if max_position_embeddings == 0 {
        return Err(BackendError::config(format!(
            "native {family_display_name} model declares zero max_position_embeddings"
        )));
    }
    let max_new_tokens = usize::try_from(max_new_tokens).map_err(|err| {
        BackendError::config(format!(
            "native {family_display_name} max_new_tokens does not fit usize: {err}"
        ))
    })?;
    let requested_context = context_tokens.checked_add(max_new_tokens).ok_or_else(|| {
        BackendError::unsupported_request(format!(
            "native {family_display_name} context length plus generation budget overflows usize"
        ))
    })?;
    if requested_context > max_position_embeddings {
        return Err(BackendError::unsupported_request(format!(
            "native {family_display_name} request needs {context_tokens} prompt tokens plus {max_new_tokens} generation tokens, exceeding model context limit {max_position_embeddings}"
        )));
    }
    let required = requested_context.max(min_cache_tokens.max(1));
    Ok(required.min(max_position_embeddings))
}

pub(crate) fn native_text_cache_namespace_token_bucket(
    cache_tokens: usize,
    max_position_embeddings: u32,
    family_display_name: &str,
) -> Result<usize, BackendError> {
    let max_position_embeddings = usize::try_from(max_position_embeddings).map_err(|err| {
        BackendError::config(format!(
            "native {family_display_name} max_position_embeddings does not fit usize: {err}"
        ))
    })?;
    Ok(cache_tokens
        .checked_next_power_of_two()
        .unwrap_or(max_position_embeddings)
        .min(max_position_embeddings))
}

#[cfg(test)]
use llm_backend::native::InferenceScratchpad;

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
        return Err(BackendError::cancelled());
    }
    let mut hidden = None;
    for chunk in context_tokens.chunks(prefill_chunk_tokens.max(1)) {
        if cancellation.is_cancelled() {
            return Err(BackendError::cancelled());
        }
        let hidden_states = prefill_chunk(chunk, caches, scratch)?;
        if cancellation.is_cancelled() {
            return Err(BackendError::cancelled());
        }
        hidden = hidden_states.last().cloned();
    }
    hidden.ok_or_else(|| {
        BackendError::internal_invariant(format!(
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
        return Err(BackendError::sampler(format!(
            "{family_display_name} lm head returned no logits"
        )));
    }
    match sampling {
        SamplingConfig::Greedy => llm_sampler::GreedySampler
            .sample(logits)
            .map_err(|err| BackendError::sampler(err.to_string())),
        SamplingConfig::TopP { temperature, top_p } => {
            llm_sampler::TopPSampler { temperature, top_p }
                .sample_with_scratch(logits, draw, top_p_scratch)
                .map_err(|err| BackendError::sampler(err.to_string()))
        }
        _ => Err(BackendError::invalid_sampling_config(
            "unsupported sampling configuration",
        )),
    }
}

fn native_text_lm_head_candidate_count(
    configured_top_k: usize,
    vocab_size: usize,
    sampling: SamplingConfig,
) -> usize {
    let candidate_top_k = if sampling.is_greedy() {
        configured_top_k
    } else {
        // Top-p needs enough headroom to represent a useful nucleus without
        // falling back to a full-vocab logits buffer on every sampled token.
        configured_top_k.max(NATIVE_TEXT_TOP_P_PREFILTER_TOP_K)
    };
    candidate_top_k.min(vocab_size).max(1)
}

fn sample_token_id_from_top_logits_with_draw(
    top_logits: &[TopKLogit],
    sampling: SamplingConfig,
    sampling_draw: Option<f32>,
    family_display_name: &str,
    top_p_scratch: &mut TopPSamplerScratch,
) -> Result<usize, BackendError> {
    if top_logits.is_empty() {
        return Err(BackendError::sampler(format!(
            "{family_display_name} lm head returned no logits"
        )));
    }

    match sampling {
        SamplingConfig::Greedy => Ok(top_logits[0].index),
        SamplingConfig::TopP { .. } => {
            let sampling_draw = sampling_draw.ok_or_else(|| {
                BackendError::sampler(format!(
                    "{family_display_name} non-greedy sampling requires an RNG draw"
                ))
            })?;
            let candidate_logits = top_logits
                .iter()
                .map(|candidate| candidate.logit)
                .collect::<Vec<_>>();
            let candidate_index = sample_token_id_with_draw_with_scratch(
                &candidate_logits,
                sampling,
                sampling_draw,
                family_display_name,
                top_p_scratch,
            )?;
            top_logits
                .get(candidate_index)
                .map(|candidate| candidate.index)
                .ok_or_else(|| {
                    BackendError::sampler(format!(
                        "{family_display_name} sampled invalid top-k candidate index {candidate_index}"
                    ))
                })
        }
        _ => Err(BackendError::invalid_sampling_config(
            "unsupported sampling configuration",
        )),
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
        let final_norm = native_final_norm_for_spec_ref(self.store, self.spec, hidden, self.matvec)
            .await
            .map_err(BackendError::from)?;
        let top_k = native_text_lm_head_candidate_count(
            self.top_k,
            self.spec.vocab_size() as usize,
            sampling,
        );
        let top_logits = native_lm_head_top_k_for_spec_ref(
            self.store,
            self.spec,
            &final_norm,
            top_k,
            self.chunk_rows,
            self.matvec,
        )
        .await
        .map_err(BackendError::from)?;
        let token_id = sample_token_id_from_top_logits_with_draw(
            &top_logits,
            sampling,
            sampling_draw,
            self.family_display_name,
            top_p_scratch,
        )?;
        ensure_token_id_fits_u32(token_id, self.family_display_name)?;
        Ok(token_id)
    }
}

fn ensure_token_id_fits_u32(
    token_id: usize,
    family_display_name: &str,
) -> Result<(), BackendError> {
    u32::try_from(token_id).map_err(|err| {
        BackendError::internal_invariant(format!(
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
    use llm_backend::native::{CpuNativeMatvecBackend, MathError, TensorLoadError, TopKLogit};
    use llm_models::{ModelFamily, QwenModelSpec};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test]
    async fn top_p_sampling_uses_prefiltered_lm_head_candidates() {
        let snapshot = llm_test_support::safetensors::temp_snapshot_dir(
            "llm-native-runtime",
            "top-p-prefilter",
        );
        std::fs::remove_dir_all(&snapshot).ok();
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");
        let spec = zero_layer_qwen_spec(1, 300);
        let mut lm_head = vec![-100.0_f32; spec.vocab_size as usize];
        lm_head[42] = 100.0;
        lm_head[17] = 99.0;
        llm_test_support::safetensors::TinySafetensorsSnapshot::new()
            .with_bf16_tensor(
                "model.safetensors",
                spec.final_norm_weight(),
                vec![1],
                vec![0.0],
            )
            .with_bf16_tensor(
                "model.safetensors",
                spec.lm_head_weight(),
                vec![spec.vocab_size as usize, 1],
                lm_head,
            )
            .write(&snapshot)
            .expect("write snapshot");

        let store = SafeTensorShardStore::open(&snapshot).expect("store opens");
        let matvec = RecordingLmHeadMatvec::default();
        let mut scratch = TopPSamplerScratch::new();

        let token_id = NativeTextNextTokenContext {
            store: &store,
            spec: (&spec).into(),
            top_k: 2,
            chunk_rows: 64,
            matvec: &matvec,
            family_display_name: "Qwen",
        }
        .select_next_token(
            &[1.0],
            SamplingConfig::TopP {
                temperature: 1.0,
                top_p: 0.5,
            },
            Some(0.0),
            &mut scratch,
        )
        .await
        .expect("top-p token");

        assert_eq!(token_id, 42);
        assert_eq!(matvec.full_vocab_lm_head_calls(), 0);
        assert_eq!(matvec.top_k_lm_head_calls(), 1);
        assert_eq!(matvec.last_lm_head_top_k(), 256);
        std::fs::remove_dir_all(snapshot).ok();
    }

    #[derive(Default)]
    struct RecordingLmHeadMatvec {
        full_vocab_lm_head_calls: AtomicUsize,
        top_k_lm_head_calls: AtomicUsize,
        last_lm_head_top_k: AtomicUsize,
    }

    impl RecordingLmHeadMatvec {
        fn full_vocab_lm_head_calls(&self) -> usize {
            self.full_vocab_lm_head_calls.load(Ordering::SeqCst)
        }

        fn top_k_lm_head_calls(&self) -> usize {
            self.top_k_lm_head_calls.load(Ordering::SeqCst)
        }

        fn last_lm_head_top_k(&self) -> usize {
            self.last_lm_head_top_k.load(Ordering::SeqCst)
        }
    }

    impl NativeMatvecBackend for RecordingLmHeadMatvec {
        async fn bf16_matvec_row_major_f32_in_place(
            &self,
            store: &SafeTensorShardStore,
            tensor: &str,
            input: &[f32],
            output: &mut [f32],
        ) -> Result<(), TensorLoadError> {
            CpuNativeMatvecBackend
                .bf16_matvec_row_major_f32_in_place(store, tensor, input, output)
                .await
        }

        async fn bf16_matvec_rows_f32_in_place(
            &self,
            store: &SafeTensorShardStore,
            tensor: &str,
            input: &[f32],
            chunk_rows: usize,
            output: &mut [f32],
        ) -> Result<(), TensorLoadError> {
            if tensor == "lm_head.weight" {
                self.full_vocab_lm_head_calls.fetch_add(1, Ordering::SeqCst);
            }
            CpuNativeMatvecBackend
                .bf16_matvec_rows_f32_in_place(store, tensor, input, chunk_rows, output)
                .await
        }

        async fn bf16_matvec_top_k_rows_f32(
            &self,
            store: &SafeTensorShardStore,
            tensor: &str,
            input: &[f32],
            top_k: usize,
            chunk_rows: usize,
        ) -> Result<Vec<TopKLogit>, TensorLoadError> {
            if tensor == "lm_head.weight" {
                self.top_k_lm_head_calls.fetch_add(1, Ordering::SeqCst);
                self.last_lm_head_top_k.store(top_k, Ordering::SeqCst);
            }
            CpuNativeMatvecBackend
                .bf16_matvec_top_k_rows_f32(store, tensor, input, top_k, chunk_rows)
                .await
        }

        async fn matvec_row_major_f32_in_place(
            &self,
            input: &[f32],
            weights: &[f32],
            rows: usize,
            columns: usize,
            output: &mut [f32],
        ) -> Result<(), MathError> {
            CpuNativeMatvecBackend
                .matvec_row_major_f32_in_place(input, weights, rows, columns, output)
                .await
        }

        async fn rms_norm_one_centered_f32_in_place(
            &self,
            input: &[f32],
            weight: &[f32],
            eps: f32,
            output: &mut [f32],
        ) -> Result<(), MathError> {
            CpuNativeMatvecBackend
                .rms_norm_one_centered_f32_in_place(input, weight, eps, output)
                .await
        }

        async fn softmax_f32_in_place(
            &self,
            scores: &[f32],
            output: &mut [f32],
        ) -> Result<(), MathError> {
            CpuNativeMatvecBackend
                .softmax_f32_in_place(scores, output)
                .await
        }

        async fn linear_attention_conv1d_silu_f32_in_place(
            &self,
            window: &[f32],
            weights: &[f32],
            conv_dim: usize,
            kernel_size: usize,
            output: &mut [f32],
        ) -> Result<(), MathError> {
            CpuNativeMatvecBackend
                .linear_attention_conv1d_silu_f32_in_place(
                    window,
                    weights,
                    conv_dim,
                    kernel_size,
                    output,
                )
                .await
        }

        async fn weighted_sum_f32_in_place(
            &self,
            values: &[f32],
            weights: &[f32],
            vector_len: usize,
            output: &mut [f32],
        ) -> Result<(), MathError> {
            CpuNativeMatvecBackend
                .weighted_sum_f32_in_place(values, weights, vector_len, output)
                .await
        }

        #[allow(clippy::too_many_arguments)]
        async fn linear_attention_recurrent_update_f32_in_place(
            &self,
            state: &[f32],
            key: &[f32],
            value: &[f32],
            memory: &[f32],
            beta: f32,
            decay: f32,
            key_head_dim: usize,
            value_head_dim: usize,
            output: &mut [f32],
        ) -> Result<(), MathError> {
            CpuNativeMatvecBackend
                .linear_attention_recurrent_update_f32_in_place(
                    state,
                    key,
                    value,
                    memory,
                    beta,
                    decay,
                    key_head_dim,
                    value_head_dim,
                    output,
                )
                .await
        }

        async fn select_head_rows_f32_in_place(
            &self,
            values: &[f32],
            row_count: usize,
            row_len: usize,
            head_start: usize,
            head_len: usize,
            output: &mut [f32],
        ) -> Result<(), MathError> {
            CpuNativeMatvecBackend
                .select_head_rows_f32_in_place(
                    values, row_count, row_len, head_start, head_len, output,
                )
                .await
        }
    }

    fn zero_layer_qwen_spec(hidden_size: u32, vocab_size: u32) -> QwenModelSpec {
        QwenModelSpec {
            family: ModelFamily::Qwen,
            architecture: "Qwen3_5MoeForConditionalGeneration".to_owned(),
            model_type: "qwen3_5_moe".to_owned(),
            text_model_type: "qwen3_5_moe_text".to_owned(),
            hidden_size,
            rms_norm_eps: 0.0,
            tie_word_embeddings: false,
            rope_theta: 1_000_000.0,
            partial_rotary_factor: 1.0,
            num_hidden_layers: 0,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: hidden_size,
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_key_head_dim: 1,
            linear_value_head_dim: hidden_size,
            linear_conv_kernel_dim: 1,
            num_experts: 1,
            num_experts_per_tok: 1,
            moe_intermediate_size: 1,
            shared_expert_intermediate_size: 1,
            max_position_embeddings: 1,
            vocab_size,
            layer_kinds: Vec::new(),
        }
    }

    fn assert_unsupported_message(err: BackendError, expected: &str) {
        assert!(
            err.is_unsupported_request(),
            "expected UnsupportedRequest, got {err}"
        );
        assert_eq!(err.to_string(), expected);
    }

    #[test]
    fn native_text_cache_token_capacity_rejects_zero_max_tokens_consistently() {
        let qwen = resolve_native_text_max_tokens(Some(0), 8, "Qwen")
            .expect_err("zero Qwen max_tokens fails closed");
        let gemma = resolve_native_text_max_tokens(Some(0), 8, "Gemma")
            .expect_err("zero Gemma max_tokens fails closed");

        assert_eq!(qwen, gemma);
        assert_unsupported_message(
            qwen,
            "unsupported backend request: max_tokens must be greater than 0",
        );
    }

    #[test]
    fn native_text_cache_token_capacity_clamps_zero_configured_generation_limit() {
        assert_eq!(
            resolve_native_text_max_tokens(None, 0, "Qwen")
                .expect("omitted max_tokens clamps to one token"),
            1
        );
        assert_eq!(
            resolve_native_text_max_tokens(Some(1), 0, "Gemma")
                .expect("one requested token fits clamped limit"),
            1
        );

        let err = resolve_native_text_max_tokens(Some(2), 0, "Gemma")
            .expect_err("request above clamped native limit fails closed");
        assert_unsupported_message(
            err,
            "unsupported backend request: requested max_tokens 2 exceeds configured native Gemma limit 1",
        );
    }

    #[test]
    fn native_text_cache_token_capacity_formats_limit_errors_by_family() {
        let qwen = resolve_native_text_max_tokens(Some(9), 8, "Qwen")
            .expect_err("Qwen request above configured limit fails closed");
        let gemma = resolve_native_text_max_tokens(Some(9), 8, "Gemma")
            .expect_err("Gemma request above configured limit fails closed");

        assert_unsupported_message(
            qwen,
            "unsupported backend request: requested max_tokens 9 exceeds configured native Qwen limit 8",
        );
        assert_unsupported_message(
            gemma,
            "unsupported backend request: requested max_tokens 9 exceeds configured native Gemma limit 8",
        );
    }

    #[test]
    fn native_text_cache_token_capacity_rejects_context_generation_overflow() {
        let err = native_text_cache_token_capacity(usize::MAX, 1, 1, u32::MAX, "Qwen")
            .expect_err("context plus generation overflow fails closed");

        assert_unsupported_message(
            err,
            "unsupported backend request: native Qwen context length plus generation budget overflows usize",
        );
    }

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
