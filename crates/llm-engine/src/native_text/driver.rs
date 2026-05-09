use super::{
    NativeStreamTextDeltas, NativeTextPrefixCache, NativeTextPrefixCacheMetrics,
    NativeTextPrefixCacheNamespace, NativeTextPrefixCacheValue, native_text_cache_token_capacity,
    native_text_prefill_context_with_cache, native_text_worker_stream,
    resolve_native_text_max_tokens, send_backend_stream_chunk,
};
use crate::native_matvec::NativeTextCacheMirrorSource;
use async_trait::async_trait;
use futures::stream::BoxStream;
use llm_backend::{
    BackendError, BackendModelMetadata, BackendOutput, BackendRequest, BackendStreamChunk,
    ModelBackend, SamplingConfig,
};
use llm_tokenizer::HuggingFaceTokenizer;
use tokio_util::sync::CancellationToken;

pub(crate) trait NativeTextAdapter: Clone + Send + Sync + 'static {
    type DecodeSession: Send + 'static;
    type LayerCache: NativeTextPrefixCacheValue + NativeTextCacheMirrorSource + Send + 'static;

    fn family_display_name(&self) -> &'static str;
    fn worker_label(&self) -> &'static str;
    fn set_max_prefill_tokens(&mut self, max_prefill_tokens: usize);
    fn encode_prompt(
        &self,
        tokenizer: &HuggingFaceTokenizer,
        request: &BackendRequest,
    ) -> Result<Vec<u32>, BackendError>;
    fn decode_output(
        &self,
        tokenizer: &HuggingFaceTokenizer,
        output_ids: &[u32],
    ) -> Result<String, BackendError>;
    fn stop_tokens(&self) -> NativeTextStopTokens {
        NativeTextStopTokens::default()
    }
    fn observe_candidate(
        &self,
        tokenizer: &HuggingFaceTokenizer,
        _emitted_tokens: &[u32],
        token_id: usize,
    ) -> Result<NativeTextCandidateDecision, BackendError> {
        Ok(native_text_candidate_decision_for_stop_tokens(
            self.stop_tokens(),
            tokenizer,
            token_id,
        ))
    }
    fn max_position_embeddings(&self) -> u32;
    fn max_prefill_tokens(&self) -> usize;
    fn prefix_cache(&self) -> &NativeTextPrefixCache<Self::LayerCache>;
    fn prefix_cache_metrics(&self) -> &NativeTextPrefixCacheMetrics;
    fn prefix_cache_namespace(
        &self,
        request: &BackendRequest,
        cache_tokens: usize,
    ) -> NativeTextPrefixCacheNamespace;
    fn layer_count(&self) -> usize;
    fn allocate_caches(&self, cache_tokens: usize) -> Result<Vec<Self::LayerCache>, BackendError>;
    fn prefill_chunk_with_cache(
        &self,
        token_ids: &[usize],
        caches: &mut [Self::LayerCache],
    ) -> Result<Vec<Vec<f32>>, BackendError>;
    fn make_decode_session(
        &self,
        hidden: Vec<f32>,
        caches: Vec<Self::LayerCache>,
    ) -> Self::DecodeSession;
    fn cleanup_cache_mirrors(&self, _caches: &[Self::LayerCache]) {}
    fn hidden<'a>(&self, session: &'a Self::DecodeSession) -> &'a [f32];
    fn step(&self, session: &mut Self::DecodeSession, token_id: usize) -> Result<(), BackendError>;
    fn next_token_from_hidden(
        &self,
        hidden: &[f32],
        sampling: SamplingConfig,
    ) -> Result<usize, BackendError>;
}

pub(crate) enum NativeTextCandidateDecision {
    Emit(usize),
    Stop,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct NativeTextStopTokens {
    pub(crate) token_ids: &'static [usize],
    pub(crate) token_strings: &'static [&'static str],
}

impl NativeTextStopTokens {
    pub(crate) fn contains(self, tokenizer: &HuggingFaceTokenizer, token_id: usize) -> bool {
        self.token_ids.contains(&token_id)
            || self.token_strings.iter().any(|token| {
                tokenizer
                    .token_to_id(token)
                    .is_some_and(|stop_id| token_id == stop_id as usize)
            })
    }
}

fn native_text_candidate_decision_for_stop_tokens(
    stop_tokens: NativeTextStopTokens,
    tokenizer: &HuggingFaceTokenizer,
    token_id: usize,
) -> NativeTextCandidateDecision {
    if stop_tokens.contains(tokenizer, token_id) {
        NativeTextCandidateDecision::Stop
    } else {
        NativeTextCandidateDecision::Emit(token_id)
    }
}

struct NativeTextCacheMirrorCleanupGuard<'a, A>
where
    A: NativeTextAdapter,
{
    adapter: &'a A,
    disarmed: bool,
}

impl<'a, A> NativeTextCacheMirrorCleanupGuard<'a, A>
where
    A: NativeTextAdapter,
{
    fn new(adapter: &'a A) -> Self {
        Self {
            adapter,
            disarmed: false,
        }
    }

    fn disarm(&mut self) {
        self.disarmed = true;
    }

    fn cleanup(&self, caches: &[A::LayerCache]) {
        if !self.disarmed {
            self.adapter.cleanup_cache_mirrors(caches);
        }
    }
}

#[derive(Clone)]
pub(crate) struct NativeTextDriver<A>
where
    A: NativeTextAdapter,
{
    pub(crate) model_id: String,
    pub(crate) metadata: BackendModelMetadata,
    pub(crate) tokenizer: HuggingFaceTokenizer,
    pub(crate) adapter: A,
    pub(crate) max_new_tokens: u32,
}

impl<A> NativeTextDriver<A>
where
    A: NativeTextAdapter,
{
    pub(crate) fn new(
        model_id: String,
        metadata: BackendModelMetadata,
        tokenizer: HuggingFaceTokenizer,
        adapter: A,
        max_new_tokens: u32,
    ) -> Self {
        Self {
            model_id,
            metadata,
            tokenizer,
            adapter,
            max_new_tokens: max_new_tokens.max(1),
        }
    }

    pub(crate) fn with_max_new_tokens(mut self, max_new_tokens: u32) -> Self {
        self.max_new_tokens = max_new_tokens.max(1);
        self
    }

    pub(crate) fn with_max_prefill_tokens(mut self, max_prefill_tokens: usize) -> Self {
        self.adapter
            .set_max_prefill_tokens(max_prefill_tokens.max(1));
        self
    }

    pub(crate) fn generate_blocking(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        if cancellation.is_cancelled() {
            return Err(BackendError::Cancelled);
        }
        self.validate_model(&request)?;
        if cancellation.is_cancelled() {
            return Err(BackendError::Cancelled);
        }
        let prompt_tokens = self.encode_prompt(&request)?;
        let context_tokens = prompt_tokens
            .iter()
            .map(|token| *token as usize)
            .collect::<Vec<_>>();
        if context_tokens.is_empty() {
            return Err(BackendError::Other(format!(
                "{} prompt encoded to zero tokens",
                self.adapter.family_display_name()
            )));
        }
        let mut output_ids = Vec::new();
        let mut finish_reason = llm_api::FinishReason::Length;
        let requested = resolve_native_text_max_tokens(
            request.max_tokens,
            self.max_new_tokens,
            self.adapter.family_display_name(),
        )?;
        let mut decode =
            self.start_decode_session(&context_tokens, requested, &request, &cancellation)?;
        if cancellation.is_cancelled() {
            return Err(BackendError::Cancelled);
        }

        for step_idx in 0..requested {
            if cancellation.is_cancelled() {
                return Err(BackendError::Cancelled);
            }
            let candidate = self
                .adapter
                .next_token_from_hidden(self.adapter.hidden(&decode), request.sampling)?;
            if cancellation.is_cancelled() {
                return Err(BackendError::Cancelled);
            }
            let token_id =
                match self
                    .adapter
                    .observe_candidate(&self.tokenizer, &output_ids, candidate)?
                {
                    NativeTextCandidateDecision::Emit(token_id) => token_id,
                    NativeTextCandidateDecision::Stop => {
                        finish_reason = llm_api::FinishReason::Stop;
                        break;
                    }
                };
            output_ids.push(u32::try_from(token_id).map_err(|err| {
                BackendError::Other(format!(
                    "{} token id does not fit u32: {err}",
                    self.adapter.family_display_name()
                ))
            })?);
            if step_idx + 1 < requested {
                self.adapter.step(&mut decode, token_id)?;
            }
        }

        let text = self.adapter.decode_output(&self.tokenizer, &output_ids)?;
        Ok(BackendOutput {
            text,
            prompt_tokens: prompt_tokens.len() as u64,
            completion_tokens: output_ids.len() as u64,
            finish_reason,
        })
    }

    pub(crate) fn generate_blocking_stream(
        &self,
        request: BackendRequest,
        tx: tokio::sync::mpsc::Sender<Result<BackendStreamChunk, BackendError>>,
        cancellation: CancellationToken,
    ) -> Result<(), BackendError> {
        if cancellation.is_cancelled() {
            return Err(BackendError::Cancelled);
        }
        self.validate_model(&request)?;
        if cancellation.is_cancelled() {
            return Err(BackendError::Cancelled);
        }
        let prompt_tokens = self.encode_prompt(&request)?;
        let context_tokens = prompt_tokens
            .iter()
            .map(|token| *token as usize)
            .collect::<Vec<_>>();
        if context_tokens.is_empty() {
            return Err(BackendError::Other(format!(
                "{} prompt encoded to zero tokens",
                self.adapter.family_display_name()
            )));
        }
        let mut output_ids = Vec::new();
        let mut text_deltas = NativeStreamTextDeltas::default();
        let mut unreported_completion_tokens = 0_u64;
        let mut finish_reason = llm_api::FinishReason::Length;
        let requested = resolve_native_text_max_tokens(
            request.max_tokens,
            self.max_new_tokens,
            self.adapter.family_display_name(),
        )?;
        let mut decode =
            match self.start_decode_session(&context_tokens, requested, &request, &cancellation) {
                Ok(decode) => decode,
                Err(BackendError::Cancelled) if cancellation.is_cancelled() => {
                    return Err(BackendError::Cancelled);
                }
                Err(err) => return Err(err),
            };
        if cancellation.is_cancelled() {
            return Err(BackendError::Cancelled);
        }

        for step_idx in 0..requested {
            if cancellation.is_cancelled() {
                return Err(BackendError::Cancelled);
            }
            let candidate = self
                .adapter
                .next_token_from_hidden(self.adapter.hidden(&decode), request.sampling)?;
            if cancellation.is_cancelled() {
                return Err(BackendError::Cancelled);
            }
            let token_id =
                match self
                    .adapter
                    .observe_candidate(&self.tokenizer, &output_ids, candidate)?
                {
                    NativeTextCandidateDecision::Emit(token_id) => token_id,
                    NativeTextCandidateDecision::Stop => {
                        finish_reason = llm_api::FinishReason::Stop;
                        break;
                    }
                };
            output_ids.push(u32::try_from(token_id).map_err(|err| {
                BackendError::Other(format!(
                    "{} token id does not fit u32: {err}",
                    self.adapter.family_display_name()
                ))
            })?);
            unreported_completion_tokens += 1;
            let next_decoded = self.adapter.decode_output(&self.tokenizer, &output_ids)?;
            let delta = text_deltas.observe(next_decoded)?;
            if cancellation.is_cancelled() {
                return Err(BackendError::Cancelled);
            }
            if let Some(delta) = delta {
                send_backend_stream_chunk(
                    &tx,
                    BackendStreamChunk {
                        text: delta,
                        prompt_tokens: prompt_tokens.len() as u64,
                        completion_tokens: std::mem::take(&mut unreported_completion_tokens),
                        finish_reason: None,
                    },
                )?;
            }
            if step_idx + 1 < requested {
                if cancellation.is_cancelled() {
                    return Err(BackendError::Cancelled);
                }
                self.adapter.step(&mut decode, token_id)?;
            }
        }

        if cancellation.is_cancelled() {
            return Err(BackendError::Cancelled);
        }
        let final_text = if output_ids.is_empty() {
            None
        } else {
            let final_decoded = self.adapter.decode_output(&self.tokenizer, &output_ids)?;
            text_deltas.finish(final_decoded)?
        };
        send_backend_stream_chunk(
            &tx,
            BackendStreamChunk {
                text: final_text.unwrap_or_default(),
                prompt_tokens: prompt_tokens.len() as u64,
                completion_tokens: std::mem::take(&mut unreported_completion_tokens),
                finish_reason: Some(finish_reason),
            },
        )
    }

    fn validate_model(&self, request: &BackendRequest) -> Result<(), BackendError> {
        if request.model != self.model_id {
            return Err(BackendError::ModelNotFound {
                requested: request.model.clone(),
                available: self.model_id.clone(),
            });
        }
        Ok(())
    }

    fn encode_prompt(&self, request: &BackendRequest) -> Result<Vec<u32>, BackendError> {
        self.adapter.encode_prompt(&self.tokenizer, request)
    }

    pub(crate) fn start_decode_session(
        &self,
        context_tokens: &[usize],
        max_new_tokens: u32,
        request: &BackendRequest,
        cancellation: &CancellationToken,
    ) -> Result<A::DecodeSession, BackendError> {
        if cancellation.is_cancelled() {
            return Err(BackendError::Cancelled);
        }
        let cache_tokens = native_text_cache_token_capacity(
            context_tokens.len(),
            max_new_tokens,
            self.adapter.max_prefill_tokens(),
            self.adapter.max_position_embeddings(),
            self.adapter.family_display_name(),
        )?;
        let namespace = self.adapter.prefix_cache_namespace(request, cache_tokens);
        let layer_count = self.adapter.layer_count();
        let mut cached_prefix_len = 0_usize;
        let (mut hidden, mut caches) = if let Some(hit) = self.adapter.prefix_cache().lookup(
            &namespace,
            context_tokens,
            self.adapter.prefix_cache_metrics(),
        ) {
            if hit.caches.len() != layer_count {
                return Err(BackendError::Other(format!(
                    "native {} prefix cache entry had {} layers, expected {layer_count}",
                    self.adapter.family_display_name(),
                    hit.caches.len()
                )));
            }
            cached_prefix_len = hit.token_count;
            (Some(hit.hidden), hit.caches)
        } else {
            (None, self.adapter.allocate_caches(cache_tokens)?)
        };
        let mut cache_cleanup = NativeTextCacheMirrorCleanupGuard::new(&self.adapter);
        if cancellation.is_cancelled() {
            cache_cleanup.cleanup(&caches);
            return Err(BackendError::Cancelled);
        }
        if cached_prefix_len < context_tokens.len() {
            hidden = match self.prefill_context_with_cache(
                &context_tokens[cached_prefix_len..],
                &mut caches,
                cancellation,
            ) {
                Ok(hidden) => Some(hidden),
                Err(err) => {
                    cache_cleanup.cleanup(&caches);
                    return Err(err);
                }
            };
        }
        let hidden = match hidden {
            Some(hidden) => hidden,
            None => {
                cache_cleanup.cleanup(&caches);
                return Err(BackendError::Other(format!(
                    "{} prefill returned no hidden states",
                    self.adapter.family_display_name()
                )));
            }
        };
        if cancellation.is_cancelled() {
            cache_cleanup.cleanup(&caches);
            return Err(BackendError::Cancelled);
        }
        self.adapter.prefix_cache().store(
            namespace,
            context_tokens,
            &hidden,
            &caches,
            self.adapter.prefix_cache_metrics(),
        );
        cache_cleanup.disarm();
        Ok(self.adapter.make_decode_session(hidden, caches))
    }

    pub(crate) fn prefill_context_with_cache(
        &self,
        context_tokens: &[usize],
        caches: &mut [A::LayerCache],
        cancellation: &CancellationToken,
    ) -> Result<Vec<f32>, BackendError> {
        native_text_prefill_context_with_cache(
            self.adapter.family_display_name(),
            self.adapter.max_prefill_tokens(),
            context_tokens,
            caches,
            cancellation,
            |chunk, caches| self.adapter.prefill_chunk_with_cache(chunk, caches),
        )
    }
}

#[async_trait]
impl<A> ModelBackend for NativeTextDriver<A>
where
    A: NativeTextAdapter,
{
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        self.metadata.clone()
    }

    async fn generate(&self, request: BackendRequest) -> Result<BackendOutput, BackendError> {
        self.generate_with_cancel(request, CancellationToken::new())
            .await
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        let driver = self.clone();
        let label = driver.worker_label();
        tokio::task::spawn_blocking(move || driver.generate_blocking(request, cancellation))
            .await
            .map_err(|err| BackendError::Other(format!("{label} worker failed: {err}")))?
    }

    fn generate_stream<'a>(
        &'a self,
        request: BackendRequest,
    ) -> BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        self.generate_stream_with_cancel(request, CancellationToken::new())
    }

    fn generate_stream_with_cancel<'a>(
        &'a self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        let driver = self.clone();
        let label = driver.worker_label();
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let worker = tokio::task::spawn_blocking(move || {
            let err_tx = tx.clone();
            if let Err(err) = driver.generate_blocking_stream(request, tx, cancellation) {
                let _ = err_tx.blocking_send(Err(err));
            }
        });
        native_text_worker_stream(label, rx, worker)
    }
}

impl<A> NativeTextDriver<A>
where
    A: NativeTextAdapter,
{
    fn worker_label(&self) -> &'static str {
        self.adapter.worker_label()
    }
}
