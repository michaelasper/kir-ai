use super::{
    NativeStreamTextDeltas, NativeTextPrefixCache, NativeTextPrefixCacheMetrics,
    NativeTextPrefixCacheNamespace, NativeTextPrefixCacheValue, NativeTextSamplingRng,
    native_text_cache_token_capacity, native_text_worker_stream, resolve_native_text_max_tokens,
};
use crate::native_matvec::NativeTextCacheMirrorSource;
use async_trait::async_trait;
use futures::stream::BoxStream;
use llm_backend::{
    BackendError, BackendModelMetadata, BackendOutput, BackendRequest, BackendStreamChunk,
    InferenceScratchpad, ModelBackend, SamplingConfig,
};
use llm_sampler::TopPSamplerScratch;
use llm_tokenizer::HuggingFaceTokenizer;
use std::{
    cell::RefCell,
    collections::HashSet,
    future::Future,
    ops::{Deref, DerefMut},
    sync::Arc,
};
use tokio_util::sync::CancellationToken;

thread_local! {
    static NATIVE_TEXT_WORKER_RUNTIME: RefCell<Option<tokio::runtime::Runtime>> =
        const { RefCell::new(None) };
}

#[async_trait]
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
        stop_tokens: &NativeTextResolvedStopTokens,
        _emitted_tokens: &[u32],
        token_id: usize,
    ) -> Result<NativeTextCandidateDecision, BackendError> {
        Ok(native_text_candidate_decision_for_stop_tokens(
            stop_tokens,
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
    async fn prefill_chunk_with_cache(
        &self,
        token_ids: &[usize],
        caches: &mut [Self::LayerCache],
        scratch: &mut InferenceScratchpad,
    ) -> Result<Vec<Vec<f32>>, BackendError>;
    fn make_decode_session(
        &self,
        hidden: Vec<f32>,
        caches: Vec<Self::LayerCache>,
    ) -> Self::DecodeSession;
    fn cleanup_cache_mirrors(&self, _caches: &[Self::LayerCache]) {}
    fn hidden<'a>(&self, session: &'a Self::DecodeSession) -> &'a [f32];
    async fn step(
        &self,
        session: &mut Self::DecodeSession,
        token_id: usize,
        scratch: &mut InferenceScratchpad,
    ) -> Result<(), BackendError>;
    async fn next_token_from_hidden(
        &self,
        hidden: &[f32],
        sampling: SamplingConfig,
        sampling_draw: Option<f32>,
        scratch: &mut InferenceScratchpad,
        sampling_scratch: &mut TopPSamplerScratch,
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
    pub(crate) fn resolve(&self, tokenizer: &HuggingFaceTokenizer) -> NativeTextResolvedStopTokens {
        let mut token_ids = self.token_ids.to_vec();
        for token_string in self.token_strings {
            if let Some(token_id) = tokenizer.token_to_id(token_string) {
                token_ids.push(token_id as usize);
            } else if let Ok(encoded) = tokenizer.encode(token_string, false) {
                token_ids.extend(encoded.into_iter().map(|token_id| token_id as usize));
            }
        }
        NativeTextResolvedStopTokens {
            token_ids: token_ids.into_iter().collect(),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct NativeTextResolvedStopTokens {
    token_ids: HashSet<usize>,
}

impl NativeTextResolvedStopTokens {
    pub(crate) fn contains(&self, token_id: usize) -> bool {
        self.token_ids.contains(&token_id)
    }

    #[cfg(test)]
    pub(crate) fn token_ids(&self) -> Vec<usize> {
        let mut token_ids = self.token_ids.iter().copied().collect::<Vec<_>>();
        token_ids.sort_unstable();
        token_ids
    }
}

pub(crate) fn native_text_candidate_decision_for_stop_tokens(
    stop_tokens: &NativeTextResolvedStopTokens,
    token_id: usize,
) -> NativeTextCandidateDecision {
    if stop_tokens.contains(token_id) {
        NativeTextCandidateDecision::Stop
    } else {
        NativeTextCandidateDecision::Emit(token_id)
    }
}

pub(crate) struct NativeTextDriver<A>
where
    A: NativeTextAdapter,
{
    inner: Arc<NativeTextDriverInner<A>>,
}

#[derive(Clone)]
pub(crate) struct NativeTextDriverInner<A>
where
    A: NativeTextAdapter,
{
    pub(crate) model_id: String,
    pub(crate) metadata: BackendModelMetadata,
    pub(crate) tokenizer: HuggingFaceTokenizer,
    pub(crate) adapter: A,
    pub(crate) stop_tokens: NativeTextResolvedStopTokens,
    pub(crate) max_new_tokens: u32,
}

struct NativeTextDecodeStart<S> {
    session: S,
    cache_report: NativeTextRequestCacheReport,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NativeTextRequestCacheReport {
    lookup_result: NativeTextPrefixLookupResult,
    prompt_tokens: u64,
    reused_tokens: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeTextPrefixLookupResult {
    Hit,
    Miss,
}

impl NativeTextRequestCacheReport {
    fn hit(prompt_tokens: usize, reused_tokens: usize) -> Self {
        Self {
            lookup_result: NativeTextPrefixLookupResult::Hit,
            prompt_tokens: prompt_tokens as u64,
            reused_tokens: reused_tokens as u64,
        }
    }

    fn miss(prompt_tokens: usize) -> Self {
        Self {
            lookup_result: NativeTextPrefixLookupResult::Miss,
            prompt_tokens: prompt_tokens as u64,
            reused_tokens: 0,
        }
    }

    fn prompt_cached_tokens(self) -> Option<u64> {
        Some(self.reused_tokens)
    }

    fn trace(self, namespace: &NativeTextPrefixCacheNamespace) {
        tracing::debug!(
            lookup_result = self.lookup_result.as_str(),
            reuse_source = "in_memory_prefix_cache",
            reused_tokens = self.reused_tokens,
            prompt_tokens = self.prompt_tokens,
            model_id = %namespace.model_id,
            backend = %namespace.backend,
            family = ?namespace.family,
            loader = ?namespace.loader,
            quantization = ?namespace.quantization,
            prompt_template = %namespace.prompt_template,
            request_mode = %namespace.request_mode,
            cache_layout_version = namespace.cache_layout_version,
            cache_tokens = namespace.cache_tokens,
            max_prefill_tokens = namespace.max_prefill_tokens,
            "native text prefix cache request"
        );
    }
}

impl NativeTextPrefixLookupResult {
    fn as_str(self) -> &'static str {
        match self {
            Self::Hit => "hit",
            Self::Miss => "miss",
        }
    }
}

impl<A> Clone for NativeTextDriver<A>
where
    A: NativeTextAdapter,
{
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<A> Deref for NativeTextDriver<A>
where
    A: NativeTextAdapter,
{
    type Target = NativeTextDriverInner<A>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<A> DerefMut for NativeTextDriver<A>
where
    A: NativeTextAdapter,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(&mut self.inner)
    }
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
        let stop_tokens = adapter.stop_tokens().resolve(&tokenizer);
        Self {
            inner: Arc::new(NativeTextDriverInner {
                model_id,
                metadata,
                tokenizer,
                adapter,
                stop_tokens,
                max_new_tokens,
            }),
        }
    }

    #[cfg(test)]
    pub(crate) fn shares_inner_state_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    pub(crate) fn with_max_new_tokens(mut self, max_new_tokens: u32) -> Self {
        self.max_new_tokens = max_new_tokens;
        self
    }

    pub(crate) fn with_max_prefill_tokens(mut self, max_prefill_tokens: usize) -> Self {
        self.adapter
            .set_max_prefill_tokens(max_prefill_tokens.max(1));
        self
    }

    #[cfg(test)]
    pub(crate) fn generate_blocking(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        tokio::task::block_in_place(|| {
            self.block_on_worker(self.generate_async(request, cancellation))?
        })
    }

    #[cfg(test)]
    pub(crate) fn generate_blocking_stream(
        &self,
        request: BackendRequest,
        tx: tokio::sync::mpsc::Sender<Result<BackendStreamChunk, BackendError>>,
        cancellation: CancellationToken,
    ) -> Result<(), BackendError> {
        tokio::task::block_in_place(|| {
            self.block_on_worker(self.generate_stream_async(request, tx, cancellation))?
        })
    }

    #[cfg(test)]
    pub(crate) fn block_on_worker<F>(&self, future: F) -> Result<F::Output, BackendError>
    where
        F: Future,
    {
        block_on_native_text_worker(future)
    }

    pub async fn generate_async(
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
        let mut scratch = InferenceScratchpad::new();
        let mut sampling_scratch = TopPSamplerScratch::new();
        let mut sampling_rng = native_text_sampling_rng_for_config(request.sampling);
        let start = self
            .start_decode_session_with_cache_report(
                &context_tokens,
                requested,
                &request,
                &cancellation,
                &mut scratch,
            )
            .await?;
        let cache_report = start.cache_report;
        let mut decode = start.session;
        if cancellation.is_cancelled() {
            return Err(BackendError::Cancelled);
        }

        for step_idx in 0..requested {
            if cancellation.is_cancelled() {
                return Err(BackendError::Cancelled);
            }
            let candidate = self
                .adapter
                .next_token_from_hidden(
                    self.adapter.hidden(&decode),
                    request.sampling,
                    native_text_sampling_draw_for_config(request.sampling, &mut sampling_rng),
                    &mut scratch,
                    &mut sampling_scratch,
                )
                .await?;
            if cancellation.is_cancelled() {
                return Err(BackendError::Cancelled);
            }
            let token_id =
                match self
                    .adapter
                    .observe_candidate(&self.stop_tokens, &output_ids, candidate)?
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
                self.adapter
                    .step(&mut decode, token_id, &mut scratch)
                    .await?;
            }
        }

        let text = self.adapter.decode_output(&self.tokenizer, &output_ids)?;
        Ok(BackendOutput {
            text,
            prompt_tokens: prompt_tokens.len() as u64,
            prompt_cached_tokens: cache_report.prompt_cached_tokens(),
            completion_tokens: output_ids.len() as u64,
            finish_reason,
        })
    }

    pub async fn generate_stream_async(
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
        let mut scratch = InferenceScratchpad::new();
        let mut sampling_scratch = TopPSamplerScratch::new();
        let mut sampling_rng = native_text_sampling_rng_for_config(request.sampling);
        let start = match self
            .start_decode_session_with_cache_report(
                &context_tokens,
                requested,
                &request,
                &cancellation,
                &mut scratch,
            )
            .await
        {
            Ok(start) => start,
            Err(BackendError::Cancelled) if cancellation.is_cancelled() => {
                return Err(BackendError::Cancelled);
            }
            Err(err) => return Err(err),
        };
        let cache_report = start.cache_report;
        let mut decode = start.session;
        if cancellation.is_cancelled() {
            return Err(BackendError::Cancelled);
        }

        for step_idx in 0..requested {
            if cancellation.is_cancelled() {
                return Err(BackendError::Cancelled);
            }
            let candidate = self
                .adapter
                .next_token_from_hidden(
                    self.adapter.hidden(&decode),
                    request.sampling,
                    native_text_sampling_draw_for_config(request.sampling, &mut sampling_rng),
                    &mut scratch,
                    &mut sampling_scratch,
                )
                .await?;
            if cancellation.is_cancelled() {
                return Err(BackendError::Cancelled);
            }
            let token_id =
                match self
                    .adapter
                    .observe_candidate(&self.stop_tokens, &output_ids, candidate)?
                {
                    NativeTextCandidateDecision::Emit(token_id) => token_id,
                    NativeTextCandidateDecision::Stop => {
                        finish_reason = llm_api::FinishReason::Stop;
                        break;
                    }
                };
            let output_id = u32::try_from(token_id).map_err(|err| {
                BackendError::Other(format!(
                    "{} token id does not fit u32: {err}",
                    self.adapter.family_display_name()
                ))
            })?;
            output_ids.push(output_id);
            unreported_completion_tokens += 1;
            let token_decoded = self.adapter.decode_output(&self.tokenizer, &[output_id])?;
            let delta = text_deltas.observe_incremental(token_decoded);
            if cancellation.is_cancelled() {
                return Err(BackendError::Cancelled);
            }
            if let Some(delta) = delta {
                tx.send(Ok(BackendStreamChunk {
                    text: delta,
                    tool_call_deltas: Vec::new(),
                    prompt_tokens: prompt_tokens.len() as u64,
                    prompt_cached_tokens: cache_report.prompt_cached_tokens(),
                    completion_tokens: std::mem::take(&mut unreported_completion_tokens),
                    finish_reason: None,
                }))
                .await
                .map_err(|err| BackendError::Other(err.to_string()))?;
            }
            if step_idx + 1 < requested {
                if cancellation.is_cancelled() {
                    return Err(BackendError::Cancelled);
                }
                self.adapter
                    .step(&mut decode, token_id, &mut scratch)
                    .await?;
            }
        }

        if cancellation.is_cancelled() {
            return Err(BackendError::Cancelled);
        }
        let final_text = if output_ids.is_empty() {
            None
        } else {
            text_deltas.finish_incremental()
        };
        tx.send(Ok(BackendStreamChunk {
            text: final_text.unwrap_or_default(),
            tool_call_deltas: Vec::new(),
            prompt_tokens: prompt_tokens.len() as u64,
            prompt_cached_tokens: cache_report.prompt_cached_tokens(),
            completion_tokens: std::mem::take(&mut unreported_completion_tokens),
            finish_reason: Some(finish_reason),
        }))
        .await
        .map_err(|err| BackendError::Other(err.to_string()))?;
        Ok(())
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

    #[cfg(test)]
    pub(crate) async fn start_decode_session(
        &self,
        context_tokens: &[usize],
        max_new_tokens: u32,
        request: &BackendRequest,
        cancellation: &CancellationToken,
        scratch: &mut InferenceScratchpad,
    ) -> Result<A::DecodeSession, BackendError> {
        self.start_decode_session_with_cache_report(
            context_tokens,
            max_new_tokens,
            request,
            cancellation,
            scratch,
        )
        .await
        .map(|start| start.session)
    }

    async fn start_decode_session_with_cache_report(
        &self,
        context_tokens: &[usize],
        max_new_tokens: u32,
        request: &BackendRequest,
        cancellation: &CancellationToken,
        scratch: &mut InferenceScratchpad,
    ) -> Result<NativeTextDecodeStart<A::DecodeSession>, BackendError> {
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
        let mut cache_report = NativeTextRequestCacheReport::miss(context_tokens.len());
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
            cache_report = NativeTextRequestCacheReport::hit(context_tokens.len(), hit.token_count);
            (Some(hit.hidden), hit.caches)
        } else {
            (None, self.adapter.allocate_caches(cache_tokens)?)
        };
        cache_report.trace(&namespace);
        let mut cache_cleanup = NativeTextCacheMirrorCleanupGuard::new(&self.adapter);
        if cancellation.is_cancelled() {
            cache_cleanup.cleanup(&caches);
            return Err(BackendError::Cancelled);
        }
        if cached_prefix_len < context_tokens.len() {
            let mut prefill_hidden = None;
            let prefill_chunk_tokens = self.adapter.max_prefill_tokens();
            for chunk in context_tokens[cached_prefix_len..].chunks(prefill_chunk_tokens.max(1)) {
                if cancellation.is_cancelled() {
                    cache_cleanup.cleanup(&caches);
                    return Err(BackendError::Cancelled);
                }
                let hidden_states = match self
                    .adapter
                    .prefill_chunk_with_cache(chunk, &mut caches, scratch)
                    .await
                {
                    Ok(hs) => hs,
                    Err(err) => {
                        cache_cleanup.cleanup(&caches);
                        return Err(err);
                    }
                };
                if cancellation.is_cancelled() {
                    cache_cleanup.cleanup(&caches);
                    return Err(BackendError::Cancelled);
                }
                prefill_hidden = hidden_states.last().cloned();
            }
            hidden = Some(prefill_hidden.ok_or_else(|| {
                cache_cleanup.cleanup(&caches);
                BackendError::Other(format!(
                    "{} prefill returned no hidden states",
                    self.adapter.family_display_name()
                ))
            })?);
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
        Ok(NativeTextDecodeStart {
            session: self.adapter.make_decode_session(hidden, caches),
            cache_report,
        })
    }
}

fn block_on_native_text_worker<F>(future: F) -> Result<F::Output, BackendError>
where
    F: Future,
{
    NATIVE_TEXT_WORKER_RUNTIME.with(|runtime| {
        if runtime.borrow().is_none() {
            *runtime.borrow_mut() = Some(build_native_text_worker_runtime()?);
        }
        let runtime = runtime.borrow();
        let Some(runtime) = runtime.as_ref() else {
            return Err(BackendError::Other(
                "native text worker runtime was not initialized".to_owned(),
            ));
        };
        Ok(runtime.block_on(future))
    })
}

fn build_native_text_worker_runtime() -> Result<tokio::runtime::Runtime, BackendError> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|err| {
            BackendError::Other(format!("native text worker runtime build failed: {err}"))
        })
}

fn native_text_sampling_draw_for_config(
    sampling: SamplingConfig,
    sampling_rng: &mut Option<NativeTextSamplingRng>,
) -> Option<f32> {
    if sampling.is_greedy() {
        None
    } else {
        sampling_rng.as_mut().map(NativeTextSamplingRng::draw_f32)
    }
}

fn native_text_sampling_rng_for_config(sampling: SamplingConfig) -> Option<NativeTextSamplingRng> {
    (!sampling.is_greedy()).then(NativeTextSamplingRng::from_entropy)
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
        let label = driver.adapter.worker_label();
        let mut cancel_on_drop = CancelOnDrop::new(cancellation.clone());
        let result = tokio::task::spawn_blocking(move || {
            block_on_native_text_worker(async move {
                driver.generate_async(request, cancellation).await
            })?
        })
        .await
        .map_err(|err| BackendError::Other(format!("{label} generation worker failed: {err}")))?;
        cancel_on_drop.disarm();
        result
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
        let label = driver.adapter.worker_label();
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let worker = tokio::task::spawn_blocking(move || {
            let runtime_error_tx = tx.clone();
            if let Err(err) = block_on_native_text_worker(async move {
                if let Err(err) = driver
                    .generate_stream_async(request, tx.clone(), cancellation)
                    .await
                {
                    let _ = tx.send(Err(err)).await;
                }
            }) {
                let _ = runtime_error_tx.blocking_send(Err(err));
            }
        });
        native_text_worker_stream(label, rx, worker)
    }
}

struct CancelOnDrop {
    cancellation: CancellationToken,
    armed: bool,
}

impl CancelOnDrop {
    fn new(cancellation: CancellationToken) -> Self {
        Self {
            cancellation,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.cancellation.cancel();
        }
    }
}

pub(crate) struct NativeTextCacheMirrorCleanupGuard<'a, A: NativeTextAdapter> {
    adapter: &'a A,
    armed: bool,
}

impl<'a, A: NativeTextAdapter> NativeTextCacheMirrorCleanupGuard<'a, A> {
    pub(crate) fn new(adapter: &'a A) -> Self {
        Self {
            adapter,
            armed: true,
        }
    }

    pub(crate) fn disarm(&mut self) {
        self.armed = false;
    }

    pub(crate) fn cleanup(&self, caches: &[A::LayerCache]) {
        if self.armed {
            self.adapter.cleanup_cache_mirrors(caches);
        }
    }
}
