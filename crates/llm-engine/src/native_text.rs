use crate::{
    DEFAULT_NATIVE_QWEN_MAX_NEW_TOKENS, NativeGemmaAdapter, NativeGemmaBackend,
    NativeGemmaLoadOptions, NativeQwenAdapter, NativeQwenBackend, NativeQwenLoadOptions,
};
use async_trait::async_trait;
use futures::stream::BoxStream;
use llm_backend::{
    BackendError, BackendModelMetadata, BackendOutput, BackendRequest, BackendStreamChunk,
    ModelBackend, SamplingConfig,
};
use llm_models::{ModelFamily, NativeTextModelSpec};
use llm_tokenizer::HuggingFaceTokenizer;
use std::path::Path;
use tokio_util::sync::CancellationToken;

mod generation;
mod prefix_cache;
mod streaming;

#[allow(unused_imports)]
pub(crate) use generation::{
    NativeTextNextTokenContext, native_text_cache_token_capacity,
    native_text_prefill_context_with_cache, resolve_native_text_max_tokens,
    sample_token_id_with_draw,
};
#[allow(unused_imports)]
pub(crate) use prefix_cache::{
    NativeTextPrefixCache, NativeTextPrefixCacheCounters, NativeTextPrefixCacheEntry,
    NativeTextPrefixCacheHit, NativeTextPrefixCacheInner, NativeTextPrefixCacheKey,
    NativeTextPrefixCacheMetrics, NativeTextPrefixCacheNamespace, NativeTextPrefixCacheValue,
    NativeTextPrefixNamespaceContext, native_text_prefix_namespace,
    native_text_prefix_request_mode,
};
pub(crate) use streaming::{
    NativeStreamTextDeltas, native_text_worker_stream, send_backend_stream_chunk,
};

pub const DEFAULT_NATIVE_TEXT_MAX_NEW_TOKENS: u32 = DEFAULT_NATIVE_QWEN_MAX_NEW_TOKENS;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativeTextLoadOptions {
    pub family: Option<ModelFamily>,
    pub qwen: NativeQwenLoadOptions,
    pub gemma: NativeGemmaLoadOptions,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativeTextRuntimeOptions {
    pub eager_materialize_shards: bool,
    pub metal_weight_cache_bytes: Option<u64>,
    pub warm_metal_weight_cache: bool,
}

impl From<NativeTextRuntimeOptions> for NativeQwenLoadOptions {
    fn from(value: NativeTextRuntimeOptions) -> Self {
        Self {
            eager_materialize_shards: value.eager_materialize_shards,
            metal_weight_cache_bytes: value.metal_weight_cache_bytes,
            warm_metal_weight_cache: value.warm_metal_weight_cache,
        }
    }
}

impl From<NativeTextRuntimeOptions> for NativeGemmaLoadOptions {
    fn from(value: NativeTextRuntimeOptions) -> Self {
        Self {
            eager_materialize_shards: value.eager_materialize_shards,
            metal_weight_cache_bytes: value.metal_weight_cache_bytes,
            warm_metal_weight_cache: value.warm_metal_weight_cache,
        }
    }
}

impl NativeTextLoadOptions {
    pub fn with_runtime_options(runtime: NativeTextRuntimeOptions) -> Self {
        Self {
            family: None,
            qwen: runtime.into(),
            gemma: runtime.into(),
        }
    }

    pub fn with_qwen_options(qwen: NativeQwenLoadOptions) -> Self {
        Self {
            family: None,
            gemma: NativeGemmaLoadOptions {
                eager_materialize_shards: qwen.eager_materialize_shards,
                metal_weight_cache_bytes: qwen.metal_weight_cache_bytes,
                warm_metal_weight_cache: qwen.warm_metal_weight_cache,
            },
            qwen,
        }
    }

    pub fn with_family(mut self, family: ModelFamily) -> Self {
        self.family = Some(family);
        self
    }
}

pub(crate) fn native_text_metal_metrics_snapshot() -> serde_json::Value {
    crate::native_matvec::native_text_metal_metrics_snapshot()
}

pub(crate) fn native_text_prefix_cache_metrics_snapshot(
    qwen_snapshot: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({
        "qwen": qwen_snapshot,
        "gemma": crate::native_gemma::native_gemma_prefix_cache_metrics_snapshot(),
    })
}

#[derive(Clone)]
pub struct NativeTextBackend {
    inner: NativeTextBackendInner,
}

#[derive(Clone)]
enum NativeTextBackendInner {
    Qwen(NativeTextDriver<NativeQwenAdapter>),
    Gemma(NativeTextDriver<NativeGemmaAdapter>),
}

impl NativeTextBackend {
    pub fn open(
        model_id: impl Into<String>,
        snapshot_path: impl AsRef<Path>,
    ) -> anyhow::Result<Self> {
        Self::open_with_options(model_id, snapshot_path, NativeTextLoadOptions::default())
    }

    pub fn open_with_options(
        model_id: impl Into<String>,
        snapshot_path: impl AsRef<Path>,
        options: NativeTextLoadOptions,
    ) -> anyhow::Result<Self> {
        let snapshot_path = snapshot_path.as_ref();
        let family = match options.family {
            Some(family) => family,
            None => infer_native_text_family(snapshot_path)?,
        };
        match family {
            ModelFamily::Gemma => {
                let driver =
                    NativeGemmaBackend::open_with_options(model_id, snapshot_path, options.gemma)?
                        .into_driver();
                Ok(Self {
                    inner: NativeTextBackendInner::Gemma(driver),
                })
            }
            ModelFamily::DeepSeek => {
                anyhow::bail!(
                    "native text execution for family `deep_seek` is deferred until native DeepSeek tensor support exists"
                );
            }
            ModelFamily::Llama => {
                anyhow::bail!(
                    "native text execution for family `llama` is deferred until native Llama tensor support exists"
                );
            }
            ModelFamily::Qwen => {
                let driver =
                    NativeQwenBackend::open_with_options(model_id, snapshot_path, options.qwen)?
                        .into_driver();
                Ok(Self {
                    inner: NativeTextBackendInner::Qwen(driver),
                })
            }
        }
    }

    pub fn with_max_new_tokens(mut self, max_new_tokens: u32) -> Self {
        self.inner = match self.inner {
            NativeTextBackendInner::Qwen(driver) => {
                NativeTextBackendInner::Qwen(driver.with_max_new_tokens(max_new_tokens))
            }
            NativeTextBackendInner::Gemma(driver) => {
                NativeTextBackendInner::Gemma(driver.with_max_new_tokens(max_new_tokens))
            }
        };
        self
    }

    pub fn with_max_prefill_tokens(mut self, max_prefill_tokens: usize) -> Self {
        self.inner = match self.inner {
            NativeTextBackendInner::Qwen(driver) => {
                NativeTextBackendInner::Qwen(driver.with_max_prefill_tokens(max_prefill_tokens))
            }
            NativeTextBackendInner::Gemma(driver) => {
                NativeTextBackendInner::Gemma(driver.with_max_prefill_tokens(max_prefill_tokens))
            }
        };
        self
    }
}

pub(crate) fn infer_native_text_family(snapshot_path: &Path) -> anyhow::Result<ModelFamily> {
    let config_path = snapshot_path.join("config.json");
    let config_json = std::fs::read_to_string(&config_path).map_err(|err| {
        anyhow::anyhow!(
            "native text snapshot without explicit family metadata requires readable config.json for family detection at `{}`: {err}",
            config_path.display()
        )
    })?;
    Ok(NativeTextModelSpec::infer_from_config_json(&config_json)?.family())
}

#[async_trait]
impl ModelBackend for NativeTextBackend {
    fn model_id(&self) -> &str {
        match &self.inner {
            NativeTextBackendInner::Qwen(backend) => backend.model_id(),
            NativeTextBackendInner::Gemma(backend) => backend.model_id(),
        }
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        match &self.inner {
            NativeTextBackendInner::Qwen(backend) => backend.model_metadata(),
            NativeTextBackendInner::Gemma(backend) => backend.model_metadata(),
        }
    }

    async fn generate(&self, request: BackendRequest) -> Result<BackendOutput, BackendError> {
        match &self.inner {
            NativeTextBackendInner::Qwen(backend) => backend.generate(request).await,
            NativeTextBackendInner::Gemma(backend) => backend.generate(request).await,
        }
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        match &self.inner {
            NativeTextBackendInner::Qwen(backend) => {
                backend.generate_with_cancel(request, cancellation).await
            }
            NativeTextBackendInner::Gemma(backend) => {
                backend.generate_with_cancel(request, cancellation).await
            }
        }
    }

    fn generate_stream<'a>(
        &'a self,
        request: BackendRequest,
    ) -> BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        match &self.inner {
            NativeTextBackendInner::Qwen(backend) => backend.generate_stream(request),
            NativeTextBackendInner::Gemma(backend) => backend.generate_stream(request),
        }
    }

    fn generate_stream_with_cancel<'a>(
        &'a self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        match &self.inner {
            NativeTextBackendInner::Qwen(backend) => {
                backend.generate_stream_with_cancel(request, cancellation)
            }
            NativeTextBackendInner::Gemma(backend) => {
                backend.generate_stream_with_cancel(request, cancellation)
            }
        }
    }
}

pub(crate) trait NativeTextAdapter: Clone + Send + Sync + 'static {
    type DecodeSession: Send + 'static;
    type LayerCache: NativeTextPrefixCacheValue + Send + 'static;

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
    fn contains(self, tokenizer: &HuggingFaceTokenizer, token_id: usize) -> bool {
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
        if cancellation.is_cancelled() {
            return Err(BackendError::Cancelled);
        }
        if cached_prefix_len < context_tokens.len() {
            hidden = Some(self.prefill_context_with_cache(
                &context_tokens[cached_prefix_len..],
                &mut caches,
                cancellation,
            )?);
        }
        let hidden = hidden.ok_or_else(|| {
            BackendError::Other(format!(
                "{} prefill returned no hidden states",
                self.adapter.family_display_name()
            ))
        })?;
        if cancellation.is_cancelled() {
            return Err(BackendError::Cancelled);
        }
        self.adapter.prefix_cache().store(
            namespace,
            context_tokens,
            &hidden,
            &caches,
            self.adapter.prefix_cache_metrics(),
        );
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

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TestCache {
        bytes: u64,
        marker: u32,
    }

    impl NativeTextPrefixCacheValue for TestCache {
        fn prefix_cache_entry_bytes(hidden: &[f32], caches: &[Self]) -> u64 {
            std::mem::size_of_val(hidden) as u64
                + caches.iter().map(|cache| cache.bytes).sum::<u64>()
        }
    }

    #[derive(Clone)]
    struct TestDecodeSession {
        hidden: Vec<f32>,
    }

    #[derive(Clone)]
    struct TestAdapter {
        script: std::sync::Arc<[usize]>,
        stop_tokens: NativeTextStopTokens,
        max_prefill_tokens: usize,
        prefix_cache: std::sync::Arc<NativeTextPrefixCache<TestCache>>,
        prefix_cache_metrics: std::sync::Arc<NativeTextPrefixCacheMetrics>,
    }

    impl TestAdapter {
        fn new(script: impl Into<std::sync::Arc<[usize]>>) -> Self {
            Self {
                script: script.into(),
                stop_tokens: NativeTextStopTokens::default(),
                max_prefill_tokens: 4,
                prefix_cache: std::sync::Arc::new(NativeTextPrefixCache::new(1024)),
                prefix_cache_metrics: std::sync::Arc::new(NativeTextPrefixCacheMetrics::default()),
            }
        }

        fn with_stop_tokens(mut self, stop_tokens: NativeTextStopTokens) -> Self {
            self.stop_tokens = stop_tokens;
            self
        }
    }

    impl NativeTextAdapter for TestAdapter {
        type DecodeSession = TestDecodeSession;
        type LayerCache = TestCache;

        fn family_display_name(&self) -> &'static str {
            "Test"
        }

        fn worker_label(&self) -> &'static str {
            "native test"
        }

        fn set_max_prefill_tokens(&mut self, max_prefill_tokens: usize) {
            self.max_prefill_tokens = max_prefill_tokens;
        }

        fn encode_prompt(
            &self,
            _tokenizer: &HuggingFaceTokenizer,
            _request: &BackendRequest,
        ) -> Result<Vec<u32>, BackendError> {
            Ok(vec![42])
        }

        fn decode_output(
            &self,
            _tokenizer: &HuggingFaceTokenizer,
            output_ids: &[u32],
        ) -> Result<String, BackendError> {
            Ok(output_ids
                .iter()
                .map(|token_id| format!("<{token_id}>"))
                .collect::<String>())
        }

        fn stop_tokens(&self) -> NativeTextStopTokens {
            self.stop_tokens
        }

        fn max_position_embeddings(&self) -> u32 {
            16
        }

        fn max_prefill_tokens(&self) -> usize {
            self.max_prefill_tokens
        }

        fn prefix_cache(&self) -> &NativeTextPrefixCache<Self::LayerCache> {
            &self.prefix_cache
        }

        fn prefix_cache_metrics(&self) -> &NativeTextPrefixCacheMetrics {
            &self.prefix_cache_metrics
        }

        fn prefix_cache_namespace(
            &self,
            _request: &BackendRequest,
            cache_tokens: usize,
        ) -> NativeTextPrefixCacheNamespace {
            NativeTextPrefixCacheNamespace {
                cache_tokens,
                ..namespace("driver-test")
            }
        }

        fn layer_count(&self) -> usize {
            1
        }

        fn allocate_caches(
            &self,
            _cache_tokens: usize,
        ) -> Result<Vec<Self::LayerCache>, BackendError> {
            Ok(vec![TestCache {
                bytes: 0,
                marker: 0,
            }])
        }

        fn prefill_chunk_with_cache(
            &self,
            token_ids: &[usize],
            _caches: &mut [Self::LayerCache],
        ) -> Result<Vec<Vec<f32>>, BackendError> {
            Ok(token_ids.iter().map(|_| vec![0.0]).collect())
        }

        fn make_decode_session(
            &self,
            hidden: Vec<f32>,
            _caches: Vec<Self::LayerCache>,
        ) -> Self::DecodeSession {
            TestDecodeSession { hidden }
        }

        fn hidden<'a>(&self, session: &'a Self::DecodeSession) -> &'a [f32] {
            &session.hidden
        }

        fn step(
            &self,
            session: &mut Self::DecodeSession,
            _token_id: usize,
        ) -> Result<(), BackendError> {
            session.hidden[0] += 1.0;
            Ok(())
        }

        fn next_token_from_hidden(
            &self,
            hidden: &[f32],
            _sampling: SamplingConfig,
        ) -> Result<usize, BackendError> {
            let script_index = hidden[0] as usize;
            Ok(*self
                .script
                .get(script_index)
                .expect("test script includes requested token"))
        }
    }

    #[derive(Clone)]
    struct ContextSensitiveTestAdapter {
        base: TestAdapter,
        stop_after_emitted: usize,
    }

    impl ContextSensitiveTestAdapter {
        fn new(script: impl Into<std::sync::Arc<[usize]>>, stop_after_emitted: usize) -> Self {
            Self {
                base: TestAdapter::new(script),
                stop_after_emitted,
            }
        }
    }

    impl NativeTextAdapter for ContextSensitiveTestAdapter {
        type DecodeSession = TestDecodeSession;
        type LayerCache = TestCache;

        fn family_display_name(&self) -> &'static str {
            self.base.family_display_name()
        }

        fn worker_label(&self) -> &'static str {
            self.base.worker_label()
        }

        fn set_max_prefill_tokens(&mut self, max_prefill_tokens: usize) {
            self.base.set_max_prefill_tokens(max_prefill_tokens);
        }

        fn encode_prompt(
            &self,
            tokenizer: &HuggingFaceTokenizer,
            request: &BackendRequest,
        ) -> Result<Vec<u32>, BackendError> {
            self.base.encode_prompt(tokenizer, request)
        }

        fn decode_output(
            &self,
            tokenizer: &HuggingFaceTokenizer,
            output_ids: &[u32],
        ) -> Result<String, BackendError> {
            self.base.decode_output(tokenizer, output_ids)
        }

        fn observe_candidate(
            &self,
            tokenizer: &HuggingFaceTokenizer,
            emitted_tokens: &[u32],
            token_id: usize,
        ) -> Result<NativeTextCandidateDecision, BackendError> {
            if emitted_tokens.len() >= self.stop_after_emitted {
                Ok(NativeTextCandidateDecision::Stop)
            } else {
                self.base
                    .observe_candidate(tokenizer, emitted_tokens, token_id)
            }
        }

        fn max_position_embeddings(&self) -> u32 {
            self.base.max_position_embeddings()
        }

        fn max_prefill_tokens(&self) -> usize {
            self.base.max_prefill_tokens()
        }

        fn prefix_cache(&self) -> &NativeTextPrefixCache<Self::LayerCache> {
            self.base.prefix_cache()
        }

        fn prefix_cache_metrics(&self) -> &NativeTextPrefixCacheMetrics {
            self.base.prefix_cache_metrics()
        }

        fn prefix_cache_namespace(
            &self,
            request: &BackendRequest,
            cache_tokens: usize,
        ) -> NativeTextPrefixCacheNamespace {
            self.base.prefix_cache_namespace(request, cache_tokens)
        }

        fn layer_count(&self) -> usize {
            self.base.layer_count()
        }

        fn allocate_caches(
            &self,
            cache_tokens: usize,
        ) -> Result<Vec<Self::LayerCache>, BackendError> {
            self.base.allocate_caches(cache_tokens)
        }

        fn prefill_chunk_with_cache(
            &self,
            token_ids: &[usize],
            caches: &mut [Self::LayerCache],
        ) -> Result<Vec<Vec<f32>>, BackendError> {
            self.base.prefill_chunk_with_cache(token_ids, caches)
        }

        fn make_decode_session(
            &self,
            hidden: Vec<f32>,
            caches: Vec<Self::LayerCache>,
        ) -> Self::DecodeSession {
            self.base.make_decode_session(hidden, caches)
        }

        fn hidden<'a>(&self, session: &'a Self::DecodeSession) -> &'a [f32] {
            self.base.hidden(session)
        }

        fn step(
            &self,
            session: &mut Self::DecodeSession,
            token_id: usize,
        ) -> Result<(), BackendError> {
            self.base.step(session, token_id)
        }

        fn next_token_from_hidden(
            &self,
            hidden: &[f32],
            sampling: SamplingConfig,
        ) -> Result<usize, BackendError> {
            self.base.next_token_from_hidden(hidden, sampling)
        }
    }

    fn namespace(label: &str) -> NativeTextPrefixCacheNamespace {
        NativeTextPrefixCacheNamespace {
            model_id: format!("model-{label}"),
            backend: "native-test".to_owned(),
            family: Some("test".to_owned()),
            loader: Some("native-metal".to_owned()),
            quantization: Some("bf16".to_owned()),
            repo_id: Some("org/model".to_owned()),
            resolved_commit: Some("abc123".to_owned()),
            profile: Some(label.to_owned()),
            manifest_digest: Some("digest".to_owned()),
            prompt_template: "raw".to_owned(),
            tool_schema: None,
            request_mode: "conversation=false,json_object=false,required_tool=None".to_owned(),
            cache_layout_version: 1,
            cache_tokens: 16,
            max_prefill_tokens: 4,
        }
    }

    fn driver_test_tokenizer() -> HuggingFaceTokenizer {
        let tokenizer_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/qwen36/tokenizer.json");
        HuggingFaceTokenizer::from_file(tokenizer_path).expect("tokenizer loads")
    }

    fn driver_test_request(max_tokens: u32) -> BackendRequest {
        BackendRequest {
            model: "model-test".to_owned(),
            prompt: "test".to_owned(),
            chat_context: None,
            max_tokens: Some(max_tokens),
            sampling: SamplingConfig::Greedy,
            required_tool_choice: None,
            json_object_mode: false,
            conversation_mode: false,
            cache_context: llm_backend::BackendCacheContext::default(),
        }
    }

    fn driver_for_test<A>(adapter: A) -> NativeTextDriver<A>
    where
        A: NativeTextAdapter,
    {
        NativeTextDriver::new(
            "model-test".to_owned(),
            BackendModelMetadata::new("model-test", "native-test").with_family("test"),
            driver_test_tokenizer(),
            adapter,
            8,
        )
    }

    #[test]
    fn runtime_options_apply_to_supported_native_text_families() {
        let options = NativeTextLoadOptions::with_runtime_options(NativeTextRuntimeOptions {
            eager_materialize_shards: true,
            metal_weight_cache_bytes: Some(4096),
            warm_metal_weight_cache: true,
        });

        assert!(options.qwen.eager_materialize_shards);
        assert_eq!(options.qwen.metal_weight_cache_bytes, Some(4096));
        assert!(options.qwen.warm_metal_weight_cache);
        assert!(options.gemma.eager_materialize_shards);
        assert_eq!(options.gemma.metal_weight_cache_bytes, Some(4096));
        assert!(options.gemma.warm_metal_weight_cache);
    }

    #[test]
    fn prefix_namespace_copies_metadata_and_request_context() {
        let mut metadata = BackendModelMetadata::new("model-a", "native-test").with_family("test");
        metadata.loader = Some("native-metal".to_owned());
        metadata.quantization = Some("bf16".to_owned());
        metadata.repo_id = Some("org/model".to_owned());
        metadata.resolved_commit = Some("abc123".to_owned());
        metadata.profile = Some("profile-a".to_owned());
        metadata.manifest_digest = Some("digest-a".to_owned());
        let request = BackendRequest {
            model: "model-a".to_owned(),
            prompt: "hello".to_owned(),
            chat_context: None,
            max_tokens: Some(1),
            sampling: SamplingConfig::Greedy,
            required_tool_choice: None,
            json_object_mode: true,
            conversation_mode: true,
            cache_context: llm_backend::BackendCacheContext {
                prompt_template: String::new(),
                tool_schema: Some("schema-a".to_owned()),
            },
        };

        let namespace = native_text_prefix_namespace(NativeTextPrefixNamespaceContext {
            model_id: "model-a",
            metadata: &metadata,
            request: &request,
            cache_layout_version: 7,
            cache_tokens: 64,
            max_prefill_tokens: 8,
        });

        assert_eq!(namespace.model_id, "model-a");
        assert_eq!(namespace.backend, "native-test");
        assert_eq!(namespace.family.as_deref(), Some("test"));
        assert_eq!(namespace.loader.as_deref(), Some("native-metal"));
        assert_eq!(namespace.quantization.as_deref(), Some("bf16"));
        assert_eq!(namespace.repo_id.as_deref(), Some("org/model"));
        assert_eq!(namespace.resolved_commit.as_deref(), Some("abc123"));
        assert_eq!(namespace.profile.as_deref(), Some("profile-a"));
        assert_eq!(namespace.manifest_digest.as_deref(), Some("digest-a"));
        assert_eq!(namespace.prompt_template, "raw-prompt/v1");
        assert_eq!(namespace.tool_schema.as_deref(), Some("schema-a"));
        assert_eq!(
            namespace.request_mode,
            "conversation=true,json_object=true,required_tool=None"
        );
        assert_eq!(namespace.cache_layout_version, 7);
        assert_eq!(namespace.cache_tokens, 64);
        assert_eq!(namespace.max_prefill_tokens, 8);
    }

    #[test]
    fn stop_tokens_match_literal_ids_and_tokenizer_tokens() {
        let tokenizer_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/qwen36/tokenizer.json");
        let tokenizer = HuggingFaceTokenizer::from_file(tokenizer_path).expect("tokenizer loads");
        let im_end = tokenizer
            .token_to_id("<|im_end|>")
            .expect("qwen tokenizer has im_end token") as usize;
        let stop_tokens = NativeTextStopTokens {
            token_ids: &[1],
            token_strings: &["<|im_end|>"],
        };
        let non_stop = (0..16)
            .find(|token_id| *token_id != 1 && *token_id != im_end)
            .expect("small non-stop token id exists");

        assert!(stop_tokens.contains(&tokenizer, 1));
        assert!(stop_tokens.contains(&tokenizer, im_end));
        assert!(!stop_tokens.contains(&tokenizer, non_stop));
    }

    #[test]
    fn driver_stop_token_candidate_is_not_emitted_for_blocking_generation() {
        let driver = driver_for_test(TestAdapter::new([1_usize]).with_stop_tokens(
            NativeTextStopTokens {
                token_ids: &[1],
                token_strings: &[],
            },
        ));

        let output = driver
            .generate_blocking(driver_test_request(4), CancellationToken::new())
            .expect("generation stops cleanly");

        assert_eq!(output.text, "");
        assert_eq!(output.completion_tokens, 0);
        assert_eq!(output.finish_reason, llm_api::FinishReason::Stop);
    }

    #[test]
    fn driver_stop_token_candidate_is_not_emitted_for_streaming_generation() {
        let driver = driver_for_test(TestAdapter::new([1_usize]).with_stop_tokens(
            NativeTextStopTokens {
                token_ids: &[1],
                token_strings: &[],
            },
        ));
        let (tx, mut rx) = tokio::sync::mpsc::channel(2);

        driver
            .generate_blocking_stream(driver_test_request(4), tx, CancellationToken::new())
            .expect("streaming generation stops cleanly");
        let final_chunk = rx
            .blocking_recv()
            .expect("final chunk is sent")
            .expect("final chunk is ok");
        assert_eq!(final_chunk.text, "");
        assert_eq!(final_chunk.completion_tokens, 0);
        assert_eq!(final_chunk.finish_reason, Some(llm_api::FinishReason::Stop));
        assert!(rx.blocking_recv().is_none());
    }

    #[test]
    fn driver_allows_adapter_context_sensitive_candidate_observation() {
        let driver = driver_for_test(ContextSensitiveTestAdapter::new([7_usize, 8_usize], 1));

        let output = driver
            .generate_blocking(driver_test_request(4), CancellationToken::new())
            .expect("generation stops through adapter hook");

        assert_eq!(output.text, "<7>");
        assert_eq!(output.completion_tokens, 1);
        assert_eq!(output.finish_reason, llm_api::FinishReason::Stop);
    }

    #[test]
    fn prefill_context_returns_last_hidden_from_last_chunk() {
        let cancellation = CancellationToken::new();
        let mut observed_chunks = Vec::new();

        let hidden = native_text_prefill_context_with_cache(
            "Test",
            2,
            &[1, 2, 3],
            &mut [TestCache {
                bytes: 0,
                marker: 0,
            }],
            &cancellation,
            |chunk, _caches| {
                observed_chunks.push(chunk.to_vec());
                Ok(chunk
                    .iter()
                    .map(|token| vec![*token as f32, (*token * 10) as f32])
                    .collect())
            },
        )
        .expect("prefill succeeds");

        assert_eq!(observed_chunks, vec![vec![1, 2], vec![3]]);
        assert_eq!(hidden, vec![3.0, 30.0]);
    }

    #[test]
    fn prefill_context_observes_cancellation_between_chunks() {
        let cancellation = CancellationToken::new();
        let mut calls = 0;

        let err = native_text_prefill_context_with_cache(
            "Test",
            1,
            &[1, 2],
            &mut [TestCache {
                bytes: 0,
                marker: 0,
            }],
            &cancellation,
            |chunk, _caches| {
                calls += 1;
                assert_eq!(chunk, &[1]);
                cancellation.cancel();
                Ok(vec![vec![1.0]])
            },
        )
        .expect_err("cancelled after first chunk");

        assert!(matches!(err, BackendError::Cancelled));
        assert_eq!(calls, 1);
    }

    #[test]
    fn cache_token_capacity_rounds_budget_within_position_limit() {
        let capacity = native_text_cache_token_capacity(40, 8, 32, 64, "Test")
            .expect("context and generation budget fits");

        assert_eq!(capacity, 64);
    }

    #[test]
    fn cache_token_capacity_rejects_invalid_position_limits() {
        let err = native_text_cache_token_capacity(0, 1, 1, 0, "Test")
            .expect_err("zero position limit fails closed");

        assert!(matches!(err, BackendError::UnsupportedRequest(_)));
        assert!(
            err.to_string()
                .contains("native Test model declares zero max_position_embeddings"),
            "error should identify the invalid model position limit: {err}"
        );
    }

    #[test]
    fn prefix_cache_reuses_longest_namespace_compatible_prefix() {
        let cache = NativeTextPrefixCache::new(1024);
        let metrics = NativeTextPrefixCacheMetrics::default();
        let namespace = namespace("base");
        let caches = vec![TestCache {
            bytes: 11,
            marker: 7,
        }];

        cache.store(namespace.clone(), &[1, 2], &[0.5, 1.5], &caches, &metrics);

        let hit = cache
            .lookup(&namespace, &[1, 2, 3], &metrics)
            .expect("longer prompt reuses compatible prefix");
        assert_eq!(hit.token_count, 2);
        assert_eq!(hit.hidden, vec![0.5, 1.5]);
        assert_eq!(hit.caches, caches);

        let incompatible = NativeTextPrefixCacheNamespace {
            prompt_template: "different".to_owned(),
            ..namespace
        };
        assert!(cache.lookup(&incompatible, &[1, 2], &metrics).is_none());
    }

    #[test]
    fn prefix_cache_uses_value_sizing_for_eviction_budget() {
        let cache = NativeTextPrefixCache::new(32);
        let metrics = NativeTextPrefixCacheMetrics::default();
        let namespace = namespace("budget");
        let hidden = vec![1.0; 4];

        cache.store(
            namespace.clone(),
            &[1],
            &hidden,
            &[TestCache {
                bytes: 8,
                marker: 1,
            }],
            &metrics,
        );
        cache.store(
            namespace.clone(),
            &[2],
            &hidden,
            &[TestCache {
                bytes: 8,
                marker: 2,
            }],
            &metrics,
        );

        assert!(cache.lookup(&namespace, &[1], &metrics).is_none());
        assert!(cache.lookup(&namespace, &[2], &metrics).is_some());
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot["evictions"], 1);
        assert_eq!(snapshot["resident_bytes"], 24);
    }
}
