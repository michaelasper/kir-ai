use crate::{
    DEFAULT_NATIVE_QWEN_MAX_NEW_TOKENS, NativeGemmaAdapter, NativeGemmaBackend,
    NativeGemmaLoadOptions, NativeQwenAdapter, NativeQwenBackend, NativeQwenLoadOptions,
    sync_ext::RecoverPoisonedMutex,
};
use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};
use llm_backend::{
    BackendError, BackendModelMetadata, BackendOutput, BackendRequest, BackendStreamChunk,
    ModelBackend, NativeMatvecBackend, NativeTextModelSpecRef, SafeTensorShardStore,
    SamplingConfig, native_final_norm_for_spec_ref_with_matvec,
    native_lm_head_logits_for_spec_ref_with_matvec, native_lm_head_top_k_for_spec_ref_with_matvec,
};
use llm_models::{ModelFamily, NativeTextModelSpec};
use llm_tokenizer::HuggingFaceTokenizer;
use std::{
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};
use tokio_util::sync::CancellationToken;

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
    fn observe_candidate(
        &self,
        tokenizer: &HuggingFaceTokenizer,
        emitted_tokens: &[u32],
        token_id: usize,
    ) -> Result<NativeTextCandidateDecision, BackendError>;
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

#[derive(Default)]
pub(crate) struct NativeStreamTextDeltas {
    emitted: String,
    pending: Option<String>,
}

impl NativeStreamTextDeltas {
    pub(crate) fn observe(&mut self, decoded: String) -> Result<Option<String>, BackendError> {
        if !decoded.starts_with(&self.emitted) {
            return Err(non_prefix_stream_error());
        }
        let Some(pending) = self.pending.take() else {
            self.pending = Some(decoded);
            return Ok(None);
        };
        let delta = if pending.starts_with(&self.emitted) && decoded.starts_with(&pending) {
            let delta = pending[self.emitted.len()..].to_owned();
            self.emitted = pending;
            non_empty(delta)
        } else {
            None
        };
        self.pending = Some(decoded);
        Ok(delta)
    }

    pub(crate) fn finish(&mut self, decoded: String) -> Result<Option<String>, BackendError> {
        self.pending = None;
        if !decoded.starts_with(&self.emitted) {
            return Err(non_prefix_stream_error());
        }
        let delta = decoded[self.emitted.len()..].to_owned();
        self.emitted = decoded;
        Ok(non_empty(delta))
    }
}

fn non_empty(value: String) -> Option<String> {
    (!value.is_empty()).then_some(value)
}

fn non_prefix_stream_error() -> BackendError {
    BackendError::Other(
        "native tokenizer streaming decode became non-prefix after emitted delta".to_owned(),
    )
}

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

pub(crate) fn native_text_prefill_context_with_cache<C>(
    family_display_name: &str,
    prefill_chunk_tokens: usize,
    context_tokens: &[usize],
    caches: &mut [C],
    cancellation: &CancellationToken,
    mut prefill_chunk: impl FnMut(&[usize], &mut [C]) -> Result<Vec<Vec<f32>>, BackendError>,
) -> Result<Vec<f32>, BackendError> {
    if cancellation.is_cancelled() {
        return Err(BackendError::Cancelled);
    }
    let mut hidden = None;
    for chunk in context_tokens.chunks(prefill_chunk_tokens.max(1)) {
        if cancellation.is_cancelled() {
            return Err(BackendError::Cancelled);
        }
        let hidden_states = prefill_chunk(chunk, caches)?;
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
    pub(crate) fn select_next_token(
        &self,
        hidden: &[f32],
        sampling: SamplingConfig,
    ) -> Result<usize, BackendError> {
        let final_norm =
            native_final_norm_for_spec_ref_with_matvec(self.store, self.spec, hidden, self.matvec)
                .map_err(|err| BackendError::Other(err.to_string()))?;
        if !sampling.is_greedy() {
            let logits = native_lm_head_logits_for_spec_ref_with_matvec(
                self.store,
                self.spec,
                &final_norm,
                self.chunk_rows,
                self.matvec,
            )
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

pub(crate) fn native_text_worker_stream(
    label: &'static str,
    rx: tokio::sync::mpsc::Receiver<Result<BackendStreamChunk, BackendError>>,
    worker: tokio::task::JoinHandle<()>,
) -> BoxStream<'static, Result<BackendStreamChunk, BackendError>> {
    async_stream::stream! {
        let mut rx = rx;
        let mut worker = Some(worker);
        loop {
            let Some(handle) = worker.as_mut() else {
                match rx.recv().await {
                    Some(item) => {
                        yield item;
                        continue;
                    }
                    None => break,
                }
            };
            tokio::select! {
                item = rx.recv() => {
                    match item {
                        Some(item) => yield item,
                        None => {
                            let result = worker
                                .take()
                                .expect("worker handle exists while stream watches it")
                                .await;
                            if let Err(err) = result {
                                yield Err(BackendError::Other(format!(
                                    "{label} streaming worker failed: {err}"
                                )));
                            }
                            break;
                        }
                    }
                }
                result = handle => {
                    worker = None;
                    if let Err(err) = result {
                        yield Err(BackendError::Other(format!(
                            "{label} streaming worker failed: {err}"
                        )));
                        break;
                    }
                }
            }
        }
    }
    .boxed()
}

pub(crate) fn send_backend_stream_chunk(
    tx: &tokio::sync::mpsc::Sender<Result<BackendStreamChunk, BackendError>>,
    chunk: BackendStreamChunk,
) -> Result<(), BackendError> {
    tx.blocking_send(Ok(chunk))
        .map_err(|_| BackendError::Other("stream receiver dropped".to_owned()))
}

pub(crate) trait NativeTextPrefixCacheValue: Clone {
    fn prefix_cache_entry_bytes(hidden: &[f32], caches: &[Self]) -> u64;
}

#[derive(Debug)]
pub(crate) struct NativeTextPrefixCache<C> {
    pub(crate) max_bytes: u64,
    pub(crate) inner: std::sync::Mutex<NativeTextPrefixCacheInner<C>>,
}

#[derive(Debug)]
pub(crate) struct NativeTextPrefixCacheInner<C> {
    pub(crate) entries:
        std::collections::HashMap<NativeTextPrefixCacheKey, NativeTextPrefixCacheEntry<C>>,
    pub(crate) used_bytes: u64,
    pub(crate) next_access: u64,
}

impl<C> Default for NativeTextPrefixCacheInner<C> {
    fn default() -> Self {
        Self {
            entries: std::collections::HashMap::new(),
            used_bytes: 0,
            next_access: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct NativeTextPrefixCacheNamespace {
    pub(crate) model_id: String,
    pub(crate) backend: String,
    pub(crate) family: Option<String>,
    pub(crate) loader: Option<String>,
    pub(crate) quantization: Option<String>,
    pub(crate) repo_id: Option<String>,
    pub(crate) resolved_commit: Option<String>,
    pub(crate) profile: Option<String>,
    pub(crate) manifest_digest: Option<String>,
    pub(crate) prompt_template: String,
    pub(crate) tool_schema: Option<String>,
    pub(crate) request_mode: String,
    pub(crate) cache_layout_version: u32,
    pub(crate) cache_tokens: usize,
    pub(crate) max_prefill_tokens: usize,
}

pub(crate) struct NativeTextPrefixNamespaceContext<'a> {
    pub(crate) model_id: &'a str,
    pub(crate) metadata: &'a BackendModelMetadata,
    pub(crate) request: &'a BackendRequest,
    pub(crate) cache_layout_version: u32,
    pub(crate) cache_tokens: usize,
    pub(crate) max_prefill_tokens: usize,
}

pub(crate) fn native_text_prefix_namespace(
    context: NativeTextPrefixNamespaceContext<'_>,
) -> NativeTextPrefixCacheNamespace {
    NativeTextPrefixCacheNamespace {
        model_id: context.model_id.to_owned(),
        backend: context.metadata.backend.clone(),
        family: context.metadata.family.clone(),
        loader: context.metadata.loader.clone(),
        quantization: context.metadata.quantization.clone(),
        repo_id: context.metadata.repo_id.clone(),
        resolved_commit: context.metadata.resolved_commit.clone(),
        profile: context.metadata.profile.clone(),
        manifest_digest: context.metadata.manifest_digest.clone(),
        prompt_template: native_text_cache_prompt_template(context.request),
        tool_schema: context.request.cache_context.tool_schema.clone(),
        request_mode: native_text_prefix_request_mode(context.request),
        cache_layout_version: context.cache_layout_version,
        cache_tokens: context.cache_tokens,
        max_prefill_tokens: context.max_prefill_tokens,
    }
}

pub(crate) fn native_text_prefix_request_mode(request: &BackendRequest) -> String {
    format!(
        "conversation={},json_object={},required_tool={:?}",
        request.conversation_mode, request.json_object_mode, request.required_tool_choice
    )
}

fn native_text_cache_prompt_template(request: &BackendRequest) -> String {
    if request.cache_context.prompt_template.is_empty() {
        llm_backend::BackendCacheContext::raw_prompt().prompt_template
    } else {
        request.cache_context.prompt_template.clone()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct NativeTextPrefixCacheKey {
    pub(crate) namespace: NativeTextPrefixCacheNamespace,
    pub(crate) tokens: Vec<usize>,
}

#[derive(Debug, Clone)]
pub(crate) struct NativeTextPrefixCacheEntry<C> {
    pub(crate) hidden: Vec<f32>,
    pub(crate) caches: Vec<C>,
    pub(crate) byte_len: u64,
    pub(crate) last_used: u64,
}

#[derive(Debug)]
pub(crate) struct NativeTextPrefixCacheHit<C> {
    pub(crate) token_count: usize,
    pub(crate) hidden: Vec<f32>,
    pub(crate) caches: Vec<C>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct NativeTextPrefixCacheCounters {
    pub(crate) hits: u64,
    pub(crate) misses: u64,
    pub(crate) stores: u64,
    pub(crate) evictions: u64,
    pub(crate) rejected: u64,
    pub(crate) reused_tokens: u64,
    pub(crate) bytes_stored: u64,
    pub(crate) bytes_evicted: u64,
    pub(crate) resident_bytes: u64,
    pub(crate) resident_entries: u64,
}

#[derive(Debug, Default)]
pub(crate) struct NativeTextPrefixCacheMetrics {
    counters: std::sync::Mutex<NativeTextPrefixCacheCounters>,
}

impl<C> NativeTextPrefixCache<C>
where
    C: NativeTextPrefixCacheValue,
{
    pub(crate) fn new(max_bytes: u64) -> Self {
        Self {
            max_bytes,
            inner: std::sync::Mutex::new(NativeTextPrefixCacheInner::default()),
        }
    }

    pub(crate) fn lookup(
        &self,
        namespace: &NativeTextPrefixCacheNamespace,
        tokens: &[usize],
        metrics: &NativeTextPrefixCacheMetrics,
    ) -> Option<NativeTextPrefixCacheHit<C>> {
        let mut inner = self.inner.lock_or_recover("native text prefix cache");
        let mut best_key = None;
        let mut best_len = 0;
        for key in inner.entries.keys() {
            if key.namespace == *namespace
                && key.tokens.len() > best_len
                && tokens.starts_with(&key.tokens)
            {
                best_len = key.tokens.len();
                best_key = Some(key.clone());
            }
        }
        let Some(best_key) = best_key else {
            metrics.record_miss();
            return None;
        };
        let access = inner.next_access();
        let entry = inner
            .entries
            .get_mut(&best_key)
            .expect("best prefix key came from cache entries");
        entry.last_used = access;
        metrics.record_hit(best_len as u64);
        Some(NativeTextPrefixCacheHit {
            token_count: best_len,
            hidden: entry.hidden.clone(),
            caches: entry.caches.clone(),
        })
    }

    pub(crate) fn store(
        &self,
        namespace: NativeTextPrefixCacheNamespace,
        tokens: &[usize],
        hidden: &[f32],
        caches: &[C],
        metrics: &NativeTextPrefixCacheMetrics,
    ) {
        if tokens.is_empty() {
            return;
        }
        let byte_len = C::prefix_cache_entry_bytes(hidden, caches);
        if byte_len > self.max_bytes {
            metrics.record_rejected();
            return;
        }
        let key = NativeTextPrefixCacheKey {
            namespace,
            tokens: tokens.to_vec(),
        };
        let mut inner = self.inner.lock_or_recover("native text prefix cache");
        if let Some(existing) = inner.entries.remove(&key) {
            inner.used_bytes = inner.used_bytes.saturating_sub(existing.byte_len);
        }
        while inner.used_bytes.saturating_add(byte_len) > self.max_bytes {
            let Some(lru_key) = inner
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            let Some(evicted) = inner.entries.remove(&lru_key) else {
                break;
            };
            inner.used_bytes = inner.used_bytes.saturating_sub(evicted.byte_len);
            metrics.record_eviction(evicted.byte_len);
        }
        let access = inner.next_access();
        inner.entries.insert(
            key,
            NativeTextPrefixCacheEntry {
                hidden: hidden.to_vec(),
                caches: caches.to_vec(),
                byte_len,
                last_used: access,
            },
        );
        inner.used_bytes = inner.used_bytes.saturating_add(byte_len);
        metrics.record_store(byte_len);
        metrics.record_residency(inner.used_bytes, inner.entries.len() as u64);
    }
}

impl<C> NativeTextPrefixCacheInner<C> {
    fn next_access(&mut self) -> u64 {
        let access = self.next_access;
        self.next_access = self.next_access.saturating_add(1);
        access
    }
}

impl NativeTextPrefixCacheMetrics {
    pub(crate) fn record_hit(&self, tokens: u64) {
        self.update(|counters| {
            counters.hits += 1;
            counters.reused_tokens += tokens;
        });
    }

    pub(crate) fn record_miss(&self) {
        self.update(|counters| counters.misses += 1);
    }

    pub(crate) fn record_store(&self, byte_len: u64) {
        self.update(|counters| {
            counters.stores += 1;
            counters.bytes_stored += byte_len;
        });
    }

    pub(crate) fn record_eviction(&self, byte_len: u64) {
        self.update(|counters| {
            counters.evictions += 1;
            counters.bytes_evicted += byte_len;
        });
    }

    pub(crate) fn record_rejected(&self) {
        self.update(|counters| counters.rejected += 1);
    }

    pub(crate) fn record_residency(&self, bytes: u64, entries: u64) {
        self.update(|counters| {
            counters.resident_bytes = bytes;
            counters.resident_entries = entries;
        });
    }

    pub(crate) fn snapshot(&self) -> serde_json::Value {
        let counters = *self
            .counters
            .lock_or_recover("native text prefix cache metrics");
        serde_json::json!({
            "hits": counters.hits,
            "misses": counters.misses,
            "stores": counters.stores,
            "evictions": counters.evictions,
            "rejected": counters.rejected,
            "reused_tokens": counters.reused_tokens,
            "bytes_stored": counters.bytes_stored,
            "bytes_evicted": counters.bytes_evicted,
            "resident_bytes": counters.resident_bytes,
            "resident_entries": counters.resident_entries,
        })
    }

    fn update(&self, update: impl FnOnce(&mut NativeTextPrefixCacheCounters)) {
        let mut counters = self
            .counters
            .lock_or_recover("native text prefix cache metrics");
        update(&mut counters);
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
