#[cfg(feature = "native-gemma")]
use crate::{NativeGemmaAdapter, NativeGemmaBackend, NativeGemmaLoadOptions};
#[cfg(feature = "native-qwen")]
use crate::{NativeQwenAdapter, NativeQwenBackend, NativeQwenLoadOptions};
use async_trait::async_trait;
use futures::stream::BoxStream;
#[allow(unused_imports)]
use llm_backend::{
    BackendError, BackendFinishReason, BackendModelMetadata, BackendOutput, BackendRequest,
    BackendStreamChunk, InferenceScratchpad, ModelBackend,
};
use llm_models::{ModelFamily, NativeTextModelSpec};
use std::path::Path;
use tokio_util::sync::CancellationToken;

mod driver;
mod generation;
mod prefix_cache;
mod streaming;

#[allow(unused_imports)]
pub(crate) use driver::{
    NativeTextAdapter, NativeTextCandidateDecision, NativeTextDriver, NativeTextResolvedStopTokens,
    NativeTextStopTokens,
};
#[cfg(test)]
pub(crate) use generation::native_text_prefill_context_with_cache;
#[cfg(test)]
pub(crate) use generation::sample_token_id_with_draw;
#[allow(unused_imports)]
pub(crate) use generation::{
    NativeTextNextTokenContext, NativeTextSamplingRng, native_text_cache_namespace_token_bucket,
    native_text_cache_token_capacity, resolve_native_text_max_tokens,
};
#[allow(unused_imports)]
pub(crate) use prefix_cache::{
    NativeTextPrefixCache, NativeTextPrefixCacheCounters, NativeTextPrefixCacheEntry,
    NativeTextPrefixCacheHit, NativeTextPrefixCacheInner, NativeTextPrefixCacheKey,
    NativeTextPrefixCacheMetrics, NativeTextPrefixCacheNamespace, NativeTextPrefixCacheValue,
    NativeTextPrefixNamespaceContext, native_text_prefix_namespace,
    native_text_prefix_request_mode,
};
pub(crate) use streaming::{NativeStreamTextDeltas, native_text_worker_stream};

pub const DEFAULT_NATIVE_TEXT_MAX_NEW_TOKENS: u32 = 256;
pub const DEFAULT_NATIVE_TEXT_MAX_PREFILL_TOKENS: usize = 2048;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativeTextLoadOptions {
    pub family: Option<ModelFamily>,
    #[cfg(feature = "native-qwen")]
    pub qwen: NativeQwenLoadOptions,
    #[cfg(feature = "native-gemma")]
    pub gemma: NativeGemmaLoadOptions,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativeTextRuntimeOptions {
    pub eager_materialize_shards: bool,
    pub metal_weight_cache_bytes: Option<u64>,
    pub warm_metal_weight_cache: bool,
}

#[cfg(feature = "native-qwen")]
impl From<NativeTextRuntimeOptions> for NativeQwenLoadOptions {
    fn from(value: NativeTextRuntimeOptions) -> Self {
        Self {
            eager_materialize_shards: value.eager_materialize_shards,
            metal_weight_cache_bytes: value.metal_weight_cache_bytes,
            warm_metal_weight_cache: value.warm_metal_weight_cache,
        }
    }
}

#[cfg(feature = "native-gemma")]
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
            #[cfg(feature = "native-qwen")]
            qwen: runtime.into(),
            #[cfg(feature = "native-gemma")]
            gemma: runtime.into(),
        }
    }

    #[cfg(feature = "native-qwen")]
    pub fn with_qwen_options(qwen: NativeQwenLoadOptions) -> Self {
        Self {
            family: None,
            #[cfg(feature = "native-gemma")]
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
    let mut metrics = serde_json::Map::new();
    #[cfg(feature = "native-qwen")]
    metrics.insert("qwen".to_owned(), qwen_snapshot);
    #[cfg(not(feature = "native-qwen"))]
    let _ = qwen_snapshot;
    #[cfg(feature = "native-gemma")]
    metrics.insert(
        "gemma".to_owned(),
        crate::native_gemma::native_gemma_prefix_cache_metrics_snapshot(),
    );
    serde_json::Value::Object(metrics)
}

#[derive(Clone)]
pub struct NativeTextBackend {
    inner: NativeTextBackendInner,
}

#[derive(Clone)]
enum NativeTextBackendInner {
    #[cfg(feature = "native-qwen")]
    Qwen(NativeTextDriver<NativeQwenAdapter>),
    #[cfg(feature = "native-gemma")]
    Gemma(NativeTextDriver<NativeGemmaAdapter>),
}

impl NativeTextBackend {
    pub async fn open(
        model_id: impl Into<String>,
        snapshot_path: impl AsRef<Path>,
    ) -> anyhow::Result<Self> {
        Self::open_with_options(model_id, snapshot_path, NativeTextLoadOptions::default()).await
    }

    pub async fn open_with_options(
        model_id: impl Into<String>,
        snapshot_path: impl AsRef<Path>,
        options: NativeTextLoadOptions,
    ) -> anyhow::Result<Self> {
        let model_id = model_id.into();
        let snapshot_path = snapshot_path.as_ref();
        let family = match options.family {
            Some(family) => family,
            None => infer_native_text_family(snapshot_path).await?,
        };
        match family {
            ModelFamily::Gemma => {
                #[cfg(feature = "native-gemma")]
                {
                    let driver = NativeGemmaBackend::open_with_options(
                        model_id,
                        snapshot_path,
                        options.gemma,
                    )
                    .await?
                    .into_driver();
                    Ok(Self {
                        inner: NativeTextBackendInner::Gemma(driver),
                    })
                }
                #[cfg(not(feature = "native-gemma"))]
                {
                    anyhow::bail!(
                        "native text execution for family `gemma` is disabled; rebuild llm-engine with --features native-gemma"
                    );
                }
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
                #[cfg(feature = "native-qwen")]
                {
                    let driver =
                        NativeQwenBackend::open_with_options(model_id, snapshot_path, options.qwen)
                            .await?
                            .into_driver();
                    Ok(Self {
                        inner: NativeTextBackendInner::Qwen(driver),
                    })
                }
                #[cfg(not(feature = "native-qwen"))]
                {
                    anyhow::bail!(
                        "native text execution for family `qwen` is disabled; rebuild llm-engine with --features native-qwen"
                    );
                }
            }
        }
    }

    pub fn with_max_new_tokens(mut self, max_new_tokens: u32) -> Self {
        self.inner = match self.inner {
            #[cfg(feature = "native-qwen")]
            NativeTextBackendInner::Qwen(driver) => {
                NativeTextBackendInner::Qwen(driver.with_max_new_tokens(max_new_tokens))
            }
            #[cfg(feature = "native-gemma")]
            NativeTextBackendInner::Gemma(driver) => {
                NativeTextBackendInner::Gemma(driver.with_max_new_tokens(max_new_tokens))
            }
        };
        self
    }

    pub fn with_max_prefill_tokens(mut self, max_prefill_tokens: usize) -> Self {
        self.inner = match self.inner {
            #[cfg(feature = "native-qwen")]
            NativeTextBackendInner::Qwen(driver) => {
                NativeTextBackendInner::Qwen(driver.with_max_prefill_tokens(max_prefill_tokens))
            }
            #[cfg(feature = "native-gemma")]
            NativeTextBackendInner::Gemma(driver) => {
                NativeTextBackendInner::Gemma(driver.with_max_prefill_tokens(max_prefill_tokens))
            }
        };
        self
    }
}

pub(crate) async fn infer_native_text_family(snapshot_path: &Path) -> anyhow::Result<ModelFamily> {
    let config_path = snapshot_path.join("config.json");
    let config_json = tokio::fs::read_to_string(&config_path).await.map_err(|err| {
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
            #[cfg(feature = "native-qwen")]
            NativeTextBackendInner::Qwen(backend) => backend.model_id(),
            #[cfg(feature = "native-gemma")]
            NativeTextBackendInner::Gemma(backend) => backend.model_id(),
        }
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        match &self.inner {
            #[cfg(feature = "native-qwen")]
            NativeTextBackendInner::Qwen(backend) => backend.model_metadata(),
            #[cfg(feature = "native-gemma")]
            NativeTextBackendInner::Gemma(backend) => backend.model_metadata(),
        }
    }

    async fn generate(&self, request: BackendRequest) -> Result<BackendOutput, BackendError> {
        match &self.inner {
            #[cfg(feature = "native-qwen")]
            NativeTextBackendInner::Qwen(backend) => backend.generate(request).await,
            #[cfg(feature = "native-gemma")]
            NativeTextBackendInner::Gemma(backend) => backend.generate(request).await,
        }
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        match &self.inner {
            #[cfg(feature = "native-qwen")]
            NativeTextBackendInner::Qwen(backend) => {
                backend.generate_with_cancel(request, cancellation).await
            }
            #[cfg(feature = "native-gemma")]
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
            #[cfg(feature = "native-qwen")]
            NativeTextBackendInner::Qwen(backend) => backend.generate_stream(request),
            #[cfg(feature = "native-gemma")]
            NativeTextBackendInner::Gemma(backend) => backend.generate_stream(request),
        }
    }

    fn generate_stream_with_cancel<'a>(
        &'a self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        match &self.inner {
            #[cfg(feature = "native-qwen")]
            NativeTextBackendInner::Qwen(backend) => {
                backend.generate_stream_with_cancel(request, cancellation)
            }
            #[cfg(feature = "native-gemma")]
            NativeTextBackendInner::Gemma(backend) => {
                backend.generate_stream_with_cancel(request, cancellation)
            }
        }
    }
}

#[cfg(all(test, feature = "native-qwen", feature = "native-gemma"))]
mod tests {
    use super::*;
    use crate::native_matvec::{NativeTextCacheMirrorIds, NativeTextCacheMirrorSource};
    use llm_backend::{BackendStreamProgress, SamplingConfig};
    use llm_tokenizer::HuggingFaceTokenizer;
    use std::{
        sync::{
            Arc, Mutex, Weak,
            atomic::{AtomicUsize, Ordering},
        },
        time::Duration,
    };

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

    impl NativeTextCacheMirrorSource for TestCache {
        fn append_cache_mirror_ids(&self, _ids: &mut NativeTextCacheMirrorIds) {}
    }

    #[derive(Debug)]
    struct LockObservingCache {
        bytes: u64,
        cache: Weak<NativeTextPrefixCache<LockObservingCache>>,
        cloned_while_locked: Arc<AtomicUsize>,
    }

    impl Clone for LockObservingCache {
        fn clone(&self) -> Self {
            if let Some(cache) = self.cache.upgrade()
                && cache.inner.try_lock().is_err()
            {
                self.cloned_while_locked.fetch_add(1, Ordering::SeqCst);
            }
            Self {
                bytes: self.bytes,
                cache: self.cache.clone(),
                cloned_while_locked: self.cloned_while_locked.clone(),
            }
        }
    }

    impl NativeTextPrefixCacheValue for LockObservingCache {
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
        cleanup_calls: Arc<AtomicUsize>,
        next_token_calls: Arc<AtomicUsize>,
        sampling_draws: Arc<Mutex<Vec<Option<f32>>>>,
        decoded_token_total: Arc<AtomicUsize>,
        encoded_prompt: std::sync::Arc<[u32]>,
        next_token_delay: Option<Duration>,
        fail_prefill: bool,
    }

    impl TestAdapter {
        fn new(script: impl Into<std::sync::Arc<[usize]>>) -> Self {
            Self {
                script: script.into(),
                stop_tokens: NativeTextStopTokens::default(),
                max_prefill_tokens: 4,
                prefix_cache: std::sync::Arc::new(NativeTextPrefixCache::new(1024)),
                prefix_cache_metrics: std::sync::Arc::new(NativeTextPrefixCacheMetrics::default()),
                cleanup_calls: Arc::new(AtomicUsize::new(0)),
                next_token_calls: Arc::new(AtomicUsize::new(0)),
                sampling_draws: Arc::new(Mutex::new(Vec::new())),
                decoded_token_total: Arc::new(AtomicUsize::new(0)),
                encoded_prompt: std::sync::Arc::from([42_u32]),
                next_token_delay: None,
                fail_prefill: false,
            }
        }

        fn with_stop_tokens(mut self, stop_tokens: NativeTextStopTokens) -> Self {
            self.stop_tokens = stop_tokens;
            self
        }

        fn with_prefill_failure(mut self) -> Self {
            self.fail_prefill = true;
            self
        }

        fn with_next_token_delay(mut self, delay: Duration) -> Self {
            self.next_token_delay = Some(delay);
            self
        }

        fn with_encoded_prompt(mut self, encoded_prompt: impl Into<std::sync::Arc<[u32]>>) -> Self {
            self.encoded_prompt = encoded_prompt.into();
            self
        }

        fn cleanup_calls(&self) -> Arc<AtomicUsize> {
            Arc::clone(&self.cleanup_calls)
        }

        fn next_token_calls(&self) -> Arc<AtomicUsize> {
            Arc::clone(&self.next_token_calls)
        }

        fn sampling_draws(&self) -> Arc<Mutex<Vec<Option<f32>>>> {
            Arc::clone(&self.sampling_draws)
        }

        fn decoded_token_total(&self) -> Arc<AtomicUsize> {
            Arc::clone(&self.decoded_token_total)
        }
    }

    #[async_trait]
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
            Ok(self.encoded_prompt.to_vec())
        }

        fn decode_output(
            &self,
            _tokenizer: &HuggingFaceTokenizer,
            output_ids: &[u32],
        ) -> Result<String, BackendError> {
            self.decoded_token_total
                .fetch_add(output_ids.len(), Ordering::SeqCst);
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

        async fn prefill_chunk_with_cache(
            &self,
            token_ids: &[usize],
            _caches: &mut [Self::LayerCache],
            _scratch: &mut InferenceScratchpad,
        ) -> Result<Vec<Vec<f32>>, BackendError> {
            if self.fail_prefill {
                return Err(BackendError::other("test prefill failed".to_owned()));
            }
            Ok(token_ids.iter().map(|_| vec![0.0]).collect())
        }

        fn make_decode_session(
            &self,
            hidden: Vec<f32>,
            _caches: Vec<Self::LayerCache>,
        ) -> Self::DecodeSession {
            TestDecodeSession { hidden }
        }

        fn cleanup_cache_mirrors(&self, _caches: &[Self::LayerCache]) {
            self.cleanup_calls.fetch_add(1, Ordering::SeqCst);
        }

        fn hidden<'a>(&self, session: &'a Self::DecodeSession) -> &'a [f32] {
            &session.hidden
        }

        async fn step(
            &self,
            session: &mut Self::DecodeSession,
            _token_id: usize,
            _scratch: &mut InferenceScratchpad,
        ) -> Result<(), BackendError> {
            session.hidden[0] += 1.0;
            Ok(())
        }

        async fn next_token_from_hidden(
            &self,
            hidden: &[f32],
            _sampling: SamplingConfig,
            sampling_draw: Option<f32>,
            _scratch: &mut InferenceScratchpad,
            _sampling_scratch: &mut llm_sampler::TopPSamplerScratch,
        ) -> Result<usize, BackendError> {
            self.next_token_calls.fetch_add(1, Ordering::SeqCst);
            self.sampling_draws
                .lock()
                .expect("sampling draws lock is not poisoned")
                .push(sampling_draw);
            if let Some(delay) = self.next_token_delay {
                std::thread::sleep(delay);
            }
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

    #[async_trait]
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
            stop_tokens: &NativeTextResolvedStopTokens,
            emitted_tokens: &[u32],
            token_id: usize,
        ) -> Result<NativeTextCandidateDecision, BackendError> {
            if emitted_tokens.len() >= self.stop_after_emitted {
                Ok(NativeTextCandidateDecision::Stop)
            } else {
                self.base
                    .observe_candidate(stop_tokens, emitted_tokens, token_id)
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

        async fn prefill_chunk_with_cache(
            &self,
            token_ids: &[usize],
            caches: &mut [Self::LayerCache],
            scratch: &mut InferenceScratchpad,
        ) -> Result<Vec<Vec<f32>>, BackendError> {
            self.base
                .prefill_chunk_with_cache(token_ids, caches, scratch)
                .await
        }

        fn make_decode_session(
            &self,
            hidden: Vec<f32>,
            caches: Vec<Self::LayerCache>,
        ) -> Self::DecodeSession {
            self.base.make_decode_session(hidden, caches)
        }

        fn cleanup_cache_mirrors(&self, caches: &[Self::LayerCache]) {
            self.base.cleanup_cache_mirrors(caches);
        }

        fn hidden<'a>(&self, session: &'a Self::DecodeSession) -> &'a [f32] {
            self.base.hidden(session)
        }

        async fn step(
            &self,
            session: &mut Self::DecodeSession,
            token_id: usize,
            scratch: &mut InferenceScratchpad,
        ) -> Result<(), BackendError> {
            self.base.step(session, token_id, scratch).await
        }

        async fn next_token_from_hidden(
            &self,
            hidden: &[f32],
            sampling: SamplingConfig,
            sampling_draw: Option<f32>,
            scratch: &mut InferenceScratchpad,
            sampling_scratch: &mut llm_sampler::TopPSamplerScratch,
        ) -> Result<usize, BackendError> {
            self.base
                .next_token_from_hidden(hidden, sampling, sampling_draw, scratch, sampling_scratch)
                .await
        }
    }

    fn namespace(label: &str) -> NativeTextPrefixCacheNamespace {
        NativeTextPrefixCacheNamespace {
            model_id: format!("model-{label}"),
            backend: "native-test".to_owned(),
            family: Some("test".to_owned()),
            quantization: Some("bf16".to_owned()),
            repo_id: Some("org/model".to_owned()),
            resolved_commit: Some("abc123".to_owned()),
            profile: Some(label.to_owned()),
            cache_key: format!("cache-key-{label}"),
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

    fn stream_final_chunk<A>(
        driver: &NativeTextDriver<A>,
        request: BackendRequest,
    ) -> BackendStreamChunk
    where
        A: NativeTextAdapter,
    {
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        driver
            .generate_blocking_stream(request, tx, CancellationToken::new())
            .expect("streaming generation succeeds");
        let mut final_chunk = None;
        while let Some(chunk) = rx.blocking_recv() {
            let chunk = chunk.expect("stream chunk is ok");
            if chunk.finish_reason.is_some() {
                final_chunk = Some(chunk);
            }
        }
        final_chunk.expect("streaming generation emits a final chunk")
    }

    #[test]
    fn native_text_driver_clone_shares_inner_state() {
        let driver = driver_for_test(TestAdapter::new([1_usize]));
        let clone = driver.clone();

        assert!(driver.shares_inner_state_with(&clone));
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
        metadata.quantization = Some("bf16".to_owned());
        metadata.repo_id = Some("org/model".to_owned());
        metadata.resolved_commit = Some("abc123".to_owned());
        metadata.profile = Some("profile-a".to_owned());
        let request = BackendRequest {
            model: "model-a".to_owned(),
            prompt: "hello".to_owned(),
            chat_context: None,
            max_tokens: Some(1),
            sampling: SamplingConfig::Greedy,
            required_tool_choice: None,
            json_object_mode: true,
            conversation_mode: true,
            cache_context: llm_backend::BackendCacheContext::chat_template_with_kwargs(
                "chatml/qwen/v1",
                Some("schema-a".to_owned()),
                Some(r#"{"enable_thinking":false}"#.to_owned()),
            ),
        };
        let expected_cache_key = request.cache_context.key.as_str().to_owned();

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
        assert_eq!(namespace.quantization.as_deref(), Some("bf16"));
        assert_eq!(namespace.repo_id.as_deref(), Some("org/model"));
        assert_eq!(namespace.resolved_commit.as_deref(), Some("abc123"));
        assert_eq!(namespace.profile.as_deref(), Some("profile-a"));
        assert_eq!(namespace.cache_key, expected_cache_key);
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
    fn prefix_namespace_identity_changes_with_chat_template_kwargs() {
        let metadata = BackendModelMetadata::new("model-a", "native-test").with_family("qwen");
        let mut request = driver_test_request(1);
        request.cache_context = llm_backend::BackendCacheContext::chat_template_with_kwargs(
            "chatml/qwen/v1",
            None,
            Some(r#"{"enable_thinking":false}"#.to_owned()),
        );

        let no_thinking = native_text_prefix_namespace(NativeTextPrefixNamespaceContext {
            model_id: "model-a",
            metadata: &metadata,
            request: &request,
            cache_layout_version: 1,
            cache_tokens: 16,
            max_prefill_tokens: 8,
        });
        request.cache_context = llm_backend::BackendCacheContext::chat_template_with_kwargs(
            "chatml/qwen/v1",
            None,
            Some(r#"{"enable_thinking":true}"#.to_owned()),
        );
        let thinking = native_text_prefix_namespace(NativeTextPrefixNamespaceContext {
            model_id: "model-a",
            metadata: &metadata,
            request: &request,
            cache_layout_version: 1,
            cache_tokens: 16,
            max_prefill_tokens: 8,
        });

        assert_ne!(no_thinking, thinking);
        assert_ne!(no_thinking.cache_key, thinking.cache_key);
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
            encoded_token_strings: &[],
        };
        let non_stop = (0..16)
            .find(|token_id| *token_id != 1 && *token_id != im_end)
            .expect("small non-stop token id exists");

        let resolved = stop_tokens.resolve(&tokenizer);

        assert_eq!(resolved.token_ids(), vec![1, im_end]);
        assert!(resolved.contains(1));
        assert!(resolved.contains(im_end));
        assert!(!resolved.contains(non_stop));
        assert!(
            !resolved.contains(im_end + (u32::MAX as usize) + 1),
            "candidate ids above u32::MAX must not wrap into a tokenizer stop token"
        );

        const PHRASE_STOP_STRINGS: &[&str] = &["hello rust tokenizer"];
        let missing_literal = PHRASE_STOP_STRINGS[0];
        assert!(
            tokenizer.token_to_id(missing_literal).is_none(),
            "fixture phrase should not be treated as a single literal stop token"
        );
        let literal_only_stop_tokens = NativeTextStopTokens {
            token_ids: &[],
            token_strings: PHRASE_STOP_STRINGS,
            encoded_token_strings: &[],
        }
        .resolve(&tokenizer);
        assert_eq!(literal_only_stop_tokens.token_ids(), Vec::<usize>::new());

        let phrase = PHRASE_STOP_STRINGS[0];
        assert!(
            tokenizer.token_to_id(phrase).is_none(),
            "fixture phrase should exercise encode fallback for non-vocabulary stop strings"
        );
        let phrase_ids = tokenizer
            .encode(phrase, false)
            .expect("fixture phrase encodes");
        let phrase_stop_tokens = NativeTextStopTokens {
            token_ids: &[],
            token_strings: &[],
            encoded_token_strings: PHRASE_STOP_STRINGS,
        }
        .resolve(&tokenizer);
        for token_id in phrase_ids {
            assert!(phrase_stop_tokens.contains(token_id as usize));
        }
    }

    #[test]
    fn driver_stop_token_candidate_is_not_emitted_for_blocking_generation() {
        let driver = driver_for_test(TestAdapter::new([1_usize]).with_stop_tokens(
            NativeTextStopTokens {
                token_ids: &[1],
                token_strings: &[],
                encoded_token_strings: &[],
            },
        ));

        let output = driver
            .generate_blocking(driver_test_request(4), CancellationToken::new())
            .expect("generation stops cleanly");

        assert_eq!(output.text, "");
        assert_eq!(output.completion_tokens, 0);
        assert_eq!(output.finish_reason, BackendFinishReason::Stop);
    }

    #[test]
    fn driver_stop_token_candidate_is_not_emitted_for_streaming_generation() {
        let driver = driver_for_test(TestAdapter::new([1_usize]).with_stop_tokens(
            NativeTextStopTokens {
                token_ids: &[1],
                token_strings: &[],
                encoded_token_strings: &[],
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
        assert_eq!(final_chunk.finish_reason, Some(BackendFinishReason::Stop));
        assert!(rx.blocking_recv().is_none());
    }

    #[test]
    fn streaming_generation_decodes_each_output_token_once() {
        let adapter = TestAdapter::new([1_usize, 2, 3, 4]);
        let decoded_token_total = adapter.decoded_token_total();
        let driver = driver_for_test(adapter);
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);

        driver
            .generate_blocking_stream(driver_test_request(4), tx, CancellationToken::new())
            .expect("streaming generation succeeds");

        let mut text = String::new();
        while let Some(chunk) = rx.blocking_recv() {
            let chunk = chunk.expect("stream chunk is ok");
            text.push_str(&chunk.text);
        }

        assert_eq!(text, "<1><2><3><4>");
        assert_eq!(decoded_token_total.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn driver_supplies_rng_draws_only_for_non_greedy_sampling() {
        let greedy_adapter = TestAdapter::new([1_usize, 2]);
        let greedy_draws = greedy_adapter.sampling_draws();
        let greedy_driver = driver_for_test(greedy_adapter);

        greedy_driver
            .generate_blocking(driver_test_request(2), CancellationToken::new())
            .expect("greedy generation succeeds");

        assert_eq!(
            *greedy_draws
                .lock()
                .expect("greedy sampling draws lock is not poisoned"),
            vec![None, None]
        );

        let top_p_adapter = TestAdapter::new([1_usize, 2]);
        let top_p_draws = top_p_adapter.sampling_draws();
        let top_p_driver = driver_for_test(top_p_adapter);
        let mut request = driver_test_request(2);
        request.sampling = SamplingConfig::TopP {
            temperature: 1.0,
            top_p: 0.9,
        };

        top_p_driver
            .generate_blocking(request, CancellationToken::new())
            .expect("top-p generation succeeds");

        let top_p_draws = top_p_draws
            .lock()
            .expect("top-p sampling draws lock is not poisoned")
            .clone();
        assert_eq!(top_p_draws.len(), 2);
        assert!(
            top_p_draws
                .into_iter()
                .all(|draw| { matches!(draw, Some(value) if (0.0..1.0).contains(&value)) })
        );
    }

    #[test]
    fn driver_reports_prefix_cache_miss_and_hit_for_blocking_generation() {
        let driver = driver_for_test(TestAdapter::new([1_usize]));

        let first = driver
            .generate_blocking(driver_test_request(1), CancellationToken::new())
            .expect("first generation succeeds");
        let second = driver
            .generate_blocking(driver_test_request(1), CancellationToken::new())
            .expect("second generation succeeds");

        assert_eq!(first.prompt_cached_tokens, Some(0));
        assert_eq!(second.prompt_cached_tokens, Some(1));
    }

    #[test]
    fn driver_reports_prefix_cache_miss_and_hit_for_streaming_generation() {
        let driver = driver_for_test(TestAdapter::new([1_usize]));

        let first = stream_final_chunk(&driver, driver_test_request(1));
        let second = stream_final_chunk(&driver, driver_test_request(1));

        assert_eq!(first.prompt_cached_tokens, Some(0));
        assert_eq!(second.prompt_cached_tokens, Some(1));
    }

    #[test]
    fn streaming_generation_emits_prefill_progress_after_each_uncached_chunk() {
        let driver = driver_for_test(
            TestAdapter::new([1_usize]).with_encoded_prompt([10_u32, 11, 12, 13, 14]),
        )
        .with_max_prefill_tokens(2);
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);

        driver
            .generate_blocking_stream(driver_test_request(1), tx, CancellationToken::new())
            .expect("streaming generation succeeds");

        let mut progress = Vec::new();
        while let Some(chunk) = rx.blocking_recv() {
            let chunk = chunk.expect("stream chunk is ok");
            if let Some(event) = chunk.progress {
                progress.push(event);
            }
        }

        assert_eq!(
            progress,
            vec![
                BackendStreamProgress::PrefillProgress {
                    chunk: 1,
                    total: 3,
                    tokens: 2,
                    total_tokens: 5,
                },
                BackendStreamProgress::PrefillProgress {
                    chunk: 2,
                    total: 3,
                    tokens: 4,
                    total_tokens: 5,
                },
                BackendStreamProgress::PrefillProgress {
                    chunk: 3,
                    total: 3,
                    tokens: 5,
                    total_tokens: 5,
                },
            ]
        );
    }

    #[test]
    fn driver_allows_adapter_context_sensitive_candidate_observation() {
        let driver = driver_for_test(ContextSensitiveTestAdapter::new([7_usize, 8_usize], 1));

        let output = driver
            .generate_blocking(driver_test_request(4), CancellationToken::new())
            .expect("generation stops through adapter hook");

        assert_eq!(output.text, "<7>");
        assert_eq!(output.completion_tokens, 1);
        assert_eq!(output.finish_reason, BackendFinishReason::Stop);
    }

    #[test]
    fn driver_cleans_cache_mirrors_when_prefill_fails_before_session_handoff() {
        let adapter = TestAdapter::new([1_usize]).with_prefill_failure();
        let cleanup_calls = adapter.cleanup_calls();
        let driver = driver_for_test(adapter);

        let err = driver
            .generate_blocking(driver_test_request(1), CancellationToken::new())
            .expect_err("prefill failure is returned");

        assert!(err.to_string().contains("test prefill failed"));
        assert_eq!(cleanup_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn driver_does_not_clean_cache_mirrors_after_successful_session_handoff() {
        let adapter = TestAdapter::new([1_usize]);
        let cleanup_calls = adapter.cleanup_calls();
        let driver = driver_for_test(adapter);

        driver
            .generate_blocking(driver_test_request(1), CancellationToken::new())
            .expect("generation succeeds");

        assert_eq!(cleanup_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn driver_generate_with_cancel_runs_native_work_off_async_runtime() {
        let adapter = TestAdapter::new([1_usize]).with_next_token_delay(Duration::from_millis(750));
        let next_token_calls = adapter.next_token_calls();
        let driver = driver_for_test(adapter);

        let generation = tokio::spawn(async move {
            driver
                .generate_with_cancel(driver_test_request(1), CancellationToken::new())
                .await
        });

        let deadline = tokio::time::Instant::now() + Duration::from_millis(200);
        while next_token_calls.load(Ordering::SeqCst) == 0 && tokio::time::Instant::now() < deadline
        {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        assert_eq!(next_token_calls.load(Ordering::SeqCst), 1);
        assert!(
            !generation.is_finished(),
            "native generation should not block the async runtime while CPU work is running"
        );
        let output = generation
            .await
            .expect("generation task joins")
            .expect("generation succeeds");
        assert_eq!(output.completion_tokens, 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn driver_generate_with_cancel_cancels_worker_when_future_is_dropped() {
        let adapter = TestAdapter::new([1_usize]).with_next_token_delay(Duration::from_millis(750));
        let next_token_calls = adapter.next_token_calls();
        let driver = driver_for_test(adapter);
        let cancellation = CancellationToken::new();
        let worker_cancellation = cancellation.clone();

        let generation = tokio::spawn(async move {
            driver
                .generate_with_cancel(driver_test_request(1), worker_cancellation)
                .await
        });

        let deadline = tokio::time::Instant::now() + Duration::from_millis(200);
        while next_token_calls.load(Ordering::SeqCst) == 0 && tokio::time::Instant::now() < deadline
        {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(next_token_calls.load(Ordering::SeqCst), 1);

        generation.abort();
        assert!(
            generation
                .await
                .expect_err("generation task is aborted")
                .is_cancelled()
        );
        assert!(
            cancellation.is_cancelled(),
            "dropping the async request future should signal the blocking native worker"
        );
    }

    #[test]
    fn prefill_context_returns_last_hidden_from_last_chunk() {
        let cancellation = CancellationToken::new();
        let mut observed_chunks = Vec::new();

        let mut prefill_caches = [TestCache {
            bytes: 0,
            marker: 0,
        }];
        let mut prefill_scratch = InferenceScratchpad::new();
        let hidden = native_text_prefill_context_with_cache(
            "Test",
            2,
            &[1, 2, 3],
            &mut prefill_caches,
            &cancellation,
            &mut prefill_scratch,
            |chunk, _caches, _scratch| {
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

        let mut cancel_caches = [TestCache {
            bytes: 0,
            marker: 0,
        }];
        let mut cancel_scratch = InferenceScratchpad::new();
        let err = native_text_prefill_context_with_cache(
            "Test",
            1,
            &[1, 2],
            &mut cancel_caches,
            &cancellation,
            &mut cancel_scratch,
            |chunk, _caches, _scratch| {
                calls += 1;
                assert_eq!(chunk, &[1]);
                cancellation.cancel();
                Ok(vec![vec![1.0]])
            },
        )
        .expect_err("cancelled after first chunk");

        assert!(err.is_cancelled());
        assert_eq!(calls, 1);
    }

    #[test]
    fn cache_token_capacity_uses_exact_budget_within_position_limit() {
        let capacity = native_text_cache_token_capacity(40, 8, 32, 64, "Test")
            .expect("context and generation budget fits");

        assert_eq!(capacity, 48);
    }

    #[test]
    fn cache_namespace_token_bucket_keeps_prefix_identity_stable() {
        let capacity = native_text_cache_token_capacity(40, 8, 32, 64, "Test")
            .expect("context and generation budget fits");
        let bucket = native_text_cache_namespace_token_bucket(capacity, 64, "Test")
            .expect("namespace bucket fits");

        assert_eq!(capacity, 48);
        assert_eq!(bucket, 64);
    }

    #[test]
    fn cache_token_capacity_rejects_invalid_position_limits() {
        let err = native_text_cache_token_capacity(0, 1, 1, 0, "Test")
            .expect_err("zero position limit fails closed");

        assert!(err.is_unsupported_request());
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
            cache_key: "different".to_owned(),
            ..namespace
        };
        assert!(cache.lookup(&incompatible, &[1, 2], &metrics).is_none());
    }

    #[test]
    fn prefix_cache_lookup_skips_capacity_incompatible_entries() {
        let cache = NativeTextPrefixCache::new(1024);
        let metrics = NativeTextPrefixCacheMetrics::default();
        let namespace = namespace("capacity");
        let caches = vec![TestCache {
            bytes: 11,
            marker: 7,
        }];

        cache.store(namespace.clone(), &[1, 2], &[0.5, 1.5], &caches, &metrics);

        assert!(
            cache
                .lookup_compatible(&namespace, &[1, 2, 3], &metrics, |caches| {
                    caches.iter().all(|cache| cache.marker != 7)
                })
                .is_none()
        );

        let hit = cache
            .lookup_compatible(&namespace, &[1, 2, 3], &metrics, |caches| {
                caches.iter().all(|cache| cache.marker == 7)
            })
            .expect("compatible entry is reusable");
        assert_eq!(hit.token_count, 2);
    }

    #[test]
    fn prefix_cache_clones_payloads_outside_global_lock() {
        let cache = Arc::new(NativeTextPrefixCache::new(1024));
        let metrics = NativeTextPrefixCacheMetrics::default();
        let namespace = namespace("clone-lock");
        let cloned_while_locked = Arc::new(AtomicUsize::new(0));
        let caches = vec![LockObservingCache {
            bytes: 8,
            cache: Arc::downgrade(&cache),
            cloned_while_locked: cloned_while_locked.clone(),
        }];

        cache.store(namespace.clone(), &[1, 2], &[0.5, 1.5], &caches, &metrics);
        let hit = cache
            .lookup(&namespace, &[1, 2, 3], &metrics)
            .expect("compatible longer prompt reuses stored prefix");

        assert_eq!(hit.token_count, 2);
        assert_eq!(
            cloned_while_locked.load(Ordering::SeqCst),
            0,
            "prefix cache must not clone layer-cache payloads while holding its global lock"
        );
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
