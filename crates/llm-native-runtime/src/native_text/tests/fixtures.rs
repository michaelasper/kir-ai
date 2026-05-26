use super::*;
use crate::native_matvec::{NativeTextCacheMirrorIds, NativeTextCacheMirrorSource};
use llm_backend_contracts::{
    BackendChatContext, BackendChatMessage, BackendChatRole, BackendFailureClass,
    BackendPrefillChunkAdmission, BackendPrefillChunkAdmissionHook, BackendStreamProgress,
    BackendToolChoice, SamplingConfig,
};
use llm_tokenizer::{HuggingFaceTokenizer, HuggingFaceTokenizerIdentity};
use std::{
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::sync::Notify;

#[derive(Debug, Clone, PartialEq, Eq)]
struct TestCache {
    bytes: u64,
    marker: u32,
}
impl NativeTextPrefixCacheValue for TestCache {
    type PrefixCacheState = Self;

    fn prefix_cache_state(caches: &[Self]) -> Vec<Self::PrefixCacheState> {
        caches.to_vec()
    }

    fn prefix_cache_from_state(states: &[Self::PrefixCacheState]) -> Option<Vec<Self>> {
        Some(states.to_vec())
    }

    fn prefix_cache_entry_bytes(hidden: &[f32], states: &[Self::PrefixCacheState]) -> u64 {
        std::mem::size_of_val(hidden) as u64
            + states.iter().map(|cache| cache.bytes).sum::<u64>()
    }
}

impl NativeTextDiskCacheValue for TestCache {
    fn encode_disk_block_states(
        states: &[Self::PrefixCacheState],
        block_start: usize,
        block_token_count: usize,
        sink: &mut NativeTextDiskCacheTensorSink,
    ) -> Result<Vec<NativeTextDiskCacheLayerLayout>, NativeTextDiskCacheError> {
        let values = states[block_start..block_start + block_token_count]
            .iter()
            .map(|state| state.marker as f32)
            .collect::<Vec<_>>();
        sink.push_f32("test.markers", vec![values.len()], values)?;
        Ok(vec![NativeTextDiskCacheLayerLayout::test_marker_tensor(
            "test.markers",
        )])
    }

    fn decode_disk_states(
        layouts: &[NativeTextDiskCacheLayerLayout],
        archive: &NativeTextDiskCacheTensorArchive<'_>,
    ) -> Result<Vec<Self::PrefixCacheState>, NativeTextDiskCacheError> {
        let Some(layout) = layouts.first() else {
            return Err(NativeTextDiskCacheError::integrity(
                "missing test marker layout",
            ));
        };
        let tensor = layout
            .test_marker_tensor_name()
            .ok_or_else(|| NativeTextDiskCacheError::integrity("wrong test marker layout"))?;
        archive
            .f32_tensor(tensor)?
            .into_iter()
            .map(|marker| {
                if marker.fract() != 0.0 || marker < 0.0 {
                    return Err(NativeTextDiskCacheError::integrity(
                        "test marker must be a non-negative integer",
                    ));
                }
                Ok(TestCache {
                    bytes: std::mem::size_of::<TestCache>() as u64,
                    marker: marker as u32,
                })
            })
            .collect()
    }

    fn assemble_disk_block_states(
        blocks: &[NativeTextDiskCacheStateBlock<Self::PrefixCacheState>],
    ) -> Result<Vec<Self::PrefixCacheState>, NativeTextDiskCacheError> {
        Ok(blocks
            .iter()
            .flat_map(|block| block.states.iter().cloned())
            .collect())
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
    type PrefixCacheState = Self;

    fn prefix_cache_state(caches: &[Self]) -> Vec<Self::PrefixCacheState> {
        caches.to_vec()
    }

    fn prefix_cache_from_state(states: &[Self::PrefixCacheState]) -> Option<Vec<Self>> {
        Some(states.to_vec())
    }

    fn prefix_cache_entry_bytes(hidden: &[f32], states: &[Self::PrefixCacheState]) -> u64 {
        std::mem::size_of_val(hidden) as u64
            + states.iter().map(|cache| cache.bytes).sum::<u64>()
    }
}

#[derive(Clone)]
struct TestDecodeSession {
    hidden: Vec<f32>,
}

#[derive(Clone)]
enum TestDecodeOutput {
    TokenTags,
    UnicodeBoundary,
}

#[derive(Clone)]
struct TestAdapter {
    script: std::sync::Arc<[usize]>,
    stop_tokens: NativeTextStopTokens,
    max_prefill_tokens: usize,
    max_position_embeddings: u32,
    decode_output: TestDecodeOutput,
    prefix_cache: std::sync::Arc<NativeTextPrefixCache<TestCache>>,
    prefix_cache_metrics: std::sync::Arc<NativeTextPrefixCacheMetrics>,
    cleanup_calls: Arc<AtomicUsize>,
    cleanup_markers: Arc<Mutex<Vec<Vec<u32>>>>,
    next_token_calls: Arc<AtomicUsize>,
    sampling_draws: Arc<Mutex<Vec<Option<f32>>>>,
    decoded_token_total: Arc<AtomicUsize>,
    stream_decoded_token_total: Arc<AtomicUsize>,
    encoded_prompt: std::sync::Arc<[u32]>,
    next_token_delay: Option<Duration>,
    fail_prefill: bool,
    fail_after_prefill_chunk: Option<usize>,
    cancel_on_prefill: Option<CancellationToken>,
    cancel_after_prefill_chunk: Option<(CancellationToken, usize)>,
    prefill_chunk_calls: Arc<AtomicUsize>,
    blocking_prefill: Option<Arc<BlockingPrefill>>,
}

impl TestAdapter {
    fn new(script: impl Into<std::sync::Arc<[usize]>>) -> Self {
        Self {
            script: script.into(),
            stop_tokens: NativeTextStopTokens::default(),
            max_prefill_tokens: 4,
            max_position_embeddings: 16,
            decode_output: TestDecodeOutput::TokenTags,
            prefix_cache: std::sync::Arc::new(NativeTextPrefixCache::new(1024)),
            prefix_cache_metrics: std::sync::Arc::new(NativeTextPrefixCacheMetrics::default()),
            cleanup_calls: Arc::new(AtomicUsize::new(0)),
            cleanup_markers: Arc::new(Mutex::new(Vec::new())),
            next_token_calls: Arc::new(AtomicUsize::new(0)),
            sampling_draws: Arc::new(Mutex::new(Vec::new())),
            decoded_token_total: Arc::new(AtomicUsize::new(0)),
            stream_decoded_token_total: Arc::new(AtomicUsize::new(0)),
            encoded_prompt: std::sync::Arc::from([42_u32]),
            next_token_delay: None,
            fail_prefill: false,
            fail_after_prefill_chunk: None,
            cancel_on_prefill: None,
            cancel_after_prefill_chunk: None,
            prefill_chunk_calls: Arc::new(AtomicUsize::new(0)),
            blocking_prefill: None,
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

    fn with_prefill_failure_after_chunk(mut self, chunk: usize) -> Self {
        self.fail_after_prefill_chunk = Some(chunk);
        self
    }

    fn with_prefill_cancellation(mut self, cancellation: CancellationToken) -> Self {
        self.cancel_on_prefill = Some(cancellation);
        self
    }

    fn with_prefill_cancellation_after_chunk(
        mut self,
        cancellation: CancellationToken,
        chunk: usize,
    ) -> Self {
        self.cancel_after_prefill_chunk = Some((cancellation, chunk));
        self
    }

    fn with_prefix_cache_bytes(mut self, prefix_cache_bytes: u64) -> Self {
        self.prefix_cache = std::sync::Arc::new(NativeTextPrefixCache::new(prefix_cache_bytes));
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

    fn with_unicode_boundary_decode(mut self) -> Self {
        self.decode_output = TestDecodeOutput::UnicodeBoundary;
        self
    }

    fn with_max_position_embeddings(mut self, max_position_embeddings: u32) -> Self {
        self.max_position_embeddings = max_position_embeddings;
        self
    }

    fn cleanup_calls(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.cleanup_calls)
    }

    fn cleanup_markers(&self) -> Arc<Mutex<Vec<Vec<u32>>>> {
        Arc::clone(&self.cleanup_markers)
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

    fn stream_decoded_token_total(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.stream_decoded_token_total)
    }

    fn prefill_chunk_calls(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.prefill_chunk_calls)
    }

    fn with_blocking_prefill(mut self, blocking_prefill: Arc<BlockingPrefill>) -> Self {
        self.blocking_prefill = Some(blocking_prefill);
        self
    }
}

#[derive(Debug)]
struct BlockingPrefillAdmission {
    release: Notify,
    calls: AtomicUsize,
}

impl BlockingPrefillAdmission {
    fn new() -> Self {
        Self {
            release: Notify::new(),
            calls: AtomicUsize::new(0),
        }
    }
}

#[derive(Debug)]
struct BlockingPrefill {
    release: Notify,
    started: Notify,
    dropped: Notify,
    started_calls: AtomicUsize,
    dropped_calls: AtomicUsize,
}

impl BlockingPrefill {
    fn new() -> Self {
        Self {
            release: Notify::new(),
            started: Notify::new(),
            dropped: Notify::new(),
            started_calls: AtomicUsize::new(0),
            dropped_calls: AtomicUsize::new(0),
        }
    }

    async fn wait_started(&self) {
        while self.started_calls.load(Ordering::SeqCst) == 0 {
            self.started.notified().await;
        }
    }

    async fn wait_dropped(&self) {
        while self.dropped_calls.load(Ordering::SeqCst) == 0 {
            self.dropped.notified().await;
        }
    }

    fn dropped_calls(&self) -> usize {
        self.dropped_calls.load(Ordering::SeqCst)
    }
}

struct BlockingPrefillGuard {
    blocking_prefill: Arc<BlockingPrefill>,
}

impl Drop for BlockingPrefillGuard {
    fn drop(&mut self) {
        self.blocking_prefill
            .dropped_calls
            .fetch_add(1, Ordering::SeqCst);
        self.blocking_prefill.dropped.notify_waiters();
    }
}

#[async_trait]
impl BackendPrefillChunkAdmission for BlockingPrefillAdmission {
    async fn wait_for_next_chunk(
        &self,
        progress: BackendStreamProgress,
    ) -> Result<(), BackendError> {
        assert_eq!(
            progress,
            BackendStreamProgress::PrefillProgress {
                chunk: 1,
                total: 2,
                tokens: 2,
                total_tokens: 4,
            }
        );
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.release.notified().await;
        Ok(())
    }
}

struct TestStreamDecoder {
    decode_output: TestDecodeOutput,
    decoded_token_total: Arc<AtomicUsize>,
    unicode_boundary_started: bool,
}

impl NativeTextStreamDecoder for TestStreamDecoder {
    fn step(&mut self, token_id: u32) -> Result<Option<String>, BackendError> {
        self.decoded_token_total.fetch_add(1, Ordering::SeqCst);
        Ok(match self.decode_output {
            TestDecodeOutput::TokenTags => Some(format!("<{token_id}>")),
            TestDecodeOutput::UnicodeBoundary => {
                if self.unicode_boundary_started && token_id == 2 {
                    self.unicode_boundary_started = false;
                    Some("é".to_owned())
                } else if token_id == 1 {
                    self.unicode_boundary_started = true;
                    None
                } else {
                    Some(format!("<{token_id}>"))
                }
            }
        })
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
        Ok(match self.decode_output {
            TestDecodeOutput::TokenTags => output_ids
                .iter()
                .map(|token_id| format!("<{token_id}>"))
                .collect::<String>(),
            TestDecodeOutput::UnicodeBoundary => match output_ids {
                [1] | [2] => "�".to_owned(),
                [1, 2] => "é".to_owned(),
                _ => output_ids
                    .iter()
                    .map(|token_id| format!("<{token_id}>"))
                    .collect::<String>(),
            },
        })
    }

    fn stream_decoder<'tokenizer>(
        &self,
        _tokenizer: &'tokenizer HuggingFaceTokenizer,
    ) -> Box<dyn NativeTextStreamDecoder + 'tokenizer> {
        Box::new(TestStreamDecoder {
            decode_output: self.decode_output.clone(),
            decoded_token_total: Arc::clone(&self.stream_decoded_token_total),
            unicode_boundary_started: false,
        })
    }

    fn stop_tokens(&self) -> NativeTextStopTokens {
        self.stop_tokens
    }

    fn max_position_embeddings(&self) -> u32 {
        self.max_position_embeddings
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
        tokenizer_identity: &HuggingFaceTokenizerIdentity,
        _request: &BackendRequest,
        cache_tokens: usize,
    ) -> NativeTextPrefixCacheNamespace {
        NativeTextPrefixCacheNamespace {
            cache_tokens,
            tokenizer_kind: tokenizer_identity.kind.clone(),
            tokenizer_hash: tokenizer_identity.content_hash.clone(),
            tokenizer_normalization: tokenizer_identity.normalization.clone(),
            adapter_settings: self.prefix_cache_adapter_settings().to_owned(),
            ..namespace("driver-test")
        }
    }

    fn prefix_cache_adapter_settings(&self) -> &'static str {
        "native-test-adapter/v1"
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
        cancellation: &CancellationToken,
    ) -> Result<Vec<Vec<f32>>, BackendError> {
        let chunk_call = self.prefill_chunk_calls.fetch_add(1, Ordering::SeqCst) + 1;
        if let Some(cancellation) = &self.cancel_on_prefill {
            cancellation.cancel();
        }
        if let Some((cancellation, chunk)) = &self.cancel_after_prefill_chunk
            && chunk_call == *chunk
        {
            cancellation.cancel();
        }
        if self.fail_prefill {
            return Err(BackendError::other("test prefill failed".to_owned()));
        }
        if let Some(chunk) = self.fail_after_prefill_chunk
            && chunk_call == chunk
        {
            return Err(BackendError::other("test prefill failed".to_owned()));
        }
        if let Some(blocking_prefill) = &self.blocking_prefill {
            blocking_prefill
                .started_calls
                .fetch_add(1, Ordering::SeqCst);
            blocking_prefill.started.notify_waiters();
            let _guard = BlockingPrefillGuard {
                blocking_prefill: Arc::clone(blocking_prefill),
            };
            tokio::select! {
                biased;
                () = cancellation.cancelled() => return Err(BackendError::cancelled()),
                () = blocking_prefill.release.notified() => {}
            }
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

    fn cleanup_cache_mirrors(&self, caches: &[Self::LayerCache]) {
        let markers = caches.iter().map(|cache| cache.marker).collect();
        self.cleanup_markers
            .lock()
            .expect("cleanup markers lock is not poisoned")
            .push(markers);
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
        tokenizer_identity: &HuggingFaceTokenizerIdentity,
        request: &BackendRequest,
        cache_tokens: usize,
    ) -> NativeTextPrefixCacheNamespace {
        self.base
            .prefix_cache_namespace(tokenizer_identity, request, cache_tokens)
    }

    fn prefix_cache_adapter_settings(&self) -> &'static str {
        self.base.prefix_cache_adapter_settings()
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
        cancellation: &CancellationToken,
    ) -> Result<Vec<Vec<f32>>, BackendError> {
        self.base
            .prefill_chunk_with_cache(token_ids, caches, scratch, cancellation)
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
        sampling_scratch: &mut llm_sampler::TopPSamplerScratch,
    ) -> Result<usize, BackendError> {
        self.base
            .next_token_from_hidden(hidden, sampling, sampling_draw, sampling_scratch)
            .await
    }
}
