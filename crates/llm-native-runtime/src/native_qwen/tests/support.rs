use super::*;
use crate::native_matvec::{
    Bf16MatrixBufferCache, Bf16MatrixCacheKey,
    DEFAULT_NATIVE_TEXT_METAL_WEIGHT_CACHE_BYTES as DEFAULT_NATIVE_QWEN_METAL_WEIGHT_CACHE_BYTES,
    NativeTextMetalWarmup as NativeQwenMetalWarmup,
};
use crate::native_text::{
    NativeStreamTextDeltas, NativeTextCandidateDecision, NativeTextStopTokens,
    native_text_cache_token_capacity, native_text_prefill_context_with_cache,
    native_text_worker_stream, resolve_native_text_max_tokens, sample_token_id_with_draw,
};
use crate::sync_ext::FailPoisonedMutex;
use futures::StreamExt;
use llm_backend::native::{
    CpuNativeMatvecBackend, InferenceScratchpad, LayerKvCache, MathError, NativeMatvecBackend,
    SafeTensorShardStore, TensorLoadError, qwen_layer_caches_for_spec,
    qwen_prefill_sequence_with_cache, qwen_static_f32_tensors_for_spec,
};
use llm_backend_contracts::{BackendCacheContext, BackendToolChoice};
use llm_models::QwenModelSpec;
use llm_models::{ModelFamilyAdapter, QwenFamilyAdapter};
use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

type TinyBf16Tensor = (String, Vec<usize>, Vec<f32>);
type TinyBf16ShardMap = std::collections::BTreeMap<String, Vec<TinyBf16Tensor>>;

fn test_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime")
}

fn open_qwen_backend_blocking(model_id: &str, snapshot: &Path) -> NativeQwenBackend {
    test_runtime()
        .block_on(NativeQwenBackend::open(model_id, snapshot))
        .expect("backend opens snapshot")
}

fn open_qwen_backend_with_options_blocking(
    model_id: &str,
    snapshot: &Path,
    options: NativeQwenLoadOptions,
) -> NativeQwenBackend {
    test_runtime()
        .block_on(NativeQwenBackend::open_with_options(
            model_id, snapshot, options,
        ))
        .expect("backend opens snapshot")
}

async fn start_qwen_decode_session(
    backend: &NativeQwenBackend,
    context_tokens: &[usize],
    max_new_tokens: u32,
    request: &BackendRequest,
    cancellation: &CancellationToken,
) -> Result<NativeQwenDecodeSession, BackendError> {
    let mut scratch = InferenceScratchpad::new();
    backend
        .driver
        .start_decode_session(
            context_tokens,
            max_new_tokens,
            request,
            cancellation,
            &mut scratch,
        )
        .await
}

async fn select_qwen_token(
    backend: &NativeQwenBackend,
    hidden: &[f32],
    sampling: SamplingConfig,
) -> Result<usize, BackendError> {
    let sampling_draw = if sampling.is_greedy() {
        None
    } else {
        let mut sampling_rng = crate::native_text::NativeTextSamplingRng::from_entropy();
        Some(sampling_rng.draw_f32())
    };
    backend
        .driver
        .adapter
        .next_token_from_hidden(
            hidden,
            sampling,
            sampling_draw,
            &mut llm_sampler::TopPSamplerScratch::new(),
        )
        .await
}

#[derive(Default)]
struct TestQwenCacheMirrorCleaner {
    calls: AtomicUsize,
    cache_count: AtomicUsize,
}

impl NativeTextCacheMirrorCleaner<QwenLayerCache> for TestQwenCacheMirrorCleaner {
    fn cleanup_cache_mirrors(&self, caches: &[QwenLayerCache]) {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.cache_count.fetch_add(caches.len(), Ordering::SeqCst);
    }
}
