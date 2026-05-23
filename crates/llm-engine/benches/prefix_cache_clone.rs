//! Prefix-cache hit clone microbenchmarks.
//!
//! These benches construct Qwen3.6-like cache payloads from local cache types
//! only. They do not open model snapshots, load weights, contact the network,
//! or run inference. Cold-miss cases hold the same large payload but avoid the
//! hit path, making the hit cases comparable as payload clone measurements.

use std::{
    hint::black_box,
    time::{Duration, Instant},
};

use llm_backend::native::{
    LayerKvCache, LinearAttentionCache, QwenLayerCache, QwenLayerCachePrefixState,
};

mod sync_ext {
    pub(crate) use llm_util::sync_ext::FailPoisonedMutex;
}

#[allow(dead_code)]
#[path = "../src/native_text/prefix_cache.rs"]
mod prefix_cache;

use prefix_cache::{
    NativeTextPrefixCache, NativeTextPrefixCacheMetrics, NativeTextPrefixCacheNamespace,
    NativeTextPrefixCacheValue,
};

const HIDDEN_SIZE: usize = 2_048;
const PREFIX_TOKENS: usize = 256;

// Qwen3.6 35B-A3B text-cache dimensions from the local qwen36 fixture:
// 40 layers total, with 10 full-attention layers and 30 linear-attention layers.
const FULL_LAYER_COUNT: usize = 10;
const FULL_CACHE_TOKENS: usize = 2_048;
const FULL_KV_HEADS: usize = 2;
const FULL_HEAD_DIM: usize = 256;

const LINEAR_LAYER_COUNT: usize = 30;
const LINEAR_CONV_KERNEL: usize = 4;
const LINEAR_KEY_HEADS: usize = 16;
const LINEAR_VALUE_HEADS: usize = 32;
const LINEAR_KEY_HEAD_DIM: usize = 128;
const LINEAR_VALUE_HEAD_DIM: usize = 128;

const COLD_MISS_ITERATIONS: usize = 20_000;
const HIT_CLONE_ITERATIONS: usize = 8;

impl NativeTextPrefixCacheValue for QwenLayerCache {
    type PrefixCacheState = QwenLayerCachePrefixState;

    fn prefix_cache_state(caches: &[Self]) -> Vec<Self::PrefixCacheState> {
        caches
            .iter()
            .map(QwenLayerCache::prefix_cache_state)
            .collect()
    }

    fn prefix_cache_from_state(states: &[Self::PrefixCacheState]) -> Option<Vec<Self>> {
        states
            .iter()
            .map(QwenLayerCache::from_prefix_cache_state)
            .collect::<Result<Vec<_>, _>>()
            .ok()
    }

    fn prefix_cache_entry_bytes(hidden: &[f32], states: &[Self::PrefixCacheState]) -> u64 {
        let hidden_bytes = std::mem::size_of_val(hidden) as u64;
        states.iter().fold(hidden_bytes, |total, state| {
            total.saturating_add(match state {
                QwenLayerCachePrefixState::Full(state) => state.metadata_bytes(),
                QwenLayerCachePrefixState::Linear(state) => {
                    ((state.conv_window.len() + state.recurrent_state.len())
                        * std::mem::size_of::<f32>()) as u64
                }
            })
        })
    }
}

struct CloneFixture {
    cache: NativeTextPrefixCache<QwenLayerCache>,
    namespace: NativeTextPrefixCacheNamespace,
    metrics: NativeTextPrefixCacheMetrics,
    query: Vec<usize>,
    payload_bytes: u64,
}

fn main() {
    println!("prefix_cache_clone: Qwen-like hit clone benches; no snapshots or inference");
    println!(
        "{:<44} {:>12} {:>10} {:>14} {:>12}",
        "case", "payload_mb", "iters", "total_ms", "ns/iter"
    );

    run_shape_cases("qwen36_full_layers", qwen36_full_layer_caches());
    run_shape_cases("qwen36_linear_layers", qwen36_linear_layer_caches());
}

fn run_shape_cases(label: &str, caches: Vec<QwenLayerCache>) {
    let cold_miss = CloneFixture::new(label, caches.clone(), false);
    run_clone_case(
        &format!("{label}/cold_miss"),
        &cold_miss,
        COLD_MISS_ITERATIONS,
    );

    let hit = CloneFixture::new(label, caches, true);
    run_clone_case(
        &format!("{label}/longest_prefix_hit"),
        &hit,
        HIT_CLONE_ITERATIONS,
    );
}

impl CloneFixture {
    fn new(label: &str, caches: Vec<QwenLayerCache>, hit: bool) -> Self {
        let cache = NativeTextPrefixCache::new(u64::MAX);
        let namespace = namespace(label);
        let metrics = NativeTextPrefixCacheMetrics::default();
        let hidden = vec![0.0; HIDDEN_SIZE];
        let query = query_tokens();
        let entry_tokens = if hit {
            query[..PREFIX_TOKENS].to_vec()
        } else {
            vec![usize::MAX - 1]
        };
        let states = <QwenLayerCache as NativeTextPrefixCacheValue>::prefix_cache_state(&caches);
        let payload_bytes =
            <QwenLayerCache as NativeTextPrefixCacheValue>::prefix_cache_entry_bytes(
                &hidden, &states,
            );

        cache.store(namespace.clone(), &entry_tokens, &hidden, &caches, &metrics);

        Self {
            cache,
            namespace,
            metrics,
            query,
            payload_bytes,
        }
    }
}

fn run_clone_case(label: &str, fixture: &CloneFixture, iterations: usize) {
    let mut checksum = 0_usize;
    for _ in 0..2 {
        checksum ^= clone_lookup_once(fixture);
    }

    let started = Instant::now();
    for _ in 0..iterations {
        checksum = checksum.wrapping_add(clone_lookup_once(fixture));
    }
    let elapsed = started.elapsed();

    black_box(checksum);
    print_result(label, fixture.payload_bytes, iterations, elapsed);
}

fn clone_lookup_once(fixture: &CloneFixture) -> usize {
    match fixture.cache.lookup_compatible(
        &fixture.namespace,
        &fixture.query,
        &fixture.metrics,
        |_| true,
    ) {
        Some(hit) => {
            let checksum = hit
                .token_count
                .wrapping_add(hit.hidden.len())
                .wrapping_add(hit.caches.len());
            black_box(hit);
            checksum
        }
        None => 0,
    }
}

fn print_result(label: &str, payload_bytes: u64, iterations: usize, elapsed: Duration) {
    let payload_mb = payload_bytes as f64 / (1024.0 * 1024.0);
    let total_ms = elapsed.as_secs_f64() * 1_000.0;
    let ns_per_iter = elapsed.as_secs_f64() * 1_000_000_000.0 / iterations as f64;
    println!(
        "{:<44} {:>12.1} {:>10} {:>14.3} {:>12.1}",
        format!("prefix_cache_clone/{label}"),
        payload_mb,
        iterations,
        total_ms,
        ns_per_iter
    );
}

fn qwen36_full_layer_caches() -> Vec<QwenLayerCache> {
    (0..FULL_LAYER_COUNT)
        .map(|_| {
            LayerKvCache::new(FULL_CACHE_TOKENS, FULL_KV_HEADS, FULL_HEAD_DIM)
                .map(QwenLayerCache::Full)
                .expect("Qwen-like full cache shape is valid")
        })
        .collect()
}

fn qwen36_linear_layer_caches() -> Vec<QwenLayerCache> {
    let conv_dim =
        LINEAR_KEY_HEADS * LINEAR_KEY_HEAD_DIM * 2 + LINEAR_VALUE_HEADS * LINEAR_VALUE_HEAD_DIM;
    (0..LINEAR_LAYER_COUNT)
        .map(|_| {
            LinearAttentionCache::new(
                LINEAR_CONV_KERNEL,
                conv_dim,
                LINEAR_VALUE_HEADS,
                LINEAR_KEY_HEAD_DIM,
                LINEAR_VALUE_HEAD_DIM,
            )
            .map(QwenLayerCache::Linear)
            .expect("Qwen-like linear cache shape is valid")
        })
        .collect()
}

fn query_tokens() -> Vec<usize> {
    (0..=PREFIX_TOKENS).collect()
}

fn namespace(label: &str) -> NativeTextPrefixCacheNamespace {
    NativeTextPrefixCacheNamespace {
        model_id: format!("bench-{label}"),
        backend: "native-qwen".to_owned(),
        family: Some("qwen".to_owned()),
        quantization: Some("bf16".to_owned()),
        repo_id: Some("Qwen/Qwen3.6-35B-A3B".to_owned()),
        resolved_commit: Some("bench".to_owned()),
        profile: Some("prefix-cache-bench".to_owned()),
        cache_key: label.to_owned(),
        tool_schema: None,
        request_mode: "raw_completion".to_owned(),
        cache_layout_version: 1,
        cache_tokens: FULL_CACHE_TOKENS,
        max_prefill_tokens: FULL_CACHE_TOKENS,
    }
}
