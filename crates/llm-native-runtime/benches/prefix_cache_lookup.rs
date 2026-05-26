//! Prefix-cache lookup index microbenchmarks.
//!
//! These benches include the current prefix-cache implementation directly so
//! they measure lookup behavior without opening model snapshots or running
//! inference. Payloads are intentionally empty: the cold-miss and
//! longest-prefix hit cases isolate trie traversal and candidate lookup cost
//! from cache payload clone cost.

use std::{
    hint::black_box,
    time::{Duration, Instant},
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

const ENTRY_COUNTS: [usize; 4] = [10, 100, 1_000, 10_000];
const MAX_MATCHING_PREFIX_TOKENS: usize = 256;

#[derive(Debug, Clone)]
struct LookupPayload;

impl NativeTextPrefixCacheValue for LookupPayload {
    type PrefixCacheState = Self;

    fn prefix_cache_state(caches: &[Self]) -> Vec<Self::PrefixCacheState> {
        caches.to_vec()
    }

    fn prefix_cache_from_state(states: &[Self::PrefixCacheState]) -> Option<Vec<Self>> {
        Some(states.to_vec())
    }

    fn prefix_cache_entry_bytes(hidden: &[f32], states: &[Self::PrefixCacheState]) -> u64 {
        std::mem::size_of_val(hidden) as u64 + states.len() as u64
    }
}

struct LookupFixture {
    cache: NativeTextPrefixCache<LookupPayload>,
    namespace: NativeTextPrefixCacheNamespace,
    metrics: NativeTextPrefixCacheMetrics,
    query: Vec<usize>,
}

fn main() {
    println!("prefix_cache_lookup: indexed benches; no snapshots or inference");
    println!(
        "{:<44} {:>8} {:>10} {:>14} {:>12} {:>12}",
        "case", "entries", "iters", "total_ms", "ns/iter", "scans/iter"
    );

    for entries in ENTRY_COUNTS {
        let miss = LookupFixture::cold_miss(entries);
        run_lookup_case("cold_miss", entries, &miss, lookup_iterations(entries));

        let hit = LookupFixture::longest_prefix_hit(entries);
        run_lookup_case(
            "longest_prefix_hit",
            entries,
            &hit,
            lookup_iterations(entries),
        );
    }
}

impl LookupFixture {
    fn cold_miss(entry_count: usize) -> Self {
        let cache = NativeTextPrefixCache::new(u64::MAX);
        let namespace = namespace("lookup-cold-miss");
        let metrics = NativeTextPrefixCacheMetrics::default();
        let hidden = [];
        let payload = [];

        for index in 0..entry_count {
            let tokens = vec![1_000_000 + index];
            cache.store(namespace.clone(), &tokens, &hidden, &payload, &metrics);
        }

        Self {
            cache,
            namespace,
            metrics,
            query: query_tokens(),
        }
    }

    fn longest_prefix_hit(entry_count: usize) -> Self {
        let cache = NativeTextPrefixCache::new(u64::MAX);
        let namespace = namespace("lookup-longest-prefix-hit");
        let metrics = NativeTextPrefixCacheMetrics::default();
        let hidden = [];
        let payload = [];
        let query = query_tokens();
        let matching_prefixes = entry_count.min(MAX_MATCHING_PREFIX_TOKENS);

        for prefix_len in 1..=matching_prefixes {
            cache.store(
                namespace.clone(),
                &query[..prefix_len],
                &hidden,
                &payload,
                &metrics,
            );
        }

        for index in matching_prefixes..entry_count {
            let tokens = vec![1_000_000 + index];
            cache.store(namespace.clone(), &tokens, &hidden, &payload, &metrics);
        }

        Self {
            cache,
            namespace,
            metrics,
            query,
        }
    }
}

fn run_lookup_case(label: &str, entries: usize, fixture: &LookupFixture, iterations: usize) {
    let mut checksum = 0_usize;
    for _ in 0..32 {
        checksum ^= lookup_once(fixture);
    }

    let scans_before = lookup_scan_count(&fixture.metrics);
    let started = Instant::now();
    for _ in 0..iterations {
        checksum = checksum.wrapping_add(lookup_once(fixture));
    }
    let elapsed = started.elapsed();
    let scans_after = lookup_scan_count(&fixture.metrics);
    let scans_per_iter = scans_after.saturating_sub(scans_before) as f64 / iterations as f64;

    black_box(checksum);
    print_result(label, entries, iterations, elapsed, scans_per_iter);
}

fn lookup_once(fixture: &LookupFixture) -> usize {
    fixture
        .cache
        .lookup_compatible(&fixture.namespace, &fixture.query, &fixture.metrics, |_| {
            true
        })
        .map_or(0, |hit| hit.token_count)
}

fn lookup_iterations(entries: usize) -> usize {
    match entries {
        0..=10 => 50_000,
        11..=100 => 25_000,
        101..=1_000 => 5_000,
        _ => 1_000,
    }
}

fn lookup_scan_count(metrics: &NativeTextPrefixCacheMetrics) -> u64 {
    metrics.snapshot()["entries_scanned"].as_u64().unwrap_or(0)
}

fn print_result(
    label: &str,
    entries: usize,
    iterations: usize,
    elapsed: Duration,
    scans_per_iter: f64,
) {
    let total_ms = elapsed.as_secs_f64() * 1_000.0;
    let ns_per_iter = elapsed.as_secs_f64() * 1_000_000_000.0 / iterations as f64;
    println!(
        "{:<44} {:>8} {:>10} {:>14.3} {:>12.1} {:>12.1}",
        format!("prefix_cache_lookup/{label}"),
        entries,
        iterations,
        total_ms,
        ns_per_iter,
        scans_per_iter
    );
}

fn query_tokens() -> Vec<usize> {
    (0..=MAX_MATCHING_PREFIX_TOKENS).collect()
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
        tokenizer_kind: "huggingface-tokenizer-json".to_owned(),
        tokenizer_hash: "sha256:bench-tokenizer".to_owned(),
        tokenizer_normalization: "llm-tokenizer/hf-json/v1".to_owned(),
        cache_template_id: "raw-prompt/v1".to_owned(),
        chat_template_kwargs_hash: None,
        adapter_settings: "native-qwen-prefix-adapter/v1".to_owned(),
        cache_key: label.to_owned(),
        tool_schema: None,
        request_mode: "raw_completion".to_owned(),
        cache_layout_version: 1,
        cache_tokens: 2_048,
        max_prefill_tokens: 2_048,
    }
}
