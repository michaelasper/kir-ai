use crate::sync_ext::FailPoisonedMutex;
use llm_backend_contracts::{BackendModelMetadata, BackendRequest};
use llm_tokenizer::HuggingFaceTokenizerIdentity;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeSet, HashMap, hash_map::Entry},
    sync::{Arc, Mutex},
};

pub(crate) trait NativeTextPrefixCacheValue: Sized {
    type PrefixCacheState: Clone + std::fmt::Debug;

    fn prefix_cache_state(caches: &[Self]) -> Vec<Self::PrefixCacheState>;

    fn prefix_cache_from_state(states: &[Self::PrefixCacheState]) -> Option<Vec<Self>>;

    fn prefix_cache_entry_bytes(hidden: &[f32], states: &[Self::PrefixCacheState]) -> u64;
}

#[derive(Debug)]
pub(crate) struct NativeTextPrefixCache<C: NativeTextPrefixCacheValue> {
    pub(crate) max_bytes: u64,
    pub(crate) inner: Mutex<NativeTextPrefixCacheInner<C>>,
}

#[derive(Debug)]
pub(crate) struct NativeTextPrefixCacheInner<C: NativeTextPrefixCacheValue> {
    pub(crate) entries:
        HashMap<NativeTextPrefixCacheNamespace, HashMap<Vec<usize>, NativeTextPrefixCacheEntry<C>>>,
    pub(crate) indexes: HashMap<NativeTextPrefixCacheNamespace, NativeTextPrefixCacheLengthIndex>,
    pub(crate) lru: NativeTextPrefixCacheLruIndex,
    pub(crate) used_bytes: u64,
    pub(crate) next_access: u64,
}

impl<C> Default for NativeTextPrefixCacheInner<C>
where
    C: NativeTextPrefixCacheValue,
{
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            indexes: HashMap::new(),
            lru: NativeTextPrefixCacheLruIndex::default(),
            used_bytes: 0,
            next_access: 0,
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct NativeTextPrefixCacheLengthIndex {
    lengths: Vec<usize>,
    length_counts: HashMap<usize, usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct NativeTextPrefixCacheNamespace {
    pub(crate) model_id: String,
    pub(crate) backend: String,
    pub(crate) family: Option<String>,
    pub(crate) quantization: Option<String>,
    pub(crate) repo_id: Option<String>,
    pub(crate) resolved_commit: Option<String>,
    pub(crate) profile: Option<String>,
    pub(crate) tokenizer_kind: String,
    pub(crate) tokenizer_hash: String,
    pub(crate) tokenizer_normalization: String,
    pub(crate) cache_template_id: String,
    pub(crate) chat_template_kwargs_hash: Option<String>,
    pub(crate) adapter_settings: String,
    pub(crate) cache_key: String,
    pub(crate) tool_schema: Option<String>,
    pub(crate) request_mode: String,
    pub(crate) cache_layout_version: u32,
    pub(crate) cache_tokens: usize,
    pub(crate) max_prefill_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct NativeTextPrefixCacheLruKey {
    access: u64,
    namespace: NativeTextPrefixCacheNamespace,
    tokens: Vec<usize>,
}

#[derive(Debug, Default)]
pub(crate) struct NativeTextPrefixCacheLruIndex {
    entries: BTreeSet<NativeTextPrefixCacheLruKey>,
}

pub(crate) struct NativeTextPrefixNamespaceContext<'a> {
    pub(crate) model_id: &'a str,
    pub(crate) metadata: &'a BackendModelMetadata,
    pub(crate) tokenizer_identity: &'a HuggingFaceTokenizerIdentity,
    pub(crate) adapter_settings: &'a str,
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
        quantization: context.metadata.quantization.clone(),
        repo_id: context.metadata.repo_id.clone(),
        resolved_commit: context.metadata.resolved_commit.clone(),
        profile: context.metadata.profile.clone(),
        tokenizer_kind: context.tokenizer_identity.kind.clone(),
        tokenizer_hash: context.tokenizer_identity.content_hash.clone(),
        tokenizer_normalization: context.tokenizer_identity.normalization.clone(),
        cache_template_id: context.request.cache_context().cache_template_id.clone(),
        chat_template_kwargs_hash: context
            .request
            .cache_context()
            .chat_template_kwargs
            .as_deref()
            .map(hash_str),
        adapter_settings: context.adapter_settings.to_owned(),
        cache_key: context.request.cache_context().key.as_str().to_owned(),
        tool_schema: context.request.cache_context().tool_schema.clone(),
        request_mode: native_text_prefix_request_mode(context.request),
        cache_layout_version: context.cache_layout_version,
        cache_tokens: context.cache_tokens,
        max_prefill_tokens: context.max_prefill_tokens,
    }
}

pub(crate) fn native_text_prefix_request_mode(request: &BackendRequest) -> String {
    match request.as_chat() {
        Some(chat) => format!(
            "chat,json_object={},required_tool={:?}",
            chat.json_object_mode, chat.required_tool_choice
        ),
        None => "raw_completion".to_owned(),
    }
}

fn hash_str(value: &str) -> String {
    let digest = Sha256::digest(value.as_bytes());
    format!("sha256:{digest:x}")
}

#[derive(Debug, Clone)]
pub(crate) struct NativeTextPrefixCacheEntry<C: NativeTextPrefixCacheValue> {
    pub(crate) payload: Arc<NativeTextPrefixCacheEntryPayload<C>>,
    pub(crate) byte_len: u64,
    pub(crate) last_used: u64,
}

#[derive(Debug)]
pub(crate) struct NativeTextPrefixCacheEntryPayload<C: NativeTextPrefixCacheValue> {
    pub(crate) hidden: Vec<f32>,
    pub(crate) states: Vec<C::PrefixCacheState>,
}

#[derive(Debug)]
pub(crate) struct NativeTextPrefixCacheHit<C: NativeTextPrefixCacheValue> {
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
    pub(crate) prefill_chunks: u64,
    pub(crate) prefill_tokens: u64,
    pub(crate) checkpoint_stores: u64,
    pub(crate) checkpoint_store_tokens: u64,
    pub(crate) checkpoint_reuse_hits: u64,
    pub(crate) checkpoint_reused_tokens: u64,
    pub(crate) shared_prefix_hits: u64,
    pub(crate) shared_prefix_reused_tokens: u64,
    pub(crate) hit_tokens: u64,
    pub(crate) miss_tokens: u64,
    pub(crate) avoided_prefill_tokens: u64,
    pub(crate) entries_scanned: u64,
    pub(crate) namespace_entries_scanned: u64,
    pub(crate) hit_clone_bytes: u64,
    pub(crate) bytes_stored: u64,
    pub(crate) bytes_evicted: u64,
    pub(crate) resident_bytes: u64,
    pub(crate) resident_entries: u64,
}

#[derive(Debug, Default)]
pub(crate) struct NativeTextPrefixCacheMetrics {
    counters: Mutex<NativeTextPrefixCacheCounters>,
}

impl<C> NativeTextPrefixCache<C>
where
    C: NativeTextPrefixCacheValue,
{
    pub(crate) fn new(max_bytes: u64) -> Self {
        Self {
            max_bytes,
            inner: Mutex::new(NativeTextPrefixCacheInner::default()),
        }
    }

    #[cfg(test)]
    pub(crate) fn lookup(
        &self,
        namespace: &NativeTextPrefixCacheNamespace,
        tokens: &[usize],
        metrics: &NativeTextPrefixCacheMetrics,
    ) -> Option<NativeTextPrefixCacheHit<C>> {
        self.lookup_compatible(namespace, tokens, metrics, |_| true)
    }

    pub(crate) fn lookup_compatible(
        &self,
        namespace: &NativeTextPrefixCacheNamespace,
        tokens: &[usize],
        metrics: &NativeTextPrefixCacheMetrics,
        mut is_compatible: impl FnMut(&[C::PrefixCacheState]) -> bool,
    ) -> Option<NativeTextPrefixCacheHit<C>> {
        let (hit, entries_scanned, namespace_entries_scanned) = {
            let mut inner = self.inner.lock_or_panic("native text prefix cache");
            let candidates = inner
                .indexes
                .get(namespace)
                .map(|index| index.prefix_candidate_lengths(tokens.len()))
                .unwrap_or_default();
            let mut entries_scanned = 0;
            let mut namespace_entries_scanned = 0;
            let mut hit = None;
            let NativeTextPrefixCacheInner {
                entries,
                lru,
                next_access,
                ..
            } = &mut *inner;
            if let Some(bucket) = entries.get_mut(namespace) {
                for token_count in candidates {
                    let entry_tokens = &tokens[..token_count];
                    let Some(entry) = bucket.get_mut(entry_tokens) else {
                        continue;
                    };
                    entries_scanned += 1;
                    namespace_entries_scanned += 1;
                    if is_compatible(&entry.payload.states) {
                        let access = *next_access;
                        *next_access = next_access.saturating_add(1);
                        let previous_access = entry.last_used;
                        entry.last_used = access;
                        lru.promote(namespace, entry_tokens, previous_access, access);
                        hit = Some((token_count, entry.byte_len, Arc::clone(&entry.payload)));
                        break;
                    }
                }
            }
            (hit, entries_scanned, namespace_entries_scanned)
        };
        metrics.record_lookup_scan(entries_scanned, namespace_entries_scanned);
        let Some((best_len, clone_bytes, payload)) = hit else {
            metrics.record_miss();
            metrics.record_miss_tokens(tokens.len() as u64);
            return None;
        };
        let caches = match C::prefix_cache_from_state(&payload.states) {
            Some(caches) => caches,
            None => {
                metrics.record_miss();
                metrics.record_miss_tokens(tokens.len() as u64);
                return None;
            }
        };
        let hit_tokens = best_len as u64;
        metrics.record_hit(hit_tokens);
        metrics.record_miss_tokens((tokens.len() as u64).saturating_sub(hit_tokens));
        metrics.record_hit_clone_bytes(clone_bytes);
        Some(NativeTextPrefixCacheHit {
            token_count: best_len,
            hidden: payload.hidden.clone(),
            caches,
        })
    }

    pub(crate) fn store(
        &self,
        namespace: NativeTextPrefixCacheNamespace,
        tokens: &[usize],
        hidden: &[f32],
        caches: &[C],
        metrics: &NativeTextPrefixCacheMetrics,
    ) -> bool {
        if tokens.is_empty() {
            return false;
        }
        let states = C::prefix_cache_state(caches);
        let byte_len = C::prefix_cache_entry_bytes(hidden, &states);
        if byte_len > self.max_bytes {
            metrics.record_rejected();
            return false;
        }
        let payload = Arc::new(NativeTextPrefixCacheEntryPayload {
            hidden: hidden.to_vec(),
            states,
        });
        let entry_tokens = tokens.to_vec();
        let mut removed_entries = Vec::new();
        let mut evicted_byte_lens = Vec::new();
        let (resident_bytes, resident_entries) = {
            let mut inner = self.inner.lock_or_panic("native text prefix cache");
            if let Some(existing) = inner.remove_entry(&namespace, &entry_tokens) {
                inner.used_bytes = inner.used_bytes.saturating_sub(existing.byte_len);
                removed_entries.push(existing);
            }
            while inner.used_bytes.saturating_add(byte_len) > self.max_bytes {
                let Some(evicted) = inner.remove_lru_entry() else {
                    break;
                };
                inner.used_bytes = inner.used_bytes.saturating_sub(evicted.byte_len);
                evicted_byte_lens.push(evicted.byte_len);
                removed_entries.push(evicted);
            }
            let access = inner.next_access();
            inner.entries.entry(namespace.clone()).or_default().insert(
                entry_tokens.clone(),
                NativeTextPrefixCacheEntry {
                    payload,
                    byte_len,
                    last_used: access,
                },
            );
            inner.lru.insert(&namespace, &entry_tokens, access);
            inner
                .indexes
                .entry(namespace)
                .or_default()
                .insert(&entry_tokens);
            inner.used_bytes = inner.used_bytes.saturating_add(byte_len);
            (inner.used_bytes, inner.entry_count())
        };
        // Drop replaced or evicted payloads after releasing the global cache lock.
        drop(removed_entries);
        for byte_len in evicted_byte_lens {
            metrics.record_eviction(byte_len);
        }
        metrics.record_store(byte_len);
        metrics.record_residency(resident_bytes, resident_entries);
        true
    }
}

impl<C> NativeTextPrefixCacheInner<C>
where
    C: NativeTextPrefixCacheValue,
{
    fn next_access(&mut self) -> u64 {
        let access = self.next_access;
        self.next_access = self.next_access.saturating_add(1);
        access
    }

    fn entry_count(&self) -> u64 {
        self.lru.len() as u64
    }

    fn remove_entry(
        &mut self,
        namespace: &NativeTextPrefixCacheNamespace,
        tokens: &[usize],
    ) -> Option<NativeTextPrefixCacheEntry<C>> {
        let (entry, remove_bucket) = {
            let bucket = self.entries.get_mut(namespace)?;
            let entry = bucket.remove(tokens);
            (entry, bucket.is_empty())
        };
        if entry.is_some() {
            if let Some(entry) = &entry {
                self.lru.remove(namespace, tokens, entry.last_used);
            }
            let remove_index = self.indexes.get_mut(namespace).is_some_and(|index| {
                index.remove(tokens);
                index.is_empty()
            });
            if remove_index {
                self.indexes.remove(namespace);
            }
        }
        if remove_bucket {
            self.entries.remove(namespace);
        }
        entry
    }

    fn remove_lru_entry(&mut self) -> Option<NativeTextPrefixCacheEntry<C>> {
        let oldest = self.lru.oldest()?.clone();
        self.remove_entry(&oldest.namespace, &oldest.tokens)
    }

    #[cfg(test)]
    pub(crate) fn assert_lru_index_consistent(&self) {
        assert_eq!(
            self.lru.len() as u64,
            self.entries
                .values()
                .map(|bucket| bucket.len() as u64)
                .sum::<u64>(),
            "LRU index should contain one key per resident prefix cache entry"
        );
        for (namespace, bucket) in &self.entries {
            for (tokens, entry) in bucket {
                assert!(
                    self.lru.contains(namespace, tokens, entry.last_used),
                    "LRU index missing key for namespace {namespace:?} tokens {tokens:?}"
                );
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn lru_order(&self) -> Vec<(NativeTextPrefixCacheNamespace, Vec<usize>)> {
        self.lru
            .iter()
            .map(|entry| (entry.namespace.clone(), entry.tokens.clone()))
            .collect()
    }
}

impl NativeTextPrefixCacheLruIndex {
    fn insert(
        &mut self,
        namespace: &NativeTextPrefixCacheNamespace,
        tokens: &[usize],
        access: u64,
    ) {
        self.entries.insert(NativeTextPrefixCacheLruKey {
            access,
            namespace: namespace.clone(),
            tokens: tokens.to_vec(),
        });
    }

    fn remove(
        &mut self,
        namespace: &NativeTextPrefixCacheNamespace,
        tokens: &[usize],
        access: u64,
    ) {
        self.entries.remove(&NativeTextPrefixCacheLruKey {
            access,
            namespace: namespace.clone(),
            tokens: tokens.to_vec(),
        });
    }

    fn promote(
        &mut self,
        namespace: &NativeTextPrefixCacheNamespace,
        tokens: &[usize],
        previous_access: u64,
        next_access: u64,
    ) {
        self.remove(namespace, tokens, previous_access);
        self.insert(namespace, tokens, next_access);
    }

    fn oldest(&self) -> Option<&NativeTextPrefixCacheLruKey> {
        self.entries.iter().next()
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    #[cfg(test)]
    fn contains(
        &self,
        namespace: &NativeTextPrefixCacheNamespace,
        tokens: &[usize],
        access: u64,
    ) -> bool {
        self.entries.contains(&NativeTextPrefixCacheLruKey {
            access,
            namespace: namespace.clone(),
            tokens: tokens.to_vec(),
        })
    }

    #[cfg(test)]
    fn iter(&self) -> impl Iterator<Item = &NativeTextPrefixCacheLruKey> {
        self.entries.iter()
    }
}

impl NativeTextPrefixCacheLengthIndex {
    fn insert(&mut self, tokens: &[usize]) {
        let token_count = tokens.len();
        let count = self.length_counts.entry(token_count).or_insert(0);
        if *count == 0 {
            let index = self.lengths.partition_point(|length| *length < token_count);
            self.lengths.insert(index, token_count);
        }
        *count += 1;
    }

    fn remove(&mut self, tokens: &[usize]) {
        let token_count = tokens.len();
        let Entry::Occupied(mut count) = self.length_counts.entry(token_count) else {
            return;
        };
        if *count.get() > 1 {
            *count.get_mut() -= 1;
            return;
        }
        count.remove();
        if let Ok(index) = self.lengths.binary_search(&token_count) {
            self.lengths.remove(index);
        }
    }

    fn prefix_candidate_lengths(&self, max_token_count: usize) -> Vec<usize> {
        let upper_bound = self
            .lengths
            .partition_point(|token_count| *token_count <= max_token_count);
        self.lengths[..upper_bound].iter().rev().copied().collect()
    }

    fn is_empty(&self) -> bool {
        self.length_counts.is_empty()
    }
}

impl NativeTextPrefixCacheMetrics {
    pub(crate) fn record_hit(&self, tokens: u64) {
        self.update(|counters| {
            counters.hits += 1;
            counters.reused_tokens = counters.reused_tokens.saturating_add(tokens);
            counters.hit_tokens = counters.hit_tokens.saturating_add(tokens);
            counters.avoided_prefill_tokens =
                counters.avoided_prefill_tokens.saturating_add(tokens);
        });
    }

    pub(crate) fn record_miss(&self) {
        self.update(|counters| counters.misses += 1);
    }

    pub(crate) fn record_miss_tokens(&self, tokens: u64) {
        self.update(|counters| {
            counters.miss_tokens = counters.miss_tokens.saturating_add(tokens);
        });
    }

    pub(crate) fn record_prefill_chunk(&self, tokens: u64) {
        self.update(|counters| {
            counters.prefill_chunks += 1;
            counters.prefill_tokens = counters.prefill_tokens.saturating_add(tokens);
        });
    }

    pub(crate) fn record_checkpoint_store(&self, tokens: u64) {
        self.update(|counters| {
            counters.checkpoint_stores += 1;
            counters.checkpoint_store_tokens =
                counters.checkpoint_store_tokens.saturating_add(tokens);
        });
    }

    pub(crate) fn record_checkpoint_reuse(&self, tokens: u64) {
        self.update(|counters| {
            counters.checkpoint_reuse_hits += 1;
            counters.checkpoint_reused_tokens =
                counters.checkpoint_reused_tokens.saturating_add(tokens);
        });
    }

    pub(crate) fn record_shared_prefix_reuse(&self, tokens: u64) {
        self.update(|counters| {
            counters.shared_prefix_hits += 1;
            counters.shared_prefix_reused_tokens =
                counters.shared_prefix_reused_tokens.saturating_add(tokens);
        });
    }

    pub(crate) fn record_lookup_scan(&self, entries_scanned: u64, namespace_entries_scanned: u64) {
        self.update(|counters| {
            counters.entries_scanned = counters.entries_scanned.saturating_add(entries_scanned);
            counters.namespace_entries_scanned = counters
                .namespace_entries_scanned
                .saturating_add(namespace_entries_scanned);
        });
    }

    pub(crate) fn record_hit_clone_bytes(&self, byte_len: u64) {
        self.update(|counters| {
            counters.hit_clone_bytes = counters.hit_clone_bytes.saturating_add(byte_len);
        });
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
            .lock_or_panic("native text prefix cache metrics");
        serde_json::json!({
            "hits": counters.hits,
            "misses": counters.misses,
            "stores": counters.stores,
            "evictions": counters.evictions,
            "rejected": counters.rejected,
            "reused_tokens": counters.reused_tokens,
            "prefill_chunks": counters.prefill_chunks,
            "prefill_tokens": counters.prefill_tokens,
            "checkpoint_stores": counters.checkpoint_stores,
            "checkpoint_store_tokens": counters.checkpoint_store_tokens,
            "checkpoint_reuse_hits": counters.checkpoint_reuse_hits,
            "checkpoint_reused_tokens": counters.checkpoint_reused_tokens,
            "shared_prefix_hits": counters.shared_prefix_hits,
            "shared_prefix_reused_tokens": counters.shared_prefix_reused_tokens,
            "hit_tokens": counters.hit_tokens,
            "miss_tokens": counters.miss_tokens,
            "avoided_prefill_tokens": counters.avoided_prefill_tokens,
            "entries_scanned": counters.entries_scanned,
            "namespace_entries_scanned": counters.namespace_entries_scanned,
            "hit_clone_bytes": counters.hit_clone_bytes,
            "bytes_stored": counters.bytes_stored,
            "bytes_evicted": counters.bytes_evicted,
            "resident_bytes": counters.resident_bytes,
            "resident_entries": counters.resident_entries,
        })
    }

    fn update(&self, update: impl FnOnce(&mut NativeTextPrefixCacheCounters)) {
        let mut counters = self
            .counters
            .lock_or_panic("native text prefix cache metrics");
        update(&mut counters);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn native_text_prefix_cache_metrics_saturate_token_counters() {
        let metrics = super::NativeTextPrefixCacheMetrics::default();

        metrics.record_hit(u64::MAX - 1);
        metrics.record_hit(2);
        metrics.record_miss_tokens(u64::MAX - 2);
        metrics.record_miss_tokens(3);
        metrics.record_prefill_chunk(u64::MAX - 3);
        metrics.record_prefill_chunk(4);
        metrics.record_checkpoint_store(u64::MAX - 4);
        metrics.record_checkpoint_store(5);
        metrics.record_checkpoint_reuse(u64::MAX - 5);
        metrics.record_checkpoint_reuse(6);
        metrics.record_shared_prefix_reuse(u64::MAX - 6);
        metrics.record_shared_prefix_reuse(7);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot["reused_tokens"], u64::MAX);
        assert_eq!(snapshot["hit_tokens"], u64::MAX);
        assert_eq!(snapshot["avoided_prefill_tokens"], u64::MAX);
        assert_eq!(snapshot["miss_tokens"], u64::MAX);
        assert_eq!(snapshot["prefill_tokens"], u64::MAX);
        assert_eq!(snapshot["checkpoint_stores"], 2);
        assert_eq!(snapshot["checkpoint_store_tokens"], u64::MAX);
        assert_eq!(snapshot["checkpoint_reuse_hits"], 2);
        assert_eq!(snapshot["checkpoint_reused_tokens"], u64::MAX);
        assert_eq!(snapshot["shared_prefix_hits"], 2);
        assert_eq!(snapshot["shared_prefix_reused_tokens"], u64::MAX);
    }
}
