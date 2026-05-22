use crate::sync_ext::FailPoisonedMutex;
use llm_backend::{BackendModelMetadata, BackendRequest};
use std::{
    collections::HashMap,
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
    pub(crate) quantization: Option<String>,
    pub(crate) repo_id: Option<String>,
    pub(crate) resolved_commit: Option<String>,
    pub(crate) profile: Option<String>,
    pub(crate) cache_key: String,
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
        quantization: context.metadata.quantization.clone(),
        repo_id: context.metadata.repo_id.clone(),
        resolved_commit: context.metadata.resolved_commit.clone(),
        profile: context.metadata.profile.clone(),
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
            let NativeTextPrefixCacheInner {
                entries,
                next_access,
                ..
            } = &mut *inner;
            let mut best_len = 0;
            let mut best_entry = None;
            let mut entries_scanned = 0;
            let mut namespace_entries_scanned = 0;
            if let Some(bucket) = entries.get_mut(namespace) {
                namespace_entries_scanned = bucket.len() as u64;
                for (entry_tokens, entry) in bucket.iter_mut() {
                    entries_scanned += 1;
                    if entry_tokens.len() > best_len
                        && tokens.starts_with(entry_tokens)
                        && is_compatible(&entry.payload.states)
                    {
                        best_len = entry_tokens.len();
                        best_entry = Some(entry);
                    }
                }
            }
            let hit = if let Some(entry) = best_entry {
                let access = *next_access;
                *next_access = next_access.saturating_add(1);
                entry.last_used = access;
                Some((best_len, entry.byte_len, Arc::clone(&entry.payload)))
            } else {
                None
            };
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
    ) {
        if tokens.is_empty() {
            return;
        }
        let states = C::prefix_cache_state(caches);
        let byte_len = C::prefix_cache_entry_bytes(hidden, &states);
        if byte_len > self.max_bytes {
            metrics.record_rejected();
            return;
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
            inner.entries.entry(namespace).or_default().insert(
                entry_tokens,
                NativeTextPrefixCacheEntry {
                    payload,
                    byte_len,
                    last_used: access,
                },
            );
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
        self.entries
            .values()
            .map(|bucket| bucket.len() as u64)
            .sum()
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
        if remove_bucket {
            self.entries.remove(namespace);
        }
        entry
    }

    fn remove_lru_entry(&mut self) -> Option<NativeTextPrefixCacheEntry<C>> {
        let (namespace, tokens) = self
            .entries
            .iter()
            .flat_map(|(namespace, bucket)| {
                bucket
                    .iter()
                    .map(move |(tokens, entry)| (namespace, tokens, entry))
            })
            .min_by_key(|(_, _, entry)| entry.last_used)
            .map(|(namespace, tokens, _)| (namespace.clone(), tokens.clone()))?;
        self.remove_entry(&namespace, &tokens)
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

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot["reused_tokens"], u64::MAX);
        assert_eq!(snapshot["hit_tokens"], u64::MAX);
        assert_eq!(snapshot["avoided_prefill_tokens"], u64::MAX);
        assert_eq!(snapshot["miss_tokens"], u64::MAX);
        assert_eq!(snapshot["prefill_tokens"], u64::MAX);
    }
}
