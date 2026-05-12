use crate::sync_ext::FailPoisonedMutex;
use llm_backend::{BackendCacheContext, BackendModelMetadata, BackendRequest};
use std::{collections::HashMap, sync::Mutex};

pub(crate) trait NativeTextPrefixCacheValue: Clone {
    fn prefix_cache_entry_bytes(hidden: &[f32], caches: &[Self]) -> u64;
}

#[derive(Debug)]
pub(crate) struct NativeTextPrefixCache<C> {
    pub(crate) max_bytes: u64,
    pub(crate) inner: Mutex<NativeTextPrefixCacheInner<C>>,
}

#[derive(Debug)]
pub(crate) struct NativeTextPrefixCacheInner<C> {
    pub(crate) entries: HashMap<NativeTextPrefixCacheKey, NativeTextPrefixCacheEntry<C>>,
    pub(crate) used_bytes: u64,
    pub(crate) next_access: u64,
}

impl<C> Default for NativeTextPrefixCacheInner<C> {
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
        BackendCacheContext::raw_prompt().prompt_template
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

    pub(crate) fn lookup(
        &self,
        namespace: &NativeTextPrefixCacheNamespace,
        tokens: &[usize],
        metrics: &NativeTextPrefixCacheMetrics,
    ) -> Option<NativeTextPrefixCacheHit<C>> {
        let mut inner = self.inner.lock_or_panic("native text prefix cache");
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
        let entry = inner.entries.get_mut(&best_key)?;
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
        let mut inner = self.inner.lock_or_panic("native text prefix cache");
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
            .lock_or_panic("native text prefix cache metrics");
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
            .lock_or_panic("native text prefix cache metrics");
        update(&mut counters);
    }
}
