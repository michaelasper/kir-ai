use std::{
    collections::HashMap,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
};

use crate::{BlockId, CacheBlock, CacheBlockHash, KvCacheError, SessionBlockTable, SessionId};
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Eq)]
struct PrefixEntry {
    blocks: Vec<PrefixBlockSnapshot>,
}

impl PrefixEntry {
    fn block_ids(&self) -> Vec<BlockId> {
        self.blocks
            .iter()
            .map(|snapshot| snapshot.block_id)
            .collect()
    }

    fn terminal_hash(&self) -> Option<&CacheBlockHash> {
        self.blocks.last().map(|snapshot| &snapshot.content_hash)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PrefixBlockSnapshot {
    block_id: BlockId,
    content_hash: CacheBlockHash,
}

/// Shared paged-KV block allocator.
///
/// Cloned handles coordinate through one mutex-protected allocator while session
/// block tables remain owned by the pool and are exposed only as snapshots.
#[derive(Debug, Clone)]
pub struct BlockPool {
    inner: Arc<Mutex<BlockPoolInner>>,
}

#[derive(Debug, Clone, PartialEq)]
struct BlockPoolInner {
    blocks: HashMap<BlockId, CacheBlock>,
    free_list: Vec<BlockId>,
    prefixes: HashMap<CacheBlockHash, PrefixEntry>,
    sessions: HashMap<SessionId, SessionBlockTable>,
    next_session_id: SessionId,
    access_clock: u64,
    metrics: PagedKvCacheMetrics,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct PagedKvCacheMetrics {
    total_refcount_increments: u64,
    total_cow_clones: u64,
    cow_bytes_saved: u64,
    blocks_evicted_lru: u64,
    sessions_evicted_lru: u64,
    eviction_high_water_mark: u64,
    pool_high_water_blocks: u64,
    max_refcount_seen: u64,
}

impl PagedKvCacheMetrics {
    fn record_refcount(&mut self, ref_count: usize) {
        self.max_refcount_seen = self.max_refcount_seen.max(ref_count as u64);
    }

    fn record_shared_refcount_increment(&mut self, ref_count: usize, bytes_saved: u64) {
        self.total_refcount_increments = self.total_refcount_increments.saturating_add(1);
        self.cow_bytes_saved = self.cow_bytes_saved.saturating_add(bytes_saved);
        self.record_refcount(ref_count);
    }

    fn record_cow_clone(&mut self) {
        self.total_cow_clones = self.total_cow_clones.saturating_add(1);
    }

    fn record_eviction(&mut self, evicted_blocks: usize, high_water_blocks: usize) {
        self.sessions_evicted_lru = self.sessions_evicted_lru.saturating_add(1);
        self.blocks_evicted_lru = self
            .blocks_evicted_lru
            .saturating_add(evicted_blocks as u64);
        self.eviction_high_water_mark = self.eviction_high_water_mark.max(high_water_blocks as u64);
    }

    fn record_pool_residency(&mut self, resident_blocks: usize) {
        self.pool_high_water_blocks = self.pool_high_water_blocks.max(resident_blocks as u64);
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PagedKvCacheMetricsSnapshot {
    pub total_blocks: u64,
    pub resident_blocks: u64,
    pub active_blocks: u64,
    pub blocks_in_use: u64,
    pub free_blocks: u64,
    pub free_list_blocks: u64,
    pub pool_utilization_pct: f64,
    pub sessions: u64,
    pub session_block_tables: u64,
    pub shared_blocks: u64,
    pub refcount_total: u64,
    pub avg_refcount: f64,
    pub max_refcount: u64,
    pub max_refcount_seen: u64,
    pub total_refcount_increments: u64,
    pub total_cow_clones: u64,
    pub cow_bytes_saved: u64,
    pub blocks_evicted_lru: u64,
    pub sessions_evicted_lru: u64,
    pub eviction_high_water_mark: u64,
    pub pool_high_water_blocks: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BlockPoolSnapshot {
    pub object: String,
    pub metrics: PagedKvCacheMetricsSnapshot,
    pub sessions: Vec<SessionBlockTableSnapshot>,
    pub blocks: Vec<CacheBlockSnapshot>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SessionBlockTableSnapshot {
    pub session_id: SessionId,
    pub created_at: u64,
    pub last_access: u64,
    pub block_table: Vec<SessionBlockSnapshot>,
    pub owned_blocks: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SessionBlockSnapshot {
    pub index: usize,
    pub block_id: u64,
    pub ref_count: u64,
    pub token_count: u64,
    pub content_hash: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CacheBlockSnapshot {
    pub block_id: u64,
    pub revision: u64,
    pub capacity_tokens: u64,
    pub vector_len: u64,
    pub token_count: u64,
    pub ref_count: u64,
    pub storage_ref_count: u64,
    pub content_hash: Option<String>,
    pub last_access: u64,
    pub resident_bytes: u64,
}

impl BlockPool {
    pub fn new(
        block_count: usize,
        block_capacity_tokens: usize,
        vector_len: usize,
    ) -> Result<Self, KvCacheError> {
        Ok(Self {
            inner: Arc::new(Mutex::new(BlockPoolInner::new(
                block_count,
                block_capacity_tokens,
                vector_len,
            )?)),
        })
    }

    pub fn total_blocks(&self) -> usize {
        self.lock_inner().total_blocks()
    }

    pub fn free_blocks(&self) -> usize {
        self.lock_inner().free_blocks()
    }

    pub fn allocated_blocks(&self) -> usize {
        self.lock_inner().allocated_blocks()
    }

    pub fn metrics_snapshot(&self) -> PagedKvCacheMetricsSnapshot {
        self.lock_inner().metrics_snapshot()
    }

    pub fn snapshot(&self) -> BlockPoolSnapshot {
        self.lock_inner().snapshot()
    }

    pub fn session_count(&self) -> usize {
        self.lock_inner().session_count()
    }

    pub fn session(&self, session_id: SessionId) -> Option<SessionBlockTable> {
        self.lock_inner().session(session_id).cloned()
    }

    pub fn create_session(&self) -> Result<SessionId, KvCacheError> {
        self.lock_inner().create_session()
    }

    pub fn release_session(&self, session_id: SessionId) -> bool {
        self.lock_inner().release_session(session_id)
    }

    pub fn allocate_for_session(&self, session_id: SessionId) -> Option<BlockId> {
        self.lock_inner().allocate_for_session(session_id)
    }

    pub fn append_block_to_session(
        &self,
        session_id: SessionId,
        block_id: BlockId,
    ) -> Result<(), KvCacheError> {
        self.lock_inner()
            .append_block_to_session(session_id, block_id)
    }

    pub fn read_session_block(&self, session_id: SessionId, index: usize) -> Option<BlockId> {
        self.lock_inner().read_session_block(session_id, index)
    }

    pub fn register_prefix(&self, prefix_hash: CacheBlockHash, block_ids: Vec<BlockId>) {
        self.lock_inner().register_prefix(prefix_hash, block_ids);
    }

    pub fn lookup_prefix(&self, prefix_hash: &CacheBlockHash) -> Option<Vec<BlockId>> {
        self.lock_inner().lookup_prefix(prefix_hash)
    }

    pub fn attach_prefix_to_session(
        &self,
        session_id: SessionId,
        prefix_hash: &CacheBlockHash,
    ) -> Option<Vec<BlockId>> {
        self.lock_inner()
            .attach_prefix_to_session(session_id, prefix_hash)
    }

    pub fn copy_on_write_session_block(
        &self,
        session_id: SessionId,
        index: usize,
    ) -> Result<BlockId, KvCacheError> {
        self.lock_inner()
            .copy_on_write_session_block(session_id, index)
    }

    pub fn lru_session(&self) -> Option<SessionId> {
        self.lock_inner().lru_session()
    }

    pub fn allocate(&self) -> Option<BlockId> {
        self.lock_inner().allocate()
    }

    pub fn deallocate(&self, block_id: BlockId) -> bool {
        self.lock_inner().deallocate(block_id)
    }

    pub fn retain(&self, block_id: BlockId) -> bool {
        self.lock_inner().retain(block_id)
    }

    pub fn release(&self, block_id: BlockId) -> bool {
        self.lock_inner().release(block_id)
    }

    pub fn touch(&self, block_id: BlockId) -> Option<()> {
        self.lock_inner().touch(block_id)
    }

    pub fn lru_block(&self) -> Option<BlockId> {
        self.lock_inner().lru_block()
    }

    pub fn block(&self, block_id: BlockId) -> Option<CacheBlock> {
        self.lock_inner().block(block_id).cloned()
    }

    pub fn block_mut(&mut self, block_id: BlockId) -> Option<&mut CacheBlock> {
        self.unique_inner_mut()?.block_mut(block_id)
    }

    pub fn with_block_mut<R>(
        &self,
        block_id: BlockId,
        f: impl FnOnce(&mut CacheBlock) -> R,
    ) -> Option<R> {
        let mut inner = self.lock_inner();
        let block = inner.block_mut(block_id)?;
        Some(f(block))
    }

    pub fn with_block<R>(&self, block_id: BlockId, f: impl FnOnce(&CacheBlock) -> R) -> Option<R> {
        let inner = self.lock_inner();
        let block = inner.block(block_id)?;
        Some(f(block))
    }

    fn lock_inner(&self) -> MutexGuard<'_, BlockPoolInner> {
        match self.inner.lock() {
            Ok(guard) => guard,
            Err(poisoned) => recover_poisoned_lock(poisoned),
        }
    }

    fn unique_inner_mut(&mut self) -> Option<&mut BlockPoolInner> {
        let inner = Arc::get_mut(&mut self.inner)?;
        Some(match inner.get_mut() {
            Ok(inner) => inner,
            Err(poisoned) => recover_poisoned_lock(poisoned),
        })
    }
}

impl PartialEq for BlockPool {
    fn eq(&self, other: &Self) -> bool {
        if Arc::ptr_eq(&self.inner, &other.inner) {
            return true;
        }
        let self_addr = Arc::as_ptr(&self.inner) as usize;
        let other_addr = Arc::as_ptr(&other.inner) as usize;
        if self_addr < other_addr {
            let self_inner = self.lock_inner();
            let other_inner = other.lock_inner();
            *self_inner == *other_inner
        } else {
            let other_inner = other.lock_inner();
            let self_inner = self.lock_inner();
            *self_inner == *other_inner
        }
    }
}

fn recover_poisoned_lock<T>(poisoned: PoisonError<T>) -> T {
    tracing::warn!("block pool mutex poisoned; recovering allocator state");
    poisoned.into_inner()
}

impl BlockPoolInner {
    pub fn new(
        block_count: usize,
        block_capacity_tokens: usize,
        vector_len: usize,
    ) -> Result<Self, KvCacheError> {
        if block_count == 0 {
            return Err(KvCacheError::InvalidShape);
        }
        let mut blocks = HashMap::with_capacity(block_count);
        let mut free_list = Vec::with_capacity(block_count);
        for _ in 0..block_count {
            let block = CacheBlock::new(block_capacity_tokens, vector_len)?;
            let block_id = block.id();
            free_list.push(block_id);
            blocks.insert(block_id, block);
        }
        Ok(Self {
            blocks,
            free_list,
            prefixes: HashMap::new(),
            sessions: HashMap::new(),
            next_session_id: 1,
            access_clock: 0,
            metrics: PagedKvCacheMetrics::default(),
        })
    }

    pub fn total_blocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn free_blocks(&self) -> usize {
        self.free_list.len()
    }

    pub fn allocated_blocks(&self) -> usize {
        self.blocks
            .values()
            .filter(|block| block.ref_count() > 0)
            .count()
    }

    pub fn metrics_snapshot(&self) -> PagedKvCacheMetricsSnapshot {
        let mut resident_blocks = 0_u64;
        let mut shared_blocks = 0_u64;
        let mut refcount_total = 0_u64;
        let mut max_refcount = 0_u64;
        for block in self.blocks.values() {
            let ref_count = block.ref_count() as u64;
            if ref_count == 0 {
                continue;
            }
            resident_blocks = resident_blocks.saturating_add(1);
            refcount_total = refcount_total.saturating_add(ref_count);
            max_refcount = max_refcount.max(ref_count);
            if ref_count > 1 {
                shared_blocks = shared_blocks.saturating_add(1);
            }
        }
        let total_blocks = self.blocks.len() as u64;
        let active_blocks = self.sessions.values().fold(0_u64, |total, session| {
            total.saturating_add(session.block_count() as u64)
        });
        let pool_utilization_pct = if total_blocks == 0 {
            0.0
        } else {
            (resident_blocks as f64 / total_blocks as f64) * 100.0
        };
        let avg_refcount = if resident_blocks == 0 {
            0.0
        } else {
            refcount_total as f64 / resident_blocks as f64
        };
        PagedKvCacheMetricsSnapshot {
            total_blocks,
            resident_blocks,
            active_blocks,
            blocks_in_use: resident_blocks,
            free_blocks: self.free_list.len() as u64,
            free_list_blocks: self.free_list.len() as u64,
            pool_utilization_pct,
            sessions: self.sessions.len() as u64,
            session_block_tables: self.sessions.len() as u64,
            shared_blocks,
            refcount_total,
            avg_refcount,
            max_refcount,
            max_refcount_seen: self.metrics.max_refcount_seen.max(max_refcount),
            total_refcount_increments: self.metrics.total_refcount_increments,
            total_cow_clones: self.metrics.total_cow_clones,
            cow_bytes_saved: self.metrics.cow_bytes_saved,
            blocks_evicted_lru: self.metrics.blocks_evicted_lru,
            sessions_evicted_lru: self.metrics.sessions_evicted_lru,
            eviction_high_water_mark: self.metrics.eviction_high_water_mark,
            pool_high_water_blocks: self.metrics.pool_high_water_blocks,
        }
    }

    pub fn snapshot(&self) -> BlockPoolSnapshot {
        let mut sessions = self
            .sessions
            .values()
            .map(|session| self.session_snapshot(session))
            .collect::<Vec<_>>();
        sessions.sort_by_key(|session| session.session_id);
        let mut blocks = self.blocks.values().map(block_snapshot).collect::<Vec<_>>();
        blocks.sort_by_key(|block| block.block_id);
        BlockPoolSnapshot {
            object: "kv_cache.block_pool".to_owned(),
            metrics: self.metrics_snapshot(),
            sessions,
            blocks,
        }
    }

    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    pub fn session(&self, session_id: SessionId) -> Option<&SessionBlockTable> {
        self.sessions.get(&session_id)
    }

    pub fn create_session(&mut self) -> Result<SessionId, KvCacheError> {
        let session_id = self.next_session_id()?;
        let access = self.next_access();
        let session = SessionBlockTable::new(session_id, access)?;
        self.sessions.insert(session_id, session);
        Ok(session_id)
    }

    pub fn release_session(&mut self, session_id: SessionId) -> bool {
        self.release_session_inner(session_id).is_some()
    }

    pub fn allocate_for_session(&mut self, session_id: SessionId) -> Option<BlockId> {
        if !self.sessions.contains_key(&session_id) {
            return None;
        }
        if self.free_list.is_empty() {
            self.evict_sessions_until_free(session_id);
        }
        let block_id = self.allocate()?;
        self.append_block_to_session_inner(session_id, block_id, false)
            .ok()?;
        Some(block_id)
    }

    pub fn append_block_to_session(
        &mut self,
        session_id: SessionId,
        block_id: BlockId,
    ) -> Result<(), KvCacheError> {
        self.append_block_to_session_inner(session_id, block_id, true)
    }

    pub fn read_session_block(&mut self, session_id: SessionId, index: usize) -> Option<BlockId> {
        let access = self.next_access();
        let block_id = {
            let session = self.sessions.get_mut(&session_id)?;
            let block_id = session.read_block(index)?;
            session.touch(access);
            block_id
        };
        let block = self.blocks.get_mut(&block_id)?;
        if block.ref_count() == 0 {
            return None;
        }
        block.touch(access);
        Some(block_id)
    }

    pub fn register_prefix(&mut self, prefix_hash: CacheBlockHash, block_ids: Vec<BlockId>) {
        if let Some(entry) = self.prefix_entry_for(&block_ids)
            && entry.terminal_hash() == Some(&prefix_hash)
        {
            self.prefixes.insert(prefix_hash, entry);
        } else {
            self.prefixes.remove(&prefix_hash);
        }
    }

    pub fn lookup_prefix(&self, prefix_hash: &CacheBlockHash) -> Option<Vec<BlockId>> {
        let entry = self.prefixes.get(prefix_hash)?;
        if self.prefix_blocks_match(prefix_hash, entry) {
            Some(entry.block_ids())
        } else {
            None
        }
    }

    pub fn attach_prefix_to_session(
        &mut self,
        session_id: SessionId,
        prefix_hash: &CacheBlockHash,
    ) -> Option<Vec<BlockId>> {
        if !self.sessions.contains_key(&session_id) {
            return None;
        }
        let block_ids = self.lookup_prefix(prefix_hash)?;
        for block_id in block_ids.iter().copied() {
            self.append_block_to_session_inner(session_id, block_id, true)
                .ok()?;
        }
        Some(block_ids)
    }

    pub fn copy_on_write_session_block(
        &mut self,
        session_id: SessionId,
        index: usize,
    ) -> Result<BlockId, KvCacheError> {
        let block_id = self
            .sessions
            .get(&session_id)
            .and_then(|session| session.read_block(index))
            .ok_or(KvCacheError::InvalidShape)?;
        let ref_count = self
            .blocks
            .get(&block_id)
            .filter(|block| block.ref_count() > 0)
            .ok_or(KvCacheError::InvalidShape)?
            .ref_count();
        if ref_count == 1 {
            self.touch_session_block(session_id, block_id)?;
            return Ok(block_id);
        }
        if self.free_list.is_empty() {
            self.evict_sessions_until_free(session_id);
        }
        let ref_count = self
            .blocks
            .get(&block_id)
            .filter(|block| block.ref_count() > 0)
            .ok_or(KvCacheError::InvalidShape)?
            .ref_count();
        if ref_count == 1 {
            self.touch_session_block(session_id, block_id)?;
            return Ok(block_id);
        }
        let new_block_id = self.allocate().ok_or(KvCacheError::CapacityExceeded {
            requested: 1,
            available: 0,
        })?;
        let access = self.next_access();
        let copy_result = {
            let [Some(source), Some(destination)] =
                self.blocks.get_disjoint_mut([&block_id, &new_block_id])
            else {
                self.release(new_block_id);
                return Err(KvCacheError::InvalidShape);
            };
            destination.copy_contents_from(source, access)
        };
        if let Err(error) = copy_result {
            self.release(new_block_id);
            return Err(error);
        }
        let replace_result = {
            let session = self
                .sessions
                .get_mut(&session_id)
                .ok_or(KvCacheError::InvalidShape)?;
            let result = session.replace_owned_block(index, new_block_id);
            session.touch(access);
            result
        };
        let (_, should_release_previous) = match replace_result {
            Ok(result) => result,
            Err(error) => {
                self.release(new_block_id);
                return Err(error);
            }
        };
        if should_release_previous {
            self.release(block_id);
        }
        self.metrics.record_cow_clone();
        Ok(new_block_id)
    }

    pub fn lru_session(&self) -> Option<SessionId> {
        self.lru_session_except(None)
    }

    pub fn allocate(&mut self) -> Option<BlockId> {
        let block_id = self.free_list.pop()?;
        let access = self.next_access();
        let block = self.blocks.get_mut(&block_id)?;
        block.reset_for_allocation(access);
        self.metrics.record_refcount(block.ref_count());
        self.record_pool_residency();
        Some(block_id)
    }

    pub fn deallocate(&mut self, block_id: BlockId) -> bool {
        self.release(block_id)
    }

    pub fn retain(&mut self, block_id: BlockId) -> bool {
        let Some(block) = self.blocks.get_mut(&block_id) else {
            return false;
        };
        if block.ref_count() == 0 {
            return false;
        }
        let bytes_saved = block.payload_bytes();
        let ref_count = block.increment_ref_count();
        self.metrics
            .record_shared_refcount_increment(ref_count, bytes_saved);
        true
    }

    pub fn release(&mut self, block_id: BlockId) -> bool {
        let Some(block) = self.blocks.get_mut(&block_id) else {
            return false;
        };
        if block.ref_count() == 0 {
            return false;
        }
        if block.decrement_ref_count() == 0 {
            block.clear();
            self.free_list.push(block_id);
        }
        true
    }

    pub fn touch(&mut self, block_id: BlockId) -> Option<()> {
        let access = self.next_access();
        let block = self.blocks.get_mut(&block_id)?;
        if block.ref_count() == 0 {
            return None;
        }
        block.touch(access);
        Some(())
    }

    pub fn lru_block(&self) -> Option<BlockId> {
        self.blocks
            .values()
            .filter(|block| block.ref_count() > 0)
            .min_by_key(|block| block.last_access())
            .map(CacheBlock::id)
    }

    pub fn block(&self, block_id: BlockId) -> Option<&CacheBlock> {
        self.blocks.get(&block_id)
    }

    pub fn block_mut(&mut self, block_id: BlockId) -> Option<&mut CacheBlock> {
        let block = self.blocks.get_mut(&block_id)?;
        if block.ref_count() == 1 {
            Some(block)
        } else {
            None
        }
    }

    fn append_block_to_session_inner(
        &mut self,
        session_id: SessionId,
        block_id: BlockId,
        retain_new_owner: bool,
    ) -> Result<(), KvCacheError> {
        let block = self
            .blocks
            .get(&block_id)
            .ok_or(KvCacheError::InvalidShape)?;
        if block.ref_count() == 0 {
            return Err(KvCacheError::InvalidShape);
        }
        let access = self.next_access();
        let newly_owned = {
            let session = self
                .sessions
                .get_mut(&session_id)
                .ok_or(KvCacheError::InvalidShape)?;
            let newly_owned = session.append_owned_block(block_id)?;
            session.touch(access);
            newly_owned
        };
        let block = self
            .blocks
            .get_mut(&block_id)
            .ok_or(KvCacheError::InvalidShape)?;
        if retain_new_owner && newly_owned {
            let bytes_saved = block.payload_bytes();
            let ref_count = block.increment_ref_count();
            self.metrics
                .record_shared_refcount_increment(ref_count, bytes_saved);
        }
        block.touch(access);
        Ok(())
    }

    fn release_session_inner(&mut self, session_id: SessionId) -> Option<usize> {
        let session = self.sessions.remove(&session_id)?;
        let owned_blocks: Vec<BlockId> = session.owned_block_ids().collect();
        let mut released_blocks = 0_usize;
        for block_id in owned_blocks {
            let was_resident = self
                .blocks
                .get(&block_id)
                .is_some_and(|block| block.ref_count() > 0);
            if self.release(block_id)
                && was_resident
                && self
                    .blocks
                    .get(&block_id)
                    .is_some_and(|block| block.ref_count() == 0)
            {
                released_blocks = released_blocks.saturating_add(1);
            }
        }
        Some(released_blocks)
    }

    fn touch_session_block(
        &mut self,
        session_id: SessionId,
        block_id: BlockId,
    ) -> Result<(), KvCacheError> {
        let access = self.next_access();
        let session = self
            .sessions
            .get_mut(&session_id)
            .ok_or(KvCacheError::InvalidShape)?;
        session.touch(access);
        let block = self
            .blocks
            .get_mut(&block_id)
            .ok_or(KvCacheError::InvalidShape)?;
        if block.ref_count() == 0 {
            return Err(KvCacheError::InvalidShape);
        }
        block.touch(access);
        Ok(())
    }

    fn prefix_entry_for(&self, block_ids: &[BlockId]) -> Option<PrefixEntry> {
        let mut blocks = Vec::with_capacity(block_ids.len());
        for block_id in block_ids {
            let block = self.blocks.get(block_id)?;
            if block.ref_count() == 0 {
                return None;
            }
            blocks.push(PrefixBlockSnapshot {
                block_id: *block_id,
                content_hash: *block.content_hash()?,
            });
        }
        Some(PrefixEntry { blocks })
    }

    fn prefix_blocks_match(&self, prefix_hash: &CacheBlockHash, entry: &PrefixEntry) -> bool {
        entry.terminal_hash() == Some(prefix_hash)
            && entry.blocks.iter().all(|snapshot| {
                self.blocks.get(&snapshot.block_id).is_some_and(|block| {
                    block.ref_count() > 0
                        && block
                            .content_hash()
                            .is_some_and(|content_hash| content_hash == &snapshot.content_hash)
                })
            })
    }

    fn evict_sessions_until_free(&mut self, excluded_session_id: SessionId) {
        while self.free_list.is_empty() {
            let Some(session_id) = self.lru_session_except(Some(excluded_session_id)) else {
                break;
            };
            let high_water_blocks = self.resident_blocks_from_free_list();
            let Some(evicted_blocks) = self.release_session_inner(session_id) else {
                break;
            };
            self.metrics
                .record_eviction(evicted_blocks, high_water_blocks);
        }
    }

    fn lru_session_except(&self, excluded_session_id: Option<SessionId>) -> Option<SessionId> {
        self.sessions
            .values()
            .filter(|session| Some(session.session_id()) != excluded_session_id)
            .filter(|session| !session.is_empty())
            .min_by_key(|session| {
                (
                    session.last_access(),
                    session.created_at(),
                    session.session_id(),
                )
            })
            .map(SessionBlockTable::session_id)
    }

    fn next_session_id(&mut self) -> Result<SessionId, KvCacheError> {
        let session_id = self.next_session_id;
        self.next_session_id = self
            .next_session_id
            .checked_add(1)
            .filter(|next| *next != 0)
            .ok_or(KvCacheError::InvalidShape)?;
        Ok(session_id)
    }

    fn next_access(&mut self) -> u64 {
        self.access_clock = self.access_clock.saturating_add(1);
        self.access_clock
    }

    fn resident_blocks_from_free_list(&self) -> usize {
        self.blocks.len().saturating_sub(self.free_list.len())
    }

    fn record_pool_residency(&mut self) {
        let resident_blocks = self.resident_blocks_from_free_list();
        self.metrics.record_pool_residency(resident_blocks);
    }

    fn session_snapshot(&self, session: &SessionBlockTable) -> SessionBlockTableSnapshot {
        let block_table = session
            .block_ids()
            .iter()
            .copied()
            .enumerate()
            .map(|(index, block_id)| {
                let block = self.blocks.get(&block_id);
                SessionBlockSnapshot {
                    index,
                    block_id: block_id.as_u64(),
                    ref_count: block.map_or(0, |block| block.ref_count() as u64),
                    token_count: block.map_or(0, |block| block.token_count() as u64),
                    content_hash: block
                        .and_then(CacheBlock::content_hash)
                        .map(cache_block_hash_hex),
                }
            })
            .collect();
        let mut owned_blocks = session
            .owned_block_ids()
            .map(BlockId::as_u64)
            .collect::<Vec<_>>();
        owned_blocks.sort_unstable();
        SessionBlockTableSnapshot {
            session_id: session.session_id(),
            created_at: session.created_at(),
            last_access: session.last_access(),
            block_table,
            owned_blocks,
        }
    }
}

fn block_snapshot(block: &CacheBlock) -> CacheBlockSnapshot {
    CacheBlockSnapshot {
        block_id: block.id().as_u64(),
        revision: block.revision(),
        capacity_tokens: block.capacity_tokens() as u64,
        vector_len: block.vector_len() as u64,
        token_count: block.token_count() as u64,
        ref_count: block.ref_count() as u64,
        storage_ref_count: block.retained_storage_ref_count() as u64,
        content_hash: block.content_hash().map(cache_block_hash_hex),
        last_access: block.last_access(),
        resident_bytes: block.payload_bytes(),
    }
}

fn cache_block_hash_hex(hash: &CacheBlockHash) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(hash.len() * 2);
    for byte in hash {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}
