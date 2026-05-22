use std::collections::HashMap;

use crate::{BlockId, CacheBlock, KvCacheError, SessionBlockTable, SessionId};

#[derive(Debug, Clone, PartialEq)]
pub struct BlockPool {
    blocks: HashMap<BlockId, CacheBlock>,
    free_list: Vec<BlockId>,
    sessions: HashMap<SessionId, SessionBlockTable>,
    next_session_id: SessionId,
    access_clock: u64,
}

impl BlockPool {
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
            sessions: HashMap::new(),
            next_session_id: 1,
            access_clock: 0,
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
        let Some(session) = self.sessions.remove(&session_id) else {
            return false;
        };
        let owned_blocks: Vec<BlockId> = session.owned_block_ids().collect();
        for block_id in owned_blocks {
            self.release(block_id);
        }
        true
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

    pub fn lru_session(&self) -> Option<SessionId> {
        self.lru_session_except(None)
    }

    pub fn allocate(&mut self) -> Option<BlockId> {
        let block_id = self.free_list.pop()?;
        let access = self.next_access();
        let block = self.blocks.get_mut(&block_id)?;
        block.reset_for_allocation(access);
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
        block.increment_ref_count();
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
        self.blocks.get_mut(&block_id)
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
            block.increment_ref_count();
        }
        block.touch(access);
        Ok(())
    }

    fn evict_sessions_until_free(&mut self, excluded_session_id: SessionId) {
        while self.free_list.is_empty() {
            let Some(session_id) = self.lru_session_except(Some(excluded_session_id)) else {
                break;
            };
            if !self.release_session(session_id) {
                break;
            }
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
}
