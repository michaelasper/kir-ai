use std::collections::HashSet;

use crate::{BlockId, KvCacheError};

pub type SessionId = u64;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BlockTable {
    blocks: Vec<BlockId>,
}

impl BlockTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            blocks: Vec::with_capacity(capacity),
        }
    }

    pub fn append(&mut self, block_id: BlockId) -> Result<(), KvCacheError> {
        if block_id.is_invalid() {
            return Err(KvCacheError::InvalidShape);
        }
        self.blocks.push(block_id);
        Ok(())
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<BlockId> {
        self.blocks.get(index).copied()
    }

    pub fn as_slice(&self) -> &[BlockId] {
        &self.blocks
    }

    pub fn remove_last(&mut self) -> Option<BlockId> {
        self.blocks.pop()
    }

    pub fn clear(&mut self) {
        self.blocks.clear();
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionBlockTable {
    session_id: SessionId,
    created_at: u64,
    last_access: u64,
    block_table: BlockTable,
    owned_blocks: HashSet<BlockId>,
}

impl SessionBlockTable {
    pub fn new(session_id: SessionId, created_at: u64) -> Result<Self, KvCacheError> {
        if session_id == 0 {
            return Err(KvCacheError::InvalidShape);
        }
        Ok(Self {
            session_id,
            created_at,
            last_access: created_at,
            block_table: BlockTable::new(),
            owned_blocks: HashSet::new(),
        })
    }

    pub fn session_id(&self) -> SessionId {
        self.session_id
    }

    pub fn created_at(&self) -> u64 {
        self.created_at
    }

    pub fn last_access(&self) -> u64 {
        self.last_access
    }

    pub fn block_count(&self) -> usize {
        self.block_table.block_count()
    }

    pub fn owned_block_count(&self) -> usize {
        self.owned_blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.block_table.is_empty()
    }

    pub fn append_block(&mut self, block_id: BlockId) -> Result<(), KvCacheError> {
        self.append_owned_block(block_id).map(|_| ())
    }

    pub(crate) fn append_owned_block(&mut self, block_id: BlockId) -> Result<bool, KvCacheError> {
        if block_id.is_invalid() {
            return Err(KvCacheError::InvalidShape);
        }
        self.block_table.append(block_id)?;
        Ok(self.owned_blocks.insert(block_id))
    }

    pub fn read_block(&self, index: usize) -> Option<BlockId> {
        self.block_table.get(index)
    }

    pub fn owns_block(&self, block_id: BlockId) -> bool {
        self.owned_blocks.contains(&block_id)
    }

    pub fn block_ids(&self) -> &[BlockId] {
        self.block_table.as_slice()
    }

    pub fn owned_block_ids(&self) -> impl Iterator<Item = BlockId> + '_ {
        self.owned_blocks.iter().copied()
    }

    pub(crate) fn touch(&mut self, last_access: u64) {
        self.last_access = last_access;
    }
}
