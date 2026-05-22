use crate::{BlockId, KvCacheError};

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
