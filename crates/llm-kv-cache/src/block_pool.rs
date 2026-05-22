use std::collections::HashMap;

use crate::{BlockId, CacheBlock, KvCacheError};

#[derive(Debug, Clone, PartialEq)]
pub struct BlockPool {
    blocks: HashMap<BlockId, CacheBlock>,
    free_list: Vec<BlockId>,
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

    fn next_access(&mut self) -> u64 {
        self.access_clock = self.access_clock.saturating_add(1);
        self.access_clock
    }
}
