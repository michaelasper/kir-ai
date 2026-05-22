use llm_kv_cache::{BlockId, BlockPool, BlockTable, CacheBlock, KvCacheError};

#[test]
fn block_id_exposes_invalid_sentinel_without_accepting_it_as_valid() {
    assert_eq!(BlockId::INVALID.as_u64(), 0);
    assert!(BlockId::INVALID.is_invalid());
    assert!(BlockId::new(0).is_none());

    let id = BlockId::new(42).expect("non-zero ids are valid");

    assert_eq!(id.as_u64(), 42);
    assert!(id.is_valid());
    assert_eq!(id.to_string(), "42");
}

#[test]
fn cache_block_appends_token_rows_and_tracks_metadata() {
    let mut block = CacheBlock::new(2, 3).expect("block shape is valid");
    let hash = [7_u8; 32];

    assert!(block.id().is_valid());
    assert_eq!(block.capacity_tokens(), 2);
    assert_eq!(block.vector_len(), 3);
    assert_eq!(block.token_count(), 0);
    assert_eq!(block.remaining_tokens(), 2);
    assert_eq!(block.ref_count(), 0);
    assert_eq!(block.content_hash(), None);
    assert_eq!(block.last_access(), 0);

    assert_eq!(
        block
            .append(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0])
            .expect("first token fits"),
        0
    );
    assert_eq!(
        block
            .append(&[4.0, 5.0, 6.0], &[40.0, 50.0, 60.0])
            .expect("second token fits"),
        1
    );
    block.increment_ref_count();
    block.set_content_hash(Some(hash));
    block.touch(9);

    assert_eq!(block.token_count(), 2);
    assert_eq!(block.remaining_tokens(), 0);
    assert!(block.is_full());
    assert_eq!(block.key(0), Some(&[1.0, 2.0, 3.0][..]));
    assert_eq!(block.value(1), Some(&[40.0, 50.0, 60.0][..]));
    assert_eq!(block.keys(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(block.values(), &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    assert_eq!(block.ref_count(), 1);
    assert_eq!(block.content_hash(), Some(&hash));
    assert_eq!(block.last_access(), 9);

    let err = block
        .append(&[7.0, 8.0, 9.0], &[70.0, 80.0, 90.0])
        .expect_err("fixed block capacity is enforced");
    assert_eq!(
        err,
        KvCacheError::CapacityExceeded {
            requested: 1,
            available: 0
        }
    );
}

#[test]
fn cache_block_clear_resets_contents_without_changing_identity() {
    let mut block = CacheBlock::new(2, 2).expect("block shape is valid");
    let id = block.id();

    block
        .append(&[1.0, 2.0], &[10.0, 20.0])
        .expect("token fits");
    block.increment_ref_count();
    block.set_content_hash(Some([1_u8; 32]));
    block.touch(11);

    block.clear();

    assert_eq!(block.id(), id);
    assert_eq!(block.token_count(), 0);
    assert_eq!(block.ref_count(), 0);
    assert_eq!(block.content_hash(), None);
    assert_eq!(block.last_access(), 0);
    assert_eq!(block.keys(), &[]);
    assert_eq!(block.values(), &[]);
    assert_eq!(block.key_storage(), &[0.0, 0.0, 0.0, 0.0]);
    assert_eq!(block.value_storage(), &[0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn block_table_appends_counts_and_indexes_valid_block_ids() {
    let first = BlockId::new(1).expect("id is valid");
    let second = BlockId::new(2).expect("id is valid");
    let mut table = BlockTable::new();

    assert!(table.is_empty());
    assert_eq!(table.block_count(), 0);
    assert_eq!(table.get(0), None);

    table.append(first).expect("valid id appends");
    table.append(second).expect("valid id appends");

    assert!(!table.is_empty());
    assert_eq!(table.block_count(), 2);
    assert_eq!(table.get(0), Some(first));
    assert_eq!(table.get(1), Some(second));
    assert_eq!(table.as_slice(), &[first, second]);
    assert_eq!(table.remove_last(), Some(second));
    assert_eq!(table.as_slice(), &[first]);

    let err = table
        .append(BlockId::INVALID)
        .expect_err("invalid sentinel cannot be inserted");
    assert_eq!(err, KvCacheError::InvalidShape);
}

#[test]
fn block_pool_allocates_deallocates_and_reuses_free_blocks() {
    let mut pool = BlockPool::new(2, 4, 3).expect("pool shape is valid");

    assert_eq!(pool.total_blocks(), 2);
    assert_eq!(pool.free_blocks(), 2);
    assert_eq!(pool.allocated_blocks(), 0);

    let first = pool.allocate().expect("first block is available");
    let second = pool.allocate().expect("second block is available");

    assert_ne!(first, second);
    assert_eq!(pool.free_blocks(), 0);
    assert_eq!(pool.allocated_blocks(), 2);
    assert!(pool.allocate().is_none());
    assert_eq!(pool.block(first).expect("block exists").ref_count(), 1);

    let first_block = pool.block_mut(first).expect("block exists");
    first_block
        .append(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0])
        .expect("token fits");
    assert_eq!(first_block.token_count(), 1);

    assert!(pool.deallocate(first));
    assert_eq!(pool.free_blocks(), 1);
    assert_eq!(pool.allocated_blocks(), 1);
    assert_eq!(
        pool.block(first)
            .expect("block remains addressable")
            .token_count(),
        0
    );
    assert_eq!(
        pool.block(first)
            .expect("block remains addressable")
            .ref_count(),
        0
    );

    let reused = pool.allocate().expect("freed block is reusable");

    assert_eq!(reused, first);
    assert_eq!(pool.block(reused).expect("block exists").token_count(), 0);
    assert_eq!(pool.block(reused).expect("block exists").ref_count(), 1);
}

#[test]
fn block_pool_maintains_lru_access_order_for_allocated_blocks() {
    let mut pool = BlockPool::new(3, 2, 2).expect("pool shape is valid");
    let first = pool.allocate().expect("first block is available");
    let second = pool.allocate().expect("second block is available");
    let third = pool.allocate().expect("third block is available");

    assert_eq!(pool.lru_block(), Some(first));

    pool.touch(first).expect("allocated block can be touched");
    assert_eq!(pool.lru_block(), Some(second));

    pool.touch(second).expect("allocated block can be touched");
    assert_eq!(pool.lru_block(), Some(third));

    assert!(pool.deallocate(third));
    assert_eq!(pool.lru_block(), Some(first));
    assert_eq!(
        pool.block(third)
            .expect("block remains addressable")
            .last_access(),
        0
    );
}
