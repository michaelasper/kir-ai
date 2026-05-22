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

#[test]
fn block_pool_creates_sessions_and_release_returns_owned_blocks() {
    let mut pool = BlockPool::new(3, 2, 2).expect("pool shape is valid");
    let first_session = pool.create_session().expect("session id is available");
    let second_session = pool.create_session().expect("session id is available");

    assert_ne!(first_session, second_session);
    assert_eq!(pool.session_count(), 2);
    assert_eq!(
        pool.session(first_session)
            .expect("session exists")
            .session_id(),
        first_session
    );
    assert!(
        pool.session(first_session)
            .expect("session exists")
            .created_at()
            < pool
                .session(second_session)
                .expect("session exists")
                .created_at()
    );

    let first_block = pool
        .allocate_for_session(first_session)
        .expect("first session can allocate");
    let second_block = pool
        .allocate_for_session(first_session)
        .expect("first session can allocate again");
    let third_block = pool
        .allocate_for_session(second_session)
        .expect("second session can allocate");

    assert_ne!(first_block, second_block);
    assert_ne!(first_block, third_block);
    assert_eq!(pool.free_blocks(), 0);
    assert_eq!(pool.allocated_blocks(), 3);
    assert_eq!(
        pool.session(first_session)
            .expect("session remains live")
            .block_count(),
        2
    );
    assert_eq!(
        pool.session(first_session)
            .expect("session remains live")
            .owned_block_count(),
        2
    );

    assert!(pool.release_session(first_session));

    assert!(pool.session(first_session).is_none());
    assert_eq!(pool.session_count(), 1);
    assert_eq!(pool.free_blocks(), 2);
    assert_eq!(pool.allocated_blocks(), 1);
    assert_eq!(
        pool.block(first_block).expect("block exists").ref_count(),
        0
    );
    assert_eq!(
        pool.block(second_block).expect("block exists").ref_count(),
        0
    );
    assert_eq!(
        pool.block(third_block).expect("block exists").ref_count(),
        1
    );

    let reused = pool
        .allocate_for_session(second_session)
        .expect("released session blocks are reusable");
    assert!([first_block, second_block].contains(&reused));
}

#[test]
fn releasing_session_decrements_owned_shared_block_refcounts_once() {
    let mut pool = BlockPool::new(1, 2, 2).expect("pool shape is valid");
    let owner = pool.create_session().expect("session id is available");
    let reader = pool.create_session().expect("session id is available");
    let shared_block = pool
        .allocate_for_session(owner)
        .expect("owner can allocate");

    pool.append_block_to_session(reader, shared_block)
        .expect("second session can share allocated block");
    pool.append_block_to_session(reader, shared_block)
        .expect("same session can map shared block more than once");

    assert_eq!(
        pool.block(shared_block).expect("block exists").ref_count(),
        2
    );
    assert_eq!(
        pool.session(reader).expect("session exists").block_count(),
        2
    );
    assert_eq!(
        pool.session(reader)
            .expect("session exists")
            .owned_block_count(),
        1
    );

    assert!(pool.release_session(owner));
    assert_eq!(
        pool.block(shared_block).expect("block exists").ref_count(),
        1
    );
    assert_eq!(pool.free_blocks(), 0);
    assert_eq!(pool.read_session_block(reader, 0), Some(shared_block));
    assert_eq!(pool.read_session_block(reader, 1), Some(shared_block));

    assert!(pool.release_session(reader));
    assert_eq!(
        pool.block(shared_block).expect("block exists").ref_count(),
        0
    );
    assert_eq!(pool.free_blocks(), 1);
}

#[test]
fn full_pool_evicts_least_recently_used_session_before_allocating() {
    let mut pool = BlockPool::new(2, 2, 2).expect("pool shape is valid");
    let cold = pool.create_session().expect("session id is available");
    let hot = pool.create_session().expect("session id is available");
    let cold_block = pool.allocate_for_session(cold).expect("cold allocates");
    let hot_block = pool.allocate_for_session(hot).expect("hot allocates");

    assert_eq!(pool.lru_session(), Some(cold));
    assert_eq!(pool.read_session_block(hot, 0), Some(hot_block));
    assert_eq!(pool.lru_session(), Some(cold));

    let newcomer = pool.create_session().expect("session id is available");
    let reused_block = pool
        .allocate_for_session(newcomer)
        .expect("LRU session eviction frees a block");

    assert_eq!(reused_block, cold_block);
    assert!(pool.session(cold).is_none());
    assert!(pool.session(hot).is_some());
    assert!(pool.session(newcomer).is_some());
    assert_eq!(
        pool.block(hot_block).expect("hot block exists").ref_count(),
        1
    );
    assert_eq!(pool.total_blocks(), 2);
    assert_eq!(pool.free_blocks(), 0);
}

#[test]
fn multiple_sessions_read_independent_block_tables_safely() {
    let mut pool = BlockPool::new(2, 2, 2).expect("pool shape is valid");
    let first = pool.create_session().expect("session id is available");
    let second = pool.create_session().expect("session id is available");
    let first_block = pool.allocate_for_session(first).expect("first allocates");
    let second_block = pool.allocate_for_session(second).expect("second allocates");

    pool.block_mut(first_block)
        .expect("first block exists")
        .append(&[1.0, 2.0], &[10.0, 20.0])
        .expect("token fits");
    pool.block_mut(second_block)
        .expect("second block exists")
        .append(&[3.0, 4.0], &[30.0, 40.0])
        .expect("token fits");

    assert_eq!(pool.read_session_block(first, 0), Some(first_block));
    assert_eq!(pool.read_session_block(second, 0), Some(second_block));
    assert_eq!(
        pool.block(first_block).expect("first block exists").key(0),
        Some(&[1.0, 2.0][..])
    );
    assert_eq!(
        pool.block(second_block)
            .expect("second block exists")
            .key(0),
        Some(&[3.0, 4.0][..])
    );

    assert!(pool.release_session(first));
    assert_eq!(pool.read_session_block(second, 0), Some(second_block));
    assert_eq!(
        pool.block(second_block)
            .expect("second block exists")
            .ref_count(),
        1
    );
}
