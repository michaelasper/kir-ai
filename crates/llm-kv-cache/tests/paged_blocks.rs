use std::{
    collections::HashSet,
    sync::{Arc, Barrier, Mutex},
    thread,
    time::Duration,
};

use llm_kv_cache::{
    BlockId, BlockPool, BlockTable, CacheBlock, KvCacheError, cache_block_chain_hash,
};

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
fn cache_block_append_invalidates_content_hash() {
    let mut block = CacheBlock::new(2, 2).expect("block shape is valid");

    block.set_content_hash(Some([3_u8; 32]));
    block
        .append(&[1.0, 2.0], &[10.0, 20.0])
        .expect("token fits");

    assert_eq!(block.content_hash(), None);
}

#[test]
fn cache_block_chain_hash_depends_on_full_prefix_identity() {
    let root_hash = [0_u8; 32];
    let first_hash =
        cache_block_chain_hash("model-a", "cache-context-a", &root_hash, &[11, 12, 13]);
    let first_hash_again =
        cache_block_chain_hash("model-a", "cache-context-a", &root_hash, &[11, 12, 13]);

    assert_eq!(first_hash, first_hash_again);
    assert_ne!(
        first_hash,
        cache_block_chain_hash("model-b", "cache-context-a", &root_hash, &[11, 12, 13])
    );
    assert_ne!(
        first_hash,
        cache_block_chain_hash("model-a", "cache-context-b", &root_hash, &[11, 12, 13])
    );
    assert_ne!(
        first_hash,
        cache_block_chain_hash("model-a", "cache-context-a", &[1_u8; 32], &[11, 12, 13])
    );
    assert_ne!(
        first_hash,
        cache_block_chain_hash("model-a", "cache-context-a", &root_hash, &[11, 12, 99])
    );

    let second_hash = cache_block_chain_hash("model-a", "cache-context-a", &first_hash, &[21, 22]);
    let changed_parent_hash =
        cache_block_chain_hash("model-a", "cache-context-a", &root_hash, &[41, 42]);
    let changed_second_hash = cache_block_chain_hash(
        "model-a",
        "cache-context-a",
        &changed_parent_hash,
        &[21, 22],
    );

    assert_ne!(second_hash, changed_second_hash);
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
fn cloned_block_pool_views_share_allocator_state() {
    let first_view = BlockPool::new(1, 4, 3).expect("pool shape is valid");
    let second_view = first_view.clone();

    let block = first_view.allocate().expect("only block is available");

    assert_eq!(second_view.allocate(), None);
    assert_eq!(second_view.allocated_blocks(), 1);
    assert_eq!(second_view.free_blocks(), 0);

    assert!(second_view.deallocate(block));
    assert_eq!(first_view.allocated_blocks(), 0);
    assert_eq!(first_view.free_blocks(), 1);
}

#[test]
fn cloned_block_pool_coordinates_concurrent_allocation_and_release() {
    const WORKERS: usize = 8;
    const ITERATIONS: usize = 64;

    let pool = BlockPool::new(WORKERS, 2, 2).expect("pool shape is valid");
    let active_blocks = Arc::new(Mutex::new(HashSet::<BlockId>::new()));
    let start = Arc::new(Barrier::new(WORKERS));
    let mut handles = Vec::with_capacity(WORKERS);

    for _ in 0..WORKERS {
        let pool = pool.clone();
        let active_blocks = Arc::clone(&active_blocks);
        let start = Arc::clone(&start);
        handles.push(thread::spawn(move || {
            start.wait();
            for _ in 0..ITERATIONS {
                let block = loop {
                    if let Some(block) = pool.allocate() {
                        break block;
                    }
                    thread::yield_now();
                };
                {
                    let mut active_blocks = active_blocks
                        .lock()
                        .expect("active block set lock is not poisoned");
                    assert!(
                        active_blocks.insert(block),
                        "block {block} was allocated concurrently more than once"
                    );
                }

                thread::sleep(Duration::from_micros(50));

                {
                    let mut active_blocks = active_blocks
                        .lock()
                        .expect("active block set lock is not poisoned");
                    assert!(
                        active_blocks.remove(&block),
                        "block {block} was missing from active set before release"
                    );
                }
                assert!(pool.deallocate(block), "allocated block should release");
            }
        }));
    }

    for handle in handles {
        handle.join().expect("allocation worker should not panic");
    }

    assert_eq!(pool.allocated_blocks(), 0);
    assert_eq!(pool.free_blocks(), WORKERS);
}

#[test]
fn block_pool_maintains_lru_access_order_for_allocated_blocks() {
    let pool = BlockPool::new(3, 2, 2).expect("pool shape is valid");
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
    let pool = BlockPool::new(3, 2, 2).expect("pool shape is valid");
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
fn block_pool_prefix_lookup_and_attach_share_registered_blocks() {
    let mut pool = BlockPool::new(3, 2, 2).expect("pool shape is valid");
    let owner = pool.create_session().expect("session id is available");
    let reader = pool.create_session().expect("session id is available");
    let root_hash = [0_u8; 32];
    let first_hash = cache_block_chain_hash("model-a", "cache-context-a", &root_hash, &[1, 2]);
    let terminal_hash = cache_block_chain_hash("model-a", "cache-context-a", &first_hash, &[3, 4]);
    let first = pool
        .allocate_for_session(owner)
        .expect("owner can allocate first prefix block");
    let second = pool
        .allocate_for_session(owner)
        .expect("owner can allocate second prefix block");

    pool.block_mut(first)
        .expect("first block is exclusive")
        .set_content_hash(Some(first_hash));
    pool.block_mut(second)
        .expect("second block is exclusive")
        .set_content_hash(Some(terminal_hash));
    pool.register_prefix(terminal_hash, vec![first, second]);

    assert_eq!(
        pool.lookup_prefix(&terminal_hash),
        Some(vec![first, second])
    );
    assert_eq!(
        pool.attach_prefix_to_session(reader, &terminal_hash),
        Some(vec![first, second])
    );
    assert_eq!(pool.read_session_block(reader, 0), Some(first));
    assert_eq!(pool.read_session_block(reader, 1), Some(second));
    assert_eq!(
        pool.block(first).expect("first block exists").ref_count(),
        2
    );
    assert_eq!(
        pool.block(second).expect("second block exists").ref_count(),
        2
    );
}

#[test]
fn block_pool_prefix_attach_misses_when_intermediate_block_is_reused_with_different_hash() {
    let mut pool = BlockPool::new(3, 2, 2).expect("pool shape is valid");
    let owner = pool.create_session().expect("session id is available");
    let root_hash = [0_u8; 32];
    let first_hash = cache_block_chain_hash("model-a", "cache-context-a", &root_hash, &[1, 2]);
    let terminal_hash = cache_block_chain_hash("model-a", "cache-context-a", &first_hash, &[3, 4]);
    let reused_hash = cache_block_chain_hash("model-a", "cache-context-a", &root_hash, &[99, 100]);
    let first = pool
        .allocate_for_session(owner)
        .expect("owner can allocate first prefix block");
    let second = pool
        .allocate_for_session(owner)
        .expect("owner can allocate second prefix block");

    pool.block_mut(first)
        .expect("first block is exclusive")
        .set_content_hash(Some(first_hash));
    pool.block_mut(second)
        .expect("second block is exclusive")
        .set_content_hash(Some(terminal_hash));
    pool.register_prefix(terminal_hash, vec![first, second]);

    assert!(pool.release(first));
    let recycler = pool.create_session().expect("session id is available");
    let reused = pool
        .allocate_for_session(recycler)
        .expect("released intermediate block can be reused");
    assert_eq!(reused, first);
    pool.block_mut(reused)
        .expect("reused block is exclusive")
        .set_content_hash(Some(reused_hash));

    let first_ref_count = pool.block(first).expect("first block exists").ref_count();
    let second_ref_count = pool.block(second).expect("second block exists").ref_count();
    let reader = pool.create_session().expect("session id is available");

    assert_eq!(pool.lookup_prefix(&terminal_hash), None);
    assert_eq!(pool.attach_prefix_to_session(reader, &terminal_hash), None);
    assert_eq!(
        pool.block(first).expect("first block exists").ref_count(),
        first_ref_count
    );
    assert_eq!(
        pool.block(second).expect("second block exists").ref_count(),
        second_ref_count
    );
    assert!(
        pool.session(reader)
            .expect("reader session exists")
            .is_empty()
    );
}

#[test]
fn releasing_session_decrements_owned_shared_block_refcounts_once() {
    let pool = BlockPool::new(1, 2, 2).expect("pool shape is valid");
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
fn shared_session_block_is_not_directly_mutable_without_cow() {
    let mut pool = BlockPool::new(2, 2, 2).expect("pool shape is valid");
    let owner = pool.create_session().expect("session id is available");
    let writer = pool.create_session().expect("session id is available");
    let shared_block = pool.allocate_for_session(owner).expect("owner allocates");

    pool.append_block_to_session(writer, shared_block)
        .expect("writer can share owner prefix block");

    assert_eq!(
        pool.block(shared_block).expect("block exists").ref_count(),
        2
    );
    assert!(pool.block_mut(shared_block).is_none());
}

#[test]
fn copy_on_write_session_block_clones_shared_prefix_on_first_write() {
    let mut pool = BlockPool::new(3, 2, 2).expect("pool shape is valid");
    let reader = pool.create_session().expect("session id is available");
    let writer = pool.create_session().expect("session id is available");
    let shared_block = pool.allocate_for_session(reader).expect("reader allocates");

    pool.block_mut(shared_block)
        .expect("exclusive block is mutable")
        .append(&[1.0, 2.0], &[10.0, 20.0])
        .expect("token fits");
    pool.append_block_to_session(writer, shared_block)
        .expect("writer can share reader prefix block");

    let writer_block = pool
        .copy_on_write_session_block(writer, 0)
        .expect("shared writer block is cloned");

    assert_ne!(writer_block, shared_block);
    assert_eq!(pool.read_session_block(reader, 0), Some(shared_block));
    assert_eq!(pool.read_session_block(writer, 0), Some(writer_block));
    assert_eq!(
        pool.block(shared_block)
            .expect("shared block exists")
            .ref_count(),
        1
    );
    assert_eq!(
        pool.block(writer_block)
            .expect("writer block exists")
            .ref_count(),
        1
    );
    assert_eq!(
        pool.block(shared_block)
            .expect("shared block exists")
            .key(0),
        Some(&[1.0, 2.0][..])
    );

    pool.block_mut(writer_block)
        .expect("writer block is now exclusive")
        .append(&[3.0, 4.0], &[30.0, 40.0])
        .expect("token fits");

    assert_eq!(
        pool.block(shared_block)
            .expect("reader block remains live")
            .token_count(),
        1
    );
    assert_eq!(
        pool.block(writer_block)
            .expect("writer block remains live")
            .token_count(),
        2
    );
    assert_eq!(
        pool.block(writer_block)
            .expect("writer block remains live")
            .key(1),
        Some(&[3.0, 4.0][..])
    );
}

#[test]
fn releasing_last_session_after_cow_returns_all_blocks_to_free_list() {
    let pool = BlockPool::new(2, 2, 2).expect("pool shape is valid");
    let reader = pool.create_session().expect("session id is available");
    let writer = pool.create_session().expect("session id is available");
    let shared_block = pool.allocate_for_session(reader).expect("reader allocates");

    pool.append_block_to_session(writer, shared_block)
        .expect("writer can share reader prefix block");
    let writer_block = pool
        .copy_on_write_session_block(writer, 0)
        .expect("shared writer block is cloned");

    assert_eq!(pool.free_blocks(), 0);
    assert!(pool.release_session(reader));
    assert_eq!(
        pool.block(shared_block).expect("block exists").ref_count(),
        0
    );
    assert_eq!(
        pool.block(writer_block).expect("block exists").ref_count(),
        1
    );
    assert_eq!(pool.free_blocks(), 1);

    assert!(pool.release_session(writer));
    assert_eq!(
        pool.block(writer_block).expect("block exists").ref_count(),
        0
    );
    assert_eq!(pool.free_blocks(), 2);
}

#[test]
fn full_pool_evicts_least_recently_used_session_before_allocating() {
    let pool = BlockPool::new(2, 2, 2).expect("pool shape is valid");
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

#[test]
fn block_pool_metrics_track_sharing_cow_eviction_and_utilization() {
    let mut pool = BlockPool::new(3, 2, 2).expect("pool shape is valid");
    let initial = pool.metrics_snapshot();

    assert_eq!(initial.total_blocks, 3);
    assert_eq!(initial.resident_blocks, 0);
    assert_eq!(initial.active_blocks, 0);
    assert_eq!(initial.free_list_blocks, 3);
    assert_eq!(initial.pool_utilization_pct, 0.0);
    assert_eq!(initial.total_refcount_increments, 0);
    assert_eq!(initial.pool_high_water_blocks, 0);

    let owner = pool
        .create_session()
        .expect("owner session id is available");
    let reader = pool
        .create_session()
        .expect("reader session id is available");
    let first = pool
        .allocate_for_session(owner)
        .expect("owner can allocate first block");
    let second = pool
        .allocate_for_session(owner)
        .expect("owner can allocate second block");

    pool.block_mut(first)
        .expect("first block is exclusive")
        .append(&[1.0, 2.0], &[10.0, 20.0])
        .expect("token fits");
    pool.append_block_to_session(reader, first)
        .expect("reader can share owner block");

    let shared = pool.metrics_snapshot();
    assert_eq!(shared.resident_blocks, 2);
    assert_eq!(shared.active_blocks, 3);
    assert_eq!(shared.shared_blocks, 1);
    assert_eq!(shared.refcount_total, 3);
    assert_eq!(shared.max_refcount_seen, 2);
    assert_eq!(shared.total_refcount_increments, 1);
    assert_eq!(shared.cow_bytes_saved, 32);
    assert_eq!(shared.free_list_blocks, 1);
    assert_eq!(shared.pool_high_water_blocks, 2);
    assert!((shared.avg_refcount - 1.5).abs() < f64::EPSILON);
    assert!((shared.pool_utilization_pct - 66.666_666_666_666_66).abs() < f64::EPSILON);

    let reader_block = pool
        .copy_on_write_session_block(reader, 0)
        .expect("shared reader block is cloned before write");
    assert_ne!(reader_block, first);

    let cow = pool.metrics_snapshot();
    assert_eq!(cow.resident_blocks, 3);
    assert_eq!(cow.active_blocks, 3);
    assert_eq!(cow.shared_blocks, 0);
    assert_eq!(cow.total_cow_clones, 1);
    assert_eq!(cow.cow_bytes_saved, 32);
    assert_eq!(cow.pool_high_water_blocks, 3);
    assert_eq!(cow.free_list_blocks, 0);

    let newcomer = pool
        .create_session()
        .expect("newcomer session id is available");
    let reused = pool
        .allocate_for_session(newcomer)
        .expect("LRU eviction frees a block for newcomer");

    assert!([first, second].contains(&reused));
    assert!(pool.session(owner).is_none());

    let evicted = pool.metrics_snapshot();
    assert_eq!(evicted.sessions, 2);
    assert_eq!(evicted.resident_blocks, 2);
    assert_eq!(evicted.active_blocks, 2);
    assert_eq!(evicted.blocks_evicted_lru, 2);
    assert_eq!(evicted.sessions_evicted_lru, 1);
    assert_eq!(evicted.eviction_high_water_mark, 3);
    assert_eq!(evicted.pool_high_water_blocks, 3);
}

#[test]
fn block_pool_metrics_reset_when_pool_is_recreated() {
    let pool = BlockPool::new(1, 2, 2).expect("pool shape is valid");
    let session = pool.create_session().expect("session id is available");

    pool.allocate_for_session(session)
        .expect("session can allocate");

    let active = pool.metrics_snapshot();
    assert_eq!(active.resident_blocks, 1);
    assert_eq!(active.pool_high_water_blocks, 1);

    let recreated = BlockPool::new(1, 2, 2).expect("pool shape is valid");
    let reset = recreated.metrics_snapshot();

    assert_eq!(reset.total_blocks, 1);
    assert_eq!(reset.resident_blocks, 0);
    assert_eq!(reset.active_blocks, 0);
    assert_eq!(reset.total_refcount_increments, 0);
    assert_eq!(reset.total_cow_clones, 0);
    assert_eq!(reset.blocks_evicted_lru, 0);
    assert_eq!(reset.pool_high_water_blocks, 0);
}

#[test]
fn block_pool_admin_snapshot_includes_session_block_tables_and_refcounts() {
    let pool = BlockPool::new(2, 2, 2).expect("pool shape is valid");
    let owner = pool
        .create_session()
        .expect("owner session id is available");
    let reader = pool
        .create_session()
        .expect("reader session id is available");
    let shared_block = pool
        .allocate_for_session(owner)
        .expect("owner can allocate");

    pool.append_block_to_session(reader, shared_block)
        .expect("reader can share owner block");

    let snapshot = pool.snapshot();

    assert_eq!(snapshot.metrics.total_blocks, 2);
    assert_eq!(snapshot.metrics.shared_blocks, 1);
    assert_eq!(snapshot.sessions.len(), 2);
    assert_eq!(snapshot.sessions[0].session_id, owner);
    assert_eq!(snapshot.sessions[0].block_table.len(), 1);
    assert_eq!(snapshot.sessions[0].block_table[0].index, 0);
    assert_eq!(
        snapshot.sessions[0].block_table[0].block_id,
        shared_block.as_u64()
    );
    assert_eq!(snapshot.sessions[0].block_table[0].ref_count, 2);
    assert!(
        snapshot.sessions[0]
            .owned_blocks
            .contains(&shared_block.as_u64())
    );
    assert_eq!(snapshot.sessions[1].session_id, reader);
    assert_eq!(
        snapshot.sessions[1].block_table[0].block_id,
        shared_block.as_u64()
    );
    assert_eq!(snapshot.sessions[1].block_table[0].ref_count, 2);
    assert_eq!(
        snapshot
            .blocks
            .iter()
            .find(|block| block.block_id == shared_block.as_u64())
            .expect("shared block is represented")
            .ref_count,
        2
    );
}
