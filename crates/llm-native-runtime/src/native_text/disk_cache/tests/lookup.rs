use super::*;

#[tokio::test]
async fn lookup_assembles_prefix_from_multiple_independent_block_entries() {
    let temp = tempfile::tempdir().expect("temp dir exists");
    let config = NativeTextDiskCacheConfig::for_root(temp.path()).with_block_token_count(2);
    let namespace = namespace("assembled", "test");
    let identity = NativeTextDiskCacheIdentity::from_namespace(&namespace, "test");
    let disk = NativeTextDiskCache::<DummyCache>::open(config.clone(), identity.clone())
        .await
        .expect("cache opens");
    let first_descriptor =
        NativeTextDiskCacheBlockDescriptor::new(&identity, &namespace, 0, &[41, 42])
            .expect("first descriptor builds");
    let second_descriptor =
        NativeTextDiskCacheBlockDescriptor::new(&identity, &namespace, 2, &[41, 42, 43, 44])
            .expect("second descriptor builds");
    std::fs::write(
        disk.path_for_descriptor_for_test(&first_descriptor),
        NativeTextDiskCacheBlock::<DummyCache>::encode(
            &first_descriptor,
            &[1.0],
            &[DummyCache { marker: 1 }, DummyCache { marker: 2 }],
        )
        .expect("first block encodes"),
    )
    .expect("first block writes");
    std::fs::write(
        disk.path_for_descriptor_for_test(&second_descriptor),
        NativeTextDiskCacheBlock::<DummyCache>::encode(
            &second_descriptor,
            &[2.0],
            &[
                DummyCache { marker: 1 },
                DummyCache { marker: 2 },
                DummyCache { marker: 3 },
                DummyCache { marker: 4 },
            ],
        )
        .expect("second block encodes"),
    )
    .expect("second block writes");
    drop(disk);

    let reindexed = NativeTextDiskCache::<DummyCache>::open(config, identity)
        .await
        .expect("cache reindexes independent blocks");
    let hit = reindexed
        .lookup(&namespace, &[41, 42, 43, 44, 45], |_| true)
        .await
        .expect("lookup succeeds")
        .expect("assembled prefix hit exists");

    assert_eq!(hit.token_count, 4);
    assert_eq!(hit.hidden, vec![2.0]);
    assert_eq!(
        hit.caches,
        vec![
            DummyCache { marker: 1 },
            DummyCache { marker: 2 },
            DummyCache { marker: 3 },
            DummyCache { marker: 4 },
        ]
    );
}

#[tokio::test]
async fn lookup_does_not_reuse_later_block_from_different_prefix_context() {
    let temp = tempfile::tempdir().expect("temp dir exists");
    let config = NativeTextDiskCacheConfig::for_root(temp.path())
        .with_writer_queue_depth(4)
        .with_block_token_count(2);
    let namespace = namespace("cross-prefix", "test");
    let identity = NativeTextDiskCacheIdentity::from_namespace(&namespace, "test");
    let disk = NativeTextDiskCache::<DummyCache>::open(config, identity)
        .await
        .expect("cache opens");

    assert_eq!(
        disk.queue_store(
            &namespace,
            &[1, 2, 3, 4],
            &[4.0],
            &[
                DummyCache { marker: 1 },
                DummyCache { marker: 2 },
                DummyCache { marker: 3 },
                DummyCache { marker: 4 },
            ],
        ),
        NativeTextDiskCacheStoreStatus::Queued
    );
    assert_eq!(
        disk.queue_store(
            &namespace,
            &[9, 9],
            &[2.0],
            &[DummyCache { marker: 9 }, DummyCache { marker: 9 }],
        ),
        NativeTextDiskCacheStoreStatus::Queued
    );
    disk.flush_for_test().await.expect("queued writes flush");

    let hit = disk
        .lookup(&namespace, &[9, 9, 3, 4, 5], |_| true)
        .await
        .expect("lookup succeeds")
        .expect("first prefix block still hits");

    assert_eq!(
        hit.token_count, 2,
        "lookup must not assemble prefix B with prefix A's contextual second block"
    );
    assert_eq!(hit.hidden, vec![2.0]);
    assert_eq!(
        hit.caches,
        vec![DummyCache { marker: 9 }, DummyCache { marker: 9 }]
    );
}

#[tokio::test]
async fn disk_hits_promote_validated_blocks_into_hot_prefix_cache() {
    let temp = tempfile::tempdir().expect("temp dir exists");
    let config = NativeTextDiskCacheConfig::for_root(temp.path()).with_block_token_count(2);
    let namespace = namespace("promote", "test");
    let identity = NativeTextDiskCacheIdentity::from_namespace(&namespace, "test");
    let disk = NativeTextDiskCache::<DummyCache>::open(config, identity)
        .await
        .expect("cache opens");
    let metrics = NativeTextPrefixCacheMetrics::default();
    let memory = NativeTextPrefixCache::<DummyCache>::new(1024);
    let hidden = vec![1.0, 2.0];
    let states = vec![DummyCache { marker: 41 }, DummyCache { marker: 42 }];

    assert_eq!(
        disk.queue_store(&namespace, &[31, 32], &hidden, &states),
        NativeTextDiskCacheStoreStatus::Queued
    );
    disk.flush_for_test().await.expect("queued write flushes");
    let hit = disk
        .lookup(&namespace, &[31, 32, 33], |_| true)
        .await
        .expect("disk lookup succeeds")
        .expect("disk prefix hit exists");

    disk.promote_hit(&memory, namespace.clone(), &[31, 32], &metrics, &hit);
    let promoted = memory
        .lookup(&namespace, &[31, 32, 33], &metrics)
        .expect("memory cache now has promoted disk hit");

    assert_eq!(promoted.token_count, 2);
    assert_eq!(promoted.hidden, hidden);
    assert_eq!(promoted.caches, states);
}
