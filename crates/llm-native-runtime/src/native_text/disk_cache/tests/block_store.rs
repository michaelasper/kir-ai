use super::*;

#[tokio::test]
async fn startup_reindex_handles_nested_block_dirs_and_stale_files() {
    let temp = tempfile::tempdir().expect("temp dir exists");
    let config = NativeTextDiskCacheConfig::for_root(temp.path()).with_block_token_count(2);
    let namespace = namespace("nested", "gemma");
    let identity = NativeTextDiskCacheIdentity::from_namespace(&namespace, "gemma");
    let descriptor = NativeTextDiskCacheBlockDescriptor::new(&identity, &namespace, 0, &[21, 22])
        .expect("descriptor builds");
    let states = vec![GemmaLayerCache::Attention(filled_layer_cache(4)).prefix_cache_state()];
    let hidden = vec![0.25, 0.5];
    let bytes = NativeTextDiskCacheBlock::<GemmaLayerCache>::encode(&descriptor, &hidden, &states)
        .expect("block encodes");
    let nested = temp
        .path()
        .join(identity.model_hash())
        .join(descriptor.namespace_hash())
        .join("aa");
    std::fs::create_dir_all(&nested).expect("nested dirs create");
    std::fs::write(
        nested.join(format!("{}.safetensors", descriptor.block_hash())),
        bytes,
    )
    .expect("nested block writes");
    std::fs::write(nested.join("README.txt"), b"stale").expect("stale file writes");

    let cache = NativeTextDiskCache::<GemmaLayerCache>::open(config, identity)
        .await
        .expect("nested reindex succeeds");

    assert_eq!(cache.indexed_entry_count_for_test(), 1);
}

#[tokio::test]
async fn block_store_writes_only_terminal_block_payload_not_accumulated_prefix() {
    let temp = tempfile::tempdir().expect("temp dir exists");
    let config = NativeTextDiskCacheConfig::for_root(temp.path())
        .with_writer_queue_depth(4)
        .with_block_token_count(2);
    let namespace = namespace("block-payload", "test");
    let identity = NativeTextDiskCacheIdentity::from_namespace(&namespace, "test");
    let disk = NativeTextDiskCache::<DummyCache>::open(config, identity.clone())
        .await
        .expect("cache opens");
    let first_hidden = vec![1.0];
    let second_hidden = vec![2.0];
    let first_states = vec![DummyCache { marker: 1 }, DummyCache { marker: 2 }];
    let second_states = vec![
        DummyCache { marker: 1 },
        DummyCache { marker: 2 },
        DummyCache { marker: 3 },
        DummyCache { marker: 4 },
    ];

    assert_eq!(
        disk.queue_store(&namespace, &[31, 32], &first_hidden, &first_states),
        NativeTextDiskCacheStoreStatus::Queued
    );
    assert_eq!(
        disk.queue_store(
            &namespace,
            &[31, 32, 33, 34],
            &second_hidden,
            &second_states
        ),
        NativeTextDiskCacheStoreStatus::Queued
    );
    disk.flush_for_test().await.expect("queued writes flush");

    let first_descriptor =
        NativeTextDiskCacheBlockDescriptor::new(&identity, &namespace, 0, &[31, 32])
            .expect("first descriptor builds");
    let second_descriptor =
        NativeTextDiskCacheBlockDescriptor::new(&identity, &namespace, 2, &[31, 32, 33, 34])
            .expect("second descriptor builds");
    let first_bytes = std::fs::read(disk.path_for_descriptor_for_test(&first_descriptor))
        .expect("first block exists");
    let second_bytes = std::fs::read(disk.path_for_descriptor_for_test(&second_descriptor))
        .expect("second block exists");
    let first_block =
        NativeTextDiskCacheBlock::<DummyCache>::decode(&first_bytes, &identity, &first_descriptor)
            .expect("first block decodes");
    let second_block = NativeTextDiskCacheBlock::<DummyCache>::decode(
        &second_bytes,
        &identity,
        &second_descriptor,
    )
    .expect("second block decodes");

    assert_eq!(first_block.block_start, 0);
    assert_eq!(first_block.token_count, 2);
    assert_eq!(first_block.states, first_states);
    assert_eq!(second_block.block_start, 2);
    assert_eq!(second_block.token_count, 2);
    assert_eq!(
        second_block.states,
        vec![DummyCache { marker: 3 }, DummyCache { marker: 4 }],
        "later block files must not duplicate the earlier prefix payload"
    );
}
