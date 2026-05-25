use super::*;

#[test]
fn snapshot_codec_round_trips_qwen_full_qwen_linear_and_gemma_attention_blocks() {
    let qwen_full = QwenLayerCache::Full(filled_layer_cache(4)).prefix_cache_state();
    let qwen_linear = QwenLayerCache::Linear(filled_linear_cache()).prefix_cache_state();
    let gemma_attention = GemmaLayerCache::Attention(filled_layer_cache(4)).prefix_cache_state();

    round_trip::<QwenLayerCache>("qwen", vec![qwen_full, qwen_linear]);
    round_trip::<GemmaLayerCache>("gemma", vec![gemma_attention]);
}

#[test]
fn test_identity_snapshot_hash_is_derived_from_namespace_fixture_context() {
    let first_namespace = namespace("first-snapshot", "qwen");
    let second_namespace = namespace("second-snapshot", "qwen");

    let first_identity = NativeTextDiskCacheIdentity::from_namespace(&first_namespace, "qwen");
    let second_identity = NativeTextDiskCacheIdentity::from_namespace(&second_namespace, "qwen");

    assert_ne!(
        first_identity.snapshot_hash(),
        second_identity.snapshot_hash()
    );
    assert_ne!(first_identity.model_hash(), second_identity.model_hash());
}

#[tokio::test]
async fn startup_reindex_ignores_wrong_namespace_model_layout_version_and_corrupt_files() {
    let temp = tempfile::tempdir().expect("temp dir exists");
    let config = NativeTextDiskCacheConfig::for_root(temp.path())
        .with_writer_queue_depth(4)
        .with_block_token_count(2);
    let valid_namespace = namespace("valid", "qwen");
    let identity = NativeTextDiskCacheIdentity::from_namespace(&valid_namespace, "qwen");
    let hidden = vec![0.25, 0.5];
    let states = vec![QwenLayerCache::Full(filled_layer_cache(4)).prefix_cache_state()];

    let cache = NativeTextDiskCache::<QwenLayerCache>::open(config.clone(), identity.clone())
        .await
        .expect("cache opens");
    assert_eq!(
        cache.queue_store(&valid_namespace, &[11, 12], &hidden, &states),
        NativeTextDiskCacheStoreStatus::Queued
    );
    cache.flush_for_test().await.expect("queued write flushes");

    let valid_descriptor =
        NativeTextDiskCacheBlockDescriptor::new(&identity, &valid_namespace, 0, &[11, 12])
            .expect("valid descriptor builds");
    let valid_bytes =
        NativeTextDiskCacheBlock::<QwenLayerCache>::encode(&valid_descriptor, &hidden, &states)
            .expect("valid block encodes");
    let wrong_namespace_value = namespace("wrong", "qwen");
    let wrong_namespace =
        NativeTextDiskCacheBlockDescriptor::new(&identity, &wrong_namespace_value, 0, &[11, 12])
            .expect("wrong namespace descriptor builds");
    std::fs::write(
        cache.path_for_descriptor_for_test(&wrong_namespace),
        NativeTextDiskCacheBlock::<QwenLayerCache>::encode(&wrong_namespace, &hidden, &states)
            .expect("wrong namespace block encodes"),
    )
    .expect("wrong namespace file writes");
    let mut wrong_model_namespace = valid_namespace.clone();
    wrong_model_namespace.model_id = "wrong-model".to_owned();
    let wrong_model = NativeTextDiskCacheIdentity::from_namespace(&wrong_model_namespace, "qwen");
    let wrong_model_descriptor =
        NativeTextDiskCacheBlockDescriptor::new(&wrong_model, &valid_namespace, 0, &[11, 12])
            .expect("wrong model descriptor builds");
    std::fs::write(
        cache.path_for_descriptor_for_test(&wrong_model_descriptor),
        NativeTextDiskCacheBlock::<QwenLayerCache>::encode(
            &wrong_model_descriptor,
            &hidden,
            &states,
        )
        .expect("wrong model block encodes"),
    )
    .expect("wrong model file writes");
    let mut wrong_version = valid_bytes;
    NativeTextDiskCacheBlock::<QwenLayerCache>::rewrite_layout_version_for_test(
        &mut wrong_version,
        99,
    )
    .expect("layout metadata is rewritten");
    std::fs::write(
        cache
            .path_for_descriptor_for_test(&valid_descriptor)
            .with_file_name("wrong-version.safetensors"),
        wrong_version,
    )
    .expect("wrong version file writes");
    let corrupt_dir = cache
        .root_for_test()
        .join(identity.model_hash())
        .join("stale");
    std::fs::create_dir_all(&corrupt_dir).expect("corrupt dir creates");
    std::fs::write(
        corrupt_dir.join("corrupt.safetensors"),
        b"not a safetensors file",
    )
    .expect("corrupt file writes");
    drop(cache);

    let reindexed = NativeTextDiskCache::<QwenLayerCache>::open(config, identity)
        .await
        .expect("corrupt or mismatched files do not fail startup");

    assert_eq!(reindexed.indexed_entry_count_for_test(), 1);
}

#[tokio::test]
async fn snapshot_identity_partitions_model_hash_and_rejects_wrong_snapshot_metadata() {
    let temp = tempfile::tempdir().expect("temp dir exists");
    let config = NativeTextDiskCacheConfig::for_root(temp.path()).with_block_token_count(2);
    let mut metadata = BackendModelMetadata::new("model-shared", "native-test").with_family("qwen");
    metadata.repo_id = Some("org/model".to_owned());
    metadata.resolved_commit = Some("abc123".to_owned());
    metadata.profile = Some("default".to_owned());
    let first_identity = NativeTextDiskCacheIdentity::from_model_metadata(
        &metadata,
        "qwen",
        Some("manifest:sha256:first"),
    );
    let second_identity = NativeTextDiskCacheIdentity::from_model_metadata(
        &metadata,
        "qwen",
        Some("manifest:sha256:second"),
    );
    let namespace = namespace("snapshot", "qwen");
    let hidden = vec![0.25, 0.5];
    let states = vec![QwenLayerCache::Full(filled_layer_cache(4)).prefix_cache_state()];

    assert_ne!(first_identity.model_hash(), second_identity.model_hash());
    assert_ne!(
        first_identity.snapshot_hash(),
        second_identity.snapshot_hash()
    );

    let first_cache =
        NativeTextDiskCache::<QwenLayerCache>::open(config.clone(), first_identity.clone())
            .await
            .expect("first snapshot cache opens");
    assert_eq!(
        first_cache.queue_store(&namespace, &[11, 12], &hidden, &states),
        NativeTextDiskCacheStoreStatus::Queued
    );
    first_cache
        .flush_for_test()
        .await
        .expect("first snapshot write flushes");

    let wrong_descriptor =
        NativeTextDiskCacheBlockDescriptor::new(&first_identity, &namespace, 0, &[11, 12])
            .expect("wrong snapshot descriptor builds");
    let wrong_bytes =
        NativeTextDiskCacheBlock::<QwenLayerCache>::encode(&wrong_descriptor, &hidden, &states)
            .expect("wrong snapshot block encodes");
    let second_descriptor =
        NativeTextDiskCacheBlockDescriptor::new(&second_identity, &namespace, 0, &[11, 12])
            .expect("second snapshot descriptor builds");
    let second_cache =
        NativeTextDiskCache::<QwenLayerCache>::open(config.clone(), second_identity.clone())
            .await
            .expect("second snapshot cache opens");
    std::fs::write(
        second_cache.path_for_descriptor_for_test(&second_descriptor),
        wrong_bytes,
    )
    .expect("wrong snapshot file writes under second model root");
    drop(second_cache);

    let reindexed = NativeTextDiskCache::<QwenLayerCache>::open(config, second_identity)
        .await
        .expect("wrong snapshot metadata does not fail startup");

    assert_eq!(reindexed.indexed_entry_count_for_test(), 0);
    assert!(
        reindexed
            .lookup(&namespace, &[11, 12, 13], |_| true)
            .await
            .expect("lookup succeeds")
            .is_none(),
        "a block encoded for another snapshot must be ignored"
    );
}

#[tokio::test]
async fn snapshot_identity_prefers_manifest_digest_without_filesystem_resolution() {
    let identity = native_text_disk_cache_snapshot_identity(
        std::path::Path::new("/snapshot/does/not/exist"),
        Some("sha256:abc123"),
    )
    .await;

    assert_eq!(identity, "manifest:sha256:abc123");
}

#[tokio::test]
async fn snapshot_identity_uses_async_canonical_raw_path_without_manifest_digest() {
    let temp = tempfile::tempdir().expect("temp dir exists");
    let canonical = temp.path().canonicalize().expect("temp dir canonicalizes");

    let identity = native_text_disk_cache_snapshot_identity(temp.path(), None).await;

    assert_eq!(identity, format!("raw-path:{}", canonical.display()));
}
