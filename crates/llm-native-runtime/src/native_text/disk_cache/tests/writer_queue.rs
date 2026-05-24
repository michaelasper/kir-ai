use super::*;

#[test]
fn bounded_writer_backpressure_drops_without_blocking_generation() {
    let (writer, _rx) = NativeTextDiskCacheWriter::detached_for_test(1);
    let job = NativeTextDiskCacheWriteJob::for_test("a.safetensors", vec![1, 2, 3]);

    assert_eq!(
        writer.try_enqueue(job.clone()),
        NativeTextDiskCacheStoreStatus::Queued
    );
    let started = Instant::now();
    assert_eq!(
        writer.try_enqueue(job),
        NativeTextDiskCacheStoreStatus::Dropped
    );

    assert!(
        started.elapsed() < Duration::from_millis(25),
        "try_enqueue must not wait for writer capacity"
    );
}

#[test]
fn queue_store_drops_before_encoding_when_writer_queue_is_full() {
    static ENCODE_CALLS: AtomicUsize = AtomicUsize::new(0);

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct EncodingProbeCache {
        marker: u32,
    }

    impl NativeTextPrefixCacheValue for EncodingProbeCache {
        type PrefixCacheState = Self;

        fn prefix_cache_state(caches: &[Self]) -> Vec<Self::PrefixCacheState> {
            caches.to_vec()
        }

        fn prefix_cache_from_state(states: &[Self::PrefixCacheState]) -> Option<Vec<Self>> {
            Some(states.to_vec())
        }

        fn prefix_cache_entry_bytes(hidden: &[f32], states: &[Self::PrefixCacheState]) -> u64 {
            std::mem::size_of_val(hidden) as u64
                + states.len() as u64 * std::mem::size_of::<Self>() as u64
        }
    }

    impl NativeTextDiskCacheValue for EncodingProbeCache {
        fn encode_disk_block_states(
            states: &[Self::PrefixCacheState],
            block_start: usize,
            block_token_count: usize,
            sink: &mut NativeTextDiskCacheTensorSink,
        ) -> Result<Vec<NativeTextDiskCacheLayerLayout>, NativeTextDiskCacheError> {
            ENCODE_CALLS.fetch_add(1, Ordering::SeqCst);
            let values = states[block_start..block_start + block_token_count]
                .iter()
                .map(|state| state.marker as f32)
                .collect::<Vec<_>>();
            sink.push_f32("probe.markers", vec![values.len()], values)?;
            Ok(vec![NativeTextDiskCacheLayerLayout::test_marker_tensor(
                "probe.markers",
            )])
        }

        fn decode_disk_states(
            _layouts: &[NativeTextDiskCacheLayerLayout],
            _archive: &NativeTextDiskCacheTensorArchive<'_>,
        ) -> Result<Vec<Self::PrefixCacheState>, NativeTextDiskCacheError> {
            Ok(Vec::new())
        }

        fn assemble_disk_block_states(
            blocks: &[NativeTextDiskCacheStateBlock<Self::PrefixCacheState>],
        ) -> Result<Vec<Self::PrefixCacheState>, NativeTextDiskCacheError> {
            Ok(blocks
                .iter()
                .flat_map(|block| block.states.iter().cloned())
                .collect())
        }
    }

    ENCODE_CALLS.store(0, Ordering::SeqCst);
    let (writer, _rx) = NativeTextDiskCacheWriter::detached_for_test(1);
    let cache = NativeTextDiskCache::<EncodingProbeCache> {
        config: NativeTextDiskCacheConfig::for_root("unused").with_block_token_count(2),
        identity: NativeTextDiskCacheIdentity::for_test("model", "test"),
        index: NativeTextDiskCacheIndex::default(),
        writer,
        _cache: PhantomData,
    };
    let namespace = namespace("encoding-probe", "test");
    let hidden = [1.0, 2.0];
    let states = vec![
        EncodingProbeCache { marker: 1 },
        EncodingProbeCache { marker: 2 },
        EncodingProbeCache { marker: 3 },
        EncodingProbeCache { marker: 4 },
    ];

    assert_eq!(
        cache.queue_store(&namespace, &[1, 2], &hidden, &states[..2]),
        NativeTextDiskCacheStoreStatus::Queued
    );
    assert_eq!(ENCODE_CALLS.load(Ordering::SeqCst), 1);

    assert_eq!(
        cache.queue_store(&namespace, &[1, 2, 3, 4], &hidden, &states),
        NativeTextDiskCacheStoreStatus::Dropped
    );
    assert_eq!(
        ENCODE_CALLS.load(Ordering::SeqCst),
        1,
        "full queue must be detected before disk payload encoding runs"
    );
}
