use super::*;

#[test]
fn cache_token_capacity_uses_exact_budget_within_position_limit() {
    let capacity = native_text_cache_token_capacity(40, 8, 32, 64, "Test")
        .expect("context and generation budget fits");

    assert_eq!(capacity, 48);
}

#[test]
fn cache_namespace_token_bucket_keeps_prefix_identity_stable() {
    let capacity = native_text_cache_token_capacity(40, 8, 32, 64, "Test")
        .expect("context and generation budget fits");
    let bucket = native_text_cache_namespace_token_bucket(capacity, 64, "Test")
        .expect("namespace bucket fits");

    assert_eq!(capacity, 48);
    assert_eq!(bucket, 64);
}

#[test]
fn cache_token_capacity_rejects_invalid_position_limits() {
    let err = native_text_cache_token_capacity(0, 1, 1, 0, "Test")
        .expect_err("zero position limit fails closed");

    assert_eq!(
        err.backend_failure_class(),
        Some(BackendFailureClass::Config)
    );
    assert_eq!(err.backend_failure_code(), Some("backend_config_failed"));
    assert!(
        err.to_string()
            .contains("native Test model declares zero max_position_embeddings"),
        "error should identify the invalid model position limit: {err}"
    );
}

#[test]
fn prefix_cache_reuses_longest_namespace_compatible_prefix() {
    let cache = NativeTextPrefixCache::new(1024);
    let metrics = NativeTextPrefixCacheMetrics::default();
    let namespace = namespace("base");
    let caches = vec![TestCache {
        bytes: 11,
        marker: 7,
    }];

    cache.store(namespace.clone(), &[1, 2], &[0.5, 1.5], &caches, &metrics);

    let hit = cache
        .lookup(&namespace, &[1, 2, 3], &metrics)
        .expect("longer prompt reuses compatible prefix");
    assert_eq!(hit.token_count, 2);
    assert_eq!(hit.hidden, vec![0.5, 1.5]);
    assert_eq!(hit.caches, caches);

    let incompatible = NativeTextPrefixCacheNamespace {
        cache_key: "different".to_owned(),
        ..namespace
    };
    assert!(cache.lookup(&incompatible, &[1, 2], &metrics).is_none());
}

#[test]
fn prefix_cache_lookup_skips_capacity_incompatible_entries() {
    let cache = NativeTextPrefixCache::new(1024);
    let metrics = NativeTextPrefixCacheMetrics::default();
    let namespace = namespace("capacity");
    let caches = vec![TestCache {
        bytes: 11,
        marker: 7,
    }];

    cache.store(namespace.clone(), &[1, 2], &[0.5, 1.5], &caches, &metrics);

    assert!(
        cache
            .lookup_compatible(&namespace, &[1, 2, 3], &metrics, |caches| {
                caches.iter().all(|cache| cache.marker != 7)
            })
            .is_none()
    );

    let hit = cache
        .lookup_compatible(&namespace, &[1, 2, 3], &metrics, |caches| {
            caches.iter().all(|cache| cache.marker == 7)
        })
        .expect("compatible entry is reusable");
    assert_eq!(hit.token_count, 2);
}

#[test]
fn prefix_cache_reuses_longest_compatible_prefix_across_branching_index_paths() {
    let cache = NativeTextPrefixCache::new(2048);
    let metrics = NativeTextPrefixCacheMetrics::default();
    let namespace = namespace("branching");
    let hidden = [0.5, 1.5];

    cache.store(
        namespace.clone(),
        &[1, 2],
        &hidden,
        &[TestCache {
            bytes: 8,
            marker: 2,
        }],
        &metrics,
    );
    cache.store(
        namespace.clone(),
        &[1, 2, 3],
        &hidden,
        &[TestCache {
            bytes: 8,
            marker: 3,
        }],
        &metrics,
    );
    cache.store(
        namespace.clone(),
        &[1, 4, 5],
        &hidden,
        &[TestCache {
            bytes: 8,
            marker: 45,
        }],
        &metrics,
    );

    let fallback = cache
        .lookup_compatible(&namespace, &[1, 2, 3, 9], &metrics, |caches| {
            caches.iter().all(|cache| cache.marker != 3)
        })
        .expect("shorter compatible prefix is reused when longest branch is incompatible");
    assert_eq!(fallback.token_count, 2);
    assert_eq!(fallback.caches[0].marker, 2);

    let branch_hit = cache
        .lookup(&namespace, &[1, 4, 5, 6], &metrics)
        .expect("lookup follows the requested branch");
    assert_eq!(branch_hit.token_count, 3);
    assert_eq!(branch_hit.caches[0].marker, 45);
}

#[test]
fn prefix_cache_lookup_metrics_count_indexed_candidates_not_full_namespace() {
    let cache = NativeTextPrefixCache::new(1_000_000);
    let metrics = NativeTextPrefixCacheMetrics::default();
    let namespace = namespace("indexed-metrics");
    let hidden = [1.0];
    let caches = [TestCache {
        bytes: 1,
        marker: 1,
    }];

    cache.store(namespace.clone(), &[1, 2, 3], &hidden, &caches, &metrics);
    for index in 0..128 {
        let tokens = [10_000 + index];
        cache.store(namespace.clone(), &tokens, &hidden, &caches, &metrics);
    }

    let hit = cache
        .lookup(&namespace, &[1, 2, 3, 4], &metrics)
        .expect("matching prompt reuses indexed prefix");
    assert_eq!(hit.token_count, 3);

    let snapshot = metrics.snapshot();
    assert_eq!(
        snapshot["entries_scanned"], 1,
        "lookup should inspect only the terminal candidate on the query path"
    );
    assert_eq!(
        snapshot["namespace_entries_scanned"], 1,
        "lookup should not scan all unrelated namespace entries"
    );
}

#[test]
fn prefix_cache_lookup_skips_same_length_nonmatching_entries() {
    let cache = NativeTextPrefixCache::new(1_000_000);
    let metrics = NativeTextPrefixCacheMetrics::default();
    let namespace = namespace("same-length-indexed");
    let hidden = [1.0];
    let caches = [TestCache {
        bytes: 1,
        marker: 7,
    }];

    cache.store(namespace.clone(), &[1, 2, 3], &hidden, &caches, &metrics);
    for index in 0..128 {
        let tokens = [10_000 + index, 2, 3];
        cache.store(namespace.clone(), &tokens, &hidden, &caches, &metrics);
    }

    let mut compatibility_checks = 0;
    let hit = cache
        .lookup_compatible(&namespace, &[1, 2, 3, 4], &metrics, |caches| {
            compatibility_checks += 1;
            caches.iter().all(|cache| cache.marker == 7)
        })
        .expect("matching prefix is reused");

    assert_eq!(hit.token_count, 3);
    assert_eq!(
        compatibility_checks, 1,
        "lookup should not inspect unrelated entries with the same token length"
    );
    let snapshot = metrics.snapshot();
    assert_eq!(
        snapshot["entries_scanned"], 1,
        "lookup should only scan the exact candidate prefix"
    );
    assert_eq!(snapshot["namespace_entries_scanned"], 1);
}

#[test]
fn prefix_cache_replacement_keeps_lookup_index_single_candidate() {
    let cache = NativeTextPrefixCache::new(1024);
    let metrics = NativeTextPrefixCacheMetrics::default();
    let namespace = namespace("replace-index");
    let hidden = [1.0];

    cache.store(
        namespace.clone(),
        &[1, 2],
        &hidden,
        &[TestCache {
            bytes: 8,
            marker: 1,
        }],
        &metrics,
    );
    cache.store(
        namespace.clone(),
        &[1, 2],
        &hidden,
        &[TestCache {
            bytes: 8,
            marker: 2,
        }],
        &metrics,
    );

    let hit = cache
        .lookup(&namespace, &[1, 2, 3], &metrics)
        .expect("replacement entry is reusable");
    assert_eq!(hit.caches[0].marker, 2);

    let mut compatibility_checks = 0;
    assert!(
        cache
            .lookup_compatible(&namespace, &[1, 2, 3], &metrics, |_| {
                compatibility_checks += 1;
                false
            })
            .is_none(),
        "incompatible replacement entry should miss"
    );
    assert_eq!(
        compatibility_checks, 1,
        "replacement should not leave duplicate lookup index candidates"
    );
}

#[test]
fn prefix_cache_eviction_removes_lookup_index_candidate() {
    let cache = NativeTextPrefixCache::new(32);
    let metrics = NativeTextPrefixCacheMetrics::default();
    let namespace = namespace("evict-index");
    let hidden = vec![1.0; 4];

    cache.store(
        namespace.clone(),
        &[1, 2],
        &hidden,
        &[TestCache {
            bytes: 8,
            marker: 12,
        }],
        &metrics,
    );
    cache.store(
        namespace.clone(),
        &[9],
        &hidden,
        &[TestCache {
            bytes: 8,
            marker: 9,
        }],
        &metrics,
    );

    let before = metrics.snapshot();
    let mut compatibility_checks = 0;
    assert!(
        cache
            .lookup_compatible(&namespace, &[1, 2, 3], &metrics, |_| {
                compatibility_checks += 1;
                true
            })
            .is_none(),
        "evicted prefix should not be reusable"
    );
    let after = metrics.snapshot();
    assert_eq!(
        compatibility_checks, 0,
        "eviction should remove stale lookup index candidates"
    );
    assert_eq!(
        after["entries_scanned"].as_u64().unwrap_or(0)
            - before["entries_scanned"].as_u64().unwrap_or(0),
        0
    );
    assert!(
        cache.lookup(&namespace, &[9], &metrics).is_some(),
        "non-evicted entry remains indexed"
    );
}

#[test]
fn prefix_cache_stores_entries_in_namespace_buckets() {
    let cache = NativeTextPrefixCache::new(1024);
    let metrics = NativeTextPrefixCacheMetrics::default();
    let base_namespace = namespace("bucket");
    let other_namespace = namespace("other-bucket");
    let hidden = [1.0];
    let caches = [TestCache {
        bytes: 8,
        marker: 1,
    }];

    cache.store(base_namespace.clone(), &[1], &hidden, &caches, &metrics);
    cache.store(base_namespace.clone(), &[1, 2], &hidden, &caches, &metrics);
    cache.store(other_namespace.clone(), &[9], &hidden, &caches, &metrics);

    let inner = cache.inner.lock().expect("prefix cache lock is available");
    assert_eq!(inner.entries.len(), 2);
    assert_eq!(
        inner
            .entries
            .get(&base_namespace)
            .expect("namespace bucket exists")
            .len(),
        2
    );
    assert_eq!(
        inner
            .entries
            .get(&other_namespace)
            .expect("other namespace bucket exists")
            .len(),
        1
    );
}

#[test]
fn prefix_cache_prefers_longest_prefix_over_recency_and_updates_lru() {
    let cache = NativeTextPrefixCache::new(48);
    let metrics = NativeTextPrefixCacheMetrics::default();
    let base_namespace = namespace("longest-lru");
    let other_namespace = namespace("longest-lru-other");
    let hidden = [1.0, 2.0, 3.0, 4.0];

    cache.store(
        base_namespace.clone(),
        &[1, 2, 3],
        &hidden,
        &[TestCache {
            bytes: 8,
            marker: 3,
        }],
        &metrics,
    );
    cache.store(
        base_namespace.clone(),
        &[1, 2],
        &hidden,
        &[TestCache {
            bytes: 8,
            marker: 2,
        }],
        &metrics,
    );

    let hit = cache
        .lookup(&base_namespace, &[1, 2, 3, 4], &metrics)
        .expect("matching prompt reuses longest prefix");
    assert_eq!(hit.token_count, 3);
    assert_eq!(hit.caches[0].marker, 3);

    cache.store(
        other_namespace.clone(),
        &[9],
        &hidden,
        &[TestCache {
            bytes: 8,
            marker: 9,
        }],
        &metrics,
    );

    assert!(
        cache.lookup(&base_namespace, &[1, 2], &metrics).is_none(),
        "shorter prefix should be least recently used after the longest-prefix hit"
    );
    assert!(
        cache
            .lookup(&base_namespace, &[1, 2, 3], &metrics)
            .is_some(),
        "longest-prefix hit should refresh that entry before eviction"
    );
    assert!(cache.lookup(&other_namespace, &[9], &metrics).is_some());
}

#[test]
fn prefix_cache_clones_payloads_outside_global_lock() {
    let cache = Arc::new(NativeTextPrefixCache::new(1024));
    let metrics = NativeTextPrefixCacheMetrics::default();
    let namespace = namespace("clone-lock");
    let cloned_while_locked = Arc::new(AtomicUsize::new(0));
    let caches = vec![LockObservingCache {
        bytes: 8,
        cache: Arc::downgrade(&cache),
        cloned_while_locked: cloned_while_locked.clone(),
    }];

    cache.store(namespace.clone(), &[1, 2], &[0.5, 1.5], &caches, &metrics);
    let hit = cache
        .lookup(&namespace, &[1, 2, 3], &metrics)
        .expect("compatible longer prompt reuses stored prefix");

    assert_eq!(hit.token_count, 2);
    assert_eq!(
        cloned_while_locked.load(Ordering::SeqCst),
        0,
        "prefix cache must not clone layer-cache payloads while holding its global lock"
    );
}

#[test]
fn prefix_cache_metrics_record_lookup_scans_and_hit_clone_bytes() {
    let cache = NativeTextPrefixCache::new(1024);
    let metrics = NativeTextPrefixCacheMetrics::default();
    let base_namespace = namespace("scan-metrics");
    let other_namespace = namespace("scan-metrics-other");
    let hidden = [1.0, 2.0, 3.0, 4.0];

    cache.store(
        base_namespace.clone(),
        &[1],
        &hidden,
        &[TestCache {
            bytes: 5,
            marker: 1,
        }],
        &metrics,
    );
    cache.store(
        base_namespace.clone(),
        &[1, 2],
        &hidden,
        &[TestCache {
            bytes: 7,
            marker: 2,
        }],
        &metrics,
    );
    cache.store(
        other_namespace,
        &[9],
        &hidden,
        &[TestCache {
            bytes: 11,
            marker: 9,
        }],
        &metrics,
    );

    let hit = cache
        .lookup(&base_namespace, &[1, 2, 3], &metrics)
        .expect("matching prompt reuses longest stored prefix");
    assert_eq!(hit.token_count, 2);
    assert!(cache.lookup(&base_namespace, &[42], &metrics).is_none());

    let snapshot = metrics.snapshot();
    assert_eq!(
        snapshot["entries_scanned"], 1,
        "lookups only scan terminal candidates on the matching index path"
    );
    assert_eq!(snapshot["namespace_entries_scanned"], 1);
    assert_eq!(
        snapshot["hit_clone_bytes"],
        std::mem::size_of_val(&hidden) as u64 + 7
    );
}

#[test]
fn prefix_cache_uses_value_sizing_for_eviction_budget() {
    let cache = NativeTextPrefixCache::new(32);
    let metrics = NativeTextPrefixCacheMetrics::default();
    let namespace = namespace("budget");
    let hidden = vec![1.0; 4];

    cache.store(
        namespace.clone(),
        &[1],
        &hidden,
        &[TestCache {
            bytes: 8,
            marker: 1,
        }],
        &metrics,
    );
    cache.store(
        namespace.clone(),
        &[2],
        &hidden,
        &[TestCache {
            bytes: 8,
            marker: 2,
        }],
        &metrics,
    );

    assert!(cache.lookup(&namespace, &[1], &metrics).is_none());
    assert!(cache.lookup(&namespace, &[2], &metrics).is_some());
    let snapshot = metrics.snapshot();
    assert_eq!(snapshot["evictions"], 1);
    assert_eq!(snapshot["resident_bytes"], 24);
}
