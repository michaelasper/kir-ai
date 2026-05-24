use super::*;

#[test]
fn driver_reports_prefix_cache_miss_and_hit_for_blocking_generation() {
    let driver = driver_for_test(TestAdapter::new([1_usize]));

    let first = driver
        .generate_blocking(driver_test_request(1), CancellationToken::new())
        .expect("first generation succeeds");
    let second = driver
        .generate_blocking(driver_test_request(1), CancellationToken::new())
        .expect("second generation succeeds");

    assert_eq!(first.prompt_cached_tokens, Some(0));
    assert_eq!(second.prompt_cached_tokens, Some(1));
}

#[test]
fn driver_records_prefill_and_avoided_work_for_warm_prefix() {
    let adapter = TestAdapter::new([1_usize]).with_encoded_prompt([10_u32, 11, 12, 13, 14]);
    let metrics = Arc::clone(&adapter.prefix_cache_metrics);
    let driver = driver_for_test(adapter).with_max_prefill_tokens(2);

    let first = driver
        .generate_blocking(driver_test_request(1), CancellationToken::new())
        .expect("cold generation succeeds");
    let second = driver
        .generate_blocking(driver_test_request(1), CancellationToken::new())
        .expect("warm generation succeeds");

    assert_eq!(first.prompt_cached_tokens, Some(0));
    assert_eq!(second.prompt_cached_tokens, Some(5));
    let snapshot = metrics.snapshot();
    assert_eq!(snapshot["prefill_chunks"], 3);
    assert_eq!(snapshot["prefill_tokens"], 5);
    assert_eq!(snapshot["hit_tokens"], 5);
    assert_eq!(snapshot["miss_tokens"], 5);
    assert_eq!(snapshot["avoided_prefill_tokens"], 5);
}

#[test]
fn driver_records_shared_prefix_reuse_without_exposing_state() {
    let request = driver_test_request(1);
    let adapter = TestAdapter::new([1_usize]).with_encoded_prompt([10_u32, 11, 12, 13]);
    let metrics = Arc::clone(&adapter.prefix_cache_metrics);
    store_driver_prefix_hit(
        &adapter,
        &request,
        4,
        1,
        &[10, 11],
        TestCache {
            bytes: 8,
            marker: 77,
        },
    );
    let driver = driver_for_test(adapter).with_max_prefill_tokens(2);

    let output = driver
        .generate_blocking(request, CancellationToken::new())
        .expect("generation reuses compatible shared prefix");

    assert_eq!(output.prompt_cached_tokens, Some(2));
    let snapshot = metrics.snapshot();
    assert_eq!(snapshot["shared_prefix_hits"], 1);
    assert_eq!(snapshot["shared_prefix_reused_tokens"], 2);
    assert_eq!(snapshot.get("shared_prefix_states"), None);
}

#[test]
fn driver_reports_prefix_cache_miss_and_hit_for_streaming_generation() {
    let driver = driver_for_test(TestAdapter::new([1_usize]));

    let first = stream_final_chunk(&driver, driver_test_request(1));
    let second = stream_final_chunk(&driver, driver_test_request(1));

    assert_eq!(first.prompt_cached_tokens, Some(0));
    assert_eq!(second.prompt_cached_tokens, Some(1));
}

#[test]
fn streaming_generation_emits_prefill_progress_after_each_uncached_chunk() {
    let driver =
        driver_for_test(TestAdapter::new([1_usize]).with_encoded_prompt([10_u32, 11, 12, 13, 14]))
            .with_max_prefill_tokens(2);
    let (tx, mut rx) = tokio::sync::mpsc::channel(8);

    driver
        .generate_blocking_stream(driver_test_request(1), tx, CancellationToken::new())
        .expect("streaming generation succeeds");

    let mut progress = Vec::new();
    while let Some(chunk) = rx.blocking_recv() {
        let chunk = chunk.expect("stream chunk is ok");
        if let Some(event) = chunk.progress {
            progress.push(event);
        }
    }

    assert_eq!(
        progress,
        vec![
            BackendStreamProgress::PrefillProgress {
                chunk: 1,
                total: 3,
                tokens: 2,
                total_tokens: 5,
            },
            BackendStreamProgress::PrefillProgress {
                chunk: 2,
                total: 3,
                tokens: 4,
                total_tokens: 5,
            },
            BackendStreamProgress::PrefillProgress {
                chunk: 3,
                total: 3,
                tokens: 5,
                total_tokens: 5,
            },
        ]
    );
}

#[test]
fn streaming_generation_waits_for_prefill_admission_before_next_uncached_chunk() {
    let adapter = TestAdapter::new([1_usize]).with_encoded_prompt([10_u32, 11, 12, 13]);
    let prefill_chunk_calls = adapter.prefill_chunk_calls();
    let driver = driver_for_test(adapter).with_max_prefill_tokens(2);
    let admission = Arc::new(BlockingPrefillAdmission::new());
    let request = driver_test_request(1).with_prefill_chunk_admission(
        BackendPrefillChunkAdmissionHook::new(Arc::clone(&admission)),
    );
    let (tx, mut rx) = tokio::sync::mpsc::channel(8);
    let worker = std::thread::spawn({
        let driver = driver.clone();
        move || {
            driver
                .block_on_worker(driver.generate_stream_async(
                    request,
                    tx,
                    CancellationToken::new(),
                ))
                .expect("native stream worker runtime succeeds")
        }
    });

    let first = rx
        .blocking_recv()
        .expect("first prefill progress arrives")
        .expect("first prefill progress succeeds");
    assert_eq!(
        first.progress,
        Some(BackendStreamProgress::PrefillProgress {
            chunk: 1,
            total: 2,
            tokens: 2,
            total_tokens: 4,
        })
    );
    let deadline = Instant::now() + Duration::from_millis(500);
    while admission.calls.load(Ordering::SeqCst) == 0 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(admission.calls.load(Ordering::SeqCst), 1);
    assert_eq!(prefill_chunk_calls.load(Ordering::SeqCst), 1);

    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(
        prefill_chunk_calls.load(Ordering::SeqCst),
        1,
        "native worker must not start the next prefill chunk before admission"
    );

    admission.release.notify_waiters();
    let mut saw_final = false;
    while let Some(chunk) = rx.blocking_recv() {
        let chunk = chunk.expect("stream chunk succeeds after readmission");
        if chunk.finish_reason.is_some() {
            saw_final = true;
        }
    }
    worker
        .join()
        .expect("native stream worker joins")
        .expect("native stream generation succeeds");
    assert!(saw_final);
    assert_eq!(prefill_chunk_calls.load(Ordering::SeqCst), 2);
    assert_eq!(admission.calls.load(Ordering::SeqCst), 1);
}

#[test]
fn driver_reuses_mid_prefill_checkpoint_after_cancellation() {
    let request = driver_test_request(1);
    let cancellation = CancellationToken::new();
    let adapter = TestAdapter::new([1_usize])
        .with_encoded_prompt([10_u32, 11, 12, 13, 14])
        .with_prefill_cancellation_after_chunk(cancellation.clone(), 2);
    let prefix_cache = Arc::clone(&adapter.prefix_cache);
    let metrics = Arc::clone(&adapter.prefix_cache_metrics);
    let namespace = driver_prefix_namespace(&adapter, &request, 5, 1);
    let driver = driver_for_test(adapter).with_max_prefill_tokens(2);

    let err = driver
        .generate_blocking(request.clone(), cancellation)
        .expect_err("prefill cancellation is returned");
    assert!(err.is_cancelled());
    assert_prefix_cache_entry(&prefix_cache, &namespace, &[10, 11]);
    assert_prefix_cache_entry(&prefix_cache, &namespace, &[10, 11, 12, 13]);
    assert_no_prefix_cache_entry(&prefix_cache, &namespace, &[10, 11, 12, 13, 14]);

    let warm = driver
        .generate_blocking(request, CancellationToken::new())
        .expect("warm generation reuses checkpoint and completes suffix");

    assert_eq!(warm.prompt_cached_tokens, Some(4));
    assert_eq!(warm.text, "<1>");
    let snapshot = metrics.snapshot();
    assert_eq!(snapshot["checkpoint_stores"], 2);
    assert_eq!(snapshot["checkpoint_store_tokens"], 6);
    assert_eq!(snapshot["checkpoint_reuse_hits"], 1);
    assert_eq!(snapshot["checkpoint_reused_tokens"], 4);
}

#[test]
fn driver_does_not_checkpoint_failed_prefill_chunk() {
    let request = driver_test_request(1);
    let adapter = TestAdapter::new([1_usize])
        .with_encoded_prompt([30_u32, 31, 32, 33, 34])
        .with_prefill_failure_after_chunk(2);
    let prefix_cache = Arc::clone(&adapter.prefix_cache);
    let metrics = Arc::clone(&adapter.prefix_cache_metrics);
    let namespace = driver_prefix_namespace(&adapter, &request, 5, 1);
    let driver = driver_for_test(adapter).with_max_prefill_tokens(2);

    let err = driver
        .generate_blocking(request.clone(), CancellationToken::new())
        .expect_err("prefill failure is returned");
    assert!(err.to_string().contains("test prefill failed"));
    assert_prefix_cache_entry(&prefix_cache, &namespace, &[30, 31]);
    assert_no_prefix_cache_entry(&prefix_cache, &namespace, &[30, 31, 32, 33]);
    assert_no_prefix_cache_entry(&prefix_cache, &namespace, &[30, 31, 32, 33, 34]);

    let warm = driver
        .generate_blocking(request, CancellationToken::new())
        .expect("warm generation reuses only the successful checkpoint");

    assert_eq!(warm.prompt_cached_tokens, Some(2));
    let snapshot = metrics.snapshot();
    assert_eq!(snapshot["checkpoint_reuse_hits"], 1);
    assert_eq!(snapshot["checkpoint_reused_tokens"], 2);
}

#[test]
fn driver_allows_adapter_context_sensitive_candidate_observation() {
    let driver = driver_for_test(ContextSensitiveTestAdapter::new([7_usize, 8_usize], 1));

    let output = driver
        .generate_blocking(driver_test_request(4), CancellationToken::new())
        .expect("generation stops through adapter hook");

    assert_eq!(output.text, "<7>");
    assert_eq!(output.completion_tokens, 1);
    assert_eq!(output.finish_reason, BackendFinishReason::Stop);
}

#[test]
fn driver_cleans_cache_mirrors_when_prefill_fails_before_session_handoff() {
    let adapter = TestAdapter::new([1_usize]).with_prefill_failure();
    let cleanup_calls = adapter.cleanup_calls();
    let driver = driver_for_test(adapter);

    let err = driver
        .generate_blocking(driver_test_request(1), CancellationToken::new())
        .expect_err("prefill failure is returned");

    assert!(err.to_string().contains("test prefill failed"));
    assert_eq!(cleanup_calls.load(Ordering::SeqCst), 1);
}

#[test]
fn driver_cleans_cloned_prefix_cache_when_suffix_prefill_fails() {
    let request = driver_test_request(1);
    let adapter = TestAdapter::new([1_usize]).with_encoded_prompt([10_u32, 11_u32, 12_u32]);
    let cleanup_calls = adapter.cleanup_calls();
    let cleanup_markers = adapter.cleanup_markers();
    let prefix_cache = Arc::clone(&adapter.prefix_cache);
    let metrics = Arc::clone(&adapter.prefix_cache_metrics);
    let (namespace, byte_len) = store_driver_prefix_hit(
        &adapter,
        &request,
        3,
        1,
        &[10, 11],
        TestCache {
            bytes: 13,
            marker: 77,
        },
    );
    let driver = driver_for_test(adapter.with_prefill_failure());

    let err = driver
        .generate_blocking(request, CancellationToken::new())
        .expect_err("suffix prefill failure is returned");

    assert!(err.to_string().contains("test prefill failed"));
    assert_eq!(cleanup_calls.load(Ordering::SeqCst), 1);
    let markers = cleanup_markers
        .lock()
        .expect("cleanup markers lock is not poisoned")
        .clone();
    assert_eq!(markers, vec![vec![77]]);
    assert_only_prefix_cache_entry(&prefix_cache, &namespace, &[10, 11], byte_len);
    let snapshot = metrics.snapshot();
    assert_eq!(snapshot["hits"], 1);
    assert_eq!(snapshot["hit_tokens"], 2);
    assert_eq!(snapshot["miss_tokens"], 1);
    assert_eq!(snapshot["stores"], 1);
    assert_eq!(snapshot["resident_entries"], 1);
    assert_eq!(snapshot["resident_bytes"], byte_len);
}

#[test]
fn driver_cleans_cloned_prefix_cache_when_suffix_prefill_cancels() {
    let request = driver_test_request(1);
    let cancellation = CancellationToken::new();
    let adapter = TestAdapter::new([1_usize]).with_encoded_prompt([20_u32, 21_u32, 22_u32]);
    let cleanup_calls = adapter.cleanup_calls();
    let cleanup_markers = adapter.cleanup_markers();
    let prefix_cache = Arc::clone(&adapter.prefix_cache);
    let metrics = Arc::clone(&adapter.prefix_cache_metrics);
    let (namespace, byte_len) = store_driver_prefix_hit(
        &adapter,
        &request,
        3,
        1,
        &[20, 21],
        TestCache {
            bytes: 17,
            marker: 88,
        },
    );
    let driver = driver_for_test(adapter.with_prefill_cancellation(cancellation.clone()));

    let err = driver
        .generate_blocking(request, cancellation)
        .expect_err("suffix prefill cancellation is returned");

    assert!(err.is_cancelled());
    assert_eq!(cleanup_calls.load(Ordering::SeqCst), 1);
    let markers = cleanup_markers
        .lock()
        .expect("cleanup markers lock is not poisoned")
        .clone();
    assert_eq!(markers, vec![vec![88]]);
    assert_only_prefix_cache_entry(&prefix_cache, &namespace, &[20, 21], byte_len);
    let snapshot = metrics.snapshot();
    assert_eq!(snapshot["hits"], 1);
    assert_eq!(snapshot["hit_tokens"], 2);
    assert_eq!(snapshot["miss_tokens"], 1);
    assert_eq!(snapshot["stores"], 1);
    assert_eq!(snapshot["resident_entries"], 1);
    assert_eq!(snapshot["resident_bytes"], byte_len);
}

#[test]
fn driver_does_not_clean_cache_mirrors_after_successful_session_handoff() {
    let adapter = TestAdapter::new([1_usize]);
    let cleanup_calls = adapter.cleanup_calls();
    let driver = driver_for_test(adapter);

    driver
        .generate_blocking(driver_test_request(1), CancellationToken::new())
        .expect("generation succeeds");

    assert_eq!(cleanup_calls.load(Ordering::SeqCst), 0);
}
