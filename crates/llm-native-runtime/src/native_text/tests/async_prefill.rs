use super::*;

#[tokio::test(flavor = "current_thread")]
async fn native_text_open_blocking_work_runs_off_async_runtime() {
    let work_started = Arc::new(AtomicUsize::new(0));
    let work_started_for_closure = Arc::clone(&work_started);

    let open = tokio::spawn(async move {
        run_native_text_open_blocking("Test", move || {
            work_started_for_closure.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(750));
            Ok::<_, anyhow::Error>("opened")
        })
        .await
    });

    let deadline = tokio::time::Instant::now() + Duration::from_millis(200);
    while work_started.load(Ordering::SeqCst) == 0 && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    assert_eq!(work_started.load(Ordering::SeqCst), 1);
    assert!(
        !open.is_finished(),
        "native text snapshot open work should not block the async runtime"
    );
    assert_eq!(
        open.await
            .expect("open task joins")
            .expect("blocking open work succeeds"),
        "opened"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn driver_generate_with_cancel_runs_native_work_off_async_runtime() {
    let adapter = TestAdapter::new([1_usize]).with_next_token_delay(Duration::from_millis(750));
    let next_token_calls = adapter.next_token_calls();
    let driver = driver_for_test(adapter);

    let generation = tokio::spawn(async move {
        driver
            .generate_with_cancel(driver_test_request(1), CancellationToken::new())
            .await
    });

    let deadline = tokio::time::Instant::now() + Duration::from_millis(200);
    while next_token_calls.load(Ordering::SeqCst) == 0 && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    assert_eq!(next_token_calls.load(Ordering::SeqCst), 1);
    assert!(
        !generation.is_finished(),
        "native generation should not block the async runtime while CPU work is running"
    );
    let output = generation
        .await
        .expect("generation task joins")
        .expect("generation succeeds");
    assert_eq!(output.completion_tokens, 1);
}

#[tokio::test(flavor = "current_thread")]
async fn driver_generate_with_cancel_cancels_worker_when_future_is_dropped() {
    let adapter = TestAdapter::new([1_usize]).with_next_token_delay(Duration::from_millis(750));
    let next_token_calls = adapter.next_token_calls();
    let driver = driver_for_test(adapter);
    let cancellation = CancellationToken::new();
    let worker_cancellation = cancellation.clone();

    let generation = tokio::spawn(async move {
        driver
            .generate_with_cancel(driver_test_request(1), worker_cancellation)
            .await
    });

    let deadline = tokio::time::Instant::now() + Duration::from_millis(200);
    while next_token_calls.load(Ordering::SeqCst) == 0 && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(next_token_calls.load(Ordering::SeqCst), 1);

    generation.abort();
    assert!(
        generation
            .await
            .expect_err("generation task is aborted")
            .is_cancelled()
    );
    assert!(
        cancellation.is_cancelled(),
        "dropping the async request future should signal the blocking native worker"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn driver_generate_with_cancel_stops_worker_when_dropped_during_prefill() {
    let blocking_prefill = Arc::new(BlockingPrefill::new());
    let adapter = TestAdapter::new([1_usize]).with_blocking_prefill(Arc::clone(&blocking_prefill));
    let driver = driver_for_test(adapter);
    let cancellation = CancellationToken::new();
    let worker_cancellation = cancellation.clone();

    let generation = tokio::spawn(async move {
        driver
            .generate_with_cancel(driver_test_request(1), worker_cancellation)
            .await
    });

    tokio::time::timeout(Duration::from_millis(500), blocking_prefill.wait_started())
        .await
        .expect("prefill starts");

    generation.abort();
    assert!(
        generation
            .await
            .expect_err("generation task is aborted")
            .is_cancelled()
    );
    assert!(
        cancellation.is_cancelled(),
        "dropping the async request future should signal the blocking native worker"
    );

    let dropped_before_release =
        tokio::time::timeout(Duration::from_millis(200), blocking_prefill.wait_dropped())
            .await
            .is_ok();
    blocking_prefill.release.notify_waiters();
    tokio::time::timeout(Duration::from_secs(1), blocking_prefill.wait_dropped())
        .await
        .expect("blocking prefill future eventually drops after test release");

    assert_eq!(blocking_prefill.dropped_calls(), 1);
    assert!(
        dropped_before_release,
        "cancelled request futures must stop the in-flight prefill future before the chunk is released"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn driver_stream_with_cancel_cancels_worker_when_stream_is_dropped() {
    let adapter = TestAdapter::new([1_usize]).with_next_token_delay(Duration::from_millis(750));
    let next_token_calls = adapter.next_token_calls();
    let driver = driver_for_test(adapter);
    let cancellation = CancellationToken::new();
    let worker_cancellation = cancellation.clone();

    let stream = driver.generate_stream_with_cancel(driver_test_request(1), worker_cancellation);

    let deadline = tokio::time::Instant::now() + Duration::from_millis(200);
    while next_token_calls.load(Ordering::SeqCst) == 0 && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(next_token_calls.load(Ordering::SeqCst), 1);
    assert!(
        !cancellation.is_cancelled(),
        "active stream should not be cancelled before the client drops it"
    );

    drop(stream);

    tokio::time::timeout(Duration::from_millis(200), async {
        while !cancellation.is_cancelled() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("dropping the stream should signal the blocking native worker");
}

#[test]
fn prefill_context_returns_last_hidden_from_last_chunk() {
    let cancellation = CancellationToken::new();
    let mut observed_chunks = Vec::new();

    let mut prefill_caches = [TestCache {
        bytes: 0,
        marker: 0,
    }];
    let mut prefill_scratch = InferenceScratchpad::new();
    let hidden = native_text_prefill_context_with_cache(
        "Test",
        2,
        &[1, 2, 3],
        &mut prefill_caches,
        &cancellation,
        &mut prefill_scratch,
        |chunk, _caches, _scratch| {
            observed_chunks.push(chunk.to_vec());
            Ok(chunk
                .iter()
                .map(|token| vec![*token as f32, (*token * 10) as f32])
                .collect())
        },
    )
    .expect("prefill succeeds");

    assert_eq!(observed_chunks, vec![vec![1, 2], vec![3]]);
    assert_eq!(hidden, vec![3.0, 30.0]);
}

#[test]
fn prefill_context_observes_cancellation_between_chunks() {
    let cancellation = CancellationToken::new();
    let mut calls = 0;

    let mut cancel_caches = [TestCache {
        bytes: 0,
        marker: 0,
    }];
    let mut cancel_scratch = InferenceScratchpad::new();
    let err = native_text_prefill_context_with_cache(
        "Test",
        1,
        &[1, 2],
        &mut cancel_caches,
        &cancellation,
        &mut cancel_scratch,
        |chunk, _caches, _scratch| {
            calls += 1;
            assert_eq!(chunk, &[1]);
            cancellation.cancel();
            Ok(vec![vec![1.0]])
        },
    )
    .expect_err("cancelled after first chunk");

    assert!(err.is_cancelled());
    assert_eq!(calls, 1);
}
