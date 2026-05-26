use super::*;

#[tokio::test]
async fn native_qwen_generate_with_cancel_observes_pre_cancelled_token() {
    let snapshot = temp_snapshot_dir("cancelled-generate");
    std::fs::remove_dir_all(&snapshot).ok();
    std::fs::create_dir_all(&snapshot).expect("snapshot dir");
    copy_fixture("config.json", snapshot.join("config.json"));
    copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));
    copy_fixture(
        "model.safetensors.index.json",
        snapshot.join("model.safetensors.index.json"),
    );
    let backend = NativeQwenBackend::open(crate::DEFAULT_MODEL_ID, &snapshot)
        .await
        .expect("backend opens snapshot");
    let cancellation = CancellationToken::new();
    cancellation.cancel();

    let err = backend
        .generate_with_cancel(
            BackendRequest::raw_completion(
                crate::DEFAULT_MODEL_ID,
                "say hi",
                Some(1),
                SamplingConfig::Greedy,
            ),
            cancellation,
        )
        .await
        .expect_err("pre-cancelled generation fails before decode");

    assert!(err.to_string().contains("cancelled"));
    std::fs::remove_dir_all(snapshot).ok();
}

#[tokio::test]
async fn native_qwen_stream_with_cancel_observes_pre_cancelled_token() {
    let snapshot = temp_snapshot_dir("cancelled-stream");
    std::fs::remove_dir_all(&snapshot).ok();
    std::fs::create_dir_all(&snapshot).expect("snapshot dir");
    copy_fixture("config.json", snapshot.join("config.json"));
    copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));
    copy_fixture(
        "model.safetensors.index.json",
        snapshot.join("model.safetensors.index.json"),
    );
    let backend = NativeQwenBackend::open(crate::DEFAULT_MODEL_ID, &snapshot)
        .await
        .expect("backend opens snapshot");
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let mut stream = backend.generate_stream_with_cancel(
        BackendRequest::raw_completion(
            crate::DEFAULT_MODEL_ID,
            "say hi",
            Some(1),
            SamplingConfig::Greedy,
        ),
        cancellation,
    );
    let err = stream
        .next()
        .await
        .expect("stream reports cancellation")
        .expect_err("pre-cancelled stream fails before normal EOF");

    assert!(err.is_cancelled());
    assert!(stream.next().await.is_none());
    std::fs::remove_dir_all(snapshot).ok();
}

#[tokio::test]
async fn native_qwen_worker_stream_reports_join_failure_after_channel_close() {
    let (_tx, rx) = tokio::sync::mpsc::channel(1);
    let worker = tokio::task::spawn_blocking(|| panic!("stream worker panic"));
    let mut stream = native_text_worker_stream("native Qwen", rx, worker, CancellationToken::new());

    let err = stream
        .next()
        .await
        .expect("join failure event")
        .expect_err("worker panic is surfaced");

    assert!(
        err.to_string()
            .contains("native Qwen streaming worker failed")
    );
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn native_qwen_start_decode_session_observes_pre_cancelled_token() {
    let snapshot = temp_snapshot_dir("cancelled-start-decode");
    std::fs::remove_dir_all(&snapshot).ok();
    std::fs::create_dir_all(&snapshot).expect("snapshot dir");
    copy_fixture("config.json", snapshot.join("config.json"));
    copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));
    copy_fixture(
        "model.safetensors.index.json",
        snapshot.join("model.safetensors.index.json"),
    );
    let backend = NativeQwenBackend::open(crate::DEFAULT_MODEL_ID, &snapshot)
        .await
        .expect("backend opens snapshot");
    let cancellation = CancellationToken::new();
    cancellation.cancel();

    match start_qwen_decode_session(
        &backend,
        &[0],
        1,
        &native_qwen_test_request(crate::DEFAULT_MODEL_ID),
        &cancellation,
    )
    .await
    {
        Err(err) if err.is_cancelled() => {}
        Err(err) => panic!("expected cancellation before prefill, got {err}"),
        Ok(_) => panic!("pre-cancelled decode startup should fail before prefill"),
    }
    std::fs::remove_dir_all(snapshot).ok();
}
