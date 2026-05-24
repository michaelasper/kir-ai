use super::*;

#[tokio::test(start_paused = true)]
async fn dropping_unpolled_chat_stream_cancels_backend_stream() {
    let cancelled = Arc::new(Notify::new());
    let runtime = Runtime::new(CancellableStreamBackend {
        cancelled: cancelled.clone(),
    });
    let stream = runtime
        .chat_stream(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("say hi")],
            stream: true,
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("streaming chat starts");

    drop(stream);
    tokio::time::timeout(Duration::from_millis(100), cancelled.notified())
        .await
        .expect("backend cancellation token is cancelled");
}

#[tokio::test]
async fn completed_non_streaming_chat_does_not_cancel_shared_token() {
    let token = CancellationToken::new();
    let runtime = Runtime::new(ProtocolTestBackend::new("local-qwen36", "done"));

    runtime
        .chat_with_cancel(
            ChatCompletionRequest {
                model: "local-qwen36".to_owned(),
                messages: vec![ChatMessage::user("say hi")],
                ..ChatCompletionRequest::default()
            },
            token.clone(),
        )
        .await
        .expect("chat completes");

    assert!(
        !token.is_cancelled(),
        "successful chat completion must not cancel a shared token"
    );
}

#[tokio::test]
async fn completed_non_streaming_completion_does_not_cancel_shared_token() {
    let token = CancellationToken::new();
    let runtime = Runtime::new(ProtocolTestBackend::new("local-qwen36", "done"));

    runtime
        .completion_with_cancel(
            CompletionRequest {
                model: "local-qwen36".to_owned(),
                prompt: "say hi".to_owned(),
                ..CompletionRequest::default()
            },
            token.clone(),
        )
        .await
        .expect("completion completes");

    assert!(
        !token.is_cancelled(),
        "successful text completion must not cancel a shared token"
    );
}

#[tokio::test]
async fn completed_chat_stream_does_not_cancel_shared_token() {
    let token = CancellationToken::new();
    let runtime = Runtime::new(ProtocolTestBackend::new("local-qwen36", "done"));
    let stream = runtime
        .chat_stream_with_cancel(
            ChatCompletionRequest {
                model: "local-qwen36".to_owned(),
                messages: vec![ChatMessage::user("say hi")],
                stream: true,
                ..ChatCompletionRequest::default()
            },
            token.clone(),
        )
        .await
        .expect("chat stream starts");

    stream
        .collect_chunks()
        .await
        .expect("chat stream completes");

    assert!(
        !token.is_cancelled(),
        "successful chat stream must not cancel a shared token"
    );
}

#[tokio::test]
async fn completed_completion_stream_does_not_cancel_shared_token() {
    let token = CancellationToken::new();
    let runtime = Runtime::new(ProtocolTestBackend::new("local-qwen36", "done"));
    let stream = runtime
        .completion_stream_with_cancel(
            CompletionRequest {
                model: "local-qwen36".to_owned(),
                prompt: "say hi".to_owned(),
                stream: true,
                ..CompletionRequest::default()
            },
            token.clone(),
        )
        .await
        .expect("completion stream starts");

    stream
        .collect_chunks()
        .await
        .expect("completion stream completes");

    assert!(
        !token.is_cancelled(),
        "successful completion stream must not cancel a shared token"
    );
}

#[tokio::test(start_paused = true)]
async fn dropping_non_streaming_chat_future_cancels_backend_generation() {
    let started = Arc::new(Notify::new());
    let cancelled = Arc::new(Notify::new());
    let runtime = Runtime::new(CancellableGenerateBackend {
        started: started.clone(),
        cancelled: cancelled.clone(),
    });
    let mut future = Box::pin(runtime.chat(ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("say hi")],
        ..ChatCompletionRequest::default()
    }));

    tokio::select! {
        result = &mut future => panic!("backend should stay pending until cancellation: {result:?}"),
        _ = started.notified() => {}
    }

    drop(future);
    tokio::time::timeout(Duration::from_millis(100), cancelled.notified())
        .await
        .expect("backend generation cancellation token is cancelled");
}

#[tokio::test(start_paused = true)]
async fn dropping_non_streaming_completion_future_cancels_backend_generation() {
    let started = Arc::new(Notify::new());
    let cancelled = Arc::new(Notify::new());
    let runtime = Runtime::new(CancellableGenerateBackend {
        started: started.clone(),
        cancelled: cancelled.clone(),
    });
    let mut future = Box::pin(runtime.completion(CompletionRequest {
        model: "local-qwen36".to_owned(),
        prompt: "say hi".to_owned(),
        ..CompletionRequest::default()
    }));

    tokio::select! {
        result = &mut future => panic!("backend should stay pending until cancellation: {result:?}"),
        _ = started.notified() => {}
    }

    drop(future);
    tokio::time::timeout(Duration::from_millis(100), cancelled.notified())
        .await
        .expect("backend generation cancellation token is cancelled");
}
