struct ScriptedChatBackend;

#[async_trait]
impl ModelBackend for ScriptedChatBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "scripted-chat")
    }

    async fn generate(&self, request: BackendRequest) -> Result<BackendOutput, BackendError> {
        if request.model != self.model_id() {
            return Err(BackendError::model_not_found(
                request.model,
                self.model_id().to_owned(),
            ));
        }
        let text = scripted_chat_response(request.prompt());
        Ok(BackendOutput {
            prompt_tokens: test_token_count(request.prompt()),
            prompt_cached_tokens: None,
            completion_tokens: test_token_count(&text),
            text,
            finish_reason: BackendFinishReason::Stop,
        })
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }
}

struct BlockingBackend {
    entered: Arc<Notify>,
    release: Arc<Notify>,
}

#[async_trait]
impl ModelBackend for BlockingBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "blocking")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        self.entered.notify_waiters();
        self.release.notified().await;
        Ok(BackendOutput {
            text: "released".to_owned(),
            prompt_tokens: 1,
            prompt_cached_tokens: None,
            completion_tokens: 1,
            finish_reason: BackendFinishReason::Stop,
        })
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }
}

struct FairnessBackend {
    order: Arc<Mutex<Vec<String>>>,
    entered: Arc<Notify>,
    release: Arc<Semaphore>,
}

#[async_trait]
impl ModelBackend for FairnessBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "fairness")
    }

    async fn generate(&self, request: BackendRequest) -> Result<BackendOutput, BackendError> {
        let label = if request.prompt().contains("first-long") {
            "first-long"
        } else if request.prompt().contains("second-long") {
            "second-long"
        } else if request.prompt().contains("third-short") {
            "third-short"
        } else {
            "unknown"
        };
        self.order
            .lock()
            .expect("order lock is not poisoned")
            .push(label.to_owned());
        self.entered.notify_waiters();
        let _permit = self
            .release
            .acquire()
            .await
            .expect("release semaphore open");
        Ok(BackendOutput {
            text: label.to_owned(),
            prompt_tokens: 1,
            prompt_cached_tokens: None,
            completion_tokens: 1,
            finish_reason: BackendFinishReason::Stop,
        })
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }
}

struct AdminCancellableBackend {
    entered: Arc<Notify>,
    cancelled: Arc<Notify>,
}

#[async_trait]
impl ModelBackend for AdminCancellableBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "admin-cancellable")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Err(BackendError::other(
            "generate_with_cancel should be used".to_owned(),
        ))
    }

    async fn generate_with_cancel(
        &self,
        _request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        self.entered.notify_waiters();
        cancellation.cancelled().await;
        self.cancelled.notify_waiters();
        Err(BackendError::cancelled())
    }
}

struct AdminLateErrorBackend {
    entered: Arc<Notify>,
    release: Arc<Semaphore>,
}

#[async_trait]
impl ModelBackend for AdminLateErrorBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "admin-late-error")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Err(BackendError::other(
            "generate_with_cancel should be used".to_owned(),
        ))
    }

    async fn generate_with_cancel(
        &self,
        _request: BackendRequest,
        _cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        self.entered.notify_waiters();
        let _permit = self
            .release
            .acquire()
            .await
            .expect("release semaphore open");
        Err(BackendError::other("late backend failure".to_owned()))
    }
}

struct NoProgressBackend;

#[async_trait]
impl ModelBackend for NoProgressBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "no-progress")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Ok(BackendOutput {
            text: String::new(),
            prompt_tokens: 1,
            prompt_cached_tokens: None,
            completion_tokens: 4096,
            finish_reason: BackendFinishReason::Length,
        })
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }
}

struct DelayedStreamBackend {
    release: Arc<Semaphore>,
}

#[async_trait]
impl ModelBackend for DelayedStreamBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "delayed-stream")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        let _permit = self
            .release
            .acquire()
            .await
            .expect("release semaphore open");
        Ok(BackendOutput {
            text: "released".to_owned(),
            prompt_tokens: 1,
            prompt_cached_tokens: None,
            completion_tokens: 1,
            finish_reason: BackendFinishReason::Stop,
        })
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }
}

struct TwoStageStreamBackend {
    first: Arc<Notify>,
    finish: Arc<Notify>,
}

#[async_trait]
impl ModelBackend for TwoStageStreamBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "two-stage-stream")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        self.first.notified().await;
        self.finish.notified().await;
        Ok(BackendOutput {
            text: "first second".to_owned(),
            prompt_tokens: 1,
            prompt_cached_tokens: None,
            completion_tokens: 2,
            finish_reason: BackendFinishReason::Stop,
        })
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }

    fn generate_stream<'a>(
        &'a self,
        _request: BackendRequest,
    ) -> futures::stream::BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        let first = self.first.clone();
        let finish = self.finish.clone();
        async_stream::try_stream! {
            first.notified().await;
            yield BackendStreamChunk {
                text: "first".to_owned(),
                tool_call_deltas: Vec::new(),
                prompt_tokens: 1,
                prompt_cached_tokens: None,
                completion_tokens: 1,
                finish_reason: None,
                progress: None,
            };
            finish.notified().await;
            yield BackendStreamChunk {
                text: " second".to_owned(),
                tool_call_deltas: Vec::new(),
                prompt_tokens: 1,
                prompt_cached_tokens: None,
                completion_tokens: 1,
                finish_reason: Some(BackendFinishReason::Stop),
                progress: None,
            };
        }
        .boxed()
    }

    fn generate_stream_with_cancel<'a>(
        &'a self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> futures::stream::BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        if cancellation.is_cancelled() {
            return futures::stream::once(async { Err(BackendError::cancelled()) }).boxed();
        }
        self.generate_stream(request)
    }
}

struct TwoStageToolStreamBackend {
    first: Arc<Notify>,
    tool: Arc<Notify>,
}

#[async_trait]
impl ModelBackend for TwoStageToolStreamBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "two-stage-tool-stream")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Err(BackendError::other(
            "two-stage tool stream test must use generate_stream".to_owned(),
        ))
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }

    fn generate_stream<'a>(
        &'a self,
        _request: BackendRequest,
    ) -> futures::stream::BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        let first = self.first.clone();
        let tool = self.tool.clone();
        async_stream::try_stream! {
            first.notified().await;
            yield BackendStreamChunk {
                text: "decode-start".to_owned(),
                tool_call_deltas: Vec::new(),
                prompt_tokens: 1,
                prompt_cached_tokens: None,
                completion_tokens: 1,
                finish_reason: None,
                progress: None,
            };
            tool.notified().await;
            yield BackendStreamChunk {
                text: String::new(),
                tool_call_deltas: vec![BackendToolCallDelta {
                    index: 0,
                    id: Some("call_0".to_owned()),
                    call_type: Some(BackendToolCallType::Function),
                    function: Some(BackendToolCallFunctionDelta {
                        name: Some("lookup".to_owned()),
                        arguments: Some(r#"{"query":"rust"}"#.to_owned()),
                    }),
                }],
                prompt_tokens: 1,
                prompt_cached_tokens: None,
                completion_tokens: 1,
                finish_reason: Some(BackendFinishReason::ToolCalls),
                progress: None,
            };
        }
        .boxed()
    }

    fn generate_stream_with_cancel<'a>(
        &'a self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> futures::stream::BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        if cancellation.is_cancelled() {
            return futures::stream::once(async { Err(BackendError::cancelled()) }).boxed();
        }
        self.generate_stream(request)
    }
}

struct CancellableStreamBackend {
    cancelled: Arc<Notify>,
}

#[async_trait]
impl ModelBackend for CancellableStreamBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "cancellable-stream")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Ok(BackendOutput {
            text: "first".to_owned(),
            prompt_tokens: 1,
            prompt_cached_tokens: None,
            completion_tokens: 1,
            finish_reason: BackendFinishReason::Stop,
        })
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }

    fn generate_stream_with_cancel<'a>(
        &'a self,
        _request: BackendRequest,
        cancellation: CancellationToken,
    ) -> futures::stream::BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        let cancelled = self.cancelled.clone();
        tokio::spawn(async move {
            cancellation.cancelled().await;
            cancelled.notify_waiters();
        });
        async_stream::try_stream! {
            yield BackendStreamChunk {
                text: "first".to_owned(),
                tool_call_deltas: Vec::new(),
                prompt_tokens: 1,
                prompt_cached_tokens: None,
                completion_tokens: 1,
                finish_reason: None,
                progress: None,
            };
            futures::future::pending::<()>().await;
        }
        .boxed()
    }
}

struct FailingStreamBackend;

#[async_trait]
impl ModelBackend for FailingStreamBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "failing-stream")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Ok(BackendOutput {
            text: "first".to_owned(),
            prompt_tokens: 1,
            prompt_cached_tokens: None,
            completion_tokens: 1,
            finish_reason: BackendFinishReason::Stop,
        })
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }

    fn generate_stream<'a>(
        &'a self,
        _request: BackendRequest,
    ) -> futures::stream::BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        async_stream::try_stream! {
            yield BackendStreamChunk {
                text: "first".to_owned(),
                tool_call_deltas: Vec::new(),
                prompt_tokens: 1,
                prompt_cached_tokens: None,
                completion_tokens: 1,
                finish_reason: None,
                progress: None,
            };
            Err(BackendError::other(
                "stream failed in mlx parser at /srv/kir-ai/private/model.safetensors".to_owned(),
            ))?;
        }
        .boxed()
    }

    fn generate_stream_with_cancel<'a>(
        &'a self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> futures::stream::BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        if cancellation.is_cancelled() {
            return futures::stream::once(async { Err(BackendError::cancelled()) }).boxed();
        }
        self.generate_stream(request)
    }
}
