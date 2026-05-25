use super::*;

struct PrefillProgressStreamBackend;

#[async_trait]
impl ModelBackend for PrefillProgressStreamBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "prefill-progress-stream")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Err(BackendError::other(
            "prefill progress HTTP test must use generate_stream".to_owned(),
        ))
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
        if cancellation.is_cancelled() {
            return futures::stream::once(async { Err(BackendError::cancelled()) }).boxed();
        }
        async_stream::try_stream! {
            yield BackendStreamChunk {
                text: String::new(),
                tool_call_deltas: Vec::new(),
                prompt_tokens: 5,
                prompt_cached_tokens: Some(0),
                completion_tokens: 0,
                finish_reason: None,
                progress: Some(BackendStreamProgress::PrefillProgress {
                    chunk: 1,
                    total: 3,
                    tokens: 2,
                    total_tokens: 5,
                }),
            };
            yield BackendStreamChunk {
                text: "done".to_owned(),
                tool_call_deltas: Vec::new(),
                prompt_tokens: 5,
                prompt_cached_tokens: Some(0),
                completion_tokens: 1,
                finish_reason: Some(BackendFinishReason::Stop),
                progress: None,
            };
        }
        .boxed()
    }
}

struct InterleavedPrefillStreamBackend {
    order: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl ModelBackend for InterleavedPrefillStreamBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "interleaved-prefill-stream")
    }

    async fn generate(&self, request: BackendRequest) -> Result<BackendOutput, BackendError> {
        if !request.prompt().contains("short-decode") {
            return Err(BackendError::other(
                "interleaved prefill test only uses non-streaming generate for short decode"
                    .to_owned(),
            ));
        }
        self.order
            .lock()
            .expect("order lock is not poisoned")
            .push("short-decode".to_owned());
        Ok(BackendOutput {
            text: "short-decode".to_owned(),
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
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> futures::stream::BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        if cancellation.is_cancelled() {
            return futures::stream::once(async { Err(BackendError::cancelled()) }).boxed();
        }
        let order = Arc::clone(&self.order);
        let prompt = request.prompt().to_owned();
        async_stream::try_stream! {
            if !prompt.contains("long-prefill") {
                Err(BackendError::other(
                    "interleaved prefill test only uses streaming generate for long prefill"
                        .to_owned(),
                ))?;
            }
            {
                let mut order = order.lock().expect("order lock is not poisoned");
                order.push("long-prefill-start".to_owned());
            }
            let progress = BackendStreamProgress::PrefillProgress {
                chunk: 1,
                total: 2,
                tokens: 2,
                total_tokens: 4,
            };
            yield BackendStreamChunk {
                text: String::new(),
                tool_call_deltas: Vec::new(),
                prompt_tokens: 4,
                prompt_cached_tokens: Some(0),
                completion_tokens: 0,
                finish_reason: None,
                progress: Some(progress.clone()),
            };
            if let Some(admission) = request.prefill_chunk_admission() {
                admission.wait_for_next_chunk(progress).await?;
            }
            {
                let mut order = order.lock().expect("order lock is not poisoned");
                order.push("long-prefill-resume".to_owned());
            }
            yield BackendStreamChunk {
                text: "long-finished".to_owned(),
                tool_call_deltas: Vec::new(),
                prompt_tokens: 4,
                prompt_cached_tokens: Some(0),
                completion_tokens: 1,
                finish_reason: Some(BackendFinishReason::Stop),
                progress: None,
            };
        }
        .boxed()
    }
}

struct OneByteDeltaStreamBackend {
    delay: Duration,
    fragments: Vec<&'static str>,
}

#[async_trait]
impl ModelBackend for OneByteDeltaStreamBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "one-byte-delta-stream")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Err(BackendError::other(
            "generate_stream_with_cancel should be used".to_owned(),
        ))
    }

    async fn generate_with_cancel(
        &self,
        _request: BackendRequest,
        _cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        Err(BackendError::other(
            "generate_stream_with_cancel should be used".to_owned(),
        ))
    }

    fn generate_stream_with_cancel<'a>(
        &'a self,
        _request: BackendRequest,
        _cancellation: CancellationToken,
    ) -> futures::stream::BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        let delay = self.delay;
        let fragments = self.fragments.clone();
        async_stream::try_stream! {
            for (index, fragment) in fragments.iter().enumerate() {
                tokio::time::sleep(delay).await;
                yield BackendStreamChunk {
                    text: (*fragment).to_owned(),
                    tool_call_deltas: Vec::new(),
                    prompt_tokens: 1,
                    prompt_cached_tokens: None,
                    completion_tokens: 1,
                    finish_reason: (index + 1 == fragments.len()).then_some(BackendFinishReason::Stop),
                    progress: None,
                };
            }
        }
        .boxed()
    }
}

struct SlowStructuredToolArgumentBackend {
    delay: Duration,
}

#[async_trait]
impl ModelBackend for SlowStructuredToolArgumentBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "slow-structured-tool-argument-stream")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Err(BackendError::other(
            "slow structured tool argument test must use generate_stream".to_owned(),
        ))
    }

    async fn generate_with_cancel(
        &self,
        _request: BackendRequest,
        _cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        Err(BackendError::other(
            "slow structured tool argument test must use generate_stream".to_owned(),
        ))
    }

    fn generate_stream_with_cancel<'a>(
        &'a self,
        _request: BackendRequest,
        cancellation: CancellationToken,
    ) -> futures::stream::BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        if cancellation.is_cancelled() {
            return futures::stream::once(async { Err(BackendError::cancelled()) }).boxed();
        }
        let delay = self.delay;
        async_stream::try_stream! {
            yield BackendStreamChunk {
                text: String::new(),
                tool_call_deltas: vec![http_structured_tool_delta(
                    0,
                    Some("call_read_1"),
                    Some("read"),
                    Some("{"),
                )],
                prompt_tokens: 1,
                prompt_cached_tokens: None,
                completion_tokens: 1,
                finish_reason: None,
                progress: None,
            };
            for arguments in [
                r#""path""#,
                r#":"#,
                r#""calculator.py""#,
                r#"}"#,
            ] {
                tokio::time::sleep(delay).await;
                yield BackendStreamChunk {
                    text: String::new(),
                    tool_call_deltas: vec![http_structured_tool_delta(
                        0,
                        None,
                        None,
                        Some(arguments),
                    )],
                    prompt_tokens: 1,
                    prompt_cached_tokens: None,
                    completion_tokens: 1,
                    finish_reason: (arguments == "}").then_some(BackendFinishReason::ToolCalls),
                    progress: None,
                };
            }
        }
        .boxed()
    }
}

struct GemmaMlxRequiredToolRejectingStreamBackend;

#[async_trait]
impl ModelBackend for GemmaMlxRequiredToolRejectingStreamBackend {
    fn model_id(&self) -> &str {
        "gemma4-e2b-mlx-4bit"
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        BackendModelMetadata::new(self.model_id(), "mlx").with_family("gemma")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Err(BackendError::other(
            "Gemma MLX required-tool HTTP test must use generate_stream".to_owned(),
        ))
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
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> futures::stream::BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        if cancellation.is_cancelled() {
            return futures::stream::once(async { Err(BackendError::cancelled()) }).boxed();
        }
        let message = gemma_mlx_required_tool_rejection_message(self.model_id(), &request);
        futures::stream::once(async { Err(BackendError::unsupported_request(message)) }).boxed()
    }
}

fn gemma_mlx_required_tool_rejection_message(model: &str, request: &BackendRequest) -> String {
    let choice = request
        .as_chat()
        .and_then(|chat| chat.required_tool_choice.as_ref());
    let choice = match choice {
        Some(BackendToolChoice::RequiredAny) => "any declared tool".to_owned(),
        Some(BackendToolChoice::RequiredFunction(name)) => format!("function `{name}`"),
        Some(_) | None => "missing required tool choice".to_owned(),
    };
    format!(
        "MLX Gemma required tool_choice is not supported for model `{model}` \
         (backend `mlx`, family `gemma`); required tool choice {choice} cannot be enforced"
    )
}

struct PendingCancellableStreamBackend {
    cancelled: Arc<Notify>,
}

#[async_trait]
impl ModelBackend for PendingCancellableStreamBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "pending-cancellable-stream")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Err(BackendError::other(
            "generate_stream_with_cancel should be used".to_owned(),
        ))
    }

    async fn generate_with_cancel(
        &self,
        _request: BackendRequest,
        _cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        Err(BackendError::other(
            "generate_stream_with_cancel should be used".to_owned(),
        ))
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
        futures::stream::pending().boxed()
    }
}

async fn wait_for_metrics<F>(app: &Router, predicate: F) -> Value
where
    F: Fn(&Value) -> bool,
{
    tokio::time::timeout(Duration::from_millis(500), async {
        loop {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri("/admin/metrics")
                        .body(Body::empty())
                        .expect("request builds"),
                )
                .await
                .expect("metrics response");
            assert_eq!(response.status(), StatusCode::OK);
            let body = body_json(response.into_body()).await;
            if predicate(&body) {
                return body;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("metrics matched predicate")
}

struct StructuredToolDeltaHttpBackend;

#[async_trait]
impl ModelBackend for StructuredToolDeltaHttpBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "structured-tool-delta-http")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Err(BackendError::other(
            "structured tool delta HTTP test must use generate_stream".to_owned(),
        ))
    }

    async fn generate_with_cancel(
        &self,
        _request: BackendRequest,
        _cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        Err(BackendError::other(
            "structured tool delta HTTP test must use generate_stream".to_owned(),
        ))
    }

    fn generate_stream_with_cancel<'a>(
        &'a self,
        _request: BackendRequest,
        cancellation: CancellationToken,
    ) -> futures::stream::BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        if cancellation.is_cancelled() {
            return futures::stream::once(async { Err(BackendError::cancelled()) }).boxed();
        }
        async_stream::try_stream! {
            yield BackendStreamChunk {
                text: String::new(),
                tool_call_deltas: vec![http_structured_tool_delta(
                    0,
                    Some("call_read_1"),
                    Some("read"),
                    Some(r#"{"path":"#),
                )],
                prompt_tokens: 1,
                prompt_cached_tokens: None,
                completion_tokens: 1,
                finish_reason: None,
                progress: None,
            };
            yield BackendStreamChunk {
                text: String::new(),
                tool_call_deltas: vec![http_structured_tool_delta(
                    0,
                    None,
                    None,
                    Some(r#""calculator.py"}"#),
                )],
                prompt_tokens: 1,
                prompt_cached_tokens: None,
                completion_tokens: 1,
                finish_reason: Some(BackendFinishReason::ToolCalls),
                progress: None,
            };
        }
        .boxed()
    }
}

fn http_structured_tool_delta(
    index: u32,
    id: Option<&str>,
    name: Option<&str>,
    arguments: Option<&str>,
) -> BackendToolCallDelta {
    BackendToolCallDelta {
        index,
        id: id.map(str::to_owned),
        call_type: id.map(|_| BackendToolCallType::Function),
        function: (name.is_some() || arguments.is_some()).then(|| BackendToolCallFunctionDelta {
            name: name.map(str::to_owned),
            arguments: arguments.map(str::to_owned),
        }),
    }
}

fn sse_json_frames(body: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).expect("SSE data frame is JSON"))
        .collect()
}
