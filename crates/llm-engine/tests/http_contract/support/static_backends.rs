struct StaticBackend {
    text: String,
}

struct CachedUsageBackend {
    prompt_tokens: u64,
    prompt_cached_tokens: Option<u64>,
    completion_tokens: u64,
}

struct CacheTransitionBackend {
    prompt_tokens: u64,
    warm_cached_tokens: u64,
    completion_tokens: u64,
    calls: AtomicUsize,
}

struct FamilyStaticBackend {
    model_id: &'static str,
    family: &'static str,
    text: &'static str,
}

struct CapabilityBlockingBackend {
    capabilities: BackendCapabilities,
    entered: Arc<Notify>,
    release: Arc<Notify>,
}

#[async_trait]
impl ModelBackend for StaticBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "static")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Ok(BackendOutput {
            text: self.text.clone(),
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

#[async_trait]
impl ModelBackend for CachedUsageBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "cached-usage")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Ok(BackendOutput {
            text: "cached response".to_owned(),
            prompt_tokens: self.prompt_tokens,
            prompt_cached_tokens: self.prompt_cached_tokens,
            completion_tokens: self.completion_tokens,
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

#[async_trait]
impl ModelBackend for CacheTransitionBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "cache-transition")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(BackendOutput {
            text: "cached response".to_owned(),
            prompt_tokens: self.prompt_tokens,
            prompt_cached_tokens: Some(if call == 0 {
                0
            } else {
                self.warm_cached_tokens
            }),
            completion_tokens: self.completion_tokens,
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

#[async_trait]
impl ModelBackend for FamilyStaticBackend {
    fn model_id(&self) -> &str {
        self.model_id
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        BackendModelMetadata::new(self.model_id, "family-static").with_family(self.family)
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Ok(BackendOutput {
            text: self.text.to_owned(),
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

#[async_trait]
impl ModelBackend for CapabilityBlockingBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "capability-blocking")
    }

    fn capabilities(&self) -> BackendCapabilities {
        self.capabilities
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
