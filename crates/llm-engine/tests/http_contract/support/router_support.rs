struct FailingBackend;
struct UnhealthyBackend;
struct PathLeakingBackend;

fn qwen_test_metadata(model_id: &str, backend: &str) -> BackendModelMetadata {
    BackendModelMetadata::new(model_id, backend).with_family("qwen")
}

fn unauthenticated_admin_options() -> EngineOptions {
    EngineOptions::default()
}

fn build_router_with_unauthenticated_admin(backend: Box<dyn ModelBackend>) -> Router {
    build_router_with_unauthenticated_admin_and_options(backend, unauthenticated_admin_options())
        .expect("unauthenticated admin test router builds")
}

fn build_router_with_backend(backend: Box<dyn ModelBackend>) -> Router {
    router_builder(backend).build().expect("test router builds")
}

fn build_router_with_protocol_test_backend() -> Router {
    router_builder(Box::new(
        ProtocolTestBackend::new(
            llm_engine::DEFAULT_MODEL_ID,
            "hello from rust native backend",
        )
        .with_required_tool_protocol()
        .with_json_object_protocol(),
    ))
    .with_options(EngineOptions::default())
    .allow_unauthenticated_admin()
    .build()
    .expect("protocol test router builds")
}

fn build_router_with_backend_and_options(
    backend: Box<dyn ModelBackend>,
    options: EngineOptions,
) -> Result<Router, llm_engine::EngineConfigError> {
    router_builder(backend).with_options(options).build()
}

fn build_router_with_unauthenticated_admin_and_options(
    backend: Box<dyn ModelBackend>,
    options: EngineOptions,
) -> Result<Router, llm_engine::EngineConfigError> {
    build_router_with_backend_and_options_allowing_unauthenticated_admin(backend, options)
}

fn build_router_with_backend_and_options_allowing_unauthenticated_admin(
    backend: Box<dyn ModelBackend>,
    options: EngineOptions,
) -> Result<Router, llm_engine::EngineConfigError> {
    router_builder(backend)
        .with_options(options)
        .allow_unauthenticated_admin()
        .build()
}

#[async_trait]
impl ModelBackend for FailingBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "failing")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Err(BackendError::other("execution failed".to_owned()))
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
impl ModelBackend for UnhealthyBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "unhealthy")
    }

    async fn health(&self) -> BackendHealth {
        BackendHealth::unavailable("backend is offline")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Err(BackendError::other("backend is offline".to_owned()))
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
impl ModelBackend for PathLeakingBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        qwen_test_metadata(self.model_id(), "path-leaking")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Err(BackendError::other(
            "failed to open /data/kir-ai/private/model.safetensors".to_owned(),
        ))
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        generate_after_pre_cancel(self, request, cancellation).await
    }
}
