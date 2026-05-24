struct MetadataBackend;

#[async_trait]
impl ModelBackend for MetadataBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        let mut metadata = BackendModelMetadata::new(llm_engine::DEFAULT_MODEL_ID, "native-qwen")
            .with_family("qwen");
        metadata.quantization = Some("bf16".to_owned());
        metadata.repo_id = Some("Qwen/Qwen3.6-35B-A3B".to_owned());
        metadata.resolved_commit = Some("0123456789abcdef0123456789abcdef01234567".to_owned());
        metadata.profile = Some("qwen36-safetensors-bf16".to_owned());
        metadata
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Ok(BackendOutput {
            text: "metadata".to_owned(),
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

struct MlxMetadataBackend;

#[async_trait]
impl ModelBackend for MlxMetadataBackend {
    fn model_id(&self) -> &str {
        "local-qwen36-mlx"
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        let mut metadata = BackendModelMetadata::new("local-qwen36-mlx", "mlx").with_family("qwen");
        metadata.quantization = Some("4bit".to_owned());
        metadata.repo_id = Some("mlx-community/Qwen3.6-35B-A3B-4bit".to_owned());
        metadata.resolved_commit = Some("0123456789abcdef0123456789abcdef01234567".to_owned());
        metadata.profile = Some("qwen36-mlx-4bit".to_owned());
        metadata
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Ok(BackendOutput {
            text: "mlx metadata".to_owned(),
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

struct SnapshotMetadataBackend;

#[async_trait]
impl ModelBackend for SnapshotMetadataBackend {
    fn model_id(&self) -> &str {
        llm_engine::DEFAULT_MODEL_ID
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        let mut metadata = BackendModelMetadata::new(llm_engine::DEFAULT_MODEL_ID, "native-qwen")
            .with_family("qwen");
        metadata.quantization = Some("bf16".to_owned());
        metadata.repo_id = Some("Qwen/Qwen3.6-35B-A3B".to_owned());
        metadata.resolved_commit = Some("0123456789abcdef0123456789abcdef01234567".to_owned());
        metadata.profile = Some("qwen36-safetensors-bf16".to_owned());
        metadata
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Ok(BackendOutput {
            text: "metadata".to_owned(),
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

async fn generate_after_pre_cancel<B: ModelBackend + ?Sized>(
    backend: &B,
    request: BackendRequest,
    cancellation: CancellationToken,
) -> Result<BackendOutput, BackendError> {
    if cancellation.is_cancelled() {
        return Err(BackendError::cancelled());
    }
    backend.generate(request).await
}

fn runnable_qwen_files() -> Vec<HubFile> {
    vec![
        HubFile::new("config.json", 2, Some("\"cfg\"")),
        HubFile::new("tokenizer.json", 2, Some("\"tok\"")),
        HubFile::new(
            "model.safetensors",
            4,
            Some("3a6eb0790f39ac87c94f3856b2dd2c5d110e6811602261a9a923d3bb23adc8b7"),
        ),
    ]
}

async fn write_runnable_qwen_files(snapshot_path: &Path) {
    tokio::fs::write(snapshot_path.join("config.json"), "{}")
        .await
        .expect("config");
    tokio::fs::write(snapshot_path.join("tokenizer.json"), "{}")
        .await
        .expect("tokenizer");
    tokio::fs::write(snapshot_path.join("model.safetensors"), b"data")
        .await
        .expect("weights");
}

async fn write_verified_test_snapshot(root: &Path) -> PathBuf {
    write_verified_test_snapshot_with_profile(
        root,
        "Qwen/Qwen3.6-35B-A3B",
        ModelProfile::qwen36_safetensors_bf16(),
    )
    .await
}

async fn write_verified_mlx_test_snapshot(root: &Path) -> PathBuf {
    write_verified_test_snapshot_with_profile(
        root,
        "mlx-community/Qwen3.6-35B-A3B-4bit",
        ModelProfile::qwen36_mlx_4bit(),
    )
    .await
}

async fn write_verified_runnable_test_snapshot(root: &Path) -> PathBuf {
    let store = ModelStore::new(root);
    let plan = build_download_plan(
        HubRepoId::model("Qwen/Qwen3.6-35B-A3B").expect("repo id"),
        "main",
        "0123456789abcdef0123456789abcdef01234567",
        ModelProfile::qwen36_safetensors_bf16(),
        runnable_qwen_files(),
        &[],
    )
    .expect("plan builds");
    let snapshot_path = store.snapshot_path(&plan);
    tokio::fs::create_dir_all(&snapshot_path)
        .await
        .expect("snapshot dir");
    write_runnable_qwen_files(&snapshot_path).await;
    store
        .verify_existing_snapshot(&plan)
        .await
        .expect("snapshot verifies");
    snapshot_path
}

async fn write_verified_metadata_only_test_snapshot(root: &Path) -> PathBuf {
    let store = ModelStore::new(root);
    let full_plan = build_download_plan(
        HubRepoId::model("Qwen/Qwen3.6-35B-A3B").expect("repo id"),
        "main",
        "0123456789abcdef0123456789abcdef01234567",
        ModelProfile::qwen36_safetensors_bf16(),
        runnable_qwen_files(),
        &[],
    )
    .expect("plan builds");
    let metadata_plan = full_plan.metadata_only();
    let snapshot_path = store.snapshot_path(&metadata_plan);
    tokio::fs::create_dir_all(&snapshot_path)
        .await
        .expect("snapshot dir");
    tokio::fs::write(snapshot_path.join("config.json"), "{}")
        .await
        .expect("config");
    tokio::fs::write(snapshot_path.join("tokenizer.json"), "{}")
        .await
        .expect("tokenizer");
    store
        .verify_existing_snapshot(&metadata_plan)
        .await
        .expect("snapshot verifies");
    snapshot_path
}

async fn write_verified_test_snapshot_with_profile(
    root: &Path,
    repo_id: &str,
    profile: ModelProfile,
) -> PathBuf {
    let store = ModelStore::new(root);
    let plan = build_download_plan(
        HubRepoId::model(repo_id).expect("repo id"),
        "main",
        "0123456789abcdef0123456789abcdef01234567",
        profile,
        vec![HubFile::new("config.json", 2, Some("\"cfg\""))],
        &[],
    )
    .expect("plan builds");
    let snapshot_path = store.snapshot_path(&plan);
    tokio::fs::create_dir_all(&snapshot_path)
        .await
        .expect("snapshot dir");
    tokio::fs::write(snapshot_path.join("config.json"), "{}")
        .await
        .expect("config");
    store
        .verify_existing_snapshot(&plan)
        .await
        .expect("snapshot verifies");
    snapshot_path
}
