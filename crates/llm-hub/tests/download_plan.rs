use llm_hub::{ArtifactClass, HubFile, HubRepoId, ModelProfile, build_download_plan};

#[test]
fn qwen_mlx_profile_selects_static_artifacts_and_weights() {
    let files = vec![
        HubFile::new("config.json", 100, Some("\"cfg\"")),
        HubFile::new("tokenizer.json", 200, Some("\"tok\"")),
        HubFile::new("model.safetensors", 1_000, Some("\"weights\"")),
        HubFile::new("optimizer.pt", 10_000, Some("\"opt\"")),
    ];

    let plan = build_download_plan(
        HubRepoId::model("mlx-community/Qwen3.6-35B-A3B-4bit").expect("repo id"),
        "main",
        "0123456789abcdef0123456789abcdef01234567",
        ModelProfile::qwen36_mlx_4bit(),
        files,
        &[],
    )
    .expect("plan builds");

    assert_eq!(plan.files_to_download.len(), 3);
    assert_eq!(plan.skipped_files, vec!["optimizer.pt"]);
    assert_eq!(plan.total_bytes_to_download, 1_300);
    assert_eq!(plan.files_to_download[0].class, ArtifactClass::Config);
    assert_eq!(plan.files_to_download[1].class, ArtifactClass::Tokenizer);
    assert_eq!(plan.files_to_download[2].class, ArtifactClass::Weights);
    assert_eq!(plan.repo_id.as_str(), "mlx-community/Qwen3.6-35B-A3B-4bit");
}

#[test]
fn plan_rejects_mutable_revision_without_resolved_commit() {
    let err = build_download_plan(
        HubRepoId::model("Qwen/Qwen3.6-35B-A3B").expect("repo id"),
        "main",
        "main",
        ModelProfile::qwen36_mlx_4bit(),
        vec![],
        &[],
    )
    .expect_err("mutable commit identity must fail closed");

    assert_eq!(err.code(), "model_revision_unresolved");
}
