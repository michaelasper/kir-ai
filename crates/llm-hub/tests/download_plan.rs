use llm_hub::{
    ArtifactClass, HubFile, HubModelInfo, HubRepoId, ModelProfile, SnapshotManifest,
    build_download_plan,
};
use serde_json::json;

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

#[test]
fn parses_hugging_face_model_info_with_lfs_sizes() {
    let info = HubModelInfo::from_api_json(json!({
        "id": "Qwen/Qwen3.6-35B-A3B",
        "sha": "53c43178507d69762986fbfa314f6e8d4d859409",
        "siblings": [
            {"rfilename": "config.json", "size": 3690},
            {"rfilename": "model-00001-of-00026.safetensors", "lfs": {"size": 4_294_967_296_u64, "oid": "abc"}}
        ]
    }))
    .expect("hf model info parses");

    assert_eq!(
        info.resolved_commit,
        "53c43178507d69762986fbfa314f6e8d4d859409"
    );
    assert_eq!(info.files[0].path, "config.json");
    assert_eq!(info.files[1].size, 4_294_967_296);
    assert_eq!(info.files[1].etag.as_deref(), Some("abc"));
}

#[test]
fn manifest_digest_changes_with_artifact_identity() {
    let plan = build_download_plan(
        HubRepoId::model("mlx-community/Qwen3.6-35B-A3B-4bit").expect("repo id"),
        "main",
        "0123456789abcdef0123456789abcdef01234567",
        ModelProfile::qwen36_mlx_4bit(),
        vec![HubFile::new("config.json", 100, Some("\"cfg\""))],
        &[],
    )
    .expect("plan builds");

    let manifest = SnapshotManifest::from_plan(&plan, "/models/qwen/snapshots/0123");
    assert_eq!(manifest.source, "huggingface");
    assert_eq!(manifest.family, "qwen");
    assert_eq!(manifest.files.len(), 1);
    assert_eq!(manifest.digest().len(), 64);
}

#[test]
fn metadata_only_plan_excludes_weight_files() {
    let plan = build_download_plan(
        HubRepoId::model("Qwen/Qwen3.6-35B-A3B").expect("repo id"),
        "main",
        "0123456789abcdef0123456789abcdef01234567",
        ModelProfile::qwen36_mlx_4bit(),
        vec![
            HubFile::new("config.json", 100, Some("\"cfg\"")),
            HubFile::new("model.safetensors", 1_000, Some("\"weights\"")),
        ],
        &[],
    )
    .expect("plan builds");

    let metadata = plan.metadata_only();
    assert_eq!(metadata.files_to_download.len(), 1);
    assert_eq!(metadata.files_to_download[0].path, "config.json");
    assert_eq!(metadata.total_bytes_to_download, 100);
}
