use llm_hub::{
    ArtifactClass, HubFile, HubModelInfo, HubRepoId, ModelProfile, SnapshotManifest,
    build_download_plan,
};
use llm_models::{BackendKind, ModelFamily};
use serde_json::json;

#[test]
fn qwen_mlx_profile_selects_static_artifacts_and_weights() {
    let files = vec![
        HubFile::new("config.json", 100, Some("\"cfg\"")),
        HubFile::new("tokenizer.json", 200, Some("\"tok\"")),
        HubFile::new("model.safetensors", 1_000, Some("\"weights\"")),
        HubFile::new("image_processor_config.json", 300, Some("\"image\"")),
        HubFile::new("processor_config.json", 400, Some("\"processor\"")),
        HubFile::new("video_preprocessor_config.json", 500, Some("\"video-pre\"")),
        HubFile::new("vision_tower.safetensors", 600, Some("\"vision\"")),
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
    assert_eq!(
        plan.skipped_files,
        vec![
            "image_processor_config.json",
            "optimizer.pt",
            "processor_config.json",
            "video_preprocessor_config.json",
            "vision_tower.safetensors",
        ]
    );
    assert_eq!(plan.total_bytes_to_download, 1_300);
    assert_eq!(plan.files_to_download[0].class, ArtifactClass::Config);
    assert_eq!(plan.files_to_download[1].class, ArtifactClass::Tokenizer);
    assert_eq!(plan.files_to_download[2].class, ArtifactClass::Weights);
    assert_eq!(plan.repo_id.as_str(), "mlx-community/Qwen3.6-35B-A3B-4bit");
}

#[test]
fn qwen35_4b_mlx_profiles_record_practical_chat_quant_identities() {
    for (profile, name, quantization) in [
        (
            ModelProfile::qwen35_4b_mlx_4bit(),
            "qwen35-4b-mlx-4bit",
            "4bit",
        ),
        (
            ModelProfile::qwen35_4b_mlx_8bit(),
            "qwen35-4b-mlx-8bit",
            "8bit",
        ),
        (
            ModelProfile::qwen35_4b_mlx_optiq_4bit(),
            "qwen35-4b-mlx-optiq-4bit",
            "optiq-4bit",
        ),
    ] {
        assert_eq!(profile.name, name);
        assert_eq!(profile.family, ModelFamily::Qwen);
        assert_eq!(profile.loader, BackendKind::Mlx);
        assert_eq!(profile.quantization, quantization);
        assert!(profile.allow_patterns.contains(&"*.safetensors".to_owned()));
    }
}

#[test]
fn qwen_optiq_profile_selects_quantization_metadata() {
    let files = vec![
        HubFile::new("config.json", 100, Some("\"cfg\"")),
        HubFile::new("tokenizer.json", 200, Some("\"tok\"")),
        HubFile::new("optiq_metadata.json", 300, Some("\"optiq\"")),
        HubFile::new("model.safetensors", 1_000, Some("\"weights\"")),
        HubFile::new("optimizer.pt", 10_000, Some("\"opt\"")),
    ];

    let plan = build_download_plan(
        HubRepoId::model("mlx-community/Qwen3.5-4B-OptiQ-4bit").expect("repo id"),
        "main",
        "0123456789abcdef0123456789abcdef01234567",
        ModelProfile::qwen35_4b_mlx_optiq_4bit(),
        files,
        &[],
    )
    .expect("plan builds");

    assert_eq!(plan.files_to_download.len(), 4);
    assert_eq!(plan.skipped_files, vec!["optimizer.pt"]);
    assert_eq!(plan.total_bytes_to_download, 1_600);
    assert_eq!(plan.files_to_download[0].class, ArtifactClass::Config);
    assert_eq!(plan.files_to_download[1].class, ArtifactClass::Tokenizer);
    assert_eq!(plan.files_to_download[2].class, ArtifactClass::Quantization);
    assert_eq!(plan.files_to_download[3].class, ArtifactClass::Weights);
}

#[test]
fn qwen_bf16_profile_records_official_safetensors_identity() {
    let profile = ModelProfile::qwen36_safetensors_bf16();

    assert_eq!(profile.name, "qwen36-safetensors-bf16");
    assert_eq!(profile.loader, BackendKind::NativeMetal);
    assert_eq!(profile.quantization, "bf16");
    assert!(profile.allow_patterns.contains(&"*.safetensors".to_owned()));
}

#[test]
fn qwen3_dense_bf16_profile_selects_small_native_snapshots() {
    let files = vec![
        HubFile::new("config.json", 100, Some("\"cfg\"")),
        HubFile::new("tokenizer.json", 200, Some("\"tok\"")),
        HubFile::new("model.safetensors", 1_000, Some("\"weights\"")),
        HubFile::new("optimizer.pt", 10_000, Some("\"opt\"")),
    ];

    let plan = build_download_plan(
        HubRepoId::model("Qwen/Qwen3-0.6B").expect("repo id"),
        "main",
        "0123456789abcdef0123456789abcdef01234567",
        ModelProfile::qwen3_dense_safetensors_bf16(),
        files,
        &[],
    )
    .expect("plan builds");

    assert_eq!(plan.profile.name, "qwen3-dense-safetensors-bf16");
    assert_eq!(plan.profile.loader, BackendKind::NativeMetal);
    assert_eq!(plan.profile.quantization, "bf16");
    assert_eq!(plan.files_to_download.len(), 3);
    assert_eq!(plan.skipped_files, vec!["optimizer.pt"]);
}

#[test]
fn gemma_text_profile_skips_multimodal_artifacts() {
    let files = vec![
        HubFile::new("config.json", 100, Some("\"cfg\"")),
        HubFile::new("chat_template.jinja", 300, Some("\"template\"")),
        HubFile::new("tokenizer.json", 200, Some("\"tok\"")),
        HubFile::new("model-00001-of-00002.safetensors", 1_000, Some("\"w1\"")),
        HubFile::new("vision_tower.safetensors", 2_000, Some("\"vision\"")),
        HubFile::new("mm_projector.safetensors", 3_000, Some("\"projector\"")),
        HubFile::new("image_processor_config.json", 400, Some("\"image\"")),
        HubFile::new("preprocessor_config.json", 500, Some("\"pre\"")),
        HubFile::new("video_preprocessor_config.json", 600, Some("\"video-pre\"")),
    ];

    let plan = build_download_plan(
        HubRepoId::model("google/gemma-4-31b-it").expect("repo id"),
        "main",
        "0123456789abcdef0123456789abcdef01234567",
        ModelProfile::gemma4_text_safetensors_bf16(),
        files,
        &[],
    )
    .expect("plan builds");

    assert_eq!(plan.profile.family, ModelFamily::Gemma);
    assert_eq!(plan.profile.loader, BackendKind::NativeMetal);
    assert_eq!(plan.profile.quantization, "bf16");
    assert_eq!(plan.files_to_download.len(), 4);
    assert_eq!(plan.total_bytes_to_download, 1_600);
    assert_eq!(
        plan.skipped_files,
        vec![
            "image_processor_config.json",
            "mm_projector.safetensors",
            "preprocessor_config.json",
            "video_preprocessor_config.json",
            "vision_tower.safetensors",
        ]
    );
}

#[test]
fn gemma_text_profile_uses_native_loader_metadata() {
    let profile = ModelProfile::gemma4_text_safetensors_bf16();

    assert_eq!(profile.family, ModelFamily::Gemma);
    assert_eq!(profile.loader, BackendKind::NativeMetal);
    assert_eq!(profile.quantization, "bf16");
}

#[test]
fn gemma4_e2b_mlx_4bit_profile_records_practical_chat_identity() {
    let profile = ModelProfile::gemma4_e2b_it_mlx_4bit();

    assert_eq!(profile.name, "gemma4-e2b-it-mlx-4bit");
    assert_eq!(profile.family, ModelFamily::Gemma);
    assert_eq!(profile.loader, BackendKind::Mlx);
    assert_eq!(profile.quantization, "4bit");
    assert!(profile.allow_patterns.contains(&"*.safetensors".to_owned()));
    assert!(
        profile
            .ignore_patterns
            .contains(&"processor_config.json".to_owned())
    );
}

#[test]
fn llama32_mlx_profile_records_practical_chat_identity() {
    let profile = ModelProfile::llama32_3b_instruct_mlx_4bit();

    assert_eq!(profile.name, "llama32-3b-instruct-mlx-4bit");
    assert_eq!(profile.family, ModelFamily::Llama);
    assert_eq!(profile.loader, BackendKind::Mlx);
    assert_eq!(profile.quantization, "4bit");
    assert!(profile.allow_patterns.contains(&"*.safetensors".to_owned()));
    assert!(
        profile
            .ignore_patterns
            .contains(&"processor_config.json".to_owned())
    );
}

#[test]
fn llama_text_profile_selects_text_artifacts_and_weights() {
    let files = vec![
        HubFile::new("config.json", 100, Some("\"cfg\"")),
        HubFile::new("tokenizer.json", 200, Some("\"tok\"")),
        HubFile::new("tokenizer_config.json", 250, Some("\"tok-cfg\"")),
        HubFile::new("model.safetensors", 1_000, Some("\"weights\"")),
        HubFile::new("preprocessor_config.json", 300, Some("\"pre\"")),
        HubFile::new("vision_tower.safetensors", 600, Some("\"vision\"")),
        HubFile::new("optimizer.pt", 10_000, Some("\"opt\"")),
    ];

    let plan = build_download_plan(
        HubRepoId::model("mlx-community/Llama-3.2-3B-Instruct-4bit").expect("repo id"),
        "main",
        "0123456789abcdef0123456789abcdef01234567",
        ModelProfile::llama32_3b_instruct_mlx_4bit(),
        files,
        &[],
    )
    .expect("plan builds");

    assert_eq!(plan.profile.family, ModelFamily::Llama);
    assert_eq!(plan.profile.loader, BackendKind::Mlx);
    assert_eq!(plan.files_to_download.len(), 4);
    assert_eq!(plan.total_bytes_to_download, 1_550);
    assert_eq!(
        plan.skipped_files,
        vec![
            "optimizer.pt",
            "preprocessor_config.json",
            "vision_tower.safetensors",
        ]
    );
}

#[test]
fn builtin_profile_lookup_includes_all_supported_profiles() {
    for name in [
        "gemma4-e2b-it-mlx-4bit",
        "gemma4-text-safetensors-bf16",
        "llama32-3b-instruct-mlx-4bit",
        "qwen35-4b-mlx-4bit",
        "qwen35-4b-mlx-8bit",
        "qwen35-4b-mlx-optiq-4bit",
        "qwen3-dense-safetensors-bf16",
        "qwen36-safetensors-bf16",
        "qwen36-mlx-4bit",
    ] {
        let profile = ModelProfile::builtin(name).expect("profile exists");

        assert_eq!(profile.name, name);
    }

    assert!(ModelProfile::builtin("missing-profile").is_none());
}

#[test]
fn builtin_profile_names_match_lookup_table() {
    for name in ModelProfile::builtin_names() {
        let profile = ModelProfile::builtin(name).expect("profile exists");

        assert_eq!(profile.name, name);
    }
}

#[test]
fn builtin_profile_fields_are_typed_but_serialize_as_profile_slugs() {
    let profile = ModelProfile::qwen36_safetensors_bf16();

    assert_eq!(profile.family, ModelFamily::Qwen);
    assert_eq!(profile.loader, BackendKind::NativeMetal);

    let value = serde_json::to_value(&profile).expect("profile serializes");
    assert_eq!(value["family"], "qwen");
    assert_eq!(value["loader"], "native-metal");
    assert_eq!(value["quantization"], "bf16");

    let decoded: ModelProfile = serde_json::from_value(value).expect("profile deserializes");
    assert_eq!(decoded.family, ModelFamily::Qwen);
    assert_eq!(decoded.loader, BackendKind::NativeMetal);
    assert_eq!(decoded.quantization, "bf16");

    let decoded_alias: ModelProfile = serde_json::from_value(json!({
        "name": "qwen36-safetensors-bf16",
        "family": "qwen",
        "loader": "native_metal",
        "quantization": "bf16",
        "allow_patterns": ["*.safetensors"],
        "ignore_patterns": []
    }))
    .expect("profile deserializes legacy loader alias");
    assert_eq!(decoded_alias.family, ModelFamily::Qwen);
    assert_eq!(decoded_alias.loader, BackendKind::NativeMetal);
}

#[test]
fn builtin_profiles_are_compatible_with_model_family_backend_support() {
    for name in ModelProfile::builtin_names() {
        let profile = ModelProfile::builtin(name)
            .unwrap_or_else(|| panic!("profile `{name}` must exist in the built-in lookup table"));

        assert_eq!(
            profile.name, name,
            "profile `{name}` lookup returned mismatched profile name `{}`",
            profile.name
        );
        assert_profile_field_is_known(name, "quantization", &profile.quantization);
        assert!(
            profile
                .allow_patterns
                .iter()
                .any(|pattern| pattern == "*.safetensors"),
            "profile `{name}` must allow safetensors weights for planning; allow patterns: {:?}",
            profile.allow_patterns
        );

        let family = profile.family;
        let backend = profile.loader;
        let supported_backends = family.adapter().production_backends();

        assert!(
            family.supports_backend(backend),
            "profile `{name}` declares loader `{backend}` for family `{family}`, but supported loaders are: {}",
            backend_list(supported_backends)
        );

        let plan = build_download_plan(
            HubRepoId::model(format!("tests/{name}")).unwrap_or_else(|err| {
                panic!("profile `{name}` test repo id should be safe: {err}")
            }),
            "main",
            "0123456789abcdef0123456789abcdef01234567",
            profile,
            representative_profile_files(),
            &[],
        )
        .unwrap_or_else(|err| panic!("profile `{name}` should build a representative plan: {err}"));

        assert_eq!(
            plan.profile.name, name,
            "profile `{name}` plan recorded mismatched profile name `{}`",
            plan.profile.name
        );
        assert!(
            plan.files_to_download
                .iter()
                .any(|file| file.class == ArtifactClass::Weights),
            "profile `{name}` representative plan must include weights; files: {:?}",
            plan.files_to_download
        );
    }
}

fn assert_profile_field_is_known(profile: &str, field: &str, value: &str) {
    const KNOWN_QUANTIZATIONS: &[&str] = &["4bit", "8bit", "optiq-4bit", "bf16"];

    let known_values = match field {
        "quantization" => KNOWN_QUANTIZATIONS,
        _ => unreachable!("test only validates explicit profile fields"),
    };

    assert!(
        known_values.contains(&value),
        "profile `{profile}` declares unknown {field} `{value}`; known values: {}",
        known_values.join(", ")
    );
}

fn backend_list(backends: &[BackendKind]) -> String {
    backends
        .iter()
        .map(|backend| backend.canonical_slug())
        .collect::<Vec<_>>()
        .join(", ")
}

fn representative_profile_files() -> Vec<HubFile> {
    vec![
        HubFile::new("config.json", 100, Some("\"cfg\"")),
        HubFile::new("tokenizer.json", 200, Some("\"tok\"")),
        HubFile::new("model.safetensors", 1_000, Some("\"weights\"")),
        HubFile::new("processor_config.json", 400, Some("\"processor\"")),
    ]
}

#[test]
fn repo_id_rejects_ambiguous_or_unsafe_components() {
    for repo_id in [
        "Qwen//Qwen3.6-35B-A3B",
        "Qwen/../Qwen3.6-35B-A3B",
        "Qwen/./Qwen3.6-35B-A3B",
        "Qwen/Qwen3.6-35B-A3B/extra",
        "Qwen/Qwen3.6-35B-A3B\n",
    ] {
        let err = HubRepoId::model(repo_id).expect_err("unsafe repo id fails closed");

        assert_eq!(err.code(), "invalid_request");
    }
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
    let lfs_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    let info = HubModelInfo::from_api_json(json!({
        "id": "Qwen/Qwen3.6-35B-A3B",
        "sha": "53c43178507d69762986fbfa314f6e8d4d859409",
        "siblings": [
            {"rfilename": "config.json", "size": 3690},
            {
                "rfilename": "model-00001-of-00026.safetensors",
                "blobId": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "lfs": {"size": 4_294_967_296_u64, "oid": lfs_sha256}
            }
        ]
    }))
    .expect("hf model info parses");

    assert_eq!(
        info.resolved_commit,
        "53c43178507d69762986fbfa314f6e8d4d859409"
    );
    assert_eq!(info.files[0].path, "config.json");
    assert_eq!(info.files[1].size, 4_294_967_296);
    assert_eq!(info.files[1].etag.as_deref(), Some(lfs_sha256));

    let plan = build_download_plan(
        HubRepoId::model("Qwen/Qwen3.6-35B-A3B").expect("repo id"),
        "main",
        &info.resolved_commit,
        ModelProfile::qwen36_mlx_4bit(),
        info.files,
        &[],
    )
    .expect("plan builds with LFS sha256");
    let weights = plan
        .files_to_download
        .iter()
        .find(|file| file.path.ends_with(".safetensors"))
        .expect("weight file is planned");
    assert_eq!(weights.sha256.as_deref(), Some(lfs_sha256));
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

#[test]
fn plan_rejects_unsafe_hub_artifact_paths() {
    for path in [
        "../config.json",
        "/tmp/config.json",
        "nested/../config.json",
        "",
    ] {
        let err = build_download_plan(
            HubRepoId::model("Qwen/Qwen3.6-35B-A3B").expect("repo id"),
            "main",
            "0123456789abcdef0123456789abcdef01234567",
            ModelProfile::qwen36_mlx_4bit(),
            vec![HubFile::new(path, 100, Some("\"cfg\""))],
            &[],
        )
        .expect_err("unsafe artifact path fails closed");

        assert_eq!(err.code(), "invalid_request");
    }
}

#[test]
fn manifest_records_lfs_sha256_identity() {
    let sha = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    let plan = build_download_plan(
        HubRepoId::model("Qwen/Qwen3.6-35B-A3B").expect("repo id"),
        "main",
        "0123456789abcdef0123456789abcdef01234567",
        ModelProfile::qwen36_mlx_4bit(),
        vec![HubFile::new("config.json", 2, Some(sha))],
        &[],
    )
    .expect("plan builds");

    assert_eq!(plan.files_to_download[0].sha256.as_deref(), Some(sha));
    let manifest = SnapshotManifest::from_plan(&plan, "/snapshot");
    assert_eq!(manifest.files[0].sha256.as_deref(), Some(sha));
}
