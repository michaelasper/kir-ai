use super::*;

#[tokio::test]
async fn qwen_mlx_tool_normalized_raw_hf_snapshot_identity_does_not_require_kir_manifest() {
    let temp = tempfile::tempdir().expect("tempdir");
    let snapshot = temp
        .path()
        .join("huggingface")
        .join("models--mlx-community--Qwen3.6-35B-A3B-4bit")
        .join("snapshots")
        .join("abcdef1234567890");
    tokio::fs::create_dir_all(&snapshot)
        .await
        .expect("raw snapshot dir");
    tokio::fs::write(snapshot.join("config.json"), "{}")
        .await
        .expect("config");

    let lane = lane(&format!(
        "name=direct,endpoint=http://127.0.0.1:8080/v1,model=qwen-loaded,snapshot={},kind=direct_mlx",
        snapshot.display()
    ));
    let identity = load_lane_snapshot_identity(&lane, false)
        .await
        .expect("raw HF snapshot identity should not require llm-engine-manifest.json")
        .expect("snapshot identity");

    let snapshot_display = snapshot.display().to_string();
    assert_eq!(identity.id, snapshot_display);
    assert_eq!(
        identity.snapshot_path.as_deref(),
        Some(snapshot_display.as_str())
    );
    assert_eq!(
        identity.repo_id.as_deref(),
        Some("mlx-community/Qwen3.6-35B-A3B-4bit")
    );
    assert_eq!(
        identity.resolved_commit.as_deref(),
        Some("abcdef1234567890")
    );
    assert_eq!(identity.manifest_digest, None);
}

#[test]
fn qwen_mlx_tool_normalized_probe_plan_expands_schema_and_tool_choice_variants() {
    let probes = NormalizedProbePlan::all();

    assert_eq!(probes.len(), 25);
    assert_eq!(
        probes
            .iter()
            .filter(|probe| probe.case == NormalizedCaseKind::JsonObject)
            .collect::<Vec<_>>(),
        vec![&NormalizedProbePlan::new(
            NormalizedCaseKind::JsonObject,
            SchemaVariant::None,
            ToolChoiceVariant::None,
        )]
    );
    assert_eq!(
        probes
            .iter()
            .filter(|probe| {
                probe.case == NormalizedCaseKind::OmpRepeatedPrefix
                    && probe.schema_variant == SchemaVariant::CanonicalPermutedEquivalent
                    && probe.tool_choice_variant == ToolChoiceVariant::Function
            })
            .count(),
        1
    );
    assert!(
        probes.iter().any(|probe| {
            probe.case == NormalizedCaseKind::ToolRequiredStream
                && probe.schema_variant == SchemaVariant::BaselineCurrent
                && probe.tool_choice_variant == ToolChoiceVariant::Required
        }),
        "streamed tool probes should participate in the schema/tool-choice matrix"
    );
}

#[test]
fn qwen_mlx_tool_normalized_canonical_and_permuted_schema_hashes_capture_equivalence() {
    let baseline = tool_schema_metadata(NormalizedProbePlan::new(
        NormalizedCaseKind::ToolRequired,
        SchemaVariant::BaselineCurrent,
        ToolChoiceVariant::Required,
    ));
    let baseline_permuted = tool_schema_metadata(NormalizedProbePlan::new(
        NormalizedCaseKind::ToolRequired,
        SchemaVariant::BaselinePermutedEquivalent,
        ToolChoiceVariant::Required,
    ));
    let canonical = tool_schema_metadata(NormalizedProbePlan::new(
        NormalizedCaseKind::ToolRequired,
        SchemaVariant::CanonicalCurrent,
        ToolChoiceVariant::Required,
    ));
    let canonical_permuted = tool_schema_metadata(NormalizedProbePlan::new(
        NormalizedCaseKind::ToolRequired,
        SchemaVariant::CanonicalPermutedEquivalent,
        ToolChoiceVariant::Required,
    ));

    assert_ne!(baseline.sha256, baseline_permuted.sha256);
    assert_eq!(canonical.sha256, canonical_permuted.sha256);
    assert_ne!(baseline.sha256, canonical.sha256);
    assert!(canonical.bytes.expect("canonical bytes") > 0);
}

#[test]
fn qwen_mlx_tool_normalized_request_bodies_cover_tool_stream_and_json_with_default_no_thinking_kwargs()
 {
    let lane = lane("name=direct,endpoint=http://127.0.0.1:8080/v1,model=qwen,kind=direct_mlx");

    let tool = probe_request_body(
        &lane,
        NormalizedProbePlan::new(
            NormalizedCaseKind::ToolRequired,
            SchemaVariant::BaselineCurrent,
            ToolChoiceVariant::Required,
        ),
        ProbePrompt::measured(128, 0, None),
    );
    assert_eq!(tool["model"], "qwen");
    assert_eq!(tool["max_tokens"], 512);
    assert_eq!(tool["tool_choice"], "required");
    assert_eq!(
        tool["tools"][0]["function"]["name"],
        "record_qwen_tool_probe"
    );
    assert_eq!(tool["chat_template_kwargs"]["enable_thinking"], false);
    assert!(tool.get("stream").is_none());

    let stream = probe_request_body(
        &lane,
        NormalizedProbePlan::new(
            NormalizedCaseKind::ToolRequiredStream,
            SchemaVariant::CanonicalPermutedEquivalent,
            ToolChoiceVariant::Function,
        ),
        ProbePrompt::measured(128, 0, None),
    );
    assert_eq!(stream["max_tokens"], 512);
    assert_eq!(stream["stream"], true);
    assert_eq!(stream["stream_options"]["include_usage"], true);
    assert_eq!(stream["chat_template_kwargs"]["enable_thinking"], false);
    assert_eq!(
        stream["tool_choice"],
        json!({"type":"function","function":{"name":"record_qwen_tool_probe"}})
    );
    assert_eq!(
        stream["tools"][0]["function"]["parameters"]["required"],
        json!(["case", "probe_id"])
    );

    let json_body = probe_request_body(
        &lane,
        NormalizedProbePlan::new(
            NormalizedCaseKind::JsonObject,
            SchemaVariant::None,
            ToolChoiceVariant::None,
        ),
        ProbePrompt::measured(128, 0, None),
    );
    assert_eq!(json_body["response_format"]["type"], "json_object");
    assert_eq!(json_body["chat_template_kwargs"]["enable_thinking"], false);
    assert!(
        json_body["messages"]
            .as_array()
            .expect("messages array")
            .iter()
            .any(|message| message["content"]
                .as_str()
                .unwrap_or_default()
                .contains("KIR_QWEN_MLX_TOOL_NORMALIZED_JSON_OBJECT"))
    );

    let synthetic = probe_request_body(
        &lane,
        NormalizedProbePlan::new(
            NormalizedCaseKind::OmpRepeatedPrefix,
            SchemaVariant::BaselinePermutedEquivalent,
            ToolChoiceVariant::Function,
        ),
        ProbePrompt::measured(512, 7, Some(2)),
    );
    let messages = synthetic["messages"].as_array().expect("OMP messages");
    assert_eq!(
        messages
            .iter()
            .map(|message| message["role"].as_str().expect("message role"))
            .collect::<Vec<_>>(),
        ["system", "user", "assistant", "tool", "user"]
    );
    assert_eq!(messages[2]["tool_calls"][0]["type"], "function");
    assert_eq!(
        messages[2]["tool_calls"][0]["function"]["name"],
        "record_qwen_tool_probe"
    );
    assert_eq!(
        messages[3]["tool_call_id"],
        messages[2]["tool_calls"][0]["id"]
    );
    let final_user = messages[4]["content"].as_str().expect("final OMP user");
    assert!(final_user.contains("OMP final delta"));
    assert!(final_user.contains("sample=7 request=2"));
    assert_eq!(
        synthetic["tool_choice"],
        json!({"type":"function","function":{"name":"record_qwen_tool_probe"}})
    );
}

#[test]
fn qwen_mlx_tool_normalized_prefill_sweep_stream_bodies_use_expected_tools_and_markers() {
    let lane = lane("name=direct,endpoint=http://127.0.0.1:8080/v1,model=qwen,kind=direct_mlx");

    let chat = probe_request_body(
        &lane,
        NormalizedProbePlan::new(
            NormalizedCaseKind::ChatStream,
            SchemaVariant::None,
            ToolChoiceVariant::None,
        ),
        ProbePrompt::measured(128, 0, None),
    );
    assert_eq!(chat["stream"], true);
    assert_eq!(chat["stream_options"]["include_usage"], true);
    assert!(
        chat.get("tools").is_none(),
        "plain chat stream must not send tools: {chat}"
    );
    assert!(
        chat["messages"]
            .as_array()
            .expect("chat messages")
            .iter()
            .any(|message| message["content"]
                .as_str()
                .unwrap_or_default()
                .contains("KIR_QWEN_MLX_PREFILL_135K_CHAT_STREAM_QUARTZ_2741"))
    );

    let recall = probe_request_body(
        &lane,
        NormalizedProbePlan::new(
            NormalizedCaseKind::ContextRecallStream135k,
            SchemaVariant::CanonicalCurrent,
            ToolChoiceVariant::Required,
        ),
        ProbePrompt::measured(256, 0, None),
    );
    assert_eq!(recall["stream"], true);
    assert_eq!(recall["stream_options"]["include_usage"], true);
    assert_eq!(recall["tool_choice"], "required");
    assert_eq!(
        recall["tools"][0]["function"]["name"],
        "report_long_context_recall"
    );
    assert_eq!(
        recall["tools"][0]["function"]["parameters"]["required"],
        json!(["case", "marker", "profile"])
    );
    assert!(
        recall["messages"]
            .as_array()
            .expect("recall messages")
            .iter()
            .any(|message| message["content"]
                .as_str()
                .unwrap_or_default()
                .contains("KIR_LONG_CONTEXT_135K_CONTEXT_RECALL_STREAM_135K_QUARTZ_2741"))
    );

    let warm_prefix = probe_request_body(
        &lane,
        NormalizedProbePlan::new(
            NormalizedCaseKind::WarmPrefixRepeatedTurnStream,
            SchemaVariant::CanonicalCurrent,
            ToolChoiceVariant::Required,
        ),
        ProbePrompt::measured(256, 3, Some(1)),
    );
    assert_eq!(warm_prefix["stream"], true);
    assert_eq!(warm_prefix["stream_options"]["include_usage"], true);
    assert_eq!(
        warm_prefix["tools"][0]["function"]["name"],
        "record_qwen_tool_probe"
    );
    let messages = warm_prefix["messages"]
        .as_array()
        .expect("warm-prefix messages");
    assert_eq!(
        messages
            .iter()
            .map(|message| message["role"].as_str().expect("message role"))
            .collect::<Vec<_>>(),
        ["system", "user", "assistant", "tool", "user"]
    );
    assert!(
        messages[4]["content"]
            .as_str()
            .expect("final user")
            .contains("sample=3 request=1")
    );
}

#[test]
fn qwen_mlx_tool_normalized_stable_suite_bodies_are_streamed_and_canonical() {
    let direct = lane("name=direct,endpoint=http://127.0.0.1:8080/v1,model=qwen,kind=direct_mlx");
    let chat = probe_request_body(
        &direct,
        NormalizedProbePlan::new(
            NormalizedCaseKind::ChatStream,
            SchemaVariant::None,
            ToolChoiceVariant::None,
        ),
        ProbePrompt::measured(128, 0, None),
    );
    assert_eq!(chat["stream"], true);
    assert_eq!(chat["stream_options"]["include_usage"], true);
    assert_eq!(
        chat["chat_template_kwargs"],
        json!({"enable_thinking": false})
    );
    assert!(chat.get("tools").is_none());

    let tool = probe_request_body(
        &direct,
        NormalizedProbePlan::new(
            NormalizedCaseKind::ToolRequiredStream,
            SchemaVariant::CanonicalCurrent,
            ToolChoiceVariant::Required,
        ),
        ProbePrompt::measured(128, 0, None),
    );
    assert_eq!(tool["stream"], true);
    assert_eq!(tool["stream_options"]["include_usage"], true);
    assert_eq!(tool["tool_choice"], "required");
    assert_eq!(
        tool["tools"][0]["function"]["parameters"]["required"],
        json!(["case", "probe_id"])
    );
    assert_eq!(
        tool["chat_template_kwargs"],
        json!({"enable_thinking": false})
    );

    let warm = probe_request_body(
        &direct,
        NormalizedProbePlan::new(
            NormalizedCaseKind::WarmPrefixRepeatedTurnStream,
            SchemaVariant::CanonicalCurrent,
            ToolChoiceVariant::Required,
        ),
        ProbePrompt::measured(128, 1, None),
    );
    assert_eq!(warm["stream"], true);
    assert_eq!(warm["stream_options"]["include_usage"], true);
    assert_eq!(
        warm["tools"][0]["function"]["parameters"]["required"],
        json!(["case", "probe_id"])
    );
    assert_eq!(
        warm["messages"]
            .as_array()
            .expect("warm messages")
            .iter()
            .map(|message| message["role"].as_str().expect("role"))
            .collect::<Vec<_>>(),
        ["system", "user", "assistant", "tool", "user"]
    );
}

#[test]
fn qwen_mlx_tool_normalized_chat_completions_url_accepts_openai_base_with_or_without_v1() {
    assert_eq!(
        chat_completions_url("http://127.0.0.1:8080/v1"),
        "http://127.0.0.1:8080/v1/chat/completions"
    );
    assert_eq!(
        chat_completions_url("http://127.0.0.1:3000"),
        "http://127.0.0.1:3000/v1/chat/completions"
    );
}
