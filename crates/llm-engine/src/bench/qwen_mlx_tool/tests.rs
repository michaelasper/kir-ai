use super::super::{StreamAssembly, StreamTimingReport, apply_sse_frame, usage_from_value};
use super::*;
use crate::DEFAULT_MODEL_ID;
use serde_json::json;

fn lane(spec: &str) -> NormalizedLaneConfig {
    parse_lane_spec(spec).expect("lane spec parses")
}

fn args(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|part| (*part).to_owned()).collect()
}

#[test]
fn qwen_mlx_tool_normalized_lane_spec_defaults_to_qwen_no_thinking_and_rejects_unknown_keys() {
    let lane = lane("name=direct,endpoint=http://127.0.0.1:8080/v1/,model=qwen-loaded");

    assert_eq!(lane.name, "direct");
    assert_eq!(lane.endpoint, "http://127.0.0.1:8080/v1");
    assert_eq!(lane.kind, NormalizedLaneKind::Other);
    assert_eq!(
        lane.model_addressing,
        NormalizedModelAddressing::LoadedModelId
    );
    assert_eq!(lane.template, NormalizedTemplatePolicy::QwenNoThinking);
    assert_eq!(lane.effective_request_model_id(), "qwen-loaded");

    let err =
        parse_lane_spec("name=direct,endpoint=http://127.0.0.1:8080/v1,model=qwen,unknown=value")
            .expect_err("unknown keys fail");
    assert!(
        err.to_string().contains("unknown keys: unknown"),
        "error: {err}"
    );
}

#[test]
fn qwen_mlx_tool_normalized_cache_prefill_profile_expands_default_lanes() {
    let snapshot =
        "/tmp/huggingface/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/abcdef1234567890";
    let lanes = parse_lane_specs(&args(&[
        "--sweep-profile",
        "qwen-mlx-cache-prefill",
        "--snapshot",
        snapshot,
    ]))
    .expect("profile expands");

    assert_eq!(
        lanes
            .iter()
            .map(|lane| lane.name.as_str())
            .collect::<Vec<_>>(),
        [
            "mlx-default",
            "mlx-cache-size-4096",
            "mlx-cache-bytes-1g",
            "mlx-prefill-2048",
            "mlx-prefill-4096",
            "mlx-prefill-8192",
            "mlx-concurrent-4x2",
            "kir-proxy",
        ]
    );
    assert_eq!(
        lanes
            .iter()
            .map(|lane| lane.endpoint.as_str())
            .collect::<Vec<_>>(),
        [
            "http://127.0.0.1:8080/v1",
            "http://127.0.0.1:8081/v1",
            "http://127.0.0.1:8082/v1",
            "http://127.0.0.1:8083/v1",
            "http://127.0.0.1:8084/v1",
            "http://127.0.0.1:8085/v1",
            "http://127.0.0.1:8086/v1",
            "http://127.0.0.1:3000",
        ]
    );
    assert!(
        lanes
            .iter()
            .all(|lane| lane.launched_model_id.as_deref() == Some(snapshot))
    );
    assert!(
        lanes
            .iter()
            .all(|lane| lane.snapshot_path.as_deref() == Some(Path::new(snapshot)))
    );

    let default = &lanes[0];
    assert_eq!(default.kind, NormalizedLaneKind::DirectMlx);
    assert_eq!(default.declared_model_id, snapshot);
    assert_eq!(
        default.model_addressing,
        NormalizedModelAddressing::ServerDefault
    );
    let direct_body = probe_request_body(
        default,
        NormalizedProbePlan::new(
            NormalizedCaseKind::JsonObject,
            SchemaVariant::None,
            ToolChoiceVariant::None,
        ),
        ProbePrompt::measured(128, 0, None),
    );
    assert!(
        direct_body.get("model").is_none(),
        "plain mlx_lm.server treats unknown model ids as Hugging Face repos"
    );
    assert_eq!(
        default.template,
        NormalizedTemplatePolicy::SidecarChatTemplateArgs
    );
    assert_eq!(
        default.mlx_lm_settings.prompt_cache_size,
        DefaultOrU64::Default
    );
    assert_eq!(
        lanes[1].mlx_lm_settings.prompt_cache_size,
        DefaultOrU64::Value(4096)
    );
    assert_eq!(
        lanes[2].mlx_lm_settings.prompt_cache_bytes,
        UnsetOrU64::Value(1_073_741_824)
    );
    assert_eq!(
        lanes[3].mlx_lm_settings.prefill_step_size,
        DefaultOrU64::Value(2048)
    );
    assert_eq!(
        lanes[4].mlx_lm_settings.prefill_step_size,
        DefaultOrU64::Value(4096)
    );
    assert_eq!(
        lanes[5].mlx_lm_settings.prefill_step_size,
        DefaultOrU64::Value(8192)
    );
    assert_eq!(
        lanes[6].mlx_lm_settings.prompt_concurrency,
        DefaultOrU32::Value(4)
    );
    assert_eq!(
        lanes[6].mlx_lm_settings.decode_concurrency,
        DefaultOrU32::Value(2)
    );

    let proxy = &lanes[7];
    assert_eq!(proxy.kind, NormalizedLaneKind::KirAiProxy);
    assert_eq!(proxy.declared_model_id, "local-qwen36-mlx");
    assert_eq!(proxy.effective_request_model_id(), DEFAULT_MODEL_ID);
    assert_eq!(
        proxy.model_addressing,
        NormalizedModelAddressing::DefaultModel
    );
    assert_eq!(
        proxy.template,
        NormalizedTemplatePolicy::SidecarChatTemplateArgs
    );
}

#[test]
fn qwen_mlx_tool_normalized_cache_prefill_profile_requires_snapshot() {
    let err = parse_lane_specs(&args(&["--sweep-profile", "qwen-mlx-cache-prefill"]))
        .expect_err("profile requires snapshot");

    assert!(
        err.to_string().contains("--snapshot"),
        "error should mention --snapshot: {err}"
    );
}

#[test]
fn qwen_mlx_tool_normalized_explicit_lane_mode_remains_available() {
    let lanes = parse_lane_specs(&args(&[
        "--lane",
        "name=custom,endpoint=http://127.0.0.1:9090/v1,model=qwen-custom,kind=direct_mlx",
    ]))
    .expect("explicit lane mode parses");

    assert_eq!(lanes.len(), 1);
    assert_eq!(lanes[0].name, "custom");
    assert_eq!(lanes[0].endpoint, "http://127.0.0.1:9090/v1");
    assert_eq!(lanes[0].declared_model_id, "qwen-custom");
}

#[test]
fn qwen_mlx_tool_normalized_lane_spec_parses_mlx_lm_sweep_knobs_and_serializes_metadata() {
    let parsed_lane = lane(
        "name=direct,endpoint=http://127.0.0.1:8080/v1,model=qwen,mlx_prompt_cache_size=4096,mlx_prompt_cache_bytes=unset,mlx_prefill_step_size=8192,mlx_prompt_concurrency=4,mlx_decode_concurrency=2",
    );

    assert_eq!(
        parsed_lane.mlx_lm_settings.prompt_cache_size,
        DefaultOrU64::Value(4096)
    );
    assert_eq!(
        parsed_lane.mlx_lm_settings.prompt_cache_bytes,
        UnsetOrU64::Unset
    );
    assert_eq!(
        parsed_lane.mlx_lm_settings.prefill_step_size,
        DefaultOrU64::Value(8192)
    );
    assert_eq!(
        parsed_lane.mlx_lm_settings.prompt_concurrency,
        DefaultOrU32::Value(4)
    );
    assert_eq!(
        parsed_lane.mlx_lm_settings.decode_concurrency,
        DefaultOrU32::Value(2)
    );

    let defaulted = lane("name=defaulted,endpoint=http://127.0.0.1:8081/v1,model=qwen-default");
    assert_eq!(
        defaulted.mlx_lm_settings.prompt_cache_size,
        DefaultOrU64::Default
    );
    assert_eq!(
        defaulted.mlx_lm_settings.prompt_cache_bytes,
        UnsetOrU64::Unset
    );
    assert_eq!(
        defaulted.mlx_lm_settings.prefill_step_size,
        DefaultOrU64::Default
    );
    assert_eq!(
        defaulted.mlx_lm_settings.prompt_concurrency,
        DefaultOrU32::Default
    );
    assert_eq!(
        defaulted.mlx_lm_settings.decode_concurrency,
        DefaultOrU32::Default
    );

    let report = NormalizedLaneReport::dry_run(
        &parsed_lane,
        NormalizedRunConfig::new(1, 1, 128, 1, 0),
        None,
        &NormalizedProbePlan::all(),
    );
    let value = serde_json::to_value(report).expect("lane report serializes");
    assert_eq!(value["mlx_lm_settings"]["mlx_prompt_cache_size"], 4096);
    assert_eq!(value["mlx_lm_settings"]["mlx_prompt_cache_bytes"], "unset");
    assert_eq!(value["mlx_lm_settings"]["mlx_prefill_step_size"], 8192);
    assert_eq!(value["mlx_lm_settings"]["mlx_prompt_concurrency"], 4);
    assert_eq!(value["mlx_lm_settings"]["mlx_decode_concurrency"], 2);
}

#[test]
fn qwen_mlx_tool_normalized_lane_spec_rejects_invalid_mlx_lm_sweep_knobs() {
    for spec in [
        "name=bad,endpoint=http://127.0.0.1:8080/v1,model=qwen,mlx_prompt_cache_size=0",
        "name=bad,endpoint=http://127.0.0.1:8080/v1,model=qwen,mlx_prompt_cache_size=-1",
        "name=bad,endpoint=http://127.0.0.1:8080/v1,model=qwen,mlx_prompt_cache_bytes=default",
        "name=bad,endpoint=http://127.0.0.1:8080/v1,model=qwen,mlx_prefill_step_size=0",
        "name=bad,endpoint=http://127.0.0.1:8080/v1,model=qwen,mlx_prompt_concurrency=0",
        "name=bad,endpoint=http://127.0.0.1:8080/v1,model=qwen,mlx_decode_concurrency=-2",
    ] {
        let err = parse_lane_spec(spec).expect_err("invalid MLX knob should fail");
        assert!(
            err.to_string().contains("mlx_"),
            "error should name MLX knob for `{spec}`: {err}"
        );
    }
}

#[test]
fn qwen_mlx_tool_normalized_sidecar_template_policy_declares_assumption_without_request_kwargs() {
    let lane = lane(
        "name=sidecar,endpoint=http://127.0.0.1:8080/v1,model=qwen,template=sidecar-chat-template-args",
    );

    let body = probe_request_body(
        &lane,
        NormalizedProbePlan::new(
            NormalizedCaseKind::ToolRequired,
            SchemaVariant::BaselineCurrent,
            ToolChoiceVariant::Required,
        ),
        ProbePrompt::measured(128, 0, None),
    );
    assert!(
        body.get("chat_template_kwargs").is_none(),
        "sidecar template policy must not inject request kwargs: {body}"
    );

    let policy = lane.thinking_policy_report();
    assert_eq!(policy["template"], "sidecar-chat-template-args");
    assert_eq!(policy["enable_thinking"], false);
    assert_eq!(
        policy["source"],
        "sidecar_chat_template_args_declared_by_lane"
    );
}

#[test]
fn qwen_mlx_tool_normalized_model_addressing_controls_effective_request_model_id_and_serializes() {
    let loaded = lane(
        "name=loaded,endpoint=http://127.0.0.1:8080/v1,model=qwen-loaded,model_addressing=loaded_model_id",
    );
    let default_model = lane(
        "name=default,endpoint=http://127.0.0.1:8081/v1,model=qwen-loaded,model_addressing=default_model",
    );
    let custom = lane(
        "name=custom,endpoint=http://127.0.0.1:8082/v1,model=qwen-custom,model_addressing=custom",
    );
    let server_default = lane(
        "name=server-default,endpoint=http://127.0.0.1:8083/v1,model=qwen-loaded,snapshot=/models/qwen-snapshot,model_addressing=server_default",
    );

    assert_eq!(loaded.effective_request_model_id(), "qwen-loaded");
    assert_eq!(default_model.effective_request_model_id(), DEFAULT_MODEL_ID);
    assert_eq!(custom.effective_request_model_id(), "qwen-custom");
    assert_eq!(
        server_default.effective_request_model_id(),
        "/models/qwen-snapshot"
    );
    assert_eq!(server_default.request_model_id(), None);

    let report = NormalizedLaneReport::dry_run(
        &default_model,
        NormalizedRunConfig::new(1, 1, 128, 1, 0),
        None,
        &NormalizedProbePlan::all(),
    );
    let value = serde_json::to_value(report).expect("lane report serializes");
    assert_eq!(value["declared_model_id"], "qwen-loaded");
    assert_eq!(value["effective_request_model_id"], DEFAULT_MODEL_ID);
    assert_eq!(value["model_addressing"], "default_model");

    let body = probe_request_body(
        &server_default,
        NormalizedProbePlan::new(
            NormalizedCaseKind::JsonObject,
            SchemaVariant::None,
            ToolChoiceVariant::None,
        ),
        ProbePrompt::measured(128, 0, None),
    );
    assert!(body.get("model").is_none());
}

#[test]
fn qwen_mlx_tool_normalized_lane_can_pin_launched_model_identity() {
    let lane = lane(
        "name=direct,endpoint=http://127.0.0.1:8080/v1,model=default_model,launched_model_id=/models/qwen-snapshot,kind=direct_mlx,model_addressing=loaded_model_id",
    );

    assert_eq!(lane.effective_request_model_id(), "default_model");
    assert_eq!(lane.identity_model_id(), "/models/qwen-snapshot");

    let report = NormalizedLaneReport::dry_run(
        &lane,
        NormalizedRunConfig::new(0, 1, 128, 1, 0),
        None,
        &NormalizedProbePlan::all(),
    );
    let value = serde_json::to_value(report).expect("lane report serializes");
    assert_eq!(value["declared_model_id"], "default_model");
    assert_eq!(value["effective_request_model_id"], "default_model");
    assert_eq!(value["launched_model_id"], "/models/qwen-snapshot");
    assert_eq!(value["model_identity_source"], "lane_launched_model_id");
}

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

#[test]
fn qwen_mlx_tool_normalized_cache_phase_plan_excludes_warmups_from_measured_samples() {
    let plan = phase_plan(2, 3);
    let measured = plan
        .iter()
        .filter(|run| run.kind == PlannedRunKind::Measured)
        .collect::<Vec<_>>();
    let warmups = plan
        .iter()
        .filter(|run| run.kind == PlannedRunKind::Warmup)
        .collect::<Vec<_>>();

    assert_eq!(measured.len(), 9);
    assert_eq!(warmups.len(), 4);
    assert!(warmups.iter().all(|run| run.phase != CachePhase::Cold));
    assert_eq!(
        measured
            .iter()
            .map(|run| (run.run_mode, run.phase, run.sample_index, run.request_index))
            .collect::<Vec<_>>(),
        vec![
            (RunMode::Sequential, CachePhase::Cold, Some(0), None),
            (RunMode::Sequential, CachePhase::Cold, Some(1), None),
            (RunMode::Sequential, CachePhase::Cold, Some(2), None),
            (
                RunMode::Sequential,
                CachePhase::WarmSamePrompt,
                Some(0),
                None
            ),
            (
                RunMode::Sequential,
                CachePhase::WarmSamePrompt,
                Some(1),
                None
            ),
            (
                RunMode::Sequential,
                CachePhase::WarmSamePrompt,
                Some(2),
                None
            ),
            (
                RunMode::Sequential,
                CachePhase::WarmSameToolSchema,
                Some(0),
                None
            ),
            (
                RunMode::Sequential,
                CachePhase::WarmSameToolSchema,
                Some(1),
                None
            ),
            (
                RunMode::Sequential,
                CachePhase::WarmSameToolSchema,
                Some(2),
                None
            ),
        ]
    );
}

#[test]
fn qwen_mlx_tool_normalized_concurrent_phase_plan_preserves_sample_and_request_indexes() {
    assert_eq!(effective_concurrent_samples(1, 2, 0), 0);
    assert_eq!(effective_concurrent_samples(3, 2, 0), 2);
    assert_eq!(effective_concurrent_samples(1, 2, 4), 4);

    let plan = concurrent_phase_plan(3, 2);

    assert_eq!(plan.len(), 18);
    assert!(plan.iter().all(|run| run.kind == PlannedRunKind::Measured));
    assert!(plan.iter().all(|run| run.run_mode == RunMode::Concurrent));
    assert_eq!(
        plan.iter()
            .filter(|run| run.phase == CachePhase::Cold)
            .map(|run| (run.sample_index, run.request_index))
            .collect::<Vec<_>>(),
        vec![
            (Some(0), Some(0)),
            (Some(0), Some(1)),
            (Some(0), Some(2)),
            (Some(1), Some(0)),
            (Some(1), Some(1)),
            (Some(1), Some(2)),
        ]
    );
}

#[test]
fn qwen_mlx_tool_normalized_focused_agentic_gate_uses_small_probe_plan() {
    let suite = parse_probe_suite_flag(&args(&["--focused-agentic-gate"]));
    let probes = suite.probes();

    assert_eq!(suite.name(), "focused_agentic_gate");
    assert_eq!(
        probes,
        vec![
            NormalizedProbePlan::new(
                NormalizedCaseKind::ToolRequired,
                SchemaVariant::CanonicalCurrent,
                ToolChoiceVariant::Required,
            ),
            NormalizedProbePlan::new(
                NormalizedCaseKind::ToolRequiredStream,
                SchemaVariant::CanonicalCurrent,
                ToolChoiceVariant::Required,
            ),
            NormalizedProbePlan::new(
                NormalizedCaseKind::OmpRepeatedPrefix,
                SchemaVariant::CanonicalCurrent,
                ToolChoiceVariant::Required,
            ),
        ]
    );

    let lane = lane("name=direct,endpoint=http://127.0.0.1:8080/v1,model=qwen");
    let report = NormalizedLaneReport::dry_run(
        &lane,
        NormalizedRunConfig::new(0, 1, 128, 1, 0),
        None,
        &probes,
    );
    assert_eq!(report.samples.len(), 9);
    assert!(
        report
            .samples
            .iter()
            .all(|sample| sample.case != "json_object")
    );
}

#[test]
fn qwen_mlx_tool_normalized_aggregate_summary_rows_group_by_lane_case_phase_and_run_mode() {
    let lane_a = lane("name=a,endpoint=http://127.0.0.1:8080/v1,model=qwen-a");
    let lane_b = lane("name=b,endpoint=http://127.0.0.1:8081/v1,model=qwen-b");
    let mut report_a = NormalizedLaneReport::planned(&lane_a, 0, 0, None);
    let mut report_b = NormalizedLaneReport::planned(&lane_b, 0, 0, None);

    report_a.samples = vec![
        passed_sample(
            NormalizedCaseKind::ToolRequired,
            CachePhase::Cold,
            RunMode::Sequential,
            0,
            None,
            100,
            10,
        ),
        passed_sample(
            NormalizedCaseKind::ToolRequired,
            CachePhase::Cold,
            RunMode::Sequential,
            1,
            None,
            200,
            20,
        ),
        passed_sample(
            NormalizedCaseKind::ToolRequired,
            CachePhase::Cold,
            RunMode::Sequential,
            2,
            None,
            400,
            30,
        ),
        failed_summary_sample(
            NormalizedCaseKind::ToolRequired,
            CachePhase::Cold,
            RunMode::Sequential,
            3,
            None,
        ),
    ];
    report_b.samples = vec![passed_sample(
        NormalizedCaseKind::ToolRequired,
        CachePhase::Cold,
        RunMode::Sequential,
        0,
        None,
        50,
        5,
    )];

    let probes = NormalizedProbePlan::all();
    let summary = aggregate_normalized_summary(&[report_a, report_b], &probes);
    let a_row = summary
        .iter()
        .find(|row| {
            row.lane == "a"
                && row.case == "tool_required"
                && row.cache_phase == "cold"
                && row.run_mode == "sequential"
        })
        .expect("lane a summary row");

    assert_eq!(a_row.pass_count, 3);
    assert_eq!(a_row.schema_variant, "baseline_current");
    assert_eq!(a_row.tool_choice_variant, "required");
    assert_eq!(a_row.fail_count, 1);
    assert_eq!(a_row.p50_latency_ms, Some(200));
    assert_eq!(a_row.p95_latency_ms, Some(400));
    assert_eq!(a_row.avg_cached_tokens, Some(20.0));
    assert_eq!(a_row.avg_prompt_tokens, Some(1000.0));
    assert_eq!(a_row.avg_completion_tokens, Some(10.0));
    assert_eq!(a_row.avg_total_tokens, Some(1010.0));
    assert_eq!(a_row.fastest_lane.as_deref(), Some("b"));
}

#[test]
fn qwen_mlx_tool_normalized_agentic_gate_reports_warm_stream_cache_and_lane_deltas() {
    let lane_a = lane("name=direct,endpoint=http://127.0.0.1:8080/v1,model=qwen-a");
    let lane_b = lane("name=proxy,endpoint=http://127.0.0.1:3000,model=qwen-b");
    let mut report_a = NormalizedLaneReport::planned(&lane_a, 0, 0, None);
    let mut report_b = NormalizedLaneReport::planned(&lane_b, 0, 0, None);

    let mut direct = passed_sample(
        NormalizedCaseKind::ToolRequiredStream,
        CachePhase::WarmSamePrompt,
        RunMode::Sequential,
        0,
        None,
        1000,
        64,
    );
    direct.schema_variant = SchemaVariant::CanonicalCurrent.name();
    direct.schema_canonicalized = true;
    direct.stream_timing = StreamTimingReport {
        first_byte_latency_ms: Some(120),
        first_sse_data_latency_ms: Some(125),
        first_content_delta_latency_ms: None,
        first_tool_delta_latency_ms: Some(700),
        first_semantic_delta_latency_ms: Some(700),
    };
    direct.tokens_per_second = Some(33.0);
    report_a.samples = vec![direct];

    let mut proxy = passed_sample(
        NormalizedCaseKind::ToolRequiredStream,
        CachePhase::WarmSamePrompt,
        RunMode::Sequential,
        0,
        None,
        1125,
        60,
    );
    proxy.schema_variant = SchemaVariant::CanonicalCurrent.name();
    proxy.schema_canonicalized = true;
    proxy.stream_timing = StreamTimingReport {
        first_byte_latency_ms: Some(150),
        first_sse_data_latency_ms: Some(155),
        first_content_delta_latency_ms: None,
        first_tool_delta_latency_ms: Some(760),
        first_semantic_delta_latency_ms: Some(760),
    };
    proxy.tokens_per_second = Some(31.0);
    report_b.samples = vec![proxy];

    let gate = agentic_gate_report(&[report_a, report_b]);
    let row = gate
        .rows
        .iter()
        .find(|row| {
            row.case == "tool_required_stream"
                && row.cache_phase == "warm_same_prompt"
                && row.run_mode == "sequential"
        })
        .expect("warm stream gate row");

    assert_eq!(gate.status, "reported");
    assert_eq!(row.fastest_lane.as_deref(), Some("direct"));
    assert_eq!(row.lanes[0].p50_first_byte_latency_ms, Some(120));
    assert_eq!(row.lanes[0].p50_first_semantic_delta_latency_ms, Some(700));
    assert_eq!(row.lanes[0].p50_first_tool_delta_latency_ms, Some(700));
    assert_eq!(row.lanes[0].avg_cached_tokens, Some(64.0));
    assert_eq!(row.lanes[1].latency_delta_vs_fastest_ms, Some(125));
}

#[test]
fn qwen_mlx_tool_normalized_cached_tokens_usage_parses_present_null_and_missing_shapes() {
    let present = usage_from_value(Some(&json!({
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
        "prompt_tokens_details": {"cached_tokens": 7}
    })));
    assert_eq!(present.cached_tokens, Some(7));
    assert_eq!(present.cached_tokens_status, Some("present"));

    let null = usage_from_value(Some(&json!({
        "prompt_tokens_details": {"cached_tokens": null}
    })));
    assert_eq!(null.cached_tokens, None);
    assert_eq!(null.cached_tokens_status, Some("null"));

    let missing = usage_from_value(Some(&json!({
        "prompt_tokens": 10
    })));
    assert_eq!(missing.cached_tokens, None);
    assert_eq!(missing.cached_tokens_status, Some("missing"));
}

#[test]
fn qwen_mlx_tool_normalized_stream_usage_merges_across_frames() {
    let mut assembly = StreamAssembly::default();
    apply_sse_frame(
        &json!({
            "choices": [{"delta": {"role": "assistant"}, "finish_reason": null}],
            "usage": {
                "prompt_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 80}
            }
        }),
        &mut assembly,
    );
    apply_sse_frame(
        &json!({
            "choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "record_qwen_tool_probe", "arguments": "{}"}}]}, "finish_reason": "tool_calls"}],
            "usage": {"completion_tokens": 12}
        }),
        &mut assembly,
    );

    assert_eq!(assembly.usage.prompt_tokens, Some(100));
    assert_eq!(assembly.usage.cached_tokens_status, Some("present"));
    assert_eq!(assembly.usage.cached_tokens, Some(80));
    assert_eq!(assembly.usage.completion_tokens, Some(12));
    assert_eq!(assembly.usage.total_tokens, Some(112));
}

#[test]
fn qwen_mlx_tool_normalized_validation_classifies_buffered_tool_json_and_stream_responses() {
    let tool = json!({
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "tool_calls": [{
                    "function": {
                        "name": "record_qwen_tool_probe",
                        "arguments": "{\"probe_id\":\"KIR_QWEN_MLX_TOOL_NORMALIZED_TOOL_REQUIRED\",\"case\":\"tool_required\"}"
                    }
                }]
            }
        }]
    });
    assert_eq!(
        validate_buffered_probe(
            NormalizedCaseKind::ToolRequired,
            &tool,
            NormalizedCaseKind::ToolRequired.probe_id()
        ),
        Ok(())
    );

    let json_response = json!({
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "content": "{\"probe_id\":\"KIR_QWEN_MLX_TOOL_NORMALIZED_JSON_OBJECT\",\"case\":\"json_object\"}"
            }
        }]
    });
    assert_eq!(
        validate_buffered_probe(
            NormalizedCaseKind::JsonObject,
            &json_response,
            NormalizedCaseKind::JsonObject.probe_id()
        ),
        Ok(())
    );

    let assembly = StreamAssembly {
        tool_name: Some("record_qwen_tool_probe".to_owned()),
        tool_arguments:
            "{\"probe_id\":\"KIR_QWEN_MLX_TOOL_NORMALIZED_TOOL_REQUIRED_STREAM\",\"case\":\"tool_required_stream\"}"
                .to_owned(),
        finish_reason: Some("tool_calls".to_owned()),
        ..StreamAssembly::default()
    };
    assert_eq!(
        validate_streaming_probe(
            NormalizedCaseKind::ToolRequiredStream,
            &assembly,
            NormalizedCaseKind::ToolRequiredStream.probe_id()
        ),
        Ok(())
    );
}

fn passed_sample(
    case: NormalizedCaseKind,
    phase: CachePhase,
    run_mode: RunMode,
    sample_index: usize,
    request_index: Option<usize>,
    latency_ms: u128,
    cached_tokens: u64,
) -> NormalizedSampleReport {
    let mut sample = NormalizedSampleReport::base(
        NormalizedProbePlan::new(
            case,
            SchemaVariant::BaselineCurrent,
            ToolChoiceVariant::Required,
        ),
        phase,
        run_mode,
        sample_index,
        request_index,
        false,
        128,
    );
    sample.status = "passed".to_owned();
    sample.classification = "passed".to_owned();
    sample.latency_ms = Some(latency_ms);
    sample.prompt_tokens = Some(1000);
    sample.completion_tokens = Some(10);
    sample.total_tokens = Some(1010);
    sample.cached_tokens_status = "present";
    sample.cached_tokens = Some(cached_tokens);
    sample
}

fn failed_summary_sample(
    case: NormalizedCaseKind,
    phase: CachePhase,
    run_mode: RunMode,
    sample_index: usize,
    request_index: Option<usize>,
) -> NormalizedSampleReport {
    let mut sample = NormalizedSampleReport::base(
        NormalizedProbePlan::new(
            case,
            SchemaVariant::BaselineCurrent,
            ToolChoiceVariant::Required,
        ),
        phase,
        run_mode,
        sample_index,
        request_index,
        false,
        128,
    );
    sample.status = "failed".to_owned();
    sample.classification = "http_status_failed".to_owned();
    sample.latency_ms = Some(900);
    sample
}
