use super::super::{
    StreamAssembly, StreamTimingReport, StreamTimingTracker, apply_sse_frame, consume_sse_buffer,
    usage_from_value,
};
use super::*;
use crate::{DEFAULT_MODEL_ID, MlxToolParserMode};
use serde_json::json;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

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
fn stable_prefix_counts_use_request_cache_observation_when_usage_omits_cached_tokens() {
    let probe = NormalizedProbePlan::new(
        NormalizedCaseKind::WarmPrefixRepeatedTurnStream,
        SchemaVariant::CanonicalCurrent,
        ToolChoiceVariant::Required,
    );
    let lane_config =
        lane("name=kir,endpoint=http://127.0.0.1:8080,model=local-qwen36-mlx,kind=kir_ai_proxy");
    let mut lane = NormalizedLaneReport::planned(&lane_config, 0, 0, None);
    let mut sample = NormalizedSampleReport::base(
        probe,
        CachePhase::WarmSameToolSchema,
        RunMode::Sequential,
        0,
        None,
        true,
        128,
    );
    sample.status = "passed".to_owned();
    sample.classification = "passed".to_owned();
    sample.latency_ms = Some(42);
    sample.prompt_tokens = Some(100);
    sample.completion_tokens = Some(5);
    sample.total_tokens = Some(105);
    sample.cached_tokens_status = "missing";
    sample.request_id = Some("req-2".to_owned());
    lane.samples.push(sample);
    lane.admin_metrics.after = Some(json!({
        "request_cache": {
            "recent": [{
                "request_id": "req-2",
                "model": "local-qwen36-mlx",
                "streamed": true,
                "prompt_tokens": 100,
                "cached_tokens": null,
                "uncached_tokens": null,
                "cache_status": "partial",
                "latency_ms": 7
            }]
        }
    }));

    let metric = stable_prefix_lane_metric(
        &lane,
        probe,
        CachePhase::WarmSameToolSchema,
        RunMode::Sequential,
    )
    .expect("stable prefix metric");

    assert_eq!(metric.cache_status_counts.get("partial"), Some(&1));
    assert!(!metric.cache_status_counts.contains_key("unknown"));
    assert_eq!(metric.request_cache_observations.len(), 1);
    assert_eq!(metric.request_cache_observations[0].cache_status, "partial");
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
fn qwen_mlx_tool_normalized_stable_prefix_profile_expands_expected_lanes() {
    let snapshot =
        "/tmp/huggingface/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/abcdef1234567890";
    let args = args(&[
        "--sweep-profile",
        "qwen-mlx-stable-prefix",
        "--snapshot",
        snapshot,
    ]);
    let lanes = parse_lane_specs(&args).expect("stable profile expands");

    assert_eq!(
        lanes
            .iter()
            .map(|lane| lane.name.as_str())
            .collect::<Vec<_>>(),
        ["mlx-stable-prefix", "kir-stable-prefix"]
    );
    assert_eq!(
        lanes
            .iter()
            .map(|lane| lane.endpoint.as_str())
            .collect::<Vec<_>>(),
        ["http://127.0.0.1:8080/v1", "http://127.0.0.1:3000"]
    );
    assert!(
        lanes
            .iter()
            .all(|lane| lane.launched_model_id.as_deref() == Some(snapshot))
    );
    assert_eq!(lanes[0].kind, NormalizedLaneKind::DirectMlx);
    assert_eq!(lanes[0].template, NormalizedTemplatePolicy::QwenNoThinking);
    assert_eq!(
        lanes[0].model_addressing,
        NormalizedModelAddressing::ServerDefault
    );
    assert_eq!(lanes[1].kind, NormalizedLaneKind::KirAiProxy);
    assert_eq!(
        lanes[1].template,
        NormalizedTemplatePolicy::SidecarChatTemplateArgs
    );
    assert_eq!(lanes[1].effective_request_model_id(), DEFAULT_MODEL_ID);

    let suite = parse_probe_suite_flag(&args, Some(NormalizedSweepProfile::QwenMlxStablePrefix))
        .expect("stable profile default suite");
    assert_eq!(suite, NormalizedProbeSuite::StableAgentPrefix);
    assert_eq!(suite.name(), "stable_agent_prefix");
}

#[test]
fn qwen_mlx_tool_normalized_profile_lane_filter_selects_after_expansion() {
    let snapshot =
        "/tmp/huggingface/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/abcdef1234567890";
    let lanes = parse_lane_specs(&args(&[
        "--sweep-profile",
        "qwen-mlx-stable-prefix",
        "--snapshot",
        snapshot,
        "--only-lanes",
        "kir-stable-prefix",
    ]))
    .expect("lane filter applies after profile expansion");

    assert_eq!(lanes.len(), 1);
    assert_eq!(lanes[0].name, "kir-stable-prefix");
    assert_eq!(lanes[0].endpoint, "http://127.0.0.1:3000");

    let alias_lanes = parse_lane_specs(&args(&[
        "--sweep-profile",
        "qwen-mlx-stable-prefix",
        "--snapshot",
        snapshot,
        "--profile-lanes",
        "mlx-stable-prefix,kir-stable-prefix",
    ]))
    .expect("profile-lanes alias applies");
    assert_eq!(
        alias_lanes
            .iter()
            .map(|lane| lane.name.as_str())
            .collect::<Vec<_>>(),
        ["mlx-stable-prefix", "kir-stable-prefix"]
    );

    let err = parse_lane_specs(&args(&[
        "--sweep-profile",
        "qwen-mlx-stable-prefix",
        "--snapshot",
        snapshot,
        "--only-lanes",
        "missing-lane",
    ]))
    .expect_err("unknown lane filter fails");
    assert!(
        err.to_string().contains("missing-lane"),
        "error should mention missing lane: {err}"
    );
}

#[test]
fn qwen_mlx_tool_normalized_prefill_135k_profile_expands_direct_proxy_pairs() {
    let snapshot =
        "/tmp/huggingface/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/abcdef1234567890";
    let lanes = parse_lane_specs(&args(&[
        "--sweep-profile",
        "qwen-mlx-prefill-135k",
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
            "mlx-prefill-default",
            "kir-prefill-default",
            "mlx-prefill-512",
            "kir-prefill-512",
            "mlx-prefill-1024",
            "kir-prefill-1024",
            "mlx-prefill-2048",
            "kir-prefill-2048",
            "mlx-prefill-4096",
            "kir-prefill-4096",
            "mlx-prefill-8192",
            "kir-prefill-8192",
        ]
    );
    assert_eq!(
        lanes
            .iter()
            .map(|lane| lane.endpoint.as_str())
            .collect::<Vec<_>>(),
        [
            "http://127.0.0.1:8080/v1",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8081/v1",
            "http://127.0.0.1:3001",
            "http://127.0.0.1:8082/v1",
            "http://127.0.0.1:3002",
            "http://127.0.0.1:8083/v1",
            "http://127.0.0.1:3003",
            "http://127.0.0.1:8084/v1",
            "http://127.0.0.1:3004",
            "http://127.0.0.1:8085/v1",
            "http://127.0.0.1:3005",
        ]
    );
    assert!(lanes.iter().enumerate().all(|(index, lane)| {
        let direct = index % 2 == 0;
        matches!(
            (direct, lane.kind),
            (true, NormalizedLaneKind::DirectMlx) | (false, NormalizedLaneKind::KirAiProxy)
        )
    }));
    assert!(lanes.iter().all(|lane| {
        lane.launched_model_id.as_deref() == Some(snapshot)
            && lane.snapshot_path.as_deref() == Some(Path::new(snapshot))
            && lane.template == NormalizedTemplatePolicy::SidecarChatTemplateArgs
    }));
    assert_eq!(
        lanes
            .iter()
            .map(|lane| lane.mlx_lm_settings.prefill_step_size)
            .collect::<Vec<_>>(),
        [
            DefaultOrU64::Default,
            DefaultOrU64::Default,
            DefaultOrU64::Value(512),
            DefaultOrU64::Value(512),
            DefaultOrU64::Value(1024),
            DefaultOrU64::Value(1024),
            DefaultOrU64::Value(2048),
            DefaultOrU64::Value(2048),
            DefaultOrU64::Value(4096),
            DefaultOrU64::Value(4096),
            DefaultOrU64::Value(8192),
            DefaultOrU64::Value(8192),
        ]
    );
    assert_eq!(
        lanes[1].model_addressing,
        NormalizedModelAddressing::DefaultModel
    );
    assert_eq!(lanes[1].declared_model_id, "local-qwen36-mlx");
    assert_eq!(lanes[1].effective_request_model_id(), DEFAULT_MODEL_ID);
}

#[test]
fn qwen_mlx_tool_normalized_prefill_135k_experimental_profile_is_context_recall_only() {
    let snapshot =
        "/tmp/huggingface/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/abcdef1234567890";
    let args = args(&[
        "--sweep-profile",
        "qwen-mlx-prefill-135k-experimental",
        "--snapshot",
        snapshot,
    ]);
    let lanes = parse_lane_specs(&args).expect("experimental profile expands");

    assert_eq!(
        lanes
            .iter()
            .map(|lane| lane.name.as_str())
            .collect::<Vec<_>>(),
        [
            "mlx-prefill-8192-control",
            "kir-prefill-8192-control",
            "mlx-prefill-experimental-12288",
            "kir-prefill-experimental-12288",
            "mlx-prefill-experimental-16384",
            "kir-prefill-experimental-16384",
            "mlx-prefill-experimental-32768",
            "kir-prefill-experimental-32768",
        ]
    );
    assert_eq!(
        lanes
            .iter()
            .map(|lane| lane.endpoint.as_str())
            .collect::<Vec<_>>(),
        [
            "http://127.0.0.1:8080/v1",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8081/v1",
            "http://127.0.0.1:3001",
            "http://127.0.0.1:8082/v1",
            "http://127.0.0.1:3002",
            "http://127.0.0.1:8083/v1",
            "http://127.0.0.1:3003",
        ]
    );
    assert_eq!(
        lanes
            .iter()
            .map(|lane| lane.experimental)
            .collect::<Vec<_>>(),
        [false, false, true, true, true, true, true, true]
    );
    assert_eq!(
        lanes
            .iter()
            .map(|lane| lane.mlx_lm_settings.prefill_step_size)
            .collect::<Vec<_>>(),
        [
            DefaultOrU64::Value(8192),
            DefaultOrU64::Value(8192),
            DefaultOrU64::Value(12288),
            DefaultOrU64::Value(12288),
            DefaultOrU64::Value(16384),
            DefaultOrU64::Value(16384),
            DefaultOrU64::Value(32768),
            DefaultOrU64::Value(32768),
        ]
    );

    let suite = parse_probe_suite_flag(
        &args,
        Some(NormalizedSweepProfile::QwenMlxPrefill135kExperimental),
    )
    .expect("experimental profile default suite");
    assert_eq!(suite, NormalizedProbeSuite::PrefillSweep135kContextRecall);
    assert_eq!(suite.name(), "prefill_sweep_135k_context_recall");
    let probes = suite.probes();
    assert_eq!(probes.len(), 1);
    assert_eq!(probes[0].case, NormalizedCaseKind::ContextRecallStream135k);
    assert_eq!(probes[0].max_tokens, DEFAULT_MAX_TOKENS);
    assert!(sweep_profile_requires_exact_token_prompt(Some(
        NormalizedSweepProfile::QwenMlxPrefill135kExperimental
    )));
    assert!(!NormalizedLaneReport::planned(&lanes[0], 0, 0, None).experimental);
    assert!(NormalizedLaneReport::planned(&lanes[2], 0, 0, None).experimental);
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
        &NormalizedRunConfig::new(1, 1, 128, 1, 0),
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
fn qwen_mlx_tool_normalized_lane_spec_parses_tool_parser_metadata() {
    let parsed_lane = lane(
        "name=xml,endpoint=http://127.0.0.1:3000,model=local-qwen36,kind=kir_ai_proxy,tool_parser=qwen-xml",
    );
    assert_eq!(parsed_lane.tool_parser, MlxToolParserMode::QwenXml);

    let report = NormalizedLaneReport::dry_run(
        &parsed_lane,
        &NormalizedRunConfig::new(0, 1, 128, 1, 0),
        None,
        &NormalizedProbePlan::all(),
    );
    let value = serde_json::to_value(report).expect("lane report serializes");
    assert_eq!(value["tool_parser"], "qwen-xml");

    let defaulted = lane("name=json,endpoint=http://127.0.0.1:3000,model=local-qwen36");
    let value = serde_json::to_value(NormalizedLaneReport::dry_run(
        &defaulted,
        &NormalizedRunConfig::new(0, 1, 128, 1, 0),
        None,
        &NormalizedProbePlan::all(),
    ))
    .expect("default lane report serializes");
    assert!(
        value.get("tool_parser").is_none(),
        "auto parser mode should be omitted unless explicitly requested: {value}"
    );

    let err = parse_lane_spec(
        "name=bad,endpoint=http://127.0.0.1:3000,model=local-qwen36,tool_parser=xml",
    )
    .expect_err("invalid tool parser fails");
    assert!(err.to_string().contains("auto, json, or qwen-xml"));
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
        &NormalizedRunConfig::new(1, 1, 128, 1, 0),
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
        &NormalizedRunConfig::new(0, 1, 128, 1, 0),
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

#[test]
fn qwen_mlx_tool_normalized_cache_phase_plan_excludes_warmups_from_measured_samples() {
    let plan = phase_plan(&CachePhase::all(), 2, 3);
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
fn qwen_mlx_tool_normalized_cache_phase_flag_selects_warm_phases() {
    let phases = parse_cache_phases_flag(&args(&[
        "--cache-phases",
        "warm_same_prompt,warm_same_tool_schema",
    ]))
    .expect("cache phases parse");

    assert_eq!(
        phases,
        vec![CachePhase::WarmSamePrompt, CachePhase::WarmSameToolSchema]
    );
    assert_eq!(
        phase_plan(&phases, 1, 1)
            .iter()
            .map(|run| (run.kind, run.phase))
            .collect::<Vec<_>>(),
        vec![
            (PlannedRunKind::Warmup, CachePhase::WarmSamePrompt),
            (PlannedRunKind::Measured, CachePhase::WarmSamePrompt),
            (PlannedRunKind::Warmup, CachePhase::WarmSameToolSchema),
            (PlannedRunKind::Measured, CachePhase::WarmSameToolSchema),
        ]
    );

    let err = parse_cache_phases_flag(&args(&["--cache-phases", "warm_same_prompt,coldish"]))
        .expect_err("unknown cache phase fails");
    assert!(
        err.to_string().contains("coldish"),
        "error should mention unknown phase: {err}"
    );
}

#[test]
fn qwen_mlx_tool_normalized_concurrent_phase_plan_preserves_sample_and_request_indexes() {
    assert_eq!(effective_concurrent_samples(1, 2, 0), 0);
    assert_eq!(effective_concurrent_samples(3, 2, 0), 2);
    assert_eq!(effective_concurrent_samples(1, 2, 4), 4);

    let plan = concurrent_phase_plan(&CachePhase::all(), 3, 2);

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
fn qwen_mlx_tool_normalized_stable_prefix_smoke_uses_single_probe() {
    let suite = parse_probe_suite_flag(&args(&["--probe-suite", "stable-prefix-smoke"]), None)
        .expect("stable prefix smoke parses");

    assert_eq!(suite, NormalizedProbeSuite::StablePrefixSmoke);
    assert_eq!(suite.name(), "stable_prefix_smoke");
    assert_eq!(
        suite.probes(),
        vec![NormalizedProbePlan::new(
            NormalizedCaseKind::WarmPrefixRepeatedTurnStream,
            SchemaVariant::CanonicalCurrent,
            ToolChoiceVariant::Required,
        )]
    );
}

#[test]
fn qwen_mlx_tool_normalized_focused_agentic_gate_uses_small_probe_plan() {
    let suite = parse_probe_suite_flag(&args(&["--focused-agentic-gate"]), None)
        .expect("focused suite parses");
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
        &NormalizedRunConfig::new(0, 1, 128, 1, 0),
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
fn qwen_mlx_tool_normalized_required_tool_ttft_matrix_uses_bounded_grid() {
    let suite =
        parse_probe_suite_flag(&args(&["--probe-suite", "required-tool-ttft-matrix"]), None)
            .expect("required-tool TTFT matrix parses");
    let probes = suite.probes();

    assert_eq!(suite, NormalizedProbeSuite::RequiredToolTtftMatrix);
    assert_eq!(suite.name(), "required_tool_ttft_matrix");
    assert_eq!(probes.len(), 24);
    assert!(
        probes
            .iter()
            .all(|probe| probe.case == NormalizedCaseKind::ToolRequiredStream)
    );
    assert_eq!(
        suite.case_names(&probes),
        vec![NormalizedCaseKind::ToolRequiredStream.name()]
    );
    assert_eq!(
        suite.schema_variant_names(&probes),
        vec![
            "minimal_shallow",
            "canonical_current",
            "omp_style_i",
            "large_stress"
        ]
    );
    assert_eq!(
        suite.tool_choice_variant_names(&probes),
        vec!["required", "function"]
    );
    assert_eq!(
        unique_probe_max_tokens(&probes),
        vec![24, 48, 96],
        "required-tool TTFT matrix should keep small generation limits explicit"
    );

    let lanes = vec![
        lane("name=auto,endpoint=http://127.0.0.1:3000,model=qwen,kind=kir_ai_proxy"),
        lane(
            "name=json,endpoint=http://127.0.0.1:3001,model=qwen,kind=kir_ai_proxy,tool_parser=json",
        ),
        lane(
            "name=xml,endpoint=http://127.0.0.1:3002,model=qwen,kind=kir_ai_proxy,tool_parser=qwen-xml",
        ),
    ];
    let run_config = default_run_config_for_probe_suite(suite);
    let summary = normalized_plan_summary(&lanes, &probes, &run_config);

    assert_eq!(run_config.warmups, 0);
    assert_eq!(run_config.cache_phases, vec![CachePhase::Cold]);
    assert_eq!(summary.total_http_requests, 72);
    assert_eq!(summary.warmup_requests, 0);
    assert_eq!(summary.measured_requests, 72);
    assert_eq!(summary.probes[0].max_tokens, 24);
}

#[test]
fn qwen_mlx_tool_normalized_required_tool_ttft_request_bodies_cover_schema_and_token_variants() {
    let lane = lane("name=direct,endpoint=http://127.0.0.1:8080/v1,model=qwen,kind=direct_mlx");

    let minimal = probe_request_body(
        &lane,
        NormalizedProbePlan::new(
            NormalizedCaseKind::ToolRequiredStream,
            SchemaVariant::MinimalShallow,
            ToolChoiceVariant::Required,
        )
        .with_max_tokens(24),
        ProbePrompt::measured(128, 0, None),
    );
    assert_eq!(minimal["max_tokens"], 24);
    assert_eq!(minimal["tool_choice"], "required");
    assert_eq!(minimal["stream"], true);
    assert_eq!(
        minimal["tools"][0]["function"]["parameters"]["required"],
        json!(["probe_id", "case"])
    );
    assert!(
        minimal["tools"][0]["function"].get("description").is_none(),
        "minimal shallow schema should omit tool descriptions"
    );

    let omp = probe_request_body(
        &lane,
        NormalizedProbePlan::new(
            NormalizedCaseKind::ToolRequiredStream,
            SchemaVariant::OmpStyleI,
            ToolChoiceVariant::Function,
        )
        .with_max_tokens(48),
        ProbePrompt::measured(128, 0, None),
    );
    assert_eq!(omp["max_tokens"], 48);
    assert_eq!(
        omp["tool_choice"],
        json!({"type":"function","function":{"name":"record_qwen_tool_probe"}})
    );
    assert_eq!(
        omp["tools"][0]["function"]["parameters"]["properties"]["_i"]["type"],
        "integer"
    );

    let canonical = tool_schema_metadata(NormalizedProbePlan::new(
        NormalizedCaseKind::ToolRequiredStream,
        SchemaVariant::CanonicalCurrent,
        ToolChoiceVariant::Required,
    ));
    let large = tool_schema_metadata(NormalizedProbePlan::new(
        NormalizedCaseKind::ToolRequiredStream,
        SchemaVariant::LargeStress,
        ToolChoiceVariant::Required,
    ));
    assert!(
        large.bytes.expect("large schema bytes") > canonical.bytes.expect("canonical bytes"),
        "large stress schema should materially expand schema bytes"
    );
}

#[test]
fn qwen_mlx_tool_normalized_required_tool_ttft_report_includes_per_sample_fields_and_fastest_delta()
{
    let auto_lane = lane("name=auto,endpoint=http://127.0.0.1:3000,model=qwen,kind=kir_ai_proxy");
    let xml_lane = lane(
        "name=xml,endpoint=http://127.0.0.1:3001,model=qwen,kind=kir_ai_proxy,tool_parser=qwen-xml",
    );
    let mut auto = NormalizedLaneReport::planned(&auto_lane, 0, 0, None);
    let mut xml = NormalizedLaneReport::planned(&xml_lane, 0, 0, None);

    let probe = NormalizedProbePlan::new(
        NormalizedCaseKind::ToolRequiredStream,
        SchemaVariant::MinimalShallow,
        ToolChoiceVariant::Required,
    )
    .with_max_tokens(24);
    auto.samples = vec![required_tool_ttft_sample(probe, 120, 125, 300, 360)];
    xml.samples = vec![required_tool_ttft_sample(probe, 100, 105, 240, 310)];
    xml.samples[0].tool_required_stream_admin_metrics = Some(NormalizedAdminMetricsCapture {
        before: Some(json!({
            "stream_stalled_requests": 1,
            "no_progress_failures": 2,
            "validated_tool_call_ms": {"count": 4, "min": 70.0, "max": 70.0, "avg": 70.0}
        })),
        after: Some(json!({
            "stream_stalled_requests": 1,
            "no_progress_failures": 2,
            "validated_tool_call_ms": {"count": 5, "min": 70.0, "max": 290.0, "avg": 114.0}
        })),
        error: None,
    });

    let report = required_tool_ttft_matrix_report(&[auto, xml], &[probe], &[CachePhase::Cold]);

    assert_eq!(report.status, "reported");
    assert_eq!(report.rows.len(), 2);
    let auto_row = report
        .rows
        .iter()
        .find(|row| row.lane == "auto")
        .expect("auto row");
    assert_eq!(auto_row.max_tokens, 24);
    assert_eq!(auto_row.first_response_byte_ms, Some(120));
    assert_eq!(auto_row.first_parsed_sse_chunk_ms, Some(125));
    assert_eq!(auto_row.first_tool_delta_ms, Some(300));
    assert_eq!(auto_row.tool_finish_ms, Some(360));
    assert_eq!(auto_row.latency_delta_vs_fastest_lane_ms, Some(60));
    assert_eq!(auto_row.finish_reason.as_deref(), Some("tool_calls"));
    assert_eq!(auto_row.classification, "passed");

    let xml_row = report
        .rows
        .iter()
        .find(|row| row.lane == "xml")
        .expect("xml row");
    assert_eq!(xml_row.tool_parser, Some("qwen-xml"));
    assert_eq!(xml_row.latency_delta_vs_fastest_lane_ms, Some(0));
    assert_eq!(xml_row.validated_tool_call_ms, Some(290.0));
    assert_eq!(xml_row.stream_stalled_requests_delta, Some(0));
    assert_eq!(xml_row.no_progress_failures_delta, Some(0));
}

#[test]
fn qwen_mlx_tool_normalized_smoke_plan_summary_counts_warmups_and_tokens() {
    let lanes = vec![
        lane("name=direct,endpoint=http://127.0.0.1:8080/v1,model=qwen-a"),
        lane("name=proxy,endpoint=http://127.0.0.1:3000,model=qwen-b"),
    ];
    let probes = NormalizedProbeSuite::StablePrefixSmoke.probes();
    let run_config = NormalizedRunConfig::new(1, 1, 135_000, 1, 0)
        .with_cache_phases(vec![CachePhase::WarmSamePrompt]);

    let summary = normalized_plan_summary(&lanes, &probes, &run_config);

    assert_eq!(summary.probe_count, 1);
    assert_eq!(summary.lane_count, 2);
    assert_eq!(summary.warmup_requests, 2);
    assert_eq!(summary.measured_requests, 2);
    assert_eq!(summary.total_http_requests, 4);
    assert_eq!(summary.planned_prompt_token_budget, 540_000);
    assert_eq!(summary.cache_phases, vec!["warm_same_prompt"]);
    assert_eq!(summary.lanes, vec!["direct", "proxy"]);
    assert_eq!(summary.probes[0].case, "warm_prefix_repeated_turn_stream");
}

#[test]
fn qwen_mlx_tool_normalized_dry_run_records_warmups_as_planned_requests() {
    let lane = lane("name=direct,endpoint=http://127.0.0.1:8080/v1,model=qwen-a");
    let probes = NormalizedProbeSuite::StablePrefixSmoke.probes();
    let run_config = NormalizedRunConfig::new(1, 1, 256, 1, 0)
        .with_cache_phases(vec![CachePhase::WarmSamePrompt]);

    let report = NormalizedLaneReport::dry_run(&lane, &run_config, None, &probes);

    assert_eq!(report.samples.len(), 1);
    assert_eq!(report.planned_requests.len(), 2);
    assert_eq!(report.planned_requests[0].request_kind, "warmup");
    assert_eq!(report.planned_requests[0].cache_phase, "warm_same_prompt");
    assert_eq!(report.planned_requests[0].warmup_index, Some(0));
    assert_eq!(report.planned_requests[1].request_kind, "measured");
    assert_eq!(report.planned_requests[1].sample_index, Some(0));
}

#[test]
fn qwen_mlx_tool_normalized_live_and_dry_run_reports_share_selected_plan() {
    let lane = lane("name=direct,endpoint=http://127.0.0.1:8080/v1,model=qwen-a");
    let probes = NormalizedProbeSuite::StablePrefixSmoke.probes();
    let run_config = NormalizedRunConfig::new(1, 1, 256, 1, 0)
        .with_cache_phases(vec![CachePhase::WarmSamePrompt]);

    let dry_run = NormalizedLaneReport::dry_run(&lane, &run_config, None, &probes);
    let live_plan = NormalizedLaneReport::planned_with_requests(
        &lane,
        run_config.warmups,
        run_config.samples,
        &run_config,
        None,
        &probes,
    );

    assert_eq!(
        serde_json::to_value(&dry_run.planned_requests).expect("dry-run plan serializes"),
        serde_json::to_value(&live_plan.planned_requests).expect("live plan serializes")
    );
}

#[test]
fn qwen_mlx_tool_normalized_budget_guards_reject_oversized_plan() {
    let lanes = vec![
        lane("name=direct,endpoint=http://127.0.0.1:8080/v1,model=qwen-a"),
        lane("name=proxy,endpoint=http://127.0.0.1:3000,model=qwen-b"),
    ];
    let probes = NormalizedProbeSuite::StablePrefixSmoke.probes();
    let run_config = NormalizedRunConfig::new(1, 1, 135_000, 1, 0)
        .with_cache_phases(vec![CachePhase::WarmSamePrompt]);
    let summary = normalized_plan_summary(&lanes, &probes, &run_config);

    let requests_err =
        enforce_plan_budget(&summary, Some(3), None).expect_err("request budget rejects plan");
    assert!(
        requests_err.to_string().contains("--max-requests"),
        "error should mention request guard: {requests_err}"
    );

    let tokens_err =
        enforce_plan_budget(&summary, None, Some(539_999)).expect_err("token budget rejects plan");
    assert!(
        tokens_err
            .to_string()
            .contains("--max-planned-prompt-tokens"),
        "error should mention token guard: {tokens_err}"
    );
}

#[test]
fn qwen_mlx_tool_normalized_probe_suite_flag_and_profile_defaults() {
    let default_suite = parse_probe_suite_flag(&args(&[]), None).expect("default suite");
    assert_eq!(default_suite, NormalizedProbeSuite::FullMatrix);
    assert_eq!(default_suite.name(), "full_matrix");

    let focused = parse_probe_suite_flag(&args(&["--probe-suite", "focused-agentic-gate"]), None)
        .expect("focused suite");
    assert_eq!(focused, NormalizedProbeSuite::FocusedAgenticGate);

    let alias =
        parse_probe_suite_flag(&args(&["--focused-agentic-gate"]), None).expect("focused alias");
    assert_eq!(alias, NormalizedProbeSuite::FocusedAgenticGate);

    let prefill_default =
        parse_probe_suite_flag(&args(&[]), Some(NormalizedSweepProfile::QwenMlxPrefill135k))
            .expect("prefill profile suite");
    assert_eq!(prefill_default, NormalizedProbeSuite::PrefillSweep135k);

    let stable_default = parse_probe_suite_flag(
        &args(&[]),
        Some(NormalizedSweepProfile::QwenMlxStablePrefix),
    )
    .expect("stable profile suite");
    assert_eq!(stable_default, NormalizedProbeSuite::StableAgentPrefix);

    let prefill_explicit =
        parse_probe_suite_flag(&args(&["--probe-suite", "prefill-sweep-135k"]), None)
            .expect("prefill suite");
    assert_eq!(prefill_explicit.name(), "prefill_sweep_135k");
    assert_eq!(
        prefill_explicit.probes(),
        vec![
            NormalizedProbePlan::new(
                NormalizedCaseKind::ChatStream,
                SchemaVariant::None,
                ToolChoiceVariant::None,
            ),
            NormalizedProbePlan::new(
                NormalizedCaseKind::ToolRequiredStream,
                SchemaVariant::CanonicalCurrent,
                ToolChoiceVariant::Required,
            ),
            NormalizedProbePlan::new(
                NormalizedCaseKind::ContextRecallStream135k,
                SchemaVariant::CanonicalCurrent,
                ToolChoiceVariant::Required,
            ),
            NormalizedProbePlan::new(
                NormalizedCaseKind::WarmPrefixRepeatedTurnStream,
                SchemaVariant::CanonicalCurrent,
                ToolChoiceVariant::Required,
            ),
        ]
    );

    let stable_explicit =
        parse_probe_suite_flag(&args(&["--probe-suite", "stable-agent-prefix"]), None)
            .expect("stable suite");
    assert_eq!(stable_explicit.name(), "stable_agent_prefix");
    assert_eq!(
        stable_explicit.probes(),
        vec![
            NormalizedProbePlan::new(
                NormalizedCaseKind::ChatStream,
                SchemaVariant::None,
                ToolChoiceVariant::None,
            ),
            NormalizedProbePlan::new(
                NormalizedCaseKind::ToolRequiredStream,
                SchemaVariant::CanonicalCurrent,
                ToolChoiceVariant::Required,
            ),
            NormalizedProbePlan::new(
                NormalizedCaseKind::WarmPrefixRepeatedTurnStream,
                SchemaVariant::CanonicalCurrent,
                ToolChoiceVariant::Required,
            ),
        ]
    );

    let err = parse_probe_suite_flag(
        &args(&["--probe-suite", "full-matrix", "--focused-agentic-gate"]),
        None,
    )
    .expect_err("conflicting suite flags fail");
    assert!(
        err.to_string().contains("--focused-agentic-gate"),
        "error should mention alias conflict: {err}"
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
        tool_finish_latency_ms: Some(900),
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
        tool_finish_latency_ms: Some(950),
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
fn qwen_mlx_tool_normalized_agentic_streaming_ab_passes_when_proxy_first_delta_advances() {
    let direct = lane("name=direct,endpoint=http://127.0.0.1:8080/v1,model=qwen,kind=direct_mlx");
    let proxy = lane("name=kir-proxy,endpoint=http://127.0.0.1:3000,model=qwen,kind=kir_ai_proxy");
    let mut baseline_direct = NormalizedLaneReport::planned(&direct, 0, 0, None);
    let mut baseline_proxy = NormalizedLaneReport::planned(&proxy, 0, 0, None);
    let mut candidate_direct = NormalizedLaneReport::planned(&direct, 0, 0, None);
    let mut candidate_proxy = NormalizedLaneReport::planned(&proxy, 0, 0, None);

    baseline_direct.samples = vec![ab_tool_stream_sample(100, 130)];
    candidate_direct.samples = vec![ab_tool_stream_sample(101, 131)];
    baseline_proxy.samples = vec![ab_tool_stream_sample(120, 150)];
    candidate_proxy.samples = vec![ab_tool_stream_sample(80, 150)];

    let baseline_lanes = comparable_lanes_from_normalized(&[baseline_direct, baseline_proxy]);
    let candidate_lanes = comparable_lanes_from_normalized(&[candidate_direct, candidate_proxy]);
    let report = agentic_streaming_fast_path_ab_report(
        Some("baseline.json".to_owned()),
        &baseline_lanes,
        &candidate_lanes,
    );

    assert_eq!(report.status, "passed");
    assert_eq!(report.baseline_path.as_deref(), Some("baseline.json"));
    let proxy_row = report
        .rows
        .iter()
        .find(|row| row.lane == "kir-proxy")
        .expect("proxy row");
    assert_eq!(proxy_row.assertion_role, "fast_path_candidate");
    assert_eq!(
        proxy_row.baseline_p50_first_tool_delta_latency_ms,
        Some(120)
    );
    assert_eq!(
        proxy_row.candidate_p50_first_tool_delta_latency_ms,
        Some(80)
    );
    assert_eq!(proxy_row.first_tool_delta_delta_ms, Some(-40));
    assert_eq!(proxy_row.first_tool_delta_advanced, Some(true));
    assert_eq!(proxy_row.candidate_p50_tool_finish_latency_ms, Some(150));
    assert!(proxy_row.final_validation_unchanged);
    assert!(proxy_row.failure_reasons.is_empty());

    let direct_row = report
        .rows
        .iter()
        .find(|row| row.lane == "direct")
        .expect("direct control row");
    assert_eq!(direct_row.assertion_role, "control");
    assert_eq!(direct_row.first_tool_delta_advanced, None);
    assert!(direct_row.final_validation_unchanged);
}

#[test]
fn qwen_mlx_tool_normalized_agentic_streaming_ab_fails_when_proxy_first_delta_regresses() {
    let proxy = lane("name=kir-proxy,endpoint=http://127.0.0.1:3000,model=qwen,kind=kir_ai_proxy");
    let mut baseline_proxy = NormalizedLaneReport::planned(&proxy, 0, 0, None);
    let mut candidate_proxy = NormalizedLaneReport::planned(&proxy, 0, 0, None);
    baseline_proxy.samples = vec![ab_tool_stream_sample(80, 120)];
    candidate_proxy.samples = vec![ab_tool_stream_sample(90, 120)];

    let baseline_lanes = comparable_lanes_from_normalized(&[baseline_proxy]);
    let candidate_lanes = comparable_lanes_from_normalized(&[candidate_proxy]);
    let report = agentic_streaming_fast_path_ab_report(None, &baseline_lanes, &candidate_lanes);

    assert_eq!(report.status, "failed");
    let row = report.rows.first().expect("comparison row");
    assert_eq!(row.first_tool_delta_advanced, Some(false));
    assert!(
        row.failure_reasons
            .contains(&"first_tool_delta_not_advanced".to_owned())
    );
}

#[test]
fn qwen_mlx_tool_normalized_agentic_streaming_ab_fails_when_validation_changes() {
    let proxy = lane("name=kir-proxy,endpoint=http://127.0.0.1:3000,model=qwen,kind=kir_ai_proxy");
    let mut baseline_proxy = NormalizedLaneReport::planned(&proxy, 0, 0, None);
    let mut candidate_proxy = NormalizedLaneReport::planned(&proxy, 0, 0, None);
    baseline_proxy.samples = vec![ab_tool_stream_sample(120, 150)];
    let mut failed = ab_tool_stream_sample(80, 150);
    failed.status = "failed".to_owned();
    failed.classification = "response_validation_failed".to_owned();
    failed.finish_reason = Some("stop".to_owned());
    failed.error = Some("streamed tool arguments were not JSON".to_owned());
    candidate_proxy.samples = vec![failed];

    let baseline_lanes = comparable_lanes_from_normalized(&[baseline_proxy]);
    let candidate_lanes = comparable_lanes_from_normalized(&[candidate_proxy]);
    let report = agentic_streaming_fast_path_ab_report(None, &baseline_lanes, &candidate_lanes);

    assert_eq!(report.status, "failed");
    let row = report.rows.first().expect("comparison row");
    assert_eq!(row.first_tool_delta_advanced, Some(true));
    assert!(!row.final_validation_unchanged);
    assert!(
        row.failure_reasons
            .contains(&"final_validation_changed".to_owned())
    );
}

#[test]
fn qwen_mlx_tool_normalized_prefill_sweep_report_ranks_and_flags_invalid_lanes() {
    let lane_512 = lane(
        "name=mlx-prefill-512,endpoint=http://127.0.0.1:8081/v1,model=qwen,kind=direct_mlx,mlx_prefill_step_size=512",
    );
    let lane_1024 = lane(
        "name=mlx-prefill-1024,endpoint=http://127.0.0.1:8082/v1,model=qwen,kind=direct_mlx,mlx_prefill_step_size=1024",
    );
    let lane_proxy = lane(
        "name=kir-prefill-512,endpoint=http://127.0.0.1:3001,model=qwen,kind=kir_ai_proxy,mlx_prefill_step_size=512",
    );
    let mut report_512 = NormalizedLaneReport::planned(&lane_512, 0, 0, None);
    let mut report_1024 = NormalizedLaneReport::planned(&lane_1024, 0, 0, None);
    let mut report_proxy = NormalizedLaneReport::planned(&lane_proxy, 0, 0, None);
    let probe = NormalizedProbePlan::new(
        NormalizedCaseKind::ChatStream,
        SchemaVariant::None,
        ToolChoiceVariant::None,
    );

    let mut sample_512 = prefill_sweep_sample(
        NormalizedCaseKind::ChatStream,
        CachePhase::Cold,
        RunMode::Sequential,
        120,
    );
    sample_512.response_headers = Some(BTreeMap::from([(
        "content-type".to_owned(),
        "text/event-stream".to_owned(),
    )]));
    let sample_1024 = prefill_sweep_sample(
        NormalizedCaseKind::ChatStream,
        CachePhase::Cold,
        RunMode::Sequential,
        90,
    );
    let mut invalid_proxy = NormalizedSampleReport::base(
        probe,
        CachePhase::Cold,
        RunMode::Sequential,
        0,
        None,
        false,
        135_000,
    );
    invalid_proxy.status = "failed".to_owned();
    invalid_proxy.classification = "response_validation_failed".to_owned();
    invalid_proxy.failure_classification = Some("progress_validation_failed".to_owned());
    invalid_proxy.latency_ms = Some(140);
    invalid_proxy.stream_timing = StreamTimingReport {
        first_byte_latency_ms: Some(95),
        first_sse_data_latency_ms: Some(96),
        first_content_delta_latency_ms: None,
        first_tool_delta_latency_ms: None,
        tool_finish_latency_ms: None,
        first_semantic_delta_latency_ms: None,
    };
    report_512.samples = vec![sample_512];
    report_1024.samples = vec![sample_1024];
    report_proxy.samples = vec![invalid_proxy];
    report_proxy.admin_metrics = NormalizedAdminMetricsCapture {
        before: Some(json!({
            "stream_stalled_requests": 2,
            "no_progress_failures": 4,
            "process_rss_bytes": 100,
            "mlx": {
                "stream_first_upstream_byte_ms": {"count": 1, "min": 10.0, "max": 10.0, "avg": 10.0},
                "stream_first_parsed_chunk_ms": {"count": 1, "min": 15.0, "max": 15.0, "avg": 15.0},
                "stream_first_tool_delta_ms": {"count": 1, "min": 40.0, "max": 40.0, "avg": 40.0}
            }
        })),
        after: Some(json!({
            "stream_stalled_requests": 3,
            "no_progress_failures": 6,
            "process_rss_bytes": 160,
            "mlx": {
                "stream_first_upstream_byte_ms": {"count": 2, "min": 10.0, "max": 30.0, "avg": 20.0},
                "stream_first_parsed_chunk_ms": {"count": 2, "min": 15.0, "max": 35.0, "avg": 25.0},
                "stream_first_tool_delta_ms": {"count": 2, "min": 40.0, "max": 80.0, "avg": 60.0}
            }
        })),
        error: None,
    };

    let report = prefill_sweep_report(&[report_512, report_1024, report_proxy], &[probe]);
    let row = report
        .rows
        .iter()
        .find(|row| {
            row.case == "chat_stream" && row.cache_phase == "cold" && row.run_mode == "sequential"
        })
        .expect("cold chat stream row");

    assert_eq!(report.status, "reported");
    assert_eq!(row.fastest_lane.as_deref(), Some("mlx-prefill-1024"));
    assert_eq!(row.lanes[0].lane, "mlx-prefill-1024");
    assert_eq!(row.lanes[0].p50_first_semantic_delta_latency_ms, Some(90));
    assert_eq!(row.lanes[0].prefill_step_size, DefaultOrU64::Value(1024));
    assert_eq!(row.lanes[0].lane_kind, "direct_mlx");
    assert_eq!(row.lanes[1].latency_delta_vs_fastest_ms, Some(30));
    assert_eq!(
        row.lanes[1].response_headers[0]["content-type"],
        "text/event-stream"
    );

    let invalid = row
        .lanes
        .iter()
        .find(|metric| metric.lane == "kir-prefill-512")
        .expect("invalid proxy metric");
    assert!(!invalid.valid);
    assert!(
        invalid
            .invalid_reasons
            .contains(&"sample_failed".to_owned())
    );
    assert!(invalid.invalid_reasons.contains(&"missing_ttft".to_owned()));
    assert!(
        invalid
            .invalid_reasons
            .contains(&"missing_stream_delta".to_owned())
    );
    assert!(
        invalid
            .invalid_reasons
            .contains(&"admin_stalled_request_delta".to_owned())
    );
    assert!(
        invalid
            .invalid_reasons
            .contains(&"admin_no_progress_delta".to_owned())
    );
    assert!(
        invalid
            .invalid_reasons
            .contains(&"progress_validation_failed".to_owned())
    );
    assert_eq!(
        invalid
            .failure_classifications
            .get("progress_validation_failed"),
        Some(&1)
    );
    assert_eq!(invalid.stream_stalled_requests_delta, Some(1));
    assert_eq!(invalid.no_progress_failures_delta, Some(2));
    assert_eq!(invalid.process_rss_bytes_after, Some(160));
    assert_eq!(
        invalid
            .admin_mlx_upstream_timing
            .as_ref()
            .expect("admin mlx timing")
            .stream_first_upstream_byte_ms
            .count_delta,
        Some(1)
    );
}

#[test]
fn qwen_mlx_tool_normalized_prefill_concurrency_report_covers_acceptance_matrix() {
    let lane_config = lane(
        "name=kir-prefill,endpoint=http://127.0.0.1:3000,model=local-qwen36-mlx,kind=kir_ai_proxy,mlx_prefill_step_size=2048",
    );
    let mut lane_report = NormalizedLaneReport::planned(&lane_config, 0, 0, None);
    let probe = NormalizedProbePlan::new(
        NormalizedCaseKind::ChatStream,
        SchemaVariant::None,
        ToolChoiceVariant::None,
    );
    let mut cold = prefill_sweep_sample(
        NormalizedCaseKind::ChatStream,
        CachePhase::Cold,
        RunMode::Sequential,
        180,
    );
    cold.cached_tokens = Some(0);
    cold.cached_tokens_status = "present";
    let mut warm = prefill_sweep_sample(
        NormalizedCaseKind::ChatStream,
        CachePhase::WarmSamePrompt,
        RunMode::Sequential,
        75,
    );
    warm.cached_tokens = Some(128_000);
    warm.cached_tokens_status = "present";
    let mut mixed = prefill_sweep_sample(
        NormalizedCaseKind::ChatStream,
        CachePhase::Cold,
        RunMode::Concurrent,
        210,
    );
    mixed.request_index = Some(1);
    mixed.cached_tokens = Some(0);
    mixed.cached_tokens_status = "present";
    lane_report.samples = vec![cold, warm];
    lane_report.concurrent_samples = vec![mixed];
    lane_report.admin_metrics = NormalizedAdminMetricsCapture {
        before: Some(json!({
            "scheduler_prefill_yields": 2,
            "scheduler_prefill_yields_to_decode": 1,
            "scheduler_prefill_yield_reacquire_waits": 2,
            "scheduler_prefill_yield_reacquire_wait_ms_total": 10.5,
            "scheduler_prefill_yield_reacquire_wait_ms_max": 8.0,
            "backend_metrics": {
                "native_text_prefix_cache": {
                    "qwen": {
                        "checkpoint_reuse_hits": 1,
                        "checkpoint_reused_tokens": 2048,
                        "avoided_prefill_tokens": 4096
                    }
                }
            }
        })),
        after: Some(json!({
            "scheduler_prefill_yields": 5,
            "scheduler_prefill_yields_to_decode": 3,
            "scheduler_prefill_yield_reacquire_waits": 5,
            "scheduler_prefill_yield_reacquire_wait_ms_total": 28.5,
            "scheduler_prefill_yield_reacquire_wait_ms_max": 12.0,
            "backend_metrics": {
                "native_text_prefix_cache": {
                    "qwen": {
                        "checkpoint_reuse_hits": 4,
                        "checkpoint_reused_tokens": 8192,
                        "avoided_prefill_tokens": 12000
                    }
                }
            }
        })),
        error: None,
    };

    let report = prefill_concurrency_report(&[lane_report], &[probe]);

    assert_eq!(report.status, "reported");
    assert_eq!(
        report
            .scenarios
            .iter()
            .map(|scenario| scenario.scenario)
            .collect::<Vec<_>>(),
        [
            "cold_long_context_prefill",
            "warm_checkpoint_reuse",
            "mixed_long_prefill_short_decode_concurrency",
        ]
    );

    let cold = &report.scenarios[0].lanes[0];
    assert_eq!(cold.lane, "kir-prefill");
    assert_eq!(cold.sample_count, 1);
    assert_eq!(cold.p50_first_semantic_delta_latency_ms, Some(180));
    assert_eq!(cold.avg_cached_tokens, Some(0.0));
    assert_eq!(cold.scheduler_prefill.prefill_yields_delta, Some(3));
    assert_eq!(
        cold.scheduler_prefill.prefill_yields_to_decode_delta,
        Some(2)
    );
    assert_eq!(
        cold.scheduler_prefill.prefill_yield_reacquire_waits_delta,
        Some(3)
    );
    assert_eq!(cold.checkpoint_reuse.checkpoint_reuse_hits_delta, Some(3));
    assert_eq!(
        cold.checkpoint_reuse.checkpoint_reused_tokens_delta,
        Some(6144)
    );

    let warm = &report.scenarios[1].lanes[0];
    assert_eq!(warm.p50_first_semantic_delta_latency_ms, Some(75));
    assert_eq!(warm.avg_cached_tokens, Some(128_000.0));
    assert_eq!(
        warm.checkpoint_reuse.avoided_prefill_tokens_delta,
        Some(7904)
    );

    let mixed = &report.scenarios[2].lanes[0];
    assert_eq!(mixed.run_mode, "concurrent");
    assert_eq!(mixed.request_count, 1);
    assert_eq!(mixed.p50_first_semantic_delta_latency_ms, Some(210));
}

#[test]
fn qwen_mlx_tool_normalized_prefill_concurrency_report_keeps_missing_counters_null() {
    let lane_config =
        lane("name=direct,endpoint=http://127.0.0.1:8080/v1,model=qwen,kind=direct_mlx");
    let mut lane_report = NormalizedLaneReport::planned(&lane_config, 0, 0, None);
    let probe = NormalizedProbePlan::new(
        NormalizedCaseKind::ChatStream,
        SchemaVariant::None,
        ToolChoiceVariant::None,
    );
    lane_report.samples = vec![prefill_sweep_sample(
        NormalizedCaseKind::ChatStream,
        CachePhase::Cold,
        RunMode::Sequential,
        120,
    )];

    let report = prefill_concurrency_report(&[lane_report], &[probe]);
    let metric = &report.scenarios[0].lanes[0];

    assert_eq!(metric.scheduler_prefill.prefill_yields_delta, None);
    assert_eq!(metric.scheduler_prefill.prefill_yields_after, None);
    assert_eq!(metric.checkpoint_reuse.checkpoint_reuse_hits_delta, None);
    assert_eq!(metric.checkpoint_reuse.checkpoint_reused_tokens_delta, None);
}

#[test]
fn qwen_mlx_tool_normalized_failed_samples_classify_safety_failures() {
    let probe = NormalizedProbePlan::new(
        NormalizedCaseKind::ContextRecallStream135k,
        SchemaVariant::CanonicalCurrent,
        ToolChoiceVariant::Required,
    );
    let context = SampleContext {
        probe,
        phase: CachePhase::Cold,
        run_mode: RunMode::Sequential,
        sample_index: 0,
        request_index: None,
        planned_prompt_tokens: 135_000,
        prewarmed: false,
        expected_probe_id: probe.case.probe_id().to_owned(),
        expected_marker: Some(CONTEXT_RECALL_STREAM_135K_MARKER.to_owned()),
    };

    let oom = failed_sample(
        context.clone(),
        "stream_body_failed",
        Duration::from_millis(10),
        Some(500),
        None,
        "MLX Metal command buffer failed: out of memory".to_owned(),
        StreamTimingReport::default(),
    );
    assert_eq!(oom.failure_classification.as_deref(), Some("oom"));

    let metal = failed_sample(
        context.clone(),
        "stream_body_failed",
        Duration::from_millis(10),
        Some(500),
        None,
        "MTLCommandBufferErrorDomain: command buffer execution failed".to_owned(),
        StreamTimingReport::default(),
    );
    assert_eq!(
        metal.failure_classification.as_deref(),
        Some("metal_failure")
    );

    let timeout = failed_sample(
        context.clone(),
        "stream_http_request_failed",
        Duration::from_millis(30 * 60 * 1000),
        None,
        None,
        "operation timed out".to_owned(),
        StreamTimingReport::default(),
    );
    assert_eq!(
        timeout.failure_classification.as_deref(),
        Some("resource_limit_exceeded")
    );

    let validation = sample_from_validation(
        context,
        Err("streamed recall tool arguments were not JSON".to_owned()),
        ProbeResponseMetadata {
            latency: Duration::from_millis(100),
            stream_timing: StreamTimingReport::default(),
            http_status: Some(200),
            response_headers: None,
            finish_reason: Some("stop".to_owned()),
            usage: usage_from_value(None),
        },
    );
    assert_eq!(
        validation.failure_classification.as_deref(),
        Some("progress_validation_failed")
    );
}

#[test]
fn qwen_mlx_tool_normalized_stable_prefix_report_tracks_warm_reuse_and_admin_observations() {
    let direct =
        lane("name=mlx-stable-prefix,endpoint=http://127.0.0.1:8080/v1,model=qwen,kind=direct_mlx");
    let proxy =
        lane("name=kir-stable-prefix,endpoint=http://127.0.0.1:3000,model=qwen,kind=kir_ai_proxy");
    let mut direct_report = NormalizedLaneReport::planned(&direct, 0, 0, None);
    let mut proxy_report = NormalizedLaneReport::planned(&proxy, 0, 0, None);
    let probe = NormalizedProbePlan::new(
        NormalizedCaseKind::WarmPrefixRepeatedTurnStream,
        SchemaVariant::CanonicalCurrent,
        ToolChoiceVariant::Required,
    );

    direct_report.samples = vec![
        stable_prefix_sample(probe, CachePhase::Cold, 130, Some(0), None),
        stable_prefix_sample(probe, CachePhase::WarmSamePrompt, 65, Some(1000), None),
    ];
    proxy_report.samples = vec![stable_prefix_sample(
        probe,
        CachePhase::WarmSamePrompt,
        80,
        Some(750),
        Some("proxy-warm"),
    )];
    proxy_report.admin_metrics = NormalizedAdminMetricsCapture {
        before: None,
        after: Some(json!({
            "request_cache": {
                "capacity": 128,
                "recent": [
                    {
                        "request_id": "unrelated",
                        "model": "local-qwen36",
                        "streamed": true,
                        "prompt_tokens": 1000,
                        "cached_tokens": 0,
                        "uncached_tokens": 1000,
                        "cache_status": "miss",
                        "latency_ms": 200
                    },
                    {
                        "request_id": "proxy-warm",
                        "model": "local-qwen36",
                        "streamed": true,
                        "prompt_tokens": 1000,
                        "cached_tokens": 750,
                        "uncached_tokens": 250,
                        "cache_status": "partial",
                        "latency_ms": 95
                    }
                ]
            }
        })),
        error: None,
    };

    let report = stable_prefix_report(&[direct_report, proxy_report], &[probe]);
    let row = report
        .rows
        .iter()
        .find(|row| {
            row.case == "warm_prefix_repeated_turn_stream"
                && row.cache_phase == "warm_same_prompt"
                && row.run_mode == "sequential"
        })
        .expect("warm stable-prefix row");

    assert_eq!(report.status, "reported");
    assert_eq!(row.fastest_lane.as_deref(), Some("mlx-stable-prefix"));
    assert_eq!(row.lanes[0].lane, "mlx-stable-prefix");
    assert_eq!(row.lanes[0].p50_first_semantic_delta_latency_ms, Some(65));
    assert_eq!(row.lanes[0].p50_first_tool_delta_latency_ms, Some(65));
    assert_eq!(row.lanes[0].avg_cached_tokens, Some(1000.0));
    assert_eq!(row.lanes[0].cache_status_counts.get("hit"), Some(&1));
    assert_eq!(row.lanes[1].latency_delta_vs_fastest_ms, Some(15));
    assert_eq!(row.lanes[1].cache_status_counts.get("partial"), Some(&1));
    assert_eq!(row.lanes[1].request_cache_observations.len(), 1);
    assert_eq!(
        row.lanes[1].request_cache_observations[0].request_id,
        "proxy-warm"
    );
    assert_eq!(
        row.lanes[1].request_cache_observations[0].cache_status,
        "partial"
    );
}

#[test]
fn qwen_mlx_tool_normalized_latest_performance_comparison_reports_required_sources() {
    let direct =
        lane("name=mlx-latest,endpoint=http://127.0.0.1:8080/v1,model=qwen,kind=direct_mlx");
    let proxy = lane("name=kir-latest,endpoint=http://127.0.0.1:3000,model=qwen,kind=kir_ai_proxy");
    let mut direct_report = NormalizedLaneReport::planned(&direct, 0, 0, None);
    let mut proxy_report = NormalizedLaneReport::planned(&proxy, 0, 0, None);

    direct_report.samples = vec![
        latest_plain_stream_sample(164, 5_384, 35.7),
        latest_tool_stream_sample(1_940, 1_969, 36.0),
        latest_cache_sample(CachePhase::Cold, 10_000, 170, 35.0, Some(0)),
        latest_cache_sample(CachePhase::WarmSamePrompt, 250, 174, 41.2, Some(1_000)),
    ];
    proxy_report.samples = vec![
        latest_plain_stream_sample(166, 5_383, 32.5),
        latest_tool_stream_sample(1_960, 1_989, 32.0),
        latest_cache_sample(CachePhase::Cold, 10_500, 175, 32.0, None),
        latest_cache_sample(CachePhase::WarmSamePrompt, 230, 172, 41.0, None),
    ];

    let baselines = EngineDbBaselineExport {
        source: Some("reports/benchmarks/benchmarks.sqlite".to_owned()),
        rows: vec![EngineDbBaselineRow {
            engine: "Rapid-MLX".to_owned(),
            profile: "rapid-0615-qwen35-kv4-135k".to_owned(),
            model: Some("Qwen3.6 35B A3B 4bit".to_owned()),
            probe: "chat_stream".to_owned(),
            ttfi_ms: Some(80.6),
            first_tool_delta_ms: None,
            validated_tool_call_ms: None,
            total_latency_ms: None,
            tokens_per_second: Some(26.3),
            cache_cold_latency_ms: None,
            cache_warm_latency_ms: None,
            cache_speedup: None,
            cached_tokens: None,
            notes: Some("DB row 2026-05-07".to_owned()),
        }],
    };

    let report =
        latest_performance_comparison_report(&[direct_report, proxy_report], Some(&baselines));
    let value = serde_json::to_value(&report).expect("comparison serializes");
    let rows = value["rows"].as_array().expect("comparison rows");

    assert_eq!(report.status, "reported");
    assert_eq!(
        value["engine_db_baseline_source"],
        "reports/benchmarks/benchmarks.sqlite"
    );
    assert_eq!(value["evidence"]["has_kir_latest"], true);
    assert_eq!(value["evidence"]["has_direct_mlx_latest"], true);
    assert_eq!(value["evidence"]["has_engine_db_baselines"], true);
    assert_eq!(value["evidence"]["has_ttfi_ms"], true);
    assert_eq!(value["evidence"]["has_cache_metrics"], true);
    assert_eq!(value["evidence"]["has_tokens_per_second"], true);

    let kir_plain = rows
        .iter()
        .find(|row| row["source_kind"] == "latest_kir" && row["probe"] == "plain_stream")
        .expect("Kir plain-stream row");
    assert_eq!(kir_plain["ttfi_ms"], 166.0);
    assert_eq!(kir_plain["tokens_per_second"], 32.5);

    let direct_cache = rows
        .iter()
        .find(|row| row["source_kind"] == "direct_mlx" && row["probe"] == "prefix_cache")
        .expect("direct MLX prefix-cache row");
    assert_eq!(direct_cache["cache_cold_latency_ms"], 10_000.0);
    assert_eq!(direct_cache["cache_warm_latency_ms"], 250.0);
    assert_eq!(direct_cache["cache_speedup"], 40.0);
    assert_eq!(direct_cache["cached_tokens"], 1_000);

    assert!(rows.iter().any(|row| {
        row["source_kind"] == "engine_db_baseline"
            && row["engine"] == "Rapid-MLX"
            && row["ttfi_ms"] == 80.6
            && row["tokens_per_second"] == 26.3
    }));
}

#[tokio::test]
async fn qwen_mlx_tool_normalized_dry_run_loads_engine_db_baseline_export() {
    let temp = tempfile::tempdir().expect("tempdir");
    let baseline = temp.path().join("engine-db-baselines.json");
    let output = temp.path().join("trace.json");
    tokio::fs::write(
        &baseline,
        serde_json::json!({
            "source": "reports/benchmarks/benchmarks.sqlite",
            "rows": [{
                "engine": "oMLX",
                "profile": "omlx-qwen-a3b-ssd-cache-ctx-135k",
                "model": "Qwen3.6 35B A3B UD Q4",
                "probe": "chat_stream",
                "ttft_ms": 10.8,
                "tok_s": 23.4,
                "notes": "DB row 2026-05-04"
            }]
        })
        .to_string(),
    )
    .await
    .expect("write baseline fixture");

    run_qwen_mlx_tool_normalized_bench(&args(&[
        "--dry-run",
        "--engine-db-baselines",
        baseline.to_str().expect("utf8 baseline path"),
        "--output",
        output.to_str().expect("utf8 output path"),
        "--probe-suite",
        "stable-agent-prefix",
        "--lane",
        "name=direct,endpoint=http://127.0.0.1:8080/v1,model=qwen,kind=direct_mlx",
        "--lane",
        "name=proxy,endpoint=http://127.0.0.1:3000,model=qwen,kind=kir_ai_proxy",
    ]))
    .await
    .expect("dry-run benchmark with engine DB baseline export");

    let value: Value =
        serde_json::from_slice(&tokio::fs::read(&output).await.expect("read dry-run output"))
            .expect("dry-run JSON");
    let comparison = &value["latest_performance_comparison"];

    assert_eq!(
        comparison["engine_db_baseline_source"],
        "reports/benchmarks/benchmarks.sqlite"
    );
    assert_eq!(comparison["status"], "partial");
    assert_eq!(comparison["evidence"]["has_engine_db_baselines"], true);
    assert_eq!(comparison["rows"][0]["source_kind"], "engine_db_baseline");
    assert_eq!(comparison["rows"][0]["ttfi_ms"], 10.8);
    assert_eq!(comparison["rows"][0]["tokens_per_second"], 23.4);
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
fn qwen_mlx_tool_normalized_sse_parser_records_tool_finish_latency() {
    let mut buffer = concat!(
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,",
        "\"function\":{\"name\":\"record_qwen_tool_probe\",\"arguments\":\"{}\"}}]},",
        "\"finish_reason\":\"tool_calls\"}]}\n\n",
        "data: [DONE]\n\n"
    )
    .to_owned();
    let mut assembly = StreamAssembly::default();
    let mut timings = StreamTimingTracker::default();

    consume_sse_buffer(
        &mut buffer,
        &mut assembly,
        &mut timings,
        std::time::Duration::from_millis(42),
    );

    let report = timings.to_report();
    assert_eq!(report.first_tool_delta_latency_ms, Some(42));
    assert_eq!(report.tool_finish_latency_ms, Some(42));
    assert_eq!(assembly.finish_reason.as_deref(), Some("tool_calls"));
}

#[test]
fn qwen_mlx_tool_normalized_admin_metrics_url_uses_server_root() {
    assert_eq!(
        admin_metrics_url("http://127.0.0.1:3000"),
        "http://127.0.0.1:3000/admin/metrics"
    );
    assert_eq!(
        admin_metrics_url("http://127.0.0.1:8080/v1"),
        "http://127.0.0.1:8080/admin/metrics"
    );
}

#[tokio::test]
async fn qwen_mlx_tool_normalized_admin_metrics_skips_non_proxy_lanes() {
    let lane_config = lane("name=direct,endpoint=http://127.0.0.1:9/v1,model=qwen,kind=direct_mlx");
    let mut lane_report = NormalizedLaneReport::planned(&lane_config, 0, 0, None);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(50))
        .build()
        .expect("client builds");
    let run_config = NormalizedRunConfig::new(0, 1, 128, 1, 0);
    let progress = NormalizedProgress::new(0);

    run_lane(
        &lane_config,
        &mut lane_report,
        LaneRunContext {
            client: &client,
            run_config: &run_config,
            probes: &[],
            admin_token: None,
            prompt_tokenizer: None,
            progress: &progress,
        },
    )
    .await;

    assert!(lane_report.admin_metrics.before.is_none());
    assert!(lane_report.admin_metrics.after.is_none());
    assert!(lane_report.admin_metrics.error.is_none());
}

#[tokio::test]
async fn qwen_mlx_tool_normalized_admin_metrics_uses_independent_short_timeout() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener binds");
    let addr = listener.local_addr().expect("listener address");
    let server = tokio::spawn(async move {
        let (_socket, _) = listener.accept().await.expect("accepts connection");
        tokio::time::sleep(Duration::from_secs(1)).await;
    });
    let lane_config = lane(&format!(
        "name=proxy,endpoint=http://{addr},model=qwen,kind=kir_ai_proxy"
    ));
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .expect("client builds");

    let started = Instant::now();
    let result = tokio::time::timeout(
        Duration::from_millis(500),
        capture_normalized_admin_metrics(&client, &lane_config, None),
    )
    .await;
    server.abort();

    let err = result
        .expect("admin metrics uses a short independent timeout")
        .expect_err("hung admin metrics request fails");
    assert!(
        started.elapsed() < Duration::from_millis(500),
        "admin metrics elapsed {:?}",
        started.elapsed()
    );
    assert!(
        err.contains("admin metrics request failed"),
        "unexpected error: {err}"
    );
}

#[test]
fn qwen_mlx_tool_normalized_tool_stream_timing_report_includes_admin_stage_deltas() {
    let lane_config =
        lane("name=proxy,endpoint=http://127.0.0.1:3000,model=qwen,kind=kir_ai_proxy");
    let mut lane_report = NormalizedLaneReport::planned(&lane_config, 0, 0, None);
    let mut sample = passed_sample(
        NormalizedCaseKind::ToolRequiredStream,
        CachePhase::Cold,
        RunMode::Sequential,
        0,
        None,
        800,
        16,
    );
    sample.stream_timing = StreamTimingReport {
        first_byte_latency_ms: Some(12),
        first_sse_data_latency_ms: Some(13),
        first_content_delta_latency_ms: None,
        first_tool_delta_latency_ms: Some(30),
        tool_finish_latency_ms: Some(70),
        first_semantic_delta_latency_ms: Some(30),
    };
    sample.request_id = Some("tool-req-1".to_owned());
    lane_report.samples.push(sample);
    lane_report.admin_metrics = NormalizedAdminMetricsCapture {
        before: Some(json!({
            "first_tool_delta_ms": {"count": 2, "min": 1.0, "max": 3.0, "avg": 2.0},
            "first_tool_delta_after_ttft_ms": {"count": 2, "min": 1.0, "max": 2.0, "avg": 1.5},
            "tool_argument_assembly_ms": {"count": 2, "min": 1.0, "max": 3.0, "avg": 2.0},
            "tool_intent_fill_ms": {"count": 2, "min": 1.0, "max": 3.0, "avg": 2.0},
            "tool_schema_validation_ms": {"count": 2, "min": 1.0, "max": 3.0, "avg": 2.0},
            "tool_finish_ms": {"count": 2, "min": 1.0, "max": 3.0, "avg": 2.0},
            "validated_tool_call_ms": {"count": 2, "min": 1.0, "max": 3.0, "avg": 2.0},
            "backend_metrics": {
                "mlx": {
                    "stream_first_upstream_byte_ms": {"count": 2, "min": 1.0, "max": 4.0, "avg": 2.5},
                    "stream_first_parsed_chunk_ms": {"count": 2, "min": 2.0, "max": 5.0, "avg": 3.5},
                    "stream_first_tool_delta_ms": {"count": 2, "min": 3.0, "max": 6.0, "avg": 4.5}
                }
            }
        })),
        after: Some(json!({
            "first_tool_delta_ms": {"count": 3, "min": 1.0, "max": 30.0, "avg": 11.0},
            "first_tool_delta_after_ttft_ms": {"count": 3, "min": 1.0, "max": 20.0, "avg": 8.0},
            "tool_argument_assembly_ms": {"count": 3, "min": 1.0, "max": 40.0, "avg": 14.0},
            "tool_intent_fill_ms": {"count": 3, "min": 1.0, "max": 50.0, "avg": 17.0},
            "tool_schema_validation_ms": {"count": 3, "min": 1.0, "max": 60.0, "avg": 20.0},
            "tool_finish_ms": {"count": 3, "min": 1.0, "max": 70.0, "avg": 23.0},
            "validated_tool_call_ms": {"count": 3, "min": 1.0, "max": 70.0, "avg": 23.0},
            "backend_metrics": {
                "mlx": {
                    "stream_first_upstream_byte_ms": {"count": 3, "min": 1.0, "max": 10.0, "avg": 5.0},
                    "stream_first_parsed_chunk_ms": {"count": 3, "min": 2.0, "max": 20.0, "avg": 8.0},
                    "stream_first_tool_delta_ms": {"count": 3, "min": 3.0, "max": 25.0, "avg": 10.0}
                }
            },
            "tool_stream": {
                "capacity": 128,
                "recent": [
                    {
                        "request_id": "unrelated",
                        "model": "qwen",
                        "streamed": true,
                        "kir_first_tool_delta_ms": 99,
                        "validated_tool_call_ms": 100,
                        "latency_ms": 120
                    },
                    {
                        "request_id": "tool-req-1",
                        "model": "qwen",
                        "streamed": true,
                        "kir_first_tool_delta_ms": 30,
                        "kir_first_tool_delta_after_ttft_ms": 20,
                        "tool_argument_assembly_ms": 40,
                        "tool_intent_fill_ms": 50,
                        "tool_schema_validation_ms": 60,
                        "tool_finish_ms": 70,
                        "validated_tool_call_ms": 70,
                        "mlx_response_headers_ms": 8,
                        "mlx_first_upstream_byte_ms": 10,
                        "mlx_first_parsed_chunk_ms": 20,
                        "mlx_first_tool_delta_ms": 25,
                        "mlx_upstream_complete_ms": 65,
                        "latency_ms": 80
                    }
                ]
            }
        })),
        error: None,
    };

    let report = tool_required_stream_timing_report(&[lane_report]);
    assert_eq!(report.status, "reported");
    assert_eq!(report.lanes[0].p50_first_tool_delta_latency_ms, Some(30));
    assert_eq!(report.lanes[0].p50_tool_finish_latency_ms, Some(70));
    let admin = report.lanes[0]
        .admin_metrics
        .as_ref()
        .expect("admin metrics");
    assert_eq!(admin.tool_argument_assembly_ms.count_delta, Some(1));
    assert_eq!(admin.tool_finish_ms.count_delta, Some(1));
    let report_json = serde_json::to_value(&report).expect("report serializes");
    assert_eq!(
        report_json
            .pointer("/lanes/0/admin_metrics/first_tool_delta_after_ttft_ms/count_delta")
            .and_then(serde_json::Value::as_i64),
        Some(1),
        "bench report should expose the server's after-TTFT tool delta metric"
    );
    assert_eq!(admin.mlx_stream_first_upstream_byte_ms.count_delta, Some(1));
    let observations = report_json
        .pointer("/lanes/0/tool_stream_observations")
        .and_then(serde_json::Value::as_array)
        .expect("tool-stream observations are included");
    assert_eq!(observations.len(), 1);
    assert_eq!(observations[0]["request_id"], "tool-req-1");
    assert_eq!(observations[0]["client_first_byte_ms"], 12);
    assert_eq!(observations[0]["client_first_sse_data_ms"], 13);
    assert_eq!(observations[0]["client_visible_first_tool_delta_ms"], 30);
    assert_eq!(observations[0]["kir_first_tool_delta_ms"], 30);
    assert_eq!(observations[0]["kir_first_tool_delta_after_ttft_ms"], 20);
    assert_eq!(observations[0]["mlx_first_tool_delta_ms"], 25);
    assert_eq!(observations[0]["validated_tool_call_ms"], 70);
}

#[test]
fn qwen_mlx_tool_normalized_tool_stream_attribution_reports_per_sample_admin_timing() {
    let lane_config =
        lane("name=proxy,endpoint=http://127.0.0.1:3000,model=qwen,kind=kir_ai_proxy");
    let mut lane_report = NormalizedLaneReport::planned(&lane_config, 0, 0, None);
    let mut sample = passed_sample(
        NormalizedCaseKind::ToolRequiredStream,
        CachePhase::Cold,
        RunMode::Sequential,
        0,
        None,
        800,
        16,
    );
    sample.stream_timing = StreamTimingReport {
        first_byte_latency_ms: Some(12),
        first_sse_data_latency_ms: Some(13),
        first_content_delta_latency_ms: None,
        first_tool_delta_latency_ms: Some(30),
        tool_finish_latency_ms: Some(70),
        first_semantic_delta_latency_ms: Some(30),
    };
    sample.tool_required_stream_admin_metrics = Some(NormalizedAdminMetricsCapture {
        before: Some(json!({
            "first_tool_delta_ms": {"count": 0, "min": 0.0, "max": 0.0, "avg": 0.0},
            "validated_tool_call_ms": {"count": 0, "min": 0.0, "max": 0.0, "avg": 0.0},
            "backend_metrics": {
                "mlx": {
                    "stream_first_upstream_byte_ms": {"count": 0, "min": 0.0, "max": 0.0, "avg": 0.0},
                    "stream_first_parsed_chunk_ms": {"count": 0, "min": 0.0, "max": 0.0, "avg": 0.0},
                    "stream_first_tool_delta_ms": {"count": 0, "min": 0.0, "max": 0.0, "avg": 0.0}
                }
            }
        })),
        after: Some(json!({
            "first_tool_delta_ms": {"count": 1, "min": 29.0, "max": 29.0, "avg": 29.0},
            "validated_tool_call_ms": {"count": 1, "min": 68.0, "max": 68.0, "avg": 68.0},
            "backend_metrics": {
                "mlx": {
                    "stream_first_upstream_byte_ms": {"count": 1, "min": 8.0, "max": 8.0, "avg": 8.0},
                    "stream_first_parsed_chunk_ms": {"count": 1, "min": 12.0, "max": 12.0, "avg": 12.0},
                    "stream_first_tool_delta_ms": {"count": 1, "min": 30.0, "max": 30.0, "avg": 30.0}
                }
            }
        })),
        error: None,
    });
    lane_report.samples.push(sample);

    let report = tool_required_stream_timing_report(&[lane_report]);

    assert_eq!(report.attribution.status, "reported");
    let row = report.attribution.rows.first().expect("attribution row");
    assert_eq!(row.admin_metrics_scope, "per_sample");
    assert_eq!(row.client.first_byte_ms, Some(12));
    assert_eq!(row.client.first_sse_data_ms, Some(13));
    assert_eq!(row.client.first_tool_delta_ms, Some(30));
    assert_eq!(row.client.tool_finish_ms, Some(70));
    let admin = row.admin_metrics.as_ref().expect("admin attribution");
    assert_eq!(
        admin.mlx_stream_first_tool_delta_ms.window_avg_ms,
        Some(30.0)
    );
    assert_eq!(
        row.first_tool_delta_gap_ms.mlx_stream_to_client_ms,
        Some(0.0)
    );
    assert_eq!(row.decision, "upstream_aligned_with_client");
}

#[test]
fn qwen_mlx_tool_normalized_tool_stream_attribution_flags_kir_buffering_gap() {
    let lane_config =
        lane("name=proxy,endpoint=http://127.0.0.1:3000,model=qwen,kind=kir_ai_proxy");
    let mut lane_report = NormalizedLaneReport::planned(&lane_config, 0, 0, None);
    let mut sample = passed_sample(
        NormalizedCaseKind::ToolRequiredStream,
        CachePhase::Cold,
        RunMode::Sequential,
        0,
        None,
        800,
        16,
    );
    sample.stream_timing = StreamTimingReport {
        first_byte_latency_ms: Some(100),
        first_sse_data_latency_ms: Some(120),
        first_content_delta_latency_ms: None,
        first_tool_delta_latency_ms: Some(700),
        tool_finish_latency_ms: Some(760),
        first_semantic_delta_latency_ms: Some(700),
    };
    sample.tool_required_stream_admin_metrics = Some(NormalizedAdminMetricsCapture {
        before: Some(json!({
            "backend_metrics": {
                "mlx": {
                    "stream_first_tool_delta_ms": {"count": 3, "min": 10.0, "max": 10.0, "avg": 10.0}
                }
            }
        })),
        after: Some(json!({
            "backend_metrics": {
                "mlx": {
                    "stream_first_tool_delta_ms": {"count": 4, "min": 10.0, "max": 10.0, "avg": 10.0}
                }
            }
        })),
        error: None,
    });
    lane_report.samples.push(sample);

    let report = tool_required_stream_timing_report(&[lane_report]);
    let row = report.attribution.rows.first().expect("attribution row");

    assert_eq!(
        row.first_tool_delta_gap_ms.mlx_stream_to_client_ms,
        Some(690.0)
    );
    assert_eq!(row.decision, "kir_buffering_or_validation_gap");
}

#[test]
fn qwen_mlx_tool_normalized_tool_stream_timing_report_keeps_admin_errors_nonfatal() {
    let lane_config = lane("name=direct,endpoint=http://127.0.0.1:8080/v1,model=qwen");
    let mut lane_report = NormalizedLaneReport::planned(&lane_config, 0, 0, None);
    lane_report.admin_metrics.error = Some("before admin metrics HTTP 401".to_owned());

    let report = tool_required_stream_timing_report(&[lane_report]);

    assert_eq!(report.status, "admin_metrics_unavailable");
    assert!(report.lanes[0].admin_metrics.is_none());
    assert_eq!(
        report.lanes[0].admin_metrics_error.as_deref(),
        Some("before admin metrics HTTP 401")
    );
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
            NormalizedCaseKind::ToolRequiredStream.probe_id(),
            None,
        ),
        Ok(())
    );
}

#[test]
fn qwen_mlx_tool_normalized_prefill_stream_validation_checks_markers_and_tool_arguments() {
    let chat_marker = "KIR_QWEN_MLX_PREFILL_135K_CHAT_STREAM_QUARTZ_2741";
    let chat = StreamAssembly {
        content: format!("The recalled marker is {chat_marker}."),
        finish_reason: Some("stop".to_owned()),
        ..StreamAssembly::default()
    };
    assert_eq!(
        validate_streaming_probe(
            NormalizedCaseKind::ChatStream,
            &chat,
            NormalizedCaseKind::ChatStream.probe_id(),
            Some(chat_marker),
        ),
        Ok(())
    );

    let missing_marker = StreamAssembly {
        content: "no marker here".to_owned(),
        finish_reason: Some("stop".to_owned()),
        ..StreamAssembly::default()
    };
    let missing_marker_err = validate_streaming_probe(
        NormalizedCaseKind::ChatStream,
        &missing_marker,
        NormalizedCaseKind::ChatStream.probe_id(),
        Some(chat_marker),
    )
    .expect_err("chat stream must contain marker");
    assert!(
        missing_marker_err.contains("marker"),
        "error should mention marker: {missing_marker_err}"
    );

    let recall_marker = "KIR_LONG_CONTEXT_135K_CONTEXT_RECALL_STREAM_135K_QUARTZ_2741";
    let recall = StreamAssembly {
        tool_name: Some("report_long_context_recall".to_owned()),
        tool_arguments: json!({
            "marker": recall_marker,
            "profile": "qwen-prefill-sweep-135k",
            "case": "context_recall_stream_135k"
        })
        .to_string(),
        finish_reason: Some("tool_calls".to_owned()),
        ..StreamAssembly::default()
    };
    assert_eq!(
        validate_streaming_probe(
            NormalizedCaseKind::ContextRecallStream135k,
            &recall,
            NormalizedCaseKind::ContextRecallStream135k.probe_id(),
            Some(recall_marker),
        ),
        Ok(())
    );

    let bad_finish = StreamAssembly {
        finish_reason: Some("stop".to_owned()),
        ..recall.clone()
    };
    let bad_finish_err = validate_streaming_probe(
        NormalizedCaseKind::ContextRecallStream135k,
        &bad_finish,
        NormalizedCaseKind::ContextRecallStream135k.probe_id(),
        Some(recall_marker),
    )
    .expect_err("recall tool stream must finish with tool_calls");
    assert!(
        bad_finish_err.contains("tool_calls"),
        "error should mention tool_calls: {bad_finish_err}"
    );

    let malformed_args = StreamAssembly {
        tool_arguments: "{".to_owned(),
        ..recall
    };
    let malformed_args_err = validate_streaming_probe(
        NormalizedCaseKind::ContextRecallStream135k,
        &malformed_args,
        NormalizedCaseKind::ContextRecallStream135k.probe_id(),
        Some(recall_marker),
    )
    .expect_err("recall tool stream must send JSON arguments");
    assert!(
        malformed_args_err.contains("JSON"),
        "error should mention JSON: {malformed_args_err}"
    );

    let warm_prefix = StreamAssembly {
        tool_name: Some("record_qwen_tool_probe".to_owned()),
        tool_arguments:
            "{\"probe_id\":\"KIR_QWEN_MLX_TOOL_NORMALIZED_WARM_PREFIX_REPEATED_TURN_STREAM\",\"case\":\"warm_prefix_repeated_turn_stream\"}"
                .to_owned(),
        finish_reason: Some("tool_calls".to_owned()),
        ..StreamAssembly::default()
    };
    assert_eq!(
        validate_streaming_probe(
            NormalizedCaseKind::WarmPrefixRepeatedTurnStream,
            &warm_prefix,
            NormalizedCaseKind::WarmPrefixRepeatedTurnStream.probe_id(),
            None,
        ),
        Ok(())
    );
}

fn required_tool_ttft_sample(
    probe: NormalizedProbePlan,
    first_byte_ms: u128,
    first_sse_ms: u128,
    first_tool_delta_ms: u128,
    tool_finish_ms: u128,
) -> NormalizedSampleReport {
    let mut sample = NormalizedSampleReport::base(
        probe,
        CachePhase::Cold,
        RunMode::Sequential,
        0,
        None,
        false,
        128,
    );
    sample.status = "passed".to_owned();
    sample.classification = "passed".to_owned();
    sample.latency_ms = Some(tool_finish_ms);
    sample.stream_timing = StreamTimingReport {
        first_byte_latency_ms: Some(first_byte_ms),
        first_sse_data_latency_ms: Some(first_sse_ms),
        first_content_delta_latency_ms: None,
        first_tool_delta_latency_ms: Some(first_tool_delta_ms),
        tool_finish_latency_ms: Some(tool_finish_ms),
        first_semantic_delta_latency_ms: Some(first_tool_delta_ms),
    };
    sample.finish_reason = Some("tool_calls".to_owned());
    sample
}

fn prefill_sweep_sample(
    case: NormalizedCaseKind,
    phase: CachePhase,
    run_mode: RunMode,
    first_semantic_ms: u128,
) -> NormalizedSampleReport {
    let mut sample = NormalizedSampleReport::base(
        NormalizedProbePlan::new(case, SchemaVariant::None, ToolChoiceVariant::None),
        phase,
        run_mode,
        0,
        None,
        false,
        135_000,
    );
    sample.status = "passed".to_owned();
    sample.classification = "passed".to_owned();
    sample.latency_ms = Some(first_semantic_ms + 40);
    sample.stream_timing = StreamTimingReport {
        first_byte_latency_ms: Some(first_semantic_ms - 20),
        first_sse_data_latency_ms: Some(first_semantic_ms - 10),
        first_content_delta_latency_ms: Some(first_semantic_ms),
        first_tool_delta_latency_ms: None,
        tool_finish_latency_ms: None,
        first_semantic_delta_latency_ms: Some(first_semantic_ms),
    };
    sample.prompt_tokens = Some(135_000);
    sample.completion_tokens = Some(8);
    sample.total_tokens = Some(135_008);
    sample.cached_tokens_status = "present";
    sample.cached_tokens = Some(120_000);
    sample
}

fn latest_plain_stream_sample(
    ttfi_ms: u128,
    latency_ms: u128,
    tokens_per_second: f64,
) -> NormalizedSampleReport {
    let mut sample = NormalizedSampleReport::base(
        NormalizedProbePlan::new(
            NormalizedCaseKind::ChatStream,
            SchemaVariant::None,
            ToolChoiceVariant::None,
        ),
        CachePhase::Cold,
        RunMode::Sequential,
        0,
        None,
        false,
        512,
    );
    sample.status = "passed".to_owned();
    sample.classification = "passed".to_owned();
    sample.latency_ms = Some(latency_ms);
    sample.stream_timing = StreamTimingReport {
        first_byte_latency_ms: ttfi_ms.checked_sub(20),
        first_sse_data_latency_ms: ttfi_ms.checked_sub(10),
        first_content_delta_latency_ms: Some(ttfi_ms),
        first_tool_delta_latency_ms: None,
        tool_finish_latency_ms: None,
        first_semantic_delta_latency_ms: Some(ttfi_ms),
    };
    sample.tokens_per_second = Some(tokens_per_second);
    sample.prompt_tokens = Some(512);
    sample.completion_tokens = Some(192);
    sample.total_tokens = Some(704);
    sample.cached_tokens_status = "missing";
    sample
}

fn latest_tool_stream_sample(
    first_tool_delta_ms: u128,
    latency_ms: u128,
    tokens_per_second: f64,
) -> NormalizedSampleReport {
    let mut sample = NormalizedSampleReport::base(
        NormalizedProbePlan::new(
            NormalizedCaseKind::ToolRequiredStream,
            SchemaVariant::CanonicalCurrent,
            ToolChoiceVariant::Required,
        ),
        CachePhase::Cold,
        RunMode::Sequential,
        0,
        None,
        false,
        512,
    );
    sample.status = "passed".to_owned();
    sample.classification = "passed".to_owned();
    sample.latency_ms = Some(latency_ms);
    sample.stream_timing = StreamTimingReport {
        first_byte_latency_ms: first_tool_delta_ms.checked_sub(80),
        first_sse_data_latency_ms: first_tool_delta_ms.checked_sub(70),
        first_content_delta_latency_ms: None,
        first_tool_delta_latency_ms: Some(first_tool_delta_ms),
        tool_finish_latency_ms: Some(latency_ms),
        first_semantic_delta_latency_ms: Some(first_tool_delta_ms),
    };
    sample.tokens_per_second = Some(tokens_per_second);
    sample.prompt_tokens = Some(512);
    sample.completion_tokens = Some(64);
    sample.total_tokens = Some(576);
    sample.cached_tokens_status = "missing";
    sample
}

fn latest_cache_sample(
    phase: CachePhase,
    latency_ms: u128,
    ttfi_ms: u128,
    tokens_per_second: f64,
    cached_tokens: Option<u64>,
) -> NormalizedSampleReport {
    let probe = NormalizedProbePlan::new(
        NormalizedCaseKind::WarmPrefixRepeatedTurnStream,
        SchemaVariant::CanonicalCurrent,
        ToolChoiceVariant::Required,
    );
    let mut sample =
        NormalizedSampleReport::base(probe, phase, RunMode::Sequential, 0, None, false, 1_000);
    sample.status = "passed".to_owned();
    sample.classification = "passed".to_owned();
    sample.latency_ms = Some(latency_ms);
    sample.stream_timing = StreamTimingReport {
        first_byte_latency_ms: ttfi_ms.checked_sub(20),
        first_sse_data_latency_ms: ttfi_ms.checked_sub(10),
        first_content_delta_latency_ms: None,
        first_tool_delta_latency_ms: Some(ttfi_ms),
        tool_finish_latency_ms: Some(latency_ms),
        first_semantic_delta_latency_ms: Some(ttfi_ms),
    };
    sample.tokens_per_second = Some(tokens_per_second);
    sample.prompt_tokens = Some(1_000);
    sample.completion_tokens = Some(10);
    sample.total_tokens = Some(1_010);
    sample.cached_tokens_status = if cached_tokens.is_some() {
        "present"
    } else {
        "missing"
    };
    sample.cached_tokens = cached_tokens;
    sample
}

fn stable_prefix_sample(
    probe: NormalizedProbePlan,
    phase: CachePhase,
    first_semantic_ms: u128,
    cached_tokens: Option<u64>,
    request_id: Option<&str>,
) -> NormalizedSampleReport {
    let mut sample =
        NormalizedSampleReport::base(probe, phase, RunMode::Sequential, 0, None, false, 1000);
    sample.status = "passed".to_owned();
    sample.classification = "passed".to_owned();
    sample.latency_ms = Some(first_semantic_ms + 15);
    sample.stream_timing = StreamTimingReport {
        first_byte_latency_ms: Some(first_semantic_ms - 10),
        first_sse_data_latency_ms: Some(first_semantic_ms - 5),
        first_content_delta_latency_ms: None,
        first_tool_delta_latency_ms: Some(first_semantic_ms),
        tool_finish_latency_ms: Some(first_semantic_ms + 10),
        first_semantic_delta_latency_ms: Some(first_semantic_ms),
    };
    sample.prompt_tokens = Some(1000);
    sample.completion_tokens = Some(10);
    sample.total_tokens = Some(1010);
    sample.cached_tokens_status = if cached_tokens.is_some() {
        "present"
    } else {
        "missing"
    };
    sample.cached_tokens = cached_tokens;
    sample.request_id = request_id.map(str::to_owned);
    sample
}

fn ab_tool_stream_sample(
    first_tool_delta_ms: u128,
    tool_finish_ms: u128,
) -> NormalizedSampleReport {
    let probe = NormalizedProbePlan::new(
        NormalizedCaseKind::ToolRequiredStream,
        SchemaVariant::CanonicalCurrent,
        ToolChoiceVariant::Required,
    );
    let mut sample = NormalizedSampleReport::base(
        probe,
        CachePhase::WarmSamePrompt,
        RunMode::Sequential,
        0,
        None,
        true,
        128,
    );
    sample.status = "passed".to_owned();
    sample.classification = "passed".to_owned();
    sample.latency_ms = Some(tool_finish_ms + 5);
    sample.stream_timing = StreamTimingReport {
        first_byte_latency_ms: Some(first_tool_delta_ms.saturating_sub(20)),
        first_sse_data_latency_ms: Some(first_tool_delta_ms.saturating_sub(10)),
        first_content_delta_latency_ms: None,
        first_tool_delta_latency_ms: Some(first_tool_delta_ms),
        tool_finish_latency_ms: Some(tool_finish_ms),
        first_semantic_delta_latency_ms: Some(first_tool_delta_ms),
    };
    sample.prompt_tokens = Some(1835);
    sample.completion_tokens = Some(64);
    sample.total_tokens = Some(1899);
    sample.cached_tokens_status = "present";
    sample.cached_tokens = Some(1834);
    sample.finish_reason = Some("tool_calls".to_owned());
    sample
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
