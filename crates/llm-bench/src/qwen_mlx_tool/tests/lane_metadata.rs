use super::*;

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
