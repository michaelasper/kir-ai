use super::*;

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
