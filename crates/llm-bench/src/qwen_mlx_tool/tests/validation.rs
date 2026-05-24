use super::*;

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
