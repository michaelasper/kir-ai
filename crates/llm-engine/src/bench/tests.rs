use super::*;

const MARKER: &str = "KIR_LONG_CONTEXT_135K_JSON_OBJECT_RECALL_QUARTZ_2741";

#[test]
fn lane_artifact_mismatch_fails_gate_status() {
    let lanes = vec![
        passed_lane(
            "native",
            "michaelasper/qwen",
            "commit-a",
            "qwen3-bf16",
            "bf16",
        ),
        passed_lane("mlx", "michaelasper/qwen", "commit-b", "qwen3-bf16", "bf16"),
    ];
    let comparison = compare_bench_lanes(&lanes);

    assert_eq!(comparison.status, "artifact_identity_mismatch");
    assert_eq!(
        bench_gate_failure_classification(false, &comparison),
        Some("lane_artifact_identity_mismatch")
    );
    assert_eq!(bench_gate_status(false, &comparison), "failed");
}

#[test]
fn all_profile_selection_includes_256k_characterization() {
    let profiles = selected_profiles("all").expect("all profiles");

    assert_eq!(
        profiles
            .iter()
            .map(|profile| profile.name())
            .collect::<Vec<_>>(),
        [
            "qwen-135k-promotion",
            "qwen-200k-characterization",
            "qwen-256k-characterization"
        ]
    );
    assert_eq!(
        BenchProfileKind::Characterization256k.target_tokens(),
        256_000
    );
    assert!(!BenchProfileKind::Characterization256k.release_blocking());
}

#[test]
fn cache_metrics_summary_extracts_admin_cache_counters() {
    let admin = serde_json::json!({
        "native_qwen_prefix_cache": {
            "hits": 3,
            "misses": 1,
            "stores": 2,
            "evictions": 1,
            "rejected": 0,
            "reused_tokens": 42,
            "resident_bytes": 1024,
            "resident_entries": 2
        },
        "native_qwen_metal": {
            "bf16_matrix_cache": {
                "hits": 7,
                "misses": 3,
                "uploads": 3,
                "bytes_uploaded": 2048,
                "evictions": 1,
                "bytes_evicted": 512,
                "resident_bytes": 1536,
                "resident_buffers": 4,
                "budget_bytes": 4096
            },
            "kv_cache": {
                "allocations": 2,
                "syncs": 4,
                "evictions": 1,
                "bytes_uploaded": 4096,
                "bytes_evicted": 1024,
                "resident_bytes": 3072,
                "resident_buffers": 2
            },
            "linear_attention_cache": {
                "allocations": 1,
                "syncs": 3,
                "evictions": 0,
                "bytes_uploaded": 2048,
                "bytes_evicted": 0,
                "resident_bytes": 2048,
                "resident_buffers": 1
            }
        }
    });

    let summary = cache_metrics_from_admin(&admin).expect("cache summary");

    assert_eq!(summary.prefix_cache.hit_rate, Some(0.75));
    assert!((summary.weight_cache.hit_rate.expect("weight hit rate") - 0.7).abs() < f64::EPSILON);
    assert_eq!(summary.kv_cache.resident_bytes, 3072);
    assert_eq!(summary.linear_attention_cache.syncs, 3);
    assert_eq!(summary.readiness.status, "observable");
    assert!(summary.readiness.missing_signals.is_empty());
}

#[test]
fn json_object_recall_rejects_marker_only_contract() {
    let value = buffered_content_response(serde_json::json!({"marker": MARKER}).to_string());

    let err = validate_buffered_case(
        BenchProfileKind::Promotion135k,
        BenchCaseKind::JsonObjectRecall,
        MARKER,
        &value,
    )
    .expect_err("marker-only JSON must fail");

    assert!(err.contains("profile"), "error: {err}");
}

#[test]
fn json_object_recall_rejects_wrong_profile_or_case() {
    let value = buffered_content_response(
        serde_json::json!({
            "marker": MARKER,
            "profile": "qwen-200k-characterization",
            "case": "json-object-recall"
        })
        .to_string(),
    );

    let err = validate_buffered_case(
        BenchProfileKind::Promotion135k,
        BenchCaseKind::JsonObjectRecall,
        MARKER,
        &value,
    )
    .expect_err("wrong profile must fail");

    assert!(err.contains("profile"), "error: {err}");
}

#[test]
fn required_tool_recall_requires_tool_finish_reason_and_full_arguments() {
    let marker_only = buffered_tool_response(
        "tool_calls",
        serde_json::json!({"marker": MARKER}).to_string(),
    );
    let marker_only_err = validate_buffered_case(
        BenchProfileKind::Promotion135k,
        BenchCaseKind::RequiredToolRecall,
        MARKER,
        &marker_only,
    )
    .expect_err("marker-only tool arguments must fail");
    assert!(
        marker_only_err.contains("profile"),
        "error: {marker_only_err}"
    );

    let wrong_finish = buffered_tool_response(
        "stop",
        serde_json::json!({
            "marker": MARKER,
            "profile": "qwen-135k-promotion",
            "case": "required-tool-recall"
        })
        .to_string(),
    );
    let finish_err = validate_buffered_case(
        BenchProfileKind::Promotion135k,
        BenchCaseKind::RequiredToolRecall,
        MARKER,
        &wrong_finish,
    )
    .expect_err("wrong finish_reason must fail");
    assert!(finish_err.contains("finish_reason"), "error: {finish_err}");
}

#[test]
fn streamed_required_tool_recall_requires_tool_finish_reason_and_full_arguments() {
    let marker_only = StreamAssembly {
        tool_name: Some("report_long_context_recall".to_owned()),
        tool_arguments: serde_json::json!({"marker": MARKER}).to_string(),
        finish_reason: Some("tool_calls".to_owned()),
        ..StreamAssembly::default()
    };
    let marker_only_err = validate_streaming_case(
        BenchProfileKind::Promotion135k,
        BenchCaseKind::StreamedRequiredToolRecall,
        MARKER,
        &marker_only,
    )
    .expect_err("marker-only streamed tool arguments must fail");
    assert!(
        marker_only_err.contains("profile"),
        "error: {marker_only_err}"
    );

    let wrong_finish = StreamAssembly {
        tool_name: Some("report_long_context_recall".to_owned()),
        tool_arguments: serde_json::json!({
            "marker": MARKER,
            "profile": "qwen-135k-promotion",
            "case": "streamed-required-tool-recall"
        })
        .to_string(),
        finish_reason: Some("stop".to_owned()),
        ..StreamAssembly::default()
    };
    let finish_err = validate_streaming_case(
        BenchProfileKind::Promotion135k,
        BenchCaseKind::StreamedRequiredToolRecall,
        MARKER,
        &wrong_finish,
    )
    .expect_err("wrong streamed finish_reason must fail");
    assert!(finish_err.contains("finish_reason"), "error: {finish_err}");
}

fn passed_lane(
    name: &str,
    repo_id: &str,
    resolved_commit: &str,
    profile: &str,
    quantization: &str,
) -> BenchLaneReport {
    let mut report = profile_report(BenchProfileKind::Promotion135k);
    report.status = "passed".to_owned();
    for case in &mut report.cases {
        case.status = "passed".to_owned();
        case.classification = "passed".to_owned();
        case.latency_ms = Some(100);
    }
    BenchLaneReport {
        name: name.to_owned(),
        status: "passed".to_owned(),
        model: ModelIdentityReport {
            id: name.to_owned(),
            endpoint: None,
            snapshot_path: None,
            repo_id: Some(repo_id.to_owned()),
            requested_revision: Some(resolved_commit.to_owned()),
            resolved_commit: Some(resolved_commit.to_owned()),
            profile: Some(profile.to_owned()),
            family: Some("qwen".to_owned()),
            loader: Some("native-metal".to_owned()),
            quantization: Some(quantization.to_owned()),
            manifest_digest: None,
        },
        profiles: vec![report],
        cache_metrics: None,
        admin_metrics: None,
        admin_metrics_error: None,
    }
}

fn buffered_content_response(content: String) -> Value {
    serde_json::json!({
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": content
            }
        }]
    })
}

fn buffered_tool_response(finish_reason: &str, arguments: String) -> Value {
    serde_json::json!({
        "choices": [{
            "finish_reason": finish_reason,
            "message": {
                "role": "assistant",
                "tool_calls": [{
                    "type": "function",
                    "function": {
                        "name": "report_long_context_recall",
                        "arguments": arguments
                    }
                }]
            }
        }]
    })
}
