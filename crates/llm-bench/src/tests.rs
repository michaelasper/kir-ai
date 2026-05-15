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
        "backend_metrics": {
            "native_text_prefix_cache": {
                "qwen": {
                    "hits": 3,
                    "misses": 1,
                    "stores": 2,
                    "evictions": 1,
                    "rejected": 0,
                    "reused_tokens": 42,
                    "resident_bytes": 1024,
                    "resident_entries": 2
                }
            },
            "native_text_metal": {
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

#[tokio::test]
async fn qwen_long_context_dry_run_report_includes_analysis_schema() {
    let temp = tempfile::tempdir().expect("tempdir");
    let output = temp.path().join("bench-report.json");
    let args = vec![
        "--dry-run".to_owned(),
        "--output".to_owned(),
        output.display().to_string(),
        "--max-tokens".to_owned(),
        "64".to_owned(),
        "--scheduler-concurrency".to_owned(),
        "2".to_owned(),
        "--scheduler-queue-limit".to_owned(),
        "8".to_owned(),
        "--scheduler-prefill-threshold-chars".to_owned(),
        "8192".to_owned(),
        "--scheduler-prefill-burst".to_owned(),
        "3".to_owned(),
    ];

    run_qwen_long_context_bench(&args)
        .await
        .expect("dry-run benchmark report");

    let report: Value =
        serde_json::from_slice(&std::fs::read(&output).expect("dry-run report file is written"))
            .expect("dry-run report JSON");
    assert_eq!(report["run_controls"]["warmup_count"], 0);
    assert_eq!(report["run_controls"]["repetitions"], 1);
    assert_eq!(report["run_controls"]["max_tokens"], 64);
    assert_eq!(report["scheduler"]["concurrency_limit"], 2);
    assert_eq!(report["scheduler"]["queue_limit"], 8);
    assert_eq!(report["scheduler"]["prefill_threshold_chars"], 8192);
    assert_eq!(report["scheduler"]["prefill_burst"], 3);

    let case = &report["profiles"][0]["cases"][0];
    assert_eq!(case["prompt_identity"]["profile"], "qwen-135k-promotion");
    assert_eq!(case["prompt_identity"]["context_tokens"], 135_000);
    assert!(
        case["prompt_identity"]["prompt_hash"]
            .as_str()
            .expect("prompt hash")
            .starts_with("sha256:")
    );
    assert_eq!(
        case["prompt_identity"]["prompt_hash_source"],
        "planned_identity"
    );
    assert_eq!(case["prefill"]["planned_prompt_tokens"], Value::Null);
    assert_eq!(case["decode"]["max_tokens"], 64);
    assert_eq!(case["cache"]["lookup_result"], Value::Null);
    assert_eq!(case["summary"]["sample_count"], 0);
}

#[test]
fn case_run_populates_structured_prefill_decode_cache_and_summary() {
    let mut case = case_report(
        BenchProfileKind::Promotion135k,
        BenchCaseKind::StreamedRequiredToolRecall,
        DEFAULT_MAX_TOKENS,
    );
    apply_case_run(
        &mut case,
        CaseRun {
            status: "passed",
            classification: "passed".to_owned(),
            planned_prompt_tokens: 100,
            latency_ms: Some(250),
            stream_timing: StreamTimingReport {
                first_byte_latency_ms: Some(10),
                first_sse_data_latency_ms: Some(20),
                first_content_delta_latency_ms: None,
                first_tool_delta_latency_ms: Some(75),
                tool_finish_latency_ms: Some(125),
                first_semantic_delta_latency_ms: Some(75),
            },
            tokens_per_second: Some(12.5),
            prompt_tokens: Some(100),
            completion_tokens: Some(25),
            total_tokens: Some(125),
            cached_tokens_status: Some("present"),
            cached_tokens: Some(40),
            prompt_hash: Some("sha256:actual-prompt-body".to_owned()),
            http_status: Some(200),
            finish_reason: Some("tool_calls".to_owned()),
            error: None,
        },
    );

    let value = serde_json::to_value(&case).expect("serialize case");
    assert_eq!(
        value["prompt_identity"]["prompt_hash"],
        "sha256:actual-prompt-body"
    );
    assert_eq!(
        value["prompt_identity"]["prompt_hash_source"],
        "prompt_body"
    );
    assert_eq!(value["prefill"]["planned_prompt_tokens"], 100);
    assert_eq!(value["prefill"]["prompt_tokens"], 100);
    assert_eq!(value["prefill"]["cached_tokens"], 40);
    assert_eq!(value["prefill"]["uncached_tokens"], 60);
    assert_eq!(value["prefill"]["time_to_first_token_ms"], 75);
    assert_eq!(value["decode"]["completion_tokens"], 25);
    assert_eq!(value["decode"]["total_latency_ms"], 250);
    assert_eq!(value["decode"]["time_to_first_token_ms"], 75);
    assert_eq!(value["decode"]["tokens_per_second"], 12.5);
    assert_eq!(value["cache"]["lookup_result"], "hit");
    assert_eq!(value["cache"]["reused_tokens"], 40);
    assert_eq!(value["summary"]["sample_count"], 1);
    assert_eq!(value["summary"]["latency_ms_p50"], 250);
    assert_eq!(value["summary"]["latency_ms_p95"], 250);
    assert_eq!(value["summary"]["tokens_per_second_p50"], 12.5);
    assert_eq!(value["summary"]["tokens_per_second_p95"], 12.5);
    assert_eq!(value["summary"]["ttft_ms_p50"], 75);
    assert_eq!(value["summary"]["ttft_ms_p95"], 75);
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

#[test]
fn sse_comments_and_done_do_not_record_semantic_timings() {
    let mut buffer = ": keepalive\n\ndata: [DONE]\n".to_owned();
    let mut assembly = StreamAssembly::default();
    let mut timings = StreamTimingTracker::default();

    consume_sse_buffer(
        &mut buffer,
        &mut assembly,
        &mut timings,
        Duration::from_millis(37),
    );

    let report = timings.to_report();
    assert_eq!(report.first_sse_data_latency_ms, None);
    assert_eq!(report.first_content_delta_latency_ms, None);
    assert_eq!(report.first_tool_delta_latency_ms, None);
    assert_eq!(report.first_semantic_delta_latency_ms, None);
}

#[test]
fn sse_comments_before_text_delta_record_content_semantics_at_text_frame() {
    let mut assembly = StreamAssembly::default();
    let mut timings = StreamTimingTracker::default();
    let mut comments = ": keepalive\n\n".to_owned();
    consume_sse_buffer(
        &mut comments,
        &mut assembly,
        &mut timings,
        Duration::from_millis(11),
    );

    let mut delta = format!("data: {}\n", streamed_content_delta("marker text"));
    consume_sse_buffer(
        &mut delta,
        &mut assembly,
        &mut timings,
        Duration::from_millis(29),
    );

    let report = timings.to_report();
    assert_eq!(assembly.content, "marker text");
    assert_eq!(report.first_sse_data_latency_ms, Some(29));
    assert_eq!(report.first_content_delta_latency_ms, Some(29));
    assert_eq!(report.first_tool_delta_latency_ms, None);
    assert_eq!(report.first_semantic_delta_latency_ms, Some(29));
}

#[test]
fn sse_comments_before_tool_delta_record_tool_semantics_at_tool_frame() {
    let mut assembly = StreamAssembly::default();
    let mut timings = StreamTimingTracker::default();
    let mut comments = ": keepalive\n\n".to_owned();
    consume_sse_buffer(
        &mut comments,
        &mut assembly,
        &mut timings,
        Duration::from_millis(13),
    );

    let mut delta = format!("data: {}\n", streamed_tool_delta("call_1", "report", "{\""));
    consume_sse_buffer(
        &mut delta,
        &mut assembly,
        &mut timings,
        Duration::from_millis(41),
    );

    let report = timings.to_report();
    assert_eq!(assembly.tool_name.as_deref(), Some("report"));
    assert_eq!(assembly.tool_arguments, "{\"");
    assert_eq!(report.first_sse_data_latency_ms, Some(41));
    assert_eq!(report.first_content_delta_latency_ms, None);
    assert_eq!(report.first_tool_delta_latency_ms, Some(41));
    assert_eq!(report.first_semantic_delta_latency_ms, Some(41));
}

#[test]
fn usage_only_sse_data_records_sse_latency_without_semantic_timing() {
    let mut buffer = format!(
        "data: {}\n",
        serde_json::json!({
            "choices": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12
            }
        })
    );
    let mut assembly = StreamAssembly::default();
    let mut timings = StreamTimingTracker::default();

    consume_sse_buffer(
        &mut buffer,
        &mut assembly,
        &mut timings,
        Duration::from_millis(53),
    );

    let report = timings.to_report();
    assert_eq!(report.first_sse_data_latency_ms, Some(53));
    assert_eq!(report.first_content_delta_latency_ms, None);
    assert_eq!(report.first_tool_delta_latency_ms, None);
    assert_eq!(report.first_semantic_delta_latency_ms, None);
    assert_eq!(assembly.usage.prompt_tokens, Some(10));
    assert_eq!(assembly.usage.completion_tokens, Some(2));
    assert_eq!(assembly.usage.total_tokens, Some(12));
    assert_eq!(assembly.usage.cached_tokens_status, Some("missing"));
}

#[test]
fn lane_comparison_serializes_explicit_stream_timing_fields_without_ttft() {
    let mut lane = passed_lane(
        "native",
        "michaelasper/qwen",
        "commit-a",
        "qwen3-bf16",
        "bf16",
    );
    let stream_case = lane.profiles[0]
        .cases
        .iter_mut()
        .find(|case| case.stream)
        .expect("streaming case");
    stream_case.stream_timing = StreamTimingReport {
        first_byte_latency_ms: Some(7),
        first_sse_data_latency_ms: Some(13),
        first_content_delta_latency_ms: None,
        first_tool_delta_latency_ms: Some(21),
        tool_finish_latency_ms: Some(34),
        first_semantic_delta_latency_ms: Some(21),
    };

    let comparison = compare_bench_lanes(&[lane]);
    let value = serde_json::to_value(&comparison).expect("serialize comparison");
    let stream_case = value
        .get("cases")
        .and_then(Value::as_array)
        .and_then(|cases| {
            cases.iter().find(|case| {
                case.get("case").and_then(Value::as_str) == Some("streamed-required-tool-recall")
            })
        })
        .expect("streaming case comparison");
    let lane_metrics = stream_case
        .pointer("/lanes/0")
        .expect("streaming lane metrics")
        .as_object()
        .expect("metrics object");

    assert!(!lane_metrics.contains_key("ttft_ms"));
    assert_eq!(
        lane_metrics
            .get("first_byte_latency_ms")
            .and_then(Value::as_u64),
        Some(7)
    );
    assert_eq!(
        lane_metrics
            .get("first_sse_data_latency_ms")
            .and_then(Value::as_u64),
        Some(13)
    );
    assert_eq!(
        lane_metrics
            .get("first_tool_delta_latency_ms")
            .and_then(Value::as_u64),
        Some(21)
    );
    assert_eq!(
        lane_metrics
            .get("first_semantic_delta_latency_ms")
            .and_then(Value::as_u64),
        Some(21)
    );
    assert!(!lane_metrics.contains_key("first_content_delta_latency_ms"));
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

fn streamed_content_delta(content: &str) -> Value {
    serde_json::json!({
        "choices": [{
            "delta": {
                "content": content
            }
        }]
    })
}

fn streamed_tool_delta(id: &str, name: &str, arguments: &str) -> Value {
    serde_json::json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments
                    }
                }]
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
