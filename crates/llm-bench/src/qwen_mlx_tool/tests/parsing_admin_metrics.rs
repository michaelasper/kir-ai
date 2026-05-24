use super::*;

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
