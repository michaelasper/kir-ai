use super::*;

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
