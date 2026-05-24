use super::*;

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
