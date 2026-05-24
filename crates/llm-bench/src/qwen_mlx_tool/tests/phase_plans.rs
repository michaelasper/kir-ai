use super::*;

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
