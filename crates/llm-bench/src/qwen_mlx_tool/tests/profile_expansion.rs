use super::*;

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
