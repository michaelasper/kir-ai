use super::*;

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedPlanSummaryReport {
    pub(in crate::qwen_mlx_tool) probe_count: usize,
    pub(in crate::qwen_mlx_tool) lane_count: usize,
    pub(in crate::qwen_mlx_tool) warmups_per_warm_phase: usize,
    pub(in crate::qwen_mlx_tool) samples_per_phase: usize,
    pub(in crate::qwen_mlx_tool) concurrent_requests: usize,
    pub(in crate::qwen_mlx_tool) concurrent_samples: usize,
    pub(in crate::qwen_mlx_tool) effective_concurrent_samples: usize,
    pub(in crate::qwen_mlx_tool) cache_phases: Vec<&'static str>,
    pub(in crate::qwen_mlx_tool) probes: Vec<NormalizedPlanProbeReport>,
    pub(in crate::qwen_mlx_tool) lanes: Vec<String>,
    pub(in crate::qwen_mlx_tool) warmup_requests: usize,
    pub(in crate::qwen_mlx_tool) measured_requests: usize,
    pub(in crate::qwen_mlx_tool) sequential_measured_requests: usize,
    pub(in crate::qwen_mlx_tool) concurrent_measured_requests: usize,
    pub(in crate::qwen_mlx_tool) total_http_requests: usize,
    pub(in crate::qwen_mlx_tool) planned_prompt_token_budget: usize,
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedPlanProbeReport {
    pub(in crate::qwen_mlx_tool) case: &'static str,
    pub(in crate::qwen_mlx_tool) schema_variant: &'static str,
    pub(in crate::qwen_mlx_tool) tool_choice_variant: &'static str,
    pub(in crate::qwen_mlx_tool) max_tokens: u32,
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedBenchReport {
    pub(in crate::qwen_mlx_tool) benchmark: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::qwen_mlx_tool) sweep_profile: Option<&'static str>,
    pub(in crate::qwen_mlx_tool) status: String,
    pub(in crate::qwen_mlx_tool) generated_at_unix_ms: u128,
    pub(in crate::qwen_mlx_tool) trace_output_path: Option<String>,
    pub(in crate::qwen_mlx_tool) warmups: usize,
    pub(in crate::qwen_mlx_tool) samples: usize,
    pub(in crate::qwen_mlx_tool) context_tokens: usize,
    pub(in crate::qwen_mlx_tool) concurrent_requests: usize,
    pub(in crate::qwen_mlx_tool) concurrent_samples: usize,
    pub(in crate::qwen_mlx_tool) effective_concurrent_samples: usize,
    pub(in crate::qwen_mlx_tool) timeout_ms: u64,
    pub(in crate::qwen_mlx_tool) connect_timeout_ms: u64,
    pub(in crate::qwen_mlx_tool) probe_suite: &'static str,
    pub(in crate::qwen_mlx_tool) repo_revision: RepoRevisionReport,
    pub(in crate::qwen_mlx_tool) cases: Vec<&'static str>,
    pub(in crate::qwen_mlx_tool) schema_variants: Vec<&'static str>,
    pub(in crate::qwen_mlx_tool) tool_choice_variants: Vec<&'static str>,
    pub(in crate::qwen_mlx_tool) cache_phases: Vec<&'static str>,
    pub(in crate::qwen_mlx_tool) plan_summary: NormalizedPlanSummaryReport,
    pub(in crate::qwen_mlx_tool) summary: Vec<NormalizedAggregateSummaryRow>,
    pub(in crate::qwen_mlx_tool) tool_required_stream: NormalizedToolRequiredStreamTimingReport,
    pub(in crate::qwen_mlx_tool) required_tool_ttft_matrix: NormalizedRequiredToolTtftMatrixReport,
    pub(in crate::qwen_mlx_tool) lanes: Vec<NormalizedLaneReport>,
    pub(in crate::qwen_mlx_tool) hardware: HardwareReport,
    pub(in crate::qwen_mlx_tool) comparison: NormalizedComparisonReport,
    pub(in crate::qwen_mlx_tool) agentic_gate: NormalizedAgenticGateReport,
    pub(in crate::qwen_mlx_tool) agentic_streaming_fast_path_ab:
        NormalizedAgenticStreamingFastPathAbReport,
    pub(in crate::qwen_mlx_tool) prefill_concurrency: NormalizedPrefillConcurrencyReport,
    pub(in crate::qwen_mlx_tool) prefill_sweep: NormalizedPrefillSweepReport,
    pub(in crate::qwen_mlx_tool) stable_prefix: NormalizedStablePrefixReport,
    pub(in crate::qwen_mlx_tool) latest_performance_comparison:
        NormalizedLatestPerformanceComparisonReport,
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct RepoRevisionReport {
    pub(in crate::qwen_mlx_tool) branch: Option<String>,
    pub(in crate::qwen_mlx_tool) commit_sha: Option<String>,
    pub(in crate::qwen_mlx_tool) dirty: bool,
}

impl RepoRevisionReport {
    pub(in crate::qwen_mlx_tool) fn detect() -> Self {
        if let Some(report) = Self::from_env() {
            return report;
        }
        if let Some(report) = Self::from_origin_file() {
            return report;
        }
        Self {
            branch: git_output(&["branch", "--show-current"]).filter(|branch| !branch.is_empty()),
            commit_sha: git_output(&["rev-parse", "HEAD"]),
            dirty: git_dirty(),
        }
    }

    pub(in crate::qwen_mlx_tool) fn from_env() -> Option<Self> {
        let branch = env_string(BENCH_REPO_BRANCH_ENV);
        let commit_sha = env_string(BENCH_REPO_COMMIT_ENV);
        let dirty = env_bool(BENCH_REPO_DIRTY_ENV);
        if branch.is_none() && commit_sha.is_none() && dirty.is_none() {
            return None;
        }
        Some(Self {
            branch,
            commit_sha,
            dirty: dirty.unwrap_or(false),
        })
    }

    pub(in crate::qwen_mlx_tool) fn from_origin_file() -> Option<Self> {
        let path = benchmark_repo_dir().join(BENCH_REPO_ORIGIN_FILE);
        let value = serde_json::from_slice::<Value>(&std::fs::read(path).ok()?).ok()?;
        let revision = value.get("repo_revision").unwrap_or(&value);
        let branch = origin_string(revision, &["branch", "git_branch", "ref"]);
        let commit_sha = origin_string(
            revision,
            &[
                "commit_sha",
                "commit",
                "git_commit",
                "revision",
                "source_commit",
            ],
        );
        let dirty = origin_bool(revision, &["dirty", "git_dirty"]);
        if branch.is_none() && commit_sha.is_none() && dirty.is_none() {
            return None;
        }
        Some(Self {
            branch,
            commit_sha,
            dirty: dirty.unwrap_or(false),
        })
    }
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedLaneReport {
    pub(in crate::qwen_mlx_tool) name: String,
    pub(in crate::qwen_mlx_tool) status: String,
    pub(in crate::qwen_mlx_tool) endpoint: String,
    pub(in crate::qwen_mlx_tool) kind: &'static str,
    pub(in crate::qwen_mlx_tool) experimental: bool,
    pub(in crate::qwen_mlx_tool) declared_model_id: String,
    pub(in crate::qwen_mlx_tool) effective_request_model_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::qwen_mlx_tool) launched_model_id: Option<String>,
    pub(in crate::qwen_mlx_tool) model_identity_source: &'static str,
    pub(in crate::qwen_mlx_tool) model_addressing: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::qwen_mlx_tool) tool_parser: Option<&'static str>,
    pub(in crate::qwen_mlx_tool) mlx_lm_settings: MlxLmSettings,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::qwen_mlx_tool) snapshot_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::qwen_mlx_tool) snapshot_identity: Option<ModelIdentityReport>,
    pub(in crate::qwen_mlx_tool) qwen_thinking_policy: Value,
    pub(in crate::qwen_mlx_tool) warmups: usize,
    pub(in crate::qwen_mlx_tool) sample_count: usize,
    pub(in crate::qwen_mlx_tool) planned_requests: Vec<NormalizedPlannedRequestReport>,
    pub(in crate::qwen_mlx_tool) samples: Vec<NormalizedSampleReport>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(in crate::qwen_mlx_tool) concurrent_samples: Vec<NormalizedSampleReport>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(in crate::qwen_mlx_tool) warmup_failures: Vec<NormalizedWarmupFailure>,
    #[serde(skip)]
    pub(in crate::qwen_mlx_tool) admin_metrics: NormalizedAdminMetricsCapture,
}

impl NormalizedLaneReport {
    #[cfg(test)]
    pub(in crate::qwen_mlx_tool) fn planned(
        lane: &NormalizedLaneConfig,
        warmups: usize,
        samples: usize,
        snapshot_identity: Option<ModelIdentityReport>,
    ) -> Self {
        let run_config = NormalizedRunConfig::new(
            warmups,
            samples,
            DEFAULT_CONTEXT_TOKENS,
            DEFAULT_CONCURRENT_REQUESTS,
            DEFAULT_CONCURRENT_SAMPLES,
        );
        Self::planned_with_requests(lane, warmups, samples, &run_config, snapshot_identity, &[])
    }

    pub(in crate::qwen_mlx_tool) fn planned_with_requests(
        lane: &NormalizedLaneConfig,
        warmups: usize,
        samples: usize,
        run_config: &NormalizedRunConfig,
        snapshot_identity: Option<ModelIdentityReport>,
        probes: &[NormalizedProbePlan],
    ) -> Self {
        Self {
            name: lane.name.clone(),
            status: "planned".to_owned(),
            endpoint: lane.endpoint.clone(),
            kind: lane.kind.as_str(),
            experimental: lane.experimental,
            declared_model_id: lane.declared_model_id.clone(),
            effective_request_model_id: lane.effective_request_model_id().to_owned(),
            launched_model_id: lane.launched_model_id.clone(),
            model_identity_source: lane.model_identity_source(),
            model_addressing: lane.model_addressing.as_str(),
            tool_parser: lane.tool_parser_report(),
            mlx_lm_settings: lane.mlx_lm_settings,
            snapshot_path: lane
                .snapshot_path
                .as_ref()
                .map(|path| path.display().to_string()),
            snapshot_identity,
            qwen_thinking_policy: lane.thinking_policy_report(),
            warmups,
            sample_count: samples,
            planned_requests: planned_requests_for(probes, run_config),
            samples: Vec::new(),
            concurrent_samples: Vec::new(),
            warmup_failures: Vec::new(),
            admin_metrics: NormalizedAdminMetricsCapture::default(),
        }
    }

    pub(in crate::qwen_mlx_tool) fn dry_run(
        lane: &NormalizedLaneConfig,
        run_config: &NormalizedRunConfig,
        snapshot_identity: Option<ModelIdentityReport>,
        probes: &[NormalizedProbePlan],
    ) -> Self {
        let mut report = Self::planned_with_requests(
            lane,
            run_config.warmups,
            run_config.samples,
            run_config,
            snapshot_identity,
            probes,
        );
        report.status = "dry_run".to_owned();
        for &probe in probes {
            for planned in phase_plan(
                &run_config.cache_phases,
                run_config.warmups,
                run_config.samples,
            ) {
                if planned.kind == PlannedRunKind::Measured {
                    let mut sample = NormalizedSampleReport::base(
                        probe,
                        planned.phase,
                        planned.run_mode,
                        planned.sample_index.unwrap_or_default(),
                        planned.request_index,
                        planned.phase.warms_before_samples() && run_config.warmups > 0,
                        run_config.context_tokens,
                    );
                    sample.status = "dry_run".to_owned();
                    sample.classification = "planned".to_owned();
                    sample.cached_tokens_status = "not_measured";
                    report.samples.push(sample);
                }
            }
            for planned in concurrent_phase_plan(
                &run_config.cache_phases,
                run_config.concurrent_requests,
                run_config.effective_concurrent_samples,
            ) {
                let mut sample = NormalizedSampleReport::base(
                    probe,
                    planned.phase,
                    planned.run_mode,
                    planned.sample_index.unwrap_or_default(),
                    planned.request_index,
                    planned.phase.warms_before_samples() && run_config.warmups > 0,
                    run_config.context_tokens,
                );
                sample.status = "dry_run".to_owned();
                sample.classification = "planned".to_owned();
                sample.cached_tokens_status = "not_measured";
                report.concurrent_samples.push(sample);
            }
        }
        report
    }
}

#[derive(Debug, Default)]
pub(in crate::qwen_mlx_tool) struct NormalizedAdminMetricsCapture {
    pub(in crate::qwen_mlx_tool) before: Option<Value>,
    pub(in crate::qwen_mlx_tool) after: Option<Value>,
    pub(in crate::qwen_mlx_tool) error: Option<String>,
}

impl NormalizedAdminMetricsCapture {
    pub(in crate::qwen_mlx_tool) fn record_before(&mut self, result: Result<Value, String>) {
        match result {
            Ok(metrics) => self.before = Some(metrics),
            Err(err) => self.push_error(format!("before {err}")),
        }
    }

    pub(in crate::qwen_mlx_tool) fn record_after(&mut self, result: Result<Value, String>) {
        match result {
            Ok(metrics) => self.after = Some(metrics),
            Err(err) => self.push_error(format!("after {err}")),
        }
    }

    pub(in crate::qwen_mlx_tool) fn push_error(&mut self, err: String) {
        match &mut self.error {
            Some(existing) => {
                existing.push_str("; ");
                existing.push_str(&err);
            }
            None => self.error = Some(err),
        }
    }
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedWarmupFailure {
    pub(in crate::qwen_mlx_tool) case: &'static str,
    pub(in crate::qwen_mlx_tool) schema_variant: &'static str,
    pub(in crate::qwen_mlx_tool) tool_choice_variant: &'static str,
    pub(in crate::qwen_mlx_tool) max_tokens: u32,
    pub(in crate::qwen_mlx_tool) cache_phase: &'static str,
    pub(in crate::qwen_mlx_tool) warmup_index: usize,
    pub(in crate::qwen_mlx_tool) classification: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::qwen_mlx_tool) http_status: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::qwen_mlx_tool) error: Option<String>,
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedPlannedRequestReport {
    pub(in crate::qwen_mlx_tool) case: &'static str,
    pub(in crate::qwen_mlx_tool) schema_variant: &'static str,
    pub(in crate::qwen_mlx_tool) tool_choice_variant: &'static str,
    pub(in crate::qwen_mlx_tool) max_tokens: u32,
    pub(in crate::qwen_mlx_tool) cache_phase: &'static str,
    pub(in crate::qwen_mlx_tool) run_mode: &'static str,
    pub(in crate::qwen_mlx_tool) request_kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::qwen_mlx_tool) sample_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::qwen_mlx_tool) request_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::qwen_mlx_tool) warmup_index: Option<usize>,
    pub(in crate::qwen_mlx_tool) planned_prompt_tokens: usize,
    pub(in crate::qwen_mlx_tool) prewarmed: bool,
}

impl NormalizedPlannedRequestReport {
    pub(in crate::qwen_mlx_tool) fn new(
        probe: NormalizedProbePlan,
        planned: PlannedRun,
        run_config: &NormalizedRunConfig,
    ) -> Self {
        Self {
            case: probe.case.name(),
            schema_variant: probe.schema_variant.name(),
            tool_choice_variant: probe.tool_choice_variant.name(),
            max_tokens: probe.max_tokens,
            cache_phase: planned.phase.name(),
            run_mode: planned.run_mode.name(),
            request_kind: planned.kind.name(),
            sample_index: planned.sample_index,
            request_index: planned.request_index,
            warmup_index: planned.warmup_index,
            planned_prompt_tokens: run_config.context_tokens,
            prewarmed: planned.phase.warms_before_samples() && run_config.warmups > 0,
        }
    }
}

#[derive(Debug, Serialize)]
pub(in crate::qwen_mlx_tool) struct NormalizedSampleReport {
    pub(in crate::qwen_mlx_tool) case: &'static str,
    pub(in crate::qwen_mlx_tool) schema_variant: &'static str,
    pub(in crate::qwen_mlx_tool) tool_choice_variant: &'static str,
    pub(in crate::qwen_mlx_tool) max_tokens: u32,
    pub(in crate::qwen_mlx_tool) schema_canonicalized: bool,
    pub(in crate::qwen_mlx_tool) schema_permuted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::qwen_mlx_tool) tool_schema_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::qwen_mlx_tool) tool_schema_bytes: Option<usize>,
    pub(in crate::qwen_mlx_tool) cache_phase: &'static str,
    pub(in crate::qwen_mlx_tool) run_mode: &'static str,
    pub(in crate::qwen_mlx_tool) sample_index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::qwen_mlx_tool) request_index: Option<usize>,
    pub(in crate::qwen_mlx_tool) planned_prompt_tokens: usize,
    pub(in crate::qwen_mlx_tool) prewarmed: bool,
    pub(in crate::qwen_mlx_tool) status: String,
    pub(in crate::qwen_mlx_tool) classification: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::qwen_mlx_tool) failure_classification: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::qwen_mlx_tool) latency_ms: Option<u128>,
    #[serde(flatten)]
    pub(in crate::qwen_mlx_tool) stream_timing: StreamTimingReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::qwen_mlx_tool) tokens_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::qwen_mlx_tool) prompt_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::qwen_mlx_tool) completion_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::qwen_mlx_tool) total_tokens: Option<u64>,
    pub(in crate::qwen_mlx_tool) cached_tokens_status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::qwen_mlx_tool) cached_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::qwen_mlx_tool) http_status: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::qwen_mlx_tool) request_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::qwen_mlx_tool) response_headers: Option<BTreeMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::qwen_mlx_tool) finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::qwen_mlx_tool) error: Option<String>,
    #[serde(skip)]
    pub(in crate::qwen_mlx_tool) tool_required_stream_admin_metrics:
        Option<NormalizedAdminMetricsCapture>,
}

impl NormalizedSampleReport {
    pub(in crate::qwen_mlx_tool) fn base(
        probe: NormalizedProbePlan,
        phase: CachePhase,
        run_mode: RunMode,
        sample_index: usize,
        request_index: Option<usize>,
        prewarmed: bool,
        planned_prompt_tokens: usize,
    ) -> Self {
        let tool_schema = tool_schema_metadata(probe);
        Self {
            case: probe.case.name(),
            schema_variant: probe.schema_variant.name(),
            tool_choice_variant: probe.tool_choice_variant.name(),
            max_tokens: probe.max_tokens,
            schema_canonicalized: probe.schema_variant.canonicalized(),
            schema_permuted: probe.schema_variant.permuted(),
            tool_schema_sha256: tool_schema.sha256,
            tool_schema_bytes: tool_schema.bytes,
            cache_phase: phase.name(),
            run_mode: run_mode.name(),
            sample_index,
            request_index,
            planned_prompt_tokens,
            prewarmed,
            status: "planned".to_owned(),
            classification: "planned".to_owned(),
            failure_classification: None,
            latency_ms: None,
            stream_timing: StreamTimingReport::default(),
            tokens_per_second: None,
            prompt_tokens: None,
            completion_tokens: None,
            total_tokens: None,
            cached_tokens_status: "missing",
            cached_tokens: None,
            http_status: None,
            request_id: None,
            response_headers: None,
            finish_reason: None,
            error: None,
            tool_required_stream_admin_metrics: None,
        }
    }
}
