use super::{HardwareReport, ModelIdentityReport, StreamTimingReport};

mod config;
mod metrics;
mod planning;
mod report;
mod runner;

use planning::{
    CachePhase, NormalizedCaseKind, NormalizedProbePlan, NormalizedProbeSuite, PlannedRun,
    PlannedRunKind, RunMode, concurrent_phase_plan, phase_plan, tool_schema_metadata,
};
pub(super) use runner::run_qwen_mlx_tool_normalized_bench;

#[cfg(test)]
use config::{
    DefaultOrU32, DefaultOrU64, NormalizedLaneConfig, NormalizedLaneKind,
    NormalizedModelAddressing, NormalizedRunConfig, NormalizedSweepProfile,
    NormalizedTemplatePolicy, UnsetOrU64, default_run_config_for_probe_suite,
    effective_concurrent_samples, parse_cache_phases_flag, parse_lane_spec, parse_lane_specs,
    parse_probe_suite_flag, sweep_profile_requires_exact_token_prompt,
};
#[cfg(test)]
use metrics::*;
#[cfg(test)]
use planning::{
    CONTEXT_RECALL_STREAM_135K_MARKER, DEFAULT_MAX_TOKENS, ProbePrompt, SampleContext,
    SchemaVariant, ToolChoiceVariant, probe_request_body,
};
#[cfg(test)]
use report::*;
#[cfg(test)]
use runner::{
    LaneRunContext, NormalizedProgress, ProbeResponseMetadata, admin_metrics_url,
    capture_normalized_admin_metrics, chat_completions_url, failed_sample,
    load_lane_snapshot_identity, run_lane, sample_from_validation, validate_buffered_probe,
    validate_streaming_probe,
};
#[cfg(test)]
use serde_json::Value;
#[cfg(test)]
use std::path::Path;

const BENCH_REPO_DIR_ENV: &str = "LLM_ENGINE_BENCH_REPO_DIR";
const BENCH_REPO_COMMIT_ENV: &str = "LLM_ENGINE_BENCH_REPO_COMMIT";
const BENCH_REPO_BRANCH_ENV: &str = "LLM_ENGINE_BENCH_REPO_BRANCH";
const BENCH_REPO_DIRTY_ENV: &str = "LLM_ENGINE_BENCH_REPO_DIRTY";
const BENCH_REPO_ORIGIN_FILE: &str = ".kir-ai-origin.json";
const AGENTIC_AB_CASE: &str = "tool_required_stream";
const AGENTIC_AB_SCHEMA_VARIANT: &str = "canonical_current";
const AGENTIC_AB_TOOL_CHOICE_VARIANT: &str = "required";
const AGENTIC_AB_FAST_PATH_KIND: &str = "kir_ai_proxy";

#[cfg(test)]
mod tests;
