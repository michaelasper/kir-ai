use super::super::{
    StreamAssembly, StreamTimingReport, StreamTimingTracker, apply_sse_frame, consume_sse_buffer,
    usage_from_value,
};
use super::*;
use crate::{DEFAULT_MODEL_ID, MlxToolParserMode};
use serde_json::json;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

fn lane(spec: &str) -> NormalizedLaneConfig {
    parse_lane_spec(spec).expect("lane spec parses")
}

fn args(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|part| (*part).to_owned()).collect()
}

fn required_tool_ttft_sample(
    probe: NormalizedProbePlan,
    first_byte_ms: u128,
    first_sse_ms: u128,
    first_tool_delta_ms: u128,
    tool_finish_ms: u128,
) -> NormalizedSampleReport {
    let mut sample = NormalizedSampleReport::base(
        probe,
        CachePhase::Cold,
        RunMode::Sequential,
        0,
        None,
        false,
        128,
    );
    sample.status = "passed".to_owned();
    sample.classification = "passed".to_owned();
    sample.latency_ms = Some(tool_finish_ms);
    sample.stream_timing = StreamTimingReport {
        first_byte_latency_ms: Some(first_byte_ms),
        first_sse_data_latency_ms: Some(first_sse_ms),
        first_content_delta_latency_ms: None,
        first_tool_delta_latency_ms: Some(first_tool_delta_ms),
        tool_finish_latency_ms: Some(tool_finish_ms),
        first_semantic_delta_latency_ms: Some(first_tool_delta_ms),
    };
    sample.finish_reason = Some("tool_calls".to_owned());
    sample
}

fn prefill_sweep_sample(
    case: NormalizedCaseKind,
    phase: CachePhase,
    run_mode: RunMode,
    first_semantic_ms: u128,
) -> NormalizedSampleReport {
    let mut sample = NormalizedSampleReport::base(
        NormalizedProbePlan::new(case, SchemaVariant::None, ToolChoiceVariant::None),
        phase,
        run_mode,
        0,
        None,
        false,
        135_000,
    );
    sample.status = "passed".to_owned();
    sample.classification = "passed".to_owned();
    sample.latency_ms = Some(first_semantic_ms + 40);
    sample.stream_timing = StreamTimingReport {
        first_byte_latency_ms: Some(first_semantic_ms - 20),
        first_sse_data_latency_ms: Some(first_semantic_ms - 10),
        first_content_delta_latency_ms: Some(first_semantic_ms),
        first_tool_delta_latency_ms: None,
        tool_finish_latency_ms: None,
        first_semantic_delta_latency_ms: Some(first_semantic_ms),
    };
    sample.prompt_tokens = Some(135_000);
    sample.completion_tokens = Some(8);
    sample.total_tokens = Some(135_008);
    sample.cached_tokens_status = "present";
    sample.cached_tokens = Some(120_000);
    sample
}

fn latest_plain_stream_sample(
    ttfi_ms: u128,
    latency_ms: u128,
    tokens_per_second: f64,
) -> NormalizedSampleReport {
    let mut sample = NormalizedSampleReport::base(
        NormalizedProbePlan::new(
            NormalizedCaseKind::ChatStream,
            SchemaVariant::None,
            ToolChoiceVariant::None,
        ),
        CachePhase::Cold,
        RunMode::Sequential,
        0,
        None,
        false,
        512,
    );
    sample.status = "passed".to_owned();
    sample.classification = "passed".to_owned();
    sample.latency_ms = Some(latency_ms);
    sample.stream_timing = StreamTimingReport {
        first_byte_latency_ms: ttfi_ms.checked_sub(20),
        first_sse_data_latency_ms: ttfi_ms.checked_sub(10),
        first_content_delta_latency_ms: Some(ttfi_ms),
        first_tool_delta_latency_ms: None,
        tool_finish_latency_ms: None,
        first_semantic_delta_latency_ms: Some(ttfi_ms),
    };
    sample.tokens_per_second = Some(tokens_per_second);
    sample.prompt_tokens = Some(512);
    sample.completion_tokens = Some(192);
    sample.total_tokens = Some(704);
    sample.cached_tokens_status = "missing";
    sample
}

fn latest_tool_stream_sample(
    first_tool_delta_ms: u128,
    latency_ms: u128,
    tokens_per_second: f64,
) -> NormalizedSampleReport {
    let mut sample = NormalizedSampleReport::base(
        NormalizedProbePlan::new(
            NormalizedCaseKind::ToolRequiredStream,
            SchemaVariant::CanonicalCurrent,
            ToolChoiceVariant::Required,
        ),
        CachePhase::Cold,
        RunMode::Sequential,
        0,
        None,
        false,
        512,
    );
    sample.status = "passed".to_owned();
    sample.classification = "passed".to_owned();
    sample.latency_ms = Some(latency_ms);
    sample.stream_timing = StreamTimingReport {
        first_byte_latency_ms: first_tool_delta_ms.checked_sub(80),
        first_sse_data_latency_ms: first_tool_delta_ms.checked_sub(70),
        first_content_delta_latency_ms: None,
        first_tool_delta_latency_ms: Some(first_tool_delta_ms),
        tool_finish_latency_ms: Some(latency_ms),
        first_semantic_delta_latency_ms: Some(first_tool_delta_ms),
    };
    sample.tokens_per_second = Some(tokens_per_second);
    sample.prompt_tokens = Some(512);
    sample.completion_tokens = Some(64);
    sample.total_tokens = Some(576);
    sample.cached_tokens_status = "missing";
    sample
}

fn latest_cache_sample(
    phase: CachePhase,
    latency_ms: u128,
    ttfi_ms: u128,
    tokens_per_second: f64,
    cached_tokens: Option<u64>,
) -> NormalizedSampleReport {
    let probe = NormalizedProbePlan::new(
        NormalizedCaseKind::WarmPrefixRepeatedTurnStream,
        SchemaVariant::CanonicalCurrent,
        ToolChoiceVariant::Required,
    );
    let mut sample =
        NormalizedSampleReport::base(probe, phase, RunMode::Sequential, 0, None, false, 1_000);
    sample.status = "passed".to_owned();
    sample.classification = "passed".to_owned();
    sample.latency_ms = Some(latency_ms);
    sample.stream_timing = StreamTimingReport {
        first_byte_latency_ms: ttfi_ms.checked_sub(20),
        first_sse_data_latency_ms: ttfi_ms.checked_sub(10),
        first_content_delta_latency_ms: None,
        first_tool_delta_latency_ms: Some(ttfi_ms),
        tool_finish_latency_ms: Some(latency_ms),
        first_semantic_delta_latency_ms: Some(ttfi_ms),
    };
    sample.tokens_per_second = Some(tokens_per_second);
    sample.prompt_tokens = Some(1_000);
    sample.completion_tokens = Some(10);
    sample.total_tokens = Some(1_010);
    sample.cached_tokens_status = if cached_tokens.is_some() {
        "present"
    } else {
        "missing"
    };
    sample.cached_tokens = cached_tokens;
    sample
}

fn stable_prefix_sample(
    probe: NormalizedProbePlan,
    phase: CachePhase,
    first_semantic_ms: u128,
    cached_tokens: Option<u64>,
    request_id: Option<&str>,
) -> NormalizedSampleReport {
    let mut sample =
        NormalizedSampleReport::base(probe, phase, RunMode::Sequential, 0, None, false, 1000);
    sample.status = "passed".to_owned();
    sample.classification = "passed".to_owned();
    sample.latency_ms = Some(first_semantic_ms + 15);
    sample.stream_timing = StreamTimingReport {
        first_byte_latency_ms: Some(first_semantic_ms - 10),
        first_sse_data_latency_ms: Some(first_semantic_ms - 5),
        first_content_delta_latency_ms: None,
        first_tool_delta_latency_ms: Some(first_semantic_ms),
        tool_finish_latency_ms: Some(first_semantic_ms + 10),
        first_semantic_delta_latency_ms: Some(first_semantic_ms),
    };
    sample.prompt_tokens = Some(1000);
    sample.completion_tokens = Some(10);
    sample.total_tokens = Some(1010);
    sample.cached_tokens_status = if cached_tokens.is_some() {
        "present"
    } else {
        "missing"
    };
    sample.cached_tokens = cached_tokens;
    sample.request_id = request_id.map(str::to_owned);
    sample
}

fn ab_tool_stream_sample(
    first_tool_delta_ms: u128,
    tool_finish_ms: u128,
) -> NormalizedSampleReport {
    let probe = NormalizedProbePlan::new(
        NormalizedCaseKind::ToolRequiredStream,
        SchemaVariant::CanonicalCurrent,
        ToolChoiceVariant::Required,
    );
    let mut sample = NormalizedSampleReport::base(
        probe,
        CachePhase::WarmSamePrompt,
        RunMode::Sequential,
        0,
        None,
        true,
        128,
    );
    sample.status = "passed".to_owned();
    sample.classification = "passed".to_owned();
    sample.latency_ms = Some(tool_finish_ms + 5);
    sample.stream_timing = StreamTimingReport {
        first_byte_latency_ms: Some(first_tool_delta_ms.saturating_sub(20)),
        first_sse_data_latency_ms: Some(first_tool_delta_ms.saturating_sub(10)),
        first_content_delta_latency_ms: None,
        first_tool_delta_latency_ms: Some(first_tool_delta_ms),
        tool_finish_latency_ms: Some(tool_finish_ms),
        first_semantic_delta_latency_ms: Some(first_tool_delta_ms),
    };
    sample.prompt_tokens = Some(1835);
    sample.completion_tokens = Some(64);
    sample.total_tokens = Some(1899);
    sample.cached_tokens_status = "present";
    sample.cached_tokens = Some(1834);
    sample.finish_reason = Some("tool_calls".to_owned());
    sample
}

fn passed_sample(
    case: NormalizedCaseKind,
    phase: CachePhase,
    run_mode: RunMode,
    sample_index: usize,
    request_index: Option<usize>,
    latency_ms: u128,
    cached_tokens: u64,
) -> NormalizedSampleReport {
    let mut sample = NormalizedSampleReport::base(
        NormalizedProbePlan::new(
            case,
            SchemaVariant::BaselineCurrent,
            ToolChoiceVariant::Required,
        ),
        phase,
        run_mode,
        sample_index,
        request_index,
        false,
        128,
    );
    sample.status = "passed".to_owned();
    sample.classification = "passed".to_owned();
    sample.latency_ms = Some(latency_ms);
    sample.prompt_tokens = Some(1000);
    sample.completion_tokens = Some(10);
    sample.total_tokens = Some(1010);
    sample.cached_tokens_status = "present";
    sample.cached_tokens = Some(cached_tokens);
    sample
}

fn failed_summary_sample(
    case: NormalizedCaseKind,
    phase: CachePhase,
    run_mode: RunMode,
    sample_index: usize,
    request_index: Option<usize>,
) -> NormalizedSampleReport {
    let mut sample = NormalizedSampleReport::base(
        NormalizedProbePlan::new(
            case,
            SchemaVariant::BaselineCurrent,
            ToolChoiceVariant::Required,
        ),
        phase,
        run_mode,
        sample_index,
        request_index,
        false,
        128,
    );
    sample.status = "failed".to_owned();
    sample.classification = "http_status_failed".to_owned();
    sample.latency_ms = Some(900);
    sample
}
