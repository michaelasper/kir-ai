use crate::{BenchCaseKind, BenchProfileKind, CaseRun, StreamTimingReport};
use serde_json::Value;
use std::time::Duration;

#[derive(Debug, Clone)]
pub(crate) struct UsageMetrics {
    pub(crate) prompt_tokens: Option<u64>,
    pub(crate) completion_tokens: Option<u64>,
    pub(crate) total_tokens: Option<u64>,
    pub(crate) cached_tokens_status: Option<&'static str>,
    pub(crate) cached_tokens: Option<u64>,
}

impl Default for UsageMetrics {
    fn default() -> Self {
        Self {
            prompt_tokens: None,
            completion_tokens: None,
            total_tokens: None,
            cached_tokens_status: Some("missing"),
            cached_tokens: None,
        }
    }
}

impl UsageMetrics {
    pub(crate) fn merge(&mut self, next: Self) {
        self.prompt_tokens = max_optional_u64(self.prompt_tokens, next.prompt_tokens);
        self.completion_tokens = max_optional_u64(self.completion_tokens, next.completion_tokens);
        self.total_tokens = max_optional_u64(self.total_tokens, next.total_tokens);
        self.cached_tokens = max_optional_u64(self.cached_tokens, next.cached_tokens);
        self.cached_tokens_status =
            merge_cached_tokens_status(self.cached_tokens_status, next.cached_tokens_status);
        if self.total_tokens.is_none()
            && let (Some(prompt), Some(completion)) = (self.prompt_tokens, self.completion_tokens)
        {
            self.total_tokens = prompt.checked_add(completion);
        }
    }
}

fn max_optional_u64(current: Option<u64>, next: Option<u64>) -> Option<u64> {
    match (current, next) {
        (Some(current), Some(next)) => Some(current.max(next)),
        (Some(current), None) => Some(current),
        (None, Some(next)) => Some(next),
        (None, None) => None,
    }
}

pub(crate) fn sum_optional_u64(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.saturating_add(right)),
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (None, None) => None,
    }
}

pub(crate) fn sum_optional_u128(left: Option<u128>, right: Option<u128>) -> Option<u128> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.saturating_add(right)),
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (None, None) => None,
    }
}

pub(crate) fn merge_cached_tokens_status(
    current: Option<&'static str>,
    next: Option<&'static str>,
) -> Option<&'static str> {
    match (
        cached_tokens_status_rank(current),
        cached_tokens_status_rank(next),
    ) {
        (Some((current_rank, current)), Some((next_rank, next))) => {
            if next_rank > current_rank {
                Some(next)
            } else {
                Some(current)
            }
        }
        (Some((_rank, current)), None) => Some(current),
        (None, Some((_rank, next))) => Some(next),
        (None, None) => None,
    }
}

fn cached_tokens_status_rank(status: Option<&'static str>) -> Option<(u8, &'static str)> {
    status.map(|status| {
        let rank = match status {
            "present" => 4,
            "invalid" => 3,
            "null" => 2,
            "missing" => 1,
            _ => 0,
        };
        (rank, status)
    })
}

#[derive(Debug, Default)]
pub(crate) struct StreamTimingTracker {
    pub(crate) first_byte_latency: Option<Duration>,
    pub(crate) first_sse_data_latency: Option<Duration>,
    pub(crate) first_content_delta_latency: Option<Duration>,
    pub(crate) first_tool_delta_latency: Option<Duration>,
    pub(crate) tool_finish_latency: Option<Duration>,
    pub(crate) first_semantic_delta_latency: Option<Duration>,
}

impl StreamTimingTracker {
    pub(crate) fn record_first_byte(&mut self, elapsed: Duration) {
        if self.first_byte_latency.is_none() {
            self.first_byte_latency = Some(elapsed);
        }
    }

    pub(crate) fn record_sse_frame(&mut self, elapsed: Duration, delta: StreamFrameDelta) {
        if self.first_sse_data_latency.is_none() {
            self.first_sse_data_latency = Some(elapsed);
        }
        if delta.content && self.first_content_delta_latency.is_none() {
            self.first_content_delta_latency = Some(elapsed);
        }
        if delta.tool && self.first_tool_delta_latency.is_none() {
            self.first_tool_delta_latency = Some(elapsed);
        }
        if delta.tool_finish && self.tool_finish_latency.is_none() {
            self.tool_finish_latency = Some(elapsed);
        }
        if delta.semantic() && self.first_semantic_delta_latency.is_none() {
            self.first_semantic_delta_latency = Some(elapsed);
        }
    }

    pub(crate) fn to_report(&self) -> StreamTimingReport {
        StreamTimingReport {
            first_byte_latency_ms: self.first_byte_latency.map(|duration| duration.as_millis()),
            first_sse_data_latency_ms: self
                .first_sse_data_latency
                .map(|duration| duration.as_millis()),
            first_content_delta_latency_ms: self
                .first_content_delta_latency
                .map(|duration| duration.as_millis()),
            first_tool_delta_latency_ms: self
                .first_tool_delta_latency
                .map(|duration| duration.as_millis()),
            tool_finish_latency_ms: self
                .tool_finish_latency
                .map(|duration| duration.as_millis()),
            first_semantic_delta_latency_ms: self
                .first_semantic_delta_latency
                .map(|duration| duration.as_millis()),
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct StreamFrameDelta {
    pub(crate) content: bool,
    pub(crate) tool: bool,
    pub(crate) tool_finish: bool,
}

impl StreamFrameDelta {
    pub(crate) fn semantic(&self) -> bool {
        self.content || self.tool
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct StreamAssembly {
    pub(crate) content: String,
    pub(crate) tool_name: Option<String>,
    pub(crate) tool_arguments: String,
    pub(crate) finish_reason: Option<String>,
    pub(crate) usage: UsageMetrics,
}

pub(crate) fn validate_buffered_case(
    profile: BenchProfileKind,
    case: BenchCaseKind,
    marker: &str,
    value: &Value,
) -> Result<(), String> {
    match case {
        BenchCaseKind::PlainRecall
        | BenchCaseKind::MultiTurnLifecycle
        | BenchCaseKind::SameLongPromptTwice
        | BenchCaseKind::SharedPrefixShortSuffixVariation => {
            let content = value
                .pointer("/choices/0/message/content")
                .and_then(Value::as_str)
                .ok_or_else(|| "missing assistant content".to_owned())?;
            if content.contains(marker) {
                Ok(())
            } else {
                Err(format!(
                    "assistant content did not contain marker `{marker}`"
                ))
            }
        }
        BenchCaseKind::JsonObjectRecall => {
            let content = value
                .pointer("/choices/0/message/content")
                .and_then(Value::as_str)
                .ok_or_else(|| "missing assistant JSON content".to_owned())?;
            let parsed = serde_json::from_str::<Value>(content)
                .map_err(|err| format!("assistant content was not valid JSON: {err}"))?;
            parsed
                .as_object()
                .ok_or_else(|| "assistant JSON content was not an object".to_owned())?;
            validate_recall_arguments(&parsed, profile, case, marker, "JSON")
        }
        BenchCaseKind::RequiredToolRecall => {
            let finish_reason = value
                .pointer("/choices/0/finish_reason")
                .and_then(Value::as_str);
            validate_tool_finish_reason(finish_reason, "tool call")?;
            let tool_call = value
                .pointer("/choices/0/message/tool_calls/0")
                .ok_or_else(|| "missing required tool call".to_owned())?;
            validate_tool_call(tool_call, profile, case, marker)
        }
        BenchCaseKind::StreamedRequiredToolRecall => {
            Err("streamed tool case was routed through buffered validator".to_owned())
        }
    }
}

pub(crate) fn validate_streaming_case(
    profile: BenchProfileKind,
    case: BenchCaseKind,
    marker: &str,
    assembly: &StreamAssembly,
) -> Result<(), String> {
    if !case.streams() {
        return Err("non-streaming case was routed through streaming validator".to_owned());
    }
    let name = assembly
        .tool_name
        .as_deref()
        .ok_or_else(|| "missing streamed tool name".to_owned())?;
    if name != "report_long_context_recall" {
        return Err(format!(
            "streamed tool name `{name}` did not match expected"
        ));
    }
    validate_tool_finish_reason(assembly.finish_reason.as_deref(), "streamed tool call")?;
    let args = serde_json::from_str::<Value>(&assembly.tool_arguments)
        .map_err(|err| format!("streamed tool arguments were not JSON: {err}"))?;
    validate_recall_arguments(&args, profile, case, marker, "streamed tool")
}

pub(crate) fn validate_tool_call(
    tool_call: &Value,
    profile: BenchProfileKind,
    case: BenchCaseKind,
    marker: &str,
) -> Result<(), String> {
    let name = tool_call
        .pointer("/function/name")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing tool function name".to_owned())?;
    if name != "report_long_context_recall" {
        return Err(format!("tool function `{name}` did not match expected"));
    }
    let args_text = tool_call
        .pointer("/function/arguments")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing tool function arguments".to_owned())?;
    let args = serde_json::from_str::<Value>(args_text)
        .map_err(|err| format!("tool arguments were not JSON: {err}"))?;
    validate_recall_arguments(&args, profile, case, marker, "tool")
}

pub(crate) fn validate_tool_finish_reason(
    finish_reason: Option<&str>,
    label: &str,
) -> Result<(), String> {
    match finish_reason {
        Some("tool_calls") => Ok(()),
        Some(other) => Err(format!(
            "{label} finish_reason `{other}` did not equal `tool_calls`"
        )),
        None => Err(format!("{label} response was missing finish_reason")),
    }
}

pub(crate) fn validate_recall_arguments(
    args: &Value,
    profile: BenchProfileKind,
    case: BenchCaseKind,
    marker: &str,
    label: &str,
) -> Result<(), String> {
    let object = args
        .as_object()
        .ok_or_else(|| format!("{label} arguments were not a JSON object"))?;
    validate_recall_argument(object.get("marker"), marker, label, "marker")?;
    validate_recall_argument(object.get("profile"), profile.name(), label, "profile")?;
    validate_recall_argument(object.get("case"), case.name(), label, "case")?;
    if object.len() != 3 {
        return Err(format!(
            "{label} arguments must contain exactly marker, profile, and case"
        ));
    }
    Ok(())
}

fn validate_recall_argument(
    value: Option<&Value>,
    expected: &str,
    label: &str,
    key: &str,
) -> Result<(), String> {
    match value.and_then(Value::as_str) {
        Some(actual) if actual == expected => Ok(()),
        Some(actual) => Err(format!(
            "{label} {key} `{actual}` did not equal `{expected}`"
        )),
        None => Err(format!("{label} arguments missing string `{key}`")),
    }
}

pub(crate) fn consume_sse_buffer(
    buffer: &mut String,
    assembly: &mut StreamAssembly,
    timings: &mut StreamTimingTracker,
    elapsed: Duration,
) {
    while let Some(index) = buffer.find('\n') {
        let mut line = buffer[..index].trim_end_matches('\r').to_owned();
        buffer.drain(..=index);
        if !line.starts_with("data:") {
            continue;
        }
        line.drain(..5);
        let data = line.trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        let delta = apply_sse_frame(&value, assembly);
        timings.record_sse_frame(elapsed, delta);
    }
}

pub(crate) fn apply_sse_frame(value: &Value, assembly: &mut StreamAssembly) -> StreamFrameDelta {
    let mut delta = StreamFrameDelta::default();
    if let Some(usage) = value.get("usage") {
        assembly.usage.merge(usage_from_value(Some(usage)));
    }
    if let Some(choice) = value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
    {
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            if reason == "tool_calls" {
                delta.tool_finish = true;
            }
            assembly.finish_reason = Some(reason.to_owned());
        }
        if let Some(content) = choice.pointer("/delta/content").and_then(Value::as_str) {
            if !content.is_empty() {
                delta.content = true;
            }
            assembly.content.push_str(content);
        }
        if let Some(tool_calls) = choice
            .pointer("/delta/tool_calls")
            .and_then(Value::as_array)
        {
            for tool_call in tool_calls {
                if tool_call
                    .get("id")
                    .and_then(Value::as_str)
                    .is_some_and(|id| !id.is_empty())
                {
                    delta.tool = true;
                }
                if let Some(name) = tool_call.pointer("/function/name").and_then(Value::as_str) {
                    if !name.is_empty() {
                        delta.tool = true;
                    }
                    assembly.tool_name = Some(name.to_owned());
                }
                if let Some(arguments) = tool_call
                    .pointer("/function/arguments")
                    .and_then(Value::as_str)
                {
                    if !arguments.is_empty() {
                        delta.tool = true;
                    }
                    assembly.tool_arguments.push_str(arguments);
                }
            }
        }
    }
    delta
}

pub(crate) fn usage_from_value(value: Option<&Value>) -> UsageMetrics {
    let Some(value) = value else {
        return UsageMetrics::default();
    };
    UsageMetrics {
        prompt_tokens: value.get("prompt_tokens").and_then(Value::as_u64),
        completion_tokens: value.get("completion_tokens").and_then(Value::as_u64),
        total_tokens: value.get("total_tokens").and_then(Value::as_u64),
        cached_tokens_status: cached_tokens_status(value),
        cached_tokens: value
            .pointer("/prompt_tokens_details/cached_tokens")
            .and_then(Value::as_u64),
    }
}

fn cached_tokens_status(value: &Value) -> Option<&'static str> {
    match value.pointer("/prompt_tokens_details/cached_tokens") {
        Some(Value::Number(number)) if number.as_u64().is_some() => Some("present"),
        Some(Value::Null) => Some("null"),
        Some(_) => Some("invalid"),
        None => Some("missing"),
    }
}

pub(crate) fn case_from_validation(
    validation: Result<(), String>,
    planned_prompt_tokens: usize,
    latency: Duration,
    stream_timing: StreamTimingReport,
    http_status: Option<u16>,
    finish_reason: Option<String>,
    usage: UsageMetrics,
) -> CaseRun {
    let latency_ms = latency.as_millis();
    let tokens_per_second = usage.completion_tokens.and_then(|tokens| {
        (latency.as_secs_f64() > 0.0).then_some(tokens as f64 / latency.as_secs_f64())
    });
    match validation {
        Ok(()) => CaseRun {
            status: "passed",
            classification: "passed".to_owned(),
            planned_prompt_tokens,
            latency_ms: Some(latency_ms),
            stream_timing,
            tokens_per_second,
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
            cached_tokens_status: usage.cached_tokens_status,
            cached_tokens: usage.cached_tokens,
            prompt_hash: None,
            http_status,
            finish_reason,
            error: None,
        },
        Err(err) => CaseRun {
            status: "failed",
            classification: "response_validation_failed".to_owned(),
            planned_prompt_tokens,
            latency_ms: Some(latency_ms),
            stream_timing,
            tokens_per_second,
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
            cached_tokens_status: usage.cached_tokens_status,
            cached_tokens: usage.cached_tokens,
            prompt_hash: None,
            http_status,
            finish_reason,
            error: Some(err),
        },
    }
}
