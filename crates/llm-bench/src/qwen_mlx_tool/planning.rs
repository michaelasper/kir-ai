use super::config::NormalizedLaneConfig;
use llm_api::canonicalize_json_value;
use llm_tokenizer::HuggingFaceTokenizer;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

pub(in crate::qwen_mlx_tool) const DEFAULT_MAX_TOKENS: u32 = 512;
const REQUIRED_TOOL_TTFT_MAX_TOKENS: [u32; 3] = [24, 48, 96];
pub(in crate::qwen_mlx_tool) const PREFILL_SWEEP_135K_PROFILE_NAME: &str =
    "qwen-prefill-sweep-135k";
const CHAT_STREAM_MARKER: &str = "KIR_QWEN_MLX_PREFILL_135K_CHAT_STREAM_QUARTZ_2741";
pub(in crate::qwen_mlx_tool) const CONTEXT_RECALL_STREAM_135K_MARKER: &str =
    "KIR_LONG_CONTEXT_135K_CONTEXT_RECALL_STREAM_135K_QUARTZ_2741";

pub(in crate::qwen_mlx_tool) fn probe_request_body(
    lane: &NormalizedLaneConfig,
    probe: NormalizedProbePlan,
    prompt: ProbePrompt,
) -> Value {
    let mut body = json!({
        "max_tokens": probe.max_tokens,
        "temperature": 0,
        "top_p": 1,
        "messages": probe_messages(probe.case, prompt)
    });
    if let Some(model_id) = lane.request_model_id() {
        body["model"] = json!(model_id);
    }
    match probe.case {
        NormalizedCaseKind::ToolRequired
        | NormalizedCaseKind::ToolRequiredStream
        | NormalizedCaseKind::ContextRecallStream135k
        | NormalizedCaseKind::OmpRepeatedPrefix
        | NormalizedCaseKind::WarmPrefixRepeatedTurnStream => {
            body["tools"] = json!([probe_tool_schema(probe)]);
            body["tool_choice"] = probe.tool_choice_variant.request_value(probe.case);
            if probe.case.streams() {
                body["stream"] = json!(true);
                body["stream_options"] = json!({"include_usage": true});
            }
        }
        NormalizedCaseKind::ChatStream => {
            body["stream"] = json!(true);
            body["stream_options"] = json!({"include_usage": true});
        }
        NormalizedCaseKind::JsonObject => {
            body["response_format"] = json!({"type": "json_object"});
        }
    }
    lane.template.apply_request_kwargs(&mut body);
    body
}

fn probe_messages(case: NormalizedCaseKind, prompt: ProbePrompt) -> Value {
    if matches!(
        case,
        NormalizedCaseKind::OmpRepeatedPrefix | NormalizedCaseKind::WarmPrefixRepeatedTurnStream
    ) {
        let history_probe_id = format!("{}_HISTORY", case.probe_id());
        let history_arguments =
            json!({"probe_id": history_probe_id.clone(), "case": case.name()}).to_string();
        return json!([
            {"role": "system", "content": case.system_prompt()},
            {"role": "user", "content": stable_context_prefix(prompt.context_tokens, case)},
            {
                "role": "assistant",
                "content": Value::Null,
                "tool_calls": [{
                    "id": "call_qwen_tool_probe_history",
                    "type": "function",
                    "function": {
                        "name": "record_qwen_tool_probe",
                        "arguments": history_arguments
                    }
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_qwen_tool_probe_history",
                "content": json!({"status": "recorded", "probe_id": history_probe_id}).to_string()
            },
            {"role": "user", "content": prompt.user_prompt(case)}
        ]);
    }
    json!([
        {"role": "system", "content": case.system_prompt()},
        {"role": "user", "content": prompt.user_prompt(case)}
    ])
}

fn probe_tool_schema(probe: NormalizedProbePlan) -> Value {
    match probe.case {
        NormalizedCaseKind::ContextRecallStream135k => {
            recall_probe_tool_schema(probe.schema_variant)
        }
        _ => qwen_probe_tool_schema(probe.schema_variant),
    }
}

fn qwen_probe_tool_schema(variant: SchemaVariant) -> Value {
    let minimal = json!({
        "type": "function",
        "function": {
            "name": "record_qwen_tool_probe",
            "parameters": {
                "type": "object",
                "properties": {
                    "probe_id": {"type": "string"},
                    "case": {"type": "string"}
                },
                "required": ["probe_id", "case"]
            }
        }
    });
    let current = json!({
        "type": "function",
        "function": {
            "name": "record_qwen_tool_probe",
            "description": "Record the normalized Qwen tool benchmark probe.",
            "parameters": {
                "type": "object",
                "properties": {
                    "probe_id": {"type": "string"},
                    "case": {"type": "string"}
                },
                "required": ["probe_id", "case"],
                "additionalProperties": false
            }
        }
    });
    let permuted = json!({
        "function": {
            "parameters": {
                "additionalProperties": false,
                "required": ["case", "probe_id"],
                "properties": {
                    "case": {"type": "string"},
                    "probe_id": {"type": "string"}
                },
                "type": "object"
            },
            "description": "Record the normalized Qwen tool benchmark probe.",
            "name": "record_qwen_tool_probe"
        },
        "type": "function"
    });
    let omp_style_i = json!({
        "type": "function",
        "function": {
            "name": "record_qwen_tool_probe",
            "description": "Record an OpenManus-style Qwen tool probe with a call index field.",
            "parameters": {
                "type": "object",
                "properties": {
                    "_i": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Optional tool-call index emitted by OMP-style agents."
                    },
                    "probe_id": {"type": "string"},
                    "case": {"type": "string"}
                },
                "required": ["probe_id", "case"],
                "additionalProperties": false
            }
        }
    });
    match variant {
        SchemaVariant::MinimalShallow => minimal,
        SchemaVariant::BaselineCurrent => current,
        SchemaVariant::CanonicalCurrent => canonicalize_json_value(&current),
        SchemaVariant::BaselinePermutedEquivalent => permuted,
        SchemaVariant::CanonicalPermutedEquivalent => canonicalize_json_value(&permuted),
        SchemaVariant::OmpStyleI => omp_style_i,
        SchemaVariant::LargeStress => large_stress_qwen_probe_tool_schema(),
        SchemaVariant::None => Value::Null,
    }
}

fn large_stress_qwen_probe_tool_schema() -> Value {
    let mut properties = Map::new();
    properties.insert("probe_id".to_owned(), json!({"type": "string"}));
    properties.insert("case".to_owned(), json!({"type": "string"}));
    properties.insert(
        "agent_context".to_owned(),
        json!({
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "step": {"type": "integer"},
                "source": {"type": "string"}
            },
            "additionalProperties": false
        }),
    );
    properties.insert(
        "evidence".to_owned(),
        json!({
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                    "score": {"type": "number"}
                },
                "required": ["key", "value"],
                "additionalProperties": false
            },
            "maxItems": 8
        }),
    );
    for index in 0..32 {
        properties.insert(
            format!("stress_field_{index:02}"),
            json!({
                "type": ["string", "null"],
                "description": format!("Optional distractor schema field {index:02} for required-tool TTFT stress.")
            }),
        );
    }

    let mut parameters = Map::new();
    parameters.insert("type".to_owned(), json!("object"));
    parameters.insert("properties".to_owned(), Value::Object(properties));
    parameters.insert("required".to_owned(), json!(["probe_id", "case"]));
    parameters.insert("additionalProperties".to_owned(), Value::Bool(false));

    json!({
        "type": "function",
        "function": {
            "name": "record_qwen_tool_probe",
            "description": "Record the normalized Qwen tool benchmark probe with a deliberately large schema.",
            "parameters": Value::Object(parameters)
        }
    })
}

fn recall_probe_tool_schema(variant: SchemaVariant) -> Value {
    let current = json!({
        "type": "function",
        "function": {
            "name": "report_long_context_recall",
            "description": "Report a recalled long-context benchmark marker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case": {"type": "string"},
                    "marker": {"type": "string"},
                    "profile": {"type": "string"}
                },
                "required": ["case", "marker", "profile"],
                "additionalProperties": false
            }
        }
    });
    let permuted = json!({
        "function": {
            "parameters": {
                "additionalProperties": false,
                "required": ["profile", "marker", "case"],
                "properties": {
                    "profile": {"type": "string"},
                    "marker": {"type": "string"},
                    "case": {"type": "string"}
                },
                "type": "object"
            },
            "description": "Report a recalled long-context benchmark marker.",
            "name": "report_long_context_recall"
        },
        "type": "function"
    });
    match variant {
        SchemaVariant::MinimalShallow | SchemaVariant::OmpStyleI | SchemaVariant::LargeStress => {
            current
        }
        SchemaVariant::BaselineCurrent => current,
        SchemaVariant::CanonicalCurrent => canonicalize_json_value(&current),
        SchemaVariant::BaselinePermutedEquivalent => permuted,
        SchemaVariant::CanonicalPermutedEquivalent => canonicalize_json_value(&permuted),
        SchemaVariant::None => Value::Null,
    }
}

pub(in crate::qwen_mlx_tool) fn tool_schema_metadata(
    probe: NormalizedProbePlan,
) -> ToolSchemaMetadata {
    if probe.schema_variant == SchemaVariant::None {
        return ToolSchemaMetadata {
            sha256: None,
            bytes: None,
        };
    }
    let schema_json = serde_json::to_string(&json!([probe_tool_schema(probe)]))
        .expect("benchmark tool schema serializes");
    let digest = Sha256::digest(schema_json.as_bytes());
    ToolSchemaMetadata {
        sha256: Some(format!("{digest:x}")),
        bytes: Some(schema_json.len()),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::qwen_mlx_tool) struct ToolSchemaMetadata {
    pub(in crate::qwen_mlx_tool) sha256: Option<String>,
    pub(in crate::qwen_mlx_tool) bytes: Option<usize>,
}

pub(in crate::qwen_mlx_tool) fn phase_plan(
    phases: &[CachePhase],
    warmups: usize,
    samples: usize,
) -> Vec<PlannedRun> {
    let mut runs = Vec::new();
    for &phase in phases {
        if phase.warms_before_samples() {
            for warmup_index in 0..warmups {
                runs.push(PlannedRun {
                    phase,
                    kind: PlannedRunKind::Warmup,
                    run_mode: RunMode::Sequential,
                    sample_index: None,
                    request_index: None,
                    warmup_index: Some(warmup_index),
                });
            }
        }
        for sample_index in 0..samples {
            runs.push(PlannedRun {
                phase,
                kind: PlannedRunKind::Measured,
                run_mode: RunMode::Sequential,
                sample_index: Some(sample_index),
                request_index: None,
                warmup_index: None,
            });
        }
    }
    runs
}

pub(in crate::qwen_mlx_tool) fn concurrent_phase_plan(
    phases: &[CachePhase],
    concurrent_requests: usize,
    concurrent_samples: usize,
) -> Vec<PlannedRun> {
    let mut runs = Vec::new();
    for &phase in phases {
        for sample_index in 0..concurrent_samples {
            for request_index in 0..concurrent_requests {
                runs.push(PlannedRun {
                    phase,
                    kind: PlannedRunKind::Measured,
                    run_mode: RunMode::Concurrent,
                    sample_index: Some(sample_index),
                    request_index: Some(request_index),
                    warmup_index: None,
                });
            }
        }
    }
    runs
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::qwen_mlx_tool) enum NormalizedCaseKind {
    ToolRequired,
    ToolRequiredStream,
    JsonObject,
    OmpRepeatedPrefix,
    ChatStream,
    ContextRecallStream135k,
    WarmPrefixRepeatedTurnStream,
}

impl NormalizedCaseKind {
    pub(in crate::qwen_mlx_tool) fn all() -> [Self; 4] {
        [
            Self::ToolRequired,
            Self::ToolRequiredStream,
            Self::JsonObject,
            Self::OmpRepeatedPrefix,
        ]
    }

    pub(in crate::qwen_mlx_tool) fn name(self) -> &'static str {
        match self {
            Self::ToolRequired => "tool_required",
            Self::ToolRequiredStream => "tool_required_stream",
            Self::JsonObject => "json_object",
            Self::OmpRepeatedPrefix => "omp_repeated_prefix",
            Self::ChatStream => "chat_stream",
            Self::ContextRecallStream135k => "context_recall_stream_135k",
            Self::WarmPrefixRepeatedTurnStream => "warm_prefix_repeated_turn_stream",
        }
    }

    pub(in crate::qwen_mlx_tool) fn probe_id(self) -> &'static str {
        match self {
            Self::ToolRequired => "KIR_QWEN_MLX_TOOL_NORMALIZED_TOOL_REQUIRED",
            Self::ToolRequiredStream => "KIR_QWEN_MLX_TOOL_NORMALIZED_TOOL_REQUIRED_STREAM",
            Self::JsonObject => "KIR_QWEN_MLX_TOOL_NORMALIZED_JSON_OBJECT",
            Self::OmpRepeatedPrefix => "KIR_QWEN_MLX_TOOL_NORMALIZED_OMP_REPEATED_PREFIX",
            Self::ChatStream => "KIR_QWEN_MLX_TOOL_NORMALIZED_CHAT_STREAM",
            Self::ContextRecallStream135k => {
                "KIR_QWEN_MLX_TOOL_NORMALIZED_CONTEXT_RECALL_STREAM_135K"
            }
            Self::WarmPrefixRepeatedTurnStream => {
                "KIR_QWEN_MLX_TOOL_NORMALIZED_WARM_PREFIX_REPEATED_TURN_STREAM"
            }
        }
    }

    pub(in crate::qwen_mlx_tool) fn system_prompt(self) -> &'static str {
        match self {
            Self::ToolRequired | Self::ToolRequiredStream => {
                "You are a tool-call conformance probe. Use the provided function exactly once."
            }
            Self::ChatStream => {
                "You are a streaming chat latency probe. Return the requested marker in assistant content."
            }
            Self::ContextRecallStream135k => {
                "You are a long-context streaming tool-call evaluator. Use the provided function to report the recalled marker."
            }
            Self::JsonObject => {
                "You are a JSON conformance probe. Return one JSON object and no prose."
            }
            Self::OmpRepeatedPrefix => {
                "You are an OMP-style repeated-prefix workflow probe. Continue the tool workflow and use the provided function exactly once."
            }
            Self::WarmPrefixRepeatedTurnStream => {
                "You are a warm-prefix repeated-turn streaming workflow probe. Continue the tool workflow and use the provided function exactly once."
            }
        }
    }

    pub(in crate::qwen_mlx_tool) fn streams(self) -> bool {
        matches!(
            self,
            Self::ToolRequiredStream
                | Self::ChatStream
                | Self::ContextRecallStream135k
                | Self::WarmPrefixRepeatedTurnStream
        )
    }

    pub(in crate::qwen_mlx_tool) fn tool_function_name(self) -> &'static str {
        match self {
            Self::ContextRecallStream135k => "report_long_context_recall",
            _ => "record_qwen_tool_probe",
        }
    }

    pub(in crate::qwen_mlx_tool) fn requires_tool_delta(self) -> bool {
        matches!(
            self,
            Self::ToolRequiredStream
                | Self::ContextRecallStream135k
                | Self::WarmPrefixRepeatedTurnStream
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::qwen_mlx_tool) enum SchemaVariant {
    None,
    MinimalShallow,
    BaselineCurrent,
    CanonicalCurrent,
    BaselinePermutedEquivalent,
    CanonicalPermutedEquivalent,
    OmpStyleI,
    LargeStress,
}

impl SchemaVariant {
    pub(in crate::qwen_mlx_tool) fn all() -> [Self; 4] {
        [
            Self::BaselineCurrent,
            Self::CanonicalCurrent,
            Self::BaselinePermutedEquivalent,
            Self::CanonicalPermutedEquivalent,
        ]
    }

    pub(in crate::qwen_mlx_tool) fn required_tool_ttft_matrix() -> [Self; 4] {
        [
            Self::MinimalShallow,
            Self::CanonicalCurrent,
            Self::OmpStyleI,
            Self::LargeStress,
        ]
    }

    pub(in crate::qwen_mlx_tool) fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::MinimalShallow => "minimal_shallow",
            Self::BaselineCurrent => "baseline_current",
            Self::CanonicalCurrent => "canonical_current",
            Self::BaselinePermutedEquivalent => "baseline_permuted_equivalent",
            Self::CanonicalPermutedEquivalent => "canonical_permuted_equivalent",
            Self::OmpStyleI => "omp_style_i",
            Self::LargeStress => "large_stress",
        }
    }

    pub(in crate::qwen_mlx_tool) fn canonicalized(self) -> bool {
        matches!(
            self,
            Self::CanonicalCurrent | Self::CanonicalPermutedEquivalent
        )
    }

    pub(in crate::qwen_mlx_tool) fn permuted(self) -> bool {
        matches!(
            self,
            Self::BaselinePermutedEquivalent | Self::CanonicalPermutedEquivalent
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::qwen_mlx_tool) enum ToolChoiceVariant {
    None,
    Required,
    Function,
}

impl ToolChoiceVariant {
    pub(in crate::qwen_mlx_tool) fn all() -> [Self; 2] {
        [Self::Required, Self::Function]
    }

    pub(in crate::qwen_mlx_tool) fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Required => "required",
            Self::Function => "function",
        }
    }

    pub(in crate::qwen_mlx_tool) fn request_value(self, case: NormalizedCaseKind) -> Value {
        match self {
            Self::Required => json!("required"),
            Self::Function => {
                json!({"type": "function", "function": {"name": case.tool_function_name()}})
            }
            Self::None => Value::Null,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::qwen_mlx_tool) enum NormalizedProbeSuite {
    FullMatrix,
    FocusedAgenticGate,
    RequiredToolTtftMatrix,
    PrefillSweep135k,
    PrefillSweep135kContextRecall,
    StableAgentPrefix,
    StablePrefixSmoke,
}

impl NormalizedProbeSuite {
    pub(in crate::qwen_mlx_tool) fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "full-matrix" | "full_matrix" => Ok(Self::FullMatrix),
            "focused-agentic-gate" | "focused_agentic_gate" => Ok(Self::FocusedAgenticGate),
            "required-tool-ttft-matrix" | "required_tool_ttft_matrix" => {
                Ok(Self::RequiredToolTtftMatrix)
            }
            "prefill-sweep-135k" | "prefill_sweep_135k" => Ok(Self::PrefillSweep135k),
            "prefill-sweep-135k-context-recall" | "prefill_sweep_135k_context_recall" => {
                Ok(Self::PrefillSweep135kContextRecall)
            }
            "stable-agent-prefix" | "stable_agent_prefix" => Ok(Self::StableAgentPrefix),
            "stable-prefix-smoke" | "stable_prefix_smoke" => Ok(Self::StablePrefixSmoke),
            other => anyhow::bail!(
                "unknown --probe-suite `{other}`; expected full-matrix, focused-agentic-gate, required-tool-ttft-matrix, prefill-sweep-135k, prefill-sweep-135k-context-recall, stable-agent-prefix, or stable-prefix-smoke"
            ),
        }
    }

    pub(in crate::qwen_mlx_tool) fn name(self) -> &'static str {
        match self {
            Self::FullMatrix => "full_matrix",
            Self::FocusedAgenticGate => "focused_agentic_gate",
            Self::RequiredToolTtftMatrix => "required_tool_ttft_matrix",
            Self::PrefillSweep135k => "prefill_sweep_135k",
            Self::PrefillSweep135kContextRecall => "prefill_sweep_135k_context_recall",
            Self::StableAgentPrefix => "stable_agent_prefix",
            Self::StablePrefixSmoke => "stable_prefix_smoke",
        }
    }

    pub(in crate::qwen_mlx_tool) fn probes(self) -> Vec<NormalizedProbePlan> {
        match self {
            Self::FullMatrix => NormalizedProbePlan::all(),
            Self::FocusedAgenticGate => NormalizedProbePlan::focused_agentic_gate(),
            Self::RequiredToolTtftMatrix => NormalizedProbePlan::required_tool_ttft_matrix(),
            Self::PrefillSweep135k => NormalizedProbePlan::prefill_sweep_135k(),
            Self::PrefillSweep135kContextRecall => {
                NormalizedProbePlan::prefill_sweep_135k_context_recall()
            }
            Self::StableAgentPrefix => NormalizedProbePlan::stable_agent_prefix(),
            Self::StablePrefixSmoke => NormalizedProbePlan::stable_prefix_smoke(),
        }
    }

    pub(in crate::qwen_mlx_tool) fn case_names(
        self,
        probes: &[NormalizedProbePlan],
    ) -> Vec<&'static str> {
        match self {
            Self::FullMatrix => NormalizedCaseKind::all()
                .iter()
                .map(|case| case.name())
                .collect(),
            Self::FocusedAgenticGate
            | Self::RequiredToolTtftMatrix
            | Self::PrefillSweep135k
            | Self::PrefillSweep135kContextRecall
            | Self::StableAgentPrefix
            | Self::StablePrefixSmoke => probe_case_names(probes),
        }
    }

    pub(in crate::qwen_mlx_tool) fn schema_variant_names(
        self,
        probes: &[NormalizedProbePlan],
    ) -> Vec<&'static str> {
        match self {
            Self::FullMatrix => SchemaVariant::all()
                .iter()
                .map(|variant| variant.name())
                .collect(),
            Self::FocusedAgenticGate
            | Self::RequiredToolTtftMatrix
            | Self::PrefillSweep135k
            | Self::PrefillSweep135kContextRecall
            | Self::StableAgentPrefix
            | Self::StablePrefixSmoke => probe_schema_variant_names(probes),
        }
    }

    pub(in crate::qwen_mlx_tool) fn tool_choice_variant_names(
        self,
        probes: &[NormalizedProbePlan],
    ) -> Vec<&'static str> {
        match self {
            Self::FullMatrix => ToolChoiceVariant::all()
                .iter()
                .map(|variant| variant.name())
                .collect(),
            Self::FocusedAgenticGate
            | Self::RequiredToolTtftMatrix
            | Self::PrefillSweep135k
            | Self::PrefillSweep135kContextRecall
            | Self::StableAgentPrefix
            | Self::StablePrefixSmoke => probe_tool_choice_variant_names(probes),
        }
    }
}

fn probe_case_names(probes: &[NormalizedProbePlan]) -> Vec<&'static str> {
    unique_probe_names(probes.iter().map(|probe| probe.case.name()))
}

fn probe_schema_variant_names(probes: &[NormalizedProbePlan]) -> Vec<&'static str> {
    unique_probe_names(probes.iter().map(|probe| probe.schema_variant.name()))
}

fn probe_tool_choice_variant_names(probes: &[NormalizedProbePlan]) -> Vec<&'static str> {
    unique_probe_names(probes.iter().map(|probe| probe.tool_choice_variant.name()))
}

fn unique_probe_names(names: impl Iterator<Item = &'static str>) -> Vec<&'static str> {
    let mut unique = Vec::new();
    for name in names {
        if !unique.contains(&name) {
            unique.push(name);
        }
    }
    unique
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::qwen_mlx_tool) struct NormalizedProbePlan {
    pub(in crate::qwen_mlx_tool) case: NormalizedCaseKind,
    pub(in crate::qwen_mlx_tool) schema_variant: SchemaVariant,
    pub(in crate::qwen_mlx_tool) tool_choice_variant: ToolChoiceVariant,
    pub(in crate::qwen_mlx_tool) max_tokens: u32,
}

impl NormalizedProbePlan {
    pub(in crate::qwen_mlx_tool) fn new(
        case: NormalizedCaseKind,
        schema_variant: SchemaVariant,
        tool_choice_variant: ToolChoiceVariant,
    ) -> Self {
        Self {
            case,
            schema_variant,
            tool_choice_variant,
            max_tokens: DEFAULT_MAX_TOKENS,
        }
    }

    pub(in crate::qwen_mlx_tool) fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub(in crate::qwen_mlx_tool) fn all() -> Vec<Self> {
        let mut probes = Vec::new();
        for case in NormalizedCaseKind::all() {
            if case == NormalizedCaseKind::JsonObject {
                probes.push(Self::new(
                    case,
                    SchemaVariant::None,
                    ToolChoiceVariant::None,
                ));
                continue;
            }
            for schema_variant in SchemaVariant::all() {
                for tool_choice_variant in ToolChoiceVariant::all() {
                    probes.push(Self::new(case, schema_variant, tool_choice_variant));
                }
            }
        }
        probes
    }

    pub(in crate::qwen_mlx_tool) fn focused_agentic_gate() -> Vec<Self> {
        vec![
            Self::new(
                NormalizedCaseKind::ToolRequired,
                SchemaVariant::CanonicalCurrent,
                ToolChoiceVariant::Required,
            ),
            Self::new(
                NormalizedCaseKind::ToolRequiredStream,
                SchemaVariant::CanonicalCurrent,
                ToolChoiceVariant::Required,
            ),
            Self::new(
                NormalizedCaseKind::OmpRepeatedPrefix,
                SchemaVariant::CanonicalCurrent,
                ToolChoiceVariant::Required,
            ),
        ]
    }

    pub(in crate::qwen_mlx_tool) fn required_tool_ttft_matrix() -> Vec<Self> {
        let mut probes = Vec::new();
        for schema_variant in SchemaVariant::required_tool_ttft_matrix() {
            for tool_choice_variant in ToolChoiceVariant::all() {
                for max_tokens in REQUIRED_TOOL_TTFT_MAX_TOKENS {
                    probes.push(
                        Self::new(
                            NormalizedCaseKind::ToolRequiredStream,
                            schema_variant,
                            tool_choice_variant,
                        )
                        .with_max_tokens(max_tokens),
                    );
                }
            }
        }
        probes
    }

    pub(in crate::qwen_mlx_tool) fn prefill_sweep_135k() -> Vec<Self> {
        vec![
            Self::new(
                NormalizedCaseKind::ChatStream,
                SchemaVariant::None,
                ToolChoiceVariant::None,
            ),
            Self::new(
                NormalizedCaseKind::ToolRequiredStream,
                SchemaVariant::CanonicalCurrent,
                ToolChoiceVariant::Required,
            ),
            Self::new(
                NormalizedCaseKind::ContextRecallStream135k,
                SchemaVariant::CanonicalCurrent,
                ToolChoiceVariant::Required,
            ),
            Self::new(
                NormalizedCaseKind::WarmPrefixRepeatedTurnStream,
                SchemaVariant::CanonicalCurrent,
                ToolChoiceVariant::Required,
            ),
        ]
    }

    pub(in crate::qwen_mlx_tool) fn prefill_sweep_135k_context_recall() -> Vec<Self> {
        vec![Self::new(
            NormalizedCaseKind::ContextRecallStream135k,
            SchemaVariant::CanonicalCurrent,
            ToolChoiceVariant::Required,
        )]
    }

    pub(in crate::qwen_mlx_tool) fn stable_agent_prefix() -> Vec<Self> {
        vec![
            Self::new(
                NormalizedCaseKind::ChatStream,
                SchemaVariant::None,
                ToolChoiceVariant::None,
            ),
            Self::new(
                NormalizedCaseKind::ToolRequiredStream,
                SchemaVariant::CanonicalCurrent,
                ToolChoiceVariant::Required,
            ),
            Self::new(
                NormalizedCaseKind::WarmPrefixRepeatedTurnStream,
                SchemaVariant::CanonicalCurrent,
                ToolChoiceVariant::Required,
            ),
        ]
    }

    pub(in crate::qwen_mlx_tool) fn stable_prefix_smoke() -> Vec<Self> {
        vec![Self::new(
            NormalizedCaseKind::WarmPrefixRepeatedTurnStream,
            SchemaVariant::CanonicalCurrent,
            ToolChoiceVariant::Required,
        )]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::qwen_mlx_tool) enum CachePhase {
    Cold,
    WarmSamePrompt,
    WarmSameToolSchema,
}

impl CachePhase {
    pub(in crate::qwen_mlx_tool) fn all() -> [Self; 3] {
        [Self::Cold, Self::WarmSamePrompt, Self::WarmSameToolSchema]
    }

    pub(in crate::qwen_mlx_tool) fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "cold" => Ok(Self::Cold),
            "warm_same_prompt" => Ok(Self::WarmSamePrompt),
            "warm_same_tool_schema" => Ok(Self::WarmSameToolSchema),
            other => anyhow::bail!(
                "unknown cache phase `{other}`; expected cold, warm_same_prompt, or warm_same_tool_schema"
            ),
        }
    }

    pub(in crate::qwen_mlx_tool) fn name(self) -> &'static str {
        match self {
            Self::Cold => "cold",
            Self::WarmSamePrompt => "warm_same_prompt",
            Self::WarmSameToolSchema => "warm_same_tool_schema",
        }
    }

    pub(in crate::qwen_mlx_tool) fn warms_before_samples(self) -> bool {
        !matches!(self, Self::Cold)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::qwen_mlx_tool) enum PlannedRunKind {
    Warmup,
    Measured,
}

impl PlannedRunKind {
    pub(in crate::qwen_mlx_tool) fn name(self) -> &'static str {
        match self {
            Self::Warmup => "warmup",
            Self::Measured => "measured",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::qwen_mlx_tool) enum RunMode {
    Sequential,
    Concurrent,
}

impl RunMode {
    pub(in crate::qwen_mlx_tool) fn name(self) -> &'static str {
        match self {
            Self::Sequential => "sequential",
            Self::Concurrent => "concurrent",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(in crate::qwen_mlx_tool) struct PlannedRun {
    pub(in crate::qwen_mlx_tool) phase: CachePhase,
    pub(in crate::qwen_mlx_tool) kind: PlannedRunKind,
    pub(in crate::qwen_mlx_tool) run_mode: RunMode,
    pub(in crate::qwen_mlx_tool) sample_index: Option<usize>,
    pub(in crate::qwen_mlx_tool) request_index: Option<usize>,
    pub(in crate::qwen_mlx_tool) warmup_index: Option<usize>,
}

impl PlannedRun {
    pub(in crate::qwen_mlx_tool) fn prompt(
        self,
        context_tokens: usize,
        case: NormalizedCaseKind,
        tokenizer: Option<&HuggingFaceTokenizer>,
    ) -> anyhow::Result<ProbePrompt> {
        let long_context = if case == NormalizedCaseKind::ContextRecallStream135k {
            tokenizer
                .map(|tokenizer| build_context_recall_prompt(tokenizer, context_tokens))
                .transpose()?
        } else {
            None
        };
        match (self.kind, self.phase) {
            (PlannedRunKind::Warmup, CachePhase::WarmSameToolSchema) => {
                Ok(ProbePrompt::schema_warmup(
                    context_tokens,
                    self.warmup_index.unwrap_or_default(),
                    long_context,
                ))
            }
            _ => Ok(ProbePrompt::measured_with_long_context(
                context_tokens,
                self.sample_index.unwrap_or_default(),
                self.request_index,
                long_context,
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub(in crate::qwen_mlx_tool) struct SampleContext {
    pub(in crate::qwen_mlx_tool) probe: NormalizedProbePlan,
    pub(in crate::qwen_mlx_tool) phase: CachePhase,
    pub(in crate::qwen_mlx_tool) run_mode: RunMode,
    pub(in crate::qwen_mlx_tool) sample_index: usize,
    pub(in crate::qwen_mlx_tool) request_index: Option<usize>,
    pub(in crate::qwen_mlx_tool) planned_prompt_tokens: usize,
    pub(in crate::qwen_mlx_tool) prewarmed: bool,
    pub(in crate::qwen_mlx_tool) expected_probe_id: String,
    pub(in crate::qwen_mlx_tool) expected_marker: Option<String>,
}

#[derive(Debug, Clone)]
pub(in crate::qwen_mlx_tool) struct ProbePrompt {
    variant: ProbePromptVariant,
    context_tokens: usize,
    sample_index: usize,
    request_index: Option<usize>,
    long_context: Option<LongContextPrompt>,
}

impl ProbePrompt {
    #[cfg(test)]
    pub(in crate::qwen_mlx_tool) fn measured(
        context_tokens: usize,
        sample_index: usize,
        request_index: Option<usize>,
    ) -> Self {
        Self::measured_with_long_context(context_tokens, sample_index, request_index, None)
    }

    fn measured_with_long_context(
        context_tokens: usize,
        sample_index: usize,
        request_index: Option<usize>,
        long_context: Option<LongContextPrompt>,
    ) -> Self {
        Self {
            variant: ProbePromptVariant::Measured,
            context_tokens,
            sample_index,
            request_index,
            long_context,
        }
    }

    fn schema_warmup(
        context_tokens: usize,
        index: usize,
        long_context: Option<LongContextPrompt>,
    ) -> Self {
        Self {
            variant: ProbePromptVariant::SchemaWarmup(index),
            context_tokens,
            sample_index: 0,
            request_index: None,
            long_context,
        }
    }

    pub(in crate::qwen_mlx_tool) fn planned_prompt_tokens(&self) -> usize {
        self.long_context
            .as_ref()
            .map(|prompt| prompt.token_count)
            .unwrap_or(self.context_tokens)
    }

    pub(in crate::qwen_mlx_tool) fn probe_id(&self, case: NormalizedCaseKind) -> String {
        match self.variant {
            ProbePromptVariant::Measured => case.probe_id().to_owned(),
            ProbePromptVariant::SchemaWarmup(index) => {
                format!("{}_SCHEMA_WARMUP_{index}", case.probe_id())
            }
        }
    }

    pub(in crate::qwen_mlx_tool) fn expected_marker(
        &self,
        case: NormalizedCaseKind,
    ) -> Option<String> {
        match case {
            NormalizedCaseKind::ChatStream => Some(CHAT_STREAM_MARKER.to_owned()),
            NormalizedCaseKind::ContextRecallStream135k => Some(
                self.long_context
                    .as_ref()
                    .map(|prompt| prompt.marker.as_str())
                    .unwrap_or(CONTEXT_RECALL_STREAM_135K_MARKER)
                    .to_owned(),
            ),
            _ => None,
        }
    }

    pub(in crate::qwen_mlx_tool) fn user_prompt(&self, case: NormalizedCaseKind) -> String {
        let probe_id = self.probe_id(case);
        let prefix = stable_context_prefix(self.context_tokens, case);
        match case {
            NormalizedCaseKind::ToolRequired | NormalizedCaseKind::ToolRequiredStream => {
                format!(
                    "{prefix}\nCall record_qwen_tool_probe with probe_id `{probe_id}` and case `{}`.",
                    case.name()
                )
            }
            NormalizedCaseKind::JsonObject => {
                format!(
                    "{prefix}\nReturn exactly this JSON shape with probe_id `{probe_id}` and case `{}`: {{\"probe_id\":\"...\",\"case\":\"...\"}}",
                    case.name()
                )
            }
            NormalizedCaseKind::ChatStream => {
                format!(
                    "{prefix}\nReturn exactly this marker in assistant content and no tool call: {CHAT_STREAM_MARKER}"
                )
            }
            NormalizedCaseKind::ContextRecallStream135k => {
                let body = self
                    .long_context
                    .as_ref()
                    .map(|prompt| prompt.body.as_str())
                    .map(str::to_owned)
                    .unwrap_or_else(|| approximate_context_recall_prompt(self.context_tokens));
                format!("{body}\nCall report_long_context_recall with marker, profile, and case.")
            }
            NormalizedCaseKind::OmpRepeatedPrefix => {
                let request = self
                    .request_index
                    .map(|index| index.to_string())
                    .unwrap_or_else(|| "sequential".to_owned());
                format!(
                    "OMP final delta: sample={} request={request}. Call record_qwen_tool_probe with probe_id `{probe_id}` and case `{}`.",
                    self.sample_index,
                    case.name()
                )
            }
            NormalizedCaseKind::WarmPrefixRepeatedTurnStream => {
                let request = self
                    .request_index
                    .map(|index| index.to_string())
                    .unwrap_or_else(|| "sequential".to_owned());
                format!(
                    "Warm-prefix final delta: sample={} request={request}. Call record_qwen_tool_probe with probe_id `{probe_id}` and case `{}`.",
                    self.sample_index,
                    case.name()
                )
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum ProbePromptVariant {
    Measured,
    SchemaWarmup(usize),
}

#[derive(Debug, Clone)]
struct LongContextPrompt {
    marker: String,
    body: String,
    token_count: usize,
}

fn build_context_recall_prompt(
    tokenizer: &HuggingFaceTokenizer,
    target_tokens: usize,
) -> anyhow::Result<LongContextPrompt> {
    let marker = CONTEXT_RECALL_STREAM_135K_MARKER.to_owned();
    let mut body = context_recall_prompt_header(&marker);
    let footer = "\nEnd of benchmark context. Use the target_marker value from the first section when calling the tool.\n";
    let row_template = "Context row 000000: MLX scheduler counters, prefill chunk sizes, cache namespace fields, parser states, and trace identifiers. This row is distractor material only.\n";
    let row_tokens = tokenizer.encode(row_template, false)?.len().max(1);
    let base_tokens = tokenizer.encode(&(body.clone() + footer), false)?.len();
    let estimated_rows = target_tokens
        .saturating_sub(base_tokens)
        .div_ceil(row_tokens)
        .saturating_add(8);
    for row in 0..estimated_rows {
        body.push_str(&format!(
            "Context row {row:06}: MLX scheduler counters, prefill chunk sizes, cache namespace fields, parser states, and trace identifiers. This row is distractor material only.\n"
        ));
    }
    body.push_str(footer);
    let mut token_count = tokenizer.encode(&body, false)?.len();
    while token_count < target_tokens {
        let row = token_count;
        body.push_str(&format!(
            "Context extension {row:06}: additional non-target diagnostics for Qwen MLX prefill pressure.\n"
        ));
        token_count = tokenizer.encode(&body, false)?.len();
    }
    Ok(LongContextPrompt {
        marker,
        body,
        token_count,
    })
}

fn approximate_context_recall_prompt(context_tokens: usize) -> String {
    let marker = CONTEXT_RECALL_STREAM_135K_MARKER;
    let mut body = context_recall_prompt_header(marker);
    body.push_str(&stable_context_prefix(
        context_tokens,
        NormalizedCaseKind::ContextRecallStream135k,
    ));
    body.push_str(
        "\nEnd of benchmark context. Use the target_marker value from the first section when calling the tool.",
    );
    body
}

fn context_recall_prompt_header(marker: &str) -> String {
    format!(
        "\
Long-context benchmark profile: {PREFILL_SWEEP_135K_PROFILE_NAME}
Scenario: context_recall_stream_135k
Target marker name: target_marker
Target marker value: {marker}

Only the marker value above is correct. Later context rows are distractors and must not replace it.

"
    )
}

pub(in crate::qwen_mlx_tool) fn stable_context_prefix(
    context_tokens: usize,
    case: NormalizedCaseKind,
) -> String {
    let mut body = format!(
        "\
Qwen MLX-LM tool sweep long-context payload.
Declared context token target: {context_tokens}
Case: {case_name}
This shared prefix is stable across measured requests for cache and prefill pressure.
For OMP repeated-prefix probes, only the final user delta changes after the shared history.

",
        case_name = case.name()
    );
    let estimated_tokens_per_row = 32usize;
    let fixed_token_estimate = 80usize;
    let rows = context_tokens
        .saturating_sub(fixed_token_estimate)
        .div_ceil(estimated_tokens_per_row);
    for row in 0..rows {
        body.push_str(&format!(
            "Stable context row {row:06}: scheduler trace fields, tool schemas, repository paths, prompt-cache keys, decode counters, and parser states are distractor material.\n"
        ));
    }
    body
}
