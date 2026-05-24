use super::{CachePhase, NormalizedProbeSuite};
use crate::{DEFAULT_MODEL_ID, MlxToolParserMode};
use anyhow::Context;
use serde::{Serialize, Serializer};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

mod cli;
mod profiles;

#[cfg(test)]
pub(super) use cli::parse_lane_spec;
pub(super) use cli::{
    default_run_config_for_probe_suite, parse_cache_phases_flag, parse_count_flag,
    parse_lane_specs, parse_millis_flag, parse_optional_count_flag, parse_probe_suite_flag,
    parse_sweep_profile_flag,
};

pub(super) const DEFAULT_WARMUPS: usize = 1;
pub(super) const DEFAULT_SAMPLES: usize = 1;
pub(super) const DEFAULT_CONTEXT_TOKENS: usize = 135_000;
pub(super) const DEFAULT_CONCURRENT_REQUESTS: usize = 1;
pub(super) const DEFAULT_CONCURRENT_SAMPLES: usize = 0;

const QWEN_MLX_CACHE_PREFILL_PROFILE: &str = "qwen-mlx-cache-prefill";
pub(super) const QWEN_MLX_PREFILL_135K_PROFILE: &str = "qwen-mlx-prefill-135k";
const QWEN_MLX_PREFILL_135K_EXPERIMENTAL_PROFILE: &str = "qwen-mlx-prefill-135k-experimental";
const QWEN_MLX_STABLE_PREFIX_PROFILE: &str = "qwen-mlx-stable-prefix";

#[derive(Debug, Clone)]
pub(super) struct NormalizedRunConfig {
    pub(super) warmups: usize,
    pub(super) samples: usize,
    pub(super) context_tokens: usize,
    pub(super) concurrent_requests: usize,
    pub(super) concurrent_samples: usize,
    pub(super) effective_concurrent_samples: usize,
    pub(super) cache_phases: Vec<CachePhase>,
}

impl NormalizedRunConfig {
    pub(super) fn new(
        warmups: usize,
        samples: usize,
        context_tokens: usize,
        concurrent_requests: usize,
        concurrent_samples: usize,
    ) -> Self {
        Self {
            warmups,
            samples,
            context_tokens,
            concurrent_requests,
            concurrent_samples,
            effective_concurrent_samples: effective_concurrent_samples(
                concurrent_requests,
                samples,
                concurrent_samples,
            ),
            cache_phases: CachePhase::all().to_vec(),
        }
    }

    pub(super) fn with_cache_phases(mut self, cache_phases: Vec<CachePhase>) -> Self {
        self.cache_phases = cache_phases;
        self
    }
}
pub(super) fn sweep_profile_requires_exact_token_prompt(
    profile: Option<NormalizedSweepProfile>,
) -> bool {
    matches!(
        profile,
        Some(
            NormalizedSweepProfile::QwenMlxPrefill135k
                | NormalizedSweepProfile::QwenMlxPrefill135kExperimental
        )
    )
}
pub(super) fn effective_concurrent_samples(
    concurrent_requests: usize,
    samples: usize,
    concurrent_samples: usize,
) -> usize {
    if concurrent_samples > 0 {
        concurrent_samples
    } else if concurrent_requests > 1 {
        samples
    } else {
        0
    }
}
#[derive(Debug, Clone)]
pub(super) struct NormalizedLaneConfig {
    pub(super) name: String,
    pub(super) endpoint: String,
    pub(super) declared_model_id: String,
    pub(super) launched_model_id: Option<String>,
    pub(super) snapshot_path: Option<PathBuf>,
    pub(super) kind: NormalizedLaneKind,
    pub(super) model_addressing: NormalizedModelAddressing,
    pub(super) template: NormalizedTemplatePolicy,
    pub(super) tool_parser: MlxToolParserMode,
    pub(super) mlx_lm_settings: MlxLmSettings,
    pub(super) experimental: bool,
}

impl NormalizedLaneConfig {
    pub(super) fn effective_request_model_id(&self) -> &str {
        match self.model_addressing {
            NormalizedModelAddressing::LoadedModelId | NormalizedModelAddressing::Custom => {
                &self.declared_model_id
            }
            NormalizedModelAddressing::DefaultModel => DEFAULT_MODEL_ID,
            NormalizedModelAddressing::ServerDefault => self
                .launched_model_id
                .as_deref()
                .or_else(|| self.snapshot_path.as_deref().and_then(Path::to_str))
                .unwrap_or(&self.declared_model_id),
        }
    }

    pub(super) fn request_model_id(&self) -> Option<&str> {
        match self.model_addressing {
            NormalizedModelAddressing::ServerDefault => None,
            NormalizedModelAddressing::LoadedModelId
            | NormalizedModelAddressing::DefaultModel
            | NormalizedModelAddressing::Custom => Some(self.effective_request_model_id()),
        }
    }

    pub(super) fn identity_model_id(&self) -> String {
        self.launched_model_id
            .clone()
            .or_else(|| {
                self.snapshot_path
                    .as_ref()
                    .map(|path| path.display().to_string())
            })
            .unwrap_or_else(|| self.effective_request_model_id().to_owned())
    }

    pub(super) fn model_identity_source(&self) -> &'static str {
        if self.launched_model_id.is_some() {
            "lane_launched_model_id"
        } else if self.snapshot_path.is_some() {
            "lane_snapshot_path"
        } else {
            "effective_request_model_id"
        }
    }

    pub(super) fn thinking_policy_report(&self) -> Value {
        self.template.thinking_policy_report()
    }

    pub(super) fn tool_parser_report(&self) -> Option<&'static str> {
        (self.tool_parser != MlxToolParserMode::Auto).then(|| self.tool_parser.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NormalizedSweepProfile {
    QwenMlxCachePrefill,
    QwenMlxPrefill135k,
    QwenMlxPrefill135kExperimental,
    QwenMlxStablePrefix,
}

impl NormalizedSweepProfile {
    pub(super) fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            QWEN_MLX_CACHE_PREFILL_PROFILE => Ok(Self::QwenMlxCachePrefill),
            QWEN_MLX_PREFILL_135K_PROFILE => Ok(Self::QwenMlxPrefill135k),
            QWEN_MLX_PREFILL_135K_EXPERIMENTAL_PROFILE => Ok(Self::QwenMlxPrefill135kExperimental),
            QWEN_MLX_STABLE_PREFIX_PROFILE => Ok(Self::QwenMlxStablePrefix),
            other => anyhow::bail!(
                "unknown --sweep-profile `{other}`; expected {QWEN_MLX_CACHE_PREFILL_PROFILE}, {QWEN_MLX_PREFILL_135K_PROFILE}, {QWEN_MLX_PREFILL_135K_EXPERIMENTAL_PROFILE}, or {QWEN_MLX_STABLE_PREFIX_PROFILE}"
            ),
        }
    }

    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::QwenMlxCachePrefill => QWEN_MLX_CACHE_PREFILL_PROFILE,
            Self::QwenMlxPrefill135k => QWEN_MLX_PREFILL_135K_PROFILE,
            Self::QwenMlxPrefill135kExperimental => QWEN_MLX_PREFILL_135K_EXPERIMENTAL_PROFILE,
            Self::QwenMlxStablePrefix => QWEN_MLX_STABLE_PREFIX_PROFILE,
        }
    }

    pub(super) fn default_probe_suite(self) -> NormalizedProbeSuite {
        match self {
            Self::QwenMlxCachePrefill => NormalizedProbeSuite::FullMatrix,
            Self::QwenMlxPrefill135k => NormalizedProbeSuite::PrefillSweep135k,
            Self::QwenMlxPrefill135kExperimental => {
                NormalizedProbeSuite::PrefillSweep135kContextRecall
            }
            Self::QwenMlxStablePrefix => NormalizedProbeSuite::StableAgentPrefix,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(super) struct MlxLmSettings {
    #[serde(rename = "mlx_prompt_cache_size")]
    pub(super) prompt_cache_size: DefaultOrU64,
    #[serde(rename = "mlx_prompt_cache_bytes")]
    pub(super) prompt_cache_bytes: UnsetOrU64,
    #[serde(rename = "mlx_prefill_step_size")]
    pub(super) prefill_step_size: DefaultOrU64,
    #[serde(rename = "mlx_prompt_concurrency")]
    pub(super) prompt_concurrency: DefaultOrU32,
    #[serde(rename = "mlx_decode_concurrency")]
    pub(super) decode_concurrency: DefaultOrU32,
}

impl MlxLmSettings {
    pub(super) fn parse(values: &mut BTreeMap<String, String>) -> anyhow::Result<Self> {
        Ok(Self {
            prompt_cache_size: parse_default_or_u64(values, "mlx_prompt_cache_size")?,
            prompt_cache_bytes: parse_unset_or_u64(values, "mlx_prompt_cache_bytes")?,
            prefill_step_size: parse_default_or_u64(values, "mlx_prefill_step_size")?,
            prompt_concurrency: parse_default_or_u32(values, "mlx_prompt_concurrency")?,
            decode_concurrency: parse_default_or_u32(values, "mlx_decode_concurrency")?,
        })
    }
}

impl Default for MlxLmSettings {
    fn default() -> Self {
        Self {
            prompt_cache_size: DefaultOrU64::Default,
            prompt_cache_bytes: UnsetOrU64::Unset,
            prefill_step_size: DefaultOrU64::Default,
            prompt_concurrency: DefaultOrU32::Default,
            decode_concurrency: DefaultOrU32::Default,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DefaultOrU64 {
    Default,
    Value(u64),
}

impl Serialize for DefaultOrU64 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Default => serializer.serialize_str("default"),
            Self::Value(value) => serializer.serialize_u64(*value),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum UnsetOrU64 {
    Unset,
    Value(u64),
}

impl Serialize for UnsetOrU64 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Unset => serializer.serialize_str("unset"),
            Self::Value(value) => serializer.serialize_u64(*value),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DefaultOrU32 {
    Default,
    Value(u32),
}

impl Serialize for DefaultOrU32 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Default => serializer.serialize_str("default"),
            Self::Value(value) => serializer.serialize_u32(*value),
        }
    }
}

fn parse_default_or_u64(
    values: &mut BTreeMap<String, String>,
    key: &'static str,
) -> anyhow::Result<DefaultOrU64> {
    let Some(value) = values.remove(key) else {
        return Ok(DefaultOrU64::Default);
    };
    if value == "default" {
        return Ok(DefaultOrU64::Default);
    }
    parse_positive_u64(key, &value).map(DefaultOrU64::Value)
}

fn parse_unset_or_u64(
    values: &mut BTreeMap<String, String>,
    key: &'static str,
) -> anyhow::Result<UnsetOrU64> {
    let Some(value) = values.remove(key) else {
        return Ok(UnsetOrU64::Unset);
    };
    if value == "unset" {
        return Ok(UnsetOrU64::Unset);
    }
    parse_positive_u64(key, &value).map(UnsetOrU64::Value)
}

fn parse_default_or_u32(
    values: &mut BTreeMap<String, String>,
    key: &'static str,
) -> anyhow::Result<DefaultOrU32> {
    let Some(value) = values.remove(key) else {
        return Ok(DefaultOrU32::Default);
    };
    if value == "default" {
        return Ok(DefaultOrU32::Default);
    }
    parse_positive_u32(key, &value).map(DefaultOrU32::Value)
}

fn parse_positive_u64(key: &str, value: &str) -> anyhow::Result<u64> {
    let parsed = value
        .parse::<u64>()
        .with_context(|| format!("parse {key}; expected default/unset or a positive integer"))?;
    if parsed == 0 {
        anyhow::bail!("{key} must be greater than zero");
    }
    Ok(parsed)
}

fn parse_positive_u32(key: &str, value: &str) -> anyhow::Result<u32> {
    let parsed = value
        .parse::<u32>()
        .with_context(|| format!("parse {key}; expected default or a positive integer"))?;
    if parsed == 0 {
        anyhow::bail!("{key} must be greater than zero");
    }
    Ok(parsed)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NormalizedLaneKind {
    DirectMlx,
    KirAiProxy,
    Other,
}

impl NormalizedLaneKind {
    pub(super) fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "direct_mlx" => Ok(Self::DirectMlx),
            "kir_ai_proxy" => Ok(Self::KirAiProxy),
            "other" => Ok(Self::Other),
            other => anyhow::bail!(
                "unknown lane kind `{other}`; expected direct_mlx, kir_ai_proxy, or other"
            ),
        }
    }

    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::DirectMlx => "direct_mlx",
            Self::KirAiProxy => "kir_ai_proxy",
            Self::Other => "other",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NormalizedModelAddressing {
    LoadedModelId,
    DefaultModel,
    ServerDefault,
    Custom,
}

impl NormalizedModelAddressing {
    pub(super) fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "loaded_model_id" => Ok(Self::LoadedModelId),
            "default_model" => Ok(Self::DefaultModel),
            "server_default" => Ok(Self::ServerDefault),
            "custom" => Ok(Self::Custom),
            other => anyhow::bail!(
                "unknown model_addressing `{other}`; expected loaded_model_id, default_model, server_default, or custom"
            ),
        }
    }

    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::LoadedModelId => "loaded_model_id",
            Self::DefaultModel => "default_model",
            Self::ServerDefault => "server_default",
            Self::Custom => "custom",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NormalizedTemplatePolicy {
    QwenNoThinking,
    SidecarChatTemplateArgs,
    None,
}

impl NormalizedTemplatePolicy {
    pub(super) fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "qwen-no-thinking" => Ok(Self::QwenNoThinking),
            "sidecar-chat-template-args" => Ok(Self::SidecarChatTemplateArgs),
            "none" => Ok(Self::None),
            other => anyhow::bail!(
                "unknown template `{other}`; expected qwen-no-thinking, sidecar-chat-template-args, or none"
            ),
        }
    }

    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::QwenNoThinking => "qwen-no-thinking",
            Self::SidecarChatTemplateArgs => "sidecar-chat-template-args",
            Self::None => "none",
        }
    }

    pub(super) fn apply_request_kwargs(self, body: &mut Value) {
        if matches!(self, Self::QwenNoThinking) {
            body["chat_template_kwargs"] = json!({"enable_thinking": false});
        }
    }

    pub(super) fn thinking_policy_report(self) -> Value {
        match self {
            Self::QwenNoThinking => json!({
                "template": self.as_str(),
                "enable_thinking": false,
                "source": "request_chat_template_kwargs",
                "request_chat_template_kwargs": {"enable_thinking": false}
            }),
            Self::SidecarChatTemplateArgs => json!({
                "template": self.as_str(),
                "enable_thinking": false,
                "source": "sidecar_chat_template_args_declared_by_lane"
            }),
            Self::None => json!({
                "template": self.as_str(),
                "enable_thinking": Value::Null,
                "source": "not_configured"
            }),
        }
    }
}
