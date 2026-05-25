use serde::{Deserialize, Serialize};
use std::{fmt, str::FromStr};
use thiserror::Error;

/// Supported model families with stable API/configuration slugs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ModelFamily {
    /// Qwen dense and mixture-of-experts text models.
    Qwen,
    /// DeepSeek text models.
    DeepSeek,
    /// Gemma text models.
    Gemma,
    /// Llama instruction models.
    Llama,
}

impl ModelFamily {
    /// Parses a family from a slug.
    pub fn parse(value: &str) -> Result<Self, ModelFamilyParseError> {
        Self::parse_slug(value)
    }

    /// Parses a family from canonical or compatibility slug spelling.
    pub fn parse_slug(value: &str) -> Result<Self, ModelFamilyParseError> {
        match value {
            "qwen" => Ok(Self::Qwen),
            "deep_seek" | "deepseek" => Ok(Self::DeepSeek),
            "gemma" => Ok(Self::Gemma),
            "llama" => Ok(Self::Llama),
            other => Err(ModelFamilyParseError {
                value: other.to_owned(),
            }),
        }
    }

    /// Returns the canonical slug used in config, cache identity, and logs.
    pub fn canonical_slug(self) -> &'static str {
        match self {
            Self::Qwen => "qwen",
            Self::DeepSeek => "deep_seek",
            Self::Gemma => "gemma",
            Self::Llama => "llama",
        }
    }

    /// Returns a human-readable family name for diagnostics.
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Qwen => "Qwen",
            Self::DeepSeek => "DeepSeek",
            Self::Gemma => "Gemma",
            Self::Llama => "Llama",
        }
    }
}

impl FromStr for ModelFamily {
    type Err = ModelFamilyParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse_slug(value)
    }
}

impl fmt::Display for ModelFamily {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.canonical_slug())
    }
}

/// Error returned when a model family slug is not supported.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("unsupported model family `{value}`; expected `qwen`, `deep_seek`, `gemma`, or `llama`")]
pub struct ModelFamilyParseError {
    value: String,
}

/// Backend implementation families accepted by model profiles and loaders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum BackendKind {
    /// Native Rust/Metal tensor execution backend.
    #[serde(rename = "native-metal", alias = "native_metal")]
    NativeMetal,
    /// MLX Python bridge backend.
    #[serde(rename = "mlx")]
    Mlx,
}

impl BackendKind {
    /// Parses a backend kind from a slug.
    pub fn parse(value: &str) -> Result<Self, BackendKindParseError> {
        Self::parse_slug(value)
    }

    /// Parses a backend kind from canonical or compatibility slug spelling.
    pub fn parse_slug(value: &str) -> Result<Self, BackendKindParseError> {
        match value {
            "native-metal" | "native_metal" => Ok(Self::NativeMetal),
            "mlx" => Ok(Self::Mlx),
            other => Err(BackendKindParseError {
                value: other.to_owned(),
            }),
        }
    }

    /// Returns the canonical backend slug.
    pub fn canonical_slug(self) -> &'static str {
        match self {
            Self::NativeMetal => "native-metal",
            Self::Mlx => "mlx",
        }
    }
}

impl FromStr for BackendKind {
    type Err = BackendKindParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse_slug(value)
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.canonical_slug())
    }
}

/// Error returned when a backend kind slug is not supported.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("unsupported backend loader `{value}`; expected `native-metal` or `mlx`")]
pub struct BackendKindParseError {
    value: String,
}

/// Release readiness stage for a model family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum PromotionStage {
    /// Family is available for normal production use.
    Production,
}

/// Capabilities implied by a model family independent of a concrete backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FamilyCapabilityFlags {
    /// Family can produce text completions.
    pub text: bool,
    /// Family can expose hidden reasoning.
    pub reasoning: bool,
    /// Family has a known tool-call format.
    pub tool_calls: bool,
    /// Family supports DeepSeek DSML tool syntax.
    pub dsml_tools: bool,
    /// Family may be used for raw completion requests.
    pub raw_completion: bool,
    /// Family uses named reasoning channels rather than generic think tags.
    pub reasoning_channels: bool,
    /// Family can emit multimodal output markers.
    pub multimodal_artifacts: bool,
    /// Family has native text Rust/Metal execution support in this workspace.
    ///
    /// This reflects [`NativeTextModelSpec`](crate::NativeTextModelSpec)
    /// support, not external production backends such as MLX.
    pub backend_execution: bool,
}

/// Static adapter metadata for one model family.
///
/// Runtime prompt rendering, cache identity, model admission, and docs should
/// consume this trait instead of scattering family-specific constants.
pub trait ModelFamilyAdapter: Send + Sync {
    /// Family represented by this adapter.
    fn family(&self) -> ModelFamily;
    /// Backends promoted for production use with this family.
    fn production_backends(&self) -> &'static [BackendKind];
    /// Stable prompt/cache template identifier.
    fn cache_template_id(&self) -> &'static str;
    /// Optional JSON object passed as family-specific chat template kwargs.
    fn chat_template_kwargs_json(&self) -> Option<&'static str> {
        None
    }
    /// Safetensors tensor namespace expected by native loaders.
    fn tensor_namespace(&self) -> &'static str;
    /// Family-level behavior flags.
    fn capabilities(&self) -> FamilyCapabilityFlags;
    /// Release readiness stage.
    fn promotion_stage(&self) -> PromotionStage;
}

impl ModelFamily {
    /// Returns static adapter metadata for this family.
    pub fn adapter(self) -> &'static dyn ModelFamilyAdapter {
        match self {
            Self::Qwen => &QWEN_FAMILY_ADAPTER,
            Self::DeepSeek => &DEEPSEEK_FAMILY_ADAPTER,
            Self::Gemma => &GEMMA_FAMILY_ADAPTER,
            Self::Llama => &LLAMA_FAMILY_ADAPTER,
        }
    }

    /// Returns true when the backend is promoted for this family.
    pub fn supports_backend(self, backend: BackendKind) -> bool {
        self.adapter().production_backends().contains(&backend)
    }
}

/// Adapter metadata for Qwen models.
#[derive(Debug, Clone, Copy, Default)]
pub struct QwenFamilyAdapter;

static QWEN_FAMILY_ADAPTER: QwenFamilyAdapter = QwenFamilyAdapter;

impl ModelFamilyAdapter for QwenFamilyAdapter {
    fn family(&self) -> ModelFamily {
        ModelFamily::Qwen
    }

    fn production_backends(&self) -> &'static [BackendKind] {
        &[BackendKind::NativeMetal, BackendKind::Mlx]
    }

    fn cache_template_id(&self) -> &'static str {
        "chatml/qwen/v1"
    }

    fn chat_template_kwargs_json(&self) -> Option<&'static str> {
        Some(r#"{"enable_thinking":false}"#)
    }

    fn tensor_namespace(&self) -> &'static str {
        "qwen"
    }

    fn capabilities(&self) -> FamilyCapabilityFlags {
        FamilyCapabilityFlags {
            text: true,
            reasoning: true,
            tool_calls: true,
            dsml_tools: false,
            raw_completion: true,
            reasoning_channels: false,
            multimodal_artifacts: false,
            backend_execution: true,
        }
    }

    fn promotion_stage(&self) -> PromotionStage {
        PromotionStage::Production
    }
}

/// Adapter metadata for DeepSeek models.
#[derive(Debug, Clone, Copy, Default)]
pub struct DeepSeekFamilyAdapter;

static DEEPSEEK_FAMILY_ADAPTER: DeepSeekFamilyAdapter = DeepSeekFamilyAdapter;

impl ModelFamilyAdapter for DeepSeekFamilyAdapter {
    fn family(&self) -> ModelFamily {
        ModelFamily::DeepSeek
    }

    fn production_backends(&self) -> &'static [BackendKind] {
        &[BackendKind::Mlx]
    }

    fn cache_template_id(&self) -> &'static str {
        "deepseek/chat/v1"
    }

    fn tensor_namespace(&self) -> &'static str {
        "deepseek"
    }

    fn capabilities(&self) -> FamilyCapabilityFlags {
        FamilyCapabilityFlags {
            text: true,
            reasoning: true,
            tool_calls: true,
            dsml_tools: true,
            raw_completion: true,
            reasoning_channels: false,
            multimodal_artifacts: false,
            backend_execution: false,
        }
    }

    fn promotion_stage(&self) -> PromotionStage {
        PromotionStage::Production
    }
}

/// Adapter metadata for Gemma models.
#[derive(Debug, Clone, Copy, Default)]
pub struct GemmaFamilyAdapter;

static GEMMA_FAMILY_ADAPTER: GemmaFamilyAdapter = GemmaFamilyAdapter;

impl ModelFamilyAdapter for GemmaFamilyAdapter {
    fn family(&self) -> ModelFamily {
        ModelFamily::Gemma
    }

    fn production_backends(&self) -> &'static [BackendKind] {
        &[BackendKind::NativeMetal, BackendKind::Mlx]
    }

    fn cache_template_id(&self) -> &'static str {
        "gemma/text-it/v1"
    }

    fn chat_template_kwargs_json(&self) -> Option<&'static str> {
        Some(r#"{"enable_thinking":false}"#)
    }

    fn tensor_namespace(&self) -> &'static str {
        "gemma4_text"
    }

    fn capabilities(&self) -> FamilyCapabilityFlags {
        FamilyCapabilityFlags {
            text: true,
            reasoning: true,
            tool_calls: true,
            dsml_tools: false,
            raw_completion: true,
            reasoning_channels: true,
            multimodal_artifacts: false,
            backend_execution: true,
        }
    }

    fn promotion_stage(&self) -> PromotionStage {
        PromotionStage::Production
    }
}

/// Adapter metadata for Llama models.
#[derive(Debug, Clone, Copy, Default)]
pub struct LlamaFamilyAdapter;

static LLAMA_FAMILY_ADAPTER: LlamaFamilyAdapter = LlamaFamilyAdapter;

impl ModelFamilyAdapter for LlamaFamilyAdapter {
    fn family(&self) -> ModelFamily {
        ModelFamily::Llama
    }

    fn production_backends(&self) -> &'static [BackendKind] {
        &[BackendKind::Mlx]
    }

    fn cache_template_id(&self) -> &'static str {
        "llama3/instruct/v1"
    }

    fn tensor_namespace(&self) -> &'static str {
        "llama"
    }

    fn capabilities(&self) -> FamilyCapabilityFlags {
        FamilyCapabilityFlags {
            text: true,
            reasoning: false,
            tool_calls: true,
            dsml_tools: false,
            raw_completion: true,
            reasoning_channels: false,
            multimodal_artifacts: false,
            backend_execution: false,
        }
    }

    fn promotion_stage(&self) -> PromotionStage {
        PromotionStage::Production
    }
}
