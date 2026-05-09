use serde::{Deserialize, Serialize};
use std::{fmt, str::FromStr};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFamily {
    Qwen,
    DeepSeek,
    Gemma,
}

impl ModelFamily {
    pub fn parse(value: &str) -> Result<Self, ModelFamilyParseError> {
        Self::parse_slug(value)
    }

    pub fn parse_slug(value: &str) -> Result<Self, ModelFamilyParseError> {
        match value {
            "qwen" => Ok(Self::Qwen),
            "deep_seek" | "deepseek" => Ok(Self::DeepSeek),
            "gemma" => Ok(Self::Gemma),
            other => Err(ModelFamilyParseError {
                value: other.to_owned(),
            }),
        }
    }

    pub fn canonical_slug(self) -> &'static str {
        match self {
            Self::Qwen => "qwen",
            Self::DeepSeek => "deep_seek",
            Self::Gemma => "gemma",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::Qwen => "Qwen",
            Self::DeepSeek => "DeepSeek",
            Self::Gemma => "Gemma",
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

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("unsupported model family `{value}`; expected `qwen`, `deep_seek`, or `gemma`")]
pub struct ModelFamilyParseError {
    value: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendKind {
    #[serde(rename = "native-metal", alias = "native_metal")]
    NativeMetal,
    #[serde(rename = "mlx")]
    Mlx,
}

impl BackendKind {
    pub fn parse(value: &str) -> Result<Self, BackendKindParseError> {
        Self::parse_slug(value)
    }

    pub fn parse_slug(value: &str) -> Result<Self, BackendKindParseError> {
        match value {
            "native-metal" | "native_metal" => Ok(Self::NativeMetal),
            "mlx" => Ok(Self::Mlx),
            other => Err(BackendKindParseError {
                value: other.to_owned(),
            }),
        }
    }

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

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("unsupported backend loader `{value}`; expected `native-metal` or `mlx`")]
pub struct BackendKindParseError {
    value: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromotionStage {
    Production,
    DeferredUntilQwenParity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FamilyCapabilityFlags {
    pub text: bool,
    pub reasoning: bool,
    pub tool_calls: bool,
    pub dsml_tools: bool,
    pub raw_completion: bool,
    pub reasoning_channels: bool,
    pub multimodal_artifacts: bool,
    pub backend_execution: bool,
}

pub trait ModelFamilyAdapter: Send + Sync {
    fn family(&self) -> ModelFamily;
    fn production_backends(&self) -> &'static [BackendKind];
    fn cache_template_id(&self) -> &'static str;
    fn tensor_namespace(&self) -> &'static str;
    fn capabilities(&self) -> FamilyCapabilityFlags;
    fn promotion_stage(&self) -> PromotionStage;
}

impl ModelFamily {
    pub fn adapter(self) -> &'static dyn ModelFamilyAdapter {
        match self {
            Self::Qwen => &QWEN_FAMILY_ADAPTER,
            Self::DeepSeek => &DEEPSEEK_FAMILY_ADAPTER,
            Self::Gemma => &GEMMA_FAMILY_ADAPTER,
        }
    }
}

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
            backend_execution: true,
        }
    }

    fn promotion_stage(&self) -> PromotionStage {
        PromotionStage::Production
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GemmaFamilyAdapter;

static GEMMA_FAMILY_ADAPTER: GemmaFamilyAdapter = GemmaFamilyAdapter;

impl ModelFamilyAdapter for GemmaFamilyAdapter {
    fn family(&self) -> ModelFamily {
        ModelFamily::Gemma
    }

    fn production_backends(&self) -> &'static [BackendKind] {
        &[BackendKind::Mlx]
    }

    fn cache_template_id(&self) -> &'static str {
        "gemma/text-it/v1"
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
