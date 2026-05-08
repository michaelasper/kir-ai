use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFamily {
    Qwen,
    DeepSeek,
    Gemma,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    NativeMetal,
    Mlx,
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

#[derive(Debug, Clone, Copy, Default)]
pub struct QwenFamilyAdapter;

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

impl ModelFamilyAdapter for DeepSeekFamilyAdapter {
    fn family(&self) -> ModelFamily {
        ModelFamily::DeepSeek
    }

    fn production_backends(&self) -> &'static [BackendKind] {
        &[BackendKind::Mlx]
    }

    fn cache_template_id(&self) -> &'static str {
        "chatml/deepseek/v1"
    }

    fn tensor_namespace(&self) -> &'static str {
        "deepseek_v4"
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
        PromotionStage::DeferredUntilQwenParity
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GemmaFamilyAdapter;

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
            backend_execution: false,
        }
    }

    fn promotion_stage(&self) -> PromotionStage {
        PromotionStage::DeferredUntilQwenParity
    }
}
