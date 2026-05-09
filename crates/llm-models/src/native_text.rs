use crate::{GemmaModelSpec, ModelFamily, ModelSpecError, QwenModelSpec, SafetensorsIndex};

#[derive(Debug, Clone, PartialEq)]
pub enum NativeTextModelSpec {
    Qwen(QwenModelSpec),
    Gemma(GemmaModelSpec),
}

impl NativeTextModelSpec {
    pub fn from_config_json(family: ModelFamily, json: &str) -> Result<Self, ModelSpecError> {
        match family {
            ModelFamily::Qwen => Ok(Self::Qwen(QwenModelSpec::from_config_json(json)?)),
            ModelFamily::Gemma => Ok(Self::Gemma(GemmaModelSpec::from_config_json(json)?)),
            family => Err(ModelSpecError::unsupported(format!(
                "native text execution for family `{}` is deferred until Qwen production parity",
                family.canonical_slug()
            ))),
        }
    }

    pub fn family(&self) -> ModelFamily {
        match self {
            Self::Qwen(spec) => spec.family,
            Self::Gemma(spec) => spec.family,
        }
    }

    pub fn max_position_embeddings(&self) -> u32 {
        match self {
            Self::Qwen(spec) => spec.max_position_embeddings,
            Self::Gemma(spec) => spec.max_position_embeddings,
        }
    }

    pub fn num_hidden_layers(&self) -> u32 {
        match self {
            Self::Qwen(spec) => spec.num_hidden_layers,
            Self::Gemma(spec) => spec.num_hidden_layers,
        }
    }

    pub fn hidden_size(&self) -> u32 {
        match self {
            Self::Qwen(spec) => spec.hidden_size,
            Self::Gemma(spec) => spec.hidden_size,
        }
    }

    pub fn validate_text_weights(&self, index: &SafetensorsIndex) -> Result<(), ModelSpecError> {
        match self {
            Self::Qwen(spec) => index.validate_qwen_text_weights(spec),
            Self::Gemma(spec) => index.validate_gemma4_text_weights(spec),
        }
    }

    pub fn as_qwen(&self) -> Option<&QwenModelSpec> {
        match self {
            Self::Qwen(spec) => Some(spec),
            Self::Gemma(_) => None,
        }
    }

    pub fn as_gemma(&self) -> Option<&GemmaModelSpec> {
        match self {
            Self::Gemma(spec) => Some(spec),
            _ => None,
        }
    }

    pub fn is_qwen3_dense(&self) -> bool {
        match self {
            Self::Qwen(spec) => spec.is_qwen3_dense(),
            Self::Gemma(_) => false,
        }
    }
}
