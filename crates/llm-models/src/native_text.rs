use crate::{
    GemmaModelSpec, ModelFamily, ModelSpec, ModelSpecError, QwenModelSpec, SafetensorsIndex,
};

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
                "native text execution for family `{}` is deferred until native tensor support exists",
                family.canonical_slug()
            ))),
        }
    }

    pub fn from_config_value(
        family: ModelFamily,
        value: serde_json::Value,
    ) -> Result<Self, ModelSpecError> {
        match family {
            ModelFamily::Qwen => Ok(Self::Qwen(QwenModelSpec::from_config_value(value)?)),
            ModelFamily::Gemma => Ok(Self::Gemma(GemmaModelSpec::from_config_value(value)?)),
            family => Err(ModelSpecError::unsupported(format!(
                "native text execution for family `{}` is deferred until native tensor support exists",
                family.canonical_slug()
            ))),
        }
    }

    pub fn infer_from_config_json(json: &str) -> Result<Self, ModelSpecError> {
        let value: serde_json::Value = serde_json::from_str(json)
            .map_err(|err| ModelSpecError::invalid_request(format!("invalid JSON: {err}")))?;
        Self::infer_from_config_value(value)
    }

    pub fn infer_from_config_value(value: serde_json::Value) -> Result<Self, ModelSpecError> {
        let model_type = value
            .get("model_type")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| ModelSpecError::unsupported("native text config missing model_type"))?;
        match model_type.as_str() {
            "qwen3" | "qwen3_5_moe" => Self::from_config_value(ModelFamily::Qwen, value),
            "gemma4" | "gemma4_text" => Self::from_config_value(ModelFamily::Gemma, value),
            other => Err(ModelSpecError::unsupported(format!(
                "native text config model_type `{other}` is not supported for native tensor execution"
            ))),
        }
    }

    pub fn family(&self) -> ModelFamily {
        <Self as ModelSpec>::family(self)
    }

    pub fn max_position_embeddings(&self) -> u32 {
        <Self as ModelSpec>::max_position_embeddings(self)
    }

    pub fn num_hidden_layers(&self) -> u32 {
        <Self as ModelSpec>::num_hidden_layers(self)
    }

    pub fn hidden_size(&self) -> u32 {
        <Self as ModelSpec>::hidden_size(self)
    }

    pub fn validate_text_weights(&self, index: &SafetensorsIndex) -> Result<(), ModelSpecError> {
        <Self as ModelSpec>::validate_text_weights(self, index)
    }

    pub fn is_qwen3_dense(&self) -> bool {
        match self {
            Self::Qwen(spec) => spec.is_qwen3_dense(),
            Self::Gemma(_) => false,
        }
    }
}

impl ModelSpec for NativeTextModelSpec {
    fn family(&self) -> ModelFamily {
        match self {
            Self::Qwen(spec) => spec.family(),
            Self::Gemma(spec) => spec.family(),
        }
    }

    fn architecture(&self) -> &str {
        match self {
            Self::Qwen(spec) => spec.architecture(),
            Self::Gemma(spec) => spec.architecture(),
        }
    }

    fn model_type(&self) -> &str {
        match self {
            Self::Qwen(spec) => spec.model_type(),
            Self::Gemma(spec) => spec.model_type(),
        }
    }

    fn text_model_type(&self) -> &str {
        match self {
            Self::Qwen(spec) => spec.text_model_type(),
            Self::Gemma(spec) => spec.text_model_type(),
        }
    }

    fn max_position_embeddings(&self) -> u32 {
        match self {
            Self::Qwen(spec) => spec.max_position_embeddings(),
            Self::Gemma(spec) => spec.max_position_embeddings(),
        }
    }

    fn num_hidden_layers(&self) -> u32 {
        match self {
            Self::Qwen(spec) => spec.num_hidden_layers(),
            Self::Gemma(spec) => spec.num_hidden_layers(),
        }
    }

    fn hidden_size(&self) -> u32 {
        match self {
            Self::Qwen(spec) => spec.hidden_size(),
            Self::Gemma(spec) => spec.hidden_size(),
        }
    }

    fn vocab_size(&self) -> u32 {
        match self {
            Self::Qwen(spec) => spec.vocab_size(),
            Self::Gemma(spec) => spec.vocab_size(),
        }
    }

    fn validate_text_weights(&self, index: &SafetensorsIndex) -> Result<(), ModelSpecError> {
        match self {
            Self::Qwen(spec) => spec.validate_text_weights(index),
            Self::Gemma(spec) => spec.validate_text_weights(index),
        }
    }
}
