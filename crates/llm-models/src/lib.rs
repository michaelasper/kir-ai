use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFamily {
    Qwen,
    DeepSeek,
    Gemma,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionKind {
    LinearAttention,
    FullAttention,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QwenModelSpec {
    pub family: ModelFamily,
    pub architecture: String,
    pub model_type: String,
    pub text_model_type: String,
    pub hidden_size: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub linear_num_key_heads: u32,
    pub linear_num_value_heads: u32,
    pub linear_key_head_dim: u32,
    pub linear_value_head_dim: u32,
    pub num_experts: u32,
    pub num_experts_per_tok: u32,
    pub moe_intermediate_size: u32,
    pub shared_expert_intermediate_size: u32,
    pub max_position_embeddings: u32,
    pub vocab_size: u32,
    pub layer_kinds: Vec<AttentionKind>,
}

impl QwenModelSpec {
    pub fn from_config_json(json: &str) -> Result<Self, ModelSpecError> {
        let config: RawQwenConfig = serde_json::from_str(json)
            .map_err(|err| ModelSpecError::invalid_request(format!("invalid JSON: {err}")))?;
        let architecture = config
            .architectures
            .first()
            .ok_or_else(|| ModelSpecError::unsupported("model config missing architecture"))?
            .clone();
        if architecture != "Qwen3_5MoeForConditionalGeneration"
            || config.model_type != "qwen3_5_moe"
        {
            return Err(ModelSpecError::unsupported(
                "config is not a supported Qwen3.6/Qwen3.5 MoE architecture",
            ));
        }
        let text = config
            .text_config
            .ok_or_else(|| ModelSpecError::unsupported("qwen config missing text_config"))?;
        let layer_kinds = text
            .layer_types
            .iter()
            .map(|kind| match kind.as_str() {
                "linear_attention" => Ok(AttentionKind::LinearAttention),
                "full_attention" => Ok(AttentionKind::FullAttention),
                other => Err(ModelSpecError::unsupported(format!(
                    "unsupported qwen layer type `{other}`"
                ))),
            })
            .collect::<Result<Vec<_>, _>>()?;
        if layer_kinds.len() != text.num_hidden_layers as usize {
            return Err(ModelSpecError::invalid_request(
                "qwen layer_types length does not match num_hidden_layers",
            ));
        }
        Ok(Self {
            family: ModelFamily::Qwen,
            architecture,
            model_type: config.model_type,
            text_model_type: text.model_type,
            hidden_size: text.hidden_size,
            num_hidden_layers: text.num_hidden_layers,
            num_attention_heads: text.num_attention_heads,
            num_key_value_heads: text.num_key_value_heads,
            head_dim: text.head_dim,
            linear_num_key_heads: text.linear_num_key_heads,
            linear_num_value_heads: text.linear_num_value_heads,
            linear_key_head_dim: text.linear_key_head_dim,
            linear_value_head_dim: text.linear_value_head_dim,
            num_experts: text.num_experts,
            num_experts_per_tok: text.num_experts_per_tok,
            moe_intermediate_size: text.moe_intermediate_size,
            shared_expert_intermediate_size: text.shared_expert_intermediate_size,
            max_position_embeddings: text.max_position_embeddings,
            vocab_size: text.vocab_size,
            layer_kinds,
        })
    }
}

#[derive(Debug, Deserialize)]
struct RawQwenConfig {
    architectures: Vec<String>,
    model_type: String,
    text_config: Option<RawQwenTextConfig>,
}

#[derive(Debug, Deserialize)]
struct RawQwenTextConfig {
    model_type: String,
    hidden_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    linear_num_key_heads: u32,
    linear_num_value_heads: u32,
    linear_key_head_dim: u32,
    linear_value_head_dim: u32,
    num_experts: u32,
    num_experts_per_tok: u32,
    moe_intermediate_size: u32,
    shared_expert_intermediate_size: u32,
    max_position_embeddings: u32,
    vocab_size: u32,
    layer_types: Vec<String>,
}

#[derive(Debug, Error)]
#[error("{code}: {message}")]
pub struct ModelSpecError {
    code: &'static str,
    message: String,
}

impl ModelSpecError {
    pub fn code(&self) -> &'static str {
        self.code
    }

    fn unsupported(message: impl Into<String>) -> Self {
        Self {
            code: "unsupported_capability",
            message: message.into(),
        }
    }

    fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            code: "invalid_request",
            message: message.into(),
        }
    }
}
