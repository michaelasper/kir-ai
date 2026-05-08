use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetensorsIndex {
    pub total_size_bytes: u64,
    weight_map: BTreeMap<String, String>,
}

impl SafetensorsIndex {
    pub fn from_json(json: &str) -> Result<Self, ModelSpecError> {
        let raw: RawSafetensorsIndex = serde_json::from_str(json)
            .map_err(|err| ModelSpecError::invalid_request(format!("invalid index JSON: {err}")))?;
        let total_size_bytes = raw.metadata.total_size.round() as u64;
        Ok(Self {
            total_size_bytes,
            weight_map: raw.weight_map,
        })
    }

    pub fn tensor_count(&self) -> usize {
        self.weight_map.len()
    }

    pub fn shard_count(&self) -> usize {
        self.weight_map.values().collect::<BTreeSet<_>>().len()
    }

    pub fn contains(&self, tensor: &str) -> bool {
        self.weight_map.contains_key(tensor)
    }

    pub fn validate_qwen_text_weights(&self, spec: &QwenModelSpec) -> Result<(), ModelSpecError> {
        self.require("model.language_model.embed_tokens.weight")?;
        self.require("model.language_model.norm.weight")?;
        self.require("lm_head.weight")?;
        for (layer, kind) in spec.layer_kinds.iter().enumerate() {
            let prefix = format!("model.language_model.layers.{layer}");
            self.require(format!("{prefix}.input_layernorm.weight"))?;
            self.require(format!("{prefix}.post_attention_layernorm.weight"))?;
            self.require(format!("{prefix}.mlp.gate.weight"))?;
            self.require(format!("{prefix}.mlp.experts.down_proj"))?;
            self.require(format!("{prefix}.mlp.experts.gate_up_proj"))?;
            self.require(format!("{prefix}.mlp.shared_expert.down_proj.weight"))?;
            self.require(format!("{prefix}.mlp.shared_expert.gate_proj.weight"))?;
            self.require(format!("{prefix}.mlp.shared_expert.up_proj.weight"))?;
            self.require(format!("{prefix}.mlp.shared_expert_gate.weight"))?;
            match kind {
                AttentionKind::LinearAttention => {
                    self.require(format!("{prefix}.linear_attn.in_proj_qkv.weight"))?;
                    self.require(format!("{prefix}.linear_attn.in_proj_z.weight"))?;
                    self.require(format!("{prefix}.linear_attn.out_proj.weight"))?;
                    self.require(format!("{prefix}.linear_attn.in_proj_a.weight"))?;
                    self.require(format!("{prefix}.linear_attn.in_proj_b.weight"))?;
                    self.require(format!("{prefix}.linear_attn.dt_bias"))?;
                    self.require(format!("{prefix}.linear_attn.A_log"))?;
                    self.require(format!("{prefix}.linear_attn.conv1d.weight"))?;
                    self.require(format!("{prefix}.linear_attn.norm.weight"))?;
                }
                AttentionKind::FullAttention => {
                    self.require(format!("{prefix}.self_attn.q_proj.weight"))?;
                    self.require(format!("{prefix}.self_attn.k_proj.weight"))?;
                    self.require(format!("{prefix}.self_attn.v_proj.weight"))?;
                    self.require(format!("{prefix}.self_attn.o_proj.weight"))?;
                    self.require(format!("{prefix}.self_attn.q_norm.weight"))?;
                    self.require(format!("{prefix}.self_attn.k_norm.weight"))?;
                }
            }
        }
        Ok(())
    }

    fn require(&self, tensor: impl AsRef<str>) -> Result<(), ModelSpecError> {
        let tensor = tensor.as_ref();
        if self.contains(tensor) {
            Ok(())
        } else {
            Err(ModelSpecError::invalid_request(format!(
                "safetensors index missing required tensor `{tensor}`"
            )))
        }
    }
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

#[derive(Debug, Deserialize)]
struct RawSafetensorsIndex {
    metadata: RawSafetensorsMetadata,
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct RawSafetensorsMetadata {
    total_size: f64,
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
