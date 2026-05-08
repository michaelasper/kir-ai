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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionKind {
    LinearAttention,
    FullAttention,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QwenModelSpec {
    pub family: ModelFamily,
    pub architecture: String,
    pub model_type: String,
    pub text_model_type: String,
    pub hidden_size: u32,
    pub rms_norm_eps: f32,
    pub tie_word_embeddings: bool,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub linear_num_key_heads: u32,
    pub linear_num_value_heads: u32,
    pub linear_key_head_dim: u32,
    pub linear_value_head_dim: u32,
    pub linear_conv_kernel_dim: u32,
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
        for shard_path in raw.weight_map.values() {
            validate_safetensors_shard_path(shard_path)?;
        }
        let total_size_bytes = raw.metadata.total_size.round() as u64;
        Ok(Self {
            total_size_bytes,
            weight_map: raw.weight_map,
        })
    }

    pub fn single_file(
        total_size_bytes: u64,
        shard_path: impl Into<String>,
        tensor_names: impl IntoIterator<Item = String>,
    ) -> Result<Self, ModelSpecError> {
        let shard_path = shard_path.into();
        validate_safetensors_shard_path(&shard_path)?;
        let weight_map = tensor_names
            .into_iter()
            .map(|name| (name, shard_path.clone()))
            .collect::<BTreeMap<_, _>>();
        if weight_map.is_empty() {
            return Err(ModelSpecError::invalid_request(
                "safetensors file does not contain tensors",
            ));
        }
        Ok(Self {
            total_size_bytes,
            weight_map,
        })
    }

    pub fn tensor_count(&self) -> usize {
        self.weight_map.len()
    }

    pub fn shard_count(&self) -> usize {
        self.weight_map.values().collect::<BTreeSet<_>>().len()
    }

    pub fn shard_paths(&self) -> Vec<&str> {
        self.weight_map
            .values()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.weight_map.keys().map(String::as_str)
    }

    pub fn contains(&self, tensor: &str) -> bool {
        self.weight_map.contains_key(tensor)
    }

    pub fn shard_for(&self, tensor: &str) -> Option<&str> {
        self.weight_map.get(tensor).map(String::as_str)
    }

    pub fn validate_qwen_text_weights(&self, spec: &QwenModelSpec) -> Result<(), ModelSpecError> {
        self.require(spec.embed_tokens_weight())?;
        self.require(spec.final_norm_weight())?;
        if !spec.tie_word_embeddings {
            self.require(spec.lm_head_weight())?;
        }
        for (layer, kind) in spec.layer_kinds.iter().enumerate() {
            self.require(spec.layer_tensor(layer, "input_layernorm.weight"))?;
            self.require(spec.layer_tensor(layer, "post_attention_layernorm.weight"))?;
            if spec.is_qwen3_dense() {
                self.require(spec.mlp_tensor(layer, "gate_proj.weight"))?;
                self.require(spec.mlp_tensor(layer, "up_proj.weight"))?;
                self.require(spec.mlp_tensor(layer, "down_proj.weight"))?;
            } else {
                self.require(spec.mlp_tensor(layer, "gate.weight"))?;
                self.require(spec.mlp_tensor(layer, "experts.down_proj"))?;
                self.require(spec.mlp_tensor(layer, "experts.gate_up_proj"))?;
                self.require(spec.mlp_tensor(layer, "shared_expert.down_proj.weight"))?;
                self.require(spec.mlp_tensor(layer, "shared_expert.gate_proj.weight"))?;
                self.require(spec.mlp_tensor(layer, "shared_expert.up_proj.weight"))?;
                self.require(spec.mlp_tensor(layer, "shared_expert_gate.weight"))?;
            }
            match kind {
                AttentionKind::LinearAttention => {
                    self.require(spec.layer_tensor(layer, "linear_attn.in_proj_qkv.weight"))?;
                    self.require(spec.layer_tensor(layer, "linear_attn.in_proj_z.weight"))?;
                    self.require(spec.layer_tensor(layer, "linear_attn.out_proj.weight"))?;
                    self.require(spec.layer_tensor(layer, "linear_attn.in_proj_a.weight"))?;
                    self.require(spec.layer_tensor(layer, "linear_attn.in_proj_b.weight"))?;
                    self.require(spec.layer_tensor(layer, "linear_attn.dt_bias"))?;
                    self.require(spec.layer_tensor(layer, "linear_attn.A_log"))?;
                    self.require(spec.layer_tensor(layer, "linear_attn.conv1d.weight"))?;
                    self.require(spec.layer_tensor(layer, "linear_attn.norm.weight"))?;
                }
                AttentionKind::FullAttention => {
                    self.require(spec.self_attn_tensor(layer, "q_proj.weight"))?;
                    self.require(spec.self_attn_tensor(layer, "k_proj.weight"))?;
                    self.require(spec.self_attn_tensor(layer, "v_proj.weight"))?;
                    self.require(spec.self_attn_tensor(layer, "o_proj.weight"))?;
                    self.require(spec.self_attn_tensor(layer, "q_norm.weight"))?;
                    self.require(spec.self_attn_tensor(layer, "k_norm.weight"))?;
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

fn validate_safetensors_shard_path(path: &str) -> Result<(), ModelSpecError> {
    if path.is_empty()
        || path.starts_with('/')
        || path.contains('\\')
        || path.bytes().any(|byte| byte == 0)
        || path
            .split('/')
            .any(|component| component.is_empty() || component == "." || component == "..")
    {
        return Err(ModelSpecError::invalid_request(format!(
            "unsafe safetensors shard path `{path}`"
        )));
    }
    Ok(())
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
        if architecture == "Qwen3ForCausalLM" && config.model_type == "qwen3" {
            return Self::from_qwen3_dense_config(config, architecture);
        }
        if architecture != "Qwen3_5MoeForConditionalGeneration"
            || config.model_type != "qwen3_5_moe"
        {
            return Err(ModelSpecError::unsupported(
                "config is not a supported Qwen3 dense or Qwen3.6/Qwen3.5 MoE architecture",
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
            rms_norm_eps: text
                .rms_norm_eps
                .ok_or_else(|| ModelSpecError::unsupported("qwen config missing rms_norm_eps"))?,
            tie_word_embeddings: text.tie_word_embeddings.unwrap_or(false),
            rope_theta: text
                .rope_parameters
                .as_ref()
                .ok_or_else(|| ModelSpecError::unsupported("qwen config missing rope_parameters"))?
                .rope_theta,
            partial_rotary_factor: text
                .rope_parameters
                .as_ref()
                .and_then(|rope| rope.partial_rotary_factor)
                .unwrap_or(1.0),
            num_hidden_layers: text.num_hidden_layers,
            num_attention_heads: text.num_attention_heads,
            num_key_value_heads: text.num_key_value_heads,
            head_dim: text.head_dim,
            linear_num_key_heads: text.linear_num_key_heads,
            linear_num_value_heads: text.linear_num_value_heads,
            linear_key_head_dim: text.linear_key_head_dim,
            linear_value_head_dim: text.linear_value_head_dim,
            linear_conv_kernel_dim: text.linear_conv_kernel_dim,
            num_experts: text.num_experts,
            num_experts_per_tok: text.num_experts_per_tok,
            moe_intermediate_size: text.moe_intermediate_size,
            shared_expert_intermediate_size: text.shared_expert_intermediate_size,
            max_position_embeddings: text.max_position_embeddings,
            vocab_size: text.vocab_size,
            layer_kinds,
        })
    }

    fn from_qwen3_dense_config(
        config: RawQwenConfig,
        architecture: String,
    ) -> Result<Self, ModelSpecError> {
        validate_supported_qwen3_dense_options(&config)?;
        let hidden_size = required_root_u32(config.hidden_size, "hidden_size")?;
        let num_attention_heads =
            required_root_u32(config.num_attention_heads, "num_attention_heads")?;
        let head_dim = config
            .head_dim
            .unwrap_or_else(|| hidden_size / num_attention_heads.max(1));
        let num_hidden_layers = required_root_u32(config.num_hidden_layers, "num_hidden_layers")?;
        Ok(Self {
            family: ModelFamily::Qwen,
            architecture,
            model_type: config.model_type.clone(),
            text_model_type: config.model_type,
            hidden_size,
            rms_norm_eps: required_root_f32(config.rms_norm_eps, "rms_norm_eps")?,
            tie_word_embeddings: config.tie_word_embeddings.unwrap_or(false),
            rope_theta: required_root_f32(config.rope_theta, "rope_theta")?,
            partial_rotary_factor: 1.0,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads: required_root_u32(
                config.num_key_value_heads,
                "num_key_value_heads",
            )?,
            head_dim,
            linear_num_key_heads: 0,
            linear_num_value_heads: 0,
            linear_key_head_dim: 0,
            linear_value_head_dim: 0,
            linear_conv_kernel_dim: 0,
            num_experts: 0,
            num_experts_per_tok: 0,
            moe_intermediate_size: required_root_u32(
                config.intermediate_size,
                "intermediate_size",
            )?,
            shared_expert_intermediate_size: 0,
            max_position_embeddings: required_root_u32(
                config.max_position_embeddings,
                "max_position_embeddings",
            )?,
            vocab_size: required_root_u32(config.vocab_size, "vocab_size")?,
            layer_kinds: vec![AttentionKind::FullAttention; num_hidden_layers as usize],
        })
    }

    pub fn is_qwen3_dense(&self) -> bool {
        self.architecture == "Qwen3ForCausalLM" && self.model_type == "qwen3"
    }

    pub fn tensor_root(&self) -> &'static str {
        if self.is_qwen3_dense() {
            "model"
        } else {
            "model.language_model"
        }
    }

    pub fn embed_tokens_weight(&self) -> String {
        format!("{}.embed_tokens.weight", self.tensor_root())
    }

    pub fn final_norm_weight(&self) -> String {
        format!("{}.norm.weight", self.tensor_root())
    }

    pub fn lm_head_weight(&self) -> String {
        if self.tie_word_embeddings {
            self.embed_tokens_weight()
        } else {
            "lm_head.weight".to_owned()
        }
    }

    pub fn layer_tensor(&self, layer_idx: usize, suffix: &str) -> String {
        format!("{}.layers.{layer_idx}.{suffix}", self.tensor_root())
    }

    pub fn mlp_tensor(&self, layer_idx: usize, suffix: &str) -> String {
        self.layer_tensor(layer_idx, &format!("mlp.{suffix}"))
    }

    pub fn self_attn_tensor(&self, layer_idx: usize, suffix: &str) -> String {
        self.layer_tensor(layer_idx, &format!("self_attn.{suffix}"))
    }
}

fn required_root_u32(value: Option<u32>, field: &str) -> Result<u32, ModelSpecError> {
    value.ok_or_else(|| ModelSpecError::unsupported(format!("qwen config missing {field}")))
}

fn required_root_f32(value: Option<f32>, field: &str) -> Result<f32, ModelSpecError> {
    value.ok_or_else(|| ModelSpecError::unsupported(format!("qwen config missing {field}")))
}

fn validate_supported_qwen3_dense_options(config: &RawQwenConfig) -> Result<(), ModelSpecError> {
    if let Some(hidden_act) = config.hidden_act.as_deref()
        && hidden_act != "silu"
    {
        return Err(ModelSpecError::unsupported(format!(
            "qwen3 dense hidden_act `{hidden_act}` is unsupported; only `silu` is supported"
        )));
    }
    if config.attention_bias.unwrap_or(false) {
        return Err(ModelSpecError::unsupported(
            "qwen3 dense attention_bias=true is unsupported",
        ));
    }
    if config.use_sliding_window.unwrap_or(false) {
        return Err(ModelSpecError::unsupported(
            "qwen3 dense use_sliding_window=true is unsupported",
        ));
    }
    if config.sliding_window.is_some() {
        return Err(ModelSpecError::unsupported(
            "qwen3 dense sliding_window is unsupported",
        ));
    }
    if config.rope_scaling.is_some() {
        return Err(ModelSpecError::unsupported(
            "qwen3 dense rope_scaling is unsupported",
        ));
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
struct RawQwenConfig {
    architectures: Vec<String>,
    model_type: String,
    text_config: Option<RawQwenTextConfig>,
    hidden_act: Option<String>,
    attention_bias: Option<bool>,
    use_sliding_window: Option<bool>,
    sliding_window: Option<u32>,
    rope_scaling: Option<serde_json::Value>,
    hidden_size: Option<u32>,
    intermediate_size: Option<u32>,
    max_position_embeddings: Option<u32>,
    num_attention_heads: Option<u32>,
    num_hidden_layers: Option<u32>,
    num_key_value_heads: Option<u32>,
    head_dim: Option<u32>,
    rms_norm_eps: Option<f32>,
    rope_theta: Option<f32>,
    tie_word_embeddings: Option<bool>,
    vocab_size: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct RawQwenTextConfig {
    model_type: String,
    hidden_size: u32,
    rms_norm_eps: Option<f32>,
    tie_word_embeddings: Option<bool>,
    rope_parameters: Option<RawRopeParameters>,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    linear_num_key_heads: u32,
    linear_num_value_heads: u32,
    linear_key_head_dim: u32,
    linear_value_head_dim: u32,
    linear_conv_kernel_dim: u32,
    num_experts: u32,
    num_experts_per_tok: u32,
    moe_intermediate_size: u32,
    shared_expert_intermediate_size: u32,
    max_position_embeddings: u32,
    vocab_size: u32,
    layer_types: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RawRopeParameters {
    rope_theta: f32,
    partial_rotary_factor: Option<f32>,
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
