use crate::{ModelFamily, ModelSpec, ModelSpecError, SafetensorsIndex};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GemmaAttentionKind {
    SlidingAttention,
    FullAttention,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GemmaWeightLayout {
    ConditionalLanguageModel,
    TextOnly,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GemmaModelSpec {
    pub family: ModelFamily,
    pub architecture: String,
    pub model_type: String,
    pub text_model_type: String,
    pub weight_layout: GemmaWeightLayout,
    pub hidden_size: u32,
    pub hidden_size_per_layer_input: u32,
    pub rms_norm_eps: f32,
    pub tie_word_embeddings: bool,
    pub full_rope_theta: f32,
    pub full_partial_rotary_factor: f32,
    pub sliding_rope_theta: f32,
    pub sliding_window: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub num_global_key_value_heads: Option<u32>,
    pub num_kv_shared_layers: u32,
    pub head_dim: u32,
    pub global_head_dim: Option<u32>,
    pub intermediate_size: u32,
    pub max_position_embeddings: u32,
    pub vocab_size: u32,
    pub vocab_size_per_layer_input: u32,
    pub attention_k_eq_v: bool,
    pub enable_moe_block: bool,
    pub num_experts: Option<u32>,
    pub top_k_experts: Option<u32>,
    pub moe_intermediate_size: Option<u32>,
    pub use_double_wide_mlp: bool,
    pub layer_kinds: Vec<GemmaAttentionKind>,
}

impl GemmaModelSpec {
    pub fn from_config_json(json: &str) -> Result<Self, ModelSpecError> {
        let value: serde_json::Value = serde_json::from_str(json)
            .map_err(|err| ModelSpecError::invalid_request(format!("invalid JSON: {err}")))?;
        Self::from_config_value(value)
    }

    pub fn from_config_value(value: serde_json::Value) -> Result<Self, ModelSpecError> {
        let model_type = value
            .get("model_type")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| ModelSpecError::unsupported("gemma config missing model_type"))?;
        match model_type {
            "gemma4" => {
                let config: RawGemmaConfig = serde_json::from_value(value).map_err(|err| {
                    ModelSpecError::invalid_request(format!("invalid JSON: {err}"))
                })?;
                let architecture = config
                    .architectures
                    .as_ref()
                    .and_then(|architectures| architectures.first())
                    .ok_or_else(|| {
                        ModelSpecError::unsupported("gemma config missing architecture")
                    })?
                    .clone();
                if architecture != "Gemma4ForConditionalGeneration" {
                    return Err(ModelSpecError::unsupported(
                        "config is not a supported Gemma 4 conditional generation architecture",
                    ));
                }
                let text = config.text_config.ok_or_else(|| {
                    ModelSpecError::unsupported("gemma4 config missing text_config")
                })?;
                Self::from_text_config(
                    architecture,
                    config.model_type,
                    GemmaWeightLayout::ConditionalLanguageModel,
                    config.tie_word_embeddings,
                    text,
                )
            }
            "gemma4_text" => {
                let text: RawGemmaTextConfig = serde_json::from_value(value).map_err(|err| {
                    ModelSpecError::invalid_request(format!("invalid JSON: {err}"))
                })?;
                Self::from_text_config(
                    "Gemma4TextForCausalLM".to_owned(),
                    "gemma4_text".to_owned(),
                    GemmaWeightLayout::TextOnly,
                    None,
                    text,
                )
            }
            _ => Err(ModelSpecError::unsupported(
                "config is not a supported Gemma 4 text architecture",
            )),
        }
    }

    fn from_text_config(
        architecture: String,
        model_type: String,
        weight_layout: GemmaWeightLayout,
        top_level_tie_word_embeddings: Option<bool>,
        text: RawGemmaTextConfig,
    ) -> Result<Self, ModelSpecError> {
        if text.model_type != "gemma4_text" {
            return Err(ModelSpecError::unsupported(format!(
                "unsupported gemma text model_type `{}`",
                text.model_type
            )));
        }
        validate_supported_gemma4_text_options(&text)?;
        let layer_kinds = text
            .layer_types
            .iter()
            .map(|kind| match kind.as_str() {
                "sliding_attention" => Ok(GemmaAttentionKind::SlidingAttention),
                "full_attention" => Ok(GemmaAttentionKind::FullAttention),
                other => Err(ModelSpecError::unsupported(format!(
                    "unsupported gemma layer type `{other}`"
                ))),
            })
            .collect::<Result<Vec<_>, _>>()?;
        if layer_kinds.len() != text.num_hidden_layers as usize {
            return Err(ModelSpecError::invalid_request(
                "gemma layer_types length does not match num_hidden_layers",
            ));
        }
        let num_kv_shared_layers = text.num_kv_shared_layers.unwrap_or(0);
        if num_kv_shared_layers > text.num_hidden_layers {
            return Err(ModelSpecError::invalid_request(
                "gemma num_kv_shared_layers cannot exceed num_hidden_layers",
            ));
        }
        if text.enable_moe_block.unwrap_or(false)
            && (text.num_experts.is_none()
                || text.top_k_experts.is_none()
                || text.moe_intermediate_size.is_none())
        {
            return Err(ModelSpecError::unsupported(
                "gemma MoE config missing num_experts, top_k_experts, or moe_intermediate_size",
            ));
        }
        Ok(Self {
            family: ModelFamily::Gemma,
            architecture,
            model_type,
            text_model_type: text.model_type,
            weight_layout,
            hidden_size: text.hidden_size,
            hidden_size_per_layer_input: text.hidden_size_per_layer_input.unwrap_or(0),
            rms_norm_eps: text.rms_norm_eps,
            tie_word_embeddings: text
                .tie_word_embeddings
                .or(top_level_tie_word_embeddings)
                .unwrap_or(true),
            full_rope_theta: text.rope_parameters.full_attention.rope_theta,
            full_partial_rotary_factor: text
                .rope_parameters
                .full_attention
                .partial_rotary_factor
                .unwrap_or(1.0),
            sliding_rope_theta: text.rope_parameters.sliding_attention.rope_theta,
            sliding_window: text.sliding_window,
            num_hidden_layers: text.num_hidden_layers,
            num_attention_heads: text.num_attention_heads,
            num_key_value_heads: text.num_key_value_heads,
            num_global_key_value_heads: text.num_global_key_value_heads,
            num_kv_shared_layers,
            head_dim: text.head_dim,
            global_head_dim: text.global_head_dim,
            intermediate_size: text.intermediate_size,
            max_position_embeddings: text.max_position_embeddings,
            vocab_size: text.vocab_size,
            vocab_size_per_layer_input: text.vocab_size_per_layer_input.unwrap_or(text.vocab_size),
            attention_k_eq_v: text.attention_k_eq_v.unwrap_or(false),
            enable_moe_block: text.enable_moe_block.unwrap_or(false),
            num_experts: text.num_experts,
            top_k_experts: text.top_k_experts,
            moe_intermediate_size: text.moe_intermediate_size,
            use_double_wide_mlp: text.use_double_wide_mlp.unwrap_or(false),
            layer_kinds,
        })
    }

    pub fn tensor_root(&self) -> &'static str {
        match self.weight_layout {
            GemmaWeightLayout::ConditionalLanguageModel => "model.language_model",
            GemmaWeightLayout::TextOnly => "model",
        }
    }

    pub fn embed_tokens_weight(&self) -> String {
        format!("{}.embed_tokens.weight", self.tensor_root())
    }

    pub fn embed_tokens_per_layer_weight(&self) -> String {
        format!("{}.embed_tokens_per_layer.weight", self.tensor_root())
    }

    pub fn per_layer_model_projection_weight(&self) -> String {
        format!("{}.per_layer_model_projection.weight", self.tensor_root())
    }

    pub fn per_layer_projection_norm_weight(&self) -> String {
        format!("{}.per_layer_projection_norm.weight", self.tensor_root())
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

    pub fn uses_per_layer_input(&self) -> bool {
        self.hidden_size_per_layer_input > 0
    }

    pub fn uses_moe(&self) -> bool {
        self.enable_moe_block
    }

    pub fn is_kv_shared_layer(&self, layer_idx: usize) -> bool {
        if self.num_kv_shared_layers == 0 {
            return false;
        }
        let first_shared = self
            .num_hidden_layers
            .saturating_sub(self.num_kv_shared_layers) as usize;
        layer_idx >= first_shared
    }

    pub fn requires_key_value_projection(&self, layer_idx: usize) -> bool {
        !self.is_kv_shared_layer(layer_idx)
    }

    pub fn requires_value_projection(&self, layer_idx: usize) -> bool {
        if !self.requires_key_value_projection(layer_idx) {
            return false;
        }
        !matches!(
            self.layer_kinds[layer_idx],
            GemmaAttentionKind::FullAttention
        ) || !self.attention_k_eq_v
    }

    pub fn validate_text_weights(&self, index: &SafetensorsIndex) -> Result<(), ModelSpecError> {
        index.require(self.embed_tokens_weight())?;
        index.require(self.final_norm_weight())?;
        if !self.tie_word_embeddings {
            index.require(self.lm_head_weight())?;
        }
        if self.uses_per_layer_input() {
            index.require(self.embed_tokens_per_layer_weight())?;
            index.require(self.per_layer_model_projection_weight())?;
            index.require(self.per_layer_projection_norm_weight())?;
        }
        for layer in 0..self.num_hidden_layers as usize {
            index.require(self.layer_tensor(layer, "input_layernorm.weight"))?;
            index.require(self.layer_tensor(layer, "layer_scalar"))?;
            index.require(self.layer_tensor(layer, "post_attention_layernorm.weight"))?;
            index.require(self.layer_tensor(layer, "pre_feedforward_layernorm.weight"))?;
            index.require(self.layer_tensor(layer, "post_feedforward_layernorm.weight"))?;
            index.require(self.self_attn_tensor(layer, "q_proj.weight"))?;
            index.require(self.self_attn_tensor(layer, "o_proj.weight"))?;
            index.require(self.self_attn_tensor(layer, "q_norm.weight"))?;
            if self.requires_key_value_projection(layer) {
                index.require(self.self_attn_tensor(layer, "k_proj.weight"))?;
                index.require(self.self_attn_tensor(layer, "k_norm.weight"))?;
            }
            if self.requires_value_projection(layer) {
                index.require(self.self_attn_tensor(layer, "v_proj.weight"))?;
            }
            index.require(self.mlp_tensor(layer, "gate_proj.weight"))?;
            index.require(self.mlp_tensor(layer, "up_proj.weight"))?;
            index.require(self.mlp_tensor(layer, "down_proj.weight"))?;
            if self.uses_per_layer_input() {
                index.require(self.layer_tensor(layer, "per_layer_input_gate.weight"))?;
                index.require(self.layer_tensor(layer, "per_layer_projection.weight"))?;
                index.require(self.layer_tensor(layer, "post_per_layer_input_norm.weight"))?;
            }
            if self.uses_moe() {
                index.require(self.layer_tensor(layer, "experts.down_proj"))?;
                index.require(self.layer_tensor(layer, "experts.gate_up_proj"))?;
                index.require(self.layer_tensor(layer, "router.per_expert_scale"))?;
                index.require(self.layer_tensor(layer, "router.proj.weight"))?;
                index.require(self.layer_tensor(layer, "router.scale"))?;
                index.require(self.layer_tensor(layer, "pre_feedforward_layernorm_2.weight"))?;
                index.require(self.layer_tensor(layer, "post_feedforward_layernorm_1.weight"))?;
                index.require(self.layer_tensor(layer, "post_feedforward_layernorm_2.weight"))?;
            }
        }
        Ok(())
    }
}

impl ModelSpec for GemmaModelSpec {
    fn family(&self) -> ModelFamily {
        self.family
    }

    fn architecture(&self) -> &str {
        &self.architecture
    }

    fn model_type(&self) -> &str {
        &self.model_type
    }

    fn text_model_type(&self) -> &str {
        &self.text_model_type
    }

    fn max_position_embeddings(&self) -> u32 {
        self.max_position_embeddings
    }

    fn num_hidden_layers(&self) -> u32 {
        self.num_hidden_layers
    }

    fn hidden_size(&self) -> u32 {
        self.hidden_size
    }

    fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    fn validate_text_weights(&self, index: &SafetensorsIndex) -> Result<(), ModelSpecError> {
        GemmaModelSpec::validate_text_weights(self, index)
    }
}

fn validate_supported_gemma4_text_options(text: &RawGemmaTextConfig) -> Result<(), ModelSpecError> {
    if let Some(hidden_activation) = text.hidden_activation.as_deref()
        && hidden_activation != "gelu_pytorch_tanh"
    {
        return Err(ModelSpecError::unsupported(format!(
            "gemma4 hidden_activation `{hidden_activation}` is unsupported; only `gelu_pytorch_tanh` is supported"
        )));
    }
    if text.attention_bias.unwrap_or(false) {
        return Err(ModelSpecError::unsupported(
            "gemma4 attention_bias=true is unsupported",
        ));
    }
    if text.attention_dropout.unwrap_or(0.0) != 0.0 {
        return Err(ModelSpecError::unsupported(
            "gemma4 attention_dropout other than 0.0 is unsupported",
        ));
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
struct RawGemmaConfig {
    architectures: Option<Vec<String>>,
    model_type: String,
    text_config: Option<RawGemmaTextConfig>,
    tie_word_embeddings: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct RawGemmaTextConfig {
    model_type: String,
    attention_bias: Option<bool>,
    attention_dropout: Option<f32>,
    attention_k_eq_v: Option<bool>,
    enable_moe_block: Option<bool>,
    global_head_dim: Option<u32>,
    head_dim: u32,
    hidden_activation: Option<String>,
    hidden_size: u32,
    hidden_size_per_layer_input: Option<u32>,
    intermediate_size: u32,
    layer_types: Vec<String>,
    max_position_embeddings: u32,
    moe_intermediate_size: Option<u32>,
    num_attention_heads: u32,
    num_experts: Option<u32>,
    num_global_key_value_heads: Option<u32>,
    num_hidden_layers: u32,
    num_key_value_heads: u32,
    num_kv_shared_layers: Option<u32>,
    rms_norm_eps: f32,
    rope_parameters: RawGemmaRopeParameters,
    sliding_window: u32,
    tie_word_embeddings: Option<bool>,
    top_k_experts: Option<u32>,
    use_double_wide_mlp: Option<bool>,
    vocab_size: u32,
    vocab_size_per_layer_input: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct RawGemmaRopeParameters {
    full_attention: RawGemmaRopeKindParameters,
    sliding_attention: RawGemmaRopeKindParameters,
}

#[derive(Debug, Deserialize)]
struct RawGemmaRopeKindParameters {
    rope_theta: f32,
    partial_rotary_factor: Option<f32>,
}
