use llm_models::{AttentionKind, QwenModelSpec};

pub const QWEN_EMBED_TOKENS_WEIGHT: &str = "model.language_model.embed_tokens.weight";
pub const QWEN_LAYER0_INPUT_NORM_WEIGHT: &str =
    "model.language_model.layers.0.input_layernorm.weight";
pub(crate) const QWEN_LAYER0_LINEAR_IN_PROJ_QKV_WEIGHT: &str =
    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
pub(crate) const QWEN_LAYER0_LINEAR_IN_PROJ_Z_WEIGHT: &str =
    "model.language_model.layers.0.linear_attn.in_proj_z.weight";
pub(crate) const QWEN_LAYER0_LINEAR_IN_PROJ_B_WEIGHT: &str =
    "model.language_model.layers.0.linear_attn.in_proj_b.weight";
pub(crate) const QWEN_LAYER0_LINEAR_IN_PROJ_A_WEIGHT: &str =
    "model.language_model.layers.0.linear_attn.in_proj_a.weight";
pub(crate) const QWEN_LAYER0_LINEAR_CONV1D_WEIGHT: &str =
    "model.language_model.layers.0.linear_attn.conv1d.weight";
pub(crate) const QWEN_LAYER0_LINEAR_NORM_WEIGHT: &str =
    "model.language_model.layers.0.linear_attn.norm.weight";
pub(crate) const QWEN_LAYER0_LINEAR_OUT_PROJ_WEIGHT: &str =
    "model.language_model.layers.0.linear_attn.out_proj.weight";
pub(crate) const QWEN_LAYER0_POST_ATTENTION_NORM_WEIGHT: &str =
    "model.language_model.layers.0.post_attention_layernorm.weight";
pub(crate) const QWEN_LAYER0_MLP_GATE_WEIGHT: &str =
    "model.language_model.layers.0.mlp.gate.weight";
pub(crate) const QWEN_LAYER0_MLP_EXPERTS_GATE_UP_PROJ: &str =
    "model.language_model.layers.0.mlp.experts.gate_up_proj";
pub(crate) const QWEN_LAYER0_MLP_EXPERTS_DOWN_PROJ: &str =
    "model.language_model.layers.0.mlp.experts.down_proj";
pub(crate) const QWEN_LAYER0_MLP_SHARED_GATE_PROJ_WEIGHT: &str =
    "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight";
pub(crate) const QWEN_LAYER0_MLP_SHARED_UP_PROJ_WEIGHT: &str =
    "model.language_model.layers.0.mlp.shared_expert.up_proj.weight";
pub(crate) const QWEN_LAYER0_MLP_SHARED_DOWN_PROJ_WEIGHT: &str =
    "model.language_model.layers.0.mlp.shared_expert.down_proj.weight";
pub(crate) const QWEN_LAYER0_MLP_SHARED_EXPERT_GATE_WEIGHT: &str =
    "model.language_model.layers.0.mlp.shared_expert_gate.weight";
pub const QWEN_FINAL_NORM_WEIGHT: &str = "model.language_model.norm.weight";
pub(crate) const QWEN_LM_HEAD_WEIGHT: &str = "lm_head.weight";

pub(crate) fn qwen_layer_tensor(layer_idx: usize, suffix: &str) -> String {
    format!("model.language_model.layers.{layer_idx}.{suffix}")
}

pub(crate) fn qwen_linear_attn_tensor(layer_idx: usize, suffix: &str) -> String {
    qwen_layer_tensor(layer_idx, &format!("linear_attn.{suffix}"))
}

pub fn qwen_static_f32_tensors_for_spec(spec: &QwenModelSpec) -> Vec<String> {
    let mut tensors = Vec::new();
    tensors.push(spec.final_norm_weight());
    for (layer_idx, kind) in spec.layer_kinds.iter().enumerate() {
        tensors.push(spec.layer_tensor(layer_idx, "input_layernorm.weight"));
        tensors.push(spec.layer_tensor(layer_idx, "post_attention_layernorm.weight"));
        match kind {
            AttentionKind::LinearAttention => {
                tensors.push(spec.layer_tensor(layer_idx, "linear_attn.dt_bias"));
                tensors.push(spec.layer_tensor(layer_idx, "linear_attn.A_log"));
                tensors.push(spec.layer_tensor(layer_idx, "linear_attn.conv1d.weight"));
                tensors.push(spec.layer_tensor(layer_idx, "linear_attn.norm.weight"));
            }
            AttentionKind::FullAttention => {
                tensors.push(spec.self_attn_tensor(layer_idx, "q_norm.weight"));
                tensors.push(spec.self_attn_tensor(layer_idx, "k_norm.weight"));
            }
            _ => {}
        }
    }
    tensors
}
