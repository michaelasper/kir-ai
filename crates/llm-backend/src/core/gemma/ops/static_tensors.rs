use llm_models::GemmaModelSpec;

pub fn gemma_static_f32_tensors_for_spec(spec: &GemmaModelSpec) -> Vec<String> {
    let mut tensors = Vec::new();
    tensors.push(spec.final_norm_weight());
    if spec.uses_per_layer_input() {
        tensors.push(spec.per_layer_projection_norm_weight());
    }
    for layer_idx in 0..spec.num_hidden_layers as usize {
        tensors.push(spec.layer_tensor(layer_idx, "input_layernorm.weight"));
        tensors.push(spec.layer_tensor(layer_idx, "layer_scalar"));
        tensors.push(spec.layer_tensor(layer_idx, "post_attention_layernorm.weight"));
        tensors.push(spec.layer_tensor(layer_idx, "pre_feedforward_layernorm.weight"));
        tensors.push(spec.layer_tensor(layer_idx, "post_feedforward_layernorm.weight"));
        tensors.push(spec.self_attn_tensor(layer_idx, "q_norm.weight"));
        if spec.requires_key_value_projection(layer_idx) {
            tensors.push(spec.self_attn_tensor(layer_idx, "k_norm.weight"));
        }
        if spec.uses_per_layer_input() {
            tensors.push(spec.layer_tensor(layer_idx, "post_per_layer_input_norm.weight"));
        }
        if spec.uses_moe() {
            tensors.push(spec.layer_tensor(layer_idx, "router.per_expert_scale"));
            tensors.push(spec.layer_tensor(layer_idx, "router.scale"));
            tensors.push(spec.layer_tensor(layer_idx, "pre_feedforward_layernorm_2.weight"));
            tensors.push(spec.layer_tensor(layer_idx, "post_feedforward_layernorm_1.weight"));
            tensors.push(spec.layer_tensor(layer_idx, "post_feedforward_layernorm_2.weight"));
        }
    }
    tensors
}
