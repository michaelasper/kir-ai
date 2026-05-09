use super::*;

pub fn qwen_layer0_moe_router(
    store: &SafeTensorShardStore,
    hidden_states: &[f32],
    top_k: usize,
) -> Result<QwenMoeRouterProbe, TensorLoadError> {
    qwen_layer_moe_router(store, 0, hidden_states, top_k)
}

pub fn qwen_layer_moe_router(
    store: &SafeTensorShardStore,
    layer_idx: usize,
    hidden_states: &[f32],
    top_k: usize,
) -> Result<QwenMoeRouterProbe, TensorLoadError> {
    qwen_layer_moe_router_with_matvec(
        store,
        layer_idx,
        hidden_states,
        top_k,
        &CpuNativeMatvecBackend,
    )
}

pub fn qwen_layer_moe_router_with_matvec(
    store: &SafeTensorShardStore,
    layer_idx: usize,
    hidden_states: &[f32],
    top_k: usize,
    matvec: &impl NativeMatvecBackend,
) -> Result<QwenMoeRouterProbe, TensorLoadError> {
    let logits = matvec.bf16_matvec_row_major_f32(
        store,
        &qwen_mlp_tensor(layer_idx, "gate.weight"),
        hidden_states,
    )?;
    let selected = matvec
        .softmax_top_k_f32(&logits, top_k)
        .map_err(|err| TensorLoadError::integrity(format!("Qwen MoE router failed: {err}")))?;
    Ok(QwenMoeRouterProbe { logits, selected })
}

pub fn qwen_layer0_moe_forward(
    store: &SafeTensorShardStore,
    dims: &QwenMoeDims,
    hidden_states: &[f32],
    router: &QwenMoeRouterProbe,
) -> Result<Vec<f32>, TensorLoadError> {
    qwen_layer_moe_forward(store, 0, dims, hidden_states, router)
}

fn qwen_layer_dense_mlp_with_matvec(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[f32],
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    let hidden_size = spec.hidden_size as usize;
    let intermediate_size = spec.moe_intermediate_size as usize;
    if hidden_states.len() != hidden_size {
        return Err(TensorLoadError::integrity(format!(
            "Qwen dense MLP hidden length {} must match hidden size {hidden_size}",
            hidden_states.len()
        )));
    }
    let gate = matvec
        .bf16_matvec_row_major_f32(
            store,
            &spec.mlp_tensor(layer_idx, "gate_proj.weight"),
            hidden_states,
        )
        .map_err(|err| TensorLoadError::integrity(format!("Qwen dense MLP gate failed: {err}")))?;
    let up = matvec
        .bf16_matvec_row_major_f32(
            store,
            &spec.mlp_tensor(layer_idx, "up_proj.weight"),
            hidden_states,
        )
        .map_err(|err| TensorLoadError::integrity(format!("Qwen dense MLP up failed: {err}")))?;
    if gate.len() != intermediate_size || up.len() != intermediate_size {
        return Err(TensorLoadError::integrity(format!(
            "Qwen dense MLP gate/up lengths {}, {} must match intermediate size {intermediate_size}",
            gate.len(),
            up.len()
        )));
    }
    let activated = gate
        .iter()
        .zip(up)
        .map(|(gate, up)| silu_f32(*gate) * up)
        .collect::<Vec<_>>();
    let down = matvec
        .bf16_matvec_row_major_f32(
            store,
            &spec.mlp_tensor(layer_idx, "down_proj.weight"),
            &activated,
        )
        .map_err(|err| TensorLoadError::integrity(format!("Qwen dense MLP down failed: {err}")))?;
    if down.len() != hidden_size {
        return Err(TensorLoadError::integrity(format!(
            "Qwen dense MLP down output length {} must match hidden size {hidden_size}",
            down.len()
        )));
    }
    Ok(down)
}

pub(super) fn qwen_layer_feed_forward_with_matvec(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[f32],
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    if spec.is_qwen3_dense() {
        return qwen_layer_dense_mlp_with_matvec(store, spec, layer_idx, hidden_states, matvec);
    }
    let router = qwen_layer_moe_router_with_matvec(
        store,
        layer_idx,
        hidden_states,
        spec.num_experts_per_tok as usize,
        matvec,
    )?;
    qwen_layer_moe_forward_with_matvec(
        store,
        layer_idx,
        &QwenMoeDims::from_spec(spec),
        hidden_states,
        &router,
        matvec,
    )
}

pub fn qwen_layer_moe_forward(
    store: &SafeTensorShardStore,
    layer_idx: usize,
    dims: &QwenMoeDims,
    hidden_states: &[f32],
    router: &QwenMoeRouterProbe,
) -> Result<Vec<f32>, TensorLoadError> {
    qwen_layer_moe_forward_with_matvec(
        store,
        layer_idx,
        dims,
        hidden_states,
        router,
        &CpuNativeMatvecBackend,
    )
}

pub fn qwen_layer_moe_forward_with_matvec(
    store: &SafeTensorShardStore,
    layer_idx: usize,
    dims: &QwenMoeDims,
    hidden_states: &[f32],
    router: &QwenMoeRouterProbe,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    if hidden_states.len() != dims.hidden_size {
        return Err(TensorLoadError::integrity(format!(
            "Qwen MoE hidden length {} must match hidden size {}",
            hidden_states.len(),
            dims.hidden_size
        )));
    }
    let selected_capacity = router
        .selected
        .len()
        .checked_mul(dims.hidden_size)
        .ok_or_else(|| TensorLoadError::integrity("Qwen selected expert shape overflow"))?;
    let mut selected_outputs = Vec::with_capacity(selected_capacity);
    let mut selected_weights = Vec::with_capacity(router.selected.len());
    let gate_up_expert_elements = dims
        .moe_intermediate_size
        .checked_mul(2)
        .and_then(|rows| rows.checked_mul(dims.hidden_size))
        .ok_or_else(|| TensorLoadError::integrity("Qwen expert gate/up shape overflow"))?;
    let down_expert_elements = dims
        .hidden_size
        .checked_mul(dims.moe_intermediate_size)
        .ok_or_else(|| TensorLoadError::integrity("Qwen expert down shape overflow"))?;
    let split = dims
        .moe_intermediate_size
        .checked_mul(dims.hidden_size)
        .ok_or_else(|| TensorLoadError::integrity("Qwen expert split shape overflow"))?;
    let gate_up_tensor = qwen_mlp_tensor(layer_idx, "experts.gate_up_proj");
    let down_tensor = qwen_mlp_tensor(layer_idx, "experts.down_proj");
    for selected in &router.selected {
        if selected.index >= dims.num_experts {
            return Err(TensorLoadError::integrity(format!(
                "Qwen selected expert {} exceeds expert count {}",
                selected.index, dims.num_experts
            )));
        }
        let gate_up_offset = selected
            .index
            .checked_mul(gate_up_expert_elements)
            .ok_or_else(|| TensorLoadError::integrity("Qwen expert gate/up offset overflow"))?;
        let gate = matvec
            .bf16_matvec_range_row_major_f32(
                store,
                &gate_up_tensor,
                gate_up_offset,
                dims.moe_intermediate_size,
                dims.hidden_size,
                hidden_states,
            )
            .map_err(|err| {
                TensorLoadError::integrity(format!("Qwen selected expert gate failed: {err}"))
            })?;
        let up = matvec
            .bf16_matvec_range_row_major_f32(
                store,
                &gate_up_tensor,
                gate_up_offset
                    .checked_add(split)
                    .ok_or_else(|| TensorLoadError::integrity("Qwen expert up offset overflow"))?,
                dims.moe_intermediate_size,
                dims.hidden_size,
                hidden_states,
            )
            .map_err(|err| {
                TensorLoadError::integrity(format!("Qwen selected expert up failed: {err}"))
            })?;
        let activated = gate
            .iter()
            .zip(up)
            .map(|(gate, up)| silu_f32(*gate) * up)
            .collect::<Vec<_>>();
        let down_offset = selected
            .index
            .checked_mul(down_expert_elements)
            .ok_or_else(|| TensorLoadError::integrity("Qwen expert down offset overflow"))?;
        let expert_output = matvec
            .bf16_matvec_range_row_major_f32(
                store,
                &down_tensor,
                down_offset,
                dims.hidden_size,
                dims.moe_intermediate_size,
                &activated,
            )
            .map_err(|err| {
                TensorLoadError::integrity(format!("Qwen selected expert down failed: {err}"))
            })?;
        selected_outputs.extend_from_slice(&expert_output);
        selected_weights.push(selected.weight);
    }
    let selected_output = matvec
        .weighted_sum_f32(&selected_outputs, &selected_weights, dims.hidden_size)
        .map_err(|err| {
            TensorLoadError::integrity(format!("Qwen selected expert accumulation failed: {err}"))
        })?;

    let shared_gate = matvec
        .bf16_matvec_row_major_f32(
            store,
            &qwen_mlp_tensor(layer_idx, "shared_expert.gate_proj.weight"),
            hidden_states,
        )
        .map_err(|err| {
            TensorLoadError::integrity(format!("Qwen shared expert gate failed: {err}"))
        })?;
    let shared_up = matvec
        .bf16_matvec_row_major_f32(
            store,
            &qwen_mlp_tensor(layer_idx, "shared_expert.up_proj.weight"),
            hidden_states,
        )
        .map_err(|err| {
            TensorLoadError::integrity(format!("Qwen shared expert up failed: {err}"))
        })?;
    let shared_activated = shared_gate
        .iter()
        .zip(shared_up)
        .map(|(gate, up)| silu_f32(*gate) * up)
        .collect::<Vec<_>>();
    let shared_output = matvec
        .bf16_matvec_row_major_f32(
            store,
            &qwen_mlp_tensor(layer_idx, "shared_expert.down_proj.weight"),
            &shared_activated,
        )
        .map_err(|err| {
            TensorLoadError::integrity(format!("Qwen shared expert down failed: {err}"))
        })?;
    let shared_gate = matvec
        .bf16_matvec_row_major_f32(
            store,
            &qwen_mlp_tensor(layer_idx, "shared_expert_gate.weight"),
            hidden_states,
        )
        .map_err(|err| {
            TensorLoadError::integrity(format!("Qwen shared expert gate failed: {err}"))
        })?
        .into_iter()
        .next()
        .ok_or_else(|| TensorLoadError::integrity("Qwen shared expert gate returned no value"))?;
    let shared_gate = sigmoid_f32(shared_gate);
    let combined_capacity = dims
        .hidden_size
        .checked_mul(2)
        .ok_or_else(|| TensorLoadError::integrity("Qwen shared expert shape overflow"))?;
    let mut combined_values = Vec::with_capacity(combined_capacity);
    combined_values.extend_from_slice(&selected_output);
    combined_values.extend_from_slice(&shared_output);
    matvec
        .weighted_sum_f32(&combined_values, &[1.0, shared_gate], dims.hidden_size)
        .map_err(|err| {
            TensorLoadError::integrity(format!("Qwen shared expert accumulation failed: {err}"))
        })
}
