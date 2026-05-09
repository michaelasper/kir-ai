use super::super::math::{
    InferenceScratchpad, MathError, TopKWeight, silu_f32, softmax_top_k_f32,
};
use super::super::{
    CpuNativeMatvecBackend, NativeMatvecBackend, SafeTensorShardStore, TensorLoadError,
};
use super::{QwenMoeDims, QwenMoeRouterProbe, qwen_layer_tensor, qwen_mlp_tensor};
use llm_models::QwenModelSpec;

pub async fn qwen_layer_dense_mlp_with_matvec(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[f32],
    matvec: &impl NativeMatvecBackend,
    scratch: &mut InferenceScratchpad,
    output: &mut [f32],
) -> Result<(), TensorLoadError> {
    let hidden_size = spec.hidden_size as usize;
    let intermediate_size = spec.intermediate_size as usize;
    if hidden_states.len() != hidden_size {
        return Err(TensorLoadError::integrity(format!(
            "Qwen dense MLP hidden length {} must match hidden size {hidden_size}",
            hidden_states.len()
        )));
    }
    let gate = InferenceScratchpad::get_mut(&mut scratch.buf0, intermediate_size);
    matvec
        .bf16_matvec_row_major_f32_in_place(
            store,
            &qwen_mlp_tensor(layer_idx, "gate_proj.weight"),
            hidden_states,
            gate,
        )
        .await
        .map_err(|err| TensorLoadError::integrity(format!("Qwen dense MLP gate failed: {err}")))?;
    let up = InferenceScratchpad::get_mut(&mut scratch.buf1, intermediate_size);
    matvec
        .bf16_matvec_row_major_f32_in_place(
            store,
            &qwen_mlp_tensor(layer_idx, "up_proj.weight"),
            hidden_states,
            up,
        )
        .await
        .map_err(|err| TensorLoadError::integrity(format!("Qwen dense MLP up failed: {err}")))?;
    
    let activated = InferenceScratchpad::get_mut(&mut scratch.buf2, intermediate_size);
    for (a, (g, u)) in activated.iter_mut().zip(gate.iter().zip(up.iter())) {
        *a = silu_f32(*g) * *u;
    }
    
    matvec
        .bf16_matvec_row_major_f32_in_place(
            store,
            &qwen_mlp_tensor(layer_idx, "down_proj.weight"),
            activated,
            output,
        )
        .await
        .map_err(|err| TensorLoadError::integrity(format!("Qwen dense MLP down failed: {err}")))?;
    Ok(())
}

pub(super) async fn qwen_layer_feed_forward_with_matvec(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[f32],
    matvec: &impl NativeMatvecBackend,
    scratch: &mut InferenceScratchpad,
    output: &mut [f32],
) -> Result<(), TensorLoadError> {
    if spec.is_qwen3_dense() {
        return qwen_layer_dense_mlp_with_matvec(store, spec, layer_idx, hidden_states, matvec, scratch, output).await;
    }
    
    // MoE case
    let router = qwen_layer_moe_router_with_matvec(
        store,
        layer_idx,
        hidden_states,
        spec.num_experts_per_tok as usize,
        matvec,
    )
    .await?;
    
    qwen_layer_moe_forward_with_matvec_in_place(
        store,
        layer_idx,
        &QwenMoeDims::from_spec(spec),
        hidden_states,
        &router,
        matvec,
        scratch,
        output,
    )
    .await
}

pub async fn qwen_layer_moe_router_with_matvec(
    store: &SafeTensorShardStore,
    layer_idx: usize,
    hidden_states: &[f32],
    top_k: usize,
    matvec: &impl NativeMatvecBackend,
) -> Result<QwenMoeRouterProbe, TensorLoadError> {
    let logits = matvec
        .bf16_matvec_row_major_f32(
            store,
            &qwen_layer_tensor(layer_idx, "mlp.gate.weight"),
            hidden_states,
        )
        .await
        .map_err(|err| TensorLoadError::integrity(format!("Qwen MoE router failed: {err}")))?;
    let selected = softmax_top_k_f32(&logits, top_k)
        .map_err(|err| TensorLoadError::integrity(format!("Qwen MoE router softmax failed: {err}")))?;
    Ok(QwenMoeRouterProbe { logits, selected })
}

pub async fn qwen_layer_moe_forward_with_matvec_in_place(
    store: &SafeTensorShardStore,
    layer_idx: usize,
    dims: &QwenMoeDims,
    hidden_states: &[f32],
    router: &QwenMoeRouterProbe,
    matvec: &impl NativeMatvecBackend,
    scratch: &mut InferenceScratchpad,
    output: &mut [f32],
) -> Result<(), TensorLoadError> {
    if output.len() < dims.hidden_size {
        return Err(TensorLoadError::integrity("output buffer too small"));
    }
    output.fill(0.0);
    
    let expert_gate_up = InferenceScratchpad::get_mut(&mut scratch.buf0, dims.moe_intermediate_size * 2);
    let expert_down = InferenceScratchpad::get_mut(&mut scratch.buf1, dims.hidden_size);
    let activated = InferenceScratchpad::get_mut(&mut scratch.buf2, dims.moe_intermediate_size);

    for expert in &router.selected {
        matvec
            .bf16_matvec_range_row_major_f32_in_place(
                store,
                &qwen_layer_tensor(layer_idx, "mlp.experts.gate_up_proj"),
                expert.index * dims.moe_intermediate_size * 2,
                dims.moe_intermediate_size * 2,
                dims.hidden_size,
                hidden_states,
                expert_gate_up,
            )
            .await
            .map_err(|err| {
                TensorLoadError::integrity(format!("Qwen expert{layer_idx}.{} gate_up failed: {err}", expert.index))
            })?;
            
        for i in 0..dims.moe_intermediate_size {
            activated[i] = silu_f32(expert_gate_up[i]) * expert_gate_up[i + dims.moe_intermediate_size];
        }
        
        matvec
            .bf16_matvec_range_row_major_f32_in_place(
                store,
                &qwen_layer_tensor(layer_idx, "mlp.experts.down_proj"),
                expert.index * dims.moe_intermediate_size * dims.hidden_size,
                dims.hidden_size,
                dims.moe_intermediate_size,
                activated,
                expert_down,
            )
            .await
            .map_err(|err| {
                TensorLoadError::integrity(format!("Qwen expert{layer_idx}.{} down failed: {err}", expert.index))
            })?;
            
        for (o, d) in output.iter_mut().zip(expert_down.iter()) {
            *o += *d * expert.weight;
        }
    }
    
    let shared_output = InferenceScratchpad::get_mut(&mut scratch.buf0, dims.hidden_size);
    qwen_layer_shared_expert_forward_with_matvec(store, layer_idx, dims, hidden_states, matvec, scratch, shared_output).await?;
    
    let shared_gate_vec = matvec.bf16_matvec_row_major_f32(
        store,
        &qwen_layer_tensor(layer_idx, "mlp.shared_expert_gate.weight"),
        hidden_states,
    ).await.map_err(|err| {
        TensorLoadError::integrity(format!("Qwen shared expert gate failed: {err}"))
    })?;
    let shared_gate = sigmoid_f32(shared_gate_vec[0]);
    
    for (o, s) in output.iter_mut().zip(shared_output.iter()) {
        *o += *s * shared_gate;
    }
    Ok(())
}

async fn qwen_layer_shared_expert_forward_with_matvec(
    store: &SafeTensorShardStore,
    layer_idx: usize,
    dims: &QwenMoeDims,
    hidden_states: &[f32],
    matvec: &impl NativeMatvecBackend,
    scratch: &mut InferenceScratchpad,
    output: &mut [f32],
) -> Result<(), TensorLoadError> {
    let gate = InferenceScratchpad::get_mut(&mut scratch.buf1, dims.shared_expert_intermediate_size);
    matvec.bf16_matvec_row_major_f32_in_place(
        store,
        &qwen_layer_tensor(layer_idx, "mlp.shared_expert.gate_proj.weight"),
        hidden_states,
        gate,
    ).await?;
    let up = InferenceScratchpad::get_mut(&mut scratch.buf2, dims.shared_expert_intermediate_size);
    matvec.bf16_matvec_row_major_f32_in_place(
        store,
        &qwen_layer_tensor(layer_idx, "mlp.shared_expert.up_proj.weight"),
        hidden_states,
        up,
    ).await?;
    let activated = InferenceScratchpad::get_mut(&mut scratch.buf3, dims.shared_expert_intermediate_size);
    for i in 0..dims.shared_expert_intermediate_size {
        activated[i] = silu_f32(gate[i]) * up[i];
    }
    matvec.bf16_matvec_row_major_f32_in_place(
        store,
        &qwen_layer_tensor(layer_idx, "mlp.shared_expert.down_proj.weight"),
        activated,
        output,
    ).await?;
    Ok(())
}

fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
