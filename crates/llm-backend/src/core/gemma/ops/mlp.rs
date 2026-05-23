use super::super::super::math::InferenceScratchpad;
use super::super::super::{NativeMatvecBackend, SafeTensorShardStore, TensorLoadError};
use super::activation::gelu_pytorch_tanh_f32;
use llm_models::GemmaModelSpec;

pub(crate) async fn gemma_layer_dense_mlp(
    store: &SafeTensorShardStore,
    spec: &GemmaModelSpec,
    layer_idx: usize,
    hidden_states: &[f32],
    matvec: &impl NativeMatvecBackend,
    scratch: &mut InferenceScratchpad,
    output: &mut [f32],
) -> Result<(), TensorLoadError> {
    let hidden_size = spec.hidden_size as usize;
    let intermediate_size = gemma_intermediate_size_for_layer(spec, layer_idx);
    if hidden_states.len() != hidden_size {
        return Err(TensorLoadError::integrity(format!(
            "Gemma dense MLP hidden length {} must match hidden size {hidden_size}",
            hidden_states.len()
        )));
    }
    let gate = InferenceScratchpad::get_mut(&mut scratch.buf0, intermediate_size);
    matvec
        .bf16_matvec_row_major_f32_in_place(
            store,
            &spec.mlp_tensor(layer_idx, "gate_proj.weight"),
            hidden_states,
            gate,
        )
        .await?;

    let up = InferenceScratchpad::get_mut(&mut scratch.buf1, intermediate_size);
    matvec
        .bf16_matvec_row_major_f32_in_place(
            store,
            &spec.mlp_tensor(layer_idx, "up_proj.weight"),
            hidden_states,
            up,
        )
        .await?;

    if gate.len() != intermediate_size || up.len() != intermediate_size {
        return Err(TensorLoadError::integrity(format!(
            "Gemma dense MLP gate/up lengths {}, {} must match intermediate size {intermediate_size}",
            gate.len(),
            up.len()
        )));
    }

    let activated = InferenceScratchpad::get_mut(&mut scratch.buf2, intermediate_size);
    for (a, (g, u)) in activated.iter_mut().zip(gate.iter().zip(up.iter())) {
        *a = gelu_pytorch_tanh_f32(*g) * *u;
    }

    matvec
        .bf16_matvec_row_major_f32_in_place(
            store,
            &spec.mlp_tensor(layer_idx, "down_proj.weight"),
            activated,
            output,
        )
        .await?;

    if output.len() != hidden_size {
        return Err(TensorLoadError::integrity(format!(
            "Gemma dense MLP down output length {} must match hidden size {hidden_size}",
            output.len()
        )));
    }
    Ok(())
}

fn gemma_intermediate_size_for_layer(spec: &GemmaModelSpec, layer_idx: usize) -> usize {
    let multiplier = if spec.use_double_wide_mlp && spec.is_kv_shared_layer(layer_idx) {
        2
    } else {
        1
    };
    spec.intermediate_size as usize * multiplier
}
