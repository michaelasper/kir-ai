use super::super::super::math::rms_norm_f32_in_place;
use super::super::super::{SafeTensorShardStore, TensorLoadError};
use llm_models::GemmaModelSpec;

pub(crate) fn gemma_layer_norm_sequence(
    store: &SafeTensorShardStore,
    spec: &GemmaModelSpec,
    layer_idx: usize,
    suffix: &str,
    hidden_states: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    gemma_norm_sequence_after_projection(store, spec, layer_idx, suffix, hidden_states)
}

pub(crate) fn gemma_norm_sequence_after_projection(
    store: &SafeTensorShardStore,
    spec: &GemmaModelSpec,
    layer_idx: usize,
    suffix: &str,
    hidden_states: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    let hidden_size = spec.hidden_size as usize;
    let norm_weight = store.bf16_tensor_f32_cached_arc(&spec.layer_tensor(layer_idx, suffix))?;
    hidden_states
        .iter()
        .map(|hidden| {
            if hidden.len() != hidden_size {
                return Err(TensorLoadError::integrity(format!(
                    "Gemma hidden length {} must match hidden size {hidden_size}",
                    hidden.len()
                )));
            }
            let mut output = vec![0.0; hidden_size];
            rms_norm_f32_in_place(hidden, norm_weight.as_ref(), spec.rms_norm_eps, &mut output)
                .map_err(|err| {
                    TensorLoadError::integrity(format!("Gemma layer RMSNorm failed: {err}"))
                })?;
            Ok(output)
        })
        .collect()
}

pub(crate) fn add_sequence(
    left: &[Vec<f32>],
    right: &[Vec<f32>],
    hidden_size: usize,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    if left.len() != right.len() {
        return Err(TensorLoadError::integrity(
            "Gemma residual sequence lengths must match",
        ));
    }
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            if left.len() != hidden_size || right.len() != hidden_size {
                return Err(TensorLoadError::integrity(format!(
                    "Gemma residual hidden lengths {}, {} must match hidden size {hidden_size}",
                    left.len(),
                    right.len()
                )));
            }
            Ok(left
                .iter()
                .zip(right)
                .map(|(left, right)| left + right)
                .collect())
        })
        .collect()
}

pub(crate) fn apply_gemma_layer_scalar(
    store: &SafeTensorShardStore,
    spec: &GemmaModelSpec,
    layer_idx: usize,
    hidden_states: &mut [Vec<f32>],
) -> Result<(), TensorLoadError> {
    let scalar = store.bf16_tensor_f32_cached_arc(&spec.layer_tensor(layer_idx, "layer_scalar"))?;
    match scalar.as_ref() {
        [value] => {
            for hidden in hidden_states {
                for item in hidden {
                    *item *= *value;
                }
            }
            Ok(())
        }
        values if values.len() == spec.hidden_size as usize => {
            for hidden in hidden_states {
                for (item, scale) in hidden.iter_mut().zip(values) {
                    *item *= scale;
                }
            }
            Ok(())
        }
        values => Err(TensorLoadError::integrity(format!(
            "Gemma layer scalar length {} must be 1 or hidden size {}",
            values.len(),
            spec.hidden_size
        ))),
    }
}
