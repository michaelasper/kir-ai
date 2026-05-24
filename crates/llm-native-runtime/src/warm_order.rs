use llm_backend::native::{SafeTensorShardStore, TensorLoadError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NativeTextWarmableBf16MatrixTensor {
    pub(crate) name: String,
    pub(crate) rows: usize,
    pub(crate) columns: usize,
    pub(crate) byte_len: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct NativeTextWeightWarmOrder {
    stage: u8,
    layer: usize,
    item: u8,
}

pub(crate) fn native_text_warmable_bf16_matrix_tensors(
    store: &SafeTensorShardStore,
) -> Result<Vec<NativeTextWarmableBf16MatrixTensor>, TensorLoadError> {
    let mut tensors = Vec::new();
    for name in store.tensor_names() {
        let metadata = store.tensor_metadata(name)?;
        if metadata.dtype == "BF16" && metadata.shape.len() == 2 {
            tensors.push(NativeTextWarmableBf16MatrixTensor {
                name: name.to_owned(),
                rows: metadata.shape[0],
                columns: metadata.shape[1],
                byte_len: metadata.byte_len as u64,
            });
        }
    }
    tensors.sort_by(|left, right| {
        native_text_bf16_matrix_warm_order(&left.name)
            .cmp(&native_text_bf16_matrix_warm_order(&right.name))
            .then_with(|| left.name.cmp(&right.name))
    });
    Ok(tensors)
}

fn native_text_bf16_matrix_warm_order(name: &str) -> NativeTextWeightWarmOrder {
    let root = name
        .strip_prefix("model.language_model.")
        .or_else(|| name.strip_prefix("model."));
    if matches!(
        root,
        Some("embed_tokens.weight" | "embed_tokens_per_layer.weight")
    ) {
        return NativeTextWeightWarmOrder {
            stage: 0,
            layer: 0,
            item: 0,
        };
    }
    if matches!(root, Some("norm.weight")) || name == "lm_head.weight" {
        return NativeTextWeightWarmOrder {
            stage: 3,
            layer: 0,
            item: 0,
        };
    }
    if matches!(
        root,
        Some("per_layer_model_projection.weight" | "per_layer_projection_norm.weight")
    ) {
        return NativeTextWeightWarmOrder {
            stage: 0,
            layer: 0,
            item: 1,
        };
    }
    let Some(layer_suffix) = root.and_then(|root| root.strip_prefix("layers.")) else {
        return native_text_unknown_weight_warm_order();
    };
    let Some((layer, suffix)) = layer_suffix.split_once('.') else {
        return native_text_unknown_weight_warm_order();
    };
    let Ok(layer) = layer.parse::<usize>() else {
        return native_text_unknown_weight_warm_order();
    };
    let Some((stage, item)) = native_text_layer_bf16_matrix_warm_order(suffix) else {
        return native_text_unknown_weight_warm_order();
    };
    NativeTextWeightWarmOrder { stage, layer, item }
}

fn native_text_layer_bf16_matrix_warm_order(suffix: &str) -> Option<(u8, u8)> {
    let item = match suffix {
        "self_attn.q_proj.weight" | "linear_attn.in_proj_qkv.weight" => 0,
        "self_attn.k_proj.weight" | "linear_attn.in_proj_z.weight" => 1,
        "self_attn.v_proj.weight" | "linear_attn.in_proj_b.weight" => 2,
        "self_attn.o_proj.weight" | "linear_attn.in_proj_a.weight" => 3,
        "linear_attn.out_proj.weight" => 4,
        "input_layernorm.weight" => 5,
        "post_attention_layernorm.weight" => 6,
        "pre_feedforward_layernorm.weight" => 7,
        "post_feedforward_layernorm.weight" => 8,
        "mlp.gate.weight" => 10,
        "mlp.gate_proj.weight" => 10,
        "mlp.up_proj.weight" => 11,
        "mlp.down_proj.weight" => 12,
        "mlp.shared_expert.gate_proj.weight" => 11,
        "mlp.shared_expert.up_proj.weight" => 12,
        "mlp.shared_expert.down_proj.weight" => 13,
        "mlp.shared_expert_gate.weight" => 14,
        "input_gate.weight" => 20,
        "post_per_layer_input_norm.weight" => 21,
        _ => return None,
    };
    Some((1, item))
}

fn native_text_unknown_weight_warm_order() -> NativeTextWeightWarmOrder {
    NativeTextWeightWarmOrder {
        stage: 4,
        layer: usize::MAX,
        item: u8::MAX,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_text_warm_order_recognizes_gemma_text_roots() {
        let mut names = [
            "zz.unclassified.weight",
            "model.norm.weight",
            "model.layers.2.mlp.down_proj.weight",
            "model.layers.2.self_attn.q_proj.weight",
            "model.layers.2.input_gate.weight",
            "model.embed_tokens_per_layer.weight",
            "model.embed_tokens.weight",
            "model.per_layer_model_projection.weight",
            "model.language_model.layers.1.self_attn.o_proj.weight",
            "model.language_model.layers.1.mlp.gate_proj.weight",
        ];

        names.sort_by(|left, right| {
            native_text_bf16_matrix_warm_order(left)
                .cmp(&native_text_bf16_matrix_warm_order(right))
                .then_with(|| left.cmp(right))
        });

        assert_eq!(
            names,
            [
                "model.embed_tokens.weight",
                "model.embed_tokens_per_layer.weight",
                "model.per_layer_model_projection.weight",
                "model.language_model.layers.1.self_attn.o_proj.weight",
                "model.language_model.layers.1.mlp.gate_proj.weight",
                "model.layers.2.self_attn.q_proj.weight",
                "model.layers.2.mlp.down_proj.weight",
                "model.layers.2.input_gate.weight",
                "model.norm.weight",
                "zz.unclassified.weight",
            ]
        );
    }
}
