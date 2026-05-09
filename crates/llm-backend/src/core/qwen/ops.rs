use super::super::math::{
    MathError, TopKLogit, TopKWeight, apply_rope_to_head, require_len, rms_norm_one_centered_f32,
    sigmoid_f32, silu_f32, softplus_f32,
};
use super::super::{
    CpuNativeMatvecBackend, LayerKvCache, LinearAttentionCache, NativeFullAttentionDims,
    NativeFullAttentionSequenceParts, NativeFullAttentionStepParts, NativeMatvecBackend,
    SafeTensorShardStore, TensorLoadError, native_full_attention_sequence_from_parts_with_matvec,
    native_full_attention_sequence_with_cache_from_parts_with_matvec,
    native_full_attention_step_with_cache_from_parts_with_matvec,
};
use super::matvec::{
    l2_normalize_f32_with_matvec, l2_normalize_f32_with_matvec_and_weight_scratch,
    rms_norm_f32_with_matvec,
};
use llm_models::{AttentionKind, QwenModelSpec};

mod lm_head;
mod moe;

use moe::qwen_layer_feed_forward_with_matvec;
pub use moe::{
    qwen_layer_moe_forward, qwen_layer_moe_forward_with_matvec, qwen_layer_moe_router,
    qwen_layer_moe_router_with_matvec, qwen_layer0_moe_forward, qwen_layer0_moe_router,
};

pub use lm_head::{
    qwen_final_norm, qwen_final_norm_for_spec, qwen_final_norm_for_spec_with_matvec,
    qwen_final_norm_with_matvec, qwen_lm_head_logits, qwen_lm_head_logits_for_spec,
    qwen_lm_head_logits_for_spec_with_matvec, qwen_lm_head_logits_with_matvec, qwen_lm_head_top_k,
    qwen_lm_head_top_k_for_spec, qwen_lm_head_top_k_for_spec_with_matvec,
    qwen_lm_head_top_k_with_matvec,
};
use lm_head::{qwen_layer_tensor, qwen_linear_attn_tensor, qwen_mlp_tensor};

pub const QWEN_EMBED_TOKENS_WEIGHT: &str = "model.language_model.embed_tokens.weight";
pub const QWEN_LAYER0_INPUT_NORM_WEIGHT: &str =
    "model.language_model.layers.0.input_layernorm.weight";
pub const QWEN_LAYER0_LINEAR_IN_PROJ_QKV_WEIGHT: &str =
    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
pub const QWEN_LAYER0_LINEAR_IN_PROJ_Z_WEIGHT: &str =
    "model.language_model.layers.0.linear_attn.in_proj_z.weight";
pub const QWEN_LAYER0_LINEAR_IN_PROJ_B_WEIGHT: &str =
    "model.language_model.layers.0.linear_attn.in_proj_b.weight";
pub const QWEN_LAYER0_LINEAR_IN_PROJ_A_WEIGHT: &str =
    "model.language_model.layers.0.linear_attn.in_proj_a.weight";
pub const QWEN_LAYER0_LINEAR_CONV1D_WEIGHT: &str =
    "model.language_model.layers.0.linear_attn.conv1d.weight";
pub const QWEN_LAYER0_LINEAR_NORM_WEIGHT: &str =
    "model.language_model.layers.0.linear_attn.norm.weight";
pub const QWEN_LAYER0_LINEAR_OUT_PROJ_WEIGHT: &str =
    "model.language_model.layers.0.linear_attn.out_proj.weight";
pub const QWEN_LAYER0_POST_ATTENTION_NORM_WEIGHT: &str =
    "model.language_model.layers.0.post_attention_layernorm.weight";
pub const QWEN_LAYER0_MLP_GATE_WEIGHT: &str = "model.language_model.layers.0.mlp.gate.weight";
pub const QWEN_LAYER0_MLP_EXPERTS_GATE_UP_PROJ: &str =
    "model.language_model.layers.0.mlp.experts.gate_up_proj";
pub const QWEN_LAYER0_MLP_EXPERTS_DOWN_PROJ: &str =
    "model.language_model.layers.0.mlp.experts.down_proj";
pub const QWEN_LAYER0_MLP_SHARED_GATE_PROJ_WEIGHT: &str =
    "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight";
pub const QWEN_LAYER0_MLP_SHARED_UP_PROJ_WEIGHT: &str =
    "model.language_model.layers.0.mlp.shared_expert.up_proj.weight";
pub const QWEN_LAYER0_MLP_SHARED_DOWN_PROJ_WEIGHT: &str =
    "model.language_model.layers.0.mlp.shared_expert.down_proj.weight";
pub const QWEN_LAYER0_MLP_SHARED_EXPERT_GATE_WEIGHT: &str =
    "model.language_model.layers.0.mlp.shared_expert_gate.weight";
pub const QWEN_FINAL_NORM_WEIGHT: &str = "model.language_model.norm.weight";
pub const QWEN_LM_HEAD_WEIGHT: &str = "lm_head.weight";

#[derive(Debug, Clone, PartialEq)]
pub struct QwenEmbeddingProbe {
    pub token_id: usize,
    pub embedding: Vec<f32>,
    pub normalized: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QwenLinearAttentionProjectionProbe {
    pub qkv: Vec<f32>,
    pub z: Vec<f32>,
    pub b: Vec<f32>,
    pub a: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QwenLinearAttentionProjectionSequence {
    pub qkv: Vec<Vec<f32>>,
    pub z: Vec<Vec<f32>>,
    pub b: Vec<Vec<f32>>,
    pub a: Vec<Vec<f32>>,
}

pub struct QwenLinearAttentionFirstTokenParts<'a> {
    pub qkv: &'a [f32],
    pub z: &'a [f32],
    pub b: &'a [f32],
    pub conv1d_weight: &'a [f32],
    pub norm_weight: &'a [f32],
    pub out_proj_weight: &'a [f32],
}

pub struct QwenLinearAttentionSequenceParts<'a> {
    pub qkv: &'a [Vec<f32>],
    pub z: &'a [Vec<f32>],
    pub b: &'a [Vec<f32>],
    pub a: &'a [Vec<f32>],
    pub dt_bias: &'a [f32],
    pub a_log: &'a [f32],
    pub conv1d_weight: &'a [f32],
    pub norm_weight: &'a [f32],
    pub out_proj_weight: &'a [f32],
}

pub struct QwenLinearAttentionStepParts<'a> {
    pub qkv: &'a [f32],
    pub z: &'a [f32],
    pub b: &'a [f32],
    pub a: &'a [f32],
    pub dt_bias: &'a [f32],
    pub a_log: &'a [f32],
    pub conv1d_weight: &'a [f32],
    pub norm_weight: &'a [f32],
    pub out_proj_weight: &'a [f32],
}

pub struct QwenFullAttentionSequenceParts<'a> {
    pub q_proj: &'a [Vec<f32>],
    pub k_proj: &'a [Vec<f32>],
    pub v_proj: &'a [Vec<f32>],
    pub q_norm_weight: &'a [f32],
    pub k_norm_weight: &'a [f32],
    pub o_proj_weight: &'a [f32],
}

pub struct QwenFullAttentionStepParts<'a> {
    pub q_proj: &'a [f32],
    pub k_proj: &'a [f32],
    pub v_proj: &'a [f32],
    pub q_norm_weight: &'a [f32],
    pub k_norm_weight: &'a [f32],
    pub o_proj_weight: &'a [f32],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QwenFullAttentionSequenceConfig {
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    pub q_projection_gate: bool,
    pub one_centered_rms_norm: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QwenMoeRouterProbe {
    pub logits: Vec<f32>,
    pub selected: Vec<TopKWeight>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenMoeDims {
    pub hidden_size: usize,
    pub num_experts: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenFullAttentionDims {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
}

impl QwenFullAttentionDims {
    pub fn from_spec(spec: &QwenModelSpec) -> Self {
        Self {
            hidden_size: spec.hidden_size as usize,
            num_attention_heads: spec.num_attention_heads as usize,
            num_key_value_heads: spec.num_key_value_heads as usize,
            head_dim: spec.head_dim as usize,
        }
    }

    fn native(self) -> NativeFullAttentionDims {
        NativeFullAttentionDims {
            hidden_size: self.hidden_size,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
        }
    }

    fn attention_dim(&self) -> Result<usize, MathError> {
        self.num_attention_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| MathError::InvalidShape("Qwen attention dimension overflow".to_owned()))
    }

    fn key_value_dim(&self) -> Result<usize, MathError> {
        self.num_key_value_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| MathError::InvalidShape("Qwen KV dimension overflow".to_owned()))
    }
}

impl QwenMoeDims {
    pub fn from_spec(spec: &QwenModelSpec) -> Self {
        Self {
            hidden_size: spec.hidden_size as usize,
            num_experts: spec.num_experts as usize,
            moe_intermediate_size: spec.moe_intermediate_size as usize,
            shared_expert_intermediate_size: spec.shared_expert_intermediate_size as usize,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QwenLinearAttentionDims {
    pub hidden_size: usize,
    pub num_key_heads: usize,
    pub num_value_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub conv_kernel_size: usize,
    pub rms_norm_eps: f32,
}

impl QwenLinearAttentionDims {
    pub fn from_spec(spec: &QwenModelSpec) -> Self {
        Self {
            hidden_size: spec.hidden_size as usize,
            num_key_heads: spec.linear_num_key_heads as usize,
            num_value_heads: spec.linear_num_value_heads as usize,
            key_head_dim: spec.linear_key_head_dim as usize,
            value_head_dim: spec.linear_value_head_dim as usize,
            conv_kernel_size: spec.linear_conv_kernel_dim as usize,
            rms_norm_eps: spec.rms_norm_eps,
        }
    }

    fn key_dim(&self) -> Result<usize, MathError> {
        self.num_key_heads
            .checked_mul(self.key_head_dim)
            .ok_or_else(|| MathError::InvalidShape("Qwen key dimension overflow".to_owned()))
    }

    fn value_dim(&self) -> Result<usize, MathError> {
        self.num_value_heads
            .checked_mul(self.value_head_dim)
            .ok_or_else(|| MathError::InvalidShape("Qwen value dimension overflow".to_owned()))
    }

    fn conv_dim(&self) -> Result<usize, MathError> {
        let key_dim = self.key_dim()?;
        let value_dim = self.value_dim()?;
        key_dim
            .checked_mul(2)
            .and_then(|key_parts| key_parts.checked_add(value_dim))
            .ok_or_else(|| MathError::InvalidShape("Qwen conv dimension overflow".to_owned()))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum QwenLayerCache {
    Linear(LinearAttentionCache),
    Full(LayerKvCache),
}

pub fn qwen_layer_caches_for_spec(
    spec: &QwenModelSpec,
    max_tokens: usize,
) -> Result<Vec<QwenLayerCache>, TensorLoadError> {
    let layer_count = spec.num_hidden_layers as usize;
    if spec.layer_kinds.len() != layer_count {
        return Err(TensorLoadError::integrity(format!(
            "Qwen spec declares {layer_count} layers but has {} attention kind entries",
            spec.layer_kinds.len()
        )));
    }
    spec.layer_kinds
        .iter()
        .enumerate()
        .map(|(layer_idx, kind)| qwen_layer_cache_for_kind(spec, layer_idx, *kind, max_tokens))
        .collect()
}

fn qwen_layer_cache_for_kind(
    spec: &QwenModelSpec,
    layer_idx: usize,
    kind: AttentionKind,
    max_tokens: usize,
) -> Result<QwenLayerCache, TensorLoadError> {
    match kind {
        AttentionKind::LinearAttention => {
            let dims = QwenLinearAttentionDims::from_spec(spec);
            let conv_dim = dims.conv_dim().map_err(|err| {
                TensorLoadError::integrity(format!(
                    "Qwen layer{layer_idx} linear cache shape failed: {err}"
                ))
            })?;
            LinearAttentionCache::new(
                dims.conv_kernel_size,
                conv_dim,
                dims.num_value_heads,
                dims.key_head_dim,
                dims.value_head_dim,
            )
            .map(QwenLayerCache::Linear)
            .map_err(|err| {
                TensorLoadError::integrity(format!(
                    "Qwen layer{layer_idx} linear cache allocation failed: {err}"
                ))
            })
        }
        AttentionKind::FullAttention => {
            let dims = QwenFullAttentionDims::from_spec(spec);
            LayerKvCache::new(max_tokens, dims.num_key_value_heads, dims.head_dim)
                .map(QwenLayerCache::Full)
                .map_err(|err| {
                    TensorLoadError::integrity(format!(
                        "Qwen layer{layer_idx} full attention cache allocation failed: {err}"
                    ))
                })
        }
    }
}

pub fn qwen_embedding_and_layer0_norm(
    store: &SafeTensorShardStore,
    token_id: usize,
    hidden_size: usize,
    rms_norm_eps: f32,
) -> Result<QwenEmbeddingProbe, TensorLoadError> {
    let embedding = store.bf16_row_f32(QWEN_EMBED_TOKENS_WEIGHT, token_id)?;
    if embedding.len() != hidden_size {
        return Err(TensorLoadError::integrity(format!(
            "Qwen embedding row has length {}, expected hidden size {hidden_size}",
            embedding.len()
        )));
    }
    let norm_weight = store.bf16_tensor_f32_range(QWEN_LAYER0_INPUT_NORM_WEIGHT, 0, hidden_size)?;
    let normalized =
        rms_norm_one_centered_f32(&embedding, &norm_weight, rms_norm_eps).map_err(|err| {
            TensorLoadError::integrity(format!("Qwen layer0 input RMSNorm failed: {err}"))
        })?;
    Ok(QwenEmbeddingProbe {
        token_id,
        embedding,
        normalized,
    })
}

pub fn qwen_embedding_sequence(
    store: &SafeTensorShardStore,
    token_ids: &[usize],
    hidden_size: usize,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    token_ids
        .iter()
        .map(|token_id| {
            let embedding = store.bf16_row_f32(QWEN_EMBED_TOKENS_WEIGHT, *token_id)?;
            if embedding.len() != hidden_size {
                return Err(TensorLoadError::integrity(format!(
                    "Qwen embedding row has length {}, expected hidden size {hidden_size}",
                    embedding.len()
                )));
            }
            Ok(embedding)
        })
        .collect()
}

pub fn qwen_embedding_sequence_for_spec(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    token_ids: &[usize],
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    let hidden_size = spec.hidden_size as usize;
    let tensor = spec.embed_tokens_weight();
    token_ids
        .iter()
        .map(|token_id| {
            let embedding = store.bf16_row_f32(&tensor, *token_id)?;
            if embedding.len() != hidden_size {
                return Err(TensorLoadError::integrity(format!(
                    "Qwen embedding row has length {}, expected hidden size {hidden_size}",
                    embedding.len()
                )));
            }
            Ok(embedding)
        })
        .collect()
}

pub async fn qwen_layer_input_norm(
    store: &SafeTensorShardStore,
    layer_idx: usize,
    hidden_states: &[f32],
    hidden_size: usize,
    rms_norm_eps: f32,
) -> Result<Vec<f32>, TensorLoadError> {
    qwen_layer_input_norm_with_matvec(
        store,
        layer_idx,
        hidden_states,
        hidden_size,
        rms_norm_eps,
        &CpuNativeMatvecBackend,
    )
    .await
}

pub async fn qwen_layer_input_norm_with_matvec(
    store: &SafeTensorShardStore,
    layer_idx: usize,
    hidden_states: &[f32],
    hidden_size: usize,
    rms_norm_eps: f32,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    if hidden_states.len() != hidden_size {
        return Err(TensorLoadError::integrity(format!(
            "Qwen layer input hidden length {} must match hidden size {hidden_size}",
            hidden_states.len()
        )));
    }
    let norm_weight = store.bf16_tensor_f32_range(
        &qwen_layer_tensor(layer_idx, "input_layernorm.weight"),
        0,
        hidden_size,
    )?;
    matvec
        .rms_norm_one_centered_f32(hidden_states, &norm_weight, rms_norm_eps)
        .await
        .map_err(|err| {
            TensorLoadError::integrity(format!("Qwen layer input RMSNorm failed: {err}"))
        })
}

async fn qwen_layer_input_norm_for_spec_with_matvec(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[f32],
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    let hidden_size = spec.hidden_size as usize;
    if hidden_states.len() != hidden_size {
        return Err(TensorLoadError::integrity(format!(
            "Qwen layer input hidden length {} must match hidden size {hidden_size}",
            hidden_states.len()
        )));
    }
    let norm_weight = store.bf16_tensor_f32_range(
        &spec.layer_tensor(layer_idx, "input_layernorm.weight"),
        0,
        hidden_size,
    )?;
    qwen_rms_norm_for_spec_with_matvec(spec, hidden_states, &norm_weight, matvec)
        .await
        .map_err(|err| {
            TensorLoadError::integrity(format!("Qwen layer input RMSNorm failed: {err}"))
        })
}

async fn qwen_layer_input_norm_sequence_for_spec(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[Vec<f32>],
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    let hidden_size = spec.hidden_size as usize;
    let norm_weight = store.bf16_tensor_f32_range(
        &spec.layer_tensor(layer_idx, "input_layernorm.weight"),
        0,
        hidden_size,
    )?;
    let mut results = Vec::with_capacity(hidden_states.len());
    for hidden in hidden_states {
        if hidden.len() != hidden_size {
            return Err(TensorLoadError::integrity(format!(
                "Qwen layer input hidden length {} must match hidden size {hidden_size}",
                hidden.len()
            )));
        }
        results.push(
            qwen_rms_norm_for_spec_with_matvec(spec, hidden, &norm_weight, matvec)
                .await
                .map_err(|err| {
                    TensorLoadError::integrity(format!(
                        "Qwen layer input RMSNorm sequence failed: {err}"
                    ))
                })?,
        );
    }
    Ok(results)
}

async fn qwen_rms_norm_for_spec_with_matvec(
    spec: &QwenModelSpec,
    input: &[f32],
    weight: &[f32],
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, MathError> {
    if spec.is_qwen3_dense() {
        rms_norm_f32_with_matvec(input, weight, spec.rms_norm_eps, matvec).await
    } else {
        matvec
            .rms_norm_one_centered_f32(input, weight, spec.rms_norm_eps)
            .await
    }
}

async fn qwen_attention_rms_norm_with_matvec(
    input: &[f32],
    weight: &[f32],
    config: QwenFullAttentionSequenceConfig,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, MathError> {
    if config.one_centered_rms_norm {
        matvec
            .rms_norm_one_centered_f32(input, weight, config.rms_norm_eps)
            .await
    } else {
        rms_norm_f32_with_matvec(input, weight, config.rms_norm_eps, matvec).await
    }
}

pub async fn qwen_linear_attention_first_token_from_parts(
    dims: &QwenLinearAttentionDims,
    qkv: &[f32],
    z: &[f32],
    b: &[f32],
    conv1d_weight: &[f32],
    norm_weight: &[f32],
    out_proj_weight: &[f32],
) -> Result<Vec<f32>, MathError> {
    let parts = QwenLinearAttentionFirstTokenParts {
        qkv,
        z,
        b,
        conv1d_weight,
        norm_weight,
        out_proj_weight,
    };
    qwen_linear_attention_first_token_from_parts_with_matvec(dims, &parts, &CpuNativeMatvecBackend)
        .await
}

pub async fn qwen_linear_attention_first_token_from_parts_with_matvec(
    dims: &QwenLinearAttentionDims,
    parts: &QwenLinearAttentionFirstTokenParts<'_>,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, MathError> {
    let qkv = parts.qkv;
    let z = parts.z;
    let b = parts.b;
    let conv1d_weight = parts.conv1d_weight;
    let norm_weight = parts.norm_weight;
    let out_proj_weight = parts.out_proj_weight;
    if dims.num_key_heads == 0
        || dims.num_value_heads == 0
        || dims.key_head_dim == 0
        || dims.value_head_dim == 0
        || dims.conv_kernel_size == 0
    {
        return Err(MathError::InvalidShape(
            "Qwen linear attention dimensions must be non-zero".to_owned(),
        ));
    }
    if !dims.num_value_heads.is_multiple_of(dims.num_key_heads) {
        return Err(MathError::InvalidShape(
            "Qwen value heads must be divisible by key heads".to_owned(),
        ));
    }
    let key_dim = dims.key_dim()?;
    let value_dim = dims.value_dim()?;
    let conv_dim = dims.conv_dim()?;
    require_len("qkv projection", qkv.len(), conv_dim)?;
    require_len("z projection", z.len(), value_dim)?;
    require_len("b projection", b.len(), dims.num_value_heads)?;
    require_len("norm weight", norm_weight.len(), dims.value_head_dim)?;
    require_len(
        "conv1d weight",
        conv1d_weight.len(),
        conv_dim
            .checked_mul(dims.conv_kernel_size)
            .ok_or_else(|| MathError::InvalidShape("conv1d weight shape overflow".to_owned()))?,
    )?;
    require_len(
        "out projection weight",
        out_proj_weight.len(),
        dims.hidden_size
            .checked_mul(value_dim)
            .ok_or_else(|| MathError::InvalidShape("out projection shape overflow".to_owned()))?,
    )?;

    let mut conv_window = vec![0.0; conv_dim * dims.conv_kernel_size];
    let current_start = (dims.conv_kernel_size - 1) * conv_dim;
    conv_window[current_start..current_start + conv_dim].copy_from_slice(qkv);
    let mixed_qkv = matvec
        .linear_attention_conv1d_silu_f32(
            &conv_window,
            conv1d_weight,
            conv_dim,
            dims.conv_kernel_size,
        )
        .await?;

    let query = &mixed_qkv[..key_dim];
    let key = &mixed_qkv[key_dim..key_dim * 2];
    let value = &mixed_qkv[key_dim * 2..];
    let repeat = dims.num_value_heads / dims.num_key_heads;
    let scale = (dims.key_head_dim as f32).sqrt().recip();
    let mut gated = vec![0.0; value_dim];

    for (value_head, beta_logit) in b.iter().enumerate().take(dims.num_value_heads) {
        let key_head = value_head / repeat;
        let key_start = key_head * dims.key_head_dim;
        let value_start = value_head * dims.value_head_dim;
        let query_head = l2_normalize_f32_with_matvec(
            &query[key_start..key_start + dims.key_head_dim],
            1e-6,
            matvec,
        )
        .await?;
        let key_head_values = l2_normalize_f32_with_matvec(
            &key[key_start..key_start + dims.key_head_dim],
            1e-6,
            matvec,
        )
        .await?;
        let attention_score = query_head
            .iter()
            .zip(&key_head_values)
            .map(|(query, key)| query * key)
            .sum::<f32>()
            * scale;
        let beta = sigmoid_f32(*beta_logit);
        let mut core_head = Vec::with_capacity(dims.value_head_dim);
        for offset in 0..dims.value_head_dim {
            core_head.push(attention_score * value[value_start + offset] * beta);
        }
        let normalized =
            rms_norm_f32_with_matvec(&core_head, norm_weight, dims.rms_norm_eps, matvec).await?;
        for offset in 0..dims.value_head_dim {
            gated[value_start + offset] = normalized[offset] * silu_f32(z[value_start + offset]);
        }
    }

    matvec
        .matvec_row_major_f32(&gated, out_proj_weight, dims.hidden_size, value_dim)
        .await
}

pub async fn qwen_linear_attention_sequence_from_parts(
    dims: &QwenLinearAttentionDims,
    parts: &QwenLinearAttentionSequenceParts<'_>,
) -> Result<Vec<Vec<f32>>, MathError> {
    qwen_linear_attention_sequence_from_parts_with_matvec(dims, parts, &CpuNativeMatvecBackend)
        .await
}

pub async fn qwen_linear_attention_sequence_from_parts_with_matvec(
    dims: &QwenLinearAttentionDims,
    parts: &QwenLinearAttentionSequenceParts<'_>,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<Vec<f32>>, MathError> {
    qwen_linear_attention_sequence_from_parts_impl(dims, parts, None, matvec).await
}

pub async fn qwen_linear_attention_sequence_with_cache_from_parts(
    dims: &QwenLinearAttentionDims,
    parts: &QwenLinearAttentionSequenceParts<'_>,
    cache: &mut LinearAttentionCache,
) -> Result<Vec<Vec<f32>>, MathError> {
    qwen_linear_attention_sequence_with_cache_from_parts_with_matvec(
        dims,
        parts,
        cache,
        &CpuNativeMatvecBackend,
    )
    .await
}

pub async fn qwen_linear_attention_sequence_with_cache_from_parts_with_matvec(
    dims: &QwenLinearAttentionDims,
    parts: &QwenLinearAttentionSequenceParts<'_>,
    cache: &mut LinearAttentionCache,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<Vec<f32>>, MathError> {
    qwen_linear_attention_sequence_from_parts_impl(dims, parts, Some(cache), matvec).await
}

pub async fn qwen_linear_attention_step_with_cache_from_parts(
    dims: &QwenLinearAttentionDims,
    parts: &QwenLinearAttentionStepParts<'_>,
    cache: &mut LinearAttentionCache,
) -> Result<Vec<f32>, MathError> {
    qwen_linear_attention_step_with_cache_from_parts_with_matvec(
        dims,
        parts,
        cache,
        &CpuNativeMatvecBackend,
    )
    .await
}

pub async fn qwen_linear_attention_step_with_cache_from_parts_with_matvec(
    dims: &QwenLinearAttentionDims,
    parts: &QwenLinearAttentionStepParts<'_>,
    cache: &mut LinearAttentionCache,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, MathError> {
    let qkv = vec![parts.qkv.to_vec()];
    let z = vec![parts.z.to_vec()];
    let b = vec![parts.b.to_vec()];
    let a = vec![parts.a.to_vec()];
    let sequence_parts = QwenLinearAttentionSequenceParts {
        qkv: &qkv,
        z: &z,
        b: &b,
        a: &a,
        dt_bias: parts.dt_bias,
        a_log: parts.a_log,
        conv1d_weight: parts.conv1d_weight,
        norm_weight: parts.norm_weight,
        out_proj_weight: parts.out_proj_weight,
    };
    qwen_linear_attention_sequence_from_parts_impl(dims, &sequence_parts, Some(cache), matvec)
        .await?
        .into_iter()
        .next()
        .ok_or_else(|| {
            MathError::InvalidShape("Qwen linear attention step returned no output".to_owned())
        })
}

async fn qwen_linear_attention_sequence_from_parts_impl(
    dims: &QwenLinearAttentionDims,
    parts: &QwenLinearAttentionSequenceParts<'_>,
    mut cache: Option<&mut LinearAttentionCache>,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<Vec<f32>>, MathError> {
    let qkv = parts.qkv;
    let z = parts.z;
    let b = parts.b;
    let a = parts.a;
    if qkv.is_empty() {
        return Ok(Vec::new());
    }
    if dims.num_key_heads == 0
        || dims.num_value_heads == 0
        || dims.key_head_dim == 0
        || dims.value_head_dim == 0
        || dims.conv_kernel_size == 0
    {
        return Err(MathError::InvalidShape(
            "Qwen linear attention dimensions must be non-zero".to_owned(),
        ));
    }
    if !dims.num_value_heads.is_multiple_of(dims.num_key_heads) {
        return Err(MathError::InvalidShape(
            "Qwen value heads must be divisible by key heads".to_owned(),
        ));
    }
    let seq_len = qkv.len();
    if z.len() != seq_len || b.len() != seq_len || a.len() != seq_len {
        return Err(MathError::InvalidShape(
            "Qwen linear attention sequence inputs must have the same length".to_owned(),
        ));
    }
    let key_dim = dims.key_dim()?;
    let value_dim = dims.value_dim()?;
    let conv_dim = dims.conv_dim()?;
    if let Some(cache) = cache.as_ref() {
        require_linear_attention_cache_shape(dims, conv_dim, cache)?;
    }
    require_len("dt bias", parts.dt_bias.len(), dims.num_value_heads)?;
    require_len("A log", parts.a_log.len(), dims.num_value_heads)?;
    require_len("norm weight", parts.norm_weight.len(), dims.value_head_dim)?;
    require_len(
        "conv1d weight",
        parts.conv1d_weight.len(),
        conv_dim
            .checked_mul(dims.conv_kernel_size)
            .ok_or_else(|| MathError::InvalidShape("conv1d weight shape overflow".to_owned()))?,
    )?;
    require_len(
        "out projection weight",
        parts.out_proj_weight.len(),
        dims.hidden_size
            .checked_mul(value_dim)
            .ok_or_else(|| MathError::InvalidShape("out projection shape overflow".to_owned()))?,
    )?;
    for token_idx in 0..seq_len {
        require_len("qkv projection", qkv[token_idx].len(), conv_dim)?;
        require_len("z projection", z[token_idx].len(), value_dim)?;
        require_len("b projection", b[token_idx].len(), dims.num_value_heads)?;
        require_len("a projection", a[token_idx].len(), dims.num_value_heads)?;
    }

    let mut mixed_tokens = vec![vec![0.0; conv_dim]; seq_len];
    for token_idx in 0..seq_len {
        if let Some(cache) = cache.as_mut() {
            cache.push_conv_input(&qkv[token_idx]).map_err(|err| {
                MathError::InvalidShape(format!("linear attention cache update failed: {err}"))
            })?;
            mixed_tokens[token_idx] = matvec
                .linear_attention_conv1d_silu_f32(
                    cache.conv_window(),
                    parts.conv1d_weight,
                    conv_dim,
                    dims.conv_kernel_size,
                )
                .await?;
        } else {
            let mut conv_window = vec![0.0; conv_dim * dims.conv_kernel_size];
            for kernel_idx in 0..dims.conv_kernel_size {
                let lookback = dims.conv_kernel_size - 1 - kernel_idx;
                if token_idx >= lookback {
                    let window_start = kernel_idx * conv_dim;
                    conv_window[window_start..window_start + conv_dim]
                        .copy_from_slice(&qkv[token_idx - lookback]);
                }
            }
            mixed_tokens[token_idx] = matvec
                .linear_attention_conv1d_silu_f32(
                    &conv_window,
                    parts.conv1d_weight,
                    conv_dim,
                    dims.conv_kernel_size,
                )
                .await?;
        }
    }

    let repeat = dims.num_value_heads / dims.num_key_heads;
    let scale = (dims.key_head_dim as f32).sqrt().recip();
    let mut recurrent_state = cache
        .as_ref()
        .map(|cache| cache.recurrent_state().to_vec())
        .unwrap_or_else(|| {
            vec![0.0; dims.num_value_heads * dims.key_head_dim * dims.value_head_dim]
        });
    let value_major_len = dims
        .value_head_dim
        .checked_mul(dims.key_head_dim)
        .ok_or_else(|| MathError::InvalidShape("value-major state shape overflow".to_owned()))?;
    let mut l2_weight_scratch = Vec::with_capacity(dims.key_head_dim);
    let zero_memory = vec![0.0; dims.value_head_dim];
    let mut value_major_state = Vec::with_capacity(value_major_len);
    let mut query_scaled = vec![0.0; dims.key_head_dim];
    let mut outputs = Vec::with_capacity(seq_len);

    for token_idx in 0..seq_len {
        let mixed_qkv = &mixed_tokens[token_idx];
        let query = &mixed_qkv[..key_dim];
        let key = &mixed_qkv[key_dim..key_dim * 2];
        let value = &mixed_qkv[key_dim * 2..];
        let mut gated = vec![0.0; value_dim];

        for value_head in 0..dims.num_value_heads {
            let key_head = value_head / repeat;
            let key_start = key_head * dims.key_head_dim;
            let value_start = value_head * dims.value_head_dim;
            let query_head = l2_normalize_f32_with_matvec_and_weight_scratch(
                &query[key_start..key_start + dims.key_head_dim],
                1e-6,
                matvec,
                &mut l2_weight_scratch,
            )
            .await?;
            let key_head_values = l2_normalize_f32_with_matvec_and_weight_scratch(
                &key[key_start..key_start + dims.key_head_dim],
                1e-6,
                matvec,
                &mut l2_weight_scratch,
            )
            .await?;
            for (output, value) in query_scaled.iter_mut().zip(&query_head) {
                *output = value * scale;
            }
            let beta = sigmoid_f32(b[token_idx][value_head]);
            let decay = (-parts.a_log[value_head].exp()
                * softplus_f32(a[token_idx][value_head] + parts.dt_bias[value_head]))
            .exp();

            let state_start = value_head * dims.key_head_dim * dims.value_head_dim;
            let state_end = state_start + dims.key_head_dim * dims.value_head_dim;
            let decayed_state = if let Some(cache) = cache.as_ref() {
                matvec
                    .linear_attention_recurrent_cache_update_f32(
                        cache,
                        state_start,
                        &key_head_values,
                        &value[value_start..value_start + dims.value_head_dim],
                        &zero_memory,
                        0.0,
                        decay,
                        dims.key_head_dim,
                        dims.value_head_dim,
                    )
                    .await?
            } else {
                matvec
                    .linear_attention_recurrent_update_f32(
                        &recurrent_state[state_start..state_end],
                        &key_head_values,
                        &value[value_start..value_start + dims.value_head_dim],
                        &zero_memory,
                        0.0,
                        decay,
                        dims.key_head_dim,
                        dims.value_head_dim,
                    )
                    .await?
            };
            recurrent_state[state_start..state_end].copy_from_slice(&decayed_state);
            if let Some(cache) = cache.as_mut() {
                cache
                    .replace_recurrent_state_range(state_start, &decayed_state)
                    .map_err(|err| {
                        MathError::InvalidShape(format!(
                            "linear attention cache update failed: {err}"
                        ))
                    })?;
            }

            copy_linear_attention_value_major_state_rows(
                &recurrent_state,
                state_start,
                dims.key_head_dim,
                dims.value_head_dim,
                &mut value_major_state,
            )?;
            let memory = matvec
                .matvec_row_major_f32(
                    &key_head_values,
                    &value_major_state,
                    dims.value_head_dim,
                    dims.key_head_dim,
                )
                .await?;

            let updated_state = if let Some(cache) = cache.as_ref() {
                matvec
                    .linear_attention_recurrent_cache_update_f32(
                        cache,
                        state_start,
                        &key_head_values,
                        &value[value_start..value_start + dims.value_head_dim],
                        &memory,
                        beta,
                        1.0,
                        dims.key_head_dim,
                        dims.value_head_dim,
                    )
                    .await?
            } else {
                matvec
                    .linear_attention_recurrent_update_f32(
                        &recurrent_state[state_start..state_end],
                        &key_head_values,
                        &value[value_start..value_start + dims.value_head_dim],
                        &memory,
                        beta,
                        1.0,
                        dims.key_head_dim,
                        dims.value_head_dim,
                    )
                    .await?
            };
            recurrent_state[state_start..state_end].copy_from_slice(&updated_state);
            if let Some(cache) = cache.as_mut() {
                cache
                    .replace_recurrent_state_range(state_start, &updated_state)
                    .map_err(|err| {
                        MathError::InvalidShape(format!(
                            "linear attention cache update failed: {err}"
                        ))
                    })?;
            }

            copy_linear_attention_value_major_state_rows(
                &recurrent_state,
                state_start,
                dims.key_head_dim,
                dims.value_head_dim,
                &mut value_major_state,
            )?;
            let core_head = matvec
                .matvec_row_major_f32(
                    &query_scaled,
                    &value_major_state,
                    dims.value_head_dim,
                    dims.key_head_dim,
                )
                .await?;
            let normalized =
                rms_norm_f32_with_matvec(&core_head, parts.norm_weight, dims.rms_norm_eps, matvec)
                    .await?;
            for value_offset in 0..dims.value_head_dim {
                gated[value_start + value_offset] =
                    normalized[value_offset] * silu_f32(z[token_idx][value_start + value_offset]);
            }
        }
        outputs.push(
            matvec
                .matvec_row_major_f32(&gated, parts.out_proj_weight, dims.hidden_size, value_dim)
                .await?,
        );
    }

    Ok(outputs)
}

fn copy_linear_attention_value_major_state_rows(
    recurrent_state: &[f32],
    state_start: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    rows: &mut Vec<f32>,
) -> Result<(), MathError> {
    let row_len = value_head_dim
        .checked_mul(key_head_dim)
        .ok_or_else(|| MathError::InvalidShape("value-major state shape overflow".to_owned()))?;
    let state_end = state_start
        .checked_add(row_len)
        .ok_or_else(|| MathError::InvalidShape("recurrent state offset overflow".to_owned()))?;
    if recurrent_state.len() < state_end {
        return Err(MathError::InvalidShape(format!(
            "recurrent state slice too short: expected at least {state_end}, got {}",
            recurrent_state.len()
        )));
    }
    rows.clear();
    rows.resize(row_len, 0.0);
    for value_offset in 0..value_head_dim {
        for key_offset in 0..key_head_dim {
            rows[value_offset * key_head_dim + key_offset] =
                recurrent_state[state_start + key_offset * value_head_dim + value_offset];
        }
    }
    Ok(())
}

fn require_linear_attention_cache_shape(
    dims: &QwenLinearAttentionDims,
    conv_dim: usize,
    cache: &LinearAttentionCache,
) -> Result<(), MathError> {
    if cache.conv_kernel_size() != dims.conv_kernel_size
        || cache.conv_dim() != conv_dim
        || cache.num_value_heads() != dims.num_value_heads
        || cache.key_head_dim() != dims.key_head_dim
        || cache.value_head_dim() != dims.value_head_dim
    {
        return Err(MathError::InvalidShape(format!(
            "Qwen linear attention cache shape does not match dims: cache conv_kernel_size={}, conv_dim={}, value_heads={}, key_head_dim={}, value_head_dim={}; dims conv_kernel_size={}, conv_dim={}, value_heads={}, key_head_dim={}, value_head_dim={}",
            cache.conv_kernel_size(),
            cache.conv_dim(),
            cache.num_value_heads(),
            cache.key_head_dim(),
            cache.value_head_dim(),
            dims.conv_kernel_size,
            conv_dim,
            dims.num_value_heads,
            dims.key_head_dim,
            dims.value_head_dim
        )));
    }
    Ok(())
}

fn require_full_attention_cache_shape(
    dims: &QwenFullAttentionDims,
    cache: &LayerKvCache,
) -> Result<(), MathError> {
    if cache.key_value_heads() != dims.num_key_value_heads || cache.head_dim() != dims.head_dim {
        return Err(MathError::InvalidShape(format!(
            "Qwen full attention cache shape does not match dims: cache key_value_heads={}, head_dim={}; dims key_value_heads={}, head_dim={}",
            cache.key_value_heads(),
            cache.head_dim(),
            dims.num_key_value_heads,
            dims.head_dim
        )));
    }
    Ok(())
}

pub async fn qwen_full_attention_first_token_from_parts(
    dims: &QwenFullAttentionDims,
    q_proj: &[f32],
    v_proj: &[f32],
    o_proj_weight: &[f32],
) -> Result<Vec<f32>, MathError> {
    qwen_full_attention_first_token_from_parts_with_matvec(
        dims,
        q_proj,
        v_proj,
        o_proj_weight,
        &CpuNativeMatvecBackend,
    )
    .await
}

pub async fn qwen_full_attention_first_token_from_parts_with_matvec(
    dims: &QwenFullAttentionDims,
    q_proj: &[f32],
    v_proj: &[f32],
    o_proj_weight: &[f32],
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, MathError> {
    if dims.num_attention_heads == 0
        || dims.num_key_value_heads == 0
        || dims.head_dim == 0
        || dims.hidden_size == 0
    {
        return Err(MathError::InvalidShape(
            "Qwen full attention dimensions must be non-zero".to_owned(),
        ));
    }
    if !dims
        .num_attention_heads
        .is_multiple_of(dims.num_key_value_heads)
    {
        return Err(MathError::InvalidShape(
            "Qwen attention heads must be divisible by key/value heads".to_owned(),
        ));
    }
    let attention_dim = dims.attention_dim()?;
    let key_value_dim = dims.key_value_dim()?;
    require_len("q projection", q_proj.len(), attention_dim * 2)?;
    require_len("v projection", v_proj.len(), key_value_dim)?;
    require_len(
        "o projection weight",
        o_proj_weight.len(),
        dims.hidden_size
            .checked_mul(attention_dim)
            .ok_or_else(|| MathError::InvalidShape("Qwen o projection overflow".to_owned()))?,
    )?;

    let groups = dims.num_attention_heads / dims.num_key_value_heads;
    let mut gated = vec![0.0; attention_dim];
    for head in 0..dims.num_attention_heads {
        let q_proj_head_start = head * dims.head_dim * 2;
        let gate_start = q_proj_head_start + dims.head_dim;
        let kv_head = head / groups;
        let value_start = kv_head * dims.head_dim;
        let output_start = head * dims.head_dim;
        for offset in 0..dims.head_dim {
            gated[output_start + offset] =
                v_proj[value_start + offset] * sigmoid_f32(q_proj[gate_start + offset]);
        }
    }

    matvec
        .matvec_row_major_f32(&gated, o_proj_weight, dims.hidden_size, attention_dim)
        .await
}

pub async fn qwen_full_attention_sequence_from_parts(
    dims: &QwenFullAttentionDims,
    parts: &QwenFullAttentionSequenceParts<'_>,
    config: QwenFullAttentionSequenceConfig,
) -> Result<Vec<Vec<f32>>, MathError> {
    qwen_full_attention_sequence_from_parts_with_matvec(
        dims,
        parts,
        config,
        &CpuNativeMatvecBackend,
    )
    .await
}

pub async fn qwen_full_attention_sequence_from_parts_with_matvec(
    dims: &QwenFullAttentionDims,
    parts: &QwenFullAttentionSequenceParts<'_>,
    config: QwenFullAttentionSequenceConfig,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<Vec<f32>>, MathError> {
    qwen_full_attention_sequence_from_parts_impl(dims, parts, config, None, matvec).await
}

pub async fn qwen_full_attention_sequence_with_cache_from_parts(
    dims: &QwenFullAttentionDims,
    parts: &QwenFullAttentionSequenceParts<'_>,
    config: QwenFullAttentionSequenceConfig,
    cache: &mut LayerKvCache,
) -> Result<Vec<Vec<f32>>, MathError> {
    qwen_full_attention_sequence_with_cache_from_parts_with_matvec(
        dims,
        parts,
        config,
        cache,
        &CpuNativeMatvecBackend,
    )
    .await
}

pub async fn qwen_full_attention_sequence_with_cache_from_parts_with_matvec(
    dims: &QwenFullAttentionDims,
    parts: &QwenFullAttentionSequenceParts<'_>,
    config: QwenFullAttentionSequenceConfig,
    cache: &mut LayerKvCache,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<Vec<f32>>, MathError> {
    qwen_full_attention_sequence_from_parts_impl(dims, parts, config, Some(cache), matvec).await
}

pub async fn qwen_full_attention_step_with_cache_from_parts(
    dims: &QwenFullAttentionDims,
    parts: &QwenFullAttentionStepParts<'_>,
    config: QwenFullAttentionSequenceConfig,
    cache: &mut LayerKvCache,
) -> Result<Vec<f32>, MathError> {
    qwen_full_attention_step_with_cache_from_parts_with_matvec(
        dims,
        parts,
        config,
        cache,
        &CpuNativeMatvecBackend,
    )
    .await
}

pub async fn qwen_full_attention_step_with_cache_from_parts_with_matvec(
    dims: &QwenFullAttentionDims,
    parts: &QwenFullAttentionStepParts<'_>,
    config: QwenFullAttentionSequenceConfig,
    cache: &mut LayerKvCache,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, MathError> {
    if dims.num_attention_heads == 0
        || dims.num_key_value_heads == 0
        || dims.head_dim == 0
        || dims.hidden_size == 0
    {
        return Err(MathError::InvalidShape(
            "Qwen full attention dimensions must be non-zero".to_owned(),
        ));
    }
    if !dims
        .num_attention_heads
        .is_multiple_of(dims.num_key_value_heads)
    {
        return Err(MathError::InvalidShape(
            "Qwen attention heads must be divisible by key/value heads".to_owned(),
        ));
    }
    if config.rope_theta <= 0.0 || config.partial_rotary_factor < 0.0 {
        return Err(MathError::InvalidShape(
            "Qwen RoPE parameters must be positive".to_owned(),
        ));
    }
    require_full_attention_cache_shape(dims, cache)?;
    let attention_dim = dims.attention_dim()?;
    let key_value_dim = dims.key_value_dim()?;
    let q_projection_width = if config.q_projection_gate {
        attention_dim
            .checked_mul(2)
            .ok_or_else(|| MathError::InvalidShape("Qwen q projection overflow".to_owned()))?
    } else {
        attention_dim
    };
    require_len("q projection", parts.q_proj.len(), q_projection_width)?;
    require_len("k projection", parts.k_proj.len(), key_value_dim)?;
    require_len("v projection", parts.v_proj.len(), key_value_dim)?;
    require_len("q norm weight", parts.q_norm_weight.len(), dims.head_dim)?;
    require_len("k norm weight", parts.k_norm_weight.len(), dims.head_dim)?;
    require_len(
        "o projection weight",
        parts.o_proj_weight.len(),
        dims.hidden_size
            .checked_mul(attention_dim)
            .ok_or_else(|| MathError::InvalidShape("Qwen o projection overflow".to_owned()))?,
    )?;
    let rotary_dim = ((dims.head_dim as f32) * config.partial_rotary_factor).round() as usize;
    if rotary_dim > dims.head_dim || !rotary_dim.is_multiple_of(2) {
        return Err(MathError::InvalidShape(format!(
            "Qwen rotary dimension {rotary_dim} must be even and <= head dim {}",
            dims.head_dim
        )));
    }

    let position = cache.next_position();
    let mut query = vec![0.0; attention_dim];
    let mut gate = vec![0.0; attention_dim];
    for head in 0..dims.num_attention_heads {
        let projected_head_start = if config.q_projection_gate {
            head * dims.head_dim * 2
        } else {
            head * dims.head_dim
        };
        let q_start = head * dims.head_dim;
        let normalized = qwen_attention_rms_norm_with_matvec(
            &parts.q_proj[projected_head_start..projected_head_start + dims.head_dim],
            parts.q_norm_weight,
            config,
            matvec,
        )
        .await?;
        query[q_start..q_start + dims.head_dim].copy_from_slice(&normalized);
        if config.q_projection_gate {
            gate[q_start..q_start + dims.head_dim].copy_from_slice(
                &parts.q_proj[projected_head_start + dims.head_dim
                    ..projected_head_start + dims.head_dim * 2],
            );
        }
        apply_rope_to_head(
            &mut query[q_start..q_start + dims.head_dim],
            position,
            rotary_dim,
            config.rope_theta,
        );
    }

    let mut key = vec![0.0; key_value_dim];
    for head in 0..dims.num_key_value_heads {
        let head_start = head * dims.head_dim;
        let normalized = qwen_attention_rms_norm_with_matvec(
            &parts.k_proj[head_start..head_start + dims.head_dim],
            parts.k_norm_weight,
            config,
            matvec,
        )
        .await?;
        key[head_start..head_start + dims.head_dim].copy_from_slice(&normalized);
        apply_rope_to_head(
            &mut key[head_start..head_start + dims.head_dim],
            position,
            rotary_dim,
            config.rope_theta,
        );
    }
    native_full_attention_step_with_cache_from_parts_with_matvec(
        dims.native(),
        &NativeFullAttentionStepParts {
            query: &query,
            key: &key,
            value: parts.v_proj,
            gate: config.q_projection_gate.then_some(gate.as_slice()),
            output_projection: parts.o_proj_weight,
            score_scale: (dims.head_dim as f32).sqrt().recip(),
        },
        cache,
        matvec,
    )
    .await
}

async fn qwen_full_attention_sequence_from_parts_impl(
    dims: &QwenFullAttentionDims,
    parts: &QwenFullAttentionSequenceParts<'_>,
    config: QwenFullAttentionSequenceConfig,
    cache: Option<&mut LayerKvCache>,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<Vec<f32>>, MathError> {
    let q_proj = parts.q_proj;
    let k_proj = parts.k_proj;
    let v_proj = parts.v_proj;
    if q_proj.is_empty() {
        return Ok(Vec::new());
    }
    if dims.num_attention_heads == 0
        || dims.num_key_value_heads == 0
        || dims.head_dim == 0
        || dims.hidden_size == 0
    {
        return Err(MathError::InvalidShape(
            "Qwen full attention dimensions must be non-zero".to_owned(),
        ));
    }
    if !dims
        .num_attention_heads
        .is_multiple_of(dims.num_key_value_heads)
    {
        return Err(MathError::InvalidShape(
            "Qwen attention heads must be divisible by key/value heads".to_owned(),
        ));
    }
    if config.rope_theta <= 0.0 || config.partial_rotary_factor < 0.0 {
        return Err(MathError::InvalidShape(
            "Qwen RoPE parameters must be positive".to_owned(),
        ));
    }
    let seq_len = q_proj.len();
    if k_proj.len() != seq_len || v_proj.len() != seq_len {
        return Err(MathError::InvalidShape(
            "Qwen full attention sequence inputs must have the same length".to_owned(),
        ));
    }
    let attention_dim = dims.attention_dim()?;
    let key_value_dim = dims.key_value_dim()?;
    let q_projection_width = if config.q_projection_gate {
        attention_dim
            .checked_mul(2)
            .ok_or_else(|| MathError::InvalidShape("Qwen q projection overflow".to_owned()))?
    } else {
        attention_dim
    };
    require_len("q norm weight", parts.q_norm_weight.len(), dims.head_dim)?;
    require_len("k norm weight", parts.k_norm_weight.len(), dims.head_dim)?;
    require_len(
        "o projection weight",
        parts.o_proj_weight.len(),
        dims.hidden_size
            .checked_mul(attention_dim)
            .ok_or_else(|| MathError::InvalidShape("Qwen o projection overflow".to_owned()))?,
    )?;
    let rotary_dim = ((dims.head_dim as f32) * config.partial_rotary_factor).round() as usize;
    if rotary_dim > dims.head_dim || !rotary_dim.is_multiple_of(2) {
        return Err(MathError::InvalidShape(format!(
            "Qwen rotary dimension {rotary_dim} must be even and <= head dim {}",
            dims.head_dim
        )));
    }

    let position_offset = cache.as_deref().map_or(0, LayerKvCache::next_position);
    let mut queries = vec![vec![0.0; attention_dim]; seq_len];
    let mut gates = vec![vec![0.0; attention_dim]; seq_len];
    let mut keys = vec![vec![0.0; key_value_dim]; seq_len];
    for token_idx in 0..seq_len {
        let position = position_offset
            .checked_add(token_idx)
            .ok_or_else(|| MathError::InvalidShape("Qwen RoPE position overflow".to_owned()))?;
        require_len("q projection", q_proj[token_idx].len(), q_projection_width)?;
        require_len("k projection", k_proj[token_idx].len(), key_value_dim)?;
        require_len("v projection", v_proj[token_idx].len(), key_value_dim)?;

        for head in 0..dims.num_attention_heads {
            let projected_head_start = if config.q_projection_gate {
                head * dims.head_dim * 2
            } else {
                head * dims.head_dim
            };
            let q_start = head * dims.head_dim;
            let query = qwen_attention_rms_norm_with_matvec(
                &q_proj[token_idx][projected_head_start..projected_head_start + dims.head_dim],
                parts.q_norm_weight,
                config,
                matvec,
            )
            .await?;
            queries[token_idx][q_start..q_start + dims.head_dim].copy_from_slice(&query);
            if config.q_projection_gate {
                gates[token_idx][q_start..q_start + dims.head_dim].copy_from_slice(
                    &q_proj[token_idx][projected_head_start + dims.head_dim
                        ..projected_head_start + dims.head_dim * 2],
                );
            }
            apply_rope_to_head(
                &mut queries[token_idx][q_start..q_start + dims.head_dim],
                position,
                rotary_dim,
                config.rope_theta,
            );
        }
        for head in 0..dims.num_key_value_heads {
            let head_start = head * dims.head_dim;
            let key = qwen_attention_rms_norm_with_matvec(
                &k_proj[token_idx][head_start..head_start + dims.head_dim],
                parts.k_norm_weight,
                config,
                matvec,
            )
            .await?;
            keys[token_idx][head_start..head_start + dims.head_dim].copy_from_slice(&key);
            apply_rope_to_head(
                &mut keys[token_idx][head_start..head_start + dims.head_dim],
                position,
                rotary_dim,
                config.rope_theta,
            );
        }
    }
    let generic_parts = NativeFullAttentionSequenceParts {
        queries: &queries,
        keys: &keys,
        values: v_proj,
        gates: config.q_projection_gate.then_some(gates.as_slice()),
        output_projection: parts.o_proj_weight,
        score_scale: (dims.head_dim as f32).sqrt().recip(),
    };
    if let Some(cache) = cache {
        native_full_attention_sequence_with_cache_from_parts_with_matvec(
            dims.native(),
            &generic_parts,
            cache,
            matvec,
        )
        .await
    } else {
        native_full_attention_sequence_from_parts_with_matvec(dims.native(), &generic_parts, matvec)
            .await
    }
}

pub async fn qwen_layer0_linear_attention_first_token(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    projections: &QwenLinearAttentionProjectionProbe,
) -> Result<Vec<f32>, TensorLoadError> {
    qwen_layer_linear_attention_first_token(store, spec, 0, projections).await
}

pub async fn qwen_layer_linear_attention_first_token(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    projections: &QwenLinearAttentionProjectionProbe,
) -> Result<Vec<f32>, TensorLoadError> {
    qwen_layer_linear_attention_first_token_with_matvec(
        store,
        spec,
        layer_idx,
        projections,
        &CpuNativeMatvecBackend,
    )
    .await
}

pub async fn qwen_layer_linear_attention_first_token_with_matvec(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    projections: &QwenLinearAttentionProjectionProbe,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    let dims = QwenLinearAttentionDims::from_spec(spec);
    let dt_bias = store.bf16_tensor_f32(&qwen_linear_attn_tensor(layer_idx, "dt_bias"))?;
    let a_log = store.bf16_tensor_f32(&qwen_linear_attn_tensor(layer_idx, "A_log"))?;
    let conv1d_weight =
        store.bf16_tensor_f32(&qwen_linear_attn_tensor(layer_idx, "conv1d.weight"))?;
    let norm_weight = store.bf16_tensor_f32(&qwen_linear_attn_tensor(layer_idx, "norm.weight"))?;
    let out_proj_weight =
        store.bf16_tensor_f32(&qwen_linear_attn_tensor(layer_idx, "out_proj.weight"))?;
    let qkv = vec![projections.qkv.clone()];
    let z = vec![projections.z.clone()];
    let b = vec![projections.b.clone()];
    let a = vec![projections.a.clone()];
    qwen_linear_attention_sequence_from_parts_with_matvec(
        &dims,
        &QwenLinearAttentionSequenceParts {
            qkv: &qkv,
            z: &z,
            b: &b,
            a: &a,
            dt_bias: &dt_bias,
            a_log: &a_log,
            conv1d_weight: &conv1d_weight,
            norm_weight: &norm_weight,
            out_proj_weight: &out_proj_weight,
        },
        matvec,
    )
    .await
    .map(|mut outputs| outputs.remove(0))
    .map_err(|err| {
        TensorLoadError::integrity(format!("Qwen layer0 linear attention failed: {err}"))
    })
}

pub async fn qwen_layer_linear_attention_sequence(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    qwen_layer_linear_attention_sequence_impl(
        store,
        spec,
        layer_idx,
        hidden_states,
        None,
        &CpuNativeMatvecBackend,
    )
    .await
}

pub async fn qwen_layer_linear_attention_sequence_with_cache(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[Vec<f32>],
    cache: &mut LinearAttentionCache,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    qwen_layer_linear_attention_sequence_with_cache_with_matvec(
        store,
        spec,
        layer_idx,
        hidden_states,
        cache,
        &CpuNativeMatvecBackend,
    )
    .await
}

pub async fn qwen_layer_linear_attention_sequence_with_cache_with_matvec(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[Vec<f32>],
    cache: &mut LinearAttentionCache,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    qwen_layer_linear_attention_sequence_impl(
        store,
        spec,
        layer_idx,
        hidden_states,
        Some(cache),
        matvec,
    )
    .await
}

async fn qwen_layer_linear_attention_sequence_impl(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[Vec<f32>],
    cache: Option<&mut LinearAttentionCache>,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    let projections = QwenLinearAttentionProjectionSequence {
        qkv: matvec
            .bf16_matvecs_row_major_f32(
                store,
                &qwen_linear_attn_tensor(layer_idx, "in_proj_qkv.weight"),
                hidden_states,
            )
            .await?,
        z: matvec
            .bf16_matvecs_row_major_f32(
                store,
                &qwen_linear_attn_tensor(layer_idx, "in_proj_z.weight"),
                hidden_states,
            )
            .await?,
        b: matvec
            .bf16_matvecs_row_major_f32(
                store,
                &qwen_linear_attn_tensor(layer_idx, "in_proj_b.weight"),
                hidden_states,
            )
            .await?,
        a: matvec
            .bf16_matvecs_row_major_f32(
                store,
                &qwen_linear_attn_tensor(layer_idx, "in_proj_a.weight"),
                hidden_states,
            )
            .await?,
    };
    let dims = QwenLinearAttentionDims::from_spec(spec);
    let dt_bias = store.bf16_tensor_f32(&qwen_linear_attn_tensor(layer_idx, "dt_bias"))?;
    let a_log = store.bf16_tensor_f32(&qwen_linear_attn_tensor(layer_idx, "A_log"))?;
    let conv1d_weight =
        store.bf16_tensor_f32(&qwen_linear_attn_tensor(layer_idx, "conv1d.weight"))?;
    let norm_weight = store.bf16_tensor_f32(&qwen_linear_attn_tensor(layer_idx, "norm.weight"))?;
    let out_proj_weight =
        store.bf16_tensor_f32(&qwen_linear_attn_tensor(layer_idx, "out_proj.weight"))?;
    let parts = QwenLinearAttentionSequenceParts {
        qkv: &projections.qkv,
        z: &projections.z,
        b: &projections.b,
        a: &projections.a,
        dt_bias: &dt_bias,
        a_log: &a_log,
        conv1d_weight: &conv1d_weight,
        norm_weight: &norm_weight,
        out_proj_weight: &out_proj_weight,
    };
    let result = if let Some(cache) = cache {
        qwen_linear_attention_sequence_with_cache_from_parts_with_matvec(
            &dims, &parts, cache, matvec,
        )
        .await
    } else {
        qwen_linear_attention_sequence_from_parts_with_matvec(&dims, &parts, matvec).await
    };
    result.map_err(|err| {
        TensorLoadError::integrity(format!(
            "Qwen layer{layer_idx} linear attention sequence failed: {err}"
        ))
    })
}

pub async fn qwen_layer_linear_attention_step_with_cache(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[f32],
    cache: &mut LinearAttentionCache,
) -> Result<Vec<f32>, TensorLoadError> {
    qwen_layer_linear_attention_step_with_cache_with_matvec(
        store,
        spec,
        layer_idx,
        hidden_states,
        cache,
        &CpuNativeMatvecBackend,
    )
    .await
}

pub async fn qwen_layer_linear_attention_step_with_cache_with_matvec(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[f32],
    cache: &mut LinearAttentionCache,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    let projections = qwen_layer_linear_attention_projections_with_matvec(
        store,
        layer_idx,
        hidden_states,
        matvec,
    )
    .await?;
    let dims = QwenLinearAttentionDims::from_spec(spec);
    let dt_bias = store.bf16_tensor_f32(&qwen_linear_attn_tensor(layer_idx, "dt_bias"))?;
    let a_log = store.bf16_tensor_f32(&qwen_linear_attn_tensor(layer_idx, "A_log"))?;
    let conv1d_weight =
        store.bf16_tensor_f32(&qwen_linear_attn_tensor(layer_idx, "conv1d.weight"))?;
    let norm_weight = store.bf16_tensor_f32(&qwen_linear_attn_tensor(layer_idx, "norm.weight"))?;
    let out_proj_weight =
        store.bf16_tensor_f32(&qwen_linear_attn_tensor(layer_idx, "out_proj.weight"))?;
    qwen_linear_attention_step_with_cache_from_parts_with_matvec(
        &dims,
        &QwenLinearAttentionStepParts {
            qkv: &projections.qkv,
            z: &projections.z,
            b: &projections.b,
            a: &projections.a,
            dt_bias: &dt_bias,
            a_log: &a_log,
            conv1d_weight: &conv1d_weight,
            norm_weight: &norm_weight,
            out_proj_weight: &out_proj_weight,
        },
        cache,
        matvec,
    )
    .await
    .map_err(|err| {
        TensorLoadError::integrity(format!(
            "Qwen layer{layer_idx} linear attention step failed: {err}"
        ))
    })
}

pub async fn qwen_layer_full_attention_first_token(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[f32],
) -> Result<Vec<f32>, TensorLoadError> {
    qwen_layer_full_attention_first_token_with_matvec(
        store,
        spec,
        layer_idx,
        hidden_states,
        &CpuNativeMatvecBackend,
    )
    .await
}

pub async fn qwen_layer_full_attention_first_token_with_matvec(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[f32],
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    let dims = QwenFullAttentionDims::from_spec(spec);
    let q_proj = matvec
        .bf16_matvec_row_major_f32(
            store,
            &spec.self_attn_tensor(layer_idx, "q_proj.weight"),
            hidden_states,
        )
        .await?;
    let k_proj = matvec
        .bf16_matvec_row_major_f32(
            store,
            &spec.self_attn_tensor(layer_idx, "k_proj.weight"),
            hidden_states,
        )
        .await?;
    let v_proj = matvec
        .bf16_matvec_row_major_f32(
            store,
            &spec.self_attn_tensor(layer_idx, "v_proj.weight"),
            hidden_states,
        )
        .await?;
    let q_norm_weight =
        store.bf16_tensor_f32(&spec.self_attn_tensor(layer_idx, "q_norm.weight"))?;
    let k_norm_weight =
        store.bf16_tensor_f32(&spec.self_attn_tensor(layer_idx, "k_norm.weight"))?;
    let o_proj_weight =
        store.bf16_tensor_f32(&spec.self_attn_tensor(layer_idx, "o_proj.weight"))?;
    let q_proj = vec![q_proj];
    let k_proj = vec![k_proj];
    let v_proj = vec![v_proj];
    qwen_full_attention_sequence_from_parts_with_matvec(
        &dims,
        &QwenFullAttentionSequenceParts {
            q_proj: &q_proj,
            k_proj: &k_proj,
            v_proj: &v_proj,
            q_norm_weight: &q_norm_weight,
            k_norm_weight: &k_norm_weight,
            o_proj_weight: &o_proj_weight,
        },
        QwenFullAttentionSequenceConfig {
            rms_norm_eps: spec.rms_norm_eps,
            rope_theta: spec.rope_theta,
            partial_rotary_factor: spec.partial_rotary_factor,
            q_projection_gate: !spec.is_qwen3_dense(),
            one_centered_rms_norm: !spec.is_qwen3_dense(),
        },
        matvec,
    )
    .await
    .map(|mut outputs| outputs.remove(0))
    .map_err(|err| {
        TensorLoadError::integrity(format!(
            "Qwen layer{layer_idx} full attention failed: {err}"
        ))
    })
}

pub async fn qwen_layer_full_attention_sequence(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    qwen_layer_full_attention_sequence_impl(
        store,
        spec,
        layer_idx,
        hidden_states,
        None,
        &CpuNativeMatvecBackend,
    )
    .await
}

pub async fn qwen_layer_full_attention_sequence_with_cache(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[Vec<f32>],
    cache: &mut LayerKvCache,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    qwen_layer_full_attention_sequence_with_cache_with_matvec(
        store,
        spec,
        layer_idx,
        hidden_states,
        cache,
        &CpuNativeMatvecBackend,
    )
    .await
}

pub async fn qwen_layer_full_attention_sequence_with_cache_with_matvec(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[Vec<f32>],
    cache: &mut LayerKvCache,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    qwen_layer_full_attention_sequence_impl(
        store,
        spec,
        layer_idx,
        hidden_states,
        Some(cache),
        matvec,
    )
    .await
}

async fn qwen_layer_full_attention_sequence_impl(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[Vec<f32>],
    cache: Option<&mut LayerKvCache>,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    let dims = QwenFullAttentionDims::from_spec(spec);
    let q_proj = matvec
        .bf16_matvecs_row_major_f32(
            store,
            &spec.self_attn_tensor(layer_idx, "q_proj.weight"),
            hidden_states,
        )
        .await?;
    let k_proj = matvec
        .bf16_matvecs_row_major_f32(
            store,
            &spec.self_attn_tensor(layer_idx, "k_proj.weight"),
            hidden_states,
        )
        .await?;
    let v_proj = matvec
        .bf16_matvecs_row_major_f32(
            store,
            &spec.self_attn_tensor(layer_idx, "v_proj.weight"),
            hidden_states,
        )
        .await?;
    let q_norm_weight =
        store.bf16_tensor_f32(&spec.self_attn_tensor(layer_idx, "q_norm.weight"))?;
    let k_norm_weight =
        store.bf16_tensor_f32(&spec.self_attn_tensor(layer_idx, "k_norm.weight"))?;
    let o_proj_weight =
        store.bf16_tensor_f32(&spec.self_attn_tensor(layer_idx, "o_proj.weight"))?;
    let parts = QwenFullAttentionSequenceParts {
        q_proj: &q_proj,
        k_proj: &k_proj,
        v_proj: &v_proj,
        q_norm_weight: &q_norm_weight,
        k_norm_weight: &k_norm_weight,
        o_proj_weight: &o_proj_weight,
    };
    let config = QwenFullAttentionSequenceConfig {
        rms_norm_eps: spec.rms_norm_eps,
        rope_theta: spec.rope_theta,
        partial_rotary_factor: spec.partial_rotary_factor,
        q_projection_gate: !spec.is_qwen3_dense(),
        one_centered_rms_norm: !spec.is_qwen3_dense(),
    };
    let result = if let Some(cache) = cache {
        qwen_full_attention_sequence_with_cache_from_parts_with_matvec(
            &dims, &parts, config, cache, matvec,
        )
        .await
    } else {
        qwen_full_attention_sequence_from_parts_with_matvec(&dims, &parts, config, matvec).await
    };
    result.map_err(|err| {
        TensorLoadError::integrity(format!(
            "Qwen layer{layer_idx} full attention sequence failed: {err}"
        ))
    })
}

pub async fn qwen_layer_full_attention_step_with_cache(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[f32],
    cache: &mut LayerKvCache,
) -> Result<Vec<f32>, TensorLoadError> {
    qwen_layer_full_attention_step_with_cache_with_matvec(
        store,
        spec,
        layer_idx,
        hidden_states,
        cache,
        &CpuNativeMatvecBackend,
    )
    .await
}

pub async fn qwen_layer_full_attention_step_with_cache_with_matvec(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[f32],
    cache: &mut LayerKvCache,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    let dims = QwenFullAttentionDims::from_spec(spec);
    let q_proj = matvec
        .bf16_matvec_row_major_f32(
            store,
            &spec.self_attn_tensor(layer_idx, "q_proj.weight"),
            hidden_states,
        )
        .await?;
    let k_proj = matvec
        .bf16_matvec_row_major_f32(
            store,
            &spec.self_attn_tensor(layer_idx, "k_proj.weight"),
            hidden_states,
        )
        .await?;
    let v_proj = matvec
        .bf16_matvec_row_major_f32(
            store,
            &spec.self_attn_tensor(layer_idx, "v_proj.weight"),
            hidden_states,
        )
        .await?;
    let q_norm_weight =
        store.bf16_tensor_f32(&spec.self_attn_tensor(layer_idx, "q_norm.weight"))?;
    let k_norm_weight =
        store.bf16_tensor_f32(&spec.self_attn_tensor(layer_idx, "k_norm.weight"))?;
    let o_proj_weight =
        store.bf16_tensor_f32(&spec.self_attn_tensor(layer_idx, "o_proj.weight"))?;
    qwen_full_attention_step_with_cache_from_parts_with_matvec(
        &dims,
        &QwenFullAttentionStepParts {
            q_proj: &q_proj,
            k_proj: &k_proj,
            v_proj: &v_proj,
            q_norm_weight: &q_norm_weight,
            k_norm_weight: &k_norm_weight,
            o_proj_weight: &o_proj_weight,
        },
        QwenFullAttentionSequenceConfig {
            rms_norm_eps: spec.rms_norm_eps,
            rope_theta: spec.rope_theta,
            partial_rotary_factor: spec.partial_rotary_factor,
            q_projection_gate: !spec.is_qwen3_dense(),
            one_centered_rms_norm: !spec.is_qwen3_dense(),
        },
        cache,
        matvec,
    )
    .await
    .map_err(|err| {
        TensorLoadError::integrity(format!(
            "Qwen layer{layer_idx} full attention step failed: {err}"
        ))
    })
}

pub async fn qwen_layer0_linear_attention_projections(
    store: &SafeTensorShardStore,
    hidden_states: &[f32],
) -> Result<QwenLinearAttentionProjectionProbe, TensorLoadError> {
    qwen_layer_linear_attention_projections(store, 0, hidden_states).await
}

pub async fn qwen_layer_linear_attention_projections(
    store: &SafeTensorShardStore,
    layer_idx: usize,
    hidden_states: &[f32],
) -> Result<QwenLinearAttentionProjectionProbe, TensorLoadError> {
    qwen_layer_linear_attention_projections_with_matvec(
        store,
        layer_idx,
        hidden_states,
        &CpuNativeMatvecBackend,
    )
    .await
}

pub async fn qwen_layer_linear_attention_projections_with_matvec(
    store: &SafeTensorShardStore,
    layer_idx: usize,
    hidden_states: &[f32],
    matvec: &impl NativeMatvecBackend,
) -> Result<QwenLinearAttentionProjectionProbe, TensorLoadError> {
    Ok(QwenLinearAttentionProjectionProbe {
        qkv: matvec
            .bf16_matvec_row_major_f32(
                store,
                &qwen_linear_attn_tensor(layer_idx, "in_proj_qkv.weight"),
                hidden_states,
            )
            .await?,
        z: matvec
            .bf16_matvec_row_major_f32(
                store,
                &qwen_linear_attn_tensor(layer_idx, "in_proj_z.weight"),
                hidden_states,
            )
            .await?,
        b: matvec
            .bf16_matvec_row_major_f32(
                store,
                &qwen_linear_attn_tensor(layer_idx, "in_proj_b.weight"),
                hidden_states,
            )
            .await?,
        a: matvec
            .bf16_matvec_row_major_f32(
                store,
                &qwen_linear_attn_tensor(layer_idx, "in_proj_a.weight"),
                hidden_states,
            )
            .await?,
    })
}

pub async fn qwen_layer0_post_attention_norm(
    store: &SafeTensorShardStore,
    residual: &[f32],
    attention_output: &[f32],
    hidden_size: usize,
    rms_norm_eps: f32,
) -> Result<Vec<f32>, TensorLoadError> {
    qwen_layer_post_attention_norm(
        store,
        0,
        residual,
        attention_output,
        hidden_size,
        rms_norm_eps,
    )
    .await
}

pub async fn qwen_layer_post_attention_norm(
    store: &SafeTensorShardStore,
    layer_idx: usize,
    residual: &[f32],
    attention_output: &[f32],
    hidden_size: usize,
    rms_norm_eps: f32,
) -> Result<Vec<f32>, TensorLoadError> {
    qwen_layer_post_attention_norm_with_matvec(
        store,
        layer_idx,
        residual,
        attention_output,
        hidden_size,
        rms_norm_eps,
        &CpuNativeMatvecBackend,
    )
    .await
}

pub async fn qwen_layer_post_attention_norm_with_matvec(
    store: &SafeTensorShardStore,
    layer_idx: usize,
    residual: &[f32],
    attention_output: &[f32],
    hidden_size: usize,
    rms_norm_eps: f32,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    if residual.len() != hidden_size || attention_output.len() != hidden_size {
        return Err(TensorLoadError::integrity(format!(
            "Qwen post-attention residual lengths {}, {} must match hidden size {hidden_size}",
            residual.len(),
            attention_output.len()
        )));
    }
    let hidden_states = residual
        .iter()
        .zip(attention_output)
        .map(|(residual, attention)| residual + attention)
        .collect::<Vec<_>>();
    let norm_weight = store.bf16_tensor_f32_range(
        &qwen_layer_tensor(layer_idx, "post_attention_layernorm.weight"),
        0,
        hidden_size,
    )?;
    matvec
        .rms_norm_one_centered_f32(&hidden_states, &norm_weight, rms_norm_eps)
        .await
        .map_err(|err| {
            TensorLoadError::integrity(format!("Qwen layer post-attention RMSNorm failed: {err}"))
        })
}

async fn qwen_layer_post_attention_norm_for_spec_with_matvec(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    residual: &[f32],
    attention_output: &[f32],
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    let hidden_size = spec.hidden_size as usize;
    if residual.len() != hidden_size || attention_output.len() != hidden_size {
        return Err(TensorLoadError::integrity(format!(
            "Qwen post-attention residual lengths {}, {} must match hidden size {hidden_size}",
            residual.len(),
            attention_output.len()
        )));
    }
    let hidden_states = residual
        .iter()
        .zip(attention_output)
        .map(|(residual, attention)| residual + attention)
        .collect::<Vec<_>>();
    let norm_weight = store.bf16_tensor_f32_range(
        &spec.layer_tensor(layer_idx, "post_attention_layernorm.weight"),
        0,
        hidden_size,
    )?;
    qwen_rms_norm_for_spec_with_matvec(spec, &hidden_states, &norm_weight, matvec)
        .await
        .map_err(|err| {
            TensorLoadError::integrity(format!("Qwen layer post-attention RMSNorm failed: {err}"))
        })
}

async fn qwen_layer_post_attention_norm_sequence_for_spec(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    residual: &[Vec<f32>],
    attention_output: &[Vec<f32>],
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    let hidden_size = spec.hidden_size as usize;
    if residual.len() != attention_output.len() {
        return Err(TensorLoadError::integrity(
            "Qwen post-attention sequence lengths must match",
        ));
    }
    let norm_weight = store.bf16_tensor_f32_range(
        &spec.layer_tensor(layer_idx, "post_attention_layernorm.weight"),
        0,
        hidden_size,
    )?;
    let mut results = Vec::with_capacity(residual.len());
    for (residual, attention) in residual.iter().zip(attention_output) {
        if residual.len() != hidden_size || attention.len() != hidden_size {
            return Err(TensorLoadError::integrity(format!(
                "Qwen post-attention residual lengths {}, {} must match hidden size {hidden_size}",
                residual.len(),
                attention.len()
            )));
        }
        let hidden_states = residual
            .iter()
            .zip(attention)
            .map(|(residual, attention)| residual + attention)
            .collect::<Vec<_>>();
        results.push(
            qwen_rms_norm_for_spec_with_matvec(spec, &hidden_states, &norm_weight, matvec)
                .await
                .map_err(|err| {
                    TensorLoadError::integrity(format!(
                        "Qwen post-attention RMSNorm sequence failed: {err}"
                    ))
                })?,
        );
    }
    Ok(results)
}

pub async fn qwen_linear_decoder_layer_first_token(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[f32],
) -> Result<Vec<f32>, TensorLoadError> {
    qwen_linear_decoder_layer_first_token_with_matvec(
        store,
        spec,
        layer_idx,
        hidden_states,
        &CpuNativeMatvecBackend,
    )
    .await
}

pub async fn qwen_linear_decoder_layer_first_token_with_matvec(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[f32],
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    match spec.layer_kinds.get(layer_idx) {
        Some(AttentionKind::LinearAttention) => {}
        Some(AttentionKind::FullAttention) => {
            return Err(TensorLoadError::unsupported(format!(
                "Qwen layer {layer_idx} is full attention, not linear attention"
            )));
        }
        None => {
            return Err(TensorLoadError::missing(format!(
                "Qwen layer {layer_idx} is outside configured layer count"
            )));
        }
    }
    let input_norm = qwen_layer_input_norm_for_spec_with_matvec(store, spec, layer_idx, hidden_states, matvec)
        .await?;
    let projections = qwen_layer_linear_attention_projections_with_matvec(store, layer_idx, &input_norm, matvec)
        .await?;
    let attention_output = qwen_layer_linear_attention_first_token_with_matvec(
        store,
        spec,
        layer_idx,
        &projections,
        matvec,
    )
    .await?;
    let post_attention = qwen_layer_post_attention_norm_for_spec_with_matvec(
        store,
        spec,
        layer_idx,
        hidden_states,
        &attention_output,
        matvec,
    )
    .await?;
    let mlp_output =
        qwen_layer_feed_forward_with_matvec(store, spec, layer_idx, &post_attention, matvec).await?;
    hidden_states
        .iter()
        .zip(attention_output)
        .zip(mlp_output)
        .map(|((hidden, attention), mlp)| Ok(hidden + attention + mlp))
        .collect()
}

pub async fn qwen_full_decoder_layer_first_token(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[f32],
) -> Result<Vec<f32>, TensorLoadError> {
    qwen_full_decoder_layer_first_token_with_matvec(
        store,
        spec,
        layer_idx,
        hidden_states,
        &CpuNativeMatvecBackend,
    )
    .await
}

pub async fn qwen_full_decoder_layer_first_token_with_matvec(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[f32],
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    match spec.layer_kinds.get(layer_idx) {
        Some(AttentionKind::FullAttention) => {}
        Some(AttentionKind::LinearAttention) => {
            return Err(TensorLoadError::unsupported(format!(
                "Qwen layer {layer_idx} is linear attention, not full attention"
            )));
        }
        None => {
            return Err(TensorLoadError::missing(format!(
                "Qwen layer {layer_idx} is outside configured layer count"
            )));
        }
    }
    let input_norm = qwen_layer_input_norm_for_spec_with_matvec(store, spec, layer_idx, hidden_states, matvec)
        .await?;
    let attention_output = qwen_layer_full_attention_first_token_with_matvec(
        store,
        spec,
        layer_idx,
        &input_norm,
        matvec,
    )
    .await?;
    let post_attention = qwen_layer_post_attention_norm_for_spec_with_matvec(
        store,
        spec,
        layer_idx,
        hidden_states,
        &attention_output,
        matvec,
    )
    .await?;
    let mlp_output =
        qwen_layer_feed_forward_with_matvec(store, spec, layer_idx, &post_attention, matvec).await?;
    hidden_states
        .iter()
        .zip(attention_output)
        .zip(mlp_output)
        .map(|((hidden, attention), mlp)| Ok(hidden + attention + mlp))
        .collect()
}

pub async fn qwen_decoder_layer_first_token(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[f32],
) -> Result<Vec<f32>, TensorLoadError> {
    qwen_decoder_layer_first_token_with_matvec(
        store,
        spec,
        layer_idx,
        hidden_states,
        &CpuNativeMatvecBackend,
    )
    .await
}

pub async fn qwen_decoder_layer_first_token_with_matvec(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[f32],
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    match spec.layer_kinds.get(layer_idx) {
        Some(AttentionKind::LinearAttention) => qwen_linear_decoder_layer_first_token_with_matvec(
            store,
            spec,
            layer_idx,
            hidden_states,
            matvec,
        )
        .await,
        Some(AttentionKind::FullAttention) => qwen_full_decoder_layer_first_token_with_matvec(
            store,
            spec,
            layer_idx,
            hidden_states,
            matvec,
        )
        .await,
        None => Err(TensorLoadError::missing(format!(
            "Qwen layer {layer_idx} is outside configured layer count"
        ))),
    }
}

pub async fn qwen_decoder_layer_sequence(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    qwen_decoder_layer_sequence_impl(
        store,
        spec,
        layer_idx,
        hidden_states,
        None,
        &CpuNativeMatvecBackend,
    )
    .await
}

pub async fn qwen_decoder_layer_sequence_with_cache(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[Vec<f32>],
    cache: &mut QwenLayerCache,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    qwen_decoder_layer_sequence_with_cache_with_matvec(
        store,
        spec,
        layer_idx,
        hidden_states,
        cache,
        &CpuNativeMatvecBackend,
    )
    .await
}

pub async fn qwen_decoder_layer_sequence_with_cache_with_matvec(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[Vec<f32>],
    cache: &mut QwenLayerCache,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    qwen_decoder_layer_sequence_impl(store, spec, layer_idx, hidden_states, Some(cache), matvec).await
}

pub async fn qwen_decoder_layer_step_with_cache(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[f32],
    cache: &mut QwenLayerCache,
) -> Result<Vec<f32>, TensorLoadError> {
    qwen_decoder_layer_step_with_cache_with_matvec(
        store,
        spec,
        layer_idx,
        hidden_states,
        cache,
        &CpuNativeMatvecBackend,
    )
    .await
}

pub async fn qwen_decoder_layer_step_with_cache_with_matvec(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[f32],
    cache: &mut QwenLayerCache,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    let input_norm = qwen_layer_input_norm_for_spec_with_matvec(store, spec, layer_idx, hidden_states, matvec)
        .await?;
    let attention_output = match spec.layer_kinds.get(layer_idx) {
        Some(AttentionKind::LinearAttention) => match cache {
            QwenLayerCache::Linear(cache) => {
                qwen_layer_linear_attention_step_with_cache_with_matvec(
                    store,
                    spec,
                    layer_idx,
                    &input_norm,
                    cache,
                    matvec,
                )
                .await?
            }
            QwenLayerCache::Full(_) => {
                return Err(TensorLoadError::integrity(format!(
                    "Qwen layer{layer_idx} expected linear attention cache"
                )));
            }
        },
        Some(AttentionKind::FullAttention) => match cache {
            QwenLayerCache::Full(cache) => qwen_layer_full_attention_step_with_cache_with_matvec(
                store,
                spec,
                layer_idx,
                &input_norm,
                cache,
                matvec,
            )
            .await?,
            QwenLayerCache::Linear(_) => {
                return Err(TensorLoadError::integrity(format!(
                    "Qwen layer{layer_idx} expected full attention cache"
                )));
            }
        },
        None => {
            return Err(TensorLoadError::missing(format!(
                "Qwen layer {layer_idx} is outside configured layer count"
            )));
        }
    };
    let post_attention = qwen_layer_post_attention_norm_for_spec_with_matvec(
        store,
        spec,
        layer_idx,
        hidden_states,
        &attention_output,
        matvec,
    )
    .await?;
    let mlp_output =
        qwen_layer_feed_forward_with_matvec(store, spec, layer_idx, &post_attention, matvec).await?;
    hidden_states
        .iter()
        .zip(attention_output)
        .zip(mlp_output)
        .map(|((hidden, attention), mlp)| Ok(hidden + attention + mlp))
        .collect()
}

async fn qwen_decoder_layer_sequence_impl(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[Vec<f32>],
    cache: Option<&mut QwenLayerCache>,
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    let input_norm = qwen_layer_input_norm_sequence_for_spec(store, spec, layer_idx, hidden_states, matvec)
        .await?;
    let attention_output = match spec.layer_kinds.get(layer_idx) {
        Some(AttentionKind::LinearAttention) => match cache {
            Some(QwenLayerCache::Linear(cache)) => {
                qwen_layer_linear_attention_sequence_with_cache_with_matvec(
                    store,
                    spec,
                    layer_idx,
                    &input_norm,
                    cache,
                    matvec,
                )
                .await?
            }
            Some(QwenLayerCache::Full(_)) => {
                return Err(TensorLoadError::integrity(format!(
                    "Qwen layer{layer_idx} expected linear attention cache"
                )));
            }
            None => qwen_layer_linear_attention_sequence_impl(
                store,
                spec,
                layer_idx,
                &input_norm,
                None,
                matvec,
            )
            .await?,
        },
        Some(AttentionKind::FullAttention) => match cache {
            Some(QwenLayerCache::Full(cache)) => {
                qwen_layer_full_attention_sequence_with_cache_with_matvec(
                    store,
                    spec,
                    layer_idx,
                    &input_norm,
                    cache,
                    matvec,
                )
                .await?
            }
            Some(QwenLayerCache::Linear(_)) => {
                return Err(TensorLoadError::integrity(format!(
                    "Qwen layer{layer_idx} expected full attention cache"
                )));
            }
            None => qwen_layer_full_attention_sequence_impl(
                store,
                spec,
                layer_idx,
                &input_norm,
                None,
                matvec,
            )
            .await?,
        },
        None => {
            return Err(TensorLoadError::missing(format!(
                "Qwen layer {layer_idx} is outside configured layer count"
            )));
        }
    };
    let post_attention = qwen_layer_post_attention_norm_sequence_for_spec(
        store,
        spec,
        layer_idx,
        hidden_states,
        &attention_output,
        matvec,
    )
    .await?;
    let mut results = Vec::with_capacity(hidden_states.len());
    for ((hidden, attention), post_attention) in hidden_states.iter().zip(attention_output).zip(post_attention) {
        let mlp_output = qwen_layer_feed_forward_with_matvec(
            store,
            spec,
            layer_idx,
            &post_attention,
            matvec,
        )
        .await?;
        results.push(
            hidden
                .iter()
                .zip(attention)
                .zip(mlp_output)
                .map(|((hidden, attention), mlp)| hidden + attention + mlp)
                .collect::<Vec<_>>(),
        );
    }
    Ok(results)
}

pub async fn qwen_prefill_sequence(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    token_ids: &[usize],
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    let mut hidden_states = qwen_embedding_sequence_for_spec(store, spec, token_ids)?;
    for layer_idx in 0..spec.num_hidden_layers as usize {
        hidden_states = qwen_decoder_layer_sequence(store, spec, layer_idx, &hidden_states).await?;
    }
    Ok(hidden_states)
}

pub async fn qwen_prefill_sequence_with_cache(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    token_ids: &[usize],
    caches: &mut [QwenLayerCache],
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    qwen_prefill_sequence_with_cache_with_matvec(
        store,
        spec,
        token_ids,
        caches,
        &CpuNativeMatvecBackend,
    )
    .await
}

pub async fn qwen_prefill_sequence_with_cache_with_matvec(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    token_ids: &[usize],
    caches: &mut [QwenLayerCache],
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    let layer_count = spec.num_hidden_layers as usize;
    if caches.len() != layer_count {
        return Err(TensorLoadError::integrity(format!(
            "Qwen prefill expected {layer_count} layer caches, got {}",
            caches.len()
        )));
    }
    let mut hidden_states = qwen_embedding_sequence_for_spec(store, spec, token_ids)?;
    for (layer_idx, cache) in caches.iter_mut().enumerate().take(layer_count) {
        hidden_states = qwen_decoder_layer_sequence_with_cache_with_matvec(
            store,
            spec,
            layer_idx,
            &hidden_states,
            cache,
            matvec,
        )
        .await?;
    }
    Ok(hidden_states)
}

pub async fn qwen_decode_token_with_cache(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    token_id: usize,
    caches: &mut [QwenLayerCache],
) -> Result<Vec<f32>, TensorLoadError> {
    qwen_decode_token_with_cache_with_matvec(store, spec, token_id, caches, &CpuNativeMatvecBackend)
        .await
}

pub async fn qwen_decode_token_with_cache_with_matvec(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    token_id: usize,
    caches: &mut [QwenLayerCache],
    matvec: &impl NativeMatvecBackend,
) -> Result<Vec<f32>, TensorLoadError> {
    let layer_count = spec.num_hidden_layers as usize;
    if caches.len() != layer_count {
        return Err(TensorLoadError::integrity(format!(
            "Qwen decode expected {layer_count} layer caches, got {}",
            caches.len()
        )));
    }
    let mut hidden_states = store.bf16_row_f32(&spec.embed_tokens_weight(), token_id)?;
    if hidden_states.len() != spec.hidden_size as usize {
        return Err(TensorLoadError::integrity(format!(
            "Qwen embedding row has length {}, expected hidden size {}",
            hidden_states.len(),
            spec.hidden_size
        )));
    }
    for (layer_idx, cache) in caches.iter_mut().enumerate().take(layer_count) {
        hidden_states = qwen_decoder_layer_step_with_cache_with_matvec(
            store,
            spec,
            layer_idx,
            &hidden_states,
            cache,
            matvec,
        )
        .await?;
    }
    Ok(hidden_states)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_attention_value_major_state_rows_reuses_scratch_buffer() {
        let recurrent_state = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut scratch = Vec::with_capacity(16);

        copy_linear_attention_value_major_state_rows(&recurrent_state, 0, 2, 3, &mut scratch)
            .expect("state transpose succeeds");

        assert_eq!(scratch, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(scratch.capacity(), 16);
    }
}
