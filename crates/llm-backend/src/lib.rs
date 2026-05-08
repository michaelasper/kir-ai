use async_trait::async_trait;
use futures::{
    StreamExt,
    stream::{self, BoxStream},
};
use llm_api::FinishReason;
use llm_kv_cache::{LayerKvCache, LinearAttentionCache};
use llm_models::{AttentionKind, QwenModelSpec, SafetensorsIndex};
use memmap2::{Mmap, MmapOptions};
use safetensors::{SafeTensors, tensor::Dtype};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    ops::Range,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
use thiserror::Error;
use tokio_util::sync::CancellationToken;

const MAX_SAFETENSORS_HEADER_LEN: u64 = 64 * 1024 * 1024;
const BF16_MATVEC_CHUNK_ROWS: usize = 256;

#[derive(Debug, Clone, PartialEq)]
pub struct BackendRequest {
    pub model: String,
    pub prompt: String,
    pub max_tokens: Option<u32>,
    pub sampling: SamplingConfig,
    pub required_tool_choice: Option<String>,
    pub json_object_mode: bool,
    pub conversation_mode: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum SamplingConfig {
    #[default]
    Greedy,
    TopP {
        temperature: f32,
        top_p: f32,
    },
}

impl SamplingConfig {
    pub fn from_openai_controls(temperature: Option<f32>, top_p: Option<f32>) -> Self {
        match (temperature, top_p) {
            (None, None | Some(1.0)) | (Some(0.0), _) => Self::Greedy,
            (None, Some(top_p)) => Self::TopP {
                temperature: 1.0,
                top_p,
            },
            (Some(temperature), top_p) => Self::TopP {
                temperature,
                top_p: top_p.unwrap_or(1.0),
            },
        }
    }

    pub fn is_greedy(self) -> bool {
        matches!(self, Self::Greedy)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendOutput {
    pub text: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendStreamChunk {
    pub text: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendModelMetadata {
    pub id: String,
    pub backend: String,
    pub family: Option<String>,
    pub loader: Option<String>,
    pub quantization: Option<String>,
    pub repo_id: Option<String>,
    pub resolved_commit: Option<String>,
    pub profile: Option<String>,
    pub snapshot_path: Option<PathBuf>,
    pub manifest_digest: Option<String>,
}

impl BackendModelMetadata {
    pub fn new(id: impl Into<String>, backend: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            backend: backend.into(),
            family: None,
            loader: None,
            quantization: None,
            repo_id: None,
            resolved_commit: None,
            profile: None,
            snapshot_path: None,
            manifest_digest: None,
        }
    }
}

#[async_trait]
pub trait ModelBackend: Send + Sync + 'static {
    fn model_id(&self) -> &str;

    fn model_metadata(&self) -> BackendModelMetadata {
        BackendModelMetadata::new(self.model_id(), "unknown")
    }

    async fn generate(&self, request: BackendRequest) -> Result<BackendOutput, BackendError>;

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        _cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        self.generate(request).await
    }

    fn generate_stream<'a>(
        &'a self,
        request: BackendRequest,
    ) -> BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        stream::once(async move {
            self.generate(request)
                .await
                .map(|output| BackendStreamChunk {
                    text: output.text,
                    prompt_tokens: output.prompt_tokens,
                    completion_tokens: output.completion_tokens,
                    finish_reason: Some(output.finish_reason),
                })
        })
        .boxed()
    }

    fn generate_stream_with_cancel<'a>(
        &'a self,
        request: BackendRequest,
        _cancellation: CancellationToken,
    ) -> BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        self.generate_stream(request)
    }
}

#[async_trait]
impl<T> ModelBackend for Box<T>
where
    T: ModelBackend + ?Sized,
{
    fn model_id(&self) -> &str {
        (**self).model_id()
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        (**self).model_metadata()
    }

    async fn generate(&self, request: BackendRequest) -> Result<BackendOutput, BackendError> {
        (**self).generate(request).await
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        (**self).generate_with_cancel(request, cancellation).await
    }

    fn generate_stream<'a>(
        &'a self,
        request: BackendRequest,
    ) -> BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        (**self).generate_stream(request)
    }

    fn generate_stream_with_cancel<'a>(
        &'a self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        (**self).generate_stream_with_cancel(request, cancellation)
    }
}

#[derive(Debug, Clone)]
pub struct DeterministicBackend {
    model_id: String,
    text: String,
    required_tool_protocol: bool,
    json_object_protocol: bool,
    conversation_protocol: bool,
}

impl DeterministicBackend {
    pub fn new(model_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            text: text.into(),
            required_tool_protocol: false,
            json_object_protocol: false,
            conversation_protocol: false,
        }
    }

    pub fn with_required_tool_protocol(mut self) -> Self {
        self.required_tool_protocol = true;
        self
    }

    pub fn with_json_object_protocol(mut self) -> Self {
        self.json_object_protocol = true;
        self
    }

    pub fn with_conversation_protocol(mut self) -> Self {
        self.conversation_protocol = true;
        self
    }
}

#[async_trait]
impl ModelBackend for DeterministicBackend {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        BackendModelMetadata::new(self.model_id.clone(), "deterministic")
    }

    async fn generate(&self, request: BackendRequest) -> Result<BackendOutput, BackendError> {
        if request.model != self.model_id {
            return Err(BackendError::ModelNotFound {
                requested: request.model,
                available: self.model_id.clone(),
            });
        }
        if !request.sampling.is_greedy() {
            return Err(BackendError::UnsupportedRequest(
                "deterministic backend does not support non-greedy sampling".to_owned(),
            ));
        }
        let (text, finish_reason) = if self.required_tool_protocol
            && let Some(name) = request.required_tool_choice
        {
            (
                serde_json::json!({
                    "name": name,
                    "arguments": {},
                })
                .to_string(),
                FinishReason::ToolCalls,
            )
        } else if self.json_object_protocol && request.json_object_mode {
            (
                serde_json::json!({
                    "response": "ok",
                })
                .to_string(),
                FinishReason::Stop,
            )
        } else if self.conversation_protocol && request.conversation_mode {
            (
                deterministic_conversation_response(&request.prompt)
                    .unwrap_or_else(|| self.text.clone()),
                FinishReason::Stop,
            )
        } else {
            (self.text.clone(), FinishReason::Stop)
        };
        let text = if matches!(finish_reason, FinishReason::ToolCalls) {
            format!("<tool_call>{text}</tool_call>")
        } else {
            text
        };
        Ok(BackendOutput {
            completion_tokens: count_tokens(&text),
            text,
            prompt_tokens: count_tokens(&request.prompt),
            finish_reason,
        })
    }
}

fn deterministic_conversation_response(prompt: &str) -> Option<String> {
    let prompt = prompt.to_ascii_lowercase();
    if prompt.contains("rewrite") && prompt.contains("feedback") {
        Some(
            "Revised poem: Dogs sprint through morning light, bright paws drumming home."
                .to_owned(),
        )
    } else if prompt.contains("critique") && prompt.contains("feedback") {
        Some(
            "Feedback: The dog poem has clear motion; add sharper images and a stronger final line."
                .to_owned(),
        )
    } else if prompt.contains("poem") && prompt.contains("dog") {
        Some("Dogs flash through rain-wet grass, brave hearts chasing the sun.".to_owned())
    } else {
        None
    }
}

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("model `{requested}` is not loaded; available model is `{available}`")]
    ModelNotFound {
        requested: String,
        available: String,
    },
    #[error("unsupported backend request: {0}")]
    UnsupportedRequest(String),
    #[error("backend error: {0}")]
    Other(String),
}

fn count_tokens(text: &str) -> u64 {
    let normalized = text
        .replace("<|im_start|>system", " ")
        .replace("<|im_start|>user", " ")
        .replace("<|im_start|>assistant", " ")
        .replace("<|im_start|>tool", " ")
        .replace("<|im_end|>", " ")
        .replace("<think>", " ")
        .replace("</think>", " ");
    normalized.split_whitespace().count().max(1) as u64
}

pub fn rms_norm_f32(input: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>, MathError> {
    rms_norm_with_weight_offset_f32(input, weight, eps, 0.0)
}

pub fn qwen_rms_norm_f32(input: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>, MathError> {
    rms_norm_with_weight_offset_f32(input, weight, eps, 1.0)
}

fn rms_norm_with_weight_offset_f32(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    weight_offset: f32,
) -> Result<Vec<f32>, MathError> {
    if input.len() != weight.len() {
        return Err(MathError::InvalidShape(
            "input and weight must have the same length".to_owned(),
        ));
    }
    if input.is_empty() {
        return Ok(Vec::new());
    }
    if eps < 0.0 {
        return Err(MathError::InvalidShape(
            "rms norm epsilon must be non-negative".to_owned(),
        ));
    }
    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let scale = (mean_square + eps).sqrt().recip();
    Ok(input
        .iter()
        .zip(weight)
        .map(|(value, weight)| value * scale * (weight_offset + weight))
        .collect())
}

pub fn matvec_row_major_f32(
    input: &[f32],
    weights: &[f32],
    rows: usize,
    columns: usize,
) -> Result<Vec<f32>, MathError> {
    if input.len() != columns {
        return Err(MathError::InvalidShape(format!(
            "input length {} does not match matvec columns {columns}",
            input.len()
        )));
    }
    let expected_weights = rows
        .checked_mul(columns)
        .ok_or_else(|| MathError::InvalidShape("matvec shape overflows usize".to_owned()))?;
    if weights.len() != expected_weights {
        return Err(MathError::InvalidShape(format!(
            "weight length {} does not match rows {rows} * columns {columns}",
            weights.len()
        )));
    }
    Ok(weights
        .chunks_exact(columns)
        .map(|row| {
            row.iter()
                .zip(input)
                .map(|(weight, value)| weight * value)
                .sum()
        })
        .collect())
}

pub fn matvecs_row_major_f32(
    inputs: &[Vec<f32>],
    weights: &[f32],
    rows: usize,
    columns: usize,
) -> Result<Vec<Vec<f32>>, MathError> {
    inputs
        .iter()
        .map(|input| matvec_row_major_f32(input, weights, rows, columns))
        .collect()
}

pub fn swiglu_mlp_f32(
    input: &[f32],
    gate_weight: &[f32],
    up_weight: &[f32],
    down_weight: &[f32],
    intermediate_size: usize,
) -> Result<Vec<f32>, MathError> {
    let gate = matvec_row_major_f32(input, gate_weight, intermediate_size, input.len())?;
    let up = matvec_row_major_f32(input, up_weight, intermediate_size, input.len())?;
    let activated = gate
        .iter()
        .zip(up)
        .map(|(gate, up)| silu_f32(*gate) * up)
        .collect::<Vec<_>>();
    if !down_weight.len().is_multiple_of(intermediate_size) {
        return Err(MathError::InvalidShape(format!(
            "down projection length {} is not divisible by intermediate size {intermediate_size}",
            down_weight.len()
        )));
    }
    let rows = down_weight.len() / intermediate_size;
    matvec_row_major_f32(&activated, down_weight, rows, intermediate_size)
}

pub fn silu_f32(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopKWeight {
    pub index: usize,
    pub weight: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopKLogit {
    pub index: usize,
    pub logit: f32,
}

pub fn softmax_top_k_f32(logits: &[f32], top_k: usize) -> Result<Vec<TopKWeight>, MathError> {
    if top_k == 0 || top_k > logits.len() {
        return Err(MathError::InvalidShape(format!(
            "top_k {top_k} must be in 1..={}",
            logits.len()
        )));
    }
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(MathError::InvalidShape(
            "router logits must be finite".to_owned(),
        ));
    }
    let mut selected = logits.iter().copied().enumerate().collect::<Vec<_>>();
    selected.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    selected.truncate(top_k);
    let max = selected
        .iter()
        .map(|(_, value)| *value)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut exp_values = selected
        .iter()
        .map(|(_, value)| (*value - max).exp())
        .collect::<Vec<_>>();
    let sum = exp_values.iter().sum::<f32>();
    if sum == 0.0 || !sum.is_finite() {
        return Err(MathError::InvalidShape(
            "router softmax denominator is invalid".to_owned(),
        ));
    }
    Ok(selected
        .iter()
        .zip(exp_values.iter_mut())
        .map(|((index, _), value)| TopKWeight {
            index: *index,
            weight: *value / sum,
        })
        .collect())
}

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
    let normalized = qwen_rms_norm_f32(&embedding, &norm_weight, rms_norm_eps).map_err(|err| {
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

pub fn qwen_layer_input_norm(
    store: &SafeTensorShardStore,
    layer_idx: usize,
    hidden_states: &[f32],
    hidden_size: usize,
    rms_norm_eps: f32,
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
    qwen_rms_norm_f32(hidden_states, &norm_weight, rms_norm_eps).map_err(|err| {
        TensorLoadError::integrity(format!("Qwen layer input RMSNorm failed: {err}"))
    })
}

fn qwen_layer_input_norm_sequence(
    store: &SafeTensorShardStore,
    layer_idx: usize,
    hidden_states: &[Vec<f32>],
    hidden_size: usize,
    rms_norm_eps: f32,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    let norm_weight = store.bf16_tensor_f32_range(
        &qwen_layer_tensor(layer_idx, "input_layernorm.weight"),
        0,
        hidden_size,
    )?;
    hidden_states
        .iter()
        .map(|hidden| {
            if hidden.len() != hidden_size {
                return Err(TensorLoadError::integrity(format!(
                    "Qwen layer input hidden length {} must match hidden size {hidden_size}",
                    hidden.len()
                )));
            }
            qwen_rms_norm_f32(hidden, &norm_weight, rms_norm_eps).map_err(|err| {
                TensorLoadError::integrity(format!("Qwen layer input RMSNorm failed: {err}"))
            })
        })
        .collect()
}

pub fn qwen_linear_attention_first_token_from_parts(
    dims: &QwenLinearAttentionDims,
    qkv: &[f32],
    z: &[f32],
    b: &[f32],
    conv1d_weight: &[f32],
    norm_weight: &[f32],
    out_proj_weight: &[f32],
) -> Result<Vec<f32>, MathError> {
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

    let mut mixed_qkv = vec![0.0; conv_dim];
    for channel in 0..conv_dim {
        let kernel_last = channel * dims.conv_kernel_size + (dims.conv_kernel_size - 1);
        mixed_qkv[channel] = silu_f32(qkv[channel] * conv1d_weight[kernel_last]);
    }

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
        let query_head = l2_normalize_f32(&query[key_start..key_start + dims.key_head_dim], 1e-6)?;
        let key_head_values =
            l2_normalize_f32(&key[key_start..key_start + dims.key_head_dim], 1e-6)?;
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
        let normalized = rms_norm_f32(&core_head, norm_weight, dims.rms_norm_eps)?;
        for offset in 0..dims.value_head_dim {
            gated[value_start + offset] = normalized[offset] * silu_f32(z[value_start + offset]);
        }
    }

    matvec_row_major_f32(&gated, out_proj_weight, dims.hidden_size, value_dim)
}

pub fn qwen_linear_attention_sequence_from_parts(
    dims: &QwenLinearAttentionDims,
    parts: &QwenLinearAttentionSequenceParts<'_>,
) -> Result<Vec<Vec<f32>>, MathError> {
    qwen_linear_attention_sequence_from_parts_impl(dims, parts, None)
}

pub fn qwen_linear_attention_sequence_with_cache_from_parts(
    dims: &QwenLinearAttentionDims,
    parts: &QwenLinearAttentionSequenceParts<'_>,
    cache: &mut LinearAttentionCache,
) -> Result<Vec<Vec<f32>>, MathError> {
    if cache.token_count() != 0 {
        return Err(MathError::InvalidShape(
            "Qwen linear attention prefill cache must be empty".to_owned(),
        ));
    }
    qwen_linear_attention_sequence_from_parts_impl(dims, parts, Some(cache))
}

pub fn qwen_linear_attention_step_with_cache_from_parts(
    dims: &QwenLinearAttentionDims,
    parts: &QwenLinearAttentionStepParts<'_>,
    cache: &mut LinearAttentionCache,
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
    qwen_linear_attention_sequence_from_parts_impl(dims, &sequence_parts, Some(cache))?
        .into_iter()
        .next()
        .ok_or_else(|| {
            MathError::InvalidShape("Qwen linear attention step returned no output".to_owned())
        })
}

fn qwen_linear_attention_sequence_from_parts_impl(
    dims: &QwenLinearAttentionDims,
    parts: &QwenLinearAttentionSequenceParts<'_>,
    mut cache: Option<&mut LinearAttentionCache>,
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
        let mixed_token = &mut mixed_tokens[token_idx];
        if let Some(cache) = cache.as_mut() {
            cache.push_conv_input(&qkv[token_idx]).map_err(|err| {
                MathError::InvalidShape(format!("linear attention cache update failed: {err}"))
            })?;
            for (channel, mixed_value) in mixed_token.iter_mut().enumerate() {
                let mut mixed = 0.0;
                for kernel_idx in 0..dims.conv_kernel_size {
                    mixed += cache.conv_window()[kernel_idx * conv_dim + channel]
                        * parts.conv1d_weight[channel * dims.conv_kernel_size + kernel_idx];
                }
                *mixed_value = silu_f32(mixed);
            }
        } else {
            for (channel, mixed_value) in mixed_token.iter_mut().enumerate() {
                let mut mixed = 0.0;
                for kernel_idx in 0..dims.conv_kernel_size {
                    let lookback = dims.conv_kernel_size - 1 - kernel_idx;
                    if token_idx >= lookback {
                        mixed += qkv[token_idx - lookback][channel]
                            * parts.conv1d_weight[channel * dims.conv_kernel_size + kernel_idx];
                    }
                }
                *mixed_value = silu_f32(mixed);
            }
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
            let query_head =
                l2_normalize_f32(&query[key_start..key_start + dims.key_head_dim], 1e-6)?;
            let key_head_values =
                l2_normalize_f32(&key[key_start..key_start + dims.key_head_dim], 1e-6)?;
            let query_scaled = query_head
                .into_iter()
                .map(|value| value * scale)
                .collect::<Vec<_>>();
            let beta = sigmoid_f32(b[token_idx][value_head]);
            let decay = (-parts.a_log[value_head].exp()
                * softplus_f32(a[token_idx][value_head] + parts.dt_bias[value_head]))
            .exp();

            let state_start = value_head * dims.key_head_dim * dims.value_head_dim;
            for state in &mut recurrent_state
                [state_start..state_start + dims.key_head_dim * dims.value_head_dim]
            {
                *state *= decay;
            }

            let mut memory = vec![0.0; dims.value_head_dim];
            for (key_offset, key_value) in key_head_values.iter().enumerate() {
                let state_row = state_start + key_offset * dims.value_head_dim;
                for value_offset in 0..dims.value_head_dim {
                    memory[value_offset] += recurrent_state[state_row + value_offset] * key_value;
                }
            }

            let mut delta = vec![0.0; dims.value_head_dim];
            for value_offset in 0..dims.value_head_dim {
                delta[value_offset] =
                    (value[value_start + value_offset] - memory[value_offset]) * beta;
            }
            for (key_offset, key_value) in key_head_values.iter().enumerate() {
                let state_row = state_start + key_offset * dims.value_head_dim;
                for value_offset in 0..dims.value_head_dim {
                    recurrent_state[state_row + value_offset] += key_value * delta[value_offset];
                }
            }

            let mut core_head = vec![0.0; dims.value_head_dim];
            for (key_offset, query_value) in query_scaled.iter().enumerate() {
                let state_row = state_start + key_offset * dims.value_head_dim;
                for value_offset in 0..dims.value_head_dim {
                    core_head[value_offset] +=
                        recurrent_state[state_row + value_offset] * query_value;
                }
            }
            let normalized = rms_norm_f32(&core_head, parts.norm_weight, dims.rms_norm_eps)?;
            for value_offset in 0..dims.value_head_dim {
                gated[value_start + value_offset] =
                    normalized[value_offset] * silu_f32(z[token_idx][value_start + value_offset]);
            }
        }
        outputs.push(matvec_row_major_f32(
            &gated,
            parts.out_proj_weight,
            dims.hidden_size,
            value_dim,
        )?);
    }

    if let Some(cache) = cache {
        cache
            .replace_recurrent_state(&recurrent_state)
            .map_err(|err| {
                MathError::InvalidShape(format!("linear attention cache update failed: {err}"))
            })?;
    }

    Ok(outputs)
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

pub fn qwen_full_attention_first_token_from_parts(
    dims: &QwenFullAttentionDims,
    q_proj: &[f32],
    v_proj: &[f32],
    o_proj_weight: &[f32],
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

    matvec_row_major_f32(&gated, o_proj_weight, dims.hidden_size, attention_dim)
}

pub fn qwen_full_attention_sequence_from_parts(
    dims: &QwenFullAttentionDims,
    parts: &QwenFullAttentionSequenceParts<'_>,
    config: QwenFullAttentionSequenceConfig,
) -> Result<Vec<Vec<f32>>, MathError> {
    qwen_full_attention_sequence_from_parts_impl(dims, parts, config, None)
}

pub fn qwen_full_attention_sequence_with_cache_from_parts(
    dims: &QwenFullAttentionDims,
    parts: &QwenFullAttentionSequenceParts<'_>,
    config: QwenFullAttentionSequenceConfig,
    cache: &mut LayerKvCache,
) -> Result<Vec<Vec<f32>>, MathError> {
    if cache.token_count() != 0 {
        return Err(MathError::InvalidShape(
            "Qwen full attention prefill cache must be empty".to_owned(),
        ));
    }
    qwen_full_attention_sequence_from_parts_impl(dims, parts, config, Some(cache))
}

pub fn qwen_full_attention_step_with_cache_from_parts(
    dims: &QwenFullAttentionDims,
    parts: &QwenFullAttentionStepParts<'_>,
    config: QwenFullAttentionSequenceConfig,
    cache: &mut LayerKvCache,
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
    require_len("q projection", parts.q_proj.len(), attention_dim * 2)?;
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

    let position = cache.token_count();
    let mut query = vec![0.0; attention_dim];
    let mut gate = vec![0.0; attention_dim];
    for head in 0..dims.num_attention_heads {
        let projected_head_start = head * dims.head_dim * 2;
        let q_start = head * dims.head_dim;
        let normalized = qwen_rms_norm_f32(
            &parts.q_proj[projected_head_start..projected_head_start + dims.head_dim],
            parts.q_norm_weight,
            config.rms_norm_eps,
        )?;
        query[q_start..q_start + dims.head_dim].copy_from_slice(&normalized);
        gate[q_start..q_start + dims.head_dim].copy_from_slice(
            &parts.q_proj
                [projected_head_start + dims.head_dim..projected_head_start + dims.head_dim * 2],
        );
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
        let normalized = qwen_rms_norm_f32(
            &parts.k_proj[head_start..head_start + dims.head_dim],
            parts.k_norm_weight,
            config.rms_norm_eps,
        )?;
        key[head_start..head_start + dims.head_dim].copy_from_slice(&normalized);
        apply_rope_to_head(
            &mut key[head_start..head_start + dims.head_dim],
            position,
            rotary_dim,
            config.rope_theta,
        );
    }
    cache
        .append(&key, parts.v_proj)
        .map_err(|err| MathError::InvalidShape(format!("KV cache append failed: {err}")))?;

    let groups = dims.num_attention_heads / dims.num_key_value_heads;
    let scale = (dims.head_dim as f32).sqrt().recip();
    let mut attended = vec![0.0; attention_dim];
    for head in 0..dims.num_attention_heads {
        let kv_head = head / groups;
        let q_start = head * dims.head_dim;
        let kv_start = kv_head * dims.head_dim;
        let mut scores = Vec::with_capacity(cache.token_count());
        for source_idx in 0..cache.token_count() {
            let key_token = cache
                .key(source_idx)
                .ok_or_else(|| MathError::InvalidShape("KV cache key missing".to_owned()))?;
            let score = query[q_start..q_start + dims.head_dim]
                .iter()
                .zip(&key_token[kv_start..kv_start + dims.head_dim])
                .map(|(query, key)| query * key)
                .sum::<f32>()
                * scale;
            scores.push(score);
        }
        let weights = softmax_f32(&scores)?;
        for (source_idx, weight) in weights.into_iter().enumerate() {
            let value_token = cache
                .value(source_idx)
                .ok_or_else(|| MathError::InvalidShape("KV cache value missing".to_owned()))?;
            for offset in 0..dims.head_dim {
                attended[q_start + offset] += weight * value_token[kv_start + offset];
            }
        }
        for offset in 0..dims.head_dim {
            attended[q_start + offset] *= sigmoid_f32(gate[q_start + offset]);
        }
    }

    matvec_row_major_f32(
        &attended,
        parts.o_proj_weight,
        dims.hidden_size,
        attention_dim,
    )
}

fn qwen_full_attention_sequence_from_parts_impl(
    dims: &QwenFullAttentionDims,
    parts: &QwenFullAttentionSequenceParts<'_>,
    config: QwenFullAttentionSequenceConfig,
    mut cache: Option<&mut LayerKvCache>,
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

    let mut queries = vec![vec![0.0; attention_dim]; seq_len];
    let mut gates = vec![vec![0.0; attention_dim]; seq_len];
    let mut keys = vec![vec![0.0; key_value_dim]; seq_len];
    for token_idx in 0..seq_len {
        require_len("q projection", q_proj[token_idx].len(), attention_dim * 2)?;
        require_len("k projection", k_proj[token_idx].len(), key_value_dim)?;
        require_len("v projection", v_proj[token_idx].len(), key_value_dim)?;

        for head in 0..dims.num_attention_heads {
            let projected_head_start = head * dims.head_dim * 2;
            let q_start = head * dims.head_dim;
            let query = qwen_rms_norm_f32(
                &q_proj[token_idx][projected_head_start..projected_head_start + dims.head_dim],
                parts.q_norm_weight,
                config.rms_norm_eps,
            )?;
            queries[token_idx][q_start..q_start + dims.head_dim].copy_from_slice(&query);
            gates[token_idx][q_start..q_start + dims.head_dim].copy_from_slice(
                &q_proj[token_idx][projected_head_start + dims.head_dim
                    ..projected_head_start + dims.head_dim * 2],
            );
            apply_rope_to_head(
                &mut queries[token_idx][q_start..q_start + dims.head_dim],
                token_idx,
                rotary_dim,
                config.rope_theta,
            );
        }
        for head in 0..dims.num_key_value_heads {
            let head_start = head * dims.head_dim;
            let key = qwen_rms_norm_f32(
                &k_proj[token_idx][head_start..head_start + dims.head_dim],
                parts.k_norm_weight,
                config.rms_norm_eps,
            )?;
            keys[token_idx][head_start..head_start + dims.head_dim].copy_from_slice(&key);
            apply_rope_to_head(
                &mut keys[token_idx][head_start..head_start + dims.head_dim],
                token_idx,
                rotary_dim,
                config.rope_theta,
            );
        }
    }
    if let Some(cache) = cache.as_deref_mut() {
        for token_idx in 0..seq_len {
            cache
                .append(&keys[token_idx], &v_proj[token_idx])
                .map_err(|err| MathError::InvalidShape(format!("KV cache append failed: {err}")))?;
        }
    }

    let groups = dims.num_attention_heads / dims.num_key_value_heads;
    let scale = (dims.head_dim as f32).sqrt().recip();
    let mut outputs = Vec::with_capacity(seq_len);
    for token_idx in 0..seq_len {
        let mut attended = vec![0.0; attention_dim];
        for head in 0..dims.num_attention_heads {
            let kv_head = head / groups;
            let q_start = head * dims.head_dim;
            let kv_start = kv_head * dims.head_dim;
            let mut scores = Vec::with_capacity(token_idx + 1);
            for (source_idx, local_key) in keys.iter().enumerate().take(token_idx + 1) {
                let key_token = cache
                    .as_deref()
                    .and_then(|cache| cache.key(source_idx))
                    .unwrap_or(local_key);
                let score = queries[token_idx][q_start..q_start + dims.head_dim]
                    .iter()
                    .zip(&key_token[kv_start..kv_start + dims.head_dim])
                    .map(|(query, key)| query * key)
                    .sum::<f32>()
                    * scale;
                scores.push(score);
            }
            let weights = softmax_f32(&scores)?;
            for (source_idx, weight) in weights.into_iter().enumerate() {
                let value_token = cache
                    .as_deref()
                    .and_then(|cache| cache.value(source_idx))
                    .unwrap_or(&v_proj[source_idx]);
                for offset in 0..dims.head_dim {
                    attended[q_start + offset] += weight * value_token[kv_start + offset];
                }
            }
            for offset in 0..dims.head_dim {
                attended[q_start + offset] *= sigmoid_f32(gates[token_idx][q_start + offset]);
            }
        }
        outputs.push(matvec_row_major_f32(
            &attended,
            parts.o_proj_weight,
            dims.hidden_size,
            attention_dim,
        )?);
    }

    Ok(outputs)
}

pub fn qwen_layer0_linear_attention_first_token(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    projections: &QwenLinearAttentionProjectionProbe,
) -> Result<Vec<f32>, TensorLoadError> {
    qwen_layer_linear_attention_first_token(store, spec, 0, projections)
}

pub fn qwen_layer_linear_attention_first_token(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    projections: &QwenLinearAttentionProjectionProbe,
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
    qwen_linear_attention_sequence_from_parts(
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
    )
    .map(|mut outputs| outputs.remove(0))
    .map_err(|err| {
        TensorLoadError::integrity(format!("Qwen layer0 linear attention failed: {err}"))
    })
}

pub fn qwen_layer_linear_attention_sequence(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    qwen_layer_linear_attention_sequence_impl(store, spec, layer_idx, hidden_states, None)
}

pub fn qwen_layer_linear_attention_sequence_with_cache(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[Vec<f32>],
    cache: &mut LinearAttentionCache,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    qwen_layer_linear_attention_sequence_impl(store, spec, layer_idx, hidden_states, Some(cache))
}

fn qwen_layer_linear_attention_sequence_impl(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[Vec<f32>],
    cache: Option<&mut LinearAttentionCache>,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    let projections = QwenLinearAttentionProjectionSequence {
        qkv: store.bf16_matvecs_row_major_f32(
            &qwen_linear_attn_tensor(layer_idx, "in_proj_qkv.weight"),
            hidden_states,
        )?,
        z: store.bf16_matvecs_row_major_f32(
            &qwen_linear_attn_tensor(layer_idx, "in_proj_z.weight"),
            hidden_states,
        )?,
        b: store.bf16_matvecs_row_major_f32(
            &qwen_linear_attn_tensor(layer_idx, "in_proj_b.weight"),
            hidden_states,
        )?,
        a: store.bf16_matvecs_row_major_f32(
            &qwen_linear_attn_tensor(layer_idx, "in_proj_a.weight"),
            hidden_states,
        )?,
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
        qwen_linear_attention_sequence_with_cache_from_parts(&dims, &parts, cache)
    } else {
        qwen_linear_attention_sequence_from_parts(&dims, &parts)
    };
    result.map_err(|err| {
        TensorLoadError::integrity(format!(
            "Qwen layer{layer_idx} linear attention sequence failed: {err}"
        ))
    })
}

pub fn qwen_layer_linear_attention_step_with_cache(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[f32],
    cache: &mut LinearAttentionCache,
) -> Result<Vec<f32>, TensorLoadError> {
    let projections = qwen_layer_linear_attention_projections(store, layer_idx, hidden_states)?;
    let dims = QwenLinearAttentionDims::from_spec(spec);
    let dt_bias = store.bf16_tensor_f32(&qwen_linear_attn_tensor(layer_idx, "dt_bias"))?;
    let a_log = store.bf16_tensor_f32(&qwen_linear_attn_tensor(layer_idx, "A_log"))?;
    let conv1d_weight =
        store.bf16_tensor_f32(&qwen_linear_attn_tensor(layer_idx, "conv1d.weight"))?;
    let norm_weight = store.bf16_tensor_f32(&qwen_linear_attn_tensor(layer_idx, "norm.weight"))?;
    let out_proj_weight =
        store.bf16_tensor_f32(&qwen_linear_attn_tensor(layer_idx, "out_proj.weight"))?;
    qwen_linear_attention_step_with_cache_from_parts(
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
    )
    .map_err(|err| {
        TensorLoadError::integrity(format!(
            "Qwen layer{layer_idx} linear attention step failed: {err}"
        ))
    })
}

pub fn qwen_layer_full_attention_first_token(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[f32],
) -> Result<Vec<f32>, TensorLoadError> {
    let dims = QwenFullAttentionDims::from_spec(spec);
    let q_proj = store.bf16_matvec_row_major_f32(
        &qwen_self_attn_tensor(layer_idx, "q_proj.weight"),
        hidden_states,
    )?;
    let k_proj = store.bf16_matvec_row_major_f32(
        &qwen_self_attn_tensor(layer_idx, "k_proj.weight"),
        hidden_states,
    )?;
    let v_proj = store.bf16_matvec_row_major_f32(
        &qwen_self_attn_tensor(layer_idx, "v_proj.weight"),
        hidden_states,
    )?;
    let q_norm_weight =
        store.bf16_tensor_f32(&qwen_self_attn_tensor(layer_idx, "q_norm.weight"))?;
    let k_norm_weight =
        store.bf16_tensor_f32(&qwen_self_attn_tensor(layer_idx, "k_norm.weight"))?;
    let o_proj_weight =
        store.bf16_tensor_f32(&qwen_self_attn_tensor(layer_idx, "o_proj.weight"))?;
    let q_proj = vec![q_proj];
    let k_proj = vec![k_proj];
    let v_proj = vec![v_proj];
    qwen_full_attention_sequence_from_parts(
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
        },
    )
    .map(|mut outputs| outputs.remove(0))
    .map_err(|err| {
        TensorLoadError::integrity(format!(
            "Qwen layer{layer_idx} full attention failed: {err}"
        ))
    })
}

pub fn qwen_layer_full_attention_sequence(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    qwen_layer_full_attention_sequence_impl(store, spec, layer_idx, hidden_states, None)
}

pub fn qwen_layer_full_attention_sequence_with_cache(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[Vec<f32>],
    cache: &mut LayerKvCache,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    qwen_layer_full_attention_sequence_impl(store, spec, layer_idx, hidden_states, Some(cache))
}

fn qwen_layer_full_attention_sequence_impl(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[Vec<f32>],
    cache: Option<&mut LayerKvCache>,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    let dims = QwenFullAttentionDims::from_spec(spec);
    let q_proj = store.bf16_matvecs_row_major_f32(
        &qwen_self_attn_tensor(layer_idx, "q_proj.weight"),
        hidden_states,
    )?;
    let k_proj = store.bf16_matvecs_row_major_f32(
        &qwen_self_attn_tensor(layer_idx, "k_proj.weight"),
        hidden_states,
    )?;
    let v_proj = store.bf16_matvecs_row_major_f32(
        &qwen_self_attn_tensor(layer_idx, "v_proj.weight"),
        hidden_states,
    )?;
    let q_norm_weight =
        store.bf16_tensor_f32(&qwen_self_attn_tensor(layer_idx, "q_norm.weight"))?;
    let k_norm_weight =
        store.bf16_tensor_f32(&qwen_self_attn_tensor(layer_idx, "k_norm.weight"))?;
    let o_proj_weight =
        store.bf16_tensor_f32(&qwen_self_attn_tensor(layer_idx, "o_proj.weight"))?;
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
    };
    let result = if let Some(cache) = cache {
        qwen_full_attention_sequence_with_cache_from_parts(&dims, &parts, config, cache)
    } else {
        qwen_full_attention_sequence_from_parts(&dims, &parts, config)
    };
    result.map_err(|err| {
        TensorLoadError::integrity(format!(
            "Qwen layer{layer_idx} full attention sequence failed: {err}"
        ))
    })
}

pub fn qwen_layer0_linear_attention_projections(
    store: &SafeTensorShardStore,
    hidden_states: &[f32],
) -> Result<QwenLinearAttentionProjectionProbe, TensorLoadError> {
    qwen_layer_linear_attention_projections(store, 0, hidden_states)
}

pub fn qwen_layer_linear_attention_projections(
    store: &SafeTensorShardStore,
    layer_idx: usize,
    hidden_states: &[f32],
) -> Result<QwenLinearAttentionProjectionProbe, TensorLoadError> {
    Ok(QwenLinearAttentionProjectionProbe {
        qkv: store.bf16_matvec_row_major_f32(
            &qwen_linear_attn_tensor(layer_idx, "in_proj_qkv.weight"),
            hidden_states,
        )?,
        z: store.bf16_matvec_row_major_f32(
            &qwen_linear_attn_tensor(layer_idx, "in_proj_z.weight"),
            hidden_states,
        )?,
        b: store.bf16_matvec_row_major_f32(
            &qwen_linear_attn_tensor(layer_idx, "in_proj_b.weight"),
            hidden_states,
        )?,
        a: store.bf16_matvec_row_major_f32(
            &qwen_linear_attn_tensor(layer_idx, "in_proj_a.weight"),
            hidden_states,
        )?,
    })
}

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
    let logits = store
        .bf16_matvec_row_major_f32(&qwen_mlp_tensor(layer_idx, "gate.weight"), hidden_states)?;
    let selected = softmax_top_k_f32(&logits, top_k)
        .map_err(|err| TensorLoadError::integrity(format!("Qwen MoE router failed: {err}")))?;
    Ok(QwenMoeRouterProbe { logits, selected })
}

pub fn qwen_layer0_post_attention_norm(
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
}

pub fn qwen_layer_post_attention_norm(
    store: &SafeTensorShardStore,
    layer_idx: usize,
    residual: &[f32],
    attention_output: &[f32],
    hidden_size: usize,
    rms_norm_eps: f32,
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
    qwen_rms_norm_f32(&hidden_states, &norm_weight, rms_norm_eps).map_err(|err| {
        TensorLoadError::integrity(format!("Qwen layer0 post-attention RMSNorm failed: {err}"))
    })
}

fn qwen_layer_post_attention_norm_sequence(
    store: &SafeTensorShardStore,
    layer_idx: usize,
    residual: &[Vec<f32>],
    attention_output: &[Vec<f32>],
    hidden_size: usize,
    rms_norm_eps: f32,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    if residual.len() != attention_output.len() {
        return Err(TensorLoadError::integrity(
            "Qwen post-attention sequence lengths must match",
        ));
    }
    let norm_weight = store.bf16_tensor_f32_range(
        &qwen_layer_tensor(layer_idx, "post_attention_layernorm.weight"),
        0,
        hidden_size,
    )?;
    residual
        .iter()
        .zip(attention_output)
        .map(|(residual, attention)| {
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
            qwen_rms_norm_f32(&hidden_states, &norm_weight, rms_norm_eps).map_err(|err| {
                TensorLoadError::integrity(format!(
                    "Qwen post-attention RMSNorm sequence failed: {err}"
                ))
            })
        })
        .collect()
}

pub fn qwen_layer0_moe_forward(
    store: &SafeTensorShardStore,
    dims: &QwenMoeDims,
    hidden_states: &[f32],
    router: &QwenMoeRouterProbe,
) -> Result<Vec<f32>, TensorLoadError> {
    qwen_layer_moe_forward(store, 0, dims, hidden_states, router)
}

pub fn qwen_linear_decoder_layer_first_token(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[f32],
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
    let hidden_size = spec.hidden_size as usize;
    let input_norm = qwen_layer_input_norm(
        store,
        layer_idx,
        hidden_states,
        hidden_size,
        spec.rms_norm_eps,
    )?;
    let projections = qwen_layer_linear_attention_projections(store, layer_idx, &input_norm)?;
    let attention_output =
        qwen_layer_linear_attention_first_token(store, spec, layer_idx, &projections)?;
    let post_attention = qwen_layer_post_attention_norm(
        store,
        layer_idx,
        hidden_states,
        &attention_output,
        hidden_size,
        spec.rms_norm_eps,
    )?;
    let router = qwen_layer_moe_router(
        store,
        layer_idx,
        &post_attention,
        spec.num_experts_per_tok as usize,
    )?;
    let moe_output = qwen_layer_moe_forward(
        store,
        layer_idx,
        &QwenMoeDims::from_spec(spec),
        &post_attention,
        &router,
    )?;
    hidden_states
        .iter()
        .zip(attention_output)
        .zip(moe_output)
        .map(|((hidden, attention), moe)| Ok(hidden + attention + moe))
        .collect()
}

pub fn qwen_full_decoder_layer_first_token(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[f32],
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
    let hidden_size = spec.hidden_size as usize;
    let input_norm = qwen_layer_input_norm(
        store,
        layer_idx,
        hidden_states,
        hidden_size,
        spec.rms_norm_eps,
    )?;
    let attention_output =
        qwen_layer_full_attention_first_token(store, spec, layer_idx, &input_norm)?;
    let post_attention = qwen_layer_post_attention_norm(
        store,
        layer_idx,
        hidden_states,
        &attention_output,
        hidden_size,
        spec.rms_norm_eps,
    )?;
    let router = qwen_layer_moe_router(
        store,
        layer_idx,
        &post_attention,
        spec.num_experts_per_tok as usize,
    )?;
    let moe_output = qwen_layer_moe_forward(
        store,
        layer_idx,
        &QwenMoeDims::from_spec(spec),
        &post_attention,
        &router,
    )?;
    hidden_states
        .iter()
        .zip(attention_output)
        .zip(moe_output)
        .map(|((hidden, attention), moe)| Ok(hidden + attention + moe))
        .collect()
}

pub fn qwen_decoder_layer_first_token(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[f32],
) -> Result<Vec<f32>, TensorLoadError> {
    match spec.layer_kinds.get(layer_idx) {
        Some(AttentionKind::LinearAttention) => {
            qwen_linear_decoder_layer_first_token(store, spec, layer_idx, hidden_states)
        }
        Some(AttentionKind::FullAttention) => {
            qwen_full_decoder_layer_first_token(store, spec, layer_idx, hidden_states)
        }
        None => Err(TensorLoadError::missing(format!(
            "Qwen layer {layer_idx} is outside configured layer count"
        ))),
    }
}

pub fn qwen_decoder_layer_sequence(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    qwen_decoder_layer_sequence_impl(store, spec, layer_idx, hidden_states, None)
}

pub fn qwen_decoder_layer_sequence_with_cache(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[Vec<f32>],
    cache: &mut QwenLayerCache,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    qwen_decoder_layer_sequence_impl(store, spec, layer_idx, hidden_states, Some(cache))
}

fn qwen_decoder_layer_sequence_impl(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    layer_idx: usize,
    hidden_states: &[Vec<f32>],
    cache: Option<&mut QwenLayerCache>,
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    let hidden_size = spec.hidden_size as usize;
    let input_norm = qwen_layer_input_norm_sequence(
        store,
        layer_idx,
        hidden_states,
        hidden_size,
        spec.rms_norm_eps,
    )?;
    let attention_output = match spec.layer_kinds.get(layer_idx) {
        Some(AttentionKind::LinearAttention) => match cache {
            Some(QwenLayerCache::Linear(cache)) => qwen_layer_linear_attention_sequence_with_cache(
                store,
                spec,
                layer_idx,
                &input_norm,
                cache,
            )?,
            Some(QwenLayerCache::Full(_)) => {
                return Err(TensorLoadError::integrity(format!(
                    "Qwen layer{layer_idx} expected linear attention cache"
                )));
            }
            None => qwen_layer_linear_attention_sequence(store, spec, layer_idx, &input_norm)?,
        },
        Some(AttentionKind::FullAttention) => match cache {
            Some(QwenLayerCache::Full(cache)) => qwen_layer_full_attention_sequence_with_cache(
                store,
                spec,
                layer_idx,
                &input_norm,
                cache,
            )?,
            Some(QwenLayerCache::Linear(_)) => {
                return Err(TensorLoadError::integrity(format!(
                    "Qwen layer{layer_idx} expected full attention cache"
                )));
            }
            None => qwen_layer_full_attention_sequence(store, spec, layer_idx, &input_norm)?,
        },
        None => {
            return Err(TensorLoadError::missing(format!(
                "Qwen layer {layer_idx} is outside configured layer count"
            )));
        }
    };
    let post_attention = qwen_layer_post_attention_norm_sequence(
        store,
        layer_idx,
        hidden_states,
        &attention_output,
        hidden_size,
        spec.rms_norm_eps,
    )?;
    let moe_dims = QwenMoeDims::from_spec(spec);
    hidden_states
        .iter()
        .zip(attention_output)
        .zip(post_attention)
        .map(|((hidden, attention), post_attention)| {
            let router = qwen_layer_moe_router(
                store,
                layer_idx,
                &post_attention,
                spec.num_experts_per_tok as usize,
            )?;
            let moe_output =
                qwen_layer_moe_forward(store, layer_idx, &moe_dims, &post_attention, &router)?;
            Ok(hidden
                .iter()
                .zip(attention)
                .zip(moe_output)
                .map(|((hidden, attention), moe)| hidden + attention + moe)
                .collect::<Vec<_>>())
        })
        .collect()
}

pub fn qwen_prefill_sequence(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    token_ids: &[usize],
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    let mut hidden_states = qwen_embedding_sequence(store, token_ids, spec.hidden_size as usize)?;
    for layer_idx in 0..spec.num_hidden_layers as usize {
        hidden_states = qwen_decoder_layer_sequence(store, spec, layer_idx, &hidden_states)?;
    }
    Ok(hidden_states)
}

pub fn qwen_prefill_sequence_with_cache(
    store: &SafeTensorShardStore,
    spec: &QwenModelSpec,
    token_ids: &[usize],
    caches: &mut [QwenLayerCache],
) -> Result<Vec<Vec<f32>>, TensorLoadError> {
    let layer_count = spec.num_hidden_layers as usize;
    if caches.len() != layer_count {
        return Err(TensorLoadError::integrity(format!(
            "Qwen prefill expected {layer_count} layer caches, got {}",
            caches.len()
        )));
    }
    let mut hidden_states = qwen_embedding_sequence(store, token_ids, spec.hidden_size as usize)?;
    for (layer_idx, cache) in caches.iter_mut().enumerate().take(layer_count) {
        hidden_states =
            qwen_decoder_layer_sequence_with_cache(store, spec, layer_idx, &hidden_states, cache)?;
    }
    Ok(hidden_states)
}

pub fn qwen_layer_moe_forward(
    store: &SafeTensorShardStore,
    layer_idx: usize,
    dims: &QwenMoeDims,
    hidden_states: &[f32],
    router: &QwenMoeRouterProbe,
) -> Result<Vec<f32>, TensorLoadError> {
    if hidden_states.len() != dims.hidden_size {
        return Err(TensorLoadError::integrity(format!(
            "Qwen MoE hidden length {} must match hidden size {}",
            hidden_states.len(),
            dims.hidden_size
        )));
    }
    let mut output = vec![0.0; dims.hidden_size];
    let gate_up_expert_elements = dims
        .moe_intermediate_size
        .checked_mul(2)
        .and_then(|rows| rows.checked_mul(dims.hidden_size))
        .ok_or_else(|| TensorLoadError::integrity("Qwen expert gate/up shape overflow"))?;
    let down_expert_elements = dims
        .hidden_size
        .checked_mul(dims.moe_intermediate_size)
        .ok_or_else(|| TensorLoadError::integrity("Qwen expert down shape overflow"))?;
    for selected in &router.selected {
        if selected.index >= dims.num_experts {
            return Err(TensorLoadError::integrity(format!(
                "Qwen selected expert {} exceeds expert count {}",
                selected.index, dims.num_experts
            )));
        }
        let gate_up = store.bf16_tensor_f32_range(
            &qwen_mlp_tensor(layer_idx, "experts.gate_up_proj"),
            selected.index * gate_up_expert_elements,
            gate_up_expert_elements,
        )?;
        let split = dims
            .moe_intermediate_size
            .checked_mul(dims.hidden_size)
            .ok_or_else(|| TensorLoadError::integrity("Qwen expert split shape overflow"))?;
        let down = store.bf16_tensor_f32_range(
            &qwen_mlp_tensor(layer_idx, "experts.down_proj"),
            selected.index * down_expert_elements,
            down_expert_elements,
        )?;
        let expert_output = swiglu_mlp_f32(
            hidden_states,
            &gate_up[..split],
            &gate_up[split..],
            &down,
            dims.moe_intermediate_size,
        )
        .map_err(|err| {
            TensorLoadError::integrity(format!("Qwen selected expert MLP failed: {err}"))
        })?;
        for (output, expert) in output.iter_mut().zip(expert_output) {
            *output += expert * selected.weight;
        }
    }

    let shared_gate = store.bf16_tensor_f32(&qwen_mlp_tensor(
        layer_idx,
        "shared_expert.gate_proj.weight",
    ))?;
    let shared_up =
        store.bf16_tensor_f32(&qwen_mlp_tensor(layer_idx, "shared_expert.up_proj.weight"))?;
    let shared_down = store.bf16_tensor_f32(&qwen_mlp_tensor(
        layer_idx,
        "shared_expert.down_proj.weight",
    ))?;
    let shared_output = swiglu_mlp_f32(
        hidden_states,
        &shared_gate,
        &shared_up,
        &shared_down,
        dims.shared_expert_intermediate_size,
    )
    .map_err(|err| TensorLoadError::integrity(format!("Qwen shared expert MLP failed: {err}")))?;
    let shared_expert_gate =
        store.bf16_tensor_f32(&qwen_mlp_tensor(layer_idx, "shared_expert_gate.weight"))?;
    let shared_gate = matvec_row_major_f32(hidden_states, &shared_expert_gate, 1, dims.hidden_size)
        .map_err(|err| {
            TensorLoadError::integrity(format!("Qwen shared expert gate failed: {err}"))
        })?
        .into_iter()
        .next()
        .ok_or_else(|| TensorLoadError::integrity("Qwen shared expert gate returned no value"))?;
    let shared_gate = sigmoid_f32(shared_gate);
    for (output, shared) in output.iter_mut().zip(shared_output) {
        *output += shared_gate * shared;
    }
    Ok(output)
}

fn qwen_layer_tensor(layer_idx: usize, suffix: &str) -> String {
    format!("model.language_model.layers.{layer_idx}.{suffix}")
}

fn qwen_linear_attn_tensor(layer_idx: usize, suffix: &str) -> String {
    qwen_layer_tensor(layer_idx, &format!("linear_attn.{suffix}"))
}

fn qwen_mlp_tensor(layer_idx: usize, suffix: &str) -> String {
    qwen_layer_tensor(layer_idx, &format!("mlp.{suffix}"))
}

fn qwen_self_attn_tensor(layer_idx: usize, suffix: &str) -> String {
    qwen_layer_tensor(layer_idx, &format!("self_attn.{suffix}"))
}

pub fn qwen_final_norm(
    store: &SafeTensorShardStore,
    hidden_states: &[f32],
    hidden_size: usize,
    rms_norm_eps: f32,
) -> Result<Vec<f32>, TensorLoadError> {
    if hidden_states.len() != hidden_size {
        return Err(TensorLoadError::integrity(format!(
            "Qwen final norm hidden length {} must match hidden size {hidden_size}",
            hidden_states.len()
        )));
    }
    let norm_weight = store.bf16_tensor_f32_range(QWEN_FINAL_NORM_WEIGHT, 0, hidden_size)?;
    qwen_rms_norm_f32(hidden_states, &norm_weight, rms_norm_eps)
        .map_err(|err| TensorLoadError::integrity(format!("Qwen final RMSNorm failed: {err}")))
}

pub fn qwen_lm_head_top_k(
    store: &SafeTensorShardStore,
    hidden_states: &[f32],
    top_k: usize,
    chunk_rows: usize,
) -> Result<Vec<TopKLogit>, TensorLoadError> {
    store.bf16_matvec_top_k_rows_f32(QWEN_LM_HEAD_WEIGHT, hidden_states, top_k, chunk_rows)
}

pub fn qwen_lm_head_logits(
    store: &SafeTensorShardStore,
    hidden_states: &[f32],
    chunk_rows: usize,
) -> Result<Vec<f32>, TensorLoadError> {
    store.bf16_matvec_rows_f32(QWEN_LM_HEAD_WEIGHT, hidden_states, chunk_rows)
}

fn require_len(name: &str, actual: usize, expected: usize) -> Result<(), MathError> {
    if actual == expected {
        Ok(())
    } else {
        Err(MathError::InvalidShape(format!(
            "{name} length {actual} does not match expected {expected}"
        )))
    }
}

fn sigmoid_f32(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn softmax_f32(scores: &[f32]) -> Result<Vec<f32>, MathError> {
    if scores.is_empty() {
        return Ok(Vec::new());
    }
    if scores.iter().any(|value| !value.is_finite()) {
        return Err(MathError::InvalidShape(
            "softmax scores must be finite".to_owned(),
        ));
    }
    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_scores = scores
        .iter()
        .map(|score| (*score - max_score).exp())
        .collect::<Vec<_>>();
    let sum = exp_scores.iter().sum::<f32>();
    if sum == 0.0 || !sum.is_finite() {
        return Err(MathError::InvalidShape(
            "softmax denominator is invalid".to_owned(),
        ));
    }
    Ok(exp_scores.into_iter().map(|value| value / sum).collect())
}

fn softplus_f32(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        (1.0 + value.exp()).ln()
    }
}

fn apply_rope_to_head(head: &mut [f32], position: usize, rotary_dim: usize, theta: f32) {
    if rotary_dim == 0 {
        return;
    }
    let half = rotary_dim / 2;
    for offset in 0..half {
        let inv_freq = theta.powf(-((2 * offset) as f32) / rotary_dim as f32);
        let angle = position as f32 * inv_freq;
        let (sin, cos) = angle.sin_cos();
        let first = head[offset];
        let second = head[offset + half];
        head[offset] = first * cos - second * sin;
        head[offset + half] = second * cos + first * sin;
    }
}

fn l2_normalize_f32(input: &[f32], eps: f32) -> Result<Vec<f32>, MathError> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    if eps < 0.0 {
        return Err(MathError::InvalidShape(
            "l2 norm epsilon must be non-negative".to_owned(),
        ));
    }
    let inv_norm = (input.iter().map(|value| value * value).sum::<f32>() + eps)
        .sqrt()
        .recip();
    Ok(input.iter().map(|value| value * inv_norm).collect())
}

#[derive(Debug, Error)]
pub enum MathError {
    #[error("invalid math shape: {0}")]
    InvalidShape(String),
}

#[derive(Debug, Clone)]
pub struct SafeTensorArchive {
    bytes: Vec<u8>,
}

impl SafeTensorArchive {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TensorLoadError> {
        SafeTensors::deserialize(bytes)
            .map_err(|err| TensorLoadError::integrity(format!("invalid safetensors: {err}")))?;
        Ok(Self {
            bytes: bytes.to_vec(),
        })
    }

    pub fn tensor_metadata(&self, name: &str) -> Result<TensorMetadata, TensorLoadError> {
        let tensors = self.tensors()?;
        let view = tensors
            .tensor(name)
            .map_err(|err| TensorLoadError::missing(format!("tensor `{name}` not found: {err}")))?;
        Ok(TensorMetadata {
            name: name.to_owned(),
            dtype: format!("{:?}", view.dtype()),
            shape: view.shape().to_vec(),
            byte_len: view.data().len(),
        })
    }

    pub fn f32_tensor(&self, name: &str) -> Result<Vec<f32>, TensorLoadError> {
        let tensors = self.tensors()?;
        let view = tensors
            .tensor(name)
            .map_err(|err| TensorLoadError::missing(format!("tensor `{name}` not found: {err}")))?;
        if view.dtype() != Dtype::F32 {
            return Err(TensorLoadError::unsupported(format!(
                "tensor `{name}` has dtype {:?}, expected F32",
                view.dtype()
            )));
        }
        let data = view.data();
        if data.len() % std::mem::size_of::<f32>() != 0 {
            return Err(TensorLoadError::integrity(format!(
                "tensor `{name}` byte length is not divisible by 4"
            )));
        }
        Ok(data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk has length 4")))
            .collect())
    }

    fn tensors(&self) -> Result<SafeTensors<'_>, TensorLoadError> {
        SafeTensors::deserialize(&self.bytes)
            .map_err(|err| TensorLoadError::integrity(format!("invalid safetensors: {err}")))
    }
}

#[derive(Debug, Clone)]
pub struct SafeTensorHeader {
    source_path: Option<PathBuf>,
    file_len: u64,
    header_len: u64,
    data_start: u64,
    tensors: BTreeMap<String, TensorHeaderEntry>,
}

impl SafeTensorHeader {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TensorLoadError> {
        let file_len = bytes.len() as u64;
        let (header_len, header_end) = read_header_prefix(bytes, file_len)?;
        let header = bytes
            .get(8..header_end)
            .ok_or_else(|| TensorLoadError::integrity("safetensors header is truncated"))?;
        Self::from_header_bytes(None, file_len, header_len, header)
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TensorLoadError> {
        let path = path.as_ref();
        let mut file = File::open(path).map_err(|err| {
            TensorLoadError::missing(format!(
                "could not open safetensors file `{}`: {err}",
                path.display()
            ))
        })?;
        let file_len = file
            .metadata()
            .map_err(|err| {
                TensorLoadError::integrity(format!(
                    "could not read metadata for `{}`: {err}",
                    path.display()
                ))
            })?
            .len();
        let mut prefix = [0_u8; 8];
        file.read_exact(&mut prefix).map_err(|err| {
            TensorLoadError::integrity(format!(
                "could not read safetensors header prefix from `{}`: {err}",
                path.display()
            ))
        })?;
        let header_len = validate_header_len(u64::from_le_bytes(prefix), file_len)?;
        let mut header = vec![0_u8; usize_from_u64(header_len, "safetensors header is too large")?];
        file.read_exact(&mut header).map_err(|err| {
            TensorLoadError::integrity(format!(
                "could not read safetensors header from `{}`: {err}",
                path.display()
            ))
        })?;
        Self::from_header_bytes(Some(path.to_path_buf()), file_len, header_len, &header)
    }

    pub fn source_path(&self) -> Option<&Path> {
        self.source_path.as_deref()
    }

    pub fn file_len(&self) -> u64 {
        self.file_len
    }

    pub fn header_len(&self) -> u64 {
        self.header_len
    }

    pub fn data_start(&self) -> u64 {
        self.data_start
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn tensor_metadata(&self, name: &str) -> Result<TensorMetadata, TensorLoadError> {
        let tensor = self.tensor_entry(name)?;
        Ok(TensorMetadata {
            name: name.to_owned(),
            dtype: tensor.dtype.clone(),
            shape: tensor.shape.clone(),
            byte_len: tensor.byte_len()?,
        })
    }

    pub fn tensor_data_range(&self, name: &str) -> Result<Range<u64>, TensorLoadError> {
        let tensor = self.tensor_entry(name)?;
        let start = self
            .data_start
            .checked_add(tensor.data_offsets[0])
            .ok_or_else(|| {
                TensorLoadError::integrity(format!("tensor `{name}` offset overflow"))
            })?;
        let end = self
            .data_start
            .checked_add(tensor.data_offsets[1])
            .ok_or_else(|| {
                TensorLoadError::integrity(format!("tensor `{name}` offset overflow"))
            })?;
        Ok(start..end)
    }

    fn from_header_bytes(
        source_path: Option<PathBuf>,
        file_len: u64,
        header_len: u64,
        header: &[u8],
    ) -> Result<Self, TensorLoadError> {
        let data_start = 8_u64
            .checked_add(header_len)
            .ok_or_else(|| TensorLoadError::integrity("safetensors header length overflow"))?;
        let payload_len = file_len
            .checked_sub(data_start)
            .ok_or_else(|| TensorLoadError::integrity("safetensors payload is truncated"))?;
        let root: serde_json::Value = serde_json::from_slice(header).map_err(|err| {
            TensorLoadError::integrity(format!("invalid safetensors header json: {err}"))
        })?;
        let object = root.as_object().ok_or_else(|| {
            TensorLoadError::integrity("safetensors header must be a json object")
        })?;
        let mut tensors = BTreeMap::new();
        for (name, value) in object {
            if name == "__metadata__" {
                continue;
            }
            tensors.insert(
                name.clone(),
                TensorHeaderEntry::from_json(name, value, payload_len)?,
            );
        }
        if tensors.is_empty() {
            return Err(TensorLoadError::integrity(
                "safetensors header does not contain tensors",
            ));
        }
        Ok(Self {
            source_path,
            file_len,
            header_len,
            data_start,
            tensors,
        })
    }

    fn tensor_entry(&self, name: &str) -> Result<&TensorHeaderEntry, TensorLoadError> {
        self.tensors
            .get(name)
            .ok_or_else(|| TensorLoadError::missing(format!("tensor `{name}` not found")))
    }
}

#[derive(Debug)]
pub struct SafeTensorFile {
    header: SafeTensorHeader,
    file: File,
    mapped: Mutex<Option<Arc<Mmap>>>,
}

impl SafeTensorFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, TensorLoadError> {
        let path = path.as_ref();
        let header = SafeTensorHeader::from_file(path)?;
        let file = File::open(path).map_err(|err| {
            TensorLoadError::missing(format!(
                "could not open safetensors file `{}`: {err}",
                path.display()
            ))
        })?;
        Ok(Self {
            header,
            file,
            mapped: Mutex::new(None),
        })
    }

    pub fn header(&self) -> &SafeTensorHeader {
        &self.header
    }

    pub fn tensor_metadata(&self, name: &str) -> Result<TensorMetadata, TensorLoadError> {
        self.header.tensor_metadata(name)
    }

    pub fn materialize(&self) -> Result<usize, TensorLoadError> {
        let mapped = self.materialized_file()?;
        Ok(mapped.len())
    }

    pub fn is_materialized(&self) -> bool {
        self.mapped
            .lock()
            .map(|mapped| mapped.is_some())
            .unwrap_or(false)
    }

    pub fn tensor_bytes_range(
        &self,
        name: &str,
        tensor_byte_offset: u64,
        byte_len: usize,
    ) -> Result<Vec<u8>, TensorLoadError> {
        let file_range = self.tensor_file_byte_range(name, tensor_byte_offset, byte_len)?;
        if let Some(mapped) = self.materialized_file_if_present()? {
            let bytes = mapped.get(file_range.clone()).ok_or_else(|| {
                TensorLoadError::integrity(format!("tensor `{name}` mapped range is invalid"))
            })?;
            return Ok(bytes.to_vec());
        }
        let mut bytes = vec![0_u8; byte_len];
        let mut file = self.file.try_clone().map_err(|err| {
            TensorLoadError::integrity(format!("could not clone safetensors file handle: {err}"))
        })?;
        file.seek(SeekFrom::Start(file_range.start as u64))
            .map_err(|err| {
                TensorLoadError::integrity(format!("could not seek tensor `{name}`: {err}"))
            })?;
        file.read_exact(&mut bytes).map_err(|err| {
            TensorLoadError::integrity(format!("could not read tensor `{name}` bytes: {err}"))
        })?;
        Ok(bytes)
    }

    fn tensor_file_byte_range(
        &self,
        name: &str,
        tensor_byte_offset: u64,
        byte_len: usize,
    ) -> Result<Range<usize>, TensorLoadError> {
        let metadata = self.header.tensor_metadata(name)?;
        let tensor_byte_len = u64_from_usize(
            metadata.byte_len,
            "tensor byte length does not fit in u64 for range read",
        )?;
        let requested_end = tensor_byte_offset
            .checked_add(u64_from_usize(
                byte_len,
                "requested byte length does not fit in u64",
            )?)
            .ok_or_else(|| TensorLoadError::integrity(format!("tensor `{name}` range overflow")))?;
        if requested_end > tensor_byte_len {
            return Err(TensorLoadError::integrity(format!(
                "tensor `{name}` requested byte range exceeds tensor length"
            )));
        }
        let tensor_range = self.header.tensor_data_range(name)?;
        let file_offset = tensor_range
            .start
            .checked_add(tensor_byte_offset)
            .ok_or_else(|| {
                TensorLoadError::integrity(format!("tensor `{name}` offset overflow"))
            })?;
        let file_end = file_offset
            .checked_add(u64_from_usize(
                byte_len,
                "requested byte length does not fit in u64",
            )?)
            .ok_or_else(|| TensorLoadError::integrity(format!("tensor `{name}` range overflow")))?;
        Ok(
            usize_from_u64(file_offset, "tensor file offset does not fit in usize")?
                ..usize_from_u64(file_end, "tensor file end does not fit in usize")?,
        )
    }

    fn materialized_file(&self) -> Result<Arc<Mmap>, TensorLoadError> {
        if let Some(mapped) = self.materialized_file_if_present()? {
            return Ok(mapped);
        }
        let expected_len = usize_from_u64(
            self.header.file_len(),
            "safetensors file length does not fit in usize for mmap",
        )?;
        // SAFETY: promoted safetensors snapshots are treated as immutable by the
        // store. This read-only mapping is used only after header/range validation,
        // and callers copy bytes out before decoding.
        let mapped = Arc::new(
            unsafe { MmapOptions::new().map(&self.file) }.map_err(|err| {
                TensorLoadError::integrity(format!("could not mmap safetensors file: {err}"))
            })?,
        );
        if mapped.len() != expected_len {
            return Err(TensorLoadError::integrity(format!(
                "mmap length {} does not match safetensors header length {expected_len}",
                mapped.len()
            )));
        }
        let mut cached = self.mapped.lock().map_err(|err| {
            TensorLoadError::integrity(format!("mmap cache lock poisoned: {err}"))
        })?;
        if cached.is_none() {
            *cached = Some(Arc::clone(&mapped));
        }
        cached
            .as_ref()
            .map(Arc::clone)
            .ok_or_else(|| TensorLoadError::integrity("mmap cache was not populated"))
    }

    fn materialized_file_if_present(&self) -> Result<Option<Arc<Mmap>>, TensorLoadError> {
        self.mapped
            .lock()
            .map(|mapped| mapped.as_ref().map(Arc::clone))
            .map_err(|err| TensorLoadError::integrity(format!("mmap cache lock poisoned: {err}")))
    }

    pub fn bf16_tensor_f32_range(
        &self,
        name: &str,
        element_offset: usize,
        element_count: usize,
    ) -> Result<Vec<f32>, TensorLoadError> {
        let metadata = self.header.tensor_metadata(name)?;
        if metadata.dtype != "BF16" {
            return Err(TensorLoadError::unsupported(format!(
                "tensor `{name}` has dtype {}, expected BF16",
                metadata.dtype
            )));
        }
        let byte_offset = u64_from_usize(
            element_offset
                .checked_mul(2)
                .ok_or_else(|| TensorLoadError::integrity("BF16 element offset overflow"))?,
            "BF16 byte offset does not fit in u64",
        )?;
        let byte_len = element_count
            .checked_mul(2)
            .ok_or_else(|| TensorLoadError::integrity("BF16 element count overflow"))?;
        let bytes = self.tensor_bytes_range(name, byte_offset, byte_len)?;
        bf16_bytes_to_f32(&bytes)
    }

    pub fn bf16_row_f32(&self, name: &str, row: usize) -> Result<Vec<f32>, TensorLoadError> {
        let metadata = self.header.tensor_metadata(name)?;
        if metadata.shape.len() != 2 {
            return Err(TensorLoadError::unsupported(format!(
                "tensor `{name}` row reader expects rank 2, got rank {}",
                metadata.shape.len()
            )));
        }
        let rows = metadata.shape[0];
        let columns = metadata.shape[1];
        if row >= rows {
            return Err(TensorLoadError::integrity(format!(
                "tensor `{name}` row {row} exceeds row count {rows}"
            )));
        }
        let element_offset = row
            .checked_mul(columns)
            .ok_or_else(|| TensorLoadError::integrity("row offset overflow"))?;
        self.bf16_tensor_f32_range(name, element_offset, columns)
    }
}

#[derive(Debug, Clone)]
pub struct SafeTensorShardStore {
    root: PathBuf,
    index: SafetensorsIndex,
    shards: Arc<Mutex<BTreeMap<PathBuf, Arc<SafeTensorFile>>>>,
}

impl SafeTensorShardStore {
    pub fn open(root: impl AsRef<Path>) -> Result<Self, TensorLoadError> {
        let root = root.as_ref().to_path_buf();
        let index_path = root.join("model.safetensors.index.json");
        let index_json = fs::read_to_string(&index_path).map_err(|err| {
            TensorLoadError::missing(format!(
                "could not read safetensors index `{}`: {err}",
                index_path.display()
            ))
        })?;
        let index = SafetensorsIndex::from_json(&index_json).map_err(|err| {
            TensorLoadError::integrity(format!(
                "invalid safetensors index `{}`: {err}",
                index_path.display()
            ))
        })?;
        Ok(Self {
            root,
            index,
            shards: Arc::new(Mutex::new(BTreeMap::new())),
        })
    }

    pub fn tensor_shard_path(&self, tensor: &str) -> Result<PathBuf, TensorLoadError> {
        let shard = self.index.shard_for(tensor).ok_or_else(|| {
            TensorLoadError::missing(format!("tensor `{tensor}` not found in safetensors index"))
        })?;
        self.resolve_shard_path(shard)
    }

    fn resolve_shard_path(&self, shard: &str) -> Result<PathBuf, TensorLoadError> {
        let root = fs::canonicalize(&self.root).map_err(|err| {
            TensorLoadError::missing(format!(
                "could not resolve safetensors snapshot root `{}`: {err}",
                self.root.display()
            ))
        })?;
        let path = root.join(shard);
        let path = fs::canonicalize(&path).map_err(|err| {
            TensorLoadError::missing(format!(
                "could not resolve safetensors shard `{}`: {err}",
                path.display()
            ))
        })?;
        if !path.starts_with(&root) {
            return Err(TensorLoadError::integrity(format!(
                "safetensors shard `{}` escapes snapshot root `{}`",
                path.display(),
                root.display()
            )));
        }
        Ok(path)
    }

    pub fn tensor_metadata(&self, tensor: &str) -> Result<TensorMetadata, TensorLoadError> {
        self.open_tensor_file(tensor)?.tensor_metadata(tensor)
    }

    pub fn bf16_row_f32(&self, tensor: &str, row: usize) -> Result<Vec<f32>, TensorLoadError> {
        self.open_tensor_file(tensor)?.bf16_row_f32(tensor, row)
    }

    pub fn bf16_tensor_f32_range(
        &self,
        tensor: &str,
        element_offset: usize,
        element_count: usize,
    ) -> Result<Vec<f32>, TensorLoadError> {
        self.open_tensor_file(tensor)?
            .bf16_tensor_f32_range(tensor, element_offset, element_count)
    }

    pub fn bf16_tensor_f32(&self, tensor: &str) -> Result<Vec<f32>, TensorLoadError> {
        let metadata = self.tensor_metadata(tensor)?;
        let element_count = metadata.shape.iter().try_fold(1_usize, |acc, dim| {
            acc.checked_mul(*dim)
                .ok_or_else(|| TensorLoadError::integrity("tensor shape overflows usize"))
        })?;
        self.bf16_tensor_f32_range(tensor, 0, element_count)
    }

    pub fn bf16_matvec_row_major_f32(
        &self,
        tensor: &str,
        input: &[f32],
    ) -> Result<Vec<f32>, TensorLoadError> {
        self.bf16_matvec_rows_f32(tensor, input, BF16_MATVEC_CHUNK_ROWS)
    }

    pub fn bf16_matvec_rows_f32(
        &self,
        tensor: &str,
        input: &[f32],
        chunk_rows: usize,
    ) -> Result<Vec<f32>, TensorLoadError> {
        let file = self.open_tensor_file(tensor)?;
        let metadata = file.tensor_metadata(tensor)?;
        if metadata.shape.len() != 2 {
            return Err(TensorLoadError::unsupported(format!(
                "tensor `{tensor}` matvec expects rank 2, got rank {}",
                metadata.shape.len()
            )));
        }
        let rows = metadata.shape[0];
        let columns = metadata.shape[1];
        if input.len() != columns {
            return Err(TensorLoadError::integrity(format!(
                "input length {} does not match tensor `{tensor}` columns {columns}",
                input.len()
            )));
        }
        if chunk_rows == 0 {
            return Err(TensorLoadError::integrity(
                "chunk_rows must be greater than zero",
            ));
        }
        let mut output = Vec::with_capacity(rows);
        for row_start in (0..rows).step_by(chunk_rows) {
            let rows_in_chunk = chunk_rows.min(rows - row_start);
            let element_offset = row_start
                .checked_mul(columns)
                .ok_or_else(|| TensorLoadError::integrity("matvec offset overflow"))?;
            let element_count = rows_in_chunk
                .checked_mul(columns)
                .ok_or_else(|| TensorLoadError::integrity("matvec chunk overflow"))?;
            let weights = file.bf16_tensor_f32_range(tensor, element_offset, element_count)?;
            output.extend(weights.chunks_exact(columns).map(|row| {
                row.iter()
                    .zip(input)
                    .map(|(weight, value)| weight * value)
                    .sum::<f32>()
            }));
        }
        Ok(output)
    }

    pub fn bf16_matvecs_row_major_f32(
        &self,
        tensor: &str,
        inputs: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, TensorLoadError> {
        let file = self.open_tensor_file(tensor)?;
        let metadata = file.tensor_metadata(tensor)?;
        if metadata.shape.len() != 2 {
            return Err(TensorLoadError::unsupported(format!(
                "tensor `{tensor}` batched matvec expects rank 2, got rank {}",
                metadata.shape.len()
            )));
        }
        let rows = metadata.shape[0];
        let columns = metadata.shape[1];
        for input in inputs {
            if input.len() != columns {
                return Err(TensorLoadError::integrity(format!(
                    "input length {} does not match tensor `{tensor}` columns {columns}",
                    input.len()
                )));
            }
        }
        let mut outputs = vec![Vec::with_capacity(rows); inputs.len()];
        for row_start in (0..rows).step_by(BF16_MATVEC_CHUNK_ROWS) {
            let rows_in_chunk = BF16_MATVEC_CHUNK_ROWS.min(rows - row_start);
            let element_offset = row_start
                .checked_mul(columns)
                .ok_or_else(|| TensorLoadError::integrity("batched matvec offset overflow"))?;
            let element_count = rows_in_chunk
                .checked_mul(columns)
                .ok_or_else(|| TensorLoadError::integrity("batched matvec chunk overflow"))?;
            let weights = file.bf16_tensor_f32_range(tensor, element_offset, element_count)?;
            for input_idx in 0..inputs.len() {
                outputs[input_idx].extend(weights.chunks_exact(columns).map(|row| {
                    row.iter()
                        .zip(&inputs[input_idx])
                        .map(|(weight, value)| weight * value)
                        .sum::<f32>()
                }));
            }
        }
        Ok(outputs)
    }

    pub fn bf16_matvec_top_k_rows_f32(
        &self,
        tensor: &str,
        input: &[f32],
        top_k: usize,
        chunk_rows: usize,
    ) -> Result<Vec<TopKLogit>, TensorLoadError> {
        let file = self.open_tensor_file(tensor)?;
        let metadata = file.tensor_metadata(tensor)?;
        if metadata.shape.len() != 2 {
            return Err(TensorLoadError::unsupported(format!(
                "tensor `{tensor}` top-k matvec expects rank 2, got rank {}",
                metadata.shape.len()
            )));
        }
        let rows = metadata.shape[0];
        let columns = metadata.shape[1];
        if input.len() != columns {
            return Err(TensorLoadError::integrity(format!(
                "input length {} does not match tensor `{tensor}` columns {columns}",
                input.len()
            )));
        }
        if top_k == 0 || top_k > rows {
            return Err(TensorLoadError::integrity(format!(
                "top_k {top_k} must be in 1..={rows}"
            )));
        }
        if chunk_rows == 0 {
            return Err(TensorLoadError::integrity(
                "chunk_rows must be greater than zero",
            ));
        }
        let mut top = Vec::with_capacity(top_k);
        for row_start in (0..rows).step_by(chunk_rows) {
            let rows_in_chunk = chunk_rows.min(rows - row_start);
            let element_offset = row_start
                .checked_mul(columns)
                .ok_or_else(|| TensorLoadError::integrity("top-k matvec offset overflow"))?;
            let element_count = rows_in_chunk
                .checked_mul(columns)
                .ok_or_else(|| TensorLoadError::integrity("top-k matvec chunk overflow"))?;
            let weights = file.bf16_tensor_f32_range(tensor, element_offset, element_count)?;
            for (row_offset, row) in weights.chunks_exact(columns).enumerate() {
                let logit = row
                    .iter()
                    .zip(input)
                    .map(|(weight, value)| weight * value)
                    .sum::<f32>();
                push_top_logit(
                    &mut top,
                    TopKLogit {
                        index: row_start + row_offset,
                        logit,
                    },
                    top_k,
                );
            }
        }
        Ok(top)
    }

    pub fn cached_shard_count(&self) -> usize {
        self.shards.lock().map(|shards| shards.len()).unwrap_or(0)
    }

    pub fn materialized_shard_count(&self) -> usize {
        self.shards
            .lock()
            .map(|shards| {
                shards
                    .values()
                    .filter(|shard| shard.is_materialized())
                    .count()
            })
            .unwrap_or(0)
    }

    pub fn materialize_shard_for_tensor(&self, tensor: &str) -> Result<usize, TensorLoadError> {
        self.open_tensor_file(tensor)?.materialize()
    }

    pub fn materialize_all_shards(&self) -> Result<usize, TensorLoadError> {
        let mut total_bytes = 0_usize;
        for shard in self.index.shard_paths() {
            let shard_path = self.resolve_shard_path(shard)?;
            let file = self.open_shard_file(shard_path)?;
            total_bytes = total_bytes
                .checked_add(file.materialize()?)
                .ok_or_else(|| TensorLoadError::integrity("materialized shard bytes overflow"))?;
        }
        Ok(total_bytes)
    }

    fn open_tensor_file(&self, tensor: &str) -> Result<Arc<SafeTensorFile>, TensorLoadError> {
        let shard_path = self.tensor_shard_path(tensor)?;
        self.open_shard_file(shard_path)
    }

    fn open_shard_file(&self, shard_path: PathBuf) -> Result<Arc<SafeTensorFile>, TensorLoadError> {
        {
            let shards = self.shards.lock().map_err(|err| {
                TensorLoadError::integrity(format!("shard cache lock poisoned: {err}"))
            })?;
            if let Some(file) = shards.get(&shard_path) {
                return Ok(Arc::clone(file));
            }
        }
        let file = Arc::new(SafeTensorFile::open(&shard_path)?);
        let mut shards = self.shards.lock().map_err(|err| {
            TensorLoadError::integrity(format!("shard cache lock poisoned: {err}"))
        })?;
        Ok(Arc::clone(
            shards
                .entry(shard_path)
                .or_insert_with(|| Arc::clone(&file)),
        ))
    }
}

fn push_top_logit(top: &mut Vec<TopKLogit>, candidate: TopKLogit, top_k: usize) {
    top.push(candidate);
    top.sort_by(|left, right| {
        right
            .logit
            .total_cmp(&left.logit)
            .then_with(|| left.index.cmp(&right.index))
    });
    top.truncate(top_k);
}

pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn bf16_bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>, TensorLoadError> {
    if !bytes.len().is_multiple_of(2) {
        return Err(TensorLoadError::integrity(
            "BF16 byte length must be divisible by 2",
        ));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| bf16_bits_to_f32(u16::from_le_bytes(chunk.try_into().expect("BF16 chunk"))))
        .collect())
}

#[derive(Debug, Clone)]
struct TensorHeaderEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

impl TensorHeaderEntry {
    fn from_json(
        name: &str,
        value: &serde_json::Value,
        payload_len: u64,
    ) -> Result<Self, TensorLoadError> {
        let object = value.as_object().ok_or_else(|| {
            TensorLoadError::integrity(format!("tensor `{name}` header must be an object"))
        })?;
        let dtype = object
            .get("dtype")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| TensorLoadError::integrity(format!("tensor `{name}` is missing dtype")))?
            .to_owned();
        let shape = parse_shape(name, object.get("shape"))?;
        let data_offsets = parse_data_offsets(name, object.get("data_offsets"))?;
        if data_offsets[1] < data_offsets[0] {
            return Err(TensorLoadError::integrity(format!(
                "tensor `{name}` has inverted data offsets"
            )));
        }
        if data_offsets[1] > payload_len {
            return Err(TensorLoadError::integrity(format!(
                "tensor `{name}` data offsets exceed payload length"
            )));
        }
        Ok(Self {
            dtype,
            shape,
            data_offsets,
        })
    }

    fn byte_len(&self) -> Result<usize, TensorLoadError> {
        usize_from_u64(
            self.data_offsets[1] - self.data_offsets[0],
            "tensor byte length does not fit in usize",
        )
    }
}

fn read_header_prefix(bytes: &[u8], file_len: u64) -> Result<(u64, usize), TensorLoadError> {
    let prefix = bytes
        .get(0..8)
        .ok_or_else(|| TensorLoadError::integrity("safetensors file is missing header prefix"))?;
    let header_len = validate_header_len(
        u64::from_le_bytes(prefix.try_into().expect("prefix has length 8")),
        file_len,
    )?;
    let header_end = 8_u64
        .checked_add(header_len)
        .ok_or_else(|| TensorLoadError::integrity("safetensors header length overflow"))?;
    Ok((
        header_len,
        usize_from_u64(header_end, "safetensors header end does not fit in usize")?,
    ))
}

fn validate_header_len(header_len: u64, file_len: u64) -> Result<u64, TensorLoadError> {
    if header_len > MAX_SAFETENSORS_HEADER_LEN {
        return Err(TensorLoadError::integrity(format!(
            "safetensors header length {header_len} exceeds limit {MAX_SAFETENSORS_HEADER_LEN}"
        )));
    }
    let header_end = 8_u64
        .checked_add(header_len)
        .ok_or_else(|| TensorLoadError::integrity("safetensors header length overflow"))?;
    if header_end > file_len {
        return Err(TensorLoadError::integrity(
            "safetensors header length exceeds file length",
        ));
    }
    Ok(header_len)
}

fn parse_shape(
    name: &str,
    value: Option<&serde_json::Value>,
) -> Result<Vec<usize>, TensorLoadError> {
    let array = value
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| TensorLoadError::integrity(format!("tensor `{name}` is missing shape")))?;
    array
        .iter()
        .map(|value| {
            let dim = value.as_u64().ok_or_else(|| {
                TensorLoadError::integrity(format!("tensor `{name}` shape must contain integers"))
            })?;
            usize_from_u64(dim, "tensor shape dimension does not fit in usize")
        })
        .collect()
}

fn parse_data_offsets(
    name: &str,
    value: Option<&serde_json::Value>,
) -> Result<[u64; 2], TensorLoadError> {
    let array = value.and_then(serde_json::Value::as_array).ok_or_else(|| {
        TensorLoadError::integrity(format!("tensor `{name}` is missing data_offsets"))
    })?;
    if array.len() != 2 {
        return Err(TensorLoadError::integrity(format!(
            "tensor `{name}` data_offsets must contain two integers"
        )));
    }
    let start = array[0].as_u64().ok_or_else(|| {
        TensorLoadError::integrity(format!("tensor `{name}` data_offsets must be integers"))
    })?;
    let end = array[1].as_u64().ok_or_else(|| {
        TensorLoadError::integrity(format!("tensor `{name}` data_offsets must be integers"))
    })?;
    Ok([start, end])
}

fn usize_from_u64(value: u64, message: &str) -> Result<usize, TensorLoadError> {
    value
        .try_into()
        .map_err(|_| TensorLoadError::integrity(message))
}

fn u64_from_usize(value: usize, message: &str) -> Result<u64, TensorLoadError> {
    value
        .try_into()
        .map_err(|_| TensorLoadError::integrity(message))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorMetadata {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub byte_len: usize,
}

#[derive(Debug, Error)]
#[error("{code}: {message}")]
pub struct TensorLoadError {
    code: &'static str,
    message: String,
}

impl TensorLoadError {
    pub fn code(&self) -> &'static str {
        self.code
    }

    fn integrity(message: impl Into<String>) -> Self {
        Self {
            code: "model_integrity_failed",
            message: message.into(),
        }
    }

    fn missing(message: impl Into<String>) -> Self {
        Self {
            code: "model_artifact_missing",
            message: message.into(),
        }
    }

    fn unsupported(message: impl Into<String>) -> Self {
        Self {
            code: "unsupported_capability",
            message: message.into(),
        }
    }
}
