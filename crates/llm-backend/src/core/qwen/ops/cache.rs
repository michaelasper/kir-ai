use super::super::super::{
    KvCacheError, LayerKvCache, LayerKvCachePrefixState, LayerKvCacheSnapshot,
    LinearAttentionCache, LinearAttentionCacheSnapshot, TensorLoadError,
};
use super::attention_full::QwenFullAttentionDims;
use super::attention_linear::QwenLinearAttentionDims;
use llm_models::{AttentionKind, QwenModelSpec};

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum QwenLayerCache {
    Linear(LinearAttentionCache),
    Full(LayerKvCache),
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum QwenLayerCacheSnapshot {
    Linear(LinearAttentionCacheSnapshot),
    Full(LayerKvCacheSnapshot),
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum QwenLayerCachePrefixState {
    Linear(LinearAttentionCacheSnapshot),
    Full(LayerKvCachePrefixState),
}

impl QwenLayerCache {
    pub fn snapshot(&self) -> QwenLayerCacheSnapshot {
        match self {
            Self::Linear(cache) => QwenLayerCacheSnapshot::Linear(cache.snapshot()),
            Self::Full(cache) => QwenLayerCacheSnapshot::Full(cache.snapshot()),
        }
    }

    pub fn from_snapshot(snapshot: QwenLayerCacheSnapshot) -> Result<Self, KvCacheError> {
        match snapshot {
            QwenLayerCacheSnapshot::Linear(snapshot) => {
                LinearAttentionCache::from_snapshot(snapshot).map(Self::Linear)
            }
            QwenLayerCacheSnapshot::Full(snapshot) => {
                LayerKvCache::from_snapshot(snapshot).map(Self::Full)
            }
        }
    }

    pub fn prefix_cache_state(&self) -> QwenLayerCachePrefixState {
        match self {
            Self::Linear(cache) => QwenLayerCachePrefixState::Linear(cache.snapshot()),
            Self::Full(cache) => QwenLayerCachePrefixState::Full(cache.prefix_cache_state()),
        }
    }

    pub fn from_prefix_cache_state(
        state: &QwenLayerCachePrefixState,
    ) -> Result<Self, KvCacheError> {
        match state {
            QwenLayerCachePrefixState::Linear(snapshot) => {
                LinearAttentionCache::from_snapshot(snapshot.clone()).map(Self::Linear)
            }
            QwenLayerCachePrefixState::Full(state) => {
                LayerKvCache::from_prefix_cache_state(state).map(Self::Full)
            }
        }
    }
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
        _ => Err(TensorLoadError::unsupported(format!(
            "Qwen layer {layer_idx} uses unsupported attention kind `{kind:?}`"
        ))),
    }
}
