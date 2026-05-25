use super::super::super::{
    KvCacheError, LayerKvCache, LayerKvCachePrefixState, LayerKvCacheSnapshot, TensorLoadError,
};
use super::attention::GemmaAttentionDims;
use llm_models::{GemmaAttentionKind, GemmaModelSpec};

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum GemmaLayerCache {
    Attention(LayerKvCache),
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum GemmaLayerCacheSnapshot {
    Attention(LayerKvCacheSnapshot),
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum GemmaLayerCachePrefixState {
    Attention(LayerKvCachePrefixState),
}

impl GemmaLayerCache {
    pub fn snapshot(&self) -> GemmaLayerCacheSnapshot {
        match self {
            Self::Attention(cache) => GemmaLayerCacheSnapshot::Attention(cache.snapshot()),
        }
    }

    pub fn from_snapshot(snapshot: GemmaLayerCacheSnapshot) -> Result<Self, KvCacheError> {
        match snapshot {
            GemmaLayerCacheSnapshot::Attention(snapshot) => {
                LayerKvCache::from_snapshot(snapshot).map(Self::Attention)
            }
        }
    }

    pub fn prefix_cache_state(&self) -> GemmaLayerCachePrefixState {
        match self {
            Self::Attention(cache) => {
                GemmaLayerCachePrefixState::Attention(cache.prefix_cache_state())
            }
        }
    }

    pub fn from_prefix_cache_state(
        state: &GemmaLayerCachePrefixState,
    ) -> Result<Self, KvCacheError> {
        match state {
            GemmaLayerCachePrefixState::Attention(state) => {
                LayerKvCache::from_prefix_cache_state(state).map(Self::Attention)
            }
        }
    }
}

pub fn gemma_layer_caches_for_spec(
    spec: &GemmaModelSpec,
    max_tokens: usize,
) -> Result<Vec<GemmaLayerCache>, TensorLoadError> {
    let concrete_count = gemma_concrete_cache_count(spec)?;
    (0..concrete_count)
        .map(|layer_idx| gemma_layer_cache_for_spec(spec, layer_idx, max_tokens))
        .collect()
}

pub fn gemma_cache_count_for_spec(spec: &GemmaModelSpec) -> Result<usize, TensorLoadError> {
    gemma_concrete_cache_count(spec)
}

fn gemma_layer_cache_for_spec(
    spec: &GemmaModelSpec,
    layer_idx: usize,
    max_tokens: usize,
) -> Result<GemmaLayerCache, TensorLoadError> {
    let dims = GemmaAttentionDims::from_spec(spec, layer_idx)?;
    let kind = spec.layer_kinds[layer_idx];
    let max_tokens = match kind {
        GemmaAttentionKind::SlidingAttention => {
            let sliding_window = spec.sliding_window as usize;
            max_tokens.min(sliding_window).max(1)
        }
        GemmaAttentionKind::FullAttention => max_tokens.max(1),
        _ => {
            return Err(TensorLoadError::unsupported(format!(
                "Gemma layer {layer_idx} uses unsupported attention kind `{kind:?}`"
            )));
        }
    };
    LayerKvCache::new(max_tokens, dims.num_key_value_heads, dims.head_dim)
        .map(GemmaLayerCache::Attention)
        .map_err(|err| {
            TensorLoadError::integrity(format!(
                "Gemma layer{layer_idx} attention cache allocation failed: {err}"
            ))
        })
}

pub(crate) fn gemma_concrete_cache_count(spec: &GemmaModelSpec) -> Result<usize, TensorLoadError> {
    let layer_count = spec.num_hidden_layers as usize;
    if spec.layer_kinds.len() != layer_count {
        return Err(TensorLoadError::integrity(format!(
            "Gemma spec declares {layer_count} layers but has {} attention kind entries",
            spec.layer_kinds.len()
        )));
    }
    if spec.num_kv_shared_layers == 0 {
        return Ok(layer_count);
    }
    let first_shared = first_gemma_kv_shared_layer_idx(spec);
    if first_shared == 0 {
        return Err(TensorLoadError::unsupported(
            "Gemma KV sharing requires at least one concrete layer before shared layers",
        ));
    }
    for layer_idx in first_shared..layer_count {
        gemma_cache_index_for_layer(spec, layer_idx)?;
    }
    Ok(first_shared)
}

pub(crate) fn gemma_cache_index_for_layer(
    spec: &GemmaModelSpec,
    layer_idx: usize,
) -> Result<usize, TensorLoadError> {
    let first_shared = first_gemma_kv_shared_layer_idx(spec);
    if layer_idx < first_shared {
        return Ok(layer_idx);
    }
    let kind = *spec.layer_kinds.get(layer_idx).ok_or_else(|| {
        TensorLoadError::missing(format!(
            "Gemma layer {layer_idx} is outside configured layer count"
        ))
    })?;
    spec.layer_kinds[..first_shared]
        .iter()
        .enumerate()
        .rev()
        .find_map(|(idx, candidate)| (*candidate == kind).then_some(idx))
        .ok_or_else(|| {
            TensorLoadError::unsupported(format!(
                "Gemma shared layer {layer_idx} has no earlier concrete {kind:?} cache"
            ))
        })
}

fn first_gemma_kv_shared_layer_idx(spec: &GemmaModelSpec) -> usize {
    (spec.num_hidden_layers - spec.num_kv_shared_layers) as usize
}
