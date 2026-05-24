use super::*;
use crate::native_text::{
    NativeTextPrefixCache, NativeTextPrefixCacheMetrics, NativeTextPrefixCacheNamespace,
    NativeTextPrefixCacheValue,
};
use llm_backend::native::{
    GemmaLayerCache, LayerKvCache, LinearAttentionCache, QwenLayerCache,
};
use llm_backend_contracts::BackendModelMetadata;
use std::{
    sync::atomic::{AtomicUsize, Ordering},
    time::{Duration, Instant},
};

fn namespace(label: &str, family: &str) -> NativeTextPrefixCacheNamespace {
    NativeTextPrefixCacheNamespace {
        model_id: format!("model-{label}"),
        backend: "native-test".to_owned(),
        family: Some(family.to_owned()),
        quantization: Some("bf16".to_owned()),
        repo_id: Some("org/model".to_owned()),
        resolved_commit: Some("abc123".to_owned()),
        profile: Some(label.to_owned()),
        tokenizer_kind: "huggingface-tokenizer-json".to_owned(),
        tokenizer_hash: format!("sha256:tokenizer-{label}"),
        tokenizer_normalization: "llm-tokenizer/hf-json/v1".to_owned(),
        cache_template_id: format!("template-{label}/v1"),
        chat_template_kwargs_hash: None,
        adapter_settings: format!("native-test-adapter-{label}/v1"),
        cache_key: format!("cache-key-{label}"),
        tool_schema: None,
        request_mode: "chat,json_object=false,required_tool=None".to_owned(),
        cache_layout_version: 1,
        cache_tokens: 16,
        max_prefill_tokens: 4,
    }
}
fn filled_layer_cache(max_tokens: usize) -> LayerKvCache {
    let mut cache = LayerKvCache::new(max_tokens, 2, 3).expect("layer cache shape is valid");
    cache
        .append(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        )
        .expect("first token appends");
    cache
        .append(
            &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            &[70.0, 80.0, 90.0, 100.0, 110.0, 120.0],
        )
        .expect("second token appends");
    cache
}

fn filled_linear_cache() -> LinearAttentionCache {
    let mut cache = LinearAttentionCache::new(2, 3, 2, 2, 3)
        .expect("linear attention cache shape is valid");
    cache
        .push_conv_input(&[1.0, 2.0, 3.0])
        .expect("conv input appends");
    cache
        .push_conv_input(&[4.0, 5.0, 6.0])
        .expect("second conv input appends");
    cache
        .replace_recurrent_state(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
        .expect("recurrent state matches shape");
    cache
}

fn round_trip<C>(family: &str, states: Vec<C::PrefixCacheState>)
where
    C: NativeTextDiskCacheValue + NativeTextPrefixCacheValue,
{
    let namespace = namespace("round-trip", family);
    let identity = NativeTextDiskCacheIdentity::from_namespace(&namespace, family);
    let descriptor =
        NativeTextDiskCacheBlockDescriptor::new(&identity, &namespace, 0, &[11, 12])
            .expect("descriptor builds");
    let hidden = vec![0.25, 0.5, 0.75];

    let encoded = NativeTextDiskCacheBlock::<C>::encode(&descriptor, &hidden, &states)
        .expect("prefix block encodes");
    let decoded = NativeTextDiskCacheBlock::<C>::decode(&encoded, &identity, &descriptor)
        .expect("prefix block decodes");

    assert_eq!(decoded.token_count, 2);
    assert_eq!(decoded.hidden, hidden);
    assert_eq!(decoded.states.len(), states.len());
    assert_eq!(
        C::prefix_cache_entry_bytes(&hidden, &decoded.states),
        C::prefix_cache_entry_bytes(&hidden, &states)
    );
    assert!(
        C::prefix_cache_from_state(&decoded.states).is_some(),
        "decoded disk states restore into hot cache values"
    );

    let mut wrong_cache_layout = encoded;
    NativeTextDiskCacheBlock::<C>::rewrite_cache_layout_version_for_test(
        &mut wrong_cache_layout,
        descriptor.cache_layout_version + 1,
    )
    .expect("cache layout metadata is rewritten");
    assert!(
        NativeTextDiskCacheBlock::<C>::decode(&wrong_cache_layout, &identity, &descriptor)
            .is_err(),
        "blocks for another cache layout must not decode"
    );
}


#[derive(Debug, Clone, PartialEq, Eq)]
struct DummyCache {
    marker: u32,
}

impl NativeTextPrefixCacheValue for DummyCache {
    type PrefixCacheState = Self;

    fn prefix_cache_state(caches: &[Self]) -> Vec<Self::PrefixCacheState> {
        caches.to_vec()
    }

    fn prefix_cache_from_state(states: &[Self::PrefixCacheState]) -> Option<Vec<Self>> {
        Some(states.to_vec())
    }

    fn prefix_cache_entry_bytes(hidden: &[f32], states: &[Self::PrefixCacheState]) -> u64 {
        std::mem::size_of_val(hidden) as u64
            + states.len() as u64 * std::mem::size_of::<Self>() as u64
    }
}

impl NativeTextDiskCacheValue for DummyCache {
    fn encode_disk_block_states(
        states: &[Self::PrefixCacheState],
        block_start: usize,
        block_token_count: usize,
        sink: &mut NativeTextDiskCacheTensorSink,
    ) -> Result<Vec<NativeTextDiskCacheLayerLayout>, NativeTextDiskCacheError> {
        let values = states[block_start..block_start + block_token_count]
            .iter()
            .map(|state| state.marker as f32)
            .collect::<Vec<_>>();
        sink.push_f32("dummy.markers", vec![values.len()], values)?;
        Ok(vec![NativeTextDiskCacheLayerLayout::test_marker_tensor(
            "dummy.markers",
        )])
    }

    fn decode_disk_states(
        layouts: &[NativeTextDiskCacheLayerLayout],
        archive: &NativeTextDiskCacheTensorArchive<'_>,
    ) -> Result<Vec<Self::PrefixCacheState>, NativeTextDiskCacheError> {
        let Some(layout) = layouts.first() else {
            return Err(NativeTextDiskCacheError::integrity("missing dummy layout"));
        };
        let tensor = layout
            .test_marker_tensor_name()
            .ok_or_else(|| NativeTextDiskCacheError::integrity("wrong dummy layout"))?;
        archive
            .f32_tensor(tensor)?
            .into_iter()
            .map(|marker| {
                if marker.fract() != 0.0 || marker < 0.0 {
                    return Err(NativeTextDiskCacheError::integrity(
                        "dummy marker must be a non-negative integer",
                    ));
                }
                Ok(DummyCache {
                    marker: marker as u32,
                })
            })
            .collect()
    }

    fn assemble_disk_block_states(
        blocks: &[NativeTextDiskCacheStateBlock<Self::PrefixCacheState>],
    ) -> Result<Vec<Self::PrefixCacheState>, NativeTextDiskCacheError> {
        Ok(blocks
            .iter()
            .flat_map(|block| block.states.iter().cloned())
            .collect())
    }
}
