use super::*;
use crate::native_text::{
    NativeTextPrefixCache, NativeTextPrefixCacheMetrics, NativeTextPrefixCacheNamespace,
    NativeTextPrefixCacheValue,
};
use llm_backend::native::{
    GemmaLayerCache, LayerKvCache, LayerKvCacheSnapshot, LinearAttentionCache, QwenLayerCache,
    QwenLayerCachePrefixState, QwenLayerCacheSnapshot,
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

fn qwen_full_cache(max_tokens: usize, token_count: usize, seed: f32) -> QwenLayerCache {
    let mut cache = LayerKvCache::new(max_tokens, 2, 3).expect("layer cache shape is valid");
    for token_index in 0..token_count {
        let keys = qwen_token_values(seed, token_index, 0.0);
        let values = qwen_token_values(seed, token_index, 1000.0);
        cache.append(&keys, &values).expect("token appends");
    }
    QwenLayerCache::Full(cache)
}

fn qwen_full_states(
    max_tokens: usize,
    token_count: usize,
    seed: f32,
) -> Vec<QwenLayerCachePrefixState> {
    vec![qwen_full_cache(max_tokens, token_count, seed).prefix_cache_state()]
}

fn qwen_full_block_states(
    max_tokens: usize,
    block_start: usize,
    block_token_count: usize,
    seed: f32,
    tokens_seen: usize,
) -> Vec<QwenLayerCachePrefixState> {
    let keys = (block_start..block_start + block_token_count)
        .flat_map(|token_index| qwen_token_values(seed, token_index, 0.0))
        .collect::<Vec<_>>();
    let values = (block_start..block_start + block_token_count)
        .flat_map(|token_index| qwen_token_values(seed, token_index, 1000.0))
        .collect::<Vec<_>>();
    let config = LayerKvCache::new(max_tokens, 2, 3)
        .expect("layer cache shape is valid")
        .snapshot()
        .config;
    let snapshot = LayerKvCacheSnapshot {
        revision: tokens_seen as u64,
        config,
        max_tokens,
        key_value_heads: 2,
        head_dim: 3,
        token_count: block_token_count,
        tokens_seen,
        keys,
        values,
    };
    vec![
        QwenLayerCache::Full(
            LayerKvCache::from_snapshot(snapshot).expect("block snapshot restores"),
        )
        .prefix_cache_state(),
    ]
}

fn qwen_token_values(seed: f32, token_index: usize, offset: f32) -> Vec<f32> {
    (0..6)
        .map(|element| seed + offset + token_index as f32 * 10.0 + element as f32)
        .collect()
}

fn assert_qwen_full_states_match(
    actual: &[QwenLayerCachePrefixState],
    expected: &[QwenLayerCachePrefixState],
) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(
            qwen_full_snapshot_from_state(actual),
            qwen_full_snapshot_from_state(expected)
        );
    }
}

fn assert_qwen_full_caches_match(actual: &[QwenLayerCache], expected: &[QwenLayerCache]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.snapshot(), expected.snapshot());
    }
}

fn qwen_full_snapshot_from_state(state: &QwenLayerCachePrefixState) -> QwenLayerCacheSnapshot {
    QwenLayerCache::from_prefix_cache_state(state)
        .expect("qwen prefix state restores")
        .snapshot()
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
