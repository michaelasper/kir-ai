use super::*;

#[test]
fn native_max_tokens_defaults_to_configured_cache_limit() {
    assert_eq!(
        resolve_native_text_max_tokens(None, 4, "Qwen")
            .expect("omitted max tokens uses configured cap"),
        4
    );
}

#[test]
fn native_qwen_default_max_new_tokens_is_interactive_budget() {
    assert_eq!(DEFAULT_NATIVE_QWEN_MAX_NEW_TOKENS, 256);
    assert_eq!(
        resolve_native_text_max_tokens(None, DEFAULT_NATIVE_QWEN_MAX_NEW_TOKENS, "Qwen")
            .expect("omitted max tokens uses native default"),
        256
    );
    assert_eq!(
        resolve_native_text_max_tokens(Some(128), DEFAULT_NATIVE_QWEN_MAX_NEW_TOKENS, "Qwen")
            .expect("requests below native default are accepted"),
        128
    );
}

#[test]
fn native_max_tokens_accepts_multi_token_decode_with_cache() {
    assert_eq!(
        resolve_native_text_max_tokens(Some(2), 4, "Qwen").expect("multi-token decode uses cache"),
        2
    );
}

#[test]
fn native_max_tokens_rejects_requests_above_configured_limit() {
    let err = resolve_native_text_max_tokens(Some(5), 4, "Qwen")
        .expect_err("request above configured limit fails closed");

    assert!(err.is_unsupported_request());
    assert!(err.to_string().contains("configured native Qwen limit"));
}

#[test]
fn native_qwen_cache_capacity_preserves_prompt_and_generation_budget() {
    let capacity = native_text_cache_token_capacity(40, 8, 32, 64, "Qwen")
        .expect("prompt plus generation budget fits context");
    let spec = QwenModelSpec {
        family: llm_models::ModelFamily::Qwen,
        architecture: "Qwen3_5MoeForConditionalGeneration".to_owned(),
        model_type: "qwen3_5_moe".to_owned(),
        text_model_type: "qwen3_5_moe_text".to_owned(),
        hidden_size: 2,
        rms_norm_eps: 0.0,
        tie_word_embeddings: false,
        rope_theta: 1_000_000.0,
        partial_rotary_factor: 1.0,
        num_hidden_layers: 1,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 2,
        linear_num_key_heads: 1,
        linear_num_value_heads: 1,
        linear_key_head_dim: 1,
        linear_value_head_dim: 1,
        linear_conv_kernel_dim: 1,
        num_experts: 1,
        num_experts_per_tok: 1,
        moe_intermediate_size: 1,
        shared_expert_intermediate_size: 1,
        max_position_embeddings: 32,
        sliding_window: None,
        vocab_size: 16,
        layer_kinds: vec![llm_models::AttentionKind::FullAttention],
    };

    let caches = qwen_layer_caches_for_spec(&spec, capacity).expect("cache allocates");
    match &caches[0] {
        QwenLayerCache::Full(cache) => assert_eq!(cache.max_tokens(), 48),
        QwenLayerCache::Linear(_) => panic!("expected full-attention cache"),
        other => panic!("unexpected Qwen cache variant: {other:?}"),
    }
}

#[test]
fn native_qwen_cache_capacity_rejects_context_beyond_position_limit() {
    let err = native_text_cache_token_capacity(60, 8, 32, 64, "Qwen")
        .expect_err("context beyond model position limit fails closed");

    assert!(err.is_unsupported_request());
    assert!(
        err.to_string().contains("model context limit"),
        "error should name context limit: {err}"
    );
}

#[test]
fn native_qwen_adapter_stop_tokens_use_chatml_im_end() {
    let snapshot = temp_snapshot_dir("qwen-stop-tokens");
    std::fs::remove_dir_all(&snapshot).ok();
    std::fs::create_dir_all(&snapshot).expect("snapshot dir");
    copy_fixture("tokenizer.json", snapshot.join("tokenizer.json"));
    write_tiny_linear_decoder_snapshot(&snapshot);
    let backend = native_qwen_test_backend(
        &snapshot,
        crate::DEFAULT_MODEL_ID,
        tiny_engine_qwen_spec(llm_models::AttentionKind::LinearAttention),
        8,
        1,
        2,
        64,
    );
    let im_end = backend
        .driver
        .tokenizer
        .token_to_id("<|im_end|>")
        .expect("qwen tokenizer has im_end token") as usize;
    let non_stop = (0..16)
        .find(|token_id| *token_id != im_end)
        .expect("small non-stop token id exists");

    assert_eq!(
        backend.driver.adapter.stop_tokens(),
        NativeTextStopTokens {
            token_ids: &[],
            token_strings: &["<|im_end|>"],
            encoded_token_strings: &[],
        }
    );
    assert!(matches!(
        backend
            .driver
            .adapter
            .observe_candidate(&backend.driver.stop_tokens, &[], im_end)
            .expect("im_end candidate is observed"),
        NativeTextCandidateDecision::Stop
    ));
    assert!(matches!(
        backend
            .driver
            .adapter
            .observe_candidate(&backend.driver.stop_tokens, &[], non_stop)
            .expect("non-stop candidate is observed"),
        NativeTextCandidateDecision::Emit(token_id) if token_id == non_stop
    ));
    std::fs::remove_dir_all(snapshot).ok();
}
