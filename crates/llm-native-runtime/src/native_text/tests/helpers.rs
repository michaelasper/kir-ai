fn namespace(label: &str) -> NativeTextPrefixCacheNamespace {
    NativeTextPrefixCacheNamespace {
        model_id: format!("model-{label}"),
        backend: "native-test".to_owned(),
        family: Some("test".to_owned()),
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
        request_mode: "conversation=false,json_object=false,required_tool=None".to_owned(),
        cache_layout_version: 1,
        cache_tokens: 16,
        max_prefill_tokens: 4,
    }
}
fn driver_test_tokenizer() -> HuggingFaceTokenizer {
    let tokenizer_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/qwen36/tokenizer.json");
    HuggingFaceTokenizer::from_file(tokenizer_path).expect("tokenizer loads")
}

fn driver_test_tokenizer_identity() -> HuggingFaceTokenizerIdentity {
    driver_test_tokenizer().identity().clone()
}

fn driver_test_request(max_tokens: u32) -> BackendRequest {
    BackendRequest::raw_completion(
        "model-test",
        "test",
        Some(max_tokens),
        SamplingConfig::Greedy,
    )
}

fn driver_for_test<A>(adapter: A) -> NativeTextDriver<A>
where
    A: NativeTextAdapter,
{
    NativeTextDriver::new(
        "model-test".to_owned(),
        BackendModelMetadata::new("model-test", "native-test").with_family("test"),
        driver_test_tokenizer(),
        adapter,
        8,
    )
}

fn stream_final_chunk<A>(
    driver: &NativeTextDriver<A>,
    request: BackendRequest,
) -> BackendStreamChunk
where
    A: NativeTextAdapter,
{
    let (tx, mut rx) = tokio::sync::mpsc::channel(8);
    driver
        .generate_blocking_stream(request, tx, CancellationToken::new())
        .expect("streaming generation succeeds");
    let mut final_chunk = None;
    while let Some(chunk) = rx.blocking_recv() {
        let chunk = chunk.expect("stream chunk is ok");
        if chunk.finish_reason.is_some() {
            final_chunk = Some(chunk);
        }
    }
    final_chunk.expect("streaming generation emits a final chunk")
}

fn store_driver_prefix_hit(
    adapter: &TestAdapter,
    request: &BackendRequest,
    prompt_tokens: usize,
    max_new_tokens: u32,
    prefix_tokens: &[usize],
    cache: TestCache,
) -> (NativeTextPrefixCacheNamespace, u64) {
    let cache_tokens = native_text_cache_token_capacity(
        prompt_tokens,
        max_new_tokens,
        adapter.max_prefill_tokens(),
        adapter.max_position_embeddings(),
        adapter.family_display_name(),
    )
    .expect("test cache token capacity is valid");
    let namespace_cache_tokens = native_text_cache_namespace_token_bucket(
        cache_tokens,
        adapter.max_position_embeddings(),
        adapter.family_display_name(),
    )
    .expect("test namespace cache token bucket is valid");
    let tokenizer = driver_test_tokenizer();
    let namespace =
        adapter.prefix_cache_namespace(tokenizer.identity(), request, namespace_cache_tokens);
    let hidden = [0.25_f32];
    let caches = [cache];
    let byte_len = TestCache::prefix_cache_entry_bytes(&hidden, &caches);

    adapter.prefix_cache.store(
        namespace.clone(),
        prefix_tokens,
        &hidden,
        &caches,
        &adapter.prefix_cache_metrics,
    );

    (namespace, byte_len)
}

fn driver_prefix_namespace(
    adapter: &TestAdapter,
    request: &BackendRequest,
    prompt_tokens: usize,
    max_new_tokens: u32,
) -> NativeTextPrefixCacheNamespace {
    let cache_tokens = native_text_cache_token_capacity(
        prompt_tokens,
        max_new_tokens,
        adapter.max_prefill_tokens(),
        adapter.max_position_embeddings(),
        adapter.family_display_name(),
    )
    .expect("test cache token capacity is valid");
    let namespace_cache_tokens = native_text_cache_namespace_token_bucket(
        cache_tokens,
        adapter.max_position_embeddings(),
        adapter.family_display_name(),
    )
    .expect("test namespace cache token bucket is valid");
    let tokenizer = driver_test_tokenizer();
    adapter.prefix_cache_namespace(tokenizer.identity(), request, namespace_cache_tokens)
}

fn assert_prefix_cache_entry(
    cache: &NativeTextPrefixCache<TestCache>,
    namespace: &NativeTextPrefixCacheNamespace,
    tokens: &[usize],
) {
    let inner = cache
        .inner
        .lock()
        .expect("prefix cache lock is not poisoned");
    let bucket = inner
        .entries
        .get(namespace)
        .expect("prefix namespace remains resident");
    assert!(
        bucket.contains_key(&tokens.to_vec()),
        "expected checkpoint for tokens {tokens:?}"
    );
}

fn assert_no_prefix_cache_entry(
    cache: &NativeTextPrefixCache<TestCache>,
    namespace: &NativeTextPrefixCacheNamespace,
    tokens: &[usize],
) {
    let inner = cache
        .inner
        .lock()
        .expect("prefix cache lock is not poisoned");
    assert!(
        inner
            .entries
            .get(namespace)
            .is_none_or(|bucket| !bucket.contains_key(&tokens.to_vec())),
        "did not expect checkpoint for tokens {tokens:?}"
    );
}

fn assert_only_prefix_cache_entry(
    cache: &NativeTextPrefixCache<TestCache>,
    namespace: &NativeTextPrefixCacheNamespace,
    tokens: &[usize],
    byte_len: u64,
) {
    let inner = cache
        .inner
        .lock()
        .expect("prefix cache lock is not poisoned");
    let bucket = inner
        .entries
        .get(namespace)
        .expect("prefix namespace remains resident");

    assert_eq!(bucket.len(), 1);
    assert!(bucket.contains_key(&tokens.to_vec()));
    assert_eq!(
        inner
            .entries
            .values()
            .map(std::collections::HashMap::len)
            .sum::<usize>(),
        1
    );
    assert_eq!(inner.used_bytes, byte_len);
}
