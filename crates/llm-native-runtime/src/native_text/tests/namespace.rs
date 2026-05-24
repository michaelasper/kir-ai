use super::*;

#[test]
fn prefix_namespace_copies_metadata_and_request_context() {
    let mut metadata = BackendModelMetadata::new("model-a", "native-test").with_family("test");
    metadata.quantization = Some("bf16".to_owned());
    metadata.repo_id = Some("org/model".to_owned());
    metadata.resolved_commit = Some("abc123".to_owned());
    metadata.profile = Some("profile-a".to_owned());
    let request = BackendRequest::chat_completion(
        "model-a",
        "hello",
        BackendChatContext {
            messages: vec![BackendChatMessage {
                role: BackendChatRole::User,
                content: Some("hello".to_owned()),
                name: None,
                tool_call_id: None,
                tool_calls: Vec::new(),
            }],
            tools: Vec::new(),
        },
        Some(1),
        SamplingConfig::Greedy,
        None,
        true,
        llm_backend_contracts::BackendCacheContext::chat_template_with_kwargs(
            "chatml/qwen/v1",
            Some("schema-a".to_owned()),
            Some(r#"{"enable_thinking":false}"#.to_owned()),
        ),
    );
    let expected_cache_key = request.cache_context().key.as_str().to_owned();
    let tokenizer_identity = driver_test_tokenizer_identity();

    let namespace = native_text_prefix_namespace(NativeTextPrefixNamespaceContext {
        model_id: "model-a",
        metadata: &metadata,
        tokenizer_identity: &tokenizer_identity,
        adapter_settings: "native-test-adapter/v1",
        request: &request,
        cache_layout_version: 7,
        cache_tokens: 64,
        max_prefill_tokens: 8,
    });

    assert_eq!(namespace.model_id, "model-a");
    assert_eq!(namespace.backend, "native-test");
    assert_eq!(namespace.family.as_deref(), Some("test"));
    assert_eq!(namespace.quantization.as_deref(), Some("bf16"));
    assert_eq!(namespace.repo_id.as_deref(), Some("org/model"));
    assert_eq!(namespace.resolved_commit.as_deref(), Some("abc123"));
    assert_eq!(namespace.profile.as_deref(), Some("profile-a"));
    assert_eq!(namespace.tokenizer_kind, "huggingface-tokenizer-json");
    assert!(namespace.tokenizer_hash.starts_with("sha256:"));
    assert_eq!(
        namespace.tokenizer_normalization,
        "llm-tokenizer/hf-json/v1"
    );
    assert_eq!(namespace.cache_template_id, "chatml/qwen/v1");
    assert!(
        namespace
            .chat_template_kwargs_hash
            .as_deref()
            .is_some_and(|hash| hash.starts_with("sha256:"))
    );
    assert_eq!(namespace.adapter_settings, "native-test-adapter/v1");
    assert_eq!(namespace.cache_key, expected_cache_key);
    assert_eq!(namespace.tool_schema.as_deref(), Some("schema-a"));
    assert_eq!(
        namespace.request_mode,
        "chat,json_object=true,required_tool=None"
    );
    assert_eq!(namespace.cache_layout_version, 7);
    assert_eq!(namespace.cache_tokens, 64);
    assert_eq!(namespace.max_prefill_tokens, 8);
}

#[test]
fn prefix_cache_reuses_namespace_when_only_sampling_changes() {
    let metadata = BackendModelMetadata::new("model-a", "native-test").with_family("qwen");
    let greedy_request = BackendRequest::raw_completion_with_cache_context(
        "model-a",
        "hello",
        Some(1),
        SamplingConfig::Greedy,
        llm_backend_contracts::BackendCacheContext::chat_template_with_kwargs(
            "chatml/qwen/v1",
            Some("schema-a".to_owned()),
            Some(r#"{"enable_thinking":false}"#.to_owned()),
        ),
    );
    let mut top_p_request = greedy_request.clone();
    top_p_request.sampling = SamplingConfig::TopP {
        temperature: 0.7,
        top_p: 0.8,
    };
    let tokenizer_identity = driver_test_tokenizer_identity();
    let namespace_for = |request: &BackendRequest| {
        native_text_prefix_namespace(NativeTextPrefixNamespaceContext {
            model_id: "model-a",
            metadata: &metadata,
            tokenizer_identity: &tokenizer_identity,
            adapter_settings: "native-test-adapter/v1",
            request,
            cache_layout_version: 1,
            cache_tokens: 16,
            max_prefill_tokens: 8,
        })
    };
    let greedy_namespace = namespace_for(&greedy_request);
    let top_p_namespace = namespace_for(&top_p_request);
    let cache = NativeTextPrefixCache::new(1024);
    let metrics = NativeTextPrefixCacheMetrics::default();
    let caches = vec![TestCache {
        bytes: 8,
        marker: 1,
    }];

    cache.store(
        greedy_namespace.clone(),
        &[1, 2],
        &[0.25, 0.75],
        &caches,
        &metrics,
    );

    assert_eq!(greedy_namespace, top_p_namespace);
    assert!(
        cache
            .lookup(&top_p_namespace, &[1, 2, 3], &metrics)
            .is_some(),
        "sampling controls are intentionally outside the prefix cache namespace"
    );
}

#[test]
fn prefix_cache_namespace_separates_cache_capacity() {
    let cache = NativeTextPrefixCache::new(1024);
    let metrics = NativeTextPrefixCacheMetrics::default();
    let namespace = namespace("capacity-key");
    let larger_capacity_namespace = NativeTextPrefixCacheNamespace {
        cache_tokens: namespace.cache_tokens * 2,
        ..namespace.clone()
    };

    cache.store(
        namespace.clone(),
        &[1, 2],
        &[0.25, 0.75],
        &[TestCache {
            bytes: 8,
            marker: 1,
        }],
        &metrics,
    );

    assert_ne!(namespace, larger_capacity_namespace);
    assert!(
        cache
            .lookup(&larger_capacity_namespace, &[1, 2], &metrics)
            .is_none(),
        "cache capacity buckets are prefix cache compatibility keys"
    );
}

#[test]
fn prefix_cache_namespace_separates_manifest_identity_and_profile() {
    let cache = NativeTextPrefixCache::new(1024);
    let metrics = NativeTextPrefixCacheMetrics::default();
    let namespace = namespace("manifest");
    let different_manifest_namespace = NativeTextPrefixCacheNamespace {
        resolved_commit: Some("def456".to_owned()),
        ..namespace.clone()
    };
    let different_profile_namespace = NativeTextPrefixCacheNamespace {
        profile: Some("profile-b".to_owned()),
        ..namespace.clone()
    };

    cache.store(
        namespace.clone(),
        &[1, 2],
        &[0.25, 0.75],
        &[TestCache {
            bytes: 8,
            marker: 1,
        }],
        &metrics,
    );

    assert_ne!(namespace, different_manifest_namespace);
    assert_ne!(namespace, different_profile_namespace);
    assert!(
        cache
            .lookup(&different_manifest_namespace, &[1, 2], &metrics)
            .is_none(),
        "manifest identity changes must not reuse prefix state"
    );
    assert!(
        cache
            .lookup(&different_profile_namespace, &[1, 2], &metrics)
            .is_none(),
        "profile changes must not reuse prefix state"
    );
}

#[test]
fn prefix_cache_namespace_separates_tokenizer_template_adapter_and_bucket_identity() {
    let cache = NativeTextPrefixCache::new(1024);
    let metrics = NativeTextPrefixCacheMetrics::default();
    let namespace = namespace("shared-route");
    let mismatches = [
        NativeTextPrefixCacheNamespace {
            tokenizer_kind: "different-tokenizer-kind".to_owned(),
            ..namespace.clone()
        },
        NativeTextPrefixCacheNamespace {
            tokenizer_hash: "sha256:different-tokenizer".to_owned(),
            ..namespace.clone()
        },
        NativeTextPrefixCacheNamespace {
            tokenizer_normalization: "llm-tokenizer/hf-json/v2".to_owned(),
            ..namespace.clone()
        },
        NativeTextPrefixCacheNamespace {
            cache_template_id: "template/shared-route/v2".to_owned(),
            ..namespace.clone()
        },
        NativeTextPrefixCacheNamespace {
            chat_template_kwargs_hash: Some("sha256:different-template-kwargs".to_owned()),
            ..namespace.clone()
        },
        NativeTextPrefixCacheNamespace {
            adapter_settings: "native-test-adapter/shared-route/v2".to_owned(),
            ..namespace.clone()
        },
        NativeTextPrefixCacheNamespace {
            cache_tokens: namespace.cache_tokens * 2,
            ..namespace.clone()
        },
    ];

    cache.store(
        namespace.clone(),
        &[1, 2],
        &[0.25, 0.75],
        &[TestCache {
            bytes: 8,
            marker: 1,
        }],
        &metrics,
    );

    assert!(cache.lookup(&namespace, &[1, 2, 3], &metrics).is_some());
    for mismatch in mismatches {
        assert!(
            cache.lookup(&mismatch, &[1, 2, 3], &metrics).is_none(),
            "incompatible shared-prefix identity must miss: {mismatch:?}"
        );
    }
}

#[test]
fn prefix_namespace_identity_changes_with_chat_template_kwargs() {
    let metadata = BackendModelMetadata::new("model-a", "native-test").with_family("qwen");
    let mut request = driver_test_request(1);
    *request.cache_context_mut() =
        llm_backend_contracts::BackendCacheContext::chat_template_with_kwargs(
            "chatml/qwen/v1",
            None,
            Some(r#"{"enable_thinking":false}"#.to_owned()),
        );
    let tokenizer_identity = driver_test_tokenizer_identity();

    let no_thinking = native_text_prefix_namespace(NativeTextPrefixNamespaceContext {
        model_id: "model-a",
        metadata: &metadata,
        tokenizer_identity: &tokenizer_identity,
        adapter_settings: "native-test-adapter/v1",
        request: &request,
        cache_layout_version: 1,
        cache_tokens: 16,
        max_prefill_tokens: 8,
    });
    *request.cache_context_mut() =
        llm_backend_contracts::BackendCacheContext::chat_template_with_kwargs(
            "chatml/qwen/v1",
            None,
            Some(r#"{"enable_thinking":true}"#.to_owned()),
        );
    let thinking = native_text_prefix_namespace(NativeTextPrefixNamespaceContext {
        model_id: "model-a",
        metadata: &metadata,
        tokenizer_identity: &tokenizer_identity,
        adapter_settings: "native-test-adapter/v1",
        request: &request,
        cache_layout_version: 1,
        cache_tokens: 16,
        max_prefill_tokens: 8,
    });

    assert_ne!(no_thinking, thinking);
    assert_ne!(no_thinking.cache_key, thinking.cache_key);
}

#[test]
fn prefix_namespace_identity_changes_with_tool_schema_and_request_mode() {
    fn chat_request(
        tool_schema: &str,
        required_tool_choice: Option<BackendToolChoice>,
        json_object_mode: bool,
    ) -> BackendRequest {
        BackendRequest::chat_completion(
            "model-a",
            "hello",
            BackendChatContext {
                messages: vec![BackendChatMessage {
                    role: BackendChatRole::User,
                    content: Some("hello".to_owned()),
                    name: None,
                    tool_call_id: None,
                    tool_calls: Vec::new(),
                }],
                tools: Vec::new(),
            },
            Some(1),
            SamplingConfig::Greedy,
            required_tool_choice,
            json_object_mode,
            llm_backend_contracts::BackendCacheContext::chat_template(
                "chatml/qwen/v1",
                Some(tool_schema.to_owned()),
            ),
        )
    }

    let metadata = BackendModelMetadata::new("model-a", "native-test").with_family("qwen");
    let base_request = chat_request("schema-a", None, false);
    let different_schema_request = chat_request("schema-b", None, false);
    let required_tool_request = chat_request(
        "schema-a",
        Some(BackendToolChoice::RequiredFunction("lookup".to_owned())),
        false,
    );
    let tokenizer_identity = driver_test_tokenizer_identity();

    let namespace_for = |request: &BackendRequest| {
        native_text_prefix_namespace(NativeTextPrefixNamespaceContext {
            model_id: "model-a",
            metadata: &metadata,
            tokenizer_identity: &tokenizer_identity,
            adapter_settings: "native-test-adapter/v1",
            request,
            cache_layout_version: 1,
            cache_tokens: 16,
            max_prefill_tokens: 8,
        })
    };
    let base = namespace_for(&base_request);
    let different_schema = namespace_for(&different_schema_request);
    let required_tool = namespace_for(&required_tool_request);

    assert_ne!(base, different_schema);
    assert_ne!(base.cache_key, different_schema.cache_key);
    assert_ne!(base.tool_schema, different_schema.tool_schema);
    assert_ne!(base, required_tool);
    assert_ne!(base.request_mode, required_tool.request_mode);
}

#[test]
fn prefix_cache_namespace_separates_required_tool_choice_names() {
    fn chat_request(required_tool_name: &str) -> BackendRequest {
        BackendRequest::chat_completion(
            "model-a",
            "hello",
            BackendChatContext {
                messages: vec![BackendChatMessage {
                    role: BackendChatRole::User,
                    content: Some("hello".to_owned()),
                    name: None,
                    tool_call_id: None,
                    tool_calls: Vec::new(),
                }],
                tools: Vec::new(),
            },
            Some(1),
            SamplingConfig::Greedy,
            Some(BackendToolChoice::RequiredFunction(
                required_tool_name.to_owned(),
            )),
            false,
            llm_backend_contracts::BackendCacheContext::chat_template(
                "chatml/qwen/v1",
                Some("schema-a".to_owned()),
            ),
        )
    }

    let metadata = BackendModelMetadata::new("model-a", "native-test").with_family("qwen");
    let lookup_request = chat_request("lookup");
    let search_request = chat_request("search");
    let tokenizer_identity = driver_test_tokenizer_identity();
    let namespace_for = |request: &BackendRequest| {
        native_text_prefix_namespace(NativeTextPrefixNamespaceContext {
            model_id: "model-a",
            metadata: &metadata,
            tokenizer_identity: &tokenizer_identity,
            adapter_settings: "native-test-adapter/v1",
            request,
            cache_layout_version: 1,
            cache_tokens: 16,
            max_prefill_tokens: 8,
        })
    };
    let lookup_namespace = namespace_for(&lookup_request);
    let search_namespace = namespace_for(&search_request);
    let cache = NativeTextPrefixCache::new(1024);
    let metrics = NativeTextPrefixCacheMetrics::default();

    cache.store(
        lookup_namespace.clone(),
        &[1, 2],
        &[0.25, 0.75],
        &[TestCache {
            bytes: 8,
            marker: 1,
        }],
        &metrics,
    );

    assert_ne!(lookup_namespace, search_namespace);
    assert_ne!(lookup_namespace.request_mode, search_namespace.request_mode);
    assert!(
        cache.lookup(&search_namespace, &[1, 2], &metrics).is_none(),
        "required tool-choice names are prefix cache compatibility keys"
    );
}
