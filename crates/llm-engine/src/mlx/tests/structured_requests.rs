use super::*;

#[tokio::test]
async fn mlx_backend_posts_gemma_structured_messages_to_chat_completion_endpoint() {
    let server = FakeMlxServer::start(
        "data: {\"choices\":[{\"delta\":{\"content\":\"gemma says hi\"},\"finish_reason\":\"stop\"}],\"usage\":{\"input_tokens\":6,\"output_tokens\":4}}\n\ndata: [DONE]\n\n",
    );
    let backend = MlxBackend::open_with_options(
        "local-mlx",
        server.snapshot_path(),
        MlxBackendOptions {
            endpoint: server.endpoint(),
            family: Some(ModelFamily::Gemma),
            ..MlxBackendOptions::default()
        },
    )
    .await
    .expect("backend opens");

    let output = backend
        .generate(BackendRequest::chat_completion(
            "local-mlx",
            "<bos><|turn>user\nhello gemma<turn|>\n<|turn>model\n",
            backend_chat_context(vec![
                ChatMessage::system("You are Kir."),
                ChatMessage::user("hello gemma"),
            ]),
            Some(12),
            SamplingConfig::Greedy,
            None,
            false,
            BackendCacheContext::raw_prompt(),
        ))
        .await
        .expect("mlx generation succeeds");

    assert_eq!(output.text, "gemma says hi");
    assert_eq!(output.prompt_tokens, 6);
    assert_eq!(output.completion_tokens, 4);
    let request = server.received_body();
    assert_eq!(
        request["model"].as_str(),
        Some(backend.upstream_model.as_str())
    );
    assert_eq!(request["messages"][0]["role"], "system");
    assert_eq!(request["messages"][0]["content"], "You are Kir.");
    assert_eq!(request["messages"][1]["role"], "user");
    assert_eq!(request["messages"][1]["content"], "hello gemma");
    assert_eq!(request["stream"], false);
    assert_eq!(
        request["chat_template_kwargs"],
        serde_json::json!({"enable_thinking": false})
    );
}

#[tokio::test]
async fn mlx_backend_posts_tool_schema_with_structured_chat_messages() {
    let server = FakeMlxServer::start(
        "data: {\"choices\":[{\"delta\":{\"content\":\"tool fallback\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
    );
    let backend = MlxBackend::open_with_options(
        "local-mlx",
        server.snapshot_path(),
        MlxBackendOptions {
            endpoint: server.endpoint(),
            family: Some(ModelFamily::Gemma),
            ..MlxBackendOptions::default()
        },
    )
    .await
    .expect("backend opens");

    let output = backend
        .generate(BackendRequest::chat_completion(
            "local-mlx",
            "<bos><|turn>user\nuse lookup<turn|>\n<|turn>model\n",
            backend_chat_context_with_tools(
                vec![ChatMessage::user("use lookup")],
                vec![BackendToolDefinition::function(
                    "lookup",
                    "Lookup docs.",
                    serde_json::json!({}),
                )],
            ),
            Some(12),
            SamplingConfig::Greedy,
            None,
            false,
            BackendCacheContext::chat_template(
                "gemma/gemma4/v1",
                Some("tool-schema-compatibility-v1".to_owned()),
            ),
        ))
        .await
        .expect("mlx generation succeeds");

    assert_eq!(output.text, "tool fallback");
    let request = server.received_body();
    assert_eq!(request["messages"][0]["role"], "user");
    assert_eq!(request["messages"][0]["content"], "use lookup");
    assert_eq!(request["tools"][0]["type"], "function");
    assert_eq!(
        request["chat_template_kwargs"],
        serde_json::json!({"enable_thinking": false})
    );
    assert_eq!(
        request["messages"]
            .as_array()
            .expect("messages array")
            .len(),
        1
    );
}

#[test]
fn mlx_request_builder_does_not_parse_cache_tool_schema_as_request_tools() {
    let metadata = BackendModelMetadata::new("local-mlx", "mlx").with_family("gemma");
    let request = BackendRequest::chat_completion(
        "local-mlx",
        "<bos><|turn>user\nplain chat<turn|>\n<|turn>model\n",
        backend_chat_context(vec![ChatMessage::user("plain chat")]),
        Some(12),
        SamplingConfig::Greedy,
        None,
        false,
        BackendCacheContext::chat_template(
            "gemma/gemma4/v1",
            Some("opaque-cache-compatibility-token".to_owned()),
        ),
    );

    let upstream_request =
        super::request::build_upstream_request("/tmp/local-mlx", &metadata, &request, false, false)
            .expect("MLX request building does not parse cache compatibility identity");

    assert_eq!(
        upstream_request.protocol(),
        MlxUpstreamProtocol::ChatCompletions
    );
    assert_eq!(upstream_request.content_type(), "application/json");
    let request: Value =
        serde_json::from_slice(upstream_request.body()).expect("request JSON parses");
    assert!(request.get("tools").is_none());
}

#[test]
fn mlx_request_builder_rejects_gemma_required_tool_choice_with_attribution() {
    let metadata = BackendModelMetadata::new("local-gemma4-e2b", "mlx").with_family("gemma");
    let request = BackendRequest::chat_completion(
        "local-gemma4-e2b",
        "<bos><|turn>user\nrecord observation<turn|>\n<|turn>model\n",
        backend_chat_context_with_tools(
            vec![ChatMessage::user("record observation")],
            vec![BackendToolDefinition::function(
                "record_agentic_observation",
                "Record a structured benchmark observation.",
                serde_json::json!({}),
            )],
        ),
        Some(12),
        SamplingConfig::Greedy,
        Some(llm_backend_contracts::BackendToolChoice::RequiredFunction(
            "record_agentic_observation".to_owned(),
        )),
        false,
        BackendCacheContext::chat_template(
            "gemma/text-it/v1",
            Some("tool-schema-compatibility-v1".to_owned()),
        ),
    );

    let err = super::request::build_upstream_request(
        "/tmp/local-gemma4-e2b",
        &metadata,
        &request,
        true,
        true,
    )
    .expect_err("Gemma MLX required tool choice must fail closed before upstream proxying");

    assert!(err.is_unsupported_request());
    let message = err.to_string();
    assert!(message.contains("MLX Gemma"));
    assert!(message.contains("model `local-gemma4-e2b`"));
    assert!(message.contains("backend `mlx`"));
    assert!(message.contains("family `gemma`"));
    assert!(message.contains("record_agentic_observation"));
}

#[tokio::test]
async fn mlx_backend_routes_deepseek_chat_to_chat_completion_endpoint() {
    let server = FakeMlxServer::start(
        "data: {\"choices\":[{\"delta\":{\"content\":\"deepseek says hi\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":6,\"completion_tokens\":4}}\n\ndata: [DONE]\n\n",
    );
    let backend = MlxBackend::open_with_options(
        "local-mlx",
        server.snapshot_path(),
        MlxBackendOptions {
            endpoint: server.endpoint(),
            family: Some(ModelFamily::DeepSeek),
            ..MlxBackendOptions::default()
        },
    )
    .await
    .expect("backend opens");

    let output = backend
        .generate(BackendRequest::chat_completion(
            "local-mlx",
            "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜>",
            backend_chat_context_with_tools(
                vec![ChatMessage::user("hello")],
                vec![BackendToolDefinition::function(
                    "lookup",
                    "Lookup docs.",
                    serde_json::json!({}),
                )],
            ),
            Some(12),
            SamplingConfig::Greedy,
            Some(llm_backend_contracts::BackendToolChoice::RequiredFunction(
                "lookup".to_owned(),
            )),
            false,
            BackendCacheContext::chat_template(
                "deepseek/chat/v1",
                Some("tool-schema-compatibility-v1".to_owned()),
            ),
        ))
        .await
        .expect("mlx generation succeeds");

    assert_eq!(output.text, "deepseek says hi");
    assert_eq!(server.received_path(), "/v1/chat/completions");
    let request = server.received_body();
    assert_eq!(request["messages"][0]["role"], "user");
    assert_eq!(request["messages"][0]["content"], "hello");
    assert_eq!(request["tools"][0]["function"]["name"], "lookup");
    assert_eq!(
        request["tool_choice"],
        serde_json::json!({"type":"function","function":{"name":"lookup"}})
    );
    assert!(request.get("chat_template_kwargs").is_none());
}

#[tokio::test]
async fn mlx_backend_routes_llama_chat_to_chat_completion_endpoint() {
    let server = FakeMlxServer::start(
        "data: {\"choices\":[{\"delta\":{\"content\":\"llama says hi\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":6,\"completion_tokens\":4}}\n\ndata: [DONE]\n\n",
    );
    let backend = MlxBackend::open_with_options(
        "local-mlx",
        server.snapshot_path(),
        MlxBackendOptions {
            endpoint: server.endpoint(),
            family: Some(ModelFamily::Llama),
            ..MlxBackendOptions::default()
        },
    )
    .await
    .expect("backend opens");

    let output = backend
        .generate(BackendRequest::chat_completion(
            "local-mlx",
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            backend_chat_context(vec![ChatMessage::user("hello")]),
            Some(12),
            SamplingConfig::Greedy,
            None,
            false,
            BackendCacheContext::chat_template("llama3/instruct/v1", None),
        ))
        .await
        .expect("mlx generation succeeds");

    assert_eq!(output.text, "llama says hi");
    assert_eq!(server.received_path(), "/v1/chat/completions");
    let request = server.received_body();
    assert_eq!(request["messages"][0]["role"], "user");
    assert_eq!(request["messages"][0]["content"], "hello");
    assert!(request.get("chat_template_kwargs").is_none());
}

#[tokio::test]
async fn mlx_backend_routes_llama_rendered_prompt_fallback_to_completion_endpoint() {
    let server = FakeMlxServer::start(
        "data: {\"choices\":[{\"text\":\"llama says hi\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":6,\"completion_tokens\":4}}\n\ndata: [DONE]\n\n",
    );
    let backend = MlxBackend::open_with_options(
        "local-mlx",
        server.snapshot_path(),
        MlxBackendOptions {
            endpoint: server.endpoint(),
            family: Some(ModelFamily::Llama),
            ..MlxBackendOptions::default()
        },
    )
    .await
    .expect("backend opens");
    let prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nlookup rust<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{\"name\":\"lookup\",\"parameters\":{\"query\":\"rust\"}}<|eot_id|><|start_header_id|>ipython<|end_header_id|>\n\n{\"answer\":\"systems\"}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";

    let output = backend
        .generate(BackendRequest::raw_completion_with_cache_context(
            "local-mlx",
            prompt,
            Some(12),
            SamplingConfig::Greedy,
            BackendCacheContext::chat_template("llama3/instruct/v1", None),
        ))
        .await
        .expect("mlx generation succeeds");

    assert_eq!(output.text, "llama says hi");
    assert_eq!(server.received_path(), "/v1/completions");
    let request = server.received_body();
    assert_eq!(request["prompt"], prompt);
    assert!(request.get("messages").is_none());
}

#[tokio::test]
async fn mlx_backend_posts_json_object_response_format_to_chat_completion_endpoint() {
    let server = FakeMlxServer::start(
        "data: {\"choices\":[{\"delta\":{\"content\":\"{\\\"ok\\\":true}\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2}}\n\ndata: [DONE]\n\n",
    );
    let backend = MlxBackend::open_with_options(
        "local-mlx",
        server.snapshot_path(),
        MlxBackendOptions {
            endpoint: server.endpoint(),
            family: Some(ModelFamily::Qwen),
            ..MlxBackendOptions::default()
        },
    )
    .await
    .expect("backend opens");

    let output = backend
        .generate(BackendRequest::chat_completion(
            "local-mlx",
            "<|im_start|>user\nreturn json<|im_end|>\n<|im_start|>assistant\n",
            backend_chat_context(vec![ChatMessage::user("return json")]),
            Some(12),
            SamplingConfig::Greedy,
            None,
            true,
            BackendCacheContext::chat_template("chatml/qwen/v1", None),
        ))
        .await
        .expect("mlx generation succeeds");

    assert_eq!(output.text, "{\"ok\":true}");
    assert_eq!(server.received_path(), "/v1/chat/completions");
    let request = server.received_body();
    assert_eq!(
        request["response_format"],
        serde_json::json!({"type":"json_object"})
    );
    assert_eq!(
        request["chat_template_kwargs"],
        serde_json::json!({"enable_thinking": false})
    );
}

#[tokio::test]
async fn mlx_backend_uses_metadata_kwargs_for_request_body_and_fingerprint() {
    let server = FakeMlxServer::start(
        "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":1}}\n\ndata: [DONE]\n\n",
    );
    let backend = MlxBackend::open_with_options(
        "local-mlx",
        server.snapshot_path(),
        MlxBackendOptions {
            endpoint: server.endpoint(),
            family: Some(ModelFamily::Qwen),
            ..MlxBackendOptions::default()
        },
    )
    .await
    .expect("backend opens");
    let request = BackendRequest::chat_completion(
        "local-mlx",
        "<|im_start|>user\nsay ok<|im_end|>\n<|im_start|>assistant\n",
        backend_chat_context(vec![ChatMessage::user("say ok")]),
        Some(12),
        SamplingConfig::Greedy,
        None,
        false,
        BackendCacheContext::chat_template("chatml/qwen/v1", None),
    );
    let metadata = BackendModelMetadata::new("local-mlx", "mlx").with_family("qwen");

    let chunks = backend
        .generate_stream(request.clone())
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("mlx stream succeeds");

    assert!(!chunks.is_empty());
    let received = server.received_body();
    assert_eq!(
        received["chat_template_kwargs"],
        serde_json::json!({"enable_thinking": false})
    );
    assert!(received.get("cache_key").is_none());
    assert!(received.get("session_id").is_none());
    assert!(received.get("prompt_cache_key").is_none());
    let fingerprint = mlx_request_fingerprint(
        MlxUpstreamProtocol::ChatCompletions,
        true,
        &metadata,
        &request,
    );
    let expected_hash = {
        let bytes = serde_json::to_vec(&serde_json::json!({"enable_thinking": false}))
            .expect("kwargs serialize");
        let digest = Sha256::digest(&bytes);
        format!("{digest:x}")
    };
    assert_eq!(
        fingerprint["chat_template_kwargs_hash"].as_str(),
        Some(expected_hash.as_str())
    );
}

#[tokio::test]
async fn mlx_backend_uses_gemma_metadata_kwargs_for_request_body_and_fingerprint() {
    let server = FakeMlxServer::start(
        "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":1}}\n\ndata: [DONE]\n\n",
    );
    let backend = MlxBackend::open_with_options(
        "local-mlx",
        server.snapshot_path(),
        MlxBackendOptions {
            endpoint: server.endpoint(),
            family: Some(ModelFamily::Gemma),
            ..MlxBackendOptions::default()
        },
    )
    .await
    .expect("backend opens");
    let request = BackendRequest::chat_completion(
        "local-mlx",
        "<bos><|turn>user\nsay ok<turn|>\n<|turn>model\n",
        backend_chat_context(vec![ChatMessage::user("say ok")]),
        Some(12),
        SamplingConfig::Greedy,
        None,
        false,
        BackendCacheContext::chat_template("gemma/text-it/v1", None),
    );
    let metadata = BackendModelMetadata::new("local-mlx", "mlx").with_family("gemma");

    let chunks = backend
        .generate_stream(request.clone())
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("mlx stream succeeds");

    assert!(!chunks.is_empty());
    let received = server.received_body();
    assert_eq!(
        received["chat_template_kwargs"],
        serde_json::json!({"enable_thinking": false})
    );
    let fingerprint = mlx_request_fingerprint(
        MlxUpstreamProtocol::ChatCompletions,
        true,
        &metadata,
        &request,
    );
    let expected_hash = {
        let bytes = serde_json::to_vec(&serde_json::json!({"enable_thinking": false}))
            .expect("kwargs serialize");
        let digest = Sha256::digest(&bytes);
        format!("{digest:x}")
    };
    assert_eq!(
        fingerprint["chat_template_kwargs_hash"].as_str(),
        Some(expected_hash.as_str())
    );
}

#[tokio::test]
async fn mlx_backend_strips_control_stop_tokens_from_completion_text() {
    let server = FakeMlxServer::start(
        "data: {\"choices\":[{\"text\":\"otter:19<|im_end|>\\n\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":6}}\n\ndata: [DONE]\n\n",
    );
    let backend = MlxBackend::open_with_options(
        "local-mlx",
        server.snapshot_path(),
        MlxBackendOptions {
            endpoint: server.endpoint(),
            family: Some(ModelFamily::Qwen),
            ..MlxBackendOptions::default()
        },
    )
    .await
    .expect("backend opens");

    let output = backend
        .generate(BackendRequest::raw_completion(
            "local-mlx",
            "hello mlx",
            Some(12),
            SamplingConfig::Greedy,
        ))
        .await
        .expect("mlx generation succeeds");

    assert_eq!(output.text, "otter:19");
    assert_eq!(output.finish_reason, BackendFinishReason::Stop);
}

#[tokio::test]
async fn mlx_backend_strips_split_control_stop_tokens_from_stream() {
    let server = FakeMlxServer::start(
        "data:{\"choices\":[{\"text\":\"otter:19<|im\",\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":2}}\n\ndata: {\"choices\":[{\"text\":\"_end|>\\n\",\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":6}}\n\ndata: [DONE]\n\n",
    );
    let backend = MlxBackend::open_with_options(
        "local-mlx",
        server.snapshot_path(),
        MlxBackendOptions {
            endpoint: server.endpoint(),
            family: Some(ModelFamily::Qwen),
            ..MlxBackendOptions::default()
        },
    )
    .await
    .expect("backend opens");

    let chunks = backend
        .generate_stream(BackendRequest::raw_completion(
            "local-mlx",
            "hello mlx",
            Some(12),
            SamplingConfig::Greedy,
        ))
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("mlx stream succeeds");

    let text = chunks
        .iter()
        .map(|chunk| chunk.text.as_str())
        .collect::<String>();
    assert_eq!(text, "otter:19");
    assert_eq!(
        chunks.last().and_then(|chunk| chunk.finish_reason),
        Some(BackendFinishReason::Stop)
    );
}

#[tokio::test]
async fn mlx_backend_strips_gemma_control_stop_tokens_from_completion_text() {
    let server = FakeMlxServer::start(
        "data: {\"choices\":[{\"text\":\"hello from gemma<turn|>\\n\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":5}}\n\ndata: [DONE]\n\n",
    );
    let backend = MlxBackend::open_with_options(
        "local-mlx",
        server.snapshot_path(),
        MlxBackendOptions {
            endpoint: server.endpoint(),
            family: Some(ModelFamily::Gemma),
            ..MlxBackendOptions::default()
        },
    )
    .await
    .expect("backend opens");

    let output = backend
        .generate(BackendRequest::raw_completion(
            "local-mlx",
            "hello gemma",
            Some(12),
            SamplingConfig::Greedy,
        ))
        .await
        .expect("mlx generation succeeds");

    assert_eq!(output.text, "hello from gemma");
    assert_eq!(output.finish_reason, BackendFinishReason::Stop);
}

#[tokio::test]
async fn mlx_backend_strips_llama_control_stop_tokens_from_completion_text() {
    let server = FakeMlxServer::start(
        "data: {\"choices\":[{\"text\":\"hello from llama<|eot_id|>\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":5}}\n\ndata: [DONE]\n\n",
    );
    let backend = MlxBackend::open_with_options(
        "local-mlx",
        server.snapshot_path(),
        MlxBackendOptions {
            endpoint: server.endpoint(),
            family: Some(ModelFamily::Llama),
            ..MlxBackendOptions::default()
        },
    )
    .await
    .expect("backend opens");

    let output = backend
        .generate(BackendRequest::raw_completion(
            "local-mlx",
            "hello llama",
            Some(12),
            SamplingConfig::Greedy,
        ))
        .await
        .expect("mlx generation succeeds");

    assert_eq!(output.text, "hello from llama");
    assert_eq!(output.finish_reason, BackendFinishReason::Stop);
}
