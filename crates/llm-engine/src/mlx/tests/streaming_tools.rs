use super::*;

#[tokio::test]
async fn mlx_backend_streams_completion_chunks() {
    let server = FakeMlxServer::start(
        "data:{\"choices\":[{\"text\":\"one \",\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":2,\"prompt_tokens_details\":{\"cached_tokens\":1}}}\n\ndata: {\"choices\":[{\"text\":\"two\",\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":3}}\n\ndata: [DONE]\n\n",
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

    let mut stream = backend.generate_stream(BackendRequest::raw_completion(
        "local-mlx",
        "hello mlx",
        Some(12),
        SamplingConfig::Greedy,
    ));

    let first = stream
        .next()
        .await
        .expect("first stream item")
        .expect("first chunk");
    let second = stream
        .next()
        .await
        .expect("second stream item")
        .expect("second chunk");
    assert!(stream.next().await.is_none());

    assert_eq!(first.text, "one ");
    assert_eq!(first.prompt_tokens, 2);
    assert_eq!(first.prompt_cached_tokens, Some(1));
    assert_eq!(first.completion_tokens, 0);
    assert_eq!(first.finish_reason, None);
    assert_eq!(second.text, "two");
    assert_eq!(second.prompt_cached_tokens, Some(1));
    assert_eq!(second.completion_tokens, 3);
    assert_eq!(second.finish_reason, Some(BackendFinishReason::Stop));
}

#[tokio::test]
async fn mlx_backend_preserves_structured_qwen_tool_call_response() {
    let server = FakeMlxServer::start(
        "data:{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"Cargo.toml\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":5}}\n\ndata:[DONE]\n\n",
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
            "read a file",
            backend_chat_context_with_tools(
                vec![ChatMessage::user("read a file")],
                vec![BackendToolDefinition::function(
                    "read_file",
                    "Read a file.",
                    serde_json::json!({}),
                )],
            ),
            Some(12),
            SamplingConfig::Greedy,
            Some(llm_backend_contracts::BackendToolChoice::RequiredFunction(
                "read_file".to_owned(),
            )),
            false,
            BackendCacheContext::chat_template(
                "chatml/qwen/v1",
                Some("tool-schema-compatibility-v1".to_owned()),
            ),
        ))
        .await
        .expect("mlx generation succeeds");

    assert_eq!(server.received_path(), "/v1/chat/completions");
    let request = server.received_body();
    assert_eq!(request["messages"][0]["role"], "user");
    assert_eq!(request["messages"][0]["content"], "read a file");
    assert_eq!(request["tools"][0]["function"]["name"], "read_file");
    assert_eq!(
        request["tool_choice"],
        serde_json::json!({"type":"function","function":{"name":"read_file"}})
    );
    assert_eq!(output.finish_reason, BackendFinishReason::ToolCalls);
    assert!(output.text.starts_with("<tool_call>"));
    assert!(output.text.contains("\"name\":\"read_file\""));
    assert!(output.text.contains("\"path\":\"Cargo.toml\""));
}

#[tokio::test]
async fn mlx_backend_posts_lossless_qwen_tool_history_to_chat_completion_endpoint() {
    let server = FakeMlxServer::start(
        "data: {\"choices\":[{\"delta\":{\"content\":\"read complete\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3}}\n\ndata: [DONE]\n\n",
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
    let mut tool_result = ChatMessage::tool("call_read_1", "{\"contents\":\"pub mod api;\"}");
    tool_result.name = Some("read_file".to_owned());

    let output = backend
        .generate(BackendRequest::chat_completion(
            "local-mlx",
            "rendered prompt fallback should not be used for structured MLX chat",
            backend_chat_context_with_tools(
                vec![
                    ChatMessage::user("read src/lib.rs"),
                    ChatMessage::assistant_tool_call(
                        "call_read_1",
                        "read_file",
                        serde_json::json!({"path": "src/lib.rs", "_i": 2}),
                    ),
                    tool_result,
                    ChatMessage::user("summarize what you read"),
                ],
                vec![BackendToolDefinition::function(
                    "read_file",
                    "Read a file.",
                    serde_json::json!({}),
                )],
            ),
            Some(12),
            SamplingConfig::Greedy,
            None,
            false,
            BackendCacheContext::chat_template(
                "chatml/qwen/v1",
                Some("tool-schema-compatibility-v1".to_owned()),
            ),
        ))
        .await
        .expect("mlx generation succeeds");

    assert_eq!(output.text, "read complete");
    assert_eq!(server.received_path(), "/v1/chat/completions");
    let request = server.received_body();
    let messages = request["messages"].as_array().expect("messages array");
    assert_eq!(messages.len(), 4);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[0]["content"], "read src/lib.rs");
    assert_eq!(messages[1]["role"], "assistant");
    assert!(messages[1].get("content").is_none());
    assert_eq!(messages[1]["tool_calls"][0]["id"], "call_read_1");
    assert_eq!(messages[1]["tool_calls"][0]["type"], "function");
    assert_eq!(
        messages[1]["tool_calls"][0]["function"]["name"],
        "read_file"
    );
    let arguments = messages[1]["tool_calls"][0]["function"]["arguments"]
        .as_str()
        .expect("tool arguments are serialized as an OpenAI JSON string");
    assert_eq!(
        serde_json::from_str::<Value>(arguments).expect("tool arguments JSON"),
        serde_json::json!({"path": "src/lib.rs", "_i": 2})
    );
    assert_eq!(messages[2]["role"], "tool");
    assert_eq!(messages[2]["tool_call_id"], "call_read_1");
    assert_eq!(messages[2]["name"], "read_file");
    assert_eq!(messages[2]["content"], "{\"contents\":\"pub mod api;\"}");
    assert_eq!(messages[3]["role"], "user");
    assert_eq!(messages[3]["content"], "summarize what you read");
}

#[tokio::test]
async fn mlx_backend_accumulates_streamed_tool_call_fragments() {
    let server = FakeMlxServer::start(
        "data:{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"read_\",\"arguments\":\"{\\\"path\\\"\"}}]},\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":4}}\n\ndata:{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"file\",\"arguments\":\":\\\"Cargo.toml\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"completion_tokens\":5}}\n\ndata:[DONE]\n\n",
    );
    let mut backend = MlxBackend::open_with_options(
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
    backend.metrics = Arc::new(MlxBackendMetrics::default());
    let metrics = backend.metrics.clone();

    let chunks = backend
        .generate_stream(BackendRequest::raw_completion(
            "local-mlx",
            "read a file",
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
    assert!(
        !text.contains("<tool_call>"),
        "structured stream should not synthesize tool markup: {text}"
    );
    let deltas = chunks
        .iter()
        .flat_map(|chunk| &chunk.tool_call_deltas)
        .collect::<Vec<_>>();
    assert_eq!(deltas.len(), 2);
    assert_eq!(
        deltas
            .iter()
            .filter_map(|delta| delta.function.as_ref())
            .filter_map(|function| function.name.as_deref())
            .collect::<String>(),
        "read_file"
    );
    assert_eq!(
        deltas
            .iter()
            .filter_map(|delta| delta.function.as_ref())
            .filter_map(|function| function.arguments.as_deref())
            .collect::<String>(),
        r#"{"path":"Cargo.toml"}"#
    );
    assert_eq!(
        chunks.last().and_then(|chunk| chunk.finish_reason),
        Some(BackendFinishReason::ToolCalls)
    );
    let metrics = metrics.snapshot();
    assert_eq!(metrics["stream_response_headers_ms"]["count"], 1);
    assert_eq!(metrics["stream_first_upstream_byte_ms"]["count"], 1);
    assert_eq!(metrics["stream_first_parsed_chunk_ms"]["count"], 1);
    assert_eq!(metrics["stream_first_tool_delta_ms"]["count"], 1);
    assert_eq!(metrics["stream_upstream_complete_ms"]["count"], 1);
    assert_eq!(
        metrics["last_request_fingerprint"]["protocol"],
        "completions"
    );
    assert_eq!(metrics["last_request_fingerprint"]["stream"], true);
    assert!(metrics["last_request_fingerprint"]["cache_key"].is_string());
    assert!(metrics["last_request_fingerprint"]["prompt_hash"].is_string());
}

#[tokio::test]
async fn mlx_backend_streams_qwen_xml_tool_deltas_and_records_first_tool_delta() {
    let server = FakeMlxServer::start(
        "data:{\"choices\":[{\"text\":\"<tool_call><function=read_file>\",\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":4}}\n\ndata:{\"choices\":[{\"text\":\"<parameter=path>Cargo.toml</parameter></function></tool_call>\",\"finish_reason\":\"tool_calls\"}],\"usage\":{\"completion_tokens\":5}}\n\ndata:[DONE]\n\n",
    );
    let mut backend = MlxBackend::open_with_options(
        "local-mlx",
        server.snapshot_path(),
        MlxBackendOptions {
            endpoint: server.endpoint(),
            family: Some(ModelFamily::Qwen),
            tool_parser: MlxToolParserMode::QwenXml,
            ..MlxBackendOptions::default()
        },
    )
    .await
    .expect("backend opens");
    backend.metrics = Arc::new(MlxBackendMetrics::default());
    let metrics = backend.metrics.clone();

    let chunks = backend
        .generate_stream(BackendRequest::raw_completion(
            "local-mlx",
            "read a file",
            Some(12),
            SamplingConfig::Greedy,
        ))
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("mlx stream succeeds");

    assert_eq!(chunks[0].finish_reason, None);
    assert!(
        !chunks
            .iter()
            .map(|chunk| chunk.text.as_str())
            .collect::<String>()
            .contains("<tool_call>")
    );
    let deltas = chunks
        .iter()
        .flat_map(|chunk| &chunk.tool_call_deltas)
        .collect::<Vec<_>>();
    assert_generated_tool_call_id_is_opaque(deltas[0].id.as_deref().expect("generated id"));
    assert_eq!(
        deltas[0]
            .function
            .as_ref()
            .and_then(|function| function.name.as_deref()),
        Some("read_file")
    );
    let arguments = deltas
        .iter()
        .filter_map(|delta| delta.function.as_ref())
        .filter_map(|function| function.arguments.as_deref())
        .collect::<String>();
    assert_eq!(
        serde_json::from_str::<Value>(&arguments).expect("arguments JSON"),
        serde_json::json!({"path":"Cargo.toml"})
    );
    assert_eq!(
        chunks.last().and_then(|chunk| chunk.finish_reason),
        Some(BackendFinishReason::ToolCalls)
    );
    let metrics = metrics.snapshot();
    assert_eq!(metrics["stream_first_tool_delta_ms"]["count"], 1);
}

#[tokio::test]
async fn mlx_backend_preserves_structured_gemma_tool_call_response() {
    let server = FakeMlxServer::start(
        "data:{\"choices\":[{\"message\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"lookup\",\"arguments\":\"{\\\"query\\\":\\\"rust\\\",\\\"limit\\\":3}\"}}]},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"input_tokens\":4,\"output_tokens\":5}}\n\ndata:[DONE]\n\n",
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
            "lookup rust",
            backend_chat_context(vec![ChatMessage::user("lookup rust")]),
            Some(12),
            SamplingConfig::Greedy,
            None,
            false,
            BackendCacheContext::chat_template("gemma/text-it/v1", None),
        ))
        .await
        .expect("mlx generation succeeds");

    assert_eq!(output.finish_reason, BackendFinishReason::ToolCalls);
    assert!(output.text.starts_with("<|tool_call>call:lookup"));
    assert!(output.text.contains("\"query\":\"rust\""));
    assert!(output.text.contains("\"limit\":3"));
}

#[tokio::test]
async fn mlx_backend_streams_gemma_tool_deltas_without_synthetic_markup() {
    let server = FakeMlxServer::start(
        "data:{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_lookup_1\",\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"arguments\":\"{\\\"query\\\":\\\"rust\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"input_tokens\":4,\"output_tokens\":5}}\n\ndata:[DONE]\n\n",
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

    let chunks = backend
        .generate_stream(BackendRequest::chat_completion(
            "local-mlx",
            "lookup rust",
            backend_chat_context(vec![ChatMessage::user("lookup rust")]),
            Some(12),
            SamplingConfig::Greedy,
            None,
            false,
            BackendCacheContext::chat_template("gemma/text-it/v1", None),
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
    assert!(
        !text.contains("<|tool_call>"),
        "Gemma streaming should trust structured deltas instead of synthetic markup: {text}"
    );
    let deltas = chunks
        .iter()
        .flat_map(|chunk| &chunk.tool_call_deltas)
        .collect::<Vec<_>>();
    assert_eq!(deltas.len(), 1);
    assert_eq!(deltas[0].id.as_deref(), Some("call_lookup_1"));
    assert_eq!(
        deltas[0]
            .function
            .as_ref()
            .and_then(|function| function.name.as_deref()),
        Some("lookup")
    );
    assert_eq!(
        deltas[0]
            .function
            .as_ref()
            .and_then(|function| function.arguments.as_deref()),
        Some(r#"{"query":"rust"}"#)
    );
    assert_eq!(
        chunks.last().and_then(|chunk| chunk.finish_reason),
        Some(BackendFinishReason::ToolCalls)
    );
}

#[tokio::test]
async fn mlx_backend_records_zero_output_gemma_stream_success() {
    let server = FakeMlxServer::start(
        "data:{\"choices\":[{\"delta\":{\"content\":\"\"},\"finish_reason\":\"stop\"}],\"usage\":{\"input_tokens\":128000,\"output_tokens\":0}}\n\ndata:[DONE]\n\n",
    );
    let mut backend = MlxBackend::open_with_options(
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
    backend.metrics = Arc::new(MlxBackendMetrics::default());
    let metrics = backend.metrics.clone();

    let chunks = backend
        .generate_stream(BackendRequest::chat_completion(
            "local-mlx",
            "recall long context",
            backend_chat_context(vec![ChatMessage::user("recall long context")]),
            Some(64),
            SamplingConfig::Greedy,
            None,
            false,
            BackendCacheContext::chat_template("gemma/text-it/v1", None),
        ))
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("mlx stream succeeds");

    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].text, "");
    assert_eq!(chunks[0].completion_tokens, 0);
    assert_eq!(chunks[0].finish_reason, Some(BackendFinishReason::Stop));

    let metrics = metrics.snapshot();
    assert_eq!(metrics["zero_output_successes"], 1);
    let observation = &metrics["last_zero_output_success"];
    assert_eq!(observation["model"], "local-mlx");
    assert_eq!(observation["family"], "gemma");
    assert_eq!(observation["streamed"], true);
    assert_eq!(observation["prompt_tokens"], 128000);
    assert_eq!(observation["completion_tokens"], 0);
    assert_eq!(observation["finish_reason"], "stop");
    assert_eq!(observation["stream_chunks"], 1);
    assert!(observation["response_bytes"].as_u64().unwrap_or_default() > 0);
}

#[tokio::test]
async fn mlx_backend_preserves_structured_deepseek_tool_call_response() {
    let server = FakeMlxServer::start(
        "data:{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"lookup\",\"arguments\":\"{\\\"query\\\":\\\"metal\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":5}}\n\ndata:[DONE]\n\n",
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
            "lookup metal",
            backend_chat_context(vec![ChatMessage::user("lookup metal")]),
            Some(12),
            SamplingConfig::Greedy,
            None,
            false,
            BackendCacheContext::chat_template("deepseek/chat/v1", None),
        ))
        .await
        .expect("mlx generation succeeds");

    assert_eq!(output.finish_reason, BackendFinishReason::ToolCalls);
    assert!(output.text.starts_with("<｜tool▁calls▁begin｜>"));
    assert!(output.text.contains("<｜tool▁sep｜>lookup"));
    assert!(output.text.contains("\"query\":\"metal\""));
}

#[tokio::test]
async fn mlx_backend_preserves_structured_llama_tool_call_response() {
    let server = FakeMlxServer::start(
        "data:{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"lookup\",\"arguments\":\"{\\\"query\\\":\\\"llama\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":5}}\n\ndata:[DONE]\n\n",
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
            "lookup llama",
            backend_chat_context(vec![ChatMessage::user("lookup llama")]),
            Some(12),
            SamplingConfig::Greedy,
            None,
            false,
            BackendCacheContext::chat_template("llama3/instruct/v1", None),
        ))
        .await
        .expect("mlx generation succeeds");

    assert_eq!(output.finish_reason, BackendFinishReason::ToolCalls);
    assert!(output.text.starts_with("<tool_call>"));
    assert!(output.text.contains("\"name\":\"lookup\""));
    assert!(output.text.contains("\"query\":\"llama\""));
}
