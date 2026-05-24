use super::*;

#[tokio::test]
async fn mlx_backend_posts_prompt_to_completion_endpoint() {
    let server = FakeMlxServer::start(
        "data: {\"choices\":[{\"text\":\"MLX says \",\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":3,\"prompt_tokens_details\":{\"cached_tokens\":2}}}\n\ndata: {\"choices\":[{\"text\":\"hi\",\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":4}}\n\ndata: [DONE]\n\n",
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

    let output = backend
        .generate(BackendRequest::raw_completion(
            "local-mlx",
            "hello mlx",
            Some(12),
            SamplingConfig::TopP {
                temperature: 0.7,
                top_p: 0.9,
            },
        ))
        .await
        .expect("mlx generation succeeds");

    assert_eq!(output.text, "MLX says hi");
    assert_eq!(output.prompt_tokens, 3);
    assert_eq!(output.prompt_cached_tokens, Some(2));
    assert_eq!(output.completion_tokens, 4);
    let request = server.received_body();
    assert_eq!(
        request["model"],
        server
            .snapshot_path()
            .canonicalize()
            .expect("canonical snapshot")
            .display()
            .to_string()
    );
    assert_eq!(request["prompt"], "hello mlx");
    assert_eq!(request["max_tokens"], 12);
    assert_eq!(request["temperature"], 0.7);
    assert_eq!(request["top_p"], 0.9);
    assert_eq!(request["stream"], false);
    assert!(
        request.get("stream_options").is_none(),
        "non-streaming completion requests must not include stream_options: {request}"
    );

    let metrics = metrics.snapshot();
    assert_eq!(metrics["requests_total"], 1);
    assert_eq!(metrics["successful_requests"], 1);
    assert_eq!(metrics["failed_requests"], 0);
    assert_eq!(metrics["completion_requests"], 1);
    assert_eq!(metrics["chat_completion_requests"], 0);
    assert_eq!(metrics["stream_chunks"], 2);
    assert_eq!(metrics["http_error_responses"], 0);
    assert_eq!(metrics["request_latency_ms"]["count"], 1);
    assert_eq!(metrics["upstream_request_latency_ms"]["count"], 1);
    assert_eq!(metrics["blocking_upstream_request_latency_ms"]["count"], 1);
    assert_eq!(metrics["streaming_upstream_request_latency_ms"]["count"], 0);
    assert!(
        metrics["request_latency_ms"]["max"]
            .as_f64()
            .expect("MLX latency max is numeric")
            >= metrics["request_latency_ms"]["min"]
                .as_f64()
                .expect("MLX latency min is numeric")
    );
}

#[tokio::test]
async fn mlx_backend_uses_non_streaming_qwen_xml_chat_completion() {
    let server = FakeMlxServer::start(
        r#"{"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_read_1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"Cargo.toml\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":4,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":3}}}"#,
    );
    let backend = MlxBackend::open_with_options(
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
    let output = backend
        .generate(BackendRequest::chat_completion(
            "local-mlx",
            "read a file",
            backend_chat_context_with_tools(
                vec![ChatMessage::user("read Cargo.toml")],
                vec![BackendToolDefinition::function(
                    "read_file",
                    "Read a file.",
                    serde_json::json!({
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"}
                        }
                    }),
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
    assert_eq!(request["stream"], false);
    assert!(
        request.get("stream_options").is_none(),
        "non-streaming chat requests must not include stream_options: {request}"
    );
    assert_eq!(
        request["tool_choice"],
        serde_json::json!({"type":"function","function":{"name":"read_file"}})
    );
    assert!(output.text.starts_with("<tool_call>"));
    assert!(output.text.contains(r#""id":"call_read_1""#));
    assert!(output.text.contains(r#""name":"read_file""#));
    assert!(output.text.contains(r#""path":"Cargo.toml""#));
    assert_eq!(output.prompt_tokens, 4);
    assert_eq!(output.prompt_cached_tokens, Some(3));
    assert_eq!(output.completion_tokens, 5);
    assert_eq!(output.finish_reason, BackendFinishReason::ToolCalls);
}

#[tokio::test]
async fn mlx_backend_adds_qwen_tool_logits_bias_kwargs_for_required_tools() {
    let server = FakeMlxServer::start(
        r#"{"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_read_1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"Cargo.toml\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":4,"completion_tokens":5}}"#,
    );
    let backend = MlxBackend::open_with_options(
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
    let request = BackendRequest::chat_completion(
        "local-mlx",
        "read a file",
        backend_chat_context_with_tools(
            vec![ChatMessage::user("read Cargo.toml")],
            vec![BackendToolDefinition::function(
                "read_file",
                "Read a file.",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"}
                    }
                }),
            )],
        ),
        Some(12),
        SamplingConfig::Greedy,
        Some(llm_backend_contracts::BackendToolChoice::RequiredAny),
        false,
        BackendCacheContext::chat_template(
            "chatml/qwen/v1",
            Some("tool-schema-compatibility-v1".to_owned()),
        ),
    );

    let output = backend
        .generate(request.clone())
        .await
        .expect("mlx generation succeeds");

    assert_eq!(server.received_path(), "/v1/chat/completions");
    let expected_kwargs =
        serde_json::json!({"enable_thinking": false, "enable_tool_logits_bias": true});
    let received = server.received_body();
    assert_eq!(received["tool_choice"], "required");
    assert_eq!(received["chat_template_kwargs"], expected_kwargs);
    let metadata = BackendModelMetadata::new("local-mlx", "mlx").with_family("qwen");
    let fingerprint = mlx_request_fingerprint(
        MlxUpstreamProtocol::ChatCompletions,
        false,
        &metadata,
        &request,
    );
    let expected_hash = {
        let bytes = serde_json::to_vec(&expected_kwargs).expect("kwargs serialize");
        let digest = Sha256::digest(&bytes);
        format!("{digest:x}")
    };
    assert_eq!(
        fingerprint["chat_template_kwargs_hash"].as_str(),
        Some(expected_hash.as_str())
    );
    assert_eq!(output.finish_reason, BackendFinishReason::ToolCalls);
    assert!(output.text.starts_with("<tool_call>"));
    assert!(output.text.contains("\"name\":\"read_file\""));
    assert!(output.text.contains("\"path\":\"Cargo.toml\""));
}

#[tokio::test]
async fn mlx_backend_omits_qwen_tool_logits_bias_kwargs_without_required_tools() {
    let server = FakeMlxServer::start(
        r#"{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":1}}"#,
    );
    let backend = MlxBackend::open_with_options(
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

    let output = backend
        .generate(BackendRequest::chat_completion(
            "local-mlx",
            "read a file",
            backend_chat_context_with_tools(
                vec![ChatMessage::user("read Cargo.toml")],
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

    assert_eq!(output.text, "ok");
    let received = server.received_body();
    assert!(received.get("tool_choice").is_none());
    assert_eq!(
        received["chat_template_kwargs"],
        serde_json::json!({"enable_thinking": false})
    );
}

#[tokio::test]
async fn mlx_backend_streaming_completion_requests_include_usage_by_default() {
    let server = FakeMlxServer::start(
        "data:{\"choices\":[{\"text\":\"one\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":3}}\n\ndata:[DONE]\n\n",
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

    assert_eq!(chunks.len(), 1);
    assert_eq!(server.received_path(), "/v1/completions");
    let request = server.received_body();
    assert_eq!(request["stream"], true);
    assert_eq!(request["stream_options"]["include_usage"], true);
}

#[tokio::test]
async fn mlx_backend_streaming_chat_requests_include_usage_by_default() {
    let server = FakeMlxServer::start(
        "data:{\"choices\":[{\"delta\":{\"content\":\"one\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":3}}\n\ndata:[DONE]\n\n",
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
        .generate_stream(BackendRequest::chat_completion(
            "local-mlx",
            "hello mlx",
            backend_chat_context(vec![ChatMessage::user("hello mlx")]),
            Some(12),
            SamplingConfig::Greedy,
            None,
            false,
            BackendCacheContext::chat_template("chatml/qwen/v1", None),
        ))
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("mlx stream succeeds");

    assert_eq!(chunks.len(), 1);
    assert_eq!(server.received_path(), "/v1/chat/completions");
    let request = server.received_body();
    assert_eq!(request["stream"], true);
    assert_eq!(request["stream_options"]["include_usage"], true);
}

#[tokio::test]
async fn mlx_backend_streaming_requests_omit_usage_when_disabled() {
    let server = FakeMlxServer::start(
        "data:{\"choices\":[{\"text\":\"one\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":3}}\n\ndata:[DONE]\n\n",
    );
    let backend = MlxBackend::open_with_options(
        "local-mlx",
        server.snapshot_path(),
        MlxBackendOptions {
            endpoint: server.endpoint(),
            family: Some(ModelFamily::Qwen),
            include_stream_usage: false,
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

    assert_eq!(chunks.len(), 1);
    let request = server.received_body();
    assert_eq!(request["stream"], true);
    assert!(
        request.get("stream_options").is_none(),
        "stream_options must be omitted when include_stream_usage is false: {request}"
    );
}

#[tokio::test]
async fn mlx_backend_metrics_record_http_errors() {
    let server = FakeMlxServer::start_with_status(
        503,
        "Service Unavailable",
        "{\"error\":\"sidecar warming\"}",
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

    let err = backend
        .generate(BackendRequest::raw_completion(
            "local-mlx",
            "hello mlx",
            Some(12),
            SamplingConfig::Greedy,
        ))
        .await
        .expect_err("HTTP error is surfaced");

    assert!(err.to_string().contains("HTTP 503"));
    let metrics = metrics.snapshot();
    assert_eq!(metrics["requests_total"], 1);
    assert_eq!(metrics["successful_requests"], 0);
    assert_eq!(metrics["failed_requests"], 1);
    assert_eq!(metrics["completion_requests"], 1);
    assert_eq!(metrics["http_error_responses"], 1);
    assert_eq!(metrics["stream_chunks"], 0);
    assert_eq!(metrics["request_latency_ms"]["count"], 1);
}

#[tokio::test]
async fn mlx_backend_metrics_count_request_with_opaque_tool_cache_identity() {
    let server = FakeMlxServer::start(
        "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1}}\n\ndata: [DONE]\n\n",
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

    let output = backend
        .generate(BackendRequest::chat_completion(
            "local-mlx",
            "use lookup",
            backend_chat_context(vec![ChatMessage::user("use lookup")]),
            Some(12),
            SamplingConfig::Greedy,
            None,
            false,
            BackendCacheContext::chat_template("chatml/qwen/v1", Some("not json".to_owned())),
        ))
        .await
        .expect("opaque cache identity does not fail local request building");

    assert_eq!(output.text, "ok");
    let metrics = metrics.snapshot();
    assert_eq!(metrics["requests_total"], 1);
    assert_eq!(metrics["successful_requests"], 1);
    assert_eq!(metrics["failed_requests"], 0);
    assert_eq!(metrics["transport_failures"], 0);
}

#[tokio::test]
async fn mlx_backend_metrics_count_http_status_even_when_error_body_fails() {
    let server = FakeMlxServer::start_with_response_content_length(
        503,
        "Service Unavailable",
        "{\"error\":\"truncated\"}",
        1024,
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

    let err = backend
        .generate(BackendRequest::raw_completion(
            "local-mlx",
            "hello mlx",
            Some(12),
            SamplingConfig::Greedy,
        ))
        .await
        .expect_err("truncated HTTP error body is surfaced");

    assert!(err.to_string().contains("response read failed"));
    let metrics = metrics.snapshot();
    assert_eq!(metrics["requests_total"], 1);
    assert_eq!(metrics["failed_requests"], 1);
    assert_eq!(metrics["http_error_responses"], 1);
    assert_eq!(metrics["transport_failures"], 0);
}

#[tokio::test]
async fn mlx_backend_metrics_record_dropped_streams() {
    let server = FakeMlxServer::start(
        "data:{\"choices\":[{\"text\":\"one \",\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":2,\"prompt_tokens_details\":{\"cached_tokens\":1}}}\n\ndata: {\"choices\":[{\"text\":\"two\",\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":3}}\n\ndata: [DONE]\n\n",
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
    assert_eq!(first.text, "one ");
    drop(stream);

    let metrics = metrics.snapshot();
    assert_eq!(metrics["requests_total"], 1);
    assert_eq!(metrics["successful_requests"], 0);
    assert_eq!(metrics["failed_requests"], 1);
    assert_eq!(metrics["dropped_requests"], 1);
    assert_eq!(metrics["cancelled_requests"], 0);
}

#[tokio::test]
async fn mlx_backend_metrics_record_success_when_stream_stops_after_finish_chunk() {
    let server = FakeMlxServer::start(
        "data:{\"choices\":[{\"text\":\"done\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":3}}\n\ndata: [DONE]\n\n",
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

    let mut stream = backend.generate_stream(BackendRequest::raw_completion(
        "local-mlx",
        "hello mlx",
        Some(12),
        SamplingConfig::Greedy,
    ));
    let chunk = stream
        .next()
        .await
        .expect("stream item")
        .expect("finish chunk");
    assert_eq!(chunk.finish_reason, Some(BackendFinishReason::Stop));
    drop(stream);

    assert_eq!(server.received_body()["stream"], true);
    let metrics = metrics.snapshot();
    assert_eq!(metrics["requests_total"], 1);
    assert_eq!(metrics["successful_requests"], 1);
    assert_eq!(metrics["failed_requests"], 0);
    assert_eq!(metrics["dropped_requests"], 0);
    assert_eq!(metrics["upstream_request_latency_ms"]["count"], 1);
    assert_eq!(metrics["blocking_upstream_request_latency_ms"]["count"], 0);
    assert_eq!(metrics["streaming_upstream_request_latency_ms"]["count"], 1);
}

#[tokio::test]
async fn mlx_backend_metrics_record_in_flight_cancellations() {
    let server = FakeMlxServer::start_with_response_delay(
        "data:{\"choices\":[{\"text\":\"late\",\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
        Duration::from_millis(100),
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
    let cancellation = CancellationToken::new();

    let mut stream = backend.generate_stream_with_cancel(
        BackendRequest::raw_completion("local-mlx", "hello mlx", Some(12), SamplingConfig::Greedy),
        cancellation.clone(),
    );
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(20)).await;
        cancellation.cancel();
    });
    let err = stream
        .next()
        .await
        .expect("cancelled stream item")
        .expect_err("stream is cancelled");
    assert!(err.is_cancelled());

    let metrics = metrics.snapshot();
    assert_eq!(metrics["requests_total"], 1);
    assert_eq!(metrics["successful_requests"], 0);
    assert_eq!(metrics["failed_requests"], 1);
    assert_eq!(metrics["cancelled_requests"], 1);
    assert_eq!(metrics["dropped_requests"], 0);
}
