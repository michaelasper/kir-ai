use super::protocol::{
    MLX_DEEPSEEK_CONTROL_STOP_TOKENS, MLX_QWEN_CONTROL_STOP_TOKENS, MlxToolMarkup,
};
use super::*;
use llm_api::ChatMessage;
use llm_backend::{
    BackendCacheContext, BackendChatContext, BackendRequest, ModelBackend, SamplingConfig,
};
use serde_json::Value;
use std::{
    io::{Read, Write},
    net::TcpListener,
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};
use tempfile::TempDir;

type ParsedMlxChunkForTest = (String, u64, u64, Option<llm_api::FinishReason>);

fn parse_mlx_sse_for_test(
    chunks: &[&str],
    markup: MlxToolMarkup,
) -> Result<Vec<ParsedMlxChunkForTest>, BackendError> {
    let mut parser = MlxSseParser::new("hello mlx", MLX_QWEN_CONTROL_STOP_TOKENS, markup);
    let mut parsed = Vec::new();
    for chunk in chunks {
        parsed.extend(parser.push_str(chunk)?);
    }
    parsed.extend(parser.finish()?);
    Ok(parsed
        .into_iter()
        .map(|chunk| {
            (
                chunk.text,
                chunk.prompt_tokens,
                chunk.completion_tokens,
                chunk.finish_reason,
            )
        })
        .collect())
}

#[tokio::test]
async fn mlx_backend_posts_prompt_to_completion_endpoint() {
    let server = FakeMlxServer::start(
        "data: {\"choices\":[{\"text\":\"MLX says \",\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":3}}\n\ndata: {\"choices\":[{\"text\":\"hi\",\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":4}}\n\ndata: [DONE]\n\n",
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
        .generate(BackendRequest {
            model: "local-mlx".to_owned(),
            prompt: "hello mlx".to_owned(),
            chat_context: None,
            max_tokens: Some(12),
            sampling: SamplingConfig::TopP {
                temperature: 0.7,
                top_p: 0.9,
            },
            required_tool_choice: None,
            json_object_mode: false,
            conversation_mode: false,
            cache_context: BackendCacheContext::raw_prompt(),
        })
        .await
        .expect("mlx generation succeeds");

    assert_eq!(output.text, "MLX says hi");
    assert_eq!(output.prompt_tokens, 3);
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

    let metrics = metrics.snapshot();
    assert_eq!(metrics["requests_total"], 1);
    assert_eq!(metrics["successful_requests"], 1);
    assert_eq!(metrics["failed_requests"], 0);
    assert_eq!(metrics["completion_requests"], 1);
    assert_eq!(metrics["chat_completion_requests"], 0);
    assert_eq!(metrics["stream_chunks"], 2);
    assert_eq!(metrics["http_error_responses"], 0);
    assert_eq!(metrics["request_latency_ms"]["count"], 1);
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
async fn mlx_backend_uses_non_streaming_chat_completion_for_generate() {
    let server = FakeMlxServer::start(
        r#"{"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_read_1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"Cargo.toml\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":4,"completion_tokens":5}}"#,
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
        .generate(BackendRequest {
            model: "local-mlx".to_owned(),
            prompt: "read a file".to_owned(),
            chat_context: Some(BackendChatContext {
                messages: vec![ChatMessage::user("read Cargo.toml")],
            }),
            max_tokens: Some(12),
            sampling: SamplingConfig::Greedy,
            required_tool_choice: Some(llm_backend::BackendToolChoice::RequiredFunction(
                "read_file".to_owned(),
            )),
            json_object_mode: false,
            conversation_mode: true,
            cache_context: BackendCacheContext::chat_template(
                "chatml/qwen/v1",
                Some(r#"[{"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}]"#.to_owned()),
            ),
        })
        .await
        .expect("mlx generation succeeds");

    assert_eq!(server.received_path(), "/v1/chat/completions");
    let request = server.received_body();
    assert_eq!(request["stream"], false);
    assert_eq!(
        request["tool_choice"],
        serde_json::json!({"type":"function","function":{"name":"read_file"}})
    );
    assert!(output.text.starts_with("<tool_call>"));
    assert!(output.text.contains(r#""name":"read_file""#));
    assert!(output.text.contains(r#""path":"Cargo.toml""#));
    assert_eq!(output.prompt_tokens, 4);
    assert_eq!(output.completion_tokens, 5);
    assert_eq!(output.finish_reason, llm_api::FinishReason::ToolCalls);
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
        .generate(BackendRequest {
            model: "local-mlx".to_owned(),
            prompt: "hello mlx".to_owned(),
            chat_context: None,
            max_tokens: Some(12),
            sampling: SamplingConfig::Greedy,
            required_tool_choice: None,
            json_object_mode: false,
            conversation_mode: false,
            cache_context: BackendCacheContext::raw_prompt(),
        })
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
async fn mlx_backend_metrics_skip_local_request_build_errors() {
    let snapshot = tempfile::tempdir().expect("snapshot tempdir");
    let mut backend = MlxBackend::open_with_options(
        "local-mlx",
        snapshot.path(),
        MlxBackendOptions {
            endpoint: Url::parse("http://127.0.0.1:1/v1").expect("url"),
            family: Some(ModelFamily::Qwen),
            ..MlxBackendOptions::default()
        },
    )
    .await
    .expect("backend opens");
    backend.metrics = Arc::new(MlxBackendMetrics::default());
    let metrics = backend.metrics.clone();

    let err = backend
        .generate(BackendRequest {
            model: "local-mlx".to_owned(),
            prompt: "use lookup".to_owned(),
            chat_context: Some(BackendChatContext {
                messages: vec![ChatMessage::user("use lookup")],
            }),
            max_tokens: Some(12),
            sampling: SamplingConfig::Greedy,
            required_tool_choice: None,
            json_object_mode: false,
            conversation_mode: true,
            cache_context: BackendCacheContext::chat_template(
                "chatml/qwen/v1",
                Some("not json".to_owned()),
            ),
        })
        .await
        .expect_err("invalid local request build fails before HTTP");

    assert!(err.to_string().contains("tool schema"));
    let metrics = metrics.snapshot();
    assert_eq!(metrics["requests_total"], 0);
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
        .generate(BackendRequest {
            model: "local-mlx".to_owned(),
            prompt: "hello mlx".to_owned(),
            chat_context: None,
            max_tokens: Some(12),
            sampling: SamplingConfig::Greedy,
            required_tool_choice: None,
            json_object_mode: false,
            conversation_mode: false,
            cache_context: BackendCacheContext::raw_prompt(),
        })
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
        "data:{\"choices\":[{\"text\":\"one \",\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":2}}\n\ndata: {\"choices\":[{\"text\":\"two\",\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":3}}\n\ndata: [DONE]\n\n",
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

    let mut stream = backend.generate_stream(BackendRequest {
        model: "local-mlx".to_owned(),
        prompt: "hello mlx".to_owned(),
        chat_context: None,
        max_tokens: Some(12),
        sampling: SamplingConfig::Greedy,
        required_tool_choice: None,
        json_object_mode: false,
        conversation_mode: false,
        cache_context: BackendCacheContext::raw_prompt(),
    });
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

    let mut stream = backend.generate_stream(BackendRequest {
        model: "local-mlx".to_owned(),
        prompt: "hello mlx".to_owned(),
        chat_context: None,
        max_tokens: Some(12),
        sampling: SamplingConfig::Greedy,
        required_tool_choice: None,
        json_object_mode: false,
        conversation_mode: false,
        cache_context: BackendCacheContext::raw_prompt(),
    });
    let chunk = stream
        .next()
        .await
        .expect("stream item")
        .expect("finish chunk");
    assert_eq!(chunk.finish_reason, Some(llm_api::FinishReason::Stop));
    drop(stream);

    assert_eq!(server.received_body()["stream"], true);
    let metrics = metrics.snapshot();
    assert_eq!(metrics["requests_total"], 1);
    assert_eq!(metrics["successful_requests"], 1);
    assert_eq!(metrics["failed_requests"], 0);
    assert_eq!(metrics["dropped_requests"], 0);
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
        BackendRequest {
            model: "local-mlx".to_owned(),
            prompt: "hello mlx".to_owned(),
            chat_context: None,
            max_tokens: Some(12),
            sampling: SamplingConfig::Greedy,
            required_tool_choice: None,
            json_object_mode: false,
            conversation_mode: false,
            cache_context: BackendCacheContext::raw_prompt(),
        },
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
    assert!(matches!(err, BackendError::Cancelled));

    let metrics = metrics.snapshot();
    assert_eq!(metrics["requests_total"], 1);
    assert_eq!(metrics["successful_requests"], 0);
    assert_eq!(metrics["failed_requests"], 1);
    assert_eq!(metrics["cancelled_requests"], 1);
    assert_eq!(metrics["dropped_requests"], 0);
}

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
        .generate(BackendRequest {
            model: "local-mlx".to_owned(),
            prompt: "<bos><|turn>user\nhello gemma<turn|>\n<|turn>model\n".to_owned(),
            chat_context: Some(BackendChatContext {
                messages: vec![
                    ChatMessage::system("You are Kir."),
                    ChatMessage::user("hello gemma"),
                ],
            }),
            max_tokens: Some(12),
            sampling: SamplingConfig::Greedy,
            required_tool_choice: None,
            json_object_mode: false,
            conversation_mode: false,
            cache_context: BackendCacheContext::raw_prompt(),
        })
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
    assert!(request.get("chat_template_kwargs").is_none());
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
        .generate(BackendRequest {
            model: "local-mlx".to_owned(),
            prompt: "<bos><|turn>user\nuse lookup<turn|>\n<|turn>model\n".to_owned(),
            chat_context: Some(BackendChatContext {
                messages: vec![ChatMessage::user("use lookup")],
            }),
            max_tokens: Some(12),
            sampling: SamplingConfig::Greedy,
            required_tool_choice: None,
            json_object_mode: false,
            conversation_mode: true,
            cache_context: BackendCacheContext::chat_template(
                "gemma/gemma4/v1",
                Some(r#"[{"type":"function"}]"#.to_owned()),
            ),
        })
        .await
        .expect("mlx generation succeeds");

    assert_eq!(output.text, "tool fallback");
    let request = server.received_body();
    assert_eq!(request["messages"][0]["role"], "user");
    assert_eq!(request["messages"][0]["content"], "use lookup");
    assert_eq!(request["tools"][0]["type"], "function");
    assert_eq!(
        request["messages"]
            .as_array()
            .expect("messages array")
            .len(),
        1
    );
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
        .generate(BackendRequest {
            model: "local-mlx".to_owned(),
            prompt: "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜>".to_owned(),
            chat_context: Some(BackendChatContext {
                messages: vec![ChatMessage::user("hello")],
            }),
            max_tokens: Some(12),
            sampling: SamplingConfig::Greedy,
            required_tool_choice: Some(llm_backend::BackendToolChoice::RequiredFunction(
                "lookup".to_owned(),
            )),
            json_object_mode: false,
            conversation_mode: true,
            cache_context: BackendCacheContext::chat_template(
                "deepseek/chat/v1",
                Some(
                    r#"[{"type":"function","function":{"name":"lookup","parameters":{}}}]"#
                        .to_owned(),
                ),
            ),
        })
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
            .generate(BackendRequest {
                model: "local-mlx".to_owned(),
                prompt: "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n".to_owned(),
                chat_context: Some(BackendChatContext {
                    messages: vec![ChatMessage::user("hello")],
                }),
                max_tokens: Some(12),
                sampling: SamplingConfig::Greedy,
                required_tool_choice: None,
                json_object_mode: false,
                conversation_mode: true,
                cache_context: BackendCacheContext::chat_template("llama3/instruct/v1", None),
            })
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
        .generate(BackendRequest {
            model: "local-mlx".to_owned(),
            prompt: prompt.to_owned(),
            chat_context: None,
            max_tokens: Some(12),
            sampling: SamplingConfig::Greedy,
            required_tool_choice: None,
            json_object_mode: false,
            conversation_mode: true,
            cache_context: BackendCacheContext::chat_template("llama3/instruct/v1", None),
        })
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
        .generate(BackendRequest {
            model: "local-mlx".to_owned(),
            prompt: "<|im_start|>user\nreturn json<|im_end|>\n<|im_start|>assistant\n".to_owned(),
            chat_context: Some(BackendChatContext {
                messages: vec![ChatMessage::user("return json")],
            }),
            max_tokens: Some(12),
            sampling: SamplingConfig::Greedy,
            required_tool_choice: None,
            json_object_mode: true,
            conversation_mode: true,
            cache_context: BackendCacheContext::chat_template("chatml/qwen/v1", None),
        })
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
        .generate(BackendRequest {
            model: "local-mlx".to_owned(),
            prompt: "hello mlx".to_owned(),
            chat_context: None,
            max_tokens: Some(12),
            sampling: SamplingConfig::Greedy,
            required_tool_choice: None,
            json_object_mode: false,
            conversation_mode: false,
            cache_context: BackendCacheContext::raw_prompt(),
        })
        .await
        .expect("mlx generation succeeds");

    assert_eq!(output.text, "otter:19");
    assert_eq!(output.finish_reason, llm_api::FinishReason::Stop);
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
        .generate_stream(BackendRequest {
            model: "local-mlx".to_owned(),
            prompt: "hello mlx".to_owned(),
            chat_context: None,
            max_tokens: Some(12),
            sampling: SamplingConfig::Greedy,
            required_tool_choice: None,
            json_object_mode: false,
            conversation_mode: false,
            cache_context: BackendCacheContext::raw_prompt(),
        })
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
        chunks.last().and_then(|chunk| chunk.finish_reason.clone()),
        Some(llm_api::FinishReason::Stop)
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
        .generate(BackendRequest {
            model: "local-mlx".to_owned(),
            prompt: "hello gemma".to_owned(),
            chat_context: None,
            max_tokens: Some(12),
            sampling: SamplingConfig::Greedy,
            required_tool_choice: None,
            json_object_mode: false,
            conversation_mode: false,
            cache_context: BackendCacheContext::raw_prompt(),
        })
        .await
        .expect("mlx generation succeeds");

    assert_eq!(output.text, "hello from gemma");
    assert_eq!(output.finish_reason, llm_api::FinishReason::Stop);
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
        .generate(BackendRequest {
            model: "local-mlx".to_owned(),
            prompt: "hello llama".to_owned(),
            chat_context: None,
            max_tokens: Some(12),
            sampling: SamplingConfig::Greedy,
            required_tool_choice: None,
            json_object_mode: false,
            conversation_mode: false,
            cache_context: BackendCacheContext::raw_prompt(),
        })
        .await
        .expect("mlx generation succeeds");

    assert_eq!(output.text, "hello from llama");
    assert_eq!(output.finish_reason, llm_api::FinishReason::Stop);
}

#[test]
fn mlx_sse_parser_flushes_non_stop_prefix_at_done() {
    let mut parser = MlxSseParser::new(
        "hello mlx",
        MLX_QWEN_CONTROL_STOP_TOKENS,
        MlxToolMarkup::Json,
    );
    let chunks = parser
            .push_str(
                "data:{\"choices\":[{\"text\":\"keep <|im\",\"finish_reason\":null}]}\n\ndata:[DONE]\n\n",
            )
            .expect("parse chunk");
    let final_chunks = parser.finish().expect("finish parser");
    let chunks = chunks.into_iter().chain(final_chunks).collect::<Vec<_>>();

    let text = chunks
        .iter()
        .map(|chunk| chunk.text.as_str())
        .collect::<String>();
    assert_eq!(text, "keep <|im");
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| chunk.completion_tokens)
            .sum::<u64>(),
        2
    );
}

#[test]
fn mlx_sse_parser_handles_deepseek_non_ascii_stop_prefix_checks() {
    let mut parser = MlxSseParser::new(
        "hello deepseek",
        MLX_DEEPSEEK_CONTROL_STOP_TOKENS,
        MlxToolMarkup::DeepSeek,
    );
    let chunks = parser
        .push_str("data:{\"choices\":[{\"text\":\"plain answer\",\"finish_reason\":null}]}\n\n")
        .expect("DeepSeek parser does not panic while checking non-ASCII stop tokens");

    assert_eq!(chunks[0].text, "plain answer");
}

#[test]
fn mlx_sse_parser_strips_split_deepseek_control_stop_tokens() {
    let mut parser = MlxSseParser::new(
        "hello deepseek",
        MLX_DEEPSEEK_CONTROL_STOP_TOKENS,
        MlxToolMarkup::DeepSeek,
    );
    let chunks = parser
        .push_str("data:{\"choices\":[{\"text\":\"answer <｜end\",\"finish_reason\":null}]}\n\n")
        .expect("first split chunk parses");
    let next_chunks = parser
            .push_str(
                "data:{\"choices\":[{\"text\":\"▁of▁sentence｜> ignored\",\"finish_reason\":\"stop\"}]}\n\ndata:[DONE]\n\n",
            )
            .expect("second split chunk parses");
    let final_chunks = parser.finish().expect("finish parser");
    let text = chunks
        .into_iter()
        .chain(next_chunks)
        .chain(final_chunks)
        .map(|chunk| chunk.text)
        .collect::<String>();

    assert_eq!(text, "answer ");
}

#[test]
fn mlx_sse_parser_is_chunk_boundary_invariant_for_tool_calls() {
    let payload = "data:{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"read_\",\"arguments\":\"{\\\"path\\\"\"}}]},\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":4}}\n\ndata:{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"file\",\"arguments\":\":\\\"Cargo.toml\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"completion_tokens\":5}}\n\ndata:[DONE]\n\n";
    let expected =
        parse_mlx_sse_for_test(&[payload], MlxToolMarkup::Json).expect("single chunk parses");

    for split in payload
        .char_indices()
        .map(|(index, _)| index)
        .chain(std::iter::once(payload.len()))
    {
        let actual =
            parse_mlx_sse_for_test(&[&payload[..split], &payload[split..]], MlxToolMarkup::Json)
                .unwrap_or_else(|err| panic!("split at byte {split} failed: {err}"));
        assert_eq!(actual, expected, "split at byte {split}");
    }
}

#[test]
fn mlx_production_module_does_not_depend_on_protocol_test_backend() {
    const FORBIDDEN_TEST_BACKEND_SYMBOLS: &[&str] = &[
        "ProtocolTestBackend",
        "protocol_test",
        "build_router_with_protocol_test_backend",
    ];
    for (name, source) in [
        ("mlx.rs", include_str!("../mlx.rs")),
        ("mlx/client.rs", include_str!("client.rs")),
        ("mlx/metadata.rs", include_str!("metadata.rs")),
        ("mlx/metrics.rs", include_str!("metrics.rs")),
        ("mlx/protocol.rs", include_str!("protocol.rs")),
        ("mlx/request.rs", include_str!("request.rs")),
        ("mlx/sse.rs", include_str!("sse.rs")),
    ] {
        let production_source = source.split("#[cfg(test)]").next().unwrap_or(source);
        for symbol in FORBIDDEN_TEST_BACKEND_SYMBOLS {
            assert!(
                !production_source.contains(symbol),
                "{name} should not depend on test backend symbol {symbol}"
            );
        }
    }
}

#[tokio::test]
async fn mlx_backend_streams_completion_chunks() {
    let server = FakeMlxServer::start(
        "data:{\"choices\":[{\"text\":\"one \",\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":2}}\n\ndata: {\"choices\":[{\"text\":\"two\",\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":3}}\n\ndata: [DONE]\n\n",
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

    let mut stream = backend.generate_stream(BackendRequest {
        model: "local-mlx".to_owned(),
        prompt: "hello mlx".to_owned(),
        chat_context: None,
        max_tokens: Some(12),
        sampling: SamplingConfig::Greedy,
        required_tool_choice: None,
        json_object_mode: false,
        conversation_mode: false,
        cache_context: BackendCacheContext::raw_prompt(),
    });

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
    assert_eq!(first.completion_tokens, 0);
    assert_eq!(first.finish_reason, None);
    assert_eq!(second.text, "two");
    assert_eq!(second.completion_tokens, 3);
    assert_eq!(second.finish_reason, Some(llm_api::FinishReason::Stop));
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
        .generate(BackendRequest {
            model: "local-mlx".to_owned(),
            prompt: "read a file".to_owned(),
            chat_context: Some(BackendChatContext {
                messages: vec![ChatMessage::user("read a file")],
            }),
            max_tokens: Some(12),
            sampling: SamplingConfig::Greedy,
            required_tool_choice: Some(llm_backend::BackendToolChoice::RequiredFunction(
                "read_file".to_owned(),
            )),
            json_object_mode: false,
            conversation_mode: true,
            cache_context: BackendCacheContext::chat_template(
                "chatml/qwen/v1",
                Some(
                    r#"[{"type":"function","function":{"name":"read_file","parameters":{}}}]"#
                        .to_owned(),
                ),
            ),
        })
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
    assert_eq!(output.finish_reason, llm_api::FinishReason::ToolCalls);
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
        .generate(BackendRequest {
            model: "local-mlx".to_owned(),
            prompt: "rendered prompt fallback should not be used for structured MLX chat"
                .to_owned(),
            chat_context: Some(BackendChatContext {
                messages: vec![
                    ChatMessage::user("read src/lib.rs"),
                    ChatMessage::assistant_tool_call(
                        "call_read_1",
                        "read_file",
                        serde_json::json!({"path": "src/lib.rs", "_i": 2}),
                    ),
                    tool_result,
                    ChatMessage::user("summarize what you read"),
                ],
            }),
            max_tokens: Some(12),
            sampling: SamplingConfig::Greedy,
            required_tool_choice: None,
            json_object_mode: false,
            conversation_mode: true,
            cache_context: BackendCacheContext::chat_template(
                "chatml/qwen/v1",
                Some(
                    r#"[{"type":"function","function":{"name":"read_file","parameters":{}}}]"#
                        .to_owned(),
                ),
            ),
        })
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
        .generate_stream(BackendRequest {
            model: "local-mlx".to_owned(),
            prompt: "read a file".to_owned(),
            chat_context: None,
            max_tokens: Some(12),
            sampling: SamplingConfig::Greedy,
            required_tool_choice: None,
            json_object_mode: false,
            conversation_mode: false,
            cache_context: BackendCacheContext::raw_prompt(),
        })
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("mlx stream succeeds");

    let text = chunks
        .iter()
        .map(|chunk| chunk.text.as_str())
        .collect::<String>();
    assert!(text.contains("\"name\":\"read_file\""));
    assert!(text.contains("\"path\":\"Cargo.toml\""));
    assert_eq!(
        chunks.last().and_then(|chunk| chunk.finish_reason.clone()),
        Some(llm_api::FinishReason::ToolCalls)
    );
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
        .generate(BackendRequest {
            model: "local-mlx".to_owned(),
            prompt: "lookup rust".to_owned(),
            chat_context: None,
            max_tokens: Some(12),
            sampling: SamplingConfig::Greedy,
            required_tool_choice: None,
            json_object_mode: false,
            conversation_mode: true,
            cache_context: BackendCacheContext::raw_prompt(),
        })
        .await
        .expect("mlx generation succeeds");

    assert_eq!(output.finish_reason, llm_api::FinishReason::ToolCalls);
    assert!(output.text.starts_with("<|tool_call>call:lookup"));
    assert!(output.text.contains("\"query\":\"rust\""));
    assert!(output.text.contains("\"limit\":3"));
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
        .generate(BackendRequest {
            model: "local-mlx".to_owned(),
            prompt: "lookup metal".to_owned(),
            chat_context: None,
            max_tokens: Some(12),
            sampling: SamplingConfig::Greedy,
            required_tool_choice: None,
            json_object_mode: false,
            conversation_mode: true,
            cache_context: BackendCacheContext::raw_prompt(),
        })
        .await
        .expect("mlx generation succeeds");

    assert_eq!(output.finish_reason, llm_api::FinishReason::ToolCalls);
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
        .generate(BackendRequest {
            model: "local-mlx".to_owned(),
            prompt: "lookup llama".to_owned(),
            chat_context: None,
            max_tokens: Some(12),
            sampling: SamplingConfig::Greedy,
            required_tool_choice: None,
            json_object_mode: false,
            conversation_mode: true,
            cache_context: BackendCacheContext::raw_prompt(),
        })
        .await
        .expect("mlx generation succeeds");

    assert_eq!(output.finish_reason, llm_api::FinishReason::ToolCalls);
    assert!(output.text.starts_with("<tool_call>"));
    assert!(output.text.contains("\"name\":\"lookup\""));
    assert!(output.text.contains("\"query\":\"llama\""));
}

#[tokio::test]
async fn mlx_backend_rejects_model_mismatch_before_http_request() {
    let snapshot = tempfile::tempdir().expect("snapshot tempdir");
    let backend = MlxBackend::open_with_options(
        "local-mlx",
        snapshot.path(),
        MlxBackendOptions {
            family: Some(ModelFamily::Qwen),
            ..MlxBackendOptions::default()
        },
    )
    .await
    .expect("backend opens");

    let err = backend
        .generate(BackendRequest {
            model: "other-model".to_owned(),
            prompt: "hello".to_owned(),
            chat_context: None,
            max_tokens: Some(1),
            sampling: SamplingConfig::Greedy,
            required_tool_choice: None,
            json_object_mode: false,
            conversation_mode: false,
            cache_context: BackendCacheContext::raw_prompt(),
        })
        .await
        .expect_err("model mismatch fails before HTTP");

    assert!(matches!(err, BackendError::ModelNotFound { .. }));
}

#[tokio::test]
async fn mlx_backend_rejects_non_loopback_endpoint() {
    let snapshot = tempfile::tempdir().expect("snapshot tempdir");

    let err = MlxBackend::open_with_options(
        "local-mlx",
        snapshot.path(),
        MlxBackendOptions {
            endpoint: Url::parse("https://example.com/v1").expect("url"),
            ..MlxBackendOptions::default()
        },
    )
    .await
    .expect_err("remote MLX endpoint is rejected");

    assert!(err.to_string().contains("not loopback"));
}

#[tokio::test]
async fn mlx_backend_rejects_manifestless_snapshot_without_family() {
    let snapshot = tempfile::tempdir().expect("snapshot tempdir");

    let err = MlxBackend::open_with_options(
        "local-mlx",
        snapshot.path(),
        MlxBackendOptions {
            endpoint: Url::parse("http://127.0.0.1:18080/v1").expect("url"),
            ..MlxBackendOptions::default()
        },
    )
    .await
    .expect_err("raw MLX family is required");

    assert!(
        err.to_string()
            .contains("MLX backend requires model family metadata")
    );
}

#[tokio::test]
async fn mlx_backend_accepts_gemma_requested_family() {
    let snapshot = tempfile::tempdir().expect("snapshot tempdir");

    let backend = MlxBackend::open_with_options(
        "local-mlx",
        snapshot.path(),
        MlxBackendOptions {
            endpoint: Url::parse("http://127.0.0.1:18080/v1").expect("url"),
            family: Some(ModelFamily::Gemma),
            ..MlxBackendOptions::default()
        },
    )
    .await
    .expect("Gemma MLX backend opens");

    assert_eq!(backend.model_metadata().family.as_deref(), Some("gemma"));
    assert_eq!(backend.model_metadata().loader.as_deref(), Some("mlx"));
}

#[tokio::test]
async fn mlx_backend_rejects_non_mlx_manifest_loader() {
    let snapshot = tempfile::tempdir().expect("snapshot tempdir");
    write_mlx_manifest(snapshot.path(), "native-metal", "qwen");

    let err = MlxBackend::open_with_options(
        "local-mlx",
        snapshot.path(),
        MlxBackendOptions {
            endpoint: Url::parse("http://127.0.0.1:18080/v1").expect("url"),
            ..MlxBackendOptions::default()
        },
    )
    .await
    .expect_err("MLX backend rejects native manifest loader");

    assert!(
        err.to_string()
            .contains("MLX backend requires manifest loader `mlx`")
    );
}

#[tokio::test]
async fn mlx_backend_accepts_llama_requested_family() {
    let snapshot = tempfile::tempdir().expect("snapshot tempdir");

    let backend = MlxBackend::open_with_options(
        "local-mlx",
        snapshot.path(),
        MlxBackendOptions {
            endpoint: Url::parse("http://127.0.0.1:18080/v1").expect("url"),
            family: Some(ModelFamily::Llama),
            ..MlxBackendOptions::default()
        },
    )
    .await
    .expect("Llama MLX backend opens");

    assert_eq!(backend.model_metadata().family.as_deref(), Some("llama"));
    assert_eq!(backend.model_metadata().loader.as_deref(), Some("mlx"));
}

#[tokio::test]
async fn mlx_backend_rejects_unknown_manifest_family() {
    let snapshot = tempfile::tempdir().expect("snapshot tempdir");
    write_mlx_manifest(snapshot.path(), "mlx", "glm");

    let err = MlxBackend::open_with_options(
        "local-mlx",
        snapshot.path(),
        MlxBackendOptions {
            endpoint: Url::parse("http://127.0.0.1:18080/v1").expect("url"),
            ..MlxBackendOptions::default()
        },
    )
    .await
    .expect_err("unknown manifest family is rejected");

    assert!(err.to_string().contains("unsupported model family `glm`"));
}

#[tokio::test]
async fn mlx_backend_rejects_sse_without_done_marker() {
    let server = FakeMlxServer::start(
        "data: {\"choices\":[{\"text\":\"partial\",\"finish_reason\":\"stop\"}]}\n\n",
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

    let err = backend
        .generate(BackendRequest {
            model: "local-mlx".to_owned(),
            prompt: "hello".to_owned(),
            chat_context: None,
            max_tokens: Some(1),
            sampling: SamplingConfig::Greedy,
            required_tool_choice: None,
            json_object_mode: false,
            conversation_mode: false,
            cache_context: BackendCacheContext::raw_prompt(),
        })
        .await
        .expect_err("missing DONE fails closed");

    assert!(err.to_string().contains("[DONE]"));
}

#[tokio::test]
async fn mlx_backend_per_chunk_timeout_detects_stalled_stream() {
    let server = FakeMlxServer::start_with_stall(
        "data:{\"choices\":[{\"text\":\"one\",\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":2}}\n\n",
        Duration::from_secs(300),
    );
    let mut backend = MlxBackend::open_with_options(
        "local-mlx",
        server.snapshot_path(),
        MlxBackendOptions {
            endpoint: server.endpoint(),
            family: Some(ModelFamily::Qwen),
            timeouts: MlxTimeouts {
                connect: Duration::from_secs(5),
                request: Duration::from_secs(5),
                read: Duration::from_millis(100),
            },
        },
    )
    .await
    .expect("backend opens");
    backend.metrics = Arc::new(MlxBackendMetrics::default());
    let metrics = backend.metrics.clone();

    let err = backend
        .generate(BackendRequest {
            model: "local-mlx".to_owned(),
            prompt: "hello mlx".to_owned(),
            chat_context: None,
            max_tokens: Some(12),
            sampling: SamplingConfig::Greedy,
            required_tool_choice: None,
            json_object_mode: false,
            conversation_mode: false,
            cache_context: BackendCacheContext::raw_prompt(),
        })
        .await
        .expect_err("stalled stream produces timeout error");

    assert!(
        err.to_string().contains("stalled"),
        "expected stall error, got: {err}"
    );
    let metrics = metrics.snapshot();
    assert_eq!(metrics["failed_requests"], 1);
    assert_eq!(metrics["stall_failures"], 1);
}

#[tokio::test]
async fn mlx_backend_request_timeout_detects_delayed_response_headers() {
    let server = FakeMlxServer::start_with_response_delay(
        "data:{\"choices\":[{\"text\":\"late\",\"finish_reason\":null}]}\n\ndata: [DONE]\n\n",
        Duration::from_millis(300),
    );
    let mut backend = MlxBackend::open_with_options(
        "local-mlx",
        server.snapshot_path(),
        MlxBackendOptions {
            endpoint: server.endpoint(),
            family: Some(ModelFamily::Qwen),
            timeouts: MlxTimeouts {
                connect: Duration::from_secs(5),
                request: Duration::from_millis(100),
                read: Duration::from_secs(5),
            },
        },
    )
    .await
    .expect("backend opens");
    backend.metrics = Arc::new(MlxBackendMetrics::default());
    let metrics = backend.metrics.clone();

    let err = backend
        .generate(BackendRequest {
            model: "local-mlx".to_owned(),
            prompt: "hello mlx".to_owned(),
            chat_context: None,
            max_tokens: Some(12),
            sampling: SamplingConfig::Greedy,
            required_tool_choice: None,
            json_object_mode: false,
            conversation_mode: false,
            cache_context: BackendCacheContext::raw_prompt(),
        })
        .await
        .expect_err("delayed response headers produce timeout error");

    assert!(
        err.to_string().contains("timed out"),
        "expected timeout error, got: {err}"
    );
    let metrics = metrics.snapshot();
    assert_eq!(metrics["failed_requests"], 1);
    assert_eq!(metrics["stall_failures"], 1);
}

struct FakeMlxServer {
    endpoint: Url,
    snapshot: TempDir,
    received: Arc<Mutex<Option<Value>>>,
    received_path: Arc<Mutex<Option<String>>>,
    join: Option<thread::JoinHandle<()>>,
}

impl FakeMlxServer {
    fn start(response_body: &'static str) -> Self {
        Self::start_with_status(200, "OK", response_body)
    }

    fn start_with_status(
        status_code: u16,
        reason: &'static str,
        response_body: &'static str,
    ) -> Self {
        Self::start_with_response_delay_and_content_length(
            status_code,
            reason,
            response_body,
            response_body.len(),
            Duration::ZERO,
        )
    }

    fn start_with_response_delay(response_body: &'static str, delay: Duration) -> Self {
        Self::start_with_response_delay_and_content_length(
            200,
            "OK",
            response_body,
            response_body.len(),
            delay,
        )
    }

    fn start_with_response_content_length(
        status_code: u16,
        reason: &'static str,
        response_body: &'static str,
        response_content_length: usize,
    ) -> Self {
        Self::start_with_response_delay_and_content_length(
            status_code,
            reason,
            response_body,
            response_content_length,
            Duration::ZERO,
        )
    }

    fn start_with_stall(first_chunk: &'static str, stall_duration: Duration) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind fake mlx server");
        let endpoint = Url::parse(&format!(
            "http://{}/v1",
            listener.local_addr().expect("addr")
        ))
        .expect("endpoint url");
        let received = Arc::new(Mutex::new(None));
        let received_path = Arc::new(Mutex::new(None));
        let received_for_thread = received.clone();
        let received_path_for_thread = received_path.clone();
        let _join = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept fake mlx request");
            let mut bytes = Vec::new();
            let mut buffer = [0_u8; 1024];
            let header_end;
            loop {
                let read = stream.read(&mut buffer).expect("read request");
                assert!(read > 0, "client closed before headers");
                bytes.extend_from_slice(&buffer[..read]);
                if let Some(index) = find_subsequence(&bytes, b"\r\n\r\n") {
                    header_end = index + 4;
                    break;
                }
            }
            let headers = String::from_utf8_lossy(&bytes[..header_end]);
            let request_path = headers
                .lines()
                .next()
                .and_then(|line| line.split_whitespace().nth(1))
                .expect("request path")
                .to_owned();
            *received_path_for_thread.lock().expect("received path lock") = Some(request_path);
            let request_content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().expect("content length"))
                })
                .expect("content-length header");
            while bytes.len() < header_end + request_content_length {
                let read = stream.read(&mut buffer).expect("read body");
                assert!(read > 0, "client closed before body");
                bytes.extend_from_slice(&buffer[..read]);
            }
            let body = &bytes[header_end..header_end + request_content_length];
            *received_for_thread.lock().expect("received lock") =
                Some(serde_json::from_slice(body).expect("json request body"));
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
            );
            let _ = stream.flush();
            let _ = write!(stream, "{:x}\r\n{}\r\n", first_chunk.len(), first_chunk);
            let _ = stream.flush();
            thread::sleep(stall_duration);
        });
        Self {
            endpoint,
            snapshot: tempfile::tempdir().expect("snapshot tempdir"),
            received,
            received_path,
            join: None,
        }
    }

    fn start_with_response_delay_and_content_length(
        status_code: u16,
        reason: &'static str,
        response_body: &'static str,
        response_content_length: usize,
        response_delay: Duration,
    ) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind fake mlx server");
        let endpoint = Url::parse(&format!(
            "http://{}/v1",
            listener.local_addr().expect("addr")
        ))
        .expect("endpoint url");
        let received = Arc::new(Mutex::new(None));
        let received_path = Arc::new(Mutex::new(None));
        let received_for_thread = received.clone();
        let received_path_for_thread = received_path.clone();
        let join = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept fake mlx request");
            let mut bytes = Vec::new();
            let mut buffer = [0_u8; 1024];
            let header_end;
            loop {
                let read = stream.read(&mut buffer).expect("read request");
                assert!(read > 0, "client closed before headers");
                bytes.extend_from_slice(&buffer[..read]);
                if let Some(index) = find_subsequence(&bytes, b"\r\n\r\n") {
                    header_end = index + 4;
                    break;
                }
            }
            let headers = String::from_utf8_lossy(&bytes[..header_end]);
            let request_path = headers
                .lines()
                .next()
                .and_then(|line| line.split_whitespace().nth(1))
                .expect("request path")
                .to_owned();
            *received_path_for_thread.lock().expect("received path lock") = Some(request_path);
            let request_content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().expect("content length"))
                })
                .expect("content-length header");
            while bytes.len() < header_end + request_content_length {
                let read = stream.read(&mut buffer).expect("read body");
                assert!(read > 0, "client closed before body");
                bytes.extend_from_slice(&buffer[..read]);
            }
            let body = &bytes[header_end..header_end + request_content_length];
            *received_for_thread.lock().expect("received lock") =
                Some(serde_json::from_slice(body).expect("json request body"));
            thread::sleep(response_delay);
            let _ = write!(
                stream,
                "HTTP/1.1 {status_code} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response_content_length, response_body
            );
        });
        Self {
            endpoint,
            snapshot: tempfile::tempdir().expect("snapshot tempdir"),
            received,
            received_path,
            join: Some(join),
        }
    }

    fn endpoint(&self) -> Url {
        self.endpoint.clone()
    }

    fn snapshot_path(&self) -> &Path {
        self.snapshot.path()
    }

    fn received_body(&self) -> Value {
        self.received
            .lock()
            .expect("received lock")
            .clone()
            .expect("received request body")
    }

    fn received_path(&self) -> String {
        self.received_path
            .lock()
            .expect("received path lock")
            .clone()
            .expect("received request path")
    }
}

impl Drop for FakeMlxServer {
    fn drop(&mut self) {
        if let Some(join) = self.join.take() {
            join.join().expect("fake server thread");
        }
    }
}

fn find_subsequence(bytes: &[u8], needle: &[u8]) -> Option<usize> {
    bytes
        .windows(needle.len())
        .position(|window| window == needle)
}

fn write_mlx_manifest(snapshot_path: &Path, loader: &str, family: &str) {
    std::fs::write(
        snapshot_path.join("llm-engine-manifest.json"),
        serde_json::json!({
            "schema_version": 1,
            "source": "huggingface",
            "repo_type": "model",
            "repo_id": "example/model",
            "requested_revision": "main",
            "resolved_commit": "0123456789abcdef0123456789abcdef01234567",
            "profile": "test-mlx",
            "family": family,
            "loader": loader,
            "quantization": "4bit",
            "created_at": "2026-05-08T00:00:00Z",
            "snapshot_path": snapshot_path.display().to_string(),
            "files": [],
            "allow_patterns": [],
            "ignore_patterns": []
        })
        .to_string(),
    )
    .expect("manifest");
}
