use super::protocol::{
    MLX_DEEPSEEK_CONTROL_STOP_TOKENS, MLX_QWEN_CONTROL_STOP_TOKENS, MlxToolMarkup,
    MlxUpstreamProtocol,
};
use super::sse::{fold_mlx_chunks, parse_mlx_completion_body};
use super::*;
use llm_api::ChatMessage;
use llm_backend::{
    BackendCacheContext, BackendChatContext, BackendChatMessage, BackendChatRole,
    BackendFinishReason, BackendModelMetadata, BackendRequest, BackendToolCall,
    BackendToolCallDelta, BackendToolCallFunction, BackendToolCallType, BackendToolDefinition,
    ModelBackend, SamplingConfig,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    io::{Read, Write},
    net::TcpListener,
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};
use tempfile::TempDir;

type ParsedMlxChunkForTest = (
    String,
    Vec<BackendToolCallDelta>,
    u64,
    u64,
    Option<BackendFinishReason>,
);

fn backend_messages(messages: Vec<ChatMessage>) -> Vec<BackendChatMessage> {
    messages.into_iter().map(backend_message).collect()
}

fn backend_chat_context(messages: Vec<ChatMessage>) -> BackendChatContext {
    backend_chat_context_with_tools(messages, Vec::new())
}

fn backend_chat_context_with_tools(
    messages: Vec<ChatMessage>,
    tools: Vec<BackendToolDefinition>,
) -> BackendChatContext {
    BackendChatContext {
        messages: backend_messages(messages),
        tools,
    }
}

fn backend_message(message: ChatMessage) -> BackendChatMessage {
    BackendChatMessage {
        role: match message.role {
            llm_api::ChatRole::System => BackendChatRole::System,
            llm_api::ChatRole::User => BackendChatRole::User,
            llm_api::ChatRole::Assistant => BackendChatRole::Assistant,
            llm_api::ChatRole::Tool => BackendChatRole::Tool,
        },
        content: message.content,
        name: message.name,
        tool_call_id: message.tool_call_id,
        tool_calls: message
            .tool_calls
            .into_iter()
            .map(|tool_call| BackendToolCall {
                id: tool_call.id,
                call_type: match tool_call.call_type {
                    llm_api::ToolCallType::Function => BackendToolCallType::Function,
                },
                function: BackendToolCallFunction {
                    name: tool_call.function.name,
                    arguments: tool_call.function.arguments,
                },
            })
            .collect(),
    }
}

fn parse_mlx_sse_for_test(
    chunks: &[&str],
    markup: MlxToolMarkup,
) -> Result<Vec<ParsedMlxChunkForTest>, BackendError> {
    let mut parser = MlxSseParser::new_streaming("hello mlx", MLX_QWEN_CONTROL_STOP_TOKENS, markup);
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
                chunk.tool_call_deltas,
                chunk.prompt_tokens,
                chunk.completion_tokens,
                chunk.finish_reason,
            )
        })
        .collect())
}

fn parse_mlx_sse_for_test_with_tool_schema(
    chunks: &[&str],
    markup: MlxToolMarkup,
    tool_schema: &str,
) -> Result<Vec<ParsedMlxChunkForTest>, BackendError> {
    let mut parser = MlxSseParser::new_streaming_with_tool_schema(
        "hello mlx",
        MLX_QWEN_CONTROL_STOP_TOKENS,
        markup,
        Some(tool_schema),
    )?;
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
                chunk.tool_call_deltas,
                chunk.prompt_tokens,
                chunk.completion_tokens,
                chunk.finish_reason,
            )
        })
        .collect())
}

fn mlx_text_sse_frame(text: &str, finish_reason: Option<&str>) -> String {
    format!(
        "data:{}\n\n",
        serde_json::json!({
            "choices": [{
                "text": text,
                "finish_reason": finish_reason,
            }]
        })
    )
}

#[test]
fn mlx_output_observation_saturates_completion_tokens() {
    let mut observation = MlxOutputObservation {
        completion_tokens: u64::MAX - 1,
        ..MlxOutputObservation::default()
    };

    observation.observe_chunk(&BackendStreamChunk {
        text: String::new(),
        tool_call_deltas: Vec::new(),
        prompt_tokens: 0,
        prompt_cached_tokens: None,
        completion_tokens: 2,
        finish_reason: None,
        progress: None,
    });

    assert_eq!(observation.completion_tokens, u64::MAX);
}

#[test]
fn mlx_chunk_folding_saturates_completion_tokens() {
    let output = fold_mlx_chunks(
        vec![
            BackendStreamChunk {
                text: "hello".to_owned(),
                tool_call_deltas: Vec::new(),
                prompt_tokens: 1,
                prompt_cached_tokens: None,
                completion_tokens: u64::MAX - 1,
                finish_reason: None,
                progress: None,
            },
            BackendStreamChunk {
                text: " world".to_owned(),
                tool_call_deltas: Vec::new(),
                prompt_tokens: 1,
                prompt_cached_tokens: None,
                completion_tokens: 2,
                finish_reason: Some(BackendFinishReason::Stop),
                progress: None,
            },
        ],
        "prompt",
        true,
    );

    assert_eq!(output.completion_tokens, u64::MAX);
}

fn qwen_xml_test_tool_schema() -> String {
    serde_json::json!([{
        "type": "function",
        "function": {
            "name": "record",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer"},
                    "active": {"type": "boolean"}
                },
                "required": ["path", "limit", "active"]
            }
        }
    }])
    .to_string()
}

#[test]
fn mlx_sse_parser_streams_split_qwen_xml_as_schema_aware_tool_deltas() {
    let schema = qwen_xml_test_tool_schema();
    let first = mlx_text_sse_frame("Before <too", None);
    let second = mlx_text_sse_frame("l_call><function=record><parameter=path>Car", None);
    let third = mlx_text_sse_frame(
        "go.toml</parameter><parameter=limit>3</parameter><parameter=active>true</parameter></function></tool_call> after",
        Some("tool_calls"),
    );
    let done = "data:[DONE]\n\n";
    let chunks = parse_mlx_sse_for_test_with_tool_schema(
        &[&first, &second, &third, done],
        MlxToolMarkup::QwenXml,
        &schema,
    )
    .expect("Qwen XML tool stream parses");

    let text = chunks
        .iter()
        .map(|chunk| chunk.0.as_str())
        .collect::<String>();
    assert_eq!(text, "Before  after");
    assert!(!text.contains("<tool_call>"));

    let deltas = chunks.iter().flat_map(|chunk| &chunk.1).collect::<Vec<_>>();
    assert!(
        deltas.len() >= 4,
        "expected header and argument fragments, got {deltas:#?}"
    );
    let header = deltas
        .iter()
        .find(|delta| delta.id.as_deref() == Some("call_0"))
        .expect("tool header delta");
    assert_eq!(header.call_type, Some(BackendToolCallType::Function));
    assert_eq!(
        header
            .function
            .as_ref()
            .and_then(|function| function.name.as_deref()),
        Some("record")
    );
    let arguments = deltas
        .iter()
        .filter_map(|delta| delta.function.as_ref())
        .filter_map(|function| function.arguments.as_deref())
        .collect::<String>();
    assert_eq!(
        serde_json::from_str::<Value>(&arguments).expect("arguments are JSON"),
        serde_json::json!({"path":"Cargo.toml","limit":3,"active":true})
    );
    assert_eq!(
        chunks.last().and_then(|chunk| chunk.4),
        Some(BackendFinishReason::ToolCalls)
    );
}

#[test]
fn mlx_sse_parser_preserves_xml_like_prose_until_exact_qwen_tool_call() {
    let first = mlx_text_sse_frame("Use <tool_calling> and partial <tool", None);
    let second = mlx_text_sse_frame("_call prose as text", Some("stop"));
    let done = "data:[DONE]\n\n";
    let chunks = parse_mlx_sse_for_test(&[&first, &second, done], MlxToolMarkup::QwenXml)
        .expect("XML-like prose parses as content");

    let text = chunks
        .iter()
        .map(|chunk| chunk.0.as_str())
        .collect::<String>();
    assert_eq!(
        text,
        "Use <tool_calling> and partial <tool_call prose as text"
    );
    assert!(chunks.iter().all(|chunk| chunk.1.is_empty()));
}

#[test]
fn mlx_sse_parser_rejects_truncated_active_qwen_xml_tool_call() {
    let frame = mlx_text_sse_frame("<tool_call><function=record><parameter=path>Cargo", None);
    let done = "data:[DONE]\n\n";
    let err = parse_mlx_sse_for_test(&[&frame, done], MlxToolMarkup::QwenXml)
        .expect_err("truncated active XML tool call fails closed");

    assert!(
        err.to_string().contains("Qwen XML"),
        "error should identify Qwen XML parser: {err}"
    );
}

#[test]
fn mlx_sse_parser_allows_length_truncated_qwen_xml_before_function() {
    let frame = mlx_text_sse_frame("<tool_call>", Some("length"));
    let done = "data:[DONE]\n\n";
    let chunks = parse_mlx_sse_for_test(&[&frame, done], MlxToolMarkup::QwenXml)
        .expect("length-truncated XML start degrades without backend failure");

    assert!(chunks.iter().all(|chunk| chunk.1.is_empty()));
    assert_eq!(
        chunks.last().and_then(|chunk| chunk.4),
        Some(BackendFinishReason::Length)
    );
}

#[test]
fn mlx_sse_parser_finishes_length_truncated_qwen_xml_string_parameter() {
    let frame = mlx_text_sse_frame(
        "<tool_call><function=record><parameter=path>Cargo",
        Some("length"),
    );
    let done = "data:[DONE]\n\n";
    let chunks = parse_mlx_sse_for_test(&[&frame, done], MlxToolMarkup::QwenXml)
        .expect("length-truncated XML parameter produces best-effort deltas");

    let deltas = chunks.iter().flat_map(|chunk| &chunk.1).collect::<Vec<_>>();
    assert_eq!(
        deltas
            .iter()
            .filter_map(|delta| delta.function.as_ref())
            .filter_map(|function| function.name.as_deref())
            .collect::<Vec<_>>(),
        ["record"]
    );
    assert_eq!(
        deltas
            .iter()
            .filter_map(|delta| delta.function.as_ref())
            .filter_map(|function| function.arguments.as_deref())
            .collect::<String>(),
        r#"{"path":"Cargo"}"#
    );
    assert_eq!(
        chunks.last().and_then(|chunk| chunk.4),
        Some(BackendFinishReason::Length)
    );
}

#[test]
fn mlx_sse_parser_handles_adjacent_qwen_xml_tool_calls() {
    let frame = mlx_text_sse_frame(
        "<tool_call><function=first></function></tool_call><tool_call><function=second><parameter=path>src/lib.rs</parameter></function></tool_call>",
        Some("tool_calls"),
    );
    let done = "data:[DONE]\n\n";
    let chunks = parse_mlx_sse_for_test(&[&frame, done], MlxToolMarkup::QwenXml)
        .expect("adjacent Qwen XML calls parse");
    let deltas = chunks.iter().flat_map(|chunk| &chunk.1).collect::<Vec<_>>();

    let names = deltas
        .iter()
        .filter_map(|delta| delta.function.as_ref())
        .filter_map(|function| function.name.as_deref())
        .collect::<Vec<_>>();
    assert_eq!(names, ["first", "second"]);
    assert_eq!(deltas[0].index, 0);
    assert_eq!(deltas[0].id.as_deref(), Some("call_0"));
    assert!(deltas.iter().any(|delta| delta.index == 1));

    let first_arguments = deltas
        .iter()
        .filter(|delta| delta.index == 0)
        .filter_map(|delta| delta.function.as_ref())
        .filter_map(|function| function.arguments.as_deref())
        .collect::<String>();
    let second_arguments = deltas
        .iter()
        .filter(|delta| delta.index == 1)
        .filter_map(|delta| delta.function.as_ref())
        .filter_map(|function| function.arguments.as_deref())
        .collect::<String>();
    assert_eq!(
        serde_json::from_str::<Value>(&first_arguments).expect("first arguments JSON"),
        serde_json::json!({})
    );
    assert_eq!(
        serde_json::from_str::<Value>(&second_arguments).expect("second arguments JSON"),
        serde_json::json!({"path":"src/lib.rs"})
    );
}

#[test]
fn mlx_sse_parser_ignores_qwen_reasoning_markers_around_xml_tool_calls() {
    let frame = mlx_text_sse_frame(
        "<think>Need a tool; mention <tool_call> only as text.</think><tool_call><think>Pick the reader.</think>\n<function=record><parameter=path>Cargo.toml</parameter></function></tool_call>",
        Some("tool_calls"),
    );
    let done = "data:[DONE]\n\n";
    let chunks = parse_mlx_sse_for_test(&[&frame, done], MlxToolMarkup::QwenXml)
        .expect("Qwen XML tool call with reasoning parses");

    let text = chunks
        .iter()
        .map(|chunk| chunk.0.as_str())
        .collect::<String>();
    assert!(text.contains("<think>Need a tool"));
    assert!(
        !text.contains("<function=record>"),
        "XML tool markup should not leak as content: {text}"
    );

    let deltas = chunks.iter().flat_map(|chunk| &chunk.1).collect::<Vec<_>>();
    assert_eq!(
        deltas
            .iter()
            .filter_map(|delta| delta.function.as_ref())
            .filter_map(|function| function.name.as_deref())
            .collect::<Vec<_>>(),
        ["record"]
    );
    assert_eq!(
        deltas
            .iter()
            .filter_map(|delta| delta.function.as_ref())
            .filter_map(|function| function.arguments.as_deref())
            .collect::<String>(),
        r#"{"path":"Cargo.toml"}"#
    );
}

#[test]
fn mlx_non_streaming_qwen_xml_converts_to_canonical_tool_markup() {
    let schema = qwen_xml_test_tool_schema();
    let body = serde_json::json!({
        "choices": [{
            "text": "<tool_call><function=record><parameter=path>Cargo.toml</parameter><parameter=limit>3</parameter><parameter=active>true</parameter></function></tool_call>",
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 4, "completion_tokens": 5}
    })
    .to_string();
    let (output, _chunk_count) = parse_mlx_completion_body(
        &body,
        "hello mlx",
        MLX_QWEN_CONTROL_STOP_TOKENS,
        MlxToolMarkup::QwenXml,
        Some(&schema),
    )
    .expect("non-streaming XML parses");

    assert_eq!(output.finish_reason, BackendFinishReason::ToolCalls);
    assert!(output.text.starts_with("<tool_call>"));
    assert!(output.text.ends_with("</tool_call>"));
    assert!(output.text.contains("\"name\":\"record\""));
    assert!(output.text.contains("\"path\":\"Cargo.toml\""));
    assert!(output.text.contains("\"limit\":3"));
    assert!(output.text.contains("\"active\":true"));
}

#[test]
fn mlx_non_streaming_qwen_xml_structured_tool_calls_bypass_xml_reparse() {
    let body = serde_json::json!({
        "choices": [{
            "message": {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_read_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": "{\"path\":\"Cargo.toml\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 3}
        }
    })
    .to_string();
    let (output, _chunk_count) = parse_mlx_completion_body(
        &body,
        "read a file",
        MLX_QWEN_CONTROL_STOP_TOKENS,
        MlxToolMarkup::QwenXml,
        Some(&qwen_xml_test_tool_schema()),
    )
    .expect("structured MLX tool calls render without Qwen XML reparse");

    assert_eq!(output.finish_reason, BackendFinishReason::ToolCalls);
    assert!(output.text.starts_with("<tool_call>"));
    assert!(output.text.ends_with("</tool_call>"));
    assert!(
        !output.text.contains("<function="),
        "raw Qwen XML should not leak into canonical tool markup: {}",
        output.text
    );
    let payload = output
        .text
        .strip_prefix("<tool_call>")
        .and_then(|text| text.strip_suffix("</tool_call>"))
        .expect("canonical tool call wrapper");
    assert_eq!(
        serde_json::from_str::<Value>(payload).expect("canonical tool call JSON"),
        serde_json::json!({"name": "read_file", "arguments": {"path": "Cargo.toml"}})
    );
    assert_eq!(output.prompt_cached_tokens, Some(3));
    assert_eq!(output.completion_tokens, 5);
}

#[test]
fn mlx_non_streaming_qwen_xml_allows_length_truncated_raw_tool_xml() {
    for raw_text in [
        "<tool_call>",
        "<tool_call><function=record><parameter=path>Cargo",
    ] {
        let body = serde_json::json!({
            "choices": [{
                "text": raw_text,
                "finish_reason": "length"
            }],
            "usage": {"prompt_tokens": 4, "completion_tokens": 5}
        })
        .to_string();
        let (output, _chunk_count) = parse_mlx_completion_body(
            &body,
            "hello mlx",
            MLX_QWEN_CONTROL_STOP_TOKENS,
            MlxToolMarkup::QwenXml,
            Some(&qwen_xml_test_tool_schema()),
        )
        .expect("length-truncated raw Qwen XML degrades without backend failure");

        assert_eq!(output.finish_reason, BackendFinishReason::Length);
        assert!(
            !output.text.contains("<function=record>"),
            "active raw XML should not leak when length-truncated: {}",
            output.text
        );
    }
}

#[test]
fn mlx_non_streaming_qwen_xml_rejects_non_length_truncated_raw_tool_xml() {
    let body = serde_json::json!({
        "choices": [{
            "text": "<tool_call><function=record><parameter=path>Cargo",
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 4, "completion_tokens": 5}
    })
    .to_string();
    let err = parse_mlx_completion_body(
        &body,
        "hello mlx",
        MLX_QWEN_CONTROL_STOP_TOKENS,
        MlxToolMarkup::QwenXml,
        Some(&qwen_xml_test_tool_schema()),
    )
    .expect_err("non-length incomplete raw Qwen XML must fail closed");

    assert!(
        err.to_string().contains("Qwen XML"),
        "error should identify Qwen XML parser: {err}"
    );
}

#[test]
fn mlx_tool_parser_auto_detects_qwen36_and_keeps_older_qwen_json() {
    let mut qwen36 = BackendModelMetadata::new("local-qwen36", "mlx").with_family("qwen");
    qwen36.repo_id = Some("mlx-community/Qwen3.6-35B-A3B-4bit".to_owned());
    assert_eq!(
        mlx_tool_markup_for_metadata(&qwen36, None, MlxToolParserMode::Auto)
            .expect("auto resolves"),
        MlxToolMarkup::QwenXml
    );

    let mut qwen35 = BackendModelMetadata::new("local-qwen35", "mlx").with_family("qwen");
    qwen35.profile = Some("qwen3_5_moe-mlx-4bit".to_owned());
    assert_eq!(
        mlx_tool_markup_for_metadata(&qwen35, None, MlxToolParserMode::Auto)
            .expect("auto resolves"),
        MlxToolMarkup::QwenXml
    );

    let snapshot = BackendModelMetadata::new("local-qwen", "mlx").with_family("qwen");
    let snapshot_path = std::path::Path::new(
        "/models/huggingface/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/abcdef",
    );
    assert_eq!(
        mlx_tool_markup_for_metadata(&snapshot, Some(snapshot_path), MlxToolParserMode::Auto)
            .expect("auto resolves"),
        MlxToolMarkup::QwenXml
    );

    let qwen3 = BackendModelMetadata::new("local-qwen3", "mlx").with_family("qwen");
    assert_eq!(
        mlx_tool_markup_for_metadata(&qwen3, None, MlxToolParserMode::Auto).expect("auto resolves"),
        MlxToolMarkup::Json
    );
}

#[test]
fn mlx_tool_parser_override_controls_or_rejects_qwen_xml() {
    let mut qwen36 = BackendModelMetadata::new("local-qwen36", "mlx").with_family("qwen");
    qwen36.repo_id = Some("mlx-community/Qwen3.6-35B-A3B-4bit".to_owned());
    assert_eq!(
        mlx_tool_markup_for_metadata(&qwen36, None, MlxToolParserMode::Json)
            .expect("json resolves"),
        MlxToolMarkup::Json
    );
    assert_eq!(
        mlx_tool_markup_for_metadata(&qwen36, None, MlxToolParserMode::QwenXml)
            .expect("qwen xml resolves"),
        MlxToolMarkup::QwenXml
    );

    let gemma = BackendModelMetadata::new("local-gemma", "mlx").with_family("gemma");
    let err = mlx_tool_markup_for_metadata(&gemma, None, MlxToolParserMode::QwenXml)
        .expect_err("Gemma cannot use Qwen XML parser");
    assert!(
        err.to_string().contains("qwen-xml"),
        "override rejection should name parser: {err}"
    );
}

#[test]
fn mlx_sse_parser_emits_structured_tool_call_deltas_without_synthetic_markup() {
    let chunks = parse_mlx_sse_for_test(
        &[
            "data:{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_read_1\",\"type\":\"function\",\"function\":{\"name\":\"read_\",\"arguments\":\"{\\\"path\\\"\"}}]},\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":4,\"prompt_tokens_details\":{\"cached_tokens\":2}}}\n\n",
            "data:{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"file\",\"arguments\":\":\\\"Cargo.toml\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"completion_tokens\":5}}\n\n",
            "data:[DONE]\n\n",
        ],
        MlxToolMarkup::Json,
    )
    .expect("structured tool deltas parse");

    let text = chunks
        .iter()
        .map(|chunk| chunk.0.as_str())
        .collect::<String>();
    assert_eq!(text, "");
    let deltas = chunks.iter().flat_map(|chunk| &chunk.1).collect::<Vec<_>>();
    assert_eq!(deltas.len(), 2);
    assert_eq!(deltas[0].index, 0);
    assert_eq!(deltas[0].id.as_deref(), Some("call_read_1"));
    assert_eq!(
        deltas[0]
            .function
            .as_ref()
            .and_then(|function| function.name.as_deref()),
        Some("read_")
    );
    assert_eq!(
        deltas[1]
            .function
            .as_ref()
            .and_then(|function| function.arguments.as_deref()),
        Some(":\"Cargo.toml\"}")
    );
    assert_eq!(
        chunks.last().and_then(|chunk| chunk.4),
        Some(BackendFinishReason::ToolCalls)
    );
}

#[test]
fn mlx_sse_parser_accepts_usage_only_empty_choices_chunk() {
    let chunks = parse_mlx_sse_for_test(
        &[
            "data:{\"choices\":[{\"text\":\"hello\",\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":4}}\n\n",
            "data:{\"choices\":[],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2,\"prompt_tokens_details\":{\"cached_tokens\":3}}}\n\n",
            "data:{\"choices\":[{\"text\":\"\",\"finish_reason\":\"stop\"}]}\n\n",
            "data:[DONE]\n\n",
        ],
        MlxToolMarkup::Json,
    )
    .expect("usage-only empty choices chunk parses");

    assert_eq!(
        chunks
            .iter()
            .map(|chunk| chunk.0.as_str())
            .collect::<String>(),
        "hello"
    );
    assert_eq!(chunks.iter().map(|chunk| chunk.3).sum::<u64>(), 2);
    assert_eq!(
        chunks.last().and_then(|chunk| chunk.4),
        Some(BackendFinishReason::Stop)
    );
}

#[test]
fn mlx_sse_parser_uses_upstream_prompt_tokens_below_whitespace_estimate() {
    let mut parser = MlxSseParser::new_streaming(
        "one two three four five six",
        MLX_QWEN_CONTROL_STOP_TOKENS,
        MlxToolMarkup::Json,
    );
    let chunks = parser
        .push_str(
            "data:{\"choices\":[{\"text\":\"hello\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1}}\n\ndata:[DONE]\n\n",
        )
        .expect("parse upstream usage chunk");
    let final_chunks = parser.finish().expect("finish parser");
    let chunks = chunks.into_iter().chain(final_chunks).collect::<Vec<_>>();

    assert_eq!(
        chunks.last().expect("final chunk is emitted").prompt_tokens,
        2
    );
}

#[test]
fn mlx_sse_parser_leaves_missing_prompt_usage_unreported() {
    let mut parser = MlxSseParser::new_streaming(
        "one two three four",
        MLX_QWEN_CONTROL_STOP_TOKENS,
        MlxToolMarkup::Json,
    );
    let chunks = parser
        .push_str(
            "data:{\"choices\":[{\"text\":\"hello\",\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":1}}\n\ndata:[DONE]\n\n",
        )
        .expect("parse chunk without prompt usage");
    let final_chunks = parser.finish().expect("finish parser");
    let chunks = chunks.into_iter().chain(final_chunks).collect::<Vec<_>>();

    assert_eq!(
        chunks.last().expect("final chunk is emitted").prompt_tokens,
        0
    );
    assert_eq!(
        chunks
            .last()
            .expect("final chunk is emitted")
            .completion_tokens,
        1
    );
}

#[test]
fn mlx_sse_parser_leaves_missing_completion_usage_unreported() {
    let mut parser = MlxSseParser::new_streaming(
        "one two three four",
        MLX_QWEN_CONTROL_STOP_TOKENS,
        MlxToolMarkup::Json,
    );
    let chunks = parser
        .push_str(
            "data:{\"choices\":[{\"text\":\"hello world\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":4}}\n\ndata:[DONE]\n\n",
        )
        .expect("parse chunk without completion usage");
    let final_chunks = parser.finish().expect("finish parser");
    let chunks = chunks.into_iter().chain(final_chunks).collect::<Vec<_>>();
    let final_chunk = chunks.last().expect("final chunk is emitted");

    assert_eq!(final_chunk.prompt_tokens, 4);
    assert_eq!(final_chunk.completion_tokens, 0);
}

#[test]
fn mlx_completion_body_leaves_missing_usage_unreported() {
    let (output, _) = parse_mlx_completion_body(
        "{\"choices\":[{\"text\":\"hello world\",\"finish_reason\":\"stop\"}]}",
        "one two three four",
        MLX_QWEN_CONTROL_STOP_TOKENS,
        MlxToolMarkup::Json,
        None,
    )
    .expect("completion body parses");

    assert_eq!(output.text, "hello world");
    assert_eq!(output.prompt_tokens, 0);
    assert_eq!(output.completion_tokens, 0);
}

#[test]
fn mlx_completion_body_preserves_upstream_zero_prompt_tokens() {
    let (output, _) = parse_mlx_completion_body(
        "{\"choices\":[{\"text\":\"hello\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":1}}",
        "one two three four",
        MLX_QWEN_CONTROL_STOP_TOKENS,
        MlxToolMarkup::Json,
        None,
    )
    .expect("completion body parses");

    assert_eq!(output.prompt_tokens, 0);
}

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
            Some(llm_backend::BackendToolChoice::RequiredFunction(
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
        Some(llm_backend::BackendToolChoice::RequiredAny),
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
    let client = reqwest::Client::new();
    let endpoint = Url::parse("http://127.0.0.1:54321").expect("valid loopback endpoint");
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

    let (protocol, builder) = super::request::build_upstream_request(
        &client,
        &endpoint,
        "/tmp/local-mlx",
        &metadata,
        &request,
        false,
        false,
    )
    .expect("MLX request building does not parse cache compatibility identity");

    assert_eq!(protocol, MlxUpstreamProtocol::ChatCompletions);
    let request = builder.build().expect("request builder serializes body");
    let body = request
        .body()
        .and_then(reqwest::Body::as_bytes)
        .expect("JSON body is buffered");
    let request: Value = serde_json::from_slice(body).expect("request JSON parses");
    assert!(request.get("tools").is_none());
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
            Some(llm_backend::BackendToolChoice::RequiredFunction(
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
        0
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
            Some(llm_backend::BackendToolChoice::RequiredFunction(
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
    assert_eq!(deltas[0].id.as_deref(), Some("call_0"));
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
        .generate(BackendRequest::raw_completion(
            "other-model",
            "hello",
            Some(1),
            SamplingConfig::Greedy,
        ))
        .await
        .expect_err("model mismatch fails before HTTP");

    assert!(err.is_model_not_found());
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
        .generate(BackendRequest::raw_completion(
            "local-mlx",
            "hello",
            Some(1),
            SamplingConfig::Greedy,
        ))
        .await
        .expect_err("missing DONE fails closed");

    assert!(err.to_string().contains("[DONE]"));
}

#[tokio::test]
#[ignore = "slow wall-clock upstream stall coverage; run the slow timeout lane"]
async fn mlx_slow_backend_per_chunk_timeout_detects_stalled_stream() {
    let server = FakeMlxServer::start_with_stall(
        "data:{\"choices\":[{\"text\":\"one\",\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":2}}\n\n",
        Duration::from_millis(80),
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
                read: Duration::from_millis(40),
            },
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
#[ignore = "slow wall-clock upstream stall coverage; run the slow timeout lane"]
async fn mlx_slow_backend_read_timeout_allows_initial_prefill_silence() {
    let server = FakeMlxServer::start_with_initial_body_delay(
        "data:{\"choices\":[{\"text\":\"late\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":3}}\n\ndata: [DONE]\n\n",
        Duration::from_millis(60),
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
                read: Duration::from_millis(30),
            },
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
            "hello mlx",
            Some(12),
            SamplingConfig::Greedy,
        ))
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("initial prefill silence should not trip read timeout");

    let text = chunks
        .iter()
        .map(|chunk| chunk.text.as_str())
        .collect::<String>();
    assert_eq!(text, "late");
    let metrics = metrics.snapshot();
    assert_eq!(metrics["successful_requests"], 1);
    assert_eq!(metrics["failed_requests"], 0);
    assert_eq!(metrics["stall_failures"], 0);
}

#[tokio::test]
#[ignore = "slow wall-clock upstream stall coverage; run the slow timeout lane"]
async fn mlx_slow_backend_request_timeout_detects_delayed_response_headers() {
    let server = FakeMlxServer::start_with_response_delay(
        "data:{\"choices\":[{\"text\":\"late\",\"finish_reason\":null}]}\n\ndata: [DONE]\n\n",
        Duration::from_millis(60),
    );
    let mut backend = MlxBackend::open_with_options(
        "local-mlx",
        server.snapshot_path(),
        MlxBackendOptions {
            endpoint: server.endpoint(),
            family: Some(ModelFamily::Qwen),
            timeouts: MlxTimeouts {
                connect: Duration::from_secs(5),
                request: Duration::from_millis(30),
                read: Duration::from_secs(5),
            },
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
        .expect_err("delayed response headers produce timeout error");

    assert!(
        err.to_string().contains("timed out"),
        "expected timeout error, got: {err}"
    );
    let metrics = metrics.snapshot();
    assert_eq!(metrics["failed_requests"], 1);
    assert_eq!(metrics["stall_failures"], 1);
}

#[tokio::test]
async fn mlx_backend_health_reports_model_list_http_failure() {
    let server = FakeMlxServer::start_with_status(503, "Service Unavailable", "{}");
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

    let health = backend.health().await;

    assert_eq!(health.status().as_str(), "unavailable");
    assert_eq!(server.received_path(), "/v1/models");
    assert!(
        health
            .reason()
            .expect("unavailable health reports a reason")
            .contains("503"),
        "health reason should include upstream status: {health:?}"
    );
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

    fn start_with_initial_body_delay(response_body: &'static str, delay: Duration) -> Self {
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
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
            );
            let _ = stream.flush();
            thread::sleep(delay);
            let _ = write!(
                stream,
                "{:x}\r\n{}\r\n0\r\n\r\n",
                response_body.len(),
                response_body
            );
            let _ = stream.flush();
        });
        Self {
            endpoint,
            snapshot: tempfile::tempdir().expect("snapshot tempdir"),
            received,
            received_path,
            join: Some(join),
        }
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
                .unwrap_or(0);
            while bytes.len() < header_end + request_content_length {
                let read = stream.read(&mut buffer).expect("read body");
                assert!(read > 0, "client closed before body");
                bytes.extend_from_slice(&buffer[..read]);
            }
            let body = &bytes[header_end..header_end + request_content_length];
            if !body.is_empty() {
                *received_for_thread.lock().expect("received lock") =
                    Some(serde_json::from_slice(body).expect("json request body"));
            }
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
