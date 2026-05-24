use super::*;

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
        .find(|delta| delta.id.is_some())
        .expect("tool header delta");
    assert_generated_tool_call_id_is_opaque(header.id.as_deref().expect("tool header id"));
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
    assert_generated_tool_call_id_is_opaque(deltas[0].id.as_deref().expect("generated id"));
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
