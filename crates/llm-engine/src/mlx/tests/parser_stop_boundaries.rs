use super::*;

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
        ("mlx.rs", include_str!("../../mlx.rs")),
        ("mlx/client.rs", include_str!("../client.rs")),
        ("mlx/metadata.rs", include_str!("../metadata.rs")),
        ("mlx/metrics.rs", include_str!("../metrics.rs")),
        ("mlx/protocol.rs", include_str!("../protocol.rs")),
        ("mlx/request.rs", include_str!("../request.rs")),
        ("mlx/sse.rs", include_str!("../sse.rs")),
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

#[test]
fn mlx_control_stop_tokens_do_not_default_future_families_to_qwen() {
    let source = include_str!("../protocol.rs");
    let production_source = source.split("#[cfg(test)]").next().unwrap_or(source);
    let control_lookup = production_source
        .split("pub(super) fn mlx_control_stop_tokens_for_metadata")
        .nth(1)
        .and_then(|source| {
            source
                .split("pub(super) fn mlx_tool_markup_for_metadata")
                .next()
        })
        .expect("MLX control stop-token lookup should be present");

    assert!(
        !control_lookup.contains("Some(_)"),
        "future non-exhaustive model families must not silently use Qwen MLX control stop tokens"
    );
    assert!(
        control_lookup.contains("Ok(Some(family))"),
        "future non-exhaustive model families must route to an explicit unsupported-family error"
    );
}

#[test]
fn mlx_control_stop_tokens_reject_unknown_family_metadata() {
    let metadata = BackendModelMetadata::new("local-glm", "mlx").with_family("glm");
    let err = super::super::protocol::mlx_control_stop_tokens_for_metadata(&metadata)
        .expect_err("unknown family metadata must not use Qwen control stop tokens");

    assert!(
        err.to_string().contains("unsupported model family `glm`"),
        "error should preserve unsupported family detail: {err}"
    );
}
