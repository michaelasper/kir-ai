use llm_models::ModelFamily;
use llm_tool_parser::{GemmaParser, ParsedAssistant, parse_assistant_for_family};

#[test]
fn parses_gemma_plain_text() {
    let parsed = parse_assistant_for_family(ModelFamily::Gemma, "Plain Gemma text answer.")
        .expect("Gemma plain text parses");

    assert_eq!(parsed, ParsedAssistant::content("Plain Gemma text answer."));
}

#[test]
fn parses_gemma_reasoning_channel_and_content() {
    let parsed = GemmaParser
        .parse_complete("<|channel>thought\nNeed inspect.\n<channel|>Use read_file.<turn|>")
        .expect("Gemma reasoning parses");

    assert_eq!(parsed.reasoning.as_deref(), Some("Need inspect."));
    assert_eq!(parsed.content, "Use read_file.");
    assert!(parsed.tool_calls.is_empty());
}

#[test]
fn parses_gemma_tool_call_channel() {
    let parsed = GemmaParser
        .parse_complete(
            "<|channel>thought\nNeed a file read.\n<channel|><|tool_call>call:read_file{path:<|\"|>Cargo.toml<|\"|>}<tool_call|>",
        )
        .expect("Gemma tool call parses");

    assert_eq!(parsed.reasoning.as_deref(), Some("Need a file read."));
    assert_eq!(parsed.content, "");
    assert_eq!(parsed.tool_calls.len(), 1);
    assert_eq!(parsed.tool_calls[0].function.name, "read_file");
    assert_eq!(
        parsed.tool_calls[0].function.arguments["path"],
        "Cargo.toml"
    );
}

#[test]
fn parses_gemma_nested_tool_arguments() {
    let parsed = GemmaParser
        .parse_complete(
            "<|tool_call>call:lookup{query:<|\"|>rust<|\"|>,options:{limit:3,exact:true},tags:[<|\"|>code<|\"|>,<|\"|>docs<|\"|>]}<tool_call|>",
        )
        .expect("Gemma nested tool call parses");

    assert_eq!(parsed.tool_calls[0].function.name, "lookup");
    assert_eq!(
        parsed.tool_calls[0].function.arguments["options"]["limit"],
        3
    );
    assert_eq!(parsed.tool_calls[0].function.arguments["tags"][1], "docs");
}

#[test]
fn rejects_malformed_gemma_tool_call() {
    let err = GemmaParser
        .parse_complete("<|tool_call>call:read_file{path:<|\"|>Cargo.toml<tool_call|>")
        .expect_err("malformed Gemma tool call fails");

    assert_eq!(err.code(), "malformed_tool_call");
}

#[test]
fn rejects_gemma_multimodal_artifact_markers() {
    let err = GemmaParser
        .parse_complete(include_str!("fixtures/gemma/unsupported_multimodal.txt"))
        .expect_err("multimodal artifact marker fails closed");

    assert_eq!(err.code(), "unsupported_multimodal_output");
}
