use llm_api::{
    ChatCompletionRequest, ChatMessage, CompletionRequest, FinishReason, ResponseFormat,
    ToolChoice, ToolDefinition,
};
use llm_backend::DeterministicBackend;
use llm_runtime::{NoProgressClass, Runtime, RuntimeError};
use serde_json::json;

#[tokio::test]
async fn runtime_returns_non_streaming_chat_completion() {
    let backend = DeterministicBackend::new("local-qwen36", "hello from rust");
    let runtime = Runtime::new(backend);
    let response = runtime
        .chat(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("say hi")],
            max_tokens: Some(16),
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("runtime chat succeeds");

    assert_eq!(response.model, "local-qwen36");
    assert_eq!(response.choices[0].finish_reason, Some(FinishReason::Stop));
    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("hello from rust")
    );
    assert_eq!(response.usage.total_tokens, 5);
}

#[tokio::test]
async fn runtime_returns_text_completion() {
    let backend = DeterministicBackend::new("local-qwen36", "hello from completion END ignored");
    let runtime = Runtime::new(backend);
    let response = runtime
        .completion(CompletionRequest {
            model: "local-qwen36".to_owned(),
            prompt: "say hi".to_owned(),
            max_tokens: Some(8),
            stop: vec![" END".to_owned()],
            stream: false,
            stream_options: llm_api::StreamOptions::default(),
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            logprobs: None,
            n: None,
        })
        .await
        .expect("completion succeeds");

    assert_eq!(response.object, "text_completion");
    assert_eq!(response.choices[0].text, "hello from completion");
    assert_eq!(response.choices[0].finish_reason, Some(FinishReason::Stop));
    assert_eq!(response.usage.total_tokens, 7);
}

#[tokio::test]
async fn runtime_returns_streaming_text_completion_chunks() {
    let backend = DeterministicBackend::new("local-qwen36", "hello from completion END ignored");
    let runtime = Runtime::new(backend);
    let stream = runtime
        .completion_stream(CompletionRequest {
            model: "local-qwen36".to_owned(),
            prompt: "say hi".to_owned(),
            max_tokens: Some(8),
            stop: vec![" END".to_owned()],
            stream: true,
            stream_options: llm_api::StreamOptions::default(),
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            logprobs: None,
            n: None,
        })
        .await
        .expect("completion stream succeeds");

    assert_eq!(stream.chunks.len(), 2);
    assert_eq!(stream.chunks[0].choices[0].text, "hello from completion");
    assert_eq!(stream.chunks[0].choices[0].finish_reason, None);
    assert_eq!(stream.chunks[1].choices[0].text, "");
    assert_eq!(
        stream.chunks[1].choices[0].finish_reason,
        Some(FinishReason::Stop)
    );
}

#[tokio::test]
async fn runtime_appends_text_completion_stream_usage_when_requested() {
    let backend = DeterministicBackend::new("local-qwen36", "hello");
    let runtime = Runtime::new(backend);
    let stream = runtime
        .completion_stream(CompletionRequest {
            model: "local-qwen36".to_owned(),
            prompt: "say hi".to_owned(),
            stream: true,
            stream_options: llm_api::StreamOptions {
                include_usage: true,
            },
            ..CompletionRequest::default()
        })
        .await
        .expect("completion stream succeeds");

    let usage_chunk = stream.chunks.last().expect("usage chunk");
    assert!(usage_chunk.choices.is_empty());
    assert_eq!(usage_chunk.usage.as_ref().expect("usage").total_tokens, 3);
}

#[tokio::test]
async fn optional_tools_allow_text_completion() {
    let backend = DeterministicBackend::new("local-qwen36", "plain text");
    let runtime = Runtime::new(backend);
    let response = runtime
        .chat(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("say hi")],
            tools: vec![ToolDefinition::function("lookup", "lookup", json!({}))],
            tool_choice: Some(ToolChoice::Auto),
            max_tokens: Some(16),
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("optional tools do not require tool calls");

    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("plain text")
    );
    assert!(response.choices[0].message.tool_calls.is_empty());
}

#[tokio::test]
async fn required_tool_choice_rejects_text_fallback() {
    let backend = DeterministicBackend::new("local-qwen36", "plain text");
    let runtime = Runtime::new(backend);
    let err = runtime
        .chat(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("say hi")],
            tools: vec![ToolDefinition::function("lookup", "lookup", json!({}))],
            tool_choice: Some(ToolChoice::Required),
            max_tokens: Some(16),
            ..ChatCompletionRequest::default()
        })
        .await
        .expect_err("required tool choice rejects text fallback");

    assert!(matches!(
        err,
        RuntimeError::NoProgress(NoProgressClass::TextFallbackRequiredTool)
    ));
}

#[tokio::test]
async fn json_object_response_format_accepts_object_content() {
    let backend = DeterministicBackend::new("local-qwen36", r#"{"answer":"rust"}"#);
    let runtime = Runtime::new(backend);
    let response = runtime
        .chat(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("return json")],
            response_format: Some(ResponseFormat::JsonObject),
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("json object content is valid");

    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some(r#"{"answer":"rust"}"#)
    );
}

#[tokio::test]
async fn runtime_truncates_content_at_stop_sequence() {
    let backend = DeterministicBackend::new("local-qwen36", "hello END trailing");
    let runtime = Runtime::new(backend);
    let response = runtime
        .chat(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("say hi")],
            stop: vec![" END".to_owned()],
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("runtime chat succeeds");

    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("hello")
    );
    assert_eq!(response.choices[0].finish_reason, Some(FinishReason::Stop));
}

#[tokio::test]
async fn json_object_response_format_rejects_text_content() {
    let backend = DeterministicBackend::new("local-qwen36", "not json");
    let runtime = Runtime::new(backend);
    let err = runtime
        .chat(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("return json")],
            response_format: Some(ResponseFormat::JsonObject),
            ..ChatCompletionRequest::default()
        })
        .await
        .expect_err("json object mode validates assistant content");

    assert!(matches!(err, RuntimeError::JsonMode(_)));
}

#[tokio::test]
async fn streaming_json_object_response_format_rejects_text_content() {
    let backend = DeterministicBackend::new("local-qwen36", "not json");
    let runtime = Runtime::new(backend);
    let err = runtime
        .chat_stream(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("return json")],
            response_format: Some(ResponseFormat::JsonObject),
            stream: true,
            ..ChatCompletionRequest::default()
        })
        .await
        .expect_err("streaming json object mode validates before SSE assembly");

    assert!(matches!(err, RuntimeError::JsonMode(_)));
}

#[tokio::test]
async fn parses_generated_tool_calls_into_openai_message() {
    let backend = DeterministicBackend::new(
        "local-qwen36",
        r#"<tool_call>{"name":"lookup","arguments":{"query":"rust"}}</tool_call>"#,
    );
    let runtime = Runtime::new(backend);
    let response = runtime
        .chat(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("lookup rust")],
            tools: vec![ToolDefinition::function("lookup", "lookup", json!({}))],
            tool_choice: Some(ToolChoice::Required),
            max_tokens: Some(16),
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("tool call parses");

    let choice = &response.choices[0];
    assert_eq!(choice.finish_reason, Some(FinishReason::ToolCalls));
    assert_eq!(choice.message.tool_calls[0].function.name, "lookup");
    assert_eq!(
        choice.message.tool_calls[0].function.arguments,
        json!({"query": "rust"})
    );
}

#[tokio::test]
async fn runtime_returns_text_stream_chunks() {
    let backend = DeterministicBackend::new("local-qwen36", "hello");
    let runtime = Runtime::new(backend);
    let stream = runtime
        .chat_stream(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("say hi")],
            stream: true,
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("streaming text succeeds");

    assert_eq!(stream.chunks.len(), 3);
    assert_eq!(stream.chunks[0].object, "chat.completion.chunk");
    assert_eq!(
        stream.chunks[0].choices[0].delta.role,
        Some(llm_api::ChatRole::Assistant)
    );
    assert_eq!(
        stream.chunks[1].choices[0].delta.content.as_deref(),
        Some("hello")
    );
    assert_eq!(
        stream.chunks[2].choices[0].finish_reason,
        Some(FinishReason::Stop)
    );
}

#[tokio::test]
async fn runtime_appends_chat_stream_usage_when_requested() {
    let backend = DeterministicBackend::new("local-qwen36", "hello");
    let runtime = Runtime::new(backend);
    let stream = runtime
        .chat_stream(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("say hi")],
            stream: true,
            stream_options: llm_api::StreamOptions {
                include_usage: true,
            },
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("streaming text succeeds");

    let usage_chunk = stream.chunks.last().expect("usage chunk");
    assert!(usage_chunk.choices.is_empty());
    assert_eq!(usage_chunk.usage.as_ref().expect("usage").total_tokens, 3);
}

#[tokio::test]
async fn runtime_streams_generated_tool_call_delta() {
    let backend = DeterministicBackend::new(
        "local-qwen36",
        r#"<tool_call>{"name":"lookup","arguments":{"query":"rust"}}</tool_call>"#,
    );
    let runtime = Runtime::new(backend);
    let stream = runtime
        .chat_stream(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("lookup rust")],
            tools: vec![ToolDefinition::function("lookup", "lookup", json!({}))],
            tool_choice: Some(ToolChoice::Required),
            stream: true,
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("streaming tool calls assemble");

    assert_eq!(stream.chunks.len(), 3);
    let delta = &stream.chunks[1].choices[0].delta.tool_calls[0];
    assert_eq!(delta.index, 0);
    assert_eq!(delta.id.as_deref(), Some("call_0"));
    assert_eq!(
        delta
            .function
            .as_ref()
            .and_then(|function| function.name.as_deref()),
        Some("lookup")
    );
    assert_eq!(
        delta
            .function
            .as_ref()
            .and_then(|function| function.arguments.as_deref()),
        Some(r#"{"query":"rust"}"#)
    );
    assert_eq!(
        stream.chunks[2].choices[0].finish_reason,
        Some(FinishReason::ToolCalls)
    );
}

#[test]
fn classifies_high_output_empty_completion_as_no_progress() {
    let class = llm_runtime::classify_no_progress("", 4096, false);
    assert_eq!(class, Some(NoProgressClass::EmptyHighOutputCompletion));
}

#[test]
fn content_delta_is_progress_even_with_many_tokens() {
    let class = llm_runtime::classify_no_progress("patched Cargo.toml", 4096, false);
    assert_eq!(class, None);
}
