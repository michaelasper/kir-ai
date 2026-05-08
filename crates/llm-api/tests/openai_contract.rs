use llm_api::{
    ChatCompletionDelta, ChatCompletionRequest, ChatCompletionStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, ChatRole, CompletionRequest, CompletionResponse,
    FinishReason, ResponseFormat, ToolChoice, ValidateRequest,
};
use serde_json::json;

#[test]
fn validates_required_tool_choice_against_declared_tools() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "local-qwen36",
        "messages": [{"role": "user", "content": "call the calculator"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "evaluate arithmetic",
                "parameters": {
                    "type": "object",
                    "properties": {"expr": {"type": "string"}},
                    "required": ["expr"]
                }
            }
        }],
        "tool_choice": {
            "type": "function",
            "function": {"name": "calculator"}
        }
    }))
    .expect("request json should parse");

    request.validate().expect("declared required tool is valid");
}

#[test]
fn rejects_required_tool_choice_for_missing_tool() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "local-qwen36",
        "messages": [{"role": "user", "content": "call the calculator"}],
        "tools": [],
        "tool_choice": {
            "type": "function",
            "function": {"name": "calculator"}
        }
    }))
    .expect("request json should parse");

    let err = request
        .validate()
        .expect_err("missing tool must fail closed");
    assert_eq!(err.code(), "unsupported_capability");
}

#[test]
fn rejects_json_schema_when_object_mode_is_required() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("return json")],
        response_format: Some(ResponseFormat::JsonSchema {
            json_schema: json!({"name": "answer"}),
        }),
        ..ChatCompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("json_schema is not object mode");
    assert_eq!(err.code(), "unsupported_capability");
}

#[test]
fn streaming_finish_reason_serializes_as_openai_string() {
    let value = serde_json::to_value(FinishReason::ToolCalls).expect("finish reason serializes");
    assert_eq!(value, json!("tool_calls"));
}

#[test]
fn chat_completion_stream_chunk_serializes_as_openai_delta() {
    let chunk = ChatCompletionStreamResponse {
        id: "chatcmpl-test".to_owned(),
        object: "chat.completion.chunk".to_owned(),
        created: 1,
        model: "local-qwen36".to_owned(),
        choices: vec![ChatCompletionStreamChoice {
            index: 0,
            delta: ChatCompletionDelta {
                role: Some(ChatRole::Assistant),
                content: Some("hello".to_owned()),
                tool_calls: Vec::new(),
            },
            finish_reason: None,
        }],
    };

    let value = serde_json::to_value(chunk).expect("chunk serializes");

    assert_eq!(value["object"], "chat.completion.chunk");
    assert_eq!(value["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(value["choices"][0]["delta"]["content"], "hello");
    assert!(value["choices"][0]["finish_reason"].is_null());
}

#[test]
fn auto_tool_choice_is_distinct_from_none() {
    assert_ne!(ToolChoice::Auto, ToolChoice::None);
}

#[test]
fn chat_completion_stop_accepts_string_or_array() {
    let single: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "local-qwen36",
        "messages": [{"role": "user", "content": "hello"}],
        "stop": "END"
    }))
    .expect("single stop parses");
    assert_eq!(single.stop, vec!["END"]);

    let multiple: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "local-qwen36",
        "messages": [{"role": "user", "content": "hello"}],
        "stop": ["END", "<|im_end|>"]
    }))
    .expect("array stop parses");
    assert_eq!(multiple.stop, vec!["END", "<|im_end|>"]);
}

#[test]
fn text_completion_response_serializes_openai_shape() {
    let response = CompletionResponse {
        id: "cmpl-test".to_owned(),
        object: "text_completion".to_owned(),
        created: 1,
        model: "local-qwen36".to_owned(),
        choices: vec![llm_api::CompletionChoice {
            text: "hello".to_owned(),
            index: 0,
            finish_reason: Some(FinishReason::Stop),
        }],
        usage: llm_api::Usage {
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
        },
    };

    let value = serde_json::to_value(response).expect("response serializes");

    assert_eq!(value["object"], "text_completion");
    assert_eq!(value["choices"][0]["text"], "hello");
    assert_eq!(value["choices"][0]["finish_reason"], "stop");
}

#[test]
fn completion_stop_accepts_string_or_array() {
    let single: CompletionRequest = serde_json::from_value(json!({
        "model": "local-qwen36",
        "prompt": "hello",
        "stop": "END"
    }))
    .expect("single stop parses");
    assert_eq!(single.stop, vec!["END"]);

    let multiple: CompletionRequest = serde_json::from_value(json!({
        "model": "local-qwen36",
        "prompt": "hello",
        "stop": ["END", "<|im_end|>"]
    }))
    .expect("array stop parses");
    assert_eq!(multiple.stop, vec!["END", "<|im_end|>"]);
}
