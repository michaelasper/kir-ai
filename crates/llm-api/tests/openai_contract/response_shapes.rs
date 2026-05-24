use super::*;

#[test]
fn streaming_finish_reason_serializes_as_openai_string() {
    let value = serde_json::to_value(FinishReason::ToolCalls).expect("finish reason serializes");
    assert_eq!(value, json!("tool_calls"));
}

#[test]
fn chat_completion_stream_chunk_serializes_as_openai_delta() {
    let chunk = ChatCompletionStreamResponse {
        id: Arc::from("chatcmpl-test"),
        object: "chat.completion.chunk".to_owned(),
        created: 1,
        model: Arc::from("local-qwen36"),
        choices: vec![ChatCompletionStreamChoice {
            index: 0,
            delta: ChatCompletionDelta {
                role: Some(ChatRole::Assistant),
                content: Some("hello".to_owned()),
                tool_calls: Vec::new(),
            },
            finish_reason: None,
        }],
        usage: None,
    };

    let value = serde_json::to_value(chunk).expect("chunk serializes");

    assert_eq!(value["id"], "chatcmpl-test");
    assert_eq!(value["object"], "chat.completion.chunk");
    assert_eq!(value["model"], "local-qwen36");
    assert_eq!(value["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(value["choices"][0]["delta"]["content"], "hello");
    assert!(value["choices"][0]["finish_reason"].is_null());

    let parsed: ChatCompletionStreamResponse =
        serde_json::from_value(value).expect("chunk deserializes");
    assert_eq!(parsed.id.as_ref(), "chatcmpl-test");
    assert_eq!(parsed.model.as_ref(), "local-qwen36");
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
            prompt_tokens_details: None,
        },
    };

    let value = serde_json::to_value(response).expect("response serializes");

    assert_eq!(value["object"], "text_completion");
    assert_eq!(value["choices"][0]["text"], "hello");
    assert_eq!(value["choices"][0]["finish_reason"], "stop");
}

#[test]
fn usage_serializes_cached_prompt_token_details_when_present() {
    let usage = llm_api::Usage {
        prompt_tokens: 10,
        completion_tokens: 2,
        total_tokens: 12,
        prompt_tokens_details: Some(llm_api::PromptTokensDetails { cached_tokens: 7 }),
    };

    let value = serde_json::to_value(usage).expect("usage serializes");

    assert_eq!(value["prompt_tokens"], 10);
    assert_eq!(value["prompt_tokens_details"]["cached_tokens"], 7);
}

#[test]
fn usage_serialization_derives_total_tokens_from_components() {
    let usage = llm_api::Usage {
        prompt_tokens: 10,
        completion_tokens: 2,
        total_tokens: 99,
        prompt_tokens_details: None,
    };

    let value = serde_json::to_value(usage).expect("usage serializes");

    assert_eq!(value["prompt_tokens"], 10);
    assert_eq!(value["completion_tokens"], 2);
    assert_eq!(value["total_tokens"], 12);
}

#[test]
fn usage_deserialization_rejects_inconsistent_total_tokens() {
    let err = serde_json::from_value::<llm_api::Usage>(json!({
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 99
    }))
    .expect_err("usage total must match prompt plus completion tokens");

    assert!(err.to_string().contains("total_tokens"));
}

#[test]
fn usage_omits_cached_prompt_token_details_when_missing_and_deserializes_openai_shape() {
    let compact = llm_api::Usage {
        prompt_tokens: 10,
        completion_tokens: 2,
        total_tokens: 12,
        prompt_tokens_details: None,
    };
    let compact_value = serde_json::to_value(compact).expect("compact usage serializes");
    assert!(compact_value.get("prompt_tokens_details").is_none());

    let parsed: llm_api::Usage = serde_json::from_value(json!({
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
        "prompt_tokens_details": {"cached_tokens": 7}
    }))
    .expect("OpenAI usage shape deserializes");

    assert_eq!(parsed.total_tokens, 12);
    assert_eq!(
        parsed
            .prompt_tokens_details
            .as_ref()
            .map(|details| details.cached_tokens),
        Some(7)
    );
}

#[test]
fn text_completion_stream_response_serializes_without_usage() {
    let response = CompletionStreamResponse {
        id: Arc::from("cmpl-test"),
        object: "text_completion".to_owned(),
        created: 1,
        model: Arc::from("local-qwen36"),
        choices: vec![llm_api::CompletionChoice {
            text: "hello".to_owned(),
            index: 0,
            finish_reason: None,
        }],
        usage: None,
    };

    let value = serde_json::to_value(response).expect("response serializes");

    assert_eq!(value["id"], "cmpl-test");
    assert_eq!(value["object"], "text_completion");
    assert_eq!(value["model"], "local-qwen36");
    assert_eq!(value["choices"][0]["text"], "hello");
    assert!(value.get("usage").is_none());

    let parsed: CompletionStreamResponse =
        serde_json::from_value(value).expect("response deserializes");
    assert_eq!(parsed.id.as_ref(), "cmpl-test");
    assert_eq!(parsed.model.as_ref(), "local-qwen36");
}

#[test]
fn chat_stream_options_include_usage_parses() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "local-qwen36",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": true,
        "stream_options": {"include_usage": true}
    }))
    .expect("stream options parse");

    assert!(request.stream_options.include_usage);
}

#[test]
fn completion_stream_options_include_usage_parses() {
    let request: CompletionRequest = serde_json::from_value(json!({
        "model": "local-qwen36",
        "prompt": "hello",
        "stream": true,
        "stream_options": {"include_usage": true}
    }))
    .expect("stream options parse");

    assert!(request.stream_options.include_usage);
}

#[test]
fn chat_message_content_accepts_text_part_array() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "local-qwen36",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "text", "text": " world"}
            ]
        }]
    }))
    .expect("text content parts deserialize");

    assert_eq!(request.messages[0].content.as_deref(), Some("hello world"));
}

#[test]
fn chat_message_content_separates_adjacent_text_parts() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "local-qwen36",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"}
            ]
        }]
    }))
    .expect("adjacent text content parts deserialize");

    assert_eq!(request.messages[0].content.as_deref(), Some("Hello World"));
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
