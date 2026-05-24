use super::*;

#[test]
fn no_progress_threshold_defaults_match_north_star_spec() {
    assert_eq!(NO_PROGRESS_EXACT_REPEATED_INVALID_TOOL_CALL_THRESHOLD, 5);
    assert_eq!(NO_PROGRESS_FUZZY_REPEATED_INVALID_TOOL_CALL_THRESHOLD, 3);
}

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
fn rejects_user_and_system_messages_without_content() {
    for role in ["user", "system"] {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "local-qwen36",
            "messages": [{"role": role}]
        }))
        .expect("request json should parse");

        let err = request
            .validate()
            .expect_err("plain chat messages need content");

        assert_eq!(err.code(), "invalid_request");
        assert!(err.message().contains("messages[0].content"));
    }
}

#[test]
fn rejects_tool_messages_without_tool_call_id() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "local-qwen36",
        "messages": [{"role": "tool", "content": "lookup result"}]
    }))
    .expect("request json should parse");

    let err = request
        .validate()
        .expect_err("tool messages must identify their tool call");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.message().contains("messages[0].tool_call_id"));
}

#[test]
fn rejects_assistant_messages_without_content_or_tool_calls() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "local-qwen36",
        "messages": [{"role": "assistant"}]
    }))
    .expect("request json should parse");

    let err = request
        .validate()
        .expect_err("assistant messages need content or tool calls");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.message().contains("messages[0].content"));
    assert!(err.message().contains("tool_calls"));
}

#[test]
fn rejects_tool_messages_without_content() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "local-qwen36",
        "messages": [{"role": "tool", "tool_call_id": "call_1"}]
    }))
    .expect("request json should parse");

    let err = request.validate().expect_err("tool messages need content");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.message().contains("messages[0].content"));
}

#[test]
fn rejects_tool_calls_on_non_assistant_messages() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "local-qwen36",
        "messages": [{
            "role": "user",
            "content": "hello",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": {"query": "rust"}
                }
            }]
        }]
    }))
    .expect("request json should parse");

    let err = request
        .validate()
        .expect_err("only assistant messages may include tool_calls");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.message().contains("messages[0].tool_calls"));
    assert!(err.message().contains("assistant"));
}

#[test]
fn rejects_tool_call_ids_on_non_tool_messages() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "local-qwen36",
        "messages": [{
            "role": "assistant",
            "content": "done",
            "tool_call_id": "call_1"
        }]
    }))
    .expect("request json should parse");

    let err = request
        .validate()
        .expect_err("only tool messages may include tool_call_id");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.message().contains("messages[0].tool_call_id"));
    assert!(err.message().contains("tool messages"));
}

#[test]
fn rejects_system_messages_after_conversation_messages() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![
            ChatMessage::user("hello"),
            ChatMessage::system("late instruction"),
        ],
        ..ChatCompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("system messages must appear before conversation turns");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.message().contains("messages[1].role"));
    assert!(err.message().contains("system"));
}

#[test]
fn rejects_tool_messages_not_matching_pending_assistant_tool_calls() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![
            ChatMessage::user("lookup rust"),
            ChatMessage::assistant_tool_call("call_1", "lookup", json!({"query": "rust"})),
            ChatMessage::tool("call_2", "Rust result"),
        ],
        ..ChatCompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("tool results must match pending assistant tool call ids");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.message().contains("messages[2].tool_call_id"));
    assert!(err.message().contains("pending"));
}

#[test]
fn validates_tool_result_exchange_with_followup_user_message() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![
            ChatMessage::system("You answer briefly."),
            ChatMessage::user("lookup rust"),
            ChatMessage::assistant_tool_call("call_1", "lookup", json!({"query": "rust"})),
            ChatMessage::tool("call_1", "Rust is a systems programming language."),
            ChatMessage::user("summarize that"),
        ],
        tools: vec![ToolDefinition::function("lookup", "lookup docs", json!({}))],
        ..ChatCompletionRequest::default()
    };

    request
        .validate()
        .expect("complete assistant tool call and tool result exchange is valid");
}

#[test]
fn rejects_empty_declared_tool_function_name() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("use a tool")],
        tools: vec![ToolDefinition::function("", "lookup docs", json!({}))],
        ..ChatCompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("empty tool names are invalid");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.message().contains("tools[0].function.name"));
}

#[test]
fn rejects_empty_assistant_tool_call_function_name() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::assistant_tool_call("call_1", "", json!({}))],
        ..ChatCompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("empty assistant tool call names are invalid");

    assert_eq!(err.code(), "invalid_request");
    assert!(
        err.message()
            .contains("messages[0].tool_calls[0].function.name")
    );
}

#[test]
fn rejects_empty_named_tool_choice_function_name() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "local-qwen36",
        "messages": [{"role": "user", "content": "call a tool"}],
        "tools": [{
            "type": "function",
            "function": {"name": "lookup", "parameters": {}}
        }],
        "tool_choice": {
            "type": "function",
            "function": {"name": ""}
        }
    }))
    .expect("request json should parse");

    let err = request
        .validate()
        .expect_err("empty named tool choice is invalid");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.message().contains("tool_choice.function.name"));
}

#[test]
fn rejects_duplicate_tool_names_for_required_tool_choice() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("use a tool")],
        tools: vec![
            ToolDefinition::function("lookup", "first lookup", json!({})),
            ToolDefinition::function("lookup", "second lookup", json!({})),
        ],
        tool_choice: Some(ToolChoice::Required),
        ..ChatCompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("required tool choice needs unique tool names");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.message().contains("duplicate tool name"));
}

#[test]
fn rejects_duplicate_tool_names_for_named_tool_choice() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "local-qwen36",
        "messages": [{"role": "user", "content": "call lookup"}],
        "tools": [
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {}}
            },
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {}}
            }
        ],
        "tool_choice": {
            "type": "function",
            "function": {"name": "lookup"}
        }
    }))
    .expect("request json should parse");

    let err = request
        .validate()
        .expect_err("named tool choice needs unique tool names");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.message().contains("duplicate tool name"));
}
