use super::*;

#[test]
fn request_limits_reject_too_many_chat_messages() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("hello"); MAX_CHAT_MESSAGES + 1],
        ..ChatCompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("chat message count must be capped");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.message().contains("messages"));
}

#[test]
fn request_limits_reject_oversized_chat_message_content() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("x".repeat(MAX_MESSAGE_CONTENT_BYTES + 1))],
        ..ChatCompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("message content bytes must be capped");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.message().contains("messages[0].content"));
}

#[test]
fn request_limits_allow_long_context_chat_message_by_default() {
    let legacy_limit = 1024 * 1024;
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("x".repeat(legacy_limit + 1))],
        ..ChatCompletionRequest::default()
    };

    request
        .validate()
        .expect("default request limits accept long-context chat messages");

    let err = request
        .validate_with_limits(RequestLimits {
            message_content_bytes: legacy_limit,
            ..RequestLimits::default()
        })
        .expect_err("custom lower message limit rejects the same request");
    assert_eq!(err.code(), "invalid_request");
    assert!(err.message().contains("messages[0].content"));
}

#[test]
fn validated_request_wraps_after_successful_limit_validation() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("hello")],
        ..ChatCompletionRequest::default()
    };

    let validated = request
        .into_validated_with_limits(RequestLimits::default())
        .expect("valid request wraps");

    assert_eq!(validated.as_ref().model, "local-qwen36");
    assert_eq!(validated.request_limits(), RequestLimits::default());
    assert_eq!(validated.into_inner().messages.len(), 1);
}

#[test]
fn request_limits_reject_too_many_tools() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("use a tool")],
        tools: vec![
            ToolDefinition::function("lookup", "lookup docs", json!({"type": "object"}));
            MAX_TOOLS + 1
        ],
        ..ChatCompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("declared tool count must be capped");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.message().contains("tools"));
}

#[test]
fn request_limits_reject_oversized_tool_schema() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("use a tool")],
        tools: vec![ToolDefinition::function(
            "lookup",
            "lookup docs",
            json!({
                "type": "object",
                "description": "x".repeat(MAX_TOOL_SCHEMA_BYTES),
            }),
        )],
        ..ChatCompletionRequest::default()
    };

    let err = request.validate().expect_err("tool schemas must be capped");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.message().contains("parameters"));
}

#[test]
fn rejects_tool_schema_parameters_that_are_not_objects() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("use a tool")],
        tools: vec![ToolDefinition::function(
            "lookup",
            "lookup docs",
            json!("not a schema object"),
        )],
        ..ChatCompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("tool schema parameters must be objects");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.message().contains("tools[0].function.parameters"));
}

#[test]
fn rejects_tool_schema_unknown_type_keyword() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("use a tool")],
        tools: vec![ToolDefinition::function(
            "lookup",
            "lookup docs",
            json!({
                "type": "object",
                "properties": {
                    "query": { "type": "str" }
                }
            }),
        )],
        ..ChatCompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("unknown JSON Schema types must fail closed");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.message().contains("properties.query.type"));
    assert!(err.message().contains("str"));
}

#[test]
fn rejects_tool_schema_unknown_type_array_entry() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("use a tool")],
        tools: vec![ToolDefinition::function(
            "lookup",
            "lookup docs",
            json!({
                "type": "object",
                "properties": {
                    "query": { "type": ["string", "str"] }
                }
            }),
        )],
        ..ChatCompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("unknown JSON Schema union types must fail closed");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.message().contains("properties.query.type[1]"));
    assert!(err.message().contains("str"));
}

#[test]
fn rejects_malformed_nested_tool_schema_property() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("use a tool")],
        tools: vec![ToolDefinition::function(
            "lookup",
            "lookup docs",
            json!({
                "type": "object",
                "properties": {
                    "query": "string"
                }
            }),
        )],
        ..ChatCompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("nested property schemas must be schema objects");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.message().contains("properties.query"));
    assert!(err.message().contains("JSON object"));
}

#[test]
fn rejects_malformed_tool_schema_items_keyword() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("use a tool")],
        tools: vec![ToolDefinition::function(
            "lookup",
            "lookup docs",
            json!({
                "type": "array",
                "items": "string"
            }),
        )],
        ..ChatCompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("items schemas must be schema objects");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.message().contains("items"));
    assert!(err.message().contains("JSON object"));
}

#[test]
fn rejects_tool_schema_properties_over_default_depth_limit() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("use a tool")],
        tools: vec![ToolDefinition::function(
            "lookup",
            "lookup docs",
            nested_properties_tool_schema(MAX_TOOL_SCHEMA_DEPTH + 1),
        )],
        ..ChatCompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("deeply nested properties schemas must be capped");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.message().contains("schema depth"));
    assert!(
        err.message()
            .contains(&format!("maximum {MAX_TOOL_SCHEMA_DEPTH}"))
    );
}

#[test]
fn rejects_tool_schema_items_over_default_depth_limit() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("use a tool")],
        tools: vec![ToolDefinition::function(
            "lookup",
            "lookup docs",
            nested_items_tool_schema(MAX_TOOL_SCHEMA_DEPTH + 1),
        )],
        ..ChatCompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("deeply nested items schemas must be capped");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.message().contains("schema depth"));
    assert!(
        err.message()
            .contains(&format!("maximum {MAX_TOOL_SCHEMA_DEPTH}"))
    );
}

#[test]
fn accepts_tool_schema_at_default_depth_limit() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("use a tool")],
        tools: vec![
            ToolDefinition::function(
                "lookup_properties",
                "lookup docs",
                nested_properties_tool_schema(MAX_TOOL_SCHEMA_DEPTH),
            ),
            ToolDefinition::function(
                "lookup_items",
                "lookup docs",
                nested_items_tool_schema(MAX_TOOL_SCHEMA_DEPTH),
            ),
        ],
        ..ChatCompletionRequest::default()
    };

    request
        .validate()
        .expect("schemas at the default depth limit remain valid");
}

#[test]
fn custom_request_limits_reject_lower_tool_schema_depth() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("use a tool")],
        tools: vec![ToolDefinition::function(
            "lookup",
            "lookup docs",
            nested_properties_tool_schema(5),
        )],
        ..ChatCompletionRequest::default()
    };

    request
        .validate()
        .expect("default tool schema depth accepts this schema");

    let err = request
        .validate_with_limits(RequestLimits {
            tool_schema_depth: 4,
            ..RequestLimits::default()
        })
        .expect_err("custom lower schema depth rejects the same request");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.message().contains("schema depth"));
    assert!(err.message().contains("maximum 4"));
}

#[test]
fn rejects_tool_schema_depth_before_serialized_byte_limit() {
    let mut schema = nested_properties_tool_schema(2);
    schema["description"] = json!("x".repeat(MAX_TOOL_SCHEMA_BYTES));
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("use a tool")],
        tools: vec![ToolDefinition::function("lookup", "lookup docs", schema)],
        ..ChatCompletionRequest::default()
    };

    let err = request
        .validate_with_limits(RequestLimits {
            tool_schema_depth: 1,
            ..RequestLimits::default()
        })
        .expect_err("schema depth must be checked before serialized byte size");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.message().contains("schema depth"));
    assert!(err.message().contains("maximum 1"));
}

#[test]
fn accepts_supported_tool_schema_types_and_unions() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("use a tool")],
        tools: vec![ToolDefinition::function(
            "lookup",
            "lookup docs",
            json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "limit": { "type": "integer" },
                    "score": { "type": "number" },
                    "exact": { "type": "boolean" },
                    "deleted_at": { "type": "null" },
                    "tags": {
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "metadata": {
                        "type": ["object", "null"],
                        "properties": {
                            "source": { "type": "string" }
                        }
                    }
                }
            }),
        )],
        ..ChatCompletionRequest::default()
    };

    request
        .validate()
        .expect("supported JSON Schema types remain valid");
}

#[test]
fn rejects_malformed_tool_schema_required_keyword() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("use a tool")],
        tools: vec![ToolDefinition::function(
            "lookup",
            "lookup docs",
            json!({
                "type": "object",
                "required": "query",
            }),
        )],
        ..ChatCompletionRequest::default()
    };

    let err = request.validate().expect_err("required must be an array");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.message().contains("required"));
}

#[test]
fn rejects_malformed_tool_schema_properties_keyword() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("use a tool")],
        tools: vec![ToolDefinition::function(
            "lookup",
            "lookup docs",
            json!({
                "type": "object",
                "properties": ["query"],
            }),
        )],
        ..ChatCompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("properties must be an object");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.message().contains("properties"));
}

#[test]
fn request_limits_reject_oversized_tool_description() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("use a tool")],
        tools: vec![ToolDefinition::function(
            "lookup",
            "x".repeat(MAX_TOOL_DESCRIPTION_BYTES + 1),
            json!({"type": "object"}),
        )],
        ..ChatCompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("tool descriptions must be capped");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.message().contains("description"));
}

#[test]
fn request_limits_reject_oversized_tool_call_arguments() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::assistant_tool_call(
            "call_1",
            "lookup",
            json!({"query": "x".repeat(MAX_TOOL_ARGUMENT_BYTES)}),
        )],
        ..ChatCompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("tool call argument bytes must be capped");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.message().contains("arguments"));
}

#[test]
fn request_limits_reject_too_many_stop_sequences() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("hello")],
        stop: vec!["END".to_owned(); MAX_STOP_SEQUENCES + 1],
        ..ChatCompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("stop sequence count must be capped");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.message().contains("stop"));
}

#[test]
fn request_limits_reject_oversized_stop_sequence() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("hello")],
        stop: vec!["x".repeat(MAX_STOP_SEQUENCE_BYTES + 1)],
        ..ChatCompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("stop sequence bytes must be capped");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.message().contains("stop[0]"));
}

#[test]
fn request_limits_reject_oversized_completion_prompt() {
    let request = CompletionRequest {
        model: "local-qwen36".to_owned(),
        prompt: "x".repeat(MAX_COMPLETION_PROMPT_BYTES + 1),
        ..CompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("completion prompt bytes must be capped");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.message().contains("prompt"));
}

fn nested_properties_tool_schema(depth: usize) -> serde_json::Value {
    let mut schema = json!({ "type": "string" });
    for index in (0..depth).rev() {
        schema = json!({
            "type": "object",
            "properties": {
                format!("level_{index}"): schema,
            },
        });
    }
    schema
}

fn nested_items_tool_schema(depth: usize) -> serde_json::Value {
    let mut schema = json!({ "type": "string" });
    for _ in 0..depth {
        schema = json!({
            "type": "array",
            "items": schema,
        });
    }
    schema
}

#[test]
fn request_limits_allow_long_context_completion_prompt_by_default() {
    let legacy_limit = 1024 * 1024;
    let request = CompletionRequest {
        model: "local-qwen36".to_owned(),
        prompt: "x".repeat(legacy_limit + 1),
        ..CompletionRequest::default()
    };

    request
        .validate()
        .expect("default request limits accept long-context completion prompts");

    let err = request
        .validate_with_limits(RequestLimits {
            completion_prompt_bytes: legacy_limit,
            ..RequestLimits::default()
        })
        .expect_err("custom lower completion prompt limit rejects the same request");
    assert_eq!(err.code(), "invalid_request");
    assert!(err.message().contains("prompt"));
}
