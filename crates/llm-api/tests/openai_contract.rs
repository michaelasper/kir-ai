use llm_api::{
    ChatCompletionDelta, ChatCompletionRequest, ChatCompletionStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, ChatRole, CompletionRequest, CompletionResponse,
    CompletionStreamResponse, FinishReason, MAX_CHAT_MESSAGES, MAX_COMPLETION_PROMPT_BYTES,
    MAX_MESSAGE_CONTENT_BYTES, MAX_STOP_SEQUENCE_BYTES, MAX_STOP_SEQUENCES,
    MAX_TOOL_ARGUMENT_BYTES, MAX_TOOL_DESCRIPTION_BYTES, MAX_TOOL_SCHEMA_BYTES, MAX_TOOLS,
    NO_PROGRESS_EXACT_REPEATED_INVALID_TOOL_CALL_THRESHOLD,
    NO_PROGRESS_FUZZY_REPEATED_INVALID_TOOL_CALL_THRESHOLD, RequestLimits, ResponseFormat,
    ToolChoice, ToolDefinition, ValidateRequest, canonical_tool_schema_json,
    canonicalize_tool_schemas,
};
use serde_json::json;

#[test]
fn canonical_tool_schema_json_matches_equivalent_property_and_required_order() {
    let current = vec![ToolDefinition::function(
        "lookup",
        "Lookup docs.",
        json!({
            "type": "object",
            "required": ["query", "source"],
            "properties": {
                "query": {"type": "string"},
                "source": {"type": "string"}
            },
            "additionalProperties": false
        }),
    )];
    let permuted = vec![ToolDefinition::function(
        "lookup",
        "Lookup docs.",
        json!({
            "additionalProperties": false,
            "properties": {
                "source": {"type": "string"},
                "query": {"type": "string"}
            },
            "required": ["source", "query"],
            "type": "object"
        }),
    )];

    assert_ne!(
        serde_json::to_string(&current).expect("current serializes"),
        serde_json::to_string(&permuted).expect("permuted serializes")
    );
    assert_eq!(
        canonical_tool_schema_json(&current).expect("current canonicalizes"),
        canonical_tool_schema_json(&permuted).expect("permuted canonicalizes")
    );

    let canonical = canonicalize_tool_schemas(&permuted);
    assert_eq!(
        canonical[0].function.parameters["required"],
        json!(["query", "source"])
    );
    assert_eq!(
        canonical[0].function.parameters["properties"]
            .as_object()
            .expect("properties object")
            .keys()
            .cloned()
            .collect::<Vec<_>>(),
        ["query", "source"]
    );
}

#[test]
fn canonical_tool_schema_preserves_tool_order_names_and_descriptions() {
    let tools = vec![
        ToolDefinition::function("second", "Second tool.", json!({"type": "object"})),
        ToolDefinition::function("first", "First tool.", json!({"type": "object"})),
    ];

    let canonical = canonicalize_tool_schemas(&tools);

    assert_eq!(canonical[0].function.name, "second");
    assert_eq!(
        canonical[0].function.description.as_deref(),
        Some("Second tool.")
    );
    assert_eq!(canonical[1].function.name, "first");
    assert_eq!(
        canonical[1].function.description.as_deref(),
        Some("First tool.")
    );
}

#[test]
fn canonical_tool_schema_keeps_semantic_differences_distinct() {
    let alpha_then_beta = vec![ToolDefinition::function(
        "lookup",
        "Lookup docs.",
        json!({
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["alpha", "beta"]}
            }
        }),
    )];
    let beta_then_alpha = vec![ToolDefinition::function(
        "lookup",
        "Lookup docs.",
        json!({
            "type": "object",
            "properties": {
                "mode": {"enum": ["beta", "alpha"], "type": "string"}
            }
        }),
    )];
    let different_description = vec![ToolDefinition::function(
        "lookup",
        "Lookup other docs.",
        json!({
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["alpha", "beta"]}
            }
        }),
    )];

    assert_ne!(
        canonical_tool_schema_json(&alpha_then_beta).expect("canonical alpha/beta"),
        canonical_tool_schema_json(&beta_then_alpha).expect("canonical beta/alpha")
    );
    assert_ne!(
        canonical_tool_schema_json(&alpha_then_beta).expect("canonical alpha/beta"),
        canonical_tool_schema_json(&different_description).expect("canonical description")
    );
}

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

#[test]
fn rejects_named_tool_choice_for_undeclared_tool() {
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
        .expect_err("undeclared named tool choice must fail closed");
    assert_eq!(err.code(), "invalid_request");
    assert!(err.message().contains("calculator"));
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
fn rejects_required_tool_choice_without_declared_tools() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("use a tool")],
        tool_choice: Some(ToolChoice::Required),
        ..ChatCompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("required tool choice needs tools");

    assert_eq!(err.code(), "invalid_request");
    assert!(err.message().contains("tool_choice required"));
}

#[test]
fn accepts_non_greedy_sampling_controls() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("sample")],
        temperature: Some(0.7),
        top_p: Some(0.9),
        ..ChatCompletionRequest::default()
    };

    request
        .validate()
        .expect("native sampling controls are accepted");
}

#[test]
fn rejects_invalid_sampling_controls() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("sample")],
        temperature: Some(-0.1),
        ..ChatCompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("negative temperature is invalid");
    assert_eq!(err.code(), "invalid_request");

    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("sample")],
        top_p: Some(0.0),
        ..ChatCompletionRequest::default()
    };

    let err = request.validate().expect_err("zero top_p is invalid");
    assert_eq!(err.code(), "invalid_request");
}

#[test]
fn rejects_temperature_above_2() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("sample")],
        temperature: Some(2.5),
        ..ChatCompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("temperature above 2.0 is invalid");
    assert_eq!(err.code(), "invalid_request");
}

#[test]
fn rejects_nan_temperature() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("sample")],
        temperature: Some(f32::NAN),
        ..ChatCompletionRequest::default()
    };

    let err = request.validate().expect_err("NaN temperature is invalid");
    assert_eq!(err.code(), "invalid_request");
}

#[test]
fn rejects_inf_temperature() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("sample")],
        temperature: Some(f32::INFINITY),
        ..ChatCompletionRequest::default()
    };

    let err = request.validate().expect_err("inf temperature is invalid");
    assert_eq!(err.code(), "invalid_request");
}

#[test]
fn rejects_nan_top_p() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("sample")],
        top_p: Some(f32::NAN),
        ..ChatCompletionRequest::default()
    };

    let err = request.validate().expect_err("NaN top_p is invalid");
    assert_eq!(err.code(), "invalid_request");
}

#[test]
fn rejects_inf_top_p() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("sample")],
        top_p: Some(f32::INFINITY),
        ..ChatCompletionRequest::default()
    };

    let err = request.validate().expect_err("inf top_p is invalid");
    assert_eq!(err.code(), "invalid_request");
}

#[test]
fn accepts_explicit_greedy_sampling_controls() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("sample")],
        temperature: Some(0.0),
        top_p: Some(1.0),
        ..ChatCompletionRequest::default()
    };

    request.validate().expect("greedy controls are supported");
}

#[test]
fn rejects_unsupported_chat_penalty_controls() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("sample")],
        presence_penalty: Some(0.5),
        ..ChatCompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("presence penalty is not implemented");
    assert_eq!(err.code(), "unsupported_capability");

    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("sample")],
        frequency_penalty: Some(0.5),
        ..ChatCompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("frequency penalty is not implemented");
    assert_eq!(err.code(), "unsupported_capability");
}

#[test]
fn accepts_neutral_chat_penalty_controls() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("sample")],
        presence_penalty: Some(0.0),
        frequency_penalty: Some(0.0),
        ..ChatCompletionRequest::default()
    };

    request.validate().expect("neutral penalties are no-ops");
}

#[test]
fn rejects_unsupported_chat_logprob_controls() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("sample")],
        logprobs: Some(true),
        ..ChatCompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("logprobs are not implemented");
    assert_eq!(err.code(), "unsupported_capability");

    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("sample")],
        top_logprobs: Some(1),
        ..ChatCompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("top_logprobs are not implemented");
    assert_eq!(err.code(), "unsupported_capability");
}

#[test]
fn accepts_disabled_chat_logprobs() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("sample")],
        logprobs: Some(false),
        ..ChatCompletionRequest::default()
    };

    request.validate().expect("disabled logprobs are a no-op");
}

#[test]
fn rejects_parallel_tool_calls_when_requested() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("call tools")],
        parallel_tool_calls: Some(true),
        ..ChatCompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("parallel tool calls are not implemented");
    assert_eq!(err.code(), "unsupported_capability");
}

#[test]
fn accepts_disabled_parallel_tool_calls() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("call tools")],
        parallel_tool_calls: Some(false),
        ..ChatCompletionRequest::default()
    };

    request
        .validate()
        .expect("disabled parallel tool calls are a no-op");
}

#[test]
fn completion_accepts_non_greedy_sampling_controls() {
    let request = CompletionRequest {
        model: "local-qwen36".to_owned(),
        prompt: "sample".to_owned(),
        temperature: Some(0.7),
        top_p: Some(0.9),
        ..CompletionRequest::default()
    };

    request
        .validate()
        .expect("native sampling controls are accepted");
}

#[test]
fn completion_rejects_invalid_sampling_controls() {
    let request = CompletionRequest {
        model: "local-qwen36".to_owned(),
        prompt: "sample".to_owned(),
        temperature: Some(-0.1),
        ..CompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("negative temperature is invalid");
    assert_eq!(err.code(), "invalid_request");

    let request = CompletionRequest {
        model: "local-qwen36".to_owned(),
        prompt: "sample".to_owned(),
        top_p: Some(0.0),
        ..CompletionRequest::default()
    };

    let err = request.validate().expect_err("zero top_p is invalid");
    assert_eq!(err.code(), "invalid_request");
}

#[test]
fn completion_rejects_temperature_above_2() {
    let request = CompletionRequest {
        model: "local-qwen36".to_owned(),
        prompt: "sample".to_owned(),
        temperature: Some(2.5),
        ..CompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("temperature above 2.0 is invalid");
    assert_eq!(err.code(), "invalid_request");
}

#[test]
fn completion_rejects_nan_temperature() {
    let request = CompletionRequest {
        model: "local-qwen36".to_owned(),
        prompt: "sample".to_owned(),
        temperature: Some(f32::NAN),
        ..CompletionRequest::default()
    };

    let err = request.validate().expect_err("NaN temperature is invalid");
    assert_eq!(err.code(), "invalid_request");
}

#[test]
fn completion_rejects_nan_top_p() {
    let request = CompletionRequest {
        model: "local-qwen36".to_owned(),
        prompt: "sample".to_owned(),
        top_p: Some(f32::NAN),
        ..CompletionRequest::default()
    };

    let err = request.validate().expect_err("NaN top_p is invalid");
    assert_eq!(err.code(), "invalid_request");
}

#[test]
fn completion_rejects_inf_top_p() {
    let request = CompletionRequest {
        model: "local-qwen36".to_owned(),
        prompt: "sample".to_owned(),
        top_p: Some(f32::INFINITY),
        ..CompletionRequest::default()
    };

    let err = request.validate().expect_err("inf top_p is invalid");
    assert_eq!(err.code(), "invalid_request");
}

#[test]
fn completion_accepts_explicit_greedy_sampling_controls() {
    let request = CompletionRequest {
        model: "local-qwen36".to_owned(),
        prompt: "sample".to_owned(),
        temperature: Some(0.0),
        top_p: Some(1.0),
        ..CompletionRequest::default()
    };

    request.validate().expect("greedy controls are supported");
}

#[test]
fn completion_rejects_unsupported_penalty_controls() {
    let request = CompletionRequest {
        model: "local-qwen36".to_owned(),
        prompt: "sample".to_owned(),
        presence_penalty: Some(0.5),
        ..CompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("presence penalty is not implemented");
    assert_eq!(err.code(), "unsupported_capability");

    let request = CompletionRequest {
        model: "local-qwen36".to_owned(),
        prompt: "sample".to_owned(),
        frequency_penalty: Some(0.5),
        ..CompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("frequency penalty is not implemented");
    assert_eq!(err.code(), "unsupported_capability");
}

#[test]
fn completion_rejects_unsupported_logprobs() {
    let request = CompletionRequest {
        model: "local-qwen36".to_owned(),
        prompt: "sample".to_owned(),
        logprobs: Some(0),
        ..CompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("completion logprobs are not implemented");
    assert_eq!(err.code(), "unsupported_capability");
}

#[test]
fn rejects_zero_chat_max_tokens() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("hello")],
        max_tokens: Some(0),
        ..ChatCompletionRequest::default()
    };

    let err = request.validate().expect_err("zero max_tokens is invalid");
    assert_eq!(err.code(), "invalid_request");
}

#[test]
fn chat_accepts_max_completion_tokens_alias() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "local-qwen36",
        "messages": [{"role": "user", "content": "hello"}],
        "max_completion_tokens": 12
    }))
    .expect("request parses");

    request.validate().expect("alias is valid");
    assert_eq!(request.effective_max_tokens(), Some(12));
}

#[test]
fn rejects_conflicting_chat_max_token_fields() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "local-qwen36",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 8,
        "max_completion_tokens": 12
    }))
    .expect("request parses");

    let err = request
        .validate()
        .expect_err("conflicting token limits are invalid");
    assert_eq!(err.code(), "invalid_request");
}

#[test]
fn rejects_zero_chat_max_completion_tokens() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "local-qwen36",
        "messages": [{"role": "user", "content": "hello"}],
        "max_completion_tokens": 0
    }))
    .expect("request parses");

    let err = request
        .validate()
        .expect_err("zero max_completion_tokens is invalid");
    assert_eq!(err.code(), "invalid_request");
}

#[test]
fn rejects_zero_completion_max_tokens() {
    let request = CompletionRequest {
        model: "local-qwen36".to_owned(),
        prompt: "hello".to_owned(),
        max_tokens: Some(0),
        ..CompletionRequest::default()
    };

    let err = request.validate().expect_err("zero max_tokens is invalid");
    assert_eq!(err.code(), "invalid_request");
}

#[test]
fn rejects_unsupported_multiple_chat_choices() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("hello")],
        n: Some(2),
        ..ChatCompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("multiple choices are not implemented");
    assert_eq!(err.code(), "unsupported_capability");
}

#[test]
fn rejects_zero_chat_choices() {
    let request = ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("hello")],
        n: Some(0),
        ..ChatCompletionRequest::default()
    };

    let err = request.validate().expect_err("zero choices is invalid");
    assert_eq!(err.code(), "invalid_request");
}

#[test]
fn rejects_unsupported_multiple_completion_choices() {
    let request = CompletionRequest {
        model: "local-qwen36".to_owned(),
        prompt: "hello".to_owned(),
        n: Some(2),
        ..CompletionRequest::default()
    };

    let err = request
        .validate()
        .expect_err("multiple choices are not implemented");
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
        usage: None,
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
        id: "cmpl-test".to_owned(),
        object: "text_completion".to_owned(),
        created: 1,
        model: "local-qwen36".to_owned(),
        choices: vec![llm_api::CompletionChoice {
            text: "hello".to_owned(),
            index: 0,
            finish_reason: None,
        }],
        usage: None,
    };

    let value = serde_json::to_value(response).expect("response serializes");

    assert_eq!(value["object"], "text_completion");
    assert_eq!(value["choices"][0]["text"], "hello");
    assert!(value.get("usage").is_none());
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
