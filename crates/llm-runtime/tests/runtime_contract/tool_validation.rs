use super::*;

#[tokio::test]
async fn optional_tools_allow_text_completion() {
    let backend = ProtocolTestBackend::new("local-qwen36", "plain text");
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
async fn runtime_preserves_structured_chat_context_when_tools_are_declared() {
    let observed = Arc::new(Mutex::new(None));
    let runtime = Runtime::new(RecordingChatContextBackend {
        observed: observed.clone(),
        family: "gemma",
    });

    runtime
        .chat(ChatCompletionRequest {
            model: "local-gemma4".to_owned(),
            messages: vec![ChatMessage::user("lookup rust")],
            tools: vec![ToolDefinition::function("lookup", "lookup", json!({}))],
            max_tokens: Some(16),
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("Gemma chat with tools succeeds");

    let observed = observed
        .lock()
        .expect("observed request lock")
        .clone()
        .expect("backend request captured");
    let chat_context = &observed.as_chat().expect("chat request kind").chat_context;
    assert_eq!(chat_context.messages.len(), 1);
    assert_eq!(chat_context.messages[0].role, BackendChatRole::User);
    assert_eq!(
        chat_context.messages[0].content.as_deref(),
        Some("lookup rust")
    );
    assert!(
        observed
            .cache_context()
            .tool_schema
            .as_deref()
            .is_some_and(|schema| schema.contains("lookup"))
    );
    assert_eq!(
        chat_context.tools,
        backend_tool_definitions(&[ToolDefinition::function("lookup", "lookup", json!({}))])
    );
}

#[tokio::test]
async fn required_tool_choice_rejects_text_fallback() {
    let backend = ProtocolTestBackend::new("local-qwen36", "plain text");
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
async fn protocol_test_backend_returns_tool_call_for_required_tool_choice() {
    let backend =
        ProtocolTestBackend::new("local-qwen36", "plain text").with_required_tool_protocol();
    let runtime = Runtime::new(backend);
    let response = runtime
        .chat(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("lookup rust")],
            tools: vec![ToolDefinition::function("lookup", "lookup", json!({}))],
            tool_choice: Some(ToolChoice::Required),
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("required tool choice succeeds in protocol test mode");

    assert_eq!(
        response.choices[0].finish_reason,
        Some(FinishReason::ToolCalls)
    );
    assert_eq!(response.choices[0].message.tool_calls.len(), 1);
    assert_eq!(
        response.choices[0].message.tool_calls[0].function.name,
        "lookup"
    );
}

#[tokio::test]
async fn chat_stop_sequence_suppresses_later_tool_calls() {
    let backend = ProtocolTestBackend::new(
        "local-qwen36",
        r#"content STOP <tool_call>{"name":"lookup","arguments":{"query":"rust"}}</tool_call>"#,
    );
    let runtime = Runtime::new(backend);
    let response = runtime
        .chat(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("say hi")],
            tools: vec![ToolDefinition::function("lookup", "lookup", json!({}))],
            stop: vec![" STOP".to_owned()],
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("runtime chat succeeds");

    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("content")
    );
    assert!(response.choices[0].message.tool_calls.is_empty());
    assert_eq!(response.choices[0].finish_reason, Some(FinishReason::Stop));
}

#[tokio::test]
async fn parses_generated_tool_calls_into_openai_message() {
    let backend = ProtocolTestBackend::new(
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
async fn rejects_generated_tool_call_missing_required_schema_argument() {
    let backend = ProtocolTestBackend::new(
        "local-qwen36",
        r#"<tool_call>{"name":"read_file","arguments":{}}</tool_call>"#,
    );
    let runtime = Runtime::new(backend);
    let err = runtime
        .chat(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("read Cargo.toml")],
            tools: vec![ToolDefinition::function(
                "read_file",
                "read file",
                json!({
                    "type": "object",
                    "required": ["path"],
                    "properties": {
                        "path": { "type": "string" }
                    }
                }),
            )],
            tool_choice: Some(ToolChoice::Required),
            ..ChatCompletionRequest::default()
        })
        .await
        .expect_err("missing required tool argument is rejected");

    assert!(matches!(err, RuntimeError::ToolCallValidation(_)));
    assert!(err.to_string().contains("path"));
}

#[tokio::test]
async fn repeated_empty_required_tool_call_fourth_attempt_returns_schema_hint() {
    let err = replay_read_tool_error(
        failed_read_attempts(3, json!({})),
        json!({}),
        "four total empty read attempts should remain schema validation",
    )
    .await;

    let RuntimeError::ToolCallValidation(message) = err else {
        panic!("expected tool-call validation under exact threshold, got {err:?}");
    };
    assert!(message.contains("missing required argument `path`"));
    assert!(message.contains("required arguments: `path`, `_i`"));
    assert!(message.contains("expected arguments object"));
    assert!(message.contains(r#""path":"<string>""#));
}

#[tokio::test]
async fn repeated_empty_required_tool_call_fifth_attempt_returns_no_progress() {
    let err = replay_read_tool_error(
        failed_read_attempts(4, json!({})),
        json!({}),
        "fifth empty read attempt should hit exact no-progress threshold",
    )
    .await;

    assert!(matches!(
        err,
        RuntimeError::NoProgress(NoProgressClass::RepeatedInvalidToolCall)
    ));
}

#[tokio::test]
async fn fuzzy_repeated_invalid_tool_call_third_attempt_returns_no_progress() {
    let err = replay_read_tool_error(
        vec![
            ChatMessage::user("read the first missing file"),
            ChatMessage::assistant_tool_call("call_0", "read", json!({"path": "missing-a.txt"})),
            ChatMessage::tool("call_0", "error: file not found: missing-a.txt"),
            ChatMessage::user("try the second missing file"),
            ChatMessage::assistant_tool_call("call_1", "read", json!({"path": "missing-b.txt"})),
            ChatMessage::tool("call_1", "error: file not found: missing-b.txt"),
        ],
        json!({"path": "missing-c.txt"}),
        "third same-shape failed read attempt should hit fuzzy no-progress threshold",
    )
    .await;

    assert!(matches!(
        err,
        RuntimeError::NoProgress(NoProgressClass::FuzzyRepeatedInvalidToolCall)
    ));
}

#[tokio::test]
async fn repeated_empty_required_tool_call_counts_failed_attempts_across_turns() {
    let messages = vec![
        ChatMessage::system("You are a file-reading assistant."),
        ChatMessage::user("read missing.txt"),
        ChatMessage::assistant_tool_call("call_0", "read", json!({})),
        ChatMessage::tool("call_0", "error: missing path argument"),
        ChatMessage::user("try again"),
        ChatMessage::assistant_tool_call("call_1", "read", json!({})),
        ChatMessage::tool("call_1", "error: missing path argument"),
        ChatMessage::user("try again"),
        ChatMessage::assistant_tool_call("call_2", "read", json!({})),
        ChatMessage::tool("call_2", "error: missing path argument"),
        ChatMessage::user("one more time"),
        ChatMessage::assistant_tool_call("call_3", "read", json!({})),
        ChatMessage::tool("call_3", "error: missing path argument"),
    ];
    let err = replay_read_tool_error(
        messages,
        json!({}),
        "failed attempts should count across separate user turns",
    )
    .await;

    assert!(matches!(
        err,
        RuntimeError::NoProgress(NoProgressClass::RepeatedInvalidToolCall)
    ));
}

#[tokio::test]
async fn repeated_empty_required_tool_call_ignores_successes_and_unmatched_tool_results() {
    let messages = vec![
        ChatMessage::user("read missing.txt"),
        ChatMessage::assistant_tool_call("call_0", "read", json!({})),
        ChatMessage::tool("call_0", "ok"),
        ChatMessage::user("try again"),
        ChatMessage::assistant_tool_call("call_1", "read", json!({})),
        ChatMessage::tool("other_1", "error: missing path argument"),
        ChatMessage::user("try again"),
        ChatMessage::assistant_tool_call("call_2", "read", json!({})),
        ChatMessage::tool("call_2", "read succeeded"),
        ChatMessage::user("try again"),
        ChatMessage::assistant_tool_call("call_3", "read", json!({})),
        ChatMessage::tool("other_3", "error: missing path argument"),
    ];
    let err = replay_read_tool_error(
        messages,
        json!({}),
        "successful and unmatched tool results should not count toward threshold",
    )
    .await;

    assert!(matches!(err, RuntimeError::ToolCallValidation(_)));
}

#[tokio::test]
async fn fills_missing_required_omp_intent_argument() {
    let backend = ProtocolTestBackend::new(
        "local-qwen36",
        r#"<tool_call>{"name":"read","arguments":{"path":"calculator.py"}}</tool_call>"#,
    );
    let runtime = Runtime::new(backend);
    let response = runtime
        .chat(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("read calculator.py")],
            tools: vec![ToolDefinition::function(
                "read",
                "read file",
                json!({
                    "type": "object",
                    "required": ["path", "_i"],
                    "properties": {
                        "path": { "type": "string" },
                        "_i": { "type": "string" }
                    }
                }),
            )],
            tool_choice: Some(ToolChoice::Required),
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("missing OMP intent metadata is repaired");

    let arguments = &response.choices[0].message.tool_calls[0].function.arguments;
    assert_eq!(arguments["path"], "calculator.py");
    assert!(
        arguments["_i"]
            .as_str()
            .is_some_and(|intent| !intent.is_empty())
    );
}

#[tokio::test]
async fn rejects_generated_tool_call_for_undeclared_tool() {
    let backend = ProtocolTestBackend::new(
        "local-qwen36",
        r#"<tool_call>{"name":"delete_file","arguments":{"path":"Cargo.toml"}}</tool_call>"#,
    );
    let runtime = Runtime::new(backend);
    let err = runtime
        .chat(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("lookup rust")],
            tools: vec![ToolDefinition::function("lookup", "lookup", json!({}))],
            tool_choice: Some(ToolChoice::Required),
            ..ChatCompletionRequest::default()
        })
        .await
        .expect_err("undeclared generated tool call is rejected");

    assert!(matches!(err, RuntimeError::ToolCallValidation(_)));
    assert!(err.to_string().contains("delete_file"));
}

#[tokio::test]
async fn rejects_generated_tool_call_that_mismatches_explicit_choice() {
    let backend = ProtocolTestBackend::new(
        "local-qwen36",
        r#"<tool_call>{"name":"lookup","arguments":{"query":"rust"}}</tool_call>"#,
    );
    let runtime = Runtime::new(backend);
    let err = runtime
        .chat(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("edit Cargo.toml")],
            tools: vec![
                ToolDefinition::function("lookup", "lookup", json!({})),
                ToolDefinition::function("edit_file", "edit file", json!({})),
            ],
            tool_choice: Some(ToolChoice::Function {
                name: "edit_file".to_owned(),
            }),
            ..ChatCompletionRequest::default()
        })
        .await
        .expect_err("explicit tool choice requires matching generated tool calls");

    assert!(matches!(err, RuntimeError::ToolCallValidation(_)));
    assert!(err.to_string().contains("edit_file"));
}

#[tokio::test]
async fn accepts_multiple_generated_tool_calls_when_all_are_declared() {
    let backend = ProtocolTestBackend::new(
        "local-qwen36",
        concat!(
            r#"<tool_call>{"name":"lookup","arguments":{"query":"rust"}}</tool_call>"#,
            r#"<tool_call>{"name":"edit_file","arguments":{"path":"Cargo.toml"}}</tool_call>"#
        ),
    );
    let runtime = Runtime::new(backend);
    let response = runtime
        .chat(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("lookup then edit")],
            tools: vec![
                ToolDefinition::function("lookup", "lookup", json!({})),
                ToolDefinition::function("edit_file", "edit file", json!({})),
            ],
            tool_choice: Some(ToolChoice::Required),
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("declared tool calls are accepted");

    assert_eq!(response.choices[0].message.tool_calls.len(), 2);
    assert_eq!(
        response.choices[0].message.tool_calls[0].function.name,
        "lookup"
    );
    assert_eq!(
        response.choices[0].message.tool_calls[1].function.name,
        "edit_file"
    );
}

#[tokio::test]
async fn runtime_preserves_chat_context_when_tool_messages_are_present() {
    let observed = Arc::new(Mutex::new(None));
    let runtime = Runtime::new(RecordingChatContextBackend {
        observed: observed.clone(),
        family: "gemma",
    });
    let mut tool_result = ChatMessage::tool("call_1", "Rust is a systems programming language.");
    tool_result.name = Some("lookup".to_owned());

    runtime
        .chat(ChatCompletionRequest {
            model: "local-gemma4".to_owned(),
            messages: vec![
                ChatMessage::system("You are a helpful assistant."),
                ChatMessage::user("lookup rust"),
                ChatMessage::assistant_tool_call("call_1", "lookup", json!({"query": "rust"})),
                tool_result,
                ChatMessage::user("tell me more"),
            ],
            tools: vec![ToolDefinition::function("lookup", "lookup", json!({}))],
            max_tokens: Some(16),
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("Gemma chat with tool messages succeeds");

    let observed = observed
        .lock()
        .expect("observed request lock")
        .clone()
        .expect("backend request captured");
    let chat_context = &observed.as_chat().expect("chat request kind").chat_context;
    assert_eq!(
        chat_context.messages.len(),
        5,
        "should preserve the original system, user, assistant tool call, tool result, and follow-up user messages"
    );
    assert_eq!(chat_context.messages[0].role, BackendChatRole::System);
    assert_eq!(
        chat_context.messages[0].content.as_deref(),
        Some("You are a helpful assistant.")
    );
    assert_eq!(chat_context.messages[1].role, BackendChatRole::User);
    assert_eq!(
        chat_context.messages[1].content.as_deref(),
        Some("lookup rust")
    );
    assert_eq!(chat_context.messages[2].role, BackendChatRole::Assistant);
    assert_eq!(chat_context.messages[2].content, None);
    assert_eq!(chat_context.messages[2].tool_calls.len(), 1);
    assert_eq!(chat_context.messages[2].tool_calls[0].id, "call_1");
    assert_eq!(
        chat_context.messages[2].tool_calls[0].function.name,
        "lookup"
    );
    assert_eq!(
        chat_context.messages[2].tool_calls[0].function.arguments,
        json!({"query": "rust"})
    );
    assert_eq!(chat_context.messages[3].role, BackendChatRole::Tool);
    assert_eq!(
        chat_context.messages[3].tool_call_id.as_deref(),
        Some("call_1")
    );
    assert_eq!(chat_context.messages[3].name.as_deref(), Some("lookup"));
    assert_eq!(
        chat_context.messages[3].content.as_deref(),
        Some("Rust is a systems programming language.")
    );
    assert_eq!(chat_context.messages[4].role, BackendChatRole::User);
    assert_eq!(
        chat_context.messages[4].content.as_deref(),
        Some("tell me more")
    );
}

#[tokio::test]
async fn no_progress_classifier_allows_content_tool_calls_and_json_objects() {
    let content = Runtime::new(ReplayBackend {
        output: BackendOutput {
            text: "Patched Cargo.toml and added the regression test.".to_owned(),
            prompt_tokens: 4,
            prompt_cached_tokens: None,
            completion_tokens: 8,
            finish_reason: BackendFinishReason::Stop,
        },
    })
    .chat(ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("What changed?")],
        ..ChatCompletionRequest::default()
    })
    .await
    .expect("normal content is progress");
    assert_eq!(
        content.choices[0].message.content.as_deref(),
        Some("Patched Cargo.toml and added the regression test.")
    );

    let tool = Runtime::new(ReplayBackend {
        output: BackendOutput {
            text:
                r#"<tool_call>{"name":"read_file","arguments":{"path":"Cargo.toml"}}</tool_call>"#
                    .to_owned(),
            prompt_tokens: 4,
            prompt_cached_tokens: None,
            completion_tokens: 5,
            finish_reason: BackendFinishReason::ToolCalls,
        },
    })
    .chat(ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("Read Cargo.toml")],
        tools: vec![ToolDefinition::function(
            "read_file",
            "read a file",
            json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }),
        )],
        tool_choice: Some(ToolChoice::Required),
        ..ChatCompletionRequest::default()
    })
    .await
    .expect("valid tool call is progress");
    assert_eq!(tool.choices[0].message.tool_calls.len(), 1);

    let json_response = Runtime::new(ReplayBackend {
        output: BackendOutput {
            text: r#"{"answer":"ok"}"#.to_owned(),
            prompt_tokens: 4,
            prompt_cached_tokens: None,
            completion_tokens: 3,
            finish_reason: BackendFinishReason::Stop,
        },
    })
    .chat(ChatCompletionRequest {
        model: "local-qwen36".to_owned(),
        messages: vec![ChatMessage::user("Return JSON")],
        response_format: Some(ResponseFormat::JsonObject),
        ..ChatCompletionRequest::default()
    })
    .await
    .expect("valid JSON object is progress");
    assert_eq!(
        json_response.choices[0].message.content.as_deref(),
        Some(r#"{"answer":"ok"}"#)
    );
}

fn read_tool_definition() -> ToolDefinition {
    ToolDefinition::function(
        "read",
        "read file",
        json!({
            "type": "object",
            "required": ["path", "_i"],
            "properties": {
                "path": { "type": "string" },
                "_i": { "type": "string" }
            }
        }),
    )
}

fn failed_read_attempts(count: usize, arguments: Value) -> Vec<ChatMessage> {
    let mut messages = vec![ChatMessage::user("read missing.txt")];
    for index in 0..count {
        let call_id = format!("call_{index}");
        messages.push(ChatMessage::assistant_tool_call(
            call_id.clone(),
            "read",
            arguments.clone(),
        ));
        messages.push(ChatMessage::tool(
            call_id,
            "error: missing path argument or file not found",
        ));
        messages.push(ChatMessage::user("try again"));
    }
    messages
}

async fn replay_read_tool_error(
    messages: Vec<ChatMessage>,
    generated_arguments: Value,
    expectation: &str,
) -> RuntimeError {
    let backend = ProtocolTestBackend::new(
        "local-qwen36",
        format!(r#"<tool_call>{{"name":"read","arguments":{generated_arguments}}}</tool_call>"#),
    );
    let runtime = Runtime::new(backend);
    runtime
        .chat(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages,
            tools: vec![read_tool_definition()],
            tool_choice: Some(ToolChoice::Required),
            ..ChatCompletionRequest::default()
        })
        .await
        .expect_err(expectation)
}
