use super::*;

#[tokio::test]
async fn chat_rejects_missing_model_family_before_generation() {
    let runtime = Runtime::new(FamilyMetadataBackend { family: None });
    let err = runtime
        .chat(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("say hi")],
            max_tokens: Some(16),
            ..ChatCompletionRequest::default()
        })
        .await
        .expect_err("missing family should fail before generation");

    assert!(err.to_string().contains("did not declare a model family"));
}

#[tokio::test]
async fn chat_accepts_mlx_backend_when_family_is_qwen() {
    let runtime = Runtime::new(MlxQwenMetadataBackend);
    let response = runtime
        .chat(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("say hi")],
            max_tokens: Some(16),
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("qwen MLX metadata selects Qwen adapter");

    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("hello from mlx")
    );
}

#[tokio::test]
async fn chat_accepts_mlx_backend_when_family_is_gemma() {
    let runtime = Runtime::new(MlxGemmaMetadataBackend);
    let response = runtime
        .chat(ChatCompletionRequest {
            model: "local-gemma4".to_owned(),
            messages: vec![ChatMessage::user("say hi")],
            max_tokens: Some(16),
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("Gemma MLX metadata selects Gemma adapter");

    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("hello from gemma")
    );
}

#[tokio::test]
async fn chat_accepts_mlx_backend_when_family_is_deepseek() {
    let runtime = Runtime::new(MlxDeepSeekMetadataBackend);
    let response = runtime
        .chat(ChatCompletionRequest {
            model: "local-deepseek".to_owned(),
            messages: vec![ChatMessage::user("say hi")],
            max_tokens: Some(16),
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("DeepSeek MLX metadata selects DeepSeek adapter");

    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("hello from deepseek")
    );
}

#[tokio::test]
async fn chat_accepts_mlx_backend_when_family_is_llama() {
    let runtime = Runtime::new(MlxLlamaMetadataBackend);
    let response = runtime
        .chat(ChatCompletionRequest {
            model: "local-llama".to_owned(),
            messages: vec![ChatMessage::user("say hi")],
            max_tokens: Some(16),
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("Llama MLX metadata selects Llama adapter");

    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("hello from llama")
    );
}

#[tokio::test]
async fn protocol_test_backend_exercises_required_tool_protocol_for_each_family() {
    for (model_id, family) in [
        ("local-qwen36", ModelFamily::Qwen),
        ("local-gemma4", ModelFamily::Gemma),
        ("local-deepseek", ModelFamily::DeepSeek),
        ("local-llama", ModelFamily::Llama),
    ] {
        let backend = ProtocolTestBackend::new(model_id, "plain text")
            .with_family(family)
            .with_required_tool_protocol();
        let runtime = Runtime::new(backend);
        let response = runtime
            .chat(ChatCompletionRequest {
                model: model_id.to_owned(),
                messages: vec![ChatMessage::user("read Cargo.toml")],
                tools: vec![ToolDefinition::function(
                    "read_file",
                    "read a file from the workspace",
                    json!({
                        "type": "object",
                        "properties": {
                            "path": { "type": "string" }
                        },
                        "required": ["path"]
                    }),
                )],
                tool_choice: Some(ToolChoice::Function {
                    name: "read_file".to_owned(),
                }),
                ..ChatCompletionRequest::default()
            })
            .await
            .unwrap_or_else(|err| panic!("{family:?} protocol test backend failed: {err}"));

        assert_eq!(
            response.choices[0].finish_reason,
            Some(FinishReason::ToolCalls),
            "{family:?} should produce a parsed tool-call finish"
        );
        let tool_call = response.choices[0]
            .message
            .tool_calls
            .first()
            .unwrap_or_else(|| panic!("{family:?} response should include a tool call"));
        assert_eq!(tool_call.function.name, "read_file");
        assert_eq!(
            tool_call.function.arguments,
            json!({ "path": "Cargo.toml" })
        );
    }
}
