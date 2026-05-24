use super::*;

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
