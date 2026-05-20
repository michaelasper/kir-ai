use crate::RuntimeError;
use crate::response_validation::{default_tool_intent, schema_requires_string_intent_argument};
use llm_api::{
    ApiError, ChatCompletionRequest, ToolCall, ToolCallDelta, ToolCallFunction,
    ToolCallFunctionDelta, ToolCallType, ToolChoice,
};
use llm_backend::BackendToolChoice;
use llm_tool_parser::{ParsedAssistant, split_reasoning};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ToolSchemaNormalization {
    #[default]
    Preserve,
    Canonical,
}

#[derive(Debug, Default)]
pub(crate) struct StructuredToolDeltaAssembler {
    calls: Vec<StructuredToolCallAccumulator>,
}

#[derive(Debug, Default)]
struct StructuredToolCallAccumulator {
    id: Option<String>,
    call_type: Option<ToolCallType>,
    name: String,
    arguments: String,
}

impl StructuredToolDeltaAssembler {
    pub(crate) fn is_empty(&self) -> bool {
        self.calls.is_empty()
    }

    pub(crate) fn push(&mut self, delta: &ToolCallDelta) -> Result<(), RuntimeError> {
        let index = usize::try_from(delta.index).map_err(|err| {
            ApiError::invalid_request(format!("tool call index does not fit usize: {err}"))
        })?;
        if self.calls.len() <= index {
            self.calls
                .resize_with(index + 1, StructuredToolCallAccumulator::default);
        }
        let call = &mut self.calls[index];
        if let Some(id) = &delta.id {
            call.id = Some(id.clone());
        }
        if let Some(call_type) = &delta.call_type {
            call.call_type = Some(call_type.clone());
        }
        if let Some(function) = &delta.function {
            if let Some(name) = &function.name {
                call.name.push_str(name);
            }
            if let Some(arguments) = &function.arguments {
                call.arguments.push_str(arguments);
            }
        }
        Ok(())
    }

    pub(crate) fn into_parsed(self, content: &str) -> Result<ParsedAssistant, RuntimeError> {
        let mut tool_calls = Vec::new();
        for (index, call) in self.calls.into_iter().enumerate() {
            if call.name.trim().is_empty() && call.arguments.trim().is_empty() {
                continue;
            }
            if call.name.trim().is_empty() {
                return Err(RuntimeError::ToolCallValidation(format!(
                    "streamed tool call `{index}` was missing a function name"
                )));
            }
            let arguments = parse_structured_tool_arguments(index, &call.arguments)?;
            tool_calls.push(ToolCall {
                id: call.id.unwrap_or_else(|| format!("call_{index}")),
                call_type: call.call_type.unwrap_or(ToolCallType::Function),
                function: ToolCallFunction {
                    name: call.name,
                    arguments,
                },
            });
        }
        let (reasoning, content) = split_reasoning(content)?;
        Ok(ParsedAssistant {
            reasoning,
            content,
            tool_calls,
        })
    }
}

fn parse_structured_tool_arguments(
    index: usize,
    arguments: &str,
) -> Result<serde_json::Value, RuntimeError> {
    let trimmed = arguments.trim();
    if trimmed.is_empty() {
        return Ok(serde_json::json!({}));
    }
    serde_json::from_str(trimmed).map_err(|err| {
        RuntimeError::ToolCallValidation(format!(
            "streamed tool call `{index}` arguments were not valid JSON: {err}"
        ))
    })
}

pub(crate) fn request_may_fill_tool_intent_arguments(request: &ChatCompletionRequest) -> bool {
    request
        .tools
        .iter()
        .any(|tool| schema_requires_string_intent_argument(&tool.function.parameters))
}

pub(crate) fn structured_tool_delta_without_arguments(
    delta: &ToolCallDelta,
) -> Option<ToolCallDelta> {
    let function = delta.function.as_ref().and_then(|function| {
        function.name.as_ref().map(|name| ToolCallFunctionDelta {
            name: Some(name.clone()),
            arguments: None,
        })
    });
    let stripped = ToolCallDelta {
        index: delta.index,
        id: delta.id.clone(),
        call_type: delta.call_type.clone(),
        function,
    };
    structured_tool_delta_has_progress(&stripped).then_some(stripped)
}

fn structured_tool_delta_has_progress(delta: &ToolCallDelta) -> bool {
    delta.id.is_some()
        || delta.call_type.is_some()
        || delta
            .function
            .as_ref()
            .is_some_and(|function| function.name.is_some() || function.arguments.is_some())
}

pub(crate) fn tool_call_delta(
    index: usize,
    tool_call: &ToolCall,
) -> Result<ToolCallDelta, RuntimeError> {
    Ok(ToolCallDelta {
        index: u32::try_from(index).map_err(|err| {
            ApiError::invalid_request(format!("tool call index does not fit u32: {err}"))
        })?,
        id: Some(tool_call.id.clone()),
        call_type: Some(tool_call.call_type.clone()),
        function: Some(ToolCallFunctionDelta {
            name: Some(tool_call.function.name.clone()),
            arguments: Some(serde_json::to_string(&tool_call.function.arguments)?),
        }),
    })
}

pub(crate) fn tool_call_arguments_delta(
    index: usize,
    tool_call: &ToolCall,
) -> Result<ToolCallDelta, RuntimeError> {
    Ok(ToolCallDelta {
        index: u32::try_from(index).map_err(|err| {
            ApiError::invalid_request(format!("tool call index does not fit u32: {err}"))
        })?,
        id: None,
        call_type: None,
        function: Some(ToolCallFunctionDelta {
            name: None,
            arguments: Some(serde_json::to_string(&tool_call.function.arguments)?),
        }),
    })
}

pub(crate) fn fill_missing_tool_intent_arguments(
    parsed: &mut ParsedAssistant,
    request: &ChatCompletionRequest,
) {
    for tool_call in &mut parsed.tool_calls {
        let Some(arguments) = tool_call.function.arguments.as_object_mut() else {
            continue;
        };
        if arguments.contains_key("_i") {
            continue;
        }
        let Some(tool) = request
            .tools
            .iter()
            .find(|tool| tool.function.name == tool_call.function.name)
        else {
            continue;
        };
        if schema_requires_string_intent_argument(&tool.function.parameters) {
            arguments.insert(
                "_i".to_owned(),
                serde_json::Value::String(default_tool_intent(&tool_call.function.name).to_owned()),
            );
        }
    }
}

pub(crate) fn request_requires_tool_choice(request: &ChatCompletionRequest) -> bool {
    matches!(
        request.tool_choice,
        Some(ToolChoice::Required | ToolChoice::Function { .. })
    )
}

pub(crate) fn required_backend_tool_choice(
    request: &ChatCompletionRequest,
) -> Option<BackendToolChoice> {
    match &request.tool_choice {
        Some(ToolChoice::Required) => Some(BackendToolChoice::RequiredAny),
        Some(ToolChoice::Function { name }) => {
            Some(BackendToolChoice::RequiredFunction(name.clone()))
        }
        Some(ToolChoice::Auto | ToolChoice::None) | None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structured_tool_delta_assembler_preserves_reasoning_from_visible_text() {
        let mut assembler = StructuredToolDeltaAssembler::default();
        assembler
            .push(&ToolCallDelta {
                index: 0,
                id: Some("call_0".to_owned()),
                call_type: Some(ToolCallType::Function),
                function: Some(ToolCallFunctionDelta {
                    name: Some("read_file".to_owned()),
                    arguments: Some(r#"{"path":"Cargo.toml"}"#.to_owned()),
                }),
            })
            .expect("delta is valid");

        let parsed = assembler
            .into_parsed("<think>Need the manifest.</think>")
            .expect("structured deltas parse");

        assert_eq!(parsed.reasoning.as_deref(), Some("Need the manifest."));
        assert_eq!(parsed.content, "");
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].function.name, "read_file");
    }
}
