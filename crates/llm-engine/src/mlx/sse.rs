use super::protocol::MlxToolMarkup;
use llm_backend::{
    BackendError, BackendFinishReason, BackendOutput, BackendStreamChunk, BackendToolCallDelta,
    BackendToolCallFunctionDelta, BackendToolCallType,
};
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Debug, Deserialize)]
struct MlxCompletionResponse {
    choices: Vec<MlxCompletionChoice>,
    usage: Option<MlxUsage>,
}

#[derive(Debug, Deserialize)]
struct MlxCompletionChoice {
    text: Option<String>,
    message: Option<MlxMessage>,
    delta: Option<MlxMessage>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MlxMessage {
    content: Option<String>,
    tool_calls: Option<Vec<MlxToolCall>>,
}

#[derive(Debug, Deserialize)]
struct MlxToolCall {
    index: Option<usize>,
    id: Option<String>,
    #[serde(rename = "type")]
    call_type: Option<BackendToolCallType>,
    function: Option<MlxToolCallFunction>,
}

#[derive(Debug, Deserialize)]
struct MlxToolCallFunction {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MlxUsage {
    #[serde(alias = "input_tokens")]
    prompt_tokens: Option<u64>,
    #[serde(alias = "output_tokens")]
    completion_tokens: Option<u64>,
    prompt_tokens_details: Option<MlxPromptTokensDetails>,
}

#[derive(Debug, Deserialize)]
struct MlxPromptTokensDetails {
    cached_tokens: Option<u64>,
}

#[derive(Debug)]
pub(super) struct MlxSseParser {
    prompt_tokens: u64,
    prompt_cached_tokens: Option<u64>,
    estimated_completion_tokens: u64,
    emitted_completion_tokens: u64,
    uses_upstream_usage: bool,
    saw_done: bool,
    line_buffer: String,
    stop_filter: MlxControlStopFilter,
    tool_markup: MlxToolMarkup,
    tool_calls: Vec<MlxToolCallAccumulator>,
    emit_structured_tool_deltas: bool,
    qwen_xml: Option<QwenXmlToolParser>,
}

impl MlxSseParser {
    #[cfg(test)]
    pub(super) fn new(
        prompt: &str,
        stop_tokens: &'static [&'static str],
        tool_markup: MlxToolMarkup,
    ) -> Self {
        Self::new_inner(prompt, stop_tokens, tool_markup, false, None)
            .expect("empty Qwen XML schema is always valid")
    }

    pub(super) fn new_with_tool_schema(
        prompt: &str,
        stop_tokens: &'static [&'static str],
        tool_markup: MlxToolMarkup,
        tool_schema: Option<&str>,
    ) -> Result<Self, BackendError> {
        Self::new_inner(prompt, stop_tokens, tool_markup, false, tool_schema)
    }

    fn new_inner(
        prompt: &str,
        stop_tokens: &'static [&'static str],
        tool_markup: MlxToolMarkup,
        emit_structured_tool_deltas: bool,
        tool_schema: Option<&str>,
    ) -> Result<Self, BackendError> {
        let qwen_xml = matches!(tool_markup, MlxToolMarkup::QwenXml)
            .then(|| QwenXmlToolParser::new(emit_structured_tool_deltas, tool_schema))
            .transpose()?;
        Ok(Self {
            prompt_tokens: count_whitespace_tokens(prompt),
            prompt_cached_tokens: None,
            estimated_completion_tokens: 0,
            emitted_completion_tokens: 0,
            uses_upstream_usage: false,
            saw_done: false,
            line_buffer: String::new(),
            stop_filter: MlxControlStopFilter::new(stop_tokens),
            tool_markup,
            tool_calls: Vec::new(),
            emit_structured_tool_deltas,
            qwen_xml,
        })
    }

    pub(super) fn new_streaming(
        prompt: &str,
        stop_tokens: &'static [&'static str],
        tool_markup: MlxToolMarkup,
    ) -> Self {
        Self::new_streaming_with_tool_schema(prompt, stop_tokens, tool_markup, None)
            .expect("empty Qwen XML schema is always valid")
    }

    pub(super) fn new_streaming_with_tool_schema(
        prompt: &str,
        stop_tokens: &'static [&'static str],
        tool_markup: MlxToolMarkup,
        tool_schema: Option<&str>,
    ) -> Result<Self, BackendError> {
        Self::new_inner(prompt, stop_tokens, tool_markup, true, tool_schema)
    }

    pub(super) fn push_str(
        &mut self,
        chunk: &str,
    ) -> Result<Vec<BackendStreamChunk>, BackendError> {
        self.line_buffer.push_str(chunk);
        let mut chunks = Vec::new();
        while let Some(index) = self.line_buffer.find('\n') {
            let mut line = self.line_buffer.drain(..=index).collect::<String>();
            if line.ends_with('\n') {
                line.pop();
            }
            if line.ends_with('\r') {
                line.pop();
            }
            chunks.extend(self.parse_line(&line)?);
        }
        Ok(chunks)
    }

    pub(super) fn finish(&mut self) -> Result<Vec<BackendStreamChunk>, BackendError> {
        if !self.saw_done {
            return Err(BackendError::other(
                "MLX SSE completion ended before data: [DONE]".to_owned(),
            ));
        }
        self.flush_pending()
    }

    fn finish_non_streaming(&mut self) -> Result<Vec<BackendStreamChunk>, BackendError> {
        self.flush_pending()
    }

    fn flush_pending(&mut self) -> Result<Vec<BackendStreamChunk>, BackendError> {
        let mut chunks = Vec::new();
        if !self.line_buffer.is_empty() {
            let line = std::mem::take(&mut self.line_buffer);
            chunks.extend(self.parse_line(line.trim_end_matches('\r'))?);
        }
        let text = self.stop_filter.finish();
        if !text.is_empty() {
            chunks.extend(self.push_text_chunks(&text)?);
        }
        let qwen_xml_final = if let Some(qwen_xml) = &mut self.qwen_xml {
            Some(qwen_xml.finish()?)
        } else {
            None
        };
        if let Some(emissions) = qwen_xml_final {
            chunks.extend(self.qwen_xml_emissions_to_chunks(emissions));
        }
        self.finalize_completion_chunks(&mut chunks, None, None, true);
        Ok(chunks)
    }

    fn parse_line(&mut self, line: &str) -> Result<Vec<BackendStreamChunk>, BackendError> {
        let Some(data) = mlx_sse_data(line) else {
            return Ok(Vec::new());
        };
        if data == "[DONE]" {
            self.saw_done = true;
            return Ok(Vec::new());
        }
        let completion = serde_json::from_str::<MlxCompletionResponse>(data).map_err(|err| {
            BackendError::other(format!("invalid MLX SSE completion JSON: {err}"))
        })?;
        self.parse_completion(completion)
    }

    fn parse_completion(
        &mut self,
        completion: MlxCompletionResponse,
    ) -> Result<Vec<BackendStreamChunk>, BackendError> {
        if let Some(prompt_tokens) = completion
            .usage
            .as_ref()
            .and_then(|usage| usage.prompt_tokens)
        {
            self.prompt_tokens = self.prompt_tokens.max(prompt_tokens);
        }
        if let Some(cached_tokens) = completion
            .usage
            .as_ref()
            .and_then(|usage| usage.prompt_tokens_details.as_ref())
            .and_then(|details| details.cached_tokens)
        {
            self.prompt_cached_tokens =
                max_optional_u64(self.prompt_cached_tokens, Some(cached_tokens));
        }
        let usage_completion_tokens = completion
            .usage
            .as_ref()
            .and_then(|usage| usage.completion_tokens);
        let Some(choice) = completion.choices.into_iter().next() else {
            let mut chunks = Vec::new();
            self.finalize_completion_chunks(&mut chunks, usage_completion_tokens, None, false);
            return Ok(chunks);
        };
        let tool_call_deltas = if let Some(tool_calls) = choice
            .delta
            .as_ref()
            .and_then(|message| message.tool_calls.as_ref())
            .or_else(|| {
                choice
                    .message
                    .as_ref()
                    .and_then(|message| message.tool_calls.as_ref())
            }) {
            self.push_tool_calls(tool_calls)?
        } else {
            Vec::new()
        };
        let text = choice
            .text
            .or_else(|| choice.delta.and_then(|message| message.content))
            .or_else(|| choice.message.and_then(|message| message.content))
            .unwrap_or_default();
        let text = self.stop_filter.push_str(&text);
        let finish_reason = choice
            .finish_reason
            .as_deref()
            .map(|reason| mlx_finish_reason(Some(reason)))
            .transpose()?;
        let finish_reason = finish_reason.or_else(|| {
            self.stop_filter
                .is_stopped()
                .then_some(BackendFinishReason::Stop)
        });
        let mut chunks = Vec::new();
        if !tool_call_deltas.is_empty() {
            chunks.push(self.chunk(String::new(), tool_call_deltas));
        }
        if !text.is_empty() {
            chunks.extend(self.push_text_chunks(&text)?);
        }
        if matches!(finish_reason, Some(BackendFinishReason::Length))
            && let Some(qwen_xml) = &mut self.qwen_xml
        {
            let emissions = qwen_xml.finish_truncated()?;
            chunks.extend(self.qwen_xml_emissions_to_chunks(emissions));
        }
        if matches!(finish_reason, Some(BackendFinishReason::ToolCalls))
            && !self.emit_structured_tool_deltas
            && !self.tool_calls.is_empty()
        {
            let tool_text = self.render_tool_calls()?;
            chunks.extend(self.push_text_chunks(&tool_text)?);
        }
        let is_final_chunk = finish_reason.is_some();
        self.finalize_completion_chunks(
            &mut chunks,
            usage_completion_tokens,
            finish_reason,
            is_final_chunk,
        );
        Ok(chunks)
    }

    fn push_text_chunks(&mut self, text: &str) -> Result<Vec<BackendStreamChunk>, BackendError> {
        if let Some(qwen_xml) = &mut self.qwen_xml {
            let emissions = qwen_xml.push_str(text)?;
            return Ok(self.qwen_xml_emissions_to_chunks(emissions));
        }
        self.estimated_completion_tokens += count_visible_tokens(text);
        Ok(vec![self.chunk(text.to_owned(), Vec::new())])
    }

    fn qwen_xml_emissions_to_chunks(
        &mut self,
        emissions: Vec<QwenXmlEmission>,
    ) -> Vec<BackendStreamChunk> {
        let chunks = emissions
            .into_iter()
            .map(|emission| self.chunk(emission.text, emission.tool_call_deltas))
            .collect::<Vec<_>>();
        for chunk in &chunks {
            self.estimated_completion_tokens += count_visible_tokens(&chunk.text);
        }
        chunks
    }

    fn chunk(
        &self,
        text: String,
        tool_call_deltas: Vec<BackendToolCallDelta>,
    ) -> BackendStreamChunk {
        BackendStreamChunk {
            text,
            tool_call_deltas,
            prompt_tokens: self.prompt_tokens,
            prompt_cached_tokens: self.prompt_cached_tokens,
            completion_tokens: 0,
            finish_reason: None,
        }
    }

    fn finalize_completion_chunks(
        &mut self,
        chunks: &mut Vec<BackendStreamChunk>,
        usage_completion_tokens: Option<u64>,
        finish_reason: Option<BackendFinishReason>,
        is_final_chunk: bool,
    ) {
        let completion_tokens =
            self.completion_token_delta(usage_completion_tokens, is_final_chunk);
        if let Some(last) = chunks.last_mut() {
            last.completion_tokens += completion_tokens;
            last.finish_reason = finish_reason;
        } else if finish_reason.is_some() || completion_tokens > 0 {
            chunks.push(BackendStreamChunk {
                text: String::new(),
                tool_call_deltas: Vec::new(),
                prompt_tokens: self.prompt_tokens,
                prompt_cached_tokens: self.prompt_cached_tokens,
                completion_tokens,
                finish_reason,
            });
        }
    }

    fn push_tool_calls(
        &mut self,
        tool_calls: &[MlxToolCall],
    ) -> Result<Vec<BackendToolCallDelta>, BackendError> {
        let mut deltas = Vec::new();
        for call in tool_calls {
            let index = call.index.unwrap_or(self.tool_calls.len());
            if self.tool_calls.len() <= index {
                self.tool_calls
                    .resize_with(index + 1, MlxToolCallAccumulator::default);
            }
            let accumulator = &mut self.tool_calls[index];
            let mut function_delta = None;
            if let Some(function) = &call.function {
                if let Some(name) = &function.name {
                    accumulator.name.push_str(name);
                }
                if let Some(arguments) = &function.arguments {
                    accumulator.arguments.push_str(arguments);
                }
                function_delta = Some(BackendToolCallFunctionDelta {
                    name: function.name.clone(),
                    arguments: function.arguments.clone(),
                });
            }
            if self.emit_structured_tool_deltas {
                deltas.push(BackendToolCallDelta {
                    index: u32::try_from(index).map_err(|err| {
                        BackendError::other(format!("MLX tool call index does not fit u32: {err}"))
                    })?,
                    id: call.id.clone(),
                    call_type: call.call_type.clone(),
                    function: function_delta,
                });
            }
        }
        Ok(deltas)
    }

    fn render_tool_calls(&mut self) -> Result<String, BackendError> {
        if self.tool_calls.is_empty() {
            return Ok(String::new());
        }
        let mut rendered = String::new();
        for call in std::mem::take(&mut self.tool_calls) {
            rendered.push_str(&render_mlx_tool_call(&call, self.tool_markup)?);
        }
        Ok(rendered)
    }

    fn completion_token_delta(
        &mut self,
        usage_completion_tokens: Option<u64>,
        is_final_chunk: bool,
    ) -> u64 {
        if let Some(total) = usage_completion_tokens {
            self.uses_upstream_usage = true;
            let delta = total.saturating_sub(self.emitted_completion_tokens);
            self.emitted_completion_tokens = self.emitted_completion_tokens.max(total);
            return delta;
        }
        if self.uses_upstream_usage || !is_final_chunk {
            return 0;
        }
        let delta = self
            .estimated_completion_tokens
            .saturating_sub(self.emitted_completion_tokens);
        self.emitted_completion_tokens = self.estimated_completion_tokens;
        delta
    }
}

#[derive(Debug, Default, Clone)]
struct MlxToolCallAccumulator {
    name: String,
    arguments: String,
}

const QWEN_XML_TOOL_START: &str = "<tool_call>";
const QWEN_XML_TOOL_END: &str = "</tool_call>";
const QWEN_XML_FUNCTION_START: &str = "<function=";
const QWEN_XML_FUNCTION_END: &str = "</function>";
const QWEN_XML_PARAMETER_START: &str = "<parameter=";
const QWEN_XML_PARAMETER_END: &str = "</parameter>";
const QWEN_REASONING_START: &str = "<think>";
const QWEN_REASONING_END: &str = "</think>";

#[derive(Debug)]
struct QwenXmlToolParser {
    structured_deltas: bool,
    schema: QwenXmlToolSchema,
    state: QwenXmlState,
    buffer: String,
    next_index: u32,
    current_call: Option<QwenXmlActiveCall>,
    current_parameter: Option<QwenXmlActiveParameter>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QwenXmlState {
    Outside,
    ExpectFunction,
    InFunction,
    InParameter,
    ExpectToolClose,
}

#[derive(Debug)]
struct QwenXmlEmission {
    text: String,
    tool_call_deltas: Vec<BackendToolCallDelta>,
}

impl QwenXmlEmission {
    fn text(text: String) -> Self {
        Self {
            text,
            tool_call_deltas: Vec::new(),
        }
    }

    fn delta(delta: BackendToolCallDelta) -> Self {
        Self {
            text: String::new(),
            tool_call_deltas: vec![delta],
        }
    }
}

#[derive(Debug, Default)]
struct QwenXmlToolSchema {
    functions: BTreeMap<String, BTreeMap<String, QwenXmlParameterKind>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QwenXmlParameterKind {
    String,
    Json,
}

impl QwenXmlToolSchema {
    fn parse(schema: Option<&str>) -> Result<Self, BackendError> {
        let Some(schema) = schema else {
            return Ok(Self::default());
        };
        let value = serde_json::from_str::<Value>(schema).map_err(|err| {
            BackendError::other(format!("Qwen XML tool schema was not valid JSON: {err}"))
        })?;
        let mut functions = BTreeMap::new();
        let Some(tools) = value.as_array() else {
            return Ok(Self::default());
        };
        for tool in tools {
            let Some(function) = tool.get("function") else {
                continue;
            };
            let Some(name) = function.get("name").and_then(Value::as_str) else {
                continue;
            };
            let mut parameters = BTreeMap::new();
            if let Some(properties) = function
                .get("parameters")
                .and_then(|parameters| parameters.get("properties"))
                .and_then(Value::as_object)
            {
                for (key, property) in properties {
                    let kind = if schema_property_is_string(property) {
                        QwenXmlParameterKind::String
                    } else {
                        QwenXmlParameterKind::Json
                    };
                    parameters.insert(key.clone(), kind);
                }
            }
            functions.insert(name.to_owned(), parameters);
        }
        Ok(Self { functions })
    }

    fn parameter_kind(&self, function: &str, key: &str) -> QwenXmlParameterKind {
        self.functions
            .get(function)
            .and_then(|parameters| parameters.get(key))
            .copied()
            .unwrap_or(QwenXmlParameterKind::String)
    }
}

fn schema_property_is_string(property: &Value) -> bool {
    match property.get("type") {
        Some(Value::String(kind)) => kind == "string",
        Some(Value::Array(kinds)) => kinds
            .iter()
            .any(|kind| kind.as_str().is_some_and(|kind| kind == "string")),
        _ => false,
    }
}

#[derive(Debug)]
struct QwenXmlActiveCall {
    index: u32,
    name: String,
    argument_count: usize,
    arguments_json: String,
}

impl QwenXmlActiveCall {
    fn new(index: u32, name: String) -> Self {
        Self {
            index,
            name,
            argument_count: 0,
            arguments_json: String::new(),
        }
    }

    fn next_parameter_prefix(&mut self, key: &str) -> Result<String, BackendError> {
        let mut prefix = if self.argument_count == 0 {
            "{".to_owned()
        } else {
            ",".to_owned()
        };
        self.argument_count += 1;
        prefix.push_str(&serde_json::to_string(key).map_err(|err| {
            BackendError::other(format!("Qwen XML parameter key render failed: {err}"))
        })?);
        prefix.push(':');
        Ok(prefix)
    }
}

#[derive(Debug)]
struct QwenXmlActiveParameter {
    key: String,
    prefix: String,
    kind: QwenXmlParameterKind,
    value: String,
}

impl QwenXmlToolParser {
    fn new(structured_deltas: bool, tool_schema: Option<&str>) -> Result<Self, BackendError> {
        Ok(Self {
            structured_deltas,
            schema: QwenXmlToolSchema::parse(tool_schema)?,
            state: QwenXmlState::Outside,
            buffer: String::new(),
            next_index: 0,
            current_call: None,
            current_parameter: None,
        })
    }

    fn push_str(&mut self, text: &str) -> Result<Vec<QwenXmlEmission>, BackendError> {
        self.buffer.push_str(text);
        let mut emissions = Vec::new();
        loop {
            let progressed = match self.state {
                QwenXmlState::Outside => self.process_outside(&mut emissions),
                QwenXmlState::ExpectFunction => self.process_expected_function(&mut emissions),
                QwenXmlState::InFunction => self.process_in_function(&mut emissions),
                QwenXmlState::InParameter => self.process_in_parameter(&mut emissions),
                QwenXmlState::ExpectToolClose => self.process_expected_tool_close(),
            }?;
            if !progressed {
                break;
            }
        }
        Ok(emissions)
    }

    fn finish(&mut self) -> Result<Vec<QwenXmlEmission>, BackendError> {
        if !matches!(self.state, QwenXmlState::Outside) {
            return Err(BackendError::other(format!(
                "Qwen XML tool call ended while parser was in {:?}",
                self.state
            )));
        }
        let mut emissions = Vec::new();
        if !self.buffer.is_empty() {
            emissions.push(QwenXmlEmission::text(std::mem::take(&mut self.buffer)));
        }
        Ok(emissions)
    }

    fn finish_truncated(&mut self) -> Result<Vec<QwenXmlEmission>, BackendError> {
        let mut emissions = Vec::new();
        match self.state {
            QwenXmlState::Outside => {}
            QwenXmlState::ExpectFunction => {
                self.current_call = None;
                self.current_parameter = None;
                self.buffer.clear();
                self.state = QwenXmlState::Outside;
            }
            QwenXmlState::InFunction => {
                self.buffer.clear();
                if self.current_call.is_some() {
                    self.finish_function(&mut emissions)?;
                }
                self.state = QwenXmlState::Outside;
            }
            QwenXmlState::InParameter => {
                let final_piece = std::mem::take(&mut self.buffer);
                let parameter_kind = self
                    .current_parameter
                    .as_ref()
                    .map(|parameter| parameter.kind);
                match parameter_kind {
                    Some(QwenXmlParameterKind::String) => {
                        self.finish_parameter(&final_piece, &mut emissions)?;
                    }
                    Some(QwenXmlParameterKind::Json) => {
                        let parsed = self.finish_parameter(&final_piece, &mut emissions);
                        if parsed.is_err() {
                            self.current_parameter = None;
                            if let Some(call) = &mut self.current_call {
                                call.argument_count = call.argument_count.saturating_sub(1);
                            }
                        } else {
                            parsed?;
                        }
                    }
                    None => {}
                }
                if self.current_call.is_some() {
                    self.finish_function(&mut emissions)?;
                }
                self.state = QwenXmlState::Outside;
            }
            QwenXmlState::ExpectToolClose => {
                self.buffer.clear();
                self.current_call = None;
                self.current_parameter = None;
                self.state = QwenXmlState::Outside;
            }
        }
        Ok(emissions)
    }

    fn process_outside(
        &mut self,
        emissions: &mut Vec<QwenXmlEmission>,
    ) -> Result<bool, BackendError> {
        match earliest_qwen_marker(&self.buffer) {
            Some((index, QwenXmlOutsideMarker::ToolStart)) => {
                if index > 0 {
                    emissions.push(QwenXmlEmission::text(self.buffer.drain(..index).collect()));
                    return Ok(true);
                }
                self.buffer.drain(..QWEN_XML_TOOL_START.len());
                self.state = QwenXmlState::ExpectFunction;
                return Ok(true);
            }
            Some((index, QwenXmlOutsideMarker::ReasoningStart)) => {
                if index > 0 {
                    emissions.push(QwenXmlEmission::text(self.buffer.drain(..index).collect()));
                    return Ok(true);
                }
                if self.drain_leading_reasoning_block(emissions)? {
                    return Ok(true);
                }
                return Ok(false);
            }
            None => {}
        }
        let pending = pending_tag_prefix_len(&self.buffer, QWEN_XML_TOOL_START)
            .max(pending_tag_prefix_len(&self.buffer, QWEN_REASONING_START));
        let emit_len = self.buffer.len().saturating_sub(pending);
        if emit_len > 0 {
            emissions.push(QwenXmlEmission::text(
                self.buffer.drain(..emit_len).collect(),
            ));
            return Ok(true);
        }
        Ok(false)
    }

    fn process_expected_function(
        &mut self,
        emissions: &mut Vec<QwenXmlEmission>,
    ) -> Result<bool, BackendError> {
        if self.drain_leading_whitespace() {
            return Ok(true);
        }
        if self.drain_leading_reasoning_block(emissions)? {
            return Ok(true);
        }
        if self.buffer.is_empty()
            || is_partial_tag_prefix(&self.buffer, QWEN_XML_FUNCTION_START)
            || is_partial_tag_prefix(&self.buffer, QWEN_REASONING_START)
        {
            return Ok(false);
        }
        if let Some(index) = self.buffer.find(QWEN_XML_FUNCTION_START)
            && index > 0
        {
            emissions.push(QwenXmlEmission::text(self.buffer.drain(..index).collect()));
            return Ok(true);
        }
        if !self.buffer.starts_with(QWEN_XML_FUNCTION_START) {
            return Ok(false);
        }
        let Some(end) = self.buffer.find('>') else {
            return Ok(false);
        };
        let name = self.buffer[QWEN_XML_FUNCTION_START.len()..end].to_owned();
        if name.trim().is_empty() || name.contains('<') {
            return Err(qwen_xml_error(format!(
                "invalid function tag `<function={name}>`"
            )));
        }
        self.buffer.drain(..=end);
        let index = self.next_index;
        self.next_index = self
            .next_index
            .checked_add(1)
            .ok_or_else(|| qwen_xml_error("tool call index overflow"))?;
        self.current_call = Some(QwenXmlActiveCall::new(index, name.clone()));
        if self.structured_deltas {
            emissions.push(QwenXmlEmission::delta(BackendToolCallDelta {
                index,
                id: Some(format!("call_{index}")),
                call_type: Some(BackendToolCallType::Function),
                function: Some(BackendToolCallFunctionDelta {
                    name: Some(name),
                    arguments: None,
                }),
            }));
        }
        self.state = QwenXmlState::InFunction;
        Ok(true)
    }

    fn process_in_function(
        &mut self,
        emissions: &mut Vec<QwenXmlEmission>,
    ) -> Result<bool, BackendError> {
        if self.drain_leading_whitespace() {
            return Ok(true);
        }
        if self.buffer.is_empty()
            || is_partial_tag_prefix(&self.buffer, QWEN_XML_PARAMETER_START)
            || is_partial_tag_prefix(&self.buffer, QWEN_XML_FUNCTION_END)
        {
            return Ok(false);
        }
        if self.buffer.starts_with(QWEN_XML_PARAMETER_START) {
            let Some(end) = self.buffer.find('>') else {
                return Ok(false);
            };
            let key = self.buffer[QWEN_XML_PARAMETER_START.len()..end].to_owned();
            if key.trim().is_empty() || key.contains('<') {
                return Err(qwen_xml_error(format!(
                    "invalid parameter tag `<parameter={key}>`"
                )));
            }
            self.buffer.drain(..=end);
            self.start_parameter(key, emissions)?;
            self.state = QwenXmlState::InParameter;
            return Ok(true);
        }
        if self.buffer.starts_with(QWEN_XML_FUNCTION_END) {
            self.buffer.drain(..QWEN_XML_FUNCTION_END.len());
            self.finish_function(emissions)?;
            self.state = QwenXmlState::ExpectToolClose;
            return Ok(true);
        }
        Err(qwen_xml_error(
            "expected Qwen XML parameter or function close tag",
        ))
    }

    fn process_in_parameter(
        &mut self,
        emissions: &mut Vec<QwenXmlEmission>,
    ) -> Result<bool, BackendError> {
        if let Some(index) = self.buffer.find(QWEN_XML_PARAMETER_END) {
            let final_piece = self.buffer[..index].to_owned();
            self.buffer.drain(..index + QWEN_XML_PARAMETER_END.len());
            self.finish_parameter(&final_piece, emissions)?;
            self.state = QwenXmlState::InFunction;
            return Ok(true);
        }
        let pending = pending_tag_prefix_len(&self.buffer, QWEN_XML_PARAMETER_END);
        let emit_len = self.buffer.len().saturating_sub(pending);
        if emit_len == 0 {
            return Ok(false);
        }
        let value = self.buffer.drain(..emit_len).collect::<String>();
        let kind = self
            .current_parameter
            .as_ref()
            .ok_or_else(|| qwen_xml_error("parameter value without active parameter"))?
            .kind;
        match kind {
            QwenXmlParameterKind::String => {
                self.push_argument_fragment(&escape_json_string_fragment(&value), emissions)?;
            }
            QwenXmlParameterKind::Json => self
                .current_parameter
                .as_mut()
                .ok_or_else(|| qwen_xml_error("parameter value without active parameter"))?
                .value
                .push_str(&value),
        }
        Ok(true)
    }

    fn process_expected_tool_close(&mut self) -> Result<bool, BackendError> {
        if self.drain_leading_whitespace() {
            return Ok(true);
        }
        if self.buffer.is_empty() || is_partial_tag_prefix(&self.buffer, QWEN_XML_TOOL_END) {
            return Ok(false);
        }
        if !self.buffer.starts_with(QWEN_XML_TOOL_END) {
            return Err(qwen_xml_error(format!(
                "expected `{QWEN_XML_TOOL_END}` after `{QWEN_XML_FUNCTION_END}`"
            )));
        }
        self.buffer.drain(..QWEN_XML_TOOL_END.len());
        self.state = QwenXmlState::Outside;
        Ok(true)
    }

    fn start_parameter(
        &mut self,
        key: String,
        emissions: &mut Vec<QwenXmlEmission>,
    ) -> Result<(), BackendError> {
        let call = self
            .current_call
            .as_mut()
            .ok_or_else(|| qwen_xml_error("parameter started without active function"))?;
        let kind = self.schema.parameter_kind(&call.name, &key);
        let prefix = call.next_parameter_prefix(&key)?;
        let parameter = QwenXmlActiveParameter {
            key,
            prefix,
            kind,
            value: String::new(),
        };
        if kind == QwenXmlParameterKind::String {
            let fragment = format!("{}\"", parameter.prefix);
            self.push_argument_fragment(&fragment, emissions)?;
        }
        self.current_parameter = Some(parameter);
        Ok(())
    }

    fn finish_parameter(
        &mut self,
        final_piece: &str,
        emissions: &mut Vec<QwenXmlEmission>,
    ) -> Result<(), BackendError> {
        let mut parameter = self
            .current_parameter
            .take()
            .ok_or_else(|| qwen_xml_error("parameter close without active parameter"))?;
        match parameter.kind {
            QwenXmlParameterKind::String => {
                let fragment = format!("{}\"", escape_json_string_fragment(final_piece));
                self.push_argument_fragment(&fragment, emissions)?;
            }
            QwenXmlParameterKind::Json => {
                parameter.value.push_str(final_piece);
                let trimmed = parameter.value.trim();
                let value = serde_json::from_str::<Value>(trimmed).map_err(|err| {
                    qwen_xml_error(format!(
                        "parameter `{}` was not valid JSON: {err}",
                        parameter.key
                    ))
                })?;
                let rendered = serde_json::to_string(&value).map_err(|err| {
                    qwen_xml_error(format!(
                        "parameter `{}` JSON render failed: {err}",
                        parameter.key
                    ))
                })?;
                let fragment = format!("{}{rendered}", parameter.prefix);
                self.push_argument_fragment(&fragment, emissions)?;
            }
        }
        Ok(())
    }

    fn finish_function(
        &mut self,
        emissions: &mut Vec<QwenXmlEmission>,
    ) -> Result<(), BackendError> {
        let argument_close = {
            let call = self
                .current_call
                .as_ref()
                .ok_or_else(|| qwen_xml_error("function close without active function"))?;
            if call.argument_count == 0 { "{}" } else { "}" }
        };
        self.push_argument_fragment(argument_close, emissions)?;
        if !self.structured_deltas {
            let call = self
                .current_call
                .as_ref()
                .ok_or_else(|| qwen_xml_error("function close without active function"))?;
            let arguments = serde_json::from_str::<Value>(&call.arguments_json).map_err(|err| {
                qwen_xml_error(format!(
                    "assembled function arguments were not valid JSON: {err}"
                ))
            })?;
            emissions.push(QwenXmlEmission::text(format!(
                "<tool_call>{}</tool_call>",
                serde_json::json!({
                    "name": call.name.as_str(),
                    "arguments": arguments,
                })
            )));
        }
        self.current_call = None;
        Ok(())
    }

    fn push_argument_fragment(
        &mut self,
        fragment: &str,
        emissions: &mut Vec<QwenXmlEmission>,
    ) -> Result<(), BackendError> {
        let call = self
            .current_call
            .as_mut()
            .ok_or_else(|| qwen_xml_error("argument fragment without active function"))?;
        call.arguments_json.push_str(fragment);
        if self.structured_deltas && !fragment.is_empty() {
            emissions.push(QwenXmlEmission::delta(BackendToolCallDelta {
                index: call.index,
                id: None,
                call_type: None,
                function: Some(BackendToolCallFunctionDelta {
                    name: None,
                    arguments: Some(fragment.to_owned()),
                }),
            }));
        }
        Ok(())
    }

    fn drain_leading_whitespace(&mut self) -> bool {
        let trimmed = self.buffer.trim_start_matches(char::is_whitespace);
        let trim_len = self.buffer.len() - trimmed.len();
        if trim_len > 0 {
            self.buffer.drain(..trim_len);
            true
        } else {
            false
        }
    }

    fn drain_leading_reasoning_block(
        &mut self,
        emissions: &mut Vec<QwenXmlEmission>,
    ) -> Result<bool, BackendError> {
        if !self.buffer.starts_with(QWEN_REASONING_START) {
            return Ok(false);
        }
        let body_start = QWEN_REASONING_START.len();
        let Some(end_rel) = self.buffer[body_start..].find(QWEN_REASONING_END) else {
            return Ok(false);
        };
        let end = body_start + end_rel + QWEN_REASONING_END.len();
        emissions.push(QwenXmlEmission::text(self.buffer.drain(..end).collect()));
        Ok(true)
    }
}

#[derive(Debug, Clone, Copy)]
enum QwenXmlOutsideMarker {
    ToolStart,
    ReasoningStart,
}

fn earliest_qwen_marker(buffer: &str) -> Option<(usize, QwenXmlOutsideMarker)> {
    [
        (
            buffer.find(QWEN_XML_TOOL_START),
            QwenXmlOutsideMarker::ToolStart,
        ),
        (
            buffer.find(QWEN_REASONING_START),
            QwenXmlOutsideMarker::ReasoningStart,
        ),
    ]
    .into_iter()
    .filter_map(|(index, marker)| index.map(|index| (index, marker)))
    .min_by_key(|(index, _)| *index)
}

fn pending_tag_prefix_len(buffer: &str, tag: &str) -> usize {
    (1..tag.len())
        .filter(|length| buffer.ends_with(&tag[..*length]))
        .max()
        .unwrap_or(0)
}

fn is_partial_tag_prefix(buffer: &str, tag: &str) -> bool {
    buffer.len() < tag.len() && tag.starts_with(buffer)
}

fn escape_json_string_fragment(value: &str) -> String {
    let rendered =
        serde_json::to_string(value).expect("serializing a Rust string to JSON cannot fail");
    rendered[1..rendered.len() - 1].to_owned()
}

fn qwen_xml_error(message: impl Into<String>) -> BackendError {
    BackendError::other(format!("Qwen XML tool parser error: {}", message.into()))
}

#[derive(Debug, Clone)]
struct MlxControlStopFilter {
    stop_tokens: &'static [&'static str],
    pending: String,
    stopped: bool,
}

impl MlxControlStopFilter {
    fn new(stop_tokens: &'static [&'static str]) -> Self {
        Self {
            stop_tokens,
            pending: String::new(),
            stopped: false,
        }
    }

    fn is_stopped(&self) -> bool {
        self.stopped
    }

    fn push_str(&mut self, text: &str) -> String {
        if self.stopped || text.is_empty() {
            return String::new();
        }
        self.pending.push_str(text);
        if let Some((index, token_len)) = self.first_stop_token() {
            self.stopped = true;
            let output = self.pending[..index].to_owned();
            self.pending.drain(..index + token_len);
            self.pending.clear();
            return output;
        }
        let withheld = self.pending_stop_prefix_len();
        if withheld == self.pending.len() {
            return String::new();
        }
        let split_at = self.pending.len() - withheld;
        self.pending.drain(..split_at).collect()
    }

    fn finish(&mut self) -> String {
        if self.stopped {
            self.pending.clear();
            return String::new();
        }
        std::mem::take(&mut self.pending)
    }

    fn first_stop_token(&self) -> Option<(usize, usize)> {
        self.stop_tokens
            .iter()
            .filter_map(|token| self.pending.find(token).map(|index| (index, token.len())))
            .min_by_key(|(index, _)| *index)
    }

    fn pending_stop_prefix_len(&self) -> usize {
        self.stop_tokens
            .iter()
            .flat_map(|token| {
                (1..token.len()).filter(move |length| {
                    token.is_char_boundary(*length) && self.pending.ends_with(&token[..*length])
                })
            })
            .max()
            .unwrap_or(0)
    }
}

fn mlx_sse_data(line: &str) -> Option<&str> {
    line.strip_prefix("data:")
        .map(|data| data.strip_prefix(' ').unwrap_or(data))
}

fn count_visible_tokens(text: &str) -> u64 {
    if text.trim().is_empty() {
        0
    } else {
        count_whitespace_tokens(text)
    }
}

fn mlx_finish_reason(reason: Option<&str>) -> Result<BackendFinishReason, BackendError> {
    match reason {
        Some("length") => Ok(BackendFinishReason::Length),
        Some("tool_calls") => Ok(BackendFinishReason::ToolCalls),
        Some("stop") | None => Ok(BackendFinishReason::Stop),
        Some(other) => Err(BackendError::other(format!(
            "unsupported MLX finish reason `{other}`"
        ))),
    }
}

fn render_mlx_tool_call(
    call: &MlxToolCallAccumulator,
    markup: MlxToolMarkup,
) -> Result<String, BackendError> {
    if call.name.trim().is_empty() {
        return Err(BackendError::other(
            "MLX structured tool call was missing a function name".to_owned(),
        ));
    }
    let arguments = parse_mlx_tool_arguments(&call.arguments)?;
    match markup {
        MlxToolMarkup::Json | MlxToolMarkup::QwenXml => Ok(format!(
            "<tool_call>{}</tool_call>",
            serde_json::json!({
                "name": call.name.as_str(),
                "arguments": arguments,
            })
        )),
        MlxToolMarkup::DeepSeek => Ok(format!(
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>{}\n```json\n{}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            call.name,
            serde_json::to_string(&arguments).map_err(|err| BackendError::other(format!(
                "DeepSeek tool argument render failed: {err}"
            )))?
        )),
        MlxToolMarkup::Gemma => {
            let Value::Object(arguments) = arguments else {
                return Err(BackendError::other(
                    "MLX structured Gemma tool arguments must be a JSON object".to_owned(),
                ));
            };
            Ok(format!(
                "<|tool_call>call:{}{}<tool_call|>",
                call.name,
                render_gemma_tool_object(&arguments)?
            ))
        }
    }
}

fn parse_mlx_tool_arguments(arguments: &str) -> Result<Value, BackendError> {
    let trimmed = arguments.trim();
    if trimmed.is_empty() {
        return Ok(serde_json::json!({}));
    }
    serde_json::from_str::<Value>(trimmed).map_err(|err| {
        BackendError::other(format!(
            "invalid MLX structured tool call arguments `{trimmed}`: {err}"
        ))
    })
}

fn render_gemma_tool_object(
    object: &serde_json::Map<String, Value>,
) -> Result<String, BackendError> {
    let mut rendered = String::from("{");
    for (index, (key, value)) in object.iter().enumerate() {
        if index > 0 {
            rendered.push(',');
        }
        rendered.push_str(
            &serde_json::to_string(key).map_err(|err| {
                BackendError::other(format!("Gemma tool key render failed: {err}"))
            })?,
        );
        rendered.push(':');
        rendered.push_str(&render_gemma_tool_value(value)?);
    }
    rendered.push('}');
    Ok(rendered)
}

fn render_gemma_tool_value(value: &Value) -> Result<String, BackendError> {
    match value {
        Value::Object(object) => render_gemma_tool_object(object),
        Value::Array(values) => {
            let mut rendered = String::from("[");
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    rendered.push(',');
                }
                rendered.push_str(&render_gemma_tool_value(value)?);
            }
            rendered.push(']');
            Ok(rendered)
        }
        _ => serde_json::to_string(value)
            .map_err(|err| BackendError::other(format!("Gemma tool value render failed: {err}"))),
    }
}

pub(super) fn parse_mlx_completion_body(
    body: &str,
    prompt: &str,
    stop_tokens: &'static [&'static str],
    tool_markup: MlxToolMarkup,
    tool_schema: Option<&str>,
) -> Result<(BackendOutput, usize), BackendError> {
    let mut parser =
        MlxSseParser::new_with_tool_schema(prompt, stop_tokens, tool_markup, tool_schema)?;
    let chunks = if body.trim_start().starts_with("data:") {
        let mut chunks = parser.push_str(body)?;
        chunks.extend(parser.finish()?);
        chunks
    } else {
        let completion = serde_json::from_str::<MlxCompletionResponse>(body)
            .map_err(|err| BackendError::other(format!("invalid MLX completion JSON: {err}")))?;
        let mut chunks = Vec::new();
        chunks.extend(parser.parse_completion(completion)?);
        chunks.extend(parser.finish_non_streaming()?);
        chunks
    };
    let chunk_count = chunks.len();
    Ok((fold_mlx_chunks(chunks, prompt), chunk_count))
}

pub(super) fn fold_mlx_chunks(chunks: Vec<BackendStreamChunk>, prompt: &str) -> BackendOutput {
    let mut text = String::new();
    let mut prompt_tokens = 0;
    let mut prompt_cached_tokens = None;
    let mut completion_tokens = 0;
    let mut finish_reason = BackendFinishReason::Stop;
    for chunk in chunks {
        prompt_tokens = prompt_tokens.max(chunk.prompt_tokens);
        prompt_cached_tokens = max_optional_u64(prompt_cached_tokens, chunk.prompt_cached_tokens);
        completion_tokens += chunk.completion_tokens;
        text.push_str(&chunk.text);
        if let Some(reason) = chunk.finish_reason {
            finish_reason = reason;
        }
    }
    if prompt_tokens == 0 {
        prompt_tokens = count_whitespace_tokens(prompt);
    }
    if completion_tokens == 0 && !text.is_empty() {
        completion_tokens = count_whitespace_tokens(&text);
    }
    BackendOutput {
        prompt_tokens,
        prompt_cached_tokens,
        completion_tokens,
        text,
        finish_reason,
    }
}

fn max_optional_u64(current: Option<u64>, next: Option<u64>) -> Option<u64> {
    match (current, next) {
        (Some(current), Some(next)) => Some(current.max(next)),
        (Some(current), None) => Some(current),
        (None, Some(next)) => Some(next),
        (None, None) => None,
    }
}

pub(super) fn count_whitespace_tokens(text: &str) -> u64 {
    text.split_whitespace().count().max(1) as u64
}
