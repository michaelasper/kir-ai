use super::protocol::MlxToolMarkup;
use llm_backend::{BackendError, BackendOutput, BackendStreamChunk};
use serde::Deserialize;
use serde_json::Value;

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

impl MlxCompletionResponse {
    fn first_choice(self) -> Result<MlxCompletionChoice, BackendError> {
        self.choices
            .into_iter()
            .next()
            .ok_or_else(|| BackendError::Other("MLX completion response had no choices".to_owned()))
    }
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
}

impl MlxSseParser {
    pub(super) fn new(
        prompt: &str,
        stop_tokens: &'static [&'static str],
        tool_markup: MlxToolMarkup,
    ) -> Self {
        Self {
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
        }
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
            if let Some(chunk) = self.parse_line(&line)? {
                chunks.push(chunk);
            }
        }
        Ok(chunks)
    }

    pub(super) fn finish(&mut self) -> Result<Vec<BackendStreamChunk>, BackendError> {
        if !self.saw_done {
            return Err(BackendError::Other(
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
            if let Some(chunk) = self.parse_line(line.trim_end_matches('\r'))? {
                chunks.push(chunk);
            }
        }
        let text = self.stop_filter.finish();
        if !text.is_empty() {
            self.estimated_completion_tokens += count_visible_tokens(&text);
            let completion_tokens = self.completion_token_delta(None, true);
            chunks.push(BackendStreamChunk {
                text,
                prompt_tokens: self.prompt_tokens,
                prompt_cached_tokens: self.prompt_cached_tokens,
                completion_tokens,
                finish_reason: None,
            });
        }
        Ok(chunks)
    }

    fn parse_line(&mut self, line: &str) -> Result<Option<BackendStreamChunk>, BackendError> {
        let Some(data) = mlx_sse_data(line) else {
            return Ok(None);
        };
        if data == "[DONE]" {
            self.saw_done = true;
            return Ok(None);
        }
        let completion = serde_json::from_str::<MlxCompletionResponse>(data).map_err(|err| {
            BackendError::Other(format!("invalid MLX SSE completion JSON: {err}"))
        })?;
        self.parse_completion(completion)
    }

    fn parse_completion(
        &mut self,
        completion: MlxCompletionResponse,
    ) -> Result<Option<BackendStreamChunk>, BackendError> {
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
        let choice = completion.first_choice()?;
        if let Some(tool_calls) = choice
            .delta
            .as_ref()
            .and_then(|message| message.tool_calls.as_ref())
            .or_else(|| {
                choice
                    .message
                    .as_ref()
                    .and_then(|message| message.tool_calls.as_ref())
            })
        {
            self.push_tool_calls(tool_calls);
        }
        let text = choice
            .text
            .or_else(|| choice.delta.and_then(|message| message.content))
            .or_else(|| choice.message.and_then(|message| message.content))
            .unwrap_or_default();
        let mut text = self.stop_filter.push_str(&text);
        self.estimated_completion_tokens += count_visible_tokens(&text);
        let finish_reason = choice
            .finish_reason
            .as_deref()
            .map(|reason| mlx_finish_reason(Some(reason)))
            .transpose()?;
        let finish_reason = finish_reason.or_else(|| {
            self.stop_filter
                .is_stopped()
                .then_some(llm_api::FinishReason::Stop)
        });
        if matches!(finish_reason, Some(llm_api::FinishReason::ToolCalls)) {
            let tool_text = self.render_tool_calls()?;
            self.estimated_completion_tokens += count_visible_tokens(&tool_text);
            text.push_str(&tool_text);
        }
        let completion_tokens =
            self.completion_token_delta(usage_completion_tokens, finish_reason.is_some());
        if text.is_empty() && finish_reason.is_none() && completion_tokens == 0 {
            return Ok(None);
        }
        Ok(Some(BackendStreamChunk {
            text,
            prompt_tokens: self.prompt_tokens,
            prompt_cached_tokens: self.prompt_cached_tokens,
            completion_tokens,
            finish_reason,
        }))
    }

    fn push_tool_calls(&mut self, tool_calls: &[MlxToolCall]) {
        for call in tool_calls {
            let index = call.index.unwrap_or(self.tool_calls.len());
            if self.tool_calls.len() <= index {
                self.tool_calls
                    .resize_with(index + 1, MlxToolCallAccumulator::default);
            }
            let accumulator = &mut self.tool_calls[index];
            if let Some(function) = &call.function {
                if let Some(name) = &function.name {
                    accumulator.name.push_str(name);
                }
                if let Some(arguments) = &function.arguments {
                    accumulator.arguments.push_str(arguments);
                }
            }
        }
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

fn mlx_finish_reason(reason: Option<&str>) -> Result<llm_api::FinishReason, BackendError> {
    match reason {
        Some("length") => Ok(llm_api::FinishReason::Length),
        Some("tool_calls") => Ok(llm_api::FinishReason::ToolCalls),
        Some("stop") | None => Ok(llm_api::FinishReason::Stop),
        Some(other) => Err(BackendError::Other(format!(
            "unsupported MLX finish reason `{other}`"
        ))),
    }
}

fn render_mlx_tool_call(
    call: &MlxToolCallAccumulator,
    markup: MlxToolMarkup,
) -> Result<String, BackendError> {
    if call.name.trim().is_empty() {
        return Err(BackendError::Other(
            "MLX structured tool call was missing a function name".to_owned(),
        ));
    }
    let arguments = parse_mlx_tool_arguments(&call.arguments)?;
    match markup {
        MlxToolMarkup::Json => Ok(format!(
            "<tool_call>{}</tool_call>",
            serde_json::json!({
                "name": call.name.as_str(),
                "arguments": arguments,
            })
        )),
        MlxToolMarkup::DeepSeek => Ok(format!(
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>{}\n```json\n{}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            call.name,
            serde_json::to_string(&arguments).map_err(|err| BackendError::Other(format!(
                "DeepSeek tool argument render failed: {err}"
            )))?
        )),
        MlxToolMarkup::Gemma => {
            let Value::Object(arguments) = arguments else {
                return Err(BackendError::Other(
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
        BackendError::Other(format!(
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
                BackendError::Other(format!("Gemma tool key render failed: {err}"))
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
            .map_err(|err| BackendError::Other(format!("Gemma tool value render failed: {err}"))),
    }
}

pub(super) fn parse_mlx_completion_body(
    body: &str,
    prompt: &str,
    stop_tokens: &'static [&'static str],
    tool_markup: MlxToolMarkup,
) -> Result<(BackendOutput, usize), BackendError> {
    let mut parser = MlxSseParser::new(prompt, stop_tokens, tool_markup);
    let chunks = if body.trim_start().starts_with("data:") {
        let mut chunks = parser.push_str(body)?;
        chunks.extend(parser.finish()?);
        chunks
    } else {
        let completion = serde_json::from_str::<MlxCompletionResponse>(body)
            .map_err(|err| BackendError::Other(format!("invalid MLX completion JSON: {err}")))?;
        let mut chunks = Vec::new();
        if let Some(chunk) = parser.parse_completion(completion)? {
            chunks.push(chunk);
        }
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
    let mut finish_reason = llm_api::FinishReason::Stop;
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
