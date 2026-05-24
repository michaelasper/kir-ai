use std::{
    hint::black_box,
    time::{Duration, Instant},
};

use futures::{StreamExt, stream::BoxStream};
use llm_api::{ChatCompletionRequest, ChatMessage, ToolChoice, ToolDefinition};
use llm_backend_contracts::{
    BackendError, BackendFinishReason, BackendModelMetadata, BackendOutput, BackendRequest,
    BackendStreamChunk, BackendToolCallDelta, BackendToolCallFunctionDelta, BackendToolCallType,
    ModelBackend,
};
use llm_runtime::Runtime;
use tokio::runtime::Builder;
use tokio_util::sync::CancellationToken;

#[allow(dead_code)]
#[path = "../src/stop.rs"]
mod stop;

use stop::{IncrementalStopDetector, max_stop_sequence_len, safe_stream_emit_len};

const LONG_TEXT_CHUNKS: usize = 4_096;
const LONG_TEXT_CHUNK_BYTES: usize = 32;
const LEGACY_STOP_ITERATIONS: usize = 3;
const CURRENT_STOP_ITERATIONS: usize = 100;
const LARGE_TOOL_ARGUMENT_BYTES: usize = 256 * 1024;
const LARGE_TOOL_ITERATIONS: usize = 25;

fn main() {
    println!("streaming_parsing: stop windows and large tool-call assembly");
    println!(
        "{:<26} {:<24} {:>8} {:>12} {:>12}",
        "case", "path", "iters", "total_ms", "ns/iter"
    );

    let chunks = long_text_chunks();
    let stop = vec![" STOP".to_owned(), "<|eot_id|>".to_owned()];
    let legacy_elapsed = run_stop_case(LEGACY_STOP_ITERATIONS, || {
        legacy_full_stop_scan(&chunks, &stop)
    });
    print_result(
        "long_text_stop",
        "legacy_full_scan",
        LEGACY_STOP_ITERATIONS,
        legacy_elapsed,
    );

    let current_elapsed = run_stop_case(CURRENT_STOP_ITERATIONS, || {
        incremental_stop_scan(&chunks, &stop)
    });
    print_result(
        "long_text_stop",
        "incremental_window",
        CURRENT_STOP_ITERATIONS,
        current_elapsed,
    );

    let runtime = Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("benchmark runtime builds");
    let arguments = large_tool_arguments(LARGE_TOOL_ARGUMENT_BYTES);
    let large_tool_elapsed = run_large_tool_case(&runtime, LARGE_TOOL_ITERATIONS, &arguments);
    print_result(
        "large_tool_arguments",
        "runtime_stream_collect",
        LARGE_TOOL_ITERATIONS,
        large_tool_elapsed,
    );
}

fn run_stop_case(iterations: usize, mut run: impl FnMut() -> usize) -> Duration {
    let mut checksum = 0_usize;
    for _ in 0..2 {
        checksum ^= black_box(run());
    }
    let started = Instant::now();
    for _ in 0..iterations {
        checksum ^= black_box(run());
    }
    black_box(checksum);
    started.elapsed()
}

fn run_large_tool_case(
    runtime: &tokio::runtime::Runtime,
    iterations: usize,
    arguments: &str,
) -> Duration {
    let mut checksum = 0_usize;
    for _ in 0..2 {
        checksum ^= black_box(runtime.block_on(collect_large_tool(arguments.to_owned())));
    }
    let started = Instant::now();
    for _ in 0..iterations {
        checksum ^= black_box(runtime.block_on(collect_large_tool(arguments.to_owned())));
    }
    black_box(checksum);
    started.elapsed()
}

fn print_result(case: &str, path: &str, iterations: usize, elapsed: Duration) {
    let total_ms = elapsed.as_secs_f64() * 1_000.0;
    let ns_per_iter = elapsed.as_secs_f64() * 1_000_000_000.0 / iterations as f64;
    println!(
        "{:<26} {:<24} {:>8} {:>12.3} {:>12.1}",
        case, path, iterations, total_ms, ns_per_iter
    );
}

fn legacy_full_stop_scan(chunks: &[String], stop: &[String]) -> usize {
    let mut raw_text = String::new();
    let mut emitted_len = 0;
    let mut checksum = 0_usize;
    let max_stop_len = max_stop_sequence_len(stop);

    for chunk in chunks {
        raw_text.push_str(chunk);
        if let Some(stop_at) = legacy_earliest_stop_index(&raw_text, stop) {
            checksum ^= stop_at;
            break;
        }
        let safe_len = safe_stream_emit_len(&raw_text, max_stop_len);
        if safe_len > emitted_len {
            checksum ^= safe_len - emitted_len;
            emitted_len = safe_len;
        }
    }
    checksum ^ emitted_len
}

fn incremental_stop_scan(chunks: &[String], stop: &[String]) -> usize {
    let mut raw_text = String::new();
    let mut emitted_len = 0;
    let mut checksum = 0_usize;
    let max_stop_len = max_stop_sequence_len(stop);
    let mut detector = IncrementalStopDetector::new(stop);

    for chunk in chunks {
        raw_text.push_str(chunk);
        if let Some(stop_at) = detector.observe(&raw_text, stop) {
            checksum ^= stop_at;
            break;
        }
        let safe_len = safe_stream_emit_len(&raw_text, max_stop_len);
        if safe_len > emitted_len {
            checksum ^= safe_len - emitted_len;
            emitted_len = safe_len;
        }
    }
    checksum ^ emitted_len
}

fn legacy_earliest_stop_index(content: &str, stop: &[String]) -> Option<usize> {
    stop.iter()
        .filter_map(|sequence| content.find(sequence))
        .min()
}

fn long_text_chunks() -> Vec<String> {
    let mut chunks = Vec::with_capacity(LONG_TEXT_CHUNKS + 1);
    for index in 0..LONG_TEXT_CHUNKS {
        chunks.push(format!(
            "token-{index:06}-{:0<width$}",
            "",
            width = LONG_TEXT_CHUNK_BYTES.saturating_sub("token-000000-".len())
        ));
    }
    chunks.push(" STOP ignored".to_owned());
    chunks
}

async fn collect_large_tool(arguments: String) -> usize {
    let runtime = Runtime::new(LargeStructuredToolBackend { arguments });
    let stream = runtime
        .chat_stream(ChatCompletionRequest {
            model: "local-qwen36".to_owned(),
            messages: vec![ChatMessage::user("lookup rust")],
            tools: vec![ToolDefinition::function(
                "lookup",
                "lookup",
                serde_json::json!({}),
            )],
            tool_choice: Some(ToolChoice::Required),
            stream: true,
            ..ChatCompletionRequest::default()
        })
        .await
        .expect("large tool stream starts");
    let (chunks, usage) = stream
        .collect_chunks()
        .await
        .expect("large tool stream collects");
    let argument_bytes = chunks
        .iter()
        .flat_map(|chunk| &chunk.choices)
        .flat_map(|choice| &choice.delta.tool_calls)
        .filter_map(|delta| delta.function.as_ref())
        .filter_map(|function| function.arguments.as_ref())
        .map(String::len)
        .sum::<usize>();
    argument_bytes ^ usize::try_from(usage.completion_tokens).unwrap_or(0)
}

fn large_tool_arguments(payload_bytes: usize) -> String {
    format!(r#"{{"payload":"{}"}}"#, "x".repeat(payload_bytes))
}

struct LargeStructuredToolBackend {
    arguments: String,
}

#[async_trait::async_trait]
impl ModelBackend for LargeStructuredToolBackend {
    fn model_id(&self) -> &str {
        "local-qwen36"
    }

    fn model_metadata(&self) -> BackendModelMetadata {
        BackendModelMetadata::new(self.model_id(), "streaming-parsing-bench").with_family("qwen")
    }

    async fn generate(&self, _request: BackendRequest) -> Result<BackendOutput, BackendError> {
        Err(BackendError::other(
            "streaming parsing benchmark must use generate_stream".to_owned(),
        ))
    }

    async fn generate_with_cancel(
        &self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> Result<BackendOutput, BackendError> {
        if cancellation.is_cancelled() {
            return Err(BackendError::cancelled());
        }
        self.generate(request).await
    }

    fn generate_stream<'a>(
        &'a self,
        _request: BackendRequest,
    ) -> BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        let arguments = self.arguments.clone();
        async_stream::try_stream! {
            yield BackendStreamChunk {
                text: String::new(),
                tool_call_deltas: vec![BackendToolCallDelta {
                    index: 0,
                    id: Some("call_large".to_owned()),
                    call_type: Some(BackendToolCallType::Function),
                    function: Some(BackendToolCallFunctionDelta {
                        name: Some("lookup".to_owned()),
                        arguments: Some(arguments),
                    }),
                }],
                prompt_tokens: 4,
                prompt_cached_tokens: None,
                completion_tokens: 1,
                finish_reason: Some(BackendFinishReason::ToolCalls),
                progress: None,
            };
        }
        .boxed()
    }

    fn generate_stream_with_cancel<'a>(
        &'a self,
        request: BackendRequest,
        cancellation: CancellationToken,
    ) -> BoxStream<'a, Result<BackendStreamChunk, BackendError>> {
        if cancellation.is_cancelled() {
            return futures::stream::once(async { Err(BackendError::cancelled()) }).boxed();
        }
        self.generate_stream(request)
    }
}
