# MLX Sidecar Bounded Timeouts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add bounded connect, request, and per-chunk read timeouts to the MLX sidecar HTTP client so a wedged sidecar cannot hold a scheduler permit indefinitely.

**Architecture:** Mirror the `HubTimeouts`/`HubClient` pattern from `crates/llm-hub/src/client.rs`. Introduce an `MlxTimeouts` struct, build the MLX `reqwest::Client` with those bounds, and wrap each `bytes.next()` call in `tokio::time::timeout`. Add a `Stall` failure metric kind and three CLI flags.

**Tech Stack:** Rust, reqwest, tokio::time, existing FakeMlxServer test harness.

---

### Task 1: Add MlxTimeouts type and build_http_client helper

**Files:**
- Modify: `crates/llm-engine/src/mlx/client.rs`

- [ ] **Step 1: Add MlxTimeouts, build_http_client, and format_duration to client.rs**

Add the following to the end of `crates/llm-engine/src/mlx/client.rs`:

```rust
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MlxTimeouts {
    pub connect: Duration,
    pub request: Duration,
    pub read: Duration,
}

impl Default for MlxTimeouts {
    fn default() -> Self {
        Self {
            connect: Duration::from_secs(5),
            request: Duration::from_secs(300),
            read: Duration::from_secs(60),
        }
    }
}

pub(super) fn build_http_client(timeouts: MlxTimeouts) -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(timeouts.connect)
        .timeout(timeouts.request)
        .build()
        .expect("MLX HTTP client builds")
}

pub(super) fn format_duration(duration: Duration) -> String {
    if duration.as_secs() > 0 {
        format!("{}s", duration.as_secs())
    } else {
        format!("{}ms", duration.as_millis())
    }
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check -p llm-engine`
Expected: compiles without errors

- [ ] **Step 3: Commit**

```bash
git add crates/llm-engine/src/mlx/client.rs
git commit -m "feat(mlx): add MlxTimeouts type and build_http_client helper"
```

---

### Task 2: Add MlxBackendFailureKind::Stall metric variant

**Files:**
- Modify: `crates/llm-engine/src/mlx/metrics.rs`

- [ ] **Step 1: Add Stall variant to MlxBackendFailureKind**

In `crates/llm-engine/src/mlx/metrics.rs`, add `Stall` to the `MlxBackendFailureKind` enum (after `SseParse`):

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MlxBackendFailureKind {
    HttpStatus,
    Transport,
    StreamRead,
    InvalidUtf8,
    SseParse,
    Stall,
    Cancelled,
}
```

- [ ] **Step 2: Add stall_failures counter and metric recording**

Add `stall_failures: u64` to `MlxBackendMetricCounters`:

```rust
struct MlxBackendMetricCounters {
    // ... existing fields ...
    stall_failures: u64,
}
```

Add `"stall_failures"` to the `snapshot()` method's JSON output:

```rust
"stall_failures": counters.stall_failures,
```

Add a `Stall` arm in the `record_failure` match:

```rust
MlxBackendFailureKind::Stall => counters.stall_failures += 1,
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p llm-engine`
Expected: compiles without errors (the `Stall` variant isn't constructed yet, but the match is exhaustive)

- [ ] **Step 4: Commit**

```bash
git add crates/llm-engine/src/mlx/metrics.rs
git commit -m "feat(mlx): add Stall failure metric kind"
```

---

### Task 3: Wire MlxTimeouts into MlxBackendOptions and MlxBackend construction

**Files:**
- Modify: `crates/llm-engine/src/mlx.rs`

- [ ] **Step 1: Add timeouts field to MlxBackendOptions**

In `crates/llm-engine/src/mlx.rs`, add the `timeouts` field to `MlxBackendOptions`:

```rust
#[derive(Debug, Clone)]
pub struct MlxBackendOptions {
    pub endpoint: Url,
    pub family: Option<ModelFamily>,
    pub timeouts: MlxTimeouts,
}
```

Add the import at the top of the file:

```rust
use client::{is_loopback_endpoint, build_http_client, MlxTimeouts, format_duration};
```

- [ ] **Step 2: Update MlxBackend to store timeouts and use build_http_client**

Add `timeouts` field to `MlxBackend`:

```rust
#[derive(Debug, Clone)]
pub struct MlxBackend {
    model_id: String,
    metadata: BackendModelMetadata,
    upstream_model: String,
    endpoint: Url,
    control_stop_tokens: &'static [&'static str],
    client: reqwest::Client,
    timeouts: MlxTimeouts,
    metrics: Arc<MlxBackendMetrics>,
}
```

In `open_with_options`, replace `reqwest::Client::new()` with `build_http_client(options.timeouts)` and store the timeouts:

```rust
let client = build_http_client(options.timeouts);
let timeouts = options.timeouts;
Ok(Self {
    model_id: model_id.clone(),
    metadata,
    upstream_model,
    endpoint: options.endpoint,
    control_stop_tokens,
    client,
    timeouts,
    metrics: mlx_backend_metrics(),
})
```

- [ ] **Step 3: Update MlxBackendOptions::default to include timeouts**

```rust
impl Default for MlxBackendOptions {
    fn default() -> Self {
        Self {
            endpoint: Url::parse(DEFAULT_MLX_ENDPOINT).expect(
                "DEFAULT_MLX_ENDPOINT is a valid URL verified at compile time by this assertion",
            ),
            family: None,
            timeouts: MlxTimeouts::default(),
        }
    }
}
```

- [ ] **Step 4: Update existing tests that construct MlxBackendOptions directly**

In `crates/llm-engine/src/mlx/tests.rs`, every `MlxBackendOptions { endpoint: ..., family: ... }` literal now needs a `timeouts` field. The easiest fix is to use `..MlxBackendOptions::default()` for existing tests that don't need custom timeouts. For each test that constructs `MlxBackendOptions`, change:

```rust
MlxBackendOptions {
    endpoint: server.endpoint(),
    family: Some(ModelFamily::Qwen),
}
```

to:

```rust
MlxBackendOptions {
    endpoint: server.endpoint(),
    family: Some(ModelFamily::Qwen),
    ..MlxBackendOptions::default()
}
```

Apply this pattern to every `MlxBackendOptions` construction in `tests.rs` that doesn't use `..MlxBackendOptions::default()` already.

- [ ] **Step 5: Verify all existing tests still pass**

Run: `cargo test -p llm-engine -- mlx`
Expected: all existing MLX tests pass

- [ ] **Step 6: Commit**

```bash
git add crates/llm-engine/src/mlx.rs crates/llm-engine/src/mlx/tests.rs
git commit -m "feat(mlx): wire MlxTimeouts into MlxBackendOptions and client construction"
```

---

### Task 4: Write failing stall timeout test

**Files:**
- Modify: `crates/llm-engine/src/mlx/tests.rs`

- [ ] **Step 1: Add a FakeMlxServer variant that stalls mid-stream**

Add a new constructor `start_with_stall` to `FakeMlxServer` that sends an initial SSE chunk, then sleeps for a long time before sending more data. This is a variant of the existing `start_with_response_delay_and_content_length` but splits the response:

```rust
fn start_with_stall(
    first_chunk: &'static str,
    second_chunk: &'static str,
    stall_duration: Duration,
) -> Self {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind fake mlx server");
    let endpoint = Url::parse(&format!(
        "http://{}/v1",
        listener.local_addr().expect("addr")
    ))
    .expect("endpoint url");
    let received = Arc::new(Mutex::new(None));
    let received_path = Arc::new(Mutex::new(None));
    let received_for_thread = received.clone();
    let received_path_for_thread = received_path.clone();
    let join = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept fake mlx request");
        let mut bytes = Vec::new();
        let mut buffer = [0_u8; 1024];
        let header_end;
        loop {
            let read = stream.read(&mut buffer).expect("read request");
            assert!(read > 0, "client closed before headers");
            bytes.extend_from_slice(&buffer[..read]);
            if let Some(index) = find_subsequence(&bytes, b"\r\n\r\n") {
                header_end = index + 4;
                break;
            }
        }
        let headers = String::from_utf8_lossy(&bytes[..header_end]);
        let request_path = headers
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .expect("request path")
            .to_owned();
        *received_path_for_thread.lock().expect("received path lock") = Some(request_path);
        let request_content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().expect("content length"))
            })
            .expect("content-length header");
        while bytes.len() < header_end + request_content_length {
            let read = stream.read(&mut buffer).expect("read body");
            assert!(read > 0, "client closed before body");
            bytes.extend_from_slice(&buffer[..read]);
        }
        let body = &bytes[header_end..header_end + request_content_length];
        *received_for_thread.lock().expect("received lock") =
            Some(serde_json::from_slice(body).expect("json request body"));
        let full_body = format!("{first_chunk}{second_chunk}");
        let _ = write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
        );
        let _ = stream.flush();
        let _ = write!(stream, "{:x}\r\n{}\r\n", first_chunk.len(), first_chunk);
        let _ = stream.flush();
        thread::sleep(stall_duration);
        let _ = write!(stream, "{:x}\r\n{}\r\n", second_chunk.len(), second_chunk);
        let _ = stream.flush();
        let _ = write!(stream, "0\r\n\r\n");
        let _ = stream.flush();
    });
    Self {
        endpoint,
        snapshot: tempfile::tempdir().expect("snapshot tempdir"),
        received,
        received_path,
        join: Some(join),
    }
}
```

- [ ] **Step 2: Write the failing test**

```rust
#[tokio::test]
async fn mlx_backend_per_chunk_timeout_detects_stalled_stream() {
    let server = FakeMlxServer::start_with_stall(
        "data:{\"choices\":[{\"text\":\"one\",\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":2}}\n\n",
        "data: {\"choices\":[{\"text\":\"two\",\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
        Duration::from_secs(300),
    );
    let mut backend = MlxBackend::open_with_options(
        "local-mlx",
        server.snapshot_path(),
        MlxBackendOptions {
            endpoint: server.endpoint(),
            family: Some(ModelFamily::Qwen),
            timeouts: MlxTimeouts {
                connect: Duration::from_secs(5),
                request: Duration::from_secs(300),
                read: Duration::from_millis(100),
            },
        },
    )
    .await
    .expect("backend opens");
    backend.metrics = Arc::new(MlxBackendMetrics::default());
    let metrics = backend.metrics.clone();

    let err = backend
        .generate(BackendRequest {
            model: "local-mlx".to_owned(),
            prompt: "hello mlx".to_owned(),
            chat_context: None,
            max_tokens: Some(12),
            sampling: SamplingConfig::Greedy,
            required_tool_choice: None,
            json_object_mode: false,
            conversation_mode: false,
            cache_context: BackendCacheContext::raw_prompt(),
        })
        .await
        .expect_err("stalled stream produces timeout error");

    assert!(err.to_string().contains("stalled"), "expected stall error, got: {err}");
    let metrics = metrics.snapshot();
    assert_eq!(metrics["failed_requests"], 1);
    assert_eq!(metrics["stall_failures"], 1);
}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cargo test -p llm-engine -- mlx_backend_per_chunk_timeout_detects_stalled_stream`
Expected: FAIL (the test times out at the overall request timeout of 300s, or hangs, because there's no per-chunk timeout yet — kill with Ctrl+C if it hangs)

Note: This test will likely time out or hang. That's expected. We'll implement the per-chunk timeout in the next task.

- [ ] **Step 4: Commit**

```bash
git add crates/llm-engine/src/mlx/tests.rs
git commit -m "test(mlx): add failing stall timeout test"
```

---

### Task 5: Add per-chunk timeout to stream_completion

**Files:**
- Modify: `crates/llm-engine/src/mlx.rs`
- Modify: `crates/llm-engine/src/mlx/metrics.rs`

- [ ] **Step 1: Update mlx_failure_kind_for_backend_error to detect stall errors**

In `crates/llm-engine/src/mlx.rs`, update `mlx_failure_kind_for_backend_error` to detect stall errors:

```rust
fn mlx_failure_kind_for_backend_error(err: &BackendError) -> MlxBackendFailureKind {
    if matches!(err, BackendError::Cancelled) {
        MlxBackendFailureKind::Cancelled
    } else if err.to_string().contains("stalled") {
        MlxBackendFailureKind::Stall
    } else {
        MlxBackendFailureKind::Transport
    }
}
```

- [ ] **Step 2: Replace bytes.next() select! with per-chunk timeout**

In `stream_completion`, find the `tokio::select!` block around line 161 that reads:

```rust
let item = tokio::select! {
    item = bytes.next() => Ok(item),
    _ = cancellation.cancelled() => Err(BackendError::Cancelled),
};
```

Replace with:

```rust
let item = tokio::select! {
    biased;
    _ = cancellation.cancelled() => Err(BackendError::Cancelled),
    result = tokio::time::timeout(self.timeouts.read, bytes.next()) => {
        result.map_err(|_| BackendError::Other(format!(
            "MLX stream stalled for {} without data",
            format_duration(self.timeouts.read)
        )))
    }
};
```

- [ ] **Step 3: Run the stall timeout test to verify it passes**

Run: `cargo test -p llm-engine -- mlx_backend_per_chunk_timeout_detects_stalled_stream`
Expected: PASS (the test should complete within ~200ms)

- [ ] **Step 4: Run all MLX tests**

Run: `cargo test -p llm-engine -- mlx`
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/llm-engine/src/mlx.rs crates/llm-engine/src/mlx/metrics.rs
git commit -m "feat(mlx): add per-chunk read timeout to stream_completion"
```

---

### Task 6: Add CLI flags

**Files:**
- Modify: `crates/llm-engine/src/main.rs`

- [ ] **Step 1: Parse timeout flags and pass to MlxBackendOptions**

In `main.rs`, after the `--mlx-endpoint` parsing block (around line 113), add timeout flag parsing:

```rust
let mlx_timeouts = {
    use std::time::Duration;
    let connect = flag_value(&serve_args, "--mlx-connect-timeout")
        .map(str::parse::<u64>)
        .transpose()?
        .map(Duration::from_secs);
    let request = flag_value(&serve_args, "--mlx-request-timeout")
        .map(str::parse::<u64>)
        .transpose()?
        .map(Duration::from_secs);
    let read = flag_value(&serve_args, "--mlx-read-timeout")
        .map(str::parse::<u64>)
        .transpose()?
        .map(Duration::from_secs);
    MlxTimeouts {
        connect: connect.unwrap_or(MlxTimeouts::default().connect),
        request: request.unwrap_or(MlxTimeouts::default().request),
        read: read.unwrap_or(MlxTimeouts::default().read),
    }
};
```

Note: `MlxTimeouts` needs to be exported from `llm-engine` lib.rs. Add `pub use` in `crates/llm-engine/src/mlx.rs`:

```rust
pub use client::MlxTimeouts;
```

And make `MlxTimeouts` and `build_http_client` pub(super) → update `client.rs` visibility to `pub` for `MlxTimeouts` and `format_duration`:

In `client.rs`, change `pub(super) struct MlxTimeouts` to `pub struct MlxTimeouts` and `pub(super) fn format_duration` to `pub fn format_duration`.

Update the `MlxBackendOptions` in the `serve` command to include the timeouts:

```rust
mlx: MlxBackendOptions {
    endpoint: mlx_endpoint,
    timeouts: mlx_timeouts,
    ..MlxBackendOptions::default()
},
```

- [ ] **Step 2: Update help text**

In `print_serve_help()`, add after the `--mlx-endpoint` line:

```
  --mlx-connect-timeout <secs>              MLX sidecar connect timeout [default: 5]
  --mlx-request-timeout <secs>              MLX sidecar overall request timeout [default: 300]
  --mlx-read-timeout <secs>                 MLX sidecar per-chunk read timeout [default: 60]
```

- [ ] **Step 3: Update lib.rs exports**

In `crates/llm-engine/src/lib.rs`, add `MlxTimeouts` to the public exports alongside the existing `MlxBackendOptions`:

```rust
pub use mlx::MlxTimeouts;
```

- [ ] **Step 4: Verify it compiles**

Run: `cargo check -p llm-engine`
Expected: compiles without errors

- [ ] **Step 5: Commit**

```bash
git add crates/llm-engine/src/main.rs crates/llm-engine/src/lib.rs crates/llm-engine/src/mlx.rs crates/llm-engine/src/mlx/client.rs
git commit -m "feat(mlx): add --mlx-{connect,request,read}-timeout CLI flags"
```

---

### Task 7: Run full test suite and lint

- [ ] **Step 1: Run cargo clippy**

Run: `cargo clippy --all-features`
Expected: no warnings or errors

- [ ] **Step 2: Run cargo test**

Run: `cargo test --all-features`
Expected: all tests pass

- [ ] **Step 3: Run cargo test with all targets**

Run: `cargo test --all-targets --all-features`
Expected: all tests pass

- [ ] **Step 4: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "fix: address clippy/test findings from MLX timeout implementation"
```
