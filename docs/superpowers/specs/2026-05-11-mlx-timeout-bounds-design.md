# MLX Sidecar Bounded Timeouts

**Issue:** #134
**Date:** 2026-05-11
**Status:** Approved

## Problem

The MLX sidecar backend constructs a default `reqwest::Client` (`mlx.rs:74`) and waits on upstream request send/body reads without backend-owned connect, request, or per-read timeout bounds. Non-streaming `/v1/chat/completions` and `/v1/completions` call `generate_once`, which drains the backend stream without a stall deadline. A wedged loopback MLX sidecar can hold a scheduler permit until the client disconnects or external cancellation arrives.

## Approach

Mirror the `HubTimeouts`/`HubClient` pattern from `crates/llm-hub/src/client.rs`. Introduce an `MlxTimeouts` struct with connect, request, and read durations. Build the MLX `reqwest::Client` with these bounds and wrap each `bytes.next()` call in `tokio::time::timeout`.

## Design

### 1. MlxTimeouts type

New struct in `crates/llm-engine/src/mlx/client.rs`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MlxTimeouts {
    pub connect: Duration,  // reqwest connect_timeout
    pub request: Duration,  // reqwest timeout (overall request ceiling)
    pub read: Duration,     // per-chunk tokio::time::timeout
}
```

Defaults for loopback sidecar:
- **connect:** 5s — localhost connect should be near-instant
- **request:** 300s — 5min overall ceiling covering long-context prefill
- **read:** 60s — per-chunk stall detection

A `build_http_client` helper constructs the `reqwest::Client`:

```rust
fn build_http_client(timeouts: MlxTimeouts) -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(timeouts.connect)
        .timeout(timeouts.request)
        .build()
        .expect("MLX HTTP client builds")
}
```

`MlxBackendOptions` gains a `timeouts: MlxTimeouts` field defaulting to `MlxTimeouts::default()`.

### 2. Per-chunk timeout enforcement

The `stream_completion` method's `bytes.next()` loop replaces:

```rust
let item = tokio::select! {
    item = bytes.next() => Ok(item),
    _ = cancellation.cancelled() => Err(BackendError::Cancelled),
};
```

with:

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

Key details:
- `biased` ensures cancellation always wins over timeout
- Timeout produces a `BackendError::Other` with a descriptive stall message
- `generate_once` delegates to `stream_completion`, so it inherits protection automatically
- The `upstream_request.send()` call is covered by the reqwest client's `.timeout()` setting

### 3. CLI flags

Three new flags added to the `llm-engine serve` command:

```
--mlx-connect-timeout <secs>    MLX sidecar connect timeout [default: 5]
--mlx-request-timeout <secs>    MLX sidecar overall request timeout [default: 300]
--mlx-read-timeout <secs>       MLX sidecar per-chunk read timeout [default: 60]
```

Parsed in `main.rs` into `MlxTimeouts` and passed through `MlxBackendOptions`.

### 4. Stall failure metric

Add `MlxBackendFailureKind::Stall` variant to the metrics enum in `mlx/metrics.rs`. This makes stall timeouts distinguishable from raw transport failures in the admin metrics endpoint. The `mlx_failure_kind_for_backend_error` helper maps stall errors to this variant.

## Files changed

| File | Change |
|------|--------|
| `crates/llm-engine/src/mlx/client.rs` | Add `MlxTimeouts`, `build_http_client`, `format_duration` |
| `crates/llm-engine/src/mlx.rs` | Add `timeouts` field to `MlxBackend` and `MlxBackendOptions`. Wire bounded client + per-chunk timeout in `stream_completion`. |
| `crates/llm-engine/src/mlx/metrics.rs` | Add `MlxBackendFailureKind::Stall` variant |
| `crates/llm-engine/src/main.rs` | Parse `--mlx-connect-timeout`, `--mlx-request-timeout`, `--mlx-read-timeout` CLI flags |

## Testing

- Existing `MlxBackend` tests continue to pass (defaults are permissive enough)
- New unit test: construct `MlxBackend` with `MlxTimeouts` and verify the `reqwest::Client` is configured correctly
- Integration test path: a test backend simulating a stalled sidecar should produce a stall error within the read timeout
