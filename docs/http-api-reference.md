# HTTP API Reference

`llm-engine` exposes a small OpenAI-compatible HTTP surface plus admin endpoints
for model status, metrics, snapshot verification, planning, pulls, and active
request cancellation.

Base URL in examples:

```text
http://127.0.0.1:3000
```

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Server health and runtime identity. |
| `GET` | `/v1/models` | OpenAI-compatible model list. |
| `GET` | `/admin/models` | Read-only status for served model aliases. |
| `GET` | `/admin/models/{alias}` | Read-only status for one served model alias. |
| `GET` | `/admin/metrics` | Aggregate inference counters and token totals. |
| `POST` | `/admin/models/{alias}/verify` | Verify the served snapshot from its manifest. |
| `POST` | `/admin/models/{alias}/plan` | Build a download plan for the served model alias. |
| `POST` | `/admin/models/{alias}/pull` | Pull and promote a snapshot into the configured model store. |
| `POST` | `/admin/requests/{request_id}/cancel` | Cancel an active chat or completion request. |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions. |
| `POST` | `/v1/completions` | OpenAI-compatible legacy text completions. |

Admin routes are local operational controls. Use `--admin-token` or
`LLM_ENGINE_ADMIN_TOKEN` to require `Authorization: Bearer <token>`; non-loopback
serving refuses to start unless an admin token is configured.

> [!TIP]
> JSON schemas for all admin API responses are available in [`docs/schemas/admin/`](./schemas/admin/).

## `GET /health`

Response:

```json
{
  "status": "ok",
  "runtime": "rust",
  "python_runtime": false
}
```

## `GET /v1/models`

Response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "local-qwen36",
      "object": "model",
      "owned_by": "local"
    }
  ]
}
```

The model id is the served backend model id. In native text mode, it comes from
`--model-id`.

## `GET /admin/models`

Response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "local-qwen36",
      "object": "admin.model",
      "status": "ready",
      "runtime": "rust",
      "python_runtime": false
    }
  ]
}
```

## `GET /admin/models/{alias}`

Response for a loaded alias:

```json
{
  "id": "local-qwen36",
  "object": "admin.model",
  "status": "ready",
  "runtime": "rust",
  "python_runtime": false,
  "backend": "native_qwen",
  "family": "qwen",
  "loader": "safetensors",
  "quantization": "bf16",
  "repo_id": "michaelasper/qwen36-safetensors-bf16",
  "resolved_commit": "...",
  "profile": "qwen36-safetensors-bf16",
  "snapshot_path": "/Users/...",
  "manifest_digest": "sha256:..."
}
```

Unknown aliases return `404` with `model_not_found`.

## `GET /admin/metrics`

Returns aggregate request, stream, failure, token, and scheduler counters for the running process.

### Response Fields

- `requests_total`: Total number of requests received.
- `successful_requests`: Number of requests that completed successfully.
- `failed_requests`: Number of requests that failed.
- `streamed_requests`: Number of streaming requests.
- `stream_client_disconnected_requests`: Number of streaming requests where the client disconnected early.
- `stream_stalled_requests`: Number of streaming requests that stalled.
- `active_requests`: Current number of active requests.
- `queued_requests`: Total number of requests in the scheduler queue.
- `queued_prefill_requests`: Number of prefill requests in the queue.
- `queued_decode_requests`: Number of decode requests in the queue.
- `prefill_requests`: Total cumulative prefill operations.
- `decode_requests`: Total cumulative decode operations.
- `active_prefill_requests`: Current number of requests in prefill phase.
- `active_decode_requests`: Current number of requests in decode phase.
- `scheduler_admitted_prefill_requests`: Requests admitted to prefill by the scheduler.
- `scheduler_admitted_decode_requests`: Requests admitted to decode by the scheduler.
- `scheduler_completed_requests`: Requests completed by the scheduler.
- `scheduler_cancelled_requests`: Requests cancelled while active in the scheduler.
- `scheduler_failed_requests`: Requests that failed while active in the scheduler.
- `scheduler_queued_cancelled_requests`: Requests cancelled while still in the queue.
- `scheduler_queue_timeouts`: Requests that timed out while in the queue.
- `cancelled_requests`: Cumulative admin-triggered cancellations.
- `no_progress_failures`: Cumulative failures due to lack of progress.
- `model_pull_operations`: Total model pull attempts.
- `model_pull_successes`: Successful model pulls.
- `model_pull_failures`: Failed model pulls.
- `model_pull_bytes`: Total bytes downloaded during model pulls.
- `model_store_snapshots`: Number of model snapshots in the local store.
- `model_store_bytes`: Total disk usage of the model store.
- `model_store_quarantined_snapshots`: Number of quarantined snapshots.
- `model_store_quarantined_bytes`: Disk usage of quarantined snapshots.
- `artifact_verification_failures`: Cumulative checksum/signature verification failures.
- `process_rss_bytes`: Resident set size of the process (when supported).
- `tokens_per_second`: Current aggregate throughput.
- `backend_metrics`: Platform-specific backend metrics keyed by compiled
  backend/compatibility name. Common keys include `mlx`,
  `native_text_metal`, `native_text_prefix_cache`, `native_qwen_metal`, and
  `native_qwen_prefix_cache`, when those backends are compiled in. For MLX
  sidecar diagnostics, `backend_metrics.mlx.upstream_request_latency_ms`
  reports sidecar request duration, while
  `backend_metrics.mlx.blocking_upstream_request_latency_ms` and
  `backend_metrics.mlx.streaming_upstream_request_latency_ms` split that
  duration by kir-ai blocking versus streaming generation path. Native text
  prefix-cache objects include cache counters plus `prefill_chunks`,
  `prefill_tokens`, `hit_tokens`, `miss_tokens`, and
  `avoided_prefill_tokens` so warm-prefix runs can distinguish hit rate from
  avoided prefill work.
- `request_cache`: Bounded per-request prefix-cache observations. `capacity`
  is fixed at `128`; `recent` contains successful buffered and streaming
  requests with `request_id`, `model`, `streamed`, `prompt_tokens`,
  `cached_tokens`, `uncached_tokens`, `cache_status`, and `latency_ms`.
  `cache_status` is `unknown` when upstream cached-token details are absent,
  `miss` when cached tokens are `0`, `hit` when cached tokens are greater than
  or equal to prompt tokens, and `partial` otherwise. Prompts, messages, tool
  schemas, and request bodies are not stored.
- `request_latency_ms`: Summary (count, min, max, avg) of total outer kir-ai
  request duration.
- `non_streamed_request_latency_ms`: Summary of outer kir-ai request duration
  for non-streaming responses.
- `streamed_request_latency_ms`: Summary of outer kir-ai request duration for
  streaming responses.
- `time_to_first_token_ms`: Summary of latency to the first token generated.
- `tokens`: Token usage summary (`prompt_tokens`, `completion_tokens`, `total_tokens`).

### Sample Response

```json
{
  "requests_total": 150,
  "successful_requests": 145,
  "failed_requests": 5,
  "streamed_requests": 120,
  "active_requests": 2,
  "queued_requests": 0,
  "tokens_per_second": 45.5,
  "request_latency_ms": {
    "count": 145,
    "min": 10.5,
    "max": 500.2,
    "avg": 120.4
  },
  "request_cache": {
    "capacity": 128,
    "recent": [
      {
        "request_id": "req-123",
        "model": "local-qwen36-mlx",
        "streamed": true,
        "prompt_tokens": 2048,
        "cached_tokens": 1792,
        "uncached_tokens": 256,
        "cache_status": "partial",
        "latency_ms": 95
      }
    ]
  },
  "tokens": {
    "prompt_tokens": 5000,
    "completion_tokens": 15000,
    "total_tokens": 20000
  }
}
```

## `POST /admin/models/{alias}/verify`

Verifies the currently served snapshot against its `llm-engine-manifest.json`
and returns verified file and byte counts. This endpoint requires the served
backend to expose snapshot metadata.

## `POST /admin/models/{alias}/plan`

Builds a Hugging Face download plan for the served model alias using the
configured model profile and hub endpoint. This does not mutate the model store.

## `POST /admin/models/{alias}/pull`

Downloads, verifies, and promotes a snapshot through the configured model store.
This is a mutating admin operation and should be exposed only behind the admin
Bearer token.

## `POST /admin/requests/{request_id}/cancel`

Cancels an active chat or text-completion request by request ID. Clients may set
`x-request-id` or `x-llm-request-id` on `/v1/chat/completions` and
`/v1/completions`; otherwise the server assigns an `x-request-id` response
header. Active cancellation returns:

```json
{
  "object": "admin.request_cancellation",
  "request_id": "cancel-me",
  "status": "cancelled"
}
```

Unknown or already-finished request IDs return `404` with `request_not_found`.

## `POST /v1/chat/completions`

Request fields:

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `model` | string | yes | Must match the served model id. |
| `messages` | array | yes | Must not be empty. System messages must appear first. User/system/tool messages require `content`; assistant messages require either `content` or `tool_calls`. Tool results require `tool_call_id` and must answer pending assistant `tool_calls`. |
| `tools` | array | no | Function tools only. |
| `tool_choice` | string or object | no | `auto`, `none`, `required`, or function choice object. |
| `response_format` | object | no | `{"type":"text"}` or `{"type":"json_object"}`. `json_schema` is rejected. |
| `stream` | boolean | no | Defaults to `false`. |
| `stream_options.include_usage` | boolean | no | Defaults to `false`. |
| `temperature` | number | no | Must be finite and non-negative. `0` selects greedy decode. |
| `top_p` | number | no | Must be finite and in `(0, 1]`. |
| `max_tokens` | integer | no | Omitted values use the backend default. Native text defaults to `256` and caps requests with `--max-new-tokens`. Must be greater than `0`. |
| `stop` | string or string array | no | Empty strings are rejected. |

Supported roles:

- `system`
- `user`
- `assistant`
- `tool`

Example:

```sh
curl -s http://127.0.0.1:3000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "local-qwen36",
    "messages": [{"role": "user", "content": "hello"}],
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 8
  }' | jq
```

Response shape:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1760000000,
  "model": "local-qwen36",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "hello from rust native backend"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 8,
    "completion_tokens": 5,
    "total_tokens": 13
  }
}
```

## Chat Tools

Tool definitions use OpenAI function-tool shape:

```json
{
  "type": "function",
  "function": {
    "name": "lookup",
    "description": "Look up a value",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {"type": "string"}
      },
      "required": ["query"]
    }
  }
}
```

Tool `parameters` must be JSON objects. The local schema validator accepts
`type` values `object`, `array`, `string`, `boolean`, `null`, `integer`, and
`number`, including string arrays such as `["string", "null"]`. Nested
`properties` entries and `items` schemas must also be JSON objects. Unknown
types or malformed supported keywords fail with `invalid_request` during request
validation.

`tool_choice` may be:

```json
"auto"
```

```json
"none"
```

```json
"required"
```

```json
{"type": "function", "function": {"name": "lookup"}}
```

If a function choice names a tool that was not declared in `tools`, the request
fails with `invalid_request`. `unsupported_capability` is reserved for supported
request fields or controls whose requested mode is not implemented by the local
server.

Generated Qwen tool calls are parsed from either:

```text
<tool_call>{"name":"lookup","arguments":{"query":"rust"}}</tool_call>
```

or:

```text
<tool_call><function=bash><parameter=cmd>cargo test --workspace</parameter></function></tool_call>
```

Parsed tool-call arguments must be JSON objects.

## JSON Object Mode

Use:

```json
{"response_format": {"type": "json_object"}}
```

The runtime validates assistant content as a JSON object. Tool calls also satisfy
JSON object mode when their arguments are objects. Invalid JSON content returns
`422` with `json_validation_failed`.

`json_schema` is not supported.

## Chat Streaming

Request:

```sh
curl -N http://127.0.0.1:3000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "local-qwen36",
    "messages": [{"role": "user", "content": "hello"}],
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

SSE chunks use `data:` lines. The stream emits:

1. A role delta.
2. A content delta or tool-call deltas.
3. A final choice chunk with `finish_reason`.
4. An optional usage-only chunk when `include_usage` is true.
5. Exactly one `data: [DONE]`.

Streaming is response-shape streaming. Text paths can forward backend chunks
incrementally; tool-call and JSON-object validation paths may buffer before
emitting deltas to preserve fail-closed semantics. Runtime errors that happen
after an SSE stream starts are emitted as `data:` error objects with stable
`code`, `phase`, and `retryable` fields, followed by `[DONE]`. Runtime SSE
error messages are intentionally generic for clients; detailed diagnostics stay
in server logs.

## `POST /v1/completions`

Request fields:

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `model` | string | yes | Must match the served model id. |
| `prompt` | string | yes | Must not be empty. |
| `stream` | boolean | no | Defaults to `false`. |
| `stream_options.include_usage` | boolean | no | Defaults to `false`. |
| `temperature` | number | no | Must be finite and non-negative. `0` selects greedy decode. |
| `top_p` | number | no | Must be finite and in `(0, 1]`. |
| `max_tokens` | integer | no | Omitted values use the backend default. Native text defaults to `256` and caps requests with `--max-new-tokens`. Must be greater than `0`. |
| `stop` | string or string array | no | Empty strings are rejected. |

Example:

```sh
curl -s http://127.0.0.1:3000/v1/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "local-qwen36",
    "prompt": "hello",
    "max_tokens": 8,
    "stop": " backend"
  }' | jq
```

Response shape:

```json
{
  "id": "cmpl-...",
  "object": "text_completion",
  "created": 1760000000,
  "model": "local-qwen36",
  "choices": [
    {
      "text": "hello from rust native",
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 1,
    "completion_tokens": 5,
    "total_tokens": 6
  }
}
```

## Text Completion Streaming

`/v1/completions` supports the same SSE conventions as chat streaming, with
`text_completion` chunk objects and one final `[DONE]`.

## Error Shape

Public inference endpoints (`/v1/chat/completions` and `/v1/completions`) are rate-limited globally. When the limit is exceeded, the server returns `429`, includes `Retry-After`, and does not parse or validate the request body.

All engine errors use this body shape:

```json
{
  "error": {
    "message": "...",
    "code": "invalid_request",
    "phase": "request_validation",
    "retryable": false,
    "type": "llm_engine_error"
  }
}
```

Known codes:

| Code | Typical status | Phase | Retryable |
| --- | --- | --- | --- |
| `invalid_request` | `400` | `request_validation` | `false` |
| `unsupported_capability` | `400` | `request_validation` | `false` |
| `model_not_found` | `404` | `model_resolution` | `false` |
| `rate_limited` | `429` | `rate_limit` | `true` |
| `backend_execution_failed` | `500` | `decode` | `true` |
| `cancelled` | `408` | `decode` | `false` |
| `request_not_found` | `404` | `cancellation` | `false` |
| `request_id_conflict` | `409` | `request_validation` | `false` |
| `chat_template_failed` | `422` | `prompt_rendering` | `false` |
| `malformed_tool_call` | `422` | `response_parsing` | `false` |
| `json_validation_failed` | `422` | `response_validation` | `false` |
| `no_progress` | `422` | `response_validation` | `false` |
| `response_serialization_failed` | `500` | `response_serialization` | `true` |
