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
| `GET` | `/metrics` | Prometheus text metrics scrape, including paged-KV counters when exposed by the backend. |
| `GET` | `/v1/models` | OpenAI-compatible model list. |
| `GET` | `/admin/models` | Read-only status for served model aliases. |
| `GET` | `/admin/models/{alias}` | Read-only status for one served model alias. |
| `GET` | `/admin/metrics` | Aggregate inference counters and token totals. |
| `GET` | `/admin/kv-cache` | Detailed paged-KV cache snapshot when the backend exposes one. |
| `GET` | `/admin/metrics.tool_stream` | Bounded per-request streamed tool-call timing observations. |
| `POST` | `/admin/models/{alias}/verify` | Run runnable verification for the served snapshot. |
| `POST` | `/admin/models/{alias}/plan` | Build a download plan for the served model alias. |
| `POST` | `/admin/models/{alias}/pull` | Pull and promote a snapshot into the configured model store. |
| `POST` | `/admin/requests/{request_id}/cancel` | Cancel an active chat or completion request. |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions. |
| `POST` | `/v1/completions` | OpenAI-compatible legacy text completions. |

Admin routes are local operational controls and require `Authorization: Bearer <token>`.
Use `--admin-token` or `LLM_ENGINE_ADMIN_TOKEN` to set a stable token. If neither
is set on a loopback bind, `serve` generates a temporary token for that process
and prints the required header at startup. Non-loopback serving refuses to start
unless an admin token is configured.

> [!TIP]
> JSON schemas for all admin API responses are available in [`docs/schemas/admin/`](./schemas/admin/).

## `GET /health`

Healthy response (`200 OK`):

```json
{
  "status": "ok",
  "runtime": "rust",
  "python_runtime": false
}
```

If the active backend reports itself unavailable, the endpoint returns
`503 Service Unavailable` with the same top-level shape and backend details:

```json
{
  "status": "unavailable",
  "runtime": "rust",
  "python_runtime": false,
  "backend": {
    "status": "unavailable",
    "model": "local-qwen36",
    "backend": "mlx",
    "reason": "MLX model list returned HTTP 503 Service Unavailable"
  }
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

## `GET /metrics`

Returns Prometheus text exposition for numeric server and backend metrics. This
route uses the same admin authentication policy as `/admin/metrics`.

Paged-KV cache metrics are emitted with the `kir_paged_kv_cache_` prefix when
the active backend exposes a KV cache snapshot. Examples include:

```text
kir_paged_kv_cache_resident_blocks 12
kir_paged_kv_cache_active_blocks 18
kir_paged_kv_cache_shared_blocks 4
kir_paged_kv_cache_total_cow_clones 2
kir_paged_kv_cache_cow_bytes_saved 1048576
kir_paged_kv_cache_blocks_evicted_lru 3
kir_paged_kv_cache_pool_utilization_pct 75
```

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
- `scheduler_prefill_yields`: Streaming prefill chunk boundaries where the scheduler successfully released and reacquired prefill admission.
- `scheduler_prefill_yields_to_decode`: Successful prefill yields where at least one queued decode request was admitted before prefill readmission.
- `scheduler_prefill_yield_reacquire_waits`: Successful prefill yield reacquisitions.
- `scheduler_prefill_yield_reacquire_wait_ms_total`: Total scheduler wait time, in milliseconds, for successful prefill yield reacquisitions.
- `scheduler_prefill_yield_reacquire_wait_ms_max`: Maximum scheduler wait time, in milliseconds, for a successful prefill yield reacquisition.
- `scheduler_prefill_chunk_latency_ms`: Summary of server-observed latency for streamed prefill progress chunks.
- `scheduler_decode_starvation_events`: Decode requests that entered the scheduler queue behind active or queued prefill work.
- `scheduler_decode_starvation_waits`: Starved decode requests that were eventually admitted.
- `scheduler_decode_starvation_wait_ms_total`: Total scheduler wait time, in milliseconds, for starved decode requests that were admitted.
- `scheduler_decode_starvation_wait_ms_max`: Maximum scheduler wait time, in milliseconds, for a starved decode request that was admitted.
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
  `prefill_tokens`, `checkpoint_stores`, `checkpoint_store_tokens`,
  `checkpoint_reuse_hits`, `checkpoint_reused_tokens`, `shared_prefix_hits`,
  `shared_prefix_reused_tokens`, `hit_tokens`, `miss_tokens`, and
  `avoided_prefill_tokens` so warm-prefix runs can distinguish full hits,
  shared agent-prefix reuse, checkpoint reuse, and avoided prefill work. Native
  text prefix-cache routing keys are fail-closed across model/backend/family,
  repo/commit/profile, tokenizer kind, tokenizer hash and normalization version,
  chat-template id and kwargs hash, adapter settings, request mode, cache layout, and
  cache-token bucket before KV state is reused. Native text Metal KV cache
  metrics include resident bytes/buffers, allocations, syncs, skipped syncs,
  bytes uploaded, evictions, and bytes evicted when Metal support is compiled in, plus
  `f32_*`, `f16_*`, and `int8_*` uploaded/resident byte breakdowns for cache
  precision comparisons.
- `request_cache`: Bounded per-request prefix-cache observations. `capacity`
  is fixed at `128`; `recent` contains successful buffered and streaming
  requests with `request_id`, `model`, `streamed`, `prompt_tokens`,
  `cached_tokens`, `uncached_tokens`, `cache_status`, `prompt_hash`,
  `cache_key`, `cache_template_id`, `model_family`, `tool_schema_hash`,
  `system_prompt_hash`, `chat_template_kwargs_hash`, `stable_prefix_key`, and
  `latency_ms`.
  `cache_status` is `unknown` when cached-token details are absent and no
  prior matching Kir cache identity is available, `miss` when cached tokens are
  `0`, `hit` when cached tokens are greater than or equal to prompt tokens, and
  `partial` otherwise. If upstream usage omits cached-token details, Kir derives
  a best-effort `partial`/`hit` status from prior observations with the same
  stable prefix identity; derived partial observations leave token counts
  absent. Identity fields are hashes or template identifiers only: prompts,
  messages, tool schemas, and
  request bodies are not stored. `stable_prefix_key` is a versioned hash over
  model family, chat template ID, tool schema hash, system prompt hash, and
  chat template kwargs hash so repeated agent turns can be grouped without
  exposing prompt content.
- `tool_stream`: Bounded per-request streamed tool-call timing observations.
  `capacity` is fixed at `128`; `recent` contains successful streamed tool
  requests keyed by `request_id` with scalar timing fields only. Kir-visible
  milestones use `kir_first_tool_delta_ms`,
  `kir_first_tool_delta_after_ttft_ms`, `tool_argument_assembly_ms`,
  `tool_intent_fill_ms`, `tool_schema_validation_ms`, `tool_finish_ms`, and
  `validated_tool_call_ms`. MLX upstream milestones use
  `mlx_response_headers_ms`, `mlx_first_upstream_byte_ms`,
  `mlx_first_parsed_chunk_ms`, `mlx_first_tool_delta_ms`, and
  `mlx_upstream_complete_ms`. Prompts, messages, tool schemas, tool arguments,
  and request bodies are not stored.
- `request_latency_ms`: Summary (count, min, max, avg) of total outer kir-ai
  request duration.
- `non_streamed_request_latency_ms`: Summary of outer kir-ai request duration
  for non-streaming responses.
- `streamed_request_latency_ms`: Summary of outer kir-ai request duration for
  streaming responses.
- `time_to_first_token_ms`: Summary of latency to the first token generated.
- `first_tool_delta_ms`: Summary of end-to-end latency from request start to
  the first streamed tool-call delta.
- `first_tool_delta_after_ttft_ms`: Summary of latency from the first real
  streamed decode delta, the same point recorded by `time_to_first_token_ms`, to
  the first streamed tool-call delta. This excludes scheduler wait and prompt
  prefill time so long-context runs can separate prefill cost from tool assembly
  latency.
- `tool_argument_assembly_ms`, `tool_intent_fill_ms`,
  `tool_schema_validation_ms`, `tool_finish_ms`, and
  `validated_tool_call_ms`: Summaries of tool-call validation lifecycle stages.
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
        "prompt_hash": "sha256:3b7c...",
        "cache_key": "sha256:cb45...",
        "cache_template_id": "chatml/qwen/v1",
        "model_family": "qwen",
        "tool_schema_hash": "sha256:07af...",
        "system_prompt_hash": "sha256:911d...",
        "chat_template_kwargs_hash": "sha256:38ff...",
        "stable_prefix_key": "sha256:fb52...",
        "latency_ms": 95
      }
    ]
  },
  "tool_stream": {
    "capacity": 128,
    "recent": [
      {
        "request_id": "req-456",
        "model": "local-qwen36-mlx",
        "streamed": true,
        "kir_first_tool_delta_ms": 576,
        "tool_argument_assembly_ms": 582,
        "tool_intent_fill_ms": 584,
        "tool_schema_validation_ms": 588,
        "validated_tool_call_ms": 590,
        "mlx_response_headers_ms": 80,
        "mlx_first_upstream_byte_ms": 120,
        "mlx_first_parsed_chunk_ms": 180,
        "mlx_first_tool_delta_ms": 560,
        "mlx_upstream_complete_ms": 585,
        "latency_ms": 610
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

## `GET /admin/kv-cache`

Returns the active backend's detailed paged-KV cache snapshot. Backends that
own a paged-KV block pool can include per-session block tables, block refcounts,
resident/free/shared block counts, COW clone counters, COW bytes saved,
eviction counters and high-water marks, and pool utilization. The detailed
session and block arrays are intended for explicit admin inspection; hot request
paths update counters only.

If the backend does not expose a live paged-KV snapshot, the response is still
stable:

```json
{
  "object": "kv_cache.snapshot",
  "supported": false,
  "reason": "backend did not expose a paged KV cache snapshot",
  "metrics": {},
  "sessions": [],
  "layers": [],
  "blocks": []
}
```

Example supported snapshot shape:

```json
{
  "object": "kv_cache.block_pool",
  "metrics": {
    "total_blocks": 128,
    "resident_blocks": 96,
    "active_blocks": 140,
    "free_list_blocks": 32,
    "shared_blocks": 24,
    "refcount_total": 140,
    "max_refcount_seen": 4,
    "total_cow_clones": 6,
    "cow_bytes_saved": 2097152,
    "blocks_evicted_lru": 12,
    "pool_utilization_pct": 75.0
  },
  "sessions": [
    {
      "session_id": 7,
      "block_table": [
        {
          "index": 0,
          "block_id": 42,
          "ref_count": 2,
          "token_count": 256
        }
      ],
      "owned_blocks": [42]
    }
  ],
  "blocks": [
    {
      "block_id": 42,
      "ref_count": 2,
      "token_count": 256,
      "resident_bytes": 1048576
    }
  ]
}
```

## `GET /admin/metrics.tool_stream`

Returns the bounded per-request streamed tool-call timing ring directly. This
is the same `tool_stream` snapshot embedded in `GET /admin/metrics`, without
the aggregate counters around it.

### Response Fields

- `capacity`: Maximum number of retained observations. This is fixed at `128`.
- `recent`: Successful streamed tool-call observations keyed by `request_id`,
  ordered from oldest to newest retained entry.
- Each observation includes `request_id`, `model`, `streamed`, scalar Kir
  timing milestones, scalar MLX upstream timing milestones when available, and
  `latency_ms`.
- Prompts, messages, tool schemas, tool arguments, and request bodies are not
  stored.

### Sample Response

```json
{
  "capacity": 128,
  "recent": [
    {
      "request_id": "req-456",
      "model": "local-qwen36-mlx",
      "streamed": true,
      "kir_first_tool_delta_ms": 576,
      "tool_argument_assembly_ms": 582,
      "tool_intent_fill_ms": 584,
      "tool_schema_validation_ms": 588,
      "validated_tool_call_ms": 590,
      "mlx_response_headers_ms": 80,
      "mlx_first_upstream_byte_ms": 120,
      "mlx_first_parsed_chunk_ms": 180,
      "mlx_first_tool_delta_ms": 560,
      "mlx_upstream_complete_ms": 585,
      "latency_ms": 610
    }
  ]
}
```

## `POST /admin/models/{alias}/verify`

Runs runnable verification for the currently served snapshot. This uses the same
semantics as `llm-engine model verify`: manifest file integrity plus config,
tokenizer, weight artifact, built-in profile, and safetensors index checks.
Metadata-only snapshots fail with `model_integrity_failed` and should be pulled
again without `--metadata-only` before serving.

Successful responses include `verification_mode: "runnable"` with verified file
and byte counts. This endpoint requires the served backend to expose snapshot
metadata.

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
| `temperature` | number | no | Must be finite and in `[0, 2]`. `0` selects greedy decode; omitted uses the default `1`. |
| `top_p` | number | no | Must be finite and in `(0, 1]`. Omitted uses the default `1`. |
| `max_tokens` | integer | no | Omitted values use the backend default. Native text defaults to `256` and caps requests with `--max-new-tokens`. Must be greater than `0`. |
| `stop` | string or string array | no | Empty strings are rejected. |

Supported roles:

- `system`
- `user`
- `assistant`
- `tool`

Sampling controls are request-level controls. The runtime maps `temperature: 0`
to greedy backend sampling and otherwise uses top-p sampling with OpenAI defaults
for omitted controls. Backends that do not advertise the requested sampling
capability fail closed with `unsupported_capability`.

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

Streaming is response-shape streaming over the backend stream contract. Native
text paths can forward backend chunks incrementally during decode; the protocol
test backend and default stream adapter may produce a single backend chunk after
non-streaming generation. Tool-call and JSON-object validation paths may buffer
before emitting deltas to preserve fail-closed semantics. Runtime errors that
happen after an SSE stream starts are emitted as `data:` error objects with
stable `code`, `phase`, and `retryable` fields, followed by `[DONE]`. Runtime
SSE backend-execution error messages are intentionally generic for clients;
request-validation errors may include sanitized model/backend attribution.
Detailed internal diagnostics stay in server logs.

## `POST /v1/completions`

Request fields:

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `model` | string | yes | Must match the served model id. |
| `prompt` | string | yes | Must not be empty. |
| `stream` | boolean | no | Defaults to `false`. |
| `stream_options.include_usage` | boolean | no | Defaults to `false`. |
| `temperature` | number | no | Must be finite and in `[0, 2]`. `0` selects greedy decode; omitted uses the default `1`. |
| `top_p` | number | no | Must be finite and in `(0, 1]`. Omitted uses the default `1`. |
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

Public inference endpoints (`/v1/chat/completions` and `/v1/completions`) use a per-client sliding-window rate limit. Client identity is selected from the first non-empty `X-Forwarded-For` address, then `X-Real-IP`, then the `Authorization` header; requests without those headers share an anonymous bucket. Rate-limited responses return `429`, include `Retry-After`, and do not parse or validate the request body. Public inference responses include `x-ratelimit-limit-requests`, `x-ratelimit-remaining-requests`, and `x-ratelimit-reset-requests`.

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

Client-visible error messages are sanitized so absolute filesystem paths are
reported as `[redacted path]`; detailed path context remains server-side.

When an error happens after an SSE stream has started, the HTTP status remains
`200` because response headers have already been sent. Clients should read the
structured `data:` error object and use `code`, `phase`, and `retryable` for
programmatic handling.

Known codes:

| Code | Typical status | Phase | Retryable |
| --- | --- | --- | --- |
| `invalid_request` | `400` | `request_validation` | `false` |
| `unsupported_capability` | `400` | `request_validation` | `false` |
| `model_not_found` | `404` | `model_resolution` | `false` |
| `rate_limited` | `429` | `rate_limit` | `true` |
| `model_overloaded` | `429` | `scheduler` | `true` |
| `backend_execution_failed` | `500` | `decode` | `true` |
| `model_integrity_failed` | `422` | `model_artifact_verification` | `false` |
| `model_artifact_missing` | `422` | `model_artifact_verification` | `false` |
| `tokenizer_failed` | `422` | `tokenization` | `false` |
| `sampler_failed` | `422` | `decode` | `false` |
| `metal_backend_failed` | `503` | `decode` | `true` |
| `backend_config_failed` | `422` | `model_configuration` | `false` |
| `backend_invariant_failed` | `500` | `decode` | `false` |
| `cancelled` | `408` | `decode` | `false` |
| `request_not_found` | `404` | `cancellation` | `false` |
| `request_id_conflict` | `409` | `request_validation` | `false` |
| `admin_auth_required` | `401` | `admin_auth` | `false` |
| `chat_template_failed` | `422` | `prompt_rendering` | `false` |
| `malformed_tool_call` | `422` | `response_parsing` | `false` |
| `unsupported_multimodal_output` | `422` | `response_parsing` | `false` |
| `json_validation_failed` | `422` | `response_validation` | `false` |
| `tool_call_validation_failed` | `422` | `response_validation` | `false` |
| `no_progress_empty_completion` | `422` | `response_validation` | `false` |
| `no_progress_empty_high_output_completion` | `422` | `response_validation` | `false` |
| `no_progress_hidden_only_output` | `422` | `response_validation` | `false` |
| `no_progress_missing_required_tool_call` | `422` | `response_validation` | `false` |
| `no_progress_repeated_invalid_tool_call` | `422` | `response_validation` | `false` |
| `no_progress_fuzzy_repeated_invalid_tool_call` | `422` | `response_validation` | `false` |
| `no_progress_repeated_assistant_content` | `422` | `response_validation` | `false` |
| `no_progress_stalled_assistant_turn` | `422` | `response_validation` | `false` |
| `stream_stalled` | `200` (SSE) | `streaming` | `true` |
| `stream_incomplete` | `200` (SSE) | `streaming` | `true` |
| `response_serialization_failed` | `200` (SSE) | `response_serialization` | `true` |
