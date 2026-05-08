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

The model id is the served backend model id. In native Qwen mode, it comes from
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
  "python_runtime": false
}
```

Unknown aliases return `404` with `model_not_found`.

## `GET /admin/metrics`

Returns aggregate request, stream, failure, prompt-token, completion-token, and
total-token counters for the running process. The response also includes current
`active_requests`, `queued_requests`, `prefill_requests`, `decode_requests`,
cumulative admin-triggered `cancelled_requests`, cumulative `no_progress_failures`, aggregate
`request_latency_ms`, streamed `time_to_first_token_ms`, and cumulative
`tokens_per_second`. Model-store pull counters are reported separately as
`model_pull_operations`, `model_pull_successes`, `model_pull_failures`, and
`model_pull_bytes`. The response also reports manifest-backed model-store usage
as `model_store_snapshots` and `model_store_bytes`, plus cumulative
`artifact_verification_failures` from failed admin snapshot verification. Process
resident memory is exposed as `process_rss_bytes` when supported by the host OS.

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
| `messages` | array | yes | Must not be empty. |
| `tools` | array | no | Function tools only. |
| `tool_choice` | string or object | no | `auto`, `none`, `required`, or function choice object. |
| `response_format` | object | no | `{"type":"text"}` or `{"type":"json_object"}`. `json_schema` is rejected. |
| `stream` | boolean | no | Defaults to `false`. |
| `stream_options.include_usage` | boolean | no | Defaults to `false`. |
| `temperature` | number | no | Must be finite and non-negative. `0` selects greedy decode. |
| `top_p` | number | no | Must be finite and in `(0, 1]`. |
| `max_tokens` | integer | no | Runtime default is `4096`; native backend may cap lower with `--max-new-tokens`. Must be greater than `0`. |
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
fails with `unsupported_capability`.

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
emitting deltas to preserve fail-closed semantics.

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
| `max_tokens` | integer | no | Runtime default is `4096`; must be greater than `0`. |
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
| `backend_execution_failed` | `500` | `decode` | `true` |
| `cancelled` | `408` | `decode` | `false` |
| `request_not_found` | `404` | `cancellation` | `false` |
| `request_id_conflict` | `409` | `request_validation` | `false` |
| `chat_template_failed` | `422` | `prompt_rendering` | `false` |
| `malformed_tool_call` | `422` | `response_parsing` | `false` |
| `json_validation_failed` | `422` | `response_validation` | `false` |
| `no_progress` | `422` | `response_validation` | `false` |
| `response_serialization_failed` | `500` | `response_serialization` | `true` |
