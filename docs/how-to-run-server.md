# How To Run The Server

This guide shows the practical server modes: deterministic protocol mode and
native Qwen snapshot mode.

## Run Protocol Mode

Use protocol mode when you want fast, repeatable OpenAI-compatible responses
without model artefacts:

```sh
cargo run -p llm-engine -- serve \
  --addr 127.0.0.1:3000 \
  --deterministic-test-backend
```

With `--deterministic-test-backend` and no `--snapshot`, the server uses a
deterministic Rust backend. It serves the model alias `local-qwen36` and returns
the fixed text:

```text
hello from rust native backend
```

Omitting both `--snapshot` and `--deterministic-test-backend` exits with an
explicit backend requirement.

Confirm the server:

```sh
curl -s http://127.0.0.1:3000/health | jq
curl -s http://127.0.0.1:3000/v1/models | jq
```

## Run Native Qwen Mode

Use native Qwen mode when you have a complete local Qwen snapshot containing:

- `config.json`
- `tokenizer.json`
- `model.safetensors.index.json`
- all safetensors shards referenced by `weight_map`

Start the server with the snapshot path:

```sh
SNAPSHOT=.llm-models/huggingface/models--Qwen--Qwen3.6-35B-A3B/snapshots/<resolved-commit>

cargo run -p llm-engine -- serve \
  --addr 127.0.0.1:3000 \
  --snapshot "$SNAPSHOT" \
  --model-id local-qwen36 \
  --max-new-tokens 1 \
  --max-prefill-tokens 32 \
  --native-metal-weight-cache-bytes 8589934592
```

The native path tokenises the rendered prompt, keeps a bounded tail of prompt
tokens, runs Qwen prefill, applies final norm and LM-head top-k, then returns
decoded text.

## Choose Generation Bounds

`--max-new-tokens` caps native generation per request. It defaults to `1` and is
clamped to at least `1`.

`--max-prefill-tokens` controls how many recent prompt tokens are retained for
native prefill. It defaults to `32` and is clamped to at least `1`.

`--native-metal-weight-cache-bytes` controls the per-backend LRU budget for
uploaded Metal BF16 weight buffers. It defaults to `8589934592` bytes and can be
set to `0` to disable weight-buffer caching.

`--warm-native-metal-weight-cache` preloads rank-2 BF16 tensors into that cache
at startup until the configured budget is full. Leave it off when you want
minimum startup time or when first-request latency is not the bottleneck.

Use small values while probing correctness:

```sh
cargo run -p llm-engine -- serve \
  --snapshot "$SNAPSHOT" \
  --max-new-tokens 1 \
  --max-prefill-tokens 8
```

Use larger values only when you expect the current CPU-bound path to spend more
time in prefill:

```sh
cargo run -p llm-engine -- serve \
  --snapshot "$SNAPSHOT" \
  --max-new-tokens 4 \
  --max-prefill-tokens 64
```

## Call Chat Completions

```sh
curl -s http://127.0.0.1:3000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "local-qwen36",
    "messages": [{"role": "user", "content": "Say the word test."}],
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 1
  }' | jq
```

The request `model` must match `--model-id`. `temperature: 0` selects greedy
decode. Non-greedy native Qwen sampling accepts finite non-negative
`temperature` and `top_p` in `(0, 1]`.

## Call Text Completions

```sh
curl -s http://127.0.0.1:3000/v1/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "local-qwen36",
    "prompt": "Say the word test.",
    "max_tokens": 1
  }' | jq
```

Use the legacy text completion endpoint when the caller already owns prompt
rendering and does not need chat roles or tools.

## Stream Responses

```sh
curl -N http://127.0.0.1:3000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "local-qwen36",
    "messages": [{"role": "user", "content": "hello"}],
    "stream": true,
    "stream_options": {"include_usage": true},
    "max_tokens": 1
  }'
```

The server emits JSON SSE chunks and then one `data: [DONE]` terminator. When
`stream_options.include_usage` is `true`, the usage-only chunk appears before
`[DONE]`.

Streaming is currently assembled after backend generation. It preserves the
OpenAI-compatible response shape, but it is not token-by-token decode streaming.

## Inspect Admin Status

```sh
curl -s http://127.0.0.1:3000/admin/models | jq
curl -s http://127.0.0.1:3000/admin/models/local-qwen36 | jq
```

`GET /admin/models` and `GET /admin/models/{alias}` are read-only status
endpoints. The `/admin/*` surface also includes metrics, snapshot verification,
download planning, snapshot pulls, and active request cancellation;
`/admin/models/{alias}/pull` mutates the configured model store.

To make a request cancellable by a known ID, send `x-request-id` on the
inference call, then cancel it through the admin surface:

```sh
curl -X POST http://127.0.0.1:3000/admin/requests/my-request-id/cancel
```

Use `--admin-token` or `LLM_ENGINE_ADMIN_TOKEN` to require
`Authorization: Bearer <token>` on admin routes. The server refuses non-loopback
binds unless an admin token is configured.

## Stop The Server

Press `Ctrl-C` in the terminal running `llm-engine`.
