# How To Run The Server

This guide shows the practical server modes: protocol test mode,
native text snapshot mode, and the loopback MLX sidecar mode.

## Run Protocol Mode

Use protocol mode when you want fast, repeatable OpenAI-compatible responses
without model artefacts:

```sh
cargo run -p llm-engine --features test-utils -- serve \
  --addr 127.0.0.1:3000 \
  --protocol-test-backend \
  --i-understand-this-is-not-real-inference
```

The protocol backend is compiled only with the `test-utils` feature and requires
the acknowledgement flag because it serves hardcoded fixtures, not real
inference.

With `--protocol-test-backend` and no `--snapshot`, the server uses a
protocol test backend. It serves the model alias `local-qwen36` and returns
the fixed text:

```text
hello from rust native backend
```

Omitting both `--snapshot` and an acknowledged protocol backend exits with an
explicit backend requirement.

Confirm the server:

```sh
curl -s http://127.0.0.1:3000/health | jq
curl -s http://127.0.0.1:3000/v1/models | jq
```

## Run Native Text Mode

Use native text mode when you have a complete local Qwen or Gemma snapshot
containing:

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
  --max-new-tokens 256 \
  --max-prefill-tokens 32 \
  --native-metal-weight-cache-bytes 8589934592
```

You can also serve a model-store alias created by `model pull --alias`:

```sh
cargo run -p llm-engine -- serve \
  --addr 127.0.0.1:3000 \
  --snapshot-alias local-qwen36 \
  --model-home .llm-models
```

The native path tokenises the rendered prompt, keeps a bounded tail of prompt
tokens, runs family-specific prefill, applies final norm and LM-head top-k,
then returns decoded text.

## Run MLX Sidecar Mode

Use MLX sidecar mode when the snapshot manifest has `loader: mlx`, for example
the `qwen36-mlx-4bit`, `gemma4-e2b-it-mlx-4bit`, or
`llama32-3b-instruct-mlx-4bit` profiles. Kir remains the public
OpenAI-compatible server and proxies generation to a loopback MLX sidecar. Chat
requests use `/v1/chat/completions`; legacy text completion requests use a
completions-capable sidecar endpoint when the selected family exposes one. Qwen,
DeepSeek, and Llama run through `mlx_lm.server`; Gemma 4 runs through
`mlx_vlm.server`.

For chat requests, Kir forwards the structured OpenAI message history to MLX
losslessly, including assistant `tool_calls`, `tool` role results,
`tool_call_id`, and optional `name` fields. The rendered prompt is still kept
for cache and fallback paths, but it is not the source of truth for MLX chat
requests. The only rendered-prompt MLX chat fallback is Llama conversation mode
when no structured `chat_context` is available.

Start the Qwen MLX sidecar separately:

```sh
SNAPSHOT=.llm-models/huggingface/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/<resolved-commit>
mlx_lm.server --model "$SNAPSHOT"
```

Then start Kir against the same Qwen snapshot:

```sh
cargo run -p llm-engine -- serve \
  --addr 127.0.0.1:3000 \
  --snapshot "$SNAPSHOT" \
  --model-id local-qwen36-mlx \
  --mlx-endpoint http://127.0.0.1:8080/v1
```

If the snapshot was populated by the Hugging Face cache and has no Kir manifest,
raw native Qwen and Gemma snapshots can infer family metadata from
`config.json`. Raw MLX snapshots still require selecting the MLX loader and
model family explicitly. Raw MLX snapshots without `--family` fail at startup.
Qwen, DeepSeek, Gemma, and Llama are serveable runtime chat families through
family-specific MLX sidecars:

```sh
SNAPSHOT=$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/<resolved-commit>
mlx_lm.server --model "$SNAPSHOT" --chat-template-args '{"enable_thinking":false}'
cargo run -p llm-engine -- serve \
  --addr 127.0.0.1:3000 \
  --snapshot "$SNAPSHOT" \
  --loader mlx \
  --family qwen \
  --model-id local-qwen35-4b \
  --mlx-endpoint http://127.0.0.1:8080/v1
```

For Gemma 4 E2B, use the VLM sidecar because the current MLX Gemma 4 package
exposes OpenAI chat completions rather than text completions:

```sh
SNAPSHOT=$HOME/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/<resolved-commit>
mlx_vlm.server --model "$SNAPSHOT"
cargo run -p llm-engine -- serve \
  --addr 127.0.0.1:3000 \
  --snapshot "$SNAPSHOT" \
  --loader mlx \
  --family gemma \
  --model-id local-gemma4-e2b \
  --mlx-endpoint http://127.0.0.1:8080/v1
```

For Llama 3.2 Instruct, use the standard MLX LM sidecar and `--family llama`:

```sh
SNAPSHOT=$HOME/.cache/huggingface/hub/models--mlx-community--Llama-3.2-3B-Instruct-4bit/snapshots/<resolved-commit>
mlx_lm.server --model "$SNAPSHOT"
cargo run -p llm-engine -- serve \
  --addr 127.0.0.1:3000 \
  --snapshot "$SNAPSHOT" \
  --loader mlx \
  --family llama \
  --model-id local-llama32-3b \
  --mlx-endpoint http://127.0.0.1:8080/v1
```

The MLX endpoint must be loopback. Kir rejects remote MLX endpoints and does
not fall back to protocol-test mode or native text when an MLX manifest is selected.
This is a bootstrap comparison path; the no-Python production target remains a
native MLX bridge.

## Choose Generation Bounds

`--max-new-tokens` caps native generation per request. It defaults to `256` and
is clamped to at least `1`.

`--max-prefill-tokens` controls the native prefill chunk size. It defaults to
`32` and is clamped to at least `1`. Native text backends retain the accepted
prompt context by sizing full-attention caches from prompt length plus
generation budget, and reject requests that exceed the model context limit.

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

Use a larger prefill chunk only when you expect the current CPU-bound path to
benefit from fewer prefill calls:

```sh
cargo run -p llm-engine -- serve \
  --snapshot "$SNAPSHOT" \
  --max-new-tokens 256 \
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
decode. Non-greedy native text sampling accepts finite non-negative
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
