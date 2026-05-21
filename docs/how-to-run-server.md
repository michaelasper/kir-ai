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
  --max-prefill-tokens 2048 \
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

For Qwen MLX requests with a required tool choice, Kir adds
`"enable_tool_logits_bias":true` to the forwarded `chat_template_kwargs`. This
is a request-level hint for sidecars that honor it to bias decoding toward
structured tool-call tokens; ordinary Qwen chat requests keep only
`"enable_thinking":false`.

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
mlx_vlm.server --model "$SNAPSHOT" --prompt-cache-size 16 --prefill-step-size 2048
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

MLX sidecar prompt-cache reuse is controlled at sidecar launch time with
`--prompt-cache-size` or `--prompt-cache-bytes`. Kir keeps request bodies
OpenAI/MLX-compatible and does not send request-level cache/session fields that
the MLX sidecars do not advertise. Stable-prefix serving should instead keep
system prompts, tools, chat-template kwargs, and tool schema serialization stable
so the sidecar cache can match common prefixes when enabled.

## Choose Generation Bounds

`--max-new-tokens` caps native generation per request. It defaults to `256` and
is clamped to at least `1`.

`--max-prefill-tokens` controls the native prefill chunk size. It defaults to
`2048` and is clamped to at least `1`. Long-context native serving depends on a
large value here because prompt prefill runs sequentially by chunk. Native text
backends retain the accepted prompt context by sizing full-attention caches from
prompt length plus generation budget, and reject requests that exceed the model
context limit.

`--native-prefix-cache-bytes` controls the per-backend Qwen/Gemma prefix-cache
budget. It defaults to `536870912` bytes. Set `0` to reject prefix-cache stores
while still allowing generation without prefix reuse. `LLM_ENGINE_PREFIX_CACHE_BYTES`
provides the same setting when the flag is omitted.

`--native-metal-weight-cache-bytes` controls the per-backend LRU budget for
uploaded Metal BF16 weight buffers. It defaults to `8589934592` bytes and can be
set to `0` to disable weight-buffer caching.

`--warm-native-metal-weight-cache` preloads rank-2 BF16 tensors into that cache
at startup until the configured budget is full. Leave it off when you want
minimum startup time or when first-request latency is not the bottleneck.

Override to small values only while probing correctness or reducing memory
pressure:

```sh
cargo run -p llm-engine -- serve \
  --snapshot "$SNAPSHOT" \
  --max-new-tokens 1 \
  --max-prefill-tokens 8
```

Keep the default, or tune upward when the host has enough memory and long-context
TTFT is the bottleneck:

```sh
cargo run -p llm-engine -- serve \
  --snapshot "$SNAPSHOT" \
  --max-new-tokens 256 \
  --max-prefill-tokens 4096
```

## Enable HTTPS

Plain HTTP remains the default so existing loopback workflows keep working. To
serve HTTPS directly from `llm-engine`, provide both a PEM certificate chain and
a PEM private key at startup:

```sh
cargo run -p llm-engine -- serve \
  --addr 0.0.0.0:3000 \
  --snapshot "$SNAPSHOT" \
  --admin-token "$LLM_ENGINE_ADMIN_TOKEN" \
  --tls-cert /etc/kir-ai/tls/fullchain.pem \
  --tls-key /etc/kir-ai/tls/privkey.pem
```

The server validates both files before accepting requests. Provide the two
flags together; `--tls-cert` without `--tls-key`, or the reverse, fails at
startup. If you bind a non-loopback address without these flags, keep TLS at a
local reverse proxy such as Caddy or nginx and forward only trusted traffic to
Kir.

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
ADMIN_TOKEN=... # token from --admin-token, LLM_ENGINE_ADMIN_TOKEN, or startup output
curl -s -H "Authorization: Bearer $ADMIN_TOKEN" http://127.0.0.1:3000/admin/models | jq
curl -s -H "Authorization: Bearer $ADMIN_TOKEN" http://127.0.0.1:3000/admin/models/local-qwen36 | jq
```

`GET /admin/models` and `GET /admin/models/{alias}` are read-only status
endpoints. The `/admin/*` surface also includes metrics, snapshot verification,
download planning, snapshot pulls, and active request cancellation;
`/admin/models/{alias}/pull` mutates the configured model store.

To make a request cancellable by a known ID, send `x-request-id` on the
inference call, then cancel it through the admin surface:

```sh
curl -X POST \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  http://127.0.0.1:3000/admin/requests/my-request-id/cancel
```

Admin routes require `Authorization: Bearer <token>`. Use `--admin-token` or
`LLM_ENGINE_ADMIN_TOKEN` to set a stable token. If neither is set on a loopback
bind, `serve` generates a temporary token for that process and prints the
required header at startup. The server refuses non-loopback binds unless an admin
token is configured. Use HTTPS or reverse-proxy TLS for non-loopback binds so
bearer tokens, model pulls, prompts, and responses are not sent over cleartext
HTTP.

## Stop The Server

Press `Ctrl-C` in the terminal running `llm-engine`.
