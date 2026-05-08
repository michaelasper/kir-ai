# How To Run The Server

This guide shows the practical server modes: deterministic protocol mode and
native Qwen snapshot mode.

## Run Protocol Mode

Use protocol mode when you want fast, repeatable OpenAI-compatible responses
without model artefacts:

```sh
cargo run -p llm-engine -- serve --addr 127.0.0.1:3000
```

With no `--snapshot`, the server uses a deterministic Rust backend. It serves
the model alias `local-qwen36` and returns the fixed text:

```text
hello from rust native backend
```

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
  --max-prefill-tokens 32
```

The native path tokenises the rendered prompt, keeps a bounded tail of prompt
tokens, runs Qwen prefill, applies final norm and LM-head top-k, then returns
decoded text.

## Choose Generation Bounds

`--max-new-tokens` caps native generation per request. It defaults to `1` and is
clamped to at least `1`.

`--max-prefill-tokens` controls how many recent prompt tokens are retained for
native prefill. It defaults to `32` and is clamped to at least `1`.

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

The request `model` must match `--model-id`. Non-greedy sampling is not
implemented, so omit `temperature` and `top_p` or use exactly `0` and `1`.

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

The admin model status is read-only and reports the currently served alias.

## Stop The Server

Press `Ctrl-C` in the terminal running `llm-engine`.
