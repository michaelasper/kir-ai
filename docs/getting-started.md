# Getting Started

This tutorial gets you from a fresh checkout to a working OpenAI-compatible
server response. It uses the deterministic Rust backend so the steps are small,
repeatable, and do not require a model download.

You will:

1. Install the pinned Rust toolchain.
2. Run a focused server test.
3. Start `llm-engine`.
4. Call health, models, chat, completion, and streaming endpoints.

## Prerequisites

Install `mise` and make sure `curl` is available. `jq` is optional, but it makes
the JSON responses easier to read.

The workspace pins Rust `1.95.0` in [../mise.toml](../mise.toml). The Cargo
workspace requires Rust `1.95` and edition `2024`.

## 1. Install The Toolchain

From the repository root:

```sh
mise install
```

This installs the Rust version used by the workspace tasks.

## 2. Run One Contract Test

Run the HTTP contract test before starting the server:

```sh
cargo test -p llm-engine --test http_contract
```

The test exercises the same deterministic router you will call below. It should
finish without a model snapshot.

## 3. Start The Server

Open a terminal and run:

```sh
cargo run -p llm-engine -- serve --addr 127.0.0.1:3000
```

You should see a log line similar to:

```text
llm-engine listening addr=127.0.0.1:3000
```

Keep this terminal running.

## 4. Check Health

In a second terminal:

```sh
curl -s http://127.0.0.1:3000/health | jq
```

You should see:

```json
{
  "python_runtime": false,
  "runtime": "rust",
  "status": "ok"
}
```

## 5. List The Served Model

```sh
curl -s http://127.0.0.1:3000/v1/models | jq
```

The deterministic backend serves the `local-qwen36` alias:

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

## 6. Send A Chat Request

```sh
curl -s http://127.0.0.1:3000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "local-qwen36",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 8
  }' | jq
```

Notice that the response has the OpenAI chat shape:

```json
{
  "object": "chat.completion",
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
  ]
}
```

The exact `id`, `created`, and `usage` values vary by run.

## 7. Send A Text Completion Request

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

The stop sequence truncates the deterministic text:

```json
{
  "object": "text_completion",
  "model": "local-qwen36",
  "choices": [
    {
      "text": "hello from rust native",
      "index": 0,
      "finish_reason": "stop"
    }
  ]
}
```

## 8. Try Streaming With Usage

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

You should see `data:` lines containing `chat.completion.chunk` JSON, then a
usage-only chunk with `"choices":[]`, then one final:

```text
data: [DONE]
```

## What You Built

You have run the Rust HTTP edge, confirmed that it does not depend on Python at
request time, and exercised the OpenAI-compatible chat, text completion, and SSE
shapes. Use [how-to-run-server.md](how-to-run-server.md) when you want to switch
from the deterministic backend to a native Qwen snapshot.
