# kir-ai

`kir-ai` is a Rust-first local inference engine workspace. Its current centre is
`llm-engine`, an OpenAI-compatible HTTP server and model tooling CLI for native,
no-Python local inference experiments on Apple Silicon.

The project is intentionally explicit about its current state:

- Running `llm-engine serve` without a snapshot starts a deterministic Rust
  backend for protocol and client integration work.
- Running `llm-engine serve --snapshot <path>` starts the constrained native
  Qwen path, backed by local Hugging Face safetensors artefacts.
- The native Qwen path is a correctness and integration path, not a production
  throughput path. It uses bounded prefill, conservative generation defaults,
  CPU-oriented BF16 tensor reads, and no reusable KV cache yet.

## Quick Start

Install the pinned Rust toolchain with `mise`:

```sh
mise install
```

Run the workspace checks:

```sh
mise run check
```

Start the deterministic protocol server:

```sh
cargo run -p llm-engine -- serve --addr 127.0.0.1:3000
```

In another terminal, make a chat request:

```sh
curl -s http://127.0.0.1:3000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "local-qwen36",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 8
  }' | jq
```

You should see an OpenAI-shaped `chat.completion` response from the
deterministic Rust backend.

## Native Qwen Snapshot Flow

Plan the default Qwen BF16 native profile before downloading it:

```sh
cargo run -p llm-engine -- model plan Qwen/Qwen3.6-35B-A3B \
  --revision main \
  --profile qwen36-safetensors-bf16
```

Pull metadata only when you want to inspect manifests and static artefacts
without downloading weight shards:

```sh
cargo run -p llm-engine -- model pull Qwen/Qwen3.6-35B-A3B \
  --metadata-only \
  --model-home .llm-models
```

Pulling the full BF16 profile is large, approximately 72 GB of selected
artefacts for the current Qwen3.6 fixture. After a full pull, serve the snapshot:

```sh
cargo run -p llm-engine -- serve \
  --addr 127.0.0.1:3000 \
  --snapshot .llm-models/huggingface/models--Qwen--Qwen3.6-35B-A3B/snapshots/<resolved-commit> \
  --model-id local-qwen36 \
  --max-new-tokens 1 \
  --max-prefill-tokens 32
```

## Documentation Map

| Need | Document |
| --- | --- |
| Learn the first working flow | [Getting started](docs/getting-started.md) |
| Set up a developer machine | [Setup](docs/setup.md) |
| Run the server for protocol or native Qwen testing | [How to run the server](docs/how-to-run-server.md) |
| Plan, pull, inspect, and verify model snapshots | [How to manage model snapshots](docs/how-to-manage-models.md) |
| Look up CLI commands and flags | [CLI reference](docs/cli-reference.md) |
| Look up HTTP endpoints, request fields, streaming, and errors | [HTTP API reference](docs/http-api-reference.md) |
| Look up configuration, snapshot, and model format facts | [Configuration reference](docs/configuration-reference.md) |
| Understand crate boundaries and request flow | [Architecture](docs/architecture.md) |
| Work on the codebase safely | [Development guide](docs/development.md) |

The north-star product direction and implementation tracker live in
[rust-metal-inference-engine-north-star.md](rust-metal-inference-engine-north-star.md).

## Current Limitations

- Qwen3.5/Qwen3.6 MoE text loading is the only native model-family path.
- The server does not execute `generation_config.json` or the downloaded
  `chat_template.jinja`; chat prompts use the Rust Qwen ChatML renderer.
- Streaming responses are OpenAI-shaped SSE, but chunks are assembled after a
  backend generation rather than emitted token by token during decode.
- Non-greedy sampling is rejected. Use `temperature: 0` and `top_p: 1`, or omit
  those fields.
- Metal exists as a smoke-tested vector-add crate; Qwen inference kernels are
  not wired to Metal yet.
