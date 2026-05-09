# Architecture

This explanation describes how the workspace is divided, how a request becomes
tokens, and why some pieces are deliberately narrow today.

## Purpose

The project is building a no-Python, Rust-owned local inference engine for
agentic coding workflows. The first working surface is an OpenAI-compatible
server with strict runtime semantics and a constrained native Qwen backend.

The architecture favours clear boundaries over early throughput optimisation.
The code separates API contracts, runtime validation, prompt rendering, tool
parsing, model acquisition, tensor access, and future acceleration work.

## Request Flow

```text
HTTP client
  -> llm-engine Axum route
  -> llm-api request type
  -> llm-runtime validation and orchestration
  -> llm-tokenizer prompt rendering for chat
  -> llm-backend ModelBackend
  -> llm-runtime stop/tool/json/no-progress handling
  -> llm-engine JSON or SSE response
```

For native Qwen:

```text
NativeQwenBackend
  -> load config.json into QwenModelSpec
  -> load tokenizer.json
  -> open model.safetensors.index.json through SafeTensorShardStore
  -> tokenise prompt
  -> keep bounded context tail
  -> run Qwen prefill layers
  -> apply final norm
  -> stream LM-head rows in chunks
  -> choose a decoded top candidate
  -> return BackendOutput
```

The runtime then applies stop sequences, parses Qwen tool calls, validates JSON
object mode, classifies no-progress completions, and builds OpenAI-shaped
responses.

## Crate Map

| Crate | Responsibility | Current status |
| --- | --- | --- |
| `llm-api` | OpenAI-compatible request and response structs, tool schema, finish reasons, usage, and validation. | Implements the supported API subset and fails closed for unsupported request features. |
| `llm-engine` | HTTP and CLI edge. Owns routing, SSE framing, admin endpoints, error-to-HTTP mapping, and manifest-based backend selection. | Serving requires an explicit backend: deterministic protocol mode uses `--deterministic-test-backend`, native Qwen is isolated under the native backend module, and MLX manifests proxy through the MLX backend module. |
| `llm-runtime` | Semantic orchestration between API and backend. | Handles chat and text completions, streaming chunk assembly, stop truncation, tool parsing, JSON-object validation, and no-progress classification. |
| `llm-backend` | Backend trait, protocol-test backend, safetensors loading, BF16 tensor access, generic backend cache identity, and CPU Qwen math behind Qwen-specific functions. | Contains the active native inference code: embeddings, RMSNorm, linear/full attention paths, MoE, final norm, and LM-head top-k. |
| `llm-tokenizer` | Hugging Face tokenizer wrapper and family chat-template selection. | Supports Qwen ChatML and Gemma 4 text/tool chat templates. DeepSeek template selection is explicit and fails closed until production parity. |
| `llm-tool-parser` | Family assistant output parser selection. | Supports Qwen reasoning tags and JSON/XML tool-call forms plus Gemma 4 thought/tool-call channels. DeepSeek parser execution fails closed until production parity. |
| `llm-models` | Model config, family adapters, production backend declarations, and safetensors index interpretation. | Supports dense Qwen3 plus Qwen3.5/Qwen3.6 MoE text config, declares Qwen production backends as native Metal plus MLX, declares Gemma production serving through MLX, and keeps DeepSeek deferred. |
| `llm-hub` | Hugging Face planning, download, snapshot promotion, and verification. | Requires immutable resolved commits, validates paths, supports resumable downloads, writes engine manifests, and includes Gemma text-chat acquisition profiles that skip vision/projector artifacts. |
| `llm-metal` | Metal device and kernel experiments. | Smoke-tested vector add only. Not wired into Qwen inference yet. |
| `llm-sampler` | Greedy argmax sampler. | Standalone and tested. Native backend currently chooses from LM-head top-k directly. |
| `llm-kv-cache` | KV-cache budget accounting. | Placeholder utility. No actual key/value tensor cache yet. |
| `llm-telemetry` | Token counters and request metrics. | Standalone metrics primitives. Runtime currently constructs API usage directly. |

## Protocol Test Backend

The deterministic backend is a protocol test stub, not a chat model and not an
inference path. It lets the HTTP contract mature separately from model execution
with fast, stable responses for:

- request validation
- OpenAI response shape
- SSE framing
- error metadata
- client compatibility tests

It must not grow prompt-specific chat behavior. Real generation belongs behind
snapshot-backed native backends such as `NativeQwenBackend`.

## Why Native Qwen Is Opt-In

Native Qwen serving requires a complete local snapshot and currently runs a
bounded correctness path. The server does real tokenisation, safetensors reads,
Qwen layer execution, final norm, and LM-head top-k, but it does not yet have the
performance properties expected from production serving.

The opt-in `--snapshot` boundary keeps protocol work easy while making native
model execution explicit.

Rust callers follow the same rule. `build_router()` fails closed when no backend
is provided, so callers must migrate to `build_router_with_backend(...)` or
`build_router_with_backend_and_options(...)` for inference. Protocol-only tests
can opt into `build_router_with_deterministic_test_backend()`.

## Fail-Closed Semantics

The runtime rejects unsupported behaviour instead of accepting and ignoring it.
Examples:

- `temperature` must be absent or `0`.
- `top_p` must be absent or `1`.
- `json_schema` response format is rejected.
- Required function tool choices must name a declared tool.
- Empty chat requests, empty completion prompts, zero `max_tokens`, and empty
  stop sequences are invalid.
- Malformed generated tool calls become structured errors.

This is important for agentic workflows because silent approximation is harder
to diagnose than a clear unsupported-capability error.

## Streaming Today

The HTTP stream is OpenAI-compatible SSE, including one `[DONE]` terminator and
optional usage-only chunks. The runtime currently assembles streaming chunks
after backend generation completes.

That means streaming is correct at the protocol level, but it is not yet
incremental token streaming from the decoder. Future work needs cancellation,
stall detection, token-by-token deltas, and long-prefill visibility.

## Model Acquisition Boundary

`llm-hub` treats model acquisition as a product surface:

- mutable revisions are resolved to immutable commits
- selected artefacts are planned before download
- unsafe artefact paths are rejected
- downloads are written to staging before promotion
- sizes and SHA-256 values are verified when available
- promoted snapshots record manifest identity

This keeps model identity auditable and separates model artefact storage from
runtime cache work.

## Acceleration Boundary

`llm-backend` currently carries correctness-first CPU BF16 math. `llm-metal`,
`llm-kv-cache`, and `llm-sampler` indicate the intended acceleration and decode
boundaries, but they are not yet the serving hot path.

The current shape makes it possible to promote individual operations from CPU
probes to Metal kernels without changing API or model-store semantics.

## Current Design Constraints

- The runtime selects chat rendering and parser behaviour from backend model
  metadata; Qwen is implemented, while non-Qwen families fail closed until their
  adapters exist.
- Native model execution is BF16 safetensors-oriented.
- Native Qwen uses `--max-prefill-tokens` as a prefill chunk size; retained prompt context is sized from the accepted prompt plus generation budget and fails closed at the model context limit.
- Multi-token decode recomputes bounded context instead of maintaining reusable
  KV or recurrent state caches.
- The server does not use downloaded `generation_config.json` sampling settings.
- The downloaded `chat_template.jinja` is a fixture and artefact, not runtime
  template code.
