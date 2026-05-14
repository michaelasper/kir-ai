<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/kir-ai.png">
    <source media="(prefers-color-scheme: light)" srcset="assets/kir-ai.png">
    <img alt="kir-ai" src="assets/kir-ai.png" width="120">
  </picture>

  <h1>kir-ai</h1>
  <p>Rust-first local inference on Apple Silicon with explicit, OpenAI-compatible runtime boundaries.</p>
</div>

<div align="center">

[![License][license-shield]][license-url]
[![CI][ci-shield]][ci-url]
[![Release][release-shield]][release-url]
[![Rust][rust-shield]][rust-url]
[![Apple Metal][metal-shield]][metal-url]
[![Local Inference][inference-shield]][docs-setup]

</div>

<div align="center">
  <a href="#quick-start">Quick Start</a> ·
  <a href="#features--highlights">Features</a> ·
  <a href="#usage">Usage</a> ·
  <a href="#documentation-map">Docs</a> ·
  <a href="https://github.com/michaelasper/kir-ai/issues">Report Bug</a>
</div>

**kir-ai** is an OpenAI-shaped local inference workspace for Apple Silicon that keeps core inference, request contracts, and safety checks in Rust. The project is built around explicit runtime selection: protocol verification, native Metal execution, and MLX sidecar interop all live behind the same CLI/server surface with strict capability boundaries.

## Why / The Problem

Many local inference stacks are easiest to ship with ad-hoc Python glue, but that coupling makes behaviour harder to audit and scale. `kir-ai` addresses this by making protocol handling and runtime orchestration explicit in a Rust workspace while preserving the API shape your clients already expect.

You get an engine that:
- exposes OpenAI-style endpoints consistently,
- fails closed for unsupported request features,
- separates testing pathways from model-serving pathways,
- and keeps model lifecycle (plan/pull/verify/serve) under explicit commands.

## Features / Highlights

- **OpenAI-compatible edge** for `/v1/chat/completions`, `/v1/completions`, streaming SSE, and model listing.
- **Strict capability gating** in request validation and runtime mapping; unsupported features return stable errors instead of silent fallback behaviour.
- **Two serving modes**: protocol-test mode for client contract work and snapshot-backed serving for native Metal/MLX paths.
- **Native Metal first-class support** for Qwen and Gemma text pipelines with bounded prefill and typed cache identities.
- **Model lifecycle tooling** in `llm-engine`: `model plan`, `model list`, `model inspect`, `model verify`, and `model pull`.
- **Operational controls** with admin endpoints for metrics, snapshot verification/pull, lane-level request cancellation, and model metadata.
- **Failure-safe semantics** including request validation for unsafe fields (`max_tokens`, sampling controls, stop sequences, tool schemas, malformed JSON, and token budgets).

## When to Use

Use `kir-ai` when you want a local inference server that is explicit about execution mode and protocol behaviour. If you are iterating on client integration, choose protocol-test mode first. If you are preparing model-backed inference runs, switch to snapshot-based serving.

Avoid `kir-ai` as a first step if your immediate need is a managed multi-user cloud inference platform.

## Quick Start

1. Install and prepare the workspace.

   ```sh
   curl -fsSL https://raw.githubusercontent.com/michaelasper/kir-ai/main/scripts/install-macos.sh | bash
   ```

2. Start the protocol test backend.

   ```sh
   kirai
   ```

3. Send a smoke request.

   ```sh
   curl -s http://127.0.0.1:3000/v1/chat/completions \
     -H 'content-type: application/json' \
     -d '{
       "model": "local-qwen36",
       "messages": [{"role": "user", "content": "hello"}],
       "max_tokens": 8
     }' | jq
   ```

Expected response: OpenAI-shaped `chat.completion` JSON with `local-qwen36`.

### Install and Runtime Options

- `KIR_AI_DIR`, `KIR_AI_REF` choose install location and revision.
- `KIR_AI_SKIP_BUILD=1` for dependency setup without compile.
- `KIR_AI_SKIP_PYTHON=1` for Rust-only install paths.
- `KIR_AI_FORCE_CLONE=1` to force a fresh checkout path.

For full script controls, see [`docs/ci-and-release.md`][docs-setup].

### Serve with a Snapshot

```sh
kirai serve \
  --snapshot .llm-models/<manifest-snapshot-path> \
  --model-id local-qwen36 \
  --max-new-tokens 256 \
  --max-prefill-tokens 32
```

For MLX manifests, set the loopback endpoint:

```sh
kirai serve \
  --snapshot .llm-models/<mlx-snapshot-path> \
  --loader mlx \
  --family qwen \
  --model-id local-qwen35-4b \
  --mlx-endpoint http://127.0.0.1:8080/v1
```

## Usage

### Core Endpoints

- `GET /health`
- `GET /v1/models`
- `GET /admin/models` and `/admin/models/{alias}`
- `POST /v1/chat/completions` and `POST /v1/completions`
- `POST /admin/models/{alias}/verify`
- `POST /admin/models/{alias}/plan`
- `POST /admin/models/{alias}/pull`
- `POST /admin/requests/{request_id}/cancel`
- `GET /admin/metrics`

For request and response examples, see [`docs/getting-started.md`][docs-getting-started].
For the full HTTP contract, see [`docs/http-api-reference.md`][http-api-doc].

## Native Text Snapshot Flow

Use `kirai` model commands to plan, inspect, verify, and pull profiles before serving.

```sh
kirai model plan Qwen/Qwen3-0.6B \
  --revision main \
  --profile qwen3-dense-safetensors-bf16

kirai model pull Qwen/Qwen3.6-35B-A3B \
  --metadata-only \
  --model-home .llm-models

kirai model inspect .llm-models/<snapshot-path>
```

Want direct source commands? Use `cargo run -p llm-engine -- ...` from a local checkout (development mode).

## Documentation Map

| Need | Document |
| --- | --- |
| Start with a working response | [`docs/getting-started.md`][docs-getting-started] |
| Developer machine setup | [`docs/setup.md`][docs-setup] |
| Run server and native text paths | [`docs/how-to-run-server.md`][docs-run-server] |
| Model snapshot lifecycle | [`docs/how-to-manage-models.md`][docs-models] |
| CLI reference | [`docs/cli-reference.md`][docs-cli] |
| HTTP API reference | [`docs/http-api-reference.md`][http-api-doc] |
| Configuration and formats | [`docs/configuration-reference.md`][docs-config] |
| Project architecture | [`docs/architecture.md`][docs-architecture] |
| CI and release details | [`docs/ci-and-release.md`][docs-ci-release] |
| Development guide | [`docs/development.md`][docs-dev] |

The product direction and implementation milestones are tracked in [`rust-metal-inference-engine-north-star.md`][north-star].

## Current Limitations

- Native Metal text execution currently covers dense Qwen, Qwen3/Qwen3.6 MoE, and Gemma 4 paths.
- Native paths are correctness-first and intentionally conservative for sampling and throughput.
- The server does not execute `generation_config.json` or downloaded chat templates (`chat_template.jinja`) as runtime config.
- Tool-call and JSON-object validation paths may buffer to preserve fail-closed semantics.
- Snapshot serving requires explicit backend mode; implicit no-snapshot stub serving is not supported.

## Compatibility

- Rust workspace version: `1.95`
- Runtime target profile: Apple Silicon first-class, macOS-first CI.

## License

This project is licensed under MIT. See upstream license terms at the official MIT license text.

[ci-shield]: https://img.shields.io/github/actions/workflow/status/michaelasper/kir-ai/ci.yml?branch=main&style=flat-square&label=ci
[ci-url]: https://github.com/michaelasper/kir-ai/actions/workflows/ci.yml
[release-shield]: https://img.shields.io/github/actions/workflow/status/michaelasper/kir-ai/release.yml?label=release&style=flat-square
[release-url]: https://github.com/michaelasper/kir-ai/actions/workflows/release.yml
[rust-shield]: https://img.shields.io/badge/rust-1.95-f5a97f?style=flat-square&logo=rust&logoColor=white
[rust-url]: https://www.rust-lang.org/
[metal-shield]: https://img.shields.io/badge/apple%20metal-native-c6a0f6?style=flat-square&logo=apple&logoColor=white
[metal-url]: https://developer.apple.com/metal/
[license-shield]: https://img.shields.io/badge/license-MIT-a6da95?style=flat-square&logo=opensourceinitiative&logoColor=white
[license-url]: https://opensource.org/licenses/MIT
[inference-shield]: https://img.shields.io/badge/local-inference-91d7e3?style=flat-square
[docs-getting-started]: docs/getting-started.md
[docs-setup]: docs/setup.md
[docs-run-server]: docs/how-to-run-server.md
[docs-models]: docs/how-to-manage-models.md
[docs-cli]: docs/cli-reference.md
[http-api-doc]: docs/http-api-reference.md
[docs-config]: docs/configuration-reference.md
[docs-architecture]: docs/architecture.md
[docs-ci-release]: docs/ci-and-release.md
[docs-dev]: docs/development.md
[north-star]: rust-metal-inference-engine-north-star.md
