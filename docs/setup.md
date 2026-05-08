# Setup

This guide shows how to set up a machine for developing and running `kir-ai`.
It assumes you are working from the repository root.

## Install With Mise

The preferred setup path uses `mise`, because the workspace pins Rust `1.95.0`
in [../mise.toml](../mise.toml).

```sh
mise install
```

Then confirm the toolchain:

```sh
rustc --version
cargo --version
```

## Install Without Mise

If you do not use `mise`, install Rust `1.95.x` with your normal Rust toolchain
manager, then run Cargo directly.

```sh
rustup toolchain install 1.95.0
rustup override set 1.95.0
```

The workspace uses edition `2024`, so older compilers are not supported.

## Build

Build the whole workspace:

```sh
cargo build --workspace
```

Build only the server and CLI binary:

```sh
cargo build -p llm-engine
```

The binary target is `llm-engine`.

## Run Checks

Use the `mise` task aliases when available:

```sh
mise run fmt-check
mise run test
mise run clippy
```

Run the full gate:

```sh
mise run check
```

Run the north-star promotion gates locally:

```sh
mise run gates-ci
mise run gates-nightly
```

Both gate profiles write JSON, Markdown, and per-gate logs under
`target/north-star-gates/`. The nightly profile skips real long-context
inference unless `LLM_BENCH_ENDPOINT` and `LLM_BENCH_SNAPSHOT` are configured.

The equivalent Cargo commands are:

```sh
cargo fmt --all --check
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

## Apple Silicon And Metal

The project targets Apple Silicon first. The current native Qwen path is still
CPU-oriented, but the workspace includes a Metal smoke crate. The Metal smoke
test skips itself when no Metal device is available.

For the intended environment, use:

- macOS on Apple Silicon.
- Rust `1.95.x`.
- Enough disk space for local model snapshots. The full Qwen3.6 BF16 selected
  artefacts are approximately 72 GB.

## Configure Hugging Face Access

Anonymous model planning and downloads work only for public models. For gated or
private repositories, export `HF_TOKEN` before planning or pulling:

```sh
export HF_TOKEN=hf_...
```

The token is read by `model plan` and `model pull`.

## Choose A Model Store

Model commands use `.llm-models` by default. Use either `LLM_MODEL_HOME`:

```sh
export LLM_MODEL_HOME=/Volumes/models/kir-ai
```

or pass `--model-home` on commands that support it:

```sh
cargo run -p llm-engine -- model list --model-home /Volumes/models/kir-ai
```

`model list` and `model pull` use `--model-home`. `model inspect`,
`model verify`, and serving with `--snapshot` take explicit snapshot paths.

## Run The Protocol Server

The fastest server path does not require a model:

```sh
cargo run -p llm-engine -- serve \
  --addr 127.0.0.1:3000 \
  --deterministic-test-backend
```

This starts the deterministic Rust backend explicitly. Use it for HTTP contract
work, client integration, and API shape checks.

## Common Setup Problems

If `cargo` reports an unsupported edition, check the Rust version first. The
workspace needs Rust `1.95`.

If a model pull fails with `model_auth_failed`, set `HF_TOKEN` and retry.

If a model verify command reports `model_integrity_failed`, treat the snapshot
as untrusted. Re-run `model pull` for the same repo, revision, profile, and
model home. Existing valid files are reused when their size and SHA-256 match,
and corrupt existing snapshots encountered by pull are moved to quarantine.

If the server returns `model_not_found`, check that the request `model` matches
the served `--model-id`. The default alias is `local-qwen36`.
