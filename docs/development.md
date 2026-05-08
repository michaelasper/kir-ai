# Development Guide

This guide covers day-to-day development commands and the safest places to make
common changes.

## Run The Full Gate

```sh
mise run check
```

This runs formatting checks, all tests, and strict clippy.

Without `mise`:

```sh
cargo fmt --all --check
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

## Run Focused Tests

API request and response contracts:

```sh
cargo test -p llm-api --test openai_contract
```

Runtime orchestration:

```sh
cargo test -p llm-runtime --test runtime_contract
```

HTTP server contract:

```sh
cargo test -p llm-engine --test http_contract
```

Model CLI commands:

```sh
cargo test -p llm-engine --test model_cli
```

Qwen config and safetensors index validation:

```sh
cargo test -p llm-models --test qwen36_config
```

Tokenizer and prompt rendering:

```sh
cargo test -p llm-tokenizer --test qwen_template
```

Tool-call parsing:

```sh
cargo test -p llm-tool-parser --test qwen_parser
```

Safetensors and native math probes:

```sh
cargo test -p llm-backend
```

## Add Or Change HTTP Fields

Change request and response structs in `crates/llm-api/src/lib.rs`.

Then update validation in the same crate. Runtime code calls `validate()` before
backend execution, so request-shape failures should usually be represented as
`ApiError`.

Add or update tests in:

- `crates/llm-api/tests/openai_contract.rs`
- `crates/llm-runtime/tests/runtime_contract.rs`
- `crates/llm-engine/tests/http_contract.rs`

Keep unsupported features fail-closed. If the runtime cannot honour a field,
return `unsupported_capability` instead of accepting it silently.

## Add Or Change Runtime Behaviour

Use `crates/llm-runtime/src/lib.rs` for behaviour between validated API requests
and backend calls:

- chat prompt rendering
- text completion orchestration
- stop sequence handling
- tool-call parsing
- JSON object validation
- no-progress classification
- stream chunk assembly

When changing runtime behaviour, add tests to `crates/llm-runtime/tests` first,
then cover the HTTP shape in `crates/llm-engine/tests/http_contract.rs` when the
wire response changes.

## Add Or Change CLI Flags

The CLI is currently parsed manually in `crates/llm-engine/src/main.rs`.

When adding flags:

1. Keep the `--flag value` convention unless the flag is boolean.
2. Add usage errors for required values.
3. Update [cli-reference.md](cli-reference.md).
4. Add a focused test in `crates/llm-engine/tests/model_cli.rs` when the command
   produces stable output.

Unknown extra flags may currently be ignored by recognised subcommands. Be
careful when relying on strict CLI rejection until argument parsing is replaced
or hardened.

## Add Or Change Model Acquisition Profiles

Profiles live in `crates/llm-hub/src/lib.rs`.

When adding a profile:

1. Define the profile identity: name, family, loader, and quantisation.
2. Set allow and ignore patterns.
3. Add tests in `crates/llm-hub/tests/download_plan.rs`.
4. Update [configuration-reference.md](configuration-reference.md) and
   [cli-reference.md](cli-reference.md).

Remember that pattern matching is simple exact, prefix, or suffix matching, not
full globbing.

## Add Or Change Model Config Support

Model-family config parsing lives in `crates/llm-models/src/lib.rs`.

Qwen support currently validates:

- root architecture and model type
- text config dimensions
- linear/full attention layer types
- safetensors index tensor names expected by the text loader

Add fixture-driven tests under `crates/llm-models/tests` before broadening model
families or layer formats.

## Work On Native Tensor Math

Most native Qwen math lives in `crates/llm-backend/src/lib.rs`.

Use the existing tests in `crates/llm-backend/tests/safetensors_loader.rs` as
the pattern for:

- tiny safetensors fixtures
- BF16 row reads
- indexed shard access
- Qwen embedding and norm probes
- attention path probes
- MoE router and expert execution
- LM-head top-k checks

Prefer small shape-specific tests before touching full snapshot probes.

## Work On Metal

Metal code lives in `crates/llm-metal/src/lib.rs`.

The current crate compiles and runs a vector-add kernel. The smoke test skips if
no Metal device exists:

```sh
cargo test -p llm-metal
```

Do not assume Metal code is part of the Qwen serving path until it is explicitly
wired through `llm-backend` or a backend implementation.

## Keep Docs In Sync

When changing user-visible behaviour, update the relevant document:

- CLI flags: [cli-reference.md](cli-reference.md)
- HTTP fields, responses, or errors: [http-api-reference.md](http-api-reference.md)
- model profiles, snapshot layout, config fields: [configuration-reference.md](configuration-reference.md)
- server workflows: [how-to-run-server.md](how-to-run-server.md)
- model workflows: [how-to-manage-models.md](how-to-manage-models.md)

Use the docs split by user need:

- tutorials teach a first success
- how-to guides solve practical tasks
- reference pages describe facts
- explanations discuss design and trade-offs
