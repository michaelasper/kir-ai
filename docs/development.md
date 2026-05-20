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

## Run The Fast Local Baseline

Use the fast baseline for local iteration before broad gates:

```sh
mise run check-fast
```

This runs a compile baseline plus the fastest unit and contract slices:

```sh
cargo check --workspace --all-targets
cargo test -p llm-kv-cache
cargo test -p llm-sampler
cargo test -p llm-api --test openai_contract
```

`check-fast` does not run `cargo test --workspace`, clippy, or release gates.

## Run Targeted Validation

Pick the narrowest command that matches the files changed, then add adjacent
contract tests when behavior crosses crate boundaries.

| Changed files | Mise task | Underlying command |
| --- | --- | --- |
| `crates/llm-kv-cache/**` | `mise run test-cache` | `cargo test -p llm-kv-cache` |
| `crates/llm-sampler/**` | `mise run test-sampler` | `cargo test -p llm-sampler` |
| `crates/llm-api/**` or request validation | `mise run test-api-contract` | `cargo test -p llm-api --test openai_contract` |
| Broad runtime behavior in `crates/llm-runtime/**` | `mise run test-runtime-contract` | `cargo test -p llm-runtime --test runtime_contract` |
| Runtime chat request flow, prompt rendering, stop handling | `mise run test-runtime-chat` | `cargo test -p llm-runtime --test runtime_contract chat::` |
| Runtime text completion flow | `mise run test-runtime-completion` | `cargo test -p llm-runtime --test runtime_contract completion::` |
| Runtime stream assembly, cancellation, streaming tool deltas | `mise run test-runtime-streaming` | `cargo test -p llm-runtime --test runtime_contract streaming::` |
| Runtime tool-choice validation and tool-call retry behavior | `mise run test-runtime-tools` | `cargo test -p llm-runtime --test runtime_contract tool_validation::` |
| Runtime JSON-object response mode | `mise run test-runtime-json` | `cargo test -p llm-runtime --test runtime_contract json_mode::` |
| Runtime no-progress classification | `mise run test-runtime-no-progress` | `cargo test -p llm-runtime --test runtime_contract no_progress::` |
| `crates/llm-tool-parser/**` | `mise run test-parser` | `cargo test -p llm-tool-parser` |
| `crates/llm-tokenizer/**` | `mise run test-tokenizer` | `cargo test -p llm-tokenizer` |
| Parser and tokenizer family changes together | `mise run test-parser-tokenizer` | `cargo test -p llm-tool-parser` and `cargo test -p llm-tokenizer` |
| `crates/llm-hub/**` | `mise run test-hub` | `cargo test -p llm-hub` |
| `crates/llm-backend/src/core/**` or safetensors CPU paths | `mise run test-backend-cpu` | `cargo test -p llm-backend --test safetensors_loader` |
| `crates/llm-metal/**` | `mise run test-metal-smoke` | `cargo test -p llm-metal --test metal_smoke -- --test-threads=1` |

Metal smoke tests are serialized because they use the host GPU. If they fail
because Metal or sandbox permissions are unavailable, record the result as
blocked and rerun on a host with Metal access.

## Common Focused Commands

API request and response contracts:

```sh
mise run test-api-contract
```

Runtime orchestration:

```sh
mise run test-runtime-contract
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
mise run test-tokenizer
```

Tool-call parsing:

```sh
mise run test-parser
```

Safetensors and native math probes:

```sh
mise run test-backend-cpu
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

Profiles live in `crates/llm-hub/src/profile.rs`.

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

Deferred families should still have explicit adapters and fail-closed template
and parser selectors. Add golden fixtures under `crates/llm-tokenizer/tests`
and `crates/llm-tool-parser/tests/fixtures` before advertising backend
execution for a new family.

Add fixture-driven tests under `crates/llm-models/tests` before broadening model
families or layer formats.

## Work On Native Tensor Math

Most native tensor math lives under `crates/llm-backend/src/core`.

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
mise run test-metal-smoke
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
