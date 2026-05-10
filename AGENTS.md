# kir-ai AGENTS

This repo uses a local Rust codebase.

## Architecture

- `src/lib.rs` — Library API surface and shared types.
- `src/main.rs` — CLI entrypoint (`serve`, `bench`, `model`) and process orchestration.
- `crates/llm-engine/src` and related crates — inference runtime and backends.
- `crates/llm-runtime` — protocol/runtime flow.
- `crates/llm-api` — request types and validation.
- `tests` / `crates/*/tests` — contract/integration tests.

## Testing

```bash
cargo test                  # run unit tests
cargo test --all-targets     # run all test targets
cargo clippy                # lint
```

## Using `kt`

The `kt` MCP server is configured for this repo.

Use `kt` in this order for code understanding and maintenance:

1. Use `kt_search` before reading files directly.
2. Use `kt_read_file` on the specific paths returned by search.
3. Use `kt_sync` when changing runtime/inference paths to refresh local discovery state.
4. Use `kt_git_status` before committing.
5. Use `kt_git_commit <message>` for targeted, scoped patches.

Suggested protocol-safe workflow:

- `kt_search "protocol"`
- `kt_sync <repo>`
- `kt_read_file <path>`
- implement minimal changes
- `kt_git_status` and review
- if behavior changed, run the protocol validation path
- `kt_git_commit "..."`

Guidance:
- Prefer narrow edits and avoid broad refactors.
- Keep shared defaults centralized and documented.
- Keep behavior changes paired with matching assertion updates in contract tests.
