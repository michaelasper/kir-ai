# CI and Release Reference

This page describes the project automation that supports working on kir-ai.
It is reference material for contributors who need to understand what runs,
what each workflow produces, and how releases are cut.

## GitHub Actions

| Workflow | File | Trigger | Purpose | Primary output |
| --- | --- | --- | --- | --- |
| CI | `.github/workflows/ci.yml` | Pull requests and pushes to `main` | Formatting, clippy, focused admin schema drift validation, release hygiene checks, named north-star product gates, and macOS build packaging | Gate reports, release-notes preview, and build artifact |
| Nightly | `.github/workflows/nightly.yml` | Daily schedule and manual dispatch | Broad workspace tests, nightly north-star gates, long-context planning, optional live bench gates, and nightly build packaging | Nightly gate report and nightly build artifact |
| Release | `.github/workflows/release.yml` | `v*.*.*` tags and manual dispatch | Validate tag/version, run release checks, build `llm-engine`, generate notes, publish GitHub release | macOS release archive, SHA-256 file, release notes |

All workflows run on macOS because Metal smoke coverage and Apple Silicon
serving are first-class project concerns.

## CI Jobs

The CI workflow uses separate job names for separate responsibilities. It does
not run `cargo test --workspace` before the north-star gates; the named
north-star product gates are the PR source of truth for required contract
coverage.

- `Static Analysis` runs `cargo fmt --all -- --check` and `cargo clippy --workspace --all-targets --all-features -- -D warnings`.
- `Admin Schema Drift` runs `cargo test -p llm-server --all-features --lib generate_admin_api_schemas`, then fails if `docs/schemas/admin/` has uncommitted changes.
- `Build` runs `cargo build --release --target aarch64-apple-darwin -p llm-engine` and uploads the packaged macOS binary.
- `North-Star Gate Report` runs versioning, conventional commits, release-note preview generation, and `scripts/north-star-gates.sh ci`.

The north-star gate script writes JSON, Markdown, and per-gate logs under
`target/north-star-gates/`. CI uploads those reports as artifacts and appends
the Markdown summary to the workflow step summary.

The PR `ci` north-star profile runs the required product contract gates by
name:

- `cargo test -p llm-api --test openai_contract`
- `cargo test -p llm-runtime --test runtime_contract --all-features`
- `cargo test -p llm-engine --test http_contract --all-features`
- `cargo test -p llm-engine --test model_cli --all-features`
- `cargo test -p llm-hub`
- `cargo test -p llm-models --test family_adapter`
- Deferred family tokenizer/parser template tests for DeepSeek, Gemma, and Llama.
- Full tokenizer and tool-parser crate tests.

`cargo test --workspace --all-features` covers those same API, runtime,
engine, hub, model, tokenizer, and parser suites plus the rest of the
workspace. That broad validation belongs to nightly and explicit release/deep
validation, not the PR path.

## Nightly Validation

The nightly workflow has a `Nightly Validation` job that runs
`scripts/north-star-gates.sh nightly`. The nightly profile first runs
`workspace_tests` with:

```sh
cargo test --workspace --all-features
```

After that broad gate passes, the report records the PR product gates and other
workspace-covered test gates as `covered` instead of rerunning the same suites.
That includes the no-progress replay classifier, native backend, and Metal
smoke test gates. The nightly profile then runs non-workspace gates such as
long-context dry-run planning and optional live long-context inference gates
when `LLM_BENCH_ENDPOINT` and `LLM_BENCH_SNAPSHOT` are set.

If `workspace_tests` fails, the covered PR gate rows are marked skipped with a
reason that broad workspace coverage could not be credited, and the nightly
report fails because `workspace_tests` is required.

## Local Validation Tasks

`mise run check-fast` is the default local iteration baseline before broader
gates. It compiles all workspace targets and runs only the fastest cache,
sampler, and API contract slices:

```sh
cargo check --workspace --all-targets
cargo test -p llm-kv-cache
cargo test -p llm-sampler
cargo test -p llm-api --test openai_contract
```

It intentionally does not run `cargo test --workspace`.

Use `mise run gates-ci` to reproduce the PR north-star product gate report.
Use `mise run gates-nightly` for the broad nightly profile, including
`cargo test --workspace --all-features` and the additional nightly-only gates.
Use `mise run test` when you specifically want broad workspace tests without
the north-star report.

Use targeted tasks when a change is isolated to one area:

| Changed area | Mise task | Underlying command |
| --- | --- | --- |
| KV cache | `mise run test-cache` | `cargo test -p llm-kv-cache` |
| Sampler | `mise run test-sampler` | `cargo test -p llm-sampler` |
| API contract | `mise run test-api-contract` | `cargo test -p llm-api --test openai_contract` |
| Runtime contracts | `mise run test-runtime-contract` or a runtime subset task | `cargo test -p llm-runtime --test runtime_contract` |
| Tool parser | `mise run test-parser` | `cargo test -p llm-tool-parser` |
| Parser and tokenizer family changes | `mise run test-parser-tokenizer` | `cargo test -p llm-tool-parser` and `cargo test -p llm-tokenizer` |
| Hub download planning | `mise run test-hub` | `cargo test -p llm-hub --test download_plan` |
| Backend CPU ops | `mise run test-backend-cpu` | `cargo test -p llm-backend --test safetensors_loader` |
| Metal smoke | `mise run test-metal-smoke` | `cargo test -p llm-metal --test metal_smoke -- --test-threads=1` |

These tasks support local triage. CI still runs the workflow jobs listed above,
and release candidates still use the full release checklist.

## Version And Tag Rules

The workspace version is defined once in `[workspace.package]` in
`Cargo.toml`. Crate manifests must inherit `version.workspace = true`.

Release tags must match the workspace version exactly:

```sh
scripts/check-versioning.sh --tag v0.1.0
```

Conventional commit subjects are checked with:

```sh
scripts/check-conventional-commits.sh HEAD~1..HEAD
```

Accepted commit types are `build`, `chore`, `ci`, `docs`, `feat`, `fix`,
`perf`, `refactor`, `revert`, `style`, and `test`.

## Release Notes

Release notes are generated from conventional commit subjects:

```sh
scripts/generate-release-notes.sh v0.1.0 > target/release-notes.md
```

When there is a previous tag, the script uses `previous-tag..current-tag`.
Without a previous tag, it reports all commits reachable from the tag. In CI
without a tag, it previews unreleased notes from the latest tag to `HEAD`.

## Cutting A Release

1. Update `Cargo.toml` `[workspace.package].version`.
2. Run `mise run check`.
3. Create an annotated tag matching the version:

```sh
git tag -a v0.1.0 -m "v0.1.0"
git push origin v0.1.0
```

The Release workflow validates the tag, runs release checks, builds
`llm-engine`, packages the macOS binary, writes a SHA-256 file, generates
release notes, and creates the GitHub release.

## macOS Installer

The public install command is:

```sh
curl -fsSL https://raw.githubusercontent.com/michaelasper/kir-ai/main/scripts/install-macos.sh | bash
```

By default it installs `kirai` into `~/.local/bin` (or `KIR_AI_BIN_DIR`) and
starts the protocol backend with:

```sh
kirai
```

Useful environment controls:

| Variable | Meaning |
| --- | --- |
| `KIR_AI_DIR` | Checkout/install directory. Defaults to `$HOME/.kir-ai/kir-ai` for pipe installs. |
| `KIR_AI_REF` | Branch or tag to install. Defaults to `main`. |
| `KIR_AI_REPO_URL` | Git remote to clone. Defaults to the public repository. |
| `KIR_AI_RUST_TOOLCHAIN` | Rust toolchain. Defaults to `1.95.0`. |
| `KIR_AI_VENV` | Python virtual environment path. Defaults to `.venv` under the checkout. |
| `KIR_AI_BIN_DIR` | Install directory for the `kirai` wrapper. |
| `KIR_AI_FORCE_CLONE` | Set to `1` to exercise clone/ref checkout even when running the script from an existing checkout. |
| `KIR_AI_SKIP_PYTHON` | Set to `1` to skip virtualenv and MLX package installation during installer smoke tests. |
| `KIR_AI_SKIP_BUILD` | Set to `1` to install dependencies without building or running smoke tests. |
| `KIR_AI_SKIP_TESTS` | Set to `1` to skip parser/tokenizer checks. |

For CI or script validation without installing dependencies:

```sh
bash scripts/install-macos.sh --check
```
