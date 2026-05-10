# CI and Release Reference

This page describes the project automation that supports working on kir-ai.
It is reference material for contributors who need to understand what runs,
what each workflow produces, and how releases are cut.

## GitHub Actions

| Workflow | File | Trigger | Purpose | Primary output |
| --- | --- | --- | --- | --- |
| CI | `.github/workflows/ci.yml` | Pull requests and pushes to `main` | Formatting, build, clippy, workspace tests, installer smoke test, and north-star gates | Gate reports and release-notes preview |
| Nightly | `.github/workflows/nightly.yml` | Daily schedule and manual dispatch | Full north-star gate profile plus long-context planning and optional live bench gates | Nightly gate report |
| Release | `.github/workflows/release.yml` | `v*.*.*` tags and manual dispatch | Validate tag/version, run release checks, build `llm-engine`, generate notes, publish GitHub release | macOS release archive, SHA-256 file, release notes |

All workflows run on macOS because Metal smoke coverage and Apple Silicon
serving are first-class project concerns. Actions are pinned by SHA with the
source major version noted in comments.

## CI Jobs

The CI workflow uses separate job names for separate responsibilities:

- `Formatting` runs `cargo fmt --all -- --check`.
- `Workspace Build` runs `cargo build --workspace`.
- `Clippy` runs `cargo clippy --workspace --all-targets --all-features -- -D warnings`.
- `Workspace Tests` runs `cargo test --workspace`.
- `Installer Smoke Test` runs `bash scripts/install-macos.sh --check`.
- `North-Star Gate Report` runs versioning, conventional commits, release-note preview generation, and `scripts/north-star-gates.sh ci`.

The north-star gate script writes JSON, Markdown, and per-gate logs under
`target/north-star-gates/`. CI uploads those reports as artifacts and appends
the Markdown summary to the workflow step summary.

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
