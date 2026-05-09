#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rust_toolchain="${KIR_AI_RUST_TOOLCHAIN:-1.95.0}"
python_bin="${PYTHON:-python3}"
venv_dir="${KIR_AI_VENV:-$repo_root/.venv}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "kir-ai macOS installer must be run on macOS." >&2
  exit 1
fi

require_command() {
  local command="$1"
  local install_hint="$2"
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Missing required command: $command" >&2
    echo "$install_hint" >&2
    exit 1
  fi
}

require_command git "Install Xcode Command Line Tools with: xcode-select --install"
require_command curl "Install Xcode Command Line Tools with: xcode-select --install"

if ! command -v rustup >/dev/null 2>&1; then
  echo "Installing rustup..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env"
fi

rustup toolchain install "$rust_toolchain" --profile minimal
rustup component add rustfmt clippy --toolchain "$rust_toolchain"

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Missing Python: $python_bin" >&2
  echo "Install Python 3.12+ with Homebrew, mise, pyenv, or python.org." >&2
  exit 1
fi

"$python_bin" -m venv "$venv_dir"
# shellcheck disable=SC1091
source "$venv_dir/bin/activate"
python -m pip install --upgrade pip
python -m pip install --upgrade mlx-lm mlx-vlm

cd "$repo_root"
cargo +"$rust_toolchain" build --workspace
cargo +"$rust_toolchain" test -p llm-tool-parser -p llm-tokenizer

cat <<'EOF'
kir-ai macOS setup complete.

Common next steps:
  source .venv/bin/activate
  cargo run -p llm-engine -- model list
  cargo run -p llm-engine -- serve --snapshot <snapshot-path> --model-id local-model

For MLX-backed Gemma or Qwen, start the matching MLX server first and then run llm-engine with --loader mlx.
EOF
