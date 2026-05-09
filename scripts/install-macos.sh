#!/usr/bin/env bash
set -euo pipefail

check_only=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --check)
      check_only=1
      shift
      ;;
    *)
      echo "usage: bash scripts/install-macos.sh [--check]" >&2
      exit 2
      ;;
  esac
done

repo_url="${KIR_AI_REPO_URL:-https://github.com/michaelasper/kir-ai.git}"
repo_ref="${KIR_AI_REF:-main}"
install_dir="${KIR_AI_DIR:-$HOME/.kir-ai/kir-ai}"
rust_toolchain="${KIR_AI_RUST_TOOLCHAIN:-1.95.0}"
python_bin="${PYTHON:-python3}"

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

script_source="${BASH_SOURCE[0]:-}"
candidate_root=""
if [[ -n "$script_source" && -f "$script_source" ]]; then
  candidate_root="$(cd "$(dirname "$script_source")/.." && pwd)"
fi

using_local_source=0
if [[ "${KIR_AI_FORCE_CLONE:-0}" != "1" && -n "$candidate_root" && -f "$candidate_root/Cargo.toml" ]]; then
  repo_root="$candidate_root"
  using_local_source=1
else
  repo_root="$install_dir"
fi

venv_dir="${KIR_AI_VENV:-$repo_root/.venv}"

if [[ "$check_only" -eq 1 ]]; then
  cat <<EOF
kir-ai installer check passed.

Repository: $repo_url
Reference:  $repo_ref
Target:     $repo_root
Toolchain:  $rust_toolchain
Python:     $python_bin
EOF
  exit 0
fi

if [[ "$using_local_source" -eq 1 ]]; then
  echo "Using local source tree at $repo_root"
elif [[ ! -f "$repo_root/Cargo.toml" ]]; then
  mkdir -p "$(dirname "$repo_root")"
  git clone --filter=blob:none "$repo_url" "$repo_root"
  git -C "$repo_root" fetch --tags origin "$repo_ref" || git -C "$repo_root" fetch --tags origin
  if git -C "$repo_root" show-ref --verify --quiet "refs/remotes/origin/$repo_ref"; then
    git -C "$repo_root" checkout -B "$repo_ref" "origin/$repo_ref"
  else
    git -C "$repo_root" checkout --detach "$repo_ref"
  fi
elif [[ -d "$repo_root/.git" ]]; then
  git -C "$repo_root" fetch --tags origin "$repo_ref" || git -C "$repo_root" fetch --tags origin
  if git -C "$repo_root" show-ref --verify --quiet "refs/remotes/origin/$repo_ref"; then
    git -C "$repo_root" checkout -B "$repo_ref" "origin/$repo_ref"
    git -C "$repo_root" pull --ff-only origin "$repo_ref"
  else
    git -C "$repo_root" checkout --detach "$repo_ref"
  fi
else
  echo "Using existing source tree at $repo_root"
fi

if ! command -v rustup >/dev/null 2>&1; then
  echo "Installing rustup..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env"
fi

rustup toolchain install "$rust_toolchain" --profile minimal
rustup component add rustfmt clippy --toolchain "$rust_toolchain"

if [[ "${KIR_AI_SKIP_PYTHON:-0}" != "1" ]]; then
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
fi

cd "$repo_root"
if [[ "${KIR_AI_SKIP_BUILD:-0}" != "1" ]]; then
  cargo +"$rust_toolchain" build --workspace
  cargo +"$rust_toolchain" test -p llm-tool-parser -p llm-tokenizer
fi

cat <<'EOF'
kir-ai macOS setup complete.

Common next steps:
  source .venv/bin/activate
  cargo run -p llm-engine -- model list
  cargo run -p llm-engine -- serve --snapshot <snapshot-path> --model-id local-model

For MLX-backed Gemma or Qwen, start the matching MLX server first and then run llm-engine with --loader mlx.
EOF
