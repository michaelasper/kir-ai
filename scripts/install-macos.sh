#!/usr/bin/env bash
set -euo pipefail

check_only=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --check)
      check_only=1
      shift
      ;;
    --help|-h)
      cat <<'USAGE'
Usage: bash scripts/install-macos.sh [--check]

Installs kirai to a local bin directory and builds/installs llm-engine.

Environment variables:
  KIR_AI_INSTALL_ROOT      Base directory for checkout + binaries (default: $HOME/.kir-ai)
  KIR_AI_DIR               Override checkout path
  KIR_AI_REF               Git ref to install (default: main)
  KIR_AI_REPO_URL          Repository URL
  KIR_AI_RUST_TOOLCHAIN    Rust toolchain version (default: 1.95.0)
  KIR_AI_BIN_DIR           Binary install directory (default: ~/.local/bin)
  KIR_AI_SKIP_PYTHON       Set to 1 to skip Python/MLX package setup
  KIR_AI_SKIP_BUILD        Set to 1 to skip compiling llm-engine
  KIR_AI_SKIP_TESTS        Set to 1 to skip parser/tokenizer checks
USAGE
      exit 0
      ;;
    *)
      echo "usage: bash scripts/install-macos.sh [--check]" >&2
      exit 2
      ;;
  esac
done

repo_url="${KIR_AI_REPO_URL:-https://github.com/michaelasper/kir-ai.git}"
repo_ref="${KIR_AI_REF:-main}"
install_root="${KIR_AI_INSTALL_ROOT:-$HOME/.kir-ai}"
repo_root="${KIR_AI_DIR:-$install_root/kir-ai}"
rust_toolchain="${KIR_AI_RUST_TOOLCHAIN:-1.95.0}"
python_bin="${PYTHON:-python3}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "kir-ai macOS installer must be run on macOS." >&2
  exit 1
fi

resolve_bin_dir() {
  if [[ -n "${KIR_AI_BIN_DIR:-}" ]]; then
    printf '%s' "${KIR_AI_BIN_DIR}"
    return
  fi

  local local_bin="$HOME/.local/bin"
  if [[ -w "$(dirname "$local_bin")" ]]; then
    mkdir -p "$local_bin"
    printf '%s' "$local_bin"
    return
  fi

  printf '%s' "$install_root/bin"
}

install_bin_dir="$(resolve_bin_dir)"

require_command() {
  local command_name="$1"
  local install_hint="$2"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing required command: $command_name" >&2
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
Binary dir: $install_bin_dir
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
fi

if [[ -s "$HOME/.cargo/env" ]]; then
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
  cargo +"$rust_toolchain" build -p llm-engine --release
fi
if [[ "${KIR_AI_SKIP_TESTS:-0}" != "1" ]]; then
  cargo +"$rust_toolchain" test -p llm-tool-parser -p llm-tokenizer
fi

mkdir -p "$install_bin_dir"
engine_bin="$repo_root/target/release/llm-engine"
if [[ ! -x "$engine_bin" ]]; then
  echo "llm-engine binary was not found at $engine_bin" >&2
  exit 1
fi

kirai_bin="$install_bin_dir/kirai"
cat > "$kirai_bin" <<SHIM
#!/usr/bin/env bash
set -euo pipefail

ENGINE_BIN="\${KIR_AI_ENGINE_BIN:-$engine_bin}"

if [[ ! -x "$ENGINE_BIN" ]]; then
  echo "kirai is installed, but llm-engine is not executable: ${ENGINE_BIN}" >&2
  exit 1
fi

if [[ "$#" -eq 0 ]]; then
  exec "$ENGINE_BIN" serve --protocol-test-backend
fi

case "$1" in
  -h|--help|help)
    exec "$ENGINE_BIN" --help
    ;;
  protocol|run-protocol|protocol-backend)
    shift
    exec "$ENGINE_BIN" serve --protocol-test-backend "$@"
    ;;
  *)
    exec "$ENGINE_BIN" "$@"
    ;;
esac
SHIM
chmod +x "$kirai_bin"

printf '\nkirai command installed at: %s\n' "$kirai_bin"
printf 'Quick start (protocol backend):\n  kirai\n\n'
printf 'Run with explicit arguments to llm-engine:\n  kirai serve --help\n'

case ":$PATH:" in
  *":$install_bin_dir:"*)
    ;;
  *)
    printf '\nAdd this directory to your shell path if needed:\n  export PATH=\"%s:$PATH\"\\n' "$install_bin_dir"
    ;;
esac
