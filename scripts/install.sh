#!/usr/bin/env bash
set -euo pipefail

# kir-ai installation script
# Clones the repository, builds the release binary, and installs it.

repo_url="https://github.com/michaelasper/kir-ai.git"
repo_ref="${KIR_AI_REF:-main}"
install_root="${KIR_AI_INSTALL_ROOT:-$HOME/.kir-ai}"
repo_root="${KIR_AI_DIR:-$install_root/kir-ai}"
rust_toolchain="${KIR_AI_RUST_TOOLCHAIN:-1.95.0}"
python_bin="${PYTHON:-python3}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
RESET='\033[0m'

echo -e "${BOLD}${BLUE}🚀 Installing kir-ai...${RESET}"

# Determine installation directory
resolve_bin_dir() {
  if [[ -n "${KIR_AI_BIN_DIR:-}" ]]; then
    printf '%s' "${KIR_AI_BIN_DIR}"
    return
  fi

  local local_bin="$HOME/.local/bin"
  mkdir -p "$local_bin"
  printf '%s' "$local_bin"
}

install_bin_dir="$(resolve_bin_dir)"

# Check requirements
require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo -e "${RED}Error: Missing required command: $1${RESET}"
    echo -e "$2"
    exit 1
  fi
}

require_command git "Please install git."
require_command curl "Please install curl."

# Clone or update repository
if [[ ! -d "$repo_root" ]]; then
  echo -e "${BLUE}Cloning repository to ${repo_root}...${RESET}"
  mkdir -p "$(dirname "$repo_root")"
  git clone --filter=blob:none "$repo_url" "$repo_root"
fi

cd "$repo_root"
echo -e "${BLUE}Checking out ${repo_ref}...${RESET}"
git fetch origin "$repo_ref"
git checkout "$repo_ref"
git pull origin "$repo_ref" --ff-only || true

# Install Rust if missing
if ! command -v rustup >/dev/null 2>&1; then
  echo -e "${BLUE}Installing rustup...${RESET}"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  source "$HOME/.cargo/env"
fi

# Build binary
echo -e "${BLUE}Building kir-ai release binary...${RESET}"
cargo build --release -p llm-engine

# Install binary and shim
echo -e "${BLUE}Installing binary to ${install_bin_dir}...${RESET}"
mkdir -p "$install_bin_dir"
engine_bin="$repo_root/target/release/llm-engine"

kirai_bin="$install_bin_dir/kirai"
cat > "$kirai_bin" <<SHIM
#!/usr/bin/env bash
set -euo pipefail
ENGINE_BIN="${engine_bin}"
if [[ ! -x "\$ENGINE_BIN" ]]; then
  echo "kirai error: engine binary not found or not executable at \$ENGINE_BIN" >&2
  exit 1
fi
if [[ "\$#" -eq 0 ]]; then
  exec "\$ENGINE_BIN" serve --protocol-test-backend
fi
case "\$1" in
  -h|--help|help)
    exec "\$ENGINE_BIN" --help
    ;;
  protocol|run-protocol|protocol-backend)
    shift
    exec "\$ENGINE_BIN" serve --protocol-test-backend "\$@"
    ;;
  *)
    exec "\$ENGINE_BIN" "\$@"
    ;;
esac
SHIM
chmod +x "$kirai_bin"

echo -e "\n${BOLD}${GREEN}✓ kir-ai installed successfully!${RESET}"
echo -e "\n${BOLD}Quick Start:${RESET}"
echo -e "  • Run protocol backend: ${BLUE}kirai${RESET}"
echo -e "  • Show help:           ${BLUE}kirai --help${RESET}"

if [[ ":$PATH:" != *":$install_bin_dir:"* ]]; then
  echo -e "\n${YELLOW}Note: Add ${install_bin_dir} to your PATH to run 'kirai' from anywhere:${RESET}"
  echo -e "  ${BLUE}export PATH=\"$install_bin_dir:\$PATH\"${RESET}"
fi
