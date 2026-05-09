#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

version="$(awk '
  /^\[workspace\.package\]$/ { in_workspace_package = 1; next }
  /^\[/ { in_workspace_package = 0 }
  in_workspace_package && /^version = / {
    gsub(/"/, "", $3)
    print $3
  }
' Cargo.toml)"

if [[ -z "$version" ]]; then
  echo "workspace.package.version is missing from Cargo.toml" >&2
  exit 1
fi

if [[ ! "$version" =~ ^([0-9]|[1-9][0-9]*)\.([0-9]|[1-9][0-9]*)\.([0-9]|[1-9][0-9]*)([-+][0-9A-Za-z.-]+)?$ ]]; then
  echo "workspace.package.version is not valid SemVer: $version" >&2
  exit 1
fi

if [[ "$version" != "0.1.0" ]]; then
  echo "workspace SemVer baseline must start at 0.1.0; found $version" >&2
  exit 1
fi

while IFS= read -r manifest; do
  if ! grep -q '^version\.workspace = true$' "$manifest"; then
    echo "$manifest must inherit version.workspace = true" >&2
    exit 1
  fi
done < <(find crates -mindepth 2 -maxdepth 2 -name Cargo.toml | sort)

echo "workspace SemVer baseline verified: $version"
