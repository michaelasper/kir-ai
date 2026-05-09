#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

expected_tag=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --tag)
      expected_tag="${2:-}"
      if [[ -z "$expected_tag" ]]; then
        echo "--tag requires a tag value" >&2
        exit 2
      fi
      shift 2
      ;;
    *)
      echo "usage: bash scripts/check-versioning.sh [--tag vX.Y.Z]" >&2
      exit 2
      ;;
  esac
done

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

if [[ -n "$expected_tag" ]]; then
  expected_tag="${expected_tag#refs/tags/}"
  if [[ "$expected_tag" != "v$version" ]]; then
    echo "release tag $expected_tag does not match workspace version $version; expected v$version" >&2
    exit 1
  fi
fi

if [[ ! -f Cargo.lock ]]; then
  echo "Cargo.lock is required for reproducible builds" >&2
  exit 1
fi

while IFS= read -r manifest; do
  if ! grep -q '^version\.workspace = true$' "$manifest"; then
    echo "$manifest must inherit version.workspace = true" >&2
    exit 1
  fi
done < <(find crates -mindepth 2 -maxdepth 2 -name Cargo.toml | sort)

echo "workspace SemVer verified: $version"
