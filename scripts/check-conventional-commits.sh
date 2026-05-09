#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

range="${1:-}"
if [[ -z "$range" ]]; then
  if [[ -n "${GITHUB_BASE_REF:-}" ]]; then
    git fetch --no-tags --depth=100 origin "$GITHUB_BASE_REF" >/dev/null 2>&1 || true
    range="origin/${GITHUB_BASE_REF}..HEAD"
  elif git rev-parse HEAD~1 >/dev/null 2>&1; then
    range="HEAD~1..HEAD"
  else
    range="HEAD"
  fi
fi

pattern='^(build|chore|ci|docs|feat|fix|perf|refactor|revert|style|test)(\([A-Za-z0-9._/-]+\))?!?: .+'
failed=0

while IFS= read -r subject; do
  [[ -z "$subject" ]] && continue
  [[ "$subject" == Merge\ * ]] && continue
  [[ "$subject" == Revert\ \"* ]] && continue
  if [[ ! "$subject" =~ $pattern ]]; then
    echo "Non-conventional commit subject: $subject" >&2
    failed=1
  fi
done < <(git log --format=%s "$range")

if [[ "$failed" -ne 0 ]]; then
  echo "Expected: type(scope)!: summary, where type is build|chore|ci|docs|feat|fix|perf|refactor|revert|style|test" >&2
  exit 1
fi

echo "conventional commit subjects verified for $range"
