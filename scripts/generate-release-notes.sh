#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

tag="${1:-${GITHUB_REF_NAME:-unreleased}}"
generated_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

if [[ "$tag" != "unreleased" ]] && git rev-parse -q --verify "refs/tags/$tag" >/dev/null; then
  previous_tag="$(git describe --tags --abbrev=0 "$tag^" 2>/dev/null || true)"
  if [[ -n "$previous_tag" ]]; then
    range="$previous_tag..$tag"
  else
    range="$tag"
  fi
else
  previous_tag="$(git describe --tags --abbrev=0 2>/dev/null || true)"
  if [[ -n "$previous_tag" ]]; then
    range="$previous_tag..HEAD"
  else
    range="HEAD"
  fi
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

features="$tmp_dir/features"
fixes="$tmp_dir/fixes"
docs="$tmp_dir/docs"
ci="$tmp_dir/ci"
maintenance="$tmp_dir/maintenance"
breaking="$tmp_dir/breaking"
touch "$features" "$fixes" "$docs" "$ci" "$maintenance" "$breaking"

while IFS= read -r subject; do
  [[ -z "$subject" ]] && continue
  [[ "$subject" == Merge\ * ]] && continue
  summary="$(printf '%s' "$subject" | sed -E 's/^[a-z]+(\([^)]+\))?!?: //')"
  prefix="${subject%%:*}"
  if [[ "$prefix" == *"!" ]]; then
    printf -- '- %s\n' "$summary" >>"$breaking"
  fi
  case "$subject" in
    feat* ) printf -- '- %s\n' "$summary" >>"$features" ;;
    fix* | perf* ) printf -- '- %s\n' "$summary" >>"$fixes" ;;
    docs* ) printf -- '- %s\n' "$summary" >>"$docs" ;;
    ci* | build* ) printf -- '- %s\n' "$summary" >>"$ci" ;;
    * ) printf -- '- %s\n' "$subject" >>"$maintenance" ;;
  esac
done < <(git log --format=%s --reverse "$range")

emit_section() {
  local title="$1"
  local file="$2"
  if [[ -s "$file" ]]; then
    printf '\n## %s\n\n' "$title"
    cat "$file"
  fi
}

cat <<EOF
# $tag

Generated: $generated_at
Range: \`$range\`

EOF

if [[ -s "$breaking" || -s "$features" || -s "$fixes" || -s "$docs" || -s "$ci" || -s "$maintenance" ]]; then
  emit_section "Breaking Changes" "$breaking"
  emit_section "Features" "$features"
  emit_section "Fixes and Performance" "$fixes"
  emit_section "Documentation" "$docs"
  emit_section "CI and Build" "$ci"
  emit_section "Maintenance" "$maintenance"
else
  echo "No commit subjects were found for this range."
fi
