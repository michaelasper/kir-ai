#!/usr/bin/env bash
set -uo pipefail

profile="${1:-ci}"
case "$profile" in
  ci|nightly) ;;
  *)
    echo "usage: bash scripts/north-star-gates.sh <ci|nightly>" >&2
    exit 2
    ;;
esac

out_dir="${NORTH_STAR_GATE_DIR:-target/north-star-gates}"
log_dir="$out_dir/logs"
mkdir -p "$out_dir"
mkdir -p "$log_dir"
json_report="$out_dir/${profile}-report.json"
markdown_report="$out_dir/${profile}-report.md"
commit="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
started_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
os_name="$(uname -s)"
arch_name="$(uname -m)"
rust_version="$(rustc --version 2>/dev/null || echo unknown)"
cargo_version="$(cargo --version 2>/dev/null || echo unknown)"
failures=0
first_gate=1

json_escape() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g; s/	/\\t/g'
}

safe_name() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9_.-' '_'
}

json_field() {
  json_escape "$1"
}

begin_reports() {
  cat >"$json_report" <<JSON
{
  "schema_version": 1,
  "profile": "$(json_field "$profile")",
  "commit": "$(json_field "$commit")",
  "branch": "$(json_field "$branch")",
  "started_at": "$(json_field "$started_at")",
  "hardware": {
    "os": "$(json_field "$os_name")",
    "arch": "$(json_field "$arch_name")"
  },
  "toolchain": {
    "rustc": "$(json_field "$rust_version")",
    "cargo": "$(json_field "$cargo_version")"
  },
  "cache": {
    "LLM_MODEL_HOME": "$(json_field "${LLM_MODEL_HOME:-}")",
    "LLM_BENCH_ENDPOINT": "$(json_field "${LLM_BENCH_ENDPOINT:-}")",
    "LLM_BENCH_MODEL": "$(json_field "${LLM_BENCH_MODEL:-}")",
    "LLM_BENCH_SNAPSHOT": "$(json_field "${LLM_BENCH_SNAPSHOT:-}")",
    "LLM_BENCH_BASELINE": "$(json_field "${LLM_BENCH_BASELINE:-}")"
  },
  "gates": [
JSON

  cat >"$markdown_report" <<MD
# North-Star ${profile} Gate Report

- Commit: \`${commit}\`
- Branch: \`${branch}\`
- Started: \`${started_at}\`
- Host: \`${os_name}/${arch_name}\`
- Toolchain: \`${rust_version}\`; \`${cargo_version}\`

| Gate | Status | Required | Duration | Class | Log | Reason |
| --- | --- | --- | ---: | --- | --- | --- |
MD
}

append_gate() {
  local name="$1"
  local status="$2"
  local required="$3"
  local failure_class="$4"
  local reason="$5"
  local command="$6"
  local duration="$7"
  local log_path="$8"

  if [ "$first_gate" -eq 0 ]; then
    printf ',\n' >>"$json_report"
  fi
  first_gate=0
  cat >>"$json_report" <<JSON
    {
      "name": "$(json_field "$name")",
      "status": "$(json_field "$status")",
      "required": $required,
      "failure_class": "$(json_field "$failure_class")",
      "reason": "$(json_field "$reason")",
      "command": "$(json_field "$command")",
      "log_path": "$(json_field "$log_path")",
      "duration_seconds": $duration
    }
JSON
  local log_cell=" "
  if [ -n "$log_path" ]; then
    log_cell="\`${log_path}\`"
  fi
  printf '| `%s` | `%s` | `%s` | `%ss` | `%s` | %s | %s |\n' \
    "$name" "$status" "$required" "$duration" "${failure_class:- }" "$log_cell" "${reason:- }" >>"$markdown_report"
}

run_gate() {
  local name="$1"
  local required="$2"
  shift 2
  local command="$*"
  local log_path="$log_dir/$(safe_name "$name").log"
  local start_seconds
  local end_seconds
  local duration
  start_seconds="$(date +%s)"
  echo "==> north-star gate: $name"
  {
    echo "gate: $name"
    echo "required: $required"
    echo "command: $command"
    echo "started_at: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo
  } >"$log_path"
  if "$@" >>"$log_path" 2>&1; then
    end_seconds="$(date +%s)"
    duration=$((end_seconds - start_seconds))
    append_gate "$name" "passed" "$required" "none" "" "$command" "$duration" "$log_path"
    return 0
  else
    end_seconds="$(date +%s)"
    duration=$((end_seconds - start_seconds))
    append_gate "$name" "failed" "$required" "command_failed" "command exited non-zero" "$command" "$duration" "$log_path"
    echo "north-star gate failed: $name" >&2
    tail -40 "$log_path" >&2
    if [ "$required" = "true" ]; then
      failures=$((failures + 1))
    fi
    return 1
  fi
}

skip_gate() {
  local name="$1"
  local required="$2"
  local reason="$3"
  local command="$4"
  echo "==> north-star gate: $name skipped: $reason"
  append_gate "$name" "skipped" "$required" "skipped" "$reason" "$command" 0 ""
}

covered_gate() {
  local name="$1"
  local required="$2"
  local reason="$3"
  local command="$4"
  echo "==> north-star gate: $name covered: $reason"
  append_gate "$name" "covered" "$required" "covered_by_workspace" "$reason" "$command" 0 ""
}

run_ci_gates() {
  run_gate "protocol_api_contracts" true cargo test -p llm-api --test openai_contract
  run_gate "runtime_agentic_contracts" true cargo test -p llm-runtime --test runtime_contract --all-features
  run_gate "engine_http_contracts" true cargo test -p llm-engine --test http_contract --all-features
  run_gate "engine_model_cli_contracts" true cargo test -p llm-engine --test model_cli --all-features
  run_gate "model_acquisition_contracts" true cargo test -p llm-hub
  run_gate "model_family_backend_profiles" true cargo test -p llm-models --test family_adapter
  run_gate "deferred_family_contracts" true bash -lc 'cargo test -p llm-tokenizer --test deepseek_template && cargo test -p llm-tokenizer --test gemma_template && cargo test -p llm-tokenizer --test llama_template && cargo test -p llm-tool-parser --test deepseek_parser && cargo test -p llm-tool-parser --test gemma_parser && cargo test -p llm-tool-parser --test llama_parser'
  run_gate "tokenizer_parser_contracts" true bash -lc 'cargo test -p llm-tokenizer && cargo test -p llm-tool-parser'
}

record_workspace_covered_ci_gates() {
  local reason="covered by workspace_tests (cargo test --workspace --all-features)"
  covered_gate "protocol_api_contracts" true "$reason" "cargo test -p llm-api --test openai_contract"
  covered_gate "runtime_agentic_contracts" true "$reason" "cargo test -p llm-runtime --test runtime_contract --all-features"
  covered_gate "engine_http_contracts" true "$reason" "cargo test -p llm-engine --test http_contract --all-features"
  covered_gate "engine_model_cli_contracts" true "$reason" "cargo test -p llm-engine --test model_cli --all-features"
  covered_gate "model_acquisition_contracts" true "$reason" "cargo test -p llm-hub"
  covered_gate "model_family_backend_profiles" true "$reason" "cargo test -p llm-models --test family_adapter"
  covered_gate "deferred_family_contracts" true "$reason" "cargo test -p llm-tokenizer --test deepseek_template && cargo test -p llm-tokenizer --test gemma_template && cargo test -p llm-tokenizer --test llama_template && cargo test -p llm-tool-parser --test deepseek_parser && cargo test -p llm-tool-parser --test gemma_parser && cargo test -p llm-tool-parser --test llama_parser"
  covered_gate "tokenizer_parser_contracts" true "$reason" "cargo test -p llm-tokenizer && cargo test -p llm-tool-parser"
}

skip_workspace_covered_ci_gates() {
  local reason="workspace_tests failed before coverage could be credited"
  skip_gate "protocol_api_contracts" true "$reason" "cargo test -p llm-api --test openai_contract"
  skip_gate "runtime_agentic_contracts" true "$reason" "cargo test -p llm-runtime --test runtime_contract --all-features"
  skip_gate "engine_http_contracts" true "$reason" "cargo test -p llm-engine --test http_contract --all-features"
  skip_gate "engine_model_cli_contracts" true "$reason" "cargo test -p llm-engine --test model_cli --all-features"
  skip_gate "model_acquisition_contracts" true "$reason" "cargo test -p llm-hub"
  skip_gate "model_family_backend_profiles" true "$reason" "cargo test -p llm-models --test family_adapter"
  skip_gate "deferred_family_contracts" true "$reason" "cargo test -p llm-tokenizer --test deepseek_template && cargo test -p llm-tokenizer --test gemma_template && cargo test -p llm-tokenizer --test llama_template && cargo test -p llm-tool-parser --test deepseek_parser && cargo test -p llm-tool-parser --test gemma_parser && cargo test -p llm-tool-parser --test llama_parser"
  skip_gate "tokenizer_parser_contracts" true "$reason" "cargo test -p llm-tokenizer && cargo test -p llm-tool-parser"
}

record_workspace_covered_nightly_gates() {
  local reason="covered by workspace_tests (cargo test --workspace --all-features)"
  covered_gate "no_progress_replay_classifiers" true "$reason" "cargo test -p llm-runtime --test runtime_contract no_progress_transcript_replay_fixtures_return_stable_codes"
  covered_gate "native_backend_contracts" true "$reason" "cargo test -p llm-backend --tests"
  if [ "$os_name" = "Darwin" ]; then
    covered_gate "metal_smoke_contracts" true "$reason" "cargo test -p llm-metal --test metal_smoke"
  else
    skip_gate "metal_smoke_contracts" false "Metal smoke tests require a macOS runner" "cargo test -p llm-metal --test metal_smoke"
  fi
}

skip_workspace_covered_nightly_gates() {
  local reason="workspace_tests failed before coverage could be credited"
  skip_gate "no_progress_replay_classifiers" true "$reason" "cargo test -p llm-runtime --test runtime_contract no_progress_transcript_replay_fixtures_return_stable_codes"
  skip_gate "native_backend_contracts" true "$reason" "cargo test -p llm-backend --tests"
  if [ "$os_name" = "Darwin" ]; then
    skip_gate "metal_smoke_contracts" true "$reason" "cargo test -p llm-metal --test metal_smoke"
  else
    skip_gate "metal_smoke_contracts" false "Metal smoke tests require a macOS runner" "cargo test -p llm-metal --test metal_smoke"
  fi
}

run_nightly_gates() {
  if run_gate "workspace_tests" true cargo test --workspace --all-features; then
    record_workspace_covered_ci_gates
    record_workspace_covered_nightly_gates
  else
    skip_workspace_covered_ci_gates
    skip_workspace_covered_nightly_gates
  fi
  run_gate "slow_timeout_stall_contracts" true cargo test -p llm-engine --lib mlx_slow_ -- --ignored --test-threads=1
  run_gate "qwen_long_context_plan" true cargo run -p llm-engine -- bench qwen-long-context --dry-run --profile all --output "$out_dir/qwen-long-context-plan.json"

  if [ -n "${LLM_BENCH_ENDPOINT:-}" ] && [ -n "${LLM_BENCH_SNAPSHOT:-}" ]; then
    local baseline_args=()
    if [ -n "${LLM_BENCH_BASELINE:-}" ]; then
      baseline_args=(--baseline "$LLM_BENCH_BASELINE")
    fi
    run_gate "qwen_135k_release_gate" true cargo run -p llm-engine -- bench qwen-long-context --profile 135k --endpoint "$LLM_BENCH_ENDPOINT" --model "${LLM_BENCH_MODEL:-local-qwen36}" --snapshot "$LLM_BENCH_SNAPSHOT" --output "$out_dir/qwen-135k-report.json" "${baseline_args[@]}"
    run_gate "qwen_200k_characterization" false cargo run -p llm-engine -- bench qwen-long-context --profile 200k --endpoint "$LLM_BENCH_ENDPOINT" --model "${LLM_BENCH_MODEL:-local-qwen36}" --snapshot "$LLM_BENCH_SNAPSHOT" --output "$out_dir/qwen-200k-report.json" "${baseline_args[@]}"
  else
    skip_gate "qwen_135k_release_gate" false "LLM_BENCH_ENDPOINT and LLM_BENCH_SNAPSHOT are required for real long-context inference" "cargo run -p llm-engine -- bench qwen-long-context --profile 135k ..."
    skip_gate "qwen_200k_characterization" false "LLM_BENCH_ENDPOINT and LLM_BENCH_SNAPSHOT are required for real long-context inference" "cargo run -p llm-engine -- bench qwen-long-context --profile 200k ..."
  fi
}

finish_reports() {
  local finished_at
  local status
  finished_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  if [ "$failures" -eq 0 ]; then
    status="passed"
  else
    status="failed"
  fi
  cat >>"$json_report" <<JSON

  ],
  "finished_at": "$(json_field "$finished_at")",
  "status": "$(json_field "$status")",
  "required_failures": $failures
}
JSON
  {
    echo
    echo "- Finished: \`${finished_at}\`"
    echo "- Status: \`${status}\`"
    echo "- Required failures: \`${failures}\`"
  } >>"$markdown_report"
  echo "north-star gate report: $json_report"
}

begin_reports
if [ "$profile" = "ci" ]; then
  run_ci_gates
else
  run_nightly_gates
fi
finish_reports

if [ "$failures" -ne 0 ]; then
  exit 1
fi
