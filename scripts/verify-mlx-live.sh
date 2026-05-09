#!/usr/bin/env bash
set -euo pipefail

endpoint="${LLM_ENGINE_ENDPOINT:-http://127.0.0.1:3000/v1/chat/completions}"
model="${LLM_ENGINE_MODEL:-local-gemma}"
expected="${LLM_ENGINE_EXPECTED_ADAPTIVE_REPLY:-circle/blue}"

payload="$(
  jq -n \
    --arg model "$model" \
    --arg prompt "Remember: shape=circle, color=red. Now change color to blue. Reply exactly as shape/color." \
    '{
      model: $model,
      messages: [{role: "user", content: $prompt}],
      max_tokens: 16,
      temperature: 0
    }'
)"

response="$(curl -fsS "$endpoint" -H 'content-type: application/json' -d "$payload")"
content="$(printf '%s' "$response" | jq -r '.choices[0].message.content // empty')"

if [[ "$content" != "$expected" ]]; then
  echo "MLX live verification failed: expected '$expected', got '$content'" >&2
  printf '%s\n' "$response" >&2
  exit 1
fi

usage="$(printf '%s' "$response" | jq -c '.usage')"
echo "MLX live verification passed: $content"
echo "usage: $usage"
