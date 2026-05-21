# CLI Reference

The `llm-engine` binary is the HTTP server and model tooling CLI. Argument
parsing is manual. Flags use `--flag value`; boolean flags are present or absent.

## Synopsis

```sh
llm-engine [serve]
llm-engine serve [--addr <host:port>] [--tls-cert <path> --tls-key <path>] [--protocol-test-backend --i-understand-this-is-not-real-inference | --snapshot <path> | --snapshot-alias <alias>] [--snapshot-readiness <fast|deep>] [--loader <native-metal|mlx>] [--family <qwen|deep_seek|gemma|llama>] [--model-id <id>] [--max-new-tokens <n>] [--max-prefill-tokens <n>] [--max-json-body-bytes <bytes>] [--max-message-content-bytes <bytes>] [--max-completion-prompt-bytes <bytes>] [--max-public-inference-requests-per-second <n>] [--mlx-endpoint <url>] [--native-prefix-cache-bytes <bytes>] [--native-metal-weight-cache-bytes <bytes>] [--warm-native-metal-weight-cache] [--canonical-tool-schemas]
llm-engine bench qwen-long-context [--endpoint <url> --snapshot <path> | --lane <spec> ...]
llm-engine bench qwen-mlx-tool-normalized --lane <spec> [--lane <spec> ...]
llm-engine model <subcommand> ...
```

If no command is provided, `llm-engine` defaults to `serve`, which still
requires either `--snapshot <path>` or acknowledged protocol test mode.

When running through Cargo:

```sh
cargo run -p llm-engine --features test-utils -- serve \
  --protocol-test-backend \
  --i-understand-this-is-not-real-inference
cargo run -p llm-engine -- model list
```

## `serve`

Starts the Axum HTTP server.

```sh
llm-engine serve \
  --addr 127.0.0.1:3000 \
  --snapshot <snapshot-path> \
  --model-id local-qwen36 \
  --max-new-tokens 256 \
  --max-prefill-tokens 2048 \
  --native-metal-weight-cache-bytes 8589934592
```

| Flag | Default | Description |
| --- | --- | --- |
| `--addr <host:port>` | `127.0.0.1:3000` | Socket address to bind. |
| `--tls-cert <path>` | none | PEM certificate chain for built-in HTTPS. Must be provided with `--tls-key`. When omitted, the server keeps the existing plain HTTP behavior. |
| `--tls-key <path>` | none | PEM private key for built-in HTTPS. Must be provided with `--tls-cert`. The key material is read at startup and is not logged. |
| `--protocol-test-backend` | absent | Enables protocol test mode without model artifacts. Intended for tests and client integration. Requires the `test-utils` feature and `--i-understand-this-is-not-real-inference`. |
| `--deterministic-test-backend` | absent | Deprecated compatibility alias for `--protocol-test-backend`; it has the same feature and acknowledgement requirements. |
| `--snapshot <path>` | none | Enables manifest-selected serving from a local snapshot directory. `loader: native-metal` opens native text execution for supported families; `loader: mlx` opens the loopback MLX sidecar backend. |
| `--snapshot-alias <alias>` / `--model-alias <alias>` | none | Resolves a snapshot path from the model store alias records and verifies the recorded manifest digest before serving. |
| `--snapshot-readiness <fast\|deep>` | `fast` | Selects the startup readiness check for promoted manifest snapshots. Fast parses the manifest and checks required file classes, file presence/sizes, and safetensors index coverage without hashing weights. Deep hashes every manifest file and performs the same runnable readiness checks before opening the backend. |
| `--loader <native-metal\|mlx>` / `--backend <native-metal\|mlx>` | manifest or `native-metal` | Overrides the snapshot loader when no Kir manifest is present. Fails if it conflicts with an existing manifest. |
| `--family <qwen\|deep_seek\|gemma\|llama>` | manifest metadata or native `config.json` detection | Supplies model-family metadata for raw snapshots without a Kir manifest. Raw native snapshots infer Qwen or Gemma from `config.json` when omitted. Raw MLX snapshots must set this explicitly. |
| `--model-id <id>` | `local-qwen36` or snapshot alias | Served model alias. Used with `--snapshot`; protocol test mode also serves `local-qwen36`. |
| `--max-new-tokens <u32>` | `256` | Native text generation cap per request. Values below `1` are clamped to `1`. |
| `--max-prefill-tokens <usize>` | `2048` | Native text prefill chunk size. Long-context native serving depends on a large value here because prefill runs sequentially by chunk; values below `1` are clamped to `1`, and prompt retention is sized from the accepted prompt plus generation budget and fails closed at the model context limit. Lower this only for memory-constrained correctness probes. |
| `--max-json-body-bytes <usize>` | `16777216` | HTTP JSON request body cap for API and admin JSON routes. Values below `1` are rejected. |
| `--max-message-content-bytes <usize>` | `8388608` | Per-message chat `content` byte cap after JSON parsing. Values below `1` are rejected. |
| `--max-completion-prompt-bytes <usize>` | `8388608` | Text completion `prompt` byte cap after JSON parsing. Values below `1` are rejected. |
| `--max-public-inference-requests-per-second <usize>` | `60` | Global fixed-window rate limit for public `/v1/chat/completions` and `/v1/completions` requests. Values below `1` are rejected. Exceeded requests return `429` with `Retry-After`. |
| `--mlx-endpoint <url>` | `http://127.0.0.1:8080/v1` | Loopback `mlx_lm.server` or `mlx_vlm.server` `/v1` endpoint for MLX manifests. Remote endpoints are rejected. `MLX_LM_ENDPOINT` is used when this flag is omitted. |
| `--native-prefix-cache-bytes <u64>` | `536870912` | Per-backend native Qwen/Gemma prefix-cache budget. Set `0` to reject stores while still allowing generation without prefix reuse. `LLM_ENGINE_PREFIX_CACHE_BYTES` is used when this flag is omitted. |
| `--native-metal-weight-cache-bytes <u64>` | `8589934592` | Per-backend Metal BF16 weight-buffer LRU budget. Set `0` to disable weight-buffer caching. |
| `--warm-native-metal-weight-cache` | absent | Preloads rank-2 BF16 tensors into the Metal weight-buffer cache at startup until the configured budget is full. |
| `--canonical-tool-schemas` | absent | Opts production serving into canonical tool schema rendering/cache keys. Equivalent JSON object key order and string-only `required` array order normalize to one minified schema. |

Set `LLM_ENGINE_CANONICAL_TOOL_SCHEMAS=1` to enable the same tool-schema
canonicalization without the flag. Omit the flag/env to preserve existing
OpenAI-compatible request serialization.

HTTP remains the default transport. To serve HTTPS directly, pass both
`--tls-cert` and `--tls-key`; the certificate file may contain a PEM chain and
the key file must contain one PEM private key. For non-loopback deployments that
omit these flags, terminate TLS at a reverse proxy such as Caddy, nginx, or an
equivalent local ingress.

Stable-prefix serving for Qwen agent traffic is `--canonical-tool-schemas` plus
Qwen family metadata. Kir records Qwen chat-template kwargs as
`{"enable_thinking":false}` in cache identity and forwards the same kwargs to
MLX chat-completion sidecars.

MLX prompt-cache control is a sidecar launch policy, not a Kir request-body
field. Launch `mlx_lm.server`/`mlx_vlm.server` with `--prompt-cache-size` or
`--prompt-cache-bytes` and keep request serialization stable. Kir records the
stable cache identity in backend/admin metrics and consumes upstream
`usage.prompt_tokens_details.cached_tokens` when the sidecar reports it, but it
does not send unsupported `cache_key`, `session_id`, or `prompt_cache_key`
fields to MLX request bodies.

The default request-size limits accept the current long-context benchmark
payloads, including the synthetic 135k stable-prefix probe. Lower the three
request-limit flags for small deployments that need stricter ingress bounds.

Without `--snapshot`, `serve` exits unless protocol test mode is explicitly
selected and acknowledged. Implicit no-snapshot stub serving was removed.

With a native-metal snapshot, the directory must contain `config.json`,
`tokenizer.json`, `model.safetensors.index.json`, and all referenced shard
files. With an MLX snapshot promoted by `llm-engine model pull`, the directory
must include an `llm-engine-manifest.json` whose loader is `mlx`, and a
compatible MLX sidecar must already be listening on the configured loopback
endpoint. Chat requests for Qwen, DeepSeek, Gemma, and Llama use OpenAI-compatible
`/v1/chat/completions` so the MLX sidecar owns model-specific chat templating
and receives lossless structured tool history; legacy text completion requests use a
completions-capable sidecar endpoint when the selected family exposes one. Raw
Hugging Face cache snapshots served through native-metal infer Qwen or Gemma
from `config.json` when no manifest is present. Raw Hugging Face cache
snapshots served through MLX need both `--loader mlx` and a serveable `--family`
such as `qwen`, `deep_seek`, `gemma`, or `llama` so family metadata and parser
policy are explicit. `--loader mlx` without a family fails at startup for raw
snapshots.

## `bench qwen-long-context`

Runs or plans the Qwen long-context promotion and characterization benchmark.

Single-lane usage keeps the original flags:

```sh
llm-engine bench qwen-long-context \
  --profile 135k \
  --endpoint http://127.0.0.1:3000 \
  --model local-qwen36 \
  --snapshot "$SNAPSHOT"
```

Named lanes run the same profile and cases against multiple backends and emit
side-by-side comparison metadata:

```sh
llm-engine bench qwen-long-context \
  --profile 135k \
  --lane name=native-metal,endpoint=http://127.0.0.1:3000,snapshot="$NATIVE_SNAPSHOT",model=local-native \
  --lane name=mlx,endpoint=http://127.0.0.1:3001,snapshot="$MLX_SNAPSHOT",model=local-mlx
```

| Flag | Default | Description |
| --- | --- | --- |
| `--endpoint <url>` | none | OpenAI-compatible base URL for the primary single-lane run. |
| `--snapshot <path>` | none | Snapshot used for tokenizer planning and model identity in a single-lane run. |
| `--lane <spec>` | none | Adds a named lane. Specs are comma-separated `key=value` pairs: `name`, `endpoint`, `snapshot`, and optional `model`/`model_id`. |
| `--profile <135k\|200k\|256k\|all>` | `135k` | Selects the release-blocking 135K profile, frontier 200K profile, max-context 256K profile, or all profiles. |
| `--baseline <path>` | none | Previous trace JSON to compare against on matching hardware/model class. |
| `--output <path>` | none | Writes the full JSON trace to disk as well as stdout. |
| `--admin-token <token>` | none | Optional bearer token used when capturing each lane's `/admin/metrics` snapshot. |
| `--dry-run` | absent | Emits the planned profiles, cases, lanes, model identities, and cache policy without HTTP requests. |

The trace keeps top-level `model` and `profiles` for compatibility with older
single-lane consumers. New consumers should read `lanes[*].profiles` and
`comparison`, which includes per-case latency, token throughput, pass/fail
classification, and fastest-lane summaries. Streaming cases also include explicit
timing fields where observed: `first_byte_latency_ms` for the first non-empty HTTP
body chunk, `first_sse_data_latency_ms` for the first valid non-empty SSE JSON
`data:` frame, `first_content_delta_latency_ms` for the first non-empty content
delta, `first_tool_delta_latency_ms` for the first non-empty tool-call delta, and
`first_semantic_delta_latency_ms` for the first content or tool delta. SSE
comments, blank data frames, `[DONE]`, and usage-only frames do not count as
semantic output. The case matrix includes native prefix-cache probes for running
the same long prompt twice and for varying only a short suffix after a shared long
prefix. `cache_policy.env` records `LLM_ENGINE_PREFIX_CACHE_BYTES` when set so
benchmark traces identify the native prefix-cache budget used by the served lane.
When `/admin/metrics` is available, each lane also includes `cache_metrics` with
prefix-cache hit rate, hit-token and miss-token counters, residency, Metal BF16
weight-cache hit rate/residency, KV-cache residency, recurrent
linear-attention-cache residency, and eviction churn signals. Each executed case
records `admin_metrics.prefix_cache.before`, `after`, and `delta`; the two
prefix-cache probe cases fail if their delta does not show increased
`hit_tokens`. Lane comparison reports `artifact_identity_mismatch` unless
repo, commit, profile, and quantization are identical across lanes; that mismatch
fails the promotion gate and is emitted as
`failure_classification: "lane_artifact_identity_mismatch"`. JSON and tool-call
recall cases validate the full benchmark contract: `marker`, `profile`, `case`,
and `finish_reason: "tool_calls"` for tool responses.

## `bench qwen-mlx-tool-normalized`

Runs direct Qwen tool/JSON probes across comparable OpenAI-compatible lanes
without changing the `qwen-long-context` promotion gate. The command does not
start MLX sidecars; each lane records the endpoint, request model, optional
launch model identity, model addressing mode, template/thinking assumption,
optional snapshot identity, declared MLX-LM sweep knobs, repo revision metadata,
measured cache phase, aggregate summary rows, and the structured
`prefill_sweep` and `stable_prefix` ranking reports.

Start the sidecars in separate terminals. Direct MLX-LM lanes for Qwen must
disable thinking with `--chat-template-args '{"enable_thinking":false}'` or an
equivalent request-side `chat_template_kwargs` policy.

Default prompt-cache size with unbounded prompt-cache bytes:

```sh
SNAPSHOT=.llm-models/huggingface/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/<resolved-commit>
mlx_lm.server \
  --model "$SNAPSHOT" \
  --host 127.0.0.1 \
  --port 8080 \
  --chat-template-args '{"enable_thinking":false}'
```

Larger prompt-cache size:

```sh
mlx_lm.server \
  --model "$SNAPSHOT" \
  --host 127.0.0.1 \
  --port 8081 \
  --prompt-cache-size 4096 \
  --chat-template-args '{"enable_thinking":false}'
```

Bounded prompt-cache bytes:

```sh
mlx_lm.server \
  --model "$SNAPSHOT" \
  --host 127.0.0.1 \
  --port 8082 \
  --prompt-cache-bytes 1073741824 \
  --chat-template-args '{"enable_thinking":false}'
```

Prefill step-size sweep:

```sh
mlx_lm.server --model "$SNAPSHOT" --host 127.0.0.1 --port 8083 --prefill-step-size 2048 --chat-template-args '{"enable_thinking":false}'
mlx_lm.server --model "$SNAPSHOT" --host 127.0.0.1 --port 8084 --prefill-step-size 4096 --chat-template-args '{"enable_thinking":false}'
mlx_lm.server --model "$SNAPSHOT" --host 127.0.0.1 --port 8085 --prefill-step-size 8192 --chat-template-args '{"enable_thinking":false}'
```

Prompt/decode concurrency:

```sh
mlx_lm.server \
  --model "$SNAPSHOT" \
  --host 127.0.0.1 \
  --port 8086 \
  --prompt-concurrency 4 \
  --decode-concurrency 2 \
  --chat-template-args '{"enable_thinking":false}'
```

Kir proxy lanes are also externally started and should point at one of the MLX
sidecars:

```sh
cargo run -p llm-engine -- serve \
  --addr 127.0.0.1:3000 \
  --snapshot "$SNAPSHOT" \
  --loader mlx \
  --family qwen \
  --model-id local-qwen36-mlx \
  --mlx-endpoint http://127.0.0.1:8080/v1
```

```sh
llm-engine bench qwen-mlx-tool-normalized \
  --sweep-profile qwen-mlx-cache-prefill \
  --snapshot "$SNAPSHOT" \
  --warmups 1 \
  --samples 3 \
  --context-tokens 135000 \
  --concurrent-requests 4 \
  --concurrent-samples 1 \
  --output qwen-mlx-tool-sweep.json
```

Use `--dry-run` with the same cache-prefill profile to print the exact
lane/sample matrix without issuing HTTP requests. The profile expands the eight
fixed lanes `mlx-default`, `mlx-cache-size-4096`, `mlx-cache-bytes-1g`,
`mlx-prefill-2048`, `mlx-prefill-4096`, `mlx-prefill-8192`,
`mlx-concurrent-4x2`, and `kir-proxy` on ports `8080` through `8086` and
`3000`. Use explicit `--lane` specs instead of `--sweep-profile` when a sidecar
uses custom ports or experiment-specific knobs.

The focused Qwen3.6 35B A3B 135K prefill sweep uses paired direct MLX and Kir
proxy lanes. The benchmark still does not launch sidecars; start one direct MLX
server per prefill setting and one Kir proxy per matching upstream:

```sh
SNAPSHOT=.llm-models/huggingface/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/<resolved-commit>

mlx_lm.server --model "$SNAPSHOT" --host 127.0.0.1 --port 8080 --chat-template-args '{"enable_thinking":false}'
mlx_lm.server --model "$SNAPSHOT" --host 127.0.0.1 --port 8081 --prefill-step-size 512 --chat-template-args '{"enable_thinking":false}'
mlx_lm.server --model "$SNAPSHOT" --host 127.0.0.1 --port 8082 --prefill-step-size 1024 --chat-template-args '{"enable_thinking":false}'
mlx_lm.server --model "$SNAPSHOT" --host 127.0.0.1 --port 8083 --prefill-step-size 2048 --chat-template-args '{"enable_thinking":false}'
mlx_lm.server --model "$SNAPSHOT" --host 127.0.0.1 --port 8084 --prefill-step-size 4096 --chat-template-args '{"enable_thinking":false}'
mlx_lm.server --model "$SNAPSHOT" --host 127.0.0.1 --port 8085 --prefill-step-size 8192 --chat-template-args '{"enable_thinking":false}'
```

```sh
cargo run -p llm-engine -- serve --addr 127.0.0.1:3000 --snapshot "$SNAPSHOT" --loader mlx --family qwen --model-id local-qwen36-mlx --mlx-endpoint http://127.0.0.1:8080/v1
cargo run -p llm-engine -- serve --addr 127.0.0.1:3001 --snapshot "$SNAPSHOT" --loader mlx --family qwen --model-id local-qwen36-mlx --mlx-endpoint http://127.0.0.1:8081/v1
cargo run -p llm-engine -- serve --addr 127.0.0.1:3002 --snapshot "$SNAPSHOT" --loader mlx --family qwen --model-id local-qwen36-mlx --mlx-endpoint http://127.0.0.1:8082/v1
cargo run -p llm-engine -- serve --addr 127.0.0.1:3003 --snapshot "$SNAPSHOT" --loader mlx --family qwen --model-id local-qwen36-mlx --mlx-endpoint http://127.0.0.1:8083/v1
cargo run -p llm-engine -- serve --addr 127.0.0.1:3004 --snapshot "$SNAPSHOT" --loader mlx --family qwen --model-id local-qwen36-mlx --mlx-endpoint http://127.0.0.1:8084/v1
cargo run -p llm-engine -- serve --addr 127.0.0.1:3005 --snapshot "$SNAPSHOT" --loader mlx --family qwen --model-id local-qwen36-mlx --mlx-endpoint http://127.0.0.1:8085/v1
```

Run the repeatable prefill sweep with:

```sh
llm-engine bench qwen-mlx-tool-normalized \
  --sweep-profile qwen-mlx-prefill-135k \
  --snapshot "$SNAPSHOT" \
  --samples 3 \
  --output qwen-mlx-prefill-135k.json
```

The stable agent-prefix profile compares one direct MLX sidecar on `8080/v1`
with one Kir proxy on `3000`. The benchmark does not launch either process.
Start the direct sidecar with Qwen no-thinking chat-template args:

```sh
mlx_lm.server \
  --model "$SNAPSHOT" \
  --host 127.0.0.1 \
  --port 8080 \
  --chat-template-args '{"enable_thinking":false}'
```

Start the Kir proxy with canonical tool schemas and the same Qwen/MLX upstream:

```sh
cargo run -p llm-engine -- serve \
  --addr 127.0.0.1:3000 \
  --snapshot "$SNAPSHOT" \
  --loader mlx \
  --family qwen \
  --model-id local-qwen36-mlx \
  --mlx-endpoint http://127.0.0.1:8080/v1 \
  --canonical-tool-schemas
```

Run the stable prefix probes with:

```sh
llm-engine bench qwen-mlx-tool-normalized \
  --sweep-profile qwen-mlx-stable-prefix \
  --snapshot "$SNAPSHOT" \
  --samples 3 \
  --engine-db-baselines reports/benchmarks/engine-db-baselines.json \
  --output qwen-mlx-stable-prefix.json
```

The `qwen-mlx-prefill-135k` profile defaults to `--probe-suite
prefill-sweep-135k` and expands `mlx-prefill-default`, `kir-prefill-default`,
`mlx-prefill-512`, `kir-prefill-512`, `mlx-prefill-1024`,
`kir-prefill-1024`, `mlx-prefill-2048`, `kir-prefill-2048`,
`mlx-prefill-4096`, `kir-prefill-4096`, `mlx-prefill-8192`, and
`kir-prefill-8192` on direct ports `8080` through `8085` and proxy ports
`3000` through `3005`.

The `qwen-mlx-stable-prefix` profile defaults to `--probe-suite
stable-agent-prefix` and expands `mlx-stable-prefix` on
`http://127.0.0.1:8080/v1` plus `kir-stable-prefix` on
`http://127.0.0.1:3000`.

| Flag | Default | Description |
| --- | --- | --- |
| `--sweep-profile <name>` | none | Expands a built-in lane matrix. `qwen-mlx-cache-prefill`, `qwen-mlx-prefill-135k`, and `qwen-mlx-stable-prefix` require `--snapshot` and use the default MLX/Kir proxy ports above. |
| `--probe-suite <name>` | profile default | Selects `full-matrix`, `focused-agentic-gate`, `prefill-sweep-135k`, or `stable-agent-prefix`. `qwen-mlx-prefill-135k` defaults to `prefill-sweep-135k`; `qwen-mlx-stable-prefix` defaults to `stable-agent-prefix`; other modes default to `full-matrix`. |
| `--snapshot <path>` | none | Raw Hugging Face snapshot path used by built-in sweep profiles. The profile records it as `snapshot`, `launched_model_id`, and raw snapshot identity. |
| `--lane <spec>` | none | Adds an explicit lane. Specs are comma-separated `key=value` pairs: required `name`, `endpoint`, `model`; optional `launched_model_id`, `snapshot`, `kind=direct_mlx\|kir_ai_proxy\|other`, `model_addressing=loaded_model_id\|default_model\|server_default\|custom`, `template=qwen-no-thinking\|sidecar-chat-template-args\|none`, `tool_parser=auto\|json\|qwen-xml`, `mlx_prompt_cache_size=default\|<u64>`, `mlx_prompt_cache_bytes=unset\|<u64>`, `mlx_prefill_step_size=default\|<u64>`, `mlx_prompt_concurrency=default\|<u32>`, and `mlx_decode_concurrency=default\|<u32>`. Do not combine explicit lanes with `--sweep-profile`. |
| `--warmups <n>` | `1` | Warmup requests issued before measured samples for `warm_same_prompt` and `warm_same_tool_schema`. `cold` never performs command-issued warmups. |
| `--samples <n>` | `1` | Sequential measured samples per lane, case, schema variant, tool-choice variant, and cache phase. |
| `--context-tokens <n>` | `135000` | Stable long-context prompt target for all probes. |
| `--concurrent-requests <n>` | `1` | Requests issued together for the separate concurrent pass. If this is greater than `1` and `--concurrent-samples` is `0`, the concurrent pass uses `--samples` batches. |
| `--concurrent-samples <n>` | `0` | Concurrent sample batches per lane, case, schema variant, tool-choice variant, and cache phase. Values greater than `0` enable the concurrent pass even when `--concurrent-requests` is `1`. |
| `--ab-baseline <path>` | none | Loads a previous `qwen-mlx-tool-normalized` JSON trace and emits `agentic_streaming_fast_path_ab`. The command fails when a `kir_ai_proxy` lane does not advance p50 `tool_required_stream` first tool delta versus the baseline, or when final validation signatures change. |
| `--output <path>` | none | Writes the full JSON trace to disk as well as stdout. |
| `--engine-db-baselines <path>` | none | Reads a JSON export of benchmark DB baseline rows and includes them in `latest_performance_comparison` beside latest direct MLX and Kir proxy lane metrics. |
| `--timeout-ms <n>` | `1800000` | Whole request timeout. |
| `--connect-timeout-ms <n>` | `10000` | HTTP connect timeout. |
| `--dry-run` | absent | Emits the planned cases, phases, lanes, model/template assumptions, and sample grid without HTTP requests. |

Cases are `tool_required`, `tool_required_stream`, `json_object`, and
`omp_repeated_prefix`. Tool cases run the schema variants `baseline_current`,
`canonical_current`, `baseline_permuted_equivalent`, and
`canonical_permuted_equivalent` across `required` and function-specific
`tool_choice` variants. `json_object` remains a control with `schema_variant:
"none"` and `tool_choice_variant: "none"`. The OMP case uses a multi-turn
history with stable system/user context, an assistant tool call, a tool result,
and a small final user delta requiring `record_qwen_tool_probe`. Cache phases are
`cold`, `warm_same_prompt`, and `warm_same_tool_schema`; warmups are excluded
from measured `samples`, and concurrent measurements are reported in a separate
`concurrent_samples` array. The default `template=qwen-no-thinking` injects
`chat_template_kwargs: {"enable_thinking": false}` into requests.
`template=sidecar-chat-template-args` records that the sidecar is expected to
have been launched with equivalent chat-template args and does not inject
request kwargs.

Each measured sample reports `schema_variant`, `tool_choice_variant`,
`schema_canonicalized`, `schema_permuted`, `tool_schema_sha256`,
`tool_schema_bytes`, `cache_phase`, `prewarmed`, latency, HTTP status, finish
reason, prompt/completion/total tokens, cached-token status/count when provided
by upstream `usage.prompt_tokens_details.cached_tokens`, validation
classification, and the stream timing fields when observed. The top-level
`repo_revision` records the kir-ai source checkout branch, commit SHA, and dirty
status. Exported benchmark harnesses without a `.git` directory should set
`LLM_ENGINE_BENCH_REPO_COMMIT`, `LLM_ENGINE_BENCH_REPO_BRANCH`, and
`LLM_ENGINE_BENCH_REPO_DIRTY`, or include a `.kir-ai-origin.json` file with
`repo_revision.commit_sha`, `repo_revision.branch`, and `repo_revision.dirty`;
`LLM_ENGINE_BENCH_REPO_DIR` is used only when it points at the actual kir-ai Git
root or exported workspace so parent harness repositories are not reported as
the benchmark source.
Manifestless Hugging Face cache snapshots are recorded as raw snapshot identity
with inferred `repo_id` and resolved commit when the path follows
`models--<owner>--<repo>/snapshots/<commit>`. Use `launched_model_id` to pin the
model identity from the sidecar launch command when `/v1/models` reports a
generic or unrelated ID. `model_addressing=server_default` omits the request
payload `model` field and lets an externally launched MLX-LM sidecar use its
loaded model; the built-in `qwen-mlx-cache-prefill` direct lanes use this mode.
Use `--focused-agentic-gate` to run the smaller Qwen MLX agentic subset; the
top-level `agentic_gate` report summarizes warm-prefix latency, first-byte and
first-semantic/tool-delta timing, token throughput, cached-token counts, and
lane deltas without requiring the full schema/tool-choice matrix.
Use `--ab-baseline <trace.json>` with the focused agentic gate to produce the
top-level `agentic_streaming_fast_path_ab` report. It compares matching lanes for
the canonical required `tool_required_stream` probe, records baseline and
candidate p50 first tool delta and tool-finish timings, requires `kir_ai_proxy`
lanes to move first tool delta earlier, and requires candidate pass/fail,
classification, and `tool_calls` finish signatures to remain unchanged.
The `prefill_sweep` report ranks lanes by p50 first semantic delta for each
probe, cache phase, and run mode, while preserving lane kind, prefill step size,
response headers, first response byte, first parsed SSE chunk, first tool delta,
elapsed latency, token and cached-token stats, optional Kir MLX upstream admin
timing, process RSS, stalled-request deltas, and no-progress deltas. Runs are
flagged invalid when samples fail, TTFT or required stream/tool deltas are
missing, or Kir admin metrics report stalled/no-progress deltas.
The `stable_prefix` report groups by probe, cache phase, run mode, and lane. It
reports p50 first semantic/tool delta, p50 elapsed latency, average
prompt/cached/uncached tokens, cache status counts (`unknown`, `miss`,
`partial`, `hit`), lane latency deltas, and matching
`/admin/metrics.request_cache` observations for Kir proxy samples when
`x-request-id` and admin access are available.
The `latest_performance_comparison` report condenses the latest live lane
samples into plain stream, required-tool stream, and prefix-cache rows for
`kind=direct_mlx` and `kind=kir_ai_proxy`, then appends rows from
`--engine-db-baselines`. Each row carries stable `ttfi_ms`,
`tokens_per_second`, `cache_cold_latency_ms`, `cache_warm_latency_ms`,
`cache_speedup`, and tool-stream timing fields, using `null` when a metric does
not apply. The top-level `evidence` object records whether Kir, direct MLX,
engine DB baselines, TTFI, cache, and tok/s metrics are all present.

The engine baseline file is a JSON export from
`reports/benchmarks/benchmarks.sqlite` or another benchmark DB source. The CLI
consumes JSON so dry-run and CI report-shaping tests do not require SQLite or
live engines. Common DB export names such as `ttft_ms`, `latency_ms`, and
`tok_s` are accepted as aliases for the stable output fields:

```json
{
  "source": "reports/benchmarks/benchmarks.sqlite",
  "rows": [
    {
      "engine": "Rapid-MLX",
      "profile": "rapid-0615-qwen35-kv4-135k",
      "model": "Qwen3.6 35B A3B 4bit",
      "probe": "chat_stream",
      "ttfi_ms": 80.6,
      "tokens_per_second": 26.3,
      "notes": "DB row 2026-05-07"
    }
  ]
}
```
The top-level `summary` groups rows by lane, case, schema variant, tool-choice
variant, cache phase, and run mode with pass/fail counts, p50/p95 latency,
average cached/token usage, and the fastest lane for that group.

For long unattended agentic workflow runs across Qwen 27B, Qwen 35B, Gemma 4,
direct stable-prefix probes, and opencode coding tasks, use
[`scripts/agentic_overnight_benchmark.py`](../scripts/agentic_overnight_benchmark.py)
as documented in [Agentic Overnight Benchmark](agentic-overnight-benchmark.md).
That script intentionally launches sidecars and Kir proxies itself; the Rust
benchmark command above remains a focused externally managed harness.

## `model list`

Lists snapshots from a model home. The command reconciles promoted snapshots
before reporting them with fast readiness by default: it parses manifests,
checks required config/tokenizer/weight classes, verifies manifest file presence
and sizes, and validates safetensors index coverage without hashing weights.
Promoted snapshots that fail the selected readiness check are moved to
quarantine, while intentional metadata-only snapshots are reported separately
and are not advertised as ready for serving. Default fast readiness does not
detect same-size SHA corruption; use `--snapshot-readiness deep` or
`model verify` when checksum verification is required.

```sh
llm-engine model list [--model-home <path>] [--snapshot-readiness <fast|deep>]
```

Readiness modes:

- `fast` (default): parses the manifest and checks required file classes, file
  presence/sizes, and safetensors index coverage without hashing model weights.
- `deep`: hashes every manifest file and performs the same runnable readiness
  checks before reporting or quarantining snapshots.

Model home resolution:

1. `--model-home <path>`
2. `LLM_MODEL_HOME`
3. `.llm-models`

Output shape:

```json
{
  "snapshots": [
    {
      "status": "ready",
      "path": "...",
      "repo_id": "Qwen/Qwen3.6-35B-A3B",
      "requested_revision": "main",
      "resolved_commit": "...",
      "profile": "qwen36-safetensors-bf16",
      "family": "qwen",
      "loader": "native-metal",
      "quantization": "bf16",
      "manifest_digest": "...",
      "files": 39,
      "readiness_reason": null,
      "aliases": ["local-qwen36"]
    }
  ],
  "metadata_only_snapshots": [],
  "quarantined_snapshots": []
}
```

## `model inspect`

Reads a promoted or quarantined snapshot manifest and prints a summary.

```sh
llm-engine model inspect <snapshot-path>
```

Promoted snapshots contain `llm-engine-manifest.json`. Quarantined snapshots
contain `llm-engine-quarantine.json` and report `status: "quarantined"`.

Output fields:

- `status`
- `readiness_reason`
- `snapshot_path`
- `repo_id`
- `requested_revision`
- `resolved_commit`
- `profile`
- `family`
- `loader`
- `quantization`
- `manifest_digest`
- `files`
- `total_bytes`

## `model verify`

Verifies files recorded in a promoted snapshot manifest and checks that the
snapshot is runnable. Runnable snapshots must include config, tokenizer, and
weight artifacts. Safetensors indexes must reference weight shards recorded in
the manifest.

```sh
llm-engine model verify <snapshot-path>
```

Output shape:

```json
{
  "status": "ok",
  "snapshot_path": "...",
  "repo_id": "Qwen/Qwen3.6-35B-A3B",
  "resolved_commit": "...",
  "manifest_digest": "...",
  "verified_files": 39,
  "verified_bytes": 71926864255
}
```

Verification checks file presence, file type, size, SHA-256 when available,
readiness metadata for built-in profiles, and safetensors index shard coverage.
Successful verification records the snapshot as recently used for prune
retention.

## `model plan`

Plans a Hugging Face model download without writing a snapshot.

```sh
llm-engine model plan <repo> \
  [--revision <rev>] \
  [--profile <profile>] \
  [--metadata-only]
```

| Option | Default | Description |
| --- | --- | --- |
| `<repo>` | required | Hugging Face model repo in `org/name` form. |
| `--revision <rev>` | `main` | Branch, tag, or commit to resolve. The resolved commit must be immutable. |
| `--profile <profile>` | `qwen36-safetensors-bf16` | Built-in model acquisition profile. |
| `--metadata-only` | `false` | Excludes files classified as weights from the plan. |

`HF_TOKEN` is used when present.

Supported profiles:

- `gemma4-e2b-it-mlx-4bit`
- `gemma4-text-safetensors-bf16`
- `llama32-3b-instruct-mlx-4bit`
- `qwen35-4b-mlx-4bit`
- `qwen35-4b-mlx-8bit`
- `qwen35-4b-mlx-optiq-4bit`
- `qwen3-dense-safetensors-bf16`
- `qwen36-safetensors-bf16`
- `qwen36-mlx-4bit`

`gemma4-e2b-it-mlx-4bit` targets a practical Gemma 4 MLX text-chat snapshot.
`gemma4-text-safetensors-bf16` targets BF16 Gemma 4 text artifacts for native
text serving and excludes vision and projector artifacts.
`llama32-3b-instruct-mlx-4bit` targets practical Llama 3.2 Instruct MLX
chat snapshots.
`qwen35-4b-mlx-optiq-4bit` targets the Apple-silicon OptiQ mixed 4/8-bit MLX
snapshot family.
`qwen3-dense-safetensors-bf16` targets standard dense Qwen3 text checkpoints
such as `Qwen/Qwen3-0.6B` and `Qwen/Qwen3-4B`.

## `model pull`

Plans and downloads selected Hugging Face artefacts into the model store.

```sh
llm-engine model pull <repo> \
  [--revision <rev>] \
  [--profile <profile>] \
  [--metadata-only] \
  [--alias <model-id>] \
  [--model-home <path>]
```

`model pull` uses the same planning flags as `model plan`, plus model home
resolution. `--alias` records an active model alias that protects the pulled
snapshot from prune.

Output shape:

```json
{
  "snapshot_path": "...",
  "manifest_digest": "...",
  "resolved_commit": "...",
  "files": 39
}
```

Pulls write to:

```text
<model-home>/huggingface/models--<org>--<name>/staging/<commit>.partial
```

and promote to:

```text
<model-home>/huggingface/models--<org>--<name>/snapshots/<commit>
```

Metadata-only snapshots promote to:

```text
snapshots/<commit>.metadata-only
```

## `model prune`

Plans or applies snapshot deletion using the model-store retention policy.

```sh
llm-engine model prune --dry-run \
  [--older-than-days <days>] \
  [--keep-min-per-profile <n>] \
  [--profile <profile>] \
  [--model-home <path>]

llm-engine model prune --confirm-delete \
  [--older-than-days <days>] \
  [--keep-min-per-profile <n>] \
  [--profile <profile>] \
  [--model-home <path>]
```

`--dry-run` and `--confirm-delete` use the same candidate planner. Destructive
mode requires the explicit `--confirm-delete` flag and reports the candidate set
that was computed before deletion.

Retention protects:

- snapshots referenced by recorded aliases
- snapshots used within `--older-than-days`
- the newest `--keep-min-per-profile` snapshots per profile

Before deleting a candidate, destructive mode verifies the manifest files. A
candidate that fails verification is moved to quarantine instead of deleted.

Output includes `candidates`, `protected`, `deleted`, `quarantined`,
`reclaimable_bytes`, and `deleted_bytes`.

## `model inspect-safetensors`

Reads safetensors header metadata from one shard file.

```sh
llm-engine model inspect-safetensors <path> \
  [--tensor <name>] \
  [--bf16-row <row>] \
  [--limit <n>]
```

| Flag | Default | Description |
| --- | --- | --- |
| `--tensor <name>` | none | Adds metadata for one tensor. |
| `--bf16-row <row>` | none | Reads one BF16 row as f32 values. Requires `--tensor`. |
| `--limit <n>` | `8` | Limits displayed row values. |

Output includes file length, header length, data start offset, tensor count, and
sample tensor names.

## `model inspect-tensor`

Resolves a tensor through a snapshot safetensors index.

```sh
llm-engine model inspect-tensor <snapshot-path> \
  --tensor <name> \
  [--bf16-row <row>] \
  [--limit <n>]
```

The command prints the tensor name, dtype, shape, byte length, shard path, and
optional row prefix.

## `model inspect-qwen-input`

Runs native Qwen probing for one token id.

```sh
llm-engine model inspect-qwen-input <snapshot-path> \
  --token-id <id> \
  [--limit <n>] \
  [--linear-layers <n>] \
  [--layers <n>] \
  [--lm-head-top-k <k>] \
  [--chunk-rows <n>] \
  [--layer0-projections] \
  [--layer0-attention] \
  [--layer0-router] \
  [--layer0-moe] \
  [--top-k <k>]
```

| Flag | Default | Description |
| --- | --- | --- |
| `--token-id <id>` | required | Token id to read from embedding rows. |
| `--limit <n>` | `8` | Prefix length for displayed vectors. |
| `--linear-layers <n>` | none | Runs the linear decoder layer helper repeatedly. |
| `--layers <n>` | none | Runs full Qwen decoder layer selection for the first `n` layers. |
| `--lm-head-top-k <k>` | none | Adds final norm and LM-head top-k output. |
| `--chunk-rows <n>` | `512` | Rows per chunk for this command's LM-head probe. |
| `--layer0-projections` | `false` | Reads layer-0 linear-attention projections. |
| `--layer0-attention` | `false` | Runs layer-0 attention and implies projections. |
| `--layer0-router` | `false` | Runs layer-0 router and implies attention. |
| `--layer0-moe` | `false` | Runs selected layer-0 MoE execution and implies router. |
| `--top-k <k>` | `num_experts_per_tok` | Router top-k when layer-0 routing is enabled. |

This command is for model-loader and math inspection, not normal serving.
