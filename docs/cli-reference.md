# CLI Reference

The `llm-engine` binary is the HTTP server and model tooling CLI. Argument
parsing is manual. Flags use `--flag value`; boolean flags are present or absent.

## Synopsis

```sh
llm-engine [serve]
llm-engine serve [--addr <host:port>] [--protocol-test-backend | --snapshot <path> | --snapshot-alias <alias>] [--loader <native-metal|mlx>] [--family <qwen|deep_seek|gemma|llama>] [--model-id <id>] [--max-new-tokens <n>] [--max-prefill-tokens <n>] [--mlx-endpoint <url>] [--native-metal-weight-cache-bytes <bytes>] [--warm-native-metal-weight-cache]
llm-engine bench qwen-long-context [--endpoint <url> --snapshot <path> | --lane <spec> ...]
llm-engine model <subcommand> ...
```

If no command is provided, `llm-engine` defaults to `serve`, which still
requires either `--snapshot <path>` or `--protocol-test-backend`.

When running through Cargo:

```sh
cargo run -p llm-engine -- serve --protocol-test-backend
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
  --max-prefill-tokens 32 \
  --native-metal-weight-cache-bytes 8589934592
```

| Flag | Default | Description |
| --- | --- | --- |
| `--addr <host:port>` | `127.0.0.1:3000` | Socket address to bind. |
| `--protocol-test-backend` | absent | Enables protocol test mode without model artifacts. Intended for tests and client integration. |
| `--snapshot <path>` | none | Enables manifest-selected serving from a local snapshot directory. `loader: native-metal` opens native Qwen; `loader: mlx` opens the loopback MLX sidecar backend. |
| `--snapshot-alias <alias>` / `--model-alias <alias>` | none | Resolves a snapshot path from the model store alias records and verifies the recorded manifest digest before serving. |
| `--loader <native-metal\|mlx>` / `--backend <native-metal\|mlx>` | manifest or `native-metal` | Overrides the snapshot loader when no Kir manifest is present. Fails if it conflicts with an existing manifest. |
| `--family <qwen\|deep_seek\|gemma\|llama>` | manifest metadata | Supplies model-family metadata for raw snapshots without a Kir manifest. Raw MLX snapshots must set this explicitly. Qwen, DeepSeek, Gemma, and Llama are serveable through the MLX sidecar. |
| `--model-id <id>` | `local-qwen36` or snapshot alias | Served model alias. Used with `--snapshot`; protocol test mode also serves `local-qwen36`. |
| `--max-new-tokens <u32>` | `256` | Native Qwen generation cap per request. Values below `1` are clamped to `1`. |
| `--max-prefill-tokens <usize>` | `32` | Native Qwen prefill chunk size. Values below `1` are clamped to `1`; prompt retention is sized from the accepted prompt plus generation budget and fails closed at the model context limit. |
| `--mlx-endpoint <url>` | `http://127.0.0.1:8080/v1` | Loopback `mlx_lm.server` or `mlx_vlm.server` `/v1` endpoint for MLX manifests. Remote endpoints are rejected. `MLX_LM_ENDPOINT` is used when this flag is omitted. |
| `--native-metal-weight-cache-bytes <u64>` | `8589934592` | Per-backend Metal BF16 weight-buffer LRU budget. Set `0` to disable weight-buffer caching. |
| `--warm-native-metal-weight-cache` | absent | Preloads rank-2 BF16 tensors into the Metal weight-buffer cache at startup until the configured budget is full. |

Without `--snapshot`, `serve` exits unless `--protocol-test-backend` is
present. Implicit no-snapshot stub serving was removed.

With a native-metal snapshot, the directory must contain `config.json`,
`tokenizer.json`, `model.safetensors.index.json`, and all referenced shard
files. With an MLX snapshot promoted by `llm-engine model pull`, the directory
must include an `llm-engine-manifest.json` whose loader is `mlx`, and a
compatible MLX sidecar must already be listening on the configured loopback
endpoint. Chat requests for Qwen, DeepSeek, Gemma, and Llama use OpenAI-compatible
`/v1/chat/completions` so the MLX sidecar owns model-specific chat templating
and structured tool metadata; legacy text completion requests use a
completions-capable sidecar endpoint when the selected family exposes one. Raw
Hugging Face cache snapshots need both `--loader mlx` and a serveable `--family`
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
`comparison`, which includes per-case latency, TTFT, token throughput, pass/fail
classification, and fastest-lane summaries. When `/admin/metrics` is available,
each lane also includes `cache_metrics` with prefix-cache hit rate/residency,
Metal BF16 weight-cache hit rate/residency, KV-cache residency, recurrent
linear-attention-cache residency, and eviction churn signals. Lane comparison reports
`artifact_identity_mismatch` unless repo, commit, profile, and quantization are
identical across lanes; that mismatch fails the promotion gate and is emitted as
`failure_classification: "lane_artifact_identity_mismatch"`. JSON and tool-call
recall cases validate the full benchmark contract: `marker`, `profile`, `case`,
and `finish_reason: "tool_calls"` for tool responses.

## `model list`

Lists runnable snapshots from a model home. The command reconciles promoted
snapshots before reporting them: stale or corrupt promoted snapshots are moved
to quarantine, while intentional metadata-only snapshots are reported
separately and are not advertised as ready for serving.

```sh
llm-engine model list [--model-home <path>]
```

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
`gemma4-text-safetensors-bf16` targets BF16 Gemma 4 text artifacts for MLX
sidecar serving and excludes vision and projector artifacts.
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
