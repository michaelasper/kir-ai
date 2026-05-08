# CLI Reference

The `llm-engine` binary is the HTTP server and model tooling CLI. Argument
parsing is manual. Flags use `--flag value`; boolean flags are present or absent.

## Synopsis

```sh
llm-engine [serve]
llm-engine serve [--addr <host:port>] [--snapshot <path>] [--model-id <id>] [--max-new-tokens <n>] [--max-prefill-tokens <n>]
llm-engine model <subcommand> ...
```

If no command is provided, `llm-engine` defaults to `serve`.

When running through Cargo:

```sh
cargo run -p llm-engine -- serve
cargo run -p llm-engine -- model list
```

## `serve`

Starts the Axum HTTP server.

```sh
llm-engine serve \
  --addr 127.0.0.1:3000 \
  --snapshot <snapshot-path> \
  --model-id local-qwen36 \
  --max-new-tokens 1 \
  --max-prefill-tokens 32
```

| Flag | Default | Description |
| --- | --- | --- |
| `--addr <host:port>` | `127.0.0.1:3000` | Socket address to bind. |
| `--snapshot <path>` | none | Enables native Qwen backend from a local snapshot directory. |
| `--model-id <id>` | `local-qwen36` | Served model alias. Only used with `--snapshot`; deterministic mode also uses `local-qwen36`. |
| `--max-new-tokens <u32>` | `1` | Native Qwen generation cap per request. Values below `1` are clamped to `1`. |
| `--max-prefill-tokens <usize>` | `32` | Number of recent prompt tokens retained for native Qwen prefill. Values below `1` are clamped to `1`. |

Without `--snapshot`, the deterministic backend is used.

With `--snapshot`, the directory must contain `config.json`, `tokenizer.json`,
`model.safetensors.index.json`, and all referenced shard files.

## `model list`

Lists promoted snapshots from a model home.

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
      "path": "...",
      "repo_id": "Qwen/Qwen3.6-35B-A3B",
      "requested_revision": "main",
      "resolved_commit": "...",
      "profile": "qwen36-safetensors-bf16",
      "family": "qwen",
      "loader": "native-metal",
      "quantization": "bf16",
      "manifest_digest": "...",
      "files": 39
    }
  ]
}
```

## `model inspect`

Reads a promoted snapshot manifest and prints a summary.

```sh
llm-engine model inspect <snapshot-path>
```

The snapshot path must contain `llm-engine-manifest.json`.

Output fields:

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

Verifies files recorded in a promoted snapshot manifest.

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

Verification checks file presence, file type, size, and SHA-256 when available.

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

- `qwen36-safetensors-bf16`
- `qwen36-mlx-4bit`

## `model pull`

Plans and downloads selected Hugging Face artefacts into the model store.

```sh
llm-engine model pull <repo> \
  [--revision <rev>] \
  [--profile <profile>] \
  [--metadata-only] \
  [--model-home <path>]
```

`model pull` uses the same planning flags as `model plan`, plus model home
resolution.

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
