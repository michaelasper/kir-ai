# How To Manage Model Snapshots

This guide shows how to plan, pull, list, inspect, verify, and use model
snapshots with the native Rust model store.

## Pick A Model Home

The default model home is `.llm-models` under the repository root. For large
snapshots, use a dedicated volume:

```sh
export LLM_MODEL_HOME=/Volumes/models/kir-ai
```

You can also pass `--model-home` to `model pull` and `model list`.

## Authenticate For Gated Models

If the repository is gated or private:

```sh
export HF_TOKEN=hf_...
```

`model plan` and `model pull` use this token for Hugging Face API and download
requests.

## Plan A Download

Plan the default native BF16 profile:

```sh
cargo run -p llm-engine -- model plan Qwen/Qwen3.6-35B-A3B \
  --revision main \
  --profile qwen36-safetensors-bf16
```

The plan resolves `main` to an immutable 40-character commit SHA, selects files
from the profile allow patterns, reports skipped files, and prints selected byte
counts.

Use `--metadata-only` to see the static artefact subset:

```sh
cargo run -p llm-engine -- model plan Qwen/Qwen3.6-35B-A3B \
  --metadata-only
```

Metadata-only plans exclude files classified as `Weights`. The safetensors index
is not classified as a weight and remains in the metadata snapshot.

## Pull Metadata Only

Use this when you want manifests, config, tokenizer, template, and index files
without full weight shards:

```sh
cargo run -p llm-engine -- model pull Qwen/Qwen3.6-35B-A3B \
  --metadata-only \
  --model-home "$LLM_MODEL_HOME"
```

The promoted snapshot path ends with:

```text
snapshots/<resolved-commit>.metadata-only
```

Metadata-only snapshots are useful for inspecting configuration, but they cannot
serve native Qwen requests because shard files are missing.

## Pull A Full Native Snapshot

Pull the full BF16 profile only when you have enough disk space:

```sh
cargo run -p llm-engine -- model pull Qwen/Qwen3.6-35B-A3B \
  --revision main \
  --profile qwen36-safetensors-bf16 \
  --model-home "$LLM_MODEL_HOME"
```

The pull writes files into a staging directory and promotes the snapshot only
after selected artefacts are present and verified. Interrupted downloads can
resume from partially downloaded files.

## List Local Snapshots

```sh
cargo run -p llm-engine -- model list --model-home "$LLM_MODEL_HOME"
```

The command prints promoted snapshots with repo identity, requested revision,
resolved commit, profile, family, loader, quantisation, manifest digest, and file
count.

## Inspect A Snapshot

```sh
SNAPSHOT="$LLM_MODEL_HOME/huggingface/models--Qwen--Qwen3.6-35B-A3B/snapshots/<resolved-commit>"

cargo run -p llm-engine -- model inspect "$SNAPSHOT"
```

This reads `llm-engine-manifest.json` without network access and reports the
snapshot identity, file count, total bytes, and manifest digest.

## Verify A Snapshot

```sh
cargo run -p llm-engine -- model verify "$SNAPSHOT"
```

Verification checks every manifest file for:

- presence
- file type
- expected size
- SHA-256 digest when the manifest contains a normalised 64-character SHA

Treat `model_integrity_failed` as a signal to pull or restore the snapshot
again.

## Inspect Safetensors Metadata

Read a safetensors header without loading the full payload:

```sh
cargo run -p llm-engine -- model inspect-safetensors \
  "$SNAPSHOT/model-00001-of-00026.safetensors"
```

Inspect a named tensor:

```sh
cargo run -p llm-engine -- model inspect-safetensors \
  "$SNAPSHOT/model-00001-of-00026.safetensors" \
  --tensor model.language_model.embed_tokens.weight
```

Read a BF16 row prefix:

```sh
cargo run -p llm-engine -- model inspect-safetensors \
  "$SNAPSHOT/model-00001-of-00026.safetensors" \
  --tensor model.language_model.embed_tokens.weight \
  --bf16-row 0 \
  --limit 8
```

`--bf16-row` requires `--tensor`.

## Inspect A Tensor Through The Snapshot Index

Use `inspect-tensor` when you know the tensor name but not its shard:

```sh
cargo run -p llm-engine -- model inspect-tensor "$SNAPSHOT" \
  --tensor model.language_model.embed_tokens.weight \
  --bf16-row 0 \
  --limit 8
```

The command resolves the tensor through `model.safetensors.index.json`.

## Probe Native Qwen Inputs

Probe embedding and layer-0 normalisation for one token:

```sh
cargo run -p llm-engine -- model inspect-qwen-input "$SNAPSHOT" \
  --token-id 0 \
  --limit 8
```

Run all decoder layers and inspect top LM-head candidates:

```sh
cargo run -p llm-engine -- model inspect-qwen-input "$SNAPSHOT" \
  --token-id 0 \
  --layers 40 \
  --lm-head-top-k 5 \
  --chunk-rows 2048 \
  --limit 2
```

Layer-0 flags build on one another:

- `--layer0-projections` reads linear-attention projections.
- `--layer0-attention` also runs layer-0 linear attention.
- `--layer0-router` also runs post-attention norm and MoE routing.
- `--layer0-moe` also runs selected expert execution and residual merge.

## Serve A Verified Snapshot

After verification, pass the snapshot to the server:

```sh
cargo run -p llm-engine -- serve \
  --snapshot "$SNAPSHOT" \
  --model-id local-qwen36 \
  --max-new-tokens 1 \
  --max-prefill-tokens 32
```
