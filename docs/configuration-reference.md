# Configuration Reference

This reference describes the configuration surfaces that exist in the current
workspace: environment variables, server flags, model acquisition profiles,
snapshot layout, Qwen config requirements, safetensors requirements, and request
generation options.

## Environment Variables

| Variable | Used By | Description |
| --- | --- | --- |
| `HF_TOKEN` | `model plan`, `model pull` | Hugging Face bearer token for gated or private repositories. Anonymous access is used when unset. |
| `LLM_MODEL_HOME` | `model list`, `model pull` | Model store root when `--model-home` is not passed. Defaults to `.llm-models`. |

## Workspace Tooling

| File | Setting |
| --- | --- |
| [../mise.toml](../mise.toml) | Pins Rust `1.95.0` and defines workspace tasks. |
| [../Cargo.toml](../Cargo.toml) | Workspace resolver `3`, edition `2024`, Rust version `1.95`, MIT licence. |

Mise tasks:

| Task | Command |
| --- | --- |
| `mise run fmt` | `cargo fmt --all` |
| `mise run fmt-check` | `cargo fmt --all --check` |
| `mise run test` | `cargo test --workspace` |
| `mise run clippy` | `cargo clippy --workspace --all-targets --all-features -- -D warnings` |
| `mise run check` | `fmt-check`, `test`, and `clippy` |
| `mise run run` | `cargo run -p llm-engine -- serve --deterministic-test-backend` |

## Server Configuration

`llm-engine serve` flags:

| Flag | Type | Default | Behaviour |
| --- | --- | --- | --- |
| `--addr` | socket address | `127.0.0.1:3000` | Address bound by Axum. |
| `--deterministic-test-backend` | boolean | unset | Enables deterministic protocol mode without model artifacts. |
| `--snapshot` | path | unset | Enables native Qwen backend. Without this flag, `serve` requires `--deterministic-test-backend`. |
| `--model-id` | string | `local-qwen36` | Served model id for native Qwen mode. |
| `--max-new-tokens` | `u32` | `256` | Native backend generation cap. Clamped to at least `1`. |
| `--max-prefill-tokens` | `usize` | `32` | Number of recent prompt tokens retained for native prefill. Clamped to at least `1`. |
| `--native-metal-weight-cache-bytes` | `u64` | `8589934592` | Per-backend Metal BF16 weight-buffer LRU budget. Set `0` to disable weight-buffer caching. |
| `--warm-native-metal-weight-cache` | boolean | unset | Preloads rank-2 BF16 tensors into the Metal weight-buffer cache at startup until the configured budget is full. |

Native Qwen backend internal defaults:

| Field | Default | Notes |
| --- | --- | --- |
| `top_k` | `16` | Number of LM-head candidates inspected internally. |
| `chunk_rows` | `2048` | Rows per chunk for native server LM-head matvecs. |

These internals are not exposed as `serve` flags.

## Model Acquisition Profiles

`ModelProfile` fields:

| Field | Meaning |
| --- | --- |
| `name` | Profile id passed to `--profile`. |
| `family` | Model family label recorded in manifests. |
| `loader` | Intended loader label recorded in manifests. |
| `quantization` | Quantisation label recorded in manifests. |
| `allow_patterns` | File path patterns selected for planning. |
| `ignore_patterns` | File path patterns excluded even if allowed. |

Built-in profiles:

| Profile | Family | Loader | Quantisation |
| --- | --- | --- | --- |
| `gemma4-text-safetensors-bf16` | `gemma` | `mlx` | `bf16` |
| `qwen3-dense-safetensors-bf16` | `qwen` | `native-metal` | `bf16` |
| `qwen36-safetensors-bf16` | `qwen` | `native-metal` | `bf16` |
| `qwen36-mlx-4bit` | `qwen` | `mlx` | `4bit` |

Qwen built-ins allow:

- `*.json`
- `*.jinja`
- `*.txt`
- `tokenizer*`
- `README.md`
- `LICENSE*`
- `*.safetensors`
- `*.safetensors.index.json`

All built-ins ignore:

- `*.bin`
- `*.pt`
- `optimizer*`
- `training_args.bin`

The Gemma text-only profile also ignores image processor, preprocessor,
vision-tower, and projector artifacts.

Pattern matching is simple exact matching, suffix matching for patterns that
start with `*`, and prefix matching for patterns that end with `*`. It is not a
full glob implementation.

## Model Store Layout

The store root defaults to `.llm-models`.

```text
<model-home>/
  huggingface/
    models--<org>--<name>/
      staging/
        <resolved-commit>.partial/
      snapshots/
        <resolved-commit>/
          llm-engine-manifest.json
          config.json
          tokenizer.json
          model.safetensors.index.json
          model-00001-of-00026.safetensors
          ...
        <resolved-commit>.metadata-only/
          llm-engine-manifest.json
          config.json
          tokenizer.json
          model.safetensors.index.json
          ...
      quarantine/
        <snapshot>.quarantined.<timestamp>.<counter>/
          llm-engine-quarantine.json
          llm-engine-manifest.json
          ...
  aliases/
    <alias>.<digest>.json
```

Repository ids must be `org/name`. Snapshot commit ids must be 40-character
immutable hex SHAs.

## Snapshot Manifest

Promoted snapshots contain `llm-engine-manifest.json`.

Manifest fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | Manifest schema version. Current value is `1`. |
| `source` | Current value is `huggingface`. |
| `repo_type` | Current value is `model`. |
| `repo_id` | Hugging Face repo id. |
| `requested_revision` | User-requested revision such as `main`. |
| `resolved_commit` | Immutable 40-character commit SHA. |
| `profile` | Acquisition profile name. |
| `family` | Profile family. |
| `loader` | Profile loader. |
| `quantization` | Profile quantisation. |
| `created_at` | Manifest creation timestamp. |
| `snapshot_path` | Path recorded at promotion time. |
| `files` | Selected files with size and identity. |
| `allow_patterns` | Profile allow patterns used to select files. |
| `ignore_patterns` | Profile ignore patterns used to exclude files. |

Manifest file entries:

| Field | Meaning |
| --- | --- |
| `path` | Relative artefact path inside the snapshot. |
| `size` | Expected file size in bytes. |
| `etag` | Source ETag or LFS object id when present. |
| `sha256` | Normalised SHA-256 when the source ETag is 64 hex characters. |
| `class` | `config`, `tokenizer`, `weights`, `quantization`, `license`, or `other`. |

SHA-256 verification happens only when `sha256` is present.

## Native Snapshot Requirements

`NativeQwenBackend::open` requires these files:

- `config.json`
- `tokenizer.json`
- `model.safetensors.index.json`
- every shard file referenced by the index `weight_map`

`llm-engine-manifest.json` is required for `model list`, `model inspect`, and
`model verify`, but the native server does not read it when opening a snapshot.

`generation_config.json` and `chat_template.jinja` may be present, but the
runtime does not read them.

## Qwen Config Requirements

Supported root fields:

| Field | Requirement |
| --- | --- |
| `architectures` | First item must be `Qwen3ForCausalLM` or `Qwen3_5MoeForConditionalGeneration`. |
| `model_type` | Must be `qwen3` or `qwen3_5_moe`. |
| `text_config` | Required for `qwen3_5_moe`; absent for standard dense `qwen3`. |

Required dense `qwen3` root fields:

- `hidden_size`
- `intermediate_size`
- `rms_norm_eps`
- `rope_theta`
- `num_hidden_layers`
- `num_attention_heads`
- `num_key_value_heads`
- `head_dim`
- `max_position_embeddings`
- `vocab_size`

Required `text_config` fields:

- `model_type`
- `hidden_size`
- `rms_norm_eps`
- `rope_parameters.rope_theta`
- `num_hidden_layers`
- `num_attention_heads`
- `num_key_value_heads`
- `head_dim`
- `linear_num_key_heads`
- `linear_num_value_heads`
- `linear_key_head_dim`
- `linear_value_head_dim`
- `linear_conv_kernel_dim`
- `num_experts`
- `num_experts_per_tok`
- `moe_intermediate_size`
- `shared_expert_intermediate_size`
- `max_position_embeddings`
- `vocab_size`
- `layer_types`

Optional defaults:

| Field | Default |
| --- | --- |
| `tie_word_embeddings` | `false` |
| `rope_parameters.partial_rotary_factor` | `1.0` |

`layer_types` must be exactly `num_hidden_layers` long. Each entry must be
`linear_attention` or `full_attention`.

The committed Qwen3.6 fixture has:

- 40 layers
- 30 linear-attention layers
- 10 full-attention layers
- hidden size `2048`
- 256 experts
- 8 experts per token
- max position embeddings `262144`

## Safetensors Index Requirements

The index schema is:

```json
{
  "metadata": {
    "total_size": 71903645408
  },
  "weight_map": {
    "model.language_model.embed_tokens.weight": "model-00001-of-00026.safetensors"
  }
}
```

The native text loader expects:

- embedding weight
- final norm weight
- `lm_head.weight`
- per-layer input and post-attention norms
- per-layer MoE gate and expert tensors
- `linear_attn.*` tensors for `linear_attention` layers
- `self_attn.*` tensors for `full_attention` layers

Safetensors headers are capped at 64 MiB. Native row and matvec readers are
BF16-oriented and expect rank-2 tensors for row-major operations.

## HTTP Generation Options

Chat request defaults:

| Field | Default |
| --- | --- |
| `messages` | `[]`, but validation rejects empty chat requests |
| `tools` | `[]` |
| `tool_choice` | unset |
| `response_format` | unset, equivalent to text |
| `stream` | `false` |
| `stream_options.include_usage` | `false` |
| `temperature` | unset |
| `top_p` | unset |
| `max_tokens` | backend default; native Qwen uses `256` unless `--max-new-tokens` changes it |
| `stop` | `[]` |

Completion request defaults:

| Field | Default |
| --- | --- |
| `stream` | `false` |
| `stream_options.include_usage` | `false` |
| `max_tokens` | backend default; native Qwen uses `256` unless `--max-new-tokens` changes it |
| `stop` | `[]` |

Validation rules:

- `model` must not be blank.
- Chat `messages` must not be empty.
- Completion `prompt` must not be empty.
- `max_tokens` must be greater than `0`.
- `stop` may be missing, `null`, a string, or an array of strings.
- Stop strings must not be empty.
- `temperature` must be absent or exactly `0.0`.
- `top_p` must be absent or exactly `1.0`.
- `response_format.type=json_schema` is rejected.

## Tokenizer And Template Behaviour

The runtime chat path uses:

```text
enable_thinking = false
add_generation_prompt = true
```

The Rust renderer emits simplified Qwen ChatML and inserts:

```text
<think>

</think>

```

before assistant generation when thinking is disabled.

The renderer supports system, user, assistant, and tool messages. It serialises
prior assistant tool calls as JSON `<tool_call>` blocks.

The runtime does not execute downloaded Jinja templates and does not support
multimodal content arrays.

## Parser Behaviour

The Qwen parser recognises:

- `<think>...</think>` reasoning tags
- JSON `<tool_call>{"name":"...","arguments":{...}}</tool_call>` blocks
- XML-style `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>` blocks

Reasoning is parsed but not returned in the OpenAI message body. Assistant
content outside recognized reasoning and tool-call tags is preserved. Malformed
tool markup returns `malformed_tool_call`.

## Current Limits

- Dense Qwen3 and Qwen3.5/Qwen3.6 MoE text loading are supported.
- `generation_config.json` sampling settings are ignored.
- Non-greedy sampling is not implemented.
- Streaming chunks are assembled after backend generation.
- Unknown JSON request fields are not denied by the current request structs.
- The native loader does not validate every required indexed text weight at open
  time; missing tensors fail during execution.
