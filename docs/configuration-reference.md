# Configuration Reference

This reference describes the configuration surfaces that exist in the current
workspace: environment variables, server flags, model acquisition profiles,
snapshot layout, Qwen config requirements, safetensors requirements, and request
generation options.

## Environment Variables

| Variable | Used By | Description |
| --- | --- | --- |
| `HF_TOKEN` | `serve`, `model plan`, `model pull` | Hugging Face bearer token for gated or private repositories. Anonymous access is used when unset. When `serve` uses a configured hub endpoint with this token, the endpoint must be HTTPS. |
| `LLM_HUB_ENDPOINT` | `serve` | Hugging Face-compatible Hub endpoint for admin model plan/pull routes when `--hub-endpoint` is not passed. Tokenless local HTTP mirrors are allowed; endpoints used with `HF_TOKEN` must be HTTPS. |
| `LLM_MODEL_HOME` | `model list`, `model pull` | Model store root when `--model-home` is not passed. Defaults to `.llm-models`. |
| `LLM_ENGINE_SNAPSHOT` | `mise run run-inference` | Raw snapshot path to serve. Mutually exclusive with `LLM_ENGINE_SNAPSHOT_ALIAS`. |
| `LLM_ENGINE_SNAPSHOT_ALIAS` | `mise run run-inference` | Model-store alias to resolve and serve. |
| `LLM_ENGINE_LOADER` | `mise run run-inference` | Optional raw snapshot loader override such as `mlx`. |
| `LLM_ENGINE_FAMILY` | `mise run run-inference` | Optional raw snapshot family override such as `qwen`. |
| `LLM_ENGINE_MODEL` | `mise run run-inference` | Served model id. Defaults to the snapshot alias or `local-qwen36`. |
| `LLM_ENGINE_ADDR` | `mise run run-inference` | Listen address. Defaults to `127.0.0.1:3000`. |
| `LLM_ENGINE_MAX_NEW_TOKENS` | `mise run run-inference` | Generation cap passed to `--max-new-tokens`. Defaults to `256`. |
| `LLM_ENGINE_MAX_PREFILL_TOKENS` | `mise run run-inference` | Native prefill chunk size passed to `--max-prefill-tokens`. Defaults to `2048`; lowering it is mainly for memory-constrained correctness probes. |
| `LLM_ENGINE_PREFIX_CACHE_BYTES` | `serve`, `bench qwen-long-context`, `mise run run-inference` | Native Qwen/Gemma prefix-cache byte budget when `--native-prefix-cache-bytes` is omitted. Defaults to `536870912`; `0` rejects stores while generation continues without prefix reuse. Benchmark traces record this value under `cache_policy.env` when set. |
| `MLX_LM_ENDPOINT` | `serve`, `mise run run-inference` | Loopback MLX sidecar `/v1` endpoint when `--mlx-endpoint` is omitted. |

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
| `mise run run` | Delegates to `mise run run-inference`; requires `LLM_ENGINE_SNAPSHOT` or `LLM_ENGINE_SNAPSHOT_ALIAS`. |
| `mise run run-protocol` | `cargo run -p llm-engine --features test-utils -- serve --protocol-test-backend --i-understand-this-is-not-real-inference` |

## Server Configuration

`llm-engine serve` flags:

| Flag | Type | Default | Behaviour |
| --- | --- | --- | --- |
| `--addr` | socket address | `127.0.0.1:3000` | Address bound by Axum. |
| `--protocol-test-backend` | boolean | unset | Enables protocol test mode without model artifacts. Requires `test-utils` and `--i-understand-this-is-not-real-inference`. |
| `--deterministic-test-backend` | boolean | unset | Deprecated compatibility alias for `--protocol-test-backend`; it has the same feature and acknowledgement requirements. |
| `--snapshot` | path | unset | Enables manifest-selected serving. Without this flag, `serve` requires an acknowledged protocol backend. |
| `--snapshot-alias` / `--model-alias` | string | unset | Resolves a promoted snapshot from the model store alias records. |
| `--loader` / `--backend` | `native-metal` or `mlx` | manifest or `native-metal` | Selects the snapshot loader for raw snapshots without a Kir manifest. Conflicting manifest metadata is rejected. |
| `--family` | `qwen`, `deep_seek`, `gemma`, or `llama` | manifest metadata or native `config.json` detection | Supplies model-family metadata for raw snapshots. Raw native snapshots infer Qwen or Gemma from `config.json` when omitted. Raw MLX snapshots must set this explicitly. Conflicting manifest metadata is rejected. |
| `--model-id` | string | `local-qwen36` | Served model id for snapshot mode. |
| `--hub-endpoint` | URL | `https://huggingface.co` | Hugging Face-compatible Hub endpoint for admin model plan/pull routes. `LLM_HUB_ENDPOINT` is used when omitted. If `HF_TOKEN` is set, this endpoint must be HTTPS; omit `HF_TOKEN` for tokenless local HTTP mirrors. |
| `--max-new-tokens` | `u32` | `256` | Native backend generation cap. Clamped to at least `1`. |
| `--max-prefill-tokens` | `usize` | `2048` | Native prefill chunk size. Long-context native serving depends on keeping this large enough to avoid thousands of tiny prefill steps. Clamped to at least `1`; context retention is allocated from prompt length plus generation budget and rejects requests beyond the model context limit. |
| `--max-public-inference-requests-per-second` | `usize` | `60` | Global fixed-window rate limit for `/v1/chat/completions` and `/v1/completions`. Values below `1` are rejected. |
| `--mlx-endpoint` | URL | `http://127.0.0.1:8080/v1` | Loopback MLX sidecar `/v1` endpoint for MLX snapshot manifests. Chat requests use `/v1/chat/completions` with lossless OpenAI message history; legacy text completions use a completions-capable sidecar endpoint when the selected family exposes one. Qwen, DeepSeek, and Llama use `mlx_lm.server`; Gemma 4 uses `mlx_vlm.server`. `MLX_LM_ENDPOINT` is used when this flag is omitted. |
| `--native-prefix-cache-bytes` | `u64` | `536870912` | Per-backend Qwen/Gemma prefix-cache budget. Set `0` to reject stores while generation continues without prefix reuse. `LLM_ENGINE_PREFIX_CACHE_BYTES` is used when omitted. |
| `--native-metal-weight-cache-bytes` | `u64` | `8589934592` | Per-backend Metal BF16 weight-buffer LRU budget. Set `0` to disable weight-buffer caching. |
| `--warm-native-metal-weight-cache` | boolean | unset | Preloads rank-2 BF16 tensors into the Metal weight-buffer cache at startup until the configured budget is full. |

Native text backend internal defaults:

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
| `gemma4-e2b-it-mlx-4bit` | `gemma` | `mlx` | `4bit` |
| `gemma4-text-safetensors-bf16` | `gemma` | `native-metal` | `bf16` |
| `llama32-3b-instruct-mlx-4bit` | `llama` | `mlx` | `4bit` |
| `qwen35-4b-mlx-4bit` | `qwen` | `mlx` | `4bit` |
| `qwen35-4b-mlx-8bit` | `qwen` | `mlx` | `8bit` |
| `qwen35-4b-mlx-optiq-4bit` | `qwen` | `mlx` | `optiq-4bit` |
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

Text acquisition profiles also ignore image processor, video preprocessor,
processor config, and vision-tower artifacts that are present in some
multimodal Hugging Face repos but are not part of Kir's text runtime contract.
Gemma text-chat profiles additionally ignore projector artifacts.

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

Native text backends require these files:

- `config.json`
- `tokenizer.json`
- `model.safetensors.index.json`
- every shard file referenced by the index `weight_map`

`llm-engine-manifest.json` is required for `model list`, `model inspect`, and
`model verify`. Serving validates runnable readiness for manifest-bearing
snapshots before opening a backend. Raw native snapshots without a Kir manifest
infer Qwen or Gemma from `config.json` when `--family` is omitted; raw MLX
snapshots still use the explicit `--loader mlx --family ...` path.

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

Qwen native text indexes must include:

- embedding weight
- final norm weight
- `lm_head.weight`
- per-layer input and post-attention norms
- per-layer MoE gate and expert tensors
- `linear_attn.*` tensors for `linear_attention` layers
- `self_attn.*` tensors for `full_attention` layers

Gemma native text indexes must include:

- embedding and final norm weights
- `lm_head.weight` unless embeddings are tied
- per-layer attention, feedforward, and scalar norm tensors
- per-layer `self_attn` q/o projections, q norm, and required k/v projections
- per-layer dense MLP gate/up/down projections
- optional per-layer-input embedding, projection, gate, and norm tensors
- optional Gemma MoE expert and router tensors

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
| `max_tokens` | backend default; native text uses `256` unless `--max-new-tokens` changes it |
| `stop` | `[]` |

Completion request defaults:

| Field | Default |
| --- | --- |
| `stream` | `false` |
| `stream_options.include_usage` | `false` |
| `max_tokens` | backend default; native text uses `256` unless `--max-new-tokens` changes it |
| `stop` | `[]` |

Validation rules:

- `model` must not be blank.
- Chat `messages` must not be empty.
- Chat `messages` must use role-consistent fields: system messages appear before conversation turns, user/system/tool messages include `content`, assistant messages include `content` or `tool_calls`, `tool_calls` appear only on assistant messages, and `tool_call_id` appears only on tool result messages that answer pending assistant tool calls.
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
