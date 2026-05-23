# Disk KV Snapshot Header

This reference defines the compatibility contract for future disk-backed KV or
recurrent-state snapshots. Kir does not currently persist native KV snapshots to
disk. Any implementation that adds disk persistence must write this header before
payload bytes and must reject incompatible snapshots before mapping or
deserialising the payload.

The header is intentionally stricter than a file path or model id check. A disk
KV payload is only reusable when the model artefacts, tokenizer, chat template,
runtime ABI, cache layout, capacity, and request mode match the request that
will consume it.

## Container

Disk snapshot files must start with:

1. ASCII magic `KIR-DISK-KV-SNAPSHOT\n`
2. one little-endian `u32` header schema version
3. one little-endian `u64` header byte length
4. a bounded UTF-8 JSON header
5. payload bytes

Readers must parse and validate the complete header before reading payload
bytes. The first supported JSON schema version is `1`. Unsupported major schema
versions are hard rejects.

## Header Schema

This is the canonical shape for schema version `1`. Field names are stable. New
optional fields may be added only when older readers can ignore them safely.

```json
{
  "object": "disk_kv_snapshot.header",
  "schema_version": 1,
  "created_at": "2026-05-22T00:00:00Z",
  "snapshot_id": "sha256:...",
  "model": {
    "model_id": "local-qwen36",
    "repo_id": "Qwen/Qwen3.6-35B-A3B",
    "resolved_commit": "0123456789abcdef0123456789abcdef01234567",
    "manifest_digest": "sha256:...",
    "artifact_fingerprint": null,
    "profile": "qwen36-safetensors-bf16",
    "family": "qwen",
    "loader": "native-metal",
    "quantization": {
      "label": "bf16",
      "schema_hash": "sha256:..."
    }
  },
  "tokenizer": {
    "kind": "huggingface-tokenizer-json",
    "hash": "sha256:...",
    "normalization_version": "llm-tokenizer/hf-json/v1"
  },
  "chat_template": {
    "cache_template_id": "chatml/qwen/v1",
    "source_hash": null,
    "runtime_template_version": "llm-tokenizer/qwen-chatml/v1",
    "chat_template_kwargs_hash": "sha256:..."
  },
  "runtime": {
    "backend": "native-qwen",
    "backend_kind": "native-metal",
    "runtime_abi": "kir-ai/native-qwen/qwen/v1",
    "target_triple": "aarch64-apple-darwin",
    "endianness": "little",
    "state_dtype": "bf16"
  },
  "cache": {
    "layout_version": 1,
    "payload_kind": "native-prefix-state",
    "layer_count": 40,
    "model_context_length": 262144,
    "cache_tokens": 4096,
    "max_prefill_tokens": 2048,
    "prefix_token_count": 1024,
    "prefix_token_hash": "sha256:..."
  },
  "request": {
    "cache_key": "sha256:...",
    "stable_prefix_key": "sha256:...",
    "mode": "chat",
    "json_object": false,
    "required_tool": null,
    "tool_schema_hash": null,
    "system_prompt_hash": "sha256:...",
    "chat_template_kwargs_hash": "sha256:..."
  },
  "integrity": {
    "payload_bytes": 123456789,
    "payload_hash": "sha256:..."
  }
}
```

### Model Identity

`model.manifest_digest` is required for promoted snapshots that contain
`llm-engine-manifest.json`. The digest must be the same digest reported by the
model store for the loaded snapshot.

`model.artifact_fingerprint` is required only for raw manifestless snapshots. It
must be a hash over a canonical list of required artefacts, including relative
path, size, and content digest for `config.json`, `tokenizer.json`, the
safetensors index, weight shards, and any quantisation metadata. Raw snapshot
paths alone are never a valid fingerprint.

`repo_id`, `resolved_commit`, `profile`, `family`, `loader`, and
`quantization.label` mirror the promoted model manifest and backend metadata.
They are duplicated in the disk header so operators can diagnose misses without
opening the model manifest.

### Tokenizer And Template Identity

`tokenizer.hash` must identify the tokenizer content that produced
`cache.prefix_token_hash`. For Hugging Face tokenizers this is the SHA-256 of
`tokenizer.json` as loaded by Kir, not the upstream ETag unless the ETag is
already a normalised SHA-256.

`chat_template.cache_template_id` must match the family adapter template id used
by `BackendCacheContext`. If a future backend loads an external template source,
`chat_template.source_hash` must contain the source content hash. Built-in
templates may keep `source_hash` as `null` but must set a stable
`runtime_template_version`.

`chat_template.chat_template_kwargs_hash` and
`request.chat_template_kwargs_hash` are the same value. The duplicate keeps the
template section self-contained while preserving the request-cache identity
shape used by admin metrics.

### Runtime And Cache Layout

`runtime.runtime_abi` must change whenever a persisted payload written by one
Kir binary could be misread by another binary, including changes to tensor
ordering, dtype encoding, recurrent-state representation, block-table encoding,
RoPE position handling, or backend-specific cache structs.

`cache.layout_version` is the cache payload layout version for the family and
backend. It is not the JSON header schema version. `cache.payload_kind` names the
logical payload, such as native prefix state or a future paged-KV block pool.

`cache.cache_tokens` is the same token-capacity bucket currently stored in the
native prefix-cache namespace. `cache.max_prefill_tokens` is also part of the
compatibility key because it affects how native prefill state is produced and
observed. Implementations must not silently reuse a snapshot written for a
different bucket or prefill chunk size.

`cache.prefix_token_hash` is a hash of the exact token ids covered by the
payload, encoded with an explicit length prefix per token. It is not a prompt
hash and must not store prompt text.

### Request Mode Identity

The request section records only fields that can affect reusable prefix state:

| Field | Meaning |
| --- | --- |
| `cache_key` | `BackendCacheContext` hash over template id, tool schema, and chat-template kwargs. |
| `stable_prefix_key` | Runtime grouping hash over family, template id, tool schema hash, system prompt hash, and chat-template kwargs hash. |
| `mode` | `chat` or `raw_completion`. |
| `json_object` | Chat JSON-object response mode. Must be absent or `false` for raw completions. |
| `required_tool` | Required function tool name, or `null`. |
| `tool_schema_hash` | Hash of the canonical tool schema used for rendering and parser constraints. |
| `system_prompt_hash` | Hash of system messages that affected prompt rendering, or `null`. |
| `chat_template_kwargs_hash` | Hash of effective chat-template kwargs. |

Sampling controls such as `temperature`, `top_p`, and random seed are not part
of disk KV compatibility because they affect decode selection after reusable
prefix state. `max_tokens` is represented only through the derived
`cache.cache_tokens` bucket; if the bucket differs, the snapshot is a miss.

## Alignment With Native Prefix Cache

Future disk snapshots must be at least as strict as the current in-memory native
prefix-cache namespace.

| Native prefix namespace field | Disk header field |
| --- | --- |
| `model_id` | `model.model_id` |
| `backend` | `runtime.backend`, using the same `BackendModelMetadata.backend` value |
| `family` | `model.family` |
| `quantization` | `model.quantization.label` |
| `repo_id` | `model.repo_id` |
| `resolved_commit` | `model.resolved_commit` |
| `profile` | `model.profile` |
| `cache_key` | `request.cache_key` |
| `tool_schema` | `request.tool_schema_hash` of the same canonical schema |
| `request_mode` | `request.mode`, `request.json_object`, and `request.required_tool` |
| `cache_layout_version` | `cache.layout_version` |
| `cache_tokens` | `cache.cache_tokens` |
| `max_prefill_tokens` | `cache.max_prefill_tokens` |

Disk headers add manifest digest, raw artefact fingerprint, tokenizer hash, chat
template source/runtime version, runtime ABI, payload integrity, and token-prefix
hashes because those checks are either implicit in memory or impossible to infer
after process restart.

## Compatibility Outcomes

There are three outcomes when a reader considers a snapshot for a request.

`accept`: every required field matches and payload integrity checks pass. The
reader may map or deserialize the payload.

`recoverable_miss`: the header is well-formed and safe to inspect, but it does
not match the current request or loaded model. The reader must skip the payload
and generate from a cold prefill path.

`hard_reject`: the file is malformed, unsafe, corrupt, or written for an
unsupported payload ABI. The reader must not load it. If the snapshot was
selected explicitly by an admin command or startup flag, the command must fail.
If the snapshot was discovered opportunistically as a cache candidate, the
request may continue as a cold miss, but the rejection must be reported.

| Check | Mismatch outcome |
| --- | --- |
| Magic, binary header length, JSON syntax, required fields | Hard reject |
| Header schema major version | Hard reject |
| `runtime.runtime_abi`, `target_triple`, `endianness`, `state_dtype` | Hard reject |
| `cache.layout_version`, `payload_kind`, layer count, tensor shape metadata | Hard reject |
| Payload length or payload hash | Hard reject |
| `model.manifest_digest` or `model.artifact_fingerprint` | Recoverable miss |
| `model.family`, `model.loader`, `model.quantization`, `profile`, `repo_id`, `resolved_commit` | Recoverable miss |
| `tokenizer.hash` or tokenizer normalisation version | Recoverable miss |
| `chat_template.cache_template_id`, source hash, runtime template version, kwargs hash | Recoverable miss |
| `request.cache_key`, `stable_prefix_key`, mode, JSON-object flag, required tool, tool schema hash, system prompt hash | Recoverable miss |
| `cache.cache_tokens`, `cache.max_prefill_tokens`, `model_context_length` | Recoverable miss |
| `created_at`, `snapshot_id`, diagnostic duplicate fields | Not compatibility-bearing |

## Error Reporting

Hard rejects must report stable error metadata. The future implementation should
use an error shape equivalent to:

```json
{
  "code": "disk_kv_snapshot_incompatible",
  "phase": "disk_kv_snapshot_load",
  "retryable": false,
  "snapshot_id": "sha256:...",
  "field": "runtime.runtime_abi",
  "expected": "kir-ai/native-text/qwen/v2",
  "actual": "kir-ai/native-text/qwen/v1"
}
```

Use `disk_kv_snapshot_invalid_header` for malformed headers,
`disk_kv_snapshot_unsupported` for unsupported schema or ABI,
`disk_kv_snapshot_corrupt` for payload integrity failures, and
`disk_kv_snapshot_incompatible` for explicit loads whose valid header does not
match the active model or request. Report only ids, hashes, enum values, sizes,
and paths; never report prompt text, message content, raw tool schemas, or token
arrays.

Recoverable misses should be observable through counters and structured debug
logs with the same `field`, `expected`, and `actual` metadata, but they must not
surface as OpenAI request failures.

## Test Plan

The first disk KV implementation must cover these cases before enabling
persistence by default:

- Stale snapshot: write a valid header with an old `manifest_digest` or
  `resolved_commit`, then load the current model. The candidate must be a
  recoverable miss, payload bytes must not be read, and generation must continue
  through cold prefill.
- Cross-model snapshot: write a Qwen header and attempt lookup for Gemma, or use
  two different `repo_id` values with the same served alias. The candidate must
  miss on family or model artefact identity.
- Cross-template snapshot: write a header for `chatml/qwen/v1` or one
  `chat_template_kwargs_hash`, then request a different template id, system
  prompt hash, tool schema hash, or kwargs hash. The candidate must miss before
  payload access.
- Cross-quant snapshot: write a BF16 native header and attempt lookup for a
  4-bit MLX or other quantisation profile. The candidate must miss on
  quantisation, loader, or backend identity.
- Runtime/layout incompatibility: write an otherwise matching header with a
  different `runtime.runtime_abi` or `cache.layout_version`. Explicit loading
  must hard reject with stable error metadata.
- Corrupt payload: keep the header compatible but truncate the payload or change
  `integrity.payload_hash`. Loading must hard reject and must not return partial
  cache state.
- Capacity mismatch: write a header with a different `cache.cache_tokens` bucket
  or `cache.max_prefill_tokens`. The candidate must be a recoverable miss,
  matching the existing native prefix-cache namespace behaviour.
- Ignored decode controls: write a compatible header and change only sampling
  controls. The snapshot remains eligible, proving decode-only controls are not
  part of the disk compatibility key.

Existing in-memory prefix-cache tests already protect part of this contract:
they separate capacity, manifest/profile identity, required tool names, tool
schemas, and chat-template kwargs. Disk tests should reuse the same namespace
fixtures where possible and add manifest digest, tokenizer, payload integrity,
and runtime ABI coverage.
