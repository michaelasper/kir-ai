# Rust Metal Inference Engine North Star

## Purpose

This document defines the north star for a no-Python, Rust-first local inference engine for Apple Silicon. The engine exists to replace the current stack of Python servers, Python proxies, and upstream-specific tool parsers with a single native runtime that is optimized for agentic coding workflows.

The product goal is not "an LLM server that can answer prompts." The product goal is a native engine that can run long, growing OMP-style coding sessions against frontier local models while preserving tool-call structure, JSON correctness, cache reuse, streaming semantics, and predictable latency.

## Implementation Tracker

Last updated: 2026-05-08.

Current commits:

- `7fe9d9c` - Rust workspace scaffold with north-star crate layout and mise tasks.
- `9dae41e` - Rust-owned OpenAI runtime/server skeleton with deterministic native backend.
- `3630acd` - Native Hugging Face model planning and immutable manifest identity.
- `1f6488e` - Qwen3.6 config parsing for hybrid Gated DeltaNet plus MoE topology.
- `52c78bc` - Safetensors fixture loading and direct Metal smoke compute.
- `ccab198` - Official Qwen3.6 tokenizer artifact fixture and tokenizer wrapper.
- `becf073` - Official Qwen3.6 safetensors index validation.
- `c5f147c` - Native model store pull with staged snapshot promotion.
- `199bf83` - Resumable download and artifact-size verification.
- `e255017` - Ignore generated local model stores.
- `e79e855` - Separate MLX 4-bit profile from native BF16 safetensors profile.
- `3d46d5a` - Header-only safetensors inspection for multi-GB shards.
- `c65c892` - Idempotent verified snapshot reuse and native BF16 default profile.
- `80b4324` - Direct BF16 tensor range reading from file-backed safetensors.
- `3784a5a` - Indexed safetensor shard resolution by tensor name.
- `2929990` - Backend dependency lock refresh.
- `3f74400` - Native Qwen embedding and layer0 input norm probe.
- `8f9aaa3` - Centered Qwen RMSNorm semantics from the official implementation.
- `681d02c` - Layer0 linear-attention input projections.
- `38cc5fd` - Layer0 linear-attention first-token execution.
- `e4ea9b6` - Layer0 MoE router top-expert selection.
- `f0cfbfa` - Selected layer0 MoE expert execution.
- `8c8cd47` - Streaming BF16 matvec top-k over large row matrices.
- `39e85ae` - Native Qwen final norm plus lm-head top logits.
- `56f1b0f` - Generic Qwen layer tensor path helpers.
- `1d52515` - Native linear-attention decoder layer loop.
- `289c1de` - Native full-attention first-token decoder layers.
- `d4cb27d` - Decoded top-token candidates after a full native layer pass.
- `3804e42` - Tracker update for the native Qwen full-pass milestone.
- `26ed009` - OpenAI server path for a constrained native Qwen token.
- `8593a74` - Tested Qwen linear-attention sequence recurrence.
- `2e0a54a` - Tested Qwen full-attention sequence math with RoPE and causal softmax.
- `d7c10eb` - Shard-backed Qwen prefill layer infrastructure and parsed RoPE parameters.
- `6bb686d` - Native Qwen backend uses bounded prompt prefill before lm-head decode.
- `7dcbda2` - Tracker update for bounded Qwen prefill server smoke.
- `5e71661` - Hardened Hugging Face snapshot artifact path and digest integrity.
- `d3b091c` - Correct OpenAI runtime behavior for optional tools, generated tool calls, streaming rejection, and backend error status mapping.
- `608dafe` - Qwen first-token attention paths now require and use full-attention norms/keys plus Gated DeltaNet A/dt parameters; safetensors shard files are cached and matvecs read in chunks.
- `01f6ce4` - Metal vector-add kernel compilation is cached per device, generated placeholder crates were replaced, and unused API/engine dependencies were pruned.
- `cd3cd41` - Hub download input grouping keeps the strict clippy gate clean.
- `87d560d` - Runtime parser construction now satisfies strict clippy checks.
- `a3120e5` - OpenAI chat completions can now return native SSE text chunks with `[DONE]` while streaming tool calls fail closed until delta assembly is implemented.
- `986f580` - Streaming Qwen tool calls now emit structured OpenAI tool-call deltas with JSON argument strings and final `tool_calls` finish reason.
- `24fca9c` - JSON object response mode is now enforced for non-streaming and streaming completions, and parsed tool-call arguments must be JSON objects.
- `2c8d4b5` - OpenAI chat `stop` accepts string or array forms and truncates parsed assistant content at the earliest stop sequence.
- `7b96723` - Added non-streaming `/v1/completions` with OpenAI `text_completion` response shape and stop-sequence support.
- `4ef489e` - Added streaming `/v1/completions` SSE chunks with `text_completion` objects and `[DONE]` termination.
- `6907375` - Engine error responses now include stable machine-readable error codes for API, backend, parser, JSON, serialization, and no-progress failures.
- `bf5ac3e` - The model store can list promoted local snapshots, and `llm-engine model list --model-home <path>` reports snapshot identity, profile, manifest digest, and file counts.
- `45bf64a` - Chat requests now fail closed for unsupported non-greedy `temperature`/`top_p` values instead of silently ignoring sampling controls.
- `4cf4cf6` - Engine error bodies now include failure phase and retryability metadata alongside stable error codes.
- `fdafefd` - Chat and text-completion streams honor `stream_options.include_usage` by emitting a usage-only chunk before `[DONE]`.
- `1c8faef` - Added read-only admin model status endpoints for the currently served model with stable missing-model errors.
- `ca7c097` - Added offline local snapshot `model inspect` and `model verify` commands backed by engine manifests.
- `d49ee2b` - Chat and text completion requests now reject `max_tokens: 0` during request validation.
- `358196e` - Backends now expose model metadata, and admin model status reports artifact identity when native snapshots are serving.
- `3991363` - Legacy text completions now fail closed for unsupported non-greedy `temperature` and `top_p` sampling controls.
- `3dc2083` - Chat and text completions now reject unsupported `n` values instead of silently ignoring multiple-choice requests.
- `1b76a55` - Chat requests now support `max_completion_tokens` as the OpenAI token-limit alias and reject conflicting token limit fields.
- `dbeffc8` - Added non-destructive `llm-engine model prune --dry-run` snapshot usage reporting.
- `17e3fd8` - Added aggregate inference metrics and `GET /admin/metrics` for request counts, stream counts, failures, and token totals.
- `5d4b371` - Malformed JSON request bodies now return stable request-validation error envelopes for chat and text completion routes.
- `bbb4da1` - Chat and text completions now fail closed for unsupported nonzero presence/frequency penalties while accepting neutral zero values.
- `eee90ab` - Chat and text completions now fail closed for unsupported log probability controls instead of ignoring them.
- `5c496d6` - Chat completions now fail closed for explicit parallel tool-call requests until parallel execution policy exists.
- `90e4988` - Added admin snapshot verification for the currently served snapshot-backed model.
- `efd5537` - Request-controlled prompt text now rejects reserved ChatML and tool-call control tokens before rendering.
- `3885568` - Hub planning and download HTTP requests now have explicit connect/request/read timeout bounds.
- `7a20446` - Parsed generated tool calls are validated against declared tools and explicit tool choices.
- `dd34c95` - Runtime generation limits now preserve omitted max-token requests and native Qwen rejects explicit requests above its configured cap.
- `913e25b` - Added model-level concurrency backpressure with structured retryable 429 overload errors.
- `e3ef5d4` - `tool_choice: required` now fails request validation when no tools are declared.
- `562b3dd` - Chat stop sequences are applied to raw backend output before tool-call parsing.
- `33b3954` - Admin endpoints support configured Bearer authentication and non-loopback serving requires an admin token.
- `07073a1` - Chat messages accept OpenAI text content-part arrays and normalize them to internal text.
- `12d8cdf` - Deterministic protocol mode now emits structured tool calls for required tool-choice requests.
- `ded78eb` - Safetensors index shard paths are validated and shard opens are confined to the snapshot root.
- `66f1dd2` - Deterministic protocol mode now emits valid JSON objects for `response_format: json_object`.
- `d6bdd00` - Native Qwen snapshots can be opened for serving without an engine manifest.
- `bfb20f0` - Deterministic protocol mode can return prompt-conditioned chat responses for multi-turn smoke tests.
- `b6df02d` - Model-store snapshot and staging paths include profile identity to avoid cross-profile collisions.
- `bdc18ed` - HTTP SSE responses are constructed before backend completion and hold model permits inside the body stream.
- `f9fe943` - Admin HTTP endpoints can plan and pull model snapshots through the native HubClient and ModelStore paths.
- `9b54269` - Runtime and HTTP streaming consume backend chunks incrementally instead of prebuilding full SSE vectors.
- `dc0b86e` - HTTP request validation runs before SSE construction or model scheduling.
- `e8e51b3` - Model pulls use unique staging directories and clean loser staging after concurrent promotion races.
- `b005401` - Native Qwen multi-token decode fails closed until reusable KV/recurrent cache exists.
- `337c107` - SSE streams emit engine heartbeats while backend generation is stalled before the next data chunk.
- `f0a5e5a` - SSE streams report a structured `stream_stalled` error when backend output exceeds the configured stall timeout.
- `311325f` - Native Qwen startup supports opt-in eager safetensors shard materialization.
- `5af89bc` - Runtime non-streaming chat and text completion generation uses cancellable backend tokens.
- `93470ec` - Text chat SSE applies stop sequences incrementally across backend chunks.
- `e0bc485` - OpenAI sampling controls flow through runtime requests and native Qwen full-logit top-p sampling.
- `b31c1f5` - Production serve startup requires an explicit snapshot unless the deterministic test backend is requested.
- `2cfbf2d` - Native Qwen non-greedy sampling uses full lm-head logits instead of a top-k shortlist.
- `c0d8a74` - Metal includes a Qwen-centered RMSNorm kernel covered against the CPU reference.
- `ac91d0b` - Full-attention sequence prefill has a cache-backed `LayerKvCache` CPU path.
- `a77c129` - Linear-attention sequence prefill has a cache-backed `LinearAttentionCache` CPU path.
- `c2ec88f` - Linear-attention single-token decode has a cache-backed `LinearAttentionCache` CPU step.
- `b1b8f82` - Full-attention single-token decode has a cache-backed `LayerKvCache` CPU step.
- `9934f3e` - Hybrid Qwen specs allocate typed per-layer attention caches.
- `6535150` - Shard-backed full-attention layer prefill can write `LayerKvCache`.
- `30a9583` - Shard-backed linear-attention layer prefill can write `LinearAttentionCache`.
- `5cfb4b0` - Qwen bounded prefill can run through typed per-layer caches.
- `c1dbb12` - Shard-backed linear-attention layer decode can step with `LinearAttentionCache`.
- `79352bc` - Shard-backed full-attention layer decode can step with `LayerKvCache`.
- `1d111cb` - Qwen token decode can step typed per-layer caches after cached prefill.
- `7449bb6` - Native Qwen generation reuses typed layer caches for bounded multi-token decode.
- `2e7c802` - Metal includes a row-major `f32` matvec kernel.
- `86f64ff` - Metal includes a row-major BF16-weight matvec kernel.
- `60d5328` - Safetensors exposes raw BF16 tensor ranges for acceleration paths.
- `a316fbb` - Metal includes a batched BF16-weight matvec kernel.
- `53e7fcd` - Open GitHub issues #35 through #40 are addressed and documented.
- `a6f92f3` - Metal includes chunked `f32` argmax and top-k logits kernels.
- `445c5ff` - Tracker updated with Metal logits kernel commit identity.
- `c48be82` - Admin request cancellation is wired through runtime/backend cancellation tokens.
- `cbe0ba6` - Tracker updated with admin cancellation commit identity.
- `838f218` - Admin metrics report active requests and admin-triggered cancellations.
- `a43547a` - Tracker updated with active/cancellation metrics commit identity.
- `d72c499` - Admin metrics report queue depth and no-progress failures.
- `56808e9` - Tracker updated with no-progress metrics commit identity.
- `36acd23` - Admin metrics report request latency summaries and tokens/sec throughput.
- `a19a5cc` - Tracker updated with latency metrics commit identity.
- `4f8cddf` - Admin metrics report streaming time-to-first-token summaries.
- `c76ce8d` - Tracker updated with streaming TTFT metrics commit identity.
- `b08bdbb` - Admin metrics expose current prefill/decode scheduler phase gauges.
- `7a9198b` - Tracker updated with scheduler phase metrics commit identity.
- `4201417` - Admin metrics report model pull operation counts and promoted manifest bytes.
- `48a3aa5` - Admin metrics report manifest-backed model-store usage.
- `392da7b` - Admin metrics report artifact verification failure counts.
- `a80cd60` - Admin metrics report process RSS bytes on macOS and Linux.
- `3e50842` - Streaming runtime errors include stable error metadata.
- `1822a58` - Backend cancellation support is an explicit trait requirement.
- `6a898ad` - Hub repo IDs are validated as safe two-component paths, and Hub request paths encode revisions.
- `5fb7d9e` - Native Qwen serving routes prefill, decode, MoE dense projections, and lm-head matvecs through a Metal-capable executor.
- `ff60424` - Native Qwen decode-session startup checks cancellation before and between prefill layers.
- `4a5cfbd` - Native Qwen final RMSNorm routes through the Metal-capable execution backend.
- `773dac0` - GitHub issues #45 through #47 are fixed, and native Qwen layer input/post-attention RMSNorm routes through the Metal-capable execution backend.
- `f89343f` - Docs and `mise run run` now require explicit deterministic serve mode, fixing GitHub issue #48.
- `59b5a39` - Native Qwen full-attention q/k RMSNorm and linear-attention q/k/head normalization route through the Metal-capable execution backend.
- `47619c7` - Native Qwen full-attention softmax routes through a Metal softmax kernel with CPU fallback.
- `758845e` - Native Qwen full-attention q/k score dot products route through the Metal-capable matvec executor.
- `95dbe89` - Native Qwen linear-attention recurrent memory/core dot products route through the Metal-capable matvec executor.
- `251c505` - Native Qwen linear-attention convolution+silu mixing routes through a Metal kernel with CPU fallback.
- `bc5a867` - Native Qwen MoE router top-k/softmax selection routes through the Metal-capable executor.
- `37b589b` - Native Qwen MoE selected/shared expert accumulation routes through a Metal weighted-sum kernel with CPU fallback.
- `650b614` - Native Qwen full-attention value mixing routes through the Metal-capable weighted-sum executor.
- `d74fcec` - Native Qwen linear-attention recurrent state updates route through a Metal kernel with CPU fallback.
- `c269d67` - Native Qwen linear-attention recurrent state decay routes through the recurrent-update executor hook.
- `4e2607b` - Native Qwen full-attention cache key/value row gathering routes through a Metal head-row selection kernel with CPU fallback.
- `d27a0a4` - Metal command-buffer execution status is checked before reading outputs, fixing GitHub issue #49.
- `e8b2bad` - Native Qwen Metal CPU fallbacks are logged, metered, and exposed through admin metrics, fixing GitHub issue #50.
- `e476210` - Admin metrics now track active streaming generation prefill/decode phases instead of reporting placeholder gauges.
- `4f70955` - Full-attention KV cache storage now has a tested sliding append primitive that evicts the oldest stored token when full.
- `4d4d237` - Cache-backed full-attention sequence prefill now uses the sliding KV append path when the cache is smaller than the sequence.

Current verified state:

- `mise run fmt-check`, `mise run test`, and `mise run clippy` pass for the workspace.
- `mise exec -- cargo run -p llm-engine -- model plan Qwen/Qwen3.6-35B-A3B --revision main` resolves `main` to commit `995ad96eacd98c81ed38be0c5b274b04031597b0` under profile `qwen36-safetensors-bf16` and plans 71,926,864,255 bytes of selected artifacts without Python.
- Official Qwen3.6 config/template fixtures are stored under `fixtures/qwen36/`.
- `llm-models` parses the official Qwen3.6 hybrid Gated DeltaNet plus MoE topology: 40 layers, 30 linear-attention layers, 10 full-attention layers, 256 experts, 8 routed experts per token, 262,144 native context.
- `llm-tokenizer` renders the Qwen no-thinking assistant prefix as `<think>\n\n</think>\n\n`, matching the official template behavior.
- `llm-engine model pull Qwen/Qwen3.6-35B-A3B --metadata-only --model-home .llm-models` downloads 13 non-weight artifacts through native Rust HTTP, writes a manifest, and promotes snapshot `995ad96eacd98c81ed38be0c5b274b04031597b0` with manifest digest `99e9dbff8de1b239063b12421f276c0b5f67c206844471360a8c69d9a502b825`.
- `llm-engine model pull Qwen/Qwen3.6-35B-A3B --revision main --model-home .llm-models-full` verifies the full existing 39-file BF16 snapshot through the Rust pull path, rewrites the native manifest, and reports manifest digest `e99b85a85a4a7b2fbd971f8a0be12ea32e35a9a83a9aca075b771273f3be652e`.
- `llm-engine model inspect-safetensors .llm-models-full/.../model-00001-of-00026.safetensors --tensor model.language_model.embed_tokens.weight` reads the 3,996,199,712-byte shard header without loading payload bytes and validates the embedding tensor as BF16 `[248320, 2048]` over file byte range `2848..1017121568`.
- `llm-engine model inspect-qwen-input .llm-models-full/.../snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 --token-id 0 --limit 2 --layers 40` executes all 40 native Qwen decoder layers for a single no-cache token from the real BF16 shards. Layer 39 is full attention and ends with hidden prefix `[0.27011680603027344, -0.20515947043895721]`.
- `llm-engine model inspect-qwen-input .llm-models-full/.../snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 --token-id 0 --limit 2 --layers 40 --lm-head-top-k 5 --chunk-rows 2048` applies final norm and streams the full `lm_head.weight`; the current top decoded candidates are token `353` `" I"` at logit `11.456915855407715`, token `49276` `"[]("` at `11.124484062194824`, token `1249` `"[]"` at `10.706412315368652`, token `198` newline at `10.236197471618652`, and token `271` double-newline at `10.230167388916016`.
- The native Qwen executor now covers embedding lookup, centered RMSNorm, hybrid linear/full attention first-token paths, full-attention q/k RMSNorm, cache key/value row gathering, q/k score dot products, softmax, and value mixing, linear-attention q/k/head normalization plus recurrent memory/core dot products, state decay, and state updates, routed MoE expert slices, MoE router top-k/softmax selection, selected/shared MoE accumulation, layer input/post-attention RMSNorm, final norm, streamed lm-head logits, and tokenizer decode for the first token with no Python or external inference engine.
- `llm-engine serve --addr 127.0.0.1:3017 --snapshot .llm-models-full/.../snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 --model-id local-qwen36 --max-new-tokens 1` starts an OpenAI-compatible server backed by the native Qwen executor. `GET /health` reports `python_runtime: false`, `GET /v1/models` lists `local-qwen36`, and `POST /v1/chat/completions` for `Say the word test.` returns a real Qwen-decoded one-token assistant response with usage `{prompt_tokens: 17, completion_tokens: 1, total_tokens: 18}`.
- `llm-engine serve --addr 127.0.0.1:3018 --snapshot .llm-models-full/.../snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 --model-id local-qwen36 --max-new-tokens 1 --max-prefill-tokens 2` exercises the sequence prefill path through the OpenAI-compatible endpoint. The `Say the word test.` smoke returned decoded Qwen token `"#"` with usage `{prompt_tokens: 17, completion_tokens: 1, total_tokens: 18}`.
- The sequence path has unit coverage for Gated DeltaNet recurrent state updates, full-attention RoPE plus causal softmax, and indexed BF16 batched matvecs. Workspace `fmt-check`, `test`, and `clippy` pass after the bounded-prefill backend change.
- GitHub issues #1 through #13 have local fixes committed. The fixes cover hub artifact path sanitization, SHA-256 verification, metadata-only cache isolation, optional tool semantics, generated tool-call parsing, fail-closed streaming behavior before native SSE support, backend error status mapping, Metal kernel reuse, placeholder crate replacement, dependency pruning, Qwen full-attention norm/key usage, Gated DeltaNet A/dt usage, and safetensors shard reuse/chunked matvecs. Workspace `mise run fmt-check`, `mise run test`, and `mise run clippy` pass after the issue pass.
- `/v1/chat/completions` now supports `stream: true` through native Rust SSE for text completions and parsed Qwen tool calls. The stream emits OpenAI-compatible `chat.completion.chunk` events, preserves a stable completion ID across role/content/tool/final chunks, emits tool-call deltas with JSON argument strings, and emits `data: [DONE]` exactly once.
- `response_format: {"type":"json_object"}` is now validated in the Rust runtime before non-streaming responses or SSE streams are returned. Assistant content must parse as a JSON object, and parsed tool-call arguments must be JSON objects.
- OpenAI chat `stop` supports both string and string-array request forms. The runtime applies the earliest stop sequence to parsed assistant content and reports `finish_reason: stop`.
- `/v1/completions` now serves non-streaming OpenAI text completions through the Rust runtime, including usage accounting and stop-sequence truncation.
- `/v1/completions` also supports `stream: true` with native Rust SSE chunks and exactly one `[DONE]` terminator.
- HTTP error bodies now include a stable `error.code` field, so clients can classify model-not-found, backend execution, parser, JSON validation, serialization, and no-progress failures without parsing human-readable messages.
- `llm-engine model list --model-home <path>` enumerates promoted engine-owned snapshots from local manifests, including repo ID, resolved commit, profile, family, loader, quantization, manifest digest, and file count.
- Chat sampling controls are fail-closed: explicit greedy settings `temperature: 0` and `top_p: 1` are accepted, while unsupported non-greedy sampling settings return an `unsupported_capability` validation error.
- HTTP failure bodies now include `error.phase` and `error.retryable` in addition to `error.code`, covering request validation, model resolution, prompt rendering, decode, response parsing, response validation, and serialization phases.
- Streaming chat and legacy text completions accept OpenAI `stream_options.include_usage` and append a final usage-only chunk with `choices: []` before the single `[DONE]` terminator.
- `GET /admin/models` and `GET /admin/models/{alias}` report read-only status for the currently served Rust model. Unknown aliases return the same stable `model_not_found` error metadata as inference requests.
- `llm-engine model inspect <snapshot-path>` reads the engine manifest without network access and reports artifact identity, profile, manifest digest, file count, and total manifest bytes. `llm-engine model verify <snapshot-path>` rechecks manifest file sizes and recorded SHA-256 digests and reports verified file/byte counts.
- Chat and text completion request validation now rejects `max_tokens: 0` with a stable `invalid_request` error before backend execution.
- Backend model metadata is now part of the Rust backend contract. Native Qwen serving reads `llm-engine-manifest.json` at startup and surfaces repo ID, resolved commit, profile, family, loader, quantization, snapshot path, and manifest digest through admin model status.
- Legacy `/v1/completions` sampling controls now match chat validation: explicit greedy `temperature: 0` and `top_p: 1` are accepted, while unsupported non-greedy values return `unsupported_capability`.
- Chat and text completions accept `n: 1`, reject `n: 0` as `invalid_request`, and reject multiple-choice requests as `unsupported_capability` until multi-choice generation is implemented.
- Chat completions accept `max_completion_tokens` as an alias for `max_tokens`, reject zero values, and reject conflicting `max_tokens`/`max_completion_tokens` values before backend execution.
- `llm-engine model prune --dry-run --model-home <path>` reports local snapshot count, per-snapshot manifest byte totals, and zero reclaimable bytes without deleting files. Destructive pruning remains disabled until a retention policy is implemented.
- `GET /admin/metrics` exposes aggregate Rust inference metrics for total, successful, failed, and streamed requests plus prompt/completion/total token counters. Chat and text completion handlers record metrics from runtime usage.
- Malformed JSON bodies on `/v1/chat/completions` and `/v1/completions` now map through the same stable HTTP error envelope as other request-validation failures, including `error.code`, `error.phase`, and `error.retryable`.
- Chat and text completions now accept neutral `presence_penalty: 0` and `frequency_penalty: 0`, and reject unsupported nonzero or non-finite penalty values as `unsupported_capability` instead of ignoring them.
- Chat accepts `logprobs: false` as a no-op and rejects enabled `logprobs`/`top_logprobs`; legacy text completions reject requested `logprobs` until log probability output is implemented.
- Chat completions accept `parallel_tool_calls: false` as a no-op and reject explicit `parallel_tool_calls: true` as `unsupported_capability` until the scheduler has a parallel tool execution policy.
- `POST /admin/models/{alias}/verify` verifies the currently served snapshot from backend metadata via the engine manifest and reports status, snapshot path, repo ID, resolved commit, manifest digest, verified file count, and verified bytes.
- The Qwen ChatML renderer fails closed when request-controlled message content, tool schemas, or prior tool-call payloads contain reserved prompt control tokens such as `<|im_start|>`, `<|im_end|>`, `<tool_call>`, or thinking tags. HTTP chat requests surface this as `chat_template_failed` in the `prompt_rendering` phase.
- `HubClient` builds reqwest clients with configurable connect and whole-request timeouts, and wraps streamed download body reads in a per-chunk deadline. Local socket tests cover a stalled model-info response and a stalled artifact body.
- Parsed generated tool calls must now match the request tool contract before any response is returned. The runtime rejects undeclared tool names, rejects tool calls when `tool_choice` is `none`, rejects names that differ from an explicit function choice, and still accepts multiple generated tool calls when each name was declared.
- Runtime backend requests now carry `max_tokens` as `Option<u32>`, preserving omitted OpenAI token limits as backend defaults instead of converting them to an arbitrary numeric request. Native Qwen uses its configured `max_new_tokens` only for omitted limits and rejects explicit requests above that cap as `unsupported_capability`.
- The HTTP engine state now uses a model-level semaphore. The default serve path allows one concurrent generation, `--max-concurrent-requests` can raise that limit, and requests received while all permits are busy return a stable retryable `model_overloaded` error with HTTP 429.
- Chat request validation rejects `tool_choice: "required"` when the request has no declared tools, returning `invalid_request` before any prompt rendering or backend generation.
- Chat stop sequences now truncate raw model output before Qwen tool-call parsing. Tool calls after the earliest stop marker are suppressed, and the response finish reason remains `stop`.
- `/admin/*` routes can be protected with `--admin-token` or `LLM_ENGINE_ADMIN_TOKEN`; configured deployments require `Authorization: Bearer <token>` and return `admin_auth_required` otherwise. `serve --addr` refuses non-loopback binds unless an admin token is configured.
- `POST /admin/models/{alias}/plan` returns a native Rust download plan for the served model alias, and `POST /admin/models/{alias}/pull` pulls and promotes a snapshot through `ModelStore`. Both routes reuse admin Bearer-token enforcement when configured; `serve` exposes `--model-home`, `--hub-endpoint`, and `HF_TOKEN` wiring for these operations.
- Chat message deserialization accepts plain string content, `null`, and text-only OpenAI content-part arrays such as `[{"type":"text","text":"hello"}]`; text parts are concatenated before prompt rendering.
- The default deterministic/protocol backend now threads required tool choice to the backend and emits a valid `<tool_call>` block for declared tools; optional tools still allow text fallback.
- Safetensors index parsing rejects unsafe shard paths, including absolute paths, parent traversal, Windows-style separators, empty components, and NUL bytes. The shard store also canonicalizes shard paths before opening and rejects symlink escapes outside the snapshot root.
- The default deterministic/protocol backend now returns valid JSON object content when `response_format.type` is `json_object`, while fixed-text backends still fail response validation if they emit invalid JSON.
- `NativeQwenBackend::open` treats `llm-engine-manifest.json` as optional for serving. Missing manifests now yield base native metadata with `snapshot_path`, while present manifests still populate artifact identity and digest fields.
- The default deterministic/protocol chat path now recognizes the poem/critique/rewrite smoke-flow intents in rendered prompts and returns distinct prompt-conditioned responses. Plain deterministic backends and legacy text completions still retain fixed-output behavior.
- Model-store promoted snapshot and staging directory names now include a sanitized profile name in addition to repo and resolved commit. Metadata-only snapshots still use a distinct suffix, and full profiles at the same commit no longer share one manifest directory.
- Streaming HTTP handlers now return SSE responses before backend generation completes, keep the model concurrency permit alive inside the body stream, and forward runtime stream events without prebuilding an SSE vector. `ModelBackend::generate_stream` exposes backend text deltas; runtime and HTTP tests verify that a backend chunk reaches the client before the backend releases its final chunk. Native Qwen serving sends decoded per-token deltas through the same path.
- Chat and text completion handlers now validate parsed request semantics before acquiring the model semaphore, so malformed or unsupported requests return stable 4xx JSON errors even while the model is busy. Streaming request-validation failures and buffered streaming response-validation failures return JSON errors before SSE starts.
- Model-store staging directories now include a per-request unique suffix instead of sharing one deterministic `.partial` path. If another pull has already promoted the target snapshot, the losing staging directory is removed and the existing snapshot is verified and reused.
- Native Qwen serving no longer reruns bounded prefill for every generated token. It now pre-fills once into typed per-layer caches, reuses those caches for bounded multi-token decode, defaults omitted native token limits to the configured `max_new_tokens`, and rejects explicit requests only above that cap.
- SSE streaming responses now use Axum keep-alive frames with an `llm-engine-heartbeat` marker. HTTP contract coverage holds the backend before its first content chunk and verifies a heartbeat reaches the client before generation is released.
- Streaming handlers now enforce a configurable backend-output stall timeout, defaulting to 300 seconds through `EngineOptions`. If the runtime stream does not produce the next backend event before the timeout, the SSE body emits a retryable `stream_stalled` error event followed by `[DONE]`, records failure metrics, and releases the model permit.
- Runtime errors that occur after an SSE stream has started now emit the same stable `error.code`, `error.phase`, and `error.retryable` metadata as non-streaming JSON errors before the terminal `[DONE]`.
- Safetensors shards can now be materialized through a read-only mmap cache. The shard store exposes per-tensor materialization, counts materialized cached shards, and serves validated tensor byte ranges from the mmap once populated.
- Safetensors BF16 tensor ranges can now be read either as expanded `f32` values or as raw little-endian BF16 bit words, giving acceleration paths a lossless weight representation without re-reading shard bytes.
- Streaming backend requests now carry a cancellation token. Dropping an HTTP SSE body cancels the runtime/backend stream, and the native Qwen stream path checks cancellation before and after bounded blocking decode steps.
- `ModelBackend::generate_with_cancel` is now an explicit trait requirement, and the default cancellable stream adapter delegates through it instead of silently falling back to non-cancellable generation.
- Native Qwen decode-session startup now accepts the request cancellation token and checks it before cache allocation, before embedding/prefill work, and between each prefill layer. Streaming generation treats cancellation during startup as a clean stop instead of emitting a backend error.
- GitHub issues #45 through #47 have local fixes: native greedy decoding keeps whitespace-only top logits, snapshot verification rejects symlinked manifest artifacts before hashing, and invalid hub endpoints return configuration errors instead of panicking during router construction.
- GitHub issue #48 has a local docs/config fix: protocol-mode serve examples and `mise run run` use `--deterministic-test-backend`, and the docs state that no-snapshot implicit serving was intentionally removed.
- GitHub issues #49 and #50 have local fixes: Metal command-buffer status failures now surface as `MetalError::Execution` before shared-buffer reads, and native Qwen Metal fallbacks emit de-duplicated tracing plus per-kernel attempt/success/fallback counters under `GET /admin/metrics`.
- `llm-kv-cache` now includes a reusable fixed-shape full-attention layer KV cache with contiguous key/value storage, append/read APIs, shape validation, capacity enforcement, and clear/reset behavior.
- `llm-kv-cache` also includes a linear-attention cache primitive with padded rolling convolution history, recurrent-state storage, shape validation, state replacement, mutation access, and clear/reset behavior.
- `SafeTensorShardStore` can now eagerly materialize every unique indexed shard through the same read-only mmap cache, reusing already materialized shards and reporting total mapped bytes.
- `llm-sampler` now includes a deterministic-draw temperature/top-p sampler primitive with stable nucleus ordering, probability validation, and coverage for low/high draws, minimum one-token nuclei, and invalid controls.
- Legacy completion SSE now applies stop sequences on the incremental backend stream path, including stop strings split across backend chunks, without falling back to non-streaming generation.
- Native Qwen startup now has an opt-in eager shard materialization policy through `NativeQwenLoadOptions` and `serve --eager-materialize-shards`, mmap-loading every indexed safetensors shard before advertising the backend.
- Non-streaming chat and text completion generation now use a cancellable backend contract. Dropping the runtime future cancels the backend token, and native Qwen checks the token before and after bounded blocking decode steps.
- Text-only chat SSE now applies stop sequences on the incremental backend stream path, including stop strings split across backend chunks, while still reserving buffered fail-closed paths for tool-call and JSON-object validation.
- OpenAI temperature/top_p controls now validate as native sampling inputs, flow through `BackendRequest` as `SamplingConfig`, and drive native Qwen top-p selection from full lm-head logits with a Rust RNG draw. Backends that do not implement non-greedy sampling fail closed.
- `llm-engine serve` no longer silently starts the deterministic protocol backend when no snapshot is provided. Production serving requires `--snapshot <path>`, while the deterministic backend is explicitly gated behind `--deterministic-test-backend`.
- `llm-metal` now includes a Qwen-centered RMSNorm Metal kernel with smoke coverage against the CPU reference, in addition to the direct vector-add compute smoke.
- `llm-metal` now includes a row-major `f32` matvec Metal kernel with smoke coverage against the CPU reference.
- `llm-metal` now includes a row-major BF16-weight to `f32` matvec Metal kernel with smoke coverage against the CPU reference.
- `llm-metal` now includes a batched row-major BF16-weight to `f32` matvec Metal kernel with input-major output coverage against the CPU reference.
- `llm-metal` now includes chunked `f32` argmax and top-k logits kernels with stable lower-index tie handling and smoke coverage across chunk boundaries.
- `llm-metal` now includes a single-dispatch `f32` softmax kernel with smoke coverage against the CPU reference.
- `llm-metal` now includes a Gated DeltaNet linear-attention convolution+silu kernel with smoke coverage against the CPU reference.
- `llm-metal` now includes a single-dispatch `f32` weighted-sum kernel with smoke coverage against the CPU reference.
- `llm-metal` now includes a Gated DeltaNet recurrent-state update kernel with smoke coverage against the CPU reference.
- `llm-metal` now includes a full-attention cache head-row selection kernel with smoke coverage against the CPU reference.
- Native Qwen serving now routes bounded prefill, decode, MoE dense projections, router top-k/softmax selection, and selected/shared expert accumulation, full-attention q/k RMSNorm, cache key/value row gathering, q/k score dot products, softmax, and value mixing, linear-attention convolution+silu, q/k/head normalization, recurrent memory/core dot products, recurrent-state decay, and recurrent-state updates, layer input/post-attention RMSNorm, final RMSNorm, and lm-head matvecs through a configurable executor. The production executor uses Metal BF16/f32 matvec, Qwen RMSNorm, softmax, linear convolution, weighted-sum, recurrent-update, head-row selection, and top-k kernels when a Metal device is available, with CPU fallback for unsupported shapes or non-Metal hosts.
- Unbuffered chat streams now withhold Qwen `<tool_call>` marker spans from content deltas and validate the complete assistant message before returning terminal stream events.
- Failed model pulls now clean their unique staging directory before returning the original download or integrity error.
- Existing snapshot verification now reuses a matching manifest instead of rewriting timestamp-only metadata, keeping no-op manifest digests stable.
- Public docs now describe supported temperature/top-p sampling controls and the full admin endpoint surface, including mutating model-store operations and admin Bearer-token expectations.
- Hub model IDs are now constrained to exactly two safe path components, and Hub API request paths are built with encoded URL path segments so revisions containing `/` cannot alter route structure.
- `POST /admin/requests/{request_id}/cancel` can cancel active chat and text-completion requests registered with `x-request-id`/`x-llm-request-id` or a generated request ID. Cancelled backend requests return stable `cancelled` error metadata.
- `GET /admin/metrics` now reports current active request count and cumulative admin-triggered cancellation count alongside request and token counters.
- `GET /admin/metrics` now also reports explicit queue depth and cumulative no-progress failure count.
- `GET /admin/metrics` now includes completed-request latency summaries and cumulative tokens/sec throughput.
- `GET /admin/metrics` now records streamed time-to-first-token summaries from the first real content/tool/text delta.
- `GET /admin/metrics` now exposes explicit prefill and decode phase gauges for active generation work. Streaming requests enter prefill before the first real delta, transition to decode after the first content/tool/text delta, and clear the gauge when the stream completes or is dropped.
- `GET /admin/metrics` now reports model pull operation counts, success/failure counts, and promoted manifest bytes for admin pull operations.
- `GET /admin/metrics` now reports manifest-backed model-store snapshot count and total artifact bytes from the configured model home.
- `GET /admin/metrics` now reports cumulative artifact verification failures from failed admin snapshot verification.
- `GET /admin/metrics` now reports process resident memory as `process_rss_bytes` on macOS and Linux.
- `GET /admin/metrics` now exposes native Qwen Metal per-kernel attempts, successes, and CPU fallback counters under `native_qwen_metal`.
- Full-attention sequence prefill now has a cache-backed CPU path that appends normalized RoPE keys and values into `LayerKvCache` and reads that cache for causal attention outputs.
- Linear-attention sequence prefill now has a cache-backed CPU path that updates `LinearAttentionCache` convolution history and recurrent state while matching the existing sequence output.
- Linear-attention single-token decode now has a cache-backed CPU primitive that consumes existing `LinearAttentionCache` state, emits the same next-token output as full cached sequence prefill, and leaves matching convolution/recurrent cache state.
- Full-attention single-token decode now has a cache-backed CPU primitive that uses the cache token count for RoPE position, appends the normalized key/value, attends across the full `LayerKvCache`, and matches full cached sequence prefill.
- `LayerKvCache` now supports strict appends for fixed-capacity prefill/decode and sliding appends that evict the oldest stored key/value row when full, giving the long-context path a tested local eviction primitive.
- Hybrid Qwen layer cache allocation now derives one cache per parsed layer kind, using `LinearAttentionCache` for Gated DeltaNet layers and fixed-capacity `LayerKvCache` for full-attention layers.
- Shard-backed full-attention sequence execution now has a cache-aware layer path that reads indexed safetensors projections, writes `LayerKvCache`, and matches the existing uncached layer output.
- Cache-backed full-attention prefill now also supports local sliding-window execution when the `LayerKvCache` capacity is smaller than the sequence, evicting older key/value rows and attending over the retained window.
- Shard-backed linear-attention sequence execution now has a cache-aware layer path that reads indexed safetensors projections, updates `LinearAttentionCache`, and matches the existing uncached layer output.
- Qwen bounded prefill now has a cache-aware decoder loop that consumes typed per-layer caches, matches the existing uncached prefill output on a tiny shard-backed fixture, and leaves layer cache state populated.
- Shard-backed linear-attention single-token decode now has a cache-aware layer path that reads indexed safetensors projections, consumes existing `LinearAttentionCache`, and matches the corresponding token from full cached layer prefill.
- Shard-backed full-attention single-token decode now has a cache-aware layer path that reads indexed safetensors projections, consumes existing `LayerKvCache`, and matches the corresponding token from full cached layer prefill.
- Qwen single-token decode now has a cache-aware decoder loop that embeds a new token, steps typed per-layer caches, and matches the corresponding suffix from full cached prefill on a tiny shard-backed fixture.
- Native Qwen generation now pre-fills the retained prompt window once into typed per-layer caches, samples from the current hidden state, and steps those caches between generated tokens. Omitted native token limits resolve to the configured `max_new_tokens`, and explicit requests only fail closed above that cap.

Known incomplete items:

- The native Qwen server path currently tokenizes the rendered prompt, keeps a configurable tail window, and reuses typed per-layer caches across bounded multi-token decode. It defaults to 32 retained prompt tokens. Full-attention cache storage and sequence prefill now have local sliding-window behavior, but the server still needs longer-context cache paging and generation-time sliding integration with correct position handling.
- Native Qwen multi-token decode is wired through backend caches and a Metal-capable executor, but attention cache storage/lifetime, recurrent-state cache storage, MoE expert dispatch, and remaining control flow are still CPU-owned. The remaining Qwen Metal kernels are not complete.
- Text and parsed tool-call SSE are implemented, including requested final usage chunks, aggregate streamed-request counts, incremental backend text chunks, heartbeat frames while waiting on backend output, configured stream stall detection, stream-drop backend cancellation, and incremental legacy-completion/text-chat stop handling. Chat tool-call and JSON-object validation paths still buffer where fail-closed semantics require a complete assistant message.
- Full-attention prefill math has RoPE, grouped-query expansion, causal softmax coverage, plus cache-backed `LayerKvCache` math, shard-backed layer prefill, and shard-backed layer step paths. The native Qwen server path now routes projection/output matvecs, q/k RMSNorm, cache key/value row gathering, q/k score dot products, attention softmax, and value mixing through the Metal-capable executor, but attention cache storage and lifetime remain CPU-owned.
- Linear Gated DeltaNet sequence math has recurrent state coverage for bounded prefill plus cache-backed `LinearAttentionCache` math, shard-backed layer prefill, and shard-backed layer step paths. The native Qwen server path now routes projection/output matvecs, convolution+silu mixing, recurrent memory/core dot products, recurrent-state decay, and recurrent-state updates through the Metal-capable executor, but recurrent-state cache storage remains CPU-owned.
- Safetensors metadata, F32 tensor loading, header-only BF16 shard inspection, targeted BF16 f32/raw-bit reads, shard-file/header caching, per-shard and all-shard mmap materialization, native startup eager materialization policy, chunked BF16 matvecs, and full lm-head logit materialization are implemented.
- Direct Metal smoke compute, a Qwen RMSNorm kernel, `f32` softmax, `f32` weighted sum, full-attention head-row selection, linear-attention convolution+silu, linear-attention recurrent update, row-major `f32` matvec, row-major BF16-weight matvec, batched BF16-weight matvec, and `f32` argmax/top-k logits selection are implemented. Native serving now calls full-attention q/k RMSNorm, cache key/value row gathering, q/k score dot products, softmax, and value mixing, linear-attention convolution+silu, q/k/head normalization plus recurrent memory/core dot products, state decay, and state updates, MoE router top-k/softmax selection, selected/shared MoE accumulation, layer input/post-attention RMSNorm, final RMSNorm, and the matvec/top-k subset through a CPU-fallback executor; the remaining Qwen kernels are not complete.
- Large projection reads can now use Metal chunked BF16 matvecs when a device is available, but weights are still copied into temporary Metal buffers per call. The current full 40-layer plus lm-head probe is correctness evidence, not a persistent GPU-resident serving-performance path.
- Admin status, metrics, served snapshot verification, model plan/pull, and active request cancellation HTTP endpoints exist. Non-streaming and streaming decode cancellation is wired through runtime/backend tokens, including checks before native prefill and between prefill layers. Interruption inside a single long Metal command or CPU layer kernel remains non-preemptive.

The first-class model families are:

- Qwen, especially Qwen3.5, Qwen3.6, Qwen3-Coder, and Qwen3-Coder-Next.
- DeepSeek, especially DeepSeek V4 Flash/Pro and legacy DeepSeek V3/R1 tool formats.
- Gemma, especially Gemma 4 text-only inference and Gemma 4 tool/reasoning channels.

The reference engines we currently use are:

- `vllm-mlx`
- `rapid-mlx`
- `oMLX`
- `llama.cpp`
- `SwiftLM`
- `pmetal`

Those engines are reference points, benchmark baselines, and sources of design lessons. The Rust engine is not a wrapper around any of them. The serving runtime must not import Python, start Python, depend on Python object lifetimes, or delegate request lifecycle decisions to a Python process.

## Multi-Agent Orchestration Used For This Spec

This spec was assembled from four independent specialist perspectives and then combined into one implementation direction.

### Runtime Architecture Agent

Scope:

- Rust server architecture.
- Request lifecycle.
- Scheduler.
- SSE and cancellation.
- Tokenizer and sampler boundaries.
- MLX C++ bridge.
- Direct Metal layer.
- KV cache ownership.
- Crate/module layout.

Core contribution:

- Treat inference as an explicit, cancellable state machine.
- Put OpenAI-compatible streaming and agentic invariants in Rust.
- Use MLX C++ as a bootstrap backend, direct Metal for hot paths, and Rust-owned tokenization/sampling/tool parsing.

### Model-Family Support Agent

Scope:

- Qwen, DeepSeek, and Gemma model-family requirements.
- Tokenizer and chat-template details.
- Tool-call and reasoning parser differences.
- Family-specific kernels and cache state.
- Acceptance criteria for first-class support.

Core contribution:

- Do not pretend these models are generic Llama variants.
- Treat architecture, templates, tools, reasoning channels, and cache behavior as model-family contracts.
- Fail closed when a model/template/parser combination is not explicitly validated.

### Existing-Engine Lessons Agent

Scope:

- Lessons from `vllm-mlx`, `oMLX`, `Rapid-MLX`, `llama.cpp`, `SwiftLM`, and `pmetal`.
- What to copy.
- What to avoid.
- Compatibility obligations.
- Migration parity gates.

Core contribution:

- Start from the current production `vllm-mlx` Qwen35 surface as the migration floor.
- Copy `oMLX` cache lifecycle ideas, Rapid KV4/PFlash telemetry discipline, `llama.cpp` GGUF/Jinja/metrics maturity, SwiftLM native-binary direction, and `pmetal` Rust/Metal packaging direction.
- Do not inherit their parser instability, Python dependence, ambiguous cache wins, or speed-first failure modes.

### Verification Agent

Scope:

- Benchmark design.
- Agentic workflow reliability.
- No-progress detection.
- Streaming tool-call invariants.
- Long-context gates.
- Telemetry requirements.
- Launch milestones.

Core contribution:

- Promotion is gate-based, not score-based.
- Direct API probes, protocol conformance, OMP transcript behavior, long-context lifecycle, and no-progress classification must pass independently.
- A request that emits thousands of output tokens with no content/tool deltas is a failed agent turn, even if the backend says it finished normally.

## Hard Product Requirements

### No Python In The Serving Runtime

The serving process tree must contain no Python interpreter and no Python subprocess. Python may remain in the external benchmark harness while the Rust engine is under development, but the engine itself must not rely on Python for:

- HTTP routing.
- OpenAI request parsing.
- Chat-template rendering.
- Tokenization.
- Model loading.
- Tool parsing.
- JSON mode.
- Scheduling.
- Prefill.
- Decode.
- Sampling.
- KV cache management.
- Metrics export.
- Streaming assembly.
- Cancellation.

Python is allowed only as an offline development aid for generating golden fixtures or comparing known outputs. Those fixtures must be committed as static artifacts or generated by explicit developer commands, never by the production runtime.

### Model Acquisition Is A Product Surface

The engine must not assume that models are preinstalled by hand. Model acquisition must be native, reproducible, observable, and explicit.

Required behavior:

- Pull supported model artifacts from Hugging Face.
- Resolve mutable revisions to immutable commits.
- Store verified local snapshots.
- Expose dry-run planning before large downloads.
- Support authenticated, gated, and offline workflows.
- Record artifact identity in every benchmark and request trace.
- Keep model artifact cache separate from runtime KV cache.
- Fail closed when artifacts are missing, corrupt, unauthorized, or revision-ambiguous.

### Native Apple Silicon Runtime

The engine targets Apple Silicon first. It should exploit:

- Unified memory.
- Metal compute.
- Apple GPU-friendly memory layout.
- `libmlx.dylib` and MLX C++ APIs where they accelerate bootstrap.
- Direct Metal kernels where MLX is insufficient or too opaque.
- macOS memory pressure and Metal allocation telemetry.

Cross-platform support is a later concern. Linux/CUDA compatibility is not a launch requirement.

### Agentic Correctness Beats Synthetic Throughput

Raw tokens/sec is secondary to:

- Required tool-call correctness.
- Streaming tool-call reconstruction.
- JSON object correctness.
- Long-context cache correctness.
- Valid final state after tool execution.
- No repeated invalid tool loops.
- No empty/no-progress model turns.
- Bounded and explainable latency.
- Reproducible traces for every failure.

An optimization that improves cold recall but corrupts tool prompts is not a promotion candidate.

### OpenAI Compatibility Is A Contract, Not A Skin

The engine must implement OpenAI-compatible behavior at the semantic level, not only HTTP route names.

Required surfaces:

- `GET /v1/models`
- `POST /v1/chat/completions`
- non-streaming chat completions
- streaming chat completions
- `POST /v1/completions` for raw prompt models and DeepSeek DSML probes
- JSON object mode
- tools
- required tool choice
- automatic tool choice
- tool streaming deltas
- usage accounting
- finish reasons
- model aliases
- health/status endpoint
- metrics endpoint
- explicit request cancellation endpoint

The exact compatibility envelope should be documented by conformance tests. Unsupported OpenAI features must fail explicitly.

## Current Evidence And Motivation

The current local production preset uses `vllm-mlx` with `mlx-community/Qwen3.6-35B-A3B-4bit`, a 200K server context, a conservative 135168-token OMP advertisement, no-thinking chat-template override, `stream-interval=1`, and heartbeat chunks through the local proxy.

The benchmark understanding as of this spec:

- `vllm-mlx` is the production baseline because Qwen35 135K and 200K profiles pass the broadest OMP workloads.
- `oMLX` is the strongest cache challenger because its SSD cache design targets the growing-prefix problem directly.
- `Rapid-MLX` became newly viable at 135K with KV4 prefix reuse, but PFlash is useful only on protected no-tool/no-JSON prompts unless proven otherwise.
- `llama.cpp` is a critical comparator with mature Metal/GGUF/grammar behavior, but its long 200K latency has been slower than leading MLX contenders.
- `SwiftLM` is worth watching for native binary deployment, TurboKV, stream experts, and MoE ideas, but it has not beaten the current Qwen35 production evidence.
- `pmetal` is aligned with the desired Rust/Metal direction, but existing benchmark evidence showed poor OpenAI tool/JSON behavior and server instability.

The recent OMP wedge sharpened the requirement. The server generated 4096 completion tokens and returned a normal finish, while the client saw no useful assistant content or tool calls. The agent then sat after repeated todo reminders. That kind of "successful no-progress turn" must be impossible to classify as success in the Rust engine.

## System Architecture

### High-Level Shape

The engine is a native Rust service with a narrow C++ MLX bridge and a direct Metal layer.

```
Client / OMP / OpenAI SDK
        |
        v
Rust HTTP API
        |
        v
OpenAI Request Normalizer
        |
        v
Model Artifact Resolver
        |
        v
Model Capability Resolver
        |
        v
Template + Tokenizer
        |
        v
Scheduler + KV Cache Manager
        |
        +----------------------+
        |                      |
        v                      v
MLX C++ Backend          Direct Metal Kernels
        |                      |
        +----------+-----------+
                   |
                   v
Sampler + Stop Detector
                   |
                   v
Tool / Reasoning / JSON Assembler
                   |
                   v
SSE / JSON Response Writer
```

The Rust runtime owns all product semantics. The compute backend only computes logits, updates model state, and exposes enough memory/cache hooks for the runtime to make scheduling decisions.

### Request Lifecycle

Every request moves through a typed lifecycle:

1. `accepted`
2. `validated`
3. `model_resolved`
4. `template_rendered`
5. `tokenized`
6. `prefix_lookup`
7. `queued`
8. `prefill`
9. `decode`
10. `finishing`
11. `completed`, `cancelled`, or `failed`

Each phase emits telemetry. The request ID remains stable across all logs, metrics, stream chunks, and trace artifacts.

Detailed lifecycle:

1. Parse HTTP request and capture raw request metadata.
2. Validate route, model alias, parameter ranges, and feature compatibility.
3. Resolve the requested alias to a local artifact snapshot, or fail closed if required artifacts are not present and downloads are disabled.
4. Resolve model family, architecture, quantization, tokenizer, chat template, reasoning mode, tool parser, JSON mode, cache namespace, and context policy.
5. Normalize OpenAI messages without losing type information.
6. Render chat messages and tools into model-native prompt tokens or prompt text.
7. Tokenize in Rust.
8. Check prompt length against model context, advertised context, and runtime memory policy.
9. Hash prefix blocks with model/template/tool-schema compatibility keys.
10. Ask KV manager for reusable blocks.
11. Create a `GenerateRequest` with cancellation token and response event channel.
12. Enqueue into scheduler.
13. Run prefill for uncached spans.
14. Emit prefill progress/heartbeat events while no model deltas are available.
15. Decode tokens or microbatches.
16. Sample in Rust.
17. Feed tokens into stop detector, reasoning splitter, tool parser, JSON validator, and streaming assembler.
18. Emit typed response events.
19. Persist eligible cache blocks.
20. Release temporary buffers and KV references.
21. Write final usage, finish reason, and request summary.

### Scheduler

The scheduler must support continuous batching without making response assembly nondeterministic.

Required scheduling properties:

- Separate admission queues for prefill-heavy and decode-heavy work.
- Configurable prefill chunk size.
- Fair decode scheduling so long-context requests cannot starve short interactive requests indefinitely.
- Request priority, timeout, and cancellation.
- Per-request max prompt tokens, max generated tokens, and memory budget.
- Per-model concurrency limits.
- Bounded queue depth with explicit overload errors.
- Explicit state transitions for every request.
- No detached decode job after client disconnect.

Prefill chunking is an internal policy. It must never alter streamed tool-call semantics.

### Streaming

Streaming is a first-class subsystem.

The model loop emits typed events:

- `Role`
- `ContentDelta`
- `ReasoningDelta`
- `ToolCallDelta`
- `JsonDelta`
- `Heartbeat`
- `Usage`
- `Finish`
- `Error`

The SSE writer converts typed events to OpenAI-compatible chunks. It must never parse raw model text by looking at already serialized SSE strings.

Required streaming invariants:

- Initial role chunk is optional by policy but must be consistent per request.
- Heartbeats are valid empty-delta chunks.
- Heartbeats do not increment token counts.
- Heartbeats do not count as TTFT.
- TTFT is measured from the first content, reasoning, or tool delta.
- Tool-call names and arguments remain reconstructable across arbitrary chunk boundaries.
- `[DONE]` is emitted exactly once on successful stream completion.
- Error streams terminate cleanly and classify the failure.
- Client disconnect cancels the underlying request.

### Cancellation

Cancellation must be cooperative and bounded.

Cancellation sources:

- Client disconnect.
- Explicit admin cancellation endpoint.
- Request timeout.
- Queue timeout.
- Model unload.
- Server shutdown.
- Memory pressure emergency.

Cancellation obligations:

- Release scheduler slot.
- Release KV references.
- Cancel pending response channel.
- Drop temporary Metal buffers.
- Avoid committing partial prefill cache from invalid requests.
- Record cancellation reason.
- Preserve enough trace data to debug the cancellation.

In-flight Metal kernels may not be preemptible. Cancellation latency must therefore be bounded by prefill chunk size and decode step duration.

## Rust Crate Layout

The initial workspace should be multi-crate, even if many crates are small at first.

### `llm-api`

OpenAI-compatible request and response types.

Responsibilities:

- Deserialize/serialize API payloads.
- Validate request fields.
- Preserve unknown fields where useful for diagnostics.
- Own API versioning and compatibility tests.

Key types:

- `ChatCompletionRequest`
- `CompletionRequest`
- `ToolDefinition`
- `ToolChoice`
- `ResponseFormat`
- `ChatCompletionChunk`
- `Usage`
- `FinishReason`
- `ApiError`

### `llm-server`

HTTP service.

Responsibilities:

- `axum`/`hyper` routes.
- SSE response writing.
- Request ID creation.
- Connection lifecycle and disconnect cancellation.
- Health, status, metrics, and admin endpoints.
- Config loading and model registry boot.

### `llm-runtime`

Request lifecycle and scheduler integration.

Responsibilities:

- `GenerateRequest`
- runtime event channels
- request state machine
- scheduler admission
- cancellation propagation
- per-request telemetry

### `llm-tokenizer`

Tokenizer, chat templates, and prompt rendering.

Responsibilities:

- Hugging Face tokenizer loading.
- Special token preservation.
- Chat template implementations.
- Model-native prompt builders.
- Prompt token accounting.
- Stop token and stop string registration.
- Golden byte/token fixtures.

### `llm-hub`

Native model acquisition and artifact store.

Responsibilities:

- Resolve Hugging Face model IDs, revisions, commits, and file manifests.
- Download model, tokenizer, config, license, and template artifacts without Python.
- Support authenticated and unauthenticated Hub access.
- Support dry-run planning before large downloads.
- Support include and exclude patterns for artifact selection.
- Support resumable range downloads for large weight shards.
- Support Xet-backed repositories when a Rust integration is available.
- Verify file sizes, ETags, commit hashes, and optional SHA256 checksums.
- Store snapshots in a versioned local model cache.
- Expose offline-only mode.
- Produce artifact manifests consumed by `llm-models`, `llm-tokenizer`, and `llm-backend`.

Key types:

- `HubRepoId`
- `HubRevision`
- `HubCommit`
- `HubManifest`
- `HubFile`
- `ArtifactSnapshot`
- `DownloadPlan`
- `DownloadProgress`
- `ModelStore`

### `llm-models`

Model-family metadata and architecture declarations.

Responsibilities:

- Qwen family descriptors.
- DeepSeek family descriptors.
- Gemma family descriptors.
- Architecture capability flags.
- Required operators.
- Loader routing.

### `llm-tool-parser`

Reasoning, tool-call, and JSON assembly.

Responsibilities:

- Streaming parser state machines.
- Non-streaming parser normalization.
- OpenAI `tool_calls` normalization.
- Reasoning/content separation.
- JSON mode validation.
- Fail-closed parser selection.

### `llm-sampler`

Sampling and logits processors.

Responsibilities:

- Greedy.
- Temperature.
- Top-p.
- Top-k.
- Min-p.
- Repetition penalties.
- Seed handling.
- Stop detection.
- Logprobs.

### `llm-kv-cache`

KV and recurrent-state cache manager.

Responsibilities:

- Block allocator.
- Prefix hashes.
- Cache namespaces.
- Refcounts.
- Active/reusable/evictable state.
- RAM memory budgets.
- Optional SSD cold tier.
- Cache telemetry.

### `llm-backend`

Backend trait and shared compute types.

Responsibilities:

- `ModelBackend` trait.
- `ModelHandle`.
- `ForwardInputs`.
- `ForwardOutputs`.
- `KvHandle`.
- `BackendError`.
- Backend capability checks.

### `llm-backend-mlx`

C++ MLX bridge.

Responsibilities:

- Load MLX artifacts through C++ MLX.
- Expose narrow FFI.
- Run model forward paths.
- Return logits and state handles.
- Hide all C++/MLX types behind Rust-safe handles.

### `llm-metal`

Direct Metal kernels.

Responsibilities:

- Metal device and queue setup.
- Pipeline cache.
- Kernel source management.
- Buffer lifetimes.
- KV page copy/compact kernels.
- Quant/dequant kernels.
- Attention experiments.
- Sampling/logit kernels where useful.

### `llm-telemetry`

Metrics, tracing, and resource sampling.

Responsibilities:

- Structured JSON traces.
- Prometheus metrics.
- Request phase timing.
- Memory pressure sampling.
- Metal allocation estimates.
- Cache hit/miss metrics.
- Per-token latency histograms.

### `llm-bench-adapter`

Compatibility adapter for the existing benchmark harness.

Responsibilities:

- Emit report fields expected by `llm-server`.
- Provide process audit hooks.
- Export profile metadata.
- Preserve raw request/response traces.

## Model Acquisition And Hugging Face Integration

Hugging Face integration is a first-class product requirement. The engine must be able to acquire, verify, store, inspect, and load model artifacts from Hugging Face repos without relying on Python, shelling out to `hf`, or assuming that a developer manually arranged the filesystem first.

The target user workflow is:

1. Configure a model alias that points at a Hugging Face repo and revision.
2. Run a native Rust pull command or enable an explicit admin pull operation.
3. See a dry-run plan for disk, network, and required files before a large download.
4. Download exactly the required artifacts with resume and progress reporting.
5. Verify the final local snapshot.
6. Start serving from an immutable local snapshot path.
7. Reproduce the exact artifact set later by commit hash.

### Non-Negotiable Download Constraints

The serving runtime must preserve the broader no-Python requirement.

Allowed:

- Native Rust HTTP client.
- Native Rust TLS stack or system TLS.
- Hugging Face Hub REST endpoints.
- Git metadata endpoints where needed.
- HTTP range requests.
- Direct signed download URLs.
- Xet transfer through Rust libraries or a small native helper if it has no Python runtime dependency.
- Offline fixture generation outside the serving process.

Not allowed in the serving process:

- `python -m huggingface_hub`.
- importing `huggingface_hub`.
- shelling out to `hf download`.
- Git LFS command execution as the primary download mechanism.
- mutating the user-global Hugging Face cache without explicit configuration.
- starting a download implicitly from an inference request unless `download_on_demand` is explicitly enabled.

The benchmark harness may keep using Python while the engine is under development. The engine itself must treat Python-based Hub tooling as reference behavior only.

### Artifact Identity

Every downloadable model must resolve to a stable artifact identity before it is eligible to serve.

Required identity fields:

- source provider: `huggingface`.
- repo type: `model`.
- repo ID: for example `Qwen/Qwen3.6-35B-A3B`.
- requested revision: branch, tag, or commit.
- resolved commit hash.
- file allow patterns.
- file ignore patterns.
- expected file list.
- file sizes.
- file ETags or content hashes when available.
- resolved local snapshot path.
- model family.
- expected loader format.
- expected quantization.

The runtime must never use a mutable branch name like `main` as the cache identity. A branch or tag may be accepted in config, but it must be resolved to a commit hash during `pull`, recorded in the local manifest, and surfaced in telemetry.

### Required Artifact Classes

The model store must understand artifact classes rather than treating the repo as an opaque folder.

Required classes:

- model config: `config.json`, `generation_config.json`, architecture-specific config files.
- tokenizer: `tokenizer.json`, tokenizer model files, merges, vocab files, added tokens.
- chat template: embedded tokenizer template, external template overrides, family-specific prompt encoders.
- weights: `safetensors`, MLX arrays, GGUF files, or other supported backend artifacts.
- quantization metadata: quantization config, MLX quant metadata, GGUF metadata, MXFP4 descriptors.
- model card and license: README, license files, gated-model metadata where accessible.
- code artifacts: explicitly ignored by default unless a future sandboxed converter requires them.

The default policy must prefer safe, static data formats:

- prefer `safetensors` over pickle-like formats.
- prefer MLX-native converted artifacts for MLX backend profiles.
- prefer GGUF only for GGUF-specific backend paths.
- reject arbitrary Python model code by default.
- require explicit opt-in for repos that need custom code.

### Download Selection Policy

The puller must support profile-driven file selection.

Examples:

- Qwen MLX path:
  - include `*.safetensors`, `*.safetensors.index.json`, `config.json`, `tokenizer*`, `generation_config.json`, `*.json`.
  - include MLX-specific weight files for `mlx-community/*` repos.
  - exclude training artifacts, optimizer states, datasets, examples, images, and large unused alternate formats.
- DeepSeek DSML path:
  - include model config, tokenizer, native message encoder metadata, DSML templates, and supported quantized weights.
  - exclude unused checkpoint formats.
- Gemma text-only path:
  - include text model config, tokenizer, chat template, and supported text-only weights.
  - exclude multimodal projector or vision weights unless the selected profile needs them.
- GGUF comparator path:
  - include selected `.gguf` files only.
  - reject accidental multi-quant downloads unless explicitly requested.

The download plan must show:

- repo ID.
- requested revision.
- resolved commit.
- files already cached.
- files to download.
- files skipped by pattern.
- total bytes to download.
- total final disk usage estimate.
- required auth status.
- license/gated status when known.

Dry-run support is mandatory because Qwen, DeepSeek, and Gemma repos can be large enough that accidental full-repo downloads are unacceptable.

### Local Model Store

The engine should maintain its own local model store, while optionally reading from an existing Hugging Face cache.

Default location:

```text
$LLM_MODEL_HOME/
  huggingface/
    models--org--repo/
      refs/
        main
        pinned-production
      snapshots/
        <commit>/
          config.json
          tokenizer.json
          model-00001-of-000NN.safetensors
          ...
      blobs/
        <content-id>
      manifests/
        <commit>.json
```

The store may mirror Hugging Face cache conventions where useful, but the engine owns its own manifest format. Serving must always point at a snapshot, never at a mutable staging directory.

Snapshot rules:

- Downloads write to a staging directory.
- Staging directories include a lock file.
- Partial downloads are never considered serveable.
- Successful verification atomically promotes staging to a snapshot.
- Manifests are immutable once promoted.
- Corrupt snapshots are quarantined, not deleted silently.
- Multiple model aliases may point to the same snapshot.
- Runtime KV caches are isolated from model artifact caches.

### Existing Hugging Face Cache Interop

The engine should support read-only import from existing Hugging Face cache roots.

Interop modes:

- `isolated`: use only `$LLM_MODEL_HOME`.
- `read_hf_cache`: search configured Hugging Face cache roots and import matching snapshots by hardlink, clonefile, or copy.
- `write_hf_compatible`: write a cache layout compatible enough for external inspection, without promising full `huggingface_hub` parity.

The default should be `isolated` for reproducibility. `read_hf_cache` is useful on developer machines where large artifacts already exist.

The engine must not corrupt an external Hugging Face cache. If it writes into an HF-compatible layout, it must use atomic operations and must not edit files it does not own.

### Authentication

The puller must support private and gated models without leaking credentials.

Token sources, in priority order:

1. Explicit config field referencing a secret provider.
2. Environment variable such as `HF_TOKEN`.
3. OS keychain entry.
4. Existing token file only if the path is explicitly configured.

Token handling requirements:

- Never log tokens.
- Never include tokens in traces, benchmark artifacts, panic output, or dry-run output.
- Redact authorization headers at the HTTP layer.
- Support token validation without downloading large files.
- Report gated/license errors clearly.
- Distinguish authentication failure from missing repo, missing file, and network failure.

The server should expose read-only model status without exposing whether a private repo exists to unauthenticated remote clients.

### Network And Transfer Behavior

The download layer must be robust enough for 10GB to 400GB artifact sets.

Required behavior:

- parallel downloads with configurable concurrency.
- per-host connection limits.
- HTTP range resume.
- exponential backoff with jitter.
- checksum verification after resume.
- progress events by file and total bytes.
- cancellation.
- disk-space preflight.
- no unbounded memory buffering.
- proxy support through standard environment variables and explicit config.
- custom Hugging Face endpoint support for mirrors or enterprise deployments.
- offline mode that never performs network requests.

Xet-backed storage is now part of normal Hugging Face large-file behavior. The engine should support it in one of two ways:

- Preferred: integrate directly with a native Rust Xet/CAS client if it is available and stable enough.
- Fallback: use normal signed file URLs and HTTP range behavior when Hugging Face exposes them for the requested artifact.

The puller must not require Python `hf_xet`. If the only available implementation for a transfer mode is Python, that mode is not part of the serving runtime.

### Commands And Admin API

The engine should provide both CLI and local admin surfaces.

Required CLI commands:

```text
llm-engine model plan <alias-or-repo> [--revision <rev>] [--profile <profile>]
llm-engine model pull <alias-or-repo> [--revision <rev>] [--profile <profile>]
llm-engine model list
llm-engine model inspect <alias-or-snapshot>
llm-engine model verify <alias-or-snapshot>
llm-engine model prune [--dry-run]
```

Required local admin endpoints:

- `GET /admin/models`
- `GET /admin/models/{alias}`
- `POST /admin/models/{alias}/plan`
- `POST /admin/models/{alias}/pull`
- `POST /admin/models/{alias}/verify`

Admin endpoints must be disabled or token-protected by default. Pull endpoints must reject non-local unauthenticated callers.

Inference requests should not trigger downloads by default. The preferred flow is explicit pull before serve. On-demand download may exist for development, but it must be opt-in and must return clear progress/status rather than making a chat request hang for minutes.

### Manifest Format

Every promoted snapshot must include a native manifest.

Example:

```json
{
  "schema_version": 1,
  "source": "huggingface",
  "repo_type": "model",
  "repo_id": "Qwen/Qwen3.6-35B-A3B",
  "requested_revision": "main",
  "resolved_commit": "0123456789abcdef0123456789abcdef01234567",
  "profile": "qwen35-mlx-4bit",
  "family": "qwen",
  "loader": "mlx",
  "quantization": "4bit",
  "created_at": "<timestamp>",
  "files": [
    {
      "path": "config.json",
      "size": 12345,
      "etag": "\"abc\"",
      "sha256": "optional"
    }
  ],
  "allow_patterns": ["*.json", "*.safetensors", "tokenizer*"],
  "ignore_patterns": ["*.bin", "*.pt", "optimizer*", "training_args.bin"],
  "license": {
    "name": "unknown",
    "requires_acceptance": false
  }
}
```

The manifest must be included in:

- model load telemetry.
- benchmark report metadata.
- cache compatibility keys.
- failure reports involving model artifacts.
- reproducibility bundles.

### Loader Integration

Model acquisition is not complete until the loader can prove the snapshot is usable.

For each family/backend pair, the engine must define required file predicates:

- Qwen MLX backend: MLX weights or convertible safetensors, tokenizer, config, generation config, template/parser metadata.
- Qwen GGUF comparator: selected GGUF file, tokenizer metadata embedded or adjacent, context metadata.
- DeepSeek V4 backend: config, tokenizer, native message encoding metadata, supported quantized weights.
- Gemma 4 backend: text config, tokenizer, tool/channel metadata, supported text weights.

Load-time checks:

- architecture in config matches selected family.
- quantization metadata matches selected backend.
- tokenizer special tokens required by parser are present.
- chat template ID is known and pinned.
- context length and RoPE/YaRN parameters are readable.
- shard count matches index metadata.
- no required file points outside the snapshot.

Failure must happen before the server advertises the model as ready.

### Offline And Reproducible Operation

The production server must support fully offline operation.

Offline mode requirements:

- no network requests during boot.
- no network requests during inference.
- no background metadata refresh.
- no revision resolution against remote branches.
- all aliases must resolve to local snapshot commits.
- missing artifacts fail at boot or explicit model load, not at first request.

This matters for benchmark reproducibility. A benchmark run must record the resolved Hugging Face commit and local manifest digest so a later run can prove whether it used the same model.

### Download Telemetry

Model pulls must emit telemetry separate from inference telemetry.

Required fields:

- operation ID.
- repo ID.
- requested revision.
- resolved commit.
- auth mode: `none`, `token`, `keychain`, or `configured_secret`.
- redacted endpoint host.
- plan bytes.
- downloaded bytes.
- reused bytes.
- file count.
- skipped file count.
- resume count.
- retry count.
- transfer backend: `http`, `xet`, or `mixed`.
- elapsed time.
- final snapshot path.
- manifest digest.
- success/failure class.

Benchmark dashboards should display model artifact identity next to performance metrics. Speed comparisons are invalid if the model revision or quantization differs.

## Backend Strategy

### Phase 1: MLX C++ Bootstrap, No Python

Use `libmlx.dylib`, MLX headers, and a narrow C++ shim.

This path gives us:

- Native Apple Silicon compute.
- Existing MLX tensor runtime.
- Access to `mlx.metallib`.
- A faster path to loading MLX artifacts.
- A way to validate the Rust API/scheduler/tool/cache contracts before writing every kernel ourselves.

Rules:

- No Python MLX.
- No Python model loader.
- No C++ request parsing.
- No C++ tool parsing.
- No C++ sampling policy unless the Rust side explicitly passes a deterministic command.
- No C++ cache policy decisions beyond exposing handles and primitives.

### Phase 2: Direct Metal Hot Paths

Move hot or correctness-sensitive paths into Rust-controlled Metal kernels.

Candidate kernels:

- KV page copy.
- KV page compaction.
- Quantized KV conversion.
- RoPE/p-RoPE/YaRN.
- RMSNorm.
- Logits post-processing.
- Greedy argmax.
- Top-k/top-p support.
- Qwen Gated DeltaNet recurrent state update.
- DeepSeek Sinkhorn collapse/expand.
- DeepSeek compressed attention helpers.
- Gemma local/global attention cache helpers.

Promotion rule:

Direct Metal replaces MLX only when it is measurably faster, more reliable, or necessary for cache/control-plane correctness.

### Phase 3: Native Model Runtime

Only after the API, scheduler, cache, and parser contracts are stable should the engine consider replacing MLX as the primary model backend.

This later phase would include:

- Native safetensors model loader.
- Native GGUF model loader.
- Native module graph.
- Native quantized linear layers.
- Native attention.
- Native MoE routing.
- Native architecture implementations for Qwen, DeepSeek, and Gemma.

The launch plan should not depend on completing Phase 3.

## Memory And Cache Architecture

### Cache Goals

The cache must serve growing agent sessions, not only one-shot prompt recall.

The critical workflow:

1. OMP sends a large system prompt and tool schema.
2. The model calls a tool.
3. OMP appends tool output.
4. The model calls another tool.
5. The transcript grows for many turns.

The engine must preserve reusable prefix state across these turns without silently corrupting tool behavior.

### Cache Key

A prefix cache key must include:

- model artifact ID
- model file digests
- quantization format
- architecture family
- tokenizer version
- chat-template version
- reasoning mode
- tool-parser mode
- tool schema hash
- system prompt token hash
- token block hash
- positional encoding policy
- KV cache dtype
- recurrent-state format
- backend version
- engine version
- relevant runtime flags

If any key component changes, cache reuse must fail closed.

### Cache Contents

The cache must support more than standard Transformer KV.

Required state categories:

- Attention KV pages.
- Sliding-window KV pages.
- Global-attention KV pages.
- Qwen Gated DeltaNet recurrent state.
- DeepSeek compressed attention state.
- Gemma local/global attention state.
- MTP draft/verification rollback state.
- Prefix hash metadata.
- Tool/template compatibility metadata.

### Cache States

Cache blocks move through explicit states:

- `active`
- `reusable`
- `evictable`
- `persisting`
- `cold`
- `loading`
- `invalid`

No block can be reused unless its state and compatibility metadata are valid.

### RAM Tier

RAM tier requirements:

- Fixed-size or bounded variable-size pages.
- Refcounts.
- LRU metadata.
- Memory pressure hooks.
- Maximum active memory budget.
- Emergency eviction path.
- No unbounded memory growth.
- Per-namespace isolation.

### Optional SSD Tier

The SSD tier copies the `oMLX` lesson but must be explicit and measured.

Requirements:

- Versioned cache format.
- Block metadata validation.
- Async writes.
- Bounded outstanding write queue.
- Startup scan.
- Cache corruption detection.
- Cold load latency metrics.
- SSD bytes read/written metrics.
- Ability to disable.

The SSD tier is not a substitute for RAM cache correctness.

## Model Family Requirements

### Qwen

#### Supported Qwen Targets

Initial support:

- Qwen3.5 4B/9B as smoke models.
- Qwen3.6 27B dense.
- Qwen3.6 35B A3B.
- Qwen3-Coder and Qwen3-Coder-Next template/parser paths.

Production focus:

- `mlx-community/Qwen3.6-35B-A3B-4bit`
- GGUF Qwen35 A3B comparator compatibility.

#### Qwen Architecture Requirements

Qwen3.5/Qwen3.6 support must account for:

- Hybrid Gated DeltaNet plus attention block structure.
- Sparse MoE routing.
- 256 experts where applicable.
- `8 routed + 1 shared` active experts where applicable.
- RoPE.
- GQA.
- Hybrid recurrent state plus KV cache.
- MTP heads where present.
- Long-context scaling up to 135K and frontier 200K in our target profiles.

#### Qwen Template Requirements

The tokenizer/template layer must handle:

- ChatML role tokens.
- `enable_thinking=true`.
- `enable_thinking=false`.
- historical `reasoning_content`.
- assistant messages with prior `tool_calls`.
- role=`tool` responses.
- JSON-string OpenAI tool arguments converted back into structured template args.
- no-thinking production mode for Qwen35.

Template correctness must be fixture-tested byte-for-byte.

#### Qwen Parser Requirements

The parser must support:

- `<think>...</think>` reasoning.
- implicit thinking mode where only `</think>` appears.
- `<tool_call>` appearing before `</think>` and implicitly ending reasoning.
- Hermes-style `<tool_call>{...}</tool_call>` JSON.
- Qwen3-Coder XML:
  - `<tool_call>`
  - `<function=name>`
  - `<parameter=k>v</parameter>`
  - `</function>`
  - `</tool_call>`
- typed conversion from tool schema.
- multiple tool calls.
- missing close-tag recovery when unambiguous.
- streaming argument chunks.

The parser must never leak reasoning into `content` or tool markup into streamed text deltas.

### DeepSeek

#### Supported DeepSeek Targets

Initial support:

- DeepSeek V4 Flash MXFP4 path.
- DeepSeek V4 DSML tool prompts.
- DeepSeek DSML thinking tool prompts.
- DeepSeek legacy V3/R1 special-token tools.

Later support:

- DeepSeek V4 Pro.
- DeepSeek V3.2 variants.
- R1 distills where architecture matches supported backends.

#### DeepSeek Architecture Requirements

DeepSeek V4 support must account for:

- MoE routing.
- hash routing for early layers.
- `sqrtsoftplus` routing scores.
- limited SwiGLU.
- shared experts.
- MXFP4/MXFP8 quantized matmuls.
- local sliding-window attention.
- compressed pooled attention.
- sparse top-k pooled lookup.
- attention sinks.
- DeepSeek YaRN/RoPE variants.
- manifold hyper-connections.
- Sinkhorn collapse/expand kernels.

#### DeepSeek Template Requirements

DeepSeek V4 does not behave like a normal Jinja-chat-template model. The runtime must port the native message encoder and parser. It cannot depend on a Python encoding folder at runtime.

Required prompt support:

- Raw `/v1/completions`.
- Chat Completions normalized into native DeepSeek prompt format.
- DSML tools.
- thinking and non-thinking transitions.
- system prompt plus tool schema rendering.

#### DeepSeek Parser Requirements

DSML support:

- `<｜DSML｜tool_calls>`
- `<｜DSML｜invoke name="...">`
- `<｜DSML｜parameter name="..." string="true|false">...`
- `</｜DSML｜parameter>`
- `</｜DSML｜invoke>`
- `</｜DSML｜tool_calls>`

Legacy support:

- `<｜tool▁calls▁begin｜>`
- `<｜tool▁call▁begin｜>`
- `function<｜tool▁sep｜>name`
- fenced JSON arguments
- `<｜tool▁call▁end｜>`
- `<｜tool▁calls▁end｜>`

V3.1/R1-0528 variants:

- same unicode wrapper tokens
- no `function` prefix
- no fenced JSON where expected by the variant

All DeepSeek formats must normalize to OpenAI `tool_calls`.

### Gemma

#### Supported Gemma Targets

Initial support:

- Gemma 4 31B IT Q4 path.
- Gemma 4 31B IT BF16 path where memory allows.
- Gemma 4 text-only mode from multimodal checkpoints.

Later support:

- Gemma 3 standard transformer path.
- Gemma 4 MoE variants.

#### Gemma Architecture Requirements

Gemma 4 support must account for:

- dense 31B text path.
- MoE 26B-A4B path where applicable.
- language-model-only loading from multimodal checkpoints.
- local sliding-window attention.
- global attention.
- unified KV in global layers.
- p-RoPE.
- long context up to model-specific targets.
- clean stripping or ignoring of vision/audio weights for text-only serving.

#### Gemma Parser Requirements

Gemma 4 must not be treated as Hermes by default.

Required parser support:

- native tool calls:
  - `<|tool_call>call:name{...}<tool_call|>`
- string values delimited with:
  - `<|"|>value<|"|>`
- bare numeric args
- bare float args
- bare bool args
- bare null args
- mixed bare and quoted args
- parallel tool calls
- streaming deduplication
- text-format recovery
- reasoning channels:
  - `<|channel>thought`
  - `<|channel>content`
  - `<|channel>final`
  - `<channel|>`
  - `<turn|>`

No raw tool or channel markup may leak into final `content`.

## Operators And Kernels

### Universal Operators

The engine needs native support for:

- safetensors loading.
- sharded safetensors loading.
- GGUF metadata loading.
- tokenizer config parsing.
- quantization config parsing.
- nested `text_config` handling.
- model config validation.
- RMSNorm.
- affine linear layers.
- quantized linear layers.
- RoPE.
- GQA/MQA attention.
- full attention.
- sliding-window attention.
- paged or rotating KV cache.
- quantized KV cache.
- logits processors.
- sampling.

### Qwen-Specific Operators

- sparse MoE top-k routing.
- expert gather/scatter.
- shared expert execution.
- switch or quantized-switch linear layers.
- router score normalization.
- Gated DeltaNet recurrent update.
- hybrid recurrent state snapshot.
- hybrid recurrent state restore.
- MTP draft head where present.
- MTP verification rollback.

### DeepSeek-Specific Operators

- MXFP4/MXFP8 matmul.
- hash routing.
- `sqrtsoftplus` router scoring.
- compressed attention pooling.
- sparse pooled top-k gather.
- attention sinks.
- YaRN/RoPE variants.
- mHC Sinkhorn collapse.
- mHC Sinkhorn expand.

### Gemma-Specific Operators

- local/global attention scheduling.
- unified global KV handling.
- p-RoPE.
- Gemma MoE router path.
- text-only checkpoint filtering.

### Direct Metal Kernel Candidates

Direct Metal should be prioritized for:

- KV page copy and compaction.
- quantized KV conversion.
- Qwen recurrent state update.
- DeepSeek Sinkhorn collapse/expand.
- RoPE/p-RoPE/YaRN.
- logits post-processing.
- greedy argmax.
- top-k candidate selection.
- cache compression/decompression.

Every direct Metal kernel needs:

- CPU reference implementation.
- MLX or known-good reference comparison where possible.
- dtype tolerance thresholds.
- shape coverage.
- long-context stress case.
- memory safety invariants.
- dispatch-size bounds.

## Tool, Reasoning, And JSON Semantics

### Tool Calls Are Structured Output

Tool calls must be parsed by model-family state machines, not by fragile string cleanup after the fact.

The parser must understand:

- selected model family.
- selected tool parser.
- reasoning mode.
- tool schema.
- whether streaming is enabled.
- whether raw completions or chat completions are used.

If the engine cannot select a validated parser, it must return a capability error.

### Streaming Tool Calls

Streaming tool-call state must preserve:

- call index.
- call type.
- function name fragments.
- argument fragments.
- valid JSON reconstruction.
- finish reason.
- duplicate-call prevention.
- final call normalization.

Malformed streamed tool calls are hard failures.

### Reasoning Separation

Reasoning must be separated before content emission.

Family-specific reasoning channels:

- Qwen: `<think>...</think>` and implicit think-end behavior.
- DeepSeek: `<think>...</think>` before DSML tool calls or content.
- Gemma: `<|channel>thought`, `<|channel>content`, `<|channel>final`, and related channel terminators.

No reasoning text may leak into OpenAI `message.content` unless the request explicitly asks for a reasoning field and the model family supports it.

### JSON Mode

JSON mode must enforce JSON object validity.

Rules:

- HTTP 200 with invalid JSON output is not success.
- Scalar JSON is not success when object mode is requested.
- Tool-call JSON arguments must be objects unless the schema explicitly permits another type.
- Streamed JSON must be assembled and validated before final success.
- Parser-recovered JSON must be marked as recovered in telemetry.

## Existing Engine Reference Matrix

### `vllm-mlx`

Copy:

- OpenAI-compatible serving shape.
- Continuous batching.
- `stream-interval=1`.
- conservative Qwen35 production defaults.
- 200K server context with 135K OMP advertisement.
- no-thinking Qwen template control.
- heartbeat mitigation for long prefill.

Avoid:

- treating direct API success as enough.
- aggressive settings that pass recall but damage tools.
- stream buffering that corrupts tool-call arguments.
- hiding parser behavior inside opaque serving code.

Parity gates:

- equal or beat `vllm-qwen35-overnight-135k-4096-control`.
- equal or beat `vllm-qwen35-frontier-200k-4096-control`.
- pass direct probes plus OMP read/bash/edit/fix/repo repair.
- pass 135K and 200K recall/lifecycle gates.

### `oMLX`

Copy:

- block-aware prefix cache.
- RAM plus SSD cold tier.
- safetensors block persistence.
- cache metadata validation.
- startup cache scan.
- async SSD writes.
- per-model settings.
- prefill progress tracking.

Avoid:

- Python runtime dependence.
- unclear cold-vs-warm benchmark interpretation.
- accepting high RSS without explicit memory policy.

Parity gates:

- match `omlx-qwen-a3b-ssd-cache-ctx-135k`.
- match `omlx-qwen-a3b-ssd-cache-ctx-200k`.
- match cache-lifecycle 200K behavior.
- report cache hits/misses, SSD bytes, load/save latency, RSS, cold TTFT, and warm TTFT.

### `Rapid-MLX`

Copy:

- KV4 prefix reuse behavior.
- native streaming tool-call focus.
- protected prompt policy.
- PFlash skip telemetry.
- explicit cache/profile knobs.
- repeated-tool-loop detection.

Avoid:

- relying on PFlash for tool/JSON prompts.
- stream-interval 8 in production.
- silent long prefill stalls.
- tool-parser behavior that emits valid-looking but no-progress turns.

Parity gates:

- match Rapid 32K KV4/turbo-paged gates.
- match `rapid-0615-qwen35-kv4-135k`.
- PFlash parity only applies to plain no-tool/no-JSON long prompts unless protected prompt support changes.
- log apply/skip reason for every compression decision.

### `llama.cpp`

Copy:

- GGUF compatibility.
- mature Metal backend lessons.
- flash attention.
- q8 KV cache options.
- Jinja template discipline.
- reasoning-off controls.
- metrics endpoint.
- continuous batching.
- grammar/JSON-schema/function-calling infrastructure.
- reproducible sampler behavior.
- broad architecture coverage.

Avoid:

- importing C++/ggml complexity into Rust without clear benefit.
- using it as the default production target solely because it is mature.
- ignoring long-context latency gaps.

Parity gates:

- load same Qwen35 GGUF refs.
- preserve OpenAI response shapes.
- run Qwen35 GGUF 135K/200K comparisons.
- beat or explain deltas against llama.cpp on long recall and OMP tool tasks.

### `SwiftLM`

Copy:

- native compiled binary direction.
- GPU-layer auto ideas.
- TurboKV experiments.
- stream-expert ideas.
- SSD prefetch ideas.
- Qwen thinking toggles.
- simple CLI ergonomics.

Avoid:

- extrapolating 4B/9B wins to Qwen35 production.
- prioritizing stream-experts before OpenAI tool/JSON correctness.
- speed claims without OMP task validation.

Parity gates:

- first exceed SwiftLM 4B/9B smoke while passing JSON.
- then pass Qwen35 135K read/bash/edit/fix tasks.
- retest SwiftLM only when its KV/MoE behavior changes.

### `pmetal`

Copy:

- Rust/no-Python packaging direction.
- hardware and memory introspection.
- model-fit search.
- GGUF/MLX quantization tooling ideas.
- KV quant/TurboQuant/QJL knobs.
- SSD expert packing concepts.
- continuous-batch controls.
- bit-identical greedy baseline contract.

Avoid:

- treating current server behavior as production-ready.
- prioritizing FP8/quant experiments before tool/JSON correctness.
- speed-first benchmarking when tool behavior is zero.

Parity gates:

- pass direct canaries.
- pass OMP read/grep/bash/edit on 4B/9B.
- only then test Qwen35 32K/135K.
- speed does not count until structured tool behavior and JSON pass.

## Production Compatibility Floor

The first Rust production candidate should emulate the current local production preset:

- Model: Qwen3.6 35B A3B.
- Server context: 200K.
- Advertised OMP context: 135168.
- Temperature: 0.
- `top_p`: 1.
- `max_tokens`: 4096.
- Qwen thinking: disabled by default.
- OpenAI streaming: enabled.
- Heartbeats: 30s default during long silence.
- Tool surface: compatible with read/bash/edit/find/grep and full OMP AST sessions.
- Runtime cache namespace isolated from model cache.

Only after this floor passes should the engine branch into Gemma and DeepSeek production profiles.

## Benchmark And Verification Strategy

### Gate Philosophy

Promotion is gate-based.

Aggregate score is useful for ranking, but not for launch. A profile cannot launch if it fails any hard gate, even with a high score.

Hard gates:

- no-Python runtime gate.
- model acquisition gate.
- protocol gate.
- streaming tool-call gate.
- agentic workflow gate.
- no-progress gate.
- long-context gate.
- telemetry gate.
- transcript regression gate.

### No-Python Runtime Gate

Requirements:

- process tree audit shows no Python.
- loaded libraries do not include Python runtime libraries.
- engine logs show native model loading.
- no shell-out to Python.
- no Python tokenizer.
- no Python parser.
- no Python metrics exporter.

### Model Acquisition Gate

Must pass:

- native puller can dry-run a public Hugging Face repo.
- native puller can dry-run a gated/private repo with a configured token.
- dry-run reports resolved commit, file count, cached bytes, missing bytes, and total disk requirement.
- pull writes only to staging until verification succeeds.
- interrupted pull resumes without corrupting the snapshot.
- verified snapshot is immutable and loadable.
- offline boot uses an existing snapshot without network access.
- inference request fails clearly when artifacts are missing and on-demand download is disabled.
- benchmark report includes model repo, requested revision, resolved commit, manifest digest, and local snapshot path.

Required fixture repos:

- one tiny public model for CI.
- one multi-file safetensors model for shard/index behavior.
- one GGUF fixture for single-file behavior.
- one mocked gated repo response for auth/error classification.

### Protocol Gate

Must pass:

- `chat_short`
- `chat_stream`
- `multi_turn`
- `json_object`
- `tool_shape`
- `tool_required`
- `tool_required_stream`
- `tool_no_choice`
- a multi-parameter tool schema
- malformed request handling
- cancellation handling
- timeout handling

Text fallback for required tools is a hard failure.

### Streaming Tool-Call Gate

Required invariants:

- stable `index`.
- valid incremental `function.name`.
- valid incremental `function.arguments`.
- no prose fallback.
- no duplicate terminal call.
- valid JSON object arguments.
- valid finish behavior.
- exactly one `[DONE]`.
- TTFT measured from first real delta.

### Agentic Workflow Gate

Required tasks:

- `read_code_value`
- `read_code_body_value`
- `state_tracking_log`
- `bash_sum_required`
- `bash_filter_count`
- `edit_code_constant`
- `edit_json`
- `fix_test_minimal`
- `fix_test`
- `agentic_sequence`
- `repo_config_repair`
- `deep_config_repair_96k`

The validation target is final workspace state, not plausible final text.

### No-Progress Gate

The runtime and benchmark classifier must flag:

- empty assistant content with high output token count.
- repeated same tool name and similar args.
- repeated failed tool calls.
- repeated text-only "I will now do X" turns with no tool call while work remains.
- max-output stop with no useful delta.
- assistant stop while required tool/task state remains unresolved.
- no validation progress after tool results.

Initial thresholds:

- exact repeated failed tool call: 5.
- fuzzy similar failed tool call: 3.
- assistant no-progress turns with outstanding task state: 3.
- high-output empty/no-delta turn: immediate hard failure.

### Long-Context Gate

Contexts:

- 32K canary.
- 135K production minimum.
- 200K frontier.

Each context runs:

- plain recall.
- streamed recall.
- JSON mode.
- required tool.
- streaming required tool.
- multi-turn lifecycle.
- post-task recall.
- protected prompt path.

Any prompt compression or PFlash-like feature must record:

- applied or skipped.
- reason.
- protected prompt classification.
- input tokens.
- kept tokens.
- compressed/skipped tokens.
- quality check result.

### Telemetry Gate

Every request emits:

- request ID.
- model ID.
- model family.
- template ID.
- parser ID.
- prompt tokens.
- generated tokens.
- max tokens.
- queue delay.
- prefill time.
- TTFT.
- decode time.
- total elapsed.
- per-token latency distribution.
- tokens/sec.
- cache hit tokens.
- cache miss tokens.
- cache write tokens.
- cache namespace.
- KV memory active bytes.
- reusable cache bytes.
- evicted cache bytes.
- SSD cache bytes read/written.
- memory pressure.
- RSS.
- compressed memory.
- swap in/out.
- Metal allocation estimate.
- stream chunk count.
- heartbeat count.
- first-delta type.
- stop reason.
- finish reason.
- no-progress classification.
- error class.

### Transcript Regression Gate

Historical transcripts and traces should become golden inputs.

Regression classes:

- stream text fallback.
- zero-tool OMP answer.
- repeated invalid edit calls.
- required-tool miss.
- context-limit error.
- stream stall.
- parser incompatibility.
- direct API pass but OMP fail.
- empty 4096-token no-content turn.
- repeated "now writing..." no-tool turns.
- protected prompt skip.
- PFlash apply/skip behavior.

The replay suite should run without a model and verify classification logic.

## Benchmark Suites

### `rust-engine-smoke`

Purpose:

- Verify the no-Python server boots and basic protocol works.

Includes:

- process audit.
- `/v1/models`.
- `chat_short`.
- `chat_stream`.
- `json_object`.
- `tool_required`.
- one OMP read task.
- one OMP edit task.

Pass rule:

- 3 consecutive runs pass.

### `protocol-conformance`

Purpose:

- Verify OpenAI-compatible semantics.

Includes:

- streaming and non-streaming chat.
- JSON mode.
- auto tools.
- required tools.
- missing `tool_choice`.
- multiple schemas.
- malformed backend fixtures.
- heartbeat handling.
- cancellation.
- timeout.
- error body preservation.

Pass rule:

- 100%.

### `model-acquisition`

Purpose:

- Verify native Hugging Face artifact planning, download, verification, and offline loading.

Includes:

- public repo dry-run.
- authenticated/gated repo dry-run with mocked Hub responses.
- revision-to-commit resolution.
- include/exclude pattern selection.
- shard index validation.
- partial download resume.
- failed checksum quarantine.
- disk-space preflight failure.
- offline boot from a local snapshot.
- missing artifact failure when downloads are disabled.
- model manifest fields in benchmark metadata.

Pass rule:

- 100% for CI fixtures.
- no network access in offline cases.
- no token leakage in logs or trace artifacts.

### `model-family-golden`

Purpose:

- Verify tokenizer/template/parser correctness for Qwen, DeepSeek, and Gemma.

Includes:

- Qwen thinking true/false.
- Qwen historical tool calls.
- Qwen3-Coder XML.
- DeepSeek DSML.
- DeepSeek DSML thinking.
- DeepSeek legacy.
- DeepSeek V3.1 variant.
- Gemma 4 tool calls.
- Gemma 4 channels.
- streaming fragments for every parser.

Pass rule:

- byte-for-byte template fixtures.
- exact parsed tool-call structures.
- no reasoning/content leakage.

### `kernel-correctness`

Purpose:

- Verify model-family operators.

Includes:

- CPU reference comparisons.
- MLX reference comparisons where possible.
- dtype tolerances.
- shape coverage.
- long-context shape cases.
- error-path tests.

Pass rule:

- all required kernels pass for launch model family.

### `agentic-core`

Purpose:

- Verify real OMP coding behavior.

Includes:

- deterministic OMP tasks listed in the agentic workflow gate.
- transcript parsing.
- workspace validation.
- tool budget validation.

Pass rule:

- critical tasks pass 10/10 at temperature 0.
- no hard no-progress events.

### `agentic-stress`

Purpose:

- Verify long-running workflow stability.

Includes:

- `agentic_sequence`.
- `repo_config_repair`.
- `deep_config_repair_96k`.
- `deep_config_repair_160k`.
- post-task `tool_required_stream`.
- post-task JSON.
- post-task context probes.

Pass rule:

- no unclassified failure.
- no no-progress hard failure.
- no cache correctness failure.

### `long-context-matrix`

Purpose:

- Verify context behavior by length and request class.

Matrix:

- 32K
- 135K
- 200K

Request classes:

- plain recall.
- streamed recall.
- JSON.
- required tools.
- streaming required tools.
- multi-turn lifecycle.

Pass rule:

- 135K passes all classes.
- 200K is either pass or explicitly frontier with classified capacity failures.

### `transcript-regression-replay`

Purpose:

- Preserve lessons from historical incidents.

Runs:

- raw OMP JSONL transcripts.
- proxy traces.
- synthetic malformed stream fixtures.

Pass rule:

- every fixture receives the expected class.

### `soak-and-concurrency`

Purpose:

- Verify long-running service behavior.

Includes:

- 8-hour mixed load.
- short chat.
- long prompts.
- JSON.
- streaming tools.
- OMP tasks.
- cancellations.
- timeouts.

Pass rule:

- no memory leak trend.
- no stuck streams.
- no orphaned decode jobs.
- clean cancellation.
- bounded latency distribution.

## Launch Milestones

### M0: Harness Parity

Deliverables:

- Rust server skeleton.
- OpenAI type definitions.
- mocked backend.
- model acquisition manifest schema.
- mocked Hugging Face fixture server.
- report schema matching existing benchmark artifacts.
- protocol fixture tests.

Exit criteria:

- benchmark harness can run against mocked Rust server.
- all mocked protocol fixtures pass.

### M1: Native Smoke

Deliverables:

- no-Python runtime boot.
- native Hugging Face dry-run and pull for tiny CI model.
- local model store with immutable snapshots.
- one small model path.
- tokenizer in Rust.
- sampler in Rust.
- SSE in Rust.
- basic telemetry.

Exit criteria:

- `rust-engine-smoke` passes on target hardware.
- `model-acquisition` passes for CI fixtures.

### M2: Qwen Agentic Beta

Deliverables:

- Hugging Face profile for Qwen35 MLX artifacts.
- Qwen3.5/Qwen3.6 template/parser support.
- Qwen35 MLX C++ backend.
- KV cache manager.
- no-progress classifier.
- OMP transcript summaries.

Exit criteria:

- `agentic-core` passes 10/10.
- no zero-tool critical tasks.
- no hard no-progress events.

### M3: Long-Context Candidate

Deliverables:

- 135K Qwen35 production path.
- 200K frontier path.
- cache lifecycle telemetry.
- long-context matrix.

Exit criteria:

- 135K passes plain, JSON, tools, streaming tools, and lifecycle.
- 200K results are recorded with explicit pass/fail labels.

### M4: DeepSeek And Gemma Expansion

Deliverables:

- Hugging Face profiles for DeepSeek V4 and Gemma 4.
- DeepSeek DSML and legacy parsers.
- DeepSeek V4 Flash load/generate path.
- Gemma 4 parser.
- Gemma 4 text-only load/generate path.
- model-family golden suite.

Exit criteria:

- DeepSeek probes pass structured tool calls.
- Gemma probes pass structured tool calls where supported.
- no parser leaks into content.

### M5: Release Candidate

Deliverables:

- direct Metal hot kernels for measured bottlenecks.
- 8-hour soak.
- transcript regression replay.
- benchmark dashboard integration.
- production preset.

Exit criteria:

- two consecutive nightly runs pass all hard gates.
- performance is within 10% of frozen baseline or better on target model/hardware.
- any remaining failures are classified as non-launch blockers.

### M6: Native Metal Runtime Expansion

Deliverables:

- native model loader expansion.
- more direct Metal attention and MoE kernels.
- optional SSD cache tier.
- GGUF path parity.

Exit criteria:

- selected paths beat MLX bridge on speed or reliability.
- no regression in agentic gates.

## Failure Classes

The engine and benchmark harness must use stable failure classes.

Required classes:

- `http_error`
- `invalid_request`
- `unsupported_capability`
- `context_limit`
- `model_download_disabled`
- `model_artifact_missing`
- `model_not_found`
- `model_revision_unresolved`
- `model_license_unaccepted`
- `model_auth_failed`
- `model_download_interrupted`
- `model_integrity_failed`
- `model_snapshot_corrupt`
- `prefill_timeout`
- `stream_stall`
- `empty_completion`
- `empty_high_output_completion`
- `text_fallback_required_tool`
- `malformed_tool_call`
- `malformed_stream_tool_call`
- `json_mode_invalid`
- `reasoning_leak`
- `tool_markup_leak`
- `repeated_tool_loop`
- `assistant_no_progress`
- `cache_mismatch`
- `cache_unsafe_reuse`
- `memory_pressure`
- `metal_kernel_error`
- `backend_error`
- `cancelled`
- `client_disconnected`

Each failure class must have:

- stable string ID.
- human-readable explanation.
- relevant request IDs.
- relevant model IDs.
- excerpt or structured evidence.
- retryability classification.
- launch-blocker classification.

## Configuration Model

The Rust engine should support profile-like config so it can be compared against current benchmark presets.

Key config objects:

- model registry.
- model aliases.
- model source provider.
- Hugging Face repo ID.
- Hugging Face revision.
- resolved snapshot commit.
- artifact selection profile.
- download policy.
- local model store root.
- model artifact paths.
- model family.
- tokenizer path.
- chat template ID.
- parser ID.
- reasoning mode.
- context window.
- advertised context window.
- max output tokens.
- backend type.
- quantization.
- KV dtype.
- cache namespace.
- cache memory budget.
- SSD cache settings.
- heartbeat interval.
- scheduler policy.
- telemetry output paths.

Example conceptual preset:

```yaml
name: qwen35-rust-metal-200k
model:
  family: qwen
  source:
    provider: huggingface
    repo: mlx-community/Qwen3.6-35B-A3B-4bit
    revision: main
    profile: qwen35-mlx-4bit
    download: explicit
    allow:
      - "*.json"
      - "tokenizer*"
      - "*.safetensors"
      - "*.safetensors.index.json"
    ignore:
      - "*.bin"
      - "*.pt"
      - "optimizer*"
  artifact: snapshots/<resolved-commit>
  alias: local-qwen35-rust-200k
  context_window: 200000
  advertised_context_window: 135168
template:
  id: qwen3_coder_xml
  enable_thinking: false
parser:
  reasoning: qwen3
  tools: qwen3_coder_xml
backend:
  type: mlx-cpp
  direct_metal_kernels:
    - kv_page_copy
    - rope
cache:
  namespace: production-qwen35-rust-200k
  kv_dtype: q8
  memory_budget: auto
model_store:
  root: ~/.cache/llm-engine/models
  offline: false
  download_on_demand: false
  hf_cache_interop: isolated
streaming:
  heartbeat_seconds: 30
sampling:
  temperature: 0
  top_p: 1
  max_tokens: 4096
```

## Security And Safety

The server is local-first, but it still needs explicit safety boundaries.

Requirements:

- Bind to localhost by default.
- No unauthenticated non-local binding by default.
- Admin routes disabled or token-protected by default.
- Model paths must be validated and canonicalized.
- Cache paths must stay inside configured cache roots.
- Prompt and trace logging must allow redaction.
- No tool execution inside the inference server.
- No arbitrary shell execution.
- No dynamic Python import.
- No runtime downloading unless explicitly enabled.
- No Hugging Face token logging.
- No model download from non-allowlisted endpoints unless configured.
- No serving from partially downloaded artifact directories.
- No execution of remote model code by default.
- No writes outside configured model-store and staging roots.

## Observability

The Rust engine should emit both high-level and low-level telemetry.

### Request Trace

One JSON record per request:

- request metadata.
- model artifact identity.
- source repo and resolved commit.
- local snapshot path.
- model manifest digest.
- prompt metrics.
- runtime phase timings.
- cache metrics.
- stream metrics.
- tool parser metrics.
- final outcome.
- error class.

### Live Metrics

Metrics endpoint:

- active requests.
- queued requests.
- prefill requests.
- decode requests.
- request latency histograms.
- TTFT histograms.
- tokens/sec.
- cache hit ratio.
- cache memory.
- evictions.
- Metal memory estimate.
- process RSS.
- memory pressure.
- cancellations.
- no-progress failures.
- model pull operations.
- model pull bytes/sec.
- model store disk usage.
- artifact verification failures.

### Debug Artifacts

For failed benchmark requests:

- request JSON.
- rendered prompt metadata.
- token counts.
- stream event log.
- parser state transitions.
- cache decision trace.
- model manifest excerpt.
- final classification.

Artifacts must be small enough to include in bug reports without copying full proprietary prompts unless explicitly requested.

## Risks

### MLX C++ Feature Gaps

MLX Python may expose loaders/templates not available directly through C++.

Mitigation:

- keep bridge narrow.
- port only needed loaders.
- use static fixture parity.
- do not rely on Python behavior at runtime.

### FFI And Metal Lifetime Bugs

Rust safety can be lost at the C++/Metal boundary.

Mitigation:

- isolate `unsafe`.
- document safety invariants.
- no C++ references crossing async boundaries.
- use opaque handles.
- add stress tests for load/unload/cancel.

### Parser Complexity

Qwen, DeepSeek, and Gemma each have different tool and reasoning syntax.

Mitigation:

- model-family parser state machines.
- golden streaming fixtures.
- fail closed.
- never use generic parser fallback in production.

### Cache Incorrectness

Incorrect prefix reuse can produce plausible but wrong output.

Mitigation:

- include template/tool/reasoning/cache format in cache key.
- verify recurrent state compatibility.
- expose cache decisions in telemetry.
- allow cache disable to isolate bugs.

### Model Artifact Drift

Mutable Hugging Face branches can change between benchmark runs.

Mitigation:

- resolve revisions to commits during pull.
- serve only immutable local snapshots.
- include manifest digest in every benchmark report.
- fail offline boot if an alias points to a mutable remote revision without a resolved local snapshot.
- treat model revision differences as separate benchmark profiles.

### Download And Auth Complexity

Hugging Face access involves public repos, private repos, gated repos, large sharded files, and Xet-backed storage.

Mitigation:

- keep acquisition in `llm-hub`, separate from inference.
- make pull explicit by default.
- implement dry-run before download.
- support mocked Hub responses in CI.
- fail closed on ambiguous auth or license state.
- support native Xet only when it does not compromise the no-Python runtime rule.

### Optimization Regressions

PFlash-like or stream-buffering optimizations can pass recall while damaging agentic workflows.

Mitigation:

- protected prompt classification.
- apply/skip telemetry.
- tool/JSON gates.
- OMP workflow gates before promotion.

### Scope Creep Into Full ML Runtime

Writing a full native tensor runtime too early can delay the product goal.

Mitigation:

- MLX C++ bridge first.
- direct Metal only for measured hot paths.
- native runtime expansion after API/scheduler/cache correctness.

## Open Questions

These require implementation discovery, not product debate:

- Which MLX C++ APIs are stable enough for production model loading?
- How much of Qwen3.6 Gated DeltaNet state can be exposed through MLX without custom kernels?
- What is the minimum direct Metal kernel set needed to beat current `vllm-mlx` TTFT?
- Can DeepSeek V4 MXFP4 inference be implemented through MLX C++ without Python model glue?
- What tokenizer crate or binding best preserves fullwidth DeepSeek special tokens?
- Should GGUF support launch via native Rust loader or a separate milestone after MLX artifacts?
- What is the correct default KV dtype for agentic Qwen35 on M3 Ultra?
- How should cache persistence be encrypted or isolated if traces contain sensitive prompt-derived state?
- Which native Rust Hugging Face Hub client should be adopted or forked for production?
- Is direct native Xet integration stable enough, or should Xet-backed downloads start as an optional transfer backend?
- Should the model store mirror Hugging Face cache layout exactly, or use a stricter engine-owned snapshot layout with import/export compatibility?

## Reference Corpus

### Local References

The spec is grounded in the current benchmark harness and production server artifacts:

- [README.md](../README.md)
- [docs/performance-understanding.md](performance-understanding.md)
- [docs/local-llm-production-server.md](local-llm-production-server.md)
- [configs/engines.yml](../configs/engines.yml)
- [configs/experiments.yml](../configs/experiments.yml)
- [configs/production.yml](../configs/production.yml)
- [benchmarks/openai_probe.py](../benchmarks/openai_probe.py)
- [benchmarks/omp_runner.py](../benchmarks/omp_runner.py)
- [benchmarks/omp_tasks.py](../benchmarks/omp_tasks.py)
- [benchmarks/proxy.py](../benchmarks/proxy.py)
- [benchmarks/analysis.py](../benchmarks/analysis.py)
- [research-pro-3-050426.md](../research-pro-3-050426.md)

### External Model References

These model-family references should be rechecked when implementation starts, because tokenizer, template, and architecture details can change with model revisions:

- [Qwen3.5 model card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- [Qwen3.6 model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [Qwen function-calling documentation](https://qwen.readthedocs.io/en/stable/framework/function_call.html)
- [DeepSeek V4 Flash model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- [DeepSeek V3.2 model card](https://huggingface.co/deepseek-ai/DeepSeek-V3.2)
- [Gemma 4 model card](https://huggingface.co/google/gemma-4-31B-it)

### External Hub References

The Hugging Face acquisition design should be rechecked before implementation because Hub transfer behavior and Xet integration continue to evolve:

- [Hugging Face Hub download guide](https://huggingface.co/docs/huggingface_hub/guides/download)
- [Hugging Face Hub file download API reference](https://huggingface.co/docs/huggingface_hub/main/package_reference/file_download)
- [Hugging Face Hub CLI download reference](https://huggingface.co/docs/huggingface_hub/main/package_reference/cli#hf-download)

## Decision Rule

The Rust engine becomes the default local backend only when it:

- serves without Python.
- matches the current production Qwen35 OpenAI/OMP surface.
- passes direct protocol gates.
- passes streaming tool-call gates.
- passes agentic OMP gates.
- passes 135K long-context lifecycle.
- classifies no-progress cases correctly.
- emits complete telemetry.
- is at least performance-competitive with `vllm-mlx` on the same model/hardware.

Until then, it is a research and migration track. The implementation should still be built as a production system from day one, because the failure modes we care about are contract, state, and lifecycle failures that are hardest to retrofit later.
