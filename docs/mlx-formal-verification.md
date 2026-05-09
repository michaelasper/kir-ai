# MLX Sidecar Verification

This document records the formal contract used to keep the MLX backend honest.
The MLX path is a production sidecar adapter: it sends OpenAI-compatible
requests to a loopback `mlx_lm` or `mlx_vlm` server, parses SSE responses, and
returns the same backend contract as native inference. It must not call the
protocol-test backend or return prompt-specific fixtures.

## Invariants

1. The production MLX module has no dependency on `ProtocolTestBackend`,
   `protocol_test`, or fixture-response code.
2. SSE parsing is chunk-boundary invariant: splitting an upstream MLX stream at
   any byte boundary that is a UTF-8 character boundary produces the same
   backend chunks as reading the stream in one pass.
3. Structured `tool_calls` from OpenAI-compatible MLX responses survive both
   non-streaming and streaming delta paths.
4. Control stop tokens are filtered without deleting non-stop prefixes.
5. MLX metadata validation fails closed for missing family, unknown family,
   non-MLX manifests, and non-loopback endpoints.

## Automated Evidence

Run these checks before claiming MLX correctness:

```bash
cargo test -p llm-engine mlx_ --lib
cargo test -p llm-engine mlx_reference --test mlx_reference
cargo test -p llm-runtime chat_accepts_mlx_backend --test runtime_contract
```

The bounded formal check is
`mlx_sse_parser_is_chunk_boundary_invariant_for_tool_calls`. It enumerates every
valid two-way split of a structured MLX SSE stream and proves the parser output
is identical to the unsplit stream for that fixture class.

The no-stub guard is
`mlx_production_module_does_not_depend_on_protocol_test_backend`. It statically
scans the production half of `crates/llm-engine/src/mlx.rs` and fails if the MLX
adapter starts depending on protocol-test or fixture response code.
