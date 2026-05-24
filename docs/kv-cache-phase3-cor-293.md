# COR-293 Phase 3 KV Cache Format

This report covers the first production-facing Phase 3 KV cache slice.
The feature is opt-in and does not change default serving behavior.

## Implemented Scope

- Added explicit cache format configuration for `f32`, `f16`, `int8`, and
  `asymmetric_vq`.
- Kept the default `LayerKvCache::new` path on `f32`.
- Added `LayerKvCache::new_with_config` for format selection.
- Added a fail-closed response for `f16` and `int8` CPU cache formats. Those
  formats are declared for routing and metrics compatibility but still need the
  Phase 2 compressed storage work before they can be used as CPU storage.
- Added an opt-in `asymmetric_vq` sidecar that quantizes value rows to INT4 or
  3-bit payloads with per-row asymmetric zero point and scale metadata inside
  each cache block. Keys remain in the existing f32 storage for score stability.
- Preserved the existing f32 key/value cache surface, including
  `keys()`, `values()`, `key()`, `value()`, active block views, snapshots, and
  prefix-cache reuse.
- Added format metrics for f32 resident bytes, f32/f16/int8 comparable uploaded
  bytes, Phase 3 resident bytes, Phase 3 uploaded bytes, value payload bytes,
  metadata bytes, selected Phase 3 bit width, and reconstruction error.
  Uploaded-byte comparisons count active keys and values for f32/f16/int8; the
  Phase 3 upload estimate counts FP16 keys plus quantized value payload and
  metadata.

The Phase 3 sidecar is not used by native Qwen or Gemma serving by default. It
is currently a production-owned cache representation and measurement path for
small deterministic fixtures.

## Local Fixture Report

Generated with:

```bash
cargo run -p llm-kv-cache --release --example kv_quantization_prototype
```

Run date: 2026-05-24.

| fixture | scheme | bits | rotation | payload bytes | memory ratio | recon mse | attention mse | decode ops | decode ns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3-tiny | UniformAffine | 8 | false | 64 | 0.25000 | 0.00001065 | 0.00000462 | 128 | 1988 |
| qwen3-tiny | UniformAffine | 4 | false | 32 | 0.12500 | 0.00338419 | 0.00161281 | 128 | 1199 |
| qwen3-tiny | UniformAffine | 3 | false | 24 | 0.09375 | 0.02147337 | 0.00901478 | 128 | 1011 |
| qwen3-tiny | LloydMaxCodebook | 4 | false | 32 | 0.12500 | 0.00009137 | 0.00002719 | 64 | 2300 |
| qwen3-tiny | LloydMaxCodebook | 4 | true | 32 | 0.12500 | 0.00728668 | 0.00178102 | 1088 | 3460 |
| qwen3-tiny | LloydMaxCodebook | 3 | false | 24 | 0.09375 | 0.00047748 | 0.00026935 | 64 | 1935 |
| qwen3-tiny | LloydMaxCodebook | 3 | true | 24 | 0.09375 | 0.02152329 | 0.00758221 | 1088 | 3100 |
| gemma4-tiny | UniformAffine | 8 | false | 64 | 0.25000 | 0.00001173 | 0.00000612 | 128 | 1804 |
| gemma4-tiny | UniformAffine | 4 | false | 32 | 0.12500 | 0.00341054 | 0.00164725 | 128 | 1194 |
| gemma4-tiny | UniformAffine | 3 | false | 24 | 0.09375 | 0.02144611 | 0.00909186 | 128 | 1005 |
| gemma4-tiny | LloydMaxCodebook | 4 | false | 32 | 0.12500 | 0.00009459 | 0.00003109 | 64 | 2098 |
| gemma4-tiny | LloydMaxCodebook | 4 | true | 32 | 0.12500 | 0.00727819 | 0.00179479 | 1088 | 3251 |
| gemma4-tiny | LloydMaxCodebook | 3 | false | 24 | 0.09375 | 0.00049708 | 0.00029935 | 64 | 1926 |
| gemma4-tiny | LloydMaxCodebook | 3 | true | 24 | 0.09375 | 0.02148126 | 0.00756685 | 1088 | 3069 |

The local fixture evidence still supports keeping Phase 3 off by default.
Unrotated Lloyd-Max/codebook rows produce lower drift than uniform INT4 and
3-bit rows on the clustered deterministic fixtures. Random rotation increases
decode work and worsens these fixtures, so it remains prototype evidence from
COR-292 rather than a serving default.

## Serving Benchmark Status

This patch does not claim Qwen 27B, Qwen 35B, or Gemma 4 serving throughput
improvements. The current local fixture can report memory ratio, reconstruction
drift, attention-output drift, and decode nanoseconds for tiny deterministic
rows. It does not measure TTFT or decode tokens per second because the fused
Metal value-mix path is not implemented in this slice.

## Follow-Up Work

- Implement Phase 3 Metal value payload buffers and upload accounting.
- Add fused Metal attention kernels that read quantized V payloads and
  dequantize inline during value mixing.
- Route Qwen full attention and Gemma attention cache paths through the fused
  Phase 3 kernels behind an explicit runtime option.
- Add large local benchmark profiles only after fused serving is available:
  memory, TTFT, decode tok/s, and output drift for feasible Qwen/Gemma fixtures.
- Decide whether Lloyd-Max/codebook metadata should graduate from the offline
  prototype into the serving `asymmetric_vq` format after broader calibration.
