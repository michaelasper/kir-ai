# COR-292 KV Quantization Prototype

This report covers the offline Phase 3 KV value-cache quantization prototype.
It does not enable a serving path.

Regenerate the table with:

```bash
cargo run -p llm-kv-cache --release --example kv_quantization_prototype
```

The deterministic fixtures use tiny Qwen/Gemma-shaped value rows, causal
attention output drift, and 64 decode repetitions per row. Values below are
from a local optimized run on 2026-05-24.

| fixture | scheme | bits | rotation | payload bytes | memory ratio | recon mse | attention mse | decode ops | decode ns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3-tiny | UniformAffine | 8 | false | 64 | 0.25000 | 0.00001065 | 0.00000462 | 128 | 1957 |
| qwen3-tiny | UniformAffine | 4 | false | 32 | 0.12500 | 0.00338419 | 0.00161281 | 128 | 786 |
| qwen3-tiny | UniformAffine | 3 | false | 24 | 0.09375 | 0.02147337 | 0.00901478 | 128 | 646 |
| qwen3-tiny | LloydMaxCodebook | 4 | false | 32 | 0.12500 | 0.00009137 | 0.00002719 | 64 | 1522 |
| qwen3-tiny | LloydMaxCodebook | 4 | true | 32 | 0.12500 | 0.00728668 | 0.00178102 | 1088 | 2087 |
| qwen3-tiny | LloydMaxCodebook | 3 | false | 24 | 0.09375 | 0.00047748 | 0.00026935 | 64 | 1220 |
| qwen3-tiny | LloydMaxCodebook | 3 | true | 24 | 0.09375 | 0.02152329 | 0.00758221 | 1088 | 1932 |
| gemma4-tiny | UniformAffine | 8 | false | 64 | 0.25000 | 0.00001173 | 0.00000612 | 128 | 1832 |
| gemma4-tiny | UniformAffine | 4 | false | 32 | 0.12500 | 0.00341054 | 0.00164725 | 128 | 1187 |
| gemma4-tiny | UniformAffine | 3 | false | 24 | 0.09375 | 0.02144611 | 0.00909186 | 128 | 1009 |
| gemma4-tiny | LloydMaxCodebook | 4 | false | 32 | 0.12500 | 0.00009459 | 0.00003109 | 64 | 2114 |
| gemma4-tiny | LloydMaxCodebook | 4 | true | 32 | 0.12500 | 0.00727819 | 0.00179479 | 1088 | 3247 |
| gemma4-tiny | LloydMaxCodebook | 3 | false | 24 | 0.09375 | 0.00049708 | 0.00029935 | 64 | 1940 |
| gemma4-tiny | LloydMaxCodebook | 3 | true | 24 | 0.09375 | 0.02148126 | 0.00756685 | 1088 | 3062 |

## Metadata Decision

The prototype treats rotation and Lloyd-Max codebooks as layer-head scoped
artifacts keyed by model family, layer index, head index, bit width, and a stable
fingerprint. Quantized blocks carry the model/layer/head/block identity plus the
rotation/codebook metadata fingerprint, so decode fails closed on mismatched
metadata instead of silently using the wrong transform.

Per-block metadata is still needed for block identity and payload shape, but the
rotation matrix and codebook should not be per-block unless a later benchmark
shows a clear quality gain. Per-block rotation/codebook storage would add
metadata and training overhead on the hot cache surface.

## Interpretation

Lloyd-Max/codebook quantization wins on these clustered deterministic fixtures
at equal bit width. Random orthogonal rotation is implemented and validated, but
on these fixtures it increases decode work and worsens error relative to
unrotated codebooks. These results should feed #334 as evidence to keep
rotation/codebook quantization behind offline evaluation until broader
calibration data justifies the added complexity.
