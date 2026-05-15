# Agentic Overnight Benchmark

Use `scripts/agentic_overnight_benchmark.py` for long, externally managed
agentic runs that are too broad for the focused Rust benchmark profiles. Unlike
`llm-engine bench qwen-mlx-tool-normalized`, this script launches one MLX
sidecar and one Kir proxy per lane, rotates models sequentially to avoid memory
contention, and writes artifacts under `target/agentic-bench-runs`.

The default eight-hour run covers:

- Qwen3.6 27B MLX 8-bit.
- Qwen3.6 35B A3B MLX 4-bit.
- Gemma 4 E2B MLX 4-bit.
- Optional Gemma 4 31B MLX with `--include-heavy-gemma31`.
- Direct streaming probes for chat, required tools, and stable-prefix context
  sizes up to each lane's declared maximum.
- Opencode tasks for canary CLI creation, Snake from scratch, Snake
  enhancement, seeded bug fixes, long-context rule discovery, and recovery
  debugging.

Run a plan-only check first:

```sh
python3 scripts/agentic_overnight_benchmark.py \
  --dry-run \
  --run-root target/agentic-bench-runs/overnight-plan \
  --context-sizes-k 8,32,64,96,135,200,256
```

Run the overnight benchmark:

```sh
python3 scripts/agentic_overnight_benchmark.py \
  --hours 8 \
  --context-sizes-k 8,32,64,96,135,200,256
```

Snapshot paths default to the local cache paths used on the benchmark machine.
Override them when running elsewhere:

```sh
KIR_BENCH_QWEN27_SNAPSHOT=/models/qwen27 \
KIR_BENCH_QWEN35_SNAPSHOT=/models/qwen35 \
KIR_BENCH_GEMMA4_E2B_SNAPSHOT=/models/gemma4-e2b \
python3 scripts/agentic_overnight_benchmark.py --hours 8
```

The script records `manifest.json`, per-lane sidecar/Kir logs, direct SSE
traces, opencode JSONL/stdout/stderr, per-task judges, admin metrics before and
after each sample, `samples.jsonl`, and a top-level `summary.json`.

All built-in MLX lanes launch their sidecars with prompt-cache capacity. Qwen
lanes use MLX-LM with `--prompt-cache-size 16`, no-thinking chat-template args,
and `--prefill-step-size 2048`; Gemma VLM lanes use `--prompt-cache-size 16`
and `--prefill-step-size 2048`. The benchmark does not inject request-level
cache/session fields because the MLX sidecar API does not advertise a stable
cache-key contract. Cache evidence comes from upstream usage when present and
from Kir `/admin/metrics.request_cache` observations.

Key fields to review after a run:

- `by_model[*].first_semantic_delta_ms` for chat TTFI.
- `by_model[*].first_tool_delta_ms` for required-tool first event latency.
- `by_model[*].latency_ms` for wall-clock task/probe time.
- `by_model[*].opencode_passed` and `failures` for real task quality.
- `by_model[*].cache_status_counts` plus per-task admin captures for prefix
  cache observability.

Useful focused variants:

```sh
# Synthetic cache/context sweep only.
python3 scripts/agentic_overnight_benchmark.py --hours 2 --skip-opencode

# Opencode-only quality and reliability pass.
python3 scripts/agentic_overnight_benchmark.py --hours 2 --skip-direct

# One model while debugging harness setup.
python3 scripts/agentic_overnight_benchmark.py --hours 1 --only qwen35-mlx-4bit
```
