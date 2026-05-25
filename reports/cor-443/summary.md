# COR-443 hardware validation report

Result: PASS for validation evidence handoff; product/harness findings are tracked as follow-up issues.

Date: 2026-05-25
Host: macOS 26.4.1 / Darwin 25.4.0 / Apple M3 Ultra / 256 GiB RAM
Worktree: `/private/tmp/kir-ai-cor-443`
Baseline report commit: `d7d8a4d1c1719341e3f5ffc36c775e4911820e4d`

## Follow-up issues

- `COR-514`: Gemma 4 E2B MLX required-tool probes return no tool-call delta and Kir classifies the stream as `no_progress_missing_required_tool_call`.
- `COR-515`: Agentic stable-prefix generation still needs tokenizer-budget-aware sizing; the COR-443 rework lowers density and records a Qwen 256k fit check.
- `COR-516`: Native-metal Qwen 35B BF16 135k long-context validation left Kir listening but returning an empty HTTP reply after benchmark failures.

## Coverage matrix

| Target | Snapshot | Coverage | Result |
| --- | --- | --- | --- |
| Qwen 27B MLX 8-bit | `/Users/michaelasper/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-MLX-8bit/snapshots/78067073d2bf9795e5aabcfcd647bd36cf43c0b5` | Agentic harness chat/tool/8k direct probes; explicit `qwen-mlx-tool-normalized` 32k stable-prefix smoke; 256k tokenizer fit check after harness density fix | PASS for chat/tool/8k/32k warm-prefix. Original 256k profile overfilled at `426156 > 262144`; rework artifact now records `236930 < 262144` tokens. Tokenizer-budget hardening is tracked by `COR-515`. |
| Qwen 35B MLX 4-bit | `/Users/michaelasper/source/llm-server/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46` | Explicit `qwen-mlx-tool-normalized` 32k stable-prefix smoke | PASS |
| Qwen 35B native-metal BF16 | `/Users/michaelasper/source/kir-ai/.llm-models-full/huggingface/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0` via `target/cor-443/qwen35-bf16-raw-view` | `qwen-long-context` profile `qwen-135k-promotion` | FAIL finding: all 135k cases failed with HTTP request errors/timeouts and the orphaned server later returned an empty reply while still listening. Tracked by `COR-516`. |
| Gemma 4 E2B MLX 4-bit | `/Users/michaelasper/source/kir-ai/.llm-models/huggingface/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd.gemma4-e2b-it-mlx-4bit` | Direct chat, required-tool, and repeated stable-prefix probes through `mlx_vlm.server` plus Kir | Direct chat PASS. Required-tool probes FAIL with no tool-call delta and `no_progress_missing_required_tool_call`; tracked by `COR-514`. Harness no longer passes unsupported `mlx_vlm.server --prompt-cache-size`. |

## Metrics summary

| Lane | Probe | Status | First byte | First semantic/tool delta | Elapsed | Cache evidence |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Qwen 27B 8-bit | 32k warm same prompt, required tool | passed | 37 ms | 4,062 ms | 4,123 ms | Kir request cache observation `hit`; sidecar prompt cache reached 8.38 GB after measured request |
| Qwen 35B 4-bit | 32k warm same prompt, required tool | passed | 37 ms | 1,140 ms | 1,155 ms | Kir request cache observation `hit`; sidecar prompt cache reached 2.69 GB after warmup |
| Qwen 35B BF16 native-metal | 135k long-context promotion | failed | n/a | none | 900,013 ms to 1,800,033 ms per failed non-stream case | No metrics returned; admin metrics also failed after the server stopped responding |
| Gemma 4 E2B 4-bit | direct chat | passed | n/a | 1,934.641 ms in committed smoke; 158.992 ms in rerun | 2,182.376 ms in committed smoke | usage cache status `miss` |
| Gemma 4 E2B 4-bit | required tool + stable-prefix required tool | failed | n/a | none | p50 2,000.126 ms in committed smoke; p50 1,312.39 ms across the rerun | no tool calls; cache status `unknown` for tool probes |

MLX proxy lanes exposed request-cache observations, upstream stream timing, process RSS, and sidecar prompt-cache log lines. Native Metal KV precision/residency/upload metric fields were present in the `qwen-long-context` report schema, but no native metrics were returned because the 135k requests failed before usable responses.

## Commands run

- `python3 scripts/agentic_overnight_benchmark.py --dry-run --run-root target/cor-443/agentic-smoke-plan --hours 1 --context-sizes-k 8 --skip-opencode`: PASS, plan selected all required default lanes.
- `python3 scripts/agentic_overnight_benchmark.py --run-root target/cor-443/agentic-smoke-real --hours 1 --context-sizes-k 8 --skip-opencode --sidecar-ready-timeout 1800 --kir-ready-timeout 600 --direct-timeout 600 --long-direct-timeout 900`: manually stopped during the original oversized Qwen 27B 256k probe after prefill continued past 190k tokens. Before stop, Qwen 27B completed chat, required-tool, and 8k stable-prefix probes with 0 failures.
- `cargo run -p llm-bench --features bench-server -- qwen-mlx-tool-normalized ... --snapshot <qwen27> --sweep-profile qwen-mlx-stable-prefix ...`: FAIL due built-in profile sending `local-qwen36` while Kir served `local-qwen36-27b-mlx`; preserved as `qwen27-mlx-stable-prefix-32k-model-id-fail.json`.
- `cargo run -p llm-bench --features bench-server -- qwen-mlx-tool-normalized --lane name=kir-qwen27-32k,... --probe-suite stable-prefix-smoke --cache-phases warm_same_prompt --context-tokens 32768 --max-requests 2`: PASS.
- `cargo run -p llm-bench --features bench-server -- qwen-mlx-tool-normalized --lane name=kir-qwen35-32k,... --probe-suite stable-prefix-smoke --cache-phases warm_same_prompt --context-tokens 32768 --max-requests 2`: PASS.
- `uvx --from mlx-vlm mlx_vlm.server ... --prompt-cache-size 16`: FAIL, installed `mlx_vlm.server` rejected `--prompt-cache-size`; harness rework removes this flag for VLM lanes and validates future misuse.
- Gemma rerun with `mlx_vlm.server --prefill-step-size 2048` plus Kir and direct agentic probes: FAIL for required-tool and stable-prefix tool calls.
- Native Qwen server cleanup evidence: PID 5496 was `llm-engine serve` on `127.0.0.1:3000`; `curl -sS -m 5 http://127.0.0.1:3000/v1/models` returned `curl: (52) Empty reply from server`; `kill -TERM 5496` stopped the orphan and freed the port.

## Artifacts

- `reports/cor-443/qwen27-mlx-stable-prefix-32k-explicit.json`
- `reports/cor-443/qwen35-mlx-stable-prefix-32k-explicit.json`
- `reports/cor-443/qwen27-mlx-stable-prefix-32k-model-id-fail.json`
- `reports/cor-443/qwen27-agentic-256k-tokenization-after-fix.json`
- `reports/cor-443/qwen-long-context-135k-native-bf16.json`
- `reports/cor-443/agentic-smoke-real/`
- `reports/cor-443/gemma4-agentic-smoke/`

## Notes

- The Qwen 27B max-context profile overfill is no longer a COR-443 blocker because the rework reduces synthetic prefix density and records an actual Qwen tokenizer fit check. `COR-515` tracks a stronger tokenizer-budget-aware generator.
- Gemma Kir startup used `target/cor-443/gemma4-e2b-raw-view`, a manifest-free symlink view over the same snapshot files, because the provided snapshot contains `llm-engine-manifest.json` without a sha256 digest for `model.safetensors`.
- The native Qwen long-context finding is classified as follow-up product/runtime work, not a COR-443 evidence-collection blocker, because the failed run is preserved with exact host, model, request, and cleanup evidence.
