# COR-443 hardware validation report

Result: FAIL

Date: 2026-05-25
Host: macOS 26.4.1 / Darwin 25.4.0 / Apple M3 Ultra / 256 GiB RAM
Worktree: `/private/tmp/kir-ai-cor-443`
Commit tested: `073a72f0171967190ec0c5f69ed95f7378f817a5`

## Coverage matrix

| Target | Snapshot | Coverage | Result |
| --- | --- | --- | --- |
| Qwen 27B MLX 8-bit | `/Users/michaelasper/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-MLX-8bit/snapshots/78067073d2bf9795e5aabcfcd647bd36cf43c0b5` | Agentic harness chat/tool/8k direct probes; oversized max-context attempt; explicit `qwen-mlx-tool-normalized` 32k stable-prefix smoke | PASS for chat/tool/8k/32k warm-prefix; max-context harness generated 426,156 tokens for a 262,144-token model window and was manually stopped after continued prefill |
| Qwen 35B MLX 4-bit | `/Users/michaelasper/source/llm-server/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46` | Explicit `qwen-mlx-tool-normalized` 32k stable-prefix smoke | PASS |
| Gemma 4 E2B MLX 4-bit | `/Users/michaelasper/source/kir-ai/.llm-models/huggingface/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd.gemma4-e2b-it-mlx-4bit` | Direct chat, required-tool, and repeated 8k stable-prefix probes through `mlx_vlm.server` plus Kir | FAIL: 3/3 tool-required probes had no tool-call delta; Kir emitted `no_progress_missing_required_tool_call` |

## Metrics summary

| Lane | Probe | Status | First byte | First semantic/tool delta | Elapsed | Cache evidence |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Qwen 27B 8-bit | 32k warm same prompt, required tool | passed | 37 ms | 4,062 ms | 4,123 ms | Kir request cache observation `hit`; sidecar prompt cache reached 8.38 GB after measured request |
| Qwen 35B 4-bit | 32k warm same prompt, required tool | passed | 37 ms | 1,140 ms | 1,155 ms | Kir request cache observation `hit`; sidecar prompt cache reached 2.69 GB after warmup |
| Gemma 4 E2B 4-bit | direct chat | passed | n/a | 1,934.641 ms | 2,182.376 ms | usage cache status `miss` |
| Gemma 4 E2B 4-bit | required tool + repeated 8k stable-prefix | failed | n/a | none | p50 2,000.126 ms | no tool calls; cache status `unknown` for tool probes |

MLX proxy lanes exposed request-cache observations, upstream stream timing, process RSS, and sidecar prompt-cache log lines. Native Metal KV precision/residency/upload counters were not exposed by these MLX sidecar lanes. The native BF16 `qwen-long-context` lane was not completed in this retry because the run already produced a Gemma product/harness FAIL and the full agentic max-context profile was not bounded.

## Commands run

- `python3 scripts/agentic_overnight_benchmark.py --dry-run --run-root target/cor-443/agentic-smoke-plan --hours 1 --context-sizes-k 8 --skip-opencode`: PASS, plan selected all required default lanes.
- `python3 scripts/agentic_overnight_benchmark.py --run-root target/cor-443/agentic-smoke-real --hours 1 --context-sizes-k 8 --skip-opencode --sidecar-ready-timeout 1800 --kir-ready-timeout 600 --direct-timeout 600 --long-direct-timeout 900`: manually stopped during Qwen 27B max-context probe after the configured timeout did not interrupt `http.client` chunked reads. Before stop, Qwen 27B completed chat, required-tool, and 8k stable-prefix probes with 0 failures.
- `cargo run -p llm-bench --features bench-server -- qwen-mlx-tool-normalized ... --snapshot <qwen27> --sweep-profile qwen-mlx-stable-prefix ...`: FAIL due built-in profile sending `local-qwen36` while Kir served `local-qwen36-27b-mlx`; preserved as `qwen27-mlx-stable-prefix-32k-model-id-fail.json`.
- `cargo run -p llm-bench --features bench-server -- qwen-mlx-tool-normalized --lane name=kir-qwen27-32k,... --probe-suite stable-prefix-smoke --cache-phases warm_same_prompt --context-tokens 32768 --max-requests 2`: PASS.
- `cargo run -p llm-bench --features bench-server -- qwen-mlx-tool-normalized --lane name=kir-qwen35-32k,... --probe-suite stable-prefix-smoke --cache-phases warm_same_prompt --context-tokens 32768 --max-requests 2`: PASS.
- `uvx --from mlx-vlm mlx_vlm.server ... --prompt-cache-size 16`: FAIL, installed `mlx_vlm.server` rejected `--prompt-cache-size`.
- Gemma rerun with `mlx_vlm.server --prefill-step-size 2048 --max-tokens 2048` plus Kir and direct agentic probes: FAIL for required-tool and stable-prefix tool calls.

## Artifacts

- `reports/cor-443/qwen27-mlx-stable-prefix-32k-explicit.json`
- `reports/cor-443/qwen35-mlx-stable-prefix-32k-explicit.json`
- `reports/cor-443/qwen27-mlx-stable-prefix-32k-model-id-fail.json`
- `reports/cor-443/agentic-smoke-real/`
- `reports/cor-443/gemma4-agentic-smoke/`

## Notes

- Qwen 27B max-context profile risk: the agentic harness's `direct_stable_prefix_256k` prompt tokenized to `426156 > 262144` and kept pre-filling past 190k tokens before operator stop. This is a harness/profile sizing problem, not missing model evidence.
- Gemma Kir startup risk: the provided snapshot contains `llm-engine-manifest.json` without a sha256 digest for `model.safetensors`; Kir rejects it under fast readiness. The Gemma Kir run used `target/cor-443/gemma4-e2b-raw-view`, a manifest-free symlink view over the exact same snapshot files.
- Gemma sidecar risk: current `mlx_vlm.server` does not support the agentic harness's `--prompt-cache-size` flag.
