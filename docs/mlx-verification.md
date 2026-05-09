# MLX Verification

Use this checklist to verify that MLX-backed serving is using a real upstream
MLX server and that adaptive chat behavior is not coming from a fixture.

## Live Adaptive Check

Start the appropriate MLX server for the snapshot, then start `llm-engine`
against the same snapshot with loader `mlx`.

For Gemma 4 MLX:

```sh
python -m mlx_vlm.server \
  --model .llm-models/huggingface/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/<snapshot> \
  --host 127.0.0.1 \
  --port 8080 \
  --prefill-step-size 512

cargo run -p llm-engine -- serve \
  --snapshot .llm-models/huggingface/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/<snapshot> \
  --model-id gemma4-local \
  --addr 127.0.0.1:3000
```

In another terminal:

```sh
LLM_ENGINE_ENDPOINT=http://127.0.0.1:3000/v1/chat/completions \
LLM_ENGINE_MODEL=gemma4-local \
LLM_ENGINE_EXPECTED_ADAPTIVE_REPLY=circle/blue \
scripts/verify-mlx-live.sh
```

The verifier sends a single adaptive prompt:

```text
Remember: shape=circle, color=red. Now change color to blue. Reply exactly as shape/color.
```

The expected Gemma 4 response is `circle/blue`.

## Required Evidence

Record the following when promoting an MLX path:

- MLX package and server used, such as `mlx_vlm` for Gemma or `mlx_lm` for Qwen.
- Snapshot path, manifest digest, family, loader, profile, quantization, repo id,
  and resolved commit.
- `scripts/verify-mlx-live.sh` output, including token usage.
- `cargo test -p llm-engine mlx --lib`.
- `cargo test -p llm-runtime gemma --test runtime_contract` when validating Gemma
  chat template and parser integration.
