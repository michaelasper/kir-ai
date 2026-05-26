import dataclasses
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "agentic_overnight_benchmark.py"


def load_module():
    spec = importlib.util.spec_from_file_location("agentic_overnight_benchmark", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AgenticOvernightBenchmarkTests(unittest.TestCase):
    def test_context_size_parser_dedupes_and_accepts_k_suffix(self):
        bench = load_module()

        self.assertEqual(bench.parse_context_sizes_k("8,32k,32,135"), (8, 32, 135))

    def test_lane_context_sizes_include_lane_max_and_filter_oversized_values(self):
        bench = load_module()
        gemma = next(lane for lane in bench.LANES if lane.name == "gemma4-e2b-mlx-4bit")

        self.assertEqual(bench.context_sizes_for_lane(gemma, (8, 256)), (8, 128))
        self.assertIn("direct_stable_prefix_128k", bench.direct_probe_names((8, 128)))

    def test_gemma_vlm_sidecar_command_omits_unsupported_generation_flags(self):
        bench = load_module()
        gemma_lanes = [lane for lane in bench.LANES if lane.sidecar_kind == "vlm"]

        self.assertGreaterEqual(len(gemma_lanes), 2)
        for lane in gemma_lanes:
            command = bench.sidecar_command(lane, 8123)
            self.assertIn("mlx_vlm.server", command)
            self.assertIn("--prefill-step-size", command)
            self.assertNotIn("--prompt-cache-size", command)
            self.assertNotIn("--max-tokens", command)

            invalid_lane = dataclasses.replace(lane, sidecar_extra=("--prompt-cache-size", "16"))
            with self.assertRaisesRegex(ValueError, "does not support --prompt-cache-size"):
                bench.sidecar_command(invalid_lane, 8123)

            invalid_lane = dataclasses.replace(lane, sidecar_extra=("--max-tokens", "2048"))
            with self.assertRaisesRegex(ValueError, "does not support --max-tokens"):
                bench.sidecar_command(invalid_lane, 8123)

    def test_direct_stable_prefix_probe_requests_stream_usage_and_tools(self):
        bench = load_module()

        body = bench.direct_body("local-qwen36-35b-mlx", "direct_stable_prefix_32k", repeat=7)

        self.assertTrue(body["stream"])
        self.assertEqual(body["stream_options"], {"include_usage": True})
        self.assertEqual(
            body["tool_choice"],
            {"type": "function", "function": {"name": "record_agentic_observation"}},
        )
        self.assertIn("stable-prefix-32k-shared-marker", body["messages"][1]["content"])

    def test_direct_stable_prefix_256k_stays_below_known_qwen_tokenizer_overflow_size(self):
        bench = load_module()

        body = bench.direct_body("local-qwen36-27b-mlx", "direct_stable_prefix_256k")

        self.assertLess(len(body["messages"][1]["content"]), 1_450_000)

    def test_qwen_256k_stable_prefix_uses_tokenizer_budget_guard(self):
        bench = load_module()
        qwen = next(lane for lane in bench.LANES if lane.name == "qwen27-mlx-8bit")

        def count_prompt_tokens(body):
            content = body["messages"][1]["content"]
            return 1_200 + content.count("\nsection ") * 72

        static_body = bench.direct_body(qwen.model_id, "direct_stable_prefix_256k", lane=qwen)
        budget = bench.stable_prefix_budget_for_lane(static_body, qwen)
        self.assertGreater(count_prompt_tokens(static_body), budget.prompt_budget_tokens)

        body = bench.direct_body(
            qwen.model_id,
            "direct_stable_prefix_256k",
            lane=qwen,
            token_counter=count_prompt_tokens,
        )

        prompt_tokens = count_prompt_tokens(body)
        self.assertGreater(body["messages"][1]["content"].count("\nsection "), 0)
        self.assertLess(
            body["messages"][1]["content"].count("\nsection "),
            static_body["messages"][1]["content"].count("\nsection "),
        )
        self.assertLessEqual(prompt_tokens, budget.prompt_budget_tokens)
        self.assertLessEqual(
            prompt_tokens + body["max_tokens"] + budget.guard_tokens,
            budget.context_window_tokens,
        )
        self.assertGreaterEqual(budget.guard_tokens, 2048)

    def test_gemma_128k_stable_prefix_uses_tokenizer_budget_guard(self):
        bench = load_module()
        gemma = next(lane for lane in bench.LANES if lane.name == "gemma4-e2b-mlx-4bit")

        def count_prompt_tokens(body):
            content = body["messages"][1]["content"]
            return 900 + content.count("\nsection ") * 70

        static_body = bench.direct_body(gemma.model_id, "direct_stable_prefix_128k", lane=gemma)
        budget = bench.stable_prefix_budget_for_lane(static_body, gemma)
        self.assertGreater(count_prompt_tokens(static_body), budget.prompt_budget_tokens)

        body = bench.direct_body(
            gemma.model_id,
            "direct_stable_prefix_128k",
            lane=gemma,
            token_counter=count_prompt_tokens,
        )

        prompt_tokens = count_prompt_tokens(body)
        self.assertGreater(body["messages"][1]["content"].count("\nsection "), 0)
        self.assertLess(
            body["messages"][1]["content"].count("\nsection "),
            static_body["messages"][1]["content"].count("\nsection "),
        )
        self.assertLessEqual(prompt_tokens, budget.prompt_budget_tokens)
        self.assertLessEqual(
            prompt_tokens + body["max_tokens"] + budget.guard_tokens,
            budget.context_window_tokens,
        )
        self.assertGreaterEqual(budget.guard_tokens, 2048)

    def test_oversized_stable_prefix_fails_before_streaming_request(self):
        bench = load_module()
        qwen = next(lane for lane in bench.LANES if lane.name == "qwen27-mlx-8bit")

        def oversized_prompt_tokens(_body):
            return qwen.max_context_k * 1024

        with tempfile.TemporaryDirectory() as tmp:
            run_root = pathlib.Path(tmp)
            lane_dir = run_root / qwen.name
            calls = []

            def record_admin_call(*_args, **_kwargs):
                calls.append("admin")
                return None

            def record_stream_call(*_args, **_kwargs):
                calls.append("stream")
                return {}

            original_fetch_admin = bench.fetch_admin_metrics
            original_stream = bench.stream_chat_completion
            bench.fetch_admin_metrics = record_admin_call
            bench.stream_chat_completion = record_stream_call
            try:
                with self.assertRaisesRegex(bench.StablePrefixBudgetError, "exceeds"):
                    bench.run_direct_probe(
                        run_root,
                        lane_dir,
                        "http://127.0.0.1:9",
                        qwen.model_id,
                        "direct_stable_prefix_256k",
                        1,
                        0,
                        1,
                        lane=qwen,
                        token_counter=oversized_prompt_tokens,
                    )
            finally:
                bench.fetch_admin_metrics = original_fetch_admin
                bench.stream_chat_completion = original_stream

            self.assertEqual(calls, [])

    def test_tool_fragment_diagnostics_assembles_streamed_arguments(self):
        bench = load_module()

        diagnostics = bench.tool_call_diagnostics_from_fragments(
            [
                [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "record_agentic_observation",
                            "arguments": '{"task":"agent',
                        },
                    }
                ],
                [{"index": 0, "function": {"arguments": 'ic","risk":"tool"}'}}],
            ]
        )

        self.assertEqual(diagnostics["observed"], 1)
        self.assertEqual(diagnostics["valid_json_arguments"], 1)
        self.assertEqual(diagnostics["invalid_json_arguments"], 0)
        self.assertEqual(diagnostics["names"], ["record_agentic_observation"])

    def test_sse_error_events_are_recorded_with_stable_metadata(self):
        bench = load_module()
        result = {"errors": []}

        handled = bench.record_sse_error_event(
            result,
            {
                "error": {
                    "message": "invalid request: MLX Gemma required tool_choice is not supported for model `local-gemma4-e2b`",
                    "code": "invalid_request",
                    "phase": "request_validation",
                    "retryable": False,
                    "type": "llm_engine_error",
                }
            },
        )

        self.assertTrue(handled)
        self.assertEqual(
            result["errors"],
            [
                {
                    "kind": "sse_error",
                    "message": "invalid request: MLX Gemma required tool_choice is not supported for model `local-gemma4-e2b`",
                    "code": "invalid_request",
                    "phase": "request_validation",
                    "retryable": False,
                    "type": "llm_engine_error",
                }
            ],
        )

    def test_opencode_command_diagnostics_counts_parseable_shell_commands(self):
        bench = load_module()

        diagnostics = bench.command_diagnostics_from_stdout(
            "\n".join(
                [
                    json.dumps({"type": "tool_call", "name": "bash", "command": "python -m unittest"}),
                    json.dumps({"type": "tool_call", "name": "bash", "command": "python -c 'unterminated"}),
                    json.dumps({"type": "message", "text": "not a command"}),
                ]
            )
        )

        self.assertEqual(diagnostics["observed"], 2)
        self.assertEqual(diagnostics["syntax_valid"], 1)
        self.assertEqual(diagnostics["syntax_invalid"], 1)
        self.assertEqual(diagnostics["syntax_success_rate"], 0.5)

    def test_summary_separates_speed_quality_and_failure_modes(self):
        bench = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            run_root = pathlib.Path(tmp)
            lane_dir = run_root / "qwen35-mlx-4bit"
            lane_dir.mkdir()
            samples = [
                {
                    "kind": "direct",
                    "probe": "direct_tool_required_stream",
                    "model_id": "local-qwen36-35b-mlx",
                    "result": {
                        "latency_ms": 120.0,
                        "finish_reasons": ["stop"],
                        "tool_calls": {
                            "observed": 0,
                            "valid_json_arguments": 0,
                            "invalid_json_arguments": 0,
                            "names": [],
                        },
                        "errors": [],
                    },
                },
                {
                    "kind": "opencode",
                    "task": "opencode_seeded_bugfix",
                    "model_id": "local-qwen36-35b-mlx",
                    "proc": {
                        "latency_ms": 240.0,
                        "exit_code": 0,
                        "timed_out": False,
                        "command_diagnostics": {
                            "observed": 1,
                            "syntax_valid": 1,
                            "syntax_invalid": 0,
                        },
                    },
                    "judge": {
                        "passed": False,
                        "checks": {
                            "unittest_exit_zero": False,
                            "readme": True,
                        },
                    },
                },
            ]
            (lane_dir / "samples.jsonl").write_text(
                "\n".join(json.dumps(sample) for sample in samples) + "\n",
                encoding="utf-8",
            )

            summary = bench.summarize_run(run_root)

        bucket = summary["by_model"]["local-qwen36-35b-mlx"]
        self.assertEqual(bucket["model_identity"]["lane"], "qwen35-mlx-4bit")
        self.assertEqual(bucket["model_identity"]["quantization"], "4bit")
        self.assertEqual(bucket["agentic_quality"]["opencode_pass_rate"], 0.0)
        self.assertEqual(bucket["agentic_quality"]["direct_tool_call_success_rate"], 0.0)
        self.assertEqual(bucket["agentic_quality"]["command_syntax_success_rate"], 1.0)
        self.assertEqual(bucket["failure_modes"]["tool_use"], 1)
        self.assertEqual(bucket["failure_modes"]["task_correctness"], 1)
        self.assertEqual(bucket["speed_quality"]["quality_score"], 0.0)
        self.assertEqual(bucket["speed_quality"]["throughput_quality_score"], 0.0)
        self.assertEqual(bucket["speed_quality"]["classification"], "fast_but_low_quality")

    def test_dry_run_writes_plan_without_requiring_snapshot_or_opencode(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = os.environ.copy()
            env["KIR_BENCH_QWEN35_SNAPSHOT"] = "/tmp/nonexistent-qwen35-snapshot"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--dry-run",
                    "--run-root",
                    tmp,
                    "--hours",
                    "0.1",
                    "--only",
                    "qwen35-mlx-4bit",
                    "--context-sizes-k",
                    "8,32",
                ],
                cwd=REPO_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((pathlib.Path(tmp) / "manifest.json").read_text())
            self.assertTrue(manifest["dry_run"])
            self.assertEqual(manifest["lanes"][0]["snapshot"], "/tmp/nonexistent-qwen35-snapshot")
            self.assertFalse(manifest["lanes"][0]["snapshot_exists"])
            self.assertEqual(
                manifest["workloads"]["per_lane"]["qwen35-mlx-4bit"]["context_sizes_k"],
                [8, 32, 256],
            )


if __name__ == "__main__":
    unittest.main()
