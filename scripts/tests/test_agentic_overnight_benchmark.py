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

    def test_gemma_vlm_sidecar_command_omits_unsupported_max_tokens(self):
        bench = load_module()
        gemma_lanes = [lane for lane in bench.LANES if lane.sidecar_kind == "vlm"]

        self.assertGreaterEqual(len(gemma_lanes), 2)
        for lane in gemma_lanes:
            command = bench.sidecar_command(lane, 8123)
            self.assertIn("mlx_vlm.server", command)
            self.assertIn("--prompt-cache-size", command)
            self.assertIn("--prefill-step-size", command)
            self.assertNotIn("--max-tokens", command)

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
