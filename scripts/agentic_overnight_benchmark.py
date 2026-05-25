#!/usr/bin/env python3
"""Long-running kir-ai agentic benchmark harness.

The harness is intentionally self-contained and writes all artifacts under an
ignored run directory. It rotates one model at a time so large MLX snapshots do
not compete for unified memory.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import http.client
import json
import os
import pathlib
import random
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import textwrap
import time
import traceback
import urllib.parse
import urllib.request
from typing import Any


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
HF_HOME = pathlib.Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
DEFAULT_RUN_ROOT = REPO_ROOT / "target" / "agentic-bench-runs"
DEFAULT_CONTEXT_SIZES_K = (8, 32, 64, 96, 135, 200, 256)
DEFAULT_SEED = 264259
STABLE_PREFIX_SECTIONS_PER_K = 20


def env_path(name: str, fallback: pathlib.Path) -> pathlib.Path:
    value = os.environ.get(name)
    return pathlib.Path(value).expanduser() if value else fallback


@dataclasses.dataclass(frozen=True)
class ModelLane:
    name: str
    family: str
    model_id: str
    quantization: str
    snapshot: pathlib.Path
    max_context_k: int
    sidecar_package: str
    sidecar_module: str
    sidecar_kind: str
    sidecar_extra: tuple[str, ...] = ()
    include_by_default: bool = True
    snapshot_env: str = ""
    note: str = ""


LANES: list[ModelLane] = [
    ModelLane(
        name="qwen27-mlx-8bit",
        family="qwen",
        model_id="local-qwen36-27b-mlx",
        quantization="8bit",
        snapshot=env_path(
            "KIR_BENCH_QWEN27_SNAPSHOT",
            HF_HOME
            / "models--unsloth--Qwen3.6-27B-MLX-8bit"
            / "snapshots"
            / "78067073d2bf9795e5aabcfcd647bd36cf43c0b5",
        ),
        max_context_k=256,
        sidecar_package="mlx-lm",
        sidecar_module="mlx_lm.server",
        sidecar_kind="lm",
        sidecar_extra=(
            "--chat-template-args",
            '{"enable_thinking":false}',
            "--max-tokens",
            "2048",
            "--prompt-cache-size",
            "16",
            "--prefill-step-size",
            "2048",
        ),
        snapshot_env="KIR_BENCH_QWEN27_SNAPSHOT",
        note="Qwen 27B MLX 8-bit, local HF cache",
    ),
    ModelLane(
        name="qwen35-mlx-4bit",
        family="qwen",
        model_id="local-qwen36-35b-mlx",
        quantization="4bit",
        snapshot=env_path(
            "KIR_BENCH_QWEN35_SNAPSHOT",
            HF_HOME
            / "models--mlx-community--Qwen3.6-35B-A3B-4bit"
            / "snapshots"
            / "38740b847e4cb78f352aba30aa41c76e08e6eb46",
        ),
        max_context_k=256,
        sidecar_package="mlx-lm",
        sidecar_module="mlx_lm.server",
        sidecar_kind="lm",
        sidecar_extra=(
            "--chat-template-args",
            '{"enable_thinking":false}',
            "--max-tokens",
            "2048",
            "--prompt-cache-size",
            "16",
            "--prefill-step-size",
            "2048",
        ),
        snapshot_env="KIR_BENCH_QWEN35_SNAPSHOT",
        note="Qwen 35B/A3B MLX 4-bit, local HF cache",
    ),
    ModelLane(
        name="gemma4-e2b-mlx-4bit",
        family="gemma",
        model_id="local-gemma4-e2b",
        quantization="4bit",
        snapshot=env_path(
            "KIR_BENCH_GEMMA4_E2B_SNAPSHOT",
            REPO_ROOT
            / ".llm-models"
            / "huggingface"
            / "models--mlx-community--gemma-4-e2b-it-4bit"
            / "snapshots"
            / "99d9a53ff828d365a8ecae538e45f80a08d612cd.gemma4-e2b-it-mlx-4bit",
        ),
        max_context_k=128,
        sidecar_package="mlx-vlm",
        sidecar_module="mlx_vlm.server",
        sidecar_kind="vlm",
        sidecar_extra=("--prefill-step-size", "2048"),
        snapshot_env="KIR_BENCH_GEMMA4_E2B_SNAPSHOT",
        note="Gemma 4 E2B MLX 4-bit, practical repeated-workload lane",
    ),
    ModelLane(
        name="gemma4-31b-mlx",
        family="gemma",
        model_id="local-gemma4-31b",
        quantization="unknown",
        snapshot=env_path(
            "KIR_BENCH_GEMMA4_31B_SNAPSHOT",
            HF_HOME
            / "models--prithivMLmods--gemma-4-31B-it-Uncensored-MAX-MLX"
            / "snapshots"
            / "6933ad89545897d5ab4289ac95b03cee5b87c1d7",
        ),
        max_context_k=128,
        sidecar_package="mlx-vlm",
        sidecar_module="mlx_vlm.server",
        sidecar_kind="vlm",
        sidecar_extra=("--prefill-step-size", "2048"),
        include_by_default=False,
        snapshot_env="KIR_BENCH_GEMMA4_31B_SNAPSHOT",
        note="Optional heavy Gemma 4 31B lane; enable with --include-heavy-gemma31",
    ),
]


DIRECT_CANARY_PROBES = (
    "direct_chat_stream",
    "direct_tool_required_stream",
)

OPENCODE_TASKS = (
    "opencode_canary_cli",
    "opencode_snake_from_scratch",
    "opencode_snake_enhancement",
    "opencode_seeded_bugfix",
    "opencode_long_context_rules",
    "opencode_recovery_debug",
)


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def log(run_root: pathlib.Path, message: str) -> None:
    line = f"{utc_now()} {message}"
    print(line, flush=True)
    with (run_root / "runner.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, sort_keys=True) + "\n")


def shell_quote(args: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([arg]) for arg in args)


def parse_context_sizes_k(value: str) -> tuple[int, ...]:
    sizes: list[int] = []
    for part in value.split(","):
        part = part.strip().lower().removesuffix("k")
        if not part:
            continue
        size = int(part)
        if size <= 0:
            raise ValueError("context sizes must be greater than zero")
        sizes.append(size)
    if not sizes:
        raise ValueError("at least one context size is required")
    return tuple(dict.fromkeys(sizes))


def context_sizes_for_lane(lane: ModelLane, requested: tuple[int, ...]) -> tuple[int, ...]:
    sizes = [size for size in requested if size <= lane.max_context_k]
    if lane.max_context_k not in sizes:
        sizes.append(lane.max_context_k)
    return tuple(dict.fromkeys(sizes))


def direct_probe_names(context_sizes_k: tuple[int, ...]) -> tuple[str, ...]:
    return DIRECT_CANARY_PROBES + tuple(
        f"direct_stable_prefix_{size}k" for size in context_sizes_k
    )


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def process_alive(proc: subprocess.Popen[Any] | None) -> bool:
    return proc is not None and proc.poll() is None


def stop_process(proc: subprocess.Popen[Any] | None, name: str, run_root: pathlib.Path) -> None:
    if not process_alive(proc):
        return
    assert proc is not None
    log(run_root, f"stopping {name} pid={proc.pid}")
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        pass
    log(run_root, f"force killing {name} pid={proc.pid}")
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=10)


def start_logged_process(
    cmd: list[str],
    log_path: pathlib.Path,
    cwd: pathlib.Path,
    env: dict[str, str] | None = None,
) -> subprocess.Popen[Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = log_path.open("ab", buffering=0)
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    return subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=full_env,
        stdout=f,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )


def http_json(
    method: str,
    url: str,
    body: dict[str, Any] | None = None,
    timeout: float = 30,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], Any]:
    data = None
    merged_headers = {"Content-Type": "application/json"}
    if headers:
        merged_headers.update(headers)
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method, headers=merged_headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            text = raw.decode("utf-8", errors="replace")
            try:
                parsed: Any = json.loads(text) if text else None
            except json.JSONDecodeError:
                parsed = text
            return resp.status, dict(resp.headers.items()), parsed
    except urllib.error.HTTPError as err:
        raw = err.read()
        text = raw.decode("utf-8", errors="replace")
        try:
            parsed = json.loads(text) if text else None
        except json.JSONDecodeError:
            parsed = text
        return err.code, dict(err.headers.items()), parsed


def wait_for_endpoint(
    url: str,
    timeout_sec: float,
    run_root: pathlib.Path,
    label: str,
) -> tuple[bool, Any]:
    deadline = time.monotonic() + timeout_sec
    last_error: Any = None
    while time.monotonic() < deadline:
        try:
            status, _headers, body = http_json("GET", url, timeout=10)
            if 200 <= status < 300:
                return True, body
            last_error = {"status": status, "body": body}
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
        time.sleep(5)
    log(run_root, f"{label} did not become ready: {last_error}")
    return False, last_error


def fetch_admin_metrics(base_url: str, out_path: pathlib.Path) -> dict[str, Any] | None:
    url = base_url.rstrip("/") + "/admin/metrics"
    try:
        status, headers, body = http_json("GET", url, timeout=10)
        payload = {"status": status, "headers": headers, "body": body, "captured_at": utc_now()}
        write_json(out_path, payload)
        return payload
    except Exception as exc:  # noqa: BLE001
        payload = {"error": repr(exc), "captured_at": utc_now()}
        write_json(out_path, payload)
        return None


def stream_chat_completion(
    base_v1: str,
    body: dict[str, Any],
    out_path: pathlib.Path,
    timeout: float,
) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(base_v1.rstrip("/") + "/chat/completions")
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    request_body = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": "Bearer dummy"}
    started = time.monotonic()
    result: dict[str, Any] = {
        "started_at": utc_now(),
        "ttfb_ms": None,
        "first_semantic_delta_ms": None,
        "first_tool_delta_ms": None,
        "latency_ms": None,
        "status": None,
        "response_headers": {},
        "request_id": None,
        "usage": None,
        "cache_status": "unknown",
        "chunks": 0,
        "finish_reasons": [],
        "errors": [],
    }
    raw_lines: list[str] = []
    text_fragments: list[str] = []
    tool_fragments: list[Any] = []
    try:
        conn.request("POST", parsed.path, body=request_body, headers=headers)
        resp = conn.getresponse()
        result["status"] = resp.status
        result["response_headers"] = dict(resp.getheaders())
        result["request_id"] = result["response_headers"].get("x-request-id")
        first_byte = None
        while True:
            line_bytes = resp.readline()
            if not line_bytes:
                break
            now = time.monotonic()
            if first_byte is None:
                first_byte = now
                result["ttfb_ms"] = round((now - started) * 1000, 3)
            line = line_bytes.decode("utf-8", errors="replace").rstrip("\r\n")
            raw_lines.append(line)
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                result["errors"].append({"kind": "json_decode", "line": line[:500]})
                continue
            result["chunks"] += 1
            if record_sse_error_event(result, event):
                continue
            if event.get("usage"):
                result["usage"] = event["usage"]
            for choice in event.get("choices", []):
                finish = choice.get("finish_reason")
                if finish:
                    result["finish_reasons"].append(finish)
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    text_fragments.append(content)
                    if result["first_semantic_delta_ms"] is None:
                        result["first_semantic_delta_ms"] = round((now - started) * 1000, 3)
                tool_calls = delta.get("tool_calls") or []
                if tool_calls:
                    tool_fragments.append(tool_calls)
                    if result["first_tool_delta_ms"] is None:
                        result["first_tool_delta_ms"] = round((now - started) * 1000, 3)
        result["latency_ms"] = round((time.monotonic() - started) * 1000, 3)
    except Exception as exc:  # noqa: BLE001
        result["latency_ms"] = round((time.monotonic() - started) * 1000, 3)
        result["errors"].append({"kind": type(exc).__name__, "message": repr(exc)})
    finally:
        conn.close()
    result["text"] = "".join(text_fragments)
    result["tool_fragments_count"] = len(tool_fragments)
    result["tool_calls"] = tool_call_diagnostics_from_fragments(tool_fragments)
    result["prompt_tokens"] = usage_value(result.get("usage"), "prompt_tokens")
    result["completion_tokens"] = usage_value(result.get("usage"), "completion_tokens")
    result["cached_tokens"] = cached_tokens_from_usage(result.get("usage"))
    if result["cached_tokens"] is not None and result["prompt_tokens"] is not None:
        result["uncached_tokens"] = max(0, result["prompt_tokens"] - result["cached_tokens"])
        result["cache_status"] = classify_cache_status(result["prompt_tokens"], result["cached_tokens"])
    write_json(out_path.with_suffix(".summary.json"), result)
    out_path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    return result


def usage_value(usage: Any, key: str) -> int | None:
    if not isinstance(usage, dict):
        return None
    value = usage.get(key)
    return value if isinstance(value, int) else None


def cached_tokens_from_usage(usage: Any) -> int | None:
    if not isinstance(usage, dict):
        return None
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        return None
    value = details.get("cached_tokens")
    return value if isinstance(value, int) else None


def classify_cache_status(prompt_tokens: int, cached_tokens: int | None) -> str:
    if cached_tokens is None:
        return "unknown"
    if cached_tokens == 0:
        return "miss"
    if cached_tokens >= prompt_tokens:
        return "hit"
    return "partial"


def rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 3)


def tool_call_diagnostics_from_fragments(tool_fragments: list[Any]) -> dict[str, Any]:
    calls: dict[str, dict[str, str | None]] = {}
    order: list[str] = []
    for fragment_batch in tool_fragments:
        if not isinstance(fragment_batch, list):
            continue
        for fragment in fragment_batch:
            if not isinstance(fragment, dict):
                continue
            index = fragment.get("index")
            key_value = f"index:{index}" if isinstance(index, int) else fragment.get("id")
            if not isinstance(key_value, str):
                key_value = f"call:{len(order)}"
            if key_value not in calls:
                calls[key_value] = {"name": "", "arguments": "", "id": None}
                order.append(key_value)
            call = calls[key_value]
            call_id = fragment.get("id")
            if isinstance(call_id, str):
                call["id"] = call_id
            function = fragment.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if isinstance(name, str):
                call["name"] = str(call["name"] or "") + name
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                call["arguments"] = str(call["arguments"] or "") + arguments

    names: list[str] = []
    valid_json_arguments = 0
    invalid_json_arguments = 0
    for key in order:
        call = calls[key]
        name = call.get("name")
        if isinstance(name, str) and name:
            names.append(name)
        arguments = call.get("arguments")
        arguments_text = arguments.strip() if isinstance(arguments, str) else ""
        if not arguments_text:
            invalid_json_arguments += 1
            continue
        with contextlib.suppress(json.JSONDecodeError):
            parsed = json.loads(arguments_text)
            if isinstance(parsed, dict):
                valid_json_arguments += 1
                continue
        invalid_json_arguments += 1

    return {
        "observed": len(order),
        "valid_json_arguments": valid_json_arguments,
        "invalid_json_arguments": invalid_json_arguments,
        "names": names,
    }


def record_sse_error_event(result: dict[str, Any], event: dict[str, Any]) -> bool:
    error = event.get("error")
    if not isinstance(error, dict):
        return False
    result.setdefault("errors", []).append(
        {
            "kind": "sse_error",
            "message": error.get("message"),
            "code": error.get("code"),
            "phase": error.get("phase"),
            "retryable": error.get("retryable"),
            "type": error.get("type"),
        }
    )
    return True


def command_strings_from_event(event: Any) -> list[str]:
    commands: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                key_lower = str(key).lower()
                if key_lower in {"command", "cmd", "shell_command"} and isinstance(child, str):
                    commands.append(child)
                else:
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(event)
    return commands


def command_diagnostics_from_stdout(stdout: str) -> dict[str, Any]:
    observed = 0
    syntax_valid = 0
    syntax_invalid = 0
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        for command in command_strings_from_event(event):
            observed += 1
            try:
                if shlex.split(command):
                    syntax_valid += 1
                else:
                    syntax_invalid += 1
            except ValueError:
                syntax_invalid += 1
    return {
        "observed": observed,
        "syntax_valid": syntax_valid,
        "syntax_invalid": syntax_invalid,
        "syntax_success_rate": rate(syntax_valid, observed),
    }


def tool_schema() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "record_agentic_observation",
                "description": "Record a structured benchmark observation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string"},
                        "risk": {"type": "string"},
                        "next_step": {"type": "string"},
                        "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
                    },
                    "required": ["task", "risk", "next_step", "confidence"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def long_context_text(target_k: int, marker: str) -> str:
    paragraphs: list[str] = [
        f"BEGIN_MARKER {marker}",
        "You are reading a synthetic repository investigation transcript.",
    ]
    seed = [
        "cache identity",
        "tool call validation",
        "streaming semantic delta",
        "long running coding task",
        "context compaction",
        "prefill scheduling",
        "agent recovery",
        "unpredictable workflow",
        "stable prefix",
        "request cache observation",
    ]
    # The name is an approximate target in K prompt tokens. Keep the density
    # conservative so max-context lanes do not exceed model windows.
    for i in range(target_k * STABLE_PREFIX_SECTIONS_PER_K):
        words = " ".join(seed[(i + j) % len(seed)] for j in range(9))
        paragraphs.append(
            f"section {i:05d}: {words}. Preserve the first marker and ignore distractor {i % 97}."
        )
    paragraphs.append(f"FINAL_REQUIRED_MARKER {marker}")
    return "\n".join(paragraphs)


def direct_body(model_id: str, probe: str, repeat: int = 0) -> dict[str, Any]:
    common: dict[str, Any] = {
        "model": model_id,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
        "max_tokens": 512,
    }
    if probe == "direct_chat_stream":
        return {
            **common,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a fast benchmark canary. Keep responses short and concrete.",
                },
                {
                    "role": "user",
                    "content": "In exactly two sentences, name one likely agentic-engine bottleneck and one metric that detects it.",
                },
            ],
        }
    if probe == "direct_tool_required_stream":
        return {
            **common,
            "messages": [
                {
                    "role": "system",
                    "content": "When a tool is available and required, call it with valid JSON arguments.",
                },
                {
                    "role": "user",
                    "content": "Record that malformed tool calls and first-tool latency are risks for coding agents.",
                },
            ],
            "tools": tool_schema(),
            "tool_choice": {
                "type": "function",
                "function": {"name": "record_agentic_observation"},
            },
        }
    if probe.startswith("direct_stable_prefix_"):
        size = int(probe.removeprefix("direct_stable_prefix_").removesuffix("k"))
        marker = f"stable-prefix-{size}k-run-{repeat}"
        prefix_marker = f"stable-prefix-{size}k-shared-marker"
        prompt = long_context_text(size, prefix_marker)
        return {
            **common,
            "max_tokens": 384,
            "messages": [
                {
                    "role": "system",
                    "content": "You are measuring long-context recall and cache behavior. Answer compactly.",
                },
                {
                    "role": "user",
                    "content": (
                        prompt
                        + "\n\n"
                        + f"Suffix run id: {marker}. Use the tool to report the shared marker and one cache-sensitive optimization idea."
                    ),
                },
            ],
            "tools": tool_schema(),
            "tool_choice": {
                "type": "function",
                "function": {"name": "record_agentic_observation"},
            },
        }
    raise ValueError(f"unknown direct probe {probe}")


def create_opencode_config(
    config_dir: pathlib.Path,
    base_v1: str,
    model_id: str,
    context_limit: int,
) -> None:
    opencode_dir = config_dir / "opencode"
    opencode_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "$schema": "https://opencode.ai/config.json",
        "model": f"kir/{model_id}",
        "small_model": f"kir/{model_id}",
        "provider": {
            "kir": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "kir-ai local",
                "options": {
                    "baseURL": base_v1,
                    "apiKey": "dummy",
                    "timeout": 28800000,
                    "chunkTimeout": 600000,
                },
                "models": {
                    model_id: {
                        "name": model_id,
                        "limit": {"context": context_limit, "output": 8192},
                    }
                },
            }
        },
        "autoupdate": False,
        "disabled_providers": [
            "anthropic",
            "openai",
            "gemini",
            "openrouter",
            "zai-coding-plan",
            "opencode",
            "opencode-go",
            "huggingface",
        ],
    }
    write_json(opencode_dir / "opencode.json", config)


def seed_workspace(task: str, work: pathlib.Path) -> str:
    work.mkdir(parents=True, exist_ok=True)
    if task == "opencode_canary_cli":
        return textwrap.dedent(
            """
            Create a tiny Python CLI in this empty workspace.
            Requirements:
            - Write `agent_canary.py` with a `main()` function.
            - It should print `agentic canary ok` when run.
            - Add `README.md` with one command to run it.
            Do not read or write outside this workspace.
            """
        ).strip()
    if task == "opencode_snake_from_scratch":
        return textwrap.dedent(
            """
            Build a browser Snake game in this empty workspace.
            Requirements:
            - Create `index.html`, `game.js`, and `README.md`.
            - The snake moves with arrow keys and WASD.
            - Food spawns, score increments, speed ramps up, and game over is visible.
            - Add pause/resume and restart controls.
            - Keep it dependency-free so opening `index.html` works.
            Do not read or write outside this workspace.
            """
        ).strip()
    if task == "opencode_snake_enhancement":
        (work / "index.html").write_text(
            "<!doctype html><canvas id='game' width='400' height='400'></canvas><script src='game.js'></script>\n",
            encoding="utf-8",
        )
        (work / "game.js").write_text(
            textwrap.dedent(
                """
                const canvas = document.getElementById('game');
                const ctx = canvas.getContext('2d');
                let snake = [{x: 10, y: 10}];
                let food = {x: 5, y: 5};
                let dx = 1, dy = 0;
                let score = 0;
                function tick() {
                  const head = {x: snake[0].x + dx, y: snake[0].y + dy};
                  snake.unshift(head);
                  if (head.x === food.x && head.y === food.y) score++;
                  else snake.pop();
                  ctx.clearRect(0, 0, 400, 400);
                  ctx.fillText('Score: ' + score, 10, 20);
                  requestAnimationFrame(tick);
                }
                tick();
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        return textwrap.dedent(
            """
            Improve this existing Snake game.
            Requirements:
            - Add proper grid collision and self-collision game-over behavior.
            - Add pause/resume, restart, visible score, and speed ramp.
            - Add touch/mobile controls or on-screen direction buttons.
            - Preserve dependency-free browser execution.
            - Update README.md.
            Do not read or write outside this workspace.
            """
        ).strip()
    if task == "opencode_seeded_bugfix":
        (work / "string_stats.py").write_text(
            textwrap.dedent(
                """
                def summarize_words(text):
                    words = text.split(" ")
                    return {
                        "count": len(words),
                        "unique": len(set(words)),
                        "longest": max(words),
                    }
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        (work / "test_string_stats.py").write_text(
            textwrap.dedent(
                """
                import unittest
                from string_stats import summarize_words

                class TestStringStats(unittest.TestCase):
                    def test_ignores_extra_whitespace_and_case(self):
                        got = summarize_words("  Rust rust\\nPython   ")
                        self.assertEqual(got["count"], 3)
                        self.assertEqual(got["unique"], 2)
                        self.assertEqual(got["longest"], "Python")

                    def test_empty_text(self):
                        self.assertEqual(summarize_words("   "), {"count": 0, "unique": 0, "longest": ""})

                if __name__ == "__main__":
                    unittest.main()
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        return textwrap.dedent(
            """
            Fix the failing tests in this small Python project.
            Requirements:
            - Run the tests, inspect the failure, patch the implementation.
            - Preserve the public function name `summarize_words`.
            - Do not add dependencies.
            - Explain the bug briefly in README.md.
            Do not read or write outside this workspace.
            """
        ).strip()
    if task == "opencode_long_context_rules":
        docs = work / "docs"
        docs.mkdir()
        for i in range(80):
            docs.joinpath(f"rule_{i:03d}.md").write_text(
                "\n".join(
                    [
                        f"# Rule {i:03d}",
                        "Most files here are distractors for a long-context benchmark.",
                        "The evaluator must ignore archived rules unless they are active.",
                        f"Archive checksum {i * 7919}.",
                    ]
                    * 12
                )
                + "\n",
                encoding="utf-8",
            )
        docs.joinpath("active_rules.md").write_text(
            textwrap.dedent(
                """
                # Active Rules
                A task is high priority when severity is `critical` or `high`.
                A task is stale when `days_open` is greater than 14.
                A task is escalation-ready only when it is high priority, stale, and owner is not empty.
                The result should include `label`, `stale`, and `escalate`.
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        (work / "evaluator.py").write_text(
            "def classify(task):\n    return {'label': 'todo', 'stale': False, 'escalate': False}\n",
            encoding="utf-8",
        )
        (work / "test_evaluator.py").write_text(
            textwrap.dedent(
                """
                import unittest
                from evaluator import classify

                class TestEvaluator(unittest.TestCase):
                    def test_escalates_stale_critical_owned_task(self):
                        got = classify({"severity": "critical", "days_open": 21, "owner": "ops"})
                        self.assertEqual(got, {"label": "high_priority", "stale": True, "escalate": True})

                    def test_does_not_escalate_unowned_high_task(self):
                        got = classify({"severity": "high", "days_open": 30, "owner": ""})
                        self.assertEqual(got, {"label": "high_priority", "stale": True, "escalate": False})

                    def test_low_recent_task(self):
                        got = classify({"severity": "low", "days_open": 2, "owner": "eng"})
                        self.assertEqual(got, {"label": "normal", "stale": False, "escalate": False})

                if __name__ == "__main__":
                    unittest.main()
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        return textwrap.dedent(
            """
            Implement `classify(task)` using the active rules hidden among the docs.
            Requirements:
            - Search/read the docs to find the active rule source.
            - Fix `evaluator.py` so all tests pass.
            - Do not add dependencies.
            - Add `NOTES.md` naming the active rules file you used.
            Do not read or write outside this workspace.
            """
        ).strip()
    if task == "opencode_recovery_debug":
        (work / "calc.py").write_text(
            textwrap.dedent(
                """
                def moving_average(values, window):
                    if window <= 0:
                        return []
                    out = []
                    for i in range(len(values)):
                        out.append(sum(values[i:i+window]) / window)
                    return out
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        (work / "test_calc.py").write_text(
            textwrap.dedent(
                """
                import unittest
                from calc import moving_average

                class TestCalc(unittest.TestCase):
                    def test_window_two(self):
                        self.assertEqual(moving_average([2, 4, 6, 8], 2), [3, 5, 7])

                    def test_large_window(self):
                        self.assertEqual(moving_average([1, 2], 5), [])

                    def test_invalid_window_raises(self):
                        with self.assertRaises(ValueError):
                            moving_average([1, 2, 3], 0)

                if __name__ == "__main__":
                    unittest.main()
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        return textwrap.dedent(
            """
            Debug and fix this moving-average implementation.
            Requirements:
            - Run tests first.
            - Avoid repeated identical test commands if the evidence has not changed.
            - Patch the implementation and rerun tests.
            - Write `DEBUG_NOTES.md` with the root cause and fix.
            Do not read or write outside this workspace.
            """
        ).strip()
    raise ValueError(f"unknown opencode task {task}")


def run_subprocess_capture(
    cmd: list[str],
    cwd: pathlib.Path,
    out_prefix: pathlib.Path,
    timeout_sec: int,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=full_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        timed_out = True
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            stdout, stderr = proc.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate(timeout=10)
    latency = round((time.monotonic() - started) * 1000, 3)
    out_prefix.with_suffix(".stdout.jsonl").write_text(stdout or "", encoding="utf-8")
    out_prefix.with_suffix(".stderr.log").write_text(stderr or "", encoding="utf-8")
    event_counts: dict[str, int] = {}
    for line in (stdout or "").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_type = str(event.get("type") or event.get("event") or "json")
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
    result = {
        "cmd": cmd,
        "cmd_display": shell_quote(cmd),
        "cwd": str(cwd),
        "exit_code": proc.returncode,
        "timed_out": timed_out,
        "latency_ms": latency,
        "event_counts": event_counts,
        "stdout_bytes": len((stdout or "").encode("utf-8")),
        "stderr_bytes": len((stderr or "").encode("utf-8")),
        "command_diagnostics": command_diagnostics_from_stdout(stdout or ""),
        "finished_at": utc_now(),
    }
    write_json(out_prefix.with_suffix(".summary.json"), result)
    return result


def judge_workspace(task: str, work: pathlib.Path, out_dir: pathlib.Path) -> dict[str, Any]:
    result: dict[str, Any] = {"task": task, "passed": False, "checks": {}, "captured_at": utc_now()}
    files = [str(path.relative_to(work)) for path in work.rglob("*") if path.is_file()]
    result["files"] = sorted(files)
    if task == "opencode_canary_cli":
        result["checks"]["agent_canary_py"] = (work / "agent_canary.py").exists()
        if (work / "agent_canary.py").exists():
            run = run_subprocess_capture(
                [sys.executable, "agent_canary.py"],
                work,
                out_dir / "judge-agent-canary",
                30,
            )
            result["checks"]["run_exit_zero"] = run["exit_code"] == 0
            stdout = (out_dir / "judge-agent-canary.stdout.jsonl").read_text(encoding="utf-8")
            result["checks"]["expected_output"] = "agentic canary ok" in stdout.lower()
        result["passed"] = all(bool(v) for v in result["checks"].values())
    elif task.startswith("opencode_snake"):
        content = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore")
            for path in work.rglob("*")
            if path.is_file() and path.suffix.lower() in {".html", ".js", ".css", ".md"}
        ).lower()
        result["checks"] = {
            "index_html": (work / "index.html").exists(),
            "javascript": any(path.suffix == ".js" for path in work.rglob("*")),
            "snake_keyword": "snake" in content,
            "food_keyword": "food" in content,
            "score_keyword": "score" in content,
            "pause_or_restart": "pause" in content or "restart" in content,
            "game_over": "game over" in content or "gameover" in content,
        }
        result["passed"] = all(bool(v) for v in result["checks"].values())
    elif task in {"opencode_seeded_bugfix", "opencode_long_context_rules", "opencode_recovery_debug"}:
        run = run_subprocess_capture(
            [sys.executable, "-m", "unittest", "discover", "-v"],
            work,
            out_dir / "judge-unittest",
            60,
        )
        result["checks"]["unittest_exit_zero"] = run["exit_code"] == 0
        if task == "opencode_seeded_bugfix":
            result["checks"]["readme"] = (work / "README.md").exists()
        if task == "opencode_long_context_rules":
            result["checks"]["notes"] = (work / "NOTES.md").exists()
        if task == "opencode_recovery_debug":
            result["checks"]["debug_notes"] = (work / "DEBUG_NOTES.md").exists()
        result["passed"] = all(bool(v) for v in result["checks"].values())
    write_json(out_dir / "judge.json", result)
    return result


def run_opencode_task(
    run_root: pathlib.Path,
    lane_dir: pathlib.Path,
    base_url: str,
    model_id: str,
    context_limit: int,
    opencode_bin: str,
    task: str,
    index: int,
    timeout_sec: int,
) -> dict[str, Any]:
    task_dir = lane_dir / f"{index:03d}-{task}"
    work = task_dir / "workspace"
    xdg = task_dir / "xdg"
    config_dir = xdg / "config"
    data_dir = xdg / "data"
    cache_dir = xdg / "cache"
    state_dir = xdg / "state"
    home_dir = task_dir / "home"
    for path in (config_dir, data_dir, cache_dir, state_dir, home_dir):
        path.mkdir(parents=True, exist_ok=True)
    prompt = seed_workspace(task, work)
    (task_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    create_opencode_config(config_dir, base_url.rstrip("/") + "/v1", model_id, context_limit)
    fetch_admin_metrics(base_url, task_dir / "admin-before.json")
    env = {
        "HOME": str(home_dir),
        "XDG_CONFIG_HOME": str(config_dir),
        "XDG_DATA_HOME": str(data_dir),
        "XDG_CACHE_HOME": str(cache_dir),
        "XDG_STATE_HOME": str(state_dir),
        "OPENCODE_LOCAL_API_KEY": "dummy",
    }
    cmd = [
        opencode_bin,
        "run",
        "--pure",
        "--dir",
        str(work),
        "--model",
        f"kir/{model_id}",
        "--format",
        "json",
        "--dangerously-skip-permissions",
        "--title",
        f"kir-bench-{task}-{model_id}",
        prompt,
    ]
    log(run_root, f"starting opencode task {task} model={model_id} timeout={timeout_sec}s")
    proc_result = run_subprocess_capture(cmd, work, task_dir / "opencode", timeout_sec, env=env)
    fetch_admin_metrics(base_url, task_dir / "admin-after.json")
    judge = judge_workspace(task, work, task_dir)
    summary = {
        "kind": "opencode",
        "task": task,
        "model_id": model_id,
        "proc": proc_result,
        "judge": judge,
        "artifact_dir": str(task_dir),
        "completed_at": utc_now(),
    }
    write_json(task_dir / "summary.json", summary)
    append_jsonl(lane_dir / "samples.jsonl", summary)
    return summary


def run_direct_probe(
    run_root: pathlib.Path,
    lane_dir: pathlib.Path,
    base_url: str,
    model_id: str,
    probe: str,
    index: int,
    repeat: int,
    timeout_sec: int,
) -> dict[str, Any]:
    task_dir = lane_dir / f"{index:03d}-{probe}-r{repeat}"
    task_dir.mkdir(parents=True, exist_ok=True)
    body = direct_body(model_id, probe, repeat=repeat)
    write_json(task_dir / "request.json", body)
    fetch_admin_metrics(base_url, task_dir / "admin-before.json")
    log(run_root, f"starting direct probe {probe} repeat={repeat} model={model_id}")
    result = stream_chat_completion(
        base_url.rstrip("/") + "/v1",
        body,
        task_dir / "stream.trace",
        timeout=timeout_sec,
    )
    fetch_admin_metrics(base_url, task_dir / "admin-after.json")
    summary = {
        "kind": "direct",
        "probe": probe,
        "repeat": repeat,
        "model_id": model_id,
        "result": result,
        "artifact_dir": str(task_dir),
        "completed_at": utc_now(),
    }
    write_json(task_dir / "summary.json", summary)
    append_jsonl(lane_dir / "samples.jsonl", summary)
    return summary


def model_identity_for_model_id(model_id: str) -> dict[str, Any]:
    for lane in LANES:
        if lane.model_id == model_id:
            return {
                "lane": lane.name,
                "family": lane.family,
                "quantization": lane.quantization,
                "max_context_k": lane.max_context_k,
            }
    return {
        "lane": None,
        "family": None,
        "quantization": None,
        "max_context_k": None,
    }


def increment_count(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def int_metric(value: Any) -> int:
    return value if isinstance(value, int) else 0


def direct_probe_requires_tool(probe: str) -> bool:
    return probe == "direct_tool_required_stream" or probe.startswith("direct_stable_prefix_")


def tool_call_diagnostics_from_result(result: dict[str, Any]) -> dict[str, Any]:
    diagnostics = result.get("tool_calls")
    if isinstance(diagnostics, dict):
        return diagnostics
    return {
        "observed": int_metric(result.get("tool_fragments_count")),
        "valid_json_arguments": 0,
        "invalid_json_arguments": 0,
        "names": [],
    }


def direct_tool_call_succeeded(result: dict[str, Any]) -> bool:
    diagnostics = tool_call_diagnostics_from_result(result)
    names = diagnostics.get("names")
    finish_reasons = result.get("finish_reasons")
    return (
        int_metric(diagnostics.get("observed")) > 0
        and int_metric(diagnostics.get("valid_json_arguments")) > 0
        and isinstance(names, list)
        and "record_agentic_observation" in names
        and isinstance(finish_reasons, list)
        and "tool_calls" in finish_reasons
        and not result.get("errors")
    )


def opencode_failure_modes(proc: dict[str, Any], judge: dict[str, Any]) -> list[str]:
    modes: list[str] = []
    if proc.get("timed_out"):
        modes.append("timeout")
    if proc.get("exit_code") != 0:
        modes.append("command_execution")
    checks = judge.get("checks")
    if isinstance(checks, dict):
        if checks.get("unittest_exit_zero") is False:
            modes.append("task_correctness")
        elif any(value is False for value in checks.values()):
            modes.append("workspace_artifact")
    if not modes:
        modes.append("judge_failure")
    return modes


def speed_quality_classification(quality_score: float | None, latency_p50: Any) -> str:
    if quality_score is None:
        return "no_quality_samples"
    if quality_score == 0 and isinstance(latency_p50, (int, float)):
        return "fast_but_low_quality"
    if quality_score < 0.5:
        return "low_quality"
    return "quality_viable"


def summarize_run(run_root: pathlib.Path) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    for path in sorted(run_root.glob("*/samples.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            with contextlib.suppress(json.JSONDecodeError):
                samples.append(json.loads(line))
    by_model: dict[str, dict[str, Any]] = {}
    for sample in samples:
        model = str(sample.get("model_id") or "unknown")
        bucket = by_model.setdefault(
            model,
            {
                "samples": 0,
                "model_identity": model_identity_for_model_id(model),
                "direct": 0,
                "opencode": 0,
                "opencode_passed": 0,
                "failures": 0,
                "failure_modes": {},
                "agentic_quality": {
                    "direct_tool_expected": 0,
                    "direct_tool_success": 0,
                    "opencode_process_success": 0,
                    "command_syntax_observed": 0,
                    "command_syntax_valid": 0,
                    "command_syntax_invalid": 0,
                    "unittest_exit_zero": {"passed": 0, "failed": 0},
                },
                "cache_status_counts": {},
                "latency_ms": [],
                "first_tool_delta_ms": [],
                "first_semantic_delta_ms": [],
            },
        )
        bucket["samples"] += 1
        kind = sample.get("kind")
        if kind == "direct":
            bucket["direct"] += 1
            result = sample.get("result") or {}
            direct_failed = False
            for key in ("latency_ms", "first_tool_delta_ms", "first_semantic_delta_ms"):
                value = result.get(key)
                if isinstance(value, (int, float)):
                    bucket[key].append(value)
            status = str(result.get("cache_status") or "unknown")
            counts = bucket["cache_status_counts"]
            counts[status] = counts.get(status, 0) + 1
            if result.get("errors"):
                direct_failed = True
                increment_count(bucket["failure_modes"], "protocol_or_stream")
            probe = str(sample.get("probe") or "")
            if direct_probe_requires_tool(probe):
                quality = bucket["agentic_quality"]
                quality["direct_tool_expected"] += 1
                if direct_tool_call_succeeded(result):
                    quality["direct_tool_success"] += 1
                else:
                    direct_failed = True
                    increment_count(bucket["failure_modes"], "tool_use")
            if direct_failed:
                bucket["failures"] += 1
        elif kind == "opencode":
            bucket["opencode"] += 1
            proc = sample.get("proc") or {}
            judge = sample.get("judge") or {}
            quality = bucket["agentic_quality"]
            value = proc.get("latency_ms")
            if isinstance(value, (int, float)):
                bucket["latency_ms"].append(value)
            if proc.get("exit_code") == 0 and not proc.get("timed_out"):
                quality["opencode_process_success"] += 1
            diagnostics = proc.get("command_diagnostics")
            if isinstance(diagnostics, dict):
                quality["command_syntax_observed"] += int_metric(diagnostics.get("observed"))
                quality["command_syntax_valid"] += int_metric(diagnostics.get("syntax_valid"))
                quality["command_syntax_invalid"] += int_metric(diagnostics.get("syntax_invalid"))
            checks = judge.get("checks")
            if isinstance(checks, dict) and "unittest_exit_zero" in checks:
                unittest_counts = quality["unittest_exit_zero"]
                if checks.get("unittest_exit_zero"):
                    unittest_counts["passed"] += 1
                else:
                    unittest_counts["failed"] += 1
            if judge.get("passed"):
                bucket["opencode_passed"] += 1
            else:
                bucket["failures"] += 1
                for mode in opencode_failure_modes(proc, judge):
                    increment_count(bucket["failure_modes"], mode)
    for bucket in by_model.values():
        for key in ("latency_ms", "first_tool_delta_ms", "first_semantic_delta_ms"):
            values = sorted(bucket[key])
            if values:
                bucket[key] = {
                    "count": len(values),
                    "p50": values[len(values) // 2],
                    "p95": values[min(len(values) - 1, int(len(values) * 0.95))],
                    "max": values[-1],
                }
            else:
                bucket[key] = {"count": 0}
        quality = bucket["agentic_quality"]
        quality["opencode_pass_rate"] = rate(bucket["opencode_passed"], bucket["opencode"])
        quality["opencode_process_success_rate"] = rate(
            quality["opencode_process_success"],
            bucket["opencode"],
        )
        quality["direct_tool_call_success_rate"] = rate(
            quality["direct_tool_success"],
            quality["direct_tool_expected"],
        )
        quality["command_syntax_success_rate"] = rate(
            quality["command_syntax_valid"],
            quality["command_syntax_observed"],
        )
        quality_attempts = bucket["opencode"] + quality["direct_tool_expected"]
        quality_successes = bucket["opencode_passed"] + quality["direct_tool_success"]
        quality_score = rate(quality_successes, quality_attempts)
        latency_p50 = bucket["latency_ms"].get("p50")
        if quality_score is not None and isinstance(latency_p50, (int, float)) and latency_p50 > 0:
            throughput_quality_score = round(quality_score * 1000 / latency_p50, 6)
        else:
            throughput_quality_score = None
        bucket["speed_quality"] = {
            "quality_score": quality_score,
            "quality_attempts": quality_attempts,
            "median_latency_ms": latency_p50,
            "throughput_quality_score": throughput_quality_score,
            "classification": speed_quality_classification(quality_score, latency_p50),
        }
    summary = {
        "updated_at": utc_now(),
        "sample_count": len(samples),
        "by_model": by_model,
    }
    write_json(run_root / "summary.json", summary)
    return summary


def build_engine_if_needed(run_root: pathlib.Path) -> None:
    binary = REPO_ROOT / "target" / "debug" / "llm-engine"
    if binary.exists():
        return
    log(run_root, "building llm-engine binary")
    result = run_subprocess_capture(
        ["cargo", "build", "-p", "llm-engine", "--all-features"],
        REPO_ROOT,
        run_root / "cargo-build",
        timeout_sec=1800,
    )
    if result["exit_code"] != 0:
        raise RuntimeError("cargo build failed; see cargo-build.stderr.log")


def sidecar_command(lane: ModelLane, sidecar_port: int) -> list[str]:
    unsupported_vlm_args = [
        arg
        for arg in lane.sidecar_extra
        if arg in {"--max-tokens", "--prompt-cache-size"}
        or arg.startswith("--max-tokens=")
        or arg.startswith("--prompt-cache-size=")
    ]
    if lane.sidecar_kind == "vlm" and unsupported_vlm_args:
        raise ValueError(
            f"{lane.name} uses mlx_vlm.server, which does not support "
            f"{', '.join(unsupported_vlm_args)}"
        )
    return [
        "uvx",
        "--from",
        lane.sidecar_package,
        lane.sidecar_module,
        "--model",
        str(lane.snapshot),
        "--host",
        "127.0.0.1",
        "--port",
        str(sidecar_port),
        *lane.sidecar_extra,
    ]


def start_lane(
    lane: ModelLane,
    lane_dir: pathlib.Path,
    run_root: pathlib.Path,
    sidecar_port: int,
    kir_port: int,
    sidecar_ready_timeout: int,
    kir_ready_timeout: int,
) -> tuple[subprocess.Popen[Any], subprocess.Popen[Any], str]:
    if not lane.snapshot.exists():
        raise FileNotFoundError(f"snapshot missing: {lane.snapshot}")
    sidecar_log = lane_dir / "sidecar.log"
    kir_log = lane_dir / "kir.log"
    sidecar_cmd = sidecar_command(lane, sidecar_port)
    log(run_root, f"launching sidecar {lane.name}: {shell_quote(sidecar_cmd)}")
    sidecar = start_logged_process(sidecar_cmd, sidecar_log, REPO_ROOT)
    sidecar_ok, sidecar_body = wait_for_endpoint(
        f"http://127.0.0.1:{sidecar_port}/v1/models",
        sidecar_ready_timeout,
        run_root,
        f"{lane.name} sidecar",
    )
    write_json(lane_dir / "sidecar-ready.json", {"ok": sidecar_ok, "body": sidecar_body})
    if not sidecar_ok:
        stop_process(sidecar, f"{lane.name} sidecar", run_root)
        raise RuntimeError(f"{lane.name} sidecar did not become ready")
    binary = REPO_ROOT / "target" / "debug" / "llm-engine"
    kir_cmd = [
        str(binary),
        "serve",
        "--addr",
        f"127.0.0.1:{kir_port}",
        "--snapshot",
        str(lane.snapshot),
        "--loader",
        "mlx",
        "--family",
        lane.family,
        "--model-id",
        lane.model_id,
        "--mlx-endpoint",
        f"http://127.0.0.1:{sidecar_port}/v1",
        "--mlx-read-timeout",
        "600",
        "--mlx-request-timeout",
        "3600",
        "--canonical-tool-schemas",
        "--max-concurrent-requests",
        "1",
    ]
    log(run_root, f"launching kir {lane.name}: {shell_quote(kir_cmd)}")
    kir = start_logged_process(kir_cmd, kir_log, REPO_ROOT)
    base_url = f"http://127.0.0.1:{kir_port}"
    kir_ok, kir_body = wait_for_endpoint(
        f"{base_url}/v1/models",
        kir_ready_timeout,
        run_root,
        f"{lane.name} kir",
    )
    write_json(lane_dir / "kir-ready.json", {"ok": kir_ok, "body": kir_body})
    if not kir_ok:
        stop_process(kir, f"{lane.name} kir", run_root)
        stop_process(sidecar, f"{lane.name} sidecar", run_root)
        raise RuntimeError(f"{lane.name} kir did not become ready")
    return sidecar, kir, base_url


def lane_sequence(include_heavy_gemma31: bool, only: set[str] | None) -> list[ModelLane]:
    lanes = []
    for lane in LANES:
        if only and lane.name not in only:
            continue
        if lane.include_by_default or (include_heavy_gemma31 and lane.name == "gemma4-31b-mlx"):
            lanes.append(lane)
    return lanes


def run_lane(
    lane: ModelLane,
    run_root: pathlib.Path,
    deadline: float,
    lane_budget_sec: int,
    args: argparse.Namespace,
) -> None:
    lane_dir = run_root / lane.name
    lane_dir.mkdir(parents=True, exist_ok=True)
    sidecar: subprocess.Popen[Any] | None = None
    kir: subprocess.Popen[Any] | None = None
    sidecar_port = find_free_port()
    kir_port = find_free_port()
    lane_manifest = {
        "lane": dataclasses.asdict(lane),
        "sidecar_port": sidecar_port,
        "kir_port": kir_port,
        "started_at": utc_now(),
        "lane_budget_sec": lane_budget_sec,
        "context_sizes_k": context_sizes_for_lane(lane, args.context_sizes_k),
    }
    lane_manifest["lane"]["snapshot"] = str(lane.snapshot)
    write_json(lane_dir / "manifest.json", lane_manifest)
    try:
        sidecar, kir, base_url = start_lane(
            lane,
            lane_dir,
            run_root,
            sidecar_port,
            kir_port,
            args.sidecar_ready_timeout,
            args.kir_ready_timeout,
        )
        lane_deadline = min(deadline, time.monotonic() + lane_budget_sec)
        index = 0
        lane_context_sizes = context_sizes_for_lane(lane, args.context_sizes_k)
        context_limit = lane.max_context_k * 1024
        if not args.skip_direct:
            for probe in DIRECT_CANARY_PROBES:
                if time.monotonic() > lane_deadline:
                    break
                index += 1
                run_direct_probe(
                    run_root,
                    lane_dir,
                    base_url,
                    lane.model_id,
                    probe,
                    index,
                    0,
                    args.direct_timeout,
                )
                summarize_run(run_root)
        if not args.skip_opencode and time.monotonic() < lane_deadline:
            index += 1
            run_opencode_task(
                run_root,
                lane_dir,
                base_url,
                lane.model_id,
                context_limit,
                args.opencode_bin,
                "opencode_canary_cli",
                index,
                min(args.opencode_timeout, max(120, int(lane_deadline - time.monotonic()))),
            )
            summarize_run(run_root)
        task_cycle = list(OPENCODE_TASKS[1:])
        random.shuffle(task_cycle)
        direct_cycle = list(direct_probe_names(lane_context_sizes)[len(DIRECT_CANARY_PROBES) :])
        repeat = 0
        while time.monotonic() < lane_deadline - 60:
            remaining = int(lane_deadline - time.monotonic())
            # Alternate direct context/cache probes with real opencode tasks.
            if not args.skip_direct:
                for probe in direct_cycle:
                    if time.monotonic() >= lane_deadline - 60:
                        break
                    index += 1
                    run_direct_probe(
                        run_root,
                        lane_dir,
                        base_url,
                        lane.model_id,
                        probe,
                        index,
                        repeat,
                        min(args.long_direct_timeout, max(120, remaining)),
                    )
                    summarize_run(run_root)
                    remaining = int(lane_deadline - time.monotonic())
            random.shuffle(task_cycle)
            if not args.skip_opencode:
                for task in task_cycle:
                    if time.monotonic() >= lane_deadline - 60:
                        break
                    index += 1
                    timeout = min(
                        args.opencode_timeout,
                        max(180, int(lane_deadline - time.monotonic() - 30)),
                    )
                    run_opencode_task(
                        run_root,
                        lane_dir,
                        base_url,
                        lane.model_id,
                        context_limit,
                        args.opencode_bin,
                        task,
                        index,
                        timeout,
                    )
                    summarize_run(run_root)
            repeat += 1
    except Exception as exc:  # noqa: BLE001
        error = {
            "lane": lane.name,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "captured_at": utc_now(),
        }
        write_json(lane_dir / "lane-error.json", error)
        append_jsonl(run_root / "lane-errors.jsonl", error)
        log(run_root, f"lane {lane.name} failed: {exc!r}")
    finally:
        stop_process(kir, f"{lane.name} kir", run_root)
        stop_process(sidecar, f"{lane.name} sidecar", run_root)
        write_json(lane_dir / "finished.json", {"finished_at": utc_now()})
        summarize_run(run_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--run-root", type=pathlib.Path, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--context-sizes-k",
        type=parse_context_sizes_k,
        default=DEFAULT_CONTEXT_SIZES_K,
        help="Comma-separated stable-prefix sizes in approximate K tokens. Each lane also includes its max_context_k.",
    )
    parser.add_argument("--include-heavy-gemma31", action="store_true")
    parser.add_argument("--only", action="append", default=[], help="Run only the named lane; can repeat")
    parser.add_argument(
        "--opencode-bin",
        default=shutil.which("opencode") or "/opt/homebrew/bin/opencode",
        help="opencode executable used for real agentic tasks",
    )
    parser.add_argument("--skip-direct", action="store_true", help="Skip direct streaming/cache probes")
    parser.add_argument("--skip-opencode", action="store_true", help="Skip opencode agentic tasks")
    parser.add_argument("--dry-run", action="store_true", help="Write the plan manifest and exit")
    parser.add_argument("--sidecar-ready-timeout", type=int, default=1800)
    parser.add_argument("--kir-ready-timeout", type=int, default=600)
    parser.add_argument("--direct-timeout", type=int, default=600)
    parser.add_argument("--long-direct-timeout", type=int, default=1200)
    parser.add_argument("--opencode-timeout", type=int, default=2400)
    parsed = parser.parse_args()
    if parsed.hours <= 0:
        parser.error("--hours must be greater than zero")
    if parsed.skip_direct and parsed.skip_opencode:
        parser.error("--skip-direct and --skip-opencode cannot both be set")
    return parsed


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    stamp = dt.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    run_root = args.run_root or (DEFAULT_RUN_ROOT / f"{stamp}-agentic-overnight")
    run_root.mkdir(parents=True, exist_ok=True)
    only = set(args.only) if args.only else None
    lanes = lane_sequence(args.include_heavy_gemma31, only)
    if not lanes:
        raise SystemExit("no lanes selected")
    per_lane_plan = {
        lane.name: {
            "max_context_k": lane.max_context_k,
            "context_sizes_k": list(context_sizes_for_lane(lane, args.context_sizes_k)),
            "direct_probes": list(direct_probe_names(context_sizes_for_lane(lane, args.context_sizes_k)))
            if not args.skip_direct
            else [],
            "opencode_tasks": list(OPENCODE_TASKS) if not args.skip_opencode else [],
        }
        for lane in lanes
    }
    manifest = {
        "started_at": utc_now(),
        "repo_root": str(REPO_ROOT),
        "run_root": str(run_root),
        "hours": args.hours,
        "seed": args.seed,
        "dry_run": args.dry_run,
        "opencode_bin": args.opencode_bin,
        "lanes": [
            {**dataclasses.asdict(lane), "snapshot": str(lane.snapshot), "snapshot_exists": lane.snapshot.exists()}
            for lane in lanes
        ],
        "hypotheses": [
            "Qwen 27B may be the best throughput-adjusted agentic lane if it avoids malformed tool calls.",
            "Qwen 35B should improve task quality but may lose wall-clock efficiency.",
            "Gemma 4 should expose protocol/tool robustness differences and small-model recovery behavior.",
            "Long-context latency should show size regimes; 135k is treated as one point, not the default answer.",
            "Stable prefixes should show observable cached-token reuse in Kir admin metrics after warm repeats.",
            "Opencode traces should reveal no-progress loops, command retries, and file-edit quality not visible in synthetic probes.",
        ],
        "workloads": {
            "default_context_sizes_k": list(args.context_sizes_k),
            "per_lane": per_lane_plan,
        },
    }
    write_json(run_root / "manifest.json", manifest)
    log(run_root, f"run root: {run_root}")
    log(run_root, f"selected lanes: {', '.join(lane.name for lane in lanes)}")
    if args.dry_run:
        write_json(
            run_root / "finished.json",
            {"finished_at": utc_now(), "summary": summarize_run(run_root), "dry_run": True},
        )
        log(run_root, "dry run finished")
        return 0
    build_engine_if_needed(run_root)
    deadline = time.monotonic() + args.hours * 3600
    for idx, lane in enumerate(lanes):
        remaining_lanes = max(1, len(lanes) - idx)
        remaining_sec = max(0, int(deadline - time.monotonic()))
        if remaining_sec < 300:
            log(run_root, "deadline too close; stopping before next lane")
            break
        lane_budget = max(600, remaining_sec // remaining_lanes)
        run_lane(lane, run_root, deadline, lane_budget, args)
    summary = summarize_run(run_root)
    write_json(run_root / "finished.json", {"finished_at": utc_now(), "summary": summary})
    log(run_root, "benchmark finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
