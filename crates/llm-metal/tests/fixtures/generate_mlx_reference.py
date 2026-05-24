#!/usr/bin/env python3
"""Generate MLX reference outputs for the Metal kernel smoke trace.

Run from the repository root with:

    python3 crates/llm-metal/tests/fixtures/generate_mlx_reference.py

The script prints JSON to stdout. Latencies are environment-sensitive and are
kept as trace data rather than pass/fail thresholds in the Rust test.
"""

from __future__ import annotations

import json
import statistics
import time
from datetime import date
from typing import Any, Callable

import mlx.core as mx


def main() -> None:
    cases = {
        "vector_add_f32": output_case(vector_add_f32),
        "rms_norm_one_centered_f32": output_case(rms_norm_one_centered_f32),
        "softmax_f32": output_case(softmax_f32),
        "linear_attention_conv1d_silu_f32": output_case(
            linear_attention_conv1d_silu_f32
        ),
        "matvec_f32": output_case(matvec_f32),
        "matvec_bf16_f32": output_case(matvec_bf16_f32),
        "batched_matvec_bf16_f32": output_case(batched_matvec_bf16_f32),
        "weighted_sum_f32": output_case(weighted_sum_f32),
        "linear_attention_recurrent_update_f32": output_case(
            linear_attention_recurrent_update_f32
        ),
        "select_head_rows_f32": output_case(select_head_rows_f32),
        "argmax_f32": indexed_case(argmax_f32),
        "top_k_f32": top_k_case(top_k_f32),
    }
    print(
        json.dumps(
            {
                "schema_version": 1,
                "source": "mlx.core offline reference trace generated from simple deterministic tensors",
                "generator": "crates/llm-metal/tests/fixtures/generate_mlx_reference.py",
                "generated_date": date.today().isoformat(),
                "cases": cases,
            },
            indent=2,
        )
    )


def output_case(operation: Callable[[], mx.array]) -> dict[str, Any]:
    output = operation()
    mx.eval(output)
    return {
        "output": flatten_f32(output),
        "mlx_median_us": median_us(operation),
    }


def indexed_case(operation: Callable[[], tuple[mx.array, mx.array]]) -> dict[str, Any]:
    index, value = operation()
    mx.eval(index, value)
    return {
        "index": int(index.item()),
        "value": float(value.item()),
        "mlx_median_us": median_us(operation),
    }


def top_k_case(operation: Callable[[], tuple[mx.array, mx.array]]) -> dict[str, Any]:
    indices, values = operation()
    mx.eval(indices, values)
    return {
        "indices": [int(item) for item in indices.tolist()],
        "values": [float(item) for item in values.tolist()],
        "mlx_median_us": median_us(operation),
    }


def median_us(operation: Callable[[], Any], warmup: int = 8, runs: int = 40) -> float:
    for _ in range(warmup):
        force_eval(operation())

    samples = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        force_eval(operation())
        elapsed = time.perf_counter_ns() - start
        samples.append(elapsed / 1000.0)
    return round(statistics.median(samples), 3)


def force_eval(value: Any) -> None:
    if isinstance(value, tuple):
        mx.eval(*value)
    else:
        mx.eval(value)


def flatten_f32(array: mx.array) -> list[float]:
    return [float(item) for item in mx.flatten(array).tolist()]


def vector_add_f32() -> mx.array:
    left = mx.array([1.0, 2.5, -3.0, 8.0], dtype=mx.float32)
    right = mx.array([4.0, -1.5, 3.0, 0.25], dtype=mx.float32)
    return left + right


def rms_norm_one_centered_f32() -> mx.array:
    values = mx.array([3.0, 4.0], dtype=mx.float32)
    weight = mx.array([0.0, 1.0], dtype=mx.float32)
    mean_square = mx.mean(mx.square(values))
    return values * mx.rsqrt(mean_square + 0.0) * (1.0 + weight)


def softmax_f32() -> mx.array:
    scores = mx.array([1.0, 2.0, -1.0, 0.5], dtype=mx.float32)
    return mx.softmax(scores)


def linear_attention_conv1d_silu_f32() -> mx.array:
    window = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=mx.float32)
    weights = mx.array([[0.5, 1.0], [-1.0, 0.25], [2.0, -0.5]], dtype=mx.float32)
    mixed = mx.sum(window.T * weights, axis=1)
    return mixed / (1.0 + mx.exp(-mixed))


def matvec_f32() -> mx.array:
    matrix = mx.array([[1.0, 2.0, 3.0], [4.0, -1.0, 0.5]], dtype=mx.float32)
    vector = mx.array([0.5, -2.0, 4.0], dtype=mx.float32)
    return matrix @ vector


def matvec_bf16_f32() -> mx.array:
    matrix = mx.array([[1.0, 2.0, 3.0], [4.0, -1.0, 0.5]], dtype=mx.bfloat16)
    vector = mx.array([0.5, -2.0, 4.0], dtype=mx.float32)
    return (matrix.astype(mx.float32) @ vector).astype(mx.float32)


def batched_matvec_bf16_f32() -> mx.array:
    matrix = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=mx.bfloat16)
    vectors = mx.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=mx.float32)
    return vectors @ matrix.astype(mx.float32).T


def weighted_sum_f32() -> mx.array:
    values = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=mx.float32)
    weights = mx.array([0.25, -0.5], dtype=mx.float32)
    return weights @ values


def linear_attention_recurrent_update_f32() -> mx.array:
    state = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=mx.float32)
    key = mx.array([0.5, -1.0], dtype=mx.float32)
    value = mx.array([10.0, 20.0, 30.0], dtype=mx.float32)
    memory = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
    delta = (value - memory) * 0.25
    return state * 0.5 + key[:, None] * delta


def select_head_rows_f32() -> mx.array:
    values = mx.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=mx.float32)
    return values[:, 1:3]


def argmax_f32() -> tuple[mx.array, mx.array]:
    values = [-1.0] * 600
    values[42] = 4.5
    values[311] = 4.5
    values[599] = 3.25
    logits = mx.array(values, dtype=mx.float32)
    index = mx.argmax(logits)
    return index, logits[index]


def top_k_f32() -> tuple[mx.array, mx.array]:
    values = [-10.0] * 700
    values[7] = 9.0
    values[288] = 12.0
    values[499] = 12.0
    values[612] = 5.0
    logits = mx.array(values, dtype=mx.float32)
    sorted_indices = mx.argsort(-logits)
    top_indices = sorted_indices[:3]
    return top_indices, logits[top_indices]


if __name__ == "__main__":
    main()
