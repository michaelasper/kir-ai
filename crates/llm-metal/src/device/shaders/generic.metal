#include <metal_stdlib>
using namespace metal;

kernel void vector_add(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = left[id] + right[id];
}

kernel void softmax_f32(
    device const float* scores [[buffer(0)]],
    constant uint& len [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& thread_count [[buffer(3)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint thread_id [[thread_index_in_threadgroup]]
) {
    float local_max = -INFINITY;
    for (uint index = thread_id; index < len; index += thread_count) {
        local_max = max(local_max, scores[index]);
    }
    scratch[thread_id] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = thread_count >> 1; stride > 0; stride >>= 1) {
        if (thread_id < stride) {
            scratch[thread_id] = max(scratch[thread_id], scratch[thread_id + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float max_score = scratch[0];
    float denominator = 0.0;
    for (uint index = thread_id; index < len; index += thread_count) {
        float probability = exp(scores[index] - max_score);
        output[index] = probability;
        denominator += probability;
    }
    scratch[thread_id] = denominator;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = thread_count >> 1; stride > 0; stride >>= 1) {
        if (thread_id < stride) {
            scratch[thread_id] += scratch[thread_id + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_denominator = 1.0 / scratch[0];
    for (uint index = thread_id; index < len; index += thread_count) {
        output[index] *= inv_denominator;
    }
}

kernel void weighted_sum_f32(
    device const float* values [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    constant uint& row_count [[buffer(2)]],
    constant uint& vector_len [[buffer(3)]],
    device float* output [[buffer(4)]],
    uint column [[thread_position_in_grid]]
) {
    if (column >= vector_len) {
        return;
    }
    float sum = 0.0;
    for (uint row = 0; row < row_count; row++) {
        sum += values[(row * vector_len) + column] * weights[row];
    }
    output[column] = sum;
}

kernel void select_head_rows_f32(
    device const float* values [[buffer(0)]],
    constant uint& row_len [[buffer(1)]],
    constant uint& head_start [[buffer(2)]],
    constant uint& head_len [[buffer(3)]],
    constant uint& output_len [[buffer(4)]],
    device float* output [[buffer(5)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= output_len) {
        return;
    }
    uint row = index / head_len;
    uint offset = index % head_len;
    output[index] = values[(row * row_len) + head_start + offset];
}
