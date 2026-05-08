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
    uint id [[thread_position_in_grid]]
) {
    if (id != 0 || len == 0) {
        return;
    }
    float max_score = scores[0];
    for (uint index = 1; index < len; index++) {
        max_score = max(max_score, scores[index]);
    }
    float denominator = 0.0;
    for (uint index = 0; index < len; index++) {
        denominator += exp(scores[index] - max_score);
    }
    for (uint index = 0; index < len; index++) {
        output[index] = exp(scores[index] - max_score) / denominator;
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

