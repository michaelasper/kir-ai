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

kernel void attention_scores_f32(
    device const float* query [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    constant uint& row_count [[buffer(2)]],
    constant uint& num_attention_heads [[buffer(3)]],
    constant uint& num_key_value_heads [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& groups [[buffer(6)]],
    constant float& score_scale [[buffer(7)]],
    device float* scores [[buffer(8)]],
    uint2 id [[thread_position_in_grid]]
) {
    uint row = id.x;
    uint head = id.y;
    if (row >= row_count || head >= num_attention_heads) {
        return;
    }
    uint kv_head = head / groups;
    if (kv_head >= num_key_value_heads) {
        return;
    }
    uint query_start = head * head_dim;
    uint kv_vector_len = num_key_value_heads * head_dim;
    uint key_start = (row * kv_vector_len) + (kv_head * head_dim);
    float dot = 0.0;
    for (uint offset = 0; offset < head_dim; offset++) {
        dot += query[query_start + offset] * keys[key_start + offset];
    }
    scores[(head * row_count) + row] = dot * score_scale;
}

kernel void attention_scores_f16(
    device const float* query [[buffer(0)]],
    device const half* keys [[buffer(1)]],
    constant uint& row_count [[buffer(2)]],
    constant uint& num_attention_heads [[buffer(3)]],
    constant uint& num_key_value_heads [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& groups [[buffer(6)]],
    constant float& score_scale [[buffer(7)]],
    device float* scores [[buffer(8)]],
    uint2 id [[thread_position_in_grid]]
) {
    uint row = id.x;
    uint head = id.y;
    if (row >= row_count || head >= num_attention_heads) {
        return;
    }
    uint kv_head = head / groups;
    if (kv_head >= num_key_value_heads) {
        return;
    }
    uint query_start = head * head_dim;
    uint kv_vector_len = num_key_value_heads * head_dim;
    uint key_start = (row * kv_vector_len) + (kv_head * head_dim);
    float dot = 0.0;
    for (uint offset = 0; offset < head_dim; offset++) {
        dot += query[query_start + offset] * float(keys[key_start + offset]);
    }
    scores[(head * row_count) + row] = dot * score_scale;
}

kernel void softmax_rows_f32(
    device const float* scores [[buffer(0)]],
    constant uint& row_count [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& thread_count [[buffer(3)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint head [[threadgroup_position_in_grid]],
    uint thread_id [[thread_index_in_threadgroup]]
) {
    uint row_start = head * row_count;
    float local_max = -INFINITY;
    for (uint row = thread_id; row < row_count; row += thread_count) {
        local_max = max(local_max, scores[row_start + row]);
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
    for (uint row = thread_id; row < row_count; row += thread_count) {
        float probability = exp(scores[row_start + row] - max_score);
        output[row_start + row] = probability;
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
    for (uint row = thread_id; row < row_count; row += thread_count) {
        output[row_start + row] *= inv_denominator;
    }
}

kernel void attention_weighted_sum_f32(
    device const float* values [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    constant uint& row_count [[buffer(2)]],
    constant uint& num_attention_heads [[buffer(3)]],
    constant uint& num_key_value_heads [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& groups [[buffer(6)]],
    device float* output [[buffer(7)]],
    uint2 id [[thread_position_in_grid]]
) {
    uint offset = id.x;
    uint head = id.y;
    if (offset >= head_dim || head >= num_attention_heads) {
        return;
    }
    uint kv_head = head / groups;
    if (kv_head >= num_key_value_heads) {
        return;
    }
    uint kv_vector_len = num_key_value_heads * head_dim;
    uint value_offset = (kv_head * head_dim) + offset;
    float sum = 0.0;
    for (uint row = 0; row < row_count; row++) {
        float weight = weights[(head * row_count) + row];
        sum += values[(row * kv_vector_len) + value_offset] * weight;
    }
    output[(head * head_dim) + offset] = sum;
}

kernel void attention_weighted_sum_f16(
    device const half* values [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    constant uint& row_count [[buffer(2)]],
    constant uint& num_attention_heads [[buffer(3)]],
    constant uint& num_key_value_heads [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& groups [[buffer(6)]],
    device float* output [[buffer(7)]],
    uint2 id [[thread_position_in_grid]]
) {
    uint offset = id.x;
    uint head = id.y;
    if (offset >= head_dim || head >= num_attention_heads) {
        return;
    }
    uint kv_head = head / groups;
    if (kv_head >= num_key_value_heads) {
        return;
    }
    uint kv_vector_len = num_key_value_heads * head_dim;
    uint value_offset = (kv_head * head_dim) + offset;
    float sum = 0.0;
    for (uint row = 0; row < row_count; row++) {
        float weight = weights[(head * row_count) + row];
        sum += float(values[(row * kv_vector_len) + value_offset]) * weight;
    }
    output[(head * head_dim) + offset] = sum;
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

kernel void select_head_rows_f16(
    device const half* values [[buffer(0)]],
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
    output[index] = float(values[(row * row_len) + head_start + offset]);
}
