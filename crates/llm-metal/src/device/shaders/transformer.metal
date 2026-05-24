kernel void rms_norm_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    constant uint& len [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    constant float& weight_offset [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& thread_count [[buffer(6)]],
    threadgroup float* partial_sums [[threadgroup(0)]],
    uint thread_id [[thread_index_in_threadgroup]]
) {
    float sum = 0.0;
    for (uint index = thread_id; index < len; index += thread_count) {
        float value = input[index];
        sum += value * value;
    }
    partial_sums[thread_id] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = thread_count >> 1; stride > 0; stride >>= 1) {
        if (thread_id < stride) {
            partial_sums[thread_id] += partial_sums[thread_id + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_rms = rsqrt((partial_sums[0] / float(len)) + eps);
    for (uint index = thread_id; index < len; index += thread_count) {
        output[index] = input[index] * inv_rms * (weight[index] + weight_offset);
    }
}

kernel void linear_attention_conv1d_silu_f32(
    device const float* window [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    constant uint& conv_dim [[buffer(2)]],
    constant uint& kernel_size [[buffer(3)]],
    device float* output [[buffer(4)]],
    uint channel [[thread_position_in_grid]]
) {
    if (channel >= conv_dim) {
        return;
    }
    float mixed = 0.0;
    for (uint kernel_index = 0; kernel_index < kernel_size; kernel_index++) {
        mixed += window[(kernel_index * conv_dim) + channel]
            * weights[(channel * kernel_size) + kernel_index];
    }
    output[channel] = mixed / (1.0 + exp(-mixed));
}

kernel void linear_attention_recurrent_update_f32(
    device const float* state [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device const float* memory [[buffer(3)]],
    constant float& beta [[buffer(4)]],
    constant float& decay [[buffer(5)]],
    constant uint& value_head_dim [[buffer(6)]],
    constant uint& element_count [[buffer(7)]],
    device float* output [[buffer(8)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= element_count) {
        return;
    }
    uint key_index = index / value_head_dim;
    uint value_index = index % value_head_dim;
    float delta = (value[value_index] - memory[value_index]) * beta;
    output[index] = (state[index] * decay) + (key[key_index] * delta);
}

kernel void linear_attention_recurrent_update_state_f32(
    device float* state [[buffer(0)]],
    constant uint& state_offset [[buffer(1)]],
    device const float* key [[buffer(2)]],
    device const float* value [[buffer(3)]],
    device const float* memory [[buffer(4)]],
    constant float& beta [[buffer(5)]],
    constant float& decay [[buffer(6)]],
    constant uint& value_head_dim [[buffer(7)]],
    constant uint& element_count [[buffer(8)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= element_count) {
        return;
    }
    uint key_index = index / value_head_dim;
    uint value_index = index % value_head_dim;
    uint state_index = state_offset + index;
    float delta = (value[value_index] - memory[value_index]) * beta;
    state[state_index] = (state[state_index] * decay) + (key[key_index] * delta);
}
