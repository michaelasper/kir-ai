kernel void argmax_f32(
    device const float* logits [[buffer(0)]],
    constant uint& len [[buffer(1)]],
    constant uint& chunk_size [[buffer(2)]],
    device uint* chunk_indices [[buffer(3)]],
    device float* chunk_values [[buffer(4)]],
    uint chunk [[thread_position_in_grid]]
) {
    uint start = chunk * chunk_size;
    if (start >= len) {
        return;
    }
    uint end = min(start + chunk_size, len);
    uint best_index = start;
    float best_value = logits[start];
    for (uint index = start + 1; index < end; index++) {
        float value = logits[index];
        if (value > best_value || (value == best_value && index < best_index)) {
            best_value = value;
            best_index = index;
        }
    }
    chunk_indices[chunk] = best_index;
    chunk_values[chunk] = best_value;
}

constant uint MAX_TOP_K = 64;
constant uint INVALID_TOP_K_INDEX = 0xffffffff;
constant float NEGATIVE_MAX_FLOAT = -3.4028234663852886e38f;

kernel void top_k_f32(
    device const float* logits [[buffer(0)]],
    constant uint& len [[buffer(1)]],
    constant uint& chunk_size [[buffer(2)]],
    constant uint& top_k [[buffer(3)]],
    device uint* output_indices [[buffer(4)]],
    device float* output_values [[buffer(5)]],
    uint chunk [[thread_position_in_grid]]
) {
    uint output_offset = chunk * top_k;
    for (uint rank = 0; rank < top_k; rank++) {
        output_indices[output_offset + rank] = INVALID_TOP_K_INDEX;
        output_values[output_offset + rank] = NEGATIVE_MAX_FLOAT;
    }

    uint start = chunk * chunk_size;
    if (start >= len || top_k == 0 || top_k > MAX_TOP_K) {
        return;
    }
    uint end = min(start + chunk_size, len);
    uint best_indices[MAX_TOP_K];
    float best_values[MAX_TOP_K];
    for (uint rank = 0; rank < top_k; rank++) {
        best_indices[rank] = INVALID_TOP_K_INDEX;
        best_values[rank] = NEGATIVE_MAX_FLOAT;
    }

    for (uint index = start; index < end; index++) {
        float value = logits[index];
        for (uint rank = 0; rank < top_k; rank++) {
            uint current_index = best_indices[rank];
            float current_value = best_values[rank];
            if (current_index == INVALID_TOP_K_INDEX ||
                value > current_value ||
                (value == current_value && index < current_index)) {
                for (uint shift = top_k - 1; shift > rank; shift--) {
                    best_indices[shift] = best_indices[shift - 1];
                    best_values[shift] = best_values[shift - 1];
                }
                best_indices[rank] = index;
                best_values[rank] = value;
                break;
            }
        }
    }

    for (uint rank = 0; rank < top_k; rank++) {
        output_indices[output_offset + rank] = best_indices[rank];
        output_values[output_offset + rank] = best_values[rank];
    }
}