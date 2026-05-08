kernel void matvec_f32(
    device const float* matrix [[buffer(0)]],
    device const float* vector [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    device float* output [[buffer(4)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) {
        return;
    }
    float sum = 0.0;
    uint row_offset = row * cols;
    for (uint col = 0; col < cols; col++) {
        sum += matrix[row_offset + col] * vector[col];
    }
    output[row] = sum;
}

kernel void matvec_bf16_f32(
    device const ushort* matrix [[buffer(0)]],
    device const float* vector [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    device float* output [[buffer(4)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) {
        return;
    }
    float sum = 0.0;
    uint row_offset = row * cols;
    for (uint col = 0; col < cols; col++) {
        uint bits = uint(matrix[row_offset + col]) << 16;
        float weight = as_type<float>(bits);
        sum += weight * vector[col];
    }
    output[row] = sum;
}

kernel void batched_matvec_bf16_f32(
    device const ushort* matrix [[buffer(0)]],
    device const float* vectors [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& vector_count [[buffer(4)]],
    device float* output [[buffer(5)]],
    uint2 id [[thread_position_in_grid]]
) {
    uint row = id.x;
    uint vector_index = id.y;
    if (row >= rows || vector_index >= vector_count) {
        return;
    }
    float sum = 0.0;
    uint row_offset = row * cols;
    uint vector_offset = vector_index * cols;
    for (uint col = 0; col < cols; col++) {
        uint bits = uint(matrix[row_offset + col]) << 16;
        float weight = as_type<float>(bits);
        sum += weight * vectors[vector_offset + col];
    }
    output[(vector_index * rows) + row] = sum;
}

