use llm_backend::native::{
    CpuNativeMatvecBackend, NativeMatvecBackend, NativeRowMajorMatrix, Q4_0_BLOCK_BYTE_LEN,
    Q4_0_BLOCK_SIZE, Q4RowMajorMatrix, Q8_0_BLOCK_SIZE, Q8RowMajorMatrix,
};

#[test]
fn q8_0_matvec_dequantizes_blocks_during_dot_product() {
    let mut bytes = Vec::new();
    push_q8_0_block(&mut bytes, 0x3800, &[1, -2, 0, 4], 0);
    push_q8_0_block(&mut bytes, 0x4000, &[-1, 3, 1, 0], 0);
    let matrix = Q8RowMajorMatrix::from_blocks(2, Q8_0_BLOCK_SIZE, &bytes).expect("matrix");
    let mut input = vec![0.0; Q8_0_BLOCK_SIZE];
    input[..4].copy_from_slice(&[2.0, -3.0, 5.0, 0.5]);

    let output = matrix.matvec_f32(&input).expect("matvec");

    assert_eq!(output, vec![5.0, -12.0]);
}

#[test]
fn q8_0_matvec_handles_tail_columns_without_reading_padding() {
    let mut bytes = Vec::new();
    push_q8_0_block(&mut bytes, 0x3c00, &[1; Q8_0_BLOCK_SIZE], 0);
    push_q8_0_block(&mut bytes, 0x3800, &[4, -2, 1], 127);
    push_q8_0_block(&mut bytes, 0x3400, &[-2; Q8_0_BLOCK_SIZE], 0);
    push_q8_0_block(&mut bytes, 0x3c00, &[-3, 4, 0], -128);
    let matrix = Q8RowMajorMatrix::from_blocks(2, Q8_0_BLOCK_SIZE + 3, &bytes).expect("matrix");
    let input = vec![1.0; Q8_0_BLOCK_SIZE + 3];

    let output = matrix.matvec_f32(&input).expect("matvec");

    assert_eq!(output, vec![33.5, -15.0]);
}

#[test]
fn q8_0_matrix_rejects_mismatched_block_bytes() {
    let mut bytes = Vec::new();
    push_q8_0_block(&mut bytes, 0x3c00, &[1; Q8_0_BLOCK_SIZE], 0);

    let err = Q8RowMajorMatrix::from_blocks(1, Q8_0_BLOCK_SIZE + 1, &bytes)
        .expect_err("missing tail block fails");

    assert_eq!(
        err.message(),
        "Q8_0 row-major matrix byte length 34 does not match rows 1 * blocks_per_row 2 * block bytes 34"
    );
}

#[test]
fn q8_0_matvec_validates_input_and_output_shapes() {
    let mut bytes = Vec::new();
    push_q8_0_block(&mut bytes, 0x3c00, &[1; Q8_0_BLOCK_SIZE], 0);
    push_q8_0_block(&mut bytes, 0x3c00, &[1; Q8_0_BLOCK_SIZE], 0);
    let matrix = Q8RowMajorMatrix::from_blocks(2, Q8_0_BLOCK_SIZE, &bytes).expect("matrix");

    let input_err = matrix
        .matvec_f32(&[1.0; Q8_0_BLOCK_SIZE - 1])
        .expect_err("input mismatch fails");
    let mut output = [0.0];
    let output_err = matrix
        .matvec_f32_in_place(&[1.0; Q8_0_BLOCK_SIZE], &mut output)
        .expect_err("output mismatch fails");

    assert_eq!(
        input_err.message(),
        "Q8_0 matvec input length 31 does not match columns 32"
    );
    assert_eq!(
        output_err.message(),
        "output buffer too small for Q8_0 matvec"
    );
}

#[test]
fn q8_0_matvec_rejects_non_finite_block_scale() {
    let mut bytes = Vec::new();
    push_q8_0_block(&mut bytes, 0x7c00, &[1; Q8_0_BLOCK_SIZE], 0);
    let matrix = Q8RowMajorMatrix::from_blocks(1, Q8_0_BLOCK_SIZE, &bytes).expect("matrix");

    let err = matrix
        .matvec_f32(&[1.0; Q8_0_BLOCK_SIZE])
        .expect_err("infinite scale fails");

    assert_eq!(err.message(), "Q8_0 block scale must be finite");
}

#[tokio::test]
async fn native_matvec_dispatch_executes_declared_q8_0_weights() {
    let mut bytes = Vec::new();
    push_q8_0_block(&mut bytes, 0x3c00, &[2, -1, 0, 3], 0);
    push_q8_0_block(&mut bytes, 0x3800, &[-4, 0, 2, 1], 0);
    let matrix = Q8RowMajorMatrix::from_blocks(2, Q8_0_BLOCK_SIZE, &bytes).expect("matrix");
    let mut input = vec![0.0; Q8_0_BLOCK_SIZE];
    input[..4].copy_from_slice(&[1.5, -2.0, 5.0, 0.5]);
    let mut output = [0.0; 2];

    CpuNativeMatvecBackend
        .matvec_row_major_weights_f32_in_place(
            &input,
            NativeRowMajorMatrix::Q8_0(matrix),
            &mut output,
        )
        .await
        .expect("native Q8_0 matvec");

    assert_eq!(output, [6.5, 2.25]);
}

#[test]
fn q4_0_matvec_dequantizes_blocks_during_dot_product() {
    let mut row0 = [0_i8; Q4_0_BLOCK_SIZE];
    row0[..4].copy_from_slice(&[1, -2, 0, 4]);
    row0[16..18].copy_from_slice(&[3, -1]);
    let mut row1 = [0_i8; Q4_0_BLOCK_SIZE];
    row1[..4].copy_from_slice(&[-1, 3, 1, 0]);
    row1[16..18].copy_from_slice(&[2, -4]);
    let mut bytes = Vec::new();
    push_q4_0_block(&mut bytes, 0x3800, &row0, 0);
    push_q4_0_block(&mut bytes, 0x4000, &row1, 0);
    let matrix = Q4RowMajorMatrix::from_blocks(2, Q4_0_BLOCK_SIZE, &bytes).expect("matrix");
    let mut input = vec![0.0; Q4_0_BLOCK_SIZE];
    input[..4].copy_from_slice(&[2.0, -3.0, 5.0, 0.5]);
    input[16..18].copy_from_slice(&[4.0, -2.0]);

    let output = matrix.matvec_f32(&input).expect("matvec");

    assert_eq!(output, vec![12.0, 20.0]);
}

#[test]
fn q4_0_matvec_handles_tail_columns_without_reading_padding() {
    let mut row0_tail = [0_i8; 19];
    row0_tail[..3].copy_from_slice(&[4, -2, 1]);
    row0_tail[16..19].copy_from_slice(&[3, -1, 2]);
    let mut row1_tail = [0_i8; 19];
    row1_tail[..3].copy_from_slice(&[-3, 4, 0]);
    row1_tail[16..19].copy_from_slice(&[2, -2, 1]);
    let mut bytes = Vec::new();
    push_q4_0_block(&mut bytes, 0x3c00, &[1; Q4_0_BLOCK_SIZE], 0);
    push_q4_0_block(&mut bytes, 0x3800, &row0_tail, 7);
    push_q4_0_block(&mut bytes, 0x3400, &[-2; Q4_0_BLOCK_SIZE], 0);
    push_q4_0_block(&mut bytes, 0x3c00, &row1_tail, -8);
    let matrix = Q4RowMajorMatrix::from_blocks(2, Q4_0_BLOCK_SIZE + 19, &bytes).expect("matrix");
    let input = vec![1.0; Q4_0_BLOCK_SIZE + 19];

    let output = matrix.matvec_f32(&input).expect("matvec");

    assert_eq!(output, vec![35.5, -14.0]);
}

#[test]
fn q4_0_matrix_rejects_mismatched_block_bytes() {
    let mut bytes = Vec::new();
    push_q4_0_block(&mut bytes, 0x3c00, &[1; Q4_0_BLOCK_SIZE], 0);

    let err = Q4RowMajorMatrix::from_blocks(1, Q4_0_BLOCK_SIZE + 1, &bytes)
        .expect_err("missing tail block fails");

    assert_eq!(
        err.message(),
        format!(
            "Q4_0 row-major matrix byte length {} does not match rows 1 * blocks_per_row 2 * block bytes {Q4_0_BLOCK_BYTE_LEN}",
            Q4_0_BLOCK_BYTE_LEN
        )
    );
}

#[test]
fn q4_0_matvec_rejects_non_finite_block_scale() {
    let mut bytes = Vec::new();
    push_q4_0_block(&mut bytes, 0x7c00, &[1; Q4_0_BLOCK_SIZE], 0);
    let matrix = Q4RowMajorMatrix::from_blocks(1, Q4_0_BLOCK_SIZE, &bytes).expect("matrix");

    let err = matrix
        .matvec_f32(&[1.0; Q4_0_BLOCK_SIZE])
        .expect_err("infinite scale fails");

    assert_eq!(err.message(), "Q4_0 block scale must be finite");
}

#[tokio::test]
async fn native_matvec_dispatch_executes_declared_q4_0_weights() {
    let mut row0 = [0_i8; Q4_0_BLOCK_SIZE];
    row0[..4].copy_from_slice(&[2, -1, 0, 3]);
    let mut row1 = [0_i8; Q4_0_BLOCK_SIZE];
    row1[..4].copy_from_slice(&[-4, 0, 2, 1]);
    let mut bytes = Vec::new();
    push_q4_0_block(&mut bytes, 0x3c00, &row0, 0);
    push_q4_0_block(&mut bytes, 0x3800, &row1, 0);
    let matrix = Q4RowMajorMatrix::from_blocks(2, Q4_0_BLOCK_SIZE, &bytes).expect("matrix");
    let mut input = vec![0.0; Q4_0_BLOCK_SIZE];
    input[..4].copy_from_slice(&[1.5, -2.0, 5.0, 0.5]);
    let mut output = [0.0; 2];

    CpuNativeMatvecBackend
        .matvec_row_major_weights_f32_in_place(
            &input,
            NativeRowMajorMatrix::Q4_0(matrix),
            &mut output,
        )
        .await
        .expect("native Q4_0 matvec");

    assert_eq!(output, [6.5, 2.25]);
}

fn push_q8_0_block(bytes: &mut Vec<u8>, scale_bits: u16, active_quants: &[i8], pad: i8) {
    assert!(active_quants.len() <= Q8_0_BLOCK_SIZE);
    bytes.extend_from_slice(&scale_bits.to_le_bytes());
    for idx in 0..Q8_0_BLOCK_SIZE {
        let quant = active_quants.get(idx).copied().unwrap_or(pad);
        bytes.push(quant as u8);
    }
}

fn push_q4_0_block(bytes: &mut Vec<u8>, scale_bits: u16, active_quants: &[i8], pad: i8) {
    assert!(active_quants.len() <= Q4_0_BLOCK_SIZE);
    bytes.extend_from_slice(&scale_bits.to_le_bytes());
    for idx in 0..Q4_0_BLOCK_SIZE / 2 {
        let low = q4_0_nibble(active_quants.get(idx).copied().unwrap_or(pad));
        let high = q4_0_nibble(
            active_quants
                .get(idx + Q4_0_BLOCK_SIZE / 2)
                .copied()
                .unwrap_or(pad),
        );
        bytes.push(low | (high << 4));
    }
}

fn q4_0_nibble(value: i8) -> u8 {
    assert!((-8..=7).contains(&value));
    (value + 8) as u8
}
