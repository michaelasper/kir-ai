use llm_backend::native::{
    CpuNativeMatvecBackend, NativeMatvecBackend, NativeRowMajorMatrix, Q8_0_BLOCK_SIZE,
    Q8RowMajorMatrix,
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

fn push_q8_0_block(bytes: &mut Vec<u8>, scale_bits: u16, active_quants: &[i8], pad: i8) {
    assert!(active_quants.len() <= Q8_0_BLOCK_SIZE);
    bytes.extend_from_slice(&scale_bits.to_le_bytes());
    for idx in 0..Q8_0_BLOCK_SIZE {
        let quant = active_quants.get(idx).copied().unwrap_or(pad);
        bytes.push(quant as u8);
    }
}
