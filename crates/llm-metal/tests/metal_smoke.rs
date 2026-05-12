use llm_metal::MetalDevice;

#[tokio::test]
async fn metal_vector_add_matches_cpu_reference() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };

    assert!(device.vector_add_thread_execution_width() > 0);
    let mut output = vec![0.0; 4];
    device
        .add_f32(&[1.0, 2.5, -3.0, 8.0], &[4.0, -1.5, 3.0, 0.25], &mut output)
        .await
        .expect("metal add succeeds");

    assert_eq!(output, vec![5.0, 1.0, 0.0, 8.25]);
}

#[tokio::test]
async fn metal_qwen_rms_norm_matches_cpu_reference() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };

    let mut output = vec![0.0; 2];
    device
        .qwen_rms_norm_f32(&[3.0, 4.0], &[0.0, 1.0], 0.0, &mut output)
        .await
        .expect("metal qwen rms norm succeeds");

    assert_close(&output, &[0.84852815, 2.2627418], 1e-6);
}

#[tokio::test]
async fn metal_rms_norm_matches_cpu_reference() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };

    let mut output = vec![0.0; 2];
    device
        .rms_norm_f32(&[3.0, 4.0], &[1.5, 2.5], 0.0, &mut output)
        .await
        .expect("metal rms norm succeeds");

    assert_close(&output, &[1.2727922, 2.828427], 1e-6);
}

#[tokio::test]
async fn metal_rms_norm_handles_inputs_larger_than_one_threadgroup() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };

    let input = (0..2053)
        .map(|index| ((index % 23) as f32 - 11.0) / 7.0)
        .collect::<Vec<_>>();
    let weight = (0..input.len())
        .map(|index| 0.75 + (index % 5) as f32 * 0.125)
        .collect::<Vec<_>>();
    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let inv_rms = (mean_square + 1e-6).sqrt().recip();
    let expected = input
        .iter()
        .zip(&weight)
        .map(|(input, weight)| input * inv_rms * weight)
        .collect::<Vec<_>>();
    let mut output = vec![0.0; input.len()];

    device
        .rms_norm_f32(&input, &weight, 1e-6, &mut output)
        .await
        .expect("metal rms norm succeeds");

    assert_close(&output, &expected, 1e-5);
}

#[tokio::test]
async fn metal_softmax_f32_matches_cpu_reference() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };
    let scores = [1.0, 2.0, -1.0, 0.5];
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponentials = scores
        .iter()
        .map(|score| (score - max).exp())
        .collect::<Vec<_>>();
    let denominator = exponentials.iter().sum::<f32>();
    let expected = exponentials
        .iter()
        .map(|value| value / denominator)
        .collect::<Vec<_>>();

    let mut output = vec![0.0; scores.len()];
    device
        .softmax_f32(&scores, &mut output)
        .await
        .expect("metal softmax succeeds");

    assert_close(&output, &expected, 1e-6);
    assert_close(&[output.iter().sum::<f32>()], &[1.0], 1e-6);
}

#[tokio::test]
async fn metal_softmax_f32_handles_inputs_larger_than_one_threadgroup() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };

    let scores = (0..4097)
        .map(|index| ((index % 113) as f32 - 56.0) / 17.0)
        .collect::<Vec<_>>();
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponentials = scores
        .iter()
        .map(|score| (score - max).exp())
        .collect::<Vec<_>>();
    let denominator = exponentials.iter().sum::<f32>();
    let expected = exponentials
        .iter()
        .map(|value| value / denominator)
        .collect::<Vec<_>>();

    let mut output = vec![0.0; scores.len()];
    device
        .softmax_f32(&scores, &mut output)
        .await
        .expect("metal softmax succeeds");

    assert_close(&output, &expected, 1e-5);
    assert_close(&[output.iter().sum::<f32>()], &[1.0], 1e-5);
}

#[tokio::test]
async fn metal_linear_attention_conv1d_silu_f32_matches_cpu_reference() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };
    let window = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let weights = [0.5, 1.0, -1.0, 0.25, 2.0, -0.5];
    let expected = [silu(4.5), silu(-0.75), silu(3.0)];

    let mut output = vec![0.0; 3];
    device
        .linear_attention_conv1d_silu_f32(&window, &weights, 3, 2, &mut output)
        .await
        .expect("metal linear attention conv1d succeeds");

    assert_close(&output, &expected, 1e-6);
}

#[tokio::test]
async fn metal_weighted_sum_f32_matches_cpu_reference() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };

    let mut output = vec![0.0; 3];
    device
        .weighted_sum_f32(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[0.25, -0.5],
            3,
            &mut output,
        )
        .await
        .expect("metal weighted sum succeeds");

    assert_close(&output, &[-1.75, -2.0, -2.25], 1e-6);
}

#[tokio::test]
async fn metal_linear_attention_recurrent_update_f32_matches_cpu_reference() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };

    let mut output = vec![0.0; 6];
    device
        .linear_attention_recurrent_update_f32(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[0.5, -1.0],
            &[10.0, 20.0, 30.0],
            &[1.0, 2.0, 3.0],
            0.25,
            0.5,
            2,
            3,
            &mut output,
        )
        .await
        .expect("metal recurrent update succeeds");

    assert_close(&output, &[1.625, 3.25, 4.875, -0.25, -2.0, -3.75], 1e-6);
}

#[tokio::test]
async fn metal_linear_attention_recurrent_update_state_f32_reuses_state_buffer() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };
    let state = device
        .new_f32_buffer(&[100.0, 200.0, 1.0, 2.0, 3.0, 4.0, 300.0])
        .expect("state buffer uploads");

    device
        .linear_attention_recurrent_update_f32_buffered_state(
            &state,
            2,
            &[0.5, -1.0],
            &[10.0, 20.0],
            &[1.0, 2.0],
            0.25,
            0.5,
            2,
            2,
        )
        .await
        .expect("buffered recurrent update succeeds");
    let full_state = device.read_f32_buffer(&state).expect("state buffer reads");

    assert_close(
        &full_state,
        &[100.0, 200.0, 1.625, 3.25, -0.75, -2.5, 300.0],
        1e-6,
    );
}

#[tokio::test]
async fn metal_select_head_rows_f32_matches_cpu_reference() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };

    let mut output = vec![0.0; 4];
    device
        .select_head_rows_f32(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            2,
            4,
            1,
            2,
            &mut output,
        )
        .await
        .expect("metal head row selection succeeds");

    assert_close(&output, &[2.0, 3.0, 6.0, 7.0], 1e-6);
}

#[tokio::test]
async fn metal_select_head_rows_f32_reuses_value_buffer() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };
    let values = device
        .new_f32_buffer(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .expect("values buffer uploads");

    let mut output = vec![0.0; 4];
    device
        .select_head_rows_f32_buffered(&values, 2, 4, 1, 2, &mut output)
        .await
        .expect("buffered head row selection succeeds");

    assert_eq!(values.len(), 8);
    assert_eq!(values.byte_len(), 32);
    assert_close(&output, &[2.0, 3.0, 6.0, 7.0], 1e-6);
}

#[tokio::test]
async fn metal_matvec_f32_matches_cpu_reference() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };

    let mut output = vec![0.0; 2];
    device
        .matvec_f32(
            &[1.0, 2.0, 3.0, 4.0, -1.0, 0.5],
            2,
            3,
            &[0.5, -2.0, 4.0],
            &mut output,
        )
        .await
        .expect("metal matvec succeeds");

    assert_close(&output, &[8.5, 6.0], 1e-6);
}

#[tokio::test]
async fn metal_matvec_bf16_f32_matches_cpu_reference() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };
    let matrix = [1.0, 2.0, 3.0, 4.0, -1.0, 0.5].map(f32_to_bf16_bits);

    let mut output = vec![0.0; 2];
    device
        .matvec_bf16_f32(&matrix, 2, 3, &[0.5, -2.0, 4.0], &mut output)
        .await
        .expect("metal bf16 matvec succeeds");

    assert_close(&output, &[8.5, 6.0], 1e-6);
}

#[tokio::test]
async fn metal_buffered_matvec_bf16_f32_reuses_matrix_buffer() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };
    let matrix = [1.0, 2.0, 3.0, 4.0, -1.0, 0.5].map(f32_to_bf16_bits);
    let matrix_buffer = device
        .new_bf16_matrix_buffer(&matrix, 2, 3)
        .expect("bf16 matrix buffer uploads");

    assert_eq!(matrix_buffer.rows(), 2);
    assert_eq!(matrix_buffer.columns(), 3);
    assert_eq!(matrix_buffer.byte_len(), 12);

    let mut first = vec![0.0; 2];
    device
        .matvec_bf16_f32_buffered(&matrix_buffer, &[0.5, -2.0, 4.0], &mut first)
        .await
        .expect("buffered bf16 matvec succeeds");
    let mut second = vec![0.0; 2];
    device
        .matvec_bf16_f32_buffered(&matrix_buffer, &[1.0, 0.0, -1.0], &mut second)
        .await
        .expect("buffered bf16 matvec succeeds again");
    let mut batched = vec![0.0; 4];
    device
        .batched_matvec_bf16_f32_buffered(
            &matrix_buffer,
            &[0.5, -2.0, 4.0, 1.0, 0.0, -1.0],
            2,
            &mut batched,
        )
        .await
        .expect("buffered batched bf16 matvec succeeds");

    assert_close(&first, &[8.5, 6.0], 1e-6);
    assert_close(&second, &[-2.0, 3.5], 1e-6);
    assert_close(&batched, &[8.5, 6.0, -2.0, 3.5], 1e-6);
}

#[tokio::test]
async fn metal_batched_matvec_bf16_f32_matches_cpu_reference() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };
    let matrix = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map(f32_to_bf16_bits);

    let mut output = vec![0.0; 4];
    device
        .batched_matvec_bf16_f32(
            &matrix,
            2,
            3,
            &[1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            2,
            &mut output,
        )
        .await
        .expect("metal batched bf16 matvec succeeds");

    assert_close(&output, &[14.0, 32.0, 10.0, 28.0], 1e-6);
}

#[tokio::test]
async fn metal_argmax_f32_matches_cpu_reference_with_stable_ties() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };
    let mut logits = vec![-1.0; 600];
    logits[42] = 4.5;
    logits[311] = 4.5;
    logits[599] = 3.25;

    let output = device
        .argmax_f32(&logits)
        .await
        .expect("metal argmax succeeds");

    assert_eq!(output.index, 42);
    assert_eq!(output.value, 4.5);
}

#[tokio::test]
async fn metal_top_k_f32_matches_cpu_reference_with_stable_ties() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };
    let mut logits = vec![-10.0; 700];
    logits[7] = 9.0;
    logits[288] = 12.0;
    logits[499] = 12.0;
    logits[612] = 5.0;

    let mut output = vec![
        llm_metal::TopKResult {
            index: 0,
            value: 0.0
        };
        3
    ];
    device
        .top_k_f32(&logits, 3, &mut output)
        .await
        .expect("metal top-k succeeds");

    assert_eq!(output.len(), 3);
    assert_eq!(output[0].index, 288);
    assert_eq!(output[0].value, 12.0);
    assert_eq!(output[1].index, 499);
    assert_eq!(output[1].value, 12.0);
    assert_eq!(output[2].index, 7);
    assert_eq!(output[2].value, 9.0);
}

fn f32_to_bf16_bits(value: f32) -> u16 {
    (value.to_bits() >> 16) as u16
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual} expected {expected}"
        );
    }
}
