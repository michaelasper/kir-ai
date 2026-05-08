use llm_metal::MetalDevice;

#[test]
fn metal_vector_add_matches_cpu_reference() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };

    assert!(device.vector_add_thread_execution_width() > 0);
    let output = device
        .add_f32(&[1.0, 2.5, -3.0, 8.0], &[4.0, -1.5, 3.0, 0.25])
        .expect("metal add succeeds");

    assert_eq!(output, vec![5.0, 1.0, 0.0, 8.25]);
}

#[test]
fn metal_qwen_rms_norm_matches_cpu_reference() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };

    let output = device
        .qwen_rms_norm_f32(&[3.0, 4.0], &[0.0, 1.0], 0.0)
        .expect("metal qwen rms norm succeeds");

    assert_close(&output, &[0.84852815, 2.2627418], 1e-6);
}

#[test]
fn metal_matvec_f32_matches_cpu_reference() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };

    let output = device
        .matvec_f32(&[1.0, 2.0, 3.0, 4.0, -1.0, 0.5], 2, 3, &[0.5, -2.0, 4.0])
        .expect("metal matvec succeeds");

    assert_close(&output, &[8.5, 6.0], 1e-6);
}

#[test]
fn metal_matvec_bf16_f32_matches_cpu_reference() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };
    let matrix = [1.0, 2.0, 3.0, 4.0, -1.0, 0.5].map(f32_to_bf16_bits);

    let output = device
        .matvec_bf16_f32(&matrix, 2, 3, &[0.5, -2.0, 4.0])
        .expect("metal bf16 matvec succeeds");

    assert_close(&output, &[8.5, 6.0], 1e-6);
}

#[test]
fn metal_batched_matvec_bf16_f32_matches_cpu_reference() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };
    let matrix = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map(f32_to_bf16_bits);

    let output = device
        .batched_matvec_bf16_f32(&matrix, 2, 3, &[1.0, 2.0, 3.0, 3.0, 2.0, 1.0], 2)
        .expect("metal batched bf16 matvec succeeds");

    assert_close(&output, &[14.0, 32.0, 10.0, 28.0], 1e-6);
}

#[test]
fn metal_argmax_f32_matches_cpu_reference_with_stable_ties() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };
    let mut logits = vec![-1.0; 600];
    logits[42] = 4.5;
    logits[311] = 4.5;
    logits[599] = 3.25;

    let output = device.argmax_f32(&logits).expect("metal argmax succeeds");

    assert_eq!(output.index, 42);
    assert_eq!(output.value, 4.5);
}

#[test]
fn metal_top_k_f32_matches_cpu_reference_with_stable_ties() {
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

    let output = device.top_k_f32(&logits, 3).expect("metal top-k succeeds");

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

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual} expected {expected}"
        );
    }
}
