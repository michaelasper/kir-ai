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

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual} expected {expected}"
        );
    }
}
