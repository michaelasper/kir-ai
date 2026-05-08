use llm_metal::MetalDevice;

#[test]
fn metal_vector_add_matches_cpu_reference() {
    let Some(device) = MetalDevice::system_default() else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };

    let output = device
        .add_f32(&[1.0, 2.5, -3.0, 8.0], &[4.0, -1.5, 3.0, 0.25])
        .expect("metal add succeeds");

    assert_eq!(output, vec![5.0, 1.0, 0.0, 8.25]);
}
