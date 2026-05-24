use llm_kv_cache::prototype_quantization::{
    KvQuantizationBits, KvQuantizationIdentity, KvQuantizationPrototypeError, KvQuantizationScheme,
    KvQuantizationScope, KvValueQuantizerPrototype, evaluate_kv_quantization_fixture,
    tiny_gemma_value_fixture, tiny_qwen_value_fixture,
};

fn calibration_values() -> Vec<f32> {
    vec![
        -1.25, -1.20, -0.55, -0.52, -0.10, -0.08, 0.30, 0.33, -1.18, -1.22, -0.50, -0.57, -0.09,
        -0.07, 0.31, 0.35, 0.80, 0.84, 1.10, 1.13, 1.60, 1.62, 2.05, 2.10, 0.82, 0.86, 1.08, 1.14,
        1.58, 1.64, 2.00, 2.12,
    ]
}

#[test]
fn lloyd_max_quantizes_value_block_with_random_rotation() {
    let values = calibration_values();
    let identity = KvQuantizationIdentity::new("qwen3-tiny", 0, 0, 4, 8);
    let quantizer = KvValueQuantizerPrototype::train_lloyd_max(
        identity,
        KvQuantizationBits::Four,
        Some(0x292_u64),
        &values,
        12,
    )
    .expect("codebook trains");

    let block = quantizer.quantize(&values).expect("block quantizes");
    let decoded = block.dequantize(&quantizer).expect("block dequantizes");

    assert_eq!(decoded.len(), values.len());
    assert_eq!(
        block.metadata().scheme(),
        KvQuantizationScheme::LloydMaxCodebook
    );
    assert_eq!(block.metadata().bits(), KvQuantizationBits::Four);
    assert_eq!(block.payload_bytes(), 16);
    assert_eq!(
        block
            .metadata()
            .rotation()
            .expect("rotation metadata")
            .scope(),
        KvQuantizationScope::LayerHead
    );
    assert_eq!(
        block
            .metadata()
            .codebook()
            .expect("codebook metadata")
            .scope(),
        KvQuantizationScope::LayerHead
    );
    assert!(
        max_abs_delta(&values, &decoded) < 0.08,
        "decoded values should stay close to source values"
    );
}

#[test]
fn uniform_quantizer_supports_three_bit_payload_without_rotation() {
    let values = calibration_values();
    let identity = KvQuantizationIdentity::new("gemma-tiny", 0, 0, 2, 8);
    let quantizer = KvValueQuantizerPrototype::uniform(identity, KvQuantizationBits::Three, None)
        .expect("uniform quantizer builds");

    let block = quantizer.quantize(&values).expect("block quantizes");
    let decoded = block.dequantize(&quantizer).expect("block dequantizes");

    assert_eq!(decoded.len(), values.len());
    assert_eq!(
        block.metadata().scheme(),
        KvQuantizationScheme::UniformAffine
    );
    assert_eq!(block.metadata().bits(), KvQuantizationBits::Three);
    assert_eq!(block.payload_bytes(), 12);
    assert!(block.metadata().rotation().is_none());
    assert!(
        max_abs_delta(&values, &decoded) < 0.25,
        "3-bit uniform baseline should still reconstruct the deterministic block"
    );
}

#[test]
fn dequantize_rejects_mismatched_codebook_or_rotation_metadata() {
    let values = calibration_values();
    let identity = KvQuantizationIdentity::new("qwen3-tiny", 1, 2, 3, 8);
    let quantizer = KvValueQuantizerPrototype::train_lloyd_max(
        identity.clone(),
        KvQuantizationBits::Four,
        Some(123),
        &values,
        8,
    )
    .expect("codebook trains");
    let changed_rotation = KvValueQuantizerPrototype::train_lloyd_max(
        identity.clone(),
        KvQuantizationBits::Four,
        Some(124),
        &values,
        8,
    )
    .expect("codebook trains with changed rotation");
    let mut changed_training_values = values.clone();
    changed_training_values[0] = 4.0;
    let changed_codebook = KvValueQuantizerPrototype::train_lloyd_max(
        identity,
        KvQuantizationBits::Four,
        Some(123),
        &changed_training_values,
        8,
    )
    .expect("changed codebook trains");

    let block = quantizer.quantize(&values).expect("block quantizes");

    assert_eq!(
        block
            .dequantize(&changed_rotation)
            .expect_err("rotation mismatch fails"),
        KvQuantizationPrototypeError::MetadataMismatch { field: "rotation" }
    );
    assert_eq!(
        block
            .dequantize(&changed_codebook)
            .expect_err("codebook mismatch fails"),
        KvQuantizationPrototypeError::MetadataMismatch { field: "codebook" }
    );
}

#[test]
fn deterministic_qwen_and_gemma_reports_compare_baselines_with_lloyd_max() {
    for fixture in [tiny_qwen_value_fixture(), tiny_gemma_value_fixture()] {
        let report = evaluate_kv_quantization_fixture(&fixture).expect("fixture report evaluates");

        assert_eq!(report.fixture_model_family(), fixture.model_family());
        assert_eq!(report.original_bytes(), (fixture.values().len() * 4) as u64);
        assert_eq!(report.rows().len(), 7);

        let int8 = report
            .row("uniform-int8")
            .expect("INT8 uniform baseline exists");
        let int4 = report
            .row("uniform-int4")
            .expect("INT4 uniform baseline exists");
        let int3 = report
            .row("uniform-int3")
            .expect("3-bit uniform baseline exists");
        let lloyd4 = report
            .row("lloyd-max-int4")
            .expect("INT4 Lloyd-Max comparison exists");
        let rotated_lloyd4 = report
            .row("rotated-lloyd-max-int4")
            .expect("rotated INT4 Lloyd-Max comparison exists");
        let lloyd3 = report
            .row("lloyd-max-int3")
            .expect("3-bit Lloyd-Max comparison exists");
        let rotated_lloyd3 = report
            .row("rotated-lloyd-max-int3")
            .expect("rotated 3-bit Lloyd-Max comparison exists");

        assert!(int8.reconstruction_mse() <= int4.reconstruction_mse());
        assert!(int4.payload_memory_ratio() < 0.13);
        assert!(int3.payload_memory_ratio() < int4.payload_memory_ratio());
        assert!(lloyd4.reconstruction_mse() <= int4.reconstruction_mse());
        assert!(lloyd3.reconstruction_mse() <= int3.reconstruction_mse());
        assert!(rotated_lloyd4.decode_estimated_ops() > lloyd4.decode_estimated_ops());
        assert!(rotated_lloyd3.decode_estimated_ops() > lloyd3.decode_estimated_ops());
        assert!(rotated_lloyd4.attention_output_mse().is_finite());
        assert!(rotated_lloyd3.attention_output_mse().is_finite());
    }
}

fn max_abs_delta(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f32::max)
}
