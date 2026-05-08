use llm_backend::{QwenLinearAttentionDims, qwen_linear_attention_first_token_from_parts};
use llm_backend::{matvec_row_major_f32, qwen_rms_norm_f32, rms_norm_f32, silu_f32};

#[test]
fn rms_norm_matches_reference_calculation() {
    let output = rms_norm_f32(&[3.0, 4.0], &[1.0, 2.0], 0.0).expect("rms norm");

    assert_close(&output, &[0.84852815, 2.2627418], 1e-6);
}

#[test]
fn rms_norm_rejects_mismatched_weight_shape() {
    let err = rms_norm_f32(&[1.0, 2.0], &[1.0], 1e-6).expect_err("shape fails");

    assert!(err.to_string().contains("same length"));
}

#[test]
fn qwen_rms_norm_uses_one_centered_weights() {
    let output = qwen_rms_norm_f32(&[3.0, 4.0], &[0.0, 1.0], 0.0).expect("qwen rms norm");

    assert_close(&output, &[0.84852815, 2.2627418], 1e-6);
}

#[test]
fn matvec_row_major_matches_reference_calculation() {
    let output = matvec_row_major_f32(
        &[1.0, 2.0, 3.0],
        &[
            1.0, 0.0, 0.0, //
            0.0, 1.0, 1.0,
        ],
        2,
        3,
    )
    .expect("matvec");

    assert_eq!(output, vec![1.0, 5.0]);
}

#[test]
fn silu_matches_reference_values() {
    assert_close(
        &[silu_f32(0.0), silu_f32(2.0), silu_f32(-2.0)],
        &[0.0, 1.761594, -0.23840584],
        1e-6,
    );
}

#[test]
fn qwen_linear_attention_first_token_matches_simplified_reference() {
    let dims = QwenLinearAttentionDims {
        hidden_size: 1,
        num_key_heads: 1,
        num_value_heads: 1,
        key_head_dim: 1,
        value_head_dim: 1,
        conv_kernel_size: 1,
        rms_norm_eps: 0.0,
    };

    let output = qwen_linear_attention_first_token_from_parts(
        &dims,
        &[1.0, 1.0, 4.0],
        &[1.0],
        &[0.0],
        &[1.0, 1.0, 1.0],
        &[1.0],
        &[2.0],
    )
    .expect("linear attention output");

    assert_close(&output, &[1.4621172], 1e-6);
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
