use llm_backend::{
    QwenFullAttentionDims, QwenFullAttentionSequenceConfig, QwenFullAttentionSequenceParts,
    QwenLinearAttentionDims, QwenLinearAttentionSequenceParts,
    qwen_full_attention_first_token_from_parts, qwen_full_attention_sequence_from_parts,
    qwen_linear_attention_first_token_from_parts, qwen_linear_attention_sequence_from_parts,
};
use llm_backend::{
    matvec_row_major_f32, qwen_rms_norm_f32, rms_norm_f32, silu_f32, softmax_top_k_f32,
    swiglu_mlp_f32,
};

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

#[test]
fn qwen_linear_attention_sequence_updates_recurrent_state() {
    let dims = QwenLinearAttentionDims {
        hidden_size: 2,
        num_key_heads: 1,
        num_value_heads: 1,
        key_head_dim: 1,
        value_head_dim: 2,
        conv_kernel_size: 1,
        rms_norm_eps: 0.0,
    };
    let qkv = vec![vec![1.0, 1.0, 2.0, 4.0], vec![1.0, 1.0, 10.0, 0.0]];
    let z = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
    let b = vec![vec![0.0], vec![0.0]];
    let a = vec![vec![0.0], vec![0.0]];
    let dt_bias = vec![0.0];
    let a_log = vec![0.0];
    let conv1d_weight = vec![1.0, 1.0, 1.0, 1.0];
    let norm_weight = vec![1.0, 1.0];
    let out_proj_weight = vec![1.0, 0.0, 0.0, 1.0];

    let output = qwen_linear_attention_sequence_from_parts(
        &dims,
        &QwenLinearAttentionSequenceParts {
            qkv: &qkv,
            z: &z,
            b: &b,
            a: &a,
            dt_bias: &dt_bias,
            a_log: &a_log,
            conv1d_weight: &conv1d_weight,
            norm_weight: &norm_weight,
            out_proj_weight: &out_proj_weight,
        },
    )
    .expect("linear attention sequence");

    let q0 = l2_scalar(silu_f32(1.0));
    let k0 = l2_scalar(silu_f32(1.0));
    let v0 = [silu_f32(2.0), silu_f32(4.0)];
    let q1 = l2_scalar(silu_f32(1.0));
    let k1 = l2_scalar(silu_f32(1.0));
    let v1 = [silu_f32(10.0), silu_f32(0.0)];
    let beta = 0.5;
    let decay = (-std::f32::consts::LN_2).exp();
    let state0 = [k0 * v0[0] * beta, k0 * v0[1] * beta];
    let core0 = [state0[0] * q0, state0[1] * q0];
    let state1_before = [state0[0] * decay, state0[1] * decay];
    let memory1 = [state1_before[0] * k1, state1_before[1] * k1];
    let delta1 = [(v1[0] - memory1[0]) * beta, (v1[1] - memory1[1]) * beta];
    let state1 = [
        state1_before[0] + k1 * delta1[0],
        state1_before[1] + k1 * delta1[1],
    ];
    let core1 = [state1[0] * q1, state1[1] * q1];
    let gate = silu_f32(1.0);
    let expected = [rms_pair(core0, gate), rms_pair(core1, gate)];

    assert_close(&output[0], &expected[0], 1e-6);
    assert_close(&output[1], &expected[1], 1e-6);
}

#[test]
fn qwen_full_attention_first_token_matches_single_key_reference() {
    let dims = QwenFullAttentionDims {
        hidden_size: 1,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 1,
    };

    let output = qwen_full_attention_first_token_from_parts(&dims, &[0.0, 0.0], &[8.0], &[3.0])
        .expect("full attention output");

    assert_close(&output, &[12.0], 1e-6);
}

#[test]
fn qwen_full_attention_sequence_applies_rope_and_causal_softmax() {
    let dims = QwenFullAttentionDims {
        hidden_size: 2,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 2,
    };
    let q_proj = vec![vec![1.0, 0.0, 0.0, 0.0], vec![1.0, 0.0, 0.0, 0.0]];
    let k_proj = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
    let v_proj = vec![vec![2.0, 0.0], vec![0.0, 4.0]];
    let q_norm_weight = vec![0.0, 0.0];
    let k_norm_weight = vec![0.0, 0.0];
    let o_proj_weight = vec![1.0, 0.0, 0.0, 1.0];

    let output = qwen_full_attention_sequence_from_parts(
        &dims,
        &QwenFullAttentionSequenceParts {
            q_proj: &q_proj,
            k_proj: &k_proj,
            v_proj: &v_proj,
            q_norm_weight: &q_norm_weight,
            k_norm_weight: &k_norm_weight,
            o_proj_weight: &o_proj_weight,
        },
        QwenFullAttentionSequenceConfig {
            rms_norm_eps: 0.0,
            rope_theta: 10_000.0,
            partial_rotary_factor: 1.0,
        },
    )
    .expect("full attention sequence");

    let score0 = 2.0_f32.sqrt() * 1.0_f32.cos();
    let score1 = 2.0_f32.sqrt();
    let max_score = score0.max(score1);
    let exp0 = (score0 - max_score).exp();
    let exp1 = (score1 - max_score).exp();
    let sum = exp0 + exp1;
    let w0 = exp0 / sum;
    let w1 = exp1 / sum;

    assert_close(&output[0], &[1.0, 0.0], 1e-6);
    assert_close(&output[1], &[w0, 2.0 * w1], 1e-6);
}

fn rms_pair(values: [f32; 2], gate: f32) -> Vec<f32> {
    let rms = ((values[0] * values[0] + values[1] * values[1]) / 2.0).sqrt();
    vec![values[0] / rms * gate, values[1] / rms * gate]
}

fn l2_scalar(value: f32) -> f32 {
    value / (value * value + 1e-6).sqrt()
}

#[test]
fn softmax_top_k_returns_normalized_selected_weights() {
    let selected = softmax_top_k_f32(&[1.0, 3.0, 2.0, -4.0], 2).expect("top k");

    assert_eq!(selected[0].index, 1);
    assert_eq!(selected[1].index, 2);
    assert_close(
        &[selected[0].weight, selected[1].weight],
        &[0.7310586, 0.26894143],
        1e-6,
    );
}

#[test]
fn swiglu_mlp_matches_reference_calculation() {
    let output =
        swiglu_mlp_f32(&[1.0, 2.0], &[1.0, 0.0], &[0.0, 1.0], &[1.0, 2.0], 1).expect("swiglu mlp");

    assert_close(&output, &[1.4621172, 2.9242344], 1e-6);
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
