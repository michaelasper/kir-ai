use llm_backend::{
    BackendCacheContext, QwenFullAttentionDims, QwenFullAttentionSequenceConfig,
    QwenFullAttentionSequenceParts, QwenFullAttentionStepParts, QwenLinearAttentionDims,
    QwenLinearAttentionSequenceParts, QwenLinearAttentionStepParts,
    qwen_full_attention_first_token_from_parts, qwen_full_attention_sequence_from_parts,
    qwen_full_attention_sequence_with_cache_from_parts,
    qwen_full_attention_step_with_cache_from_parts, qwen_linear_attention_first_token_from_parts,
    qwen_linear_attention_sequence_from_parts,
    qwen_linear_attention_sequence_with_cache_from_parts,
    qwen_linear_attention_step_with_cache_from_parts,
};
use llm_backend::{
    matvec_row_major_f32, rms_norm_f32, rms_norm_one_centered_f32, silu_f32, softmax_top_k_f32,
    swiglu_mlp_f32,
};
use llm_kv_cache::{LayerKvCache, LinearAttentionCache};

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
fn rms_norm_one_centered_uses_one_centered_weights() {
    let output =
        rms_norm_one_centered_f32(&[3.0, 4.0], &[0.0, 1.0], 0.0).expect("one-centered rms norm");

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
fn backend_cache_context_uses_generic_chat_template_identity() {
    let context = BackendCacheContext::chat_template(
        "chatml/qwen/v1",
        Some(r#"[{"type":"function"}]"#.to_owned()),
    );

    assert_eq!(context.prompt_template, "chatml/qwen/v1");
    assert_eq!(
        context.tool_schema.as_deref(),
        Some(r#"[{"type":"function"}]"#)
    );
}

#[test]
fn silu_matches_reference_values() {
    assert_close(
        &[silu_f32(0.0), silu_f32(2.0), silu_f32(-2.0)],
        &[0.0, 1.761594, -0.23840584],
        1e-6,
    );
}

#[tokio::test]
async fn qwen_linear_attention_first_token_matches_simplified_reference() {
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
    .await
    .expect("linear attention output");

    assert_close(&output, &[1.4621172], 1e-6);
}

#[tokio::test]
async fn qwen_linear_attention_sequence_updates_recurrent_state() {
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
    .await
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

#[tokio::test]
async fn qwen_linear_attention_sequence_updates_linear_cache() {
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
    let parts = QwenLinearAttentionSequenceParts {
        qkv: &qkv,
        z: &z,
        b: &b,
        a: &a,
        dt_bias: &dt_bias,
        a_log: &a_log,
        conv1d_weight: &conv1d_weight,
        norm_weight: &norm_weight,
        out_proj_weight: &out_proj_weight,
    };
    let mut cache = LinearAttentionCache::new(1, 4, 1, 1, 2).expect("cache shape");

    let output = qwen_linear_attention_sequence_with_cache_from_parts(&dims, &parts, &mut cache)
        .await
        .expect("linear attention sequence with cache");
    let expected = qwen_linear_attention_sequence_from_parts(&dims, &parts)
        .await
        .expect("linear attention sequence");
    let k0 = l2_scalar(silu_f32(1.0));
    let v0 = [silu_f32(2.0), silu_f32(4.0)];
    let k1 = l2_scalar(silu_f32(1.0));
    let v1 = [silu_f32(10.0), silu_f32(0.0)];
    let beta = 0.5;
    let decay = (-std::f32::consts::LN_2).exp();
    let state0 = [k0 * v0[0] * beta, k0 * v0[1] * beta];
    let state1_before = [state0[0] * decay, state0[1] * decay];
    let memory1 = [state1_before[0] * k1, state1_before[1] * k1];
    let delta1 = [(v1[0] - memory1[0]) * beta, (v1[1] - memory1[1]) * beta];
    let state1 = [
        state1_before[0] + k1 * delta1[0],
        state1_before[1] + k1 * delta1[1],
    ];

    assert_close(&output[0], &expected[0], 1e-6);
    assert_close(&output[1], &expected[1], 1e-6);
    assert_eq!(cache.token_count(), 2);
    assert_close(cache.conv_window(), &[1.0, 1.0, 10.0, 0.0], 1e-6);
    assert_close(cache.recurrent_state(), &state1, 1e-6);
}

#[tokio::test]
async fn qwen_linear_attention_step_uses_existing_linear_cache() {
    let dims = QwenLinearAttentionDims {
        hidden_size: 2,
        num_key_heads: 1,
        num_value_heads: 1,
        key_head_dim: 1,
        value_head_dim: 2,
        conv_kernel_size: 2,
        rms_norm_eps: 0.0,
    };
    let qkv = vec![
        vec![1.0, 1.0, 2.0, 4.0],
        vec![1.0, 1.0, 10.0, 0.0],
        vec![2.0, 1.0, 0.0, 8.0],
    ];
    let z = vec![vec![1.0, 1.0], vec![1.0, 1.0], vec![1.0, 1.0]];
    let b = vec![vec![0.0], vec![0.0], vec![0.0]];
    let a = vec![vec![0.0], vec![0.0], vec![0.0]];
    let dt_bias = vec![0.0];
    let a_log = vec![0.0];
    let conv1d_weight = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let norm_weight = vec![1.0, 1.0];
    let out_proj_weight = vec![1.0, 0.0, 0.0, 1.0];
    let expected_parts = QwenLinearAttentionSequenceParts {
        qkv: &qkv,
        z: &z,
        b: &b,
        a: &a,
        dt_bias: &dt_bias,
        a_log: &a_log,
        conv1d_weight: &conv1d_weight,
        norm_weight: &norm_weight,
        out_proj_weight: &out_proj_weight,
    };
    let mut expected_cache = LinearAttentionCache::new(2, 4, 1, 1, 2).expect("cache shape");
    let expected_output = qwen_linear_attention_sequence_with_cache_from_parts(
        &dims,
        &expected_parts,
        &mut expected_cache,
    )
    .await
    .expect("full cached prefill");
    let prefill_qkv = qkv[..2].to_vec();
    let prefill_z = z[..2].to_vec();
    let prefill_b = b[..2].to_vec();
    let prefill_a = a[..2].to_vec();
    let prefill_parts = QwenLinearAttentionSequenceParts {
        qkv: &prefill_qkv,
        z: &prefill_z,
        b: &prefill_b,
        a: &prefill_a,
        dt_bias: &dt_bias,
        a_log: &a_log,
        conv1d_weight: &conv1d_weight,
        norm_weight: &norm_weight,
        out_proj_weight: &out_proj_weight,
    };
    let mut cache = LinearAttentionCache::new(2, 4, 1, 1, 2).expect("cache shape");
    qwen_linear_attention_sequence_with_cache_from_parts(&dims, &prefill_parts, &mut cache)
        .await
        .expect("initial cached prefill");

    let output = qwen_linear_attention_step_with_cache_from_parts(
        &dims,
        &QwenLinearAttentionStepParts {
            qkv: &qkv[2],
            z: &z[2],
            b: &b[2],
            a: &a[2],
            dt_bias: &dt_bias,
            a_log: &a_log,
            conv1d_weight: &conv1d_weight,
            norm_weight: &norm_weight,
            out_proj_weight: &out_proj_weight,
        },
        &mut cache,
    )
    .await
    .expect("linear attention decode step");

    assert_close(&output, &expected_output[2], 1e-6);
    assert_eq!(cache.token_count(), 3);
    assert_close(cache.conv_window(), expected_cache.conv_window(), 1e-6);
    assert_close(
        cache.recurrent_state(),
        expected_cache.recurrent_state(),
        1e-6,
    );
}

#[tokio::test]
async fn qwen_full_attention_first_token_matches_single_key_reference() {
    let dims = QwenFullAttentionDims {
        hidden_size: 1,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 1,
    };

    let output = qwen_full_attention_first_token_from_parts(&dims, &[0.0, 0.0], &[8.0], &[3.0])
        .await
        .expect("full attention output");

    assert_close(&output, &[12.0], 1e-6);
}

#[tokio::test]
async fn qwen_full_attention_sequence_applies_rope_and_causal_softmax() {
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
            q_projection_gate: true,
            one_centered_rms_norm: true,
        },
    )
    .await
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

#[tokio::test]
async fn qwen_full_attention_sequence_writes_and_reads_layer_kv_cache() {
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
    let config = QwenFullAttentionSequenceConfig {
        rms_norm_eps: 0.0,
        rope_theta: 10_000.0,
        partial_rotary_factor: 1.0,
        q_projection_gate: true,
        one_centered_rms_norm: true,
    };
    let mut cache = LayerKvCache::new(2, 1, 2).expect("cache shape");

    let output = qwen_full_attention_sequence_with_cache_from_parts(
        &dims,
        &QwenFullAttentionSequenceParts {
            q_proj: &q_proj,
            k_proj: &k_proj,
            v_proj: &v_proj,
            q_norm_weight: &q_norm_weight,
            k_norm_weight: &k_norm_weight,
            o_proj_weight: &o_proj_weight,
        },
        config,
        &mut cache,
    )
    .await
    .expect("full attention sequence with cache");

    let expected = qwen_full_attention_sequence_from_parts(
        &dims,
        &QwenFullAttentionSequenceParts {
            q_proj: &q_proj,
            k_proj: &k_proj,
            v_proj: &v_proj,
            q_norm_weight: &q_norm_weight,
            k_norm_weight: &k_norm_weight,
            o_proj_weight: &o_proj_weight,
        },
        config,
    )
    .await
    .expect("full attention sequence");
    assert_eq!(cache.token_count(), 2);
    assert_close(cache.key(0).expect("key 0"), &[2.0_f32.sqrt(), 0.0], 1e-6);
    assert_close(cache.value(1).expect("value 1"), &[0.0, 4.0], 1e-6);
    assert_close(&output[0], &expected[0], 1e-6);
    assert_close(&output[1], &expected[1], 1e-6);
}

#[tokio::test]
async fn qwen_full_attention_sequence_with_small_cache_uses_sliding_window() {
    let dims = QwenFullAttentionDims {
        hidden_size: 2,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 2,
    };
    let q_proj = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0, 0.0],
    ];
    let k_proj = vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![1.0, 0.0]];
    let v_proj = vec![vec![2.0, 0.0], vec![0.0, 4.0], vec![8.0, 0.0]];
    let q_norm_weight = vec![0.0, 0.0];
    let k_norm_weight = vec![0.0, 0.0];
    let o_proj_weight = vec![1.0, 0.0, 0.0, 1.0];
    let config = QwenFullAttentionSequenceConfig {
        rms_norm_eps: 0.0,
        rope_theta: 10_000.0,
        partial_rotary_factor: 0.0,
        q_projection_gate: true,
        one_centered_rms_norm: true,
    };
    let mut cache = LayerKvCache::new(2, 1, 2).expect("cache shape");

    let output = qwen_full_attention_sequence_with_cache_from_parts(
        &dims,
        &QwenFullAttentionSequenceParts {
            q_proj: &q_proj,
            k_proj: &k_proj,
            v_proj: &v_proj,
            q_norm_weight: &q_norm_weight,
            k_norm_weight: &k_norm_weight,
            o_proj_weight: &o_proj_weight,
        },
        config,
        &mut cache,
    )
    .await
    .expect("small cache uses sliding attention");

    assert_eq!(cache.token_count(), 2);
    assert_close(cache.value(0).expect("value 0"), &[0.0, 4.0], 1e-6);
    assert_close(cache.value(1).expect("value 1"), &[8.0, 0.0], 1e-6);
    assert_close(&output[2], &[2.0, 1.0], 1e-6);
}

#[tokio::test]
async fn qwen_full_attention_step_with_full_cache_uses_sliding_window() {
    let dims = QwenFullAttentionDims {
        hidden_size: 2,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 2,
    };
    let q_norm_weight = vec![0.0, 0.0];
    let k_norm_weight = vec![0.0, 0.0];
    let o_proj_weight = vec![1.0, 0.0, 0.0, 1.0];
    let config = QwenFullAttentionSequenceConfig {
        rms_norm_eps: 0.0,
        rope_theta: 10_000.0,
        partial_rotary_factor: 0.0,
        q_projection_gate: true,
        one_centered_rms_norm: true,
    };
    let mut cache = LayerKvCache::new(2, 1, 2).expect("cache shape");
    let prefill_q_proj = [vec![1.0, 0.0, 0.0, 0.0], vec![1.0, 0.0, 0.0, 0.0]];
    let prefill_k_proj = [vec![1.0, 0.0], vec![1.0, 0.0]];
    let prefill_v_proj = [vec![2.0, 0.0], vec![0.0, 4.0]];
    qwen_full_attention_sequence_with_cache_from_parts(
        &dims,
        &QwenFullAttentionSequenceParts {
            q_proj: &prefill_q_proj,
            k_proj: &prefill_k_proj,
            v_proj: &prefill_v_proj,
            q_norm_weight: &q_norm_weight,
            k_norm_weight: &k_norm_weight,
            o_proj_weight: &o_proj_weight,
        },
        config,
        &mut cache,
    )
    .await
    .expect("prefill fills cache");

    let output = qwen_full_attention_step_with_cache_from_parts(
        &dims,
        &QwenFullAttentionStepParts {
            q_proj: &[1.0, 0.0, 0.0, 0.0],
            k_proj: &[1.0, 0.0],
            v_proj: &[8.0, 0.0],
            q_norm_weight: &q_norm_weight,
            k_norm_weight: &k_norm_weight,
            o_proj_weight: &o_proj_weight,
        },
        config,
        &mut cache,
    )
    .await
    .expect("full cache evicts oldest token during decode");

    assert_eq!(cache.token_count(), 2);
    assert_close(cache.value(0).expect("value 0"), &[0.0, 4.0], 1e-6);
    assert_close(cache.value(1).expect("value 1"), &[8.0, 0.0], 1e-6);
    assert_close(&output, &[2.0, 1.0], 1e-6);
}

#[tokio::test]
async fn qwen_full_attention_step_uses_existing_layer_kv_cache() {
    let dims = QwenFullAttentionDims {
        hidden_size: 2,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 2,
    };
    let q_proj = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
    ];
    let k_proj = vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
    let v_proj = vec![vec![2.0, 0.0], vec![0.0, 4.0], vec![6.0, 8.0]];
    let q_norm_weight = vec![0.0, 0.0];
    let k_norm_weight = vec![0.0, 0.0];
    let o_proj_weight = vec![1.0, 0.0, 0.0, 1.0];
    let config = QwenFullAttentionSequenceConfig {
        rms_norm_eps: 0.0,
        rope_theta: 10_000.0,
        partial_rotary_factor: 1.0,
        q_projection_gate: true,
        one_centered_rms_norm: true,
    };
    let expected_parts = QwenFullAttentionSequenceParts {
        q_proj: &q_proj,
        k_proj: &k_proj,
        v_proj: &v_proj,
        q_norm_weight: &q_norm_weight,
        k_norm_weight: &k_norm_weight,
        o_proj_weight: &o_proj_weight,
    };
    let mut expected_cache = LayerKvCache::new(3, 1, 2).expect("cache shape");
    let expected_output = qwen_full_attention_sequence_with_cache_from_parts(
        &dims,
        &expected_parts,
        config,
        &mut expected_cache,
    )
    .await
    .expect("full cached prefill");
    let prefill_q_proj = q_proj[..2].to_vec();
    let prefill_k_proj = k_proj[..2].to_vec();
    let prefill_v_proj = v_proj[..2].to_vec();
    let prefill_parts = QwenFullAttentionSequenceParts {
        q_proj: &prefill_q_proj,
        k_proj: &prefill_k_proj,
        v_proj: &prefill_v_proj,
        q_norm_weight: &q_norm_weight,
        k_norm_weight: &k_norm_weight,
        o_proj_weight: &o_proj_weight,
    };
    let mut cache = LayerKvCache::new(3, 1, 2).expect("cache shape");
    qwen_full_attention_sequence_with_cache_from_parts(&dims, &prefill_parts, config, &mut cache)
        .await
        .expect("initial cached prefill");

    let output = qwen_full_attention_step_with_cache_from_parts(
        &dims,
        &QwenFullAttentionStepParts {
            q_proj: &q_proj[2],
            k_proj: &k_proj[2],
            v_proj: &v_proj[2],
            q_norm_weight: &q_norm_weight,
            k_norm_weight: &k_norm_weight,
            o_proj_weight: &o_proj_weight,
        },
        config,
        &mut cache,
    )
    .await
    .expect("full attention decode step");

    assert_close(&output, &expected_output[2], 1e-6);
    assert_eq!(cache.token_count(), 3);
    assert_close(cache.keys(), expected_cache.keys(), 1e-6);
    assert_close(cache.values(), expected_cache.values(), 1e-6);
}

#[tokio::test]
async fn qwen_full_attention_step_matches_sequence_with_qk_norm() {
    let dims = QwenFullAttentionDims {
        hidden_size: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 2,
    };
    let q_proj = vec![
        vec![3.0, 1.0, 1.0, 5.0],
        vec![0.5, 2.0, 4.0, 0.5],
        vec![2.0, 0.0, 0.0, 1.0],
    ];
    let k_proj = vec![vec![1.0, 2.0], vec![3.0, 0.5], vec![0.0, 1.0]];
    let v_proj = vec![vec![2.0, 1.0], vec![0.0, 4.0], vec![6.0, 8.0]];
    let q_norm_weight = vec![1.5, 0.5];
    let k_norm_weight = vec![2.0, 1.0];
    let o_proj_weight = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5];
    let config = QwenFullAttentionSequenceConfig {
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        partial_rotary_factor: 1.0,
        q_projection_gate: false,
        one_centered_rms_norm: false,
    };
    let expected_parts = QwenFullAttentionSequenceParts {
        q_proj: &q_proj,
        k_proj: &k_proj,
        v_proj: &v_proj,
        q_norm_weight: &q_norm_weight,
        k_norm_weight: &k_norm_weight,
        o_proj_weight: &o_proj_weight,
    };
    let mut expected_cache = LayerKvCache::new(3, 1, 2).expect("cache shape");
    let expected_output = qwen_full_attention_sequence_with_cache_from_parts(
        &dims,
        &expected_parts,
        config,
        &mut expected_cache,
    )
    .await
    .expect("full cached sequence with qk norm");
    let prefill_q_proj = q_proj[..2].to_vec();
    let prefill_k_proj = k_proj[..2].to_vec();
    let prefill_v_proj = v_proj[..2].to_vec();
    let prefill_parts = QwenFullAttentionSequenceParts {
        q_proj: &prefill_q_proj,
        k_proj: &prefill_k_proj,
        v_proj: &prefill_v_proj,
        q_norm_weight: &q_norm_weight,
        k_norm_weight: &k_norm_weight,
        o_proj_weight: &o_proj_weight,
    };
    let mut cache = LayerKvCache::new(3, 1, 2).expect("cache shape");
    qwen_full_attention_sequence_with_cache_from_parts(&dims, &prefill_parts, config, &mut cache)
        .await
        .expect("prefill with qk norm");

    let output = qwen_full_attention_step_with_cache_from_parts(
        &dims,
        &QwenFullAttentionStepParts {
            q_proj: &q_proj[2],
            k_proj: &k_proj[2],
            v_proj: &v_proj[2],
            q_norm_weight: &q_norm_weight,
            k_norm_weight: &k_norm_weight,
            o_proj_weight: &o_proj_weight,
        },
        config,
        &mut cache,
    )
    .await
    .expect("full attention step with qk norm");

    assert_close(&output, &expected_output[2], 1e-5);
    assert_eq!(cache.token_count(), 3);
    assert_close(cache.keys(), expected_cache.keys(), 1e-5);
    assert_close(cache.values(), expected_cache.values(), 1e-5);
}

fn rms_pair(values: [f32; 2], gate: f32) -> Vec<f32> {
    let rms = ((values[0] * values[0] + values[1] * values[1]) / 2.0).sqrt();
    vec![values[0] / rms * gate, values[1] / rms * gate]
}

#[tokio::test]
async fn qwen_linear_attention_step_matches_sequence_with_multi_dim_keys() {
    let dims = QwenLinearAttentionDims {
        hidden_size: 2,
        num_key_heads: 1,
        num_value_heads: 1,
        key_head_dim: 2,
        value_head_dim: 2,
        conv_kernel_size: 1,
        rms_norm_eps: 0.0,
    };
    let qkv = vec![
        vec![3.0, 1.0, 1.0, 5.0, 2.0, 4.0],
        vec![0.5, 2.0, 4.0, 0.5, 10.0, 0.0],
        vec![2.0, 0.0, 0.0, 1.0, 0.0, 8.0],
    ];
    let z = vec![vec![1.0, 1.0], vec![1.0, 1.0], vec![1.0, 1.0]];
    let b = vec![vec![0.0], vec![0.0], vec![0.0]];
    let a = vec![vec![0.0], vec![0.0], vec![0.0]];
    let dt_bias = vec![0.0];
    let a_log = vec![0.0];
    let conv1d_weight = vec![1.0; 6];
    let norm_weight = vec![1.0, 1.0];
    let out_proj_weight = vec![1.0, 0.0, 0.0, 1.0];
    let expected_parts = QwenLinearAttentionSequenceParts {
        qkv: &qkv,
        z: &z,
        b: &b,
        a: &a,
        dt_bias: &dt_bias,
        a_log: &a_log,
        conv1d_weight: &conv1d_weight,
        norm_weight: &norm_weight,
        out_proj_weight: &out_proj_weight,
    };
    let mut expected_cache = LinearAttentionCache::new(1, 6, 1, 2, 2).expect("cache shape");
    let expected_output = qwen_linear_attention_sequence_with_cache_from_parts(
        &dims,
        &expected_parts,
        &mut expected_cache,
    )
    .await
    .expect("full cached prefill");
    let prefill_qkv = qkv[..2].to_vec();
    let prefill_z = z[..2].to_vec();
    let prefill_b = b[..2].to_vec();
    let prefill_a = a[..2].to_vec();
    let prefill_parts = QwenLinearAttentionSequenceParts {
        qkv: &prefill_qkv,
        z: &prefill_z,
        b: &prefill_b,
        a: &prefill_a,
        dt_bias: &dt_bias,
        a_log: &a_log,
        conv1d_weight: &conv1d_weight,
        norm_weight: &norm_weight,
        out_proj_weight: &out_proj_weight,
    };
    let mut cache = LinearAttentionCache::new(1, 6, 1, 2, 2).expect("cache shape");
    qwen_linear_attention_sequence_with_cache_from_parts(&dims, &prefill_parts, &mut cache)
        .await
        .expect("initial cached prefill");

    let output = qwen_linear_attention_step_with_cache_from_parts(
        &dims,
        &QwenLinearAttentionStepParts {
            qkv: &qkv[2],
            z: &z[2],
            b: &b[2],
            a: &a[2],
            dt_bias: &dt_bias,
            a_log: &a_log,
            conv1d_weight: &conv1d_weight,
            norm_weight: &norm_weight,
            out_proj_weight: &out_proj_weight,
        },
        &mut cache,
    )
    .await
    .expect("linear attention decode step");

    assert_close(&output, &expected_output[2], 1e-5);
    assert_eq!(cache.token_count(), 3);
    assert_close(cache.conv_window(), expected_cache.conv_window(), 1e-5);
    assert_close(
        cache.recurrent_state(),
        expected_cache.recurrent_state(),
        1e-5,
    );
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
