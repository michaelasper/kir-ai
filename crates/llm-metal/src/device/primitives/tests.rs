use super::super::shaders::METAL_SOURCE;
use super::MetalDevice;

fn softmax_shader_source() -> &'static str {
    let start = METAL_SOURCE
        .find("kernel void softmax_f32")
        .expect("softmax shader exists");
    let rest = &METAL_SOURCE[start..];
    let next_kernel = rest["kernel void softmax_f32".len()..]
        .find("kernel void ")
        .map(|offset| "kernel void softmax_f32".len() + offset)
        .unwrap_or(rest.len());
    &rest[..next_kernel]
}

#[test]
fn softmax_shader_uses_threadgroup_reductions_instead_of_single_worker_thread() {
    let shader = softmax_shader_source();

    assert!(
        shader.contains("threadgroup float"),
        "softmax should reduce through threadgroup memory"
    );
    assert!(
        !shader.contains("if (id != 0"),
        "softmax must not gate all work onto one GPU thread"
    );
}

#[tokio::test]
async fn full_attention_cache_mix_matches_reference_values() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };

    let keys = device
        .new_f32_buffer(&[1.0, 0.0, 0.0, 1.0])
        .expect("key buffer");
    let values = device
        .new_f32_buffer(&[10.0, 20.0, 30.0, 40.0])
        .expect("value buffer");
    let query = [1.0, 0.0, 0.0, 1.0];
    let mut output = vec![0.0; 4];

    device
        .full_attention_cache_mix_f32_buffered(&keys, &values, &query, 2, 2, 1, 2, 1.0, &mut output)
        .await
        .expect("attention mix succeeds");

    let head0_weight_0 = 1.0_f32.exp() / (1.0_f32.exp() + 0.0_f32.exp());
    let head0_weight_1 = 1.0 - head0_weight_0;
    let head1_weight_0 = head0_weight_1;
    let head1_weight_1 = head0_weight_0;
    let expected = [
        10.0 * head0_weight_0 + 30.0 * head0_weight_1,
        20.0 * head0_weight_0 + 40.0 * head0_weight_1,
        10.0 * head1_weight_0 + 30.0 * head1_weight_1,
        20.0 * head1_weight_0 + 40.0 * head1_weight_1,
    ];
    for (actual, expected) in output.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1e-4,
            "expected {actual} to be close to {expected}"
        );
    }
}

#[tokio::test]
async fn vector_add_reuses_transient_scratch_buffers() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };

    let left = [1.0, 2.0, 3.0];
    let right = [4.0, 5.0, 6.0];
    let mut output = vec![0.0; 3];

    device
        .add_f32(&left, &right, &mut output)
        .await
        .expect("first vector add succeeds");
    device
        .add_f32(&left, &right, &mut output)
        .await
        .expect("second vector add succeeds");

    assert_eq!(
        device.scratch_buffer_count_for_test(3 * std::mem::size_of::<f32>() as u64),
        3
    );
    assert_eq!(output, [5.0, 7.0, 9.0]);
}

#[tokio::test]
async fn softmax_f32_reuses_transient_scratch_buffers() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };

    let scores = [1.0, 2.0, 3.0, 4.0];
    let mut output = vec![0.0; 4];

    device
        .softmax_f32(&scores, &mut output)
        .await
        .expect("first softmax succeeds");
    device
        .softmax_f32(&scores, &mut output)
        .await
        .expect("second softmax succeeds");

    assert_eq!(
        device.scratch_buffer_count_for_test(4 * std::mem::size_of::<f32>() as u64),
        2
    );
}

#[tokio::test]
async fn weighted_sum_f32_reuses_transient_scratch_buffers() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };

    let values = [1.0, 2.0, 3.0, 4.0, -1.0, 0.5];
    let weights = [0.5, 0.25, 0.25];
    let mut output = vec![0.0; 2];

    device
        .weighted_sum_f32(&values, &weights, 2, &mut output)
        .await
        .expect("first weighted sum succeeds");
    device
        .weighted_sum_f32(&values, &weights, 2, &mut output)
        .await
        .expect("second weighted sum succeeds");

    assert_eq!(
        device.scratch_buffer_count_for_test(6 * std::mem::size_of::<f32>() as u64),
        1
    );
    assert_eq!(
        device.scratch_buffer_count_for_test(3 * std::mem::size_of::<f32>() as u64),
        1
    );
    assert_eq!(
        device.scratch_buffer_count_for_test(2 * std::mem::size_of::<f32>() as u64),
        1
    );
    assert!((output[0] - 1.0).abs() < 1e-6);
    assert!((output[1] - 2.125).abs() < 1e-6);
}

#[tokio::test]
async fn full_attention_cache_mix_f32_reuses_transient_scratch_buffers() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };

    let keys = device
        .new_f32_buffer(&[1.0, 0.0, 0.0, 1.0])
        .expect("key buffer");
    let values = device
        .new_f32_buffer(&[10.0, 20.0, 30.0, 40.0])
        .expect("value buffer");
    let query = [1.0, 0.0, 0.0, 1.0];
    let mut output = vec![0.0; 4];

    device
        .full_attention_cache_mix_f32_buffered(&keys, &values, &query, 2, 2, 1, 2, 1.0, &mut output)
        .await
        .expect("first attention mix succeeds");
    device
        .full_attention_cache_mix_f32_buffered(&keys, &values, &query, 2, 2, 1, 2, 1.0, &mut output)
        .await
        .expect("second attention mix succeeds");

    assert_eq!(
        device.scratch_buffer_count_for_test(4 * std::mem::size_of::<f32>() as u64),
        4
    );
}

#[tokio::test]
async fn full_attention_cache_mix_f16_matches_reference_values() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };

    let keys = device
        .new_f16_buffer_from_f32(&[1.0, 0.0, 0.0, 1.0])
        .expect("key buffer");
    let values = device
        .new_f16_buffer_from_f32(&[10.0, 20.0, 30.0, 40.0])
        .expect("value buffer");
    let query = [1.0, 0.0, 0.0, 1.0];
    let mut output = vec![0.0; 4];

    device
        .full_attention_cache_mix_f16_buffered(&keys, &values, &query, 2, 2, 1, 2, 1.0, &mut output)
        .await
        .expect("attention mix succeeds");

    let head0_weight_0 = 1.0_f32.exp() / (1.0_f32.exp() + 0.0_f32.exp());
    let head0_weight_1 = 1.0 - head0_weight_0;
    let head1_weight_0 = head0_weight_1;
    let head1_weight_1 = head0_weight_0;
    let expected = [
        10.0 * head0_weight_0 + 30.0 * head0_weight_1,
        20.0 * head0_weight_0 + 40.0 * head0_weight_1,
        10.0 * head1_weight_0 + 30.0 * head1_weight_1,
        20.0 * head1_weight_0 + 40.0 * head1_weight_1,
    ];
    for (actual, expected) in output.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1e-4,
            "expected {actual} to be close to {expected}"
        );
    }
}

#[tokio::test]
async fn full_attention_cache_mix_f16_reads_from_nonzero_buffer_offsets() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };

    let keys = device
        .new_f16_buffer_from_f32(&[99.0, 99.0, 1.0, 0.0, 0.0, 1.0])
        .expect("key buffer");
    let values = device
        .new_f16_buffer_from_f32(&[99.0, 99.0, 10.0, 20.0, 30.0, 40.0])
        .expect("value buffer");
    let query = [1.0, 0.0, 0.0, 1.0];
    let mut output = vec![0.0; 4];

    device
        .full_attention_cache_mix_f16_buffered_at(
            &keys,
            2,
            &values,
            2,
            &query,
            2,
            2,
            1,
            2,
            1.0,
            &mut output,
        )
        .await
        .expect("attention mix succeeds");

    let head0_weight_0 = 1.0_f32.exp() / (1.0_f32.exp() + 0.0_f32.exp());
    let head0_weight_1 = 1.0 - head0_weight_0;
    let head1_weight_0 = head0_weight_1;
    let head1_weight_1 = head0_weight_0;
    let expected = [
        10.0 * head0_weight_0 + 30.0 * head0_weight_1,
        20.0 * head0_weight_0 + 40.0 * head0_weight_1,
        10.0 * head1_weight_0 + 30.0 * head1_weight_1,
        20.0 * head1_weight_0 + 40.0 * head1_weight_1,
    ];
    for (actual, expected) in output.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1e-4,
            "expected {actual} to be close to {expected}"
        );
    }
}

#[tokio::test]
async fn full_attention_cache_mix_int8_matches_reference_values() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };

    let keys = device.new_i8_buffer(&[127, 0, 0, 127]).expect("key buffer");
    let key_scales = device
        .new_f32_buffer(&[1.0 / 127.0, 1.0 / 127.0])
        .expect("key scale buffer");
    let values = device
        .new_i8_buffer(&[127, 0, 0, 127])
        .expect("value buffer");
    let value_scales = device
        .new_f32_buffer(&[10.0 / 127.0, 10.0 / 127.0])
        .expect("value scale buffer");
    let query = [1.0, 0.0, 0.0, 1.0];
    let mut output = vec![0.0; 4];

    device
        .full_attention_cache_mix_int8_buffered(
            &keys,
            &key_scales,
            &values,
            &value_scales,
            &query,
            2,
            2,
            1,
            2,
            1.0,
            &mut output,
        )
        .await
        .expect("attention mix succeeds");

    let head0_weight_0 = 1.0_f32.exp() / (1.0_f32.exp() + 0.0_f32.exp());
    let head0_weight_1 = 1.0 - head0_weight_0;
    let head1_weight_0 = head0_weight_1;
    let head1_weight_1 = head0_weight_0;
    let expected = [
        10.0 * head0_weight_0,
        10.0 * head0_weight_1,
        10.0 * head1_weight_0,
        10.0 * head1_weight_1,
    ];
    for (actual, expected) in output.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1e-4,
            "expected {actual} to be close to {expected}"
        );
    }
}

#[tokio::test]
async fn full_attention_cache_mix_int8_reads_from_nonzero_buffer_offsets() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };

    let keys = device
        .new_i8_buffer(&[1, 1, 127, 0, 0, 127])
        .expect("key buffer");
    let key_scales = device
        .new_f32_buffer(&[99.0, 1.0 / 127.0, 1.0 / 127.0])
        .expect("key scale buffer");
    let values = device
        .new_i8_buffer(&[1, 1, 127, 0, 0, 127])
        .expect("value buffer");
    let value_scales = device
        .new_f32_buffer(&[99.0, 10.0 / 127.0, 10.0 / 127.0])
        .expect("value scale buffer");
    let query = [1.0, 0.0, 0.0, 1.0];
    let mut output = vec![0.0; 4];

    device
        .full_attention_cache_mix_int8_buffered_at(
            &keys,
            2,
            &key_scales,
            1,
            &values,
            2,
            &value_scales,
            1,
            &query,
            2,
            2,
            1,
            2,
            1.0,
            &mut output,
        )
        .await
        .expect("attention mix succeeds");

    let head0_weight_0 = 1.0_f32.exp() / (1.0_f32.exp() + 0.0_f32.exp());
    let head0_weight_1 = 1.0 - head0_weight_0;
    let head1_weight_0 = head0_weight_1;
    let head1_weight_1 = head0_weight_0;
    let expected = [
        10.0 * head0_weight_0,
        10.0 * head0_weight_1,
        10.0 * head1_weight_0,
        10.0 * head1_weight_1,
    ];
    for (actual, expected) in output.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1e-4,
            "expected {actual} to be close to {expected}"
        );
    }
}

#[tokio::test]
async fn select_head_rows_f16_buffered_matches_reference_values() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };

    let values = device
        .new_f16_buffer_from_f32(&[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
        .expect("value buffer");
    let mut output = vec![0.0; 4];

    device
        .select_head_rows_f16_buffered(&values, 2, 4, 1, 2, &mut output)
        .await
        .expect("head row selection succeeds");

    assert_eq!(output, [2.0, 3.0, 20.0, 30.0]);
}

#[tokio::test]
async fn select_head_rows_f16_buffered_reuses_output_buffer() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };

    let values = device
        .new_f16_buffer_from_f32(&[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
        .expect("value buffer");
    let mut output = vec![0.0; 4];

    device
        .select_head_rows_f16_buffered(&values, 2, 4, 1, 2, &mut output)
        .await
        .expect("first head row selection succeeds");
    device
        .select_head_rows_f16_buffered(&values, 2, 4, 1, 2, &mut output)
        .await
        .expect("second head row selection succeeds");

    assert_eq!(
        device.scratch_buffer_count_for_test(4 * std::mem::size_of::<f32>() as u64),
        1
    );
    assert_eq!(output, [2.0, 3.0, 20.0, 30.0]);
}

#[tokio::test]
async fn select_head_rows_int8_buffered_matches_reference_values() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping smoke test");
        return;
    };

    let values = device
        .new_i8_buffer(&[10, 20, 30, 40, 5, 10, 15, 20])
        .expect("value buffer");
    let scales = device.new_f32_buffer(&[0.1, 2.0]).expect("scale buffer");
    let mut output = vec![0.0; 4];

    device
        .select_head_rows_int8_buffered(&values, &scales, 2, 4, 1, 2, &mut output)
        .await
        .expect("int8 head row selection succeeds");

    assert_eq!(output, [2.0, 3.0, 20.0, 30.0]);
}
