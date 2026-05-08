use llm_metal::MetalDevice;
use serde::Deserialize;
use std::time::{Duration, Instant};

const METAL_LATENCY_WARMUP_RUNS: usize = 1;
const METAL_LATENCY_SAMPLE_RUNS: usize = 5;

#[test]
fn metal_kernels_match_mlx_reference_trace() {
    let Some(device) = MetalDevice::system_default_result().expect("Metal device initializes")
    else {
        eprintln!("no Metal device available; skipping MLX reference test");
        return;
    };
    let fixture: MlxReferenceFixture =
        serde_json::from_str(include_str!("fixtures/mlx_reference.json"))
            .expect("MLX reference fixture parses");
    assert_eq!(fixture.schema_version, 1);
    fixture.cases.assert_latency_traces_present();

    let add_left = [1.0, 2.5, -3.0, 8.0];
    let add_right = [4.0, -1.5, 3.0, 0.25];
    let vector_add = device
        .add_f32(&add_left, &add_right)
        .expect("metal vector add succeeds");
    assert_close(&vector_add, &fixture.cases.vector_add_f32.output, 1e-6);
    assert_latency_delta(
        "vector_add_f32",
        fixture.cases.vector_add_f32.mlx_median_us,
        || {
            device
                .add_f32(&add_left, &add_right)
                .expect("metal vector add succeeds")
        },
    );

    let rms = device
        .qwen_rms_norm_f32(&[3.0, 4.0], &[0.0, 1.0], 0.0)
        .expect("metal qwen rms norm succeeds");
    assert_close(&rms, &fixture.cases.qwen_rms_norm_f32.output, 1e-6);
    assert_latency_delta(
        "qwen_rms_norm_f32",
        fixture.cases.qwen_rms_norm_f32.mlx_median_us,
        || {
            device
                .qwen_rms_norm_f32(&[3.0, 4.0], &[0.0, 1.0], 0.0)
                .expect("metal qwen rms norm succeeds")
        },
    );

    let scores = [1.0, 2.0, -1.0, 0.5];
    let softmax = device.softmax_f32(&scores).expect("metal softmax succeeds");
    assert_close(&softmax, &fixture.cases.softmax_f32.output, 1e-6);
    assert_latency_delta(
        "softmax_f32",
        fixture.cases.softmax_f32.mlx_median_us,
        || device.softmax_f32(&scores).expect("metal softmax succeeds"),
    );

    let conv_window = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let conv_weights = [0.5, 1.0, -1.0, 0.25, 2.0, -0.5];
    let conv = device
        .linear_attention_conv1d_silu_f32(&conv_window, &conv_weights, 3, 2)
        .expect("metal linear attention conv succeeds");
    assert_close(
        &conv,
        &fixture.cases.linear_attention_conv1d_silu_f32.output,
        1e-5,
    );
    assert_latency_delta(
        "linear_attention_conv1d_silu_f32",
        fixture.cases.linear_attention_conv1d_silu_f32.mlx_median_us,
        || {
            device
                .linear_attention_conv1d_silu_f32(&conv_window, &conv_weights, 3, 2)
                .expect("metal linear attention conv succeeds")
        },
    );

    let matvec_matrix = [1.0, 2.0, 3.0, 4.0, -1.0, 0.5];
    let matvec_vector = [0.5, -2.0, 4.0];
    let matvec = device
        .matvec_f32(&matvec_matrix, 2, 3, &matvec_vector)
        .expect("metal matvec succeeds");
    assert_close(&matvec, &fixture.cases.matvec_f32.output, 1e-6);
    assert_latency_delta("matvec_f32", fixture.cases.matvec_f32.mlx_median_us, || {
        device
            .matvec_f32(&matvec_matrix, 2, 3, &matvec_vector)
            .expect("metal matvec succeeds")
    });

    let matrix = matvec_matrix.map(f32_to_bf16_bits);
    let bf16_matvec = device
        .matvec_bf16_f32(&matrix, 2, 3, &matvec_vector)
        .expect("metal bf16 matvec succeeds");
    assert_close(&bf16_matvec, &fixture.cases.matvec_bf16_f32.output, 1e-6);
    assert_latency_delta(
        "matvec_bf16_f32",
        fixture.cases.matvec_bf16_f32.mlx_median_us,
        || {
            device
                .matvec_bf16_f32(&matrix, 2, 3, &matvec_vector)
                .expect("metal bf16 matvec succeeds")
        },
    );

    let batched_matrix = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map(f32_to_bf16_bits);
    let batched_vectors = [1.0, 2.0, 3.0, 3.0, 2.0, 1.0];
    let batched = device
        .batched_matvec_bf16_f32(&batched_matrix, 2, 3, &batched_vectors, 2)
        .expect("metal batched bf16 matvec succeeds");
    assert_close(
        &batched,
        &fixture.cases.batched_matvec_bf16_f32.output,
        1e-6,
    );
    assert_latency_delta(
        "batched_matvec_bf16_f32",
        fixture.cases.batched_matvec_bf16_f32.mlx_median_us,
        || {
            device
                .batched_matvec_bf16_f32(&batched_matrix, 2, 3, &batched_vectors, 2)
                .expect("metal batched bf16 matvec succeeds")
        },
    );

    let weighted_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let weighted_weights = [0.25, -0.5];
    let weighted = device
        .weighted_sum_f32(&weighted_values, &weighted_weights, 3)
        .expect("metal weighted sum succeeds");
    assert_close(&weighted, &fixture.cases.weighted_sum_f32.output, 1e-6);
    assert_latency_delta(
        "weighted_sum_f32",
        fixture.cases.weighted_sum_f32.mlx_median_us,
        || {
            device
                .weighted_sum_f32(&weighted_values, &weighted_weights, 3)
                .expect("metal weighted sum succeeds")
        },
    );

    let recurrent_state = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let recurrent_key = [0.5, -1.0];
    let recurrent_value = [10.0, 20.0, 30.0];
    let recurrent_memory = [1.0, 2.0, 3.0];
    let recurrent = device
        .linear_attention_recurrent_update_f32(
            &recurrent_state,
            &recurrent_key,
            &recurrent_value,
            &recurrent_memory,
            0.25,
            0.5,
            2,
            3,
        )
        .expect("metal recurrent update succeeds");
    assert_close(
        &recurrent,
        &fixture.cases.linear_attention_recurrent_update_f32.output,
        1e-6,
    );
    assert_latency_delta(
        "linear_attention_recurrent_update_f32",
        fixture
            .cases
            .linear_attention_recurrent_update_f32
            .mlx_median_us,
        || {
            device
                .linear_attention_recurrent_update_f32(
                    &recurrent_state,
                    &recurrent_key,
                    &recurrent_value,
                    &recurrent_memory,
                    0.25,
                    0.5,
                    2,
                    3,
                )
                .expect("metal recurrent update succeeds")
        },
    );

    let head_rows = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let selected_head_rows = device
        .select_head_rows_f32(&head_rows, 2, 4, 1, 2)
        .expect("metal head row selection succeeds");
    assert_close(
        &selected_head_rows,
        &fixture.cases.select_head_rows_f32.output,
        1e-6,
    );
    assert_latency_delta(
        "select_head_rows_f32",
        fixture.cases.select_head_rows_f32.mlx_median_us,
        || {
            device
                .select_head_rows_f32(&head_rows, 2, 4, 1, 2)
                .expect("metal head row selection succeeds")
        },
    );

    let mut argmax_logits = vec![-1.0; 600];
    argmax_logits[42] = 4.5;
    argmax_logits[311] = 4.5;
    argmax_logits[599] = 3.25;
    let argmax = device
        .argmax_f32(&argmax_logits)
        .expect("metal argmax succeeds");
    assert_eq!(argmax.index, fixture.cases.argmax_f32.index);
    assert_eq!(argmax.value, fixture.cases.argmax_f32.value);
    assert_latency_delta("argmax_f32", fixture.cases.argmax_f32.mlx_median_us, || {
        device
            .argmax_f32(&argmax_logits)
            .expect("metal argmax succeeds")
    });

    let mut top_k_logits = vec![-10.0; 700];
    top_k_logits[7] = 9.0;
    top_k_logits[288] = 12.0;
    top_k_logits[499] = 12.0;
    top_k_logits[612] = 5.0;
    let top_k = device
        .top_k_f32(&top_k_logits, 3)
        .expect("metal top-k succeeds");
    assert_eq!(
        top_k.iter().map(|item| item.index).collect::<Vec<_>>(),
        fixture.cases.top_k_f32.indices
    );
    assert_close(
        &top_k.iter().map(|item| item.value).collect::<Vec<_>>(),
        &fixture.cases.top_k_f32.values,
        1e-6,
    );
    assert_latency_delta("top_k_f32", fixture.cases.top_k_f32.mlx_median_us, || {
        device
            .top_k_f32(&top_k_logits, 3)
            .expect("metal top-k succeeds")
    });
}

#[derive(Debug, Deserialize)]
struct MlxReferenceFixture {
    schema_version: u32,
    cases: MlxReferenceCases,
}

#[derive(Debug, Deserialize)]
struct MlxReferenceCases {
    vector_add_f32: OutputCase,
    qwen_rms_norm_f32: OutputCase,
    softmax_f32: OutputCase,
    linear_attention_conv1d_silu_f32: OutputCase,
    matvec_f32: OutputCase,
    matvec_bf16_f32: OutputCase,
    batched_matvec_bf16_f32: OutputCase,
    weighted_sum_f32: OutputCase,
    linear_attention_recurrent_update_f32: OutputCase,
    select_head_rows_f32: OutputCase,
    argmax_f32: IndexedCase,
    top_k_f32: TopKCase,
}

impl MlxReferenceCases {
    fn assert_latency_traces_present(&self) {
        for (name, latency) in [
            ("vector_add_f32", self.vector_add_f32.mlx_median_us),
            ("qwen_rms_norm_f32", self.qwen_rms_norm_f32.mlx_median_us),
            ("softmax_f32", self.softmax_f32.mlx_median_us),
            (
                "linear_attention_conv1d_silu_f32",
                self.linear_attention_conv1d_silu_f32.mlx_median_us,
            ),
            ("matvec_f32", self.matvec_f32.mlx_median_us),
            ("matvec_bf16_f32", self.matvec_bf16_f32.mlx_median_us),
            (
                "batched_matvec_bf16_f32",
                self.batched_matvec_bf16_f32.mlx_median_us,
            ),
            ("weighted_sum_f32", self.weighted_sum_f32.mlx_median_us),
            (
                "linear_attention_recurrent_update_f32",
                self.linear_attention_recurrent_update_f32.mlx_median_us,
            ),
            (
                "select_head_rows_f32",
                self.select_head_rows_f32.mlx_median_us,
            ),
            ("argmax_f32", self.argmax_f32.mlx_median_us),
            ("top_k_f32", self.top_k_f32.mlx_median_us),
        ] {
            assert!(latency > 0.0, "missing MLX latency trace for {name}");
        }
    }
}

#[derive(Debug, Deserialize)]
struct OutputCase {
    output: Vec<f32>,
    mlx_median_us: f64,
}

#[derive(Debug, Deserialize)]
struct IndexedCase {
    index: usize,
    value: f32,
    mlx_median_us: f64,
}

#[derive(Debug, Deserialize)]
struct TopKCase {
    indices: Vec<usize>,
    values: Vec<f32>,
    mlx_median_us: f64,
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

fn assert_latency_delta<F, T>(name: &str, mlx_median_us: f64, mut operation: F)
where
    F: FnMut() -> T,
{
    for _ in 0..METAL_LATENCY_WARMUP_RUNS {
        std::hint::black_box(operation());
    }

    let mut samples = Vec::with_capacity(METAL_LATENCY_SAMPLE_RUNS);
    for _ in 0..METAL_LATENCY_SAMPLE_RUNS {
        let started = Instant::now();
        std::hint::black_box(operation());
        samples.push(started.elapsed());
    }
    samples.sort_unstable();

    let metal_median_us = duration_us(samples[METAL_LATENCY_SAMPLE_RUNS / 2]);
    let delta_us = metal_median_us - mlx_median_us;
    let ratio = metal_median_us / mlx_median_us;
    assert!(
        metal_median_us.is_finite() && metal_median_us > 0.0,
        "missing Metal latency trace for {name}"
    );
    assert!(
        delta_us.is_finite() && ratio.is_finite() && ratio > 0.0,
        "invalid Metal/MLX latency delta for {name}"
    );
    println!(
        "mlx-reference {name}: metal_median_us={metal_median_us:.3} mlx_median_us={mlx_median_us:.3} delta_us={delta_us:+.3} ratio={ratio:.3}"
    );
}

fn duration_us(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000_000.0
}
